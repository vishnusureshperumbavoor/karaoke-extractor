import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

UVX_PATH = shutil.which("uvx") or os.path.expanduser("~/.local/bin/uvx")
SERVER_CMD = [UVX_PATH, "--refresh", "git+https://github.com/googlecolab/colab-mcp"]

# --- Colab Seed-VC streaming settings -------------------------------------
# Seed-VC is launched as a background process on the Colab runtime so its log
# can be streamed back to the bridge (visible via /api/colab/status -> recent_logs)
# instead of looking frozen for the entire (possibly multi-minute) conversion.
SEED_VC_POLL_INTERVAL = 4.0     # seconds between log polls on the Colab runtime
SEED_VC_JOB_TIMEOUT = 3600.0    # max seconds to wait for a single Seed-VC job
SEED_VC_LOG_CHUNK = 200_000     # max bytes read from the remote log per poll

# Noise lines from the Colab runtime that are irrelevant to a user watching the
# conversion progress (pip resolver chatter, ffmpeg banners, HF/transformers warnings).
NOISY_LOG_MARKERS = (
    "It is strongly recommended to pass the `sampling_rate`",
    "ERROR: pip's dependency resolver",
    "This behaviour is the source of the following dependency conflicts",
    "requires protobuf",
    "You are sending unauthenticated requests to the HF Hub",
    "Skipped loading some keys due to shape mismatch",
    "ffmpeg version",
    "configuration: --prefix",
    "built with gcc",
    "libavutil", "libavcodec", "libavformat", "libavdevice",
    "libavfilter", "libswscale", "libswresample", "libpostproc",
    "[aist#", "Guessed Channel Layout",
    "Input #0", "Output #0", "Stream mapping:", "Press [q] to stop",
    "TSSE", "  encoder", "Metadata:",
    "size=", "bitrate=", "speed=",
)


def _is_noisy_log_line(line: str) -> bool:
    return any(marker in line for marker in NOISY_LOG_MARKERS)


def flatten_colab_output(raw: Any) -> str:
    """Convert a Colab run_code_cell result into clean, human-readable text.

    The MCP returns the cell's outputs as a JSON structure (a list of stream
    chunks). Left as-is it renders as escaped blobs like ``\\n","line\\n","``
    in the UI. This flattens every output's text into real newline-joined text.
    """
    if not isinstance(raw, str) or not raw.strip():
        return raw if isinstance(raw, str) else ""

    text = raw.strip()
    if not text.startswith("{"):
        return raw

    try:
        data = json.loads(text)
    except Exception:
        return raw
    if not isinstance(data, dict):
        return raw

    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return raw

    parts: List[str] = []
    for out in outputs:
        if not isinstance(out, dict):
            continue
        chunk = out.get("text")
        if isinstance(chunk, list):
            parts.append("".join(str(c) for c in chunk))
        elif isinstance(chunk, str):
            parts.append(chunk)
        else:
            traceback = out.get("traceback")
            if isinstance(traceback, list):
                parts.append("\n".join(str(t) for t in traceback))

    return "".join(parts) if parts else raw



def extract_text_from_result(result_obj: Any) -> str:
    """Extracts text content or raw result from an MCP tool response."""
    if not isinstance(result_obj, dict):
        return str(result_obj)

    if "structuredContent" in result_obj and "result" in result_obj["structuredContent"]:
        return str(result_obj["structuredContent"]["result"])

    contents = result_obj.get("content", [])
    text_parts = []
    for c in contents:
        if isinstance(c, dict) and c.get("type") == "text":
            text_parts.append(c.get("text", ""))
    if text_parts:
        return "\n".join(text_parts).strip()

    if "result" in result_obj:
        return str(result_obj["result"])
    return json.dumps(result_obj)


def extract_cell_id(result_obj: Any) -> str:
    """Extracts clean cell ID from add_code_cell / add_text_cell response."""
    if isinstance(result_obj, dict):
        if "structuredContent" in result_obj and isinstance(result_obj["structuredContent"], dict):
            sc = result_obj["structuredContent"]
            if "newCellId" in sc:
                return str(sc["newCellId"]).strip('"\' ')
        if "result" in result_obj and isinstance(result_obj["result"], dict):
            r = result_obj["result"]
            if "newCellId" in r:
                return str(r["newCellId"]).strip('"\' ')

    text = extract_text_from_result(result_obj)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return str(data.get("newCellId") or data.get("cellId") or data.get("id") or text).strip('"\' ')
    except Exception:
        pass

    match = re.search(r'["\']newCellId["\']\s*:\s*["\']([^"\']+)["\']', text)
    if match:
        return match.group(1).strip()

    return text.strip('"\' \n{}')


class ColabMCPClient:
    """Persistent Google Colab MCP client with a dedicated background asyncio loop for Flask."""

    def __init__(self):
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.status = "disconnected"  # disconnected, connecting, connected, error
        self.status_message = "Not started"
        self.gpu_info: Optional[str] = None
        self.selected_profile: Optional[str] = "Default"
        self.tools: Dict[str, Any] = {}
        self.logs: deque = deque(maxlen=250)

        self._lock: Optional[asyncio.Lock] = None
        self._connected_event: Optional[asyncio.Event] = None
        self._request_id = 0
        self._pending_futures: Dict[int, asyncio.Future] = {}
        self._reader_tasks: List[asyncio.Task] = []

        # Dedicated background event loop thread so synchronous Flask routes share one loop
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5.0)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._lock = asyncio.Lock()
        self._connected_event = asyncio.Event()
        self._loop_ready.set()
        self._loop.run_forever()

    def run_sync(self, coro, timeout: float = 7200.0) -> Any:
        """Run an async method on the persistent background event loop from synchronous Flask routes."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def log(self, message: str):
        print(f"[ColabMCP] {message}")
        self.logs.append(message)

    async def start(self, force_restart: bool = False):
        """Starts the persistent colab-mcp subprocess and performs MCP handshake."""
        async with self._lock:
            if self.proc and self.proc.returncode is None and not force_restart:
                return

            if self.proc and self.proc.returncode is None and force_restart:
                try:
                    self.proc.terminate()
                    await asyncio.wait_for(self.proc.wait(), timeout=2.0)
                except Exception:
                    pass

            self.status = "connecting"
            self.status_message = "Starting internal colab-mcp daemon..."
            self.log(f"Launching server: {' '.join(SERVER_CMD)}")

            try:
                from colab_bridge.chrome_profiles import browser_env_command, set_selected_profile

                set_selected_profile(self.selected_profile or "Default")
                child_env = os.environ.copy()
                child_env["BROWSER"] = browser_env_command()

                self.proc = await asyncio.create_subprocess_exec(
                    *SERVER_CMD,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=child_env,
                    limit=32 * 1024 * 1024,
                )
            except Exception as e:
                self.status = "error"
                self.status_message = f"Failed to start colab-mcp process: {e}"
                self.log(self.status_message)
                return

            self._reader_tasks = [
                asyncio.create_task(self._read_stdout()),
                asyncio.create_task(self._read_stderr()),
            ]

            try:
                self._request_id += 1
                init_msg = {
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "roots": {"listChanged": True},
                        },
                        "clientInfo": {
                            "name": "karaoke-extractor-colab-mcp",
                            "version": "1.0.0",
                        },
                    },
                }
                self.log("Sending MCP initialize...")
                res = await self._send_request(init_msg)
                self.log(f"Initialized. Server: {res.get('serverInfo', {}).get('name', 'ColabMCP')}")

                await self._send_notification("notifications/initialized", {})
                await self.refresh_tools()
                self.status_message = "Server started. Ready to pair with Colab."
                self.log("Internal colab-mcp server initialized successfully.")

            except Exception as e:
                self.status = "error"
                self.status_message = f"Handshake failed: {e}"
                self.log(self.status_message)

    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        if not self.proc or not self.proc.stdin:
            return
        msg: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        data = json.dumps(msg) + "\n"
        self.proc.stdin.write(data.encode("utf-8"))
        await self.proc.stdin.drain()

    async def _send_request(self, msg: Dict[str, Any]) -> Any:
        req_id = msg["id"]
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_futures[req_id] = future

        data = json.dumps(msg) + "\n"
        self.proc.stdin.write(data.encode("utf-8"))
        await self.proc.stdin.drain()

        try:
            return await asyncio.wait_for(future, timeout=3600.0)
        finally:
            self._pending_futures.pop(req_id, None)

    async def _read_stdout(self):
        if self.proc and self.proc.stdout and hasattr(self.proc.stdout, "_limit"):
            self.proc.stdout._limit = 32 * 1024 * 1024
        while self.proc and self.proc.stdout:
            try:
                line = await self.proc.stdout.readline()
            except Exception as e:
                self.log(f"[STDOUT READ ERROR] {e}")
                continue
            if not line:
                break
            line_str = line.decode(errors="replace").strip()
            if not line_str:
                continue

            try:
                msg = json.loads(line_str)
            except json.JSONDecodeError:
                self.log(f"[STDOUT] {line_str}")
                continue

            if "id" in msg and msg["id"] in self._pending_futures:
                fut = self._pending_futures[msg["id"]]
                if not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"].get("message", "MCP error")))
                    else:
                        fut.set_result(msg.get("result", {}))

            elif "method" in msg:
                method = msg["method"]
                if method == "notifications/tools/list_changed":
                    self.log("Tools changed. Refreshing...")
                    asyncio.create_task(self.refresh_tools())

    async def _read_stderr(self):
        noisy_patterns = [
            "FastMCP", "DeprecationWarning", "│", "╭", "╯", "╰", "───",
            "FastMCP 3.0", "Update available", "logging to", "compat", "authlib",
        ]
        while self.proc and self.proc.stderr:
            line = await self.proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace").rstrip()
            if text and not any(p in text for p in noisy_patterns):
                self.log(f"[SERVER] {text}")

    async def refresh_tools(self):
        """Fetches the updated list of tools from the MCP server."""
        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/list",
            "params": {},
        }
        res = await self._send_request(msg)
        tools_list = res.get("tools", [])
        self.tools = {t["name"]: t for t in tools_list}
        self.log(f"Active tools ({len(self.tools)}): {', '.join(self.tools.keys())}")

        if "add_code_cell" in self.tools and "run_code_cell" in self.tools:
            was_connected = self.status == "connected"
            self.status = "connected"
            self.status_message = "Connected to Google Colab GPU session!"
            self._connected_event.set()
            if not was_connected and not self.gpu_info:
                asyncio.create_task(self._probe_gpu_info())

    async def _probe_gpu_info(self):
        """Probes GPU name & VRAM on the connected Colab runtime."""
        try:
            probe_code = (
                "import torch\n"
                "if torch.cuda.is_available():\n"
                "    props = torch.cuda.get_device_properties(0)\n"
                "    print(f'GPU_INFO:{torch.cuda.get_device_name(0)} ({props.total_memory / (1024**3):.1f} GB VRAM)')\n"
                "else:\n"
                "    print('GPU_INFO:CPU Runtime (Enable GPU in Runtime -> Change runtime type)')"
            )
            res = await self.execute_code(probe_code, keep_cell=False)
            for line in res.get("output", "").splitlines():
                if line.startswith("GPU_INFO:"):
                    self.gpu_info = line.split("GPU_INFO:", 1)[1].strip()
                    self.status_message = f"Connected: {self.gpu_info}"
                    self.log(f"Detected Colab runtime: {self.gpu_info}")
                    break
        except Exception as e:
            self.log(f"GPU probe warning: {e}")

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Executes a tool on the internal Colab MCP server."""
        if not self.proc or self.proc.returncode is not None:
            await self.start()

        self._request_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {},
            },
        }
        return await self._send_request(msg)

    async def connect_browser(
        self, profile_directory: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """Calls open_colab_browser_connection to pair with browser tab."""
        from colab_bridge.chrome_profiles import set_selected_profile

        if self.status == "connected" and not force:
            return {
                "status": "already_connected",
                "connected": True,
                "gpu_info": self.gpu_info,
                "message": self.status_message,
            }

        selected = set_selected_profile(profile_directory)
        self.selected_profile = selected.get("profileDirectory") or "Default"
        self.log(
            f"Calling open_colab_browser_connection (Chrome profile={self.selected_profile})..."
        )
        self.status = "connecting"
        self.status_message = f"Waiting for Colab tab pairing ({self.selected_profile})..."
        res = await self.call_tool("open_colab_browser_connection", {})
        await self.refresh_tools()
        return {
            "status": "connected" if self.status == "connected" else "connecting",
            "profileDirectory": self.selected_profile,
            "connected": self.status == "connected",
            "gpu_info": self.gpu_info,
            "message": self.status_message,
            "result": res,
        }

    async def ensure_connected(self, timeout: float = 45.0):
        """Ensures a connected session exists."""
        if self.status == "connected":
            return

        self._connected_event.clear()
        self.status = "connecting"
        self.status_message = "Opening Google Colab tab..."
        self.log("Auto-triggering open_colab_browser_connection...")

        try:
            await self.call_tool("open_colab_browser_connection", {})
        except Exception as e:
            self.status = "error"
            self.status_message = f"Failed browser pairing: {e}"
            raise RuntimeError(self.status_message)

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            self.log("Auto-connection successful!")
        except asyncio.TimeoutError:
            self.status = "disconnected"
            self.status_message = "Connection timed out waiting for Colab tab."
            raise RuntimeError("Timed out waiting for Google Colab tab to pair.")

    async def execute_code(self, code: str, keep_cell: bool = True) -> Dict[str, Any]:
        """Inserts code cell into Colab, executes it, and returns output."""
        await self.ensure_connected()
        if "run_code_cell" not in self.tools:
            await self.refresh_tools()

        async with self._lock:
            add_res = await self.call_tool(
                "add_code_cell",
                {
                    "cellIndex": 0,
                    "language": "python",
                    "code": code,
                },
            )
            cell_id = extract_cell_id(add_res)
            exec_res = await self.call_tool(
                "run_code_cell",
                {
                    "cellId": cell_id,
                },
            )

            output_text = extract_text_from_result(exec_res)
            output_text = flatten_colab_output(output_text)
            is_error = exec_res.get("isError", False) if isinstance(exec_res, dict) else False

            if not keep_cell and "delete_cell" in self.tools:
                try:
                    await self.call_tool("delete_cell", {"cellId": cell_id})
                except Exception:
                    pass

            return {
                "cellId": cell_id,
                "output": output_text,
                "isError": is_error,
                "raw": exec_res,
            }

    async def upload_file_to_colab(self, local_path: str, remote_path: str):
        """Transfers a local binary file to Google Colab via chunked base64 cells."""
        with open(local_path, "rb") as f:
            raw_bytes = f.read()
        b64_data = base64.b64encode(raw_bytes).decode("ascii")
        chunk_size = 700_000  # ~700 KB per cell
        chunks = [b64_data[i : i + chunk_size] for i in range(0, len(b64_data), chunk_size)]
        self.log(f"Uploading {os.path.basename(local_path)} ({len(raw_bytes)/1024:.1f} KB) to Colab {remote_path} in {len(chunks)} chunk(s)...")

        tmp_b64 = f"{remote_path}.b64"
        remote_dir = os.path.dirname(remote_path) or "/content"
        for idx, chunk in enumerate(chunks):
            mode = "w" if idx == 0 else "a"
            is_last = idx == len(chunks) - 1
            code = (
                "import base64, os\n"
                f"os.makedirs({remote_dir!r}, exist_ok=True)\n"
                f"with open({tmp_b64!r}, {mode!r}) as f:\n"
                f"    f.write({chunk!r})\n"
            )
            if is_last:
                code += (
                    f"with open({tmp_b64!r}, 'r') as f_in, open({remote_path!r}, 'wb') as f_out:\n"
                    "    f_out.write(base64.b64decode(f_in.read()))\n"
                    f"os.remove({tmp_b64!r})\n"
                    f"print('UPLOAD_OK:', os.path.getsize({remote_path!r}))\n"
                )
            res = await self.execute_code(code, keep_cell=False)
            if res.get("isError") or (is_last and "UPLOAD_OK:" not in res.get("output", "")):
                raise RuntimeError(f"Failed uploading {local_path} (chunk {idx+1}/{len(chunks)}) to Colab: {res.get('output')}")

    async def download_file_from_colab(self, remote_path: str, local_path: str):
        """Downloads a binary file from Google Colab to local_path via chunked base64 slices."""
        tmp_b64 = f"{remote_path}.b64"
        prep_code = (
            "import base64, os\n"
            f"if not os.path.exists({remote_path!r}):\n"
            f"    raise FileNotFoundError({remote_path!r})\n"
            f"with open({remote_path!r}, 'rb') as f:\n"
            "    b64 = base64.b64encode(f.read()).decode('ascii')\n"
            f"with open({tmp_b64!r}, 'w') as f:\n"
            "    f.write(b64)\n"
            "print(f'B64_LEN:{len(b64)}')\n"
        )
        res = await self.execute_code(prep_code, keep_cell=False)
        out = res.get("output", "")
        match = re.search(r"B64_LEN:(\d+)", out)
        if not match:
            raise RuntimeError(f"Failed preparing download for {remote_path}: {out}")
        total_len = int(match.group(1))
        chunk_size = 150_000
        parts: List[str] = []
        self.log(f"Downloading {remote_path} ({total_len * 3 // 4 // 1024} KB) from Colab...")

        for offset in range(0, total_len, chunk_size):
            slice_code = (
                f"with open({tmp_b64!r}, 'r') as f:\n"
                f"    f.seek({offset})\n"
                f"    data = f.read({chunk_size})\n"
                "print('===B64_START===' + data + '===B64_END===')\n"
            )
            slice_res = await self.execute_code(slice_code, keep_cell=False)
            slice_out = slice_res.get("output", "")
            m = re.search(r"===B64_START===(.*?)===B64_END===", slice_out, re.S)
            if not m:
                raise RuntimeError(f"Failed reading slice at {offset} from Colab: {slice_out[:300]}")
            parts.append(m.group(1).strip())

        await self.execute_code(f"import os\nif os.path.exists({tmp_b64!r}): os.remove({tmp_b64!r})", keep_cell=False)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(base64.b64decode("".join(parts)))
        self.log(f"Downloaded {local_path} ({os.path.getsize(local_path)/1024:.1f} KB) successfully.")

    async def run_demucs_on_colab(
        self, input_audio_path: str, out_vocals_mp3_path: str, out_no_vocals_mp3_path: str
    ):
        """Runs Demucs stem separation on Google Colab GPU and downloads vocals.mp3 & no_vocals.mp3."""
        await self.ensure_connected()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
            compact_in = tmp_in.name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_audio_path, "-b:a", "160k", compact_in],
                check=True,
                capture_output=True,
            )
            await self.upload_file_to_colab(compact_in, "/content/karaoke_job/input.mp3")
        finally:
            if os.path.exists(compact_in):
                os.remove(compact_in)

        demucs_code = """import os, subprocess, sys
if not os.path.exists('/content/.demucs_installed'):
    print('Installing Demucs on Colab GPU...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'demucs'], check=True)
    open('/content/.demucs_installed', 'w').close()

subprocess.run([
    sys.executable, '-m', 'demucs.separate',
    '--two-stems=vocals',
    '-n', 'htdemucs',
    '--mp3', '--mp3-bitrate', '192',
    '-o', '/content/karaoke_job/out',
    '/content/karaoke_job/input.mp3'
], check=True)
print('DEMUCS_COLAB_DONE')
"""
        self.log("Running Demucs stem separation on Google Colab GPU...")
        res = await self.execute_code(demucs_code, keep_cell=True)
        if "DEMUCS_COLAB_DONE" not in res.get("output", ""):
            raise RuntimeError(f"Colab Demucs failed: {res.get('output')}")

        await self.download_file_from_colab(
            "/content/karaoke_job/out/htdemucs/input/vocals.mp3", out_vocals_mp3_path
        )
        await self.download_file_from_colab(
            "/content/karaoke_job/out/htdemucs/input/no_vocals.mp3", out_no_vocals_mp3_path
        )

    async def run_seed_vc_on_colab(
        self,
        vocals_path: str,
        target_voice_path: str,
        out_ai_vocals_mp3_path: str,
        diffusion_steps: int = 25,
        pitch_shift: int = 0,
        f0_condition: bool = True,
        auto_f0_adjust: bool = True,
    ):
        """Runs Seed-VC zero-shot singing voice conversion on Google Colab GPU.

        Seed-VC is launched as a background process on the Colab runtime and its
        log is streamed back into the bridge logs, so a multi-minute conversion
        is observable (via /api/colab/status -> recent_logs) instead of looking
        frozen. Diffusion steps are preserved for maximum quality.
        """
        await self.ensure_connected()

        # Compress source vocals and target voice ref into compact MP3s for fast upload
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_v, \
             tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_r:
            compact_vocals = tmp_v.name
            compact_ref = tmp_r.name

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", vocals_path, "-b:a", "128k", compact_vocals],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", target_voice_path, "-t", "12", "-ar", "44100", "-ac", "1", "-b:a", "128k", compact_ref],
                check=True,
                capture_output=True,
            )
            await self.upload_file_to_colab(compact_vocals, "/content/seedvc_job/src_vocals.mp3")
            await self.upload_file_to_colab(compact_ref, "/content/seedvc_job/ref_voice.mp3")
        finally:
            for p in (compact_vocals, compact_ref):
                if os.path.exists(p):
                    os.remove(p)

        # The runner is embedded as base64 so no quoting/escaping can break it.
        runner_code = r'''import os, glob, shutil, subprocess, sys

SEED_VC_DIR = '/content/seed-vc'
JOB_DIR = '/content/seedvc_job'
OUT_DIR = os.path.join(JOB_DIR, 'out')
DONE_FLAG = os.path.join(JOB_DIR, 'seed_vc.done')
FAIL_FLAG = os.path.join(JOB_DIR, 'seed_vc.failed')


def log(msg):
    print(msg, flush=True)


for _f in (DONE_FLAG, FAIL_FLAG):
    try:
        os.remove(_f)
    except OSError:
        pass

try:
    if not os.path.exists(os.path.join(SEED_VC_DIR, 'inference.py')):
        log('[seed-vc] Cloning Seed-VC onto Colab GPU...')
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/Plachtaa/seed-vc.git', SEED_VC_DIR], check=True)

    if not os.path.exists('/content/.seed_vc_deps_ready'):
        log('[seed-vc] Installing Seed-VC dependencies on Colab...')
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'munch', 'einops', 'descript-audio-codec', 'soundfile'
        ], check=True)
        open('/content/.seed_vc_deps_ready', 'w').close()

    # Apply compatibility patches once (PyTorch 2.6+, transformers 5.x, huggingface_hub 1.x, torchaudio 2.10+).
    if not os.path.exists('/content/.seed_vc_patched_v2'):
        log('[seed-vc] Applying compatibility patches...')
        lr_path = os.path.join(SEED_VC_DIR, 'modules', 'length_regulator.py')
        if os.path.exists(lr_path):
            code = open(lr_path).read()
            code = code.replace('\nfrom dac.nn.quantize import VectorQuantize', '')
            code = code.replace('if vector_quantize:\n                self.vq = VectorQuantize',
                                'if vector_quantize:\n                from dac.nn.quantize import VectorQuantize\n                self.vq = VectorQuantize')
            open(lr_path, 'w').write(code)

        bv_path = os.path.join(SEED_VC_DIR, 'modules', 'bigvgan', 'bigvgan.py')
        if os.path.exists(bv_path):
            bv = open(bv_path).read()
            bv = bv.replace('proxies: Optional[Dict],', 'proxies: Optional[Dict] = None,')
            bv = bv.replace('resume_download: bool,', 'resume_download: bool = False,')
            bv = bv.replace('resume_download=resume_download,', '')
            bv = bv.replace('torch.load(model_file, map_location=map_location)', 'torch.load(model_file, map_location=map_location, weights_only=False)')
            open(bv_path, 'w').write(bv)

        for rel in ['inference.py', 'modules/commons.py', 'modules/rmvpe.py']:
            p = os.path.join(SEED_VC_DIR, rel)
            if os.path.exists(p):
                c = open(p).read()
                c = c.replace('map_location="cpu")', 'map_location="cpu", weights_only=False)')
                c = c.replace("map_location='cpu')", "map_location='cpu', weights_only=False)")
                c = c.replace(
                    "whisper_model._mask_input_features(\n                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask)",
                    "(whisper_model._mask_input_features(ori_inputs.input_features, attention_mask=ori_inputs.attention_mask) if hasattr(whisper_model, '_mask_input_features') else ori_inputs.input_features)"
                )
                c = c.replace(
                    'torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)',
                    'import soundfile as sf; sf.write(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.squeeze().cpu().numpy(), sr)'
                )
                open(p, 'w').write(c)
        open('/content/.seed_vc_patched_v2', 'w').close()

    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, '-u', 'inference.py',
        '--source', os.path.join(JOB_DIR, 'src_vocals.mp3'),
        '--target', os.path.join(JOB_DIR, 'ref_voice.mp3'),
        '--output', OUT_DIR,
        '--diffusion-steps', '__DIFFUSION_STEPS__',
        '--f0-condition', '__F0_CONDITION__',
        '--auto-f0-adjust', '__AUTO_F0_ADJUST__',
        '--semi-tone-shift', '__PITCH_SHIFT__',
        '--fp16', 'True'
    ]
    log('[seed-vc] Running: ' + ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=SEED_VC_DIR, check=True)

    wavs = glob.glob(os.path.join(OUT_DIR, '*.wav'))
    if not wavs:
        raise RuntimeError('No output wav generated by Seed-VC')

    ai_vocals_mp3 = os.path.join(JOB_DIR, 'ai_vocals.mp3')
    subprocess.run(['ffmpeg', '-y', '-i', wavs[0], '-ac', '1', '-b:a', '128k', ai_vocals_mp3], check=True)
    log('SEED_VC_COLAB_DONE: ' + str(os.path.getsize(ai_vocals_mp3)))
    open(DONE_FLAG, 'w').close()
except Exception as exc:
    log('SEED_VC_COLAB_FAILED: ' + repr(exc))
    open(FAIL_FLAG, 'w').close()
    raise
'''
        runner_code = (
            runner_code
            .replace("__DIFFUSION_STEPS__", str(int(diffusion_steps)))
            .replace("__F0_CONDITION__", str(bool(f0_condition)))
            .replace("__AUTO_F0_ADJUST__", str(bool(auto_f0_adjust)))
            .replace("__PITCH_SHIFT__", str(int(pitch_shift)))
        )
        runner_b64 = base64.b64encode(runner_code.encode("utf-8")).decode("ascii")

        launch_code = """import base64, os, subprocess, sys
JOB_DIR = '/content/seedvc_job'
os.makedirs(JOB_DIR, exist_ok=True)
RUNNER_PATH = os.path.join(JOB_DIR, 'seed_vc_runner.py')
LOG_PATH = os.path.join(JOB_DIR, 'seed_vc.log')
DONE_FLAG = os.path.join(JOB_DIR, 'seed_vc.done')
FAIL_FLAG = os.path.join(JOB_DIR, 'seed_vc.failed')
for _f in (LOG_PATH, DONE_FLAG, FAIL_FLAG):
    try:
        os.remove(_f)
    except OSError:
        pass
with open(RUNNER_PATH, 'wb') as fh:
    fh.write(base64.b64decode('__RUNNER_B64__'))
log_fh = open(LOG_PATH, 'wb')
proc = subprocess.Popen(
    [sys.executable, '-u', RUNNER_PATH],
    stdout=log_fh, stderr=subprocess.STDOUT,
    cwd=JOB_DIR, start_new_session=True,
)
print('SEED_VC_LAUNCHED:', proc.pid)
""".replace("__RUNNER_B64__", runner_b64)

        self.log(f"Launching Seed-VC ({diffusion_steps} steps, fp16=True) on Google Colab GPU (streaming log)...")
        res = await self.execute_code(launch_code, keep_cell=True)
        if "SEED_VC_LAUNCHED" not in res.get("output", ""):
            raise RuntimeError(f"Failed to launch Seed-VC on Colab: {res.get('output')}")

        offset = 0
        started = time.time()
        last_line = ""
        last_progress_log = 0.0
        while True:
            poll_code = """import os, sys
JOB_DIR = '/content/seedvc_job'
LOG_PATH = os.path.join(JOB_DIR, 'seed_vc.log')
DONE_FLAG = os.path.join(JOB_DIR, 'seed_vc.done')
FAIL_FLAG = os.path.join(JOB_DIR, 'seed_vc.failed')
offset = __OFFSET__
size = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
if offset > size:
    offset = 0
with open(LOG_PATH, 'rb') as fh:
    fh.seek(offset)
    data = fh.read(__CHUNK__)
new_offset = offset + len(data)
print('===LOGCHUNK_START===')
sys.stdout.write(data.decode('utf-8', 'replace'))
print('===LOGCHUNK_END===')
print('LOG_OFFSET:', new_offset)
print('STATUS_DONE:', 1 if os.path.exists(DONE_FLAG) else 0)
print('STATUS_FAILED:', 1 if os.path.exists(FAIL_FLAG) else 0)
""".replace("__OFFSET__", str(offset)).replace("__CHUNK__", str(SEED_VC_LOG_CHUNK))

            poll_res = await self.execute_code(poll_code, keep_cell=False)
            out = poll_res.get("output", "")

            chunk_match = re.search(r"===LOGCHUNK_START===\n?(.*?)===LOGCHUNK_END===", out, re.S)
            if chunk_match:
                for raw_line in re.split(r"[\r\n]+", chunk_match.group(1)):
                    line = raw_line.strip()
                    if not line or line == last_line:
                        continue
                    last_line = line
                    if _is_noisy_log_line(line):
                        continue
                    is_progress = bool(re.search(r"\d+%\|", line)) or line.endswith(("it/s", "s/it"))
                    if is_progress and time.time() - last_progress_log < 5.0:
                        continue
                    if is_progress:
                        last_progress_log = time.time()
                    self.log(f"[Colab] {line}")

            off_match = re.search(r"LOG_OFFSET:\s*(\d+)", out)
            if off_match:
                offset = int(off_match.group(1))

            if "STATUS_DONE: 1" in out:
                self.log("Seed-VC conversion finished on Google Colab GPU.")
                break
            if "STATUS_FAILED: 1" in out:
                raise RuntimeError("Seed-VC failed on Google Colab GPU (see bridge logs above).")
            if time.time() - started > SEED_VC_JOB_TIMEOUT:
                raise TimeoutError(f"Seed-VC on Colab exceeded {SEED_VC_JOB_TIMEOUT:.0f}s.")

            await asyncio.sleep(SEED_VC_POLL_INTERVAL)

        await self.download_file_from_colab("/content/seedvc_job/ai_vocals.mp3", out_ai_vocals_mp3_path)


internal_colab_client = ColabMCPClient()
