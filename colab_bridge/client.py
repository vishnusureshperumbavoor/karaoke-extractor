"""Google Colab MCP Client powered by vspagent with Demucs & Seed-VC pipelines."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from vspagent.colab.client import (
    ColabMCPClient,
    ColabClient,
    extract_text_from_result,
    extract_cell_id,
    SERVER_CMD,
    UVX_PATH,
)

SEED_VC_POLL_INTERVAL = 4.0
SEED_VC_JOB_TIMEOUT = 3600.0
SEED_VC_LOG_CHUNK = 200_000

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


class KaraokeColabClient(ColabMCPClient):
    """Google Colab MCP Client subclassed from vspagent with audio processing pipelines."""

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
        """Runs Seed-VC zero-shot singing voice conversion on Google Colab GPU."""
        await self.ensure_connected()

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


# Persistent singleton for Karaoke Extractor
internal_colab_client = KaraokeColabClient()
