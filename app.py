"""FastAPI Backend for Karaoke Extractor & AI Voice Studio powered by vspagent."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import shutil
import subprocess
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Query, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from colab_bridge import internal_colab_client, list_chrome_accounts

BASE_IGNORED_DIR = os.path.abspath(".ignored")
UPLOAD_FOLDER = os.path.join(BASE_IGNORED_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_IGNORED_DIR, "processed")
SONGS_FOLDER = os.path.join(BASE_IGNORED_DIR, "songs")
VOICE_FOLDER = os.path.join(BASE_IGNORED_DIR, "vsp_voice")
SEED_VC_DIR = os.path.join(BASE_IGNORED_DIR, "seed-vc")
STATIC_FOLDER = os.path.abspath("static")

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, SONGS_FOLDER, VOICE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

AUDIO_EXTENSIONS = (".mp3", ".mpeg", ".mpg", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac")


def ensure_seed_vc_ready():
    """Ensure Seed-VC is cloned and patched inside .ignored/seed-vc for local zero-shot voice conversion."""
    inference_script = os.path.join(SEED_VC_DIR, "inference.py")
    if not os.path.exists(inference_script):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Plachtaa/seed-vc.git", SEED_VC_DIR],
            check=True,
            capture_output=True,
            text=True
        )

    # Ensure compatibility with PyTorch 2.6+ (weights_only=False), CPU float32 Whisper, and lazy DAC import
    lr_path = os.path.join(SEED_VC_DIR, "modules", "length_regulator.py")
    if os.path.exists(lr_path):
        with open(lr_path, "r", encoding="utf-8") as f:
            lr_code = f.read()
        if "\nfrom dac.nn.quantize import VectorQuantize" in lr_code:
            lr_code = lr_code.replace("\nfrom dac.nn.quantize import VectorQuantize", "")
            lr_code = lr_code.replace(
                "if vector_quantize:\n                self.vq = VectorQuantize",
                "if vector_quantize:\n                from dac.nn.quantize import VectorQuantize\n                self.vq = VectorQuantize"
            )
            with open(lr_path, "w", encoding="utf-8") as f:
                f.write(lr_code)

    bigvgan_utils = os.path.join(SEED_VC_DIR, "modules", "bigvgan", "utils.py")
    if os.path.exists(bigvgan_utils):
        with open(bigvgan_utils, "r", encoding="utf-8") as f:
            bu_code = f.read()
        if "\nimport matplotlib\n" in bu_code:
            bu_code = bu_code.replace(
                '\nimport matplotlib\nimport numpy as np\nfrom scipy.io.wavfile import write\nimport torch\nfrom torch.nn.utils import weight_norm\nmatplotlib.use("Agg")\nimport matplotlib.pylab as plt',
                '\nimport numpy as np\nfrom scipy.io.wavfile import write\nimport torch\nfrom torch.nn.utils import weight_norm'
            )
            bu_code = bu_code.replace(
                'def plot_spectrogram(spectrogram):\n    fig, ax = plt.subplots(figsize=(10, 2))',
                'def plot_spectrogram(spectrogram):\n    import matplotlib\n    matplotlib.use("Agg")\n    import matplotlib.pylab as plt\n    fig, ax = plt.subplots(figsize=(10, 2))'
            )
            with open(bigvgan_utils, "w", encoding="utf-8") as f:
                f.write(bu_code)

    for rel_path in ["inference.py", "modules/commons.py", "modules/rmvpe.py"]:
        target_file = os.path.join(SEED_VC_DIR, rel_path)
        if os.path.exists(target_file):
            with open(target_file, "r", encoding="utf-8") as f:
                code = f.read()
            updated = code.replace('map_location="cpu")', 'map_location="cpu", weights_only=False)')
            updated = updated.replace("map_location='cpu')", "map_location='cpu', weights_only=False)")
            updated = updated.replace(
                "WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16)",
                'WhisperModel.from_pretrained(whisper_name, torch_dtype=(torch.float16 if (fp16 and device.type != "cpu") else torch.float32))'
            )
            updated = updated.replace(
                'torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)',
                'import soundfile as sf\n    sf.write(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.squeeze().cpu().numpy(), sr)'
            )
            if updated != code:
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(updated)

    return inference_script


def find_existing_job_for_song(song_name: str) -> Optional[str]:
    """Return existing job_id if a song with song_name was already separated by Demucs."""
    if not os.path.exists(PROCESSED_FOLDER):
        return None
    for job_id in os.listdir(PROCESSED_FOLDER):
        job_dir = os.path.join(PROCESSED_FOLDER, job_id)
        meta_path = os.path.join(job_dir, "metadata.txt")
        karaoke_path = os.path.join(job_dir, "htdemucs", "input", "no_vocals.mp3")
        vocals_path = os.path.join(job_dir, "htdemucs", "input", "vocals.mp3")
        if os.path.exists(meta_path) and os.path.exists(karaoke_path) and os.path.exists(vocals_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                if f.read().strip() == song_name:
                    return job_id
    return None


async def separate_song_file(source_file_path: str, original_filename: str, use_colab: Optional[bool] = None) -> str:
    """Convert any audio file to MP3 and run Demucs stem separation (on Colab GPU if connected, else local CPU)."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp3")
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg", "-y",
            "-i", source_file_path,
            "-vn",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "192k",
            input_path
        ],
        capture_output=True, text=True, check=True
    )

    out_dir = os.path.join(PROCESSED_FOLDER, job_id)
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "metadata.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(original_filename)

    separated_dir = os.path.join(out_dir, "htdemucs", "input")
    os.makedirs(separated_dir, exist_ok=True)
    karaoke_path = os.path.join(separated_dir, "no_vocals.mp3")
    vocals_path = os.path.join(separated_dir, "vocals.mp3")

    should_use_colab = (
        use_colab if use_colab is not None else (internal_colab_client.status == "connected")
    )

    if should_use_colab and internal_colab_client.status == "connected":
        await internal_colab_client.run_demucs_on_colab(input_path, vocals_path, karaoke_path)
    else:
        cmd = [
            os.path.abspath("./venv/bin/demucs"),
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "--mp3",
            "-o", out_dir,
            input_path
        ]
        await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, check=True)

    if not os.path.exists(karaoke_path):
        raise RuntimeError("Failed to generate karaoke audio track")

    return job_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        ensure_seed_vc_ready()
    except Exception as e:
        print("Warning initializing Seed-VC:", e)
    yield
    # Shutdown
    await internal_colab_client.disconnect()


app = FastAPI(title="Karaoke Extractor & AI Voice Studio", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Google Colab MCP Endpoints ---

@app.get("/api/colab/status")
async def colab_status():
    """Returns the internal Google Colab MCP daemon status."""
    return {
        "status": internal_colab_client.status,
        "message": internal_colab_client.status_message,
        "gpu_info": internal_colab_client.gpu_info,
        "profileDirectory": internal_colab_client.selected_profile,
        "tools": list(internal_colab_client.tools.keys()),
        "connected": internal_colab_client.status == "connected",
        "colabAvailable": internal_colab_client.status == "connected",
        "recent_logs": list(internal_colab_client.logs)[-25:],
    }


@app.get("/api/colab/accounts")
async def colab_accounts():
    """List Chrome profiles / Gmail accounts available on this machine."""
    return list_chrome_accounts()


@app.post("/api/colab/connect")
async def colab_connect(request: Request):
    """Triggers pairing with Google Colab tab in the selected Chrome profile."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    profile = request.query_params.get("profileDirectory") or body.get("profileDirectory")
    force = str(request.query_params.get("force") or body.get("force", "false")).lower() == "true"
    try:
        return await internal_colab_client.connect_browser(profile_directory=profile, force=force)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "connected": False, "message": str(e)})


@app.post("/api/colab/execute")
async def colab_execute(request: Request):
    """Executes Python code in Google Colab Cloud GPU."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    code = body.get("code", "")
    keep_cell = bool(body.get("keep_cell", True))
    if not code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    try:
        return await internal_colab_client.execute_code(code, keep_cell=keep_cell)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Library & Voices ---

@app.get("/api/library")
async def list_library():
    """List songs from .ignored/songs and voice samples from .ignored/vsp_voice."""
    songs = []
    if os.path.exists(SONGS_FOLDER):
        for fname in sorted(os.listdir(SONGS_FOLDER)):
            fpath = os.path.join(SONGS_FOLDER, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(AUDIO_EXTENSIONS):
                job_id = find_existing_job_for_song(fname)
                ai_cover_url = None
                ai_vocals_url = None
                karaoke_url = None
                vocals_url = None
                if job_id:
                    job_dir = os.path.join(PROCESSED_FOLDER, job_id)
                    karaoke_url = f"/api/download/{job_id}/no_vocals.mp3"
                    vocals_url = f"/api/download/{job_id}/vocals.mp3"
                    if os.path.exists(os.path.join(job_dir, "ai_cover_song.mp3")):
                        ai_cover_url = f"/api/download/{job_id}/ai_cover_song.mp3"
                    if os.path.exists(os.path.join(job_dir, "ai_vocals.mp3")):
                        ai_vocals_url = f"/api/download/{job_id}/ai_vocals.mp3"

                songs.append({
                    "filename": fname,
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    "job_id": job_id,
                    "is_extracted": job_id is not None,
                    "karaoke_url": karaoke_url,
                    "vocals_url": vocals_url,
                    "has_ai_cover": ai_cover_url is not None,
                    "ai_cover_url": ai_cover_url,
                    "ai_vocals_url": ai_vocals_url
                })

    voices = []
    if os.path.exists(VOICE_FOLDER):
        for fname in sorted(os.listdir(VOICE_FOLDER)):
            fpath = os.path.join(VOICE_FOLDER, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(AUDIO_EXTENSIONS):
                voices.append({
                    "filename": fname,
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    "url": f"/api/voices/{fname}"
                })

    return {"songs": songs, "voices": voices}


@app.get("/api/voices/{filename:path}")
async def download_voice_sample(filename: str):
    safe_name = os.path.basename(filename)
    target = os.path.join(VOICE_FOLDER, safe_name)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Voice file not found")

    if not safe_name.lower().endswith((".mp3", ".wav")):
        preview_dir = os.path.join(BASE_IGNORED_DIR, "voice_previews")
        os.makedirs(preview_dir, exist_ok=True)
        preview_name = f"{os.path.splitext(safe_name)[0]}.mp3"
        preview_path = os.path.join(preview_dir, preview_name)
        if not os.path.exists(preview_path):
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-i", target, "-vn", "-ar", "44100", "-ac", "1", "-b:a", "192k", preview_path],
                capture_output=True, check=False
            )
        if os.path.exists(preview_path):
            return FileResponse(preview_path)

    return FileResponse(target)


@app.post("/api/process-local-song")
async def process_local_song(request: Request):
    """Extract instrumental & vocals for a song stored in .ignored/songs/."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = dict(await request.form())

    song_filename = body.get("song_filename")
    if not song_filename:
        raise HTTPException(status_code=400, detail="Missing song_filename")

    safe_name = os.path.basename(song_filename)
    song_path = os.path.join(SONGS_FOLDER, safe_name)
    if not os.path.exists(song_path):
        raise HTTPException(status_code=404, detail=f"Song '{safe_name}' not found in .ignored/songs")

    existing_job_id = find_existing_job_for_song(safe_name)
    if existing_job_id:
        job_dir = os.path.join(PROCESSED_FOLDER, existing_job_id)
        return {
            "job_id": existing_job_id,
            "filename": safe_name,
            "karaoke_url": f"/api/download/{existing_job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{existing_job_id}/vocals.mp3",
            "ai_cover_url": f"/api/download/{existing_job_id}/ai_cover_song.mp3" if os.path.exists(os.path.join(job_dir, "ai_cover_song.mp3")) else None,
            "ai_vocals_url": f"/api/download/{existing_job_id}/ai_vocals.mp3" if os.path.exists(os.path.join(job_dir, "ai_vocals.mp3")) else None,
            "message": "Loaded previously separated tracks!"
        }

    try:
        job_id = await separate_song_file(song_path, safe_name)
        return {
            "job_id": job_id,
            "filename": safe_name,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "message": "Karaoke track separated successfully!"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"error": "Audio separation failed", "details": e.stderr})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
async def process_audio(file: UploadFile = File(...)):
    """Upload audio file and separate into stems."""
    original_filename = file.filename or "uploaded_song.mp3"
    raw_ext = os.path.splitext(original_filename)[1] or ".mp3"
    temp_dir = os.path.join(UPLOAD_FOLDER, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    raw_path = os.path.join(temp_dir, f"{uuid.uuid4()}{raw_ext}")

    content = await file.read()
    with open(raw_path, "wb") as f:
        f.write(content)

    try:
        job_id = await separate_song_file(raw_path, original_filename)
        return {
            "job_id": job_id,
            "filename": original_filename,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "message": "Karaoke track separated successfully!"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"error": "Demucs audio separation failed", "details": e.stderr})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


@app.get("/api/gallery")
async def list_gallery():
    """List completed karaoke projects and recordings."""
    gallery = []
    if os.path.exists(PROCESSED_FOLDER):
        for job_id in os.listdir(PROCESSED_FOLDER):
            job_dir = os.path.join(PROCESSED_FOLDER, job_id)
            karaoke_path = os.path.join(job_dir, "htdemucs", "input", "no_vocals.mp3")
            if os.path.exists(karaoke_path):
                meta_path = os.path.join(job_dir, "metadata.txt")
                song_name = "Untitled Track"
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        song_name = f.read().strip()

                mixed_path = os.path.join(job_dir, "final_song.mp3")
                ai_cover_path = os.path.join(job_dir, "ai_cover_song.mp3")
                gallery.append({
                    "job_id": job_id,
                    "song_name": song_name,
                    "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
                    "vocals_url": f"/api/download/{job_id}/vocals.mp3",
                    "has_mixed": os.path.exists(mixed_path),
                    "mixed_url": f"/api/download/{job_id}/final_song.mp3" if os.path.exists(mixed_path) else None,
                    "has_ai_cover": os.path.exists(ai_cover_path),
                    "ai_cover_url": f"/api/download/{job_id}/ai_cover_song.mp3" if os.path.exists(ai_cover_path) else None,
                    "ai_vocals_url": f"/api/download/{job_id}/ai_vocals.mp3" if os.path.exists(os.path.join(job_dir, "ai_vocals.mp3")) else None
                })

    return {"gallery": gallery}


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download audio file assets."""
    allowed_files = [
        "no_vocals.mp3",
        "vocals.mp3",
        "user_recording.webm",
        "final_song.mp3",
        "ai_cover_song.mp3",
        "ai_vocals.mp3"
    ]
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid file request")

    file_path = os.path.join(PROCESSED_FOLDER, job_id, "htdemucs", "input", filename)
    if not os.path.exists(file_path):
        file_path = os.path.join(PROCESSED_FOLDER, job_id, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@app.post("/api/mix")
async def mix_audio(
    job_id: str = Form(...),
    vocal_volume: str = Form("1.0"),
    music_volume: str = Form("1.0"),
    user_voice: UploadFile = File(...),
):
    """Mix recorded user voice with separated backing track."""
    job_dir = os.path.join(PROCESSED_FOLDER, job_id)
    karaoke_path = os.path.join(job_dir, "htdemucs", "input", "no_vocals.mp3")

    if not os.path.exists(karaoke_path):
        raise HTTPException(status_code=404, detail="Karaoke track not found for job")

    voice_path = os.path.join(job_dir, "user_recording.webm")
    voice_content = await user_voice.read()
    with open(voice_path, "wb") as f:
        f.write(voice_content)

    output_mixed_path = os.path.join(job_dir, "final_song.mp3")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", karaoke_path,
        "-i", voice_path,
        "-filter_complex",
        f"[0:a]volume={music_volume}[a0];[1:a]volume={vocal_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.97[out]",
        "-map", "[out]",
        "-b:a", "192k",
        output_mixed_path
    ]

    try:
        await asyncio.to_thread(subprocess.run, ffmpeg_cmd, capture_output=True, text=True, check=True)
        return {
            "success": True,
            "mixed_url": f"/api/download/{job_id}/final_song.mp3",
            "message": "Recorded voice & karaoke mixed successfully!"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"error": "Audio mixing failed", "details": e.stderr})


@app.post("/api/ai-cover")
async def generate_ai_cover(
    job_id: Optional[str] = Form(None),
    song_filename: Optional[str] = Form(None),
    voice_filename: Optional[str] = Form(None),
    pitch_shift: int = Form(0),
    diffusion_steps: int = Form(10),
    vocal_volume: str = Form("1.3"),
    music_volume: str = Form("1.0"),
    use_colab: Optional[str] = Form(None),
    user_voice: Optional[UploadFile] = File(None),
):
    """Zero-Shot Singing Voice Conversion using Seed-VC on Google Colab GPU or local CPU."""
    should_use_colab = (
        (use_colab.lower() == "true")
        if use_colab is not None
        else (internal_colab_client.status == "connected")
    )

    if not job_id and song_filename:
        safe_song = os.path.basename(song_filename)
        song_path = os.path.join(SONGS_FOLDER, safe_song)
        if not os.path.exists(song_path):
            raise HTTPException(status_code=404, detail=f"Song '{safe_song}' not found in .ignored/songs")
        job_id = find_existing_job_for_song(safe_song)
        if not job_id:
            try:
                job_id = await separate_song_file(song_path, safe_song, use_colab=should_use_colab)
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail={"error": "Demucs separation failed", "details": e.stderr})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Separation failed: {e}")

    if not job_id:
        raise HTTPException(status_code=400, detail="Missing job_id or song_filename")

    job_dir = os.path.join(PROCESSED_FOLDER, job_id)
    separated_dir = os.path.join(job_dir, "htdemucs", "input")
    karaoke_path = os.path.join(separated_dir, "no_vocals.mp3")
    vocals_path = os.path.join(separated_dir, "vocals.mp3")

    if not os.path.exists(karaoke_path) or not os.path.exists(vocals_path):
        raise HTTPException(status_code=404, detail="Separated vocals or instrumental track not found for this song")

    ref_wav_path = os.path.join(job_dir, "user_voice_ref.wav")

    # Determine voice source
    if user_voice and user_voice.filename:
        raw_ext = os.path.splitext(user_voice.filename)[1] or ".webm"
        raw_voice_path = os.path.join(job_dir, f"user_voice_raw{raw_ext}")
        content = await user_voice.read()
        with open(raw_voice_path, "wb") as f:
            f.write(content)
    else:
        if not voice_filename and os.path.exists(VOICE_FOLDER):
            available = [
                f for f in sorted(os.listdir(VOICE_FOLDER))
                if os.path.isfile(os.path.join(VOICE_FOLDER, f)) and f.lower().endswith(AUDIO_EXTENSIONS)
            ]
            if available:
                voice_filename = available[0]

        if not voice_filename:
            raise HTTPException(status_code=400, detail="No voice sample provided or found in .ignored/vsp_voice")

        raw_voice_path = os.path.join(VOICE_FOLDER, os.path.basename(voice_filename))
        if not os.path.exists(raw_voice_path):
            raise HTTPException(status_code=404, detail=f"Voice file '{voice_filename}' not found in .ignored/vsp_voice")

    try:
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", raw_voice_path,
                "-vn",
                "-t", "25",
                "-ac", "1",
                "-ar", "44100",
                ref_wav_path
            ],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"error": "Failed to convert voice sample to WAV", "details": e.stderr})

    ai_vocals_mp3_path = os.path.join(job_dir, "ai_vocals.mp3")
    ai_cover_mp3_path = os.path.join(job_dir, "ai_cover_song.mp3")
    used_engine = "Local CPU"

    if should_use_colab and internal_colab_client.status == "connected":
        try:
            await internal_colab_client.run_seed_vc_on_colab(
                vocals_path=vocals_path,
                target_voice_path=ref_wav_path,
                out_ai_vocals_mp3_path=ai_vocals_mp3_path,
                diffusion_steps=max(diffusion_steps, 25),
                pitch_shift=pitch_shift,
                f0_condition=True,
                auto_f0_adjust=True,
            )
            used_engine = f"Google Colab GPU ({internal_colab_client.gpu_info or 'CUDA'})"
            vocals_to_mix = ai_vocals_mp3_path
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Google Colab GPU conversion failed: {e}")
    else:
        try:
            ensure_seed_vc_ready()
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail={"error": "Failed to initialize Seed-VC repository", "details": e.stderr})

        vc_out_dir = os.path.join(job_dir, "seed_vc_out")
        if os.path.exists(vc_out_dir):
            shutil.rmtree(vc_out_dir)
        os.makedirs(vc_out_dir, exist_ok=True)

        python_bin = os.path.abspath("./venv/bin/python") if os.path.exists("./venv/bin/python") else "python3"
        seed_vc_cmd = [
            python_bin,
            "inference.py",
            "--source", vocals_path,
            "--target", ref_wav_path,
            "--output", vc_out_dir,
            "--diffusion-steps", str(diffusion_steps),
            "--f0-condition", "True",
            "--auto-f0-adjust", "True",
            "--semi-tone-shift", str(pitch_shift),
            "--fp16", "False"
        ]

        try:
            await asyncio.to_thread(subprocess.run, seed_vc_cmd, cwd=SEED_VC_DIR, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail={"error": "Seed-VC AI voice conversion failed", "details": e.stderr or e.stdout})

        generated_wavs = [
            os.path.join(vc_out_dir, f)
            for f in os.listdir(vc_out_dir)
            if f.endswith(".wav")
        ]
        if not generated_wavs:
            raise HTTPException(status_code=500, detail="Seed-VC did not produce a converted vocal track")

        converted_wav_path = generated_wavs[0]
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", converted_wav_path,
                "-b:a", "192k",
                ai_vocals_mp3_path
            ],
            capture_output=True, text=True, check=True
        )
        vocals_to_mix = converted_wav_path

    try:
        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg", "-y",
                "-i", karaoke_path,
                "-i", vocals_to_mix,
                "-filter_complex",
                f"[0:a]volume={music_volume}[a0];[1:a]volume={vocal_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.97[out]",
                "-map", "[out]",
                "-b:a", "192k",
                ai_cover_mp3_path
            ],
            capture_output=True, text=True, check=True
        )

        return {
            "success": True,
            "job_id": job_id,
            "engine": used_engine,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "ai_cover_url": f"/api/download/{job_id}/ai_cover_song.mp3",
            "ai_vocals_url": f"/api/download/{job_id}/ai_vocals.mp3",
            "message": f"AI Voice Clone Song Cover generated via {used_engine}!"
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail={"error": "Failed to mix AI vocals with instrumental", "details": e.stderr})


# --- Static UI Delivery ---

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_FOLDER, "index.html"))

# Mount static directory for any static assets
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
