import logging
import os
import shutil
import subprocess
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from colab_bridge import internal_colab_client, list_chrome_accounts


class _SuppressColabStatusPoll(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /api/colab/status" not in record.getMessage()


logging.getLogger("werkzeug").addFilter(_SuppressColabStatusPoll())

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_IGNORED_DIR = os.path.abspath(".ignored")
UPLOAD_FOLDER = os.path.join(BASE_IGNORED_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_IGNORED_DIR, "processed")
SONGS_FOLDER = os.path.join(BASE_IGNORED_DIR, "songs")
VOICE_FOLDER = os.path.join(BASE_IGNORED_DIR, "vsp_voice")
SEED_VC_DIR = os.path.join(BASE_IGNORED_DIR, "seed-vc")

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
            if updated != code:
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(updated)

    return inference_script


def find_existing_job_for_song(song_name):
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


def separate_song_file(source_file_path, original_filename, use_colab=None):
    """Convert any audio/mpeg file to MP3 and run Demucs stem separation (on Colab GPU if connected, else local CPU)."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp3")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", source_file_path,
        "-vn",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        input_path
    ], capture_output=True, text=True, check=True)

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
        internal_colab_client.run_sync(
            internal_colab_client.run_demucs_on_colab(input_path, vocals_path, karaoke_path)
        )
    else:
        cmd = [
            os.path.abspath("./venv/bin/demucs"),
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "--mp3",
            "-o", out_dir,
            input_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    if not os.path.exists(karaoke_path):
        raise RuntimeError("Failed to generate karaoke audio track")

    return job_id


@app.route("/api/colab/status", methods=["GET"])
def colab_status():
    """Returns the internal Google Colab MCP daemon status."""
    return jsonify({
        "status": internal_colab_client.status,
        "message": internal_colab_client.status_message,
        "gpu_info": internal_colab_client.gpu_info,
        "profileDirectory": internal_colab_client.selected_profile,
        "tools": list(internal_colab_client.tools.keys()),
        "connected": internal_colab_client.status == "connected",
        "colabAvailable": internal_colab_client.status == "connected",
        "recent_logs": list(internal_colab_client.logs)[-25:],
    })


@app.route("/api/colab/accounts", methods=["GET"])
def colab_accounts():
    """List Chrome profiles / Gmail accounts available on this machine."""
    payload = list_chrome_accounts()
    return jsonify(payload)


@app.route("/api/colab/connect", methods=["POST"])
def colab_connect():
    """Triggers pairing with Google Colab tab in the selected Chrome profile."""
    data = request.get_json(silent=True) or {}
    profile = request.args.get("profileDirectory") or data.get("profileDirectory")
    force = str(request.args.get("force") or data.get("force", "false")).lower() == "true"
    try:
        res = internal_colab_client.run_sync(
            internal_colab_client.connect_browser(profile_directory=profile, force=force),
            timeout=60.0,
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "connected": False, "message": str(e)}), 500


@app.route("/api/colab/execute", methods=["POST"])
def colab_execute():
    """Executes Python code in Google Colab Cloud GPU."""
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    keep_cell = bool(data.get("keep_cell", True))
    if not code.strip():
        return jsonify({"error": "Code cannot be empty"}), 400
    try:
        res = internal_colab_client.run_sync(
            internal_colab_client.execute_code(code, keep_cell=keep_cell)
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")



@app.route("/api/library", methods=["GET"])
def list_library():
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

    return jsonify({"songs": songs, "voices": voices})


@app.route("/api/voices/<path:filename>")
def download_voice_sample(filename):
    safe_name = os.path.basename(filename)
    target = os.path.join(VOICE_FOLDER, safe_name)
    if not os.path.exists(target):
        return jsonify({"error": "Voice file not found"}), 404

    if not safe_name.lower().endswith((".mp3", ".wav")):
        preview_dir = os.path.join(BASE_IGNORED_DIR, "voice_previews")
        os.makedirs(preview_dir, exist_ok=True)
        preview_name = f"{os.path.splitext(safe_name)[0]}.mp3"
        preview_path = os.path.join(preview_dir, preview_name)
        if not os.path.exists(preview_path):
            subprocess.run([
                "ffmpeg", "-y", "-i", target, "-vn", "-ar", "44100", "-ac", "1", "-b:a", "192k", preview_path
            ], capture_output=True, check=False)
        if os.path.exists(preview_path):
            return send_from_directory(preview_dir, preview_name)

    return send_from_directory(VOICE_FOLDER, safe_name)


@app.route("/api/process-local-song", methods=["POST"])
def process_local_song():
    """Extract instrumental & vocals for a song stored in .ignored/songs/."""
    data = request.get_json(silent=True) or request.form
    song_filename = data.get("song_filename")
    if not song_filename:
        return jsonify({"error": "Missing song_filename"}), 400

    safe_name = os.path.basename(song_filename)
    song_path = os.path.join(SONGS_FOLDER, safe_name)
    if not os.path.exists(song_path):
        return jsonify({"error": f"Song '{safe_name}' not found in .ignored/songs"}), 404

    existing_job_id = find_existing_job_for_song(safe_name)
    if existing_job_id:
        job_dir = os.path.join(PROCESSED_FOLDER, existing_job_id)
        return jsonify({
            "job_id": existing_job_id,
            "filename": safe_name,
            "karaoke_url": f"/api/download/{existing_job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{existing_job_id}/vocals.mp3",
            "ai_cover_url": f"/api/download/{existing_job_id}/ai_cover_song.mp3" if os.path.exists(os.path.join(job_dir, "ai_cover_song.mp3")) else None,
            "ai_vocals_url": f"/api/download/{existing_job_id}/ai_vocals.mp3" if os.path.exists(os.path.join(job_dir, "ai_vocals.mp3")) else None,
            "message": "Loaded previously separated tracks!"
        })

    try:
        job_id = separate_song_file(song_path, safe_name)
        return jsonify({
            "job_id": job_id,
            "filename": safe_name,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "message": "Karaoke track separated successfully!"
        })
    except subprocess.CalledProcessError as e:
        print("Demucs/FFmpeg Error Output:", e.stderr)
        return jsonify({"error": "Audio separation failed", "details": e.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/process", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    original_filename = file.filename
    raw_ext = os.path.splitext(original_filename)[1] or ".mp3"
    temp_dir = os.path.join(UPLOAD_FOLDER, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    raw_path = os.path.join(temp_dir, f"{uuid.uuid4()}{raw_ext}")
    file.save(raw_path)

    try:
        job_id = separate_song_file(raw_path, original_filename)
        return jsonify({
            "job_id": job_id,
            "filename": original_filename,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "message": "Karaoke track separated successfully!"
        })
    except subprocess.CalledProcessError as e:
        print("Demucs Error Output:", e.stderr)
        return jsonify({"error": "Demucs audio separation failed", "details": e.stderr}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


@app.route("/api/gallery", methods=["GET"])
def list_gallery():
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

    return jsonify({"gallery": gallery})


@app.route("/api/download/<job_id>/<filename>")
def download_file(job_id, filename):
    allowed_files = [
        "no_vocals.mp3",
        "vocals.mp3",
        "user_recording.webm",
        "final_song.mp3",
        "ai_cover_song.mp3",
        "ai_vocals.mp3"
    ]
    if filename not in allowed_files:
        return jsonify({"error": "Invalid file request"}), 400
    
    file_path = os.path.join(PROCESSED_FOLDER, job_id, "htdemucs", "input")
    if not os.path.exists(os.path.join(file_path, filename)):
        # Check root job processed dir for mixed/AI outputs
        file_path = os.path.join(PROCESSED_FOLDER, job_id)

    return send_from_directory(file_path, filename, as_attachment=True)


@app.route("/api/mix", methods=["POST"])
def mix_audio():
    job_id = request.form.get("job_id")
    vocal_volume = request.form.get("vocal_volume", "1.0")
    music_volume = request.form.get("music_volume", "1.0")

    if not job_id or "user_voice" not in request.files:
        return jsonify({"error": "Missing job_id or voice recording file"}), 400

    job_dir = os.path.join(PROCESSED_FOLDER, job_id)
    karaoke_path = os.path.join(job_dir, "htdemucs", "input", "no_vocals.mp3")

    if not os.path.exists(karaoke_path):
        return jsonify({"error": "Karaoke track not found for job"}), 404

    voice_file = request.files["user_voice"]
    voice_path = os.path.join(job_dir, "user_recording.webm")
    voice_file.save(voice_path)

    output_mixed_path = os.path.join(job_dir, "final_song.mp3")

    # Use ffmpeg to mix microphone recording with backing track
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", karaoke_path,
        "-i", voice_path,
        "-filter_complex",
        f"[0:a]volume={music_volume}[a0];[1:a]volume={vocal_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[out]",
        "-map", "[out]",
        "-b:a", "192k",
        output_mixed_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        return jsonify({
            "success": True,
            "mixed_url": f"/api/download/{job_id}/final_song.mp3",
            "message": "Recorded voice & karaoke mixed successfully!"
        })
    except subprocess.CalledProcessError as e:
        print("FFmpeg mixing error:", e.stderr)
        return jsonify({"error": "Audio mixing failed", "details": e.stderr}), 500


@app.route("/api/ai-cover", methods=["POST"])
def generate_ai_cover():
    """
    Zero-Shot Singing Voice Conversion using Seed-VC:
    Automatically runs on Google Colab GPU (~15s) when paired via Colab MCP,
    or falls back to local CPU inference.
    """
    job_id = request.form.get("job_id")
    song_filename = request.form.get("song_filename")
    voice_filename = request.form.get("voice_filename")
    pitch_shift = int(request.form.get("pitch_shift", "0"))
    diffusion_steps = int(request.form.get("diffusion_steps", "10"))
    vocal_volume = request.form.get("vocal_volume", "1.3")
    music_volume = request.form.get("music_volume", "1.0")
    use_colab_param = request.form.get("use_colab")

    should_use_colab = (
        (use_colab_param.lower() == "true")
        if use_colab_param is not None
        else (internal_colab_client.status == "connected")
    )

    # If song_filename from .ignored/songs was provided without job_id, extract or reuse existing job_id
    if not job_id and song_filename:
        safe_song = os.path.basename(song_filename)
        song_path = os.path.join(SONGS_FOLDER, safe_song)
        if not os.path.exists(song_path):
            return jsonify({"error": f"Song '{safe_song}' not found in .ignored/songs"}), 404
        job_id = find_existing_job_for_song(safe_song)
        if not job_id:
            try:
                job_id = separate_song_file(song_path, safe_song, use_colab=should_use_colab)
            except subprocess.CalledProcessError as e:
                return jsonify({"error": "Demucs separation failed", "details": e.stderr}), 500
            except Exception as e:
                return jsonify({"error": f"Separation failed: {e}"}), 500

    if not job_id:
        return jsonify({"error": "Missing job_id or song_filename"}), 400

    job_dir = os.path.join(PROCESSED_FOLDER, job_id)
    separated_dir = os.path.join(job_dir, "htdemucs", "input")
    karaoke_path = os.path.join(separated_dir, "no_vocals.mp3")
    vocals_path = os.path.join(separated_dir, "vocals.mp3")

    if not os.path.exists(karaoke_path) or not os.path.exists(vocals_path):
        return jsonify({"error": "Separated vocals or instrumental track not found for this song"}), 404

    ref_wav_path = os.path.join(job_dir, "user_voice_ref.wav")

    # Determine voice source: uploaded/recorded file OR saved file in .ignored/vsp_voice/
    if "user_voice" in request.files and request.files["user_voice"].filename != "":
        voice_file = request.files["user_voice"]
        raw_ext = os.path.splitext(voice_file.filename or "sample.webm")[1] or ".webm"
        raw_voice_path = os.path.join(job_dir, f"user_voice_raw{raw_ext}")
        voice_file.save(raw_voice_path)
    else:
        if not voice_filename and os.path.exists(VOICE_FOLDER):
            available = [
                f for f in sorted(os.listdir(VOICE_FOLDER))
                if os.path.isfile(os.path.join(VOICE_FOLDER, f)) and f.lower().endswith(AUDIO_EXTENSIONS)
            ]
            if available:
                voice_filename = available[0]

        if not voice_filename:
            return jsonify({"error": "No voice sample provided or found in .ignored/vsp_voice"}), 400

        raw_voice_path = os.path.join(VOICE_FOLDER, os.path.basename(voice_filename))
        if not os.path.exists(raw_voice_path):
            return jsonify({"error": f"Voice file '{voice_filename}' not found in .ignored/vsp_voice"}), 404

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw_voice_path,
            "-vn",
            "-t", "25",
            "-ac", "1",
            "-ar", "44100",
            ref_wav_path
        ], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Failed to convert voice sample to WAV", "details": e.stderr}), 500

    ai_vocals_mp3_path = os.path.join(job_dir, "ai_vocals.mp3")
    ai_cover_mp3_path = os.path.join(job_dir, "ai_cover_song.mp3")
    used_engine = "Local CPU"

    if should_use_colab and internal_colab_client.status == "connected":
        try:
            internal_colab_client.run_sync(
                internal_colab_client.run_seed_vc_on_colab(
                    vocals_path=vocals_path,
                    target_voice_path=ref_wav_path,
                    out_ai_vocals_mp3_path=ai_vocals_mp3_path,
                    diffusion_steps=max(diffusion_steps, 25),
                    pitch_shift=pitch_shift,
                    f0_condition=True,
                    auto_f0_adjust=True,
                )
            )
            used_engine = f"Google Colab GPU ({internal_colab_client.gpu_info or 'CUDA'})"
            vocals_to_mix = ai_vocals_mp3_path
        except Exception as e:
            print("Colab Seed-VC Error:", e)
            return jsonify({"error": f"Google Colab GPU conversion failed: {e}"}), 500
    else:
        try:
            ensure_seed_vc_ready()
        except subprocess.CalledProcessError as e:
            return jsonify({"error": "Failed to initialize Seed-VC repository", "details": e.stderr}), 500

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
            subprocess.run(seed_vc_cmd, cwd=SEED_VC_DIR, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Seed-VC Error Output:", e.stderr)
            return jsonify({
                "error": "Seed-VC AI voice conversion failed",
                "details": e.stderr or e.stdout
            }), 500

        generated_wavs = [
            os.path.join(vc_out_dir, f)
            for f in os.listdir(vc_out_dir)
            if f.endswith(".wav")
        ]
        if not generated_wavs:
            return jsonify({"error": "Seed-VC did not produce a converted vocal track"}), 500

        converted_wav_path = generated_wavs[0]
        subprocess.run([
            "ffmpeg", "-y",
            "-i", converted_wav_path,
            "-b:a", "192k",
            ai_vocals_mp3_path
        ], capture_output=True, text=True, check=True)
        vocals_to_mix = converted_wav_path

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", karaoke_path,
            "-i", vocals_to_mix,
            "-filter_complex",
            f"[0:a]volume={music_volume}[a0];[1:a]volume={vocal_volume}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[out]",
            "-map", "[out]",
            "-b:a", "192k",
            ai_cover_mp3_path
        ], capture_output=True, text=True, check=True)

        return jsonify({
            "success": True,
            "job_id": job_id,
            "engine": used_engine,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "ai_cover_url": f"/api/download/{job_id}/ai_cover_song.mp3",
            "ai_vocals_url": f"/api/download/{job_id}/ai_vocals.mp3",
            "message": f"AI Voice Clone Song Cover generated via {used_engine}!"
        })
    except subprocess.CalledProcessError as e:
        print("FFmpeg AI cover mixing error:", e.stderr)
        return jsonify({"error": "Failed to mix AI vocals with instrumental", "details": e.stderr}), 500

if __name__ == "__main__":
    try:
        ensure_seed_vc_ready()
    except Exception as e:
        print("Warning initializing Seed-VC:", e)
    port = int(os.environ.get("BACKEND_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
