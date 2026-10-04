import os
import shutil
import subprocess
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

BASE_IGNORED_DIR = os.path.abspath(".ignored")
UPLOAD_FOLDER = os.path.join(BASE_IGNORED_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_IGNORED_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/process", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(UPLOAD_FOLDER, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input.mp3")
    file.save(input_path)

    out_dir = os.path.join(PROCESSED_FOLDER, job_id)
    os.makedirs(out_dir, exist_ok=True)

    # Run Demucs stem separation (htdemucs model isolates vocals and accompaniment)
    cmd = [
        os.path.abspath("./venv/bin/demucs"),
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "--mp3",
        "-o", out_dir,
        input_path
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Demucs output path structure: out_dir/htdemucs/input/no_vocals.mp3 and vocals.mp3
        separated_dir = os.path.join(out_dir, "htdemucs", "input")
        karaoke_path = os.path.join(separated_dir, "no_vocals.mp3")
        vocals_path = os.path.join(separated_dir, "vocals.mp3")

        if not os.path.exists(karaoke_path):
            return jsonify({"error": "Failed to generate karaoke audio track"}), 500

        return jsonify({
            "job_id": job_id,
            "karaoke_url": f"/api/download/{job_id}/no_vocals.mp3",
            "vocals_url": f"/api/download/{job_id}/vocals.mp3",
            "message": "Karaoke track separated successfully!"
        })
    except subprocess.CalledProcessError as e:
        print("Demucs Error Output:", e.stderr)
        return jsonify({"error": "Demucs audio separation failed", "details": e.stderr}), 500

@app.route("/api/download/<job_id>/<filename>")
def download_file(job_id, filename):
    if filename not in ["no_vocals.mp3", "vocals.mp3"]:
        return jsonify({"error": "Invalid file request"}), 400
    
    file_path = os.path.join(PROCESSED_FOLDER, job_id, "htdemucs", "input")
    return send_from_directory(file_path, filename, as_attachment=(filename == "no_vocals.mp3"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
