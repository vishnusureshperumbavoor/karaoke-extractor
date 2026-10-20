# 🎤 Vibe Rapper — AI Vocal Stem Isolation & Voice Studio

A modern, full-stack open-source AI web studio that isolates pure instrumental beats and rap/vocal stems from any audio track using Deep Learning AI ([Demucs](https://github.com/facebookresearch/demucs)), lets you record and spit bars live over the beat, and generates instant zero-shot AI voice conversions and song covers powered by **`vspagent`** Google Colab MCP agent orchestration!

---

## 🌟 Key Features

- **🤖 AI Beat & Vocal Stem Separation**: Uses Meta Research's **Demucs (htdemucs)** deep learning model to isolate vocal and instrumental stems offline for free or on Colab GPU.
- **🗣️ Zero-Shot AI Voice Clone Covers (Seed-VC)**: Upload or record just **10–25 seconds of your voice**, and **Seed-VC** converts vocals into your custom voice (preserving rhythm, inflection, and pitch) and mixes it with the beat!
- **🎙️ Live Rap Recording Studio**: Record live vocals and spit bars in real-time over the instrumental beat while monitoring your backing track.
- **🎛️ Auto Audio Mixing**: Powered by **FFmpeg**, automatically blends your recorded or AI-cloned voice with the beat into a studio-grade combined MP3.
- **📚 Vibe Rapper Track Gallery**: Persistent local gallery of all your previously extracted beats and covers for instant selection and playback.
- **⚡ Colab MCP Cloud GPU**: Offload Demucs separation and Seed-VC conversion to Google Colab GPUs with zero local VRAM using **`vspagent`**.
- **📱 Focused Mobile UI**: Zero-distraction studio view with streamlined **Start Recording** / **Stop & Mix** controls.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, vspagent
- **Audio & AI Processing Engine**: PyTorch, Demucs (`htdemucs`), Seed-VC (`U-DiT` + `RMVPE` + `BigVGAN`), FFmpeg
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism & Outfit Font), Web Audio / MediaRecorder API
- **Version Control**: Git, GitHub

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have **Python 3.10+** and **FFmpeg** installed on your system:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install python3 python3-venv ffmpeg -y
```

### 2. Installation & Setup

Clone the repository and set up the Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/vishnusureshperumbavoor/vibe-singer.git
cd vibe-singer

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn python-multipart vspagent demucs torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install librosa scipy munch huggingface-hub transformers accelerate soundfile pyyaml
```

### 3. Running the Backend & Frontend

You can start the backend and frontend servers using the included shell scripts:

#### Start the Backend Server (Port `5000`)
```bash
./backend_start
```

#### Start the Frontend Server (Port `3000`)
Open a second terminal and run:
```bash
./frontend_start
```

Access the application at:
- **Frontend UI (Local)**: `http://localhost:3000` (or `http://<YOUR_LOCAL_IP>:3000` on mobile Wi-Fi)
- **Backend API / Direct Server**: `http://localhost:5000` (or `http://<YOUR_LOCAL_IP>:5000`)

#### ⚡ Optional: Connect Google Colab GPU (Free T4 / L4) for ~15s Conversion
Instead of waiting several minutes on a local CPU, you can pair the app with a Google Colab GPU session via **Google Colab MCP**:
- **From the Web UI**: Choose your Chrome profile at the top of the page and click **⚡ Connect Colab GPU**.
- **Or from the Terminal**:
  ```bash
  ./colab_start
  ```
Once connected (`🟢 Google Colab GPU Active`), both **Demucs** stem separation and **Seed-VC** AI singing voice conversion automatically run on the Colab GPU in **~10–15 seconds**!

---

## 📱 Mobile Microphone Setup (Android Chrome)

To record your voice over local Wi-Fi (`http://<YOUR_LOCAL_IP>:3000` or `http://<YOUR_LOCAL_IP>:5000`), enable insecure origins in Chrome:

1. Open Chrome on Android and go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. Search for **"Insecure origins treated as secure"**.
3. Add your local URL: `http://<YOUR_LOCAL_IP>:3000` (and/or `http://<YOUR_LOCAL_IP>:5000`).
4. Change the dropdown to **Enabled** and tap **Relaunch**.

---

## 📁 Project Structure

```
vibe-rapper/
├── app.py                 # FastAPI backend server (/api/library, /api/colab/*, /api/process, /api/ai-cover, /api/mix)
├── colab_bridge/          # Google Colab MCP bridge powered by vspagent
├── colab_start            # Interactive CLI script to pair a Chrome profile with Google Colab GPU
├── backend_start          # Shell script to launch the FastAPI backend server (port 5000)
├── frontend_start         # Shell script to launch the frontend static server (port 3000)
├── static/
│   └── index.html         # Single-page web UI, Colab GPU bar, Folder Library & AI Voice Clone Studio
├── .ignored/              # Local untracked folder storing songs, voice samples, and outputs
│   ├── songs/             # Drop your MP3 / MPEG songs here for 1-click AI voice cloning
│   ├── vsp_voice/         # Drop your reference voice clip (MP3 / MPEG / WAV) here
│   ├── processed/         # Separated stems and mixed AI song covers
│   └── seed-vc/           # Local Seed-VC model & checkpoints
├── .gitignore             # Configured to exclude venv/, .ignored/, and logs
└── README.md              # Project documentation
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
