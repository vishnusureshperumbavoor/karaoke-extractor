# 🎤 Karaoke Extractor & Singer Studio AI

A modern, full-stack open-source web application that extracts pure instrumental/karaoke tracks from any MP3 song using Deep Learning AI ([Demucs](https://github.com/facebookresearch/demucs)), lets you sing along using your device microphone, and automatically mixes your recorded voice with the backing track into a downloadable final song!

---

## 🌟 Key Features

- **🤖 AI Instrumental Separation**: Uses Meta Research's **Demucs (htdemucs)** deep learning model to separate vocal and instrumental stems offline for free.
- **🎙️ Live Karaoke Studio**: Sing along in real-time using your mobile phone or laptop microphone while the instrumental backing track plays out loud.
- **🎛️ Auto Audio Mixing**: Powered by **FFmpeg**, automatically blends your recorded voice with the backing track into a studio-grade combined MP3.
- **📚 Extracted Song Gallery**: Keeps a persistent local gallery of all your previously extracted songs for quick selection and re-singing.
- **📱 Focused Mobile UI**: Zero-distraction singer view with green **Start Recording** / red **Stop & Mix** single toggle button and an instant **Cancel** option.
- **🔒 100% Free & Local**: No paid APIs, no subscriptions, and no cloud data uploads required. All storage is kept locally inside a `.ignored/` folder.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, Flask, Flask-CORS
- **Audio Processing Engine**: PyTorch, Demucs (`htdemucs`), FFmpeg
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
git clone https://github.com/vishnusureshperumbavoor/karaoke-extractor.git
cd karaoke-extractor

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install flask flask-cors demucs torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. Running the Server

Start the Flask application server:

```bash
./venv/bin/python app.py
```

The server will start running on port `5000`:
- **Local machine access**: `http://localhost:5000`
- **Mobile Wi-Fi access**: `http://<YOUR_LOCAL_IP>:5000` (e.g. `http://192.168.29.106:5000`)

---

## 📱 Mobile Microphone Setup (Android Chrome)

To record your voice over local Wi-Fi (`http://192.168.29.106:5000`), enable insecure origins in Chrome:

1. Open Chrome on Android and go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.
2. Search for **"Insecure origins treated as secure"**.
3. Add your local URL: `http://192.168.29.106:5000`.
4. Change the dropdown to **Enabled** and tap **Relaunch**.

---

## 📁 Project Structure

```
karaoke-extractor/
├── app.py                 # Flask server with /api/process and /api/mix endpoints
├── static/
│   └── index.html         # Single-page web UI & Singer Studio
├── .ignored/              # Local untracked folder storing MP3 uploads and outputs
├── .gitignore             # Configured to exclude venv/, .ignored/, and logs
└── README.md              # Project documentation
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
