#!/usr/bin/env bash
# Supreme Diffusion — bootstrap, update, and launch
# On first run: creates venv, installs PyTorch + dependencies, sets up factory dirs.
# On every run: checks for updates, then launches the app.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check for Python ─────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required but was not found. Install Python 3.10-3.12 and re-run."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "Supreme Diffusion needs Python 3.10-3.12 (found $(python3 --version 2>&1))."
    exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)'; then
    echo "WARNING: Python 3.13+ detected. Supreme Diffusion is tested on Python 3.10-3.12."
    echo "Some dependencies may not have prebuilt wheels for your Python version yet."
fi

# ── Detect CUDA version from the NVIDIA driver ─────────────────────────
# Maps the driver's max supported CUDA version to the closest PyTorch
# wheel index.  Forward-compatible: a cu126 wheel runs fine on a driver
# that reports CUDA 12.8.  But newer GPUs (Blackwell / RTX 5090) may
# need cu128+ for SM_100 compute-capability support.
detect_cuda_tag() {
    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    if [ -z "$cuda_ver" ]; then
        echo "cu126"  # sane default if nvidia-smi isn't available
        return
    fi
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    if [ "$major" -ge 13 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; }; then
        echo "cu128"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then
        echo "cu126"
    elif [ "$major" -eq 12 ]; then
        echo "cu124"
    elif [ "$major" -eq 11 ]; then
        echo "cu118"
    else
        echo "cu126"
    fi
}

CUDA_TAG=$(detect_cuda_tag)
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

# ── Check for FFmpeg ─────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo ""
    echo "FFmpeg is required but not installed."
    if command -v apt &>/dev/null; then
        read -p "Install FFmpeg via apt? [Y/n] " yn
        yn=${yn:-Y}
        if [[ "$yn" =~ ^[Yy] ]]; then
            sudo apt install -y ffmpeg
        else
            echo "Please install FFmpeg manually and re-run."
            exit 1
        fi
    elif command -v brew &>/dev/null; then
        read -p "Install FFmpeg via Homebrew? [Y/n] " yn
        yn=${yn:-Y}
        if [[ "$yn" =~ ^[Yy] ]]; then
            brew install ffmpeg
        else
            echo "Please install FFmpeg manually and re-run."
            exit 1
        fi
    else
        echo "Please install FFmpeg manually (https://ffmpeg.org/download.html) and re-run."
        exit 1
    fi
fi

# ── Create venv if it doesn't exist ──────────────────────────────────────
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
    echo "================================================"
    echo "  Supreme Diffusion — First-Time Setup"
    echo "================================================"
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
fi

# ── Activate venv ────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ── Install PyTorch if not present ───────────────────────────────────────
if ! python -c "import torch" 2>/dev/null; then
    echo ""
    echo "Detected CUDA tag: ${CUDA_TAG}"
    echo "Installing PyTorch with ${CUDA_TAG} support..."
    echo ""
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# ── Install dependencies if needed ───────────────────────────────────────
STAMP=".venv/.deps_installed"
if [ ! -f "$STAMP" ]; then
    echo ""
    echo "Installing core dependencies..."
    if ! pip install -e . --extra-index-url "$TORCH_INDEX"; then
        echo ""
        echo "First install attempt failed — clearing pip cache and retrying..."
        pip cache purge 2>/dev/null || true
        if ! pip install --no-cache-dir -e . --extra-index-url "$TORCH_INDEX"; then
            echo ""
            echo "Dependency installation failed — see the errors above."
            echo "Re-run ./start.sh to retry."
            exit 1
        fi
    fi
    echo ""

    # ── Optional features ────────────────────────────────────────────
    echo "════════════════════════════════════════════════════════"
    echo "  Optional Features"
    echo ""
    echo "  The following are optional and can be skipped."
    echo "  You can install them later by deleting .venv/.deps_installed"
    echo "  and re-running this script."
    echo "════════════════════════════════════════════════════════"
    echo ""
    read -p "Install optional components? [Y/n] " INSTALL_OPTIONAL
    INSTALL_OPTIONAL=${INSTALL_OPTIONAL:-Y}

    if [[ "$INSTALL_OPTIONAL" =~ ^[Yy] ]]; then
        echo ""

        # Face Swap (insightface only — model weights loaded via spandrel at runtime)
        read -p "  Face Swap (insightface + onnxruntime-gpu)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && {
            pip install insightface onnxruntime-gpu 2>/dev/null || echo "    (insightface install failed — face swap may not work)"
        } || true

        # 3D Modeling
        read -p "  3D Modeling (PyOpenGL, trimesh, TripoSR, FBX import)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && {
            pip install PyOpenGL trimesh xatlas pyassimp 2>/dev/null || echo "    (some 3D deps failed)"
            pip install tsr 2>/dev/null || echo "    (TripoSR install failed — mesh generation may not work)"
        } || true

        # LoRA Training (kohya sd-scripts)
        read -p "  LoRA Training (kohya sd-scripts)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && {
            pip install toml prodigy-optimizer lion-pytorch bitsandbytes dadaptation 2>/dev/null || echo "    (some training optimizer deps failed)"
            KOHYA_DIR="$SCRIPT_DIR/third_party/sd-scripts"
            if [ ! -d "$KOHYA_DIR" ]; then
                echo "    Cloning kohya sd-scripts..."
                git clone https://github.com/kohya-ss/sd-scripts.git "$KOHYA_DIR" --depth 1 2>/dev/null || echo "    (kohya clone failed — LoRA training won't work)"
                if [ -f "$KOHYA_DIR/requirements.txt" ]; then
                    pip install -r "$KOHYA_DIR/requirements.txt" 2>/dev/null || echo "    (some kohya deps failed)"
                fi
            fi
        } || true

        # Video LoRA Training (ai-toolkit)
        read -p "  Video LoRA Training (ai-toolkit)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && {
            AITK_DIR="$SCRIPT_DIR/third_party/ai-toolkit"
            if [ ! -d "$AITK_DIR" ]; then
                echo "    Cloning ai-toolkit..."
                git clone https://github.com/ostris/ai-toolkit.git "$AITK_DIR" --depth 1 2>/dev/null || echo "    (ai-toolkit clone failed)"
                if [ -f "$AITK_DIR/requirements.txt" ]; then
                    pip install -r "$AITK_DIR/requirements.txt" 2>/dev/null || echo "    (some ai-toolkit deps failed)"
                fi
            fi
        } || true

        # Audio — Chatterbox TTS
        read -p "  Chatterbox TTS (voice cloning with emotion)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && pip install chatterbox-tts 2>/dev/null || true

        # Audio — Dia TTS
        read -p "  Dia TTS (multi-speaker dialogue)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && pip install diatts 2>/dev/null || true

        # Audio — AudioCraft (MusicGen + AudioGen)
        read -p "  AudioCraft — MusicGen + AudioGen (text-to-music/sound)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && pip install audiocraft 2>/dev/null || true

        # Audio — Orpheus TTS
        read -p "  Orpheus TTS (text-to-speech)? [Y/n] " yn
        yn=${yn:-Y}
        [[ "$yn" =~ ^[Yy] ]] && pip install snac soundfile 2>/dev/null || true

        echo ""
    else
        echo ""
        echo "Skipping optional components."
        echo ""
    fi

    touch "$STAMP"
    echo "Setup complete."
    echo ""
fi

# ── Check for updates ───────────────────────────────────────────────────
if [ -d ".git" ]; then
    echo "Checking for updates..."
    git fetch origin main --quiet 2>/dev/null || true
    LOCAL=$(git rev-parse HEAD 2>/dev/null)
    REMOTE=$(git rev-parse origin/main 2>/dev/null)
    if [ "$LOCAL" != "$REMOTE" ] && [ -n "$REMOTE" ]; then
        echo "Update available — pulling latest changes..."
        git pull --ff-only origin main 2>/dev/null || {
            echo "Auto-update failed (local changes?). Run 'git pull' manually."
        }
        # Re-install in case dependencies changed
        pip install -e . --extra-index-url "$TORCH_INDEX" --quiet 2>/dev/null || true
    else
        echo "Up to date."
    fi
fi

# ── Create factory directories ───────────────────────────────────────────
mkdir -p models/{checkpoints,refiners,vae,flux,birefnet,zimage,face,rife,upscalers,mimicmotion}
mkdir -p models/{qwen3.5_4b,qwen_vl_7b,orpheus_tts,whisper}
mkdir -p models/loras/{wan,sd,flux,zimage,ltx}
mkdir -p projects
mkdir -p library/{png,faces,voices,sounds,characters,meshes}

# ── Launch ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Supreme Diffusion (CUDA: ${CUDA_TAG})..."
python run.py "$@"
