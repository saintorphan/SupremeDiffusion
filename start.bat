@echo off
REM Supreme Diffusion - bootstrap, update, and launch
REM On first run: creates venv, installs PyTorch + dependencies, sets up factory dirs.
REM On every run: checks for updates, then launches the app.
cd /d "%~dp0"

REM ── Check for Python ─────────────────────────────────────────────────
REM On a fresh Windows install, "python" may be the Microsoft Store stub,
REM which fails with a nonzero exit code for "python --version".
python --version >nul 2>nul
if errorlevel 1 goto :no_python

python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>nul
if errorlevel 1 goto :old_python

python -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo WARNING: Python 3.13+ detected. Supreme Diffusion is tested on Python 3.10-3.12.
    echo Some dependencies may not have prebuilt wheels for your Python version yet.
    echo.
)

REM ── Detect CUDA version from the NVIDIA driver ─────────────────────────
REM Uses a Python one-liner to parse nvidia-smi output - more reliable than
REM batch string parsing. Maps the driver's max CUDA version to the closest
REM PyTorch wheel index. Blackwell / RTX 5090 needs cu128+ for SM_100.
set CUDA_TAG=cu126
for /f %%T in ('python -c "import subprocess,re;m=re.search(r'CUDA Version: (\d+)\.(\d+)',subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout or '');print('cu128' if m and (int(m[1])>12 or (int(m[1])==12 and int(m[2])>=8)) else 'cu126' if m and int(m[1])==12 and int(m[2])>=6 else 'cu124' if m and int(m[1])==12 else 'cu118' if m and int(m[1])==11 else 'cu126')" 2^>nul') do set CUDA_TAG=%%T

set TORCH_INDEX=https://download.pytorch.org/whl/%CUDA_TAG%

REM ── Check for FFmpeg ─────────────────────────────────────────────────
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo FFmpeg is required but not installed.
    echo Please install FFmpeg from https://ffmpeg.org/download.html
    echo and add it to your PATH, then re-run this script.
    echo.
    pause
    exit /b 1
)

REM ── Create venv if it doesn't exist ──────────────────────────────────
if exist ".venv\Scripts\activate.bat" goto :venv_ready
if exist "venv\Scripts\activate.bat" goto :venv_ready
echo ================================================
echo   Supreme Diffusion - First-Time Setup
echo ================================================
echo.
echo Creating Python virtual environment...
python -m venv .venv
if not exist ".venv\Scripts\activate.bat" goto :venv_failed
echo Virtual environment created.

:venv_ready
REM ── Activate venv ────────────────────────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    call venv\Scripts\activate.bat
)

REM ── Install PyTorch if not present ───────────────────────────────────
python -c "import torch" >nul 2>nul
if not errorlevel 1 goto :torch_ready
echo.
echo Detected CUDA tag: %CUDA_TAG%
echo Installing PyTorch with %CUDA_TAG% support...
echo.
pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 goto :install_failed

:torch_ready
REM ── Install dependencies if needed ───────────────────────────────────
if exist ".venv\.deps_installed" goto :deps_done
echo.
echo Installing core dependencies...
pip install -e . --extra-index-url %TORCH_INDEX%
if not errorlevel 1 goto :core_ok
echo.
echo First install attempt failed - clearing pip cache and retrying...
pip cache purge >nul 2>nul
pip install --no-cache-dir -e . --extra-index-url %TORCH_INDEX%
if errorlevel 1 goto :install_failed

:core_ok
echo.
REM ── Optional features ───────────────────────────────────────────────
echo ========================================================
echo   Optional Features
echo.
echo   The following are optional and can be skipped.
echo   You can install them later by deleting .venv\.deps_installed
echo   and re-running this script.
echo ========================================================
echo.
set "INSTALL_OPT="
set /p INSTALL_OPT="Install optional components? [Y/n] "
if "%INSTALL_OPT%"=="" set INSTALL_OPT=Y
if /i not "%INSTALL_OPT%"=="Y" goto :skip_optional

echo.

REM Face Swap (insightface only - model weights loaded via spandrel at runtime)
set "YN="
set /p YN="  Face Swap (insightface + onnxruntime-gpu)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" pip install insightface onnxruntime-gpu

REM 3D Modeling
set "YN="
set /p YN="  3D Modeling (PyOpenGL, trimesh, TripoSR, FBX import)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" (
    pip install PyOpenGL trimesh xatlas pyassimp
    pip install tsr
)

REM LoRA Training (kohya sd-scripts)
set "YN="
set /p YN="  LoRA Training (kohya sd-scripts)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" (
    pip install toml prodigy-optimizer lion-pytorch bitsandbytes dadaptation
    if not exist "third_party\sd-scripts" (
        echo     Cloning kohya sd-scripts...
        git clone https://github.com/kohya-ss/sd-scripts.git third_party\sd-scripts --depth 1
        if exist "third_party\sd-scripts\requirements.txt" (
            pip install -r third_party\sd-scripts\requirements.txt
        )
    )
)

REM Video LoRA Training (ai-toolkit)
set "YN="
set /p YN="  Video LoRA Training (ai-toolkit)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" (
    if not exist "third_party\ai-toolkit" (
        echo     Cloning ai-toolkit...
        git clone https://github.com/ostris/ai-toolkit.git third_party\ai-toolkit --depth 1
        if exist "third_party\ai-toolkit\requirements.txt" (
            pip install -r third_party\ai-toolkit\requirements.txt
        )
    )
)

REM Audio - Chatterbox TTS
set "YN="
set /p YN="  Chatterbox TTS (voice cloning with emotion)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" pip install chatterbox-tts

REM Audio - Dia TTS
set "YN="
set /p YN="  Dia TTS (multi-speaker dialogue)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" pip install diatts

REM Audio - AudioCraft (MusicGen + AudioGen)
set "YN="
set /p YN="  AudioCraft - MusicGen + AudioGen (text-to-music/sound)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" pip install audiocraft

REM Audio - Orpheus TTS
set "YN="
set /p YN="  Orpheus TTS (text-to-speech)? [Y/n] "
if "%YN%"=="" set YN=Y
if /i "%YN%"=="Y" pip install snac soundfile

echo.
goto :done_optional

:skip_optional
echo.
echo Skipping optional components.
echo.

:done_optional
echo. > .venv\.deps_installed
echo Setup complete.
echo.

:deps_done
REM ── Check for updates ───────────────────────────────────────────────
if not exist ".git" goto :update_done
echo Checking for updates...
git fetch origin main --quiet 2>nul
set "LOCAL="
set "REMOTE="
for /f %%A in ('git rev-parse HEAD 2^>nul') do set LOCAL=%%A
for /f %%B in ('git rev-parse origin/main 2^>nul') do set REMOTE=%%B
if "%REMOTE%"=="" goto :update_done
if "%LOCAL%"=="%REMOTE%" (
    echo Up to date.
    goto :update_done
)
echo Update available - pulling latest changes...
git pull --ff-only origin main
if errorlevel 1 (
    echo Auto-update failed. Run 'git pull' manually.
) else (
    pip install -e . --extra-index-url %TORCH_INDEX% --quiet >nul 2>nul
)

:update_done
REM ── Create factory directories ──────────────────────────────────────
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\refiners" mkdir models\refiners
if not exist "models\vae" mkdir models\vae
if not exist "models\flux" mkdir models\flux
if not exist "models\birefnet" mkdir models\birefnet
if not exist "models\zimage" mkdir models\zimage
if not exist "models\face" mkdir models\face
if not exist "models\rife" mkdir models\rife
if not exist "models\upscalers" mkdir models\upscalers
if not exist "models\mimicmotion" mkdir models\mimicmotion
if not exist "models\qwen3.5_4b" mkdir models\qwen3.5_4b
if not exist "models\qwen_vl_7b" mkdir models\qwen_vl_7b
if not exist "models\orpheus_tts" mkdir models\orpheus_tts
if not exist "models\whisper" mkdir models\whisper
if not exist "models\loras\wan" mkdir models\loras\wan
if not exist "models\loras\sd" mkdir models\loras\sd
if not exist "models\loras\flux" mkdir models\loras\flux
if not exist "models\loras\zimage" mkdir models\loras\zimage
if not exist "models\loras\ltx" mkdir models\loras\ltx
if not exist "projects" mkdir projects
if not exist "library\png" mkdir library\png
if not exist "library\faces" mkdir library\faces
if not exist "library\voices" mkdir library\voices
if not exist "library\sounds" mkdir library\sounds
if not exist "library\characters" mkdir library\characters
if not exist "library\meshes" mkdir library\meshes

REM ── Launch ──────────────────────────────────────────────────────────
echo.
echo Starting Supreme Diffusion (CUDA: %CUDA_TAG%)...
python run.py %*
if errorlevel 1 (
    echo.
    echo Supreme Diffusion exited with an error - see messages above.
    pause
)
goto :eof

:no_python
echo.
echo Python was not found on your PATH.
echo Install Python 3.10-3.12 from https://www.python.org/downloads/
echo and make sure to check "Add python.exe to PATH" during setup,
echo then re-run this script.
echo.
pause
exit /b 1

:old_python
echo.
echo Your Python version is too old - Supreme Diffusion needs Python 3.10-3.12.
echo Install a newer Python from https://www.python.org/downloads/
echo and re-run this script.
echo.
pause
exit /b 1

:venv_failed
echo.
echo Failed to create the Python virtual environment.
echo Make sure Python 3.10-3.12 is installed correctly, then re-run this script.
echo.
pause
exit /b 1

:install_failed
echo.
echo Dependency installation failed - see the errors above.
echo Re-run this script to retry.
echo.
pause
exit /b 1
