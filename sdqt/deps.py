"""Optional dependency auto-installer — check, prompt, pip install."""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

@dataclass
class FeatureDeps:
    """Packages required for an optional feature."""
    label: str
    check_import: str              # module to test (e.g. "orpheus_speech")
    pip_packages: list[str]        # pip install args
    no_deps: list[str] = field(default_factory=list)  # packages needing --no-deps


# Central registry of optional features and their pip requirements.
# Keys match logical feature names used at call sites.
OPTIONAL_DEPS: dict[str, FeatureDeps] = {
    "orpheus_tts": FeatureDeps(
        label="Orpheus TTS",
        check_import="snac",
        pip_packages=["snac", "soundfile"],
    ),
    "openvoice": FeatureDeps(
        label="OpenVoice",
        check_import="openvoice",
        no_deps=[
            "git+https://github.com/myshell-ai/OpenVoice.git",
            "git+https://github.com/myshell-ai/MeloTTS.git",
        ],
        pip_packages=[
            "inflect", "unidecode", "pypinyin", "cn2an",
            "eng-to-ipa", "jieba", "whisper-timestamped",
            "mecab-python3", "unidic-lite", "num2words",
            "g2p_en", "anyascii", "jamo", "gruut",
            "pykakasi", "fugashi", "ipadic", "cached_path",
        ],
    ),
    "rvc": FeatureDeps(
        label="RVC Voice Conversion",
        check_import="rvc",
        no_deps=["rvc", "fairseq-fixed"],
        pip_packages=[
            "hydra-core>=1.3", "soundfile", "librosa",
            "praat-parselmouth", "pyworld", "torchcrepe",
            "faiss-cpu", "av",
        ],
    ),
    "qwen_vl": FeatureDeps(
        label="Qwen VL (Image Captioning)",
        check_import="qwen_vl_utils",
        pip_packages=["qwen-vl-utils", "bitsandbytes"],
    ),
    "transformers": FeatureDeps(
        label="Transformers (Prompt Enhancement / AI Assistant)",
        check_import="transformers",
        pip_packages=["transformers"],
    ),
    "controlnet_aux": FeatureDeps(
        label="ControlNet Preprocessors",
        check_import="controlnet_aux",
        pip_packages=["controlnet_aux"],
    ),
    "audiocraft": FeatureDeps(
        label="AudioCraft (MusicGen / AudioGen)",
        check_import="audiocraft",
        no_deps=["audiocraft"],
        pip_packages=[
            "julius", "lameenc", "flashy", "encodec", "einops",
            "torchmetrics", "demucs", "spacy", "num2words",
        ],
    ),
    "face_swap": FeatureDeps(
        label="Face Swap / Restoration",
        check_import="insightface",
        pip_packages=[
            "insightface", "onnxruntime-gpu", "gfpgan", "realesrgan",
        ],
    ),
    "adetailer_yolo": FeatureDeps(
        label="ADetailer YOLOv8 Detector",
        check_import="ultralytics",
        pip_packages=["ultralytics"],
    ),
}


def is_installed(feature: str) -> bool:
    """Check if a feature's key import is available."""
    deps = OPTIONAL_DEPS.get(feature)
    if deps is None:
        return True
    return importlib.util.find_spec(deps.check_import) is not None


def check_and_install(
    feature: str,
    parent: QWidget,
) -> bool:
    """Check if packages for *feature* are installed; prompt to install if not.

    Returns True if available (already present or successfully installed).
    Returns False if the user cancelled or install failed.
    """
    deps = OPTIONAL_DEPS.get(feature)
    if deps is None:
        return True

    if importlib.util.find_spec(deps.check_import) is not None:
        return True

    # Build display list
    all_pkgs = list(deps.no_deps) + list(deps.pip_packages)

    # Confirmation dialog
    dialog = _InstallDialog(deps.label, all_pkgs, parent=parent)
    if dialog.exec() != QDialog.Accepted:
        return False

    # Run pip install with progress
    progress = _InstallProgressDialog(deps.label, parent=parent)
    progress.show()

    worker = _PipInstallWorker(deps)
    result: dict = {"ok": False, "error": ""}

    def _on_done(_):
        result["ok"] = True

    def _on_error(msg):
        result["error"] = msg

    loop = QEventLoop()
    worker.finished_ok.connect(_on_done)
    worker.error.connect(_on_error)
    worker.status.connect(progress.set_status)
    worker.progress.connect(lambda f, d: progress.set_progress(f))
    worker.finished.connect(loop.quit)

    worker.start()
    loop.exec()
    progress.close()

    if not result["ok"]:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            parent, "Install Failed",
            f"Failed to install {deps.label}:\n\n{result['error']}",
        )
        return False

    # Verify it actually imported
    if importlib.util.find_spec(deps.check_import) is None:
        # Force reimport scan after pip install
        importlib.invalidate_caches()
        if importlib.util.find_spec(deps.check_import) is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                parent, "Install Issue",
                f"{deps.label} was installed but cannot be imported.\n"
                "You may need to restart the application.",
            )
            return False

    return True


# ---------------------------------------------------------------------------
# UI dialogs
# ---------------------------------------------------------------------------

class _InstallDialog(QDialog):
    """Confirmation dialog listing packages that will be installed."""

    def __init__(self, feature_label: str, packages: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Required Packages")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        header = QLabel(
            f"<b>{feature_label}</b> requires the following Python packages:"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        pkg_list = QLabel(
            "<pre>" + "\n".join(f"  {p}" for p in packages) + "</pre>"
        )
        pkg_list.setStyleSheet("background: #2a2a2a; padding: 8px; border-radius: 4px;")
        layout.addWidget(pkg_list)

        note = QLabel(
            "This will install these packages into the current virtual environment."
        )
        note.setStyleSheet("color: #999; font-size: 11px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Install")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _InstallProgressDialog(QDialog):
    """Non-modal progress dialog shown during pip install."""

    def __init__(self, feature_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Installing {feature_label}...")
        self.setMinimumWidth(400)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        self._label = QLabel("Preparing...")
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        layout.addWidget(self._bar)

    def set_status(self, text: str):
        self._label.setText(text)

    def set_progress(self, fraction: float):
        if fraction > 0:
            self._bar.setRange(0, 100)
            self._bar.setValue(int(fraction * 100))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _PipInstallWorker(BaseWorker):
    """Run pip install on a background thread."""

    def __init__(self, deps: FeatureDeps, parent=None):
        super().__init__(parent)
        self._deps = deps

    def do_work(self) -> bool:
        total_steps = len(self._deps.no_deps) + (1 if self._deps.pip_packages else 0)
        step = 0

        # Install --no-deps packages first
        for pkg in self._deps.no_deps:
            self.status.emit(f"Installing {pkg} (--no-deps)...")
            self._run_pip(["install", pkg, "--no-deps"])
            step += 1
            self.progress.emit(step / total_steps, f"Installed {pkg}")

        # Install regular packages
        if self._deps.pip_packages:
            pkg_str = " ".join(self._deps.pip_packages)
            self.status.emit(f"Installing {pkg_str}...")
            self._run_pip(["install"] + self._deps.pip_packages)
            step += 1
            self.progress.emit(step / total_steps, "Done")

        return True

    @staticmethod
    def _run_pip(args: list[str]) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pip", *args,
             "--quiet", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            # Extract the meaningful part of the error
            err = result.stderr.strip() or result.stdout.strip()
            # Truncate if huge
            if len(err) > 500:
                err = err[-500:]
            raise RuntimeError(f"pip {' '.join(args)} failed:\n{err}")
