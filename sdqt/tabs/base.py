"""Base class for all tabs."""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QWidget

from sdqt.state import AppState
from sdqt.utils.file_dialog import get_save_filename

logger = logging.getLogger(__name__)


class BaseTab(QWidget):
    """Common functionality shared by all tabs.

    Subclasses get:
    - Access to ``self.state`` (AppState)
    - ``self.project_name`` (updated via ``on_project_changed``)
    - ``_show_status(msg)`` for unified status bar + local label updates
    - Helper methods for frame capture, file persistence, etc.
    """

    status_message = Signal(str)  # emitted by _show_status(); MainWindow routes to status bar

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.project_name: str = state.current_project or "_default"
        self._restoring: bool = False

    def _show_status(self, msg: str) -> None:
        """Update local status label (if present) AND emit to global status bar."""
        if hasattr(self, "_status") and isinstance(self._status, QLabel):
            self._status.setText(msg)
        self.status_message.emit(msg)

    # -- GPU generation lock ----------------------------------------------

    def acquire_gpu(self, owner: str = "Image") -> bool:
        """Claim the process-wide GPU lock before starting a worker.

        Returns True if acquired; on failure shows a "GPU busy" status and
        returns False so the caller can ``return`` early. Pairs with
        ``release_gpu(owner)`` in the worker's done/error handlers.
        """
        if self.state.acquire_generation(owner):
            return True
        busy = self.state.generation_owner
        self._show_status(f"GPU is busy — {busy} is generating. Wait for it to finish.")
        return False

    def release_gpu(self, owner: str = "Image") -> None:
        """Release the process-wide GPU lock (call from done/error handlers)."""
        self.state.release_generation(owner)

    # -- Project change hook (called by MainWindow) -----------------------

    def on_project_changed(self, project_name: str) -> None:
        """Override to restore UI state from project config."""
        self.project_name = project_name

    # -- Helpers -----------------------------------------------------------

    @property
    def project_path(self) -> Path:
        return self.state.project_manager.get_project_path(self.project_name)

    def persist_file(self, src: str | None, subdir: str = "inputs") -> str:
        """Copy a file into the project directory if from a temp location."""
        if not src:
            return ""
        src_p = Path(src)
        if not src_p.is_file():
            return src
        pp = self.project_path
        if str(src_p).startswith(str(pp)):
            return src
        dest_dir = pp / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src_p.name
        if dest.exists() and dest.stat().st_size == src_p.stat().st_size:
            return str(dest)
        if dest.exists():
            stem, suffix, idx = src_p.stem, src_p.suffix, 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{idx}{suffix}"
                idx += 1
        shutil.copy2(src, dest)
        return str(dest)

    def save_frame_to_project(self, frame_path: str | None) -> str | None:
        """Save a captured frame PNG into project/frames/."""
        if not frame_path or not Path(frame_path).is_file():
            return None
        frames_dir = self.project_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.utcnow().strftime("frame_%Y%m%d_%H%M%S.png")
        dest = frames_dir / name
        shutil.copy2(frame_path, dest)
        logger.info("Saved frame: %s", dest)
        return str(dest)

    def download_frame(self, frame_path: str | None) -> None:
        """Open a native Save dialog to download a captured frame."""
        if not frame_path or not Path(frame_path).is_file():
            return
        dest, _ = get_save_filename(
            self, "Save Frame", "frame.png", "Images (*.png)"
        )
        if dest:
            shutil.copy2(frame_path, dest)
