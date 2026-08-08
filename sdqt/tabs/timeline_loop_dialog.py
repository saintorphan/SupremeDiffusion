"""Create Loop dialog — preview a forward+reverse loop and save it."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from sdqt.widgets.send_targets import CLIP_TARGETS, IMAGE_TARGETS
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.loop_build import LoopBuildWorker

logger = logging.getLogger(__name__)


class LoopDialog(QDialog):
    """Preview + accept/discard a forward+reverse loop of a clip."""

    loop_saved = Signal(str)  # saved file path

    def __init__(
        self,
        source: str,
        media_offset: float,
        duration: float,
        clip_name: str,
        state,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Create Loop — {clip_name}")
        self.setMinimumSize(720, 420)

        self._source = source
        self._media_offset = float(media_offset or 0.0)
        self._duration = float(duration or 0.0)
        self._clip_name = clip_name
        self._state = state
        self._temp_path: str | None = None
        self._worker: LoopBuildWorker | None = None

        self._build_ui()
        self._start_build()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._player = VideoPlayerWidget("Loop Preview")
        self._player.set_send_targets(
            clip_targets=CLIP_TARGETS,
            frame_targets=IMAGE_TARGETS,
        )
        self._player.send_clip_requested.connect(self._on_send_clip)
        self._player.send_frame_requested.connect(self._on_send_frame)
        self._player.set_state_ref(self._state)
        layout.addWidget(self._player, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._status = QLabel("Building loop…")
        self._status.setStyleSheet("color: #aaa; font-size: 13px;")
        layout.addWidget(self._status)

        row = QHBoxLayout()
        self._regen_btn = QPushButton("Regenerate")
        self._regen_btn.clicked.connect(self._start_build)
        self._regen_btn.setEnabled(False)
        row.addWidget(self._regen_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        self._save_btn.setEnabled(False)
        row.addWidget(self._save_btn)

        self._discard_btn = QPushButton("Discard")
        self._discard_btn.clicked.connect(self.reject)
        row.addWidget(self._discard_btn)

        row.addStretch()

        self._export_btn = QPushButton("Export…")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        row.addWidget(self._export_btn)

        self._qe_btn = QPushButton("Quick Export…")
        self._qe_btn.clicked.connect(self._on_quick_export)
        self._qe_btn.setEnabled(False)
        row.addWidget(self._qe_btn)

        layout.addLayout(row)

    # ------------------------------------------------------------------

    def _start_build(self) -> None:
        self._set_result_buttons(False)
        self._regen_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status.setText("Building loop…")

        if self._worker is not None:
            self._worker.abort()
            self._worker.deleteLater()

        self._worker = LoopBuildWorker(
            source=self._source,
            media_offset=self._media_offset,
            duration=self._duration,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_built)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, frac: float, desc: str) -> None:
        self._progress.setValue(int(max(0.0, min(1.0, frac)) * 100))
        self._status.setText(desc)

    def _on_built(self, result) -> None:
        self._temp_path = str(result)
        self._status.setText("Loop ready — preview playing")
        self._player.load_video(self._temp_path)
        self._set_result_buttons(True)
        self._regen_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._status.setText(f"Failed: {msg}")
        self._regen_btn.setEnabled(True)
        QMessageBox.critical(self, "Create Loop Failed", msg)

    # ------------------------------------------------------------------

    def _set_result_buttons(self, enabled: bool) -> None:
        self._save_btn.setEnabled(enabled)
        self._export_btn.setEnabled(enabled)
        self._qe_btn.setEnabled(enabled)

    def _on_save(self) -> None:
        if not self._temp_path or not Path(self._temp_path).is_file():
            return

        project_path = None
        try:
            name = self._state.current_project
            if name:
                project_path = self._state.project_manager.get_project_path(name)
        except Exception:
            pass
        if not project_path:
            QMessageBox.warning(
                self, "Save Loop",
                "No active project — cannot save into clips folder.",
            )
            return

        clips_dir = Path(project_path) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        base = self._clip_name or "loop"
        dest = clips_dir / f"{base}_loop.mp4"
        idx = 1
        while dest.exists():
            dest = clips_dir / f"{base}_loop_{idx}.mp4"
            idx += 1

        try:
            shutil.copy(self._temp_path, dest)
        except Exception as exc:
            QMessageBox.critical(self, "Save Loop", f"Copy failed: {exc}")
            return

        self.loop_saved.emit(str(dest))
        self.accept()

    def _on_export(self) -> None:
        if not self._temp_path:
            return
        from sdqt.tabs.clip_menu_actions import run_export_dialog
        run_export_dialog(
            self, self._state, self._temp_path,
            0.0, 0.0, f"{self._clip_name}_loop",
        )

    def _on_quick_export(self) -> None:
        if not self._temp_path:
            return
        from sdqt.tabs.clip_menu_actions import run_quick_export
        run_quick_export(
            self, self._state, self._temp_path,
            0.0, 0.0, f"{self._clip_name}_loop",
        )

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _find_main_window(self):
        """Walk up the parent chain to reach MainWindow's dispatch methods."""
        w = self.parent()
        while w is not None:
            if hasattr(w, "_dispatch_video_send") or hasattr(w, "_capture_and_send_frame"):
                return w
            w = w.parent() if hasattr(w, "parent") else None
        return None

    def _on_send_clip(self, key: str) -> None:
        if not self._temp_path:
            return
        mw = self._find_main_window()
        if mw is None or not hasattr(mw, "_dispatch_video_send"):
            QMessageBox.warning(
                self, "Send Clip",
                "Could not reach the main window to route the clip.",
            )
            return
        try:
            mw._dispatch_video_send(key, self._temp_path)
        except Exception as exc:
            logger.exception("Send clip failed")
            QMessageBox.warning(self, "Send Clip", f"Failed: {exc}")

    def _on_send_frame(self, key: str) -> None:
        mw = self._find_main_window()
        if mw is None or not hasattr(mw, "_capture_and_send_frame"):
            return
        try:
            mw._capture_and_send_frame(self._player, key)
        except Exception as exc:
            logger.exception("Send frame failed")
            QMessageBox.warning(self, "Send Frame", f"Failed: {exc}")

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.abort()
        super().closeEvent(event)
