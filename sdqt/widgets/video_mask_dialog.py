"""Modal mask-painting dialog for LTX video inpainting workflows.

Pulls the first (or user-specified) frame from a source video, drops it into
the existing ``MaskEditorWidget`` for paint tools, and on Apply saves the
mask to ``<project>/guidance/inpaint_mask.png`` so the LTX inpaint LoRA
path can pick it up.

For now we treat the mask as **static** — the same painted region applies
to every output frame. Per-frame mask sequences are a bigger UI we can
build later.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from sdqt.widgets.mask_editor import MaskEditorWidget

logger = logging.getLogger(__name__)


class VideoMaskDialog(QDialog):
    """Modal popup that lets the user paint a static mask over a video frame."""

    def __init__(
        self,
        video_path: str,
        save_path: str | Path,
        frame_index: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paint Inpaint Mask")
        self.setModal(True)
        self.resize(1000, 720)

        self._video_path = video_path
        self._save_path = Path(save_path)
        self._final_path: str | None = None
        self._frame_path: str | None = None

        self._build_ui()
        self._load_frame(frame_index)

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top row: which frame to use as the mask backdrop
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("Mask backdrop frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 9999)
        self._frame_spin.setValue(0)
        self._frame_spin.setFixedWidth(80)
        self._frame_spin.setToolTip(
            "Which frame of the source video to paint on. The painted region "
            "applies uniformly across the whole output (static mask)."
        )
        self._frame_spin.editingFinished.connect(
            lambda: self._load_frame(int(self._frame_spin.value()))
        )
        top.addWidget(self._frame_spin)
        self._frame_status = QLabel("")
        self._frame_status.setStyleSheet("color: #888; font-size: 11px;")
        top.addWidget(self._frame_status)
        top.addStretch()
        layout.addLayout(top)

        # Mask editor
        self._editor = MaskEditorWidget(self)
        layout.addWidget(self._editor, 1)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel,
            parent=self,
        )
        btns.button(QDialogButtonBox.Save).setText("Apply Mask")
        btns.accepted.connect(self._on_apply)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _load_frame(self, index: int) -> None:
        """Extract the chosen frame, drop it into the editor as the backdrop."""
        try:
            from supremediffusion.utils.video import extract_single_frame
            img = extract_single_frame(self._video_path, max(0, int(index)))
        except Exception as exc:
            logger.warning("Frame extract failed: %s", exc)
            self._frame_status.setText(f"frame extract failed: {exc}")
            return

        # Save to a temp PNG so MaskEditorWidget can load it via path API.
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=f"_frame{index}.png", prefix="vmd_")
        import os
        os.close(fd)
        img.save(tmp_path, "PNG")
        self._frame_path = tmp_path
        self._editor.load_image(tmp_path)
        self._frame_status.setText(f"{img.size[0]}×{img.size[1]} — paint white over the area to inpaint")

    # ── Apply ───────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        mask_tmp = self._editor.get_mask_path()
        if not mask_tmp:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Empty mask", "Paint at least one stroke before applying."
            )
            return
        # Persist into project guidance/ directory so the pipeline can find it.
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(mask_tmp, self._save_path)
            self._final_path = str(self._save_path)
            logger.info("Inpaint mask saved: %s", self._save_path)
            self.accept()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save failed", f"Could not save mask:\n{exc}")

    @property
    def mask_path(self) -> str | None:
        return self._final_path
