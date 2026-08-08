"""Reusable post-process widget for IMAGE batches.

Runs RealESRGAN sharpen/upscale, SCUNet denoise, and GFPGAN face restore
sequentially on a list of image paths. Used by StillGrabber + Img2Img +
Inpaint tabs to post-process generated stills.

**Not to be confused with** the unified video-output Phase 3
:class:`sdqt.workers.post_pipeline.PostProcessPipeline`, which targets video
files from generation workers and handles color anchor + ESRGAN upscale +
RIFE frame interp + audio mux as a single ordered chain. The two are
intentionally separate — different surface areas:

  - BatchPostProcessWidget     → image-batch outputs (StillGrabber, image gen)
  - PostProcessPipeline (Ph.3) → single-video outputs (Mode 1/2/3, VE save)

If you find yourself needing video-output post-processing on this widget's
callers, route through PostProcessPipeline instead.

Usage:
  Embed widget, then on your own ``_on_done(paths)`` call
  ``widget.process(paths, on_finished_cb)`` if ``widget.is_enabled()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class BatchPostProcessWidget(QWidget):
    """Compact post-process panel — checkbox header + collapsible options grid."""

    progress_changed = Signal(str)            # status string
    finished = Signal(list)                   # list[str] of result paths

    def __init__(
        self,
        state,
        title: str = "Batch Post-Process",
        default_enabled: bool = False,
        default_sharpen: bool = True,
        default_upscale: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # No explicit hide() / WA flags — Qt widgets aren't shown until .show()
        # is called or a visible parent owns them, so there's no startup flash
        # in practice. (Earlier hide() left the toggle hidden after reparent.)
        self._state = state
        self._worker = None
        self._pending: list[str] = []
        self._results: list[str] = []
        self._build_ui(title, default_enabled, default_sharpen, default_upscale)

    def _build_ui(
        self, title: str, default_enabled: bool,
        default_sharpen: bool, default_upscale: bool,
    ) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        header_row = QHBoxLayout()
        header_row.setSpacing(6)
        self._enable = QCheckBox(title)
        self._enable.setToolTip("Run AI enhancement on each result image after generation")
        self._enable.setChecked(default_enabled)
        self._enable.toggled.connect(lambda on: self._group.setVisible(on))
        header_row.addWidget(self._enable)
        header_row.addStretch()
        outer.addLayout(header_row)

        self._group = QGroupBox()
        self._group.setVisible(default_enabled)
        outer.addWidget(self._group)

        grid = QGridLayout(self._group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setSpacing(4)

        def _combo(row: int, label: str, items: list[str], tooltip: str) -> QComboBox:
            grid.addWidget(QLabel(label), row, 0)
            c = QComboBox()
            c.addItems(items)
            c.setFixedWidth(150)
            c.setToolTip(tooltip)
            grid.addWidget(c, row, 1)
            return c

        self._denoise = _combo(0, "Denoise:", ["Disabled", "SCUNet"], "Remove grain/noise")
        self._sharpen = _combo(1, "Sharpen:", ["Disabled", "RealESRGAN 2x"], "Neural detail recovery")
        if default_sharpen:
            self._sharpen.setCurrentIndex(1)
        self._upscale = _combo(2, "Upscale:", ["Disabled", "RealESRGAN 4x"], "4x neural super-resolution")
        if default_upscale:
            self._upscale.setCurrentIndex(1)
        self._face = _combo(3, "Face Restore:", ["Disabled", "GFPGAN v1.4"], "Fix face artifacts")

        grid.addWidget(QLabel("Tile Size:"), 4, 0)
        self._tile = QSpinBox()
        self._tile.setRange(0, 1024)
        self._tile.setValue(512)
        self._tile.setSingleStep(128)
        self._tile.setSuffix("px")
        self._tile.setFixedWidth(80)
        grid.addWidget(self._tile, 4, 1)

    # ── Public API ──────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        if not self._enable.isChecked():
            return False
        return (
            self._denoise.currentIndex() > 0
            or self._sharpen.currentIndex() > 0
            or self._upscale.currentIndex() > 0
            or self._face.currentIndex() > 0
        )

    def process(self, paths: list[str]) -> None:
        """Run post-processing on each path sequentially; emit ``finished`` when done."""
        if not paths or not self.is_enabled():
            self.finished.emit(list(paths))
            return
        self._pending = list(paths)
        self._results = []
        self._run_next()

    def _run_next(self) -> None:
        if not self._pending:
            self.finished.emit(self._results)
            return

        path = self._pending.pop(0)
        total = len(self._results) + len(self._pending) + 1
        current = len(self._results) + 1
        self.progress_changed.emit(f"Post-processing {current}/{total}: {Path(path).name}")

        from sdqt.workers.image_postprocess import ImagePostProcessWorker
        cfg = self._state.global_config
        upscaler_dir = cfg.model_paths.get("upscaler_dir", "")
        face_models_dir = cfg.model_paths.get("face_models_dir", "")

        worker = ImagePostProcessWorker(
            source_path=path,
            ai_denoise=self._denoise.currentIndex() > 0,
            ai_sharpen=self._sharpen.currentIndex() > 0,
            ai_enhance=self._upscale.currentIndex() > 0,
            ai_face_restore=self._face.currentIndex() > 0,
            ai_tile=self._tile.value(),
            upscaler_dir=upscaler_dir,
            face_models_dir=face_models_dir,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self.progress_changed.emit(d))
        worker.finished_ok.connect(self._on_pp_done)

        def _on_err(msg: str, src=path):
            logger.warning("Post-process failed for %s: %s", src, msg)
            self._results.append(src)  # keep original on failure
            self._run_next()
        worker.error.connect(_on_err)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_pp_done(self, result_path: str) -> None:
        self._results.append(result_path)
        self._run_next()
