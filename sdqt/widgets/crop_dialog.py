"""Crop dialog — popup for cropping a video with visual rectangle preview."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.tabs.trim_crop import CropPreviewView
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.trim_crop import CropWorker

logger = logging.getLogger(__name__)

_CROP_PRESETS = [
    ("512x512", 512, 512),
    ("640x480", 640, 480),
    ("720x480", 720, 480),
    ("768x512", 768, 512),
    ("832x480", 832, 480),
    ("480x720", 480, 720),
    ("512x768", 512, 768),
    ("480x832", 480, 832),
    ("1024x576", 1024, 576),
    ("576x1024", 576, 1024),
    ("1280x720", 1280, 720),
    ("720x1280", 720, 1280),
]


class CropDialog(QDialog):
    """Modal dialog for cropping a video."""

    def __init__(self, source_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop Video")
        self.setMinimumSize(900, 550)
        self.resize(1000, 600)

        self._source_path = source_path
        self._source_info: dict = {}
        self._worker: CropWorker | None = None
        self.result_path: str | None = None

        self._build_ui()
        self._load_source(source_path)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Top: crop preview + result
        splitter = QSplitter(Qt.Horizontal)

        left = QVBoxLayout()
        left_w = QWidget()
        left_w_layout = QVBoxLayout(left_w)
        left_w_layout.setContentsMargins(0, 0, 0, 0)
        left_w_layout.setSpacing(0)
        self._source_label = QLabel("")
        self._source_label.setStyleSheet("font-size: 11px; color: #888;")
        left_w_layout.addWidget(self._source_label)
        self._crop_preview = CropPreviewView()
        self._crop_preview.setMinimumHeight(300)
        left_w_layout.addWidget(self._crop_preview, 1)
        splitter.addWidget(left_w)

        self._result_player = VideoPlayerWidget("Cropped Result")
        splitter.addWidget(self._result_player)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Bottom controls
        bottom = QHBoxLayout()

        self._info_label = QLabel("")
        bottom.addWidget(self._info_label, 1)

        bottom.addWidget(QLabel("Crop Size:"))
        self._crop_preset = QComboBox()
        self._crop_preset.setFixedWidth(140)
        self._crop_preset.setMaxVisibleItems(14)
        for label, w, h in _CROP_PRESETS:
            self._crop_preset.addItem(label, (w, h))
        self._crop_preset.currentIndexChanged.connect(self._on_preset_changed)
        bottom.addWidget(self._crop_preset)

        self._crop_btn = QPushButton("Crop")
        self._crop_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._crop_btn.clicked.connect(self._on_crop)
        bottom.addWidget(self._crop_btn)

        self._status = QLabel("")
        bottom.addWidget(self._status)

        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self.accept)
        bottom.addWidget(self._accept_btn)

        reject_btn = QPushButton("Cancel")
        reject_btn.clicked.connect(self.reject)
        bottom.addWidget(reject_btn)

        layout.addLayout(bottom)

    def _load_source(self, path: str) -> None:
        try:
            from supremediffusion.utils.video import probe_video, extract_single_frame
            info = probe_video(path)
            self._source_info = info
            self._source_label.setText(f"Video: {Path(path).name}")
            self._info_label.setText(
                f"Duration: {info['duration']:.1f}s  |  "
                f"FPS: {info['fps']:.1f}  |  "
                f"{info['width']}\u00d7{info['height']}"
            )
            img = extract_single_frame(path, 0)
            if img:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                tmp.close()
                pixmap = QPixmap(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)
                if not pixmap.isNull():
                    self._crop_preview.set_frame(pixmap)
                    self._on_preset_changed(0)
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")

    def _on_preset_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self._crop_preset.itemData(index)
        if data:
            w, h = data
            self._crop_preview.set_crop_size(w, h)

    @Slot()
    def _on_crop(self) -> None:
        if not self._source_path or not Path(self._source_path).is_file():
            self._status.setText("No source.")
            return
        rect = self._crop_preview.get_crop_rect()
        if rect is None:
            self._status.setText("No crop region.")
            return
        x, y, w, h = rect
        if w < 1 or h < 1:
            self._status.setText("Invalid crop dimensions.")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="crop_")
        tmp.close()
        output = tmp.name

        self._crop_btn.setEnabled(False)
        self._status.setText(f"Cropping to {w}x{h}...")

        worker = CropWorker(self._source_path, output, x, y, w, h, parent=self)
        worker.progress.connect(lambda f, d: self._status.setText(d))
        worker.finished_ok.connect(self._on_crop_done)
        worker.error.connect(lambda msg: self._on_error(msg))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_crop_done(self, result: str) -> None:
        self._crop_btn.setEnabled(True)
        self.result_path = result
        self._result_player.load_video(result, auto_play=True)
        self._status.setText(f"Cropped: {Path(result).name}")
        self._accept_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._crop_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")

