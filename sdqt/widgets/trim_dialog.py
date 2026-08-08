"""Trim dialog — popup for trimming a video via range slider."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)

from sdqt.widgets.range_slider import RangeSliderWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.trim_crop import TrimWorker

logger = logging.getLogger(__name__)


class TrimDialog(QDialog):
    """Modal dialog for trimming a video clip."""

    def __init__(self, source_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trim Video")
        self.setMinimumSize(900, 550)
        self.resize(1000, 600)

        self._source_path = source_path
        self._source_info: dict = {}
        self._worker: TrimWorker | None = None
        self.result_path: str | None = None

        self._build_ui()
        self._probe_source(source_path)
        self._source_player.load_video(source_path)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Video players
        splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source")
        splitter.addWidget(self._source_player)
        self._result_player = VideoPlayerWidget("Result")
        splitter.addWidget(self._result_player)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Range slider
        trim_group = QGroupBox("Trim Range")
        tg_layout = QVBoxLayout(trim_group)
        tg_layout.setContentsMargins(4, 0, 4, 0)
        tg_layout.setSpacing(0)

        self._range_slider = RangeSliderWidget()
        self._range_slider.range_changed.connect(self._on_range_changed)
        self._range_slider.begin_dragging.connect(self._on_begin_drag)
        self._range_slider.end_dragging.connect(self._on_end_drag)
        tg_layout.addWidget(self._range_slider)

        range_row = QHBoxLayout()
        self._begin_thumb = QLabel("IN")
        self._begin_thumb.setFixedSize(80, 45)
        self._begin_thumb.setAlignment(Qt.AlignCenter)
        self._begin_thumb.setStyleSheet("background: #1a1a1a; border: 1px solid #444; color: #555;")
        self._begin_thumb.setScaledContents(True)
        range_row.addWidget(self._begin_thumb)
        self._range_begin_label = QLabel("Begin: 0.0s")
        range_row.addWidget(self._range_begin_label)
        range_row.addStretch()
        self._range_duration_label = QLabel("Duration: 0.0s")
        range_row.addWidget(self._range_duration_label)
        range_row.addStretch()
        self._range_end_label = QLabel("End: 0.0s")
        range_row.addWidget(self._range_end_label)
        self._end_thumb = QLabel("OUT")
        self._end_thumb.setFixedSize(80, 45)
        self._end_thumb.setAlignment(Qt.AlignCenter)
        self._end_thumb.setStyleSheet("background: #1a1a1a; border: 1px solid #444; color: #555;")
        self._end_thumb.setScaledContents(True)
        range_row.addWidget(self._end_thumb)
        tg_layout.addLayout(range_row)
        layout.addWidget(trim_group)

        # Bottom controls
        bottom = QHBoxLayout()
        self._info_label = QLabel("")
        bottom.addWidget(self._info_label, 1)

        self._trim_btn = QPushButton("Trim")
        self._trim_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._trim_btn.clicked.connect(self._on_trim)
        bottom.addWidget(self._trim_btn)

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

    def _probe_source(self, path: str) -> None:
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            self._source_info = info
            self._info_label.setText(
                f"Duration: {info['duration']:.1f}s  |  "
                f"FPS: {info['fps']:.1f}  |  "
                f"{info['width']}\u00d7{info['height']}"
            )
            self._range_slider.set_range(0.0, 1.0)
            self._on_range_changed(0.0, 1.0)
            self._extract_thumb(0.0, self._begin_thumb)
            self._extract_thumb(info["duration"], self._end_thumb)
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")

    def _on_range_changed(self, begin: float, end: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        b_sec = begin * duration
        e_sec = end * duration
        self._range_begin_label.setText(f"Begin: {b_sec:.2f}s")
        self._range_end_label.setText(f"End: {e_sec:.2f}s")
        self._range_duration_label.setText(f"Duration: {e_sec - b_sec:.2f}s")

    def _on_begin_drag(self, val: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._begin_thumb)

    def _on_end_drag(self, val: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._end_thumb)

    def _extract_thumb(self, sec: float, label: QLabel) -> None:
        try:
            from supremediffusion.utils.video import extract_single_frame
            fps = self._source_info.get("fps", 16)
            frame_num = max(0, int(sec * fps))
            img = extract_single_frame(self._source_path, frame_num)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()
            pixmap = QPixmap(tmp.name)
            if not pixmap.isNull():
                label.setPixmap(pixmap)
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    @Slot()
    def _on_trim(self) -> None:
        duration = self._source_info.get("duration", 0.0)
        start_sec = self._range_slider.begin() * duration
        end_sec = self._range_slider.end() * duration
        if end_sec <= start_sec:
            self._status.setText("End must be after Begin.")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="trim_")
        tmp.close()
        output = tmp.name

        self._trim_btn.setEnabled(False)
        self._status.setText("Trimming...")

        worker = TrimWorker(
            self._source_path, output, start_sec, end_sec, parent=self,
        )
        worker.progress.connect(lambda f, d: self._status.setText(d))
        worker.finished_ok.connect(self._on_trim_done)
        worker.error.connect(lambda msg: self._on_error(msg))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_trim_done(self, result: str) -> None:
        self._trim_btn.setEnabled(True)
        self.result_path = result
        self._result_player.load_video(result, auto_play=True)
        self._status.setText(f"Trimmed: {Path(result).name}")
        self._accept_btn.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._trim_btn.setEnabled(True)
        self._status.setText(f"Error: {msg}")
