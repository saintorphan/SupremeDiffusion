"""LTX Audio-to-Video dialog — lip-synced video from image+audio or video+audio."""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_RESOLUTION_PRESETS = [
    ("576x320 (16:9)", 576, 320),
    ("320x576 (9:16)", 320, 576),
    ("768x448 (16:9)", 768, 448),
    ("448x768 (9:16)", 448, 768),
    ("512x512 (1:1)", 512, 512),
    ("1024x576 (16:9)", 1024, 576),
    ("576x1024 (9:16)", 576, 1024),
    ("1024x768 (4:3)", 1024, 768),
    ("768x1024 (3:4)", 768, 1024),
]

_DURATION_PRESETS = [
    ("~2s (49 frames)", 49),
    ("~3s (73 frames)", 73),
    ("~4s (97 frames)", 97),
    ("~5s (121 frames)", 121),
    ("~8s (193 frames)", 193),
    ("~10s (241 frames)", 241),
]


class LipSyncLTXDialog(QDialog):
    """Modal dialog for LTX Audio-to-Video generation.

    Two modes:
    - **Image + Audio**: Source is an image, generates new video with lip sync
      (A2VidPipelineTwoStage).
    - **Control Video + Audio**: Source is a video, regenerates with lip sync
      using IC-LoRA control conditioning. Preserves structure/motion from the
      source while syncing lips to the audio track.
    """

    def __init__(self, source_path: str, state: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._source_path = source_path
        self._worker = None
        self.result_path: str | None = None

        ext = Path(source_path).suffix.lower()
        self._source_is_video = ext in _VIDEO_EXTS

        title = "LTX Audio-to-Video"
        if self._source_is_video:
            title += " (Control Video)"
        self.setWindowTitle(title)
        self.setMinimumSize(900, 580)
        self.resize(1000, 650)
        self._build_ui()

        self._source_player.load_video(source_path)

        # Auto-detect source resolution and FPS for seamless stitching
        if self._source_is_video:
            self._auto_detect_source()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        main_splitter = QSplitter(Qt.Vertical)

        # Video players
        video_splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source")
        self._result_player = VideoPlayerWidget("Result")
        video_splitter.addWidget(self._source_player)
        video_splitter.addWidget(self._result_player)
        main_splitter.addWidget(video_splitter)

        # Controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)
        cl.setSpacing(6)

        # Mode indicator
        if self._source_is_video:
            mode_label = QLabel(
                "<b>Control Video mode</b> — source video provides structure/motion, "
                "audio drives lip sync. Uses IC-LoRA conditioning."
            )
            mode_label.setWordWrap(True)
            mode_label.setStyleSheet("color: #1abc9c; font-size: 11px; padding: 2px;")
            cl.addWidget(mode_label)

        # Audio input — optional for video sources (can use video's own audio)
        audio_row = QHBoxLayout()
        self._audio = FileDropWidget("Audio", extensions=_AUDIO_EXTS | _VIDEO_EXTS)
        audio_row.addWidget(self._audio, 1)
        if self._source_is_video:
            self._use_source_audio = QCheckBox("Use source audio")
            self._use_source_audio.setChecked(True)
            self._use_source_audio.setToolTip(
                "Extract audio from the source video instead of a separate file"
            )
            self._use_source_audio.toggled.connect(
                lambda checked: self._audio.setEnabled(not checked)
            )
            audio_row.addWidget(self._use_source_audio)
            self._audio.setEnabled(False)
        else:
            self._use_source_audio = None
        cl.addLayout(audio_row)

        # Prompt
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Prompt:"))
        self._prompt = QTextEdit()
        self._prompt.setFixedHeight(50)
        self._prompt.setPlaceholderText("A person speaking expressively...")
        prompt_row.addWidget(self._prompt, 1)
        cl.addLayout(prompt_row)

        # Settings row 1: Resolution, Duration, FPS
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        row1.addWidget(QLabel("Resolution:"))
        self._resolution = QComboBox()
        self._resolution.setFixedWidth(120)
        for label, _, _ in _RESOLUTION_PRESETS:
            self._resolution.addItem(label)
        row1.addWidget(self._resolution)

        row1.addWidget(QLabel("Duration:"))
        self._duration = QComboBox()
        self._duration.setFixedWidth(140)
        for label, _ in _DURATION_PRESETS:
            self._duration.addItem(label)
        self._duration.setCurrentIndex(3)
        row1.addWidget(self._duration)

        row1.addWidget(QLabel("FPS:"))
        self._fps = QSpinBox()
        self._fps.setFixedWidth(60)
        self._fps.setRange(12, 50)
        self._fps.setValue(24)
        row1.addWidget(self._fps)

        row1.addStretch()
        cl.addLayout(row1)

        # Settings row 2: Steps, CFG, STG, A2V, Seed, Audio Start
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        row2.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setFixedWidth(60)
        self._steps.setRange(4, 50)
        self._steps.setValue(30)
        row2.addWidget(self._steps)

        row2.addWidget(QLabel("CFG:"))
        self._cfg = QDoubleSpinBox()
        self._cfg.setFixedWidth(70)
        self._cfg.setRange(1.0, 10.0)
        self._cfg.setDecimals(1)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(3.0)
        row2.addWidget(self._cfg)

        row2.addWidget(QLabel("STG:"))
        self._stg = QDoubleSpinBox()
        self._stg.setFixedWidth(70)
        self._stg.setRange(0.0, 3.0)
        self._stg.setDecimals(1)
        self._stg.setSingleStep(0.1)
        self._stg.setValue(1.0)
        row2.addWidget(self._stg)

        row2.addWidget(QLabel("A2V:"))
        self._a2v = QDoubleSpinBox()
        self._a2v.setFixedWidth(70)
        self._a2v.setRange(0.0, 10.0)
        self._a2v.setDecimals(1)
        self._a2v.setSingleStep(0.5)
        self._a2v.setValue(3.0)
        self._a2v.setToolTip("Audio-to-Video guidance — controls lip sync strength")
        row2.addWidget(self._a2v)

        row2.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setFixedWidth(90)
        self._seed.setRange(-1, 999999999)
        self._seed.setValue(-1)
        row2.addWidget(self._seed)

        row2.addWidget(QLabel("Audio Start:"))
        self._audio_start = QDoubleSpinBox()
        self._audio_start.setFixedWidth(70)
        self._audio_start.setRange(0.0, 600.0)
        self._audio_start.setDecimals(1)
        self._audio_start.setSingleStep(0.5)
        self._audio_start.setValue(0.0)
        row2.addWidget(self._audio_start)

        row2.addStretch()
        cl.addLayout(row2)

        # Control video settings (only for video sources)
        if self._source_is_video:
            row3 = QHBoxLayout()
            row3.setSpacing(6)

            row3.addWidget(QLabel("Control Strength:"))
            self._control_strength = QDoubleSpinBox()
            self._control_strength.setFixedWidth(70)
            self._control_strength.setRange(0.1, 1.0)
            self._control_strength.setDecimals(2)
            self._control_strength.setSingleStep(0.05)
            self._control_strength.setValue(0.85)
            self._control_strength.setToolTip(
                "How closely to follow the source video structure (0.1=loose, 1.0=exact)"
            )
            row3.addWidget(self._control_strength)

            self._loop_silence = QCheckBox("Accommodate loop silence")
            self._loop_silence.setToolTip(
                "Pad 0.5s silence at the start of the audio so the mouth\n"
                "is idle at the loop seam point (frame 0)"
            )
            self._loop_silence.setChecked(False)
            row3.addWidget(self._loop_silence)

            row3.addStretch()
            cl.addLayout(row3)
        else:
            self._control_strength = None
            self._loop_silence = None

        # Action row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        self._go_btn = QPushButton("Generate")
        self._go_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._go_btn.clicked.connect(self._on_run)
        action_row.addWidget(self._go_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet("color: #f44;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        action_row.addWidget(self._abort_btn)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 11px; color: #aaa;")
        action_row.addWidget(self._status, 1)

        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self.accept)
        action_row.addWidget(self._accept_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(cancel_btn)

        cl.addLayout(action_row)

        scroll.setWidget(ctrl)
        main_splitter.addWidget(scroll)
        main_splitter.setSizes([400, 200])
        layout.addWidget(main_splitter, 1)

    def _auto_detect_source(self) -> None:
        """Detect source video resolution and FPS, set controls to match."""
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(self._source_path)
            src_w = info.get("width", 0)
            src_h = info.get("height", 0)
            src_fps = info.get("fps", 24)

            # Set FPS to match source
            self._fps.setValue(int(round(src_fps)))

            # Find closest matching resolution preset
            if src_w and src_h:
                best_idx = 0
                best_diff = float("inf")
                for i, (_, pw, ph) in enumerate(_RESOLUTION_PRESETS):
                    diff = abs(pw - src_w) + abs(ph - src_h)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
                self._resolution.setCurrentIndex(best_idx)

                # Store source dims for output scaling
                self._source_width = src_w
                self._source_height = src_h
                self._source_fps = src_fps

                # Show info
                gen_label = _RESOLUTION_PRESETS[best_idx][0]
                if src_w != _RESOLUTION_PRESETS[best_idx][1] or src_h != _RESOLUTION_PRESETS[best_idx][2]:
                    logger.info(
                        "Source %dx%d → generating at %s, will scale output to match",
                        src_w, src_h, gen_label,
                    )
        except Exception as exc:
            logger.debug("Auto-detect failed: %s", exc)
            self._source_width = 0
            self._source_height = 0
            self._source_fps = 24

    def _on_run(self) -> None:
        # Determine audio source
        use_source = (
            self._use_source_audio is not None
            and self._use_source_audio.isChecked()
        )
        if use_source:
            audio = self._source_path
        else:
            audio = self._audio.file_path
        if not audio:
            self._status.setText("No audio file selected.")
            return

        res_idx = self._resolution.currentIndex()
        _, width, height = _RESOLUTION_PRESETS[res_idx]

        dur_idx = self._duration.currentIndex()
        _, num_frames = _DURATION_PRESETS[dur_idx]

        output = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="ltx_a2v_", delete=False,
        )
        output.close()

        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._accept_btn.setEnabled(False)

        control_strength = 0.0
        control_video = None
        loop_pad = False
        if self._source_is_video:
            control_video = self._source_path
            control_strength = (
                self._control_strength.value()
                if self._control_strength else 0.85
            )
            loop_pad = (
                self._loop_silence is not None
                and self._loop_silence.isChecked()
            )
            self._status.setText("Starting LTX Control Video generation...")
        else:
            self._status.setText("Starting LTX A2Vid generation...")

        from sdqt.workers.ltx_a2vid import LTXA2VidWorker

        worker = LTXA2VidWorker(
            source_image_path=self._source_path,
            audio_path=audio,
            output_path=output.name,
            prompt=self._prompt.toPlainText().strip(),
            negative_prompt="",
            width=width,
            height=height,
            num_frames=num_frames,
            fps=float(self._fps.value()),
            steps=self._steps.value(),
            cfg_scale=self._cfg.value(),
            stg_scale=self._stg.value(),
            a2v_scale=self._a2v.value(),
            seed=self._seed.value(),
            audio_start=self._audio_start.value(),
            control_video=control_video,
            control_strength=control_strength,
            loop_silence_pad=loop_pad,
            source_width=getattr(self, "_source_width", 0),
            source_height=getattr(self, "_source_height", 0),
            source_fps=getattr(self, "_source_fps", 0),
            state=self.state,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._status.setText(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, result: str) -> None:
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self.result_path = result
        self._result_player.load_video(result, auto_play=True)
        self._status.setText(f"Done: {Path(result).name}")
        self._accept_btn.setEnabled(True)
        self._cleanup_gpu()

    def _on_error(self, msg: str) -> None:
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._status.setText(f"Error: {msg}")
        self._cleanup_gpu()

    def _on_abort(self) -> None:
        if self._worker is not None:
            self._worker.abort()
        self._status.setText("Aborting...")

    def _cleanup_gpu(self) -> None:
        self._worker = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
