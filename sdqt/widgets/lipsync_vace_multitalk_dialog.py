"""VACE MultiTalk lip sync dialog — popup from video player context menu."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.lip_sync import VideoFaceDetectWorker

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class LipSyncVaceMultitalkDialog(QDialog):
    """Modal dialog for lip-syncing a video via VACE MultiTalk."""

    def __init__(self, source_path: str, state=None, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._source_path = source_path
        self._worker = None
        self._detect_worker: VideoFaceDetectWorker | None = None
        self._detected_faces: list[dict] = []
        self._speaker_assignments: dict[int, int] = {}
        self.result_path: str | None = None

        self.setWindowTitle("Lip Sync \u2014 VACE MultiTalk")
        self.setMinimumSize(900, 580)
        self.resize(1000, 650)

        self._build_ui()
        self._source_player.load_video(source_path)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source")
        splitter.addWidget(self._source_player)
        self._result_player = VideoPlayerWidget("Result")
        splitter.addWidget(self._result_player)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        ctrl = QWidget()
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(6)

        # Audio inputs
        audio_row = QHBoxLayout()
        self._audio_left = FileDropWidget("Audio (Speaker 1)", extensions=_AUDIO_EXTS)
        audio_row.addWidget(self._audio_left)
        self._audio_right = FileDropWidget("Audio (Speaker 2, optional)", extensions=_AUDIO_EXTS)
        audio_row.addWidget(self._audio_right)
        cl.addLayout(audio_row)

        # Face detection + speaker assignment
        fg = QGroupBox("Speaker Assignment")
        fg_l = QVBoxLayout(fg)

        dr = QHBoxLayout()
        self._detect_btn = QPushButton("Detect Faces")
        self._detect_btn.clicked.connect(self._on_detect)
        dr.addWidget(self._detect_btn)
        self._detect_status = QLabel("")
        dr.addWidget(self._detect_status, 1)
        fg_l.addLayout(dr)

        self._face_list = QListWidget()
        self._face_list.setFlow(QListWidget.LeftToRight)
        self._face_list.setWrapping(False)
        self._face_list.setIconSize(self._face_list.iconSize().__class__(80, 80))
        self._face_list.setFixedHeight(110)
        self._face_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._face_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._face_list.setSelectionMode(QListWidget.SingleSelection)
        self._face_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: 1px solid #444; }"
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background: #2a5a8a; border: 2px solid #5aa; }"
        )
        fg_l.addWidget(self._face_list)

        assign_row = QHBoxLayout()
        self._assign_s1_btn = QPushButton("Assign → Speaker 1")
        self._assign_s1_btn.clicked.connect(lambda: self._assign_speaker(1))
        assign_row.addWidget(self._assign_s1_btn)
        self._assign_s2_btn = QPushButton("Assign → Speaker 2")
        self._assign_s2_btn.clicked.connect(lambda: self._assign_speaker(2))
        assign_row.addWidget(self._assign_s2_btn)
        self._clear_assign_btn = QPushButton("Clear")
        self._clear_assign_btn.clicked.connect(self._clear_assignments)
        assign_row.addWidget(self._clear_assign_btn)
        assign_row.addStretch()
        fg_l.addLayout(assign_row)
        cl.addWidget(fg)

        # Settings
        sr = QHBoxLayout()
        sr.setSpacing(6)
        sr.addWidget(QLabel("Resolution:"))
        self._resolution = QComboBox()
        self._resolution.addItems(["832x480", "512x512", "480x832"])
        self._resolution.setFixedWidth(100)
        sr.addWidget(self._resolution)
        sr.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(4, 50)
        self._steps.setValue(20)
        self._steps.setFixedWidth(60)
        sr.addWidget(self._steps)
        sr.addWidget(QLabel("Shift:"))
        self._shift = QDoubleSpinBox()
        self._shift.setRange(1.0, 20.0)
        self._shift.setValue(5.0)
        self._shift.setSingleStep(0.5)
        self._shift.setFixedWidth(70)
        sr.addWidget(self._shift)
        sr.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 99999)
        self._seed.setValue(-1)
        self._seed.setSpecialValueText("Random")
        self._seed.setFixedWidth(75)
        sr.addWidget(self._seed)
        sr.addStretch()
        cl.addLayout(sr)

        # Action row
        ar = QHBoxLayout()
        self._go_btn = QPushButton("Generate")
        self._go_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._go_btn.clicked.connect(self._on_run)
        ar.addWidget(self._go_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet("font-weight: bold; color: #e55;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        ar.addWidget(self._abort_btn)
        self._status = QLabel("")
        ar.addWidget(self._status, 1)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self.accept)
        ar.addWidget(self._accept_btn)
        reject_btn = QPushButton("Cancel")
        reject_btn.clicked.connect(self.reject)
        ar.addWidget(reject_btn)
        cl.addLayout(ar)

        scroll.setWidget(ctrl)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        main_splitter.addWidget(splitter)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter, 1)

    # -- Face detection --------------------------------------------------------

    @Slot()
    def _on_detect(self) -> None:
        if not self._source_path:
            return
        models_dir = ""
        if self.state:
            models_dir = self.state.global_config.model_paths.get("face_models_dir", "")
        if not models_dir:
            self._detect_status.setText("Face models directory not set.")
            return
        self._detect_btn.setEnabled(False)
        self._detect_status.setText("Detecting...")
        worker = VideoFaceDetectWorker(self._source_path, models_dir, parent=self)
        worker.finished_ok.connect(self._on_detect_done)
        worker.error.connect(self._on_detect_error)
        worker.finished.connect(worker.deleteLater)
        self._detect_worker = worker
        worker.start()

    def _on_detect_done(self, faces: list[dict]) -> None:
        self._detect_btn.setEnabled(True)
        self._detected_faces = faces
        self._face_list.clear()
        self._speaker_assignments.clear()
        if not faces:
            self._detect_status.setText("No faces detected.")
            return
        for face in faces:
            thumb = face.get("thumbnail")
            if thumb is None:
                continue
            pil_img = thumb.convert("RGB")
            w, h = pil_img.size
            data = pil_img.tobytes("raw", "RGB")
            qimg = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            item.setText(f"Face {face['index']}")
            item.setData(Qt.UserRole, face["index"])
            self._face_list.addItem(item)
        self._detect_status.setText(f"Found {len(faces)} face(s). Assign speakers.")
        if faces:
            self._face_list.setCurrentRow(0)

    def _on_detect_error(self, msg: str) -> None:
        self._detect_btn.setEnabled(True)
        self._detect_status.setText(f"Error: {msg}")

    def _assign_speaker(self, speaker: int) -> None:
        row = self._face_list.currentRow()
        if row < 0:
            return
        self._speaker_assignments[row] = speaker
        item = self._face_list.item(row)
        if item:
            item.setText(f"Face {row} → S{speaker}")

    def _clear_assignments(self) -> None:
        self._speaker_assignments.clear()
        for i in range(self._face_list.count()):
            item = self._face_list.item(i)
            if item:
                item.setText(f"Face {i}")

    # -- Generation ------------------------------------------------------------

    @Slot()
    def _on_run(self) -> None:
        if not self._source_path:
            self._status.setText("No source video.")
            return
        audio1 = self._audio_left.file_path
        if not audio1:
            self._status.setText("Speaker 1 audio required.")
            return

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="vmt_")
        tmp.close()
        output = tmp.name

        # Parse resolution
        res_text = self._resolution.currentText()
        try:
            rw, rh = [int(x) for x in res_text.split("x")]
        except ValueError:
            rw, rh = 832, 480

        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._status.setText("Running VACE MultiTalk...")
        if self.state:
            self.state.unload_pipelines()

        from sdqt.workers.vace_multitalk import VaceMultitalkWorker
        worker = VaceMultitalkWorker(
            video_path=self._source_path,
            audio_path=audio1,
            audio_path_2=self._audio_right.file_path or "",
            speaker_assignments=dict(self._speaker_assignments),
            face_data=self._detected_faces,
            output_path=output,
            width=rw,
            height=rh,
            steps=self._steps.value(),
            shift=self._shift.value(),
            seed=self._seed.value(),
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

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    def _cleanup_gpu(self) -> None:
        self._worker = None
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
