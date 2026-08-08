"""MuseTalk clip dialog — modal lip sync for individual timeline clips."""

from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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

from sdqt.state import AppState
from sdqt.utils.naming import cap_stem
from sdqt.widgets.audio_player import AudioPlayerWidget
from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.lip_sync import VideoFaceDetectWorker

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class MuseTalkClipDialog(QDialog):
    """Modal dialog for lip-syncing a single timeline clip via MuseTalk."""

    def __init__(self, clip, state: AppState, project_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._clip = clip
        self.state = state
        self._project_path = project_path
        self._worker = None
        self._detect_worker: VideoFaceDetectWorker | None = None
        self._detected_faces: list[dict] = []
        self._selected_bbox: list[int] | None = None
        self._trimmed_path: str | None = None
        self.result_path: str | None = None
        self._cb_files: list[str] = []

        self.setWindowTitle(f"Lip Sync (MuseTalk) \u2014 {clip.name}")
        self.setMinimumWidth(700)
        self.resize(950, 580)
        logger.info("MuseTalkClipDialog opened for clip=%s path=%s offset=%.2f dur=%.2f",
                     clip.name, clip.path, getattr(clip, "media_offset", 0.0), clip.duration)

        self._build_ui()
        self._pre_trim()
        self._rescan_chatterbox()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Main splitter: left (dialog content) | right (Chatterbox panel) ──
        self._main_splitter = QSplitter(Qt.Horizontal)

        # ── Left: video + controls ───────────────────────────────────────────
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Video players side by side
        video_splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source (trimmed)")
        video_splitter.addWidget(self._source_player)

        self._result_player = VideoPlayerWidget("Result")
        video_splitter.addWidget(self._result_player)

        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        layout.addWidget(video_splitter, 1)

        # Controls scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setMaximumHeight(280)
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(6)

        # Audio input + Chatterbox panel toggle
        audio_row = QHBoxLayout()
        audio_row.setSpacing(4)
        self._audio_drop = FileDropWidget("Audio", extensions=_AUDIO_EXTS)
        audio_row.addWidget(self._audio_drop, 1)

        self._cb_toggle_btn = QPushButton("Audio \u25b6")
        self._cb_toggle_btn.setToolTip("Show/hide Chatterbox audio browser")
        self._cb_toggle_btn.setStyleSheet("font-size: 12px;")
        self._cb_toggle_btn.setFixedWidth(70)
        self._cb_toggle_btn.clicked.connect(self._toggle_cb_panel)
        audio_row.addWidget(self._cb_toggle_btn)

        ctrl_layout.addLayout(audio_row)

        # Face detection
        detect_row = QHBoxLayout()
        detect_row.setSpacing(6)
        self._detect_btn = QPushButton("Detect Faces")
        self._detect_btn.clicked.connect(self._on_detect_faces)
        detect_row.addWidget(self._detect_btn)
        self._detect_status = QLabel("")
        detect_row.addWidget(self._detect_status, 1)
        ctrl_layout.addLayout(detect_row)

        # Face list
        faces_group = QGroupBox("Detected Faces")
        fg_layout = QVBoxLayout(faces_group)
        fg_layout.setContentsMargins(4, 4, 4, 4)
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
        self._face_list.currentRowChanged.connect(self._on_face_selected)
        fg_layout.addWidget(self._face_list)
        ctrl_layout.addWidget(faces_group)

        # Settings row 1
        s1 = QHBoxLayout()
        s1.setSpacing(6)

        s1.addWidget(QLabel("Version:"))
        self._version = QComboBox()
        self._version.addItems(["v1.5", "v1.0"])
        self._version.setFixedWidth(80)
        s1.addWidget(self._version)

        s1.addWidget(QLabel("Blend:"))
        self._blend_mode = QComboBox()
        self._blend_mode.addItems(["jaw", "face", "full"])
        self._blend_mode.setFixedWidth(80)
        s1.addWidget(self._blend_mode)

        s1.addWidget(QLabel("Batch:"))
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 32)
        self._batch_size.setValue(8)
        self._batch_size.setFixedWidth(60)
        s1.addWidget(self._batch_size)

        self._fp16 = QCheckBox("FP16")
        self._fp16.setChecked(True)
        s1.addWidget(self._fp16)

        s1.addStretch()
        ctrl_layout.addLayout(s1)

        # Settings row 2
        s2 = QHBoxLayout()
        s2.setSpacing(6)

        s2.addWidget(QLabel("BBox:"))
        self._bbox_shift = QSpinBox()
        self._bbox_shift.setRange(-20, 20)
        self._bbox_shift.setValue(0)
        self._bbox_shift.setFixedWidth(60)
        s2.addWidget(self._bbox_shift)

        s2.addWidget(QLabel("Margin:"))
        self._extra_margin = QSpinBox()
        self._extra_margin.setRange(0, 50)
        self._extra_margin.setValue(10)
        self._extra_margin.setFixedWidth(60)
        s2.addWidget(self._extra_margin)

        s2.addWidget(QLabel("Pad:"))
        self._face_pad = QSpinBox()
        self._face_pad.setRange(20, 80)
        self._face_pad.setValue(55)
        self._face_pad.setSuffix("%")
        self._face_pad.setFixedWidth(65)
        s2.addWidget(self._face_pad)

        s2.addWidget(QLabel("L:"))
        self._cheek_left = QSpinBox()
        self._cheek_left.setRange(20, 150)
        self._cheek_left.setValue(90)
        self._cheek_left.setFixedWidth(55)
        s2.addWidget(self._cheek_left)

        s2.addWidget(QLabel("R:"))
        self._cheek_right = QSpinBox()
        self._cheek_right.setRange(20, 150)
        self._cheek_right.setValue(90)
        self._cheek_right.setFixedWidth(55)
        s2.addWidget(self._cheek_right)

        s2.addWidget(QLabel("Off:"))
        self._audio_offset = QDoubleSpinBox()
        self._audio_offset.setRange(0.0, 600.0)
        self._audio_offset.setDecimals(2)
        self._audio_offset.setSingleStep(0.1)
        self._audio_offset.setValue(0.0)
        self._audio_offset.setSuffix("s")
        self._audio_offset.setFixedWidth(80)
        s2.addWidget(self._audio_offset)

        s2.addStretch()
        ctrl_layout.addLayout(s2)

        # Generate / Abort / Status row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        self._go_btn = QPushButton("Generate")
        self._go_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._go_btn.clicked.connect(self._on_run)
        action_row.addWidget(self._go_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet("font-weight: bold; font-size: 14px; color: #e55;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        action_row.addWidget(self._abort_btn)

        self._status = QLabel("")
        action_row.addWidget(self._status, 1)

        action_row.addStretch()

        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._on_accept)
        action_row.addWidget(self._accept_btn)

        self._reject_btn = QPushButton("Reject")
        self._reject_btn.clicked.connect(self.reject)
        action_row.addWidget(self._reject_btn)

        ctrl_layout.addLayout(action_row)

        scroll.setWidget(ctrl)
        layout.addWidget(scroll)

        self._main_splitter.addWidget(left)

        # ── Right: Chatterbox audio browser ──────────────────────────────────
        self._cb_panel = self._build_chatterbox_panel()
        self._cb_panel.setVisible(False)
        self._main_splitter.addWidget(self._cb_panel)

        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)

        outer.addWidget(self._main_splitter)

    def _build_chatterbox_panel(self) -> QWidget:
        """Build the Chatterbox TTS audio browser side panel."""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(320)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        header = QLabel("<b>Chatterbox Audio</b>")
        header.setStyleSheet("font-size: 13px; color: #4ca6ff;")
        lay.addWidget(header)

        self._cb_list = QListWidget()
        self._cb_list.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: 1px solid #444; }"
            "QListWidget::item { padding: 3px; }"
            "QListWidget::item:selected { background: #2a5a8a; }"
        )
        self._cb_list.currentRowChanged.connect(self._on_cb_selected)
        lay.addWidget(self._cb_list, 1)

        self._cb_preview = AudioPlayerWidget("Preview")
        lay.addWidget(self._cb_preview)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._cb_use_btn = QPushButton("Use as Audio")
        self._cb_use_btn.setStyleSheet("font-weight: bold;")
        self._cb_use_btn.setToolTip("Load selected Chatterbox clip as MuseTalk audio source")
        self._cb_use_btn.clicked.connect(self._on_cb_use)
        btn_row.addWidget(self._cb_use_btn)

        self._cb_refresh_btn = QPushButton("\u21bb")  # ↻
        self._cb_refresh_btn.setFixedWidth(30)
        self._cb_refresh_btn.setToolTip("Refresh list")
        self._cb_refresh_btn.clicked.connect(self._rescan_chatterbox)
        btn_row.addWidget(self._cb_refresh_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

        return panel

    # -- Chatterbox panel toggle button (injected into action row) ----------

    def _toggle_cb_panel(self) -> None:
        vis = not self._cb_panel.isVisible()
        self._cb_panel.setVisible(vis)
        self._cb_toggle_btn.setText("Audio \u25c0" if vis else "Audio \u25b6")
        if vis:
            self._rescan_chatterbox()

    # -- Chatterbox audio browser ------------------------------------------

    @Slot()
    def _rescan_chatterbox(self) -> None:
        """Scan project chatterbox output dir and populate list."""
        self._cb_list.clear()
        self._cb_files = []

        # Scan all audio subdirs: chatterbox first, then dia
        for subdir in ("chatterbox", "dia"):
            audio_dir = self._project_path / "audio" / subdir
            if not audio_dir.is_dir():
                continue
            files = sorted(
                (f for f in audio_dir.iterdir()
                 if f.suffix.lower() in (".wav", ".mp3", ".flac")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for f in files:
                self._cb_files.append(str(f))
                label = f.name
                if subdir != "chatterbox":
                    label = f"[{subdir}] {f.name}"
                self._cb_list.addItem(label)
                self._cb_list.item(
                    self._cb_list.count() - 1
                ).setToolTip(str(f))

    @Slot(int)
    def _on_cb_selected(self, row: int) -> None:
        if 0 <= row < len(self._cb_files):
            self._cb_preview.load_audio(self._cb_files[row])

    @Slot()
    def _on_cb_use(self) -> None:
        """Load selected Chatterbox audio into the MuseTalk audio input."""
        row = self._cb_list.currentRow()
        if row < 0 or row >= len(self._cb_files):
            return
        path = self._cb_files[row]
        if Path(path).is_file():
            self._audio_drop.load_file(path)
            self._status.setText(f"Audio: {Path(path).name}")

    # -- Pre-trim source clip ------------------------------------------------

    def _pre_trim(self) -> None:
        """Extract the trimmed portion of the clip (stream copy, fast)."""
        clip = self._clip
        src = clip.path
        if not src or not Path(src).is_file():
            self._status.setText("Clip file not found.")
            return

        offset = getattr(clip, "media_offset", 0.0)
        duration = clip.duration

        # If no trimming needed, use source directly
        if offset < 0.01 and abs(duration - getattr(clip, "media_duration", duration)) < 0.01:
            logger.info("Pre-trim: no trim needed, using source directly")
            self._trimmed_path = src
            self._source_player.load_video(src)
            return

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix=f"{clip.name}_trim_"
        )
        tmp.close()

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", src,
                    "-ss", str(offset),
                    "-t", str(duration),
                    "-avoid_negative_ts", "make_zero",
                    tmp.name,
                ],
                capture_output=True,
                timeout=120,
            )
            if Path(tmp.name).is_file() and Path(tmp.name).stat().st_size > 0:
                logger.info("Pre-trim: exported %.2fs from offset %.2fs → %s", duration, offset, tmp.name)
                self._trimmed_path = tmp.name
                self._source_player.load_video(tmp.name)
            else:
                logger.warning("Pre-trim: output file empty or missing")
                self._status.setText("Failed to trim source clip.")
        except Exception as e:
            logger.warning("Pre-trim failed: %s", e)
            self._status.setText("Trim failed, using full source.")
            self._trimmed_path = src
            self._source_player.load_video(src)

    # -- Face Detection -------------------------------------------------------

    @Slot()
    def _on_detect_faces(self) -> None:
        src = self._trimmed_path
        if not src:
            self._detect_status.setText("No source video.")
            return

        models_dir = self.state.global_config.model_paths.get("face_models_dir", "")
        if not models_dir:
            self._detect_status.setText("Face models dir not set in Settings.")
            return

        self._detect_btn.setEnabled(False)
        self._detect_status.setText("Detecting...")
        logger.info("Face detection starting on %s", src)

        worker = VideoFaceDetectWorker(src, models_dir, parent=self)
        worker.finished_ok.connect(self._on_detect_done)
        worker.error.connect(self._on_detect_error)
        worker.finished.connect(worker.deleteLater)
        self._detect_worker = worker
        worker.start()

    def _on_detect_done(self, faces: list[dict]) -> None:
        self._detect_btn.setEnabled(True)
        self._detected_faces = faces
        self._face_list.clear()

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

        logger.info("Face detection found %d face(s)", len(faces))
        self._detect_status.setText(f"Found {len(faces)} face(s). Select one.")
        if faces:
            self._face_list.setCurrentRow(0)

    def _on_detect_error(self, msg: str) -> None:
        logger.error("Face detection failed: %s", msg)
        self._detect_btn.setEnabled(True)
        self._detect_status.setText(f"Error: {msg}")

    @Slot(int)
    def _on_face_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._detected_faces):
            self._selected_bbox = None
            return
        self._selected_bbox = self._detected_faces[row].get("bbox")

    # -- Generation -----------------------------------------------------------

    @Slot()
    def _on_run(self) -> None:
        src = self._trimmed_path
        if not src:
            self._status.setText("No source video.")
            return

        audio = self._audio_drop.file_path
        if not audio:
            self._status.setText("No audio file loaded.")
            return

        if not self._selected_bbox:
            self._status.setText("Detect faces and select one first.")
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "musetalk", self.state.model_registry, self,
            on_progress=lambda f, d: self._status.setText(d),
        ):
            return

        out_dir = self._project_path / "clips" / "lipsync"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = cap_stem(self._clip.name)
        output = str(out_dir / f"{name}_musetalk.mp4")

        # Stop any audio preview to avoid file locks / event loop contention
        self._cb_preview.clear_audio()

        logger.info("MuseTalk generate: src=%s audio=%s bbox=%s output=%s",
                     src, audio, self._selected_bbox, output)
        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._accept_btn.setEnabled(False)
        self._status.setText("Unloading pipelines...")
        self.state.unload_pipelines()
        self._status.setText("Running MuseTalk...")

        from sdqt.workers.musetalk import MuseTalkWorker

        worker = MuseTalkWorker(
            video_path=src,
            audio_path=audio,
            face_bbox=self._selected_bbox,
            output_path=output,
            version=self._version.currentText(),
            blend_mode=self._blend_mode.currentText(),
            batch_size=self._batch_size.value(),
            use_fp16=self._fp16.isChecked(),
            bbox_shift=self._bbox_shift.value(),
            extra_margin=self._extra_margin.value(),
            face_pad_pct=self._face_pad.value(),
            cheek_left=self._cheek_left.value(),
            cheek_right=self._cheek_right.value(),
            audio_offset=self._audio_offset.value(),
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._status.setText(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, result: str) -> None:
        logger.info("MuseTalk complete: %s", result)
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self.result_path = result
        self._result_player.load_video(result, auto_play=True)
        self._accept_btn.setEnabled(True)
        self._status.setText(f"Done: {Path(result).name}")
        self._cleanup_gpu()

    def _on_error(self, msg: str) -> None:
        logger.error("MuseTalk failed: %s", msg)
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._status.setText(f"Error: {msg}")
        self._cleanup_gpu()

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()
            self._status.setText("Aborting...")

    def _cleanup_gpu(self) -> None:
        self._worker = None
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    # -- Accept / Reject ------------------------------------------------------

    @Slot()
    def _on_accept(self) -> None:
        if self.result_path and Path(self.result_path).is_file():
            self.accept()

    # -- Cleanup --------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Abort any running worker
        if self._worker:
            self._worker.abort()
            self._worker = None

        # Stop players before cleanup to avoid crash
        self._cb_preview.clear_audio()
        self._source_player.clear_video()
        self._result_player.clear_video()

        # Clean up trimmed temp file (not the original clip)
        if (
            self._trimmed_path
            and self._trimmed_path != self._clip.path
            and Path(self._trimmed_path).is_file()
        ):
            try:
                Path(self._trimmed_path).unlink()
            except Exception:
                pass

        super().closeEvent(event)
