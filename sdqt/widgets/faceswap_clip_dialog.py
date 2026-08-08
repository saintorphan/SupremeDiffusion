"""Video Face Swap clip dialog — modal face swap for timeline clips."""

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
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.lip_sync import VideoFaceDetectWorker

logger = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_FACE_LIST_STYLE = (
    "QListWidget { background: #1a1a1a; border: 1px solid #444; }"
    "QListWidget::item { padding: 4px; }"
    "QListWidget::item:selected { background: #2a5a8a; border: 2px solid #5aa; }"
)


class FaceSwapClipDialog(QDialog):
    """Modal dialog for face-swapping a single timeline clip."""

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

        self.setWindowTitle(f"Face Swap \u2014 {clip.name}")
        self.setMinimumWidth(700)
        self.resize(800, 580)

        self._build_ui()
        self._pre_trim()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Video players
        video_splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source (trimmed)")
        video_splitter.addWidget(self._source_player)
        self._result_player = VideoPlayerWidget("Result")
        video_splitter.addWidget(self._result_player)
        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        # Controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        ctrl = QWidget()
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(6)

        # Source face image + library buttons
        self._face_source = ImageDropWidget("Source Face Image", thumb_height=120)
        ctrl_layout.addWidget(self._face_source)

        lib_row = QHBoxLayout()
        lib_row.setSpacing(6)
        lib_row.addWidget(QLabel("Load from:"))
        self._face_lib_btn = QPushButton("Face Library")
        self._face_lib_btn.setFixedWidth(100)
        self._face_lib_btn.clicked.connect(self._browse_face_library)
        lib_row.addWidget(self._face_lib_btn)
        self._char_lib_btn = QPushButton("Character Library")
        self._char_lib_btn.setFixedWidth(130)
        self._char_lib_btn.clicked.connect(self._browse_character_library)
        lib_row.addWidget(self._char_lib_btn)
        lib_row.addStretch()
        ctrl_layout.addLayout(lib_row)

        # Face detection (on video first frame)
        detect_row = QHBoxLayout()
        detect_row.setSpacing(6)
        self._detect_btn = QPushButton("Detect Faces in Video")
        self._detect_btn.clicked.connect(self._on_detect_faces)
        detect_row.addWidget(self._detect_btn)
        self._detect_status = QLabel("")
        detect_row.addWidget(self._detect_status, 1)
        ctrl_layout.addLayout(detect_row)

        faces_group = QGroupBox("Detected Faces (target)")
        fg_layout = QVBoxLayout(faces_group)
        fg_layout.setContentsMargins(4, 4, 4, 4)
        self._face_list = QListWidget()
        self._face_list.setFlow(QListWidget.LeftToRight)
        self._face_list.setWrapping(False)
        from PySide6.QtCore import QSize
        self._face_list.setIconSize(QSize(80, 80))
        self._face_list.setFixedHeight(110)
        self._face_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._face_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._face_list.setSelectionMode(QListWidget.SingleSelection)
        self._face_list.setStyleSheet(_FACE_LIST_STYLE)
        self._face_list.currentRowChanged.connect(self._on_face_selected)
        fg_layout.addWidget(self._face_list)
        ctrl_layout.addWidget(faces_group)

        # Settings
        s1 = QHBoxLayout()
        s1.setSpacing(6)
        s1.addWidget(QLabel("Model:"))
        self._model = QComboBox()
        face_dir = Path(self.state.global_config.model_paths.get("face_models_dir", ""))
        for m in ("inswapper_128", "reswapper_128", "reswapper_256"):
            p = face_dir / f"{m}.onnx"
            if p.is_file():  # resolves symlinks — False for broken links
                self._model.addItem(m)
        if self._model.count() == 0:
            self._model.addItem("inswapper_128")
        self._model.setFixedWidth(140)
        s1.addWidget(self._model)

        s1.addWidget(QLabel("Enhancer:"))
        self._enhancer = QComboBox()
        self._enhancer.addItems(["None", "gfpgan", "codeformer", "gpen", "restoreformer"])
        self._enhancer.setFixedWidth(140)
        s1.addWidget(self._enhancer)

        self._swap_all = QCheckBox("Swap All")
        s1.addWidget(self._swap_all)
        s1.addStretch()
        ctrl_layout.addLayout(s1)

        s2 = QHBoxLayout()
        s2.setSpacing(6)
        s2.addWidget(QLabel("Blend:"))
        self._blend = QDoubleSpinBox()
        self._blend.setRange(0, 1)
        self._blend.setDecimals(2)
        self._blend.setSingleStep(0.1)
        self._blend.setValue(0.5)
        self._blend.setFixedWidth(75)
        s2.addWidget(self._blend)

        s2.addWidget(QLabel("Strength:"))
        self._enhancer_strength = QDoubleSpinBox()
        self._enhancer_strength.setRange(0, 1)
        self._enhancer_strength.setDecimals(2)
        self._enhancer_strength.setSingleStep(0.1)
        self._enhancer_strength.setValue(0.5)
        self._enhancer_strength.setFixedWidth(75)
        self._enhancer_strength.setToolTip("CodeFormer fidelity weight")
        s2.addWidget(self._enhancer_strength)

        s2.addStretch()
        ctrl_layout.addLayout(s2)

        # Action row
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

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        main_splitter.addWidget(video_splitter)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter, 1)

    # -- Library browsers --------------------------------------------------

    def _browse_face_library(self) -> None:
        from sdqt.sequences.create_character import _collect_face_library
        pngs = _collect_face_library(self.state)
        if not pngs:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Face Library", "No faces in library yet.")
            return
        from sdqt.tabs.image_tabs.face_swap import _FaceLibraryDialog
        dialog = _FaceLibraryDialog(pngs, self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_path:
            self._face_source.load_image(dialog.selected_path)

    def _browse_character_library(self) -> None:
        from sdqt.sequences.create_character import _browse_character_image
        path = _browse_character_image(self.state, self)
        if path:
            self._face_source.load_image(path)

    # -- Pre-trim --------------------------------------------------------

    def _pre_trim(self) -> None:
        clip = self._clip
        src = clip.path
        if not src or not Path(src).is_file():
            self._status.setText("Clip file not found.")
            return

        offset = getattr(clip, "media_offset", 0.0)
        duration = clip.duration

        if offset < 0.01 and abs(duration - getattr(clip, "media_duration", duration)) < 0.01:
            self._trimmed_path = src
            self._source_player.load_video(src)
            return

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix=f"{clip.name}_trim_")
        tmp.close()

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-ss", str(offset),
                 "-t", str(duration), "-avoid_negative_ts", "make_zero",
                 tmp.name],
                capture_output=True, timeout=120)
            if Path(tmp.name).is_file() and Path(tmp.name).stat().st_size > 0:
                self._trimmed_path = tmp.name
                self._source_player.load_video(tmp.name)
            else:
                self._status.setText("Failed to trim source clip.")
        except Exception as e:
            logger.warning("Pre-trim failed: %s", e)
            self._trimmed_path = src
            self._source_player.load_video(src)

    # -- Face Detection --------------------------------------------------

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
            item.setIcon(QIcon(pixmap.scaled(
                80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            item.setText(f"Face {face['index']}")
            item.setData(Qt.UserRole, face["index"])
            self._face_list.addItem(item)

        self._detect_status.setText(f"Found {len(faces)} face(s). Select one.")
        if faces:
            self._face_list.setCurrentRow(0)

    def _on_detect_error(self, msg: str) -> None:
        self._detect_btn.setEnabled(True)
        self._detect_status.setText(f"Error: {msg}")

    @Slot(int)
    def _on_face_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._detected_faces):
            return

    # -- Generation ------------------------------------------------------

    @Slot()
    def _on_run(self) -> None:
        src_video = self._trimmed_path
        if not src_video:
            self._status.setText("No source video.")
            return

        face_img = self._face_source.image_path
        if not face_img:
            self._status.setText("Drop a source face image.")
            return

        from sdqt.deps import check_and_install
        if not check_and_install("face_swap", self):
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "face_swap", self.state.model_registry, self,
            on_progress=lambda f, d: self._status.setText(d),
        ):
            return

        models_dir = self.state.global_config.model_paths.get("face_models_dir", "")
        if not models_dir:
            self._status.setText("Face models dir not set in Settings.")
            return

        # Verify the selected swap model file exists and is readable
        swap_model_name = self._model.currentText()
        swap_model_path = Path(models_dir) / f"{swap_model_name}.onnx"
        if not swap_model_path.is_file():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Missing Model",
                f"Swap model not found or broken symlink:\n{swap_model_path}\n\n"
                f"Please download {swap_model_name}.onnx to:\n{models_dir}",
            )
            return

        out_dir = self._project_path / "clips" / "faceswap"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = self._clip.name
        output = str(out_dir / f"{name}_faceswap.mp4")

        # Build a ProjectConfig with dialog settings
        from supremediffusion.config.project_config import ProjectConfig
        cfg = ProjectConfig()
        cfg.faceswap_model = self._model.currentText()
        enhancer = self._enhancer.currentText()
        cfg.faceswap_enhancer = "" if enhancer == "None" else enhancer
        cfg.faceswap_blend_ratio = self._blend.value()
        cfg.faceswap_enhancer_strength = self._enhancer_strength.value()
        cfg.faceswap_source_face_idx = 0
        tgt_row = self._face_list.currentRow()
        cfg.faceswap_target_face_idx = tgt_row if tgt_row >= 0 else 0
        cfg.faceswap_swap_all = self._swap_all.isChecked()

        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._accept_btn.setEnabled(False)
        self._status.setText("Unloading pipelines...")
        self.state.unload_pipelines()
        self._status.setText("Running video face swap...")

        from sdqt.workers.image import VideoFaceSwapWorker

        worker = VideoFaceSwapWorker(
            source_path=face_img,
            video_path=src_video,
            project_config=cfg,
            models_dir=models_dir,
            output_path=output,
            enhancer_strength=self._enhancer_strength.value(),
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
        self._accept_btn.setEnabled(True)
        self._status.setText(f"Done: {Path(result).name}")
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

    @Slot()
    def _on_accept(self) -> None:
        if self.result_path and Path(self.result_path).is_file():
            self.accept()

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.abort()
            self._worker = None
        self._source_player.clear_video()
        self._result_player.clear_video()
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
