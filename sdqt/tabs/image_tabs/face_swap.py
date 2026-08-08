"""Face Swap sub-tab for Image Suite."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.tabs.base import BaseTab
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.mask_editor import MaskEditorWidget
from sdqt.workers.image import (
    BatchFaceSwapWorker,
    FaceDetectWorker,
    FaceSwapWorker,
)

logger = logging.getLogger(__name__)

_W_COMBO = 140
_W_DSPIN = 80

_FACE_LIST_STYLE = (
    "QListWidget { background: #1a1a1a; border: 1px solid #444; }"
    "QListWidget::item { padding: 4px; }"
    "QListWidget::item:selected { background: #2a5a8a; border: 2px solid #5aa; }"
)


# ── A/B Comparison Widget ──────────────────────────────────────────────────

class _ABCompareWidget(QWidget):
    """Before/after comparison with a draggable vertical split."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._split = 0.5  # 0..1
        self._dragging = False
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def set_images(self, before_path: str | None, after_path: str | None) -> None:
        self._before = QPixmap(before_path) if before_path else None
        self._after = QPixmap(after_path) if after_path else None
        self._split = 0.5
        self.update()

    def clear(self) -> None:
        self._before = None
        self._after = None
        self.update()

    def paintEvent(self, event) -> None:
        if not self._before or not self._after:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        # Scale both images to fit widget, maintaining aspect ratio
        bw, bh = self._before.width(), self._before.height()
        scale = min(w / bw, h / bh) if bw and bh else 1
        sw, sh = int(bw * scale), int(bh * scale)
        ox, oy = (w - sw) // 2, (h - sh) // 2

        split_x = ox + int(sw * self._split)

        # Draw "after" (full)
        after_scaled = self._after.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap(ox, oy, after_scaled)

        # Draw "before" (left of split)
        before_scaled = self._before.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.setClipRect(ox, oy, split_x - ox, sh)
        p.drawPixmap(ox, oy, before_scaled)
        p.setClipping(False)

        # Draw split line
        p.setPen(Qt.white)
        p.drawLine(split_x, oy, split_x, oy + sh)

        # Labels
        p.setPen(Qt.white)
        p.drawText(ox + 4, oy + 14, "Before")
        p.drawText(split_x + 4, oy + 14, "After")
        p.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._update_split(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_split(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False

    def _update_split(self, x: float) -> None:
        if not self._before:
            return
        bw, bh = self._before.width(), self._before.height()
        w, h = self.width(), self.height()
        scale = min(w / bw, h / bh) if bw and bh else 1
        sw = int(bw * scale)
        ox = (w - sw) // 2
        self._split = max(0.0, min(1.0, (x - ox) / sw)) if sw else 0.5
        self.update()


# ── Face Swap Tab ──────────────────────────────────────────────────────────

class FaceSwapTab(BaseTab):
    """Face swap tab with source/target face detection, multi-source,
    A/B comparison, batch swap, enhancer strength, and video swap."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: FaceSwapWorker | BatchFaceSwapWorker | None = None
        self._detect_worker: FaceDetectWorker | None = None
        self._src_detect_worker: FaceDetectWorker | None = None
        self._detected_faces: list[dict] = []
        self._detected_src_faces: list[dict] = []
        self._source_face_map: dict[int, int] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel: inputs + settings ──────────────────────────────
        left = QWidget()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Source + Target side by side
        images_row = QHBoxLayout()
        images_row.setSpacing(6)
        self._source = ImageDropWidget("Source Face", thumb_height=180)
        self._source.image_loaded.connect(
            lambda p: self._persist_source_path("faceswap_source_path", p))
        self._source.image_cleared.connect(
            lambda: self._persist_source_path("faceswap_source_path", ""))
        self._target = ImageDropWidget("Target Image", thumb_height=180)
        self._target.image_loaded.connect(
            lambda p: self._persist_source_path("faceswap_target_path", p))
        self._target.image_cleared.connect(
            lambda: self._persist_source_path("faceswap_target_path", ""))
        images_row.addWidget(self._source, 1)
        images_row.addWidget(self._target, 1)
        left_layout.addLayout(images_row)

        # Import buttons
        import_row = QHBoxLayout()
        import_row.setSpacing(6)
        self._face_lib_btn = QPushButton("Import from Face Library")
        self._face_lib_btn.setToolTip("Browse and select a face from the Face Library")
        self._face_lib_btn.clicked.connect(self._on_import_face_library)
        import_row.addWidget(self._face_lib_btn)
        self._char_lib_btn = QPushButton("Import from Character Library")
        self._char_lib_btn.setToolTip("Browse characters and load face reference as source")
        self._char_lib_btn.clicked.connect(self._on_import_character_library)
        import_row.addWidget(self._char_lib_btn)
        left_layout.addLayout(import_row)

        # ── Source Face Detection ──────────────────────────────────────
        src_detect_row = QHBoxLayout()
        src_detect_row.setSpacing(6)
        self._src_detect_btn = QPushButton("Detect Source Faces")
        self._src_detect_btn.setToolTip("Detect faces in source image to choose which face to use")
        self._src_detect_btn.clicked.connect(self._on_detect_src_faces)
        src_detect_row.addWidget(self._src_detect_btn)
        self._src_detect_status = QLabel("")
        src_detect_row.addWidget(self._src_detect_status, 1)
        left_layout.addLayout(src_detect_row)

        src_faces_group = QGroupBox("Source Faces")
        sfg_layout = QVBoxLayout(src_faces_group)
        sfg_layout.setContentsMargins(4, 4, 4, 4)
        self._src_face_list = QListWidget()
        self._src_face_list.setFlow(QListWidget.LeftToRight)
        self._src_face_list.setWrapping(False)
        self._src_face_list.setIconSize(QSize(80, 80))
        self._src_face_list.setFixedHeight(130)
        self._src_face_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._src_face_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._src_face_list.setSelectionMode(QListWidget.SingleSelection)
        self._src_face_list.setStyleSheet(_FACE_LIST_STYLE)
        self._src_face_list.currentRowChanged.connect(self._on_src_face_selected)
        sfg_layout.addWidget(self._src_face_list)
        src_hint = QLabel("Select source face for swapping (default: first face).")
        src_hint.setStyleSheet("color: #aaa; font-size: 13px;")
        sfg_layout.addWidget(src_hint)
        left_layout.addWidget(src_faces_group)

        # ── Mask Editor ────────────────────────────────────────────────
        self._mask_check = QCheckBox("Swap Face in Masked Area Only")
        self._mask_check.setToolTip(
            "Paint a mask over the face region to restrict where the swap is applied"
        )
        self._mask_check.toggled.connect(self._on_mask_toggled)
        left_layout.addWidget(self._mask_check)

        self._mask_editor = MaskEditorWidget()
        self._mask_editor.setVisible(False)
        left_layout.addWidget(self._mask_editor, 1)

        self._target.image_loaded.connect(self._on_target_loaded)

        # ── Target Face Detection ──────────────────────────────────────
        detect_row = QHBoxLayout()
        detect_row.setSpacing(6)
        self._detect_btn = QPushButton("Detect Target Faces")
        self._detect_btn.setToolTip("Detect faces in target image")
        self._detect_btn.clicked.connect(self._on_detect_faces)
        detect_row.addWidget(self._detect_btn)
        self._detect_status = QLabel("")
        detect_row.addWidget(self._detect_status, 1)
        left_layout.addLayout(detect_row)

        faces_group = QGroupBox("Target Faces")
        fg_layout = QVBoxLayout(faces_group)
        fg_layout.setContentsMargins(4, 4, 4, 4)
        self._face_list = QListWidget()
        self._face_list.setFlow(QListWidget.LeftToRight)
        self._face_list.setWrapping(False)
        self._face_list.setIconSize(QSize(80, 80))
        self._face_list.setFixedHeight(130)
        self._face_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._face_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._face_list.setSelectionMode(QListWidget.SingleSelection)
        self._face_list.setStyleSheet(_FACE_LIST_STYLE)
        self._face_list.currentRowChanged.connect(self._on_face_selected)
        fg_layout.addWidget(self._face_list)

        hint = QLabel("Select a face to set the target face index for swapping.")
        hint.setStyleSheet("color: #aaa; font-size: 13px;")
        fg_layout.addWidget(hint)

        # Multi-source mapping row (shown when both source + target faces detected)
        self._multi_src_row = QHBoxLayout()
        self._multi_src_row.setSpacing(6)
        self._multi_src_label = QLabel("Assign source:")
        self._multi_src_combo = QComboBox()
        self._multi_src_combo.setFixedWidth(120)
        self._multi_src_combo.currentIndexChanged.connect(self._on_multi_src_changed)
        self._multi_src_row.addWidget(self._multi_src_label)
        self._multi_src_row.addWidget(self._multi_src_combo)
        self._multi_src_row.addStretch()
        self._multi_src_label.setVisible(False)
        self._multi_src_combo.setVisible(False)
        fg_layout.addLayout(self._multi_src_row)

        left_layout.addWidget(faces_group)

        # ── Settings ───────────────────────────────────────────────────
        settings = QGroupBox("Settings")
        sg_layout = QVBoxLayout(settings)
        sg_layout.setContentsMargins(8, 6, 8, 6)
        sg_layout.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Model:"))
        self._model = QComboBox()
        self._model.addItems(["inswapper_128", "reswapper_128", "reswapper_256"])
        self._model.setMinimumWidth(_W_COMBO)
        row1.addWidget(self._model)
        row1.addWidget(QLabel("Enhancer:"))
        self._enhancer = QComboBox()
        self._enhancer.addItems(["None", "gfpgan", "codeformer", "gpen", "restoreformer"])
        self._enhancer.setMinimumWidth(_W_COMBO)
        row1.addWidget(self._enhancer)
        row1.addStretch()
        sg_layout.addLayout(row1)

        # Row 2: blend / strength spinboxes
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Blend:"))
        self._blend = QDoubleSpinBox()
        self._blend.setRange(0, 1)
        self._blend.setDecimals(2)
        self._blend.setSingleStep(0.1)
        self._blend.setValue(0.5)
        self._blend.setMinimumWidth(95)
        self._blend.setToolTip("Enhancer mix: 0 = raw swap only, 1 = full enhancement")
        row2.addWidget(self._blend)

        row2.addWidget(QLabel("Strength:"))
        self._enhancer_strength = QDoubleSpinBox()
        self._enhancer_strength.setRange(0, 1)
        self._enhancer_strength.setDecimals(2)
        self._enhancer_strength.setSingleStep(0.1)
        self._enhancer_strength.setValue(0.5)
        self._enhancer_strength.setMinimumWidth(95)
        self._enhancer_strength.setToolTip(
            "CodeFormer fidelity: 0 = max quality, 1 = max fidelity to original"
        )
        row2.addWidget(self._enhancer_strength)
        row2.addStretch()
        sg_layout.addLayout(row2)

        # Row 3: toggle checkboxes (split off row 2 to avoid cramming)
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self._swap_all = QCheckBox("Swap All")
        row3.addWidget(self._swap_all)
        self._restore_res = QCheckBox("Restore Res")
        self._restore_res.setChecked(True)
        self._restore_res.setToolTip("Upscale result to match the target image resolution")
        row3.addWidget(self._restore_res)

        self._enhance_source = QCheckBox("Enhance Source")
        self._enhance_source.setToolTip(
            "Run GFPGAN on the source face before extracting the identity embedding. "
            "Useful when the source is a noisy / AI-generated portrait. Adds ~1–2 sec "
            "per swap. Has no effect if the source is already very clean."
        )
        row3.addWidget(self._enhance_source)
        row3.addStretch()
        sg_layout.addLayout(row3)

        left_layout.addWidget(settings)
        left_layout.addStretch()

        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ── Right panel: result + A/B compare ──────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # A/B comparison
        self._ab_compare = _ABCompareWidget()
        self._ab_compare.setVisible(False)
        right_layout.addWidget(self._ab_compare, 1)

        # Gallery
        self._gallery = ImageGalleryWidget("Result")
        right_layout.addWidget(self._gallery, 1)

        # Compare toggle
        compare_row = QHBoxLayout()
        self._compare_btn = QPushButton("A/B Compare")
        self._compare_btn.setCheckable(True)
        self._compare_btn.setToolTip("Toggle before/after comparison view")
        self._compare_btn.toggled.connect(self._on_compare_toggled)
        compare_row.addWidget(self._compare_btn)
        compare_row.addStretch()
        right_layout.addLayout(compare_row)

        btn_row = QHBoxLayout()
        self._swap_btn = QPushButton("Swap Faces")
        self._swap_btn.setObjectName("primary")
        self._swap_btn.clicked.connect(self._on_swap)
        btn_row.addWidget(self._swap_btn)
        self._swap_again_btn = QPushButton("Swap Again")
        self._swap_again_btn.setToolTip("Re-run the swap with current settings")
        self._swap_again_btn.setVisible(False)
        self._swap_again_btn.clicked.connect(self._on_swap)
        btn_row.addWidget(self._swap_again_btn)
        self._batch_btn = QPushButton("Batch")
        self._batch_btn.setToolTip("Swap face across a folder of target images")
        self._batch_btn.clicked.connect(self._on_batch_swap)
        btn_row.addWidget(self._batch_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)
        self._unload_btn = QPushButton("Unload SD")
        self._unload_btn.setToolTip("Free SD image model from VRAM")
        self._unload_btn.clicked.connect(self._on_unload)
        btn_row.addWidget(self._unload_btn)
        right_layout.addLayout(btn_row)

        save_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Result")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self._save_btn)
        right_layout.addLayout(save_row)

        self._status = QLabel("")
        right_layout.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._result_path: str | None = None
        self._src_face_idx = 0
        self._tgt_face_idx = 0

    def _persist_source_path(self, attr: str, path: str) -> None:
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            setattr(cfg, attr, path)
            cfg.save(self.project_path)
        except Exception:
            pass

    def load_source(self, path: str) -> None:
        """Load a face image as the source."""
        self._source.load_image(path)

    def load_target(self, path: str) -> None:
        """Load an image as the swap target."""
        self._target.load_image(path)

    # -- Face Library import -------------------------------------------

    @Slot()
    def _on_import_face_library(self) -> None:
        face_dir = getattr(self.state.global_config, "face_library_dir", "")
        if not face_dir:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Face Library",
                "No face library directory configured.\nSet path in Settings.",
            )
            return
        face_path = Path(face_dir)
        if not face_path.is_dir():
            face_path.mkdir(parents=True, exist_ok=True)
        pngs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            pngs.extend(face_path.glob(ext))
        pngs = sorted(pngs)
        if not pngs:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Face Library",
                "No faces in library yet.\nSave faces from the Draw tab.",
            )
            return

        dialog = _FaceLibraryDialog(pngs, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path:
            self._source.load_image(dialog.selected_path)

    # -- Character Library import --------------------------------------

    @Slot()
    def _on_import_character_library(self) -> None:
        char_dir = getattr(self.state.global_config, "characters_dir", "")
        if not char_dir:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Character Library",
                "No character library directory configured.\nSet path in Settings.",
            )
            return
        char_path = Path(char_dir)
        if not char_path.is_dir():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Character Library",
                "Character library is empty.",
            )
            return
        from sdqt.widgets.library_dialogs import CharacterLibraryDialog
        dialog = CharacterLibraryDialog(char_path, self, prefer_face=True)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path:
            self._source.load_image(dialog.selected_path)

    # -- Mask ----------------------------------------------------------

    @Slot(bool)
    def _on_mask_toggled(self, checked: bool) -> None:
        self._mask_editor.setVisible(checked)
        if checked:
            tgt = self._target.image_path
            if tgt:
                self._mask_editor.load_image(tgt)

    @Slot(str)
    def _on_target_loaded(self, path: str) -> None:
        if self._mask_check.isChecked():
            self._mask_editor.load_image(path)

    # -- A/B Compare ---------------------------------------------------

    @Slot(bool)
    def _on_compare_toggled(self, checked: bool) -> None:
        self._ab_compare.setVisible(checked)
        self._gallery.setVisible(not checked)
        if checked and self._result_path:
            tgt = self._target.image_path
            self._ab_compare.set_images(tgt, self._result_path)

    # -- Source Face Detection -----------------------------------------

    def _get_models_dir(self) -> str | None:
        models_dir = self.state.global_config.model_paths.get("face_models_dir", "")
        if not models_dir:
            self._show_status("Face models directory not set in Settings.")
            return None
        return models_dir

    @Slot()
    def _on_detect_src_faces(self) -> None:
        src = self._source.image_path
        if not src:
            self._src_detect_status.setText("Load a source image first.")
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "face_swap", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        models_dir = self._get_models_dir()
        if not models_dir:
            return

        self._src_detect_btn.setEnabled(False)
        self._src_detect_status.setText("Detecting...")

        worker = FaceDetectWorker(src, models_dir, parent=self)
        worker.finished_ok.connect(self._on_src_detect_done)
        worker.error.connect(self._on_src_detect_error)
        worker.finished.connect(worker.deleteLater)
        self._src_detect_worker = worker
        worker.start()

    def _on_src_detect_done(self, faces: list[dict]) -> None:
        self._src_detect_btn.setEnabled(True)
        self._detected_src_faces = faces
        self._src_face_list.clear()

        if not faces:
            self._src_detect_status.setText("No faces detected in source.")
            return

        self._populate_face_list(self._src_face_list, faces)
        self._src_detect_status.setText(f"Found {len(faces)} face(s). Select one.")
        if faces:
            self._src_face_list.setCurrentRow(0)
        self._update_multi_src_ui()

    def _on_src_detect_error(self, msg: str) -> None:
        self._src_detect_btn.setEnabled(True)
        self._src_detect_status.setText(f"Error: {msg}")

    @Slot(int)
    def _on_src_face_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._detected_src_faces):
            return
        self._src_face_idx = self._detected_src_faces[row]["index"]

    # -- Target Face Detection -----------------------------------------

    @Slot()
    def _on_detect_faces(self) -> None:
        tgt = self._target.image_path
        if not tgt:
            self._detect_status.setText("Load a target image first.")
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "face_swap", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        models_dir = self._get_models_dir()
        if not models_dir:
            return

        self._detect_btn.setEnabled(False)
        self._detect_status.setText("Detecting...")

        worker = FaceDetectWorker(tgt, models_dir, parent=self)
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

        self._populate_face_list(self._face_list, faces)
        self._detect_status.setText(f"Found {len(faces)} face(s). Select one.")
        if faces:
            self._face_list.setCurrentRow(0)
        self._update_multi_src_ui()

    def _on_detect_error(self, msg: str) -> None:
        self._detect_btn.setEnabled(True)
        self._detect_status.setText(f"Error: {msg}")

    @Slot(int)
    def _on_face_selected(self, row: int) -> None:
        if row < 0:
            return
        item = self._face_list.item(row)
        if item:
            self._tgt_face_idx = item.data(Qt.UserRole)
            self._update_multi_src_combo_for_face(self._tgt_face_idx)

    # -- Shared face list helper ---------------------------------------

    @staticmethod
    def _populate_face_list(list_widget: QListWidget, faces: list[dict]) -> None:
        for face in faces:
            pil_img = face.get("crop_image")
            if pil_img is None:
                continue
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            data = pil_img.tobytes("raw", "RGB")
            qimg = QImage(data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap.scaled(
                80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            item.setText(f"Face {face['index']}")
            item.setData(Qt.UserRole, face["index"])
            list_widget.addItem(item)

    # -- Multi-source mapping ------------------------------------------

    def _update_multi_src_ui(self) -> None:
        """Show/hide multi-source combo when both sides have faces detected."""
        has_both = (
            len(self._detected_src_faces) > 1
            and len(self._detected_faces) > 0
            and self._swap_all.isChecked()
        )
        self._multi_src_label.setVisible(has_both)
        self._multi_src_combo.setVisible(has_both)
        if has_both:
            self._update_multi_src_combo_for_face(self._tgt_face_idx)

    def _update_multi_src_combo_for_face(self, tgt_idx: int) -> None:
        """Update the source face combo for the currently selected target face."""
        if not self._detected_src_faces:
            return
        self._multi_src_combo.blockSignals(True)
        self._multi_src_combo.clear()
        for f in self._detected_src_faces:
            self._multi_src_combo.addItem(f"Source Face {f['index']}", f["index"])
        # Restore mapping if exists
        mapped = self._source_face_map.get(tgt_idx, 0)
        for i in range(self._multi_src_combo.count()):
            if self._multi_src_combo.itemData(i) == mapped:
                self._multi_src_combo.setCurrentIndex(i)
                break
        self._multi_src_combo.blockSignals(False)

    @Slot(int)
    def _on_multi_src_changed(self, idx: int) -> None:
        if idx < 0:
            return
        src_idx = self._multi_src_combo.itemData(idx)
        if src_idx is not None:
            self._source_face_map[self._tgt_face_idx] = src_idx

    # -- Face Swap (single) --------------------------------------------

    def _prepare_swap(self) -> tuple[str, str, str, ProjectConfig] | None:
        """Validate inputs and return (src, tgt, models_dir, cfg) or None."""
        src = self._source.image_path
        tgt = self._target.image_path
        if not src or not tgt:
            self._show_status("Both source and target images required.")
            return None

        from sdqt.deps import check_and_install
        if not check_and_install("face_swap", self):
            return None

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "face_swap", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return None

        models_dir = self._get_models_dir()
        if not models_dir:
            return None

        cfg = ProjectConfig.load(self.project_path)
        cfg.faceswap_source_path = src
        cfg.faceswap_target_path = tgt
        cfg.faceswap_model = self._model.currentText()
        enhancer = self._enhancer.currentText()
        cfg.faceswap_enhancer = "" if enhancer == "None" else enhancer
        cfg.faceswap_blend_ratio = self._blend.value()
        cfg.faceswap_enhancer_strength = self._enhancer_strength.value()
        cfg.faceswap_source_face_idx = self._src_face_idx
        cfg.faceswap_target_face_idx = self._tgt_face_idx
        cfg.faceswap_swap_all = self._swap_all.isChecked()
        cfg.faceswap_enhance_source = self._enhance_source.isChecked()
        cfg.save(self.project_path)

        return src, tgt, models_dir, cfg

    @Slot()
    def _on_swap(self) -> None:
        prepared = self._prepare_swap()
        if not prepared:
            return
        src, tgt, models_dir, cfg = prepared

        if not self.acquire_gpu("Image"):
            return

        self._swap_btn.setVisible(False)
        self._swap_again_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Swapping faces...")

        # Build source face map for multi-source
        face_map = self._source_face_map if self._source_face_map else None

        worker = FaceSwapWorker(
            src, tgt, cfg, models_dir,
            enhancer_strength=self._enhancer_strength.value(),
            source_face_map=face_map,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, result_path: str) -> None:
        self.release_gpu("Image")
        self._abort_btn.setVisible(False)
        if not result_path:
            # Nothing was produced — keep the primary Swap button visible
            # rather than implying a successful result exists.
            self._swap_btn.setVisible(True)
            self._swap_again_btn.setVisible(False)
            self._show_status("Face swap produced no result.")
            return
        self._swap_btn.setVisible(False)
        self._swap_again_btn.setVisible(True)

        # Apply mask compositing if enabled
        if self._mask_check.isChecked():
            mask_path = self._mask_editor.get_mask_path()
            tgt_path = self._target.image_path
            if mask_path and tgt_path:
                result_path = self._apply_mask_composite(
                    result_path, tgt_path, mask_path)

        # Restore to target resolution if enabled
        if self._restore_res.isChecked():
            tgt_path = self._target.image_path
            if tgt_path:
                result_path = self._restore_resolution(result_path, tgt_path)

        # Persist result to project
        saved = self._persist_result(result_path)
        self._result_path = saved or result_path
        self._gallery.load_images(self._list_persisted_results())
        self._save_btn.setEnabled(True)
        self._show_status("Face swap complete.")

        # Update A/B compare if visible
        if self._compare_btn.isChecked():
            self._ab_compare.set_images(
                self._target.image_path, self._result_path)

    # -- Batch Swap ----------------------------------------------------

    @Slot()
    def _on_batch_swap(self) -> None:
        src = self._source.image_path
        if not src:
            self._show_status("Load a source face first.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select folder of target images")
        if not folder:
            return

        targets = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )
        if not targets:
            self._show_status("No images found in selected folder.")
            return

        from sdqt.deps import check_and_install
        if not check_and_install("face_swap", self):
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "face_swap", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        models_dir = self._get_models_dir()
        if not models_dir:
            return

        cfg = ProjectConfig.load(self.project_path)
        cfg.faceswap_model = self._model.currentText()
        enhancer = self._enhancer.currentText()
        cfg.faceswap_enhancer = "" if enhancer == "None" else enhancer
        cfg.faceswap_blend_ratio = self._blend.value()
        cfg.faceswap_enhancer_strength = self._enhancer_strength.value()
        cfg.faceswap_source_face_idx = self._src_face_idx
        cfg.faceswap_target_face_idx = self._tgt_face_idx
        cfg.faceswap_swap_all = self._swap_all.isChecked()
        cfg.faceswap_enhance_source = self._enhance_source.isChecked()

        out_dir = str(self._results_dir() / "batch")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        if not self.acquire_gpu("Image"):
            return

        self._swap_btn.setVisible(False)
        self._swap_again_btn.setVisible(False)
        self._batch_btn.setEnabled(False)
        self._abort_btn.setVisible(True)
        self._show_status(f"Batch swap: {len(targets)} images...")

        worker = BatchFaceSwapWorker(
            src, targets, cfg, models_dir, out_dir,
            enhancer_strength=self._enhancer_strength.value(),
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_batch_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_batch_done(self, results: list[str]) -> None:
        self.release_gpu("Image")
        self._swap_btn.setVisible(True)
        self._swap_again_btn.setVisible(False)
        self._batch_btn.setEnabled(True)
        self._abort_btn.setVisible(False)
        if results:
            self._result_path = results[-1]
            self._gallery.load_images(results)
            self._save_btn.setEnabled(True)
        self._show_status(f"Batch complete — {len(results)} images swapped.")

    # -- Helpers -------------------------------------------------------

    @staticmethod
    def _restore_resolution(result_path: str, target_path: str) -> str:
        import tempfile
        from PIL import Image

        result = Image.open(result_path)
        target = Image.open(target_path)
        if result.size == target.size:
            return result_path

        upscaled = result.resize(target.size, Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        upscaled.save(tmp.name)
        tmp.close()
        return tmp.name

    @staticmethod
    def _apply_mask_composite(
        swapped_path: str, original_path: str, mask_path: str,
    ) -> str:
        import tempfile
        import numpy as np
        from PIL import Image

        swapped = np.array(
            Image.open(swapped_path).convert("RGB"), dtype=np.float32)
        original = np.array(
            Image.open(original_path).convert("RGB"), dtype=np.float32)
        mask = np.array(
            Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0

        h, w = original.shape[:2]
        if mask.shape != (h, w):
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((w, h), PILImage.LANCZOS)
            mask = np.array(mask_pil, dtype=np.float32) / 255.0

        if swapped.shape[:2] != (h, w):
            from PIL import Image as PILImage
            sw_pil = PILImage.fromarray(swapped.astype(np.uint8))
            sw_pil = sw_pil.resize((w, h), PILImage.LANCZOS)
            swapped = np.array(sw_pil, dtype=np.float32)

        mask_3ch = mask[:, :, np.newaxis]
        composited = mask_3ch * swapped + (1.0 - mask_3ch) * original
        result = Image.fromarray(composited.astype(np.uint8))

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        result.save(tmp.name)
        tmp.close()
        return tmp.name

    def _on_error(self, msg: str) -> None:
        self.release_gpu("Image")
        if self._result_path:
            self._swap_again_btn.setVisible(True)
        else:
            self._swap_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._batch_btn.setEnabled(True)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    @Slot()
    def _on_unload(self) -> None:
        self.state.unload_sd_pipelines()
        self._show_status("SD pipelines unloaded.")

    def _results_dir(self) -> Path:
        d = self.project_path / "images" / "faceswap"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _persist_result(self, path: str) -> str | None:
        name = datetime.utcnow().strftime("faceswap_%Y%m%d_%H%M%S.png")
        dest = self._results_dir() / name
        shutil.copy2(path, dest)
        return str(dest)

    def _list_persisted_results(self) -> list[str]:
        d = self.project_path / "images" / "faceswap"
        if not d.is_dir():
            return []
        return sorted(str(p) for p in d.glob("*.png"))

    @Slot()
    def _on_save(self) -> None:
        if self._result_path:
            saved = self.save_frame_to_project(self._result_path)
            if saved:
                self._show_status("Saved to project.")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._gallery.clear()
        self._ab_compare.clear()
        self._face_list.clear()
        self._src_face_list.clear()
        self._detected_faces.clear()
        self._detected_src_faces.clear()
        self._source_face_map.clear()
        self._mask_editor.clear_mask()
        self._result_path = None
        self._swap_btn.setVisible(True)
        self._swap_again_btn.setVisible(False)
        self._save_btn.setEnabled(False)
        self._multi_src_label.setVisible(False)
        self._multi_src_combo.setVisible(False)
        self._restoring = True

        # Restore persisted results
        persisted = self._list_persisted_results()
        if persisted:
            self._result_path = persisted[-1]
            self._gallery.load_images(persisted)
            self._swap_btn.setVisible(False)
            self._swap_again_btn.setVisible(True)
            self._save_btn.setEnabled(True)
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        src = cfg.faceswap_source_path or ""
        tgt = cfg.faceswap_target_path or ""
        if src and Path(src).is_file():
            self._source.load_image(src)
        else:
            self._source.clear_image()
        if tgt and Path(tgt).is_file():
            self._target.load_image(tgt)
        else:
            self._target.clear_image()
        idx = self._model.findText(cfg.faceswap_model or "inswapper_128")
        if idx >= 0:
            self._model.setCurrentIndex(idx)
        enhancer = cfg.faceswap_enhancer or ""
        idx = self._enhancer.findText(enhancer if enhancer else "None")
        if idx >= 0:
            self._enhancer.setCurrentIndex(idx)
        self._blend.setValue(cfg.faceswap_blend_ratio)
        self._enhancer_strength.setValue(cfg.faceswap_enhancer_strength)
        self._src_face_idx = cfg.faceswap_source_face_idx
        self._tgt_face_idx = cfg.faceswap_target_face_idx
        self._swap_all.setChecked(cfg.faceswap_swap_all)
        self._enhance_source.setChecked(getattr(cfg, "faceswap_enhance_source", False))
        self._restoring = False


class _FaceLibraryDialog(QDialog):
    """Thumbnail grid dialog for selecting a face from the Face Library."""

    _THUMB = 80

    def __init__(self, png_paths: list[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Face Library")
        self.setMinimumSize(400, 340)
        self.selected_path: str | None = None

        layout = QVBoxLayout(self)
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(self._THUMB, self._THUMB))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setSpacing(6)
        self._grid.setWrapping(True)
        self._grid.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._grid.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._grid, 1)

        for p in png_paths:
            pixmap = QPixmap(str(p))
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                self._THUMB, self._THUMB,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb))
            item.setText(p.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setToolTip(str(p))
            self._grid.addItem(item)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        item = self._grid.currentItem()
        if item:
            self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
