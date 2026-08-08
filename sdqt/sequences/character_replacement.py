"""Character Replacement — 5-page wizard for face-swapping a character into a video."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class ReplacementState:
    # Source video
    video_path: str = ""
    # Character source
    source_mode: str = "image"  # "image" or "character"
    face_source_path: str = ""  # direct image path
    character_name: str = ""
    character_dir: str = ""
    # Face detection
    sample_frame_path: str = ""  # extracted first frame
    detected_faces: list = field(default_factory=list)
    source_face_idx: int = 0
    target_face_idx: int = 0
    # Settings
    swap_model: str = "inswapper_128"
    enhancer: str = "None"
    blend_ratio: float = 0.5
    enhancer_strength: float = 0.5
    swap_all: bool = False
    # Output
    output_path: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_path(app_state) -> Path:
    """Return the current project directory path."""
    name = app_state.current_project or "_default"
    return app_state.project_manager.get_project_path(name)


_FACE_IMAGE_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.webp"}


def _collect_face_library(app_state) -> list[Path]:
    """Collect face images from both config dir and FaceLibraryTab computed dir."""
    seen = set()
    results = []
    dirs = []
    face_dir = getattr(app_state.global_config, "face_library_dir", "")
    if face_dir:
        dirs.append(Path(face_dir))
    projects_root = getattr(app_state.global_config, "projects_root", "")
    if projects_root:
        dirs.append(Path(projects_root).parent / "library" / "faces")
    for d in dirs:
        if not d.is_dir():
            continue
        for pattern in _FACE_IMAGE_EXTS:
            for f in sorted(d.glob(pattern)):
                if f.is_file() and str(f) not in seen:
                    seen.add(str(f))
                    results.append(f)
    return results


def _extract_first_frame(video_path: str) -> str:
    """Extract the first frame of a video to a temp PNG. Returns the path."""
    from supremediffusion.utils.video import extract_single_frame
    img = extract_single_frame(video_path, 0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    img.save(tmp.name, "PNG")
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 — Video Source
# ═══════════════════════════════════════════════════════════════════════════


class VideoSourcePage(WizardPage):
    """Select a video file and preview its first frame."""

    def __init__(self, rs: ReplacementState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._rs = rs
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Browse row
        row = QHBoxLayout()
        row.addWidget(QLabel("Video:"))
        self._path_label = QLabel("(none)")
        self._path_label.setStyleSheet("color: #ccc;")
        row.addWidget(self._path_label, 1)
        browse_btn = QPushButton("Select Video")
        browse_btn.setFixedWidth(120)
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        # Preview
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(240)
        self._preview.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._preview, 1)

        layout.addStretch()

    def on_enter(self) -> None:
        if self._rs.video_path:
            self._path_label.setText(self._rs.video_path)
            self._show_frame()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)",
        )
        if path:
            self._rs.video_path = path
            self._path_label.setText(path)
            self._extract_and_show()

    def _extract_and_show(self) -> None:
        try:
            frame_path = _extract_first_frame(self._rs.video_path)
            self._rs.sample_frame_path = frame_path
            self._show_frame()
        except Exception as exc:
            logger.warning("Failed to extract first frame: %s", exc)
            self._wizard._status.setText(f"Frame extraction failed: {exc}")

    def _show_frame(self) -> None:
        if self._rs.sample_frame_path and Path(self._rs.sample_frame_path).is_file():
            pm = QPixmap(self._rs.sample_frame_path).scaled(
                640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)

    def validate(self) -> bool:
        if not self._rs.video_path or not Path(self._rs.video_path).is_file():
            QMessageBox.warning(self, "Missing", "Please select a valid video file.")
            return False
        # Ensure we have a sample frame
        if not self._rs.sample_frame_path or not Path(self._rs.sample_frame_path).is_file():
            try:
                self._rs.sample_frame_path = _extract_first_frame(self._rs.video_path)
            except Exception:
                QMessageBox.warning(self, "Error", "Could not extract a frame from the video.")
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Character / Face Source
# ═══════════════════════════════════════════════════════════════════════════


class CharacterSourcePage(WizardPage):
    """Choose a face source: direct image or saved character."""

    def __init__(self, rs: ReplacementState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._rs = rs
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Radio buttons
        self._radio_group = QButtonGroup(self)
        self._radio_image = QRadioButton("Use Image")
        self._radio_char = QRadioButton("Use Saved Character")
        self._radio_group.addButton(self._radio_image, 0)
        self._radio_group.addButton(self._radio_char, 1)
        self._radio_image.setChecked(True)
        self._radio_group.idToggled.connect(self._on_mode_changed)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self._radio_image)
        radio_row.addWidget(self._radio_char)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # --- Image mode ---
        self._image_widget = QWidget()
        img_layout = QVBoxLayout(self._image_widget)
        img_layout.setContentsMargins(0, 0, 0, 0)

        browse_row = QHBoxLayout()
        browse_row.addWidget(QLabel("Face Image:"))
        self._img_path_label = QLabel("(none)")
        self._img_path_label.setStyleSheet("color: #ccc;")
        browse_row.addWidget(self._img_path_label, 1)
        img_browse = QPushButton("Browse")
        img_browse.setFixedWidth(75)
        img_browse.clicked.connect(self._browse_image)
        browse_row.addWidget(img_browse)
        self._face_lib_btn = QPushButton("Face Library")
        self._face_lib_btn.setFixedWidth(90)
        self._face_lib_btn.clicked.connect(self._browse_face_library)
        browse_row.addWidget(self._face_lib_btn)
        self._char_lib_btn = QPushButton("Character Library")
        self._char_lib_btn.setFixedWidth(120)
        self._char_lib_btn.clicked.connect(self._browse_from_characters)
        browse_row.addWidget(self._char_lib_btn)
        img_layout.addLayout(browse_row)

        self._img_preview = QLabel()
        self._img_preview.setAlignment(Qt.AlignCenter)
        self._img_preview.setFixedHeight(200)
        self._img_preview.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        img_layout.addWidget(self._img_preview)

        layout.addWidget(self._image_widget)

        # --- Character mode ---
        self._char_widget = QWidget()
        char_layout = QVBoxLayout(self._char_widget)
        char_layout.setContentsMargins(0, 0, 0, 0)

        char_layout.addWidget(QLabel("Select a saved character:"))
        self._char_list = QListWidget()
        self._char_list.setMinimumHeight(200)
        self._char_list.currentItemChanged.connect(self._on_char_selected)
        char_layout.addWidget(self._char_list)

        self._char_preview = QLabel()
        self._char_preview.setAlignment(Qt.AlignCenter)
        self._char_preview.setFixedHeight(120)
        char_layout.addWidget(self._char_preview)

        layout.addWidget(self._char_widget)
        self._char_widget.hide()

        layout.addStretch()

    def on_enter(self) -> None:
        if self._rs.source_mode == "character":
            self._radio_char.setChecked(True)
        else:
            self._radio_image.setChecked(True)
        self._on_mode_changed(self._radio_group.checkedId(), True)

        if self._rs.face_source_path and Path(self._rs.face_source_path).is_file():
            if self._rs.source_mode == "image":
                self._img_path_label.setText(self._rs.face_source_path)
                self._show_image_preview(self._rs.face_source_path)

    def _on_mode_changed(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        if btn_id == 0:
            self._image_widget.show()
            self._char_widget.hide()
            self._rs.source_mode = "image"
        else:
            self._image_widget.hide()
            self._char_widget.show()
            self._rs.source_mode = "character"
            self._scan_characters()

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Source", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._rs.face_source_path = path
            self._img_path_label.setText(path)
            self._show_image_preview(path)

    def _browse_face_library(self) -> None:
        pngs = _collect_face_library(self._app_state)
        if not pngs:
            QMessageBox.information(self, "Face Library", "No faces in library yet.")
            return
        from sdqt.tabs.image_tabs.face_swap import _FaceLibraryDialog
        from PySide6.QtWidgets import QDialog as _QDlg
        dialog = _FaceLibraryDialog(pngs, self)
        if dialog.exec() == _QDlg.Accepted and dialog.selected_path:
            self._rs.face_source_path = dialog.selected_path
            self._img_path_label.setText(dialog.selected_path)
            self._show_image_preview(dialog.selected_path)

    def _browse_from_characters(self) -> None:
        from sdqt.sequences.create_character import _browse_character_image
        path = _browse_character_image(self._app_state, self)
        if path:
            self._rs.face_source_path = path
            self._img_path_label.setText(path)
            self._show_image_preview(path)

    def _show_image_preview(self, path: str) -> None:
        if Path(path).is_file():
            pm = QPixmap(path).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._img_preview.setPixmap(pm)

    def _scan_characters(self) -> None:
        self._char_list.clear()
        characters_dir = self._app_state.global_config.characters_dir
        if not characters_dir:
            from supremediffusion.config.defaults import APP_ROOT
            characters_dir = str(APP_ROOT / "library" / "characters")

        chars_path = Path(characters_dir)
        if not chars_path.is_dir():
            return

        for char_dir in sorted(chars_path.iterdir()):
            meta_file = char_dir / "character.json"
            if not meta_file.is_file():
                continue
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                name = meta.get("name", char_dir.name)
            except Exception:
                name = char_dir.name

            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, str(char_dir))

            # Thumbnail
            thumb_path = char_dir / "base.png"
            if thumb_path.is_file():
                from PySide6.QtGui import QIcon
                item.setIcon(QIcon(str(thumb_path)))

            self._char_list.addItem(item)

    def _on_char_selected(self, current: QListWidgetItem, previous) -> None:
        if current is None:
            return
        char_dir = Path(current.data(Qt.UserRole))
        self._rs.character_name = current.text()
        self._rs.character_dir = str(char_dir)

        # Prefer face_ref.png, fall back to base.png
        face_ref = char_dir / "face_ref.png"
        base_img = char_dir / "base.png"
        if face_ref.is_file():
            self._rs.face_source_path = str(face_ref)
        elif base_img.is_file():
            self._rs.face_source_path = str(base_img)

        # Show preview
        if self._rs.face_source_path and Path(self._rs.face_source_path).is_file():
            pm = QPixmap(self._rs.face_source_path).scaled(
                200, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._char_preview.setPixmap(pm)

    def validate(self) -> bool:
        if not self._rs.face_source_path or not Path(self._rs.face_source_path).is_file():
            QMessageBox.warning(self, "Missing", "Please select a face source image or character.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Face Detection
# ═══════════════════════════════════════════════════════════════════════════


class FaceDetectionPage(WizardPage):
    """Detect faces in the video frame and let the user pick source/target indices."""

    def __init__(self, rs: ReplacementState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._rs = rs
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Detect button
        self._detect_btn = QPushButton("Detect Faces")
        self._detect_btn.setFixedWidth(140)
        self._detect_btn.clicked.connect(self._detect)
        layout.addWidget(self._detect_btn)

        # Preview with bboxes
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(280)
        self._preview.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._preview, 1)

        # Face index controls
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)

        ctrl_row.addWidget(QLabel("Source Face Index:"))
        self._src_spin = QSpinBox()
        self._src_spin.setFixedWidth(75)
        self._src_spin.setRange(0, 0)
        self._src_spin.setToolTip("Which face in the SOURCE image (usually 0)")
        self._src_spin.valueChanged.connect(lambda v: setattr(self._rs, "source_face_idx", v))
        ctrl_row.addWidget(self._src_spin)

        ctrl_row.addWidget(QLabel("Target Face Index:"))
        self._tgt_spin = QSpinBox()
        self._tgt_spin.setFixedWidth(75)
        self._tgt_spin.setRange(0, 0)
        self._tgt_spin.setToolTip("Which face in the VIDEO to replace (0 = first detected)")
        self._tgt_spin.valueChanged.connect(lambda v: setattr(self._rs, "target_face_idx", v))
        ctrl_row.addWidget(self._tgt_spin)

        self._swap_all_cb = QCheckBox("Swap All Faces")
        self._swap_all_cb.toggled.connect(lambda v: setattr(self._rs, "swap_all", v))
        ctrl_row.addWidget(self._swap_all_cb)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        layout.addStretch()

    def on_enter(self) -> None:
        # Show the sample frame without bboxes initially
        if self._rs.sample_frame_path and Path(self._rs.sample_frame_path).is_file():
            pm = QPixmap(self._rs.sample_frame_path).scaled(
                640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)

        # Restore state
        self._src_spin.setValue(self._rs.source_face_idx)
        self._tgt_spin.setValue(self._rs.target_face_idx)
        self._swap_all_cb.setChecked(self._rs.swap_all)

        if self._rs.detected_faces:
            self._draw_bboxes()

    def _detect(self) -> None:
        if not self._rs.sample_frame_path or not Path(self._rs.sample_frame_path).is_file():
            QMessageBox.warning(self, "Missing", "No sample frame available. Go back and select a video.")
            return

        from sdqt.workers.image import FaceDetectWorker

        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")
        worker = FaceDetectWorker(
            image_path=self._rs.sample_frame_path,
            models_dir=models_dir,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_detected, on_error=self._on_error)

    @Slot(object)
    def _on_detected(self, faces) -> None:
        self._rs.detected_faces = list(faces) if faces else []
        count = len(self._rs.detected_faces)
        self._wizard._status.setText(f"Detected {count} face(s)")

        # Update spin ranges
        self._tgt_spin.setRange(0, max(0, count - 1))
        self._src_spin.setRange(0, 10)  # source image may have multiple faces

        self._draw_bboxes()

    def _draw_bboxes(self) -> None:
        if not self._rs.sample_frame_path or not Path(self._rs.sample_frame_path).is_file():
            return

        pm = QPixmap(self._rs.sample_frame_path)
        painter = QPainter(pm)
        pen = QPen(QColor("#00ff00"), 2)
        painter.setPen(pen)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)

        for i, face in enumerate(self._rs.detected_faces):
            bbox = face.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 5, str(i))

        painter.end()

        scaled = pm.scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._preview.setPixmap(scaled)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Face detection error: {msg}")

    def validate(self) -> bool:
        if not self._rs.detected_faces:
            QMessageBox.warning(self, "Missing", "Please detect faces before continuing.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Swap Settings
# ═══════════════════════════════════════════════════════════════════════════


class SwapSettingsPage(WizardPage):
    """Configure face swap model, enhancer, and blending parameters."""

    def __init__(self, rs: ReplacementState, parent=None) -> None:
        super().__init__(parent)
        self._rs = rs
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Swap model
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Swap Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setFixedWidth(160)
        self._model_combo.addItem("inswapper_128")
        row1.addWidget(self._model_combo)
        row1.addStretch()
        layout.addLayout(row1)

        # Enhancer
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Enhancer:"))
        self._enhancer_combo = QComboBox()
        self._enhancer_combo.setFixedWidth(140)
        self._enhancer_combo.addItems(["None", "gfpgan", "codeformer"])
        row2.addWidget(self._enhancer_combo)
        row2.addStretch()
        layout.addLayout(row2)

        # Blend ratio
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel("Blend Ratio:"))
        self._blend_slider = QSlider(Qt.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(50)
        self._blend_slider.setFixedWidth(200)
        row3.addWidget(self._blend_slider)
        self._blend_label = QLabel("0.50")
        self._blend_label.setFixedWidth(40)
        self._blend_slider.valueChanged.connect(
            lambda v: self._blend_label.setText(f"{v / 100:.2f}")
        )
        row3.addWidget(self._blend_label)
        row3.addStretch()
        layout.addLayout(row3)

        # Enhancer strength
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(QLabel("Enhancer Strength:"))
        self._strength_slider = QSlider(Qt.Horizontal)
        self._strength_slider.setRange(0, 100)
        self._strength_slider.setValue(50)
        self._strength_slider.setFixedWidth(200)
        row4.addWidget(self._strength_slider)
        self._strength_label = QLabel("0.50")
        self._strength_label.setFixedWidth(40)
        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        row4.addWidget(self._strength_label)
        row4.addStretch()
        layout.addLayout(row4)

        layout.addStretch()

    def on_enter(self) -> None:
        # Restore state
        idx = self._model_combo.findText(self._rs.swap_model)
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

        idx = self._enhancer_combo.findText(self._rs.enhancer)
        if idx >= 0:
            self._enhancer_combo.setCurrentIndex(idx)

        self._blend_slider.setValue(int(self._rs.blend_ratio * 100))
        self._strength_slider.setValue(int(self._rs.enhancer_strength * 100))

    def on_leave(self) -> None:
        self._rs.swap_model = self._model_combo.currentText()
        self._rs.enhancer = self._enhancer_combo.currentText()
        self._rs.blend_ratio = self._blend_slider.value() / 100.0
        self._rs.enhancer_strength = self._strength_slider.value() / 100.0


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Run Swap
# ═══════════════════════════════════════════════════════════════════════════


class RunSwapPage(WizardPage):
    """Summary, run the video face swap, and show output."""

    def __init__(self, rs: ReplacementState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._rs = rs
        self._app_state = app_state
        self._wizard = wizard
        self._running = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.RichText)
        layout.addWidget(self._summary)

        # Run button
        self._run_btn = QPushButton("Run Video Face Swap")
        self._run_btn.setFixedWidth(200)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        # Output info
        self._output_label = QLabel()
        self._output_label.setWordWrap(True)
        self._output_label.setStyleSheet("color: #8f8; margin-top: 12px;")
        self._output_label.hide()
        layout.addWidget(self._output_label)

        layout.addStretch()

    def on_enter(self) -> None:
        self._output_label.hide()
        self._run_btn.setEnabled(True)
        self._running = False

        video_name = Path(self._rs.video_path).name if self._rs.video_path else "(none)"
        source_name = Path(self._rs.face_source_path).name if self._rs.face_source_path else "(none)"
        char_info = f" ({self._rs.character_name})" if self._rs.character_name else ""

        self._summary.setText(
            f"<b>Video:</b> {video_name}<br>"
            f"<b>Face Source:</b> {source_name}{char_info}<br>"
            f"<b>Target Face:</b> {self._rs.target_face_idx} "
            f"{'(all faces)' if self._rs.swap_all else ''}<br>"
            f"<b>Swap Model:</b> {self._rs.swap_model}<br>"
            f"<b>Enhancer:</b> {self._rs.enhancer}<br>"
            f"<b>Blend Ratio:</b> {self._rs.blend_ratio:.2f}<br>"
            f"<b>Enhancer Strength:</b> {self._rs.enhancer_strength:.2f}"
        )

    def _run(self) -> None:
        if self._running:
            return
        self._running = True
        self._run_btn.setEnabled(False)

        # Build output path
        proj_path = _project_path(self._app_state)
        outputs_dir = proj_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        video_stem = Path(self._rs.video_path).stem
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in video_stem
        ).strip() or "video"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(outputs_dir / f"replacement_{safe_name}_{timestamp}.mp4")
        self._rs.output_path = output_path

        # Build ProjectConfig with faceswap settings
        cfg = ProjectConfig.load(proj_path)
        cfg.faceswap_source_face_idx = self._rs.source_face_idx
        cfg.faceswap_target_face_idx = self._rs.target_face_idx
        cfg.faceswap_model = self._rs.swap_model
        cfg.faceswap_enhancer = self._rs.enhancer
        cfg.faceswap_blend_ratio = self._rs.blend_ratio
        cfg.faceswap_swap_all = self._rs.swap_all

        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")

        from sdqt.workers.image import VideoFaceSwapWorker

        worker = VideoFaceSwapWorker(
            source_path=self._rs.face_source_path,
            video_path=self._rs.video_path,
            project_config=cfg,
            models_dir=models_dir,
            output_path=output_path,
            enhancer_strength=self._rs.enhancer_strength,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_done, on_error=self._on_error)

    @Slot(object)
    def _on_done(self, result_path) -> None:
        self._running = False
        self._rs.output_path = str(result_path)
        self._output_label.setText(
            f"Face swap complete!\n\nOutput saved to:\n{result_path}"
        )
        self._output_label.show()
        self._run_btn.setText("Re-run")
        self._run_btn.setEnabled(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._running = False
        self._run_btn.setEnabled(True)
        self._wizard._status.setText(f"Video face swap error: {msg}")


# ═══════════════════════════════════════════════════════════════════════════
# CharacterReplacementWizard — ties all 5 pages together
# ═══════════════════════════════════════════════════════════════════════════


class CharacterReplacementWizard(SequenceWizard):
    """5-page wizard: Video → Face Source → Detect → Settings → Run."""

    def __init__(self, state, *, video_path: str = "", parent=None) -> None:
        self._rs = ReplacementState()
        if video_path:
            self._rs.video_path = video_path

        super().__init__(
            "Character Replacement",
            state,
            parent=parent,
        )

        # Pre-extract first frame if video was provided
        if video_path and Path(video_path).is_file():
            self._rs.sample_frame_path = _extract_first_frame(video_path)

        # Build pages
        self._pages = [
            VideoSourcePage(self._rs, state, self),
            CharacterSourcePage(self._rs, state, self),
            FaceDetectionPage(self._rs, state, self),
            SwapSettingsPage(self._rs),
            RunSwapPage(self._rs, state, self),
        ]
        self._finish_setup()

    def _on_finish(self) -> None:
        self.accept()
