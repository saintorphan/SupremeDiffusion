"""Product Turntable — 7-page wizard sequence.

Takes a product image (imported or generated), converts to 3D mesh,
optionally refines texture, composites over a background, and renders
an animated turntable video.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage
from sdqt.workers.base import BaseWorker
from sdqt.workers.pipeline_load import PipelineLoadWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom system prompts
# ---------------------------------------------------------------------------

_POS_SYSTEM_PROMPT = (
    "You are an image prompt enhancer for SDXL. Given a description of an object, "
    "expand it into a detailed prompt. The object MUST be on a solid, high-contrast "
    "background color (e.g. pure white, solid gray, black) for easy background removal. "
    "Describe the object in detail — material, texture, lighting, shape. Include "
    "'product photography, centered, studio lighting, solid [color] background'. "
    "Output only the enhanced prompt, nothing else."
)

_NEG_SYSTEM_PROMPT = (
    "You are a negative prompt generator for SDXL image generation. Given a description "
    "of an object, produce a negative prompt that prevents unwanted elements. "
    "Always include: 'complex background, gradient, patterns, other objects, "
    "blurry, low quality, deformed, watermark'. "
    "Output only the negative prompt, nothing else."
)

_BG_SYSTEM_PROMPT = (
    "You are an image prompt enhancer for SDXL. Given a description of a background "
    "scene for product photography, expand it into a detailed prompt for generating "
    "a clean background image. Focus on environment, lighting, colors, atmosphere. "
    "The scene should be suitable as a backdrop — no foreground objects. "
    "Output only the enhanced prompt, nothing else."
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class TurntableState:
    name: str = ""
    description: str = ""
    # Source image
    source_path: str = ""       # original product image
    segmented_path: str = ""    # after BG removal
    use_existing_image: bool = True  # True=import, False=generate
    # Generation (if not importing)
    positive_prompt: str = ""
    negative_prompt: str = ""
    checkpoint: str = ""
    sampler: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    seed: int = -1
    selected_loras: list = field(default_factory=list)
    lora_multipliers: str = ""
    base_images: list[str] = field(default_factory=list)
    # Mesh
    mesh_path: str = ""
    mesh_resolution: int = 256
    mesh_format: str = "obj"
    # Texture
    tex_prompt: str = ""
    tex_neg_prompt: str = "low quality, blurry, flat color"
    tex_denoise: float = 0.35
    tex_size: int = 1024
    texture_path: str = ""
    # Background
    bg_path: str = ""  # background image for compositing
    bg_generated: bool = False
    bg_prompt: str = ""
    # Turntable
    num_frames: int = 60
    fps: int = 24
    render_width: int = 512
    render_height: int = 512
    elevation: float = -20.0  # camera elevation
    video_path: str = ""
    # Save
    save_dir: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_path(app_state) -> Path:
    name = app_state.current_project or "_default"
    return app_state.project_manager.get_project_path(name)


def _project_name(app_state) -> str:
    return app_state.current_project or "_default"


def _load_project_config(app_state) -> ProjectConfig:
    return ProjectConfig.load(_project_path(app_state))


def _safe_name(name: str) -> str:
    return "".join(
        c if c.isalnum() or c in " _-" else "_" for c in name
    ).strip() or "product"


# ---------------------------------------------------------------------------
# ObjectPromptWorker — custom Qwen chat for object prompts
# ---------------------------------------------------------------------------


class ObjectPromptWorker(BaseWorker):
    """Generate a prompt via Qwen with a custom system prompt."""

    def __init__(self, state, description: str, system_prompt: str, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._description = description
        self._system_prompt = system_prompt

    def do_work(self) -> str:
        self.status.emit("Loading Qwen...")
        self._state.load_qwen()
        self.status.emit("Generating prompt...")
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._description},
        ]
        result = self._state.generate_chat_response(messages, max_new_tokens=512, temperature=0.7)
        return result.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 — Product Info
# ═══════════════════════════════════════════════════════════════════════════


class ProductInfoPage(WizardPage):
    """Name, description, import/generate choice."""

    def __init__(self, ts: TurntableState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        row = QHBoxLayout()
        row.addWidget(QLabel("Product Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Coffee Mug, Sneaker, Watch")
        row.addWidget(self._name)
        layout.addLayout(row)

        # Description
        layout.addWidget(QLabel("Description:"))
        self._desc = QPlainTextEdit()
        self._desc.setPlaceholderText(
            "Describe the product — shape, material, color, texture, etc."
        )
        self._desc.setMaximumHeight(120)
        layout.addWidget(self._desc)

        # Import vs Generate
        mode_group = QGroupBox("Source Image")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_group = QButtonGroup(self)

        self._import_rb = QRadioButton("Import Image")
        self._import_rb.setChecked(True)
        self._mode_group.addButton(self._import_rb, 0)
        mode_layout.addWidget(self._import_rb)

        # File browser row (for import)
        self._file_row = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("Path to product image...")
        self._file_path.setReadOnly(True)
        self._file_row.addWidget(self._file_path)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setFixedWidth(90)
        self._browse_btn.clicked.connect(self._browse)
        self._file_row.addWidget(self._browse_btn)
        mode_layout.addLayout(self._file_row)

        # Preview
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(160)
        mode_layout.addWidget(self._preview)

        self._generate_rb = QRadioButton("Generate Image")
        self._mode_group.addButton(self._generate_rb, 1)
        mode_layout.addWidget(self._generate_rb)

        layout.addWidget(mode_group)

        # Connect radio to toggle import controls
        self._mode_group.idToggled.connect(self._on_mode_changed)

        layout.addStretch()

    def _on_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        import_mode = id_ == 0
        self._file_path.setEnabled(import_mode)
        self._browse_btn.setEnabled(import_mode)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Product Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._ts.source_path = path
            self._file_path.setText(path)
            pm = QPixmap(path).scaled(
                400, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)

    def on_enter(self) -> None:
        self._name.setText(self._ts.name)
        self._desc.setPlainText(self._ts.description)
        if self._ts.use_existing_image:
            self._import_rb.setChecked(True)
        else:
            self._generate_rb.setChecked(True)
        if self._ts.source_path:
            self._file_path.setText(self._ts.source_path)
            if Path(self._ts.source_path).is_file():
                pm = QPixmap(self._ts.source_path).scaled(
                    400, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self._preview.setPixmap(pm)

    def on_leave(self) -> None:
        self._ts.name = self._name.text().strip()
        self._ts.description = self._desc.toPlainText().strip()
        self._ts.use_existing_image = self._mode_group.checkedId() == 0

    def validate(self) -> bool:
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing", "Please enter a product name.")
            return False
        is_import = self._mode_group.checkedId() == 0
        if is_import:
            if not self._ts.source_path or not Path(self._ts.source_path).is_file():
                QMessageBox.warning(self, "Missing", "Please select a product image.")
                return False
        else:
            if not self._desc.toPlainText().strip():
                QMessageBox.warning(self, "Missing", "Please enter a product description for generation.")
                return False
        self.on_leave()
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Image Preparation
# ═══════════════════════════════════════════════════════════════════════════


class ImagePrepPage(WizardPage):
    """If import: remove BG. If generate: prompt gen + txt2img + pick."""

    def __init__(self, ts: TurntableState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Import mode widgets ---
        self._import_group = QWidget()
        imp_layout = QVBoxLayout(self._import_group)
        imp_layout.setContentsMargins(0, 0, 0, 0)

        self._source_preview = QLabel()
        self._source_preview.setAlignment(Qt.AlignCenter)
        self._source_preview.setFixedHeight(200)
        imp_layout.addWidget(self._source_preview)

        btn_row = QHBoxLayout()
        self._segment_btn = QPushButton("Remove Background")
        self._segment_btn.setFixedWidth(200)
        self._segment_btn.clicked.connect(self._segment)
        btn_row.addWidget(self._segment_btn)
        btn_row.addStretch()
        imp_layout.addLayout(btn_row)

        self._seg_preview = QLabel()
        self._seg_preview.setAlignment(Qt.AlignCenter)
        self._seg_preview.setFixedHeight(200)
        imp_layout.addWidget(self._seg_preview)

        self._seg_status = QLabel()
        self._seg_status.setStyleSheet("color: #aaa; font-size: 12px;")
        imp_layout.addWidget(self._seg_status)

        layout.addWidget(self._import_group)

        # --- Generate mode widgets ---
        self._gen_group = QWidget()
        gen_layout = QVBoxLayout(self._gen_group)
        gen_layout.setContentsMargins(0, 0, 0, 0)

        gen_layout.addWidget(QLabel("<b>Positive Prompt</b>"))
        self._pos_edit = QPlainTextEdit()
        self._pos_edit.setMaximumHeight(100)
        gen_layout.addWidget(self._pos_edit)

        pos_row = QHBoxLayout()
        pos_row.addStretch()
        self._enhance_btn = QPushButton("Enhance Prompt")
        self._enhance_btn.setFixedWidth(130)
        self._enhance_btn.clicked.connect(lambda: self._enhance("pos"))
        pos_row.addWidget(self._enhance_btn)
        gen_layout.addLayout(pos_row)

        gen_layout.addWidget(QLabel("<b>Negative Prompt</b>"))
        self._neg_edit = QPlainTextEdit()
        self._neg_edit.setMaximumHeight(80)
        gen_layout.addWidget(self._neg_edit)

        neg_row = QHBoxLayout()
        neg_row.addStretch()
        self._enhance_neg_btn = QPushButton("Enhance Negative")
        self._enhance_neg_btn.setFixedWidth(130)
        self._enhance_neg_btn.clicked.connect(lambda: self._enhance("neg"))
        neg_row.addWidget(self._enhance_neg_btn)
        gen_layout.addLayout(neg_row)

        self._gen_btn = QPushButton("Generate Images")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate_images)
        gen_layout.addWidget(self._gen_btn)

        self._gallery = ImageGalleryWidget("Generated Images")
        self._gallery.image_selected.connect(self._on_image_selected)
        gen_layout.addWidget(self._gallery, 1)

        layout.addWidget(self._gen_group)
        layout.addStretch()

    def on_enter(self) -> None:
        if self._ts.use_existing_image:
            self._import_group.show()
            self._gen_group.hide()
            if self._ts.source_path and Path(self._ts.source_path).is_file():
                pm = QPixmap(self._ts.source_path).scaled(
                    400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self._source_preview.setPixmap(pm)
            if self._ts.segmented_path and Path(self._ts.segmented_path).is_file():
                pm = QPixmap(self._ts.segmented_path).scaled(
                    400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self._seg_preview.setPixmap(pm)
                self._seg_status.setText(f"Segmented: {self._ts.segmented_path}")
        else:
            self._import_group.hide()
            self._gen_group.show()
            if not self._pos_edit.toPlainText().strip() and not self._ts.positive_prompt:
                self._enhance("pos")
            elif self._ts.positive_prompt:
                self._pos_edit.setPlainText(self._ts.positive_prompt)
            if self._ts.negative_prompt:
                self._neg_edit.setPlainText(self._ts.negative_prompt)
            if self._ts.base_images:
                self._gallery.load_images(self._ts.base_images)

    def _segment(self) -> None:
        if not self._ts.source_path or not Path(self._ts.source_path).is_file():
            return
        from sdqt.workers.image import AutoSegmentWorker

        worker = AutoSegmentWorker(
            image_path=self._ts.source_path,
            models_root=self._app_state.global_config.models_root,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_segment_done, on_error=self._on_error)

    @Slot(object)
    def _on_segment_done(self, mask_path) -> None:
        self._ts.segmented_path = str(mask_path)
        self._seg_status.setText(f"Segmented: {mask_path}")
        if Path(str(mask_path)).is_file():
            pm = QPixmap(str(mask_path)).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._seg_preview.setPixmap(pm)

    def _enhance(self, which: str) -> None:
        desc = self._ts.description
        if not desc:
            return
        sys_prompt = _POS_SYSTEM_PROMPT if which == "pos" else _NEG_SYSTEM_PROMPT
        worker = ObjectPromptWorker(
            state=self._app_state,
            description=desc,
            system_prompt=sys_prompt,
            parent=self,
        )
        if which == "pos":
            self._wizard._run_worker(worker, on_done=self._on_pos_done, on_error=self._on_error)
        else:
            self._wizard._run_worker(worker, on_done=self._on_neg_done, on_error=self._on_error)

    @Slot(object)
    def _on_pos_done(self, text) -> None:
        self._pos_edit.setPlainText(str(text))
        self._ts.positive_prompt = str(text)
        if not self._neg_edit.toPlainText().strip():
            self._enhance("neg")

    @Slot(object)
    def _on_neg_done(self, text) -> None:
        self._neg_edit.setPlainText(str(text))
        self._ts.negative_prompt = str(text)

    def _generate_images(self) -> None:
        self._ts.positive_prompt = self._pos_edit.toPlainText().strip()
        self._ts.negative_prompt = self._neg_edit.toPlainText().strip()
        self._gen_btn.setText("Regenerate")

        cfg = _load_project_config(self._app_state)
        cfg.img_prompt = self._ts.positive_prompt
        cfg.img_negative_prompt = self._ts.negative_prompt
        cfg.img_batch_count = 4
        cfg.img_batch_size = 1
        cfg.img_steps = self._ts.steps
        cfg.img_cfg_scale = self._ts.cfg_scale
        cfg.img_width = self._ts.width
        cfg.img_height = self._ts.height
        cfg.img_seed = self._ts.seed
        if self._ts.checkpoint:
            cfg.img_checkpoint = self._ts.checkpoint
        if self._ts.sampler:
            cfg.img_sampler = self._ts.sampler
        cfg.img_loras = self._ts.selected_loras
        cfg.save(_project_path(self._app_state))

        if self._app_state.img_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_sd_pipelines", parent=self)
            self._wizard._run_worker(loader, on_done=lambda _: self._run_txt2img(cfg))
        else:
            self._run_txt2img(cfg)

    def _run_txt2img(self, cfg) -> None:
        from sdqt.workers.image import Txt2ImgWorker

        worker = Txt2ImgWorker(
            pipeline=self._app_state.img_pipeline,
            project_name=_project_name(self._app_state),
            project_config=cfg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_images, on_error=self._on_error)

    @Slot(object)
    def _on_images(self, paths) -> None:
        self._ts.base_images = list(paths) if paths else []
        self._gallery.load_images(self._ts.base_images)
        if self._ts.base_images:
            self._ts.source_path = self._ts.base_images[0]

    @Slot(str)
    def _on_image_selected(self, path: str) -> None:
        self._ts.source_path = path

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Error: {msg}")

    def on_leave(self) -> None:
        if not self._ts.use_existing_image:
            self._ts.positive_prompt = self._pos_edit.toPlainText().strip()
            self._ts.negative_prompt = self._neg_edit.toPlainText().strip()
            self._app_state.unload_sd_pipelines()
        self._app_state.unload_qwen()

    def validate(self) -> bool:
        if self._ts.use_existing_image:
            # Use segmented if available, else original
            effective = self._ts.segmented_path or self._ts.source_path
            if not effective or not Path(effective).is_file():
                QMessageBox.warning(self, "Missing", "No source image available.")
                return False
        else:
            if not self._ts.source_path or not Path(self._ts.source_path).is_file():
                QMessageBox.warning(self, "Missing", "Please generate and select an image.")
                return False
        self.on_leave()
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Mesh Generation
# ═══════════════════════════════════════════════════════════════════════════


class MeshGenPage(WizardPage):
    """Generate a 3D mesh from the product image."""

    def __init__(self, ts: TurntableState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preview of source image
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(200)
        layout.addWidget(self._preview)

        # Settings row
        settings = QHBoxLayout()
        settings.setSpacing(6)

        settings.addWidget(QLabel("Resolution:"))
        self._res_spin = QSpinBox()
        self._res_spin.setRange(64, 512)
        self._res_spin.setValue(256)
        self._res_spin.setSingleStep(32)
        self._res_spin.setFixedWidth(75)
        settings.addWidget(self._res_spin)

        settings.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["obj", "glb", "ply", "stl"])
        self._format_combo.setFixedWidth(100)
        settings.addWidget(self._format_combo)

        self._bg_cb = QCheckBox("Remove Background")
        self._bg_cb.setChecked(True)
        settings.addWidget(self._bg_cb)

        settings.addStretch()
        layout.addLayout(settings)

        # Generate button
        self._gen_btn = QPushButton("Generate Mesh")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate)
        layout.addWidget(self._gen_btn)

        # Status
        self._mesh_status = QLabel()
        self._mesh_status.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._mesh_status)

        layout.addStretch()

    def _get_source_image(self) -> str:
        """Return the best source image path (segmented > selected > source)."""
        if self._ts.segmented_path and Path(self._ts.segmented_path).is_file():
            return self._ts.segmented_path
        return self._ts.source_path

    def on_enter(self) -> None:
        src = self._get_source_image()
        if src and Path(src).is_file():
            pm = QPixmap(src).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        if self._ts.mesh_path and Path(self._ts.mesh_path).is_file():
            self._mesh_status.setText(f"Mesh: {self._ts.mesh_path}")

    def on_leave(self) -> None:
        self._ts.mesh_resolution = self._res_spin.value()
        self._ts.mesh_format = self._format_combo.currentText()
        # Free TripoSR VRAM when leaving mesh gen page
        self._app_state.unload_triposr_pipeline()

    def _generate(self) -> None:
        self._ts.mesh_resolution = self._res_spin.value()
        self._ts.mesh_format = self._format_combo.currentText()

        src = self._get_source_image()
        if not src or not Path(src).is_file():
            QMessageBox.warning(self, "No Image", "No source image available.")
            return

        # Determine output path
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdqt_mesh_"))
        safe = _safe_name(self._ts.name)
        output_path = str(tmp_dir / f"{safe}.{self._ts.mesh_format}")

        # Check for required model downloads
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "3d_modeling", self._app_state.model_registry, self,
            on_progress=lambda f, d: self._wizard._status.setText(d),
        ):
            return

        # Free SD VRAM before loading TripoSR
        self._app_state.unload_sd_pipelines()

        remove_bg = self._bg_cb.isChecked()

        if self._app_state.triposr_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_triposr_pipeline", parent=self)
            self._wizard._run_worker(
                loader,
                on_done=lambda _: self._run_mesh_gen(src, output_path, remove_bg),
            )
        else:
            self._run_mesh_gen(src, output_path, remove_bg)

    def _run_mesh_gen(self, image_path: str, output_path: str, remove_bg: bool) -> None:
        from sdqt.workers.model3d import MeshGenerationWorker

        worker = MeshGenerationWorker(
            pipeline=self._app_state.triposr_pipeline,
            image_path=image_path,
            output_path=output_path,
            resolution=self._ts.mesh_resolution,
            output_format=self._ts.mesh_format,
            remove_bg=remove_bg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_mesh_done, on_error=self._on_error)

    @Slot(object)
    def _on_mesh_done(self, path) -> None:
        self._ts.mesh_path = str(path)
        self._mesh_status.setText(f"Mesh saved: {path}")
        self._gen_btn.setText("Regenerate Mesh")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Mesh generation error: {msg}")
        self._mesh_status.setText(f"Error: {msg}")

    def validate(self) -> bool:
        if not self._ts.mesh_path or not Path(self._ts.mesh_path).is_file():
            QMessageBox.warning(self, "Missing", "Please generate a 3D mesh first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Texture Refinement (optional)
# ═══════════════════════════════════════════════════════════════════════════


class TextureRefinePage(WizardPage):
    """Optional texture refinement via SD img2img + UV projection."""

    def __init__(self, ts: TurntableState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._viewport = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # GL viewport for mesh preview and render source
        from sdqt.widgets.gl_viewport import GLViewportWidget
        self._viewport = GLViewportWidget()
        self._viewport.setMinimumHeight(250)
        self._viewport.setMaximumHeight(350)
        layout.addWidget(self._viewport)

        # Prompt fields
        layout.addWidget(QLabel("<b>Texture Prompt</b>"))
        self._prompt = QPlainTextEdit()
        self._prompt.setMaximumHeight(80)
        layout.addWidget(self._prompt)

        layout.addWidget(QLabel("<b>Negative Prompt</b>"))
        self._neg_prompt = QPlainTextEdit()
        self._neg_prompt.setMaximumHeight(60)
        layout.addWidget(self._neg_prompt)

        # Settings row
        settings = QHBoxLayout()
        settings.setSpacing(6)

        settings.addWidget(QLabel("Denoise:"))
        self._denoise = QDoubleSpinBox()
        self._denoise.setRange(0.1, 0.8)
        self._denoise.setValue(0.35)
        self._denoise.setSingleStep(0.05)
        self._denoise.setFixedWidth(80)
        settings.addWidget(self._denoise)

        settings.addWidget(QLabel("Tex Size:"))
        self._tex_size = QComboBox()
        self._tex_size.addItems(["512", "1024", "2048"])
        self._tex_size.setCurrentText("1024")
        self._tex_size.setFixedWidth(100)
        settings.addWidget(self._tex_size)

        settings.addStretch()
        layout.addLayout(settings)

        # Refine button
        btn_row = QHBoxLayout()
        self._refine_btn = QPushButton("Refine Texture")
        self._refine_btn.setFixedWidth(200)
        self._refine_btn.clicked.connect(self._refine)
        btn_row.addWidget(self._refine_btn)

        self._skip_label = QLabel("(You can skip this step)")
        self._skip_label.setStyleSheet("color: #888; font-size: 11px;")
        btn_row.addWidget(self._skip_label)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()

    def on_enter(self) -> None:
        if self._ts.mesh_path and Path(self._ts.mesh_path).is_file():
            self._viewport.clear()
            self._viewport.load_mesh(self._ts.mesh_path)

        if not self._prompt.toPlainText().strip():
            self._prompt.setPlainText(
                self._ts.tex_prompt or self._ts.description
            )
        if not self._neg_prompt.toPlainText().strip():
            self._neg_prompt.setPlainText(self._ts.tex_neg_prompt)

    def on_leave(self) -> None:
        self._ts.tex_prompt = self._prompt.toPlainText().strip()
        self._ts.tex_neg_prompt = self._neg_prompt.toPlainText().strip()
        self._ts.tex_denoise = self._denoise.value()
        self._ts.tex_size = int(self._tex_size.currentText())
        # Free SD VRAM when leaving texture refine page
        self._app_state.unload_sd_pipelines()

    def _refine(self) -> None:
        self._ts.tex_prompt = self._prompt.toPlainText().strip()
        self._ts.tex_neg_prompt = self._neg_prompt.toPlainText().strip()
        self._ts.tex_denoise = self._denoise.value()
        self._ts.tex_size = int(self._tex_size.currentText())

        if not self._ts.mesh_path or not Path(self._ts.mesh_path).is_file():
            QMessageBox.warning(self, "No Mesh", "No mesh available for texture refinement.")
            return

        # Free TripoSR VRAM before loading SD for img2img
        self._app_state.unload_triposr_pipeline()

        from sdqt.workers.texture_refine import TextureRefineWorker

        worker = TextureRefineWorker(
            mesh_path=self._ts.mesh_path,
            render_func=self._viewport.render_single_view,
            app_state=self._app_state,
            prompt=self._ts.tex_prompt,
            negative_prompt=self._ts.tex_neg_prompt,
            denoise_strength=self._ts.tex_denoise,
            tex_size=self._ts.tex_size,
            checkpoint=self._ts.checkpoint,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_refine_done, on_error=self._on_error)

    @Slot(object)
    def _on_refine_done(self, tex_path) -> None:
        self._ts.texture_path = str(tex_path)
        self._refine_btn.setText("Re-refine Texture")
        self._viewport.reload_texture(0)
        self._wizard._status.setText(f"Texture refined: {tex_path}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Texture refinement error: {msg}")

    def get_viewport(self):
        """Return the GLViewportWidget for reuse on later pages."""
        return self._viewport


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Background
# ═══════════════════════════════════════════════════════════════════════════


class BackgroundPage(WizardPage):
    """Choose background: solid color, import image, or generate."""

    def __init__(self, ts: TurntableState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Background Source")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_group = QButtonGroup(self)

        self._solid_rb = QRadioButton("Solid Color")
        self._solid_rb.setChecked(True)
        self._mode_group.addButton(self._solid_rb, 0)
        mode_layout.addWidget(self._solid_rb)

        # Solid color controls
        self._color_row = QHBoxLayout()
        self._color_row.setSpacing(6)
        for label, hex_color in [("White", "#FFFFFF"), ("Black", "#000000"), ("Gray", "#808080")]:
            btn = QPushButton(label)
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda checked, h=hex_color: self._set_solid_color(h))
            self._color_row.addWidget(btn)
        self._custom_color_btn = QPushButton("Custom...")
        self._custom_color_btn.setFixedWidth(90)
        self._custom_color_btn.clicked.connect(self._pick_color)
        self._color_row.addWidget(self._custom_color_btn)
        self._color_row.addStretch()
        mode_layout.addLayout(self._color_row)

        self._import_rb = QRadioButton("Import Image")
        self._mode_group.addButton(self._import_rb, 1)
        mode_layout.addWidget(self._import_rb)

        # Import controls
        self._file_row = QHBoxLayout()
        self._file_path = QLineEdit()
        self._file_path.setPlaceholderText("Path to background image...")
        self._file_path.setReadOnly(True)
        self._file_row.addWidget(self._file_path)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setFixedWidth(90)
        self._browse_btn.clicked.connect(self._browse_bg)
        self._file_row.addWidget(self._browse_btn)
        mode_layout.addLayout(self._file_row)

        self._gen_rb = QRadioButton("Generate Background")
        self._mode_group.addButton(self._gen_rb, 2)
        mode_layout.addWidget(self._gen_rb)

        # Generate controls
        self._prompt_row = QHBoxLayout()
        self._bg_prompt = QLineEdit()
        self._bg_prompt.setPlaceholderText("Describe the background scene...")
        self._prompt_row.addWidget(self._bg_prompt)
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setFixedWidth(100)
        self._gen_btn.clicked.connect(self._generate_bg)
        self._prompt_row.addWidget(self._gen_btn)
        mode_layout.addLayout(self._prompt_row)

        layout.addWidget(mode_group)

        # Preview
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(200)
        layout.addWidget(self._preview)

        self._bg_status = QLabel()
        self._bg_status.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._bg_status)

        # Connect radio to toggle controls
        self._mode_group.idToggled.connect(self._on_mode_changed)

        layout.addStretch()

        # Set initial solid white background
        self._current_color = "#FFFFFF"

    def _on_mode_changed(self, id_: int, checked: bool) -> None:
        if not checked:
            return
        # Enable/disable controls based on mode
        solid = id_ == 0
        import_ = id_ == 1
        gen = id_ == 2
        self._custom_color_btn.setEnabled(solid)
        self._file_path.setEnabled(import_)
        self._browse_btn.setEnabled(import_)
        self._bg_prompt.setEnabled(gen)
        self._gen_btn.setEnabled(gen)

    def _set_solid_color(self, hex_color: str) -> None:
        self._current_color = hex_color
        self._solid_rb.setChecked(True)
        # Create solid color image
        bg = QImage(self._ts.render_width or 512, self._ts.render_height or 512, QImage.Format_RGB32)
        bg.fill(QColor(hex_color))
        # Save to temp
        tmp = Path(tempfile.mkdtemp(prefix="sdqt_bg_")) / "bg_solid.png"
        bg.save(str(tmp), "PNG")
        self._ts.bg_path = str(tmp)
        self._ts.bg_generated = False
        pm = QPixmap.fromImage(bg).scaled(
            400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self._preview.setPixmap(pm)
        self._bg_status.setText(f"Solid color: {hex_color}")

    def _pick_color(self) -> None:
        color = QColorDialog.getColor(QColor(self._current_color), self, "Background Color")
        if color.isValid():
            self._set_solid_color(color.name())

    def _browse_bg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self._ts.bg_path = path
            self._ts.bg_generated = False
            self._file_path.setText(path)
            pm = QPixmap(path).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
            self._bg_status.setText(f"Background: {path}")

    def _generate_bg(self) -> None:
        prompt = self._bg_prompt.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing", "Please enter a background description.")
            return
        self._ts.bg_prompt = prompt

        cfg = _load_project_config(self._app_state)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = "foreground objects, people, text, watermark"
        cfg.img_batch_count = 1
        cfg.img_batch_size = 1
        cfg.img_steps = self._ts.steps
        cfg.img_cfg_scale = self._ts.cfg_scale
        cfg.img_width = self._ts.render_width or 512
        cfg.img_height = self._ts.render_height or 512
        cfg.img_seed = -1
        if self._ts.checkpoint:
            cfg.img_checkpoint = self._ts.checkpoint
        if self._ts.sampler:
            cfg.img_sampler = self._ts.sampler
        cfg.save(_project_path(self._app_state))

        if self._app_state.img_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_sd_pipelines", parent=self)
            self._wizard._run_worker(loader, on_done=lambda _: self._run_bg_gen(cfg))
        else:
            self._run_bg_gen(cfg)

    def _run_bg_gen(self, cfg) -> None:
        from sdqt.workers.image import Txt2ImgWorker

        worker = Txt2ImgWorker(
            pipeline=self._app_state.img_pipeline,
            project_name=_project_name(self._app_state),
            project_config=cfg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_bg_done, on_error=self._on_error)

    @Slot(object)
    def _on_bg_done(self, paths) -> None:
        if paths:
            bg_path = list(paths)[0]
            self._ts.bg_path = bg_path
            self._ts.bg_generated = True
            pm = QPixmap(bg_path).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
            self._bg_status.setText(f"Generated background: {bg_path}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Background generation error: {msg}")

    def on_enter(self) -> None:
        if self._ts.bg_prompt:
            self._bg_prompt.setText(self._ts.bg_prompt)
        if self._ts.bg_path and Path(self._ts.bg_path).is_file():
            pm = QPixmap(self._ts.bg_path).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        elif not self._ts.bg_path:
            # Default to solid white
            self._set_solid_color("#FFFFFF")

    def on_leave(self) -> None:
        self._ts.bg_prompt = self._bg_prompt.text().strip()
        # Free SD if used for generation
        self._app_state.unload_sd_pipelines()

    def validate(self) -> bool:
        mode_id = self._mode_group.checkedId()
        if mode_id == 0:
            # Solid color — ensure bg_path created
            if not self._ts.bg_path or not Path(self._ts.bg_path).is_file():
                self._set_solid_color(self._current_color)
        elif mode_id == 1:
            if not self._ts.bg_path or not Path(self._ts.bg_path).is_file():
                QMessageBox.warning(self, "Missing", "Please select a background image.")
                return False
        elif mode_id == 2:
            if not self._ts.bg_path or not Path(self._ts.bg_path).is_file():
                QMessageBox.warning(self, "Missing", "Please generate a background first.")
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 — Turntable Render
# ═══════════════════════════════════════════════════════════════════════════


class TurntablePage(WizardPage):
    """Render turntable animation — renders frames on main thread, assembles with ffmpeg."""

    def __init__(self, ts: TurntableState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._viewport = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # GL viewport
        from sdqt.widgets.gl_viewport import GLViewportWidget
        self._viewport = GLViewportWidget()
        self._viewport.setMinimumHeight(250)
        self._viewport.setMaximumHeight(350)
        layout.addWidget(self._viewport)

        # Settings
        settings = QHBoxLayout()
        settings.setSpacing(6)

        settings.addWidget(QLabel("Frames:"))
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(30, 120)
        self._frames_spin.setValue(60)
        self._frames_spin.setFixedWidth(75)
        settings.addWidget(self._frames_spin)

        settings.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(12, 60)
        self._fps_spin.setValue(24)
        self._fps_spin.setFixedWidth(75)
        settings.addWidget(self._fps_spin)

        settings.addWidget(QLabel("Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(256, 1024)
        self._size_spin.setValue(512)
        self._size_spin.setSingleStep(64)
        self._size_spin.setFixedWidth(75)
        settings.addWidget(self._size_spin)

        settings.addWidget(QLabel("Elevation:"))
        self._elev_spin = QDoubleSpinBox()
        self._elev_spin.setRange(-60.0, 0.0)
        self._elev_spin.setValue(-20.0)
        self._elev_spin.setSingleStep(5.0)
        self._elev_spin.setFixedWidth(80)
        settings.addWidget(self._elev_spin)

        settings.addStretch()
        layout.addLayout(settings)

        # Render button
        btn_row = QHBoxLayout()
        self._render_btn = QPushButton("Render Turntable")
        self._render_btn.setFixedWidth(200)
        self._render_btn.clicked.connect(self._render)
        btn_row.addWidget(self._render_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Status
        self._render_status = QLabel()
        self._render_status.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._render_status)

        layout.addStretch()

    def on_enter(self) -> None:
        if self._ts.mesh_path and Path(self._ts.mesh_path).is_file():
            self._viewport.clear()
            self._viewport.load_mesh(self._ts.mesh_path)
        if self._ts.video_path and Path(self._ts.video_path).is_file():
            self._render_status.setText(f"Video: {self._ts.video_path}")

    def on_leave(self) -> None:
        self._ts.num_frames = self._frames_spin.value()
        self._ts.fps = self._fps_spin.value()
        self._ts.render_width = self._size_spin.value()
        self._ts.render_height = self._size_spin.value()
        self._ts.elevation = self._elev_spin.value()

    def _render(self) -> None:
        self._ts.num_frames = self._frames_spin.value()
        self._ts.fps = self._fps_spin.value()
        self._ts.render_width = self._size_spin.value()
        self._ts.render_height = self._size_spin.value()
        self._ts.elevation = self._elev_spin.value()

        if not self._ts.mesh_path or not Path(self._ts.mesh_path).is_file():
            QMessageBox.warning(self, "No Mesh", "No mesh to render.")
            return

        total = self._ts.num_frames
        w = self._ts.render_width
        h = self._ts.render_height
        elevation = self._ts.elevation

        # Create frames directory
        frames_dir = Path(tempfile.mkdtemp(prefix="sdqt_turntable_"))

        # Load background
        bg_image = None
        if self._ts.bg_path and Path(self._ts.bg_path).is_file():
            bg_image = QImage(self._ts.bg_path)
            if not bg_image.isNull():
                bg_image = bg_image.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self._wizard.set_busy(True)
        self._render_status.setText("Rendering frames...")

        # Render frames synchronously with processEvents
        for i in range(total):
            azimuth = (i / total) * 360.0
            render = self._viewport.render_single_view(elevation, azimuth, w, h)

            if render is None or render.isNull():
                continue

            # Composite over background
            if bg_image is not None and not bg_image.isNull():
                composite = QImage(bg_image)
            else:
                composite = QImage(w, h, QImage.Format_RGB32)
                composite.fill(QColor("#FFFFFF"))

            painter = QPainter(composite)
            painter.drawImage(0, 0, render)
            painter.end()

            out_path = frames_dir / f"frame_{i:05d}.png"
            composite.save(str(out_path), "PNG")

            # Update progress
            frac = (i + 1) / total
            self._wizard._status.setText(f"Rendering frame {i + 1}/{total}")
            self._wizard._progress.setValue(int(frac * 100))
            self._wizard._progress.show()
            QApplication.processEvents()

        # Assemble video with ffmpeg
        self._render_status.setText("Assembling video...")
        QApplication.processEvents()

        safe = _safe_name(self._ts.name)
        output_path = str(frames_dir / f"{safe}_turntable.mp4")

        from sdqt.utils.codec import (
            configured_codec_args,
            configured_encoder_name,
            pix_fmt_args,
        )
        codec_args = configured_codec_args()

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self._ts.fps),
            "-i", str(frames_dir / "frame_%05d.png"),
            *codec_args,
            *pix_fmt_args(configured_encoder_name()),
            "-vf", "scale=in_range=full:out_range=full",
            output_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
            self._ts.video_path = output_path
            self._render_status.setText(f"Video saved: {output_path}")
            self._render_btn.setText("Re-render Turntable")
        except subprocess.CalledProcessError as e:
            self._render_status.setText(f"FFmpeg error: {e.stderr[:200]}")
        except Exception as e:
            self._render_status.setText(f"Error: {e}")

        self._wizard.set_busy(False)

    def validate(self) -> bool:
        if not self._ts.video_path or not Path(self._ts.video_path).is_file():
            QMessageBox.warning(self, "Missing", "Please render a turntable video first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 7 — Review & Save
# ═══════════════════════════════════════════════════════════════════════════


class ReviewSavePage(WizardPage):
    """Review summary and save all assets."""

    def __init__(self, ts: TurntableState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = ts
        self._app_state = app_state
        self._wizard = wizard
        self._saved = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Summary info
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.RichText)
        self._info.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._info)
        layout.addWidget(scroll)

        # Video path display
        self._video_label = QLabel()
        self._video_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._video_label.setWordWrap(True)
        layout.addWidget(self._video_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._save_btn = QPushButton("Save && Exit")
        self._save_btn.setFixedWidth(140)
        self._save_btn.setStyleSheet(
            "QPushButton { background: #0078d4; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #1a8ae8; }"
        )
        self._save_btn.clicked.connect(self._save_and_exit)
        btn_row.addWidget(self._save_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

    def on_enter(self) -> None:
        tex_status = "Refined" if self._ts.texture_path else "Default (TripoSR)"
        bg_info = "Generated" if self._ts.bg_generated else (self._ts.bg_path or "None")

        self._info.setText(
            f"<b>Name:</b> {self._ts.name}<br>"
            f"<b>Description:</b> {self._ts.description[:200]}<br>"
            f"<b>Source:</b> {'Imported' if self._ts.use_existing_image else 'Generated'}<br>"
            f"<b>Checkpoint:</b> {self._ts.checkpoint or '(project default)'}<br>"
            f"<b>Mesh:</b> {self._ts.mesh_path}<br>"
            f"<b>Mesh Resolution:</b> {self._ts.mesh_resolution}<br>"
            f"<b>Texture:</b> {tex_status}<br>"
            f"<b>Background:</b> {bg_info}<br>"
            f"<b>Frames:</b> {self._ts.num_frames}<br>"
            f"<b>FPS:</b> {self._ts.fps}<br>"
            f"<b>Render Size:</b> {self._ts.render_width}x{self._ts.render_height}<br>"
            f"<b>Elevation:</b> {self._ts.elevation}"
        )

        if self._ts.video_path and Path(self._ts.video_path).is_file():
            self._video_label.setText(f"Video: {self._ts.video_path}")
        else:
            self._video_label.setText("No video rendered.")

        self._saved = False
        self._save_btn.setEnabled(True)

    def _get_save_dir(self) -> Path:
        """Determine and create the save directory."""
        meshes_dir = self._app_state.global_config.meshes_dir
        if not meshes_dir:
            from supremediffusion.config.defaults import APP_ROOT
            meshes_dir = str(APP_ROOT / "library" / "meshes")

        safe = _safe_name(self._ts.name)
        save_dir = Path(meshes_dir) / safe
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _save(self) -> Path:
        """Save all assets and metadata. Returns the save directory."""
        save_dir = self._get_save_dir()
        self._ts.save_dir = str(save_dir)
        safe = save_dir.name

        # Copy mesh
        if self._ts.mesh_path and Path(self._ts.mesh_path).is_file():
            ext = Path(self._ts.mesh_path).suffix
            shutil.copy2(self._ts.mesh_path, save_dir / f"{safe}{ext}")

        # Copy texture
        if self._ts.texture_path and Path(self._ts.texture_path).is_file():
            shutil.copy2(self._ts.texture_path, save_dir / f"{safe}_albedo.png")

        # Copy source image
        src = self._ts.segmented_path or self._ts.source_path
        if src and Path(src).is_file():
            shutil.copy2(src, save_dir / "source.png")

        # Copy background
        if self._ts.bg_path and Path(self._ts.bg_path).is_file():
            shutil.copy2(self._ts.bg_path, save_dir / "background.png")

        # Copy turntable video
        if self._ts.video_path and Path(self._ts.video_path).is_file():
            shutil.copy2(self._ts.video_path, save_dir / f"{safe}_turntable.mp4")

        # Save metadata
        meta = {
            "name": self._ts.name,
            "description": self._ts.description,
            "source_type": "imported" if self._ts.use_existing_image else "generated",
            "checkpoint": self._ts.checkpoint,
            "sampler": self._ts.sampler,
            "steps": self._ts.steps,
            "cfg_scale": self._ts.cfg_scale,
            "mesh_resolution": self._ts.mesh_resolution,
            "mesh_format": self._ts.mesh_format,
            "texture_refined": bool(self._ts.texture_path),
            "tex_prompt": self._ts.tex_prompt,
            "tex_denoise": self._ts.tex_denoise,
            "tex_size": self._ts.tex_size,
            "background_generated": self._ts.bg_generated,
            "bg_prompt": self._ts.bg_prompt,
            "num_frames": self._ts.num_frames,
            "fps": self._ts.fps,
            "render_width": self._ts.render_width,
            "render_height": self._ts.render_height,
            "elevation": self._ts.elevation,
        }
        with open(save_dir / "turntable.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._saved = True
        self._wizard._status.setText(f"Saved to {save_dir}")
        return save_dir

    def _save_and_exit(self) -> None:
        save_dir = self._save()
        QMessageBox.information(
            self, "Saved", f"Product '{self._ts.name}' saved to:\n{save_dir}"
        )
        self._wizard.accept()

    def validate(self) -> bool:
        return self._saved


# ═══════════════════════════════════════════════════════════════════════════
# ProductTurntableWizard — ties all 7 pages together
# ═══════════════════════════════════════════════════════════════════════════


class ProductTurntableWizard(SequenceWizard):
    """7-page wizard: Info -> ImagePrep -> Mesh -> Texture -> Background -> Turntable -> Save."""

    def __init__(self, state, *, img_params: dict | None = None, parent=None) -> None:
        self._ts = TurntableState()

        # Persistent header: checkpoint selector
        header = QWidget()
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(QLabel("Image Model:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(300)
        h_layout.addWidget(self._ckpt_combo, 1)
        h_layout.addStretch()

        super().__init__(
            "Product Turntable",
            state,
            header_widget=header,
            parent=parent,
        )
        self.setMinimumSize(850, 700)

        # Populate checkpoint combo
        self._populate_checkpoints()
        self._ckpt_combo.currentTextChanged.connect(self._on_checkpoint_changed)

        # Apply image params from Sequences tab
        if img_params:
            self._ts.checkpoint = img_params.get("checkpoint", "")
            self._ts.sampler = img_params.get("sampler", "")
            self._ts.steps = img_params.get("steps", 20)
            self._ts.cfg_scale = img_params.get("cfg_scale", 7.0)
            self._ts.width = img_params.get("width", 512)
            self._ts.height = img_params.get("height", 512)
            self._ts.seed = img_params.get("seed", -1)
        else:
            cfg = ProjectConfig.load(
                state.project_manager.get_project_path(state.current_project or "_default")
            )
            self._ts.checkpoint = cfg.img_checkpoint
            self._ts.sampler = cfg.img_sampler
            self._ts.steps = cfg.img_steps
            self._ts.cfg_scale = cfg.img_cfg_scale
            self._ts.width = cfg.img_width
            self._ts.height = cfg.img_height
            self._ts.seed = cfg.img_seed

        if self._ts.checkpoint:
            idx = self._ckpt_combo.findText(self._ts.checkpoint)
            if idx >= 0:
                self._ckpt_combo.setCurrentIndex(idx)

        # Build pages
        self._pages = [
            ProductInfoPage(self._ts, state, self),
            ImagePrepPage(self._ts, state, self, self),
            MeshGenPage(self._ts, state, self, self),
            TextureRefinePage(self._ts, state, self, self),
            BackgroundPage(self._ts, state, self, self),
            TurntablePage(self._ts, state, self, self),
            ReviewSavePage(self._ts, state, self, self),
        ]
        self._finish_setup()

    def _populate_checkpoints(self) -> None:
        ckpt_dir = self._state.global_config.model_paths.get("sd_checkpoint_dir", "")
        if not ckpt_dir:
            return
        try:
            from supremediffusion.models.sd_models import scan_checkpoints
            infos = scan_checkpoints(ckpt_dir)
            for info in infos:
                self._ckpt_combo.addItem(info.name)
        except Exception as exc:
            logger.warning("Could not scan checkpoints: %s", exc)

    @Slot(str)
    def _on_checkpoint_changed(self, name: str) -> None:
        self._ts.checkpoint = name

    def _show_page(self, idx: int) -> None:
        super()._show_page(idx)
        # Hide default Next on the review page (it has its own Save & Exit button)
        if self._current_idx == 6:
            self._next_btn.hide()
        else:
            self._next_btn.show()

    def _on_finish(self) -> None:
        self.accept()
