"""Image to 3D Model (to LoRA) — 7-page wizard sequence."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.lora_picker import LoRAPickerWidget
from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage
from sdqt.workers.base import BaseWorker
from sdqt.workers.pipeline_load import PipelineLoadWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom system prompts for object-on-solid-background
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

# ---------------------------------------------------------------------------
# Shared object state
# ---------------------------------------------------------------------------


@dataclass
class ObjectState:
    name: str = ""
    description: str = ""
    style: str = "realism"
    selected_loras: list = field(default_factory=list)
    lora_multipliers: str = ""
    positive_prompt: str = ""
    negative_prompt: str = ""
    base_images: list[str] = field(default_factory=list)
    selected_base: str = ""
    checkpoint: str = ""
    sampler: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    seed: int = -1
    # Mesh
    mesh_path: str = ""
    mesh_resolution: int = 256
    mesh_format: str = "obj"
    remove_bg: bool = True
    # Texture
    tex_prompt: str = ""
    tex_neg_prompt: str = "low quality, blurry, flat color, smooth plastic"
    tex_denoise: float = 0.35
    tex_size: int = 1024
    texture_path: str = ""
    # Output
    save_dir: str = ""
    # LoRA
    lora_dataset_dir: str = ""


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


# ---------------------------------------------------------------------------
# ObjectPromptWorker — custom Qwen chat for object prompts
# ---------------------------------------------------------------------------


class ObjectPromptWorker(BaseWorker):
    """Generate an object prompt via Qwen with a custom system prompt."""

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
# Page 1 — Object Info
# ═══════════════════════════════════════════════════════════════════════════


class ObjectInfoPage(WizardPage):
    """Name, description, style, LoRA selection."""

    def __init__(self, obj_state: ObjectState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        row = QHBoxLayout()
        row.addWidget(QLabel("Object Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Crystal Sword, Ceramic Vase")
        row.addWidget(self._name)
        layout.addLayout(row)

        # Description
        layout.addWidget(QLabel("Description:"))
        self._desc = QPlainTextEdit()
        self._desc.setPlaceholderText(
            "Describe the object in detail — shape, material, color, texture, etc."
        )
        self._desc.setMaximumHeight(120)
        layout.addWidget(self._desc)

        # Style
        style_group = QGroupBox("Style")
        style_layout = QHBoxLayout(style_group)
        self._style_group = QButtonGroup(self)
        for i, (label, value) in enumerate(
            [("Realism", "realism"), ("Cartoon", "cartoon"), ("Anime", "anime")]
        ):
            rb = QRadioButton(label)
            self._style_group.addButton(rb, i)
            style_layout.addWidget(rb)
            if value == "realism":
                rb.setChecked(True)
        style_layout.addStretch()
        layout.addWidget(style_group)

        # LoRA picker
        self._lora_picker = LoRAPickerWidget("LoRA")
        self._lora_picker.set_state(self._app_state)
        self._lora_picker.insert_to_prompt.connect(self._on_lora_insert)
        sd_lora_dir = self._app_state.global_config.model_paths.get("sd_lora_dir", "")
        if sd_lora_dir:
            self._lora_picker.set_lora_dir(sd_lora_dir)
        layout.addWidget(self._lora_picker)

        layout.addStretch()

    def _on_lora_insert(self, text: str) -> None:
        """Insert LoRA tags/triggers into the description field."""
        current = self._desc.toPlainText().strip()
        if current:
            self._desc.setPlainText(f"{current}, {text}")
        else:
            self._desc.setPlainText(text)

    def on_enter(self) -> None:
        self._name.setText(self._os.name)
        self._desc.setPlainText(self._os.description)

    def on_leave(self) -> None:
        self._os.name = self._name.text().strip()
        self._os.description = self._desc.toPlainText().strip()
        styles = ["realism", "cartoon", "anime"]
        btn_id = self._style_group.checkedId()
        self._os.style = styles[btn_id] if 0 <= btn_id < len(styles) else "realism"
        self._os.selected_loras = self._lora_picker.get_activated()
        self._os.lora_multipliers = self._lora_picker.get_multipliers()

    def validate(self) -> bool:
        if not self._name.text().strip():
            QMessageBox.warning(self, "Missing", "Please enter an object name.")
            return False
        if not self._desc.toPlainText().strip():
            QMessageBox.warning(self, "Missing", "Please enter an object description.")
            return False
        self.on_leave()
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Prompt Generation
# ═══════════════════════════════════════════════════════════════════════════


class PromptGenPage(WizardPage):
    """Auto-generate positive + negative prompts using Qwen with custom system prompt."""

    def __init__(self, obj_state: ObjectState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._wizard = wizard
        self._generated = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Positive Prompt</b>"))
        self._pos_edit = QPlainTextEdit()
        self._pos_edit.setMaximumHeight(140)
        layout.addWidget(self._pos_edit)

        pos_row = QHBoxLayout()
        pos_row.addStretch()
        self._reroll_pos = QPushButton("Reroll Positive")
        self._reroll_pos.setFixedWidth(130)
        self._reroll_pos.clicked.connect(lambda: self._enhance("pos"))
        pos_row.addWidget(self._reroll_pos)
        layout.addLayout(pos_row)

        layout.addWidget(QLabel("<b>Negative Prompt</b>"))
        self._neg_edit = QPlainTextEdit()
        self._neg_edit.setMaximumHeight(100)
        layout.addWidget(self._neg_edit)

        neg_row = QHBoxLayout()
        neg_row.addStretch()
        self._reroll_neg = QPushButton("Reroll Negative")
        self._reroll_neg.setFixedWidth(130)
        self._reroll_neg.clicked.connect(lambda: self._enhance("neg"))
        neg_row.addWidget(self._reroll_neg)
        layout.addLayout(neg_row)

        layout.addStretch()

    def on_enter(self) -> None:
        if not self._generated and not self._os.positive_prompt:
            self._enhance("pos")

    def _enhance(self, which: str) -> None:
        desc = self._os.description
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
        self._os.positive_prompt = str(text)
        self._generated = True
        if not self._neg_edit.toPlainText().strip():
            self._enhance("neg")

    @Slot(object)
    def _on_neg_done(self, text) -> None:
        self._neg_edit.setPlainText(str(text))
        self._os.negative_prompt = str(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Prompt enhance error: {msg}")

    def on_leave(self) -> None:
        self._os.positive_prompt = self._pos_edit.toPlainText().strip()
        self._os.negative_prompt = self._neg_edit.toPlainText().strip()
        # Free Qwen VRAM
        self._app_state.unload_qwen()

    def validate(self) -> bool:
        # Save fields without unloading (on_leave called by wizard after validate)
        self._os.positive_prompt = self._pos_edit.toPlainText().strip()
        self._os.negative_prompt = self._neg_edit.toPlainText().strip()
        if not self._os.positive_prompt:
            QMessageBox.warning(self, "Missing", "Please generate or enter a positive prompt.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Base Image Generation
# ═══════════════════════════════════════════════════════════════════════════


class BaseGenPage(WizardPage):
    """Generate 4 candidate base images; user picks one."""

    def __init__(self, obj_state: ObjectState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._gen_btn = QPushButton("Generate Base Images")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate)
        layout.addWidget(self._gen_btn)

        self._gallery = ImageGalleryWidget("Base Images")
        self._gallery.image_selected.connect(self._on_selected)
        layout.addWidget(self._gallery, 1)

    def on_enter(self) -> None:
        if self._os.base_images:
            self._gallery.load_images(self._os.base_images)

    def _generate(self) -> None:
        self._gen_btn.setText("Regenerate")

        cfg = _load_project_config(self._app_state)
        cfg.img_prompt = self._os.positive_prompt
        cfg.img_negative_prompt = self._os.negative_prompt
        cfg.img_batch_count = 4
        cfg.img_batch_size = 1
        cfg.img_steps = self._os.steps
        cfg.img_cfg_scale = self._os.cfg_scale
        cfg.img_width = self._os.width
        cfg.img_height = self._os.height
        cfg.img_seed = self._os.seed
        if self._os.checkpoint:
            cfg.img_checkpoint = self._os.checkpoint
        if self._os.sampler:
            cfg.img_sampler = self._os.sampler
        cfg.img_loras = self._os.selected_loras
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
        self._os.base_images = list(paths) if paths else []
        self._gallery.load_images(self._os.base_images)
        if self._os.base_images:
            self._os.selected_base = self._os.base_images[0]

    @Slot(str)
    def _on_selected(self, path: str) -> None:
        self._os.selected_base = path

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Generation error: {msg}")

    def on_leave(self) -> None:
        # Free SD VRAM when leaving base gen page
        self._app_state.unload_sd_pipelines()

    def validate(self) -> bool:
        if not self._os.selected_base or not Path(self._os.selected_base).is_file():
            QMessageBox.warning(self, "Missing", "Please generate and select a base image.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Mesh Generation
# ═══════════════════════════════════════════════════════════════════════════


class MeshGenPage(WizardPage):
    """Generate a 3D mesh from the selected base image."""

    def __init__(self, obj_state: ObjectState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preview of selected base image
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

    def on_enter(self) -> None:
        if self._os.selected_base and Path(self._os.selected_base).is_file():
            pm = QPixmap(self._os.selected_base).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        if self._os.mesh_path and Path(self._os.mesh_path).is_file():
            self._mesh_status.setText(f"Mesh: {self._os.mesh_path}")

    def on_leave(self) -> None:
        self._os.mesh_resolution = self._res_spin.value()
        self._os.mesh_format = self._format_combo.currentText()
        self._os.remove_bg = self._bg_cb.isChecked()
        # Free TripoSR VRAM when leaving mesh gen page
        self._app_state.unload_triposr_pipeline()

    def _generate(self) -> None:
        # Save field values (without triggering on_leave's TripoSR unload)
        self._os.mesh_resolution = self._res_spin.value()
        self._os.mesh_format = self._format_combo.currentText()
        self._os.remove_bg = self._bg_cb.isChecked()

        # Determine output path
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdqt_mesh_"))
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in self._os.name
        ).strip() or "object"
        output_path = str(tmp_dir / f"{safe_name}.{self._os.mesh_format}")

        # Check for required model downloads
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "3d_modeling", self._app_state.model_registry, self,
            on_progress=lambda f, d: self._wizard._status.setText(d),
        ):
            return

        # Free SD VRAM before loading TripoSR
        self._app_state.unload_sd_pipelines()

        # Load TripoSR pipeline first
        if self._app_state.triposr_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_triposr_pipeline", parent=self)
            self._wizard._run_worker(
                loader,
                on_done=lambda _: self._run_mesh_gen(output_path),
            )
        else:
            self._run_mesh_gen(output_path)

    def _run_mesh_gen(self, output_path: str) -> None:
        from sdqt.workers.model3d import MeshGenerationWorker

        worker = MeshGenerationWorker(
            pipeline=self._app_state.triposr_pipeline,
            image_path=self._os.selected_base,
            output_path=output_path,
            resolution=self._os.mesh_resolution,
            output_format=self._os.mesh_format,
            remove_bg=self._os.remove_bg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_mesh_done, on_error=self._on_error)

    @Slot(object)
    def _on_mesh_done(self, path) -> None:
        self._os.mesh_path = str(path)
        self._mesh_status.setText(f"Mesh saved: {path}")
        self._gen_btn.setText("Regenerate Mesh")
        # Notify the wizard that a mesh is ready (for pages 5-7)
        self._wizard._on_mesh_ready()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Mesh generation error: {msg}")
        self._mesh_status.setText(f"Error: {msg}")

    def validate(self) -> bool:
        if not self._os.mesh_path or not Path(self._os.mesh_path).is_file():
            QMessageBox.warning(self, "Missing", "Please generate a 3D mesh first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Texture Refinement (optional)
# ═══════════════════════════════════════════════════════════════════════════


class TextureRefinePage(WizardPage):
    """Optional texture refinement via SD img2img + UV projection."""

    def __init__(self, obj_state: ObjectState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
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
        # Load mesh into viewport if available
        if self._os.mesh_path and Path(self._os.mesh_path).is_file():
            self._viewport.clear()
            self._viewport.load_mesh(self._os.mesh_path)

        # Pre-fill prompts from object description
        if not self._prompt.toPlainText().strip():
            self._prompt.setPlainText(
                self._os.tex_prompt or self._os.description
            )
        if not self._neg_prompt.toPlainText().strip():
            self._neg_prompt.setPlainText(self._os.tex_neg_prompt)

    def on_leave(self) -> None:
        self._os.tex_prompt = self._prompt.toPlainText().strip()
        self._os.tex_neg_prompt = self._neg_prompt.toPlainText().strip()
        self._os.tex_denoise = self._denoise.value()
        self._os.tex_size = int(self._tex_size.currentText())
        # Free SD VRAM when leaving texture refine page
        self._app_state.unload_sd_pipelines()

    def _refine(self) -> None:
        # Save field values (without triggering on_leave's SD unload)
        self._os.tex_prompt = self._prompt.toPlainText().strip()
        self._os.tex_neg_prompt = self._neg_prompt.toPlainText().strip()
        self._os.tex_denoise = self._denoise.value()
        self._os.tex_size = int(self._tex_size.currentText())

        if not self._os.mesh_path or not Path(self._os.mesh_path).is_file():
            QMessageBox.warning(self, "No Mesh", "No mesh available for texture refinement.")
            return

        # Free TripoSR VRAM before loading SD for img2img
        self._app_state.unload_triposr_pipeline()

        from sdqt.workers.texture_refine import TextureRefineWorker

        worker = TextureRefineWorker(
            mesh_path=self._os.mesh_path,
            render_func=self._viewport.render_single_view,
            app_state=self._app_state,
            prompt=self._os.tex_prompt,
            negative_prompt=self._os.tex_neg_prompt,
            denoise_strength=self._os.tex_denoise,
            tex_size=self._os.tex_size,
            checkpoint=self._os.checkpoint,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_refine_done, on_error=self._on_error)

    @Slot(object)
    def _on_refine_done(self, tex_path) -> None:
        self._os.texture_path = str(tex_path)
        self._refine_btn.setText("Re-refine Texture")
        # Reload texture in viewport
        self._viewport.reload_texture(0)
        self._wizard._status.setText(f"Texture refined: {tex_path}")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Texture refinement error: {msg}")

    def get_viewport(self):
        """Return the GLViewportWidget for reuse on later pages."""
        return self._viewport


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 — Review & Save
# ═══════════════════════════════════════════════════════════════════════════


class ReviewSavePage(WizardPage):
    """Review the 3D object and save. Option to continue to LoRA training."""

    def __init__(self, obj_state: ObjectState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._wizard = wizard
        self._saved = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Small viewport for final preview (will be populated on_enter)
        from sdqt.widgets.gl_viewport import GLViewportWidget
        self._viewport = GLViewportWidget()
        self._viewport.setMaximumHeight(250)
        layout.addWidget(self._viewport)

        # Summary info
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(160)
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.RichText)
        self._info.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._info)
        layout.addWidget(scroll)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._exit_btn = QPushButton("Save && Exit")
        self._exit_btn.setFixedWidth(140)
        self._exit_btn.clicked.connect(self._save_and_exit)
        btn_row.addWidget(self._exit_btn)

        self._lora_btn = QPushButton("Save && Train LoRA")
        self._lora_btn.setFixedWidth(160)
        self._lora_btn.setStyleSheet(
            "QPushButton { background: #0078d4; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #1a8ae8; }"
        )
        self._lora_btn.clicked.connect(self._save_and_continue)
        btn_row.addWidget(self._lora_btn)

        layout.addLayout(btn_row)
        layout.addStretch()

    def on_enter(self) -> None:
        # Load mesh into this page's viewport
        if self._os.mesh_path and Path(self._os.mesh_path).is_file():
            self._viewport.clear()
            self._viewport.load_mesh(self._os.mesh_path)

        # Build summary
        tex_status = "Refined" if self._os.texture_path else "Default (TripoSR)"
        lora_str = ", ".join(self._os.selected_loras) if self._os.selected_loras else "None"
        pos_text = self._os.positive_prompt[:200]
        neg_text = self._os.negative_prompt[:100]

        self._info.setText(
            f"<b>Name:</b> {self._os.name}<br>"
            f"<b>Style:</b> {self._os.style}<br>"
            f"<b>Checkpoint:</b> {self._os.checkpoint or '(project default)'}<br>"
            f"<b>LoRAs:</b> {lora_str}<br>"
            f"<b>Mesh:</b> {self._os.mesh_path}<br>"
            f"<b>Resolution:</b> {self._os.mesh_resolution}<br>"
            f"<b>Texture:</b> {tex_status}<br>"
            f"<b>Positive:</b> {pos_text}<br>"
            f"<b>Negative:</b> {neg_text}"
        )

        self._saved = False
        self._exit_btn.setEnabled(True)
        self._lora_btn.setEnabled(True)

    def _get_save_dir(self) -> Path:
        """Determine and create the save directory."""
        meshes_dir = self._app_state.global_config.meshes_dir
        if not meshes_dir:
            from supremediffusion.config.defaults import APP_ROOT
            meshes_dir = str(APP_ROOT / "library" / "meshes")

        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in self._os.name
        ).strip() or "unnamed"

        save_dir = Path(meshes_dir) / safe_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _save(self) -> Path:
        """Save all assets and metadata. Returns the save directory."""
        save_dir = self._get_save_dir()
        self._os.save_dir = str(save_dir)

        safe_name = save_dir.name

        # Copy mesh
        if self._os.mesh_path and Path(self._os.mesh_path).is_file():
            ext = Path(self._os.mesh_path).suffix
            shutil.copy2(self._os.mesh_path, save_dir / f"{safe_name}{ext}")

        # Copy texture
        if self._os.texture_path and Path(self._os.texture_path).is_file():
            shutil.copy2(self._os.texture_path, save_dir / f"{safe_name}_albedo.png")

        # Copy source image
        if self._os.selected_base and Path(self._os.selected_base).is_file():
            shutil.copy2(self._os.selected_base, save_dir / "source.png")

        # Save metadata
        meta = {
            "name": self._os.name,
            "description": self._os.description,
            "style": self._os.style,
            "positive_prompt": self._os.positive_prompt,
            "negative_prompt": self._os.negative_prompt,
            "checkpoint": self._os.checkpoint,
            "sampler": self._os.sampler,
            "steps": self._os.steps,
            "cfg_scale": self._os.cfg_scale,
            "width": self._os.width,
            "height": self._os.height,
            "loras": self._os.selected_loras,
            "lora_multipliers": self._os.lora_multipliers,
            "mesh_resolution": self._os.mesh_resolution,
            "mesh_format": self._os.mesh_format,
            "texture_refined": bool(self._os.texture_path),
            "tex_prompt": self._os.tex_prompt,
            "tex_denoise": self._os.tex_denoise,
            "tex_size": self._os.tex_size,
        }
        with open(save_dir / "object.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._saved = True
        self._wizard._status.setText(f"Object saved to {save_dir}")
        return save_dir

    def _save_and_exit(self) -> None:
        save_dir = self._save()
        QMessageBox.information(
            self, "Saved", f"Object '{self._os.name}' saved to:\n{save_dir}"
        )
        self._wizard.accept()

    def _save_and_continue(self) -> None:
        self._save()
        self._wizard._continue_to_lora = True
        self._wizard._go_next()

    def validate(self) -> bool:
        # This page uses its own buttons, not the wizard Next
        return self._saved


# ═══════════════════════════════════════════════════════════════════════════
# Page 7 — LoRA Dataset & Training
# ═══════════════════════════════════════════════════════════════════════════

# 16 angles for LoRA training views
_LORA_ANGLES: list[tuple[float, float]] = [
    (-20, 0), (-20, 45), (-20, 90), (-20, 135),
    (-20, 180), (-20, 225), (-20, 270), (-20, 315),
    (-40, 0), (-40, 90), (-40, 180), (-40, 270),
    (0, 0), (0, 90), (0, 180), (0, 270),
]


class LoRADatasetPage(WizardPage):
    """Render multi-angle training views and send to LoRA training tab."""

    def __init__(self, obj_state: ObjectState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._os = obj_state
        self._app_state = app_state
        self._wizard = wizard
        self._rendered = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<b>LoRA Training Dataset</b><br>"
            "Render 16 views of the 3D model for LoRA training."
        ))

        # Model type + checkpoint selector for LoRA target
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("LoRA Model Type:"))
        self._model_type = QComboBox()
        self._model_type.addItems(["SD 1.5", "SDXL"])
        self._model_type.setFixedWidth(100)
        model_row.addWidget(self._model_type)

        model_row.addWidget(QLabel("Checkpoint:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(250)
        model_row.addWidget(self._ckpt_combo, 1)
        model_row.addStretch()
        layout.addLayout(model_row)

        # Viewport for rendering (will be populated on_enter)
        from sdqt.widgets.gl_viewport import GLViewportWidget
        self._viewport = GLViewportWidget()
        self._viewport.setMaximumHeight(250)
        layout.addWidget(self._viewport)

        # Render button
        self._render_btn = QPushButton("Render Training Views")
        self._render_btn.setFixedWidth(200)
        self._render_btn.clicked.connect(self._render_views)
        layout.addWidget(self._render_btn)

        # Gallery to show rendered views
        self._gallery = ImageGalleryWidget("Training Views")
        layout.addWidget(self._gallery, 1)

        # Send to LoRA button
        self._send_btn = QPushButton("Send to LoRA Training")
        self._send_btn.setFixedWidth(200)
        self._send_btn.setEnabled(False)
        self._send_btn.setStyleSheet(
            "QPushButton { background: #0078d4; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #1a8ae8; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self._send_btn.clicked.connect(self._send_to_lora)
        layout.addWidget(self._send_btn, alignment=Qt.AlignRight)

    def on_enter(self) -> None:
        if self._os.mesh_path and Path(self._os.mesh_path).is_file():
            self._viewport.clear()
            self._viewport.load_mesh(self._os.mesh_path)

        # Populate checkpoint combo if empty
        if self._ckpt_combo.count() == 0:
            self._ckpt_infos: dict[str, str] = {}  # name -> model_type
            ckpt_dir = self._app_state.global_config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                try:
                    from supremediffusion.models.sd_models import scan_checkpoints
                    for info in scan_checkpoints(ckpt_dir):
                        self._ckpt_combo.addItem(info.name, info.filename)
                        self._ckpt_infos[info.name] = info.model_type
                except Exception:
                    pass

            # Auto-set model type when checkpoint changes
            self._ckpt_combo.currentTextChanged.connect(self._on_ckpt_changed)

            # Default to the checkpoint used for generation
            if self._os.checkpoint:
                idx = self._ckpt_combo.findText(self._os.checkpoint)
                if idx >= 0:
                    self._ckpt_combo.setCurrentIndex(idx)
                    # Auto-detect type from checkpoint metadata
                    mt = self._ckpt_infos.get(self._os.checkpoint, "")
                    if mt == "sdxl":
                        self._model_type.setCurrentText("SDXL")
                    elif mt == "sd15":
                        self._model_type.setCurrentText("SD 1.5")
            else:
                # Fallback: guess from image dimensions
                if self._os.width >= 1024 or self._os.height >= 1024:
                    self._model_type.setCurrentText("SDXL")
                else:
                    self._model_type.setCurrentText("SD 1.5")

    def _on_ckpt_changed(self, name: str) -> None:
        """Auto-set model type when user changes checkpoint."""
        mt = getattr(self, "_ckpt_infos", {}).get(name, "")
        if mt == "sdxl":
            self._model_type.setCurrentText("SDXL")
        elif mt == "sd15":
            self._model_type.setCurrentText("SD 1.5")

    def _render_views(self) -> None:
        if not self._os.mesh_path or not Path(self._os.mesh_path).is_file():
            QMessageBox.warning(self, "No Mesh", "No mesh to render views from.")
            return

        # Create dataset directory
        save_dir = Path(self._os.save_dir) if self._os.save_dir else self._wizard._get_save_dir()
        dataset_dir = save_dir / "lora_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self._wizard.set_busy(True)
        self._wizard._status.setText("Rendering training views...")

        # Render views
        views = self._viewport.render_views(angles=_LORA_ANGLES, width=512, height=512)

        saved_paths = []
        for i, qimg in enumerate(views):
            if qimg is None or qimg.isNull():
                continue
            # Composite over white background
            white_bg = QImage(qimg.size(), QImage.Format_RGB32)
            white_bg.fill(Qt.white)
            painter = QPainter(white_bg)
            painter.drawImage(0, 0, qimg)
            painter.end()

            out_path = dataset_dir / f"view_{i + 1:03d}.png"
            white_bg.save(str(out_path), "PNG")
            saved_paths.append(str(out_path))

        self._os.lora_dataset_dir = str(dataset_dir)
        self._rendered = True
        self._gallery.load_images(saved_paths)
        self._send_btn.setEnabled(True)
        self._render_btn.setText("Re-render Views")

        self._wizard.set_busy(False)
        self._wizard._status.setText(
            f"Rendered {len(saved_paths)} views to {dataset_dir}"
        )

    def _send_to_lora(self) -> None:
        if not self._os.lora_dataset_dir:
            return
        model_type = self._model_type.currentText()
        checkpoint = self._ckpt_combo.currentText()
        self._wizard.send_to_lora.emit(self._os.lora_dataset_dir, model_type, checkpoint)
        self._wizard.accept()


# ═══════════════════════════════════════════════════════════════════════════
# CreateObjectWizard — ties all 7 pages together
# ═══════════════════════════════════════════════════════════════════════════


class CreateObjectWizard(SequenceWizard):
    """7-page wizard: Info -> Prompt -> Base Gen -> Mesh -> Texture -> Save -> LoRA."""

    send_to_lora = Signal(str, str, str)  # dataset_dir, model_type, checkpoint

    def __init__(self, state, *, img_params: dict | None = None, parent=None) -> None:
        self._os = ObjectState()
        self._continue_to_lora = False

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
            "Image to 3D Model (to LoRA)",
            state,
            header_widget=header,
            parent=parent,
        )
        self.setMinimumSize(850, 700)

        # Populate checkpoint combo
        self._populate_checkpoints()
        self._ckpt_combo.currentTextChanged.connect(self._on_checkpoint_changed)

        # Apply image params from Sequences tab (direct from UI controls)
        if img_params:
            self._os.checkpoint = img_params.get("checkpoint", "")
            self._os.sampler = img_params.get("sampler", "")
            self._os.steps = img_params.get("steps", 20)
            self._os.cfg_scale = img_params.get("cfg_scale", 7.0)
            self._os.width = img_params.get("width", 512)
            self._os.height = img_params.get("height", 512)
            self._os.seed = img_params.get("seed", -1)
        else:
            cfg = ProjectConfig.load(
                state.project_manager.get_project_path(state.current_project or "_default")
            )
            self._os.checkpoint = cfg.img_checkpoint
            self._os.sampler = cfg.img_sampler
            self._os.steps = cfg.img_steps
            self._os.cfg_scale = cfg.img_cfg_scale
            self._os.width = cfg.img_width
            self._os.height = cfg.img_height
            self._os.seed = cfg.img_seed

        if self._os.checkpoint:
            idx = self._ckpt_combo.findText(self._os.checkpoint)
            if idx >= 0:
                self._ckpt_combo.setCurrentIndex(idx)

        # Build pages
        self._texture_page = TextureRefinePage(self._os, state, self, self)
        self._review_page = ReviewSavePage(self._os, state, self, self)

        self._pages = [
            ObjectInfoPage(self._os, state, self),
            PromptGenPage(self._os, state, self, self),
            BaseGenPage(self._os, state, self, self),
            MeshGenPage(self._os, state, self, self),
            self._texture_page,
            self._review_page,
            LoRADatasetPage(self._os, state, self, self),
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
        self._os.checkpoint = name

    def _on_mesh_ready(self) -> None:
        """Called by MeshGenPage when a mesh is generated."""
        # Nothing extra needed — pages 5-7 load the mesh in on_enter

    def _get_save_dir(self) -> Path:
        """Get (or create) the save directory for this object."""
        return self._review_page._get_save_dir()

    def _show_page(self, idx: int) -> None:
        """Override to handle the LoRA page gating."""
        # Page 7 (LoRA) should only be reachable via the Save & Train LoRA button
        if idx == 6 and not self._continue_to_lora:
            # Don't navigate to LoRA page via normal Next
            return
        super()._show_page(idx)
        # Update nav button text for page 6 (Review)
        if self._current_idx == 5:
            # Hide the default Next button on the review page (it has its own buttons)
            self._next_btn.hide()
        else:
            self._next_btn.show()

    def _on_finish(self) -> None:
        self.accept()
