"""Talking Head — 7-page wizard sequence."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
# Custom system prompt for character portrait prompts
# ---------------------------------------------------------------------------

_POS_SYSTEM_PROMPT = (
    "You are an image prompt enhancer for SDXL. Given a character description, "
    "expand it into a detailed portrait prompt. The character should be facing the camera "
    "with a neutral or friendly expression, head and shoulders visible, well-lit face. "
    "Include 'portrait, front-facing, looking at camera, studio lighting, high quality, detailed face'. "
    "Output only the enhanced prompt, nothing else."
)

_NEG_SYSTEM_PROMPT = (
    "You are a negative prompt generator for SDXL image generation. Given a character "
    "description, produce a negative prompt that prevents unwanted elements. "
    "Always include: 'blurry, low quality, deformed, watermark, bad anatomy, "
    "bad hands, cropped, out of frame'. "
    "Output only the negative prompt, nothing else."
)

# ---------------------------------------------------------------------------
# Shared talking-head state
# ---------------------------------------------------------------------------


@dataclass
class TalkingHeadState:
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
    # Face swap
    face_swap_enabled: bool = False
    face_source_path: str = ""
    face_result_path: str = ""
    # TTS
    script_text: str = ""
    voice_ref_path: str = ""
    tts_output_path: str = ""
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    # Video
    video_path: str = ""
    # Lip sync
    face_bbox: list = field(default_factory=list)
    lipsync_output: str = ""
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


# ---------------------------------------------------------------------------
# ObjectPromptWorker — custom Qwen chat for character prompts
# ---------------------------------------------------------------------------


class CharacterPromptWorker(BaseWorker):
    """Generate a character prompt via Qwen with a custom system prompt."""

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
# Page 1 — Character Info
# ═══════════════════════════════════════════════════════════════════════════


class CharInfoPage(WizardPage):
    """Name, description, style, LoRA selection."""

    def __init__(self, th_state: TalkingHeadState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
        self._app_state = app_state
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        row = QHBoxLayout()
        row.addWidget(QLabel("Character Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Elena")
        row.addWidget(self._name)
        layout.addLayout(row)

        # Description
        layout.addWidget(QLabel("Description:"))
        self._desc = QPlainTextEdit()
        self._desc.setPlaceholderText(
            "Describe the character in detail — appearance, clothing, setting, etc."
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
        self._name.setText(self._ts.name)
        self._desc.setPlainText(self._ts.description)

    def on_leave(self) -> None:
        self._ts.name = self._name.text().strip()
        self._ts.description = self._desc.toPlainText().strip()
        styles = ["realism", "cartoon", "anime"]
        btn_id = self._style_group.checkedId()
        self._ts.style = styles[btn_id] if 0 <= btn_id < len(styles) else "realism"
        self._ts.selected_loras = self._lora_picker.get_activated()
        self._ts.lora_multipliers = self._lora_picker.get_multipliers()

    def validate(self) -> bool:
        if not self._name.text().strip():
            QMessageBox.warning(self, "Missing", "Please enter a character name.")
            return False
        if not self._desc.toPlainText().strip():
            QMessageBox.warning(self, "Missing", "Please enter a character description.")
            return False
        self.on_leave()
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Prompt Generation
# ═══════════════════════════════════════════════════════════════════════════


class PromptGenPage(WizardPage):
    """Auto-generate positive + negative prompts using Qwen with character-focused system prompt."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
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
        if not self._generated and not self._ts.positive_prompt:
            self._enhance("pos")

    def _enhance(self, which: str) -> None:
        desc = self._ts.description
        if not desc:
            return
        sys_prompt = _POS_SYSTEM_PROMPT if which == "pos" else _NEG_SYSTEM_PROMPT
        worker = CharacterPromptWorker(
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
        self._generated = True
        if not self._neg_edit.toPlainText().strip():
            self._enhance("neg")

    @Slot(object)
    def _on_neg_done(self, text) -> None:
        self._neg_edit.setPlainText(str(text))
        self._ts.negative_prompt = str(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Prompt enhance error: {msg}")

    def on_leave(self) -> None:
        self._ts.positive_prompt = self._pos_edit.toPlainText().strip()
        self._ts.negative_prompt = self._neg_edit.toPlainText().strip()
        # Free Qwen VRAM
        self._app_state.unload_qwen()

    def validate(self) -> bool:
        # Save fields without unloading (on_leave called by wizard after validate)
        self._ts.positive_prompt = self._pos_edit.toPlainText().strip()
        self._ts.negative_prompt = self._neg_edit.toPlainText().strip()
        if not self._ts.positive_prompt:
            QMessageBox.warning(self, "Missing", "Please generate or enter a positive prompt.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Base Image Generation
# ═══════════════════════════════════════════════════════════════════════════


class BaseGenPage(WizardPage):
    """Generate 4 candidate base images; user picks one."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
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
        if self._ts.base_images:
            self._gallery.load_images(self._ts.base_images)

    def _generate(self) -> None:
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
            self._ts.selected_base = self._ts.base_images[0]

    @Slot(str)
    def _on_selected(self, path: str) -> None:
        self._ts.selected_base = path

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Generation error: {msg}")

    def on_leave(self) -> None:
        # Free SD VRAM when leaving base gen page
        self._app_state.unload_sd_pipelines()

    def validate(self) -> bool:
        if not self._ts.selected_base or not Path(self._ts.selected_base).is_file():
            QMessageBox.warning(self, "Missing", "Please generate and select a base image.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Face Swap (optional)
# ═══════════════════════════════════════════════════════════════════════════


class FaceSwapPage(WizardPage):
    """Optional face swap on the selected base image."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
        self._app_state = app_state
        self._wizard = wizard
        self._swap_completed = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preview of selected base
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(200)
        layout.addWidget(self._preview)

        # Face swap controls
        face_group = QGroupBox("Face Swap")
        face_layout = QVBoxLayout(face_group)

        self._face_cb = QCheckBox("Apply Face Swap")
        self._face_cb.toggled.connect(self._on_face_toggled)
        face_layout.addWidget(self._face_cb)

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("Source Face:"))
        self._face_path = QLineEdit()
        self._face_path.setReadOnly(True)
        face_row.addWidget(self._face_path, 1)
        self._face_browse = QPushButton("Browse")
        self._face_browse.setFixedWidth(75)
        self._face_browse.clicked.connect(self._browse_face)
        self._face_browse.setEnabled(False)
        face_row.addWidget(self._face_browse)
        self._face_lib_btn = QPushButton("Face Library")
        self._face_lib_btn.setFixedWidth(90)
        self._face_lib_btn.clicked.connect(self._browse_face_library)
        self._face_lib_btn.setEnabled(False)
        face_row.addWidget(self._face_lib_btn)
        self._face_char_btn = QPushButton("Character Library")
        self._face_char_btn.setFixedWidth(120)
        self._face_char_btn.clicked.connect(self._browse_face_from_characters)
        self._face_char_btn.setEnabled(False)
        face_row.addWidget(self._face_char_btn)
        face_layout.addLayout(face_row)

        # Enhancer row
        enhance_row = QHBoxLayout()
        enhance_row.setSpacing(6)
        enhance_row.addWidget(QLabel("Face Restore:"))
        self._enhancer_combo = QComboBox()
        self._enhancer_combo.addItems(["None", "gfpgan", "codeformer"])
        self._enhancer_combo.setFixedWidth(120)
        self._enhancer_combo.setEnabled(False)
        enhance_row.addWidget(self._enhancer_combo)
        enhance_row.addWidget(QLabel("Strength:"))
        self._enhancer_strength_spin = QDoubleSpinBox()
        self._enhancer_strength_spin.setRange(0.0, 1.0)
        self._enhancer_strength_spin.setValue(0.5)
        self._enhancer_strength_spin.setSingleStep(0.1)
        self._enhancer_strength_spin.setFixedWidth(80)
        self._enhancer_strength_spin.setEnabled(False)
        enhance_row.addWidget(self._enhancer_strength_spin)
        enhance_row.addStretch()
        face_layout.addLayout(enhance_row)

        self._run_swap_btn = QPushButton("Run Swap")
        self._run_swap_btn.setFixedWidth(120)
        self._run_swap_btn.setEnabled(False)
        self._run_swap_btn.clicked.connect(self._run_face_swap)
        face_layout.addWidget(self._run_swap_btn)

        layout.addWidget(face_group)

        # Result preview
        self._result_label = QLabel()
        self._result_label.setAlignment(Qt.AlignCenter)
        self._result_label.setFixedHeight(200)
        layout.addWidget(self._result_label)

        layout.addStretch()

    def on_enter(self) -> None:
        self._swap_completed = False
        if self._ts.selected_base and Path(self._ts.selected_base).is_file():
            pm = QPixmap(self._ts.selected_base).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        self._face_cb.setChecked(self._ts.face_swap_enabled)
        if self._ts.face_source_path:
            self._face_path.setText(self._ts.face_source_path)
        if self._ts.face_result_path and Path(self._ts.face_result_path).is_file():
            pm = QPixmap(self._ts.face_result_path).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._result_label.setPixmap(pm)

    def _on_face_toggled(self, checked: bool) -> None:
        self._face_browse.setEnabled(checked)
        self._face_lib_btn.setEnabled(checked)
        self._face_char_btn.setEnabled(checked)
        self._enhancer_combo.setEnabled(checked)
        self._enhancer_strength_spin.setEnabled(checked)
        self._run_swap_btn.setEnabled(checked and bool(self._ts.face_source_path))
        self._ts.face_swap_enabled = checked
        self._swap_completed = False

    def _browse_face(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Source", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._face_path.setText(path)
            self._ts.face_source_path = path
            self._run_swap_btn.setEnabled(True)

    def _browse_face_library(self) -> None:
        pngs = _collect_face_library(self._app_state)
        if not pngs:
            QMessageBox.information(self, "Face Library", "No faces in library yet.")
            return
        from sdqt.tabs.image_tabs.face_swap import _FaceLibraryDialog
        from PySide6.QtWidgets import QDialog as _QDlg
        dialog = _FaceLibraryDialog(pngs, self)
        if dialog.exec() == _QDlg.Accepted and dialog.selected_path:
            self._face_path.setText(dialog.selected_path)
            self._ts.face_source_path = dialog.selected_path
            self._run_swap_btn.setEnabled(True)

    def _browse_face_from_characters(self) -> None:
        from sdqt.sequences.create_character import _browse_character_image
        path = _browse_character_image(self._app_state, self)
        if path:
            self._face_path.setText(path)
            self._ts.face_source_path = path
            self._run_swap_btn.setEnabled(True)

    def _run_face_swap(self) -> None:
        from sdqt.workers.image import FaceSwapWorker

        if not self._ts.face_source_path:
            QMessageBox.warning(self, "Missing", "Please select a face source image.")
            return

        cfg = _load_project_config(self._app_state)
        # Apply enhancer from wizard controls
        enhancer = self._enhancer_combo.currentText()
        cfg.faceswap_enhancer = enhancer if enhancer != "None" else ""
        cfg.faceswap_enhancer_strength = self._enhancer_strength_spin.value()
        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")

        target = self._ts.selected_base
        worker = FaceSwapWorker(
            source_path=self._ts.face_source_path,
            target_path=target,
            project_config=cfg,
            models_dir=models_dir,
            enhancer_strength=self._enhancer_strength_spin.value(),
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_swap_done, on_error=self._on_swap_error)

    @Slot(object)
    def _on_swap_done(self, result_path) -> None:
        self._ts.face_result_path = str(result_path)
        if Path(str(result_path)).is_file():
            pm = QPixmap(str(result_path)).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._result_label.setPixmap(pm)
        self._swap_completed = True
        self._wizard._status.setText("Face swap complete.")

    @Slot(str)
    def _on_swap_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Swap Error", msg)

    def validate(self) -> bool:
        # If face swap is not enabled, just pass through
        if not self._ts.face_swap_enabled:
            return True
        # If enabled and swap was run, proceed
        if self._swap_completed and self._ts.face_result_path:
            return True
        # If enabled but not yet run, warn
        if not self._ts.face_source_path:
            QMessageBox.warning(self, "Missing", "Please select a face source image.")
            return False
        if not self._swap_completed:
            QMessageBox.warning(self, "Missing", "Please run the face swap first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Script & TTS
# ═══════════════════════════════════════════════════════════════════════════


class ScriptTTSPage(WizardPage):
    """Enter dialogue text, configure voice, generate speech audio."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Script text
        layout.addWidget(QLabel("<b>Script / Dialogue</b>"))
        self._script_edit = QPlainTextEdit()
        self._script_edit.setPlaceholderText("Enter the text the character will speak...")
        self._script_edit.setMaximumHeight(160)
        layout.addWidget(self._script_edit)

        # Voice reference
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Voice Reference (optional):"))
        self._ref_path = QLineEdit()
        self._ref_path.setReadOnly(True)
        self._ref_path.setPlaceholderText("Chatterbox works without a reference")
        ref_row.addWidget(self._ref_path, 1)
        self._ref_browse = QPushButton("Browse")
        self._ref_browse.setFixedWidth(75)
        self._ref_browse.clicked.connect(self._browse_ref)
        ref_row.addWidget(self._ref_browse)
        layout.addLayout(ref_row)

        # Sliders
        slider_row = QHBoxLayout()
        slider_row.setSpacing(6)

        slider_row.addWidget(QLabel("Exaggeration:"))
        self._exag_spin = QDoubleSpinBox()
        self._exag_spin.setRange(0.0, 1.0)
        self._exag_spin.setValue(0.5)
        self._exag_spin.setSingleStep(0.05)
        self._exag_spin.setFixedWidth(80)
        slider_row.addWidget(self._exag_spin)

        slider_row.addWidget(QLabel("CFG Weight:"))
        self._cfg_spin = QDoubleSpinBox()
        self._cfg_spin.setRange(0.0, 1.0)
        self._cfg_spin.setValue(0.5)
        self._cfg_spin.setSingleStep(0.05)
        self._cfg_spin.setFixedWidth(80)
        slider_row.addWidget(self._cfg_spin)

        slider_row.addStretch()
        layout.addLayout(slider_row)

        # Generate button
        self._gen_btn = QPushButton("Generate Speech")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate_speech)
        layout.addWidget(self._gen_btn)

        # Audio output info
        self._audio_info = QLabel()
        self._audio_info.setStyleSheet("color: #aaa; font-size: 12px;")
        self._audio_info.setWordWrap(True)
        layout.addWidget(self._audio_info)

        layout.addStretch()

    def on_enter(self) -> None:
        if self._ts.script_text:
            self._script_edit.setPlainText(self._ts.script_text)
        if self._ts.voice_ref_path:
            self._ref_path.setText(self._ts.voice_ref_path)
        self._exag_spin.setValue(self._ts.exaggeration)
        self._cfg_spin.setValue(self._ts.cfg_weight)
        if self._ts.tts_output_path and Path(self._ts.tts_output_path).is_file():
            self._audio_info.setText(f"Audio: {self._ts.tts_output_path}")

    def _browse_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Reference", "", "Audio (*.wav *.mp3 *.flac *.ogg)"
        )
        if path:
            self._ref_path.setText(path)
            self._ts.voice_ref_path = path

    def _generate_speech(self) -> None:
        text = self._script_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Missing", "Please enter script text.")
            return

        self._ts.script_text = text
        self._ts.exaggeration = self._exag_spin.value()
        self._ts.cfg_weight = self._cfg_spin.value()

        # Create temp output path
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="sdqt_tts_")
        tmp.close()
        output_path = tmp.name

        from sdqt.workers.chatterbox_tts import ChatterboxTTSWorker

        ref_audio = self._ts.voice_ref_path if self._ts.voice_ref_path else None
        worker = ChatterboxTTSWorker(
            text=text,
            output_path=output_path,
            ref_audio=ref_audio,
            exaggeration=self._ts.exaggeration,
            cfg_weight=self._ts.cfg_weight,
            app_state=self._app_state,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_tts_done, on_error=self._on_tts_error)

    @Slot(object)
    def _on_tts_done(self, output_path) -> None:
        self._ts.tts_output_path = str(output_path)
        # Show duration info via ffprobe
        duration_str = ""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(output_path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                dur = float(result.stdout.strip())
                duration_str = f" ({dur:.1f}s)"
        except Exception:
            pass
        self._audio_info.setText(f"Audio saved: {output_path}{duration_str}")
        self._gen_btn.setText("Regenerate Speech")

    @Slot(str)
    def _on_tts_error(self, msg: str) -> None:
        self._wizard._status.setText(f"TTS error: {msg}")

    def on_leave(self) -> None:
        self._ts.script_text = self._script_edit.toPlainText().strip()
        self._ts.voice_ref_path = self._ref_path.text().strip()
        self._ts.exaggeration = self._exag_spin.value()
        self._ts.cfg_weight = self._cfg_spin.value()

    def validate(self) -> bool:
        self._ts.script_text = self._script_edit.toPlainText().strip()
        self._ts.exaggeration = self._exag_spin.value()
        self._ts.cfg_weight = self._cfg_spin.value()
        if not self._ts.script_text:
            QMessageBox.warning(self, "Missing", "Please enter script text.")
            return False
        if not self._ts.tts_output_path or not Path(self._ts.tts_output_path).is_file():
            QMessageBox.warning(self, "Missing", "Please generate speech audio first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 — Video Generation (Wan I2V)
# ═══════════════════════════════════════════════════════════════════════════


class VideoGenPage(WizardPage):
    """Generate a video from the base image using Wan I2V pipeline."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Preview of the image that will be animated
        layout.addWidget(QLabel("<b>Source Image for Video</b>"))
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(200)
        layout.addWidget(self._preview)

        # Generate button
        self._gen_btn = QPushButton("Generate Video")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate)
        layout.addWidget(self._gen_btn)

        # Status
        self._video_info = QLabel()
        self._video_info.setStyleSheet("color: #aaa; font-size: 12px;")
        self._video_info.setWordWrap(True)
        layout.addWidget(self._video_info)

        layout.addStretch()

    def on_enter(self) -> None:
        # Show the face-swapped result if available, otherwise the selected base
        img_path = self._ts.face_result_path if (
            self._ts.face_swap_enabled and self._ts.face_result_path
            and Path(self._ts.face_result_path).is_file()
        ) else self._ts.selected_base
        if img_path and Path(img_path).is_file():
            pm = QPixmap(img_path).scaled(
                400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        if self._ts.video_path and Path(self._ts.video_path).is_file():
            self._video_info.setText(f"Video: {self._ts.video_path}")

    def _generate(self) -> None:
        # Determine the source image
        image_path = self._ts.face_result_path if (
            self._ts.face_swap_enabled and self._ts.face_result_path
            and Path(self._ts.face_result_path).is_file()
        ) else self._ts.selected_base

        if not image_path or not Path(image_path).is_file():
            QMessageBox.warning(self, "Missing", "No source image available.")
            return

        self._gen_btn.setEnabled(False)

        # Load video pipelines first
        if self._app_state.pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_pipelines", parent=self)
            self._wizard._run_worker(
                loader,
                on_done=lambda _: self._run_inference(image_path),
            )
        else:
            self._run_inference(image_path)

    def _run_inference(self, image_path: str) -> None:
        from sdqt.workers.inference import InferenceWorker

        cfg = _load_project_config(self._app_state)
        worker = InferenceWorker(
            pipeline=self._app_state.pipeline,
            project_name=_project_name(self._app_state),
            project_config=cfg,
            mode=1,
            image_path=image_path,
            global_config=self._app_state.global_config,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_video_done, on_error=self._on_video_error)

    @Slot(object)
    def _on_video_done(self, video_path) -> None:
        self._ts.video_path = str(video_path)
        self._video_info.setText(f"Video saved: {video_path}")
        self._gen_btn.setText("Regenerate Video")
        self._gen_btn.setEnabled(True)

    @Slot(str)
    def _on_video_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Video generation error: {msg}")
        self._gen_btn.setEnabled(True)

    def on_leave(self) -> None:
        # Free video pipeline VRAM
        self._app_state.unload_video_pipelines()

    def validate(self) -> bool:
        if not self._ts.video_path or not Path(self._ts.video_path).is_file():
            QMessageBox.warning(self, "Missing", "Please generate a video first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 7 — Lip Sync (MuseTalk)
# ═══════════════════════════════════════════════════════════════════════════


class LipSyncPage(WizardPage):
    """Detect face in video, run MuseTalk lip sync, save output."""

    def __init__(self, th_state: TalkingHeadState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._ts = th_state
        self._app_state = app_state
        self._wizard = wizard
        self._face_detected = False
        self._lipsync_done = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<b>Lip Sync</b>"))

        # Face detection status
        self._face_info = QLabel("Face detection will run automatically...")
        self._face_info.setStyleSheet("color: #aaa; font-size: 12px;")
        self._face_info.setWordWrap(True)
        layout.addWidget(self._face_info)

        # Lip sync button
        self._sync_btn = QPushButton("Run Lip Sync")
        self._sync_btn.setFixedWidth(200)
        self._sync_btn.setEnabled(False)
        self._sync_btn.clicked.connect(self._run_lipsync)
        layout.addWidget(self._sync_btn)

        # Output info
        self._output_info = QLabel()
        self._output_info.setStyleSheet("color: #aaa; font-size: 12px;")
        self._output_info.setWordWrap(True)
        layout.addWidget(self._output_info)

        # Save button
        self._save_btn = QPushButton("Save to Project")
        self._save_btn.setFixedWidth(160)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_output)
        layout.addWidget(self._save_btn)

        layout.addStretch()

    def on_enter(self) -> None:
        self._face_detected = False
        self._lipsync_done = False
        self._sync_btn.setEnabled(False)
        self._save_btn.setEnabled(False)

        if self._ts.lipsync_output and Path(self._ts.lipsync_output).is_file():
            self._output_info.setText(f"Output: {self._ts.lipsync_output}")
            self._lipsync_done = True
            self._save_btn.setEnabled(True)
            return

        # Auto-detect face in first frame of the video
        if self._ts.video_path and Path(self._ts.video_path).is_file():
            self._extract_first_frame()

    def _extract_first_frame(self) -> None:
        """Extract the first frame of the video for face detection."""
        self._face_info.setText("Extracting first frame...")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="sdqt_frame_")
        tmp.close()
        self._first_frame_path = tmp.name

        try:
            from supremediffusion.utils.video import extract_single_frame
            img = extract_single_frame(self._ts.video_path, 0)
            img.save(self._first_frame_path, "PNG")
        except Exception as exc:
            self._face_info.setText(f"Failed to extract frame: {exc}")
            return

        if not Path(self._first_frame_path).is_file():
            self._face_info.setText("Failed to extract first frame.")
            return

        self._run_face_detect()

    def _run_face_detect(self) -> None:
        from sdqt.workers.image import FaceDetectWorker

        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")
        worker = FaceDetectWorker(
            image_path=self._first_frame_path,
            models_dir=models_dir,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_face_detected, on_error=self._on_detect_error)

    @Slot(object)
    def _on_face_detected(self, faces) -> None:
        if not faces:
            self._face_info.setText("No face detected in video. Lip sync may not work.")
            return

        bbox = faces[0]["bbox"]
        self._ts.face_bbox = [int(v) for v in bbox]
        self._face_detected = True
        self._sync_btn.setEnabled(True)
        self._face_info.setText(
            f"Face detected: bbox [{self._ts.face_bbox[0]}, {self._ts.face_bbox[1]}, "
            f"{self._ts.face_bbox[2]}, {self._ts.face_bbox[3]}]"
        )

    @Slot(str)
    def _on_detect_error(self, msg: str) -> None:
        self._face_info.setText(f"Face detection error: {msg}")

    def _run_lipsync(self) -> None:
        if not self._ts.face_bbox:
            QMessageBox.warning(self, "Missing", "No face bounding box detected.")
            return
        if not self._ts.tts_output_path or not Path(self._ts.tts_output_path).is_file():
            QMessageBox.warning(self, "Missing", "No TTS audio available.")
            return

        # Determine output path
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in self._ts.name
        ).strip() or "talking_head"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            _project_path(self._app_state) / "outputs" / f"talking_head_{safe_name}_{timestamp}.mp4"
        )
        # Ensure outputs dir exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        from sdqt.workers.musetalk import MuseTalkWorker

        worker = MuseTalkWorker(
            video_path=self._ts.video_path,
            audio_path=self._ts.tts_output_path,
            face_bbox=self._ts.face_bbox,
            output_path=output_path,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_lipsync_done, on_error=self._on_lipsync_error)

    @Slot(object)
    def _on_lipsync_done(self, output_path) -> None:
        self._ts.lipsync_output = str(output_path)
        self._lipsync_done = True
        self._output_info.setText(f"Lip sync output: {output_path}")
        self._save_btn.setEnabled(True)
        self._sync_btn.setText("Re-run Lip Sync")

    @Slot(str)
    def _on_lipsync_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Lip sync error: {msg}")

    def _save_output(self) -> None:
        if not self._ts.lipsync_output or not Path(self._ts.lipsync_output).is_file():
            return

        # Copy to talking_heads subdir under project
        save_dir = _project_path(self._app_state) / "talking_heads"
        save_dir.mkdir(parents=True, exist_ok=True)

        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in self._ts.name
        ).strip() or "talking_head"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = save_dir / f"{safe_name}_{timestamp}.mp4"

        shutil.copy2(self._ts.lipsync_output, dest)
        self._ts.save_dir = str(save_dir)

        self._save_btn.setEnabled(False)
        self._wizard._status.setText(f"Saved to {dest}")
        QMessageBox.information(
            self, "Saved", f"Talking head video saved to:\n{dest}"
        )

    def validate(self) -> bool:
        # Allow finishing even without saving — the output file exists
        if self._ts.lipsync_output and Path(self._ts.lipsync_output).is_file():
            return True
        QMessageBox.warning(self, "Missing", "Please run lip sync first.")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# CreateTalkingHeadWizard — ties all 7 pages together
# ═══════════════════════════════════════════════════════════════════════════


class CreateTalkingHeadWizard(SequenceWizard):
    """7-page wizard: Info -> Prompt -> Base Gen -> Face Swap -> TTS -> Video -> Lip Sync."""

    def __init__(self, state, *, img_params: dict | None = None, parent=None) -> None:
        self._ts = TalkingHeadState()

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
            "Talking Head",
            state,
            header_widget=header,
            parent=parent,
        )

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

        # Set combo to current checkpoint
        if self._ts.checkpoint:
            idx = self._ckpt_combo.findText(self._ts.checkpoint)
            if idx >= 0:
                self._ckpt_combo.setCurrentIndex(idx)

        # Build pages
        self._pages = [
            CharInfoPage(self._ts, state, self),
            PromptGenPage(self._ts, state, self, self),
            BaseGenPage(self._ts, state, self, self),
            FaceSwapPage(self._ts, state, self, self),
            ScriptTTSPage(self._ts, state, self, self),
            VideoGenPage(self._ts, state, self, self),
            LipSyncPage(self._ts, state, self, self),
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

    def _on_finish(self) -> None:
        self.accept()
