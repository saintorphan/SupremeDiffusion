"""Sequences tab — guided multi-step wizards for chaining operations."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab

logger = logging.getLogger(__name__)

# ── Card style ──────────────────────────────────────────────────────────────

_CARD_STYLE = (
    "QFrame#seqCard { background: #2a2a2a; border: 1px solid #444; border-radius: 6px; }"
    "QFrame#seqCard:hover { border-color: #0078d4; background: #2f2f2f; }"
    "QFrame#seqCard QLabel { background: transparent; border: none; padding: 0px; }"
    "QFrame#seqCard QPushButton { background: #3a3a3a; border: 1px solid #555;"
    " border-radius: 3px; padding: 4px 12px; color: #ddd; }"
    "QFrame#seqCard QPushButton:hover { background: #4a4a4a; border-color: #0078d4; }"
)

_CARD_TITLE_STYLE = "font-size: 18px; font-weight: bold; color: #ddd; background: transparent; border: none;"
_CARD_DESC_STYLE = "font-size: 14px; color: #aaa; background: transparent; border: none;"
_HEADER_STYLE = "color: #999; font-size: 12px;"

# ── Sequence definitions ───────────────────────────────────────────────────

SEQUENCES = [
    {
        "id": "create_character",
        "title": "Create a Character",
        "description": (
            "Go from a text description to a fully posed character sheet. "
            "Generates prompts, base images, optional face/body swap, "
            "pose variants, and saves everything to the character library."
        ),
    },
    {
        "id": "image_to_3d",
        "title": "Image to 3D Model (to LoRA)",
        "description": (
            "Generate an image, convert it to a 3D mesh, optionally refine "
            "textures, and save the model. Continue to render multi-angle "
            "views for LoRA training if desired."
        ),
    },
    {
        "id": "talking_head",
        "title": "Talking Head",
        "description": (
            "Create a speaking character video from scratch. Generates a "
            "portrait, optional face swap, text-to-speech with voice cloning, "
            "video animation, and lip sync — all in one guided flow."
        ),
    },
    {
        "id": "style_transfer_lora",
        "title": "Style Transfer LoRA",
        "description": (
            "Collect or generate style reference images, auto-crop and resize, "
            "batch caption with trigger words, and send the prepared dataset "
            "to LoRA training."
        ),
    },
    {
        "id": "product_turntable",
        "title": "Product Turntable",
        "description": (
            "Turn a product image into a 3D model and render an animated "
            "turntable video. Optional background removal, texture refinement, "
            "and custom background compositing."
        ),
    },
    {
        "id": "character_video",
        "title": "Character Video Series",
        "description": (
            "Load a saved character and generate multiple scene videos with "
            "consistent face identity. Describe scenes, generate images, "
            "face swap for consistency, and animate each scene."
        ),
    },
    {
        "id": "character_replacement",
        "title": "Character Replacement",
        "description": (
            "Replace a character's face in an existing video. Select a source "
            "face from an image or the character library, detect faces in the "
            "video, and run frame-by-frame face swap."
        ),
    },
    {
        "id": "character_dataset_builder",
        "title": "Character Dataset Builder",
        "description": (
            "Build a video LoRA training dataset for character consistency. "
            "Select reference stills, batch-generate I2V clips with varied "
            "actions, curate results, auto-caption with trigger tokens, and "
            "send to Video LoRA training."
        ),
    },
]


class SequencesTab(BaseTab):
    """Top-level tab showing model info header and a grid of sequence cards."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Image model settings header ───────────────────────────────────
        header = QFrame()
        header.setStyleSheet("QFrame { background: #1e1e1e; border-radius: 4px; padding: 8px; }")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(12, 6, 12, 6)
        h_layout.setSpacing(6)

        title_label = QLabel("<b>Image Generation Settings</b>")
        title_label.setStyleSheet("color: #ccc; font-size: 13px;")
        h_layout.addWidget(title_label)

        # Row 1: Checkpoint + VAE
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Checkpoint:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(200)
        row1.addWidget(self._ckpt_combo, 1)
        row1.addWidget(QLabel("VAE:"))
        self._vae_combo = QComboBox()
        self._vae_combo.addItem("Automatic")
        self._vae_combo.setFixedWidth(160)
        row1.addWidget(self._vae_combo)
        h_layout.addLayout(row1)

        # Row 2: Sampler + Steps + CFG + Dimensions + Seed
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Sampler:"))
        self._sampler_combo = QComboBox()
        self._sampler_combo.setFixedWidth(140)
        row2.addWidget(self._sampler_combo)
        row2.addWidget(QLabel("Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 150)
        self._steps_spin.setValue(20)
        self._steps_spin.setFixedWidth(75)
        row2.addWidget(self._steps_spin)
        row2.addWidget(QLabel("CFG:"))
        self._cfg_spin = QDoubleSpinBox()
        self._cfg_spin.setRange(1.0, 30.0)
        self._cfg_spin.setDecimals(1)
        self._cfg_spin.setValue(7.0)
        self._cfg_spin.setFixedWidth(80)
        row2.addWidget(self._cfg_spin)
        row2.addWidget(QLabel("W:"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(64, 4096)
        self._width_spin.setSingleStep(64)
        self._width_spin.setValue(512)
        self._width_spin.setFixedWidth(75)
        row2.addWidget(self._width_spin)
        row2.addWidget(QLabel("H:"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(64, 4096)
        self._height_spin.setSingleStep(64)
        self._height_spin.setValue(512)
        self._height_spin.setFixedWidth(75)
        row2.addWidget(self._height_spin)
        row2.addWidget(QLabel("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 999999999)
        self._seed_spin.setValue(-1)
        self._seed_spin.setFixedWidth(75)
        row2.addWidget(self._seed_spin)
        row2.addStretch()
        h_layout.addLayout(row2)

        layout.addWidget(header)

        # ── Video generation settings header ──────────────────────────────
        vid_header = QFrame()
        vid_header.setStyleSheet("QFrame { background: #1e1e1e; border-radius: 4px; padding: 8px; }")
        v_layout = QVBoxLayout(vid_header)
        v_layout.setContentsMargins(12, 6, 12, 6)
        v_layout.setSpacing(6)

        vid_title = QLabel("<b>Video Generation Settings</b>")
        vid_title.setStyleSheet("color: #ccc; font-size: 13px;")
        v_layout.addWidget(vid_title)

        # Row: Model + Resolution + Steps + Guidance + FPS
        vrow = QHBoxLayout()
        vrow.setSpacing(6)
        vrow.addWidget(QLabel("Model:"))
        self._vid_model_combo = QComboBox()
        self._vid_model_combo.setFixedWidth(200)
        from sdqt.widgets.generation_params import _MODEL_TYPES, _RESOLUTION_PRESETS
        for display, key in _MODEL_TYPES:
            self._vid_model_combo.addItem(display, userData=key)
        vrow.addWidget(self._vid_model_combo)
        vrow.addWidget(QLabel("Res:"))
        self._vid_res_combo = QComboBox()
        self._vid_res_combo.setFixedWidth(140)
        self._vid_res_combo.addItems(_RESOLUTION_PRESETS)
        vrow.addWidget(self._vid_res_combo)
        vrow.addWidget(QLabel("Steps:"))
        self._vid_steps_spin = QSpinBox()
        self._vid_steps_spin.setRange(1, 100)
        self._vid_steps_spin.setValue(4)
        self._vid_steps_spin.setFixedWidth(75)
        vrow.addWidget(self._vid_steps_spin)
        vrow.addWidget(QLabel("Guidance:"))
        self._vid_guidance_spin = QDoubleSpinBox()
        self._vid_guidance_spin.setRange(0.0, 20.0)
        self._vid_guidance_spin.setDecimals(1)
        self._vid_guidance_spin.setValue(1.0)
        self._vid_guidance_spin.setFixedWidth(80)
        vrow.addWidget(self._vid_guidance_spin)
        vrow.addWidget(QLabel("FPS:"))
        self._vid_fps_spin = QSpinBox()
        self._vid_fps_spin.setRange(1, 60)
        self._vid_fps_spin.setValue(16)
        self._vid_fps_spin.setFixedWidth(75)
        vrow.addWidget(self._vid_fps_spin)
        vrow.addStretch()
        v_layout.addLayout(vrow)

        layout.addWidget(vid_header)

        # ── Auto-save on any control change ───────────────────────────────
        self._ckpt_combo.currentTextChanged.connect(lambda: self._auto_save())
        self._vae_combo.currentTextChanged.connect(lambda: self._auto_save())
        self._sampler_combo.currentTextChanged.connect(lambda: self._auto_save())
        self._steps_spin.valueChanged.connect(lambda: self._auto_save())
        self._cfg_spin.valueChanged.connect(lambda: self._auto_save())
        self._width_spin.valueChanged.connect(lambda: self._auto_save())
        self._height_spin.valueChanged.connect(lambda: self._auto_save())
        self._seed_spin.valueChanged.connect(lambda: self._auto_save())
        self._vid_model_combo.currentIndexChanged.connect(lambda: self._auto_save())
        self._vid_res_combo.currentTextChanged.connect(lambda: self._auto_save())
        self._vid_steps_spin.valueChanged.connect(lambda: self._auto_save())
        self._vid_guidance_spin.valueChanged.connect(lambda: self._auto_save())
        self._vid_fps_spin.valueChanged.connect(lambda: self._auto_save())

        # ── Scrollable card grid ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        self._grid = QGridLayout(container)
        self._grid.setSpacing(16)
        self._grid.setContentsMargins(8, 8, 8, 8)

        # Equal column widths
        for c in range(3):
            self._grid.setColumnStretch(c, 1)

        for i, seq in enumerate(SEQUENCES):
            card = self._make_card(seq)
            col = i % 3
            row = i // 3
            self._grid.addWidget(card, row, col)

        container.setLayout(self._grid)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

    def _make_card(self, seq: dict) -> QFrame:
        card = QFrame()
        card.setObjectName("seqCard")
        card.setStyleSheet(_CARD_STYLE)
        card.setFixedHeight(180)

        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(6)

        title = QLabel(seq["title"])
        title.setStyleSheet(_CARD_TITLE_STYLE)
        v.addWidget(title)

        desc = QLabel(seq["description"])
        desc.setStyleSheet(_CARD_DESC_STYLE)
        desc.setWordWrap(True)
        desc.setMaximumHeight(70)
        v.addWidget(desc, 1)

        btn = QPushButton("Start")
        btn.setFixedWidth(80)
        btn.clicked.connect(lambda checked=False, s=seq: self._launch_sequence(s))
        v.addWidget(btn, alignment=Qt.AlignRight)

        return card

    # ── Populate controls ────────────────────────────────────────────────

    def _populate_controls(self) -> None:
        """Populate dropdowns from model dirs and restore from project config."""
        gc = self.state.global_config

        # Checkpoints
        if self._ckpt_combo.count() == 0:
            ckpt_dir = gc.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                try:
                    from supremediffusion.models.sd_models import scan_checkpoints
                    for info in scan_checkpoints(ckpt_dir):
                        self._ckpt_combo.addItem(info.name)
                except Exception:
                    pass

        # VAEs
        if self._vae_combo.count() <= 1:
            vae_dir = gc.model_paths.get("sd_vae_dir", "")
            if vae_dir:
                try:
                    from supremediffusion.models.sd_models import scan_vaes
                    for name in scan_vaes(vae_dir):
                        self._vae_combo.addItem(name)
                except Exception:
                    pass

        # Samplers
        if self._sampler_combo.count() == 0:
            try:
                from supremediffusion.models.sd_samplers import list_samplers
                self._sampler_combo.addItems(list_samplers())
            except Exception:
                self._sampler_combo.addItems(["DPM++ 2M", "Euler a", "Euler", "DDIM"])

        # Restore from project config
        self._restore_from_config()

    def _restore_from_config(self) -> None:
        """Restore control values from the current project config."""
        self._restoring = True
        from supremediffusion.config.project_config import ProjectConfig
        cfg = ProjectConfig.load(self.project_path)

        if cfg.img_checkpoint:
            idx = self._ckpt_combo.findText(cfg.img_checkpoint)
            if idx >= 0:
                self._ckpt_combo.setCurrentIndex(idx)
        if cfg.img_vae:
            idx = self._vae_combo.findText(cfg.img_vae)
            if idx >= 0:
                self._vae_combo.setCurrentIndex(idx)
        if cfg.img_sampler:
            idx = self._sampler_combo.findText(cfg.img_sampler)
            if idx >= 0:
                self._sampler_combo.setCurrentIndex(idx)
        self._steps_spin.setValue(cfg.img_steps)
        self._cfg_spin.setValue(cfg.img_cfg_scale)
        self._width_spin.setValue(cfg.img_width)
        self._height_spin.setValue(cfg.img_height)
        self._seed_spin.setValue(cfg.img_seed)

        # Video settings
        if cfg.model_type:
            idx = self._vid_model_combo.findData(cfg.model_type)
            if idx >= 0:
                self._vid_model_combo.setCurrentIndex(idx)
        if cfg.resolution:
            idx = self._vid_res_combo.findText(cfg.resolution)
            if idx >= 0:
                self._vid_res_combo.setCurrentIndex(idx)
        self._vid_steps_spin.setValue(cfg.num_inference_steps)
        self._vid_guidance_spin.setValue(cfg.guidance_scale)
        self._vid_fps_spin.setValue(cfg.fps)
        self._restoring = False

    def _save_to_config(self) -> None:
        """Persist current control values to the project config."""
        from supremediffusion.config.project_config import ProjectConfig

        cfg = ProjectConfig.load(self.project_path)
        cfg.img_checkpoint = self._ckpt_combo.currentText()
        cfg.img_vae = self._vae_combo.currentText()
        cfg.img_sampler = self._sampler_combo.currentText()
        cfg.img_steps = self._steps_spin.value()
        cfg.img_cfg_scale = self._cfg_spin.value()
        cfg.img_width = self._width_spin.value()
        cfg.img_height = self._height_spin.value()
        cfg.img_seed = self._seed_spin.value()

        # Video settings
        cfg.model_type = self._vid_model_combo.currentData() or self._vid_model_combo.currentText()
        cfg.resolution = self._vid_res_combo.currentText()
        cfg.num_inference_steps = self._vid_steps_spin.value()
        cfg.guidance_scale = self._vid_guidance_spin.value()
        cfg.fps = self._vid_fps_spin.value()
        cfg.save(self.project_path)

    def _get_image_params(self) -> dict:
        """Return current image generation settings from the UI controls."""
        return {
            "checkpoint": self._ckpt_combo.currentText(),
            "vae": self._vae_combo.currentText(),
            "sampler": self._sampler_combo.currentText(),
            "steps": self._steps_spin.value(),
            "cfg_scale": self._cfg_spin.value(),
            "width": self._width_spin.value(),
            "height": self._height_spin.value(),
            "seed": self._seed_spin.value(),
        }

    def _auto_save(self) -> None:
        """Save config whenever a control changes (skip during restore)."""
        if getattr(self, "_restoring", False):
            return
        self._save_to_config()

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._restore_from_config()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._populate_controls()

    # ── Launch sequences ─────────────────────────────────────────────────

    def _launch_sequence(self, seq: dict) -> None:
        seq_id = seq["id"]
        handler = {
            "create_character": self._launch_create_character,
            "image_to_3d": self._launch_image_to_3d,
            "talking_head": self._launch_talking_head,
            "style_transfer_lora": self._launch_style_transfer_lora,
            "product_turntable": self._launch_product_turntable,
            "character_video": self._launch_character_video,
            "character_replacement": self._launch_character_replacement,
            "character_dataset_builder": self._launch_character_dataset_builder,
        }.get(seq_id)
        if handler:
            handler()
        else:
            self._show_status(f"Unknown sequence: {seq_id}")

    def _launch_create_character(self) -> None:
        self._save_to_config()
        from sdqt.sequences.create_character import CreateCharacterWizard

        wizard = CreateCharacterWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        wizard.exec()

    def _launch_image_to_3d(self) -> None:
        self._save_to_config()
        from sdqt.sequences.image_to_3d import CreateObjectWizard

        wizard = CreateObjectWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        main_win = self.window()
        if hasattr(main_win, "_on_3d_lora_dataset_with_model"):
            wizard.send_to_lora.connect(main_win._on_3d_lora_dataset_with_model)
        wizard.exec()

    def _launch_talking_head(self) -> None:
        self._save_to_config()
        from sdqt.sequences.talking_head import CreateTalkingHeadWizard

        wizard = CreateTalkingHeadWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        wizard.exec()

    def _launch_style_transfer_lora(self) -> None:
        self._save_to_config()
        from sdqt.sequences.style_transfer_lora import StyleTransferWizard

        wizard = StyleTransferWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        main_win = self.window()
        if hasattr(main_win, "_on_3d_lora_dataset_with_model"):
            wizard.send_to_lora.connect(main_win._on_3d_lora_dataset_with_model)
        wizard.exec()

    def _launch_product_turntable(self) -> None:
        self._save_to_config()
        from sdqt.sequences.product_turntable import ProductTurntableWizard

        wizard = ProductTurntableWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        wizard.exec()

    def _launch_character_video(self) -> None:
        self._save_to_config()
        from sdqt.sequences.character_video import CharacterVideoWizard

        wizard = CharacterVideoWizard(self.state, img_params=self._get_image_params(), parent=self.window())
        wizard.status_message.connect(self._show_status)
        wizard.exec()

    def _launch_character_replacement(self) -> None:
        self._save_to_config()
        from sdqt.sequences.character_replacement import CharacterReplacementWizard

        wizard = CharacterReplacementWizard(self.state, parent=self.window())
        wizard.status_message.connect(self._show_status)
        wizard.exec()

    def _launch_character_dataset_builder(self) -> None:
        self._save_to_config()
        from sdqt.sequences.character_dataset_builder import CharacterDatasetWizard

        wizard = CharacterDatasetWizard(self.state, parent=self.window())
        wizard.status_message.connect(self._show_status)
        main_win = self.window()
        if hasattr(main_win, "_on_video_lora_dataset"):
            wizard.send_to_video_lora.connect(main_win._on_video_lora_dataset)
        wizard.exec()
