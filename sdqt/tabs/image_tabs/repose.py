"""RePose sub-tab for Image Suite.

Re-poses a person in an image using a pose from a separate reference image
while preserving the original person's clothing, shading, and appearance.
Uses ControlNet Openpose + IP-Adapter + SDXL inpainting (same backend as Body Double).
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.tabs.base import BaseTab
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.mask_editor import MaskEditorWidget
from sdqt.widgets.prompt_enhance import PromptEnhanceWidget
from sdqt.workers.image import RePoseWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

logger = logging.getLogger(__name__)

_W_SPIN = 75
_W_DSPIN = 80
_W_COMBO = 160


class RePoseTab(BaseTab):
    """RePose tab -- re-pose a person while preserving their appearance."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: RePoseWorker | None = None
        self._result_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left panel: source + pose ref + mask + settings --
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Source image + pose reference side by side
        images_row = QHBoxLayout()
        images_row.setSpacing(6)
        self._source = ImageDropWidget("Source Image", thumb_height=180)
        self._source.image_loaded.connect(lambda p: self._persist_source_path("repose_source_path", p))
        self._source.image_cleared.connect(lambda: self._persist_source_path("repose_source_path", ""))
        self._pose_ref = ImageDropWidget("Pose Reference", thumb_height=180)
        self._pose_ref.image_loaded.connect(lambda p: self._persist_source_path("repose_pose_path", p))
        self._pose_ref.image_cleared.connect(lambda: self._persist_source_path("repose_pose_path", ""))
        images_row.addWidget(self._source, 1)
        images_row.addWidget(self._pose_ref, 1)
        left_layout.addLayout(images_row)

        # Pre-processed checkbox
        preprocess_row = QHBoxLayout()
        preprocess_row.setSpacing(6)
        self._preprocessed_cb = QCheckBox("Pose is pre-processed")
        self._preprocessed_cb.setToolTip(
            "Check if the pose reference is already a skeleton/condition map (skip preprocessing)"
        )
        preprocess_row.addWidget(self._preprocessed_cb)
        preprocess_row.addStretch()
        left_layout.addLayout(preprocess_row)

        # Appearance reference (optional — defaults to source)
        self._appearance_label = QLabel("Appearance Reference (optional — defaults to source):")
        self._appearance_label.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(self._appearance_label)
        self._appearance = ImageDropWidget("Appearance Reference", thumb_height=100)
        self._appearance.image_loaded.connect(lambda p: self._persist_source_path("repose_appearance_path", p))
        self._appearance.image_cleared.connect(lambda: self._persist_source_path("repose_appearance_path", ""))
        left_layout.addWidget(self._appearance)

        # Style reference (optional — for dual IP-Adapter)
        self._style_ref_label = QLabel("Style Reference (optional — dual IP-Adapter for style transfer):")
        self._style_ref_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._style_ref_label.setVisible(False)
        left_layout.addWidget(self._style_ref_label)
        self._style_ref = ImageDropWidget("Style Reference", thumb_height=100)
        self._style_ref.image_loaded.connect(lambda p: self._persist_source_path("repose_style_ref_path", p))
        self._style_ref.image_cleared.connect(lambda: self._persist_source_path("repose_style_ref_path", ""))
        self._style_ref.setVisible(False)
        left_layout.addWidget(self._style_ref)

        # Mask editor for painting over body parts to re-pose
        mask_label = QLabel("Paint mask over body parts to re-pose:")
        mask_label.setStyleSheet("color: #aaa; font-size: 11px;")
        left_layout.addWidget(mask_label)

        self._mask_editor = MaskEditorWidget()
        left_layout.addWidget(self._mask_editor, 1)

        # Keep mask in sync with source image
        self._source.image_loaded.connect(self._on_source_loaded)

        # Magic Mask auto-segmentation
        self._mask_editor.magic_mask_requested.connect(self._on_magic_mask)
        self._auto_seg_worker = None

        # -- Settings --
        settings = QGroupBox("Settings")
        sg_layout = QVBoxLayout(settings)
        sg_layout.setContentsMargins(4, 4, 4, 4)

        # Checkpoint
        ckpt_row = QHBoxLayout()
        ckpt_row.setSpacing(6)
        ckpt_row.addWidget(QLabel("Checkpoint:"))
        self._checkpoint = QComboBox()
        self._checkpoint.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._checkpoint.setMinimumWidth(_W_COMBO)
        self._checkpoint.setMaxVisibleItems(12)
        ckpt_row.addWidget(self._checkpoint, 1)
        ckpt_row.addWidget(QLabel("Sampler:"))
        self._sampler = QComboBox()
        self._sampler.setFixedWidth(_W_COMBO)
        self._sampler.setMaxVisibleItems(12)
        ckpt_row.addWidget(self._sampler)
        ckpt_row.addWidget(QLabel("Scheduler:"))
        self._scheduler = QComboBox()
        self._scheduler.setFixedWidth(100)
        self._scheduler.setMaxVisibleItems(12)
        ckpt_row.addWidget(self._scheduler)
        ckpt_row.addStretch()
        sg_layout.addLayout(ckpt_row)

        # Prompt
        self._enhance = PromptEnhanceWidget(mode="image")
        self._enhance.enhance_requested.connect(self._on_enhance)
        self._enhance_worker = None
        sg_layout.addWidget(self._enhance)
        sg_layout.addWidget(QLabel("Prompt:"))
        self._prompt = QPlainTextEdit()
        self._prompt.setMaximumHeight(50)
        self._prompt.setPlaceholderText("Describe the person appearance...")
        sg_layout.addWidget(self._prompt)

        self._neg_prompt = QPlainTextEdit()
        self._neg_prompt.setMaximumHeight(35)
        self._neg_prompt.setPlaceholderText("Negative prompt...")
        sg_layout.addWidget(self._neg_prompt)

        # Row 1: Steps, CFG, Seed, Generations, Clip Skip
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(1, 150)
        self._steps.setValue(30)
        self._steps.setFixedWidth(_W_SPIN)
        row1.addWidget(self._steps)
        row1.addWidget(QLabel("CFG:"))
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 30.0)
        self._cfg.setDecimals(1)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(7.5)
        self._cfg.setFixedWidth(_W_DSPIN)
        row1.addWidget(self._cfg)
        row1.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2147483647)
        self._seed.setValue(-1)
        self._seed.setFixedWidth(_W_SPIN)
        row1.addWidget(self._seed)
        row1.addWidget(QLabel("Gens:"))
        self._num_generations = QSpinBox()
        self._num_generations.setRange(1, 8)
        self._num_generations.setValue(4)
        self._num_generations.setFixedWidth(_W_SPIN)
        self._num_generations.setToolTip("Number of images to generate")
        row1.addWidget(self._num_generations)
        row1.addWidget(QLabel("Clip Skip:"))
        self._clip_skip = QSpinBox()
        self._clip_skip.setRange(1, 12)
        self._clip_skip.setValue(1)
        self._clip_skip.setFixedWidth(_W_SPIN)
        row1.addWidget(self._clip_skip)
        row1.addStretch()
        sg_layout.addLayout(row1)

        # Row 2: ControlNet type selection
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("ControlNet:"))
        self._cn_type = QComboBox()
        self._cn_type.setFixedWidth(140)
        self._cn_type.setToolTip("Primary ControlNet conditioning type")
        from supremediffusion.models.controlnet_types import list_controlnet_names, list_ip_adapter_variants
        for key, name in list_controlnet_names():
            self._cn_type.addItem(name, key)
        # Default to dwpose if available, else openpose
        idx = self._cn_type.findData("dwpose")
        if idx < 0:
            idx = self._cn_type.findData("openpose")
        if idx >= 0:
            self._cn_type.setCurrentIndex(idx)
        row2.addWidget(self._cn_type)
        row2.addWidget(QLabel("Str:"))
        self._cn_strength = QDoubleSpinBox()
        self._cn_strength.setRange(0.0, 2.0)
        self._cn_strength.setDecimals(2)
        self._cn_strength.setSingleStep(0.05)
        self._cn_strength.setValue(0.7)
        self._cn_strength.setFixedWidth(_W_DSPIN)
        self._cn_strength.setToolTip("ControlNet conditioning scale")
        row2.addWidget(self._cn_strength)
        row2.addWidget(QLabel("2nd CN:"))
        self._cn2_type = QComboBox()
        self._cn2_type.setFixedWidth(140)
        self._cn2_type.setToolTip("Optional second ControlNet (multi-CN)")
        self._cn2_type.addItem("None", "")
        for key, name in list_controlnet_names():
            self._cn2_type.addItem(name, key)
        self._cn2_type.currentIndexChanged.connect(self._on_cn2_changed)
        row2.addWidget(self._cn2_type)
        self._cn2_str_label = QLabel("Str:")
        self._cn2_str_label.setVisible(False)
        row2.addWidget(self._cn2_str_label)
        self._cn2_strength = QDoubleSpinBox()
        self._cn2_strength.setRange(0.0, 2.0)
        self._cn2_strength.setDecimals(2)
        self._cn2_strength.setSingleStep(0.05)
        self._cn2_strength.setValue(0.5)
        self._cn2_strength.setFixedWidth(_W_DSPIN)
        self._cn2_strength.setVisible(False)
        row2.addWidget(self._cn2_strength)
        row2.addStretch()
        sg_layout.addLayout(row2)

        # Row 2b: IP-Adapter variant and scale
        row2b = QHBoxLayout()
        row2b.setSpacing(6)
        row2b.addWidget(QLabel("IP Variant:"))
        self._ip_variant = QComboBox()
        self._ip_variant.setFixedWidth(140)
        self._ip_variant.setToolTip("IP-Adapter weight variant")
        for key, name in list_ip_adapter_variants():
            self._ip_variant.addItem(name, key)
        idx = self._ip_variant.findData("plus")
        if idx >= 0:
            self._ip_variant.setCurrentIndex(idx)
        row2b.addWidget(self._ip_variant)
        row2b.addWidget(QLabel("IP Scale:"))
        self._ip_scale = QDoubleSpinBox()
        self._ip_scale.setRange(0.0, 2.0)
        self._ip_scale.setDecimals(2)
        self._ip_scale.setSingleStep(0.05)
        self._ip_scale.setValue(0.6)
        self._ip_scale.setFixedWidth(_W_DSPIN)
        self._ip_scale.setToolTip("IP-Adapter scale (appearance preservation strength)")
        row2b.addWidget(self._ip_scale)
        row2b.addWidget(QLabel("Style:"))
        self._style_preset = QComboBox()
        self._style_preset.setFixedWidth(120)
        self._style_preset.setToolTip(
            "InstantStyle layer targeting — controls how IP-Adapter influence is applied"
        )
        self._style_preset.addItem("Balanced", "balanced")
        self._style_preset.addItem("Style Only", "style_only")
        self._style_preset.addItem("Layout + Style", "layout_style")
        row2b.addWidget(self._style_preset)
        row2b.addStretch()
        sg_layout.addLayout(row2b)

        # Row 2c: Dual IP-Adapter (IP2) for style transfer
        row2c = QHBoxLayout()
        row2c.setSpacing(6)
        row2c.addWidget(QLabel("2nd IP:"))
        self._ip2_variant = QComboBox()
        self._ip2_variant.setFixedWidth(140)
        self._ip2_variant.setToolTip(
            "Optional second IP-Adapter for style transfer (uses Style Reference image)"
        )
        self._ip2_variant.addItem("None", "")
        # Only CLIP-based variants make sense as a second adapter
        for key, name in list_ip_adapter_variants():
            if not key.startswith("faceid"):
                self._ip2_variant.addItem(name, key)
        self._ip2_variant.currentIndexChanged.connect(self._on_ip2_changed)
        row2c.addWidget(self._ip2_variant)
        self._ip2_str_label = QLabel("Scale:")
        self._ip2_str_label.setVisible(False)
        row2c.addWidget(self._ip2_str_label)
        self._ip2_scale = QDoubleSpinBox()
        self._ip2_scale.setRange(0.0, 2.0)
        self._ip2_scale.setDecimals(2)
        self._ip2_scale.setSingleStep(0.05)
        self._ip2_scale.setValue(0.5)
        self._ip2_scale.setFixedWidth(_W_DSPIN)
        self._ip2_scale.setToolTip("Second IP-Adapter scale (style strength)")
        self._ip2_scale.setVisible(False)
        row2c.addWidget(self._ip2_scale)
        row2c.addStretch()
        sg_layout.addLayout(row2c)

        # Row 3: Denoising, Mask blur, Width, Height
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel("Denoise:"))
        self._denoising = QDoubleSpinBox()
        self._denoising.setRange(0.0, 1.0)
        self._denoising.setDecimals(2)
        self._denoising.setSingleStep(0.05)
        self._denoising.setValue(0.85)
        self._denoising.setFixedWidth(_W_DSPIN)
        row3.addWidget(self._denoising)
        row3.addWidget(QLabel("Mask Blur:"))
        self._mask_blur = QSpinBox()
        self._mask_blur.setRange(0, 64)
        self._mask_blur.setValue(8)
        self._mask_blur.setFixedWidth(_W_SPIN)
        row3.addWidget(self._mask_blur)
        row3.addWidget(QLabel("Width:"))
        self._width = QSpinBox()
        self._width.setRange(0, 2048)
        self._width.setSingleStep(64)
        self._width.setValue(0)
        self._width.setFixedWidth(_W_SPIN)
        self._width.setToolTip("Override width (0 = auto from source)")
        row3.addWidget(self._width)
        row3.addWidget(QLabel("Height:"))
        self._height = QSpinBox()
        self._height.setRange(0, 2048)
        self._height.setSingleStep(64)
        self._height.setValue(0)
        self._height.setFixedWidth(_W_SPIN)
        self._height.setToolTip("Override height (0 = auto from source)")
        row3.addWidget(self._height)
        row3.addStretch()
        sg_layout.addLayout(row3)

        # Row 4: CN guidance start/end
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(QLabel("CN Start:"))
        self._guidance_start = QDoubleSpinBox()
        self._guidance_start.setRange(0.0, 1.0)
        self._guidance_start.setDecimals(2)
        self._guidance_start.setSingleStep(0.05)
        self._guidance_start.setValue(0.0)
        self._guidance_start.setFixedWidth(_W_DSPIN)
        self._guidance_start.setToolTip("Step ratio where ControlNet begins (0.0 = from start)")
        row4.addWidget(self._guidance_start)
        row4.addWidget(QLabel("CN End:"))
        self._guidance_end = QDoubleSpinBox()
        self._guidance_end.setRange(0.0, 1.0)
        self._guidance_end.setDecimals(2)
        self._guidance_end.setSingleStep(0.05)
        self._guidance_end.setValue(1.0)
        self._guidance_end.setFixedWidth(_W_DSPIN)
        self._guidance_end.setToolTip("Step ratio where ControlNet stops (1.0 = until end)")
        row4.addWidget(self._guidance_end)
        row4.addStretch()
        sg_layout.addLayout(row4)

        left_layout.addWidget(settings)
        left_layout.addStretch()
        splitter.addWidget(left)

        # -- Right panel: result gallery + buttons --
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._gallery = ImageGalleryWidget("Result")
        self._clear_results_btn = QPushButton("Clear Results")
        self._clear_results_btn.clicked.connect(self._on_clear_results)
        right_layout.addWidget(self._clear_results_btn)
        right_layout.addWidget(self._gallery, 1)

        btn_row = QHBoxLayout()
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn)
        self._gen_again_btn = QPushButton("Generate Again")
        self._gen_again_btn.setToolTip("Re-run with current settings")
        self._gen_again_btn.setVisible(False)
        self._gen_again_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_again_btn)
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

    # -- Mask persistence --------------------------------------------------

    def _mask_file_path(self) -> Path:
        """Path to the persisted mask file inside the project directory."""
        return self.project_path / "repose_mask.png"

    def _save_mask(self) -> None:
        """Save current mask overlay to the project directory."""
        mask_file = self._mask_file_path()
        if self._mask_editor.save_mask_to_file(str(mask_file)):
            cfg = ProjectConfig.load(self.project_path)
            cfg.repose_mask_path = str(mask_file)
            cfg.save(self.project_path)

    def _restore_mask(self) -> None:
        """Restore mask from the project directory if it exists."""
        cfg = ProjectConfig.load(self.project_path)
        mask_path = cfg.repose_mask_path or ""
        if mask_path and Path(mask_path).is_file():
            self._mask_editor.load_mask_from_file(mask_path)

    def _persist_source_path(self, attr: str, path: str) -> None:
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            setattr(cfg, attr, path)
            cfg.save(self.project_path)
        except Exception:
            pass

    # -- Public API --------------------------------------------------------

    def load_source(self, path: str) -> None:
        """Load the source image (person to re-pose)."""
        self._source.load_image(path)
        self._mask_editor.load_image(path)

    def load_pose_ref(self, path: str) -> None:
        """Load a pose reference image."""
        self._pose_ref.load_image(path)

    # -- Prompt enhancement ------------------------------------------------

    def _on_enhance(self, which: str, _text: str, style: str) -> None:
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        current = self._prompt.toPlainText() if which == "pos" else self._neg_prompt.toPlainText()
        self._enhance.set_enabled_buttons(False)
        worker = PromptEnhanceWorker(current, style, which, self.state, parent=self)
        worker.finished_ok.connect(lambda r: self._on_enhance_done(which, r))
        worker.error.connect(self._on_enhance_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    def _on_enhance_done(self, which: str, result: str) -> None:
        self._enhance.set_enabled_buttons(True)
        (self._prompt if which == "pos" else self._neg_prompt).setPlainText(result)
        self._show_status("Prompt enhanced.")

    def _on_enhance_error(self, msg: str) -> None:
        self._enhance.set_enabled_buttons(True)
        self._show_status(f"Enhance error: {msg}")

    def populate_options(self, **kwargs) -> None:
        """Populate checkpoint, sampler, scheduler dropdowns."""
        if "checkpoints" in kwargs:
            prev = self._checkpoint.currentText()
            self._checkpoint.clear()
            self._checkpoint.addItems(kwargs["checkpoints"])
            if prev:
                idx = self._checkpoint.findText(prev)
                if idx >= 0:
                    self._checkpoint.setCurrentIndex(idx)
        if "samplers" in kwargs:
            prev = self._sampler.currentText()
            self._sampler.clear()
            self._sampler.addItems(kwargs["samplers"])
            if prev:
                idx = self._sampler.findText(prev)
                if idx >= 0:
                    self._sampler.setCurrentIndex(idx)
            elif not prev:
                idx = self._sampler.findText("DPM++ 3M SDE")
                if idx >= 0:
                    self._sampler.setCurrentIndex(idx)
        if "schedulers" in kwargs:
            prev = self._scheduler.currentText()
            self._scheduler.clear()
            self._scheduler.addItems(kwargs["schedulers"])
            if prev:
                idx = self._scheduler.findText(prev)
                if idx >= 0:
                    self._scheduler.setCurrentIndex(idx)
            elif not prev:
                idx = self._scheduler.findText("Karras")
                if idx >= 0:
                    self._scheduler.setCurrentIndex(idx)

    def _on_magic_mask(self) -> None:
        """Run BiRefNet auto-segmentation on the source image."""
        src = self._source.image_path
        if not src:
            self._show_status("Load a source image first.")
            return
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "auto_segment", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        self._show_status("Running auto-segmentation...")
        from sdqt.workers.image import AutoSegmentWorker
        models_root = str(self.state.model_registry.models_root)
        worker = AutoSegmentWorker(src, models_root, parent=self)
        worker.finished_ok.connect(self._on_magic_mask_done)
        worker.error.connect(lambda msg: self._show_status(f"Segmentation error: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._auto_seg_worker = worker
        worker.start()

    def _on_magic_mask_done(self, mask_path: str) -> None:
        self._mask_editor.apply_mask_from_image(mask_path)
        self._show_status("Magic Mask applied — refine with brush if needed.")

    def _on_cn2_changed(self, index: int) -> None:
        has_cn2 = bool(self._cn2_type.currentData())
        self._cn2_str_label.setVisible(has_cn2)
        self._cn2_strength.setVisible(has_cn2)

    def _on_ip2_changed(self, index: int) -> None:
        has_ip2 = bool(self._ip2_variant.currentData())
        self._ip2_str_label.setVisible(has_ip2)
        self._ip2_scale.setVisible(has_ip2)
        self._style_ref_label.setVisible(has_ip2)
        self._style_ref.setVisible(has_ip2)

    # -- Slots -------------------------------------------------------------

    @Slot(str)
    def _on_source_loaded(self, path: str) -> None:
        self._mask_editor.load_image(path)

    @Slot()
    def _on_generate(self) -> None:
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "body_double", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        # Check insightface if FaceID variant selected
        ip_variant = self._ip_variant.currentData() or "plus"
        if ip_variant.startswith("faceid"):
            from sdqt.deps import check_and_install
            if not check_and_install("face_swap", self):
                return

        # Check controlnet_aux if preprocessor needed
        skip_preprocess = self._preprocessed_cb.isChecked()
        if not skip_preprocess:
            cn_type = self._cn_type.currentData() or "openpose"
            from supremediffusion.models.controlnet_types import CONTROLNET_TYPES
            ct = CONTROLNET_TYPES.get(cn_type)
            if ct and ct.preprocessor_class:
                from sdqt.deps import check_and_install
                if not check_and_install("controlnet_aux", self):
                    return

        pipeline = self.state.img_pipeline
        if pipeline is None:
            self._show_status("Loading SD image pipelines...")
            try:
                self.state.load_sd_pipelines()
                pipeline = self.state.img_pipeline
            except Exception as exc:
                self._show_status(f"Failed to load SD pipelines: {exc}")
                return
        if pipeline is None:
            self._show_status("Image pipeline not loaded.")
            return

        src = self._source.image_path
        pose = self._pose_ref.image_path
        if not src:
            self._show_status("Load a source image.")
            return
        if not pose:
            self._show_status("Load a pose reference image.")
            return

        mask_path = self._mask_editor.get_mask_path()
        if not mask_path:
            self._show_status("Paint a mask over the body parts to re-pose.")
            return

        ckpt = self._checkpoint.currentText()
        if not ckpt:
            self._show_status("Select a checkpoint first.")
            return

        # Appearance image: use separate ref if loaded, else source
        appearance = self._appearance.image_path or None

        cfg = ProjectConfig.load(self.project_path)
        cfg.repose_checkpoint = ckpt
        cfg.repose_sampler = self._sampler.currentText()
        cfg.repose_scheduler = self._scheduler.currentText()
        cfg.repose_source_path = src
        cfg.repose_pose_path = pose
        cfg.repose_prompt = self._prompt.toPlainText()
        cfg.repose_negative_prompt = self._neg_prompt.toPlainText()
        cfg.repose_steps = self._steps.value()
        cfg.repose_cfg_scale = self._cfg.value()
        cfg.repose_seed = self._seed.value()
        cfg.repose_controlnet_type = self._cn_type.currentData() or "openpose"
        cfg.repose_controlnet_strength = self._cn_strength.value()
        cfg.repose_controlnet2_type = self._cn2_type.currentData() or ""
        cfg.repose_controlnet2_strength = self._cn2_strength.value()
        cfg.repose_ip_adapter_variant = self._ip_variant.currentData() or "plus"
        cfg.repose_ip_adapter_scale = self._ip_scale.value()
        cfg.repose_denoising_strength = self._denoising.value()
        cfg.repose_mask_blur = self._mask_blur.value()
        cfg.repose_mask_path = str(self._mask_file_path())
        cfg.repose_width = self._width.value()
        cfg.repose_height = self._height.value()
        cfg.repose_guidance_start = self._guidance_start.value()
        cfg.repose_guidance_end = self._guidance_end.value()
        cfg.repose_clip_skip = self._clip_skip.value()
        cfg.repose_preprocessed = skip_preprocess
        cfg.repose_appearance_path = appearance or ""
        cfg.repose_style_preset = self._style_preset.currentData() or "balanced"
        cfg.repose_ip_adapter2_variant = self._ip2_variant.currentData() or ""
        cfg.repose_ip_adapter2_scale = self._ip2_scale.value()
        cfg.repose_style_ref_path = self._style_ref.image_path or ""
        cfg.save(self.project_path)
        self._mask_editor.save_mask_to_file(str(self._mask_file_path()))

        self._gen_btn.setVisible(False)
        self._gen_again_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Generating re-pose...")

        worker = RePoseWorker(
            pipeline, self.project_name, src, mask_path, pose, cfg,
            num_generations=self._num_generations.value(),
            appearance_image=appearance,
            skip_preprocess=skip_preprocess,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, paths: list[str]) -> None:
        self._gen_btn.setVisible(False)
        self._gen_again_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        for p in paths:
            self._persist_result(p)
        persisted = self._list_persisted_results()
        if persisted:
            self._result_path = persisted[-1]
        self._gallery.load_images(persisted)
        self._save_btn.setEnabled(True)
        self._show_status("Re-pose complete.")

    def _on_error(self, msg: str) -> None:
        if self._result_path:
            self._gen_again_btn.setVisible(True)
        else:
            self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
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
        d = self.project_path / "images" / "repose"
        d.mkdir(parents=True, exist_ok=True)
        return d

    _persist_counter = 0

    def _persist_result(self, path: str) -> str | None:
        RePoseTab._persist_counter += 1
        name = datetime.utcnow().strftime(
            f"repose_%Y%m%d_%H%M%S_{RePoseTab._persist_counter:02d}.png"
        )
        dest = self._results_dir() / name
        shutil.copy2(path, dest)
        return str(dest)

    def _list_persisted_results(self) -> list[str]:
        d = self.project_path / "images" / "repose"
        if not d.is_dir():
            return []
        return sorted(str(p) for p in d.glob("*.png"))

    @Slot()
    def _on_clear_results(self) -> None:
        d = self.project_path / "images" / "repose"
        if d.is_dir():
            shutil.rmtree(d)
        self._gallery.clear()
        self._result_path = None
        self._save_btn.setEnabled(False)
        self._gen_btn.setVisible(True)
        self._gen_again_btn.setVisible(False)
        self._show_status("Results cleared.")

    @Slot()
    def _on_save(self) -> None:
        if self._result_path:
            saved = self.save_frame_to_project(self._result_path)
            if saved:
                self._show_status("Saved to project.")

    def on_project_changed(self, project_name: str) -> None:
        if self._mask_editor.image_path:
            self._save_mask()
        super().on_project_changed(project_name)
        self._gallery.clear()
        self._mask_editor.clear_mask()
        self._result_path = None
        self._gen_btn.setVisible(True)
        self._gen_again_btn.setVisible(False)
        self._save_btn.setEnabled(False)
        self._restoring = True

        # Restore persisted results
        persisted = self._list_persisted_results()
        if persisted:
            self._result_path = persisted[-1]
            self._gallery.load_images(persisted)
            self._gen_btn.setVisible(False)
            self._gen_again_btn.setVisible(True)
            self._save_btn.setEnabled(True)
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        src = cfg.repose_source_path or ""
        pose = cfg.repose_pose_path or ""
        if src and Path(src).is_file():
            self._source.load_image(src)
            self._mask_editor.load_image(src)
            self._restore_mask()
        else:
            self._source.clear_image()
        if pose and Path(pose).is_file():
            self._pose_ref.load_image(pose)
        else:
            self._pose_ref.clear_image()
        appearance = getattr(cfg, "repose_appearance_path", "")
        if appearance and Path(appearance).is_file():
            self._appearance.load_image(appearance)
        else:
            self._appearance.clear_image()
        ckpt = cfg.repose_checkpoint or ""
        if ckpt:
            idx = self._checkpoint.findText(ckpt)
            if idx >= 0:
                self._checkpoint.setCurrentIndex(idx)
        sampler = cfg.repose_sampler or ""
        if sampler:
            idx = self._sampler.findText(sampler)
            if idx >= 0:
                self._sampler.setCurrentIndex(idx)
        scheduler = cfg.repose_scheduler or ""
        if scheduler:
            idx = self._scheduler.findText(scheduler)
            if idx >= 0:
                self._scheduler.setCurrentIndex(idx)
        self._prompt.setPlainText(cfg.repose_prompt or "")
        self._neg_prompt.setPlainText(cfg.repose_negative_prompt or "")
        self._steps.setValue(cfg.repose_steps)
        self._cfg.setValue(cfg.repose_cfg_scale)
        self._seed.setValue(cfg.repose_seed)
        cn_type = getattr(cfg, "repose_controlnet_type", "openpose") or "openpose"
        idx = self._cn_type.findData(cn_type)
        if idx >= 0:
            self._cn_type.setCurrentIndex(idx)
        self._cn_strength.setValue(cfg.repose_controlnet_strength)
        cn2_type = getattr(cfg, "repose_controlnet2_type", "") or ""
        idx = self._cn2_type.findData(cn2_type)
        if idx >= 0:
            self._cn2_type.setCurrentIndex(idx)
        self._cn2_strength.setValue(getattr(cfg, "repose_controlnet2_strength", 0.5))
        self._on_cn2_changed(0)
        ip_var = getattr(cfg, "repose_ip_adapter_variant", "plus") or "plus"
        idx = self._ip_variant.findData(ip_var)
        if idx >= 0:
            self._ip_variant.setCurrentIndex(idx)
        self._ip_scale.setValue(cfg.repose_ip_adapter_scale)
        self._denoising.setValue(cfg.repose_denoising_strength)
        self._mask_blur.setValue(cfg.repose_mask_blur)
        self._width.setValue(getattr(cfg, "repose_width", 0))
        self._height.setValue(getattr(cfg, "repose_height", 0))
        self._guidance_start.setValue(getattr(cfg, "repose_guidance_start", 0.0))
        self._guidance_end.setValue(getattr(cfg, "repose_guidance_end", 1.0))
        self._clip_skip.setValue(getattr(cfg, "repose_clip_skip", 1))
        self._preprocessed_cb.setChecked(getattr(cfg, "repose_preprocessed", False))
        style = getattr(cfg, "repose_style_preset", "balanced") or "balanced"
        idx = self._style_preset.findData(style)
        if idx >= 0:
            self._style_preset.setCurrentIndex(idx)
        ip2_var = getattr(cfg, "repose_ip_adapter2_variant", "") or ""
        idx = self._ip2_variant.findData(ip2_var)
        if idx >= 0:
            self._ip2_variant.setCurrentIndex(idx)
        self._ip2_scale.setValue(getattr(cfg, "repose_ip_adapter2_scale", 0.5))
        self._on_ip2_changed(0)
        style_ref = getattr(cfg, "repose_style_ref_path", "")
        if style_ref and Path(style_ref).is_file():
            self._style_ref.load_image(style_ref)
        else:
            self._style_ref.clear_image()
        self._restoring = False
