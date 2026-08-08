"""ControlNet sub-tab for Image Suite.

General-purpose ControlNet txt2img, img2img, and inpaint generation.
Supports single or dual ControlNet conditioning, guidance scheduling,
guess mode, clip skip, and preprocessor bypass.
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
from sdqt.widgets.prompt_enhance import PromptEnhanceWidget
from sdqt.workers.image import ControlNetWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

logger = logging.getLogger(__name__)

_W_SPIN = 75
_W_DSPIN = 80
_W_COMBO = 160


class ControlNetTab(BaseTab):
    """ControlNet tab — general-purpose ControlNet-conditioned generation."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: ControlNetWorker | None = None
        self._result_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left panel: inputs + settings --
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # Mode toggle
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Mode:"))
        self._mode = QComboBox()
        self._mode.addItem("txt2img", "txt2img")
        self._mode.addItem("img2img", "img2img")
        self._mode.addItem("inpaint", "inpaint")
        self._mode.setFixedWidth(100)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode)
        mode_row.addStretch()
        left_layout.addLayout(mode_row)

        # Source image (img2img / inpaint)
        self._source_label = QLabel("Source Image (img2img / inpaint):")
        self._source_label.setStyleSheet("color: #aaa; font-size: 13px;")
        self._source_label.setVisible(False)
        left_layout.addWidget(self._source_label)
        self._source = ImageDropWidget("Source Image", thumb_height=140)
        self._source.image_loaded.connect(lambda p: self._persist_path("controlnet_source_path", p))
        self._source.image_cleared.connect(lambda: self._persist_path("controlnet_source_path", ""))
        self._source.setVisible(False)
        left_layout.addWidget(self._source)

        # Mask image (inpaint only)
        self._mask_label = QLabel("Mask Image (white = inpaint area):")
        self._mask_label.setStyleSheet("color: #aaa; font-size: 13px;")
        self._mask_label.setVisible(False)
        left_layout.addWidget(self._mask_label)
        self._mask = ImageDropWidget("Mask Image", thumb_height=100)
        self._mask.image_loaded.connect(lambda p: self._persist_path("controlnet_mask_path", p))
        self._mask.image_cleared.connect(lambda: self._persist_path("controlnet_mask_path", ""))
        self._mask.setVisible(False)
        left_layout.addWidget(self._mask)

        # Condition image 1
        cond_label = QLabel("Condition Image (raw photo or pre-made condition map):")
        cond_label.setStyleSheet("color: #aaa; font-size: 13px;")
        left_layout.addWidget(cond_label)
        self._condition = ImageDropWidget("Condition Image", thumb_height=160)
        self._condition.image_loaded.connect(lambda p: self._persist_path("controlnet_condition_path", p))
        self._condition.image_cleared.connect(lambda: self._persist_path("controlnet_condition_path", ""))
        left_layout.addWidget(self._condition)

        # Preprocess row
        preprocess_row = QHBoxLayout()
        preprocess_row.setSpacing(6)
        self._preprocess_btn = QPushButton("Preprocess")
        self._preprocess_btn.setToolTip("Run the selected preprocessor on the condition image to preview the condition map")
        self._preprocess_btn.clicked.connect(self._on_preprocess)
        preprocess_row.addWidget(self._preprocess_btn)
        self._preprocessed_cb = QCheckBox("Pre-processed")
        self._preprocessed_cb.setToolTip("Check if the condition image is already a condition map (skip preprocessing during generation)")
        preprocess_row.addWidget(self._preprocessed_cb)
        self._preprocess_status = QLabel("")
        self._preprocess_status.setStyleSheet("color: #aaa; font-size: 13px;")
        preprocess_row.addWidget(self._preprocess_status, 1)
        left_layout.addLayout(preprocess_row)

        # Preprocessor params — per-type controls shown/hidden via _on_cn_type_changed
        pp_row = QHBoxLayout()
        pp_row.setSpacing(6)
        # Canny thresholds
        self._pp_canny_low_label = QLabel("Canny Low:")
        pp_row.addWidget(self._pp_canny_low_label)
        self._pp_canny_low = QSpinBox()
        self._pp_canny_low.setRange(0, 500)
        self._pp_canny_low.setValue(100)
        self._pp_canny_low.setFixedWidth(_W_SPIN)
        self._pp_canny_low.setToolTip("Canny lower hysteresis threshold")
        pp_row.addWidget(self._pp_canny_low)
        self._pp_canny_high_label = QLabel("High:")
        pp_row.addWidget(self._pp_canny_high_label)
        self._pp_canny_high = QSpinBox()
        self._pp_canny_high.setRange(0, 500)
        self._pp_canny_high.setValue(200)
        self._pp_canny_high.setFixedWidth(_W_SPIN)
        self._pp_canny_high.setToolTip("Canny upper hysteresis threshold")
        pp_row.addWidget(self._pp_canny_high)
        # OpenPose / DWPose hand & face
        self._pp_include_hand = QCheckBox("Hand")
        self._pp_include_hand.setChecked(True)
        self._pp_include_hand.setToolTip("OpenPose: detect hand keypoints")
        pp_row.addWidget(self._pp_include_hand)
        self._pp_include_face = QCheckBox("Face")
        self._pp_include_face.setChecked(True)
        self._pp_include_face.setToolTip("OpenPose: detect face keypoints")
        pp_row.addWidget(self._pp_include_face)
        pp_row.addStretch()
        left_layout.addLayout(pp_row)

        # Detect / image resolution (applies to all preprocessors)
        pp_row2 = QHBoxLayout()
        pp_row2.setSpacing(6)
        pp_row2.addWidget(QLabel("Detect Res:"))
        self._pp_detect_res = QSpinBox()
        self._pp_detect_res.setRange(64, 2048)
        self._pp_detect_res.setSingleStep(64)
        self._pp_detect_res.setValue(512)
        self._pp_detect_res.setFixedWidth(_W_SPIN)
        self._pp_detect_res.setToolTip("Resolution the preprocessor detects at")
        pp_row2.addWidget(self._pp_detect_res)
        pp_row2.addWidget(QLabel("Image Res:"))
        self._pp_image_res = QSpinBox()
        self._pp_image_res.setRange(64, 2048)
        self._pp_image_res.setSingleStep(64)
        self._pp_image_res.setValue(512)
        self._pp_image_res.setFixedWidth(_W_SPIN)
        self._pp_image_res.setToolTip("Resolution of the output condition map")
        pp_row2.addWidget(self._pp_image_res)
        pp_row2.addStretch()
        left_layout.addLayout(pp_row2)

        # ControlNet type selection
        cn_row = QHBoxLayout()
        cn_row.setSpacing(6)
        cn_row.addWidget(QLabel("ControlNet:"))
        self._cn_type = QComboBox()
        self._cn_type.setFixedWidth(140)
        from supremediffusion.models.controlnet_types import list_controlnet_names
        for key, name in list_controlnet_names():
            self._cn_type.addItem(name, key)
        idx = self._cn_type.findData("canny")
        if idx >= 0:
            self._cn_type.setCurrentIndex(idx)
        self._cn_type.currentIndexChanged.connect(self._on_cn_type_changed)
        cn_row.addWidget(self._cn_type)
        cn_row.addWidget(QLabel("Str:"))
        self._cn_strength = QDoubleSpinBox()
        self._cn_strength.setRange(0.0, 2.0)
        self._cn_strength.setDecimals(2)
        self._cn_strength.setSingleStep(0.05)
        self._cn_strength.setValue(0.8)
        self._cn_strength.setFixedWidth(_W_DSPIN)
        cn_row.addWidget(self._cn_strength)
        cn_row.addStretch()
        left_layout.addLayout(cn_row)

        # 2nd ControlNet
        cn2_row = QHBoxLayout()
        cn2_row.setSpacing(6)
        cn2_row.addWidget(QLabel("2nd CN:"))
        self._cn2_type = QComboBox()
        self._cn2_type.setFixedWidth(140)
        self._cn2_type.addItem("None", "")
        for key, name in list_controlnet_names():
            self._cn2_type.addItem(name, key)
        self._cn2_type.currentIndexChanged.connect(self._on_cn2_changed)
        cn2_row.addWidget(self._cn2_type)
        self._cn2_str_label = QLabel("Str:")
        self._cn2_str_label.setVisible(False)
        cn2_row.addWidget(self._cn2_str_label)
        self._cn2_strength = QDoubleSpinBox()
        self._cn2_strength.setRange(0.0, 2.0)
        self._cn2_strength.setDecimals(2)
        self._cn2_strength.setSingleStep(0.05)
        self._cn2_strength.setValue(0.5)
        self._cn2_strength.setFixedWidth(_W_DSPIN)
        self._cn2_strength.setVisible(False)
        cn2_row.addWidget(self._cn2_strength)
        cn2_row.addStretch()
        left_layout.addLayout(cn2_row)

        # Condition image 2 (for 2nd ControlNet)
        self._condition2_label = QLabel("2nd Condition Image:")
        self._condition2_label.setStyleSheet("color: #aaa; font-size: 13px;")
        self._condition2_label.setVisible(False)
        left_layout.addWidget(self._condition2_label)
        self._condition2 = ImageDropWidget("2nd Condition Image", thumb_height=100)
        self._condition2.image_loaded.connect(lambda p: self._persist_path("controlnet_condition2_path", p))
        self._condition2.image_cleared.connect(lambda: self._persist_path("controlnet_condition2_path", ""))
        self._condition2.setVisible(False)
        left_layout.addWidget(self._condition2)

        # Prompt
        self._enhance = PromptEnhanceWidget(mode="image")
        self._enhance.enhance_requested.connect(self._on_enhance)
        self._enhance_worker = None
        left_layout.addWidget(self._enhance)
        left_layout.addWidget(QLabel("Prompt:"))
        self._prompt = QPlainTextEdit()
        self._prompt.setMaximumHeight(85)
        self._prompt.setPlaceholderText("Describe the desired image...")
        left_layout.addWidget(self._prompt)

        self._neg_prompt = QPlainTextEdit()
        self._neg_prompt.setMaximumHeight(55)
        self._neg_prompt.setPlaceholderText("Negative prompt...")
        left_layout.addWidget(self._neg_prompt)

        # Settings
        settings = QGroupBox("Settings")
        sg_layout = QVBoxLayout(settings)
        sg_layout.setContentsMargins(8, 6, 8, 6)
        sg_layout.setSpacing(8)

        # Checkpoint (own row — let it use full width)
        ckpt_row = QHBoxLayout()
        ckpt_row.setSpacing(8)
        ckpt_row.addWidget(QLabel("Checkpoint:"))
        self._checkpoint = QComboBox()
        self._checkpoint.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._checkpoint.setMinimumWidth(_W_COMBO)
        self._checkpoint.setMaxVisibleItems(12)
        ckpt_row.addWidget(self._checkpoint, 1)
        sg_layout.addLayout(ckpt_row)

        # Sampler / Scheduler (split off the checkpoint row)
        sampler_row = QHBoxLayout()
        sampler_row.setSpacing(8)
        sampler_row.addWidget(QLabel("Sampler:"))
        self._sampler = QComboBox()
        self._sampler.setMinimumWidth(_W_COMBO)
        self._sampler.setMaxVisibleItems(12)
        sampler_row.addWidget(self._sampler)
        sampler_row.addWidget(QLabel("Scheduler:"))
        self._scheduler = QComboBox()
        self._scheduler.setMinimumWidth(120)
        self._scheduler.setMaxVisibleItems(12)
        sampler_row.addWidget(self._scheduler)
        sampler_row.addStretch()
        sg_layout.addLayout(sampler_row)

        # Steps / CFG / Seed (split off Batch + Clip Skip below)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(1, 150)
        self._steps.setValue(20)
        self._steps.setMinimumWidth(90)
        row1.addWidget(self._steps)
        row1.addWidget(QLabel("CFG:"))
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 30.0)
        self._cfg.setDecimals(1)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(7.0)
        self._cfg.setMinimumWidth(95)
        row1.addWidget(self._cfg)
        row1.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2147483647)
        self._seed.setValue(-1)
        self._seed.setMinimumWidth(90)
        row1.addWidget(self._seed)
        row1.addStretch()
        sg_layout.addLayout(row1)

        # Batch / Clip Skip
        row1b = QHBoxLayout()
        row1b.setSpacing(8)
        row1b.addWidget(QLabel("Batch:"))
        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 8)
        self._batch_count.setValue(1)
        self._batch_count.setMinimumWidth(90)
        row1b.addWidget(self._batch_count)
        row1b.addWidget(QLabel("Clip Skip:"))
        self._clip_skip = QSpinBox()
        self._clip_skip.setRange(1, 12)
        self._clip_skip.setValue(1)
        self._clip_skip.setMinimumWidth(90)
        row1b.addWidget(self._clip_skip)
        row1b.addStretch()
        sg_layout.addLayout(row1b)

        # Width / Height / Denoising (img2img/inpaint only)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Width:"))
        self._width = QSpinBox()
        self._width.setRange(64, 2048)
        self._width.setSingleStep(64)
        self._width.setValue(512)
        self._width.setFixedWidth(_W_SPIN)
        row2.addWidget(self._width)
        row2.addWidget(QLabel("Height:"))
        self._height = QSpinBox()
        self._height.setRange(64, 2048)
        self._height.setSingleStep(64)
        self._height.setValue(512)
        self._height.setFixedWidth(_W_SPIN)
        row2.addWidget(self._height)
        self._denoise_label = QLabel("Denoise:")
        self._denoise_label.setVisible(False)
        row2.addWidget(self._denoise_label)
        self._denoising = QDoubleSpinBox()
        self._denoising.setRange(0.0, 1.0)
        self._denoising.setDecimals(2)
        self._denoising.setSingleStep(0.05)
        self._denoising.setValue(0.75)
        self._denoising.setFixedWidth(_W_DSPIN)
        self._denoising.setVisible(False)
        row2.addWidget(self._denoising)
        row2.addStretch()
        sg_layout.addLayout(row2)

        # Guidance scheduling + Guess mode
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel("CN Start:"))
        self._guidance_start = QDoubleSpinBox()
        self._guidance_start.setRange(0.0, 1.0)
        self._guidance_start.setDecimals(2)
        self._guidance_start.setSingleStep(0.05)
        self._guidance_start.setValue(0.0)
        self._guidance_start.setFixedWidth(_W_DSPIN)
        self._guidance_start.setToolTip("control_guidance_start: step ratio where ControlNet begins influencing (0.0 = from start)")
        row3.addWidget(self._guidance_start)
        row3.addWidget(QLabel("CN End:"))
        self._guidance_end = QDoubleSpinBox()
        self._guidance_end.setRange(0.0, 1.0)
        self._guidance_end.setDecimals(2)
        self._guidance_end.setSingleStep(0.05)
        self._guidance_end.setValue(1.0)
        self._guidance_end.setFixedWidth(_W_DSPIN)
        self._guidance_end.setToolTip("control_guidance_end: step ratio where ControlNet stops influencing (1.0 = until end)")
        row3.addWidget(self._guidance_end)
        self._guess_mode = QCheckBox("Guess Mode")
        self._guess_mode.setToolTip("Generate from control image without prompt guidance (ControlNet residuals scaled by block depth)")
        row3.addWidget(self._guess_mode)
        row3.addStretch()
        sg_layout.addLayout(row3)

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
        self._gen_btn.setObjectName("primary")
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

        # Set initial preprocessor-param visibility for the default type.
        self._on_cn_type_changed(0)

    # -- Mode toggle -------------------------------------------------------

    def _on_mode_changed(self, index: int) -> None:
        mode = self._mode.currentData()
        needs_source = mode in ("img2img", "inpaint")
        needs_mask = mode == "inpaint"
        self._source_label.setVisible(needs_source)
        self._source.setVisible(needs_source)
        self._mask_label.setVisible(needs_mask)
        self._mask.setVisible(needs_mask)
        self._denoise_label.setVisible(needs_source)
        self._denoising.setVisible(needs_source)

    def _on_cn2_changed(self, index: int) -> None:
        has_cn2 = bool(self._cn2_type.currentData())
        self._cn2_str_label.setVisible(has_cn2)
        self._cn2_strength.setVisible(has_cn2)
        self._condition2_label.setVisible(has_cn2)
        self._condition2.setVisible(has_cn2)

    def _on_cn_type_changed(self, index: int) -> None:
        """Show only the preprocessor params relevant to the selected type."""
        cn_type = self._cn_type.currentData() or "canny"
        is_canny = cn_type == "canny"
        is_pose = cn_type in ("openpose", "dwpose")
        self._pp_canny_low_label.setVisible(is_canny)
        self._pp_canny_low.setVisible(is_canny)
        self._pp_canny_high_label.setVisible(is_canny)
        self._pp_canny_high.setVisible(is_canny)
        self._pp_include_hand.setVisible(is_pose)
        self._pp_include_face.setVisible(is_pose)

    def _preprocessor_overrides(self, cn_type: str) -> dict:
        """Build call-time preprocessor overrides from the current UI controls."""
        overrides: dict = {
            "detect_resolution": self._pp_detect_res.value(),
            "image_resolution": self._pp_image_res.value(),
        }
        if cn_type == "canny":
            overrides["low_threshold"] = self._pp_canny_low.value()
            overrides["high_threshold"] = self._pp_canny_high.value()
        elif cn_type in ("openpose", "dwpose"):
            overrides["include_hand"] = self._pp_include_hand.isChecked()
            overrides["include_face"] = self._pp_include_face.isChecked()
        return overrides

    # -- Preprocess --------------------------------------------------------

    def _on_preprocess(self) -> None:
        cond = self._condition.image_path
        if not cond:
            self._preprocess_status.setText("Load a condition image first.")
            return

        from sdqt.deps import check_and_install
        if not check_and_install("controlnet_aux", self):
            return

        cn_type = self._cn_type.currentData() or "canny"
        self._preprocess_status.setText(f"Running {cn_type} preprocessor...")
        try:
            from supremediffusion.models.controlnet_types import run_preprocessor
            result = run_preprocessor(cn_type, cond, **self._preprocessor_overrides(cn_type))
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix=f"cn_{cn_type}_", delete=False)
            result.save(tmp.name)
            tmp.close()
            self._condition.load_image(tmp.name)
            self._preprocessed_cb.setChecked(True)
            self._preprocess_status.setText(f"{cn_type} preprocessing complete.")
        except Exception as exc:
            self._preprocess_status.setText(f"Preprocess error: {exc}")

    # -- Persistence -------------------------------------------------------

    def _persist_path(self, attr: str, path: str) -> None:
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            setattr(cfg, attr, path)
            cfg.save(self.project_path)
        except Exception:
            pass

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
                idx = self._sampler.findText("DPM++ 2M")
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

    # -- Public API --------------------------------------------------------

    def load_condition(self, path: str) -> None:
        """Load a condition image."""
        self._condition.load_image(path)

    def load_source(self, path: str) -> None:
        """Load a source image (switches to img2img mode)."""
        self._mode.setCurrentIndex(1)  # img2img
        self._source.load_image(path)

    # -- Generation --------------------------------------------------------

    @Slot()
    def _on_generate(self) -> None:
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

        cond = self._condition.image_path
        if not cond:
            self._show_status("Load a condition image.")
            return

        ckpt = self._checkpoint.currentText()
        if not ckpt:
            self._show_status("Select a checkpoint first.")
            return

        mode = self._mode.currentData()
        src = None
        mask = None
        if mode == "img2img":
            src = self._source.image_path
            if not src:
                self._show_status("Load a source image for img2img mode.")
                return
        elif mode == "inpaint":
            src = self._source.image_path
            mask = self._mask.image_path
            if not src:
                self._show_status("Load a source image for inpaint mode.")
                return
            if not mask:
                self._show_status("Load a mask image for inpaint mode.")
                return

        # Check if preprocessor deps are needed
        skip_preprocess = self._preprocessed_cb.isChecked()
        if not skip_preprocess:
            cn_type = self._cn_type.currentData() or "canny"
            from supremediffusion.models.controlnet_types import CONTROLNET_TYPES
            ct = CONTROLNET_TYPES.get(cn_type)
            if ct and ct.preprocessor_class:
                from sdqt.deps import check_and_install
                if not check_and_install("controlnet_aux", self):
                    return

        # Get 2nd condition image path
        cond2 = None
        cn2_type = self._cn2_type.currentData() or ""
        if cn2_type:
            cond2 = self._condition2.image_path
            # Fall back to 1st condition if no separate image loaded
            if not cond2:
                cond2 = None  # Worker will use condition1

        cfg = ProjectConfig.load(self.project_path)
        cfg.controlnet_mode = mode
        cfg.controlnet_checkpoint = ckpt
        cfg.controlnet_sampler = self._sampler.currentText()
        cfg.controlnet_scheduler = self._scheduler.currentText()
        cfg.controlnet_prompt = self._prompt.toPlainText()
        cfg.controlnet_negative_prompt = self._neg_prompt.toPlainText()
        cfg.controlnet_steps = self._steps.value()
        cfg.controlnet_cfg_scale = self._cfg.value()
        cfg.controlnet_seed = self._seed.value()
        cfg.controlnet_width = self._width.value()
        cfg.controlnet_height = self._height.value()
        cfg.controlnet_denoising_strength = self._denoising.value()
        cfg.controlnet_type = self._cn_type.currentData() or "canny"
        cfg.controlnet_type2 = cn2_type
        cfg.controlnet_strength = self._cn_strength.value()
        cfg.controlnet_strength2 = self._cn2_strength.value()
        cfg.controlnet_condition_path = cond
        cfg.controlnet_condition2_path = cond2 or ""
        cfg.controlnet_source_path = src or ""
        cfg.controlnet_mask_path = mask or ""
        cfg.controlnet_batch_count = self._batch_count.value()
        cfg.controlnet_guidance_start = self._guidance_start.value()
        cfg.controlnet_guidance_end = self._guidance_end.value()
        cfg.controlnet_guess_mode = self._guess_mode.isChecked()
        cfg.controlnet_clip_skip = self._clip_skip.value()
        cfg.controlnet_preprocessed = skip_preprocess
        # Preprocessor params (shared cn_* fields — also read by the worker)
        cfg.cn_canny_low = self._pp_canny_low.value()
        cfg.cn_canny_high = self._pp_canny_high.value()
        cfg.cn_detect_resolution = self._pp_detect_res.value()
        cfg.cn_image_resolution = self._pp_image_res.value()
        cfg.cn_openpose_include_hand = self._pp_include_hand.isChecked()
        cfg.cn_openpose_include_face = self._pp_include_face.isChecked()
        cfg.save(self.project_path)

        # All inputs validated — claim the GPU lock (released in done/error).
        if not self.acquire_gpu("Image"):
            return

        self._gen_btn.setVisible(False)
        self._gen_again_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Generating with ControlNet...")

        worker = ControlNetWorker(
            pipeline, self.project_name, cond, cond2, src, mask, cfg,
            num_generations=self._batch_count.value(),
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
        self.release_gpu("Image")
        self._abort_btn.setVisible(False)
        if not paths:
            # Nothing was generated — keep the primary Generate button visible
            # rather than implying a successful result exists.
            self._gen_btn.setVisible(True)
            self._gen_again_btn.setVisible(False)
            self._show_status("ControlNet produced no images.")
            return
        self._gen_btn.setVisible(False)
        self._gen_again_btn.setVisible(True)
        for p in paths:
            self._persist_result(p)
        persisted = self._list_persisted_results()
        if persisted:
            self._result_path = persisted[-1]
        self._gallery.load_images(persisted)
        self._save_btn.setEnabled(True)
        self._show_status("ControlNet generation complete.")

    def _on_error(self, msg: str) -> None:
        self.release_gpu("Image")
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

    # -- Results -----------------------------------------------------------

    def _results_dir(self) -> Path:
        d = self.project_path / "images" / "controlnet"
        d.mkdir(parents=True, exist_ok=True)
        return d

    _persist_counter = 0

    def _persist_result(self, path: str) -> str | None:
        ControlNetTab._persist_counter += 1
        name = datetime.utcnow().strftime(
            f"controlnet_%Y%m%d_%H%M%S_{ControlNetTab._persist_counter:02d}.png"
        )
        dest = self._results_dir() / name
        shutil.copy2(path, dest)
        return str(dest)

    def _list_persisted_results(self) -> list[str]:
        d = self.project_path / "images" / "controlnet"
        if not d.is_dir():
            return []
        return sorted(str(p) for p in d.glob("*.png"))

    @Slot()
    def _on_clear_results(self) -> None:
        d = self.project_path / "images" / "controlnet"
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

    # -- Project change ----------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._gallery.clear()
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

        # Restore mode
        mode = getattr(cfg, "controlnet_mode", "txt2img")
        idx = self._mode.findData(mode)
        if idx >= 0:
            self._mode.setCurrentIndex(idx)

        # Restore images
        cond = getattr(cfg, "controlnet_condition_path", "")
        if cond and Path(cond).is_file():
            self._condition.load_image(cond)
        else:
            self._condition.clear_image()
        cond2 = getattr(cfg, "controlnet_condition2_path", "")
        if cond2 and Path(cond2).is_file():
            self._condition2.load_image(cond2)
        else:
            self._condition2.clear_image()
        src = getattr(cfg, "controlnet_source_path", "")
        if src and Path(src).is_file():
            self._source.load_image(src)
        else:
            self._source.clear_image()
        mask = getattr(cfg, "controlnet_mask_path", "")
        if mask and Path(mask).is_file():
            self._mask.load_image(mask)
        else:
            self._mask.clear_image()

        # Restore ControlNet type
        cn_type = getattr(cfg, "controlnet_type", "canny") or "canny"
        idx = self._cn_type.findData(cn_type)
        if idx >= 0:
            self._cn_type.setCurrentIndex(idx)
        self._cn_strength.setValue(getattr(cfg, "controlnet_strength", 0.8))
        cn2_type = getattr(cfg, "controlnet_type2", "") or ""
        idx = self._cn2_type.findData(cn2_type)
        if idx >= 0:
            self._cn2_type.setCurrentIndex(idx)
        self._cn2_strength.setValue(getattr(cfg, "controlnet_strength2", 0.5))
        self._on_cn2_changed(0)

        # Restore checkpoint/sampler/scheduler
        ckpt = getattr(cfg, "controlnet_checkpoint", "")
        if ckpt:
            idx = self._checkpoint.findText(ckpt)
            if idx >= 0:
                self._checkpoint.setCurrentIndex(idx)
        sampler = getattr(cfg, "controlnet_sampler", "")
        if sampler:
            idx = self._sampler.findText(sampler)
            if idx >= 0:
                self._sampler.setCurrentIndex(idx)
        scheduler = getattr(cfg, "controlnet_scheduler", "")
        if scheduler:
            idx = self._scheduler.findText(scheduler)
            if idx >= 0:
                self._scheduler.setCurrentIndex(idx)

        # Restore generation params
        self._prompt.setPlainText(getattr(cfg, "controlnet_prompt", ""))
        self._neg_prompt.setPlainText(getattr(cfg, "controlnet_negative_prompt", ""))
        self._steps.setValue(getattr(cfg, "controlnet_steps", 20))
        self._cfg.setValue(getattr(cfg, "controlnet_cfg_scale", 7.0))
        self._seed.setValue(getattr(cfg, "controlnet_seed", -1))
        self._width.setValue(getattr(cfg, "controlnet_width", 512))
        self._height.setValue(getattr(cfg, "controlnet_height", 512))
        self._denoising.setValue(getattr(cfg, "controlnet_denoising_strength", 0.75))
        self._batch_count.setValue(getattr(cfg, "controlnet_batch_count", 1))
        self._guidance_start.setValue(getattr(cfg, "controlnet_guidance_start", 0.0))
        self._guidance_end.setValue(getattr(cfg, "controlnet_guidance_end", 1.0))
        self._guess_mode.setChecked(getattr(cfg, "controlnet_guess_mode", False))
        self._clip_skip.setValue(getattr(cfg, "controlnet_clip_skip", 1))
        self._preprocessed_cb.setChecked(getattr(cfg, "controlnet_preprocessed", False))

        # Restore preprocessor params (shared cn_* fields)
        self._pp_canny_low.setValue(getattr(cfg, "cn_canny_low", 100))
        self._pp_canny_high.setValue(getattr(cfg, "cn_canny_high", 200))
        self._pp_detect_res.setValue(getattr(cfg, "cn_detect_resolution", 512))
        self._pp_image_res.setValue(getattr(cfg, "cn_image_resolution", 512))
        self._pp_include_hand.setChecked(getattr(cfg, "cn_openpose_include_hand", True))
        self._pp_include_face.setChecked(getattr(cfg, "cn_openpose_include_face", True))
        self._on_cn_type_changed(0)

        self._restoring = False
