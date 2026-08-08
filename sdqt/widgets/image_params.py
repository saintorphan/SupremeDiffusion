"""Shared SD image generation parameters widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.lora_metadata import (
    LoRAConfigDialog,
    lora_prompt_tag,
    populate_lora_list_grouped,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.collapsible_section import CollapsibleSection

# SDXL-trained resolution presets (label, width, height)
_DIM_PRESETS: list[tuple[str, int, int]] = [
    ("1024\u00d71024 (1:1)", 1024, 1024),
    ("1152\u00d7896 (9:7)", 1152, 896),
    ("896\u00d71152 (7:9)", 896, 1152),
    ("1216\u00d7832 (3:2)", 1216, 832),
    ("832\u00d71216 (2:3)", 832, 1216),
    ("1344\u00d7768 (16:9)", 1344, 768),
    ("768\u00d71344 (9:16)", 768, 1344),
    ("1536\u00d7640 (21:9)", 1536, 640),
    ("640\u00d71536 (9:21)", 640, 1536),
    ("512\u00d7512 (1:1)", 512, 512),
    ("768\u00d7768 (1:1)", 768, 768),
]


class ImageParamsWidget(QWidget):
    """Reusable widget for SD image generation parameters.

    Shared between txt2img, img2img, inpainter, and img_edit tabs.
    Supports multiple model families via ``set_family(strategy)``.

    Signals:
        lora_toggled(str, bool): Emitted when a LoRA is checked/unchecked.
            str = prompt tag text, bool = True if checked.
    """

    lora_toggled = Signal(str, bool)
    params_changed = Signal()  # emitted when any persistent control changes

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._suppressing_lora_signals = False
        self._suppressing_params_signals = False
        self._strategy = None  # ModelFamilyStrategy, set via set_family()
        self._build_ui()
        self._wire_persistence_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Checkpoint & VAE (hidden — UnifiedGenTab handles model selection)
        self._row1_widget = QWidget()
        row1 = QHBoxLayout(self._row1_widget)
        row1.setContentsMargins(0, 0, 0, 0)
        self._lbl_checkpoint = QLabel("Checkpoint:")
        row1.addWidget(self._lbl_checkpoint)
        self.checkpoint = QComboBox()
        self.checkpoint.setMinimumWidth(200)
        row1.addWidget(self.checkpoint, 1)
        self._lbl_vae = QLabel("VAE:")
        row1.addWidget(self._lbl_vae)
        self.vae = QComboBox()
        self.vae.addItem("Automatic")
        row1.addWidget(self.vae, 1)
        self._lbl_vae_precision = QLabel("Precision:")
        self._lbl_vae_precision.setToolTip(
            "VAE compute precision. Use 32 (fp32) to escape the SDXL fp16-VAE "
            "black-image/NaN bug; 16 (fp16) is faster."
        )
        row1.addWidget(self._lbl_vae_precision)
        self.vae_precision = QComboBox()
        self.vae_precision.addItem("16", "16")
        self.vae_precision.addItem("32", "32")
        self.vae_precision.setFixedWidth(100)
        row1.addWidget(self.vae_precision)
        self._row1_widget.setVisible(False)
        layout.addWidget(self._row1_widget)

        # Sampler & Scheduler
        self._row2_widget = QWidget()
        row2 = QHBoxLayout(self._row2_widget)
        row2.setContentsMargins(0, 0, 0, 0)
        self._lbl_sampler = QLabel("Sampler:")
        row2.addWidget(self._lbl_sampler)
        self.sampler = QComboBox()
        row2.addWidget(self.sampler, 1)
        self._lbl_scheduler = QLabel("Scheduler:")
        row2.addWidget(self._lbl_scheduler)
        self.scheduler = QComboBox()
        row2.addWidget(self.scheduler, 1)
        layout.addWidget(self._row2_widget)

        # Steps, CFG, Seed (split off Clip Skip / SeqLen to a second row)
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(QLabel("Steps:"))
        self.steps = QSpinBox()
        self.steps.setRange(1, 150)
        self.steps.setValue(20)
        self.steps.setMinimumWidth(85)
        row3.addWidget(self.steps)
        row3.addWidget(QLabel("CFG:"))
        self.cfg_scale = QDoubleSpinBox()
        self.cfg_scale.setRange(1, 30)
        self.cfg_scale.setDecimals(1)
        self.cfg_scale.setValue(7.0)
        self.cfg_scale.setMinimumWidth(95)
        row3.addWidget(self.cfg_scale)
        row3.addWidget(QLabel("Seed:"))
        self.seed = QSpinBox()
        self.seed.setRange(-1, 999999999)
        self.seed.setValue(-1)
        self.seed.setMinimumWidth(110)
        row3.addWidget(self.seed)
        row3.addStretch()
        layout.addLayout(row3)

        # Clip Skip + Max Seq Len
        row3b = QHBoxLayout()
        row3b.setSpacing(8)
        self._lbl_clip_skip = QLabel("Clip Skip:")
        row3b.addWidget(self._lbl_clip_skip)
        self.clip_skip = QSpinBox()
        self.clip_skip.setRange(1, 12)
        self.clip_skip.setValue(1)
        self.clip_skip.setMinimumWidth(85)
        row3b.addWidget(self.clip_skip)
        # Max Seq Len (hidden by default, shown for FLUX)
        self._lbl_seq_len = QLabel("SeqLen:")
        self._lbl_seq_len.setVisible(False)
        row3b.addWidget(self._lbl_seq_len)
        self.max_seq_len = QSpinBox()
        self.max_seq_len.setRange(128, 4096)
        self.max_seq_len.setValue(512)
        self.max_seq_len.setMinimumWidth(90)
        self.max_seq_len.setVisible(False)
        row3b.addWidget(self.max_seq_len)
        row3b.addStretch()
        layout.addLayout(row3b)

        # Dimensions (Size preset + W/H + swap + Lock)
        row4 = QHBoxLayout()
        row4.setSpacing(8)
        row4.addWidget(QLabel("Size:"))
        self.dim_preset = QComboBox()
        self.dim_preset.setMinimumWidth(160)
        self.dim_preset.addItem("Custom")
        for label, _, _ in _DIM_PRESETS:
            self.dim_preset.addItem(label)
        self.dim_preset.currentIndexChanged.connect(self._on_dim_preset_changed)
        row4.addWidget(self.dim_preset)

        row4.addWidget(QLabel("W:"))
        self.width = QSpinBox()
        self.width.setRange(64, 4096)
        self.width.setSingleStep(64)
        self.width.setValue(512)
        self.width.setMinimumWidth(85)
        row4.addWidget(self.width)

        self._swap_btn = QPushButton("\u21c4")
        self._swap_btn.setFixedSize(30, 30)
        self._swap_btn.setToolTip("Swap width / height")
        self._swap_btn.clicked.connect(self._on_swap_dims)
        row4.addWidget(self._swap_btn)

        row4.addWidget(QLabel("H:"))
        self.height = QSpinBox()
        self.height.setRange(64, 4096)
        self.height.setSingleStep(64)
        self.height.setValue(512)
        self.height.setMinimumWidth(85)
        row4.addWidget(self.height)

        self.constrain_proportions = QCheckBox("Lock")
        self.constrain_proportions.setToolTip("Constrain proportions")
        row4.addWidget(self.constrain_proportions)

        # From Source — shown only on tabs that register a dim-source callback
        self._from_source_btn = QPushButton("From Source")
        self._from_source_btn.setToolTip(
            "Set width/height from the source image dimensions"
        )
        self._from_source_btn.clicked.connect(self._on_dims_from_source)
        self._from_source_btn.setVisible(False)
        row4.addWidget(self._from_source_btn)
        row4.addStretch()
        layout.addLayout(row4)

        # Batch / Count / Remove BG (split off the dimensions row)
        row4b = QHBoxLayout()
        row4b.setSpacing(8)
        row4b.addWidget(QLabel("Batch:"))
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(1)
        self.batch_size.setMinimumWidth(85)
        row4b.addWidget(self.batch_size)
        row4b.addWidget(QLabel("Count:"))
        self.batch_count = QSpinBox()
        self.batch_count.setRange(1, 100)
        self.batch_count.setValue(1)
        self.batch_count.setMinimumWidth(85)
        row4b.addWidget(self.batch_count)
        # Remove BG (hidden by default, shown for FLUX/Z-Image)
        self.remove_bg = QCheckBox("Remove BG")
        self.remove_bg.setVisible(False)
        row4b.addWidget(self.remove_bg)
        row4b.addStretch()
        layout.addLayout(row4b)

        # Wire constrain-proportions logic
        self._dim_aspect: float | None = None
        self._dim_updating = False
        self.width.valueChanged.connect(self._on_width_changed)
        self.height.valueChanged.connect(self._on_height_changed)

        # LoRA section
        self._lora_section = CollapsibleSection("LoRA", collapsed=True)
        self.lora_list = QListWidget()
        self.lora_list.setMinimumHeight(150)
        self.lora_list.setMaximumHeight(220)
        self.lora_list.itemDoubleClicked.connect(self._on_lora_double_click)
        self.lora_list.itemChanged.connect(self._on_lora_item_changed)
        self._lora_section.add_widget(self.lora_list)

        lora_btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh LoRAs")
        refresh_btn.clicked.connect(self._on_refresh_loras)
        lora_btn_row.addWidget(refresh_btn)
        self._lbl_lora_weights = QLabel("Weights:")
        self._lbl_lora_weights.setVisible(False)
        lora_btn_row.addWidget(self._lbl_lora_weights)
        self.lora_weights = QLineEdit()
        self.lora_weights.setPlaceholderText("1.0")
        self.lora_weights.setMinimumWidth(120)
        self.lora_weights.setToolTip("Comma-separated LoRA multipliers")
        self.lora_weights.setVisible(False)
        lora_btn_row.addWidget(self.lora_weights, 1)
        lora_btn_w = QWidget()
        lora_btn_w.setLayout(lora_btn_row)
        lora_btn_row.setContentsMargins(0, 0, 0, 0)
        self._lora_section.add_widget(lora_btn_w)
        layout.addWidget(self._lora_section)

        # SDXL Refiner section
        self._refiner_section = CollapsibleSection("SDXL Refiner", collapsed=True)

        self.refiner_enabled = QCheckBox("Enable Refiner (SDXL only)")
        self.refiner_enabled.setToolTip(
            "Run a second refinement pass using an SDXL refiner checkpoint.\n"
            "Only works with SDXL-based models; ignored for SD 1.5."
        )
        self.refiner_enabled.toggled.connect(self._on_refiner_toggled)
        self._refiner_section.add_widget(self.refiner_enabled)

        ref_ckpt_row = QHBoxLayout()
        ref_ckpt_row.setSpacing(6)
        ref_ckpt_row.addWidget(QLabel("Refiner:"))
        self.refiner_checkpoint = QComboBox()
        self.refiner_checkpoint.setMinimumWidth(200)
        self.refiner_checkpoint.setEnabled(False)
        self.refiner_checkpoint.setToolTip("SDXL refiner checkpoint (.safetensors)")
        ref_ckpt_row.addWidget(self.refiner_checkpoint, 1)
        ref_ckpt_w = QWidget()
        ref_ckpt_w.setLayout(ref_ckpt_row)
        self._refiner_section.add_widget(ref_ckpt_w)

        ref_params_row = QHBoxLayout()
        ref_params_row.setSpacing(6)

        ref_params_row.addWidget(QLabel("Switch at:"))
        self.refiner_switch_at = QDoubleSpinBox()
        self.refiner_switch_at.setRange(0.1, 0.95)
        self.refiner_switch_at.setDecimals(2)
        self.refiner_switch_at.setSingleStep(0.05)
        self.refiner_switch_at.setValue(0.80)
        self.refiner_switch_at.setEnabled(False)
        self.refiner_switch_at.setToolTip(
            "Fraction of base steps before switching to refiner (0.8 = refine last 20%)"
        )
        ref_params_row.addWidget(self.refiner_switch_at)

        ref_params_row.addWidget(QLabel("Steps:"))
        self.refiner_steps = QSpinBox()
        self.refiner_steps.setRange(1, 50)
        self.refiner_steps.setValue(10)
        self.refiner_steps.setEnabled(False)
        self.refiner_steps.setToolTip("Number of refiner denoising steps")
        ref_params_row.addWidget(self.refiner_steps)

        ref_params_row.addWidget(QLabel("CFG:"))
        self.refiner_cfg_scale = QDoubleSpinBox()
        self.refiner_cfg_scale.setRange(1, 30)
        self.refiner_cfg_scale.setDecimals(1)
        self.refiner_cfg_scale.setValue(7.0)
        self.refiner_cfg_scale.setEnabled(False)
        self.refiner_cfg_scale.setToolTip("Guidance scale for the refiner pass")
        ref_params_row.addWidget(self.refiner_cfg_scale)

        ref_params_row.addStretch()

        ref_params_w = QWidget()
        ref_params_w.setLayout(ref_params_row)
        self._refiner_section.add_widget(ref_params_w)

        layout.addWidget(self._refiner_section)

    # ── Family switching ──────────────────────────────────────────────────────

    def set_family(self, strategy) -> None:
        """Show/hide controls based on model family strategy."""
        self._strategy = strategy
        if strategy is None:
            return

        # Checkpoint & VAE row
        self._row1_widget.setVisible(strategy.has_checkpoint_selector or strategy.has_vae_selector)
        self._lbl_checkpoint.setVisible(strategy.has_checkpoint_selector)
        self.checkpoint.setVisible(strategy.has_checkpoint_selector)
        self._lbl_vae.setVisible(strategy.has_vae_selector)
        self.vae.setVisible(strategy.has_vae_selector)
        self._lbl_vae_precision.setVisible(strategy.has_vae_selector)
        self.vae_precision.setVisible(strategy.has_vae_selector)

        # Sampler & Scheduler row
        self._lbl_sampler.setVisible(strategy.has_sampler)
        self.sampler.setVisible(strategy.has_sampler)
        self._lbl_scheduler.setVisible(strategy.has_scheduler)
        self.scheduler.setVisible(strategy.has_scheduler)
        self._row2_widget.setVisible(strategy.has_sampler or strategy.has_scheduler)

        # Populate scheduler choices if strategy specifies them
        if strategy.scheduler_choices and self.scheduler.count() == 0:
            self.scheduler.clear()
            self.scheduler.addItems(strategy.scheduler_choices)

        # Clip Skip
        self._lbl_clip_skip.setVisible(strategy.has_clip_skip)
        self.clip_skip.setVisible(strategy.has_clip_skip)

        # Max Seq Len
        self._lbl_seq_len.setVisible(strategy.has_max_seq_len)
        self.max_seq_len.setVisible(strategy.has_max_seq_len)

        # Remove BG
        self.remove_bg.setVisible(strategy.has_remove_bg)

        # LoRA weights (shown for FLUX/Z-Image which use multiplier strings)
        show_weights = strategy.family in ("flux", "zimage")
        self._lbl_lora_weights.setVisible(show_weights)
        self.lora_weights.setVisible(show_weights)

        # Refiner section
        self._refiner_section.setVisible(strategy.has_refiner)

        # Update defaults
        self.steps.setValue(strategy.default_steps)
        self.cfg_scale.setRange(*strategy.cfg_range)
        self.cfg_scale.setValue(strategy.default_cfg)
        self._dim_updating = True
        self.width.setValue(strategy.default_width)
        self.height.setValue(strategy.default_height)
        self._dim_aspect = strategy.default_width / strategy.default_height if strategy.default_height else None
        self._dim_updating = False
        self._sync_preset_combo()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_refiner_toggled(self, checked: bool) -> None:
        self.refiner_checkpoint.setEnabled(checked)
        self.refiner_switch_at.setEnabled(checked)
        self.refiner_steps.setEnabled(checked)
        self.refiner_cfg_scale.setEnabled(checked)

    def set_lora_refresh_callback(self, callback) -> None:
        """Set a callback that returns list[str] of available LoRA names."""
        self._lora_refresh_cb = callback

    def _on_refresh_loras(self) -> None:
        """Rescan LoRAs via the registered callback, preserving checked items."""
        cb = getattr(self, "_lora_refresh_cb", None)
        if cb is None:
            return
        loras = cb()
        if loras is None:
            return
        checked = [
            self.lora_list.item(i).text()
            for i in range(self.lora_list.count())
            if (self.lora_list.item(i).flags() & Qt.ItemFlag.ItemIsUserCheckable)
            and self.lora_list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self._suppressing_lora_signals = True
        populate_lora_list_grouped(self.lora_list, loras, activated=checked)
        self._suppressing_lora_signals = False

    def _on_lora_double_click(self, item: QListWidgetItem) -> None:
        dlg = LoRAConfigDialog(item.text(), parent=self)
        dlg.exec()

    def _on_lora_item_changed(self, item: QListWidgetItem) -> None:
        if self._suppressing_lora_signals:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return
        checked = item.checkState() == Qt.CheckState.Checked
        tag = lora_prompt_tag(item.text())
        self.lora_toggled.emit(tag, checked)

    # ── Dimension helpers ─────────────────────────────────────────────────────

    def _on_dim_preset_changed(self, idx: int) -> None:
        if idx <= 0 or idx > len(_DIM_PRESETS):
            return
        _, w, h = _DIM_PRESETS[idx - 1]
        self._dim_updating = True
        self.width.setValue(w)
        self.height.setValue(h)
        self._dim_aspect = w / h if h else None
        self._dim_updating = False

    def _on_swap_dims(self) -> None:
        w, h = self.width.value(), self.height.value()
        self._dim_updating = True
        self.width.setValue(h)
        self.height.setValue(w)
        if self._dim_aspect:
            self._dim_aspect = 1.0 / self._dim_aspect
        self._dim_updating = False
        self._sync_preset_combo()

    def set_dim_source_callback(self, callback) -> None:
        """Register a callable returning the source image ``(w, h)`` or None.

        Registering shows the "From Source" button next to the dimension
        controls; tabs without a source image never see it.
        """
        self._dim_source_cb = callback
        self._from_source_btn.setVisible(callback is not None)

    def _on_dims_from_source(self) -> None:
        cb = getattr(self, "_dim_source_cb", None)
        wh = cb() if cb is not None else None
        if not wh:
            return
        # Snap to multiples of 8 (SD latent-space requirement)
        w = max(64, min(4096, int(round(wh[0] / 8)) * 8))
        h = max(64, min(4096, int(round(wh[1] / 8)) * 8))
        self._dim_updating = True
        self.width.setValue(w)
        self.height.setValue(h)
        self._dim_updating = False
        self._dim_aspect = w / h if h else None
        self._sync_preset_combo()
        self.params_changed.emit()

    def _on_width_changed(self, val: int) -> None:
        if self._dim_updating:
            return
        if self.constrain_proportions.isChecked() and self._dim_aspect:
            self._dim_updating = True
            self.height.setValue(max(64, int(val / self._dim_aspect)))
            self._dim_updating = False
        else:
            h = self.height.value()
            self._dim_aspect = val / h if h else None
        self._sync_preset_combo()

    def _on_height_changed(self, val: int) -> None:
        if self._dim_updating:
            return
        if self.constrain_proportions.isChecked() and self._dim_aspect:
            self._dim_updating = True
            self.width.setValue(max(64, int(val * self._dim_aspect)))
            self._dim_updating = False
        else:
            w = self.width.value()
            self._dim_aspect = w / val if val else None
        self._sync_preset_combo()

    def _sync_preset_combo(self) -> None:
        """Update preset combo to reflect current W/H if it matches one."""
        w, h = self.width.value(), self.height.value()
        self.dim_preset.blockSignals(True)
        matched = 0
        for i, (_, pw, ph) in enumerate(_DIM_PRESETS):
            if pw == w and ph == h:
                matched = i + 1
                break
        self.dim_preset.setCurrentIndex(matched)
        self.dim_preset.blockSignals(False)

    # ── Populate ──────────────────────────────────────────────────────────────

    def populate_options(
        self,
        checkpoints: list[str] | None = None,
        vaes: list[str] | None = None,
        samplers: list[str] | None = None,
        schedulers: list[str] | None = None,
        loras: list[str] | None = None,
        refiners: list[str] | None = None,
    ) -> None:
        if checkpoints is not None:
            self._repopulate_combo(self.checkpoint, checkpoints)
        if vaes is not None:
            self._repopulate_combo(self.vae, vaes, first_item="Automatic")
        if samplers is not None:
            self._repopulate_combo(self.sampler, samplers)
        if schedulers is not None:
            self._repopulate_combo(self.scheduler, schedulers)
        if loras is not None:
            self._suppressing_lora_signals = True
            populate_lora_list_grouped(self.lora_list, loras)
            self._suppressing_lora_signals = False
        if refiners is not None:
            self._repopulate_combo(self.refiner_checkpoint, refiners, first_item="(none)")

    @staticmethod
    def _repopulate_combo(combo, items: list[str], first_item: str = "") -> None:
        """Refill a combo, keeping the current selection if it still exists."""
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if first_item:
            combo.addItem(first_item)
        combo.addItems(items)
        idx = combo.findText(current) if current else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    # ── Persistence signal wiring ─────────────────────────────────────────────

    def _wire_persistence_signals(self) -> None:
        """Emit ``params_changed`` whenever any field the user cares about changes.

        Covers: checkpoint, VAE, sampler, scheduler, steps, CFG, seed,
        clip skip, width/height, batch size/count, max seq len, remove bg,
        LoRA weights string, refiner knobs. LoRA list check state is
        already handled via ``lora_toggled`` so we forward it here too.
        """
        def _emit() -> None:
            if not self._suppressing_params_signals:
                self.params_changed.emit()

        # Combos
        for combo in (self.checkpoint, self.vae, self.vae_precision,
                      self.sampler, self.scheduler,
                      getattr(self, "refiner_checkpoint", None)):
            if combo is not None:
                combo.currentIndexChanged.connect(lambda _i: _emit())

        # Spin / double-spin boxes
        for spin in (self.steps, self.cfg_scale, self.seed, self.clip_skip,
                     self.width, self.height, self.batch_size, self.batch_count,
                     self.max_seq_len,
                     getattr(self, "refiner_switch_at", None),
                     getattr(self, "refiner_steps", None),
                     getattr(self, "refiner_cfg_scale", None)):
            if spin is None:
                continue
            if hasattr(spin, "valueChanged"):
                spin.valueChanged.connect(lambda _v: _emit())

        # Checkboxes
        for cb in (self.remove_bg, getattr(self, "refiner_enabled", None)):
            if cb is not None:
                cb.toggled.connect(lambda _b: _emit())

        # LoRA multipliers text field
        if hasattr(self, "lora_weights"):
            self.lora_weights.textChanged.connect(lambda _t: _emit())

        # LoRA list check state changes are already emitted via lora_toggled
        self.lora_toggled.connect(lambda _tag, _on: _emit())

    # ── Config I/O ────────────────────────────────────────────────────────────

    def _prefix(self, strategy=None) -> str:
        s = strategy or self._strategy
        return s.config_prefix if s else "img"

    def collect_to_config(self, cfg: ProjectConfig, strategy=None) -> None:
        p = self._prefix(strategy)

        if self.checkpoint.isVisibleTo(self):
            setattr(cfg, f"{p}_checkpoint", self.checkpoint.currentText())
        if self.vae.isVisibleTo(self):
            setattr(cfg, f"{p}_vae", self.vae.currentText())
        if self.vae_precision.isVisibleTo(self):
            # Global field (shared across families) — escape hatch for the
            # SDXL fp16-VAE NaN; values "16" / "32".
            setattr(cfg, "sd_vae_precision", self.vae_precision.currentData() or "16")
        if self.sampler.isVisibleTo(self):
            setattr(cfg, f"{p}_sampler", self.sampler.currentText())
        if self.scheduler.isVisibleTo(self):
            setattr(cfg, f"{p}_scheduler", self.scheduler.currentText())

        setattr(cfg, f"{p}_steps", self.steps.value())
        setattr(cfg, f"{p}_cfg_scale", self.cfg_scale.value())
        setattr(cfg, f"{p}_seed", self.seed.value())

        if self.clip_skip.isVisibleTo(self):
            setattr(cfg, f"{p}_clip_skip", self.clip_skip.value())

        setattr(cfg, f"{p}_width", self.width.value())
        setattr(cfg, f"{p}_height", self.height.value())

        if hasattr(cfg, f"{p}_batch_size"):
            setattr(cfg, f"{p}_batch_size", self.batch_size.value())
        setattr(cfg, f"{p}_batch_count", self.batch_count.value())

        if self.max_seq_len.isVisibleTo(self):
            setattr(cfg, f"{p}_max_seq_len", self.max_seq_len.value())

        if self.remove_bg.isVisibleTo(self):
            setattr(cfg, f"{p}_remove_bg", self.remove_bg.isChecked())

        # LoRAs
        activated = []
        for i in range(self.lora_list.count()):
            item = self.lora_list.item(i)
            if item.checkState() == Qt.Checked:
                activated.append(item.text())

        # Different config field names per family
        lora_field = f"{p}_loras" if p == "img" else f"{p}_activated_loras"
        if hasattr(cfg, lora_field):
            setattr(cfg, lora_field, activated)
        elif hasattr(cfg, f"{p}_loras"):
            setattr(cfg, f"{p}_loras", activated)
        elif hasattr(cfg, f"{p}_activated_loras"):
            setattr(cfg, f"{p}_activated_loras", activated)

        # LoRA weights string (FLUX/Z-Image)
        if self.lora_weights.isVisibleTo(self):
            mf = f"{p}_lora_multipliers"
            if hasattr(cfg, mf):
                setattr(cfg, mf, self.lora_weights.text())

        # Refiner (img prefix only)
        if p == "img":
            cfg.img_refiner_enabled = self.refiner_enabled.isChecked()
            ref_ckpt = self.refiner_checkpoint.currentText()
            cfg.img_refiner_checkpoint = "" if ref_ckpt == "(none)" else ref_ckpt
            cfg.img_refiner_switch_at = self.refiner_switch_at.value()
            cfg.img_refiner_steps = self.refiner_steps.value()
            cfg.img_refiner_cfg_scale = self.refiner_cfg_scale.value()

    def restore_from_config(self, cfg: ProjectConfig, strategy=None) -> None:
        self._suppressing_params_signals = True
        try:
            self._restore_from_config_impl(cfg, strategy)
        finally:
            self._suppressing_params_signals = False

    def _restore_from_config_impl(self, cfg: ProjectConfig, strategy=None) -> None:
        p = self._prefix(strategy)

        # Checkpoint
        if self.checkpoint.isVisibleTo(self):
            ckpt = getattr(cfg, f"{p}_checkpoint", "") or ""
            if not ckpt and p == "img":
                ckpt = getattr(cfg, "default_image_model", "") or ""
            if ckpt:
                idx = self.checkpoint.findText(ckpt)
                if idx < 0:
                    # Keep the saved choice even when the scan didn't find it
                    # (e.g. model drive not mounted) so the next auto-persist
                    # doesn't stomp it with a default.
                    self.checkpoint.addItem(ckpt)
                    idx = self.checkpoint.count() - 1
                self.checkpoint.setCurrentIndex(idx)

        # VAE
        if self.vae.isVisibleTo(self):
            vae = getattr(cfg, f"{p}_vae", "Automatic") or "Automatic"
            idx = self.vae.findText(vae)
            if idx < 0 and vae != "Automatic":
                self.vae.addItem(vae)
                idx = self.vae.count() - 1
            if idx >= 0:
                self.vae.setCurrentIndex(idx)

        # VAE precision (global field)
        if self.vae_precision.isVisibleTo(self):
            prec = str(getattr(cfg, "sd_vae_precision", "16") or "16")
            idx = self.vae_precision.findData(prec)
            self.vae_precision.setCurrentIndex(idx if idx >= 0 else 0)

        # Sampler
        if self.sampler.isVisibleTo(self):
            sampler = getattr(cfg, f"{p}_sampler", "") or ""
            if not sampler and p == "img":
                sampler = getattr(cfg, "default_image_sampler", "") or ""
            idx = self.sampler.findText(sampler)
            if idx >= 0:
                self.sampler.setCurrentIndex(idx)

        # Scheduler
        if self.scheduler.isVisibleTo(self):
            sched = getattr(cfg, f"{p}_scheduler", "") or ""
            idx = self.scheduler.findText(sched)
            if idx >= 0:
                self.scheduler.setCurrentIndex(idx)

        # Steps
        s = self._strategy or strategy
        default_steps = s.default_steps if s else 20
        steps = getattr(cfg, f"{p}_steps", 0) or 0
        self.steps.setValue(steps if steps > 0 else default_steps)

        # CFG
        default_cfg = s.default_cfg if s else 7.0
        cfg_val = getattr(cfg, f"{p}_cfg_scale", None)
        if cfg_val is not None and cfg_val > 0:
            self.cfg_scale.setValue(cfg_val)
        else:
            self.cfg_scale.setValue(default_cfg)

        # Seed
        seed = getattr(cfg, f"{p}_seed", -1)
        self.seed.setValue(seed if seed is not None else -1)

        # Clip Skip
        if self.clip_skip.isVisibleTo(self):
            self.clip_skip.setValue(getattr(cfg, f"{p}_clip_skip", 1) or 1)

        # Width / Height
        default_w = s.default_width if s else 512
        default_h = s.default_height if s else 512
        w = getattr(cfg, f"{p}_width", 0) or 0
        h = getattr(cfg, f"{p}_height", 0) or 0
        self._dim_updating = True
        if w > 0:
            self.width.setValue(w)
        else:
            img_res = getattr(cfg, "image_resolution", "") or ""
            if img_res and "x" in img_res:
                self.width.setValue(int(img_res.split("x")[0]))
            else:
                self.width.setValue(default_w)
        if h > 0:
            self.height.setValue(h)
        else:
            img_res = getattr(cfg, "image_resolution", "") or ""
            if img_res and "x" in img_res:
                self.height.setValue(int(img_res.split("x")[1]))
            else:
                self.height.setValue(default_h)
        final_w = self.width.value()
        final_h = self.height.value()
        self._dim_aspect = final_w / final_h if final_h else None
        self._dim_updating = False
        self._sync_preset_combo()

        # Batch
        if hasattr(cfg, f"{p}_batch_size"):
            self.batch_size.setValue(getattr(cfg, f"{p}_batch_size", 1) or 1)
        self.batch_count.setValue(getattr(cfg, f"{p}_batch_count", 1) or 1)

        # Max Seq Len
        if self.max_seq_len.isVisibleTo(self):
            self.max_seq_len.setValue(getattr(cfg, f"{p}_max_seq_len", 512) or 512)

        # Remove BG
        if self.remove_bg.isVisibleTo(self):
            self.remove_bg.setChecked(getattr(cfg, f"{p}_remove_bg", False) or False)

        # Restore LoRA selections
        self._suppressing_lora_signals = True
        lora_field = f"{p}_loras" if p == "img" else f"{p}_activated_loras"
        loras = getattr(cfg, lora_field, None)
        if loras is None:
            loras = getattr(cfg, f"{p}_loras", None) or getattr(cfg, f"{p}_activated_loras", None)
        if loras:
            for i in range(self.lora_list.count()):
                item = self.lora_list.item(i)
                if item.text() in loras:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
        self._suppressing_lora_signals = False

        # LoRA weights
        if self.lora_weights.isVisibleTo(self):
            mf = f"{p}_lora_multipliers"
            self.lora_weights.setText(getattr(cfg, mf, "") or "")

        # Restore refiner (img prefix only)
        if p == "img":
            self.refiner_enabled.setChecked(getattr(cfg, "img_refiner_enabled", False))
            ref_ckpt = getattr(cfg, "img_refiner_checkpoint", "")
            if ref_ckpt:
                idx = self.refiner_checkpoint.findText(ref_ckpt)
                if idx >= 0:
                    self.refiner_checkpoint.setCurrentIndex(idx)
            self.refiner_switch_at.setValue(getattr(cfg, "img_refiner_switch_at", 0.8))
            self.refiner_steps.setValue(getattr(cfg, "img_refiner_steps", 10))
            self.refiner_cfg_scale.setValue(getattr(cfg, "img_refiner_cfg_scale", 7.0))

    # ── LoRA helpers ──────────────────────────────────────────────────────────

    def get_active_loras(self) -> list[str]:
        """Return list of checked LoRA names."""
        active = []
        for i in range(self.lora_list.count()):
            item = self.lora_list.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                active.append(item.text())
        return active

    def set_active_loras(self, names: list[str]) -> None:
        """Check LoRAs matching the given names."""
        self._suppressing_lora_signals = True
        name_set = set(names) if names else set()
        for i in range(self.lora_list.count()):
            item = self.lora_list.item(i)
            if item:
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in name_set else Qt.CheckState.Unchecked
                )
        self._suppressing_lora_signals = False
