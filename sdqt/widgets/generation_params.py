"""Shared generation parameters widget used by Generate and Video Extender tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.collapsible_section import CollapsibleSection

# (display_name, internal_key) — internal_key is stored in ProjectConfig
_MODEL_TYPES = [
    ("Wan 2.1 I2V 14B (480P)", "i2v_2_1_480p"),
    ("Wan 2.1 I2V 14B (720P)", "i2v_2_1_720p"),
    ("Wan 2.2 I2V 14B", "i2v_2_2"),
    ("Wan 2.2 I2V 14B Lightning v2", "i2v_2_2_lightning_v2"),
    ("Wan 2.2 I2V 14B + Lightning v2 LoRAs (tunable)", "i2v_2_2_lightning_v2_loras"),
    ("Wan 2.2 SVI 2 Pro (LoRAs)", "i2v_2_2_svi_2_pro"),
    ("Wan 2.2 SVI 2 Pro Enhanced Lightning v2", "i2v_2_2_svi_2_pro_lightning_v2"),
    ("LTX 2.3 Dev (22B)", "ltx_2_3_dev"),
    ("LTX 2.3 Distilled (22B)", "ltx_2_3_distilled"),
]

# Model keys that use SVI 2 Pro orchestration (sliding window + anchor image).
SVI_MODEL_KEYS = frozenset({"i2v_2_2_svi_2_pro", "i2v_2_2_svi_2_pro_lightning_v2"})

# Per-model-type recommended defaults for advanced settings.
# When the user switches model type, these are applied to the UI.
MODEL_DEFAULTS: dict[str, dict] = {
    "i2v_2_1_480p": {
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "guidance2_scale": 5.0,
        "flow_shift": 3.0,
        "guidance_phases": 1,
        "switch_threshold": 0,
    },
    "i2v_2_1_720p": {
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "guidance2_scale": 5.0,
        "flow_shift": 3.0,
        "guidance_phases": 1,
        "switch_threshold": 0,
    },
    "i2v_2_2": {
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "guidance2_scale": 5.0,
        "flow_shift": 3.0,
        "guidance_phases": 1,
        "switch_threshold": 0,
    },
    "i2v_2_2_lightning_v2": {
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "guidance2_scale": 1.0,
        "flow_shift": 5.0,
        "guidance_phases": 2,
        "switch_threshold": 900,
    },
    # Lightning v2 as standalone LoRAs on top of base Wan 2.2 14B. Same step
    # count as merged Lightning, but exposes per-phase weight control via
    # multipliers (default "0.7;0 0;1.0") so the user can dial down the
    # high-noise pass to reduce learned warm-bias / chroma drift.
    "i2v_2_2_lightning_v2_loras": {
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "guidance2_scale": 1.0,
        "flow_shift": 5.0,
        "guidance_phases": 2,
        "switch_threshold": 900,
    },
    # SVI 2 Pro (LoRA on top of base Wan 2.2): wan2gp defaults/i2v_2_2_svi2pro.json
    "i2v_2_2_svi_2_pro": {
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "guidance2_scale": 3.5,
        "flow_shift": 5.0,
        "guidance_phases": 2,
        "switch_threshold": 900,
    },
    # SVI 2 Pro Enhanced Lightning v2: wan2gp defaults/i2v_2_2_Enhanced_Lightning_v2_svi2pro.json
    "i2v_2_2_svi_2_pro_lightning_v2": {
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "guidance2_scale": 1.0,
        "flow_shift": 5.0,
        "guidance_phases": 2,
        "switch_threshold": 900,
    },
    "ltx_2_3_dev": {
        "num_inference_steps": 26,
        "guidance_scale": 3.0,
        "guidance2_scale": 0.0,
        "flow_shift": 1.0,
        "guidance_phases": 1,
        "switch_threshold": 0,
    },
    "ltx_2_3_distilled": {
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "guidance2_scale": 0.0,
        "flow_shift": 1.0,
        "guidance_phases": 1,
        "switch_threshold": 0,
    },
}

# Map model_type key → registry feature key for auto-download
MODEL_TYPE_TO_FEATURE = {
    "i2v_2_1_480p": "video_gen_21_480p",
    "i2v_2_1_720p": "video_gen_21_720p",
    "i2v_2_2": "video_gen",
    "i2v_2_2_lightning_v2": "video_gen",
    # LoRA variant uses base Wan 2.2 transformers + Lightning LoRAs stacked.
    "i2v_2_2_lightning_v2_loras": "video_gen",
    # SVI 2 Pro (LoRA variant) reuses the base Wan 2.2 transformers.
    "i2v_2_2_svi_2_pro": "video_gen",
    # Enhanced Lightning v2 SVI ships its own dedicated FP8 transformer pair
    # (wan22EnhancedLightning_v2I2VFP8HIGH/LOW.safetensors). The user must
    # point Settings → Video Model Paths at them; no auto-download yet.
    "i2v_2_2_svi_2_pro_lightning_v2": "video_gen",
    "ltx_2_3_dev": "ltx_video_dev",
    "ltx_2_3_distilled": "ltx_video_distilled",
}

# Default LoRA filenames + multipliers auto-applied when SVI 2 Pro (LoRA) is
# selected. Multipliers follow wan2gp's phase syntax: "1;0 0;1" = HIGH on
# phase 1, LOW on phase 2. Keep stems only — _apply_loras matches by stem.
SVI_2_PRO_LORAS = [
    "SVI_Wan2.2-I2V-A14B_high_noise_lora_v2.0_pro",
    "SVI_Wan2.2-I2V-A14B_low_noise_lora_v2.0_pro",
]
SVI_2_PRO_LORA_MULTIPLIERS = "1;0 0;1"

# Default LoRAs and multipliers auto-applied when "Lightning v2 (LoRAs)" is
# selected. HIGH at 0.7 on phase 1 (reduces high-noise pass strength —
# claude.web's recommended fix for warm/chroma drift), LOW at 1.0 on
# phase 2. Stems only — _select_loras_by_prefix matches the on-disk file
# stems case-sensitively.
LIGHTNING_V2_I2V_LORAS = [
    "Wan2.2-Lightning_I2V-A14B-4steps_HIGH",
    "Wan2.2-Lightning_I2V-A14B-4steps_LOW",
]
LIGHTNING_V2_I2V_LORA_MULTIPLIERS = "0.7;0 0;1.0"

_WAN_RESOLUTION_PRESETS = [
    "832x480 (16:9)",
    "480x832 (9:16)",
    "512x512 (1:1)",
    "720x720 (1:1)",
    "832x624 (4:3)",
    "624x832 (3:4)",
    "960x544 (16:9)",
    "544x960 (9:16)",
    "1024x1024 (1:1)",
    "1280x720 (16:9)",
    "720x1280 (9:16)",
    "1280x544 (21:9)",
    "544x1280 (9:21)",
]

_LTX_RESOLUTION_PRESETS = [
    "576x320 (16:9)",
    "320x576 (9:16)",
    "768x448 (16:9)",
    "448x768 (9:16)",
    "512x512 (1:1)",
    "1024x576 (16:9)",
    "576x1024 (9:16)",
    "1024x768 (4:3)",
    "768x1024 (3:4)",
    "1216x704 (16:9)",
    "704x1216 (9:16)",
    "1280x768 (16:9)",
    "768x1280 (9:16)",
]

_RESOLUTION_PRESETS = _WAN_RESOLUTION_PRESETS

_WAN_DURATION_PRESETS = [
    ("17 frames (1s)", 17),
    ("25 frames (1.5s)", 25),
    ("33 frames (2s)", 33),
    ("41 frames (2.5s)", 41),
    ("49 frames (3s)", 49),
    ("65 frames (4s)", 65),
    ("81 frames (5s)", 81),
]

_LTX_DURATION_PRESETS = [
    ("9 frames (0.3s)", 9),
    ("25 frames (1s)", 25),
    ("57 frames (2s)", 57),
    ("121 frames (4s)", 121),
    ("161 frames (5s)", 161),
    ("241 frames (8s)", 241),
    ("321 frames (10s)", 321),
]

_DURATION_PRESETS = _WAN_DURATION_PRESETS

# Color-match methods exposed in the Quality section. The first option is
# our drift-resistant default; the rest dispatch through color-matcher
# (same library KJNodes/ColorMatch wraps).
_CC_METHODS = [
    ("Mean-only Lab (drift-resistant)", "mean-only-lab"),
    ("Reinhard (mean + std)", "reinhard"),
    ("MKL (Monge-Kantorovich)", "mkl"),
    ("MVGD (multivariate Gaussian)", "mvgd"),
    ("HM (histogram match)", "hm"),
    ("HM-MVGD-HM (best general)", "hm-mvgd-hm"),
    ("HM-MKL-HM (fast compound)", "hm-mkl-hm"),
]


def get_video_backend(model_type: str) -> str:
    """Return 'wan' or 'ltx' based on model_type key."""
    return "ltx" if model_type.startswith("ltx_") else "wan"

# SVI 2 Pro anchor-image modes. Display label → internal key persisted in cfg.
#
# - start:                Window 0 uses source image; later windows chain
#                          from prior tail. Smooth motion, identity may drift.
# - single:               Every window starts from the same anchor (identity
#                          locked across all windows). Smooth motion +
#                          continuous identity.
# - per_window:           Hard cuts — window N's *first frame* is anchors[N].
#                          Use for staged scenes / chapters.
# - per_window_keyframes: Smooth animation — window N's *last frame* is
#                          anchors[N], chained start. Use for stretching a
#                          fast action across the full duration via
#                          progressive stage images.
_ANCHOR_MODES = [
    ("Use start frame as ref", "start"),
    ("Single anchor image", "single"),
    ("Anchor image per window (hard cuts)", "per_window"),
    ("Per-window end keyframes (smooth)", "per_window_keyframes"),
]

_TEA_CACHE_OPTIONS = ["off", "light", "medium", "heavy"]
_SELF_REFINER_OPTIONS = ["Disabled", "P1-Norm", "P2-Norm"]
_TEMPORAL_UPSAMPLING_OPTIONS = ["Disabled", "RIFE x2", "RIFE x4"]
_SPATIAL_UPSAMPLING_OPTIONS = ["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"]

# Post-processing presets: (label, temporal, spatial, grain_intensity, grain_saturation)
_PP_PRESETS: list[tuple[str, str, str, float, float]] = [
    ("— Preset —",    "Disabled",   "Disabled",      0.0,  0.0),
    ("Full Enhance",  "RIFE x2",    "Lanczos 2.0x",  0.0,  0.0),
    ("Smooth 60fps",  "RIFE x4",    "Disabled",       0.0,  0.0),
    ("Upscale Only",  "Disabled",   "Lanczos 2.0x",   0.0,  0.0),
    ("Upscale 1.5x",  "Disabled",   "Lanczos 1.5x",   0.0,  0.0),
    ("Film Look",     "Disabled",   "Disabled",       0.15, 0.40),
    ("Cinematic",     "RIFE x2",    "Lanczos 2.0x",   0.10, 0.30),
    ("Reset",         "Disabled",   "Disabled",       0.0,  0.0),
]


class GenerationParamsWidget(QWidget):
    """Reusable widget for generation parameters shared between tabs.

    Exposes:
    - Core: steps, guidance, guidance2, seed
    - Advanced: flow_shift, solver, denoising, masking, guidance_phases, etc.
    - Quality: SLG, APG, motion, color correction, self refiner
    - Post-processing: temporal/spatial upsampling, film grain
    - LoRA: list + multipliers
    """

    def __init__(self, show_lora: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._show_lora = show_lora
        self._current_backend = "wan"
        self._wan_only_widgets: list[QWidget] = []
        self._build_ui()
        self.model_type.currentIndexChanged.connect(self._on_model_type_changed)

    # House-standard minimum widths for consistent grid alignment. These are
    # min-widths (not fixed) so controls grow with the larger global font and
    # reflow on resize instead of truncating.
    _W_SPIN = 90
    _W_DSPIN = 95
    _W_COMBO_S = 120
    _W_COMBO_M = 160
    _W_COMBO_L = 200

    @staticmethod
    def _divider() -> QFrame:
        """A thin horizontal rule used to visually separate logical row groups."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #3a3a3a;")
        return line

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Model Timing
        mt_section = CollapsibleSection("Model Timing")

        # Row 1: Model + Resolution (the two wide selectors get their own line
        # so the long display names never truncate).
        mt_row1 = QHBoxLayout()
        mt_row1.setSpacing(8)
        mt_row1.addWidget(QLabel("Model:"))
        self.model_type = QComboBox()
        self.model_type.setMinimumWidth(self._W_COMBO_L)
        for display, key in _MODEL_TYPES:
            self.model_type.addItem(display, userData=key)
        mt_row1.addWidget(self.model_type, 1)
        mt_row1.addWidget(QLabel("Res:"))
        self.resolution = QComboBox()
        self.resolution.setMinimumWidth(self._W_COMBO_M)
        self.resolution.addItems(_RESOLUTION_PRESETS)
        mt_row1.addWidget(self.resolution)
        mt_row1.addStretch()
        mt_section.add_layout(mt_row1)

        # Row 2: Duration (dropdown) + Custom frame count + FPS. The duration
        # radio bank was 7+ buttons on one line; a single dropdown is far less
        # cramped and grows cleanly with the larger font.
        mt_row2 = QHBoxLayout()
        mt_row2.setSpacing(8)
        mt_row2.addWidget(QLabel("Duration:"))
        # Track the currently-active preset list so _current_duration_frames /
        # restore_from_config index into the same list the dropdown displays.
        # _reconfigure_for_backend swaps this when the model backend changes.
        self._active_duration_presets = list(_DURATION_PRESETS)
        self.duration_combo = QComboBox()
        self.duration_combo.setMinimumWidth(self._W_COMBO_M)
        for label, _ in _DURATION_PRESETS:
            self.duration_combo.addItem(label)
        self.duration_combo.addItem("Custom")
        # Default to the 81-frame preset (matches the previous default radio).
        for i, (label, _) in enumerate(_DURATION_PRESETS):
            if label.startswith("81"):
                self.duration_combo.setCurrentIndex(i)
                break
        mt_row2.addWidget(self.duration_combo)
        # Custom frame-count spinbox — for SVI long-form etc.
        self._duration_custom_frames = QSpinBox()
        self._duration_custom_frames.setMinimumWidth(self._W_SPIN)
        self._duration_custom_frames.setRange(9, 1009)
        self._duration_custom_frames.setSingleStep(4)
        self._duration_custom_frames.setValue(241)
        self._duration_custom_frames.setSuffix(" fr")
        self._duration_custom_frames.setEnabled(False)
        self._duration_custom_frames.setToolTip(
            "Custom frame count. Wan: (n-1) % 4 == 0; LTX: (n-1) % 8 == 0. "
            "Auto-snaps on save."
        )
        mt_row2.addWidget(self._duration_custom_frames)
        self.duration_combo.currentIndexChanged.connect(self._on_duration_changed)
        mt_row2.addWidget(QLabel("FPS:"))
        self.fps = QSpinBox()
        self.fps.setMinimumWidth(self._W_SPIN)
        self.fps.setRange(1, 60)
        self.fps.setValue(16)
        mt_row2.addWidget(self.fps)
        mt_row2.addStretch()
        mt_section.add_layout(mt_row2)

        layout.addWidget(mt_section)

        # Advanced Settings
        adv = CollapsibleSection("Advanced Settings", collapsed=True)

        # Row 1: Steps, Guidance, Guidance 2 (the core sampler scalars).
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Steps:"))
        self.steps = QSpinBox()
        self.steps.setMinimumWidth(self._W_SPIN)
        self.steps.setRange(1, 100)
        self.steps.setValue(4)
        row1.addWidget(self.steps)
        row1.addWidget(QLabel("Guidance:"))
        self.guidance_scale = QDoubleSpinBox()
        self.guidance_scale.setMinimumWidth(self._W_DSPIN)
        self.guidance_scale.setRange(0, 30)
        self.guidance_scale.setDecimals(1)
        self.guidance_scale.setValue(1.0)
        row1.addWidget(self.guidance_scale)
        self._guidance2_label = QLabel("Guidance 2:")
        row1.addWidget(self._guidance2_label)
        self.guidance2_scale = QDoubleSpinBox()
        self.guidance2_scale.setMinimumWidth(self._W_DSPIN)
        self.guidance2_scale.setRange(0, 30)
        self.guidance2_scale.setDecimals(1)
        self.guidance2_scale.setValue(1.0)
        row1.addWidget(self.guidance2_scale)
        self._wan_only_widgets.extend([self._guidance2_label, self.guidance2_scale])
        row1.addStretch()
        adv.add_layout(row1)

        # Row 1b (LTX-only): Alt CFG + Discard. Split off Row 1 so the Wan and
        # LTX guidance controls never crowd the same line. The whole row is a
        # single widget so it can be shown/hidden as a unit on the LTX path.
        self._ltx_guidance_row = QWidget()
        row1b = QHBoxLayout(self._ltx_guidance_row)
        row1b.setContentsMargins(0, 0, 0, 0)
        row1b.setSpacing(8)
        # Alt CFG (LTX Spatial-Temporal Guidance) — only meaningful on LTX path.
        # Distilled effective range ~1.0 (off) to ~5.0 (strong motion adherence).
        self._alt_cfg_label = QLabel("Alt CFG:")
        row1b.addWidget(self._alt_cfg_label)
        self.alt_guidance_scale = QDoubleSpinBox()
        self.alt_guidance_scale.setMinimumWidth(self._W_DSPIN)
        self.alt_guidance_scale.setRange(0, 10)
        self.alt_guidance_scale.setDecimals(1)
        self.alt_guidance_scale.setSingleStep(0.5)
        self.alt_guidance_scale.setValue(1.0)
        self.alt_guidance_scale.setToolTip(
            "LTX Spatial-Temporal Guidance (alt_guidance_scale).\n"
            "1.0 = off. 3.0–5.0 strengthens motion + prompt adherence on distilled.\n"
            "Higher values can over-saturate motion or introduce artifacts."
        )
        row1b.addWidget(self.alt_guidance_scale)
        # Discard last frames — distilled often produces collapsed tail frames.
        # Shares cfg.discard_last_frames with the Wan Quality section's spinbox.
        self._ltx_discard_label = QLabel("Discard:")
        row1b.addWidget(self._ltx_discard_label)
        self.ltx_discard_last = QSpinBox()
        self.ltx_discard_last.setMinimumWidth(self._W_SPIN)
        self.ltx_discard_last.setRange(0, 32)
        self.ltx_discard_last.setSingleStep(1)
        self.ltx_discard_last.setValue(0)
        self.ltx_discard_last.setToolTip(
            "Trim N collapsed frames from the END of LTX output.\n"
            "Distilled commonly leaves 2–6 dead frames at the tail; try 4."
        )
        row1b.addWidget(self.ltx_discard_last)
        row1b.addStretch()
        adv.add_widget(self._ltx_guidance_row)
        self._ltx_only_widgets: list[QWidget] = getattr(self, "_ltx_only_widgets", [])
        self._ltx_only_widgets.extend([
            self._alt_cfg_label, self.alt_guidance_scale,
            self._ltx_discard_label, self.ltx_discard_last,
            self._ltx_guidance_row,
        ])

        adv.add_widget(self._divider())

        # Row 2: Flow Shift, Solver, Seed
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Flow Shift:"))
        self.flow_shift = QDoubleSpinBox()
        self.flow_shift.setMinimumWidth(self._W_DSPIN)
        self.flow_shift.setRange(2, 12)
        self.flow_shift.setDecimals(1)
        self.flow_shift.setValue(3.0)
        row2.addWidget(self.flow_shift)
        row2.addWidget(QLabel("Solver:"))
        self.solver = QComboBox()
        self.solver.setMinimumWidth(self._W_COMBO_S)
        self.solver.addItems(["unipc", "euler"])
        row2.addWidget(self.solver)
        row2.addWidget(QLabel("Seed:"))
        self.seed = QSpinBox()
        self.seed.setMinimumWidth(self._W_COMBO_S)
        self.seed.setRange(-1, 999999999)
        self.seed.setValue(-1)
        row2.addWidget(self.seed)
        row2.addStretch()
        adv.add_layout(row2)

        # Row 3: Denoising
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(QLabel("Denoising:"))
        self.denoising_strength = QDoubleSpinBox()
        self.denoising_strength.setMinimumWidth(self._W_DSPIN)
        self.denoising_strength.setRange(0, 1)
        self.denoising_strength.setDecimals(2)
        self.denoising_strength.setSingleStep(0.05)
        self.denoising_strength.setValue(1.0)
        row3.addWidget(self.denoising_strength)
        row3.addStretch()
        adv.add_layout(row3)

        # Row 4: Phases, Switch (Wan-only)
        self._phases_row = QWidget()
        row4 = QHBoxLayout(self._phases_row)
        row4.setContentsMargins(0, 0, 0, 0)
        row4.setSpacing(8)
        row4.addWidget(QLabel("Phases:"))
        self.guidance_phases = QSpinBox()
        self.guidance_phases.setMinimumWidth(self._W_SPIN)
        self.guidance_phases.setRange(1, 2)
        self.guidance_phases.setValue(1)
        row4.addWidget(self.guidance_phases)
        row4.addWidget(QLabel("Switch:"))
        self.switch_threshold = QSpinBox()
        self.switch_threshold.setMinimumWidth(self._W_SPIN)
        self.switch_threshold.setRange(0, 900)
        self.switch_threshold.setValue(0)
        row4.addWidget(self.switch_threshold)
        row4.addStretch()
        adv.add_widget(self._phases_row)
        self._wan_only_widgets.append(self._phases_row)

        adv.add_widget(self._divider())

        # Track global denoising loop (standard vs wan2gp)
        self._loop_is_wan2gp = False

        # NAG / TEA enable gate
        self.nag_tea_enable = QCheckBox("Enable NAG / TEA Cache")
        self.nag_tea_enable.setChecked(False)
        adv.add_widget(self.nag_tea_enable)

        # Row 5: NAG Scale, Tau, Alpha
        row5 = QHBoxLayout()
        row5.setSpacing(8)
        row5.addWidget(QLabel("NAG Scale:"))
        self.nag_scale = QDoubleSpinBox()
        self.nag_scale.setMinimumWidth(self._W_DSPIN)
        self.nag_scale.setRange(0, 100)
        self.nag_scale.setDecimals(2)
        self.nag_scale.setValue(0.0)
        row5.addWidget(self.nag_scale)
        row5.addWidget(QLabel("Tau:"))
        self.nag_tau = QDoubleSpinBox()
        self.nag_tau.setMinimumWidth(self._W_DSPIN)
        self.nag_tau.setRange(0, 100)
        self.nag_tau.setDecimals(2)
        self.nag_tau.setValue(0.0)
        row5.addWidget(self.nag_tau)
        row5.addWidget(QLabel("Alpha:"))
        self.nag_alpha = QDoubleSpinBox()
        self.nag_alpha.setMinimumWidth(self._W_DSPIN)
        self.nag_alpha.setRange(0, 100)
        self.nag_alpha.setDecimals(2)
        self.nag_alpha.setValue(0.0)
        row5.addWidget(self.nag_alpha)
        row5.addStretch()
        adv.add_layout(row5)

        # Row 6: TEA Cache
        row6 = QHBoxLayout()
        row6.setSpacing(8)
        row6.addWidget(QLabel("TEA Cache:"))
        self.tea_cache = QComboBox()
        self.tea_cache.setMinimumWidth(self._W_COMBO_S)
        self.tea_cache.addItems(_TEA_CACHE_OPTIONS)
        row6.addWidget(self.tea_cache)
        row6.addWidget(QLabel("Start %:"))
        self.tea_start_perc = QDoubleSpinBox()
        self.tea_start_perc.setMinimumWidth(self._W_DSPIN)
        self.tea_start_perc.setRange(0, 100)
        self.tea_start_perc.setDecimals(1)
        self.tea_start_perc.setValue(0.0)
        row6.addWidget(self.tea_start_perc)
        row6.addStretch()
        adv.add_layout(row6)

        # NAG controls gated behind wan2gp checkbox; TEA Cache always available
        self._nag_controls = [
            self.nag_scale, self.nag_tau, self.nag_alpha,
        ]
        self._wan_only_widgets.append(self.nag_tea_enable)
        self.nag_tea_enable.toggled.connect(self._on_nag_tea_toggled)
        self._on_nag_tea_toggled(False)

        layout.addWidget(adv)

        # Quality (Wan-only — SLG/APG/CFG* are wan2gp-specific)
        qual = CollapsibleSection("Quality", collapsed=True)
        self._quality_section = qual
        self._wan_only_widgets.append(qual)

        self.quality_enable = QCheckBox("Enable Quality Overrides")
        self.quality_enable.setChecked(False)
        qual.add_widget(self.quality_enable)

        q1 = QHBoxLayout()
        q1.setSpacing(8)
        self.slg_switch = QCheckBox("SLG")
        q1.addWidget(self.slg_switch)
        q1.addWidget(QLabel("Layers:"))
        self.slg_layers = QLineEdit()
        self.slg_layers.setMinimumWidth(220)
        self.slg_layers.setPlaceholderText("7,8,9,10,11,12,13,14,15,16,17,18,19")
        q1.addWidget(self.slg_layers, 1)
        qual.add_layout(q1)

        q1b = QHBoxLayout()
        q1b.setSpacing(8)
        q1b.addWidget(QLabel("SLG Start %:"))
        self.slg_start_perc = QSpinBox()
        self.slg_start_perc.setMinimumWidth(self._W_SPIN)
        self.slg_start_perc.setRange(0, 100)
        q1b.addWidget(self.slg_start_perc)
        q1b.addWidget(QLabel("End %:"))
        self.slg_end_perc = QSpinBox()
        self.slg_end_perc.setMinimumWidth(self._W_SPIN)
        self.slg_end_perc.setRange(0, 100)
        self.slg_end_perc.setValue(100)
        q1b.addWidget(self.slg_end_perc)
        self.apg_switch = QCheckBox("APG")
        q1b.addWidget(self.apg_switch)
        q1b.addStretch()
        qual.add_layout(q1b)

        q1c = QHBoxLayout()
        q1c.setSpacing(8)
        q1c.addWidget(QLabel("CFG Star:"))
        self.cfg_star_switch = QSpinBox()
        self.cfg_star_switch.setMinimumWidth(self._W_SPIN)
        self.cfg_star_switch.setRange(0, 1)
        self.cfg_star_switch.setValue(0)
        q1c.addWidget(self.cfg_star_switch)
        q1c.addWidget(QLabel("CFG Zero:"))
        self.cfg_zero_step = QSpinBox()
        self.cfg_zero_step.setMinimumWidth(self._W_SPIN)
        self.cfg_zero_step.setRange(-1, 39)
        self.cfg_zero_step.setValue(-1)
        q1c.addWidget(self.cfg_zero_step)
        q1c.addStretch()
        qual.add_layout(q1c)

        q2 = QHBoxLayout()
        q2.setSpacing(8)
        q2.addWidget(QLabel("Motion:"))
        self.motion_amplitude = QDoubleSpinBox()
        self.motion_amplitude.setMinimumWidth(self._W_DSPIN)
        self.motion_amplitude.setRange(1.0, 1.4)
        self.motion_amplitude.setDecimals(2)
        self.motion_amplitude.setSingleStep(0.01)
        self.motion_amplitude.setValue(1.0)
        q2.addWidget(self.motion_amplitude)
        # Color correction widgets live in their own section below — they
        # don't depend on the wan2gp loop or quality_overrides_enabled, so
        # they shouldn't be gated by them.
        q2.addWidget(QLabel("Discard:"))
        self.discard_last_frames = QSpinBox()
        self.discard_last_frames.setMinimumWidth(self._W_SPIN)
        self.discard_last_frames.setRange(0, 20)
        self.discard_last_frames.setSingleStep(4)
        q2.addWidget(self.discard_last_frames)
        q2.addStretch()
        qual.add_layout(q2)

        # Self Refiner
        qual.add_widget(QLabel("<b>Self Refiner</b>"))

        q3 = QHBoxLayout()
        q3.setSpacing(8)
        q3.addWidget(QLabel("Refiner:"))
        self.self_refiner = QComboBox()
        self.self_refiner.setMinimumWidth(self._W_COMBO_S)
        self.self_refiner.addItems(_SELF_REFINER_OPTIONS)
        q3.addWidget(self.self_refiner)
        q3.addWidget(QLabel("Uncert:"))
        self.refiner_uncertainty = QDoubleSpinBox()
        self.refiner_uncertainty.setMinimumWidth(self._W_DSPIN)
        self.refiner_uncertainty.setRange(0, 10)
        self.refiner_uncertainty.setDecimals(3)
        q3.addWidget(self.refiner_uncertainty)
        q3.addWidget(QLabel("Cert Skip:"))
        self.refiner_certainty_skip = QDoubleSpinBox()
        self.refiner_certainty_skip.setMinimumWidth(self._W_DSPIN)
        self.refiner_certainty_skip.setRange(0, 10)
        self.refiner_certainty_skip.setDecimals(3)
        q3.addWidget(self.refiner_certainty_skip)
        q3.addStretch()
        qual.add_layout(q3)

        # Collect all Quality child controls for enable gating.
        # NOTE: color_correction + color_correction_method are intentionally
        # excluded — they're post-decode operations independent of the
        # wan2gp denoising loop, so they live in their own section.
        self._quality_controls = [
            self.slg_switch, self.slg_layers, self.slg_start_perc, self.slg_end_perc,
            self.apg_switch, self.cfg_star_switch, self.cfg_zero_step,
            self.motion_amplitude, self.discard_last_frames,
            self.self_refiner, self.refiner_uncertainty, self.refiner_certainty_skip,
        ]
        self.quality_enable.toggled.connect(self._on_quality_toggled)
        self._on_quality_toggled(False)

        layout.addWidget(qual)

        # Auto Color Match — post-decode color transfer to a reference frame.
        # Independent of the wan2gp loop / quality_overrides gate. Goal: match
        # generated output to its source frame automatically so the user
        # doesn't need post-hoc timeline grading.
        cmsec = CollapsibleSection("Auto Color Match (to Source)", collapsed=False)
        self._cc_section = cmsec
        cm_row = QHBoxLayout()
        cm_row.setSpacing(8)
        cm_row.addWidget(QLabel("Strength:"))
        self.color_correction = QDoubleSpinBox()
        self.color_correction.setMinimumWidth(self._W_DSPIN)
        self.color_correction.setRange(0, 1)
        self.color_correction.setDecimals(2)
        self.color_correction.setSingleStep(0.05)
        self.color_correction.setValue(0.0)
        self.color_correction.setToolTip(
            "0.0 = off; 1.0 = full match. Recommended 0.4–0.7 for natural look."
        )
        cm_row.addWidget(self.color_correction)
        cm_row.addWidget(QLabel("Method:"))
        self.color_correction_method = QComboBox()
        self.color_correction_method.setMinimumWidth(self._W_COMBO_L)
        for label, key in _CC_METHODS:
            self.color_correction_method.addItem(label, userData=key)
        self.color_correction_method.setToolTip(
            "Mean-only Lab: drift-resistant, additive shift only. Use for chained "
            "continuations / SVI inter-window.\n"
            "HM-MVGD-HM: best general-purpose match — eliminates timeline grading "
            "for most generations.\n"
            "Reinhard / MKL / MVGD / HM: see color-matcher docs."
        )
        cm_row.addWidget(self.color_correction_method)
        cm_row.addStretch()
        cmsec.add_layout(cm_row)

        # Anchor pre-match — PRE-generation color normalization of guidance
        # videos / last frames / keyframes to the source frame. Previously a
        # config-only field (anchor_prematch_strength) that ran silently at
        # 0.8 with no way to disable it from the UI.
        pm_row = QHBoxLayout()
        pm_row.setSpacing(8)
        pm_row.addWidget(QLabel("Pre-match:"))
        self.anchor_prematch = QDoubleSpinBox()
        self.anchor_prematch.setMinimumWidth(self._W_DSPIN)
        self.anchor_prematch.setRange(0, 1)
        self.anchor_prematch.setDecimals(2)
        self.anchor_prematch.setSingleStep(0.05)
        self.anchor_prematch.setValue(0.8)
        self.anchor_prematch.setToolTip(
            "Matches guidance videos, last/final frames, keyframes and window "
            "anchors to the source frame's colors BEFORE generation, so the "
            "model bridges content — not color — between inputs.\n"
            "0 = off. Uses the Method above. Skipped automatically for static "
            "guidance built from the source frame."
        )
        pm_row.addWidget(self.anchor_prematch)
        pm_row.addStretch()
        cmsec.add_layout(pm_row)
        layout.addWidget(cmsec)
        self._wan_only_widgets.append(cmsec)

        # Post-Processing
        pp = CollapsibleSection("Post-Processing", collapsed=True)

        pp_preset_row = QHBoxLayout()
        pp_preset_row.setSpacing(8)
        pp_preset_row.addWidget(QLabel("Preset:"))
        self.pp_preset = QComboBox()
        self.pp_preset.setMinimumWidth(self._W_COMBO_M)
        for label, *_ in _PP_PRESETS:
            self.pp_preset.addItem(label)
        self.pp_preset.currentIndexChanged.connect(self._on_pp_preset_changed)
        pp_preset_row.addWidget(self.pp_preset)
        pp_preset_row.addStretch()
        pp.add_layout(pp_preset_row)

        pp1 = QHBoxLayout()
        pp1.setSpacing(8)
        pp1.addWidget(QLabel("Temporal:"))
        self.temporal_upsampling = QComboBox()
        self.temporal_upsampling.setMinimumWidth(self._W_COMBO_M)
        self.temporal_upsampling.addItems(_TEMPORAL_UPSAMPLING_OPTIONS)
        pp1.addWidget(self.temporal_upsampling)
        pp1.addWidget(QLabel("Spatial:"))
        self.spatial_upsampling = QComboBox()
        self.spatial_upsampling.setMinimumWidth(self._W_COMBO_M)
        self.spatial_upsampling.addItems(_SPATIAL_UPSAMPLING_OPTIONS)
        pp1.addWidget(self.spatial_upsampling)
        pp1.addStretch()
        pp.add_layout(pp1)

        pp2 = QHBoxLayout()
        pp2.setSpacing(8)
        pp2.addWidget(QLabel("Film Grain:"))
        self.film_grain_intensity = QDoubleSpinBox()
        self.film_grain_intensity.setMinimumWidth(self._W_DSPIN)
        self.film_grain_intensity.setRange(0, 1)
        self.film_grain_intensity.setDecimals(2)
        self.film_grain_intensity.setSingleStep(0.05)
        pp2.addWidget(self.film_grain_intensity)
        pp2.addWidget(QLabel("Saturation:"))
        self.film_grain_saturation = QDoubleSpinBox()
        self.film_grain_saturation.setMinimumWidth(self._W_DSPIN)
        self.film_grain_saturation.setRange(0, 1)
        self.film_grain_saturation.setDecimals(2)
        self.film_grain_saturation.setSingleStep(0.05)
        pp2.addWidget(self.film_grain_saturation)
        pp2.addStretch()
        pp.add_layout(pp2)

        layout.addWidget(pp)

        # SVI 2 Pro — Sliding Window + Anchor Image
        # Visible only when an SVI model is selected (toggled in
        # _on_model_type_changed). Wan2GP exposes these as the headline
        # SVI Pro features for infinite-length video.
        # Starts collapsed so the section doesn't eat 170–200px of vertical
        # space for non-SVI models; the first time an SVI model is selected
        # it auto-expands (see _on_model_type_changed), then respects any
        # manual collapse the user makes afterward.
        svi = CollapsibleSection("SVI 2 Pro — Sliding Window", collapsed=True)
        self._svi_section = svi
        svi.setVisible(False)
        # Track whether we've ever auto-expanded this section so a manual
        # collapse by the user persists across model switches.
        self._svi_first_show = True

        sw_row1 = QHBoxLayout()
        sw_row1.setSpacing(8)
        sw_row1.addWidget(QLabel("Window Size:"))
        self.sliding_window_size = QSpinBox()
        self.sliding_window_size.setMinimumWidth(self._W_SPIN)
        # Latent-aligned values per wan2gp: (size-1) % latent_size == 0.
        # 17/25/33/41/49/65/81 are all valid Wan video lengths.
        self.sliding_window_size.setRange(17, 241)
        self.sliding_window_size.setSingleStep(8)
        self.sliding_window_size.setValue(81)
        sw_row1.addWidget(self.sliding_window_size)
        sw_row1.addWidget(QLabel("Overlap:"))
        self.sliding_window_overlap = QSpinBox()
        self.sliding_window_overlap.setMinimumWidth(self._W_SPIN)
        self.sliding_window_overlap.setRange(0, 80)
        self.sliding_window_overlap.setValue(4)
        sw_row1.addWidget(self.sliding_window_overlap)
        sw_row1.addWidget(QLabel("Discard Last:"))
        self.sliding_window_discard_last = QSpinBox()
        self.sliding_window_discard_last.setMinimumWidth(self._W_SPIN)
        self.sliding_window_discard_last.setRange(0, 40)
        self.sliding_window_discard_last.setValue(0)
        sw_row1.addWidget(self.sliding_window_discard_last)
        sw_row1.addStretch()
        svi.add_layout(sw_row1)

        sw_row2 = QHBoxLayout()
        sw_row2.setSpacing(8)
        sw_row2.addWidget(QLabel("Color Corr (per-window):"))
        self.sliding_window_color_correction = QDoubleSpinBox()
        self.sliding_window_color_correction.setMinimumWidth(self._W_DSPIN)
        self.sliding_window_color_correction.setRange(0.0, 1.0)
        self.sliding_window_color_correction.setDecimals(2)
        self.sliding_window_color_correction.setSingleStep(0.05)
        self.sliding_window_color_correction.setValue(0.0)
        sw_row2.addWidget(self.sliding_window_color_correction)
        sw_row2.addWidget(QLabel("Overlap Noise:"))
        self.sliding_window_overlap_noise = QDoubleSpinBox()
        self.sliding_window_overlap_noise.setMinimumWidth(self._W_DSPIN)
        self.sliding_window_overlap_noise.setRange(0.0, 1.0)
        self.sliding_window_overlap_noise.setDecimals(2)
        self.sliding_window_overlap_noise.setSingleStep(0.05)
        self.sliding_window_overlap_noise.setValue(0.0)
        sw_row2.addWidget(self.sliding_window_overlap_noise)
        sw_row2.addStretch()
        svi.add_layout(sw_row2)

        anchor_row = QHBoxLayout()
        anchor_row.setSpacing(8)
        anchor_row.addWidget(QLabel("Anchor:"))
        self.anchor_image_mode = QComboBox()
        self.anchor_image_mode.setMinimumWidth(self._W_COMBO_M)
        for label, _ in _ANCHOR_MODES:
            self.anchor_image_mode.addItem(label)
        anchor_row.addWidget(self.anchor_image_mode)
        self.anchor_image_path = QLineEdit()
        self.anchor_image_path.setPlaceholderText("(image path — single mode) or (semicolon-separated paths — per-window)")
        anchor_row.addWidget(self.anchor_image_path, 1)
        anchor_browse = QPushButton("Browse…")
        anchor_browse.setMinimumWidth(80)
        anchor_browse.clicked.connect(self._on_anchor_browse)
        anchor_row.addWidget(anchor_browse)
        svi.add_layout(anchor_row)

        self.anchor_image_mode.currentIndexChanged.connect(self._on_anchor_mode_changed)
        self._on_anchor_mode_changed(0)

        # Place SVI section RIGHT AFTER Model Timing (index 1) — this is
        # the most-used SVI control and shouldn't be buried below Quality,
        # Post-Processing, etc. When the user picks an SVI model from the
        # dropdown at the top, the relevant controls are immediately below.
        layout.insertWidget(1, svi)
        # NOTE: not added to _wan_only_widgets — visibility is managed
        # purely by _on_model_type_changed based on whether an SVI model
        # is selected. (SVI implies wan backend, so LTX selection naturally
        # hides it via the model-type check.)

        # LoRA
        if self._show_lora:
            # ── LTX System LoRAs (one CollapsibleSection per category) ──
            from sdqt.models.ltx_system_loras import LTX_SYSTEM_LORAS
            self._ltx_system_sections: list = []
            self._ltx_system_checkboxes: dict[str, QCheckBox] = {}  # prefix → QCheckBox
            by_cat: dict[str, list] = {}
            for cat, prefix, disp, desc in LTX_SYSTEM_LORAS:
                by_cat.setdefault(cat, []).append((prefix, disp, desc))
            for cat, items in by_cat.items():
                sec = CollapsibleSection(f"LTX System LoRAs — {cat}", collapsed=True)
                for prefix, disp, desc in items:
                    cb = QCheckBox(disp)
                    cb.setToolTip(desc)
                    cb.toggled.connect(self._on_lora_item_changed)
                    sec.add_widget(cb)
                    self._ltx_system_checkboxes[prefix] = cb
                layout.addWidget(sec)
                self._ltx_system_sections.append(sec)

            lora = CollapsibleSection("LoRA Controls", collapsed=True)
            self.lora_list = QListWidget()
            # Show ~6 rows before scrolling instead of ~2 (was capped 120px).
            self.lora_list.setMinimumHeight(180)
            self.lora_list.setMaximumHeight(220)
            self.lora_list.itemChanged.connect(self._on_lora_item_changed)
            lora.add_widget(self.lora_list)

            # Structured per-LoRA strength view. Rebuilt whenever the activated
            # LoRA set changes; each active LoRA gets a name label + strength
            # QDoubleSpinBox (and a second spinbox + phase label when the entry
            # uses wan2gp phase syntax "hi;lo"). Edits re-emit the aligned
            # loras_multipliers string in activated-LoRA order — the alignment
            # contract collect_to_config / the backend rely on.
            lora.add_widget(QLabel("<b>Per-LoRA Strength</b>"))
            self.lora_strength_container = QWidget()
            self._lora_strength_layout = QVBoxLayout(self.lora_strength_container)
            self._lora_strength_layout.setContentsMargins(0, 0, 0, 0)
            self._lora_strength_layout.setSpacing(6)
            self._lora_strength_empty = QLabel("<i>No LoRAs activated</i>")
            self._lora_strength_layout.addWidget(self._lora_strength_empty)
            lora.add_widget(self.lora_strength_container)
            # Per-active-LoRA spinbox rows, aligned to the activated order.
            # Each entry: {"name", "hi" (QDoubleSpinBox), "lo" (QDoubleSpinBox|None)}
            self._lora_strength_rows: list[dict] = []
            # Guard so programmatic spinbox/text updates don't recurse.
            self._lora_sync_guard = False

            lr = QHBoxLayout()
            lr.setSpacing(8)
            lr.addWidget(QLabel("Multipliers:"))
            self.lora_multipliers = QLineEdit()
            # Stretch so long phase strings ("1;0 0;1 0.7;0 ...") don't clip.
            self.lora_multipliers.setMinimumWidth(280)
            self.lora_multipliers.setPlaceholderText('"1.0" or "1;0 0;1"')
            self.lora_multipliers.setToolTip(
                "Raw wan2gp multiplier string (space-separated, one entry per "
                "activated LoRA; \";\" separates phase weights). Kept in sync with "
                "the Per-LoRA Strength spinboxes above."
            )
            self.lora_multipliers.editingFinished.connect(self._on_multipliers_text_edited)
            lr.addWidget(self.lora_multipliers, 1)
            lora.add_layout(lr)

            refresh = QPushButton("Refresh LoRAs")
            refresh.clicked.connect(lambda: self.refresh_loras_signal())
            lora.add_widget(refresh)

            layout.addWidget(lora)
        else:
            self.lora_list = None
            self.lora_multipliers = None
            self._ltx_system_sections = []
            self._ltx_system_checkboxes = {}
            self._lora_strength_rows = []
            self._lora_sync_guard = False

    def _on_quality_toggled(self, enabled: bool) -> None:
        allow = enabled and self._loop_is_wan2gp
        for ctrl in self._quality_controls:
            ctrl.setEnabled(allow)

    def _on_nag_tea_toggled(self, enabled: bool) -> None:
        allow = enabled and self._loop_is_wan2gp
        for ctrl in self._nag_controls:
            ctrl.setEnabled(allow)

    def _on_model_type_changed(self, _index: int) -> None:
        model_key = self.model_type.currentData() or ""
        backend = get_video_backend(model_key)
        if backend != self._current_backend:
            self._reconfigure_for_backend(backend)
        defaults = MODEL_DEFAULTS.get(model_key)
        if defaults:
            self.steps.setValue(defaults["num_inference_steps"])
            self.guidance_scale.setValue(defaults["guidance_scale"])
            self.guidance2_scale.setValue(defaults["guidance2_scale"])
            self.flow_shift.setValue(defaults["flow_shift"])
            self.guidance_phases.setValue(defaults["guidance_phases"])
            self.switch_threshold.setValue(defaults["switch_threshold"])
        # Show the SVI section only when an SVI model is active. Auto-expand
        # the very first time it's revealed so the controls are immediately
        # visible — but respect the user's collapse preference on subsequent
        # SVI re-selections.
        is_svi = model_key in SVI_MODEL_KEYS
        if hasattr(self, "_svi_section"):
            show = is_svi and backend == "wan"
            if show and getattr(self, "_svi_first_show", True):
                self._svi_section.set_collapsed(False)
                self._svi_first_show = False
            self._svi_section.setVisible(show)
        # Auto-prefill the SVI 2 Pro LoRAs (and wan2gp phase multipliers) when
        # the LoRA-based variant is selected and no LoRAs are already chosen.
        if model_key == "i2v_2_2_svi_2_pro" and self.lora_list is not None:
            already_active = [
                self.lora_list.item(i).text()
                for i in range(self.lora_list.count())
                if self.lora_list.item(i).checkState() == Qt.Checked
            ]
            if not any(name.startswith("SVI_Wan2.2-I2V-A14B") for name in already_active):
                # Trigger a refresh so the SVI files appear, then check them.
                self.refresh_loras_signal()
                self._select_loras_by_prefix(SVI_2_PRO_LORAS)
                if self.lora_multipliers is not None and not self.lora_multipliers.text().strip():
                    self.lora_multipliers.setText(SVI_2_PRO_LORA_MULTIPLIERS)
                self._rebuild_lora_strength_view()

        # Auto-prefill the standalone Lightning v2 LoRAs + per-phase weight
        # multipliers ("0.7;0 0;1.0") when that LoRA-stack model variant is
        # selected and the user hasn't already configured Lightning LoRAs.
        if model_key == "i2v_2_2_lightning_v2_loras" and self.lora_list is not None:
            already_active = [
                self.lora_list.item(i).text()
                for i in range(self.lora_list.count())
                if self.lora_list.item(i).checkState() == Qt.Checked
            ]
            if not any(
                name.startswith("Wan2.2-Lightning_I2V") for name in already_active
            ):
                self.refresh_loras_signal()
                self._select_loras_by_prefix(LIGHTNING_V2_I2V_LORAS)
                if (
                    self.lora_multipliers is not None
                    and not self.lora_multipliers.text().strip()
                ):
                    self.lora_multipliers.setText(LIGHTNING_V2_I2V_LORA_MULTIPLIERS)
                self._rebuild_lora_strength_view()

    def _select_loras_by_prefix(self, stems: list[str]) -> None:
        """Check LoRA list items whose filename matches any of the provided stems.

        Used by SVI 2 Pro auto-prefill — matches by stem so it works regardless
        of whether the LoRA list shows full filenames or stems.
        """
        if self.lora_list is None:
            return
        wanted = set(stems)
        # Block itemChanged so the auto-emit doesn't normalize (and thus
        # populate) the multiplier string before the caller's prefill text
        # has a chance to apply — the caller rebuilds the view afterward.
        self.lora_list.blockSignals(True)
        try:
            for i in range(self.lora_list.count()):
                item = self.lora_list.item(i)
                text = item.text()
                stem = text.rsplit(".", 1)[0]
                if stem in wanted or text in wanted:
                    item.setCheckState(Qt.Checked)
        finally:
            self.lora_list.blockSignals(False)

    def _on_anchor_mode_changed(self, idx: int) -> None:
        # "Use start frame as ref" hides the path field — no anchor needed.
        if idx < 0 or idx >= len(_ANCHOR_MODES):
            return
        mode_key = _ANCHOR_MODES[idx][1]
        needs_path = mode_key in ("single", "per_window", "per_window_keyframes")
        self.anchor_image_path.setEnabled(needs_path)
        if mode_key == "per_window":
            self.anchor_image_path.setPlaceholderText(
                "Semicolon-separated image paths, one per window (hard cuts at start of each)"
            )
        elif mode_key == "per_window_keyframes":
            self.anchor_image_path.setPlaceholderText(
                "Semicolon-separated keyframes — window N converges to keyframe[N] at its end"
            )
        else:
            self.anchor_image_path.setPlaceholderText("Path to anchor image")

    def _on_anchor_browse(self) -> None:
        idx = self.anchor_image_mode.currentIndex()
        mode_key = _ANCHOR_MODES[idx][1] if 0 <= idx < len(_ANCHOR_MODES) else "start"
        if mode_key in ("per_window", "per_window_keyframes"):
            label = (
                "Select keyframes (one per window — converged to at end)"
                if mode_key == "per_window_keyframes"
                else "Select anchor images (one per window)"
            )
            paths, _ = QFileDialog.getOpenFileNames(
                self, label, "", "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if paths:
                self.anchor_image_path.setText(";".join(paths))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select anchor image",
                "", "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if path:
                self.anchor_image_path.setText(path)

    def _reconfigure_for_backend(self, backend: str) -> None:
        """Swap resolution/duration presets and show/hide backend-specific controls."""
        self._current_backend = backend
        is_ltx = backend == "ltx"

        # Swap resolution presets
        current_res = self.resolution.currentText().split(" ")[0]
        presets = _LTX_RESOLUTION_PRESETS if is_ltx else _WAN_RESOLUTION_PRESETS
        self.resolution.blockSignals(True)
        self.resolution.clear()
        self.resolution.addItems(presets)
        # Try to restore previous resolution if it exists in new presets
        for i, p in enumerate(presets):
            if p.startswith(current_res):
                self.resolution.setCurrentIndex(i)
                break
        self.resolution.blockSignals(False)

        # Swap duration presets — simply repopulate the dropdown. The "Custom"
        # entry always lives at the end, preserving the spinbox state.
        dur_presets = _LTX_DURATION_PRESETS if is_ltx else _WAN_DURATION_PRESETS
        # Keep the active list in sync so duration lookups index correctly.
        self._active_duration_presets = list(dur_presets)
        prev_custom = self.duration_combo.currentText() == "Custom"
        self.duration_combo.blockSignals(True)
        self.duration_combo.clear()
        for label, _ in dur_presets:
            self.duration_combo.addItem(label)
        self.duration_combo.addItem("Custom")
        self.duration_combo.setCurrentIndex(
            len(dur_presets) if prev_custom else 0
        )
        self.duration_combo.blockSignals(False)
        # Re-snap the custom spinbox step to the new backend's latent alignment.
        self._duration_custom_frames.setSingleStep(8 if is_ltx else 4)
        self._duration_custom_frames.setEnabled(prev_custom)

        # FPS range
        if is_ltx:
            self.fps.setRange(1, 50)
            self.fps.setValue(30)
        else:
            self.fps.setRange(1, 60)
            self.fps.setValue(16)

        # Guidance scale step size
        self.guidance_scale.setSingleStep(0.5 if is_ltx else 0.1)

        # Show/hide Wan-only widgets
        for w in self._wan_only_widgets:
            w.setVisible(not is_ltx)
        for w in getattr(self, "_ltx_only_widgets", []):
            w.setVisible(is_ltx)
        for sec in getattr(self, "_ltx_system_sections", []):
            sec.setVisible(is_ltx)

    def _current_duration_frames(self) -> int:
        """Return the active duration in frames, honoring the Custom option."""
        presets = getattr(self, "_active_duration_presets", _DURATION_PRESETS)
        dur_idx = self.duration_combo.currentIndex()
        if dur_idx < 0:
            return 81
        if 0 <= dur_idx < len(presets):
            return presets[dur_idx][1]
        # Custom — snap to backend latent alignment (Wan: 4n+1, LTX: 8n+1)
        raw = self._duration_custom_frames.value()
        step = self._duration_custom_frames.singleStep() or 4
        n = max(round((raw - 1) / step), 1)
        return n * step + 1

    def _on_duration_changed(self, _idx: int) -> None:
        """Enable the Custom spinbox only while 'Custom' is selected."""
        is_custom = self.duration_combo.currentText() == "Custom"
        if hasattr(self, "_duration_custom_frames"):
            self._duration_custom_frames.setEnabled(is_custom)

    def _on_pp_preset_changed(self, idx: int) -> None:
        if idx <= 0 or idx >= len(_PP_PRESETS):
            return
        _, temporal, spatial, grain_i, grain_s = _PP_PRESETS[idx]
        ti = self.temporal_upsampling.findText(temporal)
        if ti >= 0:
            self.temporal_upsampling.setCurrentIndex(ti)
        si = self.spatial_upsampling.findText(spatial)
        if si >= 0:
            self.spatial_upsampling.setCurrentIndex(si)
        self.film_grain_intensity.setValue(grain_i)
        self.film_grain_saturation.setValue(grain_s)
        # Reset dropdown to placeholder so it can be re-selected
        self.pp_preset.blockSignals(True)
        self.pp_preset.setCurrentIndex(0)
        self.pp_preset.blockSignals(False)

    def set_denoising_loop(self, loop: str) -> None:
        """Hard-gate wan2gp-only controls based on global denoising loop setting."""
        self._loop_is_wan2gp = (loop == "wan2gp")
        # Enable/disable the gate checkboxes themselves
        self.quality_enable.setEnabled(self._loop_is_wan2gp)
        self.nag_tea_enable.setEnabled(self._loop_is_wan2gp)
        # Re-apply child gating
        self._on_quality_toggled(self.quality_enable.isChecked())
        self._on_nag_tea_toggled(self.nag_tea_enable.isChecked())

    def set_lora_refresh_callback(self, callback) -> None:
        """Set a callback that returns list[str] of available LoRA names."""
        self._lora_refresh_cb = callback

    def refresh_loras_signal(self) -> None:
        """Refresh the LoRA list using the registered callback."""
        cb = getattr(self, "_lora_refresh_cb", None)
        if cb is None:
            return
        try:
            available = cb()
        except Exception:
            available = []
        if self.lora_list is not None:
            # Preserve current selections
            activated = []
            for i in range(self.lora_list.count()):
                item = self.lora_list.item(i)
                if item.checkState() == Qt.Checked:
                    activated.append(item.text())
            self.populate_loras(available, activated)

    def collect_to_config(self, cfg: ProjectConfig) -> None:
        """Write all param values into cfg."""
        cfg.model_type = self.model_type.currentData() or self.model_type.currentText()
        # Store just "WxH" — strip the aspect ratio label e.g. "832x480 (16:9)" → "832x480"
        cfg.resolution = self.resolution.currentText().split(" ")[0]
        cfg.video_length = self._current_duration_frames()
        cfg.fps = self.fps.value()
        cfg.num_inference_steps = self.steps.value()
        cfg.guidance_scale = self.guidance_scale.value()
        cfg.seed = self.seed.value()
        cfg.guidance2_scale = self.guidance2_scale.value()
        cfg.alt_guidance_scale = self.alt_guidance_scale.value()
        # Discard last frames is written below, *after* the Quality gate, so
        # the Wan Quality-gate `else` branch can't clobber the LTX value back
        # to 0. The two controls (LTX spinbox vs the Wan Quality spinbox) are
        # reconciled there keyed on the active backend.
        cfg.flow_shift = self.flow_shift.value()
        cfg.sample_solver = self.solver.currentText()
        cfg.denoising_strength = self.denoising_strength.value()
        cfg.guidance_phases = self.guidance_phases.value()
        cfg.switch_threshold = self.switch_threshold.value()
        # TEA Cache — always saved (independent speed optimization)
        cfg.tea_cache_setting = self.tea_cache.currentText()
        cfg.tea_cache_start_step_perc = self.tea_start_perc.value()
        # NAG gate — requires wan2gp denoising loop
        cfg.nag_tea_enabled = self.nag_tea_enable.isChecked()
        if cfg.nag_tea_enabled:
            cfg.NAG_scale = self.nag_scale.value()
            cfg.NAG_tau = self.nag_tau.value()
            cfg.NAG_alpha = self.nag_alpha.value()
        else:
            cfg.NAG_scale = 0.0
            cfg.NAG_tau = 0.0
            cfg.NAG_alpha = 0.0

        # Auto Color Match (always saved — independent of the quality gate)
        cfg.color_correction_strength = self.color_correction.value()
        cfg.color_correction_method = (
            self.color_correction_method.currentData()
            or self.color_correction_method.currentText()
        )
        cfg.anchor_prematch_strength = self.anchor_prematch.value()

        # Quality gate
        cfg.quality_overrides_enabled = self.quality_enable.isChecked()
        if cfg.quality_overrides_enabled:
            cfg.cfg_star_switch = self.cfg_star_switch.value()
            cfg.cfg_zero_step = self.cfg_zero_step.value()
            cfg.slg_switch = 1 if self.slg_switch.isChecked() else 0
            layers_text = self.slg_layers.text().strip()
            if layers_text:
                try:
                    cfg.slg_layers = [int(x.strip()) for x in layers_text.split(",") if x.strip()]
                except ValueError:
                    pass
            cfg.slg_start_perc = self.slg_start_perc.value()
            cfg.slg_end_perc = self.slg_end_perc.value()
            cfg.apg_switch = 1 if self.apg_switch.isChecked() else 0
            cfg.motion_amplitude = self.motion_amplitude.value()
            cfg.self_refiner_setting = self.self_refiner.currentIndex()
            cfg.self_refiner_uncertainty = self.refiner_uncertainty.value()
            cfg.self_refiner_certainty_skip = self.refiner_certainty_skip.value()
        else:
            cfg.slg_switch = 0
            cfg.slg_layers = []
            cfg.slg_start_perc = 0
            cfg.slg_end_perc = 100
            cfg.apg_switch = 0
            cfg.cfg_star_switch = 0
            cfg.cfg_zero_step = 0
            cfg.motion_amplitude = 1.0
            cfg.self_refiner_setting = 0
            cfg.self_refiner_uncertainty = 0.0
            cfg.self_refiner_certainty_skip = 0.999

        # Discard last frames — resolved after the Quality gate so the gate's
        # `else` (Wan-only) can't reset the LTX distilled tail-trim to 0. For
        # LTX read the dedicated spinbox; for Wan read the Quality-section
        # spinbox only when overrides are enabled, otherwise 0.
        if self._current_backend == "ltx":
            cfg.discard_last_frames = int(self.ltx_discard_last.value())
        elif cfg.quality_overrides_enabled:
            cfg.discard_last_frames = self.discard_last_frames.value()
        else:
            cfg.discard_last_frames = 0

        cfg.temporal_upsampling = self.temporal_upsampling.currentText()
        cfg.spatial_upsampling = self.spatial_upsampling.currentText()
        cfg.film_grain_intensity = self.film_grain_intensity.value()
        cfg.film_grain_saturation = self.film_grain_saturation.value()

        # SVI 2 Pro sliding window + anchor
        cfg.sliding_window_size = self.sliding_window_size.value()
        cfg.sliding_window_overlap = self.sliding_window_overlap.value()
        cfg.sliding_window_discard_last_frames = self.sliding_window_discard_last.value()
        cfg.sliding_window_color_correction_strength = self.sliding_window_color_correction.value()
        cfg.sliding_window_overlap_noise = self.sliding_window_overlap_noise.value()
        idx = self.anchor_image_mode.currentIndex()
        cfg.anchor_image_mode = _ANCHOR_MODES[idx][1] if 0 <= idx < len(_ANCHOR_MODES) else "start"
        cfg.anchor_image_path = self.anchor_image_path.text().strip()

        if self.lora_list is not None:
            cfg.activated_loras = self._collect_activated_loras()
        if self.lora_multipliers is not None:
            cfg.loras_multipliers = self.lora_multipliers.text()

    def restore_from_config(self, cfg: ProjectConfig) -> None:
        """Set all controls from cfg."""
        # Match by internal key (userData) first, fall back to display text
        model_key = cfg.model_type or ""
        idx = self.model_type.findData(model_key)
        if idx < 0:
            idx = self.model_type.findText(model_key)
        if idx >= 0:
            self.model_type.setCurrentIndex(idx)
        # Match stored "WxH" against combo items like "WxH (ratio)"
        res = cfg.resolution or ""
        idx = self.resolution.findText(res)
        if idx < 0:
            # Try prefix match (stored "832x480" matches "832x480 (16:9)")
            for i in range(self.resolution.count()):
                if self.resolution.itemText(i).startswith(res):
                    idx = i
                    break
        if idx >= 0:
            self.resolution.setCurrentIndex(idx)
        matched = False
        presets = getattr(self, "_active_duration_presets", _DURATION_PRESETS)
        for i, (_, frames) in enumerate(presets):
            if frames == cfg.video_length:
                self.duration_combo.setCurrentIndex(i)
                matched = True
                break
        if not matched and cfg.video_length:
            self._duration_custom_frames.setValue(int(cfg.video_length))
            # "Custom" is the last entry in the dropdown.
            self.duration_combo.setCurrentIndex(self.duration_combo.count() - 1)
        self.fps.setValue(cfg.fps or 16)
        self.steps.setValue(cfg.num_inference_steps)
        self.guidance_scale.setValue(cfg.guidance_scale)
        self.seed.setValue(cfg.seed)
        self.guidance2_scale.setValue(cfg.guidance2_scale)
        self.alt_guidance_scale.setValue(getattr(cfg, "alt_guidance_scale", 1.0) or 1.0)
        # Mirror the same value into both Discard controls (LTX-only + Wan
        # Quality) — only one is visible at a time but both stay in sync.
        self.ltx_discard_last.setValue(int(getattr(cfg, "discard_last_frames", 0) or 0))
        self.flow_shift.setValue(cfg.flow_shift)
        idx = self.solver.findText(cfg.sample_solver or "unipc")
        if idx >= 0:
            self.solver.setCurrentIndex(idx)
        self.denoising_strength.setValue(cfg.denoising_strength)
        self.guidance_phases.setValue(cfg.guidance_phases)
        self.switch_threshold.setValue(cfg.switch_threshold)
        # NAG / TEA gate
        self.nag_tea_enable.setChecked(getattr(cfg, "nag_tea_enabled", False))
        self.nag_scale.setValue(cfg.NAG_scale)
        self.nag_tau.setValue(cfg.NAG_tau)
        self.nag_alpha.setValue(cfg.NAG_alpha)
        idx = self.tea_cache.findText(cfg.tea_cache_setting or "off")
        if idx >= 0:
            self.tea_cache.setCurrentIndex(idx)
        self.tea_start_perc.setValue(cfg.tea_cache_start_step_perc)

        # Quality gate
        self.quality_enable.setChecked(getattr(cfg, "quality_overrides_enabled", False))
        self.cfg_star_switch.setValue(cfg.cfg_star_switch)
        self.cfg_zero_step.setValue(cfg.cfg_zero_step)
        self.slg_switch.setChecked(bool(cfg.slg_switch))
        if cfg.slg_layers:
            self.slg_layers.setText(",".join(str(x) for x in cfg.slg_layers))
        self.slg_start_perc.setValue(cfg.slg_start_perc)
        self.slg_end_perc.setValue(cfg.slg_end_perc)
        self.apg_switch.setChecked(bool(cfg.apg_switch))
        self.motion_amplitude.setValue(cfg.motion_amplitude)
        self.color_correction.setValue(cfg.color_correction_strength)
        cc_method = getattr(cfg, "color_correction_method", "mean-only-lab") or "mean-only-lab"
        idx = self.color_correction_method.findData(cc_method)
        if idx < 0:
            idx = 0
        self.color_correction_method.setCurrentIndex(idx)
        self.anchor_prematch.setValue(
            float(getattr(cfg, "anchor_prematch_strength", 0.8))
        )
        self.discard_last_frames.setValue(cfg.discard_last_frames)
        self.self_refiner.setCurrentIndex(cfg.self_refiner_setting)
        self.refiner_uncertainty.setValue(cfg.self_refiner_uncertainty)
        self.refiner_certainty_skip.setValue(cfg.self_refiner_certainty_skip)

        idx = self.temporal_upsampling.findText(cfg.temporal_upsampling or "Disabled")
        if idx >= 0:
            self.temporal_upsampling.setCurrentIndex(idx)
        idx = self.spatial_upsampling.findText(cfg.spatial_upsampling or "Disabled")
        if idx >= 0:
            self.spatial_upsampling.setCurrentIndex(idx)
        self.film_grain_intensity.setValue(cfg.film_grain_intensity)
        self.film_grain_saturation.setValue(cfg.film_grain_saturation)

        # SVI 2 Pro
        self.sliding_window_size.setValue(getattr(cfg, "sliding_window_size", 81) or 81)
        self.sliding_window_overlap.setValue(getattr(cfg, "sliding_window_overlap", 4) or 4)
        self.sliding_window_discard_last.setValue(
            getattr(cfg, "sliding_window_discard_last_frames", 0) or 0
        )
        self.sliding_window_color_correction.setValue(
            getattr(cfg, "sliding_window_color_correction_strength", 0.0) or 0.0
        )
        self.sliding_window_overlap_noise.setValue(
            getattr(cfg, "sliding_window_overlap_noise", 0.0) or 0.0
        )
        anchor_mode = getattr(cfg, "anchor_image_mode", "start") or "start"
        for i, (_, key) in enumerate(_ANCHOR_MODES):
            if key == anchor_mode:
                self.anchor_image_mode.setCurrentIndex(i)
                break
        self.anchor_image_path.setText(getattr(cfg, "anchor_image_path", "") or "")

        if self.lora_multipliers is not None:
            self.lora_multipliers.setText(cfg.loras_multipliers or "")
            # Reseed the structured strength spinboxes from the restored string.
            # If the list isn't populated yet, populate_loras (called by the
            # host after refresh) will rebuild again with the same seed.
            self._rebuild_lora_strength_view()

    def populate_loras(self, available: list[str], activated: list[str] | None = None) -> None:
        """Populate LoRA list with available LoRAs.

        LTX system LoRAs (matched by registered prefix) are pulled out of the
        flat list and surfaced in their dedicated checkbox sections instead.
        Their on-disk filenames are remembered so collect_to_config can write
        them back into ``activated_loras`` if checked.
        """
        from sdqt.models.ltx_system_loras import match_system_lora
        if self.lora_list is None:
            return

        # Tracker: prefix → on-disk filename (so collect_to_config can write
        # the actual filename back when the user checks a system box).
        self._ltx_system_files: dict[str, str] = {}
        user_loras: list[str] = []
        for name in available:
            entry = match_system_lora(name)
            if entry is not None:
                self._ltx_system_files[entry[1]] = name
            else:
                user_loras.append(name)

        # Reflect activation state on system checkboxes.
        if hasattr(self, "_ltx_system_checkboxes"):
            for prefix, cb in self._ltx_system_checkboxes.items():
                fname = self._ltx_system_files.get(prefix)
                cb.setEnabled(fname is not None)
                if fname is None:
                    cb.setToolTip(cb.toolTip().split("\n")[0] +
                                  "\n(file not installed in ltx_lora_dir)")
                cb.setChecked(bool(activated and fname and fname in activated))

        # Populate the regular list with user (non-system) LoRAs only.
        # Block itemChanged while populating so we rebuild the strength view
        # exactly once at the end (and don't clobber the seed multiplier text).
        self.lora_list.blockSignals(True)
        self.lora_list.clear()
        for name in user_loras:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if activated and name in activated:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            self.lora_list.addItem(item)
        self.lora_list.blockSignals(False)
        self._rebuild_lora_strength_view()

    def _collect_activated_loras(self) -> list[str]:
        """Combine checked entries from the user list + system checkboxes."""
        names: list[str] = []
        if self.lora_list is not None:
            for i in range(self.lora_list.count()):
                item = self.lora_list.item(i)
                if item.checkState() == Qt.Checked:
                    names.append(item.text())
        sys_files = getattr(self, "_ltx_system_files", {})
        for prefix, cb in getattr(self, "_ltx_system_checkboxes", {}).items():
            if cb.isChecked() and prefix in sys_files:
                fname = sys_files[prefix]
                if fname not in names:
                    names.append(fname)
        return names

    # ── Structured per-LoRA strength view ───────────────────────────────
    #
    # The QLineEdit (self.lora_multipliers) stays the canonical persisted
    # field (collect_to_config writes cfg.loras_multipliers from it). The
    # spinbox rows are a structured editor layered on top: they parse the
    # string to seed values and re-emit it in activated-LoRA order on change.

    @staticmethod
    def _parse_multiplier_entries(text: str) -> list[str]:
        """Split a wan2gp multiplier string into per-LoRA entries.

        Space-separated entries are one-per-LoRA (each may carry ";"-separated
        phase values). Falls back to comma-separated when there are no spaces
        and no phase markers (simple "1.0,0.8" form). Phase strings are kept
        intact so the spinbox seeder can split them per phase.
        """
        s = (text or "").strip()
        if not s:
            return []
        entries = s.split()
        if len(entries) <= 1 and "," in s and ";" not in s:
            entries = [e.strip() for e in s.split(",") if e.strip()]
        return entries

    @staticmethod
    def _entry_phase_values(entry: str) -> list[float]:
        """Parse one multiplier entry into its phase float(s)."""
        vals: list[float] = []
        for p in entry.split(";"):
            p = p.strip()
            if p == "":
                continue
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(1.0)
        return vals or [1.0]

    def _clear_lora_strength_rows(self) -> None:
        """Remove every spinbox row from the strength container."""
        layout = getattr(self, "_lora_strength_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._lora_strength_rows = []
        self._lora_strength_empty = None

    def _make_strength_spinbox(self, value: float) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setMinimumWidth(self._W_DSPIN)
        sb.setRange(0.0, 4.0)
        sb.setDecimals(2)
        sb.setSingleStep(0.05)
        sb.setValue(value)
        sb.valueChanged.connect(self._on_strength_spin_changed)
        return sb

    def _rebuild_lora_strength_view(self) -> None:
        """Rebuild the per-LoRA spinbox rows aligned to the activated LoRA set.

        Seeds each row's value(s) from the current multiplier string (parsed
        positionally — entry i seeds activated LoRA i, matching the backend's
        parse_phase_multipliers alignment). A LoRA gets a second spinbox + a
        "phase 2" label when its seed entry carries phase syntax ("hi;lo").
        """
        if getattr(self, "lora_list", None) is None:
            return
        if not hasattr(self, "_lora_strength_layout"):
            return
        self._clear_lora_strength_rows()
        active = self._collect_activated_loras()
        if not active:
            self._lora_strength_empty = QLabel("<i>No LoRAs activated</i>")
            self._lora_strength_layout.addWidget(self._lora_strength_empty)
            return

        entries = self._parse_multiplier_entries(self.lora_multipliers.text())
        rows: list[dict] = []
        for i, name in enumerate(active):
            entry = entries[i] if i < len(entries) else "1.0"
            phase_vals = self._entry_phase_values(entry)
            has_phase = ";" in entry

            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(name)
            label.setToolTip(name)
            # Keep long filenames from blowing out the row width, but give them
            # room (was 220px — clipped most LoRA filenames mid-name).
            label.setMinimumWidth(280)
            label.setMaximumWidth(320)
            row.addWidget(label)
            hi = self._make_strength_spinbox(phase_vals[0])
            row.addWidget(hi)
            lo = None
            if has_phase:
                ph = QLabel("phase 2:")
                row.addWidget(ph)
                lo_val = phase_vals[1] if len(phase_vals) > 1 else phase_vals[0]
                lo = self._make_strength_spinbox(lo_val)
                row.addWidget(lo)
            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self._lora_strength_layout.addWidget(container)
            rows.append({"name": name, "hi": hi, "lo": lo})
        self._lora_strength_rows = rows
        # Normalize the canonical string to the rebuilt rows so it always
        # matches the activated order/length (padding/trimming applied).
        self._emit_lora_multipliers()

    def _emit_lora_multipliers(self) -> None:
        """Assemble the multiplier string from the spinbox rows, in order.

        Writes the aligned string into the canonical QLineEdit. One entry per
        activated LoRA, space-separated; phased rows emit "hi;lo".
        """
        if not getattr(self, "_lora_strength_rows", None):
            return
        if self.lora_multipliers is None:
            return
        parts: list[str] = []
        for row in self._lora_strength_rows:
            hi = row["hi"].value()
            lo = row["lo"]
            if lo is not None:
                parts.append(f"{hi:g};{lo.value():g}")
            else:
                parts.append(f"{hi:g}")
        text = " ".join(parts)
        self._lora_sync_guard = True
        try:
            self.lora_multipliers.setText(text)
        finally:
            self._lora_sync_guard = False

    def _on_strength_spin_changed(self, _value: float) -> None:
        if getattr(self, "_lora_sync_guard", False):
            return
        self._emit_lora_multipliers()

    def _on_lora_item_changed(self, *_args) -> None:
        """A LoRA was (un)checked — rebuild the structured strength view.

        The activated set changed, so the spinbox rows must realign. Existing
        spinbox values are preserved via the canonical multiplier string, which
        the rebuild re-parses positionally.
        """
        if getattr(self, "_lora_sync_guard", False):
            return
        self._rebuild_lora_strength_view()

    def _on_multipliers_text_edited(self) -> None:
        """User edited the raw multiplier string — reseed the spinboxes."""
        if getattr(self, "_lora_sync_guard", False):
            return
        self._rebuild_lora_strength_view()
