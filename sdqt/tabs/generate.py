"""Generate tab -- Mode 1 (single image), Mode 2 (first/last frame), Mode 3 (video clip)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.video_drop import VideoDropWidget
from sdqt.widgets.guidance_fps_check import GuidanceFpsCheck
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.widgets.prompt_enhance import PromptEnhanceWidget
from sdqt.workers.inference import InferenceWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

from sdqt.widgets.generation_params import (
    _RESOLUTION_PRESETS,
    _DURATION_PRESETS,
    _MODEL_TYPES,
    _WAN_RESOLUTION_PRESETS,
    _LTX_RESOLUTION_PRESETS,
    _WAN_DURATION_PRESETS,
    _LTX_DURATION_PRESETS,
    _ANCHOR_MODES,
    _CC_METHODS,
    SVI_MODEL_KEYS,
    SVI_2_PRO_LORAS,
    SVI_2_PRO_LORA_MULTIPLIERS,
    LIGHTNING_V2_I2V_LORAS,
    LIGHTNING_V2_I2V_LORA_MULTIPLIERS,
    get_video_backend,
)

_TEA_CACHE_OPTIONS = ["off", "light", "medium", "heavy"]

_SELF_REFINER_OPTIONS = ["Disabled", "P1-Norm", "P2-Norm"]

_TEMPORAL_UPSAMPLING_OPTIONS = ["Disabled", "RIFE x2", "RIFE x4"]

_SPATIAL_UPSAMPLING_OPTIONS = ["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"]


class GenerateTab(BaseTab):
    """Video generation tab (Mode 1: single image, Mode 2: first+last frame, Mode 3: video clip)."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: InferenceWorker | None = None
        self._static_guidance_tensor = None
        # Tracks (mode, source, num_frames) the static guidance tensor was
        # built for, so a stale tensor can't cross modes / leak into a later run.
        self._static_guidance_key = None
        self._current_backend = "wan"
        # name -> multiplier token map, kept aligned with the multipliers field
        # so collect can re-emit per-LoRA tokens in activated-list order (the
        # list widget is sorted alphabetically, which otherwise desyncs the
        # positional zip in apply_loras).
        self._lora_mult_map: dict[str, str] = {}
        # Guards immediate-persist handlers (e.g. _persist_image_path) and the
        # Wan-preset auto-resolve while _restore_from_config repopulates widgets,
        # so loading a project doesn't write the values straight back to disk.
        self._restoring = False
        self._wan_only_widgets: list = []
        self._ff_original_path: str | None = None
        self._lf_original_path: str | None = None
        # Last actually-used seed (after -1 randomization), reported by the
        # worker on gen-done; written back by the "Reuse seed" button.
        self._resolved_seed: int = -1
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        # Font sizes + input heights come from the global design system in
        # sdqt/app.py (15px base, 32px input min-height). Keep only the
        # bold group-box headers, which need no size override.
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; }"
        )

        # ── Top: Source (left) + Preview (right) — 50/50 ────────────────
        top_splitter = QSplitter(Qt.Horizontal)

        # Left: mode selector + mode-specific source widget
        source_panel = QWidget()
        source_layout = QVBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(2)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        _mode_style = "QRadioButton { padding: 4px 2px; }"
        self._mode_group = QButtonGroup(self)
        self._mode1_rb = QRadioButton("Single Image")
        self._mode1_rb.setChecked(True)
        self._mode1_rb.setStyleSheet(_mode_style)
        self._mode2_rb = QRadioButton("First/Last Frame")
        self._mode2_rb.setStyleSheet(_mode_style)
        self._mode3_rb = QRadioButton("Video Clip")
        self._mode3_rb.setStyleSheet(_mode_style)
        self._mode_group.addButton(self._mode1_rb, 1)
        self._mode_group.addButton(self._mode2_rb, 2)
        self._mode_group.addButton(self._mode3_rb, 3)
        mode_row.addWidget(self._mode1_rb)
        mode_row.addWidget(self._mode2_rb)
        mode_row.addWidget(self._mode3_rb)
        mode_row.addStretch()
        source_layout.addLayout(mode_row)
        self._mode_group.idToggled.connect(self._on_mode_change)

        # Mode 1: Source image (drag/drop)
        self._m1_group = QWidget()
        m1_layout = QVBoxLayout(self._m1_group)
        m1_layout.setContentsMargins(0, 0, 0, 0)
        self._image_source = ImageDropWidget("Source Image", thumb_height=300)
        self._image_source.image_loaded.connect(lambda p: self._persist_image_path("image_path", p))
        self._image_source.image_cleared.connect(lambda: self._persist_image_path("image_path", ""))
        m1_layout.addWidget(self._image_source)
        source_layout.addWidget(self._m1_group, 1)

        # Mode 2: First/Last frame (drag/drop) — side by side with color match buttons
        self._m2_group = QWidget()
        m2_layout = QHBoxLayout(self._m2_group)
        m2_layout.setContentsMargins(0, 0, 0, 0)
        m2_layout.setSpacing(8)

        # First Frame column
        ff_col = QVBoxLayout()
        ff_col.setSpacing(4)
        ff_header = QHBoxLayout()
        ff_header.setSpacing(6)
        ff_label = QLabel("<b>First Frame</b>")
        ff_header.addWidget(ff_label)
        self._ff_match_btn = QPushButton("Color Match LF")
        self._ff_match_btn.setMinimumHeight(34)
        self._ff_match_btn.setToolTip("Color-match the first frame to the last frame")
        self._ff_match_btn.clicked.connect(self._color_match_ff_to_lf)
        ff_header.addWidget(self._ff_match_btn)
        self._ff_undo_btn = QPushButton("Undo")
        self._ff_undo_btn.setMinimumHeight(34)
        self._ff_undo_btn.setToolTip("Restore original first frame")
        self._ff_undo_btn.clicked.connect(self._undo_ff_match)
        self._ff_undo_btn.setEnabled(False)
        ff_header.addWidget(self._ff_undo_btn)
        ff_header.addStretch()
        ff_col.addLayout(ff_header)
        self._first_frame = ImageDropWidget("", thumb_height=200)
        self._first_frame.image_loaded.connect(lambda p: self._persist_image_path("first_frame_path", p))
        self._first_frame.image_cleared.connect(lambda: self._persist_image_path("first_frame_path", ""))
        ff_col.addWidget(self._first_frame, 1)
        m2_layout.addLayout(ff_col, 1)

        # Last Frame column
        lf_col = QVBoxLayout()
        lf_col.setSpacing(4)
        lf_header = QHBoxLayout()
        lf_header.setSpacing(6)
        lf_label = QLabel("<b>Last Frame</b>")
        lf_header.addWidget(lf_label)
        self._lf_match_btn = QPushButton("Color Match FF")
        self._lf_match_btn.setMinimumHeight(34)
        self._lf_match_btn.setToolTip("Color-match the last frame to the first frame")
        self._lf_match_btn.clicked.connect(self._color_match_lf_to_ff)
        lf_header.addWidget(self._lf_match_btn)
        self._lf_undo_btn = QPushButton("Undo")
        self._lf_undo_btn.setMinimumHeight(34)
        self._lf_undo_btn.setToolTip("Restore original last frame")
        self._lf_undo_btn.clicked.connect(self._undo_lf_match)
        self._lf_undo_btn.setEnabled(False)
        lf_header.addWidget(self._lf_undo_btn)
        lf_header.addStretch()
        lf_col.addLayout(lf_header)
        self._last_frame = ImageDropWidget("", thumb_height=200)
        self._last_frame.image_loaded.connect(lambda p: self._persist_image_path("last_frame_path", p))
        self._last_frame.image_cleared.connect(lambda: self._persist_image_path("last_frame_path", ""))
        lf_col.addWidget(self._last_frame, 1)
        m2_layout.addLayout(lf_col, 1)

        self._m2_group.setVisible(False)
        source_layout.addWidget(self._m2_group, 1)

        # Mode 3: Video clip (drag/drop via FileDropWidget for video)
        self._m3_group = QWidget()
        m3_layout = QVBoxLayout(self._m3_group)
        m3_layout.setContentsMargins(0, 0, 0, 0)
        self._m3_video = FileDropWidget("Source Video")
        self._m3_video.file_loaded.connect(self._on_m3_source_loaded)
        self._m3_video.file_cleared.connect(self._on_m3_source_cleared)
        m3_layout.addWidget(self._m3_video, 1)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Start Frame:"))
        self._m3_start_frame = QSpinBox()
        self._m3_start_frame.setRange(0, 999999)
        frame_row.addWidget(self._m3_start_frame)
        frame_row.addWidget(QLabel("End Frame:"))
        self._m3_end_frame = QSpinBox()
        self._m3_end_frame.setRange(0, 999999)
        frame_row.addWidget(self._m3_end_frame)
        m3_layout.addLayout(frame_row)
        self._m3_group.setVisible(False)
        source_layout.addWidget(self._m3_group, 1)

        top_splitter.addWidget(source_panel)

        # Right: Preview video + action buttons
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(2)

        self._preview = VideoPlayerWidget("Preview")
        self._preview.set_controls_size(True)
        preview_layout.addWidget(self._preview, 1)

        # Generate / Batch Generate / Abort — main page actions use the global
        # #primary button class (44px, 16px bold, accent — defined in app.py).
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setObjectName("primary")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn, 1)
        self._batch_btn = QPushButton("Batch Generate")
        self._batch_btn.setObjectName("primary")
        self._batch_btn.clicked.connect(self._on_batch_or_sequential)
        btn_row.addWidget(self._batch_btn, 1)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setObjectName("primary")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn, 1)
        preview_layout.addLayout(btn_row)

        # Batch controller slot (populated while a batch is running)
        self._batch_controller = None

        self._status = QLabel("")
        # Secondary status text — de-emphasized with color, never a smaller size.
        self._status.setStyleSheet("color: #aaa; font-size: 13px;")
        preview_layout.addWidget(self._status)

        # Save clip + frame capture — secondary actions (global 32px button rule).
        _action_btn_style = (
            "QPushButton { font-weight: bold; padding: 8px 12px; }"
        )
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        self._clip_name = QLineEdit()
        self._clip_name.setPlaceholderText("Clip name...")
        self._clip_name.setMinimumHeight(38)
        actions_row.addWidget(self._clip_name, 1)
        self._save_clip_btn = QPushButton("Save Clip")
        self._save_clip_btn.setStyleSheet(_action_btn_style)
        self._save_clip_btn.setMinimumHeight(38)
        self._save_clip_btn.setEnabled(False)
        self._save_clip_btn.clicked.connect(self._on_save_clip)
        actions_row.addWidget(self._save_clip_btn)
        self._dl_frame_btn = QPushButton("Download Frame")
        self._dl_frame_btn.setStyleSheet(_action_btn_style)
        self._dl_frame_btn.setMinimumHeight(38)
        self._dl_frame_btn.setEnabled(False)
        self._dl_frame_btn.clicked.connect(self._on_download_frame)
        actions_row.addWidget(self._dl_frame_btn)
        self._save_frame_btn = QPushButton("Save Frame")
        self._save_frame_btn.setStyleSheet(_action_btn_style)
        self._save_frame_btn.setMinimumHeight(38)
        self._save_frame_btn.setEnabled(False)
        self._save_frame_btn.clicked.connect(self._on_save_frame)
        actions_row.addWidget(self._save_frame_btn)
        preview_layout.addLayout(actions_row)

        top_splitter.addWidget(preview_panel)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        # ── Bottom: Settings (scrollable) ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(4, 2, 4, 2)

        # Prompt
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        # Enhance widget (video mode)
        self._enhance = PromptEnhanceWidget(mode="video")
        self._enhance.enhance_requested.connect(self._on_enhance)
        prompt_layout.addWidget(self._enhance)
        self._enhance_worker = None

        # Stacked prompt: standard (0) vs Lightning per-second (1)
        self._prompt_stack = QStackedWidget()

        # Page 0: Standard single prompt
        std_page = QWidget()
        std_layout = QVBoxLayout(std_page)
        std_layout.setContentsMargins(0, 0, 0, 0)
        self._prompt = QPlainTextEdit()
        self._prompt.setMinimumHeight(120)
        self._prompt.setMaximumHeight(140)
        self._prompt.setPlaceholderText("Describe the video...")
        std_layout.addWidget(self._prompt)
        self._prompt_stack.addWidget(std_page)

        # Page 1: Lightning per-second fields (scrollable)
        lightning_page = QWidget()
        lightning_outer = QVBoxLayout(lightning_page)
        lightning_outer.setContentsMargins(0, 0, 0, 0)
        self._lightning_scroll = QScrollArea()
        self._lightning_scroll.setWidgetResizable(True)
        self._lightning_scroll.setMinimumHeight(120)
        self._lightning_scroll.setMaximumHeight(140)
        self._lightning_fields_widget = QWidget()
        self._lightning_fields_layout = QVBoxLayout(self._lightning_fields_widget)
        self._lightning_fields_layout.setContentsMargins(0, 0, 0, 0)
        self._lightning_fields_layout.setSpacing(2)
        self._lightning_scroll.setWidget(self._lightning_fields_widget)
        lightning_outer.addWidget(self._lightning_scroll)
        self._lightning_fields: list[QLineEdit] = []
        self._prompt_stack.addWidget(lightning_page)

        prompt_layout.addWidget(self._prompt_stack)

        self._neg_prompt = QLineEdit()
        self._neg_prompt.setMinimumHeight(38)
        self._neg_prompt.setPlaceholderText("Negative prompt (optional)")
        prompt_layout.addWidget(self._neg_prompt)

        # Prompt profiles
        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        self._profile_combo = QComboBox()
        self._profile_combo.setMinimumWidth(200)
        profile_row.addWidget(QLabel("Profile:"))
        profile_row.addWidget(self._profile_combo, 1)
        save_profile = QPushButton("Save")
        save_profile.clicked.connect(self._save_profile)
        profile_row.addWidget(save_profile)
        load_profile = QPushButton("Load")
        load_profile.clicked.connect(self._load_profile)
        profile_row.addWidget(load_profile)
        del_profile = QPushButton("Delete")
        del_profile.clicked.connect(self._delete_profile)
        profile_row.addWidget(del_profile)
        prompt_layout.addLayout(profile_row)
        settings_layout.addWidget(prompt_group)

        # Model & Output
        self._build_model_output_section(settings_layout)

        # SVI 2 Pro — Sliding Window (visible only when SVI model selected)
        self._build_svi_section(settings_layout)

        # Auto Color Match (always visible — replaces post-hoc timeline grading)
        self._build_color_match_section(settings_layout)

        # Advanced Settings (includes Quality + Self Refiner)
        self._build_advanced_section(settings_layout)

        # Guidance & Final Frame sections (per mode)
        self._build_guidance_section(settings_layout)
        # Default: Mode 1 selected, hide Mode 2 sections
        self._guid_ff_section_m2.setVisible(False)

        # Post-Processing
        self._build_postprocessing_section(settings_layout)

        # LoRA Controls
        self._build_lora_section(settings_layout)

        settings_layout.addStretch()
        scroll.setWidget(settings)

        # Vertical splitter: preview on top, settings on bottom — drag to resize
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter, 1)

        # Load prompt profiles
        self._refresh_profiles()

    # -- Section builders --------------------------------------------------

    # Fixed widget widths for consistent grid alignment
    _W_SPIN = 90
    _W_DSPIN = 95
    _W_COMBO_S = 120
    _W_COMBO_M = 160
    _W_COMBO_L = 200

    def _build_model_output_section(self, parent_layout: QVBoxLayout) -> None:
        section = CollapsibleSection("Model Timing")

        # Row 1: Model + Resolution
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Model:"))
        self._model_type = QComboBox()
        self._model_type.setMinimumWidth(self._W_COMBO_L)
        for display, key in _MODEL_TYPES:
            self._model_type.addItem(display, userData=key)
        self._model_type.currentIndexChanged.connect(self._on_model_type_changed)
        row.addWidget(self._model_type)
        row.addWidget(QLabel("Res:"))
        self._resolution = QComboBox()
        self._resolution.setMinimumWidth(self._W_COMBO_L)
        self._resolution.addItems(_RESOLUTION_PRESETS)
        row.addWidget(self._resolution, 1)
        row.addStretch()
        section.add_layout(row)

        # Row 2: Duration radios + Custom + FPS. Kept on its own layout so the
        # row isn't overpacked; _reconfigure_for_backend rebuilds the radios in
        # THIS layout (tracked via self._timing_dur_row), inserting before FPS.
        dur_row = QHBoxLayout()
        dur_row.setSpacing(8)
        self._timing_dur_row = dur_row
        dur_row.addWidget(QLabel("Duration:"))
        self._duration_group = QButtonGroup(self)
        for i, (label, _) in enumerate(_DURATION_PRESETS):
            rb = QRadioButton(label)
            self._duration_group.addButton(rb, i)
            dur_row.addWidget(rb)
            if i == 1:
                rb.setChecked(True)
        # Custom duration radio + frame spinbox. Wan i2v needs (n-1) % 4 == 0;
        # LTX needs (n-1) % 8 == 0. Step is set by _reconfigure_for_backend.
        self._duration_custom_rb = QRadioButton("Custom")
        custom_id = len(_DURATION_PRESETS)
        self._duration_group.addButton(self._duration_custom_rb, custom_id)
        dur_row.addWidget(self._duration_custom_rb)
        self._duration_custom_frames = QSpinBox()
        self._duration_custom_frames.setMinimumWidth(self._W_SPIN)
        self._duration_custom_frames.setRange(9, 1009)
        self._duration_custom_frames.setSingleStep(4)
        self._duration_custom_frames.setValue(241)
        self._duration_custom_frames.setSuffix(" fr")
        self._duration_custom_frames.setToolTip(
            "Custom frame count. Wan needs (n-1) % 4 == 0; LTX needs (n-1) % 8 == 0. "
            "Use 81 for one Wan window, 241 for ~15s SVI long-form."
        )
        # Disabled until Custom radio is checked
        self._duration_custom_frames.setEnabled(False)
        # Auto-select the Custom radio whenever the user touches the spinbox,
        # and refresh per-window/second prompt fields (count depends on frames).
        self._duration_custom_frames.valueChanged.connect(self._on_custom_frames_changed)
        dur_row.addWidget(self._duration_custom_frames)
        self._duration_group.idToggled.connect(self._on_duration_toggled)
        dur_row.addWidget(QLabel("FPS:"))
        self._fps = QSpinBox()
        self._fps.setMinimumWidth(self._W_SPIN)
        self._fps.setRange(1, 60)
        self._fps.setValue(16)
        dur_row.addWidget(self._fps)
        dur_row.addStretch()
        section.add_layout(dur_row)

        parent_layout.addWidget(section)

    def _build_svi_section(self, parent_layout: QVBoxLayout) -> None:
        """SVI 2 Pro — Sliding Window section. Visible only when an SVI
        model is selected (toggled in _on_model_type_changed). Starts
        expanded so the controls are immediately discoverable.
        """
        section = CollapsibleSection("SVI 2 Pro — Sliding Window", collapsed=False)
        self._svi_section = section
        section.setVisible(False)
        self._svi_first_show = True

        # Row 1: Window size, Overlap, Discard last
        r1 = QHBoxLayout()
        r1.setSpacing(6)
        r1.addWidget(QLabel("Window Size:"))
        self._sliding_window_size = QSpinBox()
        self._sliding_window_size.setMinimumWidth(self._W_SPIN)
        self._sliding_window_size.setRange(17, 241)
        self._sliding_window_size.setSingleStep(8)
        self._sliding_window_size.setValue(81)
        # Window count (and therefore the LTX per-window prompt field count)
        # depends on this — rebuild the prompt fields when it changes.
        self._sliding_window_size.valueChanged.connect(
            lambda _: self._refresh_prompt_mode()
        )
        r1.addWidget(self._sliding_window_size)
        r1.addWidget(QLabel("Overlap:"))
        self._sliding_window_overlap = QSpinBox()
        self._sliding_window_overlap.setMinimumWidth(self._W_SPIN)
        self._sliding_window_overlap.setRange(0, 80)
        self._sliding_window_overlap.setValue(4)
        r1.addWidget(self._sliding_window_overlap)
        r1.addWidget(QLabel("Discard Last:"))
        self._sliding_window_discard_last = QSpinBox()
        self._sliding_window_discard_last.setMinimumWidth(self._W_SPIN)
        self._sliding_window_discard_last.setRange(0, 40)
        self._sliding_window_discard_last.setValue(0)
        r1.addWidget(self._sliding_window_discard_last)
        r1.addStretch()
        section.add_layout(r1)

        # Row 2: Color corr (per-window), Overlap noise
        r2 = QHBoxLayout()
        r2.setSpacing(6)
        r2.addWidget(QLabel("Color Corr (per-window):"))
        self._sliding_window_color_correction = QDoubleSpinBox()
        self._sliding_window_color_correction.setMinimumWidth(self._W_DSPIN)
        self._sliding_window_color_correction.setRange(0.0, 1.0)
        self._sliding_window_color_correction.setDecimals(2)
        self._sliding_window_color_correction.setSingleStep(0.05)
        self._sliding_window_color_correction.setValue(0.0)
        r2.addWidget(self._sliding_window_color_correction)
        r2.addWidget(QLabel("Overlap Noise:"))
        self._sliding_window_overlap_noise = QDoubleSpinBox()
        self._sliding_window_overlap_noise.setMinimumWidth(self._W_DSPIN)
        self._sliding_window_overlap_noise.setRange(0.0, 1.0)
        self._sliding_window_overlap_noise.setDecimals(2)
        self._sliding_window_overlap_noise.setSingleStep(0.05)
        self._sliding_window_overlap_noise.setValue(0.0)
        r2.addWidget(self._sliding_window_overlap_noise)
        r2.addStretch()
        section.add_layout(r2)

        # Row 3: Anchor mode + path
        r3 = QHBoxLayout()
        r3.setSpacing(6)
        r3.addWidget(QLabel("Anchor:"))
        self._anchor_image_mode = QComboBox()
        self._anchor_image_mode.setMinimumWidth(self._W_COMBO_L)
        for label, _key in _ANCHOR_MODES:
            self._anchor_image_mode.addItem(label)
        r3.addWidget(self._anchor_image_mode)
        self._anchor_image_path = QLineEdit()
        self._anchor_image_path.setPlaceholderText(
            "(path) or (semicolon-separated paths for per-window modes)"
        )
        r3.addWidget(self._anchor_image_path, 1)
        anchor_browse = QPushButton("Browse…")
        anchor_browse.setFixedWidth(80)
        anchor_browse.clicked.connect(self._on_svi_anchor_browse)
        r3.addWidget(anchor_browse)
        section.add_layout(r3)

        # Row 4: per-window anchor drop slots (one drag/drop image per window;
        # shown only in per_window / per_window_keyframes modes). These drive
        # the semicolon-separated _anchor_image_path field positionally.
        self._anchor_slots: list = []
        self._anchor_slots_label = QLabel("Per-window anchors (drag/drop an image into each):")
        self._anchor_slots_label.setStyleSheet("color:#8c95a4; font-size:12px;")
        self._anchor_slots_label.setVisible(False)
        section.add_widget(self._anchor_slots_label)
        self._anchor_slots_scroll = QScrollArea()
        self._anchor_slots_scroll.setWidgetResizable(True)
        self._anchor_slots_scroll.setFixedHeight(140)
        self._anchor_slots_scroll.setVisible(False)
        self._anchor_slots_box = QWidget()
        self._anchor_slots_layout = QHBoxLayout(self._anchor_slots_box)
        self._anchor_slots_layout.setContentsMargins(0, 0, 0, 0)
        self._anchor_slots_layout.setSpacing(6)
        self._anchor_slots_scroll.setWidget(self._anchor_slots_box)
        section.add_widget(self._anchor_slots_scroll)

        self._anchor_image_mode.currentIndexChanged.connect(self._on_svi_anchor_mode_changed)
        self._on_svi_anchor_mode_changed(0)

        parent_layout.addWidget(section)

    def _build_color_match_section(self, parent_layout: QVBoxLayout) -> None:
        """Auto Color Match section — always visible, applies post-decode
        color transfer to the source frame so generations look like the
        source out of the box (eliminates need for post-hoc grading).
        """
        section = CollapsibleSection("Auto Color Match (to Source)", collapsed=False)
        self._cc_section = section

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Strength:"))
        self._color_correction = QDoubleSpinBox()
        self._color_correction.setMinimumWidth(self._W_DSPIN)
        self._color_correction.setRange(0.0, 1.0)
        self._color_correction.setDecimals(2)
        self._color_correction.setSingleStep(0.05)
        self._color_correction.setValue(0.0)
        self._color_correction.setToolTip(
            "0.0 = off; 1.0 = full match. Recommended 0.4–0.7."
        )
        row.addWidget(self._color_correction)
        row.addWidget(QLabel("Method:"))
        self._color_correction_method = QComboBox()
        self._color_correction_method.setMinimumWidth(self._W_COMBO_L)
        for label, key in _CC_METHODS:
            self._color_correction_method.addItem(label, userData=key)
        self._color_correction_method.setToolTip(
            "Mean-only Lab is drift-resistant for chained continuations.\n"
            "HM-MVGD-HM gives strongest single-shot match-to-source — pick "
            "this to eliminate timeline grading on most generations."
        )
        row.addWidget(self._color_correction_method)
        row.addWidget(QLabel("Persistence:"))
        self._color_anchor_persistence = QDoubleSpinBox()
        self._color_anchor_persistence.setMinimumWidth(self._W_DSPIN)
        self._color_anchor_persistence.setRange(0.0, 1.0)
        self._color_anchor_persistence.setDecimals(2)
        self._color_anchor_persistence.setSingleStep(0.05)
        self._color_anchor_persistence.setValue(0.6)
        self._color_anchor_persistence.setToolTip(
            "How much of Strength holds across the whole clip vs fading after ~1s.\n"
            "0.0 = legacy fade-to-zero (VAE warm bias can re-accumulate).\n"
            "0.6 = default — strong anchor throughout, drift can't sneak back.\n"
            "1.0 = full strength on every frame (may over-correct natural color evolution)."
        )
        row.addWidget(self._color_anchor_persistence)
        row.addStretch()
        section.add_layout(row)

        # Anchor reference — always visible (Phase 1 + 4b multi-ref). When
        # generating chains or sliding windows, this picks WHAT to color-match
        # each chunk to. 'start' = first source frame, 'single' = user-picked
        # image, 'per_window' = rotate through ';'-separated paths.
        from sdqt.widgets.generation_params import _ANCHOR_MODES
        row_anchor = QHBoxLayout()
        row_anchor.setSpacing(6)
        row_anchor.addWidget(QLabel("Anchor mode:"))
        self._anchor_image_mode_cc = QComboBox()
        self._anchor_image_mode_cc.setMinimumWidth(self._W_COMBO_L)
        for label, _key in _ANCHOR_MODES:
            self._anchor_image_mode_cc.addItem(label)
        self._anchor_image_mode_cc.setToolTip(
            "Reference for color anchor on chunk N:\n"
            "• Use start frame — every chunk anchors to original source (drift-resistant)\n"
            "• Single anchor — fixed user-picked image (style lock)\n"
            "• Per-window — rotate through ; -separated images per chunk\n"
            "• Per-window keyframes — SVI-only, end-of-window targets"
        )
        self._anchor_image_mode_cc.currentIndexChanged.connect(
            self._on_anchor_mode_cc_changed,
        )
        row_anchor.addWidget(self._anchor_image_mode_cc)
        self._anchor_image_path_cc = QLineEdit()
        self._anchor_image_path_cc.setPlaceholderText(
            "Reference image path(s) — required for single/per_window modes"
        )
        row_anchor.addWidget(self._anchor_image_path_cc, 1)
        self._anchor_browse_cc = QPushButton("Browse")
        self._anchor_browse_cc.clicked.connect(self._on_anchor_browse_cc)
        row_anchor.addWidget(self._anchor_browse_cc)
        section.add_layout(row_anchor)

        parent_layout.addWidget(section)

    def _on_anchor_mode_cc_changed(self, idx: int) -> None:
        """Mirror the chosen mode into the legacy SVI widget so cfg is consistent."""
        from sdqt.widgets.generation_params import _ANCHOR_MODES
        if 0 <= idx < len(_ANCHOR_MODES):
            mode_key = _ANCHOR_MODES[idx][1]
            needs_path = mode_key in ("single", "per_window", "per_window_keyframes")
            self._anchor_image_path_cc.setEnabled(needs_path)
            # Keep the legacy SVI widget synced so persistence is uniform.
            if hasattr(self, "_anchor_image_mode"):
                self._anchor_image_mode.blockSignals(True)
                self._anchor_image_mode.setCurrentIndex(idx)
                self._anchor_image_mode.blockSignals(False)

    def _on_anchor_browse_cc(self) -> None:
        from sdqt.widgets.generation_params import _ANCHOR_MODES
        idx = self._anchor_image_mode_cc.currentIndex()
        mode_key = _ANCHOR_MODES[idx][1] if 0 <= idx < len(_ANCHOR_MODES) else "start"
        if mode_key in ("per_window", "per_window_keyframes"):
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select anchor images (one per window)",
                "", "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if paths:
                self._anchor_image_path_cc.setText(";".join(paths))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select anchor image",
                "", "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if path:
                self._anchor_image_path_cc.setText(path)

    def _on_preset_changed(self, _idx: int = -1) -> None:
        """Apply Wan 2.2 Acceleration / Anti-drift preset.

        Reads the two dropdowns, resolves LoRAs from disk, and patches:
          - activated_loras list (UI checkboxes via _set_activated_loras)
          - num_inference_steps spinbox
          - guidance_scale spinbox
          - solver dropdown
        Runs lazily — only patches the visible UI widgets, the actual config
        is persisted on next save (collect_to_config / _persist_params).
        """
        if getattr(self, "_restoring", False):
            return  # skip during initial restore
        try:
            from supremediffusion.models.wan_presets import resolve_preset
        except Exception as exc:  # noqa: BLE001
            logger.warning("preset module import failed: %s", exc)
            return

        accel = self._acceleration_preset.currentData() or "off"
        antidrift = self._antidrift_preset.currentData() or "off"
        if accel == "off" and antidrift == "off":
            # User cleared presets — leave existing LoRAs in place, just blank
            # the auto-injected ones if we want. For now leave alone.
            return

        lora_dir = ""
        if hasattr(self, "state") and getattr(self.state, "global_config", None):
            lora_dir = self.state.global_config.model_paths.get("lora_dir", "") or ""
        if not lora_dir:
            self._show_status("Lora dir not configured — preset can't auto-load LoRAs.")
            return

        # Existing user-selected LoRAs (preserve them — preset prepends its own).
        existing = self._get_activated_loras()
        existing_mult = (self._lora_multipliers.text() if hasattr(self, "_lora_multipliers") else "") or ""

        result = resolve_preset(
            acceleration=accel,
            antidrift=antidrift,
            lora_dir=lora_dir,
            user_loras=existing,
            user_multipliers=existing_mult,
        )

        for w in result.get("warnings", []):
            self._show_status(w)
            logger.warning(w)

        # Patch widgets — block signals while we mass-update to avoid
        # re-triggering preset changes.
        self._set_activated_loras(result["loras"])
        if hasattr(self, "_lora_multipliers"):
            self._lora_multipliers.blockSignals(True)
            self._lora_multipliers.setText(result["multipliers"])
            self._lora_multipliers.blockSignals(False)
            # Record per-name multipliers (preset emits loras/multipliers
            # positionally aligned) so collect re-emits them in the
            # alphabetically-sorted activated-list order.
            self._record_lora_multipliers(result["loras"], result["multipliers"])
            self._rebuild_lora_strength_view()
        if result["num_steps"] is not None and hasattr(self, "_steps"):
            self._steps.setValue(int(result["num_steps"]))
        if result["guidance_scale"] is not None and hasattr(self, "_guidance_scale"):
            self._guidance_scale.setValue(float(result["guidance_scale"]))
        if result["sampler"] and hasattr(self, "_solver"):
            idx = self._solver.findText(result["sampler"])
            if idx >= 0:
                self._solver.setCurrentIndex(idx)
        self._show_status(
            f"Preset applied: accel={accel}, anti-drift={antidrift} → {len(result['loras'])} LoRAs"
        )

    def _on_svi_anchor_mode_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(_ANCHOR_MODES):
            return
        mode_key = _ANCHOR_MODES[idx][1]
        needs_path = mode_key in ("single", "per_window", "per_window_keyframes")
        self._anchor_image_path.setEnabled(needs_path)
        # Per-window modes get drag/drop slots (one per window); the raw path
        # field is hidden since the slots drive it.
        per_window = mode_key in ("per_window", "per_window_keyframes")
        # Keep the always-visible Color-Match anchor combo (which _collect_config
        # treats as the source of truth) in sync with this SVI combo.
        if hasattr(self, "_anchor_image_mode_cc") and self._anchor_image_mode_cc.currentIndex() != idx:
            self._anchor_image_mode_cc.blockSignals(True)
            self._anchor_image_mode_cc.setCurrentIndex(idx)
            self._anchor_image_mode_cc.blockSignals(False)
        if hasattr(self, "_anchor_slots_box"):
            self._anchor_slots_label.setVisible(per_window)
            self._anchor_slots_scroll.setVisible(per_window)
            self._anchor_image_path.setVisible(not per_window)
            if per_window:
                self._rebuild_anchor_slots()
        if mode_key == "per_window":
            self._anchor_image_path.setPlaceholderText(
                "Semicolon-separated images, one per window (hard cuts at start)"
            )
        elif mode_key == "per_window_keyframes":
            self._anchor_image_path.setPlaceholderText(
                "Semicolon-separated keyframes — window N converges to keyframe[N] at end"
            )
        else:
            self._anchor_image_path.setPlaceholderText("Path to anchor image")

    def _num_anchor_windows(self) -> int:
        """Window count for the current model/duration (matches the backend:
        LTX = ceil(frames/window); SVI = overlap-aware stride)."""
        import math
        if not hasattr(self, "_sliding_window_size"):
            return 1
        win = self._sliding_window_size.value()
        frames = self._current_duration_frames()
        if win <= 0 or frames <= win:
            return 1
        if get_video_backend(self._model_type.currentData() or "") == "ltx":
            return max(1, math.ceil(frames / win))
        overlap = self._sliding_window_overlap.value() if hasattr(self, "_sliding_window_overlap") else 0
        stride = max(1, win - overlap)
        return max(1, math.ceil((frames - win) / stride) + 1)

    def _rebuild_anchor_slots(self) -> None:
        """(Re)build one drag/drop anchor slot per window, preserving any loaded
        images and seeding from the current semicolon path field."""
        if not hasattr(self, "_anchor_slots_box"):
            return
        n = self._num_anchor_windows()
        # preserve existing slot images; fall back to the text field
        old = [s.image_path for s in self._anchor_slots]
        if not any(old):
            old = [p.strip() for p in self._anchor_image_path.text().split(";")]
        while self._anchor_slots_layout.count():
            it = self._anchor_slots_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        self._anchor_slots = []
        for i in range(n):
            slot = ImageDropWidget(f"Win {i + 1}", thumb_height=90)
            if i < len(old) and old[i]:
                slot.load_image(old[i])
            slot.image_loaded.connect(self._sync_anchor_slots)
            slot.image_cleared.connect(self._sync_anchor_slots)
            self._anchor_slots_layout.addWidget(slot)
            self._anchor_slots.append(slot)
        self._anchor_slots_layout.addStretch()
        self._sync_anchor_slots()

    def _sync_anchor_slots(self) -> None:
        """Write the slot images into the semicolon path field(s), positionally
        (empty slots stay empty so window N maps to slot N; the backend treats an
        empty entry as 'chain from the previous window'). Mirrors into the
        always-visible Color-Match path field, which _collect_config reads."""
        paths = [(s.image_path or "") for s in self._anchor_slots]
        while paths and not paths[-1]:
            paths.pop()
        joined = ";".join(paths)
        self._anchor_image_path.setText(joined)
        if hasattr(self, "_anchor_image_path_cc"):
            self._anchor_image_path_cc.setText(joined)

    def _maybe_rebuild_anchor_slots(self) -> None:
        """Rebuild slots when the window count may have changed, if a per-window
        anchor mode is active."""
        if not hasattr(self, "_anchor_image_mode"):
            return
        idx = self._anchor_image_mode.currentIndex()
        if 0 <= idx < len(_ANCHOR_MODES) and _ANCHOR_MODES[idx][1] in ("per_window", "per_window_keyframes"):
            self._rebuild_anchor_slots()

    def _on_svi_anchor_browse(self) -> None:
        idx = self._anchor_image_mode.currentIndex()
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
                self._anchor_image_path.setText(";".join(paths))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select anchor image",
                "", "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if path:
                self._anchor_image_path.setText(path)

    def _build_advanced_section(self, parent_layout: QVBoxLayout) -> None:
        # NOTE: this section contains Steps / Guidance / Flow Shift / Sampler /
        # Seed — all everyday-use knobs, not actually "advanced". Expanded by
        # default so users can see + change the sampler (Lightning LoRAs need
        # euler/beta or euler_a/beta and were previously invisible behind a
        # collapsed header).
        section = CollapsibleSection("Sampler & Guidance", collapsed=False)

        # Row -1: Output Quality preset — wires the unified PostProcessPipeline.
        # Draft = no post; Standard = color anchor; Quality = + upscale; Master = + RIFE.
        row_oq = QHBoxLayout()
        row_oq.setSpacing(6)
        row_oq.addWidget(QLabel("Output Quality:"))
        self._output_quality = QComboBox()
        self._output_quality.setMinimumWidth(self._W_COMBO_S + 80)
        from sdqt.workers.post_pipeline import QUALITY_PRESETS, QUALITY_LABELS
        for k in QUALITY_PRESETS:
            self._output_quality.addItem(QUALITY_LABELS[k], userData=k)
        self._output_quality.setCurrentIndex(1)  # Standard default
        self._output_quality.setToolTip(
            "Post-processing chain applied after generation.\n"
            "Draft = raw model output (no post-processing).\n"
            "Standard = color anchor (anti-drift) only.\n"
            "Quality = + 2× ESRGAN upscale.\n"
            "Master = + 4× upscale + RIFE 2× fps interpolation."
        )
        row_oq.addWidget(self._output_quality)
        row_oq.addStretch()
        section.add_layout(row_oq)

        # Row 0: Wan 2.2 Presets — Acceleration (Lightning) + Anti-drift (SVI Pro).
        # These auto-load the right LoRAs from lora_dir + adjust steps/CFG/sampler.
        row0 = QHBoxLayout()
        row0.setSpacing(6)
        row0.addWidget(QLabel("Acceleration:"))
        self._acceleration_preset = QComboBox()
        self._acceleration_preset.setMinimumWidth(self._W_COMBO_S + 80)
        # Order matches ACCELERATION_PRESETS in wan_presets.py.
        self._acceleration_preset.addItem("Off (default — 25-step)", userData="off")
        self._acceleration_preset.addItem("Lightning 4-step (fastest)", userData="lightning_4step")
        self._acceleration_preset.addItem("Lightning 6-step (recommended)", userData="lightning_6step")
        self._acceleration_preset.setToolTip(
            "Lightning 4-step LoRA distillation. Auto-loads HIGH+LOW Lightning LoRAs, "
            "drops steps to 4/6, CFG to 1.5, sampler to euler/beta. Generation 5-10× faster.\n"
            "Requires Wan2.2-Lightning_I2V-A14B-4steps_HIGH/LOW.safetensors in lora_dir."
        )
        self._acceleration_preset.currentIndexChanged.connect(self._on_preset_changed)
        row0.addWidget(self._acceleration_preset)

        row0.addWidget(QLabel("Anti-drift:"))
        self._antidrift_preset = QComboBox()
        self._antidrift_preset.setMinimumWidth(self._W_COMBO_S + 40)
        self._antidrift_preset.addItem("Off", userData="off")
        self._antidrift_preset.addItem("SVI Pro", userData="svi_pro")
        self._antidrift_preset.setToolTip(
            "SVI 2 Pro LoRA — modifies the noise schedule's padding behavior to keep "
            "chunks color/identity-stable across long generations. Auto-loads HIGH+LOW.\n"
            "Requires SVI_*_HIGH/LOW.safetensors in lora_dir."
        )
        self._antidrift_preset.currentIndexChanged.connect(self._on_preset_changed)
        row0.addWidget(self._antidrift_preset)
        row0.addStretch()
        section.add_layout(row0)

        # Row 1: Steps, Guidance, Guidance 2 (everyday Wan/LTX knobs)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setMinimumWidth(self._W_SPIN)
        self._steps.setRange(1, 100)
        self._steps.setValue(4)
        row1.addWidget(self._steps)
        row1.addWidget(QLabel("Guidance:"))
        self._guidance_scale = QDoubleSpinBox()
        self._guidance_scale.setMinimumWidth(self._W_DSPIN)
        self._guidance_scale.setRange(0, 30)
        self._guidance_scale.setDecimals(1)
        self._guidance_scale.setValue(1.0)
        row1.addWidget(self._guidance_scale)
        self._guidance2_label = QLabel("Guidance 2:")
        row1.addWidget(self._guidance2_label)
        self._guidance2_scale = QDoubleSpinBox()
        self._guidance2_scale.setMinimumWidth(self._W_DSPIN)
        self._guidance2_scale.setRange(0, 30)
        self._guidance2_scale.setDecimals(1)
        self._guidance2_scale.setValue(1.0)
        row1.addWidget(self._guidance2_scale)
        row1.addStretch()
        section.add_layout(row1)

        # Row 1b: LTX-only Alt CFG + tail Discard. Split onto its own row so
        # Row 1 isn't overpacked; the whole row hides on Wan (these widgets are
        # in _ltx_only_widgets, toggled by _reconfigure_for_backend).
        self._ltx_row1b = QWidget()
        row1b = QHBoxLayout(self._ltx_row1b)
        row1b.setContentsMargins(0, 0, 0, 0)
        row1b.setSpacing(8)
        self._alt_cfg_label = QLabel("Alt CFG:")
        row1b.addWidget(self._alt_cfg_label)
        self._alt_guidance_scale = QDoubleSpinBox()
        self._alt_guidance_scale.setMinimumWidth(self._W_DSPIN)
        self._alt_guidance_scale.setRange(0, 10)
        self._alt_guidance_scale.setDecimals(1)
        self._alt_guidance_scale.setSingleStep(0.5)
        self._alt_guidance_scale.setValue(1.0)
        self._alt_guidance_scale.setToolTip(
            "LTX Spatial-Temporal Guidance (alt_guidance_scale).\n"
            "1.0 = off. 3.0–5.0 strengthens motion + prompt adherence on distilled."
        )
        row1b.addWidget(self._alt_guidance_scale)
        self._ltx_discard_label = QLabel("Discard:")
        row1b.addWidget(self._ltx_discard_label)
        self._ltx_discard_last = QSpinBox()
        self._ltx_discard_last.setMinimumWidth(self._W_SPIN)
        self._ltx_discard_last.setRange(0, 32)
        self._ltx_discard_last.setSingleStep(1)
        self._ltx_discard_last.setValue(0)
        self._ltx_discard_last.setToolTip(
            "Trim N collapsed frames from the END of LTX output.\n"
            "Distilled commonly leaves 1–2 dead frames at the tail; try 1."
        )
        row1b.addWidget(self._ltx_discard_last)
        row1b.addStretch()
        section.add_widget(self._ltx_row1b)
        # Hide the whole LTX row (not just inner widgets) when Wan is active.
        self._ltx_only_widgets = getattr(self, "_ltx_only_widgets", [])
        self._ltx_only_widgets.append(self._ltx_row1b)

        # Row 2: Flow Shift, Solver, Seed
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Flow Shift:"))
        self._flow_shift = QDoubleSpinBox()
        self._flow_shift.setMinimumWidth(self._W_DSPIN)
        self._flow_shift.setRange(2, 12)
        self._flow_shift.setDecimals(1)
        self._flow_shift.setValue(3.0)
        row2.addWidget(self._flow_shift)
        row2.addWidget(QLabel("Solver:"))
        self._solver = QComboBox()
        # Widen for the longer Lightning-recommended sampler names.
        self._solver.setMinimumWidth(self._W_COMBO_M + 20)
        # Full set from supremediffusion.models.wan_scheduler — see that
        # module for what each maps to. **Lightning LoRAs**: pick euler/beta
        # or euler_a/beta for best results.
        from supremediffusion.models.wan_scheduler import SAMPLER_CHOICES
        self._solver.addItems(list(SAMPLER_CHOICES))
        self._solver.setToolTip(
            "Sampler / scheduler. unipc is the fast default. Lightning LoRAs "
            "require euler/beta or euler_a/beta. /beta = use_beta_sigmas. "
            "_a = ancestral (stochastic_sampling). lcm = AnimateLCM-style.")
        row2.addWidget(self._solver)
        row2.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setMinimumWidth(self._W_COMBO_S)
        self._seed.setRange(-1, 999999999)
        self._seed.setValue(-1)
        row2.addWidget(self._seed)
        row2.addStretch()
        section.add_layout(row2)

        # Row 2b: Reuse Seed on its own line (was crammed beside Solver/Seed).
        # Resolves -1 randomized runs to the concrete value that was used.
        row2b = QHBoxLayout()
        row2b.setSpacing(8)
        self._reuse_seed_btn = QPushButton("Reuse seed")
        self._reuse_seed_btn.setToolTip("Write the last generation's actual seed into the Seed field")
        self._reuse_seed_btn.setEnabled(False)
        self._reuse_seed_btn.clicked.connect(self._on_reuse_seed)
        row2b.addWidget(self._reuse_seed_btn)
        row2b.addStretch()
        section.add_layout(row2b)

        # Row 3: Denoising, Masking
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel("Denoising:"))
        self._denoising_strength = QDoubleSpinBox()
        self._denoising_strength.setMinimumWidth(self._W_DSPIN)
        self._denoising_strength.setRange(0, 1)
        self._denoising_strength.setDecimals(2)
        self._denoising_strength.setSingleStep(0.05)
        self._denoising_strength.setValue(1.0)
        row3.addWidget(self._denoising_strength)
        row3.addStretch()
        section.add_layout(row3)

        # Row 4: Phases, Switch (Wan-only)
        self._phases_row = QWidget()
        row4 = QHBoxLayout(self._phases_row)
        row4.setContentsMargins(0, 0, 0, 0)
        row4.setSpacing(6)
        row4.addWidget(QLabel("Phases:"))
        self._guidance_phases = QSpinBox()
        self._guidance_phases.setMinimumWidth(self._W_SPIN)
        self._guidance_phases.setRange(1, 2)
        self._guidance_phases.setValue(1)
        row4.addWidget(self._guidance_phases)
        row4.addWidget(QLabel("Switch:"))
        self._switch_threshold = QSpinBox()
        self._switch_threshold.setMinimumWidth(self._W_SPIN)
        self._switch_threshold.setRange(0, 900)
        self._switch_threshold.setValue(0)
        row4.addWidget(self._switch_threshold)
        row4.addStretch()
        section.add_widget(self._phases_row)

        # ── NAG / TEA Cache group ──────────────────────────────────────
        _nag_divider = QFrame()
        _nag_divider.setFrameShape(QFrame.HLine)
        _nag_divider.setFrameShadow(QFrame.Sunken)
        section.add_widget(_nag_divider)
        _nag_header = QLabel("<b>NAG / TEA Cache</b>")
        section.add_widget(_nag_header)

        # NAG / TEA enable gate
        self._loop_is_wan2gp = False
        self._nag_tea_enable = QCheckBox("Enable NAG / TEA Cache")
        self._nag_tea_enable.setChecked(False)
        section.add_widget(self._nag_tea_enable)

        # Row 5: NAG Scale, Tau, Alpha
        row5 = QHBoxLayout()
        row5.setSpacing(6)
        row5.addWidget(QLabel("NAG Scale:"))
        self._nag_scale = QDoubleSpinBox()
        self._nag_scale.setMinimumWidth(self._W_DSPIN)
        self._nag_scale.setRange(0, 100)
        self._nag_scale.setDecimals(2)
        self._nag_scale.setValue(0.0)
        row5.addWidget(self._nag_scale)
        row5.addWidget(QLabel("Tau:"))
        self._nag_tau = QDoubleSpinBox()
        self._nag_tau.setMinimumWidth(self._W_DSPIN)
        self._nag_tau.setRange(0, 100)
        self._nag_tau.setDecimals(2)
        self._nag_tau.setValue(0.0)
        row5.addWidget(self._nag_tau)
        row5.addWidget(QLabel("Alpha:"))
        self._nag_alpha = QDoubleSpinBox()
        self._nag_alpha.setMinimumWidth(self._W_DSPIN)
        self._nag_alpha.setRange(0, 100)
        self._nag_alpha.setDecimals(2)
        self._nag_alpha.setValue(0.0)
        row5.addWidget(self._nag_alpha)
        row5.addStretch()
        section.add_layout(row5)

        # Row 6: TEA Cache
        row6 = QHBoxLayout()
        row6.setSpacing(6)
        row6.addWidget(QLabel("TEA Cache:"))
        self._tea_cache = QComboBox()
        self._tea_cache.setMinimumWidth(self._W_COMBO_S)
        self._tea_cache.addItems(_TEA_CACHE_OPTIONS)
        row6.addWidget(self._tea_cache)
        row6.addWidget(QLabel("Start %:"))
        self._tea_start_perc = QDoubleSpinBox()
        self._tea_start_perc.setMinimumWidth(self._W_DSPIN)
        self._tea_start_perc.setRange(0, 100)
        self._tea_start_perc.setDecimals(1)
        self._tea_start_perc.setValue(0.0)
        row6.addWidget(self._tea_start_perc)
        row6.addStretch()
        section.add_layout(row6)

        # NAG controls gated behind wan2gp checkbox; TEA Cache always available
        self._nag_controls = [
            self._nag_scale, self._nag_tau, self._nag_alpha,
        ]
        self._nag_tea_enable.toggled.connect(self._on_nag_tea_toggled)
        self._on_nag_tea_toggled(False)

        parent_layout.addWidget(section)

        # Quality (Wan-only — SLG/APG/CFG* are wan2gp-specific)
        qual_section = CollapsibleSection("Quality", collapsed=True)
        self._quality_section = qual_section

        self._quality_enable = QCheckBox("Enable Quality Overrides")
        self._quality_enable.setChecked(False)
        qual_section.add_widget(self._quality_enable)

        slg_row = QHBoxLayout()
        slg_row.setSpacing(6)
        self._slg_switch = QCheckBox("SLG")
        slg_row.addWidget(self._slg_switch)
        slg_row.addWidget(QLabel("Layers:"))
        self._slg_layers = QLineEdit()
        self._slg_layers.setFixedWidth(220)
        self._slg_layers.setPlaceholderText("e.g. 7,8,9,10,11,12,13,14,15,16,17,18,19")
        slg_row.addWidget(self._slg_layers)
        slg_row.addStretch()
        qual_section.add_layout(slg_row)

        slg_row2 = QHBoxLayout()
        slg_row2.setSpacing(6)
        slg_row2.addWidget(QLabel("SLG Start %:"))
        self._slg_start_perc = QSpinBox()
        self._slg_start_perc.setMinimumWidth(self._W_SPIN)
        self._slg_start_perc.setRange(0, 100)
        self._slg_start_perc.setValue(0)
        slg_row2.addWidget(self._slg_start_perc)
        slg_row2.addWidget(QLabel("End %:"))
        self._slg_end_perc = QSpinBox()
        self._slg_end_perc.setMinimumWidth(self._W_SPIN)
        self._slg_end_perc.setRange(0, 100)
        self._slg_end_perc.setValue(100)
        slg_row2.addWidget(self._slg_end_perc)
        self._apg_switch = QCheckBox("APG")
        slg_row2.addWidget(self._apg_switch)
        slg_row2.addStretch()
        qual_section.add_layout(slg_row2)

        cfg_row = QHBoxLayout()
        cfg_row.setSpacing(6)
        cfg_row.addWidget(QLabel("CFG Star:"))
        self._cfg_star_switch = QSpinBox()
        self._cfg_star_switch.setMinimumWidth(self._W_SPIN)
        self._cfg_star_switch.setRange(0, 1)
        self._cfg_star_switch.setValue(0)
        cfg_row.addWidget(self._cfg_star_switch)
        cfg_row.addWidget(QLabel("CFG Zero:"))
        self._cfg_zero_step = QSpinBox()
        self._cfg_zero_step.setMinimumWidth(self._W_SPIN)
        self._cfg_zero_step.setRange(-1, 39)
        self._cfg_zero_step.setValue(-1)
        cfg_row.addWidget(self._cfg_zero_step)
        cfg_row.addStretch()
        qual_section.add_layout(cfg_row)

        motion_row = QHBoxLayout()
        motion_row.setSpacing(6)
        motion_row.addWidget(QLabel("Motion:"))
        self._motion_amplitude = QDoubleSpinBox()
        self._motion_amplitude.setMinimumWidth(self._W_DSPIN)
        self._motion_amplitude.setRange(1.0, 1.4)
        self._motion_amplitude.setDecimals(2)
        self._motion_amplitude.setSingleStep(0.01)
        self._motion_amplitude.setValue(1.0)
        motion_row.addWidget(self._motion_amplitude)
        motion_row.addWidget(QLabel("Color Corr:"))
        # Quality-section color-correction override (Wan-only). Distinct from
        # the always-visible Auto Color Match "Strength" spinbox
        # (self._color_correction) — keeping a separate attr here prevents it
        # from shadowing the visible widget that owns color_correction_strength.
        self._quality_color_correction = QDoubleSpinBox()
        self._quality_color_correction.setMinimumWidth(self._W_DSPIN)
        self._quality_color_correction.setRange(0, 1)
        self._quality_color_correction.setDecimals(2)
        self._quality_color_correction.setSingleStep(0.05)
        self._quality_color_correction.setValue(0.0)
        motion_row.addWidget(self._quality_color_correction)
        motion_row.addWidget(QLabel("Discard:"))
        self._discard_last_frames = QSpinBox()
        self._discard_last_frames.setMinimumWidth(self._W_SPIN)
        self._discard_last_frames.setRange(0, 20)
        self._discard_last_frames.setSingleStep(4)
        self._discard_last_frames.setValue(0)
        motion_row.addWidget(self._discard_last_frames)
        motion_row.addStretch()
        qual_section.add_layout(motion_row)

        # Self Refiner
        qual_section.add_widget(QLabel("<b>Self Refiner</b>"))

        refiner_row = QHBoxLayout()
        refiner_row.setSpacing(6)
        refiner_row.addWidget(QLabel("Refiner:"))
        self._self_refiner = QComboBox()
        self._self_refiner.setMinimumWidth(self._W_COMBO_S)
        self._self_refiner.addItems(_SELF_REFINER_OPTIONS)
        refiner_row.addWidget(self._self_refiner)
        refiner_row.addWidget(QLabel("Uncertainty:"))
        self._refiner_uncertainty = QDoubleSpinBox()
        self._refiner_uncertainty.setMinimumWidth(self._W_DSPIN)
        self._refiner_uncertainty.setRange(0, 10)
        self._refiner_uncertainty.setDecimals(3)
        self._refiner_uncertainty.setValue(0.0)
        refiner_row.addWidget(self._refiner_uncertainty)
        refiner_row.addWidget(QLabel("Cert Skip:"))
        self._refiner_certainty_skip = QDoubleSpinBox()
        self._refiner_certainty_skip.setMinimumWidth(self._W_DSPIN)
        self._refiner_certainty_skip.setRange(0, 10)
        self._refiner_certainty_skip.setDecimals(3)
        self._refiner_certainty_skip.setValue(0.0)
        refiner_row.addWidget(self._refiner_certainty_skip)
        refiner_row.addStretch()
        qual_section.add_layout(refiner_row)

        self._quality_controls = [
            self._slg_switch, self._slg_layers, self._slg_start_perc, self._slg_end_perc,
            self._apg_switch, self._cfg_star_switch, self._cfg_zero_step,
            self._motion_amplitude, self._quality_color_correction, self._discard_last_frames,
            self._self_refiner, self._refiner_uncertainty, self._refiner_certainty_skip,
        ]
        self._quality_enable.toggled.connect(self._on_quality_toggled)
        self._on_quality_toggled(False)

        parent_layout.addWidget(qual_section)

        # Collect Wan-only widgets for backend switching
        self._wan_only_widgets = [
            self._guidance2_label, self._guidance2_scale,
            self._phases_row,
            self._nag_tea_enable,
            self._quality_section,
        ]

    def _build_guidance_section(self, parent_layout: QVBoxLayout) -> None:
        # ── Mode 1: Guidance Video only (single image i2v) ───────────
        self._guid_section_m1 = CollapsibleSection("Guidance Video", collapsed=True)

        self._use_guidance_m1 = QCheckBox("Enable Guidance Video")
        self._use_guidance_m1.toggled.connect(self._on_guidance_m1_toggled)
        self._guid_section_m1.add_widget(self._use_guidance_m1)

        self._guidance_m1_container = QWidget()
        gm1_layout = QVBoxLayout(self._guidance_m1_container)
        gm1_layout.setContentsMargins(0, 0, 0, 0)
        gm1_layout.setSpacing(4)

        self._guidance_video_m1 = VideoDropWidget("Guidance Video", thumb_height=180)
        gm1_layout.addWidget(self._guidance_video_m1)

        self._guidance_fps_check_m1 = GuidanceFpsCheck(
            self._guidance_video_m1, lambda: self._fps.value()
        )
        gm1_layout.addWidget(self._guidance_fps_check_m1)

        gm1_opts = QHBoxLayout()
        self._gen_static_m1_btn = QPushButton("Preview Static from Source")
        self._gen_static_m1_btn.setToolTip("Pre-generate static guidance tensor and preview it")
        self._gen_static_m1_btn.clicked.connect(self._on_generate_static_guidance)
        gm1_opts.addWidget(self._gen_static_m1_btn)
        gm1_opts.addStretch()
        gm1_layout.addLayout(gm1_opts)

        self._guidance_m1_container.setVisible(False)
        self._guid_section_m1.add_widget(self._guidance_m1_container)
        parent_layout.addWidget(self._guid_section_m1)

        # ── Mode 2: Guidance Video ──────
        self._guid_ff_section_m2 = CollapsibleSection("Guidance Video", collapsed=True)

        guid_col = QVBoxLayout()
        self._use_guidance_m2 = QCheckBox("Enable Guidance Video")
        self._use_guidance_m2.toggled.connect(self._on_guidance_m2_toggled)
        guid_col.addWidget(self._use_guidance_m2)

        self._guidance_m2_container = QWidget()
        gm2_layout = QVBoxLayout(self._guidance_m2_container)
        gm2_layout.setContentsMargins(0, 0, 0, 0)
        gm2_layout.setSpacing(4)
        self._guidance_video_m2 = VideoDropWidget("Guidance Video", thumb_height=180)
        gm2_layout.addWidget(self._guidance_video_m2)
        self._guidance_fps_check_m2 = GuidanceFpsCheck(
            self._guidance_video_m2, lambda: self._fps.value()
        )
        gm2_layout.addWidget(self._guidance_fps_check_m2)
        gm2_opts = QHBoxLayout()
        gm2_opts.addWidget(QLabel("Frame:"))
        self._guidance_source_frame = QComboBox()
        self._guidance_source_frame.addItems(["first", "last"])
        gm2_opts.addWidget(self._guidance_source_frame)
        self._gen_static_m2_btn = QPushButton("Preview Static from Frame")
        self._gen_static_m2_btn.setToolTip("Pre-generate static guidance tensor and preview it")
        self._gen_static_m2_btn.clicked.connect(self._on_generate_static_guidance_m2)
        gm2_opts.addWidget(self._gen_static_m2_btn)
        gm2_opts.addStretch()
        gm2_layout.addLayout(gm2_opts)
        self._guidance_m2_container.setVisible(False)
        guid_col.addWidget(self._guidance_m2_container)
        guid_col.addStretch()

        self._guid_ff_section_m2.add_layout(guid_col)
        parent_layout.addWidget(self._guid_ff_section_m2)

        # Re-check guidance fps whenever the render frame rate changes (manual
        # edit or backend reconfigure, which calls _fps.setValue()).
        self._fps.valueChanged.connect(self._recheck_guidance_fps)

    def _build_postprocessing_section(self, parent_layout: QVBoxLayout) -> None:
        section = CollapsibleSection("Post-Processing", collapsed=True)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Temporal:"))
        self._temporal_upsampling = QComboBox()
        self._temporal_upsampling.setMinimumWidth(self._W_COMBO_M)
        self._temporal_upsampling.addItems(_TEMPORAL_UPSAMPLING_OPTIONS)
        row1.addWidget(self._temporal_upsampling)
        row1.addWidget(QLabel("Spatial:"))
        self._spatial_upsampling = QComboBox()
        self._spatial_upsampling.setMinimumWidth(self._W_COMBO_M)
        self._spatial_upsampling.addItems(_SPATIAL_UPSAMPLING_OPTIONS)
        row1.addWidget(self._spatial_upsampling)
        row1.addStretch()
        section.add_layout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Film Grain:"))
        self._film_grain_intensity = QDoubleSpinBox()
        self._film_grain_intensity.setMinimumWidth(self._W_DSPIN)
        self._film_grain_intensity.setRange(0, 1)
        self._film_grain_intensity.setDecimals(2)
        self._film_grain_intensity.setSingleStep(0.05)
        self._film_grain_intensity.setValue(0.0)
        row2.addWidget(self._film_grain_intensity)
        row2.addWidget(QLabel("Saturation:"))
        self._film_grain_saturation = QDoubleSpinBox()
        self._film_grain_saturation.setMinimumWidth(self._W_DSPIN)
        self._film_grain_saturation.setRange(0, 1)
        self._film_grain_saturation.setDecimals(2)
        self._film_grain_saturation.setSingleStep(0.05)
        self._film_grain_saturation.setValue(0.0)
        row2.addWidget(self._film_grain_saturation)
        row2.addStretch()
        section.add_layout(row2)

        parent_layout.addWidget(section)

    def _build_lora_section(self, parent_layout: QVBoxLayout) -> None:
        # ── LTX System LoRAs (one CollapsibleSection per category) ──
        from sdqt.models.ltx_system_loras import LTX_SYSTEM_LORAS
        self._ltx_system_sections: list = []
        self._ltx_system_checkboxes: dict[str, QCheckBox] = {}
        self._ltx_system_files: dict[str, str] = {}
        by_cat: dict[str, list] = {}
        for cat, prefix, disp, desc in LTX_SYSTEM_LORAS:
            by_cat.setdefault(cat, []).append((prefix, disp, desc))
        for cat, items in by_cat.items():
            sec = CollapsibleSection(f"LTX System LoRAs — {cat}", collapsed=True)
            for prefix, disp, desc in items:
                cb = QCheckBox(disp)
                cb.setToolTip(desc)
                cb.toggled.connect(self._on_lora_selection_changed)
                sec.add_widget(cb)
                self._ltx_system_checkboxes[prefix] = cb
            parent_layout.addWidget(sec)
            self._ltx_system_sections.append(sec)
        # Add to LTX-only widget list so they hide when Wan is selected.
        if not hasattr(self, "_ltx_only_widgets"):
            self._ltx_only_widgets = []
        self._ltx_only_widgets.extend(self._ltx_system_sections)

        # Per-LoRA input fields (control-video type/strength/position, inpaint
        # mask, audio) that surface only when a system LoRA needing them is on.
        self._build_system_lora_inputs(parent_layout)

        section = CollapsibleSection("LoRA Controls", collapsed=True)

        self._lora_list = QListWidget()
        self._lora_list.setMinimumHeight(200)
        self._lora_list.setMaximumHeight(220)
        self._lora_list.itemChanged.connect(self._on_lora_selection_changed)
        section.add_widget(self._lora_list)

        # Hint label — shows usage tips for selected LoRAs. Secondary text:
        # de-emphasized with color at the 13px floor, never a smaller size.
        self._lora_hint = QLabel("")
        self._lora_hint.setWordWrap(True)
        self._lora_hint.setStyleSheet("color: #aaa; font-size: 13px; padding: 2px 4px;")
        section.add_widget(self._lora_hint)

        # Per-LoRA strength table (parity with Video Extender): one selectable
        # weight spinbox per activated LoRA (two for phased "hi;lo") instead of
        # making the user hand-type the multiplier string. The QLineEdit below
        # stays as the canonical persisted field + power-user manual editor; the
        # table seeds from it and re-emits it in activated-LoRA order on change.
        self._lora_strength_rows: list[dict] = []
        self._lora_strength_empty = None
        self._lora_sync_guard = False
        section.add_widget(QLabel("<b>Per-LoRA Strength</b>"))
        self._lora_strength_container = QWidget()
        self._lora_strength_layout = QVBoxLayout(self._lora_strength_container)
        self._lora_strength_layout.setContentsMargins(0, 0, 0, 0)
        self._lora_strength_layout.setSpacing(6)
        section.add_widget(self._lora_strength_container)

        mult_row = QHBoxLayout()
        mult_row.setSpacing(8)
        mult_row.addWidget(QLabel("Multipliers:"))
        self._lora_multipliers = QLineEdit()
        self._lora_multipliers.setPlaceholderText('e.g. "1.0" or "1;0 0;1" for phased')
        self._lora_multipliers.editingFinished.connect(self._on_multipliers_text_edited)
        mult_row.addWidget(self._lora_multipliers, 1)
        section.add_layout(mult_row)
        self._rebuild_lora_strength_view()

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh LoRAs")
        refresh_btn.clicked.connect(self._refresh_loras)
        btn_row.addWidget(refresh_btn)
        btn_row.addStretch()
        section.add_layout(btn_row)

        parent_layout.addWidget(section)

    def _build_system_lora_inputs(self, parent_layout: QVBoxLayout) -> None:
        """Context-sensitive input fields for system LoRAs that take parameters.

        Each sub-panel is hidden until the matching system LoRA checkbox is on
        (driven by :meth:`_update_system_lora_panels`). The whole section rides
        ``_ltx_only_widgets`` so it disappears for the Wan backend.
        """
        self._sysctl_audio_path = ""
        sec = CollapsibleSection("System LoRA Inputs", collapsed=True)

        # ── Union Control (the Guidance Video is the control source) ──
        self._sysctl_union_panel = QWidget()
        up = QVBoxLayout(self._sysctl_union_panel)
        up.setContentsMargins(0, 0, 0, 0)
        up.setSpacing(4)
        up.addWidget(QLabel("<b>Union Control</b>"))
        ur = QHBoxLayout()
        ur.setSpacing(6)
        ur.addWidget(QLabel("Type:"))
        self._ctl_type = QComboBox()
        self._ctl_type.setMinimumWidth(self._W_COMBO_S)
        self._ctl_type.addItems(["raw", "pose", "depth", "canny"])
        self._ctl_type.setToolTip(
            "raw = feed the guidance video directly as a soft structural reference.\n"
            "pose/depth/canny = extract control maps per frame before conditioning."
        )
        ur.addWidget(self._ctl_type)
        ur.addWidget(QLabel("Strength:"))
        self._ctl_strength = QDoubleSpinBox()
        self._ctl_strength.setMinimumWidth(self._W_DSPIN)
        self._ctl_strength.setRange(0.0, 1.0)
        self._ctl_strength.setDecimals(2)
        self._ctl_strength.setSingleStep(0.05)
        self._ctl_strength.setValue(1.0)
        ur.addWidget(self._ctl_strength)
        ur.addStretch()
        up.addLayout(ur)
        ur2 = QHBoxLayout()
        ur2.setSpacing(6)
        ur2.addWidget(QLabel("Start (s):"))
        self._ctl_start_s = QDoubleSpinBox()
        self._ctl_start_s.setMinimumWidth(self._W_DSPIN)
        self._ctl_start_s.setRange(0.0, 600.0)
        self._ctl_start_s.setDecimals(2)
        self._ctl_start_s.setSingleStep(0.5)
        self._ctl_start_s.setValue(0.0)
        self._ctl_start_s.setToolTip(
            "Offset where the guide begins in the output. 0 = guide from the "
            "first frame. Frames before the guide span free-run."
        )
        ur2.addWidget(self._ctl_start_s)
        ur2.addWidget(QLabel("Ramp fr:"))
        self._ctl_ramp = QSpinBox()
        self._ctl_ramp.setMinimumWidth(self._W_SPIN)
        self._ctl_ramp.setRange(0, 30)
        self._ctl_ramp.setValue(0)
        self._ctl_ramp.setToolTip(
            "Fade control strength in/out over this many frames at each span "
            "edge to avoid a hard pose/structure snap. 0 = off."
        )
        ur2.addWidget(self._ctl_ramp)
        ur2.addStretch()
        up.addLayout(ur2)
        un_note = QLabel(
            "Uses the <b>Guidance Video</b> above as the control source. "
            "Match its FPS to the render rate for frame-accurate motion."
        )
        un_note.setWordWrap(True)
        un_note.setStyleSheet("color: #aaa; font-size: 13px;")
        up.addWidget(un_note)
        sec.add_widget(self._sysctl_union_panel)
        self._sysctl_union_panel.setVisible(False)

        # ── Inpaint mask (backend already consumes these) ──
        self._sysctl_inpaint_panel = QWidget()
        ip = QVBoxLayout(self._sysctl_inpaint_panel)
        ip.setContentsMargins(0, 0, 0, 0)
        ip.setSpacing(4)
        ip.addWidget(QLabel("<b>Inpaint Mask</b>"))
        self._ctl_mask = ImageDropWidget("Mask (white = inpaint)", thumb_height=120)
        ip.addWidget(self._ctl_mask)
        mr = QHBoxLayout()
        mr.setSpacing(6)
        mr.addWidget(QLabel("Masking Strength:"))
        self._ctl_mask_strength = QDoubleSpinBox()
        self._ctl_mask_strength.setMinimumWidth(self._W_DSPIN)
        self._ctl_mask_strength.setRange(0.0, 1.0)
        self._ctl_mask_strength.setDecimals(2)
        self._ctl_mask_strength.setSingleStep(0.05)
        self._ctl_mask_strength.setValue(1.0)
        mr.addWidget(self._ctl_mask_strength)
        mr.addStretch()
        ip.addLayout(mr)
        ip_note = QLabel("Requires a source image (Mode 1). White areas are regenerated.")
        ip_note.setWordWrap(True)
        ip_note.setStyleSheet("color: #aaa; font-size: 13px;")
        ip.addWidget(ip_note)
        sec.add_widget(self._sysctl_inpaint_panel)
        self._sysctl_inpaint_panel.setVisible(False)

        # ── AV talking-head audio ──
        self._sysctl_av_panel = QWidget()
        ap = QVBoxLayout(self._sysctl_av_panel)
        ap.setContentsMargins(0, 0, 0, 0)
        ap.setSpacing(4)
        ap.addWidget(QLabel("<b>AV Talking Head — Audio</b>"))
        ar = QHBoxLayout()
        ar.setSpacing(6)
        self._ctl_audio_label = QLabel("<i>no audio selected</i>")
        self._ctl_audio_label.setStyleSheet("color: #ccc; font-size: 13px;")
        ar.addWidget(self._ctl_audio_label, 1)
        audio_browse = QPushButton("Browse…")
        audio_browse.clicked.connect(self._on_browse_av_audio)
        ar.addWidget(audio_browse)
        audio_clear = QPushButton("Clear")
        audio_clear.clicked.connect(self._on_clear_av_audio)
        ar.addWidget(audio_clear)
        ap.addLayout(ar)
        av_note = QLabel("Drives lip-sync from a soundtrack. Trigger word: 'oh wx person'.")
        av_note.setWordWrap(True)
        av_note.setStyleSheet("color: #aaa; font-size: 13px;")
        ap.addWidget(av_note)
        sec.add_widget(self._sysctl_av_panel)
        self._sysctl_av_panel.setVisible(False)

        # ── Info for weights-only / existing-input LoRAs ──
        self._sysctl_info = QLabel("Enable a System LoRA above to configure its inputs.")
        self._sysctl_info.setWordWrap(True)
        self._sysctl_info.setStyleSheet("color: #aaa; font-size: 13px; padding: 2px 4px;")
        sec.add_widget(self._sysctl_info)

        parent_layout.addWidget(sec)
        self._sysctl_section = sec
        self._ltx_only_widgets.append(sec)

    def _on_browse_av_audio(self) -> None:
        from sdqt.utils.file_dialog import get_open_filename
        path, _ = get_open_filename(
            self, "Select Audio", "", "Audio (*.wav *.mp3 *.flac *.m4a *.aac *.ogg)"
        )
        if path:
            self._sysctl_audio_path = path
            self._ctl_audio_label.setText(Path(path).name)

    def _on_clear_av_audio(self) -> None:
        self._sysctl_audio_path = ""
        self._ctl_audio_label.setText("<i>no audio selected</i>")

    def _update_system_lora_panels(self) -> None:
        """Show only the input panels for the system LoRAs currently enabled."""
        if not hasattr(self, "_sysctl_section"):
            return
        cbs = getattr(self, "_ltx_system_checkboxes", {})
        checked = [p.lower() for p, cb in cbs.items() if cb.isChecked()]
        has = lambda frag: any(frag in p for p in checked)

        union = has("union-control")
        inpaint = has("inpaint_masked")
        av = has("talking-head")
        self._sysctl_union_panel.setVisible(union)
        self._sysctl_inpaint_panel.setVisible(inpaint)
        self._sysctl_av_panel.setVisible(av)

        notes: list[str] = []
        if has("id-lora-celebvhq"):
            notes.append(
                "ID LoRA (CelebV-HQ): VOICE-based identity (reference audio clip) — it "
                "does NOT lock visual appearance. For visual character coherence use a "
                "consistent subject start image / frame-chaining or face-swap post."
            )
        if has("outpaint"):
            notes.append("Outpaint: extends frame edges using your source image.")
        if has("transition"):
            notes.append("Transition: morphs first→last frame (Mode 2). Trigger: 'zhuanchang'.")
        weights_only = [
            name for frag, name in (
                ("hdr", "HDR"), ("refocus", "Refocus"),
                ("uncompress", "Uncompress"), ("distilled", "Distilled"),
                ("galaxy", "Galaxy Ace"),
            ) if has(frag)
        ]
        if weights_only:
            notes.append("Weights-only (no inputs): " + ", ".join(weights_only) + ".")
        if notes:
            self._sysctl_info.setText("\n".join(notes))
        elif not (union or inpaint or av):
            self._sysctl_info.setText("Enable a System LoRA above to configure its inputs.")
        else:
            self._sysctl_info.setText("")
        self._sysctl_info.setVisible(bool(self._sysctl_info.text()))

    def _on_lora_selection_changed(self, item=None) -> None:
        """Update hint text and auto-suggest multipliers when LoRAs are toggled."""
        self._update_system_lora_panels()
        activated = self._get_activated_loras()
        if not activated:
            self._lora_hint.setText("")
            self._rebuild_lora_strength_view()
            return

        hints = []
        has_orbit_pair = False
        for name in activated:
            low = name.lower()
            if "orbit" in low and "high_noise" in low:
                hints.append("Orbit (high noise) — camera orbit for primary transformer")
            elif "orbit" in low and "low_noise" in low:
                hints.append("Orbit (low noise) — camera orbit for transformer_2")
            elif "change_clothes" in low:
                hints.append("Change clothes — smoke transition clothing swap effect")
            else:
                hints.append(name)

        # Check for orbit pair
        lows = [n.lower() for n in activated]
        if any("orbit" in l and "high" in l for l in lows) and \
           any("orbit" in l and "low" in l for l in lows):
            has_orbit_pair = True

        self._lora_hint.setText("\n".join(hints))

        # Auto-suggest phased multipliers for orbit pair
        if has_orbit_pair and not self._lora_multipliers.text().strip():
            # orbit high=phase1, orbit low=phase2
            parts = []
            for name in activated:
                low = name.lower()
                if "orbit" in low and "high" in low:
                    parts.append("1;0")
                elif "orbit" in low and "low" in low:
                    parts.append("0;1")
                else:
                    parts.append("1.0")
            self._lora_multipliers.setText(" ".join(parts))
            self._record_lora_multipliers(activated, " ".join(parts))
            self._lora_multipliers.setToolTip(
                "Auto-filled: orbit high-noise active in phase 1, "
                "orbit low-noise active in phase 2"
            )

        # Realign the per-LoRA strength table to the new activation set.
        self._rebuild_lora_strength_view()

    # -- Source persistence -------------------------------------------------

    @Slot(str)
    def _on_m3_source_loaded(self, path: str) -> None:
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.m3_video_path = path
            cfg.save(self.project_path)
        except Exception:
            pass

    @Slot()
    def _on_m3_source_cleared(self) -> None:
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.m3_video_path = ""
            cfg.save(self.project_path)
        except Exception:
            pass

    # -- Mode toggle -------------------------------------------------------

    @Slot(int, bool)
    def _color_match_ff_to_lf(self) -> None:
        """Color-match the first frame image to the last frame image."""
        ff_path = self._first_frame.image_path
        lf_path = self._last_frame.image_path
        if not ff_path or not lf_path:
            self._show_status("Both frames must be loaded to color match.")
            return
        try:
            from PIL import Image
            import numpy as np
            from color_matcher import ColorMatcher

            src = np.array(Image.open(ff_path).convert("RGB"))
            ref = np.array(Image.open(lf_path).convert("RGB"))
            cm = ColorMatcher()
            result = cm.transfer(src=src, ref=ref, method="mkl")
            result = np.clip(result, 0, 255).astype(np.uint8)
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix="ff_matched_", delete=False,
            )
            Image.fromarray(result).save(tmp.name)
            tmp.close()
            self._ff_original_path = ff_path
            self._first_frame.load_image(tmp.name)
            self._ff_undo_btn.setEnabled(True)
            self._show_status("First frame color-matched to last frame.")
        except Exception as exc:
            self._show_status(f"Color match failed: {exc}")

    def _color_match_lf_to_ff(self) -> None:
        """Color-match the last frame image to the first frame image."""
        ff_path = self._first_frame.image_path
        lf_path = self._last_frame.image_path
        if not ff_path or not lf_path:
            self._show_status("Both frames must be loaded to color match.")
            return
        try:
            from PIL import Image
            import numpy as np
            from color_matcher import ColorMatcher

            src = np.array(Image.open(lf_path).convert("RGB"))
            ref = np.array(Image.open(ff_path).convert("RGB"))
            cm = ColorMatcher()
            result = cm.transfer(src=src, ref=ref, method="mkl")
            result = np.clip(result, 0, 255).astype(np.uint8)
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix="lf_matched_", delete=False,
            )
            Image.fromarray(result).save(tmp.name)
            tmp.close()
            self._lf_original_path = lf_path
            self._last_frame.load_image(tmp.name)
            self._lf_undo_btn.setEnabled(True)
            self._show_status("Last frame color-matched to first frame.")
        except Exception as exc:
            self._show_status(f"Color match failed: {exc}")

    def _undo_ff_match(self) -> None:
        if self._ff_original_path:
            self._first_frame.load_image(self._ff_original_path)
            self._ff_original_path = None
            self._ff_undo_btn.setEnabled(False)
            self._show_status("First frame restored.")

    def _undo_lf_match(self) -> None:
        if self._lf_original_path:
            self._last_frame.load_image(self._lf_original_path)
            self._lf_original_path = None
            self._lf_undo_btn.setEnabled(False)
            self._show_status("Last frame restored.")

    def _on_mode_change(self, id: int, checked: bool) -> None:
        if not checked:
            return
        self._m1_group.setVisible(id == 1)
        self._m2_group.setVisible(id == 2)
        self._m3_group.setVisible(id == 3)
        # Show relevant guidance/final frame sections per mode
        self._guid_section_m1.setVisible(id == 1)
        self._guid_ff_section_m2.setVisible(id == 2)
        # Batch (Mode 1) / Sequential Processing (Mode 2) share one button.
        # Mode 3 hides it entirely.
        if hasattr(self, "_batch_btn"):
            if id == 1:
                self._batch_btn.setText("Batch Generate")
                self._batch_btn.setVisible(True)
            elif id == 2:
                self._batch_btn.setText("Sequential Processing")
                self._batch_btn.setVisible(True)
            else:
                self._batch_btn.setVisible(False)

    # -- Guidance / Final Frame toggles ------------------------------------

    @Slot(bool)
    def _on_guidance_m1_toggled(self, checked: bool) -> None:
        self._guidance_m1_container.setVisible(checked)

    @Slot(bool)
    def _on_guidance_m2_toggled(self, checked: bool) -> None:
        self._guidance_m2_container.setVisible(checked)

    def _recheck_guidance_fps(self) -> None:
        """Re-evaluate both guidance-video fps indicators against the render fps."""
        for attr in ("_guidance_fps_check_m1", "_guidance_fps_check_m2"):
            check = getattr(self, attr, None)
            if check is not None:
                check.recheck()

    # -- Static guidance generation ----------------------------------------

    @Slot()
    def _on_generate_static_guidance(self) -> None:
        """Build a static guidance tensor directly from the Mode 1 source image.

        Uses build_static_tensor to avoid the RGB→YUV→RGB colorspace
        roundtrip that ffmpeg encoding introduces.  The tensor is stored
        in memory and passed directly to the pipeline.
        """
        src = self._image_source.image_path
        if not src or not Path(src).is_file():
            self._show_status("No source image for static guidance.")
            return
        try:
            from PIL import Image
            from supremediffusion.core.guidance import GuidanceGenerator

            img = Image.open(src).convert("RGB")
            num_frames = self._current_duration_frames()

            self._static_guidance_tensor = GuidanceGenerator.build_static_tensor(
                img, num_frames,
            )
            # Key the tensor to (mode, source, frames) so a stale Mode-1
            # tensor can't be injected into a later Mode-2/3 run or after the
            # source / duration changed.
            self._static_guidance_key = (1, src, num_frames)
            # Show source image as preview in guidance widget
            self._show_guidance_preview(img, self._guidance_video_m1)
            self._show_status(
                f"Static guidance ready — {num_frames} frames, "
                f"colors preserved (no video encode)"
            )
        except Exception as e:
            logger.error("Static guidance failed: %s", e, exc_info=True)
            self._show_status(f"Static guidance error: {e}")

    def _on_generate_static_guidance_m2(self) -> None:
        """Build a static guidance tensor from Mode 2 first or last frame."""
        which = self._guidance_source_frame.currentText()
        widget = self._first_frame if which == "first" else self._last_frame
        src = widget.image_path
        if not src or not Path(src).is_file():
            self._show_status(f"No {which} frame set for static guidance.")
            return
        try:
            from PIL import Image
            from supremediffusion.core.guidance import GuidanceGenerator

            img = Image.open(src).convert("RGB")
            num_frames = self._current_duration_frames()

            self._static_guidance_tensor = GuidanceGenerator.build_static_tensor(
                img, num_frames,
            )
            # Key the tensor to (mode, source, frames) — see Mode 1 above.
            self._static_guidance_key = (2, src, num_frames)
            self._show_guidance_preview(img, self._guidance_video_m2)
            self._show_status(
                f"Static guidance ready ({which} frame) — {num_frames} frames"
            )
        except Exception as e:
            logger.error("Static guidance M2 failed: %s", e, exc_info=True)
            self._show_status(f"Static guidance error: {e}")

    def _static_guidance_for(self, mode: int, source: str, num_frames: int):
        """Return the in-memory static guidance tensor only if it was built
        for this exact (mode, source, num_frames); otherwise None.

        Prevents a stale "Preview Static" tensor (a different mode, source
        image, or duration) from being injected into the current run.
        """
        tensor = getattr(self, "_static_guidance_tensor", None)
        if tensor is None:
            return None
        if getattr(self, "_static_guidance_key", None) != (mode, source, num_frames):
            return None
        return tensor

    def _show_guidance_preview(self, pil_img, drop_widget) -> None:
        """Show a PIL image as thumbnail preview in a VideoDropWidget."""
        from PySide6.QtGui import QImage, QPixmap

        img = pil_img.copy()
        # Scale down for thumbnail
        max_h = drop_widget._thumb_height - 4
        ratio = max_h / img.height
        new_w = int(img.width * ratio)
        img = img.resize((new_w, max_h))

        qimg = QImage(
            img.tobytes(), img.width, img.height,
            img.width * 3, QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        drop_widget._thumb.setPixmap(pixmap)
        drop_widget._thumb.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 1px solid #555; border-radius: 4px; }"
        )
        drop_widget._clear_btn.setVisible(True)
        drop_widget._browse_overlay.setVisible(False)

    # -- LoRA management ---------------------------------------------------

    def _refresh_loras(self) -> None:
        from sdqt.widgets.lora_metadata import populate_lora_list_grouped
        from sdqt.models.ltx_system_loras import match_system_lora

        previously_active = self._get_activated_loras()
        # Always scan ltx_lora_dir — system-LoRA checkboxes are LTX-only and
        # need to reflect file presence regardless of which model is currently
        # selected. For non-LTX models we still union with lora_mgr / lora_dir
        # so the user's Wan LoRAs appear in the regular list.
        available: list[str] = []
        from pathlib import Path
        ltx_dir = self.state.global_config.model_paths.get("ltx_lora_dir", "")
        if ltx_dir:
            p = Path(ltx_dir)
            # ltx_lora_dir may be configured as the folder OR as a single LoRA
            # file (e.g. the IC-LoRA path the pipeline loads). Scan the
            # containing folder either way — otherwise the system-LoRA
            # checkboxes never find the files and stay permanently disabled.
            scan_dir = p if p.is_dir() else p.parent
            if scan_dir.is_dir():
                available = sorted(
                    f.name for f in scan_dir.iterdir()
                    if f.suffix.lower() in (".safetensors", ".pt", ".pth")
                )
        is_ltx = (self._model_type.currentData() or "").startswith("ltx")
        if not is_ltx:
            extra: list[str] = []
            lora_mgr = self.state.lora_manager
            if lora_mgr is not None:
                try:
                    extra = lora_mgr.list_available()
                except Exception:
                    extra = []
            if not extra:
                from supremediffusion.models.sd_models import scan_loras
                lora_dir = self.state.global_config.model_paths.get("lora_dir", "")
                if lora_dir:
                    try:
                        extra = scan_loras(lora_dir)
                    except Exception:
                        extra = []
            seen = set(available)
            for name in extra:
                if name not in seen:
                    available.append(name)
                    seen.add(name)

        # Split system LoRAs out of the flat list so they only appear in
        # their dedicated checkbox sections.
        self._ltx_system_files = {}
        user_loras: list[str] = []
        for name in available:
            entry = match_system_lora(name)
            if entry is not None:
                self._ltx_system_files[entry[1]] = name
            else:
                user_loras.append(name)
        # Update system checkboxes — enable if file is present, reflect activation.
        for prefix, cb in self._ltx_system_checkboxes.items():
            fname = self._ltx_system_files.get(prefix)
            cb.blockSignals(True)
            cb.setEnabled(fname is not None)
            cb.setChecked(bool(fname and fname in previously_active))
            cb.blockSignals(False)

        populate_lora_list_grouped(self._lora_list, user_loras, previously_active)

    def _get_activated_loras(self) -> list[str]:
        activated = []
        for i in range(self._lora_list.count()):
            item = self._lora_list.item(i)
            if item.checkState() == Qt.Checked:
                activated.append(item.text())
        # Append any system LoRAs whose checkbox is checked.
        sys_files = getattr(self, "_ltx_system_files", {})
        for prefix, cb in getattr(self, "_ltx_system_checkboxes", {}).items():
            if cb.isChecked() and prefix in sys_files:
                fname = sys_files[prefix]
                if fname not in activated:
                    activated.append(fname)
        return activated

    def _set_activated_loras(self, loras: list[str]) -> None:
        wanted = set(loras)
        for i in range(self._lora_list.count()):
            item = self._lora_list.item(i)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        # Reflect on system checkboxes.
        sys_files = getattr(self, "_ltx_system_files", {})
        for prefix, cb in getattr(self, "_ltx_system_checkboxes", {}).items():
            fname = sys_files.get(prefix)
            cb.blockSignals(True)
            cb.setChecked(bool(fname and fname in wanted))
            cb.blockSignals(False)

    def _record_lora_multipliers(self, loras: list[str], multipliers: str) -> None:
        """Record a name -> multiplier-token map from a positionally-aligned
        (loras, multipliers) pair so collect can re-emit tokens in the
        activated-LoRA order even after the alphabetically-sorted list widget
        reorders the names. Without this the positional ``zip`` in
        ``apply_loras`` attaches scales to the wrong LoRA (e.g. Lightning + SVI
        together).
        """
        tokens = multipliers.strip().split() if multipliers.strip() else []
        if len(tokens) <= 1 and "," in multipliers and ";" not in multipliers:
            tokens = [t.strip() for t in multipliers.split(",") if t.strip()]
        for name, tok in zip(loras, tokens):
            self._lora_mult_map[name] = tok

    def _aligned_lora_multipliers(self, activated: list[str]) -> str:
        """Re-emit the multiplier string with tokens aligned to ``activated``
        order, using the recorded name->token map. Falls back to the raw field
        text when no per-name map is available (e.g. fully hand-typed input).
        """
        if not self._lora_mult_map or not any(
            n in self._lora_mult_map for n in activated
        ):
            return self._lora_multipliers.text()
        return " ".join(self._lora_mult_map.get(n, "1") for n in activated)

    # -- Structured per-LoRA strength table (parity with Video Extender) ----

    @staticmethod
    def _parse_multiplier_entries(text: str) -> list[str]:
        """Split a multiplier string into per-LoRA entries (space- or comma-sep)."""
        s = (text or "").strip()
        if not s:
            return []
        entries = s.split()
        if len(entries) <= 1 and "," in s and ";" not in s:
            entries = [e.strip() for e in s.split(",") if e.strip()]
        return entries

    @staticmethod
    def _entry_phase_values(entry: str) -> list[float]:
        """Parse one entry into its phase float(s); lenient on bad input."""
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

        Seeds each row from the aligned multiplier string (name->token map
        first, so values follow a LoRA across the alphabetical list reorder;
        falls back to the raw field positionally). A LoRA gets a second spinbox
        + a "phase 2" label when its seed entry carries phase syntax ("hi;lo").
        """
        if getattr(self, "_lora_strength_layout", None) is None:
            return
        self._clear_lora_strength_rows()
        active = self._get_activated_loras()
        if not active:
            self._lora_strength_empty = QLabel("<i>No LoRAs activated</i>")
            self._lora_strength_empty.setStyleSheet("color: #aaa; font-size: 13px;")
            self._lora_strength_layout.addWidget(self._lora_strength_empty)
            return

        entries = self._parse_multiplier_entries(self._aligned_lora_multipliers(active))
        rows: list[dict] = []
        for i, name in enumerate(active):
            entry = entries[i] if i < len(entries) else "1.0"
            phase_vals = self._entry_phase_values(entry)
            has_phase = ";" in entry

            row = QHBoxLayout()
            row.setSpacing(8)
            label = QLabel(name)
            label.setToolTip(name)
            label.setMinimumWidth(220)
            label.setMaximumWidth(340)
            row.addWidget(label)
            hi = self._make_strength_spinbox(phase_vals[0])
            row.addWidget(hi)
            lo = None
            if has_phase:
                row.addWidget(QLabel("phase 2:"))
                lo_val = phase_vals[1] if len(phase_vals) > 1 else phase_vals[0]
                lo = self._make_strength_spinbox(lo_val)
                row.addWidget(lo)
            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            self._lora_strength_layout.addWidget(container)
            rows.append({"name": name, "hi": hi, "lo": lo})
        self._lora_strength_rows = rows
        # Normalize the canonical string + map to the rebuilt rows.
        self._emit_lora_multipliers()

    def _emit_lora_multipliers(self) -> None:
        """Assemble the multiplier string from the spinbox rows, in order.

        Writes the activated-order string into the canonical QLineEdit and keeps
        the name->token map in sync so collect_to_config stays aligned.
        """
        if not getattr(self, "_lora_strength_rows", None):
            return
        parts: list[str] = []
        for row in self._lora_strength_rows:
            hi = row["hi"].value()
            lo = row["lo"]
            tok = f"{hi:g};{lo.value():g}" if lo is not None else f"{hi:g}"
            parts.append(tok)
            self._lora_mult_map[row["name"]] = tok
        text = " ".join(parts)
        self._lora_sync_guard = True
        try:
            self._lora_multipliers.setText(text)
        finally:
            self._lora_sync_guard = False

    def _on_strength_spin_changed(self, _value: float) -> None:
        if getattr(self, "_lora_sync_guard", False):
            return
        self._emit_lora_multipliers()

    def _on_multipliers_text_edited(self) -> None:
        """User edited the raw multiplier string — reseed the spinboxes.

        Record the typed tokens into the name->token map first so the rebuild
        (which seeds from the map) reflects what the user typed rather than the
        previous spinbox values — otherwise manual phased entry ("1;0 0;1")
        would be silently overridden.
        """
        if getattr(self, "_lora_sync_guard", False):
            return
        active = self._get_activated_loras()
        self._record_lora_multipliers(active, self._lora_multipliers.text())
        self._rebuild_lora_strength_view()

    # -- Prompt profiles ---------------------------------------------------

    def _profiles_path(self) -> Path:
        return self.project_path / "prompt_profiles.json"

    def _load_profiles_data(self) -> dict:
        p = self._profiles_path()
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    def _save_profiles_data(self, data: dict) -> None:
        p = self._profiles_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    def _refresh_profiles(self) -> None:
        self._profile_combo.clear()
        profiles = self._load_profiles_data()
        self._profile_combo.addItems(sorted(profiles.keys()))
        self._profile_combo.setEditable(True)

    @Slot()
    def _save_profile(self) -> None:
        name = self._profile_combo.currentText().strip()
        if not name:
            return
        profiles = self._load_profiles_data()
        if self._uses_window_fields():
            prompt_text = self._get_lightning_prompt()
        else:
            prompt_text = self._prompt.toPlainText()
        profiles[name] = {
            "prompt": prompt_text,
            "negative_prompt": self._neg_prompt.text(),
        }
        self._save_profiles_data(profiles)
        self._refresh_profiles()
        self._profile_combo.setCurrentText(name)
        self._show_status(f"Saved profile '{name}'.")

    @Slot()
    def _load_profile(self) -> None:
        name = self._profile_combo.currentText().strip()
        profiles = self._load_profiles_data()
        if name in profiles:
            prompt_text = profiles[name].get("prompt", "")
            if self._uses_window_fields():
                self._set_lightning_prompt(prompt_text)
            else:
                self._prompt.setPlainText(prompt_text)
            self._neg_prompt.setText(profiles[name].get("negative_prompt", ""))
            self._show_status(f"Loaded profile '{name}'.")

    @Slot()
    def _delete_profile(self) -> None:
        name = self._profile_combo.currentText().strip()
        profiles = self._load_profiles_data()
        if name in profiles:
            del profiles[name]
            self._save_profiles_data(profiles)
            self._refresh_profiles()
            self._show_status(f"Deleted profile '{name}'.")

    # -- Lightning prompt fields -------------------------------------------

    def apply_prompt_payload(self, payload: dict) -> None:
        """Receive a prompt + params from the Txt2Prompt tab: set the model,
        frames, sliding-window size, fps, prompt mode, and the prompt itself
        (single, per-second, or per-window fields)."""
        key = payload.get("model_key")
        if key:
            idx = self._model_type.findData(key)
            if idx >= 0:
                self._model_type.setCurrentIndex(idx)  # triggers reconfigure
        fps = payload.get("fps")
        if fps:
            self._fps.setValue(int(fps))
        frames = payload.get("frames")
        if frames and hasattr(self, "_duration_custom_frames"):
            self._duration_custom_frames.setValue(int(frames))
            if hasattr(self, "_duration_custom_rb"):
                self._duration_custom_rb.setChecked(True)
        win = payload.get("window_size")
        if win and hasattr(self, "_sliding_window_size"):
            self._sliding_window_size.setValue(int(win))
        # Generation mode (1=single image, 2=first/last frame, 3=video clip).
        tmode = payload.get("target_mode")
        if tmode and hasattr(self, "_mode_group"):
            btn = self._mode_group.button(int(tmode))
            if btn:
                btn.setChecked(True)
        prompt = payload.get("prompt", "")
        if payload.get("single_prompt"):
            # Mode 2 / "overall prompt": force the plain single-prompt box.
            self._prompt_stack.setCurrentIndex(0)
            self._prompt.setPlainText(prompt)
        else:
            if hasattr(self, "_refresh_prompt_mode"):
                self._refresh_prompt_mode()
            if hasattr(self, "_uses_window_fields") and self._uses_window_fields():
                self._set_lightning_prompt(prompt)
            else:
                self._prompt.setPlainText(prompt)
        neg = payload.get("negative_prompt")
        if neg is not None:
            self._neg_prompt.setText(neg)
        self._show_status("Prompt applied from Txt2Prompt.")

    def _is_lightning(self) -> bool:
        """Check if current model is any Lightning v2 variant (merged or LoRA-stack)."""
        return self._model_type.currentData() in (
            "i2v_2_2_lightning_v2",
            "i2v_2_2_lightning_v2_loras",
        )

    def _ltx_window_prompts_active(self) -> bool:
        """LTX with more frames than one window → expose per-window prompts.

        Mirrors the backend gate (``_should_use_sliding_window``): LTX runs an
        external sliding-window loop, so each window gets its own prompt line.
        """
        if getattr(self, "_current_backend", "wan") != "ltx":
            return False
        if not hasattr(self, "_sliding_window_size"):
            return False
        win = self._sliding_window_size.value()
        return win > 0 and self._current_duration_frames() > win

    def _num_ltx_windows(self) -> int:
        """Number of LTX windows for the current duration / window size.

        LTX chains full windows (no latent-overlap trim), so each window adds
        ``window_size`` frames → ``ceil(frames / window_size)``.
        """
        import math
        win = self._sliding_window_size.value() if hasattr(self, "_sliding_window_size") else 0
        frames = self._current_duration_frames()
        if win <= 0 or frames <= win:
            return 1
        return max(1, math.ceil(frames / win))

    def _uses_window_fields(self) -> bool:
        """Whether the per-field prompt page (Wan Lightning per-second OR LTX
        per-window) is the active prompt editor."""
        return self._is_lightning() or self._ltx_window_prompts_active()

    def _refresh_prompt_mode(self) -> None:
        """Re-evaluate which prompt page to show and rebuild its fields.

        Called on model/backend/duration/window-size changes. Safe to call
        early (during construction) — guards on widgets that may not exist yet.
        """
        if not hasattr(self, "_prompt_stack"):
            return
        use_fields = self._uses_window_fields()
        self._prompt_stack.setCurrentIndex(1 if use_fields else 0)
        if use_fields:
            self._rebuild_lightning_fields()
        # window count may have changed -> rebuild per-window anchor slots too
        self._maybe_rebuild_anchor_slots()

    @Slot()
    def _on_custom_frames_changed(self, _value: int = 0) -> None:
        """Auto-select the Custom radio and refresh per-window/second fields
        when the custom frame count changes."""
        if hasattr(self, "_duration_custom_rb"):
            self._duration_custom_rb.setChecked(True)
        self._refresh_prompt_mode()

    @Slot(int)
    def _on_model_type_changed(self, _index: int) -> None:
        """Switch prompt area and apply per-model defaults when model changes."""
        model_key = self._model_type.currentData() or ""
        backend = get_video_backend(model_key)

        # Backend switching (resolution/duration/visibility)
        if backend != self._current_backend:
            self._reconfigure_for_backend(backend)

        is_lightning = self._is_lightning()
        self._refresh_prompt_mode()
        if is_lightning:
            self._enhance.set_video_style("wan_lightning")
        elif backend == "ltx":
            self._enhance.set_video_style("ltx")
        else:
            self._enhance.set_video_style("wan")

        # Apply per-model recommended defaults
        from sdqt.widgets.generation_params import MODEL_DEFAULTS
        defaults = MODEL_DEFAULTS.get(model_key)
        if defaults:
            self._steps.setValue(defaults["num_inference_steps"])
            self._guidance_scale.setValue(defaults["guidance_scale"])
            self._guidance2_scale.setValue(defaults["guidance2_scale"])
            self._flow_shift.setValue(defaults["flow_shift"])
            self._guidance_phases.setValue(defaults["guidance_phases"])
            self._switch_threshold.setValue(defaults["switch_threshold"])

        # Show the SVI section only when an SVI model is active. Auto-expand
        # on first reveal; honor manual collapse afterward.
        is_svi = model_key in SVI_MODEL_KEYS
        if hasattr(self, "_svi_section"):
            # Sliding-window controls apply to SVI (Wan) and to LTX (which also
            # runs an external window loop). Show for either.
            show = (is_svi and backend == "wan") or backend == "ltx"
            if show and getattr(self, "_svi_first_show", True):
                self._svi_section.set_collapsed(False)
                self._svi_first_show = False
            self._svi_section.setVisible(show)

        # Auto-prefill the SVI 2 Pro LoRAs (HIGH+LOW noise) and wan2gp
        # phase multipliers when the LoRA-based variant is selected and no
        # SVI LoRAs are already activated.
        if model_key == "i2v_2_2_svi_2_pro" and getattr(self, "_lora_list", None) is not None:
            already_active = [
                self._lora_list.item(i).text()
                for i in range(self._lora_list.count())
                if self._lora_list.item(i).checkState() == Qt.Checked
            ]
            if not any(name.startswith("SVI_Wan2.2-I2V-A14B") for name in already_active):
                # Refresh the LoRA list so the SVI files appear, then check them
                if hasattr(self, "_refresh_loras"):
                    try:
                        self._refresh_loras()
                    except Exception:
                        pass
                wanted = set(SVI_2_PRO_LORAS)
                # Map each preset stem to its positionally-aligned multiplier
                # token so the on-disk filename (with extension) gets the right
                # phase weight regardless of list-widget sort order.
                svi_tokens = SVI_2_PRO_LORA_MULTIPLIERS.split()
                stem_to_token = {
                    SVI_2_PRO_LORAS[j]: svi_tokens[j]
                    for j in range(min(len(SVI_2_PRO_LORAS), len(svi_tokens)))
                }
                for i in range(self._lora_list.count()):
                    item = self._lora_list.item(i)
                    text = item.text()
                    stem = text.rsplit(".", 1)[0]
                    if stem in wanted or text in wanted:
                        item.setCheckState(Qt.Checked)
                        tok = stem_to_token.get(stem) or stem_to_token.get(text)
                        if tok:
                            self._lora_mult_map[text] = tok
                if (
                    getattr(self, "_lora_multipliers", None) is not None
                    and not self._lora_multipliers.text().strip()
                ):
                    self._lora_multipliers.setText(SVI_2_PRO_LORA_MULTIPLIERS)
                    self._rebuild_lora_strength_view()

        # Auto-prefill the standalone Lightning v2 LoRAs + per-phase weight
        # multipliers ("0.7;0 0;1.0") when the LoRA-stack Lightning variant
        # is selected and Lightning LoRAs aren't already activated.
        if (
            model_key == "i2v_2_2_lightning_v2_loras"
            and getattr(self, "_lora_list", None) is not None
        ):
            already_active = [
                self._lora_list.item(i).text()
                for i in range(self._lora_list.count())
                if self._lora_list.item(i).checkState() == Qt.Checked
            ]
            if not any(
                name.startswith("Wan2.2-Lightning_I2V") for name in already_active
            ):
                if hasattr(self, "_refresh_loras"):
                    try:
                        self._refresh_loras()
                    except Exception:
                        pass
                wanted = set(LIGHTNING_V2_I2V_LORAS)
                # Map each preset stem to its positionally-aligned multiplier
                # token (HIGH=phase1, LOW=phase2) so the list-widget sort order
                # can't desync the apply_loras positional zip.
                light_tokens = LIGHTNING_V2_I2V_LORA_MULTIPLIERS.split()
                stem_to_token = {
                    LIGHTNING_V2_I2V_LORAS[j]: light_tokens[j]
                    for j in range(min(len(LIGHTNING_V2_I2V_LORAS), len(light_tokens)))
                }
                for i in range(self._lora_list.count()):
                    item = self._lora_list.item(i)
                    text = item.text()
                    stem = text.rsplit(".", 1)[0]
                    if stem in wanted or text in wanted:
                        item.setCheckState(Qt.Checked)
                        tok = stem_to_token.get(stem) or stem_to_token.get(text)
                        if tok:
                            self._lora_mult_map[text] = tok
                if (
                    getattr(self, "_lora_multipliers", None) is not None
                    and not self._lora_multipliers.text().strip()
                ):
                    self._lora_multipliers.setText(LIGHTNING_V2_I2V_LORA_MULTIPLIERS)
                    self._rebuild_lora_strength_view()

    def _reconfigure_for_backend(self, backend: str) -> None:
        """Swap resolution/duration presets and show/hide backend-specific controls."""
        self._current_backend = backend
        is_ltx = backend == "ltx"

        # Swap resolution presets
        current_res = self._resolution.currentText().split(" ")[0]
        presets = _LTX_RESOLUTION_PRESETS if is_ltx else _WAN_RESOLUTION_PRESETS
        self._resolution.blockSignals(True)
        self._resolution.clear()
        self._resolution.addItems(presets)
        for i, p in enumerate(presets):
            if p.startswith(current_res):
                self._resolution.setCurrentIndex(i)
                break
        self._resolution.blockSignals(False)

        # Swap duration presets
        dur_presets = _LTX_DURATION_PRESETS if is_ltx else _WAN_DURATION_PRESETS
        # Remember the active list so the read-back paths resolve a ticked radio
        # against the SAME presets the radios were built from. Without this the
        # read paths used the hardcoded Wan list, so on LTX a radio mapped to the
        # Wan value at that index (ticking "121 frames" actually sent 41).
        self._active_duration_presets = dur_presets
        # Remember whether Custom was active before we tear down the group
        prev_custom = self._duration_group.checkedId() == len(_DURATION_PRESETS)
        prev_custom_value = self._duration_custom_frames.value()
        # Remove old radio buttons (presets + Custom + the spinbox).
        # The spinbox is recreated below so we delete it here to keep order.
        for btn in self._duration_group.buttons():
            self._duration_group.removeButton(btn)
            btn.deleteLater()
        # The duration radios live in their own row (self._timing_dur_row),
        # split out from the Model/Res row to avoid overpacking. Rebuild the
        # radios in THAT layout, inserting before the FPS label.
        p_layout = getattr(self, "_timing_dur_row", None)
        if p_layout:
            # Also remove the orphaned Custom spinbox (it lives in this layout
            # but isn't tracked by the QButtonGroup).
            for i in range(p_layout.count() - 1, -1, -1):
                item = p_layout.itemAt(i)
                w = item.widget() if item else None
                if w is self._duration_custom_frames:
                    p_layout.takeAt(i)
                    w.deleteLater()
                    break

            # Find FPS label to insert before it
            fps_idx = None
            for i in range(p_layout.count()):
                item = p_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    if item.widget().text() == "FPS:":
                        fps_idx = i
                        break
            insert_at = fps_idx if fps_idx is not None else p_layout.count()
            for i, (label, _) in enumerate(dur_presets):
                rb = QRadioButton(label)
                self._duration_group.addButton(rb, i)
                p_layout.insertWidget(insert_at + i, rb)
                if i == 0 and not prev_custom:
                    rb.setChecked(True)

            # Re-add Custom radio + spinbox after the presets
            self._duration_custom_rb = QRadioButton("Custom")
            custom_id = len(dur_presets)  # one past last preset
            self._duration_group.addButton(self._duration_custom_rb, custom_id)
            p_layout.insertWidget(insert_at + len(dur_presets), self._duration_custom_rb)
            self._duration_custom_frames = QSpinBox()
            self._duration_custom_frames.setMinimumWidth(self._W_SPIN)
            # Step: Wan needs (n-1) % 4 == 0; LTX needs (n-1) % 8 == 0.
            self._duration_custom_frames.setSingleStep(8 if is_ltx else 4)
            self._duration_custom_frames.setRange(9, 1009)
            self._duration_custom_frames.setValue(prev_custom_value)
            self._duration_custom_frames.setSuffix(" fr")
            self._duration_custom_frames.setEnabled(prev_custom)
            self._duration_custom_frames.valueChanged.connect(self._on_custom_frames_changed)
            p_layout.insertWidget(insert_at + len(dur_presets) + 1, self._duration_custom_frames)
            if prev_custom:
                self._duration_custom_rb.setChecked(True)

        # FPS range
        if is_ltx:
            self._fps.setRange(1, 50)
            self._fps.setValue(30)
        else:
            self._fps.setRange(1, 60)
            self._fps.setValue(16)

        # Sliding-window section serves both SVI (Wan) and LTX. Seed
        # backend-appropriate defaults + relabel. LTX ignores overlap_noise
        # (its generate() chains via the start image, no latent overlap).
        if hasattr(self, "_sliding_window_size"):
            self._sliding_window_size.blockSignals(True)
            if is_ltx:
                self._svi_section.set_title("LTX — Sliding Window (per-window prompts)")
                # Default to 121 frames (4s @ 30fps) rather than LTX's native
                # 161: the 22B LTX model is tight on 12GB cards, and a shorter
                # window cuts peak activation VRAM while staying long enough to
                # hold prompt coherence. 121 is latent-valid ((121-1) % 8 == 0).
                self._sliding_window_size.setValue(121)
                self._sliding_window_overlap.setValue(8)
                self._sliding_window_discard_last.setValue(0)
                self._sliding_window_overlap_noise.setValue(0.0)
                self._sliding_window_overlap_noise.setEnabled(False)
            else:
                self._svi_section.set_title("SVI 2 Pro — Sliding Window")
                self._sliding_window_overlap_noise.setEnabled(True)
            self._sliding_window_size.blockSignals(False)

        # Guidance scale step size
        self._guidance_scale.setSingleStep(0.5 if is_ltx else 0.1)

        # Show/hide Wan-only widgets
        for w in self._wan_only_widgets:
            w.setVisible(not is_ltx)
        for w in getattr(self, "_ltx_only_widgets", []):
            w.setVisible(is_ltx)

    def _current_duration_frames(self) -> int:
        """Return the active duration in frames, honoring the Custom radio.

        Returns 81 (default) if no radio is selected. Snaps the Custom
        spinbox to backend latent alignment so callers don't have to.
        """
        dur_id = self._duration_group.checkedId()
        if dur_id < 0:
            return 81
        presets = getattr(self, "_active_duration_presets", _DURATION_PRESETS)
        if 0 <= dur_id < len(presets):
            return presets[dur_id][1]
        # Custom — round to nearest valid k*step + 1
        raw = self._duration_custom_frames.value()
        step = self._duration_custom_frames.singleStep() or 4
        n = max(round((raw - 1) / step), 1)
        return n * step + 1

    @Slot(int, bool)
    def _on_duration_toggled(self, _id: int, checked: bool) -> None:
        """Rebuild lightning fields and toggle the Custom spinbox when the
        duration radio selection changes.
        """
        # Enable the Custom spinbox only while Custom is the selected radio.
        is_custom = (
            self._duration_group.checkedId()
            == self._duration_group.id(self._duration_custom_rb)
        )
        if hasattr(self, "_duration_custom_frames"):
            self._duration_custom_frames.setEnabled(is_custom)
        if checked:
            self._refresh_prompt_mode()

    def _rebuild_lightning_fields(self) -> None:
        """Rebuild the per-field prompt editor.

        Two layouts share the same field stack:
        - Wan Lightning: one field per second (``(at N seconds: ...)`` format).
        - LTX multi-window: one field per sliding window (one line per window).
        """
        # Preserve existing text
        old_texts = [f.text() for f in self._lightning_fields]

        # Clear existing
        while self._lightning_fields_layout.count():
            item = self._lightning_fields_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._lightning_fields.clear()

        ltx_windows = self._ltx_window_prompts_active()
        if ltx_windows:
            num_fields = self._num_ltx_windows()
        else:
            # Wan Lightning: one field per second. fps-aware (Lightning is
            # 16fps by default but the user can change it).
            frames = self._current_duration_frames()
            fps = self._fps.value() if hasattr(self, "_fps") else 16
            seconds = max((frames - 1) // max(fps, 1), 0)
            num_fields = seconds + 1  # 0s through Ns inclusive

        for i in range(num_fields):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            if ltx_windows:
                label = QLabel(f"Win {i + 1}:")
                label.setMinimumWidth(48)
                placeholder = f"Prompt for window {i + 1}..."
            else:
                label = QLabel(f"{i}s:")
                label.setMinimumWidth(32)
                placeholder = f"Action at {i} second(s)..."
            row_layout.addWidget(label)
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            row_layout.addWidget(field, 1)
            # Restore previous text if available
            if i < len(old_texts):
                field.setText(old_texts[i])
            self._lightning_fields_layout.addWidget(row_widget)
            self._lightning_fields.append(field)

    def _get_lightning_prompt(self) -> str:
        """Assemble the per-field editor into the pipeline prompt string.

        - Wan Lightning → ``(at N seconds: ...)`` lines (timed slicer).
        - LTX windows   → one line per window, blanks carried forward so a
          window left empty repeats the previous window's prompt (same scene
          continues) and line indices stay aligned to window indices.
        """
        if self._ltx_window_prompts_active():
            texts = [f.text().strip() for f in self._lightning_fields]
            if not any(texts):
                return ""
            first = next(t for t in texts if t)
            filled, prev = [], first
            for t in texts:
                if t:
                    prev = t
                filled.append(prev)
            return "\n".join(filled)

        parts = []
        for i, field in enumerate(self._lightning_fields):
            text = field.text().strip()
            if text:
                parts.append(f"(at {i} seconds: {text})")
        return "\n".join(parts)

    def _set_lightning_prompt(self, prompt: str) -> None:
        """Parse a saved prompt back into the per-field editor.

        Auto-detects format: ``(at N seconds: ...)`` anchors map to per-second
        fields; otherwise each non-empty line maps to a window field in order.
        """
        import re
        prompt = prompt or ""
        if "(at " in prompt and " seconds:" in prompt:
            field_texts: dict[int, str] = {}
            for match in re.finditer(r'\(at (\d+) seconds?: (.+?)\)', prompt):
                sec = int(match.group(1))
                text = match.group(2).strip()
                field_texts[sec] = text
            for i, field in enumerate(self._lightning_fields):
                field.setText(field_texts.get(i, ""))
            return

        # Plain multi-line → one line per window field.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        for i, field in enumerate(self._lightning_fields):
            field.setText(lines[i] if i < len(lines) else "")

    # -- Prompt enhancement ------------------------------------------------

    def _on_enhance(self, which: str, _text: str, style: str) -> None:
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        if which == "pos":
            if self._uses_window_fields():
                current = self._get_lightning_prompt()
            else:
                current = self._prompt.toPlainText()
        else:
            current = self._neg_prompt.text()

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
        if which == "pos":
            if self._uses_window_fields():
                self._set_lightning_prompt(result)
            else:
                self._prompt.setPlainText(result)
        else:
            self._neg_prompt.setText(result)
        self._show_status("Prompt enhanced.")

    def _on_enhance_error(self, msg: str) -> None:
        self._enhance.set_enabled_buttons(True)
        self._show_status(f"Enhance error: {msg}")

    # -- Denoising loop gating ---------------------------------------------

    def _on_quality_toggled(self, enabled: bool) -> None:
        allow = enabled and self._loop_is_wan2gp
        for ctrl in self._quality_controls:
            ctrl.setEnabled(allow)

    def _on_nag_tea_toggled(self, enabled: bool) -> None:
        allow = enabled and self._loop_is_wan2gp
        for ctrl in self._nag_controls:
            ctrl.setEnabled(allow)

    def set_denoising_loop(self, loop: str) -> None:
        """Hard-gate wan2gp-only controls based on global denoising loop setting."""
        self._loop_is_wan2gp = (loop == "wan2gp")
        self._quality_enable.setEnabled(self._loop_is_wan2gp)
        self._nag_tea_enable.setEnabled(self._loop_is_wan2gp)
        self._on_quality_toggled(self._quality_enable.isChecked())
        self._on_nag_tea_toggled(self._nag_tea_enable.isChecked())

    # -- Collect config from UI --------------------------------------------

    def _collect_config(self, cfg: ProjectConfig) -> ProjectConfig:
        """Populate all ProjectConfig fields from current UI state."""
        mode = self._mode_group.checkedId()
        cfg.mode = mode
        if self._uses_window_fields():
            cfg.prompt = self._get_lightning_prompt()
        else:
            cfg.prompt = self._prompt.toPlainText()
        cfg.negative_prompt = self._neg_prompt.text()

        # Model & Output — store the internal model_type KEY (e.g.
        # "i2v_2_2_svi_2_pro_lightning_v2"), not the display label, so the
        # backend's model_type checks (SVI routing, MODEL_TYPE_TO_FEATURE)
        # work. Falls back to display text if userData wasn't set.
        cfg.model_type = self._model_type.currentData() or self._model_type.currentText()
        cfg.resolution = self._resolution.currentText().split(" ")[0]
        cfg.video_length = self._current_duration_frames()
        cfg.fps = self._fps.value()

        # SVI 2 Pro — sliding window + anchor (saved regardless of which
        # model is selected, so switching models doesn't lose settings).
        if hasattr(self, "_sliding_window_size"):
            cfg.sliding_window_size = self._sliding_window_size.value()
            cfg.sliding_window_overlap = self._sliding_window_overlap.value()
            cfg.sliding_window_discard_last_frames = self._sliding_window_discard_last.value()
            cfg.sliding_window_color_correction_strength = self._sliding_window_color_correction.value()
            cfg.sliding_window_overlap_noise = self._sliding_window_overlap_noise.value()
            anchor_idx = self._anchor_image_mode.currentIndex()
            cfg.anchor_image_mode = (
                _ANCHOR_MODES[anchor_idx][1]
                if 0 <= anchor_idx < len(_ANCHOR_MODES)
                else "start"
            )
            cfg.anchor_image_path = self._anchor_image_path.text().strip()

        # Color-Match-section anchor widgets (always visible — Phase 4b multi-ref).
        # When the user changes the always-visible widget, mirror its value
        # into the cfg directly (the legacy SVI widget gets synced via the
        # change handler).
        if hasattr(self, "_anchor_image_mode_cc"):
            anchor_idx = self._anchor_image_mode_cc.currentIndex()
            cfg.anchor_image_mode = (
                _ANCHOR_MODES[anchor_idx][1]
                if 0 <= anchor_idx < len(_ANCHOR_MODES)
                else cfg.anchor_image_mode
            )
        if hasattr(self, "_anchor_image_path_cc"):
            txt = self._anchor_image_path_cc.text().strip()
            if txt:
                cfg.anchor_image_path = txt

        # Auto Color Match — always-on, post-decode color transfer
        if hasattr(self, "_color_correction"):
            cfg.color_correction_strength = self._color_correction.value()
            cfg.color_correction_method = (
                self._color_correction_method.currentData()
                or self._color_correction_method.currentText()
            )
        if hasattr(self, "_color_anchor_persistence"):
            cfg.color_anchor_persistence = self._color_anchor_persistence.value()
        # Wan 2.2 presets (Acceleration + Anti-drift)
        if hasattr(self, "_acceleration_preset"):
            cfg.acceleration_preset = self._acceleration_preset.currentData() or "off"
        if hasattr(self, "_antidrift_preset"):
            cfg.antidrift_preset = self._antidrift_preset.currentData() or "off"
        # Output Quality (post-processing pipeline preset)
        if hasattr(self, "_output_quality"):
            cfg.output_quality = self._output_quality.currentData() or "standard"

        # Advanced
        cfg.num_inference_steps = self._steps.value()
        cfg.guidance_scale = self._guidance_scale.value()
        cfg.guidance2_scale = self._guidance2_scale.value()
        cfg.alt_guidance_scale = self._alt_guidance_scale.value()
        cfg.flow_shift = self._flow_shift.value()
        cfg.sample_solver = self._solver.currentText()
        cfg.seed = self._seed.value()
        cfg.denoising_strength = self._denoising_strength.value()

        cfg.guidance_phases = self._guidance_phases.value()
        cfg.switch_threshold = self._switch_threshold.value()
        # TEA Cache — always saved (independent speed optimization)
        cfg.tea_cache_setting = self._tea_cache.currentText()
        cfg.tea_cache_start_step_perc = self._tea_start_perc.value()
        # NAG gate — requires wan2gp denoising loop
        cfg.nag_tea_enabled = self._nag_tea_enable.isChecked()
        if cfg.nag_tea_enabled:
            cfg.NAG_scale = self._nag_scale.value()
            cfg.NAG_tau = self._nag_tau.value()
            cfg.NAG_alpha = self._nag_alpha.value()
        else:
            cfg.NAG_scale = 0.0
            cfg.NAG_tau = 0.0
            cfg.NAG_alpha = 0.0

        # Quality gate
        cfg.quality_overrides_enabled = self._quality_enable.isChecked()
        if cfg.quality_overrides_enabled:
            cfg.cfg_star_switch = self._cfg_star_switch.value()
            cfg.cfg_zero_step = self._cfg_zero_step.value()
            cfg.slg_switch = 1 if self._slg_switch.isChecked() else 0
            layers_text = self._slg_layers.text().strip()
            if layers_text:
                try:
                    cfg.slg_layers = [int(x.strip()) for x in layers_text.split(",") if x.strip()]
                except ValueError:
                    pass
            cfg.slg_start_perc = self._slg_start_perc.value()
            cfg.slg_end_perc = self._slg_end_perc.value()
            cfg.apg_switch = 1 if self._apg_switch.isChecked() else 0
            cfg.motion_amplitude = self._motion_amplitude.value()
            cfg.discard_last_frames = self._discard_last_frames.value()
            cfg.self_refiner_setting = self._self_refiner.currentIndex()
            cfg.self_refiner_uncertainty = self._refiner_uncertainty.value()
            cfg.self_refiner_certainty_skip = self._refiner_certainty_skip.value()
        else:
            cfg.slg_switch = 0
            cfg.slg_layers = []
            cfg.slg_start_perc = 0
            cfg.slg_end_perc = 100
            cfg.apg_switch = 0
            cfg.cfg_star_switch = 0
            cfg.cfg_zero_step = 0
            cfg.motion_amplitude = 1.0
            cfg.discard_last_frames = 0
            cfg.self_refiner_setting = 0
            cfg.self_refiner_uncertainty = 0.0
            cfg.self_refiner_certainty_skip = 0.999

        # LTX tail-trim ("Discard") — Quality is Wan-only, so on LTX the gate's
        # else branch always runs and would zero discard_last_frames. Resolve
        # the LTX value after the gate, keyed off the active backend (not widget
        # visibility), so the distilled tail-trim actually reaches the backend.
        if self._current_backend == "ltx":
            cfg.discard_last_frames = int(self._ltx_discard_last.value())

        # Guidance
        cfg.use_guidance_m1 = self._use_guidance_m1.isChecked()
        cfg.guidance_video_path_m1 = self._guidance_video_m1.file_path or ""
        cfg.use_guidance_m2 = self._use_guidance_m2.isChecked()
        cfg.guidance_video_path_m2 = self._guidance_video_m2.file_path or ""
        cfg.guidance_video_frame = self._guidance_source_frame.currentText()

        # Post-processing
        cfg.temporal_upsampling = self._temporal_upsampling.currentText()
        cfg.spatial_upsampling = self._spatial_upsampling.currentText()
        cfg.film_grain_intensity = self._film_grain_intensity.value()
        cfg.film_grain_saturation = self._film_grain_saturation.value()

        # LoRA
        cfg.activated_loras = self._get_activated_loras()
        # Re-emit multipliers aligned to activated_loras order. The list widget
        # is sorted alphabetically while preset/auto-suggest emit tokens in
        # their own order, which would otherwise misalign apply_loras' positional
        # zip (e.g. Lightning + SVI together).
        cfg.loras_multipliers = self._aligned_lora_multipliers(cfg.activated_loras)
        self._lora_multipliers.blockSignals(True)
        self._lora_multipliers.setText(cfg.loras_multipliers)
        self._lora_multipliers.blockSignals(False)

        # System-LoRA inputs (Union Control / Inpaint / AV). Harmless for Wan
        # and for runs where the matching LoRA isn't enabled — the backend only
        # acts on these when the relevant IC-LoRA is in activated_loras.
        if hasattr(self, "_ctl_type"):
            cfg.ltx_control_type = self._ctl_type.currentText()
            cfg.ltx_control_strength = float(self._ctl_strength.value())
            cfg.ltx_control_start_seconds = float(self._ctl_start_s.value())
            cfg.ltx_control_ramp_frames = int(self._ctl_ramp.value())
            cfg.ltx_audio_path = getattr(self, "_sysctl_audio_path", "") or ""
            # The backend inpaint branch is NOT gated on the inpaint LoRA, so
            # only persist the mask when an inpaint LoRA is actually enabled —
            # otherwise a leftover mask would silently trigger inpainting.
            inpaint_on = any(
                "inpaint_masked" in str(n).lower() for n in cfg.activated_loras
            )
            if inpaint_on:
                cfg.ltx_inpaint_mask_path = self._ctl_mask.image_path or ""
                cfg.masking_strength = float(self._ctl_mask_strength.value())
            else:
                cfg.ltx_inpaint_mask_path = ""

        # Source paths
        if mode == 1:
            cfg.image_path = self._image_source.image_path or ""
        elif mode == 2:
            cfg.first_frame_path = self._first_frame.image_path or ""
            cfg.last_frame_path = self._last_frame.image_path or ""
        elif mode == 3:
            cfg.m3_video_path = self._m3_video.file_path or ""

        return cfg

    # -- Generation --------------------------------------------------------

    def _build_mode1_kwargs(self, image_path: str, use_guidance: bool) -> dict:
        """Build InferenceWorker kwargs for a Mode 1 generation.

        Called by the regular Generate button (with the current UI state)
        and by Batch Generate (with an image path override per iteration).
        """
        pipeline = self.state.pipeline
        if pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        cfg = ProjectConfig.load(self.project_path)
        cfg = self._collect_config(cfg)
        # Force the requested guidance state regardless of the tab's checkbox
        cfg.use_guidance_m1 = bool(use_guidance)
        if not use_guidance:
            # Clear any stale guidance file so the worker skips it entirely
            cfg.guidance_video_path_m1 = ""

        kwargs = dict(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=1,
            global_config=self.state.global_config,
            image_path=image_path,
        )
        if use_guidance:
            # Let the worker auto-generate a static guidance video from the image
            kwargs["generate_guidance"] = True
        return kwargs

    def _build_mode2_kwargs(
        self,
        first_path: str,
        last_path: str,
        use_guidance: bool,
        guidance_frame: str = "first",
    ) -> dict:
        """Build InferenceWorker kwargs for a Mode 2 (First/Last frame) run.

        Called by the regular Generate button (with the current UI state)
        and by Sequential Processing (with pair overrides per iteration).
        """
        pipeline = self.state.pipeline
        if pipeline is None:
            raise RuntimeError("Pipeline not loaded")

        cfg = ProjectConfig.load(self.project_path)
        cfg = self._collect_config(cfg)
        cfg.use_guidance_m2 = bool(use_guidance)
        if not use_guidance:
            cfg.guidance_video_path_m2 = ""

        kwargs = dict(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=2,
            global_config=self.state.global_config,
            first_frame_path=first_path,
            last_frame_path=last_path,
        )
        if use_guidance:
            kwargs["generate_guidance"] = True
            kwargs["guidance_frame"] = (
                guidance_frame if guidance_frame in ("first", "last") else "first"
            )
        return kwargs

    @Slot()
    def _on_generate(self) -> None:
        if not self.state.acquire_generation("Video"):
            owner = self.state.generation_owner
            self._show_status(f"GPU is busy — {owner} is generating. Wait for it to finish.")
            return
        from sdqt.models.manager import check_and_prompt_download
        from sdqt.widgets.generation_params import MODEL_TYPE_TO_FEATURE
        model_key = self._model_type.currentData() or "i2v_2_2"
        feature = MODEL_TYPE_TO_FEATURE.get(model_key, "video_gen")
        if not check_and_prompt_download(
            feature, self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            self.state.release_generation("Video")
            return

        backend = get_video_backend(model_key)
        pipeline = self.state.pipeline
        if pipeline is None or self.state.active_video_backend != backend:
            self._show_status(f"Loading {backend.upper()} video pipelines...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
                pipeline = self.state.pipeline
            except Exception as exc:
                self._show_status(f"Failed to load video pipelines: {exc}")
                self.state.release_generation("Video")
                return
        if pipeline is None:
            self._show_status("Pipeline not loaded. Check Settings.")
            self.state.release_generation("Video")
            return

        mode = self._mode_group.checkedId()
        cfg = ProjectConfig.load(self.project_path)
        cfg = self._collect_config(cfg)
        cfg.video_backend = backend
        cfg.save(self.project_path)

        kwargs = dict(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=mode,
            global_config=self.state.global_config,
        )
        if mode == 1:
            img = self._image_source.image_path
            if not img:
                self._show_status("No source image selected.")
                self.state.release_generation("Video")
                return
            kwargs["image_path"] = img
            if cfg.use_guidance_m1:
                # Prefer the in-memory tensor (no colorspace loss) — but only
                # if it was built for this exact mode/source/duration.
                static_tensor = self._static_guidance_for(1, img, cfg.video_length)
                if static_tensor is not None:
                    kwargs["guidance_tensor"] = static_tensor
                    # Built from the source image — already in its color
                    # profile, so the pipeline must skip the guidance
                    # pre-match (huge Lab allocation + pointless).
                    kwargs["guidance_from_source"] = True
                elif self._guidance_video_m1.file_path and Path(self._guidance_video_m1.file_path).is_file():
                    # Fallback: load an externally provided guidance video
                    from supremediffusion.core.guidance import GuidanceGenerator
                    kwargs["guidance_tensor"] = GuidanceGenerator.load_guidance_video(
                        self._guidance_video_m1.file_path
                    )
                else:
                    # Auto-generate static guidance from source image
                    kwargs["generate_guidance"] = True
        elif mode == 2:
            ff = self._first_frame.image_path
            lf = self._last_frame.image_path
            if not ff or not lf:
                self._show_status("Both first and last frames required.")
                self.state.release_generation("Video")
                return
            kwargs["first_frame_path"] = ff
            kwargs["last_frame_path"] = lf
            if cfg.use_guidance_m2:
                gv_path = self._guidance_video_m2.file_path
                # The Mode-2 static tensor is built from the first OR last
                # frame (whichever _guidance_source_frame selected); match the
                # key against that same source so a stale tensor is ignored.
                which = self._guidance_source_frame.currentText()
                m2_src = ff if which == "first" else lf
                static_tensor = self._static_guidance_for(2, m2_src, cfg.video_length)
                if static_tensor is not None:
                    kwargs["guidance_tensor"] = static_tensor
                    kwargs["guidance_frame"] = which
                    # Built from the first/last source frame — skip pre-match.
                    kwargs["guidance_from_source"] = True
                elif gv_path and Path(gv_path).is_file():
                    from supremediffusion.core.guidance import GuidanceGenerator
                    kwargs["guidance_tensor"] = GuidanceGenerator.load_guidance_video(gv_path)
                    kwargs["guidance_frame"] = self._guidance_source_frame.currentText()
                else:
                    # Auto-generate static guidance from selected frame
                    kwargs["generate_guidance"] = True
                    kwargs["guidance_frame"] = self._guidance_source_frame.currentText()
        elif mode == 3:
            vid = self._m3_video.file_path
            if not vid:
                self._show_status("No source video selected.")
                self.state.release_generation("Video")
                return
            kwargs["video_path"] = vid
            kwargs["start_frame"] = self._m3_start_frame.value()
            kwargs["end_frame"] = self._m3_end_frame.value()

        self._gen_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Starting generation...")

        worker = InferenceWorker(**kwargs, parent=self)
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_gen_done)
        worker.error.connect(self._on_gen_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_gen_done(self, result: object) -> None:
        # InferenceWorker emits a GenResult — a str subclass that is the output
        # path and also carries the actually-used seed (after -1 randomization)
        # so it can be reused. getattr keeps a plain str safe too.
        result_path = str(result)
        seed = getattr(result, "resolved_seed", -1)
        if isinstance(seed, int) and seed >= 0:
            self._resolved_seed = seed
            self._reuse_seed_btn.setEnabled(True)
        self.state.release_generation("Video")
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._preview.load_video(result_path, auto_play=True)
        self._show_status("Generation complete.")
        self._save_clip_btn.setEnabled(True)
        self._dl_frame_btn.setEnabled(True)
        self._save_frame_btn.setEnabled(True)
        # Persist preview path in project config
        try:
            cfg = ProjectConfig.load(self.project_path)
            mode = self._mode_group.checkedId()
            field = "preview_path_m1" if mode == 1 else "preview_path_m2"
            setattr(cfg, field, result_path)
            cfg.save(self.project_path)
        except Exception:
            pass
        # Embed generation metadata into the video file
        try:
            from supremediffusion.utils.video import write_video_metadata
            cfg = ProjectConfig.load(self.project_path)
            meta = {
                "prompt": cfg.prompt or "",
                "negative_prompt": cfg.negative_prompt or "",
                "model_type": cfg.model_type or "",
                "resolution": cfg.resolution or "",
                "fps": cfg.fps,
                "video_length": cfg.video_length,
                "steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
                "guidance2_scale": cfg.guidance2_scale,
                "flow_shift": cfg.flow_shift,
                "solver": cfg.sample_solver or "",
                # Prefer the resolved seed so randomized (-1) runs record the
                # value that was actually used, not the placeholder.
                "seed": self._resolved_seed if self._resolved_seed >= 0 else cfg.seed,
                "denoising_strength": cfg.denoising_strength,
                "mode": cfg.mode,
            }
            write_video_metadata(result_path, meta)
        except Exception:
            pass

    @Slot()
    def _on_reuse_seed(self) -> None:
        """Write the last generation's resolved seed into the seed control."""
        if self._resolved_seed >= 0:
            self._seed.setValue(self._resolved_seed)
            self._show_status(f"Reusing seed {self._resolved_seed}.")

    def _on_gen_error(self, msg: str) -> None:
        self.state.release_generation("Video")
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_batch_generate(self) -> None:
        """Open the Batch Generate dialog and kick off a batch run (Mode 1 only)."""
        try:
            self._do_batch_generate()
        except Exception as exc:
            logger.exception("Batch Generate failed to start")
            self._show_status(f"Batch failed to start: {exc}")
            try:
                self.state.release_generation("Video")
            except Exception:
                pass
            if hasattr(self, "_gen_btn"):
                self._gen_btn.setVisible(True)
            if hasattr(self, "_batch_btn"):
                self._batch_btn.setVisible(self._mode_group.checkedId() == 1)
            if hasattr(self, "_abort_btn"):
                self._abort_btn.setVisible(False)

    def _do_batch_generate(self) -> None:
        if self._mode_group.checkedId() != 1:
            self._show_status("Batch Generate is only available in Mode 1.")
            return

        from sdqt.tabs.batch_generate_dialog import BatchGenerateDialog

        saved = {}
        try:
            saved = dict(self.state.global_config.batch_generate_settings or {})
        except Exception:
            pass

        dlg = BatchGenerateDialog(self.state, last_settings=saved, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        settings = dlg.get_settings()
        if not settings or not settings.get("images"):
            self._show_status("Batch cancelled — no images resolved.")
            return

        # Persist the dialog choices for next time (drop the expanded image list)
        try:
            to_save = {k: v for k, v in settings.items() if k != "images"}
            self.state.global_config.batch_generate_settings = to_save
            self.state.global_config.save()
        except Exception as exc:
            logger.warning("Failed to persist batch settings: %s", exc)

        logger.info("Batch Generate: %d images, output=%s, prefix=%s, "
                    "guidance=%s, cc=%s, loop=%s, pp=%s, continue=%s",
                    len(settings["images"]), settings.get("output_dir"),
                    settings.get("prefix"), settings.get("use_guidance"),
                    settings.get("do_color_correct"), settings.get("do_loop"),
                    settings.get("do_pp"), settings.get("continue_numbering"))

        # Ensure the pipeline is loaded (same guard as _on_generate)
        model_key = self._model_type.currentData() or "i2v_2_2"
        backend = get_video_backend(model_key)
        if self.state.pipeline is None or self.state.active_video_backend != backend:
            self._show_status(f"Loading {backend.upper()} video pipelines...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
            except Exception as exc:
                self._show_status(f"Failed to load video pipelines: {exc}")
                return
        if self.state.pipeline is None:
            self._show_status("Pipeline not loaded. Check Settings.")
            return

        if not self.state.acquire_generation("Video"):
            owner = self.state.generation_owner
            self._show_status(f"GPU is busy — {owner} is generating. Wait for it to finish.")
            return

        from sdqt.workers.batch_generate import BatchGenerateController
        try:
            controller = BatchGenerateController(self, settings, parent=self)
        except Exception as exc:
            self.state.release_generation("Video")
            logger.exception("BatchGenerateController init failed")
            self._show_status(f"Batch init failed: {exc}")
            return

        def _on_progress(i: int, n: int, desc: str) -> None:
            self._show_status(desc)

        def _on_item_done(path: str) -> None:
            logger.info("Batch item done: %s", path)
            try:
                self._preview.load_video(path, auto_play=False)
            except Exception:
                pass

        def _on_batch_done(ok_count: int, err_count: int) -> None:
            self.state.release_generation("Video")
            self._batch_controller = None
            self._gen_btn.setVisible(True)
            self._batch_btn.setVisible(self._mode_group.checkedId() == 1)
            self._abort_btn.setVisible(False)
            self._show_status(
                f"Batch complete: {ok_count} ok, {err_count} skipped."
            )

        def _on_batch_error(msg: str) -> None:
            self.state.release_generation("Video")
            self._batch_controller = None
            self._gen_btn.setVisible(True)
            self._batch_btn.setVisible(self._mode_group.checkedId() == 1)
            self._abort_btn.setVisible(False)
            self._show_status(f"Batch failed: {msg}")

        controller.progress.connect(_on_progress)
        controller.item_done.connect(_on_item_done)
        controller.batch_done.connect(_on_batch_done)
        controller.batch_error.connect(_on_batch_error)

        self._batch_controller = controller
        self._gen_btn.setVisible(False)
        self._batch_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status(
            f"Batch starting — {len(settings['images'])} images…"
        )
        controller.start()

    @Slot()
    def _on_batch_or_sequential(self) -> None:
        """Single button dispatcher: Mode 1 → Batch Generate, Mode 2 → Sequential."""
        mode = self._mode_group.checkedId()
        if mode == 1:
            self._on_batch_generate()
        elif mode == 2:
            self._on_sequential_processing()

    @Slot()
    def _on_sequential_processing(self) -> None:
        """Open the Sequential Processing dialog and run a pair-based batch (Mode 2)."""
        try:
            self._do_sequential_processing()
        except Exception as exc:
            logger.exception("Sequential Processing failed to start")
            self._show_status(f"Sequential failed to start: {exc}")
            try:
                self.state.release_generation("Video")
            except Exception:
                pass
            if hasattr(self, "_gen_btn"):
                self._gen_btn.setVisible(True)
            if hasattr(self, "_batch_btn"):
                self._batch_btn.setVisible(self._mode_group.checkedId() in (1, 2))
            if hasattr(self, "_abort_btn"):
                self._abort_btn.setVisible(False)

    def _do_sequential_processing(self) -> None:
        if self._mode_group.checkedId() != 2:
            self._show_status("Sequential Processing is only available in Mode 2.")
            return

        from sdqt.tabs.sequential_process_dialog import SequentialProcessDialog

        saved = {}
        try:
            saved = dict(self.state.global_config.sequential_process_settings or {})
        except Exception:
            pass

        dlg = SequentialProcessDialog(self.state, last_settings=saved, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        settings = dlg.get_settings()
        if not settings or not settings.get("pairs"):
            self._show_status("Sequential cancelled — no pairs resolved.")
            return

        # Persist the dialog choices for next time (drop the expanded pair list)
        try:
            to_save = {k: v for k, v in settings.items() if k != "pairs"}
            self.state.global_config.sequential_process_settings = to_save
            self.state.global_config.save()
        except Exception as exc:
            logger.warning("Failed to persist sequential settings: %s", exc)

        logger.info(
            "Sequential: %d pairs, output=%s, prefix=%s, guidance=%s, "
            "guidance_frame=%s, cc=%s, pp=%s",
            len(settings["pairs"]), settings.get("output_dir"),
            settings.get("prefix"), settings.get("use_guidance"),
            settings.get("guidance_frame"),
            settings.get("do_color_correct"), settings.get("do_pp"),
        )

        # Pipeline guard (same as _on_generate / _on_batch_generate)
        model_key = self._model_type.currentData() or "i2v_2_2"
        backend = get_video_backend(model_key)
        if self.state.pipeline is None or self.state.active_video_backend != backend:
            self._show_status(f"Loading {backend.upper()} video pipelines...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
            except Exception as exc:
                self._show_status(f"Failed to load video pipelines: {exc}")
                return
        if self.state.pipeline is None:
            self._show_status("Pipeline not loaded. Check Settings.")
            return

        if not self.state.acquire_generation("Video"):
            owner = self.state.generation_owner
            self._show_status(f"GPU is busy — {owner} is generating. Wait for it to finish.")
            return

        from sdqt.workers.sequential_process import SequentialProcessingController
        try:
            controller = SequentialProcessingController(self, settings, parent=self)
        except Exception as exc:
            self.state.release_generation("Video")
            logger.exception("SequentialProcessingController init failed")
            self._show_status(f"Sequential init failed: {exc}")
            return

        def _on_progress(i: int, n: int, desc: str) -> None:
            self._show_status(desc)

        def _on_item_done(path: str) -> None:
            logger.info("Sequential item done: %s", path)
            try:
                self._preview.load_video(path, auto_play=False)
            except Exception:
                pass

        def _on_batch_done(ok_count: int, err_count: int) -> None:
            self.state.release_generation("Video")
            self._batch_controller = None
            self._gen_btn.setVisible(True)
            mode = self._mode_group.checkedId()
            self._batch_btn.setVisible(mode in (1, 2))
            if mode == 1:
                self._batch_btn.setText("Batch Generate")
            elif mode == 2:
                self._batch_btn.setText("Sequential Processing")
            self._abort_btn.setVisible(False)
            self._show_status(
                f"Sequential complete: {ok_count} ok, {err_count} skipped."
            )

        def _on_batch_error(msg: str) -> None:
            self.state.release_generation("Video")
            self._batch_controller = None
            self._gen_btn.setVisible(True)
            self._batch_btn.setVisible(self._mode_group.checkedId() in (1, 2))
            self._abort_btn.setVisible(False)
            self._show_status(f"Sequential failed: {msg}")

        controller.progress.connect(_on_progress)
        controller.item_done.connect(_on_item_done)
        controller.batch_done.connect(_on_batch_done)
        controller.batch_error.connect(_on_batch_error)

        self._batch_controller = controller
        self._gen_btn.setVisible(False)
        self._batch_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status(
            f"Sequential starting — {len(settings['pairs'])} pairs…"
        )
        controller.start()

    @Slot()
    def _on_abort(self) -> None:
        if self._batch_controller is not None and self._batch_controller.is_running:
            self._batch_controller.abort()
            return
        if self._worker:
            self._worker.abort()

    # -- Frame capture -----------------------------------------------------

    @Slot()
    def _on_download_frame(self) -> None:
        path = self._preview.capture_frame()
        if path:
            self.download_frame(path)

    @Slot()
    def _on_save_frame(self) -> None:
        path = self._preview.capture_frame()
        if path:
            saved = self.save_frame_to_project(path)
            if saved:
                self._show_status("Saved frame to project.")

    @Slot()
    def _on_save_clip(self) -> None:
        vid = self._preview.video_path
        if not vid or not Path(vid).is_file():
            self._show_status("No clip to save.")
            return
        name = self._clip_name.text().strip()
        if not name:
            name = Path(vid).stem
        clips_dir = self.project_path / "clips" / "generate"
        clips_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(vid).suffix
        dest = clips_dir / f"{name}{ext}"
        idx = 1
        while dest.exists():
            dest = clips_dir / f"{name}_{idx}{ext}"
            idx += 1
        shutil.copy2(vid, dest)
        self._show_status(f"Saved clip: {dest.name}")

    # -- Restore UI from config --------------------------------------------

    def _restore_from_config(self, cfg: ProjectConfig) -> None:
        """Restore all UI controls from a ProjectConfig."""
        self._restoring = True
        try:
            self._restore_from_config_inner(cfg)
        finally:
            self._restoring = False

    def _restore_from_config_inner(self, cfg: ProjectConfig) -> None:
        mode = cfg.mode if cfg.mode in (1, 2, 3) else 1
        self._mode1_rb.setChecked(mode == 1)
        self._mode2_rb.setChecked(mode == 2)
        self._mode3_rb.setChecked(mode == 3)
        self._m1_group.setVisible(mode == 1)
        self._m2_group.setVisible(mode == 2)
        self._m3_group.setVisible(mode == 3)

        self._image_source.load_image(cfg.image_path or "")
        self._first_frame.load_image(cfg.first_frame_path or "")
        self._last_frame.load_image(cfg.last_frame_path or "")
        m3_vid = cfg.m3_video_path or ""
        if m3_vid and Path(m3_vid).is_file():
            self._m3_video.load_file(m3_vid)
        else:
            self._m3_video.clear_file()
        self._neg_prompt.setText(cfg.negative_prompt or "")

        # Model & Output — try internal key first (new format), then fall
        # back to display label (legacy projects stored this way).
        model_key = cfg.model_type or ""
        idx = self._model_type.findData(model_key)
        if idx < 0:
            idx = self._model_type.findText(model_key)
        if idx >= 0:
            self._model_type.setCurrentIndex(idx)

        # Restore prompt (must happen after model_type so Lightning detection works)
        if self._is_lightning():
            self._prompt_stack.setCurrentIndex(1)
            self._rebuild_lightning_fields()
            self._set_lightning_prompt(cfg.prompt or "")
        else:
            self._prompt_stack.setCurrentIndex(0)
            self._prompt.setPlainText(cfg.prompt or "")
        res = cfg.resolution or ""
        idx = self._resolution.findText(res)
        if idx < 0:
            # Prefix match: stored "512x512" matches "512x512 (1:1)"
            for i in range(self._resolution.count()):
                if self._resolution.itemText(i).startswith(res):
                    idx = i
                    break
        if idx >= 0:
            self._resolution.setCurrentIndex(idx)
        # Match a preset radio if the saved frame count is one; otherwise
        # fall back to Custom + load the spinbox.
        matched = False
        for i, (_, frames) in enumerate(getattr(self, "_active_duration_presets", _DURATION_PRESETS)):
            if frames == cfg.video_length:
                btn = self._duration_group.button(i)
                if btn:
                    btn.setChecked(True)
                matched = True
                break
        if not matched and cfg.video_length:
            self._duration_custom_frames.setValue(int(cfg.video_length))
            self._duration_custom_rb.setChecked(True)
        self._fps.setValue(cfg.fps or 16)

        # Advanced
        self._steps.setValue(cfg.num_inference_steps)
        self._guidance_scale.setValue(cfg.guidance_scale)
        self._guidance2_scale.setValue(cfg.guidance2_scale)
        self._alt_guidance_scale.setValue(getattr(cfg, "alt_guidance_scale", 1.0) or 1.0)
        self._ltx_discard_last.setValue(int(getattr(cfg, "discard_last_frames", 0) or 0))
        self._flow_shift.setValue(cfg.flow_shift)
        idx = self._solver.findText(cfg.sample_solver or "unipc")
        if idx >= 0:
            self._solver.setCurrentIndex(idx)
        self._seed.setValue(cfg.seed)
        self._denoising_strength.setValue(cfg.denoising_strength)

        self._guidance_phases.setValue(cfg.guidance_phases)
        self._switch_threshold.setValue(cfg.switch_threshold)
        # NAG / TEA gate
        self._nag_tea_enable.setChecked(getattr(cfg, "nag_tea_enabled", False))
        self._nag_scale.setValue(cfg.NAG_scale)
        self._nag_tau.setValue(cfg.NAG_tau)
        self._nag_alpha.setValue(cfg.NAG_alpha)
        idx = self._tea_cache.findText(cfg.tea_cache_setting or "off")
        if idx >= 0:
            self._tea_cache.setCurrentIndex(idx)
        self._tea_start_perc.setValue(cfg.tea_cache_start_step_perc)

        # Quality gate
        self._quality_enable.setChecked(getattr(cfg, "quality_overrides_enabled", False))
        self._cfg_star_switch.setValue(cfg.cfg_star_switch)
        self._cfg_zero_step.setValue(cfg.cfg_zero_step)
        self._slg_switch.setChecked(bool(cfg.slg_switch))
        if cfg.slg_layers:
            self._slg_layers.setText(",".join(str(x) for x in cfg.slg_layers))
        self._slg_start_perc.setValue(cfg.slg_start_perc)
        self._slg_end_perc.setValue(cfg.slg_end_perc)
        self._apg_switch.setChecked(bool(cfg.apg_switch))
        self._motion_amplitude.setValue(cfg.motion_amplitude)
        self._color_correction.setValue(cfg.color_correction_strength)
        # Mirror into the Quality-section override widget so its displayed value
        # tracks the (authoritative) Auto Color Match strength.
        if hasattr(self, "_quality_color_correction"):
            self._quality_color_correction.setValue(cfg.color_correction_strength)
        # Auto Color Match method dropdown
        cc_method = getattr(cfg, "color_correction_method", "mean-only-lab") or "mean-only-lab"
        m_idx = self._color_correction_method.findData(cc_method)
        if m_idx < 0:
            m_idx = 0
        self._color_correction_method.setCurrentIndex(m_idx)
        # Color anchor persistence (Phase 1 rebuild — defaults to 0.6).
        if hasattr(self, "_color_anchor_persistence"):
            self._color_anchor_persistence.setValue(
                float(getattr(cfg, "color_anchor_persistence", 0.6) or 0.6)
            )
        # Wan 2.2 presets — restore without re-triggering the change handler.
        if hasattr(self, "_acceleration_preset"):
            self._acceleration_preset.blockSignals(True)
            apk = getattr(cfg, "acceleration_preset", "off") or "off"
            idx = self._acceleration_preset.findData(apk)
            if idx >= 0:
                self._acceleration_preset.setCurrentIndex(idx)
            self._acceleration_preset.blockSignals(False)
        if hasattr(self, "_antidrift_preset"):
            self._antidrift_preset.blockSignals(True)
            adk = getattr(cfg, "antidrift_preset", "off") or "off"
            idx = self._antidrift_preset.findData(adk)
            if idx >= 0:
                self._antidrift_preset.setCurrentIndex(idx)
            self._antidrift_preset.blockSignals(False)
        # Output Quality preset
        if hasattr(self, "_output_quality"):
            oqk = getattr(cfg, "output_quality", "standard") or "standard"
            idx = self._output_quality.findData(oqk)
            if idx >= 0:
                self._output_quality.setCurrentIndex(idx)

        # SVI 2 Pro — sliding window + anchor
        if hasattr(self, "_sliding_window_size"):
            self._sliding_window_size.setValue(getattr(cfg, "sliding_window_size", 81) or 81)
            self._sliding_window_overlap.setValue(getattr(cfg, "sliding_window_overlap", 4) or 4)
            self._sliding_window_discard_last.setValue(
                getattr(cfg, "sliding_window_discard_last_frames", 0) or 0
            )
            self._sliding_window_color_correction.setValue(
                getattr(cfg, "sliding_window_color_correction_strength", 0.0) or 0.0
            )
            self._sliding_window_overlap_noise.setValue(
                getattr(cfg, "sliding_window_overlap_noise", 0.0) or 0.0
            )
            anchor_mode = getattr(cfg, "anchor_image_mode", "start") or "start"
            for i, (_, key) in enumerate(_ANCHOR_MODES):
                if key == anchor_mode:
                    self._anchor_image_mode.setCurrentIndex(i)
                    break
            self._anchor_image_path.setText(getattr(cfg, "anchor_image_path", "") or "")
            # Repopulate per-window anchor slots from the just-restored path.
            self._maybe_rebuild_anchor_slots()

            # Reconcile the prompt editor now that backend + duration + window
            # size are all restored. LTX multi-window needs the per-window field
            # page populated from the saved (multi-line) prompt — the earlier
            # restore put it in the standard box. (Wan Lightning was already
            # handled above, where the model alone determines the mode.)
            if self._ltx_window_prompts_active():
                self._prompt_stack.setCurrentIndex(1)
                self._rebuild_lightning_fields()
                self._set_lightning_prompt(cfg.prompt or "")

        # Mirror to the always-visible Color-Match-section anchor widgets.
        if hasattr(self, "_anchor_image_mode_cc"):
            anchor_mode = getattr(cfg, "anchor_image_mode", "start") or "start"
            for i, (_, key) in enumerate(_ANCHOR_MODES):
                if key == anchor_mode:
                    self._anchor_image_mode_cc.blockSignals(True)
                    self._anchor_image_mode_cc.setCurrentIndex(i)
                    self._anchor_image_mode_cc.blockSignals(False)
                    # Trigger the enable/disable logic for the path field.
                    self._on_anchor_mode_cc_changed(i)
                    break
        if hasattr(self, "_anchor_image_path_cc"):
            self._anchor_image_path_cc.setText(getattr(cfg, "anchor_image_path", "") or "")

        self._discard_last_frames.setValue(cfg.discard_last_frames)
        self._self_refiner.setCurrentIndex(cfg.self_refiner_setting)
        self._refiner_uncertainty.setValue(cfg.self_refiner_uncertainty)
        self._refiner_certainty_skip.setValue(cfg.self_refiner_certainty_skip)

        # Guidance
        self._use_guidance_m1.setChecked(bool(cfg.use_guidance_m1))
        gv1 = cfg.guidance_video_path_m1 or ""
        if gv1:
            self._guidance_video_m1.load_file(gv1)
        else:
            self._guidance_video_m1.clear_file()
        self._use_guidance_m2.setChecked(bool(cfg.use_guidance_m2))
        gv2 = cfg.guidance_video_path_m2 or ""
        if gv2:
            self._guidance_video_m2.load_file(gv2)
        else:
            self._guidance_video_m2.clear_file()
        idx = self._guidance_source_frame.findText(cfg.guidance_video_frame or "first")
        if idx >= 0:
            self._guidance_source_frame.setCurrentIndex(idx)

        # Show/hide guidance sections based on active mode
        self._guid_section_m1.setVisible(mode == 1)
        self._guid_ff_section_m2.setVisible(mode == 2)

        # Post-processing
        idx = self._temporal_upsampling.findText(cfg.temporal_upsampling or "Disabled")
        if idx >= 0:
            self._temporal_upsampling.setCurrentIndex(idx)
        idx = self._spatial_upsampling.findText(cfg.spatial_upsampling or "Disabled")
        if idx >= 0:
            self._spatial_upsampling.setCurrentIndex(idx)
        self._film_grain_intensity.setValue(cfg.film_grain_intensity)
        self._film_grain_saturation.setValue(cfg.film_grain_saturation)

        # LoRA
        self._refresh_loras()
        if cfg.activated_loras:
            self._set_activated_loras(cfg.activated_loras)
        self._lora_multipliers.setText(cfg.loras_multipliers or "")
        # The saved pair is aligned (collect re-emits in activated order), so
        # rebuild the name->token map to keep future collects aligned.
        if cfg.activated_loras and (cfg.loras_multipliers or "").strip():
            self._record_lora_multipliers(cfg.activated_loras, cfg.loras_multipliers)
        self._rebuild_lora_strength_view()

        # System-LoRA inputs
        if hasattr(self, "_ctl_type"):
            idx = self._ctl_type.findText(getattr(cfg, "ltx_control_type", "") or "raw")
            self._ctl_type.setCurrentIndex(idx if idx >= 0 else 0)
            self._ctl_strength.setValue(float(getattr(cfg, "ltx_control_strength", 1.0) or 1.0))
            self._ctl_start_s.setValue(float(getattr(cfg, "ltx_control_start_seconds", 0.0) or 0.0))
            self._ctl_ramp.setValue(int(getattr(cfg, "ltx_control_ramp_frames", 0) or 0))
            mask_path = getattr(cfg, "ltx_inpaint_mask_path", "") or ""
            if mask_path and Path(mask_path).is_file():
                self._ctl_mask.load_image(mask_path)
            else:
                self._ctl_mask.clear_image()
            self._ctl_mask_strength.setValue(float(getattr(cfg, "masking_strength", 1.0) or 1.0))
            self._sysctl_audio_path = getattr(cfg, "ltx_audio_path", "") or ""
            self._ctl_audio_label.setText(
                Path(self._sysctl_audio_path).name if self._sysctl_audio_path
                else "<i>no audio selected</i>"
            )
            self._update_system_lora_panels()

        # Preview
        pv = cfg.preview_path_m1 or cfg.preview_path_m2
        if pv and Path(pv).is_file():
            self._preview.load_video(pv)
            self._save_clip_btn.setEnabled(True)
            self._dl_frame_btn.setEnabled(True)
            self._save_frame_btn.setEnabled(True)
        else:
            self._preview.clear_video()
            self._save_clip_btn.setEnabled(False)
            self._dl_frame_btn.setEnabled(False)
            self._save_frame_btn.setEnabled(False)

    # -- Project change ----------------------------------------------------

    def _persist_image_path(self, key: str, path: str) -> None:
        """Save an image path change to project config immediately."""
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            setattr(cfg, key, path)
            cfg.save(self.project_path)
        except Exception:
            pass

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._refresh_profiles()
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        self._restore_from_config(cfg)
