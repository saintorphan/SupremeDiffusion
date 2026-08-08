"""Quick Generate tab — easy-mode Image and Video generation.

Features:
- Preset-driven generation (no advanced settings clutter)
- LoRA picker with auto-prompting (checking a LoRA appends its trigger phrase)
- Qwen prompt enhancement that preserves LoRA tags like <lora:name:0.8>
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.models.pipeline_template import (
    PipelineTemplate,
    scan_templates,
    import_template,
    export_template,
)
from sdqt.state import AppState
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.lora_picker import LoRAPickerWidget
from sdqt.widgets.lora_metadata import lora_prompt_tag
from sdqt.widgets.lora_auto_prompter import (
    auto_lora_prompt_tag,
    get_intensity_label,
    parse_lora_metadata,
)
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.inference import InferenceWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

from sdqt.utils.lora_tags import extract_lora_tags, reinsert_lora_tags
from sdqt.data.presets import (
    BODY_TYPES as _BODY_TYPES,
    CAMERA_MOVEMENTS as _CAMERA_MOVEMENTS,
    CLOTHING as _CLOTHING_OPTIONS,
    EXPRESSIONS as _EXPRESSION_OPTIONS,
    FILM_STYLES as _FILM_STYLES,
    HAIR_OPTIONS as _HAIR_OPTIONS,
    IMAGE_PROMPTS as _IMAGE_PROMPTS,
    MODEL_OPTIONS_VIDEO as _MODEL_OPTIONS_VIDEO,
    POSES as _POSE_OPTIONS,
    PROPS as _PROPS_OPTIONS,
    SCENES as _SCENE_OPTIONS,
    SKIN_DETAILS as _SKIN_DETAILS,
    VIDEO_PROMPTS as _VIDEO_PROMPTS,
)

# ── Video presets built from quality tiers ──────────────────────────────────
# (label, model_key, resolution, duration_frames, steps, guidance, flow_shift, solver, phases, switch)
#
# Lightning model presets use fixed solver params (flow_shift=5.0, unipc, 2 phases, switch=900).
# Quality model presets pull steps/guidance from the "standard" quality tier.

def _build_video_presets() -> list[tuple]:
    """Build video preset list from quality tier definitions."""
    from supremediffusion.config.hardware_detect import TIER_BY_KEY

    draft = TIER_BY_KEY.get("draft")
    std = TIER_BY_KEY.get("standard")
    d_res = draft.video_resolution if draft else "512x512"
    s_res = std.video_resolution if std else "832x480"
    s_steps = std.video_steps if std else 20
    s_guidance = std.video_guidance if std else 5.0

    # Lightning presets: fast generation across resolutions and durations
    # (steps=4, guidance=1.0 are Lightning-specific, not from tiers)
    _LIGHTNING = "i2v_2_2_lightning_v2"
    _QUALITY = "i2v_2_2"
    presets = [
        (f"Fast 2s ({d_res})",     _LIGHTNING, d_res,    33, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"Fast 3s ({d_res})",     _LIGHTNING, d_res,    49, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"Fast 5s ({d_res})",     _LIGHTNING, d_res,    81, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"HD 2s ({s_res})",       _LIGHTNING, s_res,    33, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"HD 3s ({s_res})",       _LIGHTNING, s_res,    49, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"HD 5s ({s_res})",       _LIGHTNING, s_res,    81, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"Portrait 2s (480x832)", _LIGHTNING, "480x832", 33, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"Portrait 3s (480x832)", _LIGHTNING, "480x832", 49, 4, 1.0, 5.0, "unipc", 2, 900),
        (f"Portrait 5s (480x832)", _LIGHTNING, "480x832", 81, 4, 1.0, 5.0, "unipc", 2, 900),
        # Quality presets: pull steps/guidance from the standard tier
        (f"Quality 3s ({s_res})",  _QUALITY,   s_res,    49, s_steps, s_guidance, 3.0, "unipc", 1, 0),
        (f"Quality 5s ({s_res})",  _QUALITY,   s_res,    81, s_steps, s_guidance, 3.0, "unipc", 1, 0),
    ]
    return presets

_VIDEO_PRESETS = _build_video_presets()

# ── Image presets (label, width, height, steps, cfg) ──
_IMAGE_PRESETS = [
    ("Fast (512x512)",        512,  512,  20, 7.0),
    ("Square (1024x1024)",    1024, 1024, 25, 7.0),
    ("Landscape (1216x832)",  1216, 832,  25, 7.0),
    ("Portrait (832x1216)",   832,  1216, 25, 7.0),
    ("Quick (768x768)",       768,  768,  15, 5.0),
]

_MODEL_OPTIONS_IMAGE = [
    ("SDXL", "sdxl"),
    ("SD 1.5", "sd15"),
    ("Flux", "flux"),
    ("Pony (Realistic)", "pony_real"),
    ("Pony (Anime)", "pony_anime"),
]

# ── LoRA suggestion keywords — maps prompt keywords to suggested LoRA filenames ──
# (Will be populated as user adds LoRAs — for now, matches tag-based)
_LORA_KEYWORD_MAP = {
    "anime": ["anime", "cartoon", "illustration"],
    "realistic": ["realistic", "photo", "real"],
    "portrait": ["portrait", "face", "headshot"],
    "landscape": ["landscape", "scenery", "nature"],
    "fantasy": ["fantasy", "magic", "medieval"],
    "sci-fi": ["scifi", "cyber", "futuristic"],
    "fashion": ["fashion", "style", "clothing"],
    "cinematic": ["cinematic", "film", "movie"],
}


# ---------------------------------------------------------------------------
# Auto Direct — Qwen system prompt for per-second timeline generation
# ---------------------------------------------------------------------------

_AUTO_DIRECT_SYSTEM = (
    "You are a video director AI. Given a concept description and a duration in seconds, "
    "generate a detailed per-second shot breakdown for an AI video generator.\n\n"
    "Output EXACTLY this format, one line per second, starting from 0:\n"
    "(at 0 seconds: description of what happens at this moment)\n"
    "(at 1 seconds: description of what happens next)\n"
    "(at 2 seconds: ...)\n"
    "...and so on up to the final second.\n\n"
    "Rules:\n"
    "- Each description should be one clear, vivid sentence describing the visual action\n"
    "- Show progression — build the scene second by second with escalating action\n"
    "- Describe camera angles, lighting changes, and character expressions where relevant\n"
    "- Keep continuity between seconds — this is one continuous shot\n"
    "- Include motion verbs: moves, turns, walks, reaches, gestures, shifts, steps\n"
    "- DO NOT include anything outside the (at N seconds: ...) format\n"
    "- DO NOT skip any seconds from 0 to the final second"
)

class QuickGenTab(BaseTab):
    """Simplified generation tab — drop image, type prompt, pick preset, generate."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._video_worker: InferenceWorker | None = None
        self._image_worker = None
        self._enhance_worker: PromptEnhanceWorker | None = None
        self._enhance_target: str = ""  # "vid" or "img"
        self._enhance_which: str = ""   # "pos" or "neg"
        self._enhance_lora_tags: list[str] = []

        # ── Content pack state ──
        self._packs: list[PipelineTemplate] = []
        self._active_pack: PipelineTemplate | None = None
        # Active lists — start with module-level defaults, packs override
        self._cur_video_prompts = list(_VIDEO_PROMPTS)
        self._cur_image_prompts = list(_IMAGE_PROMPTS)
        self._cur_clothing = list(_CLOTHING_OPTIONS)
        self._cur_poses = list(_POSE_OPTIONS)
        self._cur_insertions = list(_PROPS_OPTIONS)
        self._cur_expressions = list(_EXPRESSION_OPTIONS)
        self._cur_lora_keywords = dict(_LORA_KEYWORD_MAP)
        self._cur_auto_direct_system = _AUTO_DIRECT_SYSTEM

        self._build_ui()
        self._load_packs()

    # ── Shared helper: build a subject description group box ─────────────

    def _build_subject_group(self, prefix: str) -> tuple[QGroupBox, dict[str, QComboBox]]:
        """Build a collapsible 'Subject Description' box with dropdowns.

        Returns (group_box, dict_of_combos) so the caller can wire signals.
        """
        group = QGroupBox("Subject Description")
        group.setCheckable(True)
        group.setChecked(True)
        group.setToolTip(
            "Build your subject description from dropdowns. "
            "Selected options are composed into the prompt when you click 'Compose Prompt'."
        )
        gl = QVBoxLayout(group)
        gl.setSpacing(4)

        combos: dict[str, QComboBox] = {}

        def _add_row(label: str, key: str, options: list[tuple[str, str]]) -> QComboBox:
            row = QHBoxLayout()
            lbl = QLabel(f"{label}:")
            lbl.setFixedWidth(90)
            row.addWidget(lbl)
            cb = QComboBox()
            cb.setMinimumWidth(180)
            for display, _ in options:
                cb.addItem(display)
            row.addWidget(cb, 1)
            gl.addLayout(row)
            combos[key] = cb
            return cb

        _add_row("Body Type", "body", _BODY_TYPES)
        _add_row("Skin", "skin", _SKIN_DETAILS)
        _add_row("Hair", "hair", _HAIR_OPTIONS)
        _add_row("Clothing", "clothing", self._cur_clothing).setToolTip(
            "Mix and match clothing. The selected item's description is added to the prompt."
        )
        _add_row("Pose", "pose", self._cur_poses)
        _add_row("Insertion", "insertion", self._cur_insertions).setToolTip(
            "Insertion device or prop — loaded from active content pack."
        )
        _add_row("Expression", "expression", self._cur_expressions)
        _add_row("Scene", "scene", _SCENE_OPTIONS)

        # Compose button
        compose_row = QHBoxLayout()
        compose_btn = QPushButton("Compose Prompt from Selections")
        compose_btn.setToolTip(
            "Build a prompt from all selected dropdown options and append it to the prompt box."
        )
        compose_btn.setStyleSheet("font-weight: bold;")
        compose_row.addStretch()
        compose_row.addWidget(compose_btn)
        compose_row.addStretch()
        gl.addLayout(compose_row)

        # Store compose button on the group for wiring
        group._compose_btn = compose_btn  # type: ignore[attr-defined]

        return group, combos

    def _compose_from_dropdowns(self, combos: dict[str, QComboBox], options_map: dict) -> str:
        """Read all dropdown selections and build a composed prompt string."""
        parts: list[str] = []
        for key, combo in combos.items():
            idx = combo.currentIndex()
            opts = options_map[key]
            if idx > 0 and idx < len(opts):
                _, text = opts[idx]
                if text:
                    parts.append(text)
        return ", ".join(parts)

    # ── Shared helper: suggested LoRAs list ──────────────────────────────

    def _build_suggested_loras(self) -> tuple[QGroupBox, QListWidget]:
        group = QGroupBox("Suggested LoRAs")
        group.setToolTip(
            "LoRAs suggested based on your prompt keywords. "
            "Check to apply, uncheck to remove. Scans your LoRA directory metadata."
        )
        layout = QVBoxLayout(group)
        lw = QListWidget()
        lw.setMaximumHeight(80)
        layout.addWidget(lw)
        return group, lw

    def _update_suggested_loras(self, prompt_text: str, list_widget: QListWidget, lora_dir: str) -> None:
        """Scan LoRAs and suggest ones matching prompt keywords."""
        list_widget.clear()
        if not lora_dir or not prompt_text.strip():
            return
        lora_path = Path(lora_dir)
        if not lora_path.is_dir():
            return

        prompt_lower = prompt_text.lower()
        # Collect matching keywords
        matched_kw: set[str] = set()
        for keyword, lora_stems in self._cur_lora_keywords.items():
            if keyword in prompt_lower:
                matched_kw.update(s.lower() for s in lora_stems)

        if not matched_kw:
            return

        # Scan LoRA files and check if their stem matches any keyword
        for f in sorted(lora_path.iterdir()):
            if f.suffix.lower() != ".safetensors":
                continue
            stem_lower = f.stem.lower()
            # Check if any keyword substring matches the LoRA filename
            for kw in matched_kw:
                if kw in stem_lower:
                    item = QListWidgetItem(f"  {f.name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setToolTip(f"Suggested because prompt contains matching keyword")
                    list_widget.addItem(item)
                    break  # Don't add same LoRA twice

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ── Content Pack selector bar ──
        pack_row = QHBoxLayout()
        pack_row.setSpacing(6)
        pack_row.addWidget(QLabel("Content Pack:"))
        self._pack_combo = QComboBox()
        self._pack_combo.setMinimumWidth(200)
        self._pack_combo.setToolTip(
            "Select a content pack to load custom prompt presets,\n"
            "subject options, LoRA keywords, and Auto Direct prompts.\n"
            "Packs are JSON files in the pipeline_templates/ folder."
        )
        self._pack_combo.currentIndexChanged.connect(self._on_pack_changed)
        pack_row.addWidget(self._pack_combo, 1)

        import_btn = QPushButton("Import")
        import_btn.setFixedHeight(24)
        import_btn.setToolTip("Import a content pack (.json) from disk")
        import_btn.clicked.connect(self._on_import_pack)
        pack_row.addWidget(import_btn)

        export_btn = QPushButton("Export")
        export_btn.setFixedHeight(24)
        export_btn.setToolTip("Export the active content pack to a file")
        export_btn.clicked.connect(self._on_export_pack)
        pack_row.addWidget(export_btn)

        folder_btn = QPushButton("\U0001F4C2")
        folder_btn.setFixedWidth(30)
        folder_btn.setFixedHeight(24)
        folder_btn.setToolTip("Open pipeline_templates folder")
        folder_btn.clicked.connect(self._on_open_packs_folder)
        pack_row.addWidget(folder_btn)

        outer.addLayout(pack_row)

        # ── Mode toggle (Image / Video) ──
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._mode_bar = QTabBar()
        self._mode_bar.addTab("Video")
        self._mode_bar.addTab("Image")
        self._mode_bar.setToolTip("Switch between Video (I2V) and Image (txt2img / img2img) generation")
        self._mode_bar.currentChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_bar)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # ── Stacked content (Video=0, Image=1) ──
        self._stack = QStackedWidget()
        outer.addWidget(self._stack, 1)

        self._build_video_page()
        self._build_image_page()

    # ── Options maps for compose helper ────────────────────────────────

    @property
    def _OPTIONS_MAP(self) -> dict:
        return {
            "body": _BODY_TYPES, "skin": _SKIN_DETAILS, "hair": _HAIR_OPTIONS,
            "clothing": self._cur_clothing, "pose": self._cur_poses,
            "insertion": self._cur_insertions, "expression": self._cur_expressions,
            "scene": _SCENE_OPTIONS,
        }

    # ── Content Pack management ──────────────────────────────────────────

    def _load_packs(self) -> None:
        """Scan pipeline_templates/ and populate the pack combo."""
        app_root = getattr(self.state, "app_root", ".")
        self._packs = [t for t in scan_templates(app_root) if t.has_quick_gen]
        self._pack_combo.blockSignals(True)
        self._pack_combo.clear()
        self._pack_combo.addItem("(Base Defaults)")
        for t in self._packs:
            self._pack_combo.addItem(f"{t.name}  — {t.author}" if t.author else t.name)
        self._pack_combo.blockSignals(False)

    @Slot(int)
    def _on_pack_changed(self, idx: int) -> None:
        """Switch active content pack and repopulate all combos."""
        if idx <= 0:
            self._active_pack = None
            self._apply_pack(None)
        else:
            pack = self._packs[idx - 1]
            self._active_pack = pack
            self._apply_pack(pack)

    def _apply_pack(self, pack: PipelineTemplate | None) -> None:
        """Replace current lists with pack content (or revert to defaults)."""
        if pack is None:
            self._cur_video_prompts = list(_VIDEO_PROMPTS)
            self._cur_image_prompts = list(_IMAGE_PROMPTS)
            self._cur_clothing = list(_CLOTHING_OPTIONS)
            self._cur_poses = list(_POSE_OPTIONS)
            self._cur_insertions = list(_PROPS_OPTIONS)
            self._cur_expressions = list(_EXPRESSION_OPTIONS)
            self._cur_lora_keywords = dict(_LORA_KEYWORD_MAP)
            self._cur_auto_direct_system = _AUTO_DIRECT_SYSTEM
        else:
            # Merge: pack items prepended after the "-- Select --" header, then defaults
            def _merge(pack_items: list[tuple[str, str]],
                       defaults: list[tuple[str, str]]) -> list[tuple[str, str]]:
                if not pack_items:
                    return list(defaults)
                # Keep the "-- Select --" / header entry from defaults
                header = [defaults[0]] if defaults and defaults[0][1] == "" else []
                return header + pack_items + defaults[1:]

            self._cur_video_prompts = _merge(
                pack.quick_gen_video_prompts,
                _VIDEO_PROMPTS,
            )
            self._cur_image_prompts = _merge(
                pack.quick_gen_image_prompts,
                _IMAGE_PROMPTS,
            )
            self._cur_clothing = _merge(pack.quick_gen_clothing, _CLOTHING_OPTIONS)
            self._cur_poses = _merge(pack.quick_gen_poses, _POSE_OPTIONS)
            self._cur_insertions = _merge(pack.quick_gen_insertions, _PROPS_OPTIONS)
            self._cur_expressions = _merge(pack.quick_gen_expressions, _EXPRESSION_OPTIONS)
            # LoRA keywords: pack overrides + defaults
            merged_kw = dict(_LORA_KEYWORD_MAP)
            merged_kw.update(pack.quick_gen_lora_keywords)
            self._cur_lora_keywords = merged_kw
            # Auto Direct system prompt
            self._cur_auto_direct_system = (
                pack.auto_direct_system_prompt or _AUTO_DIRECT_SYSTEM
            )

        # Repopulate all combo boxes
        self._repopulate_combos()

    def _repopulate_combo(self, combo: QComboBox, items: list[tuple[str, str]]) -> None:
        """Clear and refill a QComboBox from (label, value) pairs."""
        combo.blockSignals(True)
        combo.clear()
        for label, _ in items:
            combo.addItem(label)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _repopulate_combos(self) -> None:
        """Refresh all Quick Gen combos from current active lists."""
        # Prompt presets
        self._repopulate_combo(self._vid_prompt_preset, self._cur_video_prompts)
        self._repopulate_combo(self._img_prompt_preset, self._cur_image_prompts)

        # Subject description combos — clothing, pose, insertion, expression
        for prefix in ("vid", "img"):
            combos = getattr(self, f"_{prefix}_subject_combos", {})
            if "clothing" in combos:
                self._repopulate_combo(combos["clothing"], self._cur_clothing)
            if "pose" in combos:
                self._repopulate_combo(combos["pose"], self._cur_poses)
            if "insertion" in combos:
                self._repopulate_combo(combos["insertion"], self._cur_insertions)
            if "expression" in combos:
                self._repopulate_combo(combos["expression"], self._cur_expressions)

        pack_name = self._active_pack.name if self._active_pack else "Base"
        logger.info("Content pack applied: %s (%d vid, %d img prompts)",
                     pack_name,
                     len(self._cur_video_prompts),
                     len(self._cur_image_prompts))

    def _on_import_pack(self) -> None:
        """Import a content pack JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Content Pack", "", "JSON Files (*.json)"
        )
        if not path:
            return
        app_root = getattr(self.state, "app_root", ".")
        t = import_template(path, app_root)
        if t and t.has_quick_gen:
            self._packs.append(t)
            self._pack_combo.addItem(f"{t.name}  — {t.author}" if t.author else t.name)
            self._pack_combo.setCurrentIndex(self._pack_combo.count() - 1)
            logger.info("Imported content pack: %s", t.name)
        elif t:
            logger.warning("Imported template has no Quick Gen content: %s", t.name)

    def _on_export_pack(self) -> None:
        """Export the active content pack."""
        if not self._active_pack:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Content Pack",
            self._active_pack.name.replace(" ", "_") + ".json",
            "JSON Files (*.json)",
        )
        if path:
            export_template(self._active_pack, path)

    def _on_open_packs_folder(self) -> None:
        """Open the pipeline_templates folder in the system file manager."""
        import subprocess, sys
        app_root = getattr(self.state, "app_root", ".")
        d = Path(app_root) / "pipeline_templates"
        d.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(d)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    # ── Video page ───────────────────────────────────────────────────────

    def _build_video_page(self) -> None:
        from PySide6.QtWidgets import QScrollArea

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Top splitter: source (left) + preview (right)
        top_splitter = QSplitter(Qt.Horizontal)
        self._vid_source = ImageDropWidget("Drop Source Image", thumb_height=220)
        top_splitter.addWidget(self._vid_source)
        self._vid_preview = VideoPlayerWidget("Preview")
        top_splitter.addWidget(self._vid_preview)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        layout.addWidget(top_splitter, 1)

        # Scrollable middle area for all the controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll_content = QWidget()
        mid = QVBoxLayout(scroll_content)
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(6)

        # ── Model selector row ──
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("Model:"))
        self._vid_model = QComboBox()
        self._vid_model.setToolTip("Select the video generation model/pipeline.")
        for label, _ in _MODEL_OPTIONS_VIDEO:
            self._vid_model.addItem(label)
        model_row.addWidget(self._vid_model)
        model_row.addStretch()
        mid.addLayout(model_row)

        # ── Subject description dropdowns ──
        self._vid_subject_group, self._vid_subject_combos = self._build_subject_group("vid")
        self._vid_subject_group._compose_btn.clicked.connect(self._on_vid_compose)
        mid.addWidget(self._vid_subject_group)

        # ── Prompt preset + Qwen enhance row ──
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(6)
        prompt_row.addWidget(QLabel("Prompt:"))
        self._vid_prompt_preset = QComboBox()
        self._vid_prompt_preset.setMinimumWidth(200)
        self._vid_prompt_preset.setToolTip(
            "Pick a prompt preset to auto-fill the prompt box.\n"
            "[AD] presets are short concepts — click Auto Direct to expand them into per-second timelines."
        )
        for label, _ in self._cur_video_prompts:
            self._vid_prompt_preset.addItem(label)
        self._vid_prompt_preset.currentIndexChanged.connect(self._on_vid_prompt_preset)
        prompt_row.addWidget(self._vid_prompt_preset)
        prompt_row.addStretch()

        self._vid_enhance_pos = QPushButton("Enhance +")
        self._vid_enhance_pos.setFixedHeight(24)
        self._vid_enhance_pos.setToolTip("Enhance prompt with uncensored LLM (preserves LoRA tags)")
        self._vid_enhance_pos.clicked.connect(lambda: self._on_enhance("vid", "pos"))
        prompt_row.addWidget(self._vid_enhance_pos)
        self._vid_enhance_neg = QPushButton("Neg \u2013")
        self._vid_enhance_neg.setFixedHeight(24)
        self._vid_enhance_neg.setToolTip("Generate negative prompt with LLM")
        self._vid_enhance_neg.clicked.connect(lambda: self._on_enhance("vid", "neg"))
        prompt_row.addWidget(self._vid_enhance_neg)
        self._vid_auto_direct = QPushButton("Auto Direct")
        self._vid_auto_direct.setFixedHeight(24)
        self._vid_auto_direct.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        self._vid_auto_direct.setToolTip(
            "Generate per-second timeline prompts from your concept.\n"
            "Write a simple idea, pick a duration preset, then click this."
        )
        self._vid_auto_direct.clicked.connect(self._on_auto_direct)
        prompt_row.addWidget(self._vid_auto_direct)
        mid.addLayout(prompt_row)

        # ── Camera movement + film style + Remix row ──
        style_row = QHBoxLayout()
        style_row.setSpacing(6)
        style_row.addWidget(QLabel("Camera:"))
        self._vid_camera = QComboBox()
        self._vid_camera.setMinimumWidth(160)
        self._vid_camera.setToolTip(
            "Camera movement to append to your prompt.\n"
            "Select a movement, then click Remix to add it."
        )
        for label, _ in _CAMERA_MOVEMENTS:
            self._vid_camera.addItem(label)
        style_row.addWidget(self._vid_camera)

        style_row.addWidget(QLabel("Style:"))
        self._vid_film_style = QComboBox()
        self._vid_film_style.setMinimumWidth(160)
        self._vid_film_style.setToolTip(
            "Film style / look to append to your prompt.\n"
            "Found footage, security cam, candid, cinematic, etc."
        )
        for label, _ in _FILM_STYLES:
            self._vid_film_style.addItem(label)
        style_row.addWidget(self._vid_film_style)

        self._vid_remix_btn = QPushButton("Remix")
        self._vid_remix_btn.setFixedHeight(24)
        self._vid_remix_btn.setStyleSheet("font-weight: bold; color: #ffa726;")
        self._vid_remix_btn.setToolTip(
            "Append the selected camera movement and/or film style to your current prompt.\n"
            "Select one or both, then click Remix."
        )
        self._vid_remix_btn.clicked.connect(self._on_vid_remix)
        style_row.addWidget(self._vid_remix_btn)
        style_row.addStretch()
        mid.addLayout(style_row)

        self._vid_prompt = QPlainTextEdit()
        self._vid_prompt.setMaximumHeight(60)
        self._vid_prompt.setPlaceholderText("Describe what happens in the video...")
        self._vid_prompt.setToolTip(
            "Video prompt. LoRA tags are auto-inserted when you check a LoRA.\n"
            "Use Enhance+ to expand, Auto Direct for per-second timelines, or Compose from dropdowns."
        )
        mid.addWidget(self._vid_prompt)

        self._vid_neg_prompt = QPlainTextEdit()
        self._vid_neg_prompt.setMaximumHeight(30)
        self._vid_neg_prompt.setPlaceholderText("Negative prompt (optional)...")
        self._vid_neg_prompt.setToolTip("Things to avoid. Use Neg button to auto-generate.")
        mid.addWidget(self._vid_neg_prompt)

        # ── LoRA picker ──
        self._vid_lora = LoRAPickerWidget("LoRA Auto-Prompter (check to add triggers)")
        self._vid_lora.set_state(self.state)
        self._vid_lora._list.currentItemChanged.connect(
            lambda cur, _prev: self._on_lora_selected(cur, "vid")
        )
        self._vid_lora._list.itemChanged.connect(
            lambda item: self._on_lora_checked(item, self._vid_prompt, "vid")
        )
        try:
            self._vid_lora._list.itemChanged.disconnect(self._vid_lora._on_item_changed)
        except RuntimeError:
            pass
        lora_dir = self.state.global_config.model_paths.get("lora_dir", "")
        self._vid_lora_dir = lora_dir
        if lora_dir:
            self._vid_lora.set_lora_dir(lora_dir)
        mid.addWidget(self._vid_lora)

        # Intensity slider + info
        intensity_row = QHBoxLayout()
        intensity_row.setSpacing(6)
        intensity_row.addWidget(QLabel("Intensity:"))
        self._vid_intensity = QSlider(Qt.Horizontal)
        self._vid_intensity.setRange(0, 100)
        self._vid_intensity.setValue(50)
        self._vid_intensity.setFixedWidth(150)
        self._vid_intensity.setToolTip(
            "Controls LoRA trigger selection and weight.\n"
            "Low = subtle. High = extreme with all triggers + higher weight."
        )
        self._vid_intensity.valueChanged.connect(
            lambda v: self._vid_intensity_label.setText(get_intensity_label(v / 100))
        )
        intensity_row.addWidget(self._vid_intensity)
        self._vid_intensity_label = QLabel("Moderate")
        self._vid_intensity_label.setFixedWidth(60)
        intensity_row.addWidget(self._vid_intensity_label)
        intensity_row.addSpacing(12)
        self._vid_lora_info = QLabel("")
        self._vid_lora_info.setWordWrap(True)
        self._vid_lora_info.setStyleSheet("color: #aaa; font-size: 11px;")
        intensity_row.addWidget(self._vid_lora_info, 1)
        mid.addLayout(intensity_row)

        # ── Suggested LoRAs ──
        self._vid_suggested_group, self._vid_suggested_list = self._build_suggested_loras()
        mid.addWidget(self._vid_suggested_group)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 2)

        # ── Controls row: preset + generate + abort ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Preset:"))
        self._vid_preset = QComboBox()
        self._vid_preset.setMinimumWidth(220)
        self._vid_preset.setToolTip(
            "Video preset — resolution, duration, steps.\n"
            "Fast = Lightning (4 steps). Quality = full model (30 steps)."
        )
        for label, *_ in _VIDEO_PRESETS:
            self._vid_preset.addItem(label)
        self._vid_preset.setCurrentIndex(4)
        ctrl.addWidget(self._vid_preset)
        ctrl.addStretch()
        self._vid_gen_btn = QPushButton("Generate Video")
        self._vid_gen_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 6px 24px;")
        self._vid_gen_btn.setToolTip("Start video generation. Requires a source image.")
        self._vid_gen_btn.clicked.connect(self._on_video_generate)
        ctrl.addWidget(self._vid_gen_btn)
        self._vid_abort_btn = QPushButton("Abort")
        self._vid_abort_btn.setVisible(False)
        self._vid_abort_btn.clicked.connect(self._on_video_abort)
        ctrl.addWidget(self._vid_abort_btn)
        layout.addLayout(ctrl)

        self._vid_status = QLabel("")
        layout.addWidget(self._vid_status)

        self._stack.addWidget(page)

    # ── Image page ───────────────────────────────────────────────────────

    def _build_image_page(self) -> None:
        from PySide6.QtWidgets import QScrollArea

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Top splitter: source (left) + gallery (right)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._img_source = ImageDropWidget("Source Image (optional for img2img)", thumb_height=220)
        left_layout.addWidget(self._img_source)
        splitter.addWidget(left)
        self._img_gallery = ImageGalleryWidget("Results")
        splitter.addWidget(self._img_gallery)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Scrollable middle
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll_content = QWidget()
        mid = QVBoxLayout(scroll_content)
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(6)

        # ── Model selector row ──
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("Model:"))
        self._img_model = QComboBox()
        self._img_model.setToolTip(
            "Select the image generation model/pipeline.\n"
            "The pipeline must be loaded from the Image tab first."
        )
        for label, _ in _MODEL_OPTIONS_IMAGE:
            self._img_model.addItem(label)
        model_row.addWidget(self._img_model)
        model_row.addStretch()
        mid.addLayout(model_row)

        # ── Subject description dropdowns ──
        self._img_subject_group, self._img_subject_combos = self._build_subject_group("img")
        self._img_subject_group._compose_btn.clicked.connect(self._on_img_compose)
        mid.addWidget(self._img_subject_group)

        # ── Prompt preset + enhance row ──
        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(6)
        prompt_row.addWidget(QLabel("Prompt:"))
        self._img_prompt_preset = QComboBox()
        self._img_prompt_preset.setMinimumWidth(200)
        self._img_prompt_preset.setToolTip("Pick a prompt preset to auto-fill the prompt box.")
        for label, _ in self._cur_image_prompts:
            self._img_prompt_preset.addItem(label)
        self._img_prompt_preset.currentIndexChanged.connect(self._on_img_prompt_preset)
        prompt_row.addWidget(self._img_prompt_preset)
        prompt_row.addStretch()
        self._img_enhance_pos = QPushButton("Enhance +")
        self._img_enhance_pos.setFixedHeight(24)
        self._img_enhance_pos.setToolTip("Enhance prompt with uncensored LLM (preserves LoRA tags)")
        self._img_enhance_pos.clicked.connect(lambda: self._on_enhance("img", "pos"))
        prompt_row.addWidget(self._img_enhance_pos)
        self._img_enhance_neg = QPushButton("Neg \u2013")
        self._img_enhance_neg.setFixedHeight(24)
        self._img_enhance_neg.setToolTip("Generate negative prompt with LLM")
        self._img_enhance_neg.clicked.connect(lambda: self._on_enhance("img", "neg"))
        prompt_row.addWidget(self._img_enhance_neg)
        mid.addLayout(prompt_row)

        self._img_prompt = QPlainTextEdit()
        self._img_prompt.setMaximumHeight(60)
        self._img_prompt.setPlaceholderText("Describe the image...")
        self._img_prompt.setToolTip(
            "Image prompt. LoRA tags are auto-inserted when you check a LoRA.\n"
            "Use Enhance+ to expand, or Compose from dropdowns above."
        )
        mid.addWidget(self._img_prompt)

        self._img_neg_prompt = QPlainTextEdit()
        self._img_neg_prompt.setMaximumHeight(30)
        self._img_neg_prompt.setPlaceholderText("Negative prompt (optional)...")
        self._img_neg_prompt.setToolTip("Things to avoid. Use Neg button to auto-generate.")
        mid.addWidget(self._img_neg_prompt)

        # ── LoRA picker ──
        self._img_lora = LoRAPickerWidget("LoRA Auto-Prompter (check to add triggers)")
        self._img_lora.set_state(self.state)
        self._img_lora._list.currentItemChanged.connect(
            lambda cur, _prev: self._on_lora_selected(cur, "img")
        )
        self._img_lora._list.itemChanged.connect(
            lambda item: self._on_lora_checked(item, self._img_prompt, "img")
        )
        try:
            self._img_lora._list.itemChanged.disconnect(self._img_lora._on_item_changed)
        except RuntimeError:
            pass
        img_lora_dir = (
            self.state.global_config.model_paths.get("sd_lora_dir", "")
            or self.state.global_config.model_paths.get("flux_lora_dir", "")
        )
        self._img_lora_dir = img_lora_dir or ""
        if img_lora_dir:
            self._img_lora.set_lora_dir(img_lora_dir)
        mid.addWidget(self._img_lora)

        # Intensity slider + info
        intensity_row = QHBoxLayout()
        intensity_row.setSpacing(6)
        intensity_row.addWidget(QLabel("Intensity:"))
        self._img_intensity = QSlider(Qt.Horizontal)
        self._img_intensity.setRange(0, 100)
        self._img_intensity.setValue(50)
        self._img_intensity.setFixedWidth(150)
        self._img_intensity.setToolTip(
            "Controls LoRA trigger selection and weight.\n"
            "Low = subtle. High = extreme with all triggers + higher weight."
        )
        self._img_intensity.valueChanged.connect(
            lambda v: self._img_intensity_label.setText(get_intensity_label(v / 100))
        )
        intensity_row.addWidget(self._img_intensity)
        self._img_intensity_label = QLabel("Moderate")
        self._img_intensity_label.setFixedWidth(60)
        intensity_row.addWidget(self._img_intensity_label)
        intensity_row.addSpacing(12)
        self._img_lora_info = QLabel("")
        self._img_lora_info.setWordWrap(True)
        self._img_lora_info.setStyleSheet("color: #aaa; font-size: 11px;")
        intensity_row.addWidget(self._img_lora_info, 1)
        mid.addLayout(intensity_row)

        # ── Suggested LoRAs ──
        self._img_suggested_group, self._img_suggested_list = self._build_suggested_loras()
        mid.addWidget(self._img_suggested_group)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 2)

        # ── Controls row ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        ctrl.addWidget(QLabel("Preset:"))
        self._img_preset = QComboBox()
        self._img_preset.setMinimumWidth(200)
        self._img_preset.setToolTip("Image resolution and quality preset.")
        for label, *_ in _IMAGE_PRESETS:
            self._img_preset.addItem(label)
        ctrl.addWidget(self._img_preset)
        ctrl.addStretch()
        self._img_gen_btn = QPushButton("Generate Image")
        self._img_gen_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 6px 24px;")
        self._img_gen_btn.setToolTip("txt2img if no source image, img2img if source provided.")
        self._img_gen_btn.clicked.connect(self._on_image_generate)
        ctrl.addWidget(self._img_gen_btn)
        self._img_abort_btn = QPushButton("Abort")
        self._img_abort_btn.setVisible(False)
        self._img_abort_btn.clicked.connect(self._on_image_abort)
        ctrl.addWidget(self._img_abort_btn)
        layout.addLayout(ctrl)

        self._img_status = QLabel("")
        layout.addWidget(self._img_status)

        self._stack.addWidget(page)

    # ── Mode switch ──────────────────────────────────────────────────────

    @Slot(int)
    def _on_mode_changed(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)

    # ── Prompt presets ───────────────────────────────────────────────────

    @Slot(int)
    def _on_vid_prompt_preset(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._cur_video_prompts):
            return
        _, text = self._cur_video_prompts[idx]
        if text:
            self._vid_prompt.setPlainText(text)
            self._update_suggested_loras(text, self._vid_suggested_list, self._vid_lora_dir)

    @Slot(int)
    def _on_img_prompt_preset(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._cur_image_prompts):
            return
        _, text = self._cur_image_prompts[idx]
        if text:
            self._img_prompt.setPlainText(text)
            self._update_suggested_loras(text, self._img_suggested_list, self._img_lora_dir)

    # ── Subject compose ─────────────────────────────────────────────────

    @Slot()
    def _on_vid_compose(self) -> None:
        """Compose a prompt from video subject dropdowns and set it."""
        composed = self._compose_from_dropdowns(self._vid_subject_combos, self._OPTIONS_MAP)
        if composed:
            current = self._vid_prompt.toPlainText().strip()
            if current:
                self._vid_prompt.setPlainText(f"{current}, {composed}")
            else:
                self._vid_prompt.setPlainText(composed)
            self._update_suggested_loras(
                self._vid_prompt.toPlainText(), self._vid_suggested_list, self._vid_lora_dir
            )

    @Slot()
    def _on_img_compose(self) -> None:
        """Compose a prompt from image subject dropdowns and set it."""
        composed = self._compose_from_dropdowns(self._img_subject_combos, self._OPTIONS_MAP)
        if composed:
            current = self._img_prompt.toPlainText().strip()
            if current:
                self._img_prompt.setPlainText(f"{current}, {composed}")
            else:
                self._img_prompt.setPlainText(composed)
            self._update_suggested_loras(
                self._img_prompt.toPlainText(), self._img_suggested_list, self._img_lora_dir
            )

    # ── Remix — append camera movement + film style ───────────────────────

    @Slot()
    def _on_vid_remix(self) -> None:
        """Append selected camera movement and/or film style to the video prompt."""
        parts: list[str] = []
        cam_idx = self._vid_camera.currentIndex()
        if cam_idx > 0 and cam_idx < len(_CAMERA_MOVEMENTS):
            _, cam_text = _CAMERA_MOVEMENTS[cam_idx]
            if cam_text:
                parts.append(cam_text)
        style_idx = self._vid_film_style.currentIndex()
        if style_idx > 0 and style_idx < len(_FILM_STYLES):
            _, style_text = _FILM_STYLES[style_idx]
            if style_text:
                parts.append(style_text)
        if not parts:
            self._vid_status.setText("Select a camera movement and/or film style first.")
            return
        addition = ", ".join(parts)
        current = self._vid_prompt.toPlainText().strip()
        if current:
            self._vid_prompt.setPlainText(f"{current}, {addition}")
        else:
            self._vid_prompt.setPlainText(addition)
        self._vid_status.setText("Remixed!")

    # ── LoRA auto-prompting ──────────────────────────────────────────────

    def _lora_dir_for(self, target: str) -> str:
        """Return the LoRA directory for video or image."""
        return self._vid_lora_dir if target == "vid" else self._img_lora_dir

    def _on_lora_selected(self, item, target: str) -> None:
        """When a LoRA is clicked (not checked), show its metadata info."""
        if item is None:
            return
        lora_dir = self._lora_dir_for(target)
        if not lora_dir:
            return
        lora_path = str(Path(lora_dir) / item.text())
        try:
            meta = parse_lora_metadata(lora_path)
        except Exception:
            return

        info_label = self._vid_lora_info if target == "vid" else self._img_lora_info

        if meta["has_metadata"]:
            parts = []
            if meta["trigger_words"]:
                tw_preview = ", ".join(meta["trigger_words"][:6])
                if len(meta["trigger_words"]) > 6:
                    tw_preview += f" (+{len(meta['trigger_words']) - 6} more)"
                parts.append(f"Triggers: {tw_preview}")
            parts.append(f"Weight: {meta['recommended_weight']:.2f}")
            if meta["base_model"]:
                parts.append(f"Base: {meta['base_model']}")
            src = ", ".join(meta["metadata_sources"])
            parts.append(f"Source: {src}")
            if meta["example_prompts"]:
                parts.append(f"{len(meta['example_prompts'])} example prompt(s)")
            info_label.setText(" | ".join(parts))
        else:
            info_label.setText("No sidecar metadata found. Double-click to configure manually.")

    def _on_lora_checked(self, item, prompt_widget: QPlainTextEdit, target: str) -> None:
        """Smart LoRA toggle — uses auto-parsed sidecar metadata + intensity."""
        # Respect the picker's signal suppression during refresh
        picker = self._vid_lora if target == "vid" else self._img_lora
        if picker._suppressing_signals:
            return

        checked = item.checkState() == Qt.CheckState.Checked
        filename = item.text()
        lora_dir = self._lora_dir_for(target)

        # Try auto-prompter first (sidecar metadata)
        tag = ""
        if lora_dir:
            lora_path = str(Path(lora_dir) / filename)
            intensity_slider = self._vid_intensity if target == "vid" else self._img_intensity
            intensity = intensity_slider.value() / 100.0
            try:
                tag, meta = auto_lora_prompt_tag(lora_path, intensity)
            except Exception:
                logger.debug("Auto-prompter failed for %s, falling back", filename, exc_info=True)

        # Fallback to manual metadata
        if not tag:
            tag = lora_prompt_tag(filename)

        current = prompt_widget.toPlainText()
        if checked:
            if tag not in current:
                separator = ", " if current.strip() else ""
                prompt_widget.setPlainText(f"{current}{separator}{tag}")
            # Update info display
            self._on_lora_selected(item, target)
        else:
            # Remove the LoRA tag — match by <lora:stem:...> pattern
            stem = Path(filename).stem
            # Remove the full tag + optional trailing trigger text
            pattern = rf",?\s*<lora:{re.escape(stem)}:[^>]+>(?:\s+[^<,\n]+)?"
            cleaned = re.sub(pattern, "", current).strip(", ")
            prompt_widget.setPlainText(cleaned)

    # ── Qwen prompt enhancement (LoRA-tag-safe) ──────────────────────────

    def _on_enhance(self, target: str, which: str) -> None:
        """Launch Qwen enhancement. Strips LoRA tags first, re-inserts after."""
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_enhance_status(target, d),
        ):
            return

        from sdqt.deps import check_and_install
        if not check_and_install("transformers", self):
            return

        # Get current prompt text
        if target == "vid":
            prompt_widget = self._vid_prompt if which == "pos" else self._vid_neg_prompt
        else:
            prompt_widget = self._img_prompt if which == "pos" else self._img_neg_prompt
        current = prompt_widget.toPlainText()

        # Strip LoRA tags before sending to Qwen (only for positive prompts)
        if which == "pos":
            clean_text, lora_tags = extract_lora_tags(current)
        else:
            clean_text = current
            lora_tags = []

        # Determine style
        if target == "vid":
            style = "wan"
        else:
            style = "sdxl"

        # Save state for callback
        self._enhance_target = target
        self._enhance_which = which
        self._enhance_lora_tags = lora_tags

        # Disable enhance buttons
        self._set_enhance_buttons_enabled(False)
        self._show_enhance_status(target, "Enhancing prompt...")

        worker = PromptEnhanceWorker(clean_text, style, which, self.state, parent=self)
        worker.finished_ok.connect(self._on_enhance_done)
        worker.error.connect(self._on_enhance_error)
        worker.status.connect(lambda s: self._show_enhance_status(self._enhance_target, s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    def _on_enhance_done(self, result: str) -> None:
        target = self._enhance_target
        which = self._enhance_which

        # Re-insert LoRA tags that were stripped
        if which == "pos":
            result = reinsert_lora_tags(result, self._enhance_lora_tags)

        # Write result back to the correct prompt widget
        if target == "vid":
            widget = self._vid_prompt if which == "pos" else self._vid_neg_prompt
        else:
            widget = self._img_prompt if which == "pos" else self._img_neg_prompt
        widget.setPlainText(result)

        self._set_enhance_buttons_enabled(True)
        self._show_enhance_status(target, "Prompt enhanced.")

    def _on_enhance_error(self, msg: str) -> None:
        self._set_enhance_buttons_enabled(True)
        self._show_enhance_status(self._enhance_target, f"Enhance error: {msg}")

    def _set_enhance_buttons_enabled(self, enabled: bool) -> None:
        self._vid_enhance_pos.setEnabled(enabled)
        self._vid_enhance_neg.setEnabled(enabled)
        self._vid_auto_direct.setEnabled(enabled)
        self._img_enhance_pos.setEnabled(enabled)
        self._img_enhance_neg.setEnabled(enabled)

    def _show_enhance_status(self, target: str, msg: str) -> None:
        if target == "vid":
            self._vid_status.setText(msg)
        else:
            self._img_status.setText(msg)

    # ── Auto Direct — per-second timeline generation ────────────────────

    @Slot()
    def _on_auto_direct(self) -> None:
        """Use Qwen to generate per-second timeline prompts from a concept."""
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._vid_status.setText(d),
        ):
            return
        from sdqt.deps import check_and_install
        if not check_and_install("transformers", self):
            return

        concept = self._vid_prompt.toPlainText().strip()
        if not concept:
            self._vid_status.setText("Type a concept first, then click Auto Direct.")
            return

        # Strip LoRA tags — they'll be re-inserted at the front of each second
        clean_concept, lora_tags = extract_lora_tags(concept)
        self._enhance_lora_tags = lora_tags

        # Calculate duration from selected preset
        idx = self._vid_preset.currentIndex()
        _, _, _, duration_frames, *_ = _VIDEO_PRESETS[idx]
        seconds = (duration_frames - 1) // 16

        self._set_enhance_buttons_enabled(False)
        self._vid_status.setText(f"Auto-directing {seconds}s timeline...")

        user_msg = (
            f"Concept: {clean_concept}\n"
            f"Duration: {seconds} seconds (generate from 0 to {seconds} seconds inclusive)"
        )

        from sdqt.workers.base import BaseWorker

        class _AutoDirectWorker(BaseWorker):
            def __init__(self, state, system, user, parent=None):
                super().__init__(parent)
                self._state = state
                self._system = system
                self._user = user

            def do_work(self) -> str:
                self.status.emit("Loading Qwen 3.5 4B...")
                self._state.load_qwen()
                self.status.emit("Directing timeline...")
                messages = [
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": self._user},
                ]
                return self._state.generate_chat_response(messages, max_new_tokens=1024)

        worker = _AutoDirectWorker(self.state, self._cur_auto_direct_system, user_msg, parent=self)
        worker.finished_ok.connect(self._on_auto_direct_done)
        worker.error.connect(self._on_auto_direct_error)
        worker.status.connect(lambda s: self._vid_status.setText(s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    def _on_auto_direct_done(self, result: str) -> None:
        # Re-insert LoRA tags before each per-second line if present
        if self._enhance_lora_tags:
            tag_block = " ".join(self._enhance_lora_tags)
            lines = result.strip().splitlines()
            rebuilt = []
            for line in lines:
                line = line.strip()
                if line.startswith("(at "):
                    # Insert LoRA tags inside the parenthesized prompt
                    # (at N seconds: <lora tags> description)
                    match = re.match(r'\(at (\d+) seconds?:\s*', line)
                    if match:
                        prefix = match.group(0)
                        rest = line[len(prefix):]
                        # Strip trailing paren
                        if rest.endswith(")"):
                            rest = rest[:-1]
                        rebuilt.append(f"{prefix}{tag_block} {rest})")
                    else:
                        rebuilt.append(line)
                else:
                    rebuilt.append(line)
            result = "\n".join(rebuilt)

        self._vid_prompt.setPlainText(result)
        self._set_enhance_buttons_enabled(True)
        self._vid_status.setText("Timeline directed!")

    def _on_auto_direct_error(self, msg: str) -> None:
        self._set_enhance_buttons_enabled(True)
        self._vid_status.setText(f"Auto Direct error: {msg}")

    # ── Video generation ─────────────────────────────────────────────────

    @Slot()
    def _on_video_generate(self) -> None:
        from sdqt.models.manager import check_and_prompt_download
        from sdqt.widgets.generation_params import MODEL_TYPE_TO_FEATURE

        img = self._vid_source.image_path
        if not img:
            self._vid_status.setText("Drop a source image first.")
            return

        # Read preset
        idx = self._vid_preset.currentIndex()
        _, model_key, resolution, duration, steps, guidance, flow_shift, solver, phases, switch = _VIDEO_PRESETS[idx]

        # Ensure models are downloaded
        feature = MODEL_TYPE_TO_FEATURE.get(model_key, "video_gen")
        if not check_and_prompt_download(
            feature, self.state.model_registry, self,
            on_progress=lambda f, d: self._vid_status.setText(d),
        ):
            return

        # Ensure pipeline is loaded
        from sdqt.widgets.generation_params import get_video_backend
        backend = get_video_backend(model_key)
        pipeline = self.state.pipeline
        if pipeline is None or self.state.active_video_backend != backend:
            self._vid_status.setText(f"Loading {backend.upper()} video pipeline...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
                pipeline = self.state.pipeline
            except Exception as exc:
                self._vid_status.setText(f"Failed to load pipeline: {exc}")
                return
        if pipeline is None:
            self._vid_status.setText("Pipeline not loaded. Check Settings.")
            return

        # Collect LoRA selections
        activated_loras = self._vid_lora.get_activated()
        lora_multipliers = self._vid_lora.get_multipliers()

        # Build config with preset values
        cfg = ProjectConfig.load(self.project_path)
        cfg.mode = 1
        cfg.prompt = self._vid_prompt.toPlainText()
        cfg.negative_prompt = self._vid_neg_prompt.toPlainText()
        cfg.model_type = model_key
        cfg.resolution = resolution
        cfg.video_length = duration
        cfg.fps = 16
        cfg.num_inference_steps = steps
        cfg.guidance_scale = guidance
        cfg.guidance2_scale = guidance
        cfg.flow_shift = flow_shift
        cfg.sample_solver = solver
        cfg.seed = -1
        cfg.denoising_strength = 1.0
        cfg.guidance_phases = phases
        cfg.switch_threshold = switch
        # Disable all advanced features
        cfg.nag_tea_enabled = False
        cfg.NAG_scale = 0.0
        cfg.NAG_tau = 0.0
        cfg.NAG_alpha = 0.0
        cfg.tea_cache_setting = "off"
        cfg.tea_cache_start_step_perc = 0.0
        cfg.quality_overrides_enabled = False
        cfg.slg_switch = 0
        cfg.slg_layers = []
        cfg.apg_switch = 0
        cfg.cfg_star_switch = 0
        cfg.cfg_zero_step = 0
        cfg.motion_amplitude = 1.0
        cfg.color_correction_strength = 0.0
        cfg.discard_last_frames = 0
        cfg.self_refiner_setting = 0
        cfg.use_guidance_m1 = False
        cfg.temporal_upsampling = ""
        cfg.spatial_upsampling = ""
        cfg.film_grain_intensity = 0.0
        # LoRA
        cfg.activated_loras = activated_loras
        cfg.loras_multipliers = lora_multipliers
        cfg.image_path = img
        cfg.save(self.project_path)

        self._vid_gen_btn.setVisible(False)
        self._vid_abort_btn.setVisible(True)
        self._vid_status.setText("Generating video...")

        worker = InferenceWorker(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=1,
            image_path=img,
            global_config=self.state.global_config,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._vid_status.setText(d))
        worker.finished_ok.connect(self._on_video_done)
        worker.error.connect(self._on_video_error)
        worker.finished.connect(worker.deleteLater)
        self._video_worker = worker
        worker.start()

    def _on_video_done(self, result_path: str) -> None:
        self._vid_gen_btn.setVisible(True)
        self._vid_abort_btn.setVisible(False)
        self._vid_preview.load_video(result_path, auto_play=True)
        self._vid_status.setText("Done!")

    def _on_video_error(self, msg: str) -> None:
        self._vid_gen_btn.setVisible(True)
        self._vid_abort_btn.setVisible(False)
        self._vid_status.setText(f"Error: {msg}")

    @Slot()
    def _on_video_abort(self) -> None:
        if self._video_worker:
            self._video_worker.abort()

    # ── Image generation ─────────────────────────────────────────────────

    @Slot()
    def _on_image_generate(self) -> None:
        # Determine mode: txt2img if no source, img2img if source provided
        src = self._img_source.image_path
        prompt = self._img_prompt.toPlainText()
        if not prompt and not src:
            self._img_status.setText("Enter a prompt (and optionally drop a source image).")
            return

        # Get active image pipeline strategy
        from sdqt.tabs.image_tabs.model_family import STRATEGY_MAP, SDXL_STRATEGY
        strategy = SDXL_STRATEGY
        # Try to find the currently loaded strategy
        for s in STRATEGY_MAP.values():
            if s.get_pipeline(self.state) is not None:
                strategy = s
                break

        pipeline = strategy.get_pipeline(self.state)
        if pipeline is None:
            self._img_status.setText(f"Loading {strategy.display_name} pipeline...")
            try:
                getattr(self.state, strategy.load_method)()
                pipeline = strategy.get_pipeline(self.state)
            except Exception as exc:
                self._img_status.setText(f"Failed to load pipeline: {exc}")
                return
        if pipeline is None:
            self._img_status.setText("Pipeline not loaded. Check Settings.")
            return

        # Read preset
        idx = self._img_preset.currentIndex()
        _, width, height, steps, cfg_scale = _IMAGE_PRESETS[idx]

        neg = self._img_neg_prompt.toPlainText()

        # Build config
        cfg = ProjectConfig.load(self.project_path)
        strategy.set_config(cfg, "prompt", prompt)
        strategy.set_config(cfg, "negative_prompt", neg)
        strategy.set_config(cfg, "width", width)
        strategy.set_config(cfg, "height", height)
        strategy.set_config(cfg, "steps", steps)
        strategy.set_config(cfg, "cfg_scale", cfg_scale)
        strategy.set_config(cfg, "seed", -1)
        strategy.set_config(cfg, "batch_size", 1)
        strategy.set_config(cfg, "batch_count", 1)
        cfg.save(self.project_path)

        self._img_gen_btn.setVisible(False)
        self._img_abort_btn.setVisible(True)
        self._img_status.setText("Generating image...")

        if src:
            # img2img
            strategy.set_config(cfg, "denoising_strength", 0.7)
            cfg.save(self.project_path)
            WorkerCls = strategy.resolve_worker("img2img")
            worker = WorkerCls(pipeline, self.project_name, src, cfg, parent=self)
        else:
            # txt2img
            WorkerCls = strategy.resolve_worker("txt2img")
            worker = WorkerCls(pipeline, self.project_name, cfg, parent=self)

        worker.progress.connect(lambda f, d: self._img_status.setText(d))
        worker.finished_ok.connect(self._on_image_done)
        worker.error.connect(self._on_image_error)
        worker.finished.connect(worker.deleteLater)
        self._image_worker = worker
        worker.start()

    def _on_image_done(self, paths: list[str]) -> None:
        self._img_gen_btn.setVisible(True)
        self._img_abort_btn.setVisible(False)
        self._img_gallery.load_images(paths)
        self._img_status.setText(f"Generated {len(paths)} image(s).")

    def _on_image_error(self, msg: str) -> None:
        self._img_gen_btn.setVisible(True)
        self._img_abort_btn.setVisible(False)
        self._img_status.setText(f"Error: {msg}")

    @Slot()
    def _on_image_abort(self) -> None:
        if self._image_worker:
            self._image_worker.abort()
