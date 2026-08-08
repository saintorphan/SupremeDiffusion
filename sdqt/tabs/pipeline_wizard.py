"""Pipeline Wizard tab — guided divide-and-conquer video workflow.

Walks beginners through the professional workflow:
  1. Import/Generate Base Image (wide establishing shot)
  2. Realism Pass (progressive img2img)
  3. Inpaint Cleanup
  4. Orbit Clip (2-3 sec, camera orbits the scene)
  5. Extract Stills (smart pick best angles)
  6. Scene Generation (zoom in, generate action)

The Scene Graph tree on the left shows the full shot hierarchy.
The right panel changes based on the current workflow step.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.scene_graph import (
    EdgeType,
    GenerationParams,
    NodeType,
    SceneGraph,
    ShotNode,
)
from sdqt.state import AppState
from sdqt.tabs.base import BaseTab
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.scene_tree import SceneTreeWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.inference import InferenceWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker
from sdqt.tabs.still_grabber import _StillGrabWorker
from sdqt.tabs.image_tabs.model_family import SDXL_STRATEGY, STRATEGY_MAP
from sdqt.widgets.image_params import ImageParamsWidget
from sdqt.widgets.lora_metadata import lora_prompt_tag
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.grouped_checkpoint_combo import GroupedCheckpointCombo
from sdqt.widgets.quality_warning import QualityWarningBanner
from sdqt.widgets.lora_picker import LoRAPickerWidget
from sdqt.models.pipeline_template import (
    PipelineTemplate, scan_templates, import_template, export_template,
)
from supremediffusion.config.project_config import ProjectConfig

logger = logging.getLogger(__name__)

from sdqt.utils.lora_tags import extract_lora_tags, reinsert_lora_tags
from sdqt.data.presets import (
    BODY_TYPES as _BODY_TYPES,
    CAMERA_ANGLES as _CAMERA_ANGLES,
    CHARACTERS as _CHARACTERS,
    CLOTHING as _CLOTHING,
    FILM_STYLES as _STYLES,
    HAIR_OPTIONS as _HAIR_OPTIONS,
    LIGHTING as _LIGHTING,
    ORBIT_PROMPTS as _ORBIT_PROMPTS,
    PLANNER_CAMERA_MOVES as _PLANNER_CAMERA_MOVES,
    PLANNER_SUBJECT_CHANGES as _PLANNER_SUBJECT_CHANGES,
    REALISM_PRESETS as _REALISM_PRESETS,
    SCENES as _SCENES,
    DRIFT_GREEN as _DRIFT_GREEN,
    DRIFT_YELLOW as _DRIFT_YELLOW,
)


# ── Workflow steps ──
_STEPS = [
    ("1. Create Your Starting Image", (
        "WHAT TO DO: First set up your Model (checkpoint, LoRA, settings) at the top.\n"
        "Then create the wide shot that everything builds from.\n"
        "   Option A: Render from the 3D viewport (Daz2Supreme tab)\n"
        "   Option B: Pick options from the Scene Composer and click Generate\n"
        "   Option C: Drop in an image you already have\n"
        "THEN: Click 'Add to Scene Graph' and hit Next >"
    )),
    ("2. Plan Your Shot", (
        "WHAT TO DO: Decide what happens in your scene.\n"
        "   1. Pick a Camera Move (or leave Static)\n"
        "   2. Pick a Subject Change (body transformation, head turn, etc.)\n"
        "   3. If transformation: pick the Type, how far it goes (Stage), and the Ending\n"
        "   4. Click 'Compose End Prompt' to preview the text\n"
        "   5. Click 'Generate End Frame Preview' to see your target\n"
        "   6. Click 'Lock Pair' when both frames look good\n"
        "SHORTCUT: Hit 'Run Clip Chain' to auto-generate everything from here!"
    )),
    ("3. Make It Realistic", (
        "WHAT TO DO: Turn your starting image into a photorealistic photo.\n"
        "   Select your starting image in the tree on the left.\n"
        "   Click 'Auto Realism' — it runs 6 versions and picks the best one.\n"
        "   The best result gets a star. That's your clean starting point.\n"
        "SKIP IF: Your starting image already looks realistic."
    )),
    ("4. Fix Problems (Inpaint)", (
        "WHAT TO DO: Fix any weird faces, hands, or artifacts.\n"
        "   Select the best realism result in the tree.\n"
        "   Click 'Open in Image Tab (Inpaint)' to paint over problem areas.\n"
        "   The AI will regenerate just the painted region.\n"
        "SKIP IF: It already looks clean — click 'Skip (Looks Good)'."
    )),
    ("5. Face Restoration", (
        "WHAT TO DO: Enhance faces using AI face restoration.\n"
        "   Select the face restore model (GFPGAN, RestoreFormer, etc.)\n"
        "   Adjust the strength slider.\n"
        "   Click 'Run Face Restore' — the AI will detect and fix all faces.\n"
        "RESULT: Sharper, more detailed faces with fewer artifacts."
    )),
    ("6. Face Swap (Optional)", (
        "WHAT TO DO: Swap a face onto your generated character.\n"
        "   Drop in a reference face image (clear front-facing photo).\n"
        "   Pick a swap model and optional enhancer.\n"
        "   Click 'Run Face Swap' to apply.\n"
        "SKIP IF: You don't need face swapping — click 'Skip'."
    )),
    ("7. Orbit Clip", (
        "WHAT TO DO: Generate a short video where the camera moves around the scene.\n"
        "   Select your best image in the tree.\n"
        "   Pick a camera movement (Slow Orbit is recommended).\n"
        "   Set duration to 2 seconds (3 max — longer = things start to warp).\n"
        "   Click 'Generate Orbit Clip' and wait.\n"
        "RESULT: A 2-3 second clip showing your scene from multiple angles."
    )),
    ("8. Extract Stills", (
        "WHAT TO DO: Pull the best frames out of your orbit clip.\n"
        "   Select the orbit clip in the tree.\n"
        "   Click 'Extract Stills' — it pulls 12 frames and auto-picks the top 3.\n"
        "   The top 3 get a star. Each one faces a different direction.\n"
        "RESULT: A library of starting points for close-up shots."
    )),
    ("9. Scene Shot", (
        "WHAT TO DO: Pick a still and generate your close-up or action clip.\n"
        "   Select a starred still from the tree.\n"
        "   Write what happens (e.g. 'camera zooms in on the woman').\n"
        "   Click Generate, then 'Extract Best Frame' to get a new starting point.\n"
        "   Repeat: each extracted frame can start a new scene shot.\n"
        "RESULT: Short clips that chain together into your final video."
    )),
    ("10. Final Polish", (
        "WHAT TO DO: Fix clip-to-clip seams and apply Hires Fix.\n"
        "   1. INPAINT SEAMS — clip transitions have slight inconsistencies.\n"
        "      Open seam frames in the Inpaint tab to fix them.\n"
        "   2. HIRES FIX — upscale the FINAL stitched output.\n"
        "      Pick an upscale model, set the scale (2x recommended).\n"
        "      Optionally enable img2img second pass for extra detail.\n"
        "WARNING: Run Hires Fix ONLY on the final output, never per-clip!\n"
        "Per-clip upscaling causes progressive degradation."
    )),
]

# ── Shot Planner — transformation presets loaded from templates ──
# See pipeline_templates/*.json — no hardcoded transformation content.
# _TRANSFORMATION_TYPES, _TRANSFORMATION_STAGES, and _ENDING_VARIANTS
# are now loaded at runtime via PipelineTemplate.


class PipelineWizardTab(BaseTab):
    """Guided pipeline wizard with integrated scene graph."""

    # Signal to request a render from another tab's GL viewport
    request_viewport_render = Signal(int, int)  # width, height
    # Slot to receive the rendered image path
    viewport_render_ready = Signal(str)  # path to rendered PNG

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._graph = SceneGraph()
        self._selected_node_id: str | None = None
        self._current_step: int = 0
        self._worker = None
        self._enhance_worker: PromptEnhanceWorker | None = None
        self._enhance_target: str = ""
        self._enhance_lora_tags: list[str] = []
        self._checkpoint_infos: list = []
        self._active_strategy = SDXL_STRATEGY
        self._mode = "guided"  # "guided", "freestyle", "power"

        # Template system
        self._templates: list[PipelineTemplate] = []
        self._active_template: PipelineTemplate | None = None
        self._load_templates()

        self._build_ui()

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._load_graph()
        self._populate_wizard_params()

    def _populate_wizard_params(self) -> None:
        """Scan checkpoint/LoRA/VAE directories and populate all model widgets."""
        try:
            gc = self.state.global_config
            mp = gc.model_paths

            # Checkpoints — unified scan across all model dirs
            from supremediffusion.models.sd_models import (
                scan_all_image_models, scan_checkpoints, scan_vaes,
            )
            self._checkpoint_infos = scan_all_image_models(gc)
            if not self._checkpoint_infos:
                # Fallback to SD-only scan
                ckpt_dir = mp.get("sd_checkpoint_dir", "")
                if ckpt_dir:
                    self._checkpoint_infos = scan_checkpoints(ckpt_dir, recursive=True)

            # Populate grouped checkpoint combo (top-level widget)
            if hasattr(self, "_ckpt_combo"):
                self._ckpt_combo.populate(self._checkpoint_infos)

            # Also populate the hidden ImageParamsWidget for config collection
            checkpoints = [info.name for info in self._checkpoint_infos]

            # VAEs
            vae_dir = mp.get("sd_vae_dir", "")
            vaes = scan_vaes(vae_dir) if vae_dir else []

            # Samplers
            try:
                from supremediffusion.models.sd_samplers import list_samplers
                samplers = list_samplers()
            except Exception:
                samplers = ["DPM++ 2M SDE", "DPM++ 2M", "Euler a", "Euler", "DDIM"]

            # Schedulers
            schedulers = ["Karras", "Automatic", "Exponential"]

            # LoRAs — set directory on the grouped LoRA picker
            lora_dir = mp.get("sd_lora_dir", "")
            if hasattr(self, "_wizard_lora_picker"):
                self._wizard_lora_picker.set_lora_dir(lora_dir)

            # Also populate hidden ImageParamsWidget for backward compat
            loras = []
            if lora_dir:
                lora_path = Path(lora_dir)
                if lora_path.is_dir():
                    for f in sorted(lora_path.rglob("*.safetensors")):
                        rel = f.relative_to(lora_path)
                        loras.append(str(rel.with_suffix("")).replace("\\", "/"))

            self._wizard_params.populate_options(
                checkpoints=checkpoints,
                vaes=vaes,
                samplers=samplers,
                schedulers=schedulers,
                loras=loras,
            )

            # Populate visible wizard dropdowns
            if hasattr(self, "_wizard_sampler"):
                self._wizard_sampler.clear()
                self._wizard_sampler.addItems(samplers)
                idx = self._wizard_sampler.findText("DPM++ 2M SDE")
                if idx >= 0:
                    self._wizard_sampler.setCurrentIndex(idx)
            if hasattr(self, "_wizard_scheduler"):
                self._wizard_scheduler.clear()
                self._wizard_scheduler.addItems(schedulers)
                idx = self._wizard_scheduler.findText("Karras")
                if idx >= 0:
                    self._wizard_scheduler.setCurrentIndex(idx)
            if hasattr(self, "_wizard_vae"):
                self._wizard_vae.clear()
                self._wizard_vae.addItem("Automatic")
                self._wizard_vae.addItems(vaes)

            # Also set hidden widget defaults
            idx = self._wizard_params.sampler.findText("DPM++ 2M SDE")
            if idx >= 0:
                self._wizard_params.sampler.setCurrentIndex(idx)
            idx = self._wizard_params.scheduler.findText("Karras")
            if idx >= 0:
                self._wizard_params.scheduler.setCurrentIndex(idx)

        except Exception as exc:
            logger.warning("Failed to populate wizard params: %s", exc)

    # ── Template system ─────────────────────────────────────────────────

    def _app_root(self) -> Path:
        """Return the application root directory (where pipeline_templates/ lives)."""
        return Path(self.state.global_config.models_root).parent

    def _load_templates(self) -> None:
        """Scan pipeline_templates/ and load all available templates."""
        self._templates = scan_templates(self._app_root())
        if self._templates:
            self._active_template = self._templates[0]
        else:
            self._active_template = None
        logger.info("Loaded %d pipeline templates", len(self._templates))

    def _get_transformation_types(self) -> list[tuple[str, str]]:
        """Return (label, key) list from the active template."""
        if self._active_template:
            return self._active_template.transformation_types
        return [("Custom (Write Your Own)", "custom")]

    def _get_transformation_stages(self, key: str) -> list[tuple[str, str, int]]:
        """Return [(label, prompt, drift)] for a transformation key."""
        if self._active_template:
            return self._active_template.transformation_stages.get(key, [])
        return []

    def _get_ending_variants(self, key: str) -> list[tuple[str, str, str]]:
        """Return [(label, prompt, type)] for a transformation key."""
        if self._active_template:
            return self._active_template.ending_variants.get(key, [])
        return []

    def _on_template_changed(self, index: int) -> None:
        """Handle template selector change (from Step 2 panel)."""
        if 0 <= index < len(self._templates):
            self._active_template = self._templates[index]
            # Sync the top-level combo
            self._top_template_combo.blockSignals(True)
            self._top_template_combo.setCurrentIndex(index)
            self._top_template_combo.blockSignals(False)
            self._repopulate_transform_combos()
            self._show_status(f"Template: {self._active_template.name}")

    def _on_top_template_changed(self, index: int) -> None:
        """Handle template selector change (from top bar)."""
        if 0 <= index < len(self._templates):
            self._active_template = self._templates[index]
            # Sync the Step 2 panel combo if it exists
            if hasattr(self, "_template_combo"):
                self._template_combo.blockSignals(True)
                self._template_combo.setCurrentIndex(index)
                self._template_combo.blockSignals(False)
            self._repopulate_transform_combos()
            self._show_status(f"Template: {self._active_template.name}")

    def _on_open_templates_folder(self) -> None:
        """Open the pipeline_templates folder in the system file manager."""
        import subprocess, sys
        folder = str(self._app_root() / "pipeline_templates")
        if sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _repopulate_transform_combos(self) -> None:
        """Refresh the transformation type dropdown from the active template."""
        self._planner_transform_type.blockSignals(True)
        self._planner_transform_type.clear()
        for label, _ in self._get_transformation_types():
            self._planner_transform_type.addItem(label)
        self._planner_transform_type.blockSignals(False)
        self._on_transform_type_changed(0)

    def _on_import_template(self) -> None:
        """Import a template JSON file."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Pipeline Template", "",
            "JSON Templates (*.json);;All Files (*)",
        )
        if not path:
            return
        t = import_template(path, self._app_root())
        if t:
            self._templates.append(t)
            self._top_template_combo.addItem(t.name)
            if hasattr(self, "_template_combo"):
                self._template_combo.addItem(t.name)
            self._top_template_combo.setCurrentIndex(len(self._templates) - 1)
            self._show_status(f"Imported template: {t.name}")
        else:
            self._show_status("Failed to import template.")

    def _on_export_template(self) -> None:
        """Export the active template to a file."""
        if not self._active_template:
            self._show_status("No template selected.")
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pipeline Template",
            self._active_template.name.replace(" ", "_") + ".json",
            "JSON Templates (*.json);;All Files (*)",
        )
        if path:
            if export_template(self._active_template, path):
                self._show_status(f"Exported: {path}")
            else:
                self._show_status("Export failed.")

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Main splitter: tree (left) | workflow panel (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Scene Tree ──
        self._scene_tree = SceneTreeWidget()
        self._scene_tree.node_selected.connect(self._on_node_selected)
        self._scene_tree.node_action.connect(self._on_node_action)
        splitter.addWidget(self._scene_tree)

        # ── Right: Workflow panel ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Mode + step indicator row
        step_row = QHBoxLayout()
        step_row.setSpacing(8)

        # Mode toggle
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Guided", "guided")
        self._mode_combo.addItem("Freestyle", "freestyle")
        self._mode_combo.addItem("Power (Batch)", "power")
        self._mode_combo.setToolTip(
            "Guided: Step-by-step workflow for beginners\n"
            "Freestyle: All tools unlocked, drive from tree context menu\n"
            "Power: Batch queue for advanced users (coming soon)"
        )
        self._mode_combo.setFixedWidth(120)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        step_row.addWidget(self._mode_combo)

        self._step_label = QLabel("Step 1: Import Base")
        self._step_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        step_row.addWidget(self._step_label)
        step_row.addStretch()
        self._btn_prev = QPushButton("< Back")
        self._btn_prev.setFixedHeight(24)
        self._btn_prev.clicked.connect(self._prev_step)
        step_row.addWidget(self._btn_prev)
        self._btn_next = QPushButton("Next >")
        self._btn_next.setFixedHeight(24)
        self._btn_next.clicked.connect(self._next_step)
        step_row.addWidget(self._btn_next)
        right_layout.addLayout(step_row)

        # Template bar — always visible so users know templates exist
        tmpl_bar = QHBoxLayout()
        tmpl_bar.setSpacing(4)
        tmpl_bar.addWidget(QLabel("Template:"))
        self._top_template_combo = QComboBox()
        self._top_template_combo.setToolTip(
            "PIPELINE TEMPLATE — defines what transformations are available in Step 2.\n\n"
            "The Default template has general-purpose presets (aging, emotion, weather, etc.).\n"
            "Import custom templates for specialized workflows.\n\n"
            "To import: click 'Import' and pick a .json template file.\n"
            "To get templates: ask the community or create your own!"
        )
        for t in self._templates:
            self._top_template_combo.addItem(t.name)
        self._top_template_combo.currentIndexChanged.connect(self._on_top_template_changed)
        tmpl_bar.addWidget(self._top_template_combo, 1)

        self._top_import_btn = QPushButton("Import")
        self._top_import_btn.setFixedHeight(24)
        self._top_import_btn.setToolTip("Import a pipeline template (.json file)")
        self._top_import_btn.clicked.connect(self._on_import_template)
        tmpl_bar.addWidget(self._top_import_btn)

        self._top_export_btn = QPushButton("Export")
        self._top_export_btn.setFixedHeight(24)
        self._top_export_btn.setToolTip("Export the current template to share")
        self._top_export_btn.clicked.connect(self._on_export_template)
        tmpl_bar.addWidget(self._top_export_btn)

        open_folder_btn = QPushButton("📂")
        open_folder_btn.setFixedSize(24, 24)
        open_folder_btn.setToolTip("Open the pipeline_templates folder")
        open_folder_btn.clicked.connect(self._on_open_templates_folder)
        tmpl_bar.addWidget(open_folder_btn)

        right_layout.addLayout(tmpl_bar)

        # Step description
        self._step_desc = QLabel("")
        self._step_desc.setWordWrap(True)
        self._step_desc.setStyleSheet("color: #aaa; font-size: 12px; padding: 4px; "
                                      "background: #2a2a2a; border-radius: 4px;")
        right_layout.addWidget(self._step_desc)

        # ── Stacked panels for each step ──
        self._stack = QStackedWidget()
        right_layout.addWidget(self._stack, 1)

        self._build_import_panel()          # 0 — Create Starting Image
        self._build_shot_planner_panel()    # 1 — Plan Your Shot
        self._build_realism_panel()         # 2 — Make It Realistic
        self._build_inpaint_panel()         # 3 — Fix Problems (Inpaint)
        self._build_face_restore_panel()    # 4 — Face Restoration
        self._build_face_swap_panel()       # 5 — Face Swap (Optional)
        self._build_orbit_panel()           # 6 — Orbit Clip
        self._build_extract_panel()         # 7 — Extract Stills
        self._build_scene_panel()           # 8 — Scene Shot
        self._build_final_polish_panel()    # 9 — Final Polish

        # Progress bar + abort
        prog_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(18)
        prog_row.addWidget(self._progress_bar, 1)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setFixedHeight(22)
        self._abort_btn.setStyleSheet("color: #ff5252; font-weight: bold;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        prog_row.addWidget(self._abort_btn)
        right_layout.addLayout(prog_row)

        # Status
        self._status = QLabel("")
        right_layout.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        outer.addWidget(splitter)
        self._update_step_ui()

    # ── Step 1: Import Base ──────────────────────────────────────────────

    def _build_import_panel(self) -> None:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Scrollable area for all the composer options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll_inner = QWidget()
        layout = QVBoxLayout(scroll_inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ══════════════════════════════════════════════════════════════════
        # MODEL SETUP — FIRST THING IN THE UI
        # ══════════════════════════════════════════════════════════════════
        # Shared style for labels in the Model Setup section
        _LBL_STYLE = "font-size: 13px; color: #e0e0e0; font-weight: bold;"
        _FIELD_LABEL_WIDTH = 90

        model_group = QGroupBox("Model Setup — choose your AI model and settings")
        model_group.setStyleSheet(
            "QGroupBox { color: #4CAF50; border-color: #2e7d32; }"
            "QGroupBox::title { background: #1a2e1a; border-color: #2e7d32; }"
        )
        ml = QVBoxLayout(model_group)
        ml.setSpacing(8)
        ml.setContentsMargins(12, 18, 12, 12)

        # Grouped checkpoint selector
        self._ckpt_combo = GroupedCheckpointCombo()
        self._ckpt_combo.setToolTip(
            "CHECKPOINT — The AI model that generates your image.\n\n"
            "Grouped by model family (subfolder):\n"
            "  • PonyXL: stylized, anime/cartoon friendly\n"
            "  • IllustriousXL: high-detail illustration\n"
            "  • SDXL: photorealistic, general purpose\n"
            "  • Flux: next-gen architecture\n"
            "  • SD 1.5: classic, lightweight\n\n"
            "TIP: For photorealism, pick an SDXL or IllustriousXL checkpoint."
        )
        self._ckpt_combo.checkpoint_changed.connect(self._on_checkpoint_family_changed)
        ml.addWidget(self._ckpt_combo)

        # Hidden ImageParamsWidget — still used for config collection
        self._wizard_params = ImageParamsWidget()
        self._wizard_params.set_family(self._active_strategy)
        self._wizard_params.setVisible(False)
        ml.addWidget(self._wizard_params)

        # When a LoRA is checked/unchecked in the old widget, sync
        self._wizard_params.lora_toggled.connect(self._on_wizard_lora_toggled)

        # VAE row
        vae_row = QHBoxLayout()
        lbl_vae = QLabel("VAE:")
        lbl_vae.setFixedWidth(_FIELD_LABEL_WIDTH)
        lbl_vae.setStyleSheet(_LBL_STYLE)
        vae_row.addWidget(lbl_vae)
        self._wizard_vae = QComboBox()
        self._wizard_vae.addItem("Automatic")
        self._wizard_vae.setToolTip(
            "VAE — Controls color accuracy and detail sharpness.\n\n"
            "'Automatic' uses the checkpoint's built-in VAE.\n"
            "Override only if colors look washed out or oversaturated."
        )
        vae_row.addWidget(self._wizard_vae, 1)
        self._lbl_vae_row = lbl_vae  # for show/hide based on strategy
        ml.addLayout(vae_row)

        # Sampler + Scheduler row
        sampler_row = QHBoxLayout()
        lbl_sampler = QLabel("Sampler:")
        lbl_sampler.setFixedWidth(_FIELD_LABEL_WIDTH)
        lbl_sampler.setStyleSheet(_LBL_STYLE)
        sampler_row.addWidget(lbl_sampler)
        self._wizard_sampler = QComboBox()
        self._wizard_sampler.setToolTip(
            "SAMPLER — The algorithm that generates the image step by step.\n\n"
            "Recommended: DPM++ 2M SDE (best quality)"
        )
        sampler_row.addWidget(self._wizard_sampler, 1)
        lbl_sched = QLabel("Scheduler:")
        lbl_sched.setStyleSheet(_LBL_STYLE)
        sampler_row.addWidget(lbl_sched)
        self._wizard_scheduler = QComboBox()
        self._wizard_scheduler.setToolTip("SCHEDULER — Controls noise reduction. 'Karras' is the safest choice.")
        sampler_row.addWidget(self._wizard_scheduler, 1)
        self._lbl_sampler_row = lbl_sampler
        ml.addLayout(sampler_row)

        # Steps / CFG row
        param_row1 = QHBoxLayout()
        param_row1.setSpacing(12)
        lbl_steps = QLabel("Steps:")
        lbl_steps.setStyleSheet(_LBL_STYLE)
        param_row1.addWidget(lbl_steps)
        self._wizard_steps = QSpinBox()
        self._wizard_steps.setRange(1, 150)
        self._wizard_steps.setValue(35)
        self._wizard_steps.setMinimumWidth(80)
        self._wizard_steps.setToolTip(
            "STEPS — How many times the AI refines the image.\n"
            "30-35 is the sweet spot for production images."
        )
        param_row1.addWidget(self._wizard_steps)
        lbl_cfg = QLabel("CFG:")
        lbl_cfg.setStyleSheet(_LBL_STYLE)
        param_row1.addWidget(lbl_cfg)
        self._wizard_cfg = QDoubleSpinBox()
        self._wizard_cfg.setRange(0.0, 30.0)
        self._wizard_cfg.setSingleStep(0.5)
        self._wizard_cfg.setValue(7.5)
        self._wizard_cfg.setMinimumWidth(80)
        self._wizard_cfg.setToolTip(
            "CFG SCALE — How closely the AI follows your prompt.\n"
            "5-7 is recommended. 15+ causes artifacts."
        )
        param_row1.addWidget(self._wizard_cfg)
        lbl_clip = QLabel("Clip Skip:")
        lbl_clip.setStyleSheet(_LBL_STYLE)
        param_row1.addWidget(lbl_clip)
        self._wizard_clip_skip = QSpinBox()
        self._wizard_clip_skip.setRange(1, 12)
        self._wizard_clip_skip.setValue(1)
        self._wizard_clip_skip.setMinimumWidth(60)
        self._wizard_clip_skip.setToolTip("CLIP SKIP — 1 for SDXL, 2 for Pony/anime models.")
        param_row1.addWidget(self._wizard_clip_skip)
        self._lbl_clip_skip = lbl_clip
        param_row1.addStretch()
        ml.addLayout(param_row1)

        # Seed / Width / Height / Batch row
        param_row2 = QHBoxLayout()
        param_row2.setSpacing(12)
        lbl_seed = QLabel("Seed:")
        lbl_seed.setStyleSheet(_LBL_STYLE)
        param_row2.addWidget(lbl_seed)
        self._wizard_seed = QSpinBox()
        self._wizard_seed.setRange(-1, 2147483647)
        self._wizard_seed.setValue(-1)
        self._wizard_seed.setMinimumWidth(100)
        self._wizard_seed.setToolTip("SEED — -1 for random. Same seed = same image.")
        param_row2.addWidget(self._wizard_seed)
        lbl_w = QLabel("Width:")
        lbl_w.setStyleSheet(_LBL_STYLE)
        param_row2.addWidget(lbl_w)
        self._wizard_width = QSpinBox()
        self._wizard_width.setRange(256, 2048)
        self._wizard_width.setSingleStep(64)
        self._wizard_width.setValue(1024)
        self._wizard_width.setMinimumWidth(80)
        param_row2.addWidget(self._wizard_width)
        lbl_h = QLabel("Height:")
        lbl_h.setStyleSheet(_LBL_STYLE)
        param_row2.addWidget(lbl_h)
        self._wizard_height = QSpinBox()
        self._wizard_height.setRange(256, 2048)
        self._wizard_height.setSingleStep(64)
        self._wizard_height.setValue(1024)
        self._wizard_height.setMinimumWidth(80)
        param_row2.addWidget(self._wizard_height)
        lbl_batch = QLabel("Batch:")
        lbl_batch.setStyleSheet(_LBL_STYLE)
        param_row2.addWidget(lbl_batch)
        self._wizard_batch_count = QSpinBox()
        self._wizard_batch_count.setRange(1, 20)
        self._wizard_batch_count.setValue(1)
        self._wizard_batch_count.setMinimumWidth(60)
        self._wizard_batch_count.setToolTip("How many images to generate. 3-4 when experimenting.")
        param_row2.addWidget(self._wizard_batch_count)
        param_row2.addStretch()
        ml.addLayout(param_row2)

        # LoRA picker (grouped by base model)
        self._wizard_lora_picker = LoRAPickerWidget("LoRA — add style/character models")
        self._wizard_lora_picker.setToolTip(
            "LoRA — Small add-on models that change style, characters, or concepts.\n\n"
            "Check a LoRA to activate it. Double-click to configure.\n"
            "Use the filter to show only LoRAs for your checkpoint family.\n\n"
            "WARNING: Using a LoRA for the wrong model family\n"
            "(e.g., Pony LoRA with SDXL checkpoint) will produce bad results."
        )
        self._wizard_lora_picker.set_state(self.state)
        self._wizard_lora_picker.lora_toggled.connect(self._on_wizard_lora_toggled)
        ml.addWidget(self._wizard_lora_picker)

        # LoRA weights
        lora_weight_row = QHBoxLayout()
        lbl_lw = QLabel("LoRA Weights:")
        lbl_lw.setStyleSheet(_LBL_STYLE)
        lora_weight_row.addWidget(lbl_lw)
        self._wizard_lora_weights = QLineEdit()
        self._wizard_lora_weights.setPlaceholderText("e.g. 0.8, 0.6 (one per checked LoRA)")
        self._wizard_lora_weights.setToolTip(
            "LORA WEIGHTS — Comma-separated multipliers for each checked LoRA.\n"
            "1.0 = full strength, 0.7 = subtle, 1.5 = extra strong"
        )
        lora_weight_row.addWidget(self._wizard_lora_weights)
        ml.addLayout(lora_weight_row)

        # Quality warning banner
        self._quality_warning = QualityWarningBanner()
        ml.addWidget(self._quality_warning)

        # Enhance style selector
        style_row = QHBoxLayout()
        style_row.setSpacing(6)
        lbl_es = QLabel("Enhance Style:")
        lbl_es.setStyleSheet(_LBL_STYLE)
        style_row.addWidget(lbl_es)
        self._enhance_style_combo = QComboBox()
        self._enhance_style_combo.setMinimumWidth(180)
        self._enhance_style_combo.setToolTip(
            "ENHANCE STYLE — Controls how 'Enhance with AI' formats your prompt.\n\n"
            "  • Auto-Detect: guesses from your checkpoint name\n"
            "  • Pony/SDXL/Illustrious/FLUX: specific formatting"
        )
        _ENHANCE_STYLES = [
            ("Auto-Detect (from checkpoint)", "auto"),
            ("Pony (Realistic)", "pony_real"),
            ("Pony (Anime)", "pony_anime"),
            ("SDXL (Natural Language)", "sdxl"),
            ("SD 1.5 (Booru Tags)", "sd15"),
            ("Illustrious", "illustrious"),
            ("NoobAI", "noobai"),
            ("FLUX (Long Prose)", "flux"),
            ("SD3 (Natural Language)", "sd3"),
        ]
        for label, key in _ENHANCE_STYLES:
            self._enhance_style_combo.addItem(label, key)
        style_row.addWidget(self._enhance_style_combo, 1)
        ml.addLayout(style_row)

        layout.addWidget(model_group)

        # ══════════════════════════════════════════════════════════════════
        # SCENE CREATION — below model setup
        # ══════════════════════════════════════════════════════════════════

        layout.addWidget(QLabel(
            "Now create your scene — pick options and hit Generate,\n"
            "or render from the 3D viewport, or drop an existing image."
        ))

        # ── Render from 3D Viewport (Daz optional) ──
        render_group = QGroupBox("Option 1 — Render from 3D Viewport")
        render_group.setStyleSheet(
            "QGroupBox { color: #64b5f6; border-color: #1565c0; }"
            "QGroupBox::title { background: #1a1a2e; border-color: #1565c0; }"
        )
        rl = QVBoxLayout(render_group)
        rl.addWidget(QLabel(
            "If you have a scene set up in the Daz2Supreme tab,\n"
            "render it directly as your base image. No Daz3D required\n"
            "if you loaded OBJ/GLB models or used Image-to-3D."
        ))
        render_row = QHBoxLayout()
        self._render_res = QComboBox()
        self._render_res.addItem("1024 x 1024", (1024, 1024))
        self._render_res.addItem("1280 x 720", (1280, 720))
        self._render_res.addItem("1920 x 1080", (1920, 1080))
        self._render_res.addItem("768 x 768", (768, 768))
        self._render_res.addItem("512 x 512", (512, 512))
        self._render_res.setCurrentIndex(0)
        render_row.addWidget(QLabel("Resolution:"))
        render_row.addWidget(self._render_res)
        render_row.addStretch()
        self._render_viewport_btn = QPushButton("Render from 3D Viewport")
        self._render_viewport_btn.setStyleSheet(
            "font-weight: bold; padding: 8px 16px; "
            "background: #1565c0; color: white;"
        )
        self._render_viewport_btn.setToolTip(
            "Captures the current 3D viewport from Daz2Supreme tab.\n"
            "Works with Daz imports, OBJ/GLB models, or Image-to-3D meshes.\n"
            "Daz3D is optional — any 3D scene in the viewport works."
        )
        self._render_viewport_btn.clicked.connect(self._on_render_from_viewport)
        render_row.addWidget(self._render_viewport_btn)
        rl.addLayout(render_row)
        layout.addWidget(render_group)

        # Separator
        sep0 = QLabel("— OR compose with AI —")
        sep0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep0.setStyleSheet("color: #666; padding: 4px;")
        layout.addWidget(sep0)

        # ── Scene Composer ──
        composer = QGroupBox("Scene Composer — pick your options")
        cl = QVBoxLayout(composer)
        cl.setSpacing(4)

        def _add_combo(label_text, items, default=0):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(90)
            row.addWidget(lbl)
            combo = QComboBox()
            for display, _ in items:
                combo.addItem(display)
            combo.setCurrentIndex(default)
            row.addWidget(combo, 1)
            cl.addLayout(row)
            return combo

        self._comp_scene = _add_combo("Scene:", _SCENES, 0)
        self._comp_chars = _add_combo("Characters:", _CHARACTERS, 0)
        self._comp_body = _add_combo("Body Type:", _BODY_TYPES, 0)
        self._comp_hair = _add_combo("Hair:", _HAIR_OPTIONS, 0)
        self._comp_clothing = _add_combo("Clothing:", _CLOTHING, 0)
        self._comp_lighting = _add_combo("Lighting:", _LIGHTING, 0)
        self._comp_camera = _add_combo("Camera:", _CAMERA_ANGLES, 0)
        self._comp_style = _add_combo("Style:", _STYLES, 0)

        layout.addWidget(composer)

        # Composed prompt preview
        layout.addWidget(QLabel("Composed prompt (editable):"))
        self._comp_prompt = QPlainTextEdit()
        self._comp_prompt.setMaximumHeight(80)
        self._comp_prompt.setPlaceholderText("Pick options above, then click Compose...")
        layout.addWidget(self._comp_prompt)

        # Compose + Enhance + Generate buttons
        btn_row1 = QHBoxLayout()
        self._comp_compose_btn = QPushButton("Compose Prompt")
        self._comp_compose_btn.setToolTip("Build prompt from your selections above")
        self._comp_compose_btn.setStyleSheet("padding: 6px 16px;")
        self._comp_compose_btn.clicked.connect(self._on_compose_prompt)
        btn_row1.addWidget(self._comp_compose_btn)

        self._comp_enhance_btn = QPushButton("Enhance with AI")
        self._comp_enhance_btn.setToolTip(
            "Use Qwen AI to expand your prompt into a richer, more detailed description.\n"
            "This adds lighting, mood, composition, and quality details automatically.\n"
            "Tip: Compose first, then Enhance to get the best results."
        )
        self._comp_enhance_btn.setStyleSheet(
            "padding: 6px 16px; background: #6a1b9a; color: white; font-weight: bold;"
        )
        self._comp_enhance_btn.clicked.connect(lambda: self._on_enhance_prompt("comp"))
        btn_row1.addWidget(self._comp_enhance_btn)

        self._comp_generate_btn = QPushButton("Generate Base Image")
        self._comp_generate_btn.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px 20px; "
            "background: #2e7d32; color: white;"
        )
        self._comp_generate_btn.setToolTip("Generate a wide establishing shot from the prompt above")
        self._comp_generate_btn.clicked.connect(self._on_generate_base)
        btn_row1.addWidget(self._comp_generate_btn)
        layout.addLayout(btn_row1)

        # Generated preview
        self._comp_preview = QLabel("Generated image will appear here")
        self._comp_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._comp_preview.setMinimumHeight(200)
        self._comp_preview.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._comp_preview)

        # Add generated image to graph
        btn_row2 = QHBoxLayout()
        btn_row2.addStretch()
        self._comp_add_btn = QPushButton("Add Generated Image to Scene Graph")
        self._comp_add_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self._comp_add_btn.clicked.connect(self._on_add_generated_base)
        self._comp_add_btn.setEnabled(False)
        btn_row2.addWidget(self._comp_add_btn)
        btn_row2.addStretch()
        layout.addLayout(btn_row2)

        # Separator
        sep = QLabel("— OR drop an existing image —")
        sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sep.setStyleSheet("color: #666; padding: 8px;")
        layout.addWidget(sep)

        # Drop zone (existing)
        self._import_drop = ImageDropWidget("Drop Existing Image Here", thumb_height=180)
        layout.addWidget(self._import_drop)

        btn_row3 = QHBoxLayout()
        btn_row3.addStretch()
        self._import_btn = QPushButton("Add Dropped Image to Scene Graph")
        self._import_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self._import_btn.clicked.connect(self._on_import_base)
        btn_row3.addWidget(self._import_btn)
        btn_row3.addStretch()
        layout.addLayout(btn_row3)

        layout.addStretch()
        scroll.setWidget(scroll_inner)
        outer.addWidget(scroll)
        self._stack.addWidget(panel)

        # Track last generated base image path
        self._generated_base_path: str | None = None

    # ── Step 2: Shot Planner ────────────────────────────────────────────

    def _build_shot_planner_panel(self) -> None:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll_inner = QWidget()
        layout = QVBoxLayout(scroll_inner)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # ── Side-by-side frame previews ──
        frames_group = QGroupBox("Start Frame → End Frame")
        # Global card style applies from app.py stylesheet
        frames_layout = QHBoxLayout(frames_group)
        frames_layout.setSpacing(8)

        # Start frame
        start_col = QVBoxLayout()
        start_col.addWidget(QLabel("START (from Step 1)"))
        self._planner_start_img = QLabel("Add a base image in Step 1 first")
        self._planner_start_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._planner_start_img.setMinimumSize(280, 180)
        self._planner_start_img.setStyleSheet(
            "background: #1a1a1a; border: 2px solid #2e7d32; border-radius: 4px;"
        )
        start_col.addWidget(self._planner_start_img)
        frames_layout.addLayout(start_col)

        # Arrow
        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow.setStyleSheet("font-size: 28px; font-weight: bold; color: #aaa;")
        frames_layout.addWidget(arrow)

        # End frame
        end_col = QVBoxLayout()
        end_col.addWidget(QLabel("END (your target)"))
        self._planner_end_img = QLabel("Configure below, then Generate")
        self._planner_end_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._planner_end_img.setMinimumSize(280, 180)
        self._planner_end_img.setStyleSheet(
            "background: #1a1a1a; border: 2px dashed #666; border-radius: 4px;"
        )
        end_col.addWidget(self._planner_end_img)
        frames_layout.addLayout(end_col)

        layout.addWidget(frames_group)

        # ── Drift Risk Meter ──
        risk_group = QGroupBox("Drift Risk")
        risk_layout = QVBoxLayout(risk_group)
        self._drift_bar = QProgressBar()
        self._drift_bar.setRange(0, 6)
        self._drift_bar.setValue(0)
        self._drift_bar.setTextVisible(False)
        self._drift_bar.setFixedHeight(14)
        self._drift_bar.setStyleSheet(
            "QProgressBar { background: #333; border-radius: 7px; }"
            "QProgressBar::chunk { background: #4caf50; border-radius: 7px; }"
        )
        risk_layout.addWidget(self._drift_bar)
        self._drift_label = QLabel("Select options below to see drift risk.")
        self._drift_label.setWordWrap(True)
        self._drift_label.setStyleSheet("color: #aaa; font-size: 11px;")
        risk_layout.addWidget(self._drift_label)
        self._drift_split_suggestion = QLabel("")
        self._drift_split_suggestion.setWordWrap(True)
        self._drift_split_suggestion.setStyleSheet(
            "color: #ffb74d; font-size: 11px; font-weight: bold;"
        )
        self._drift_split_suggestion.setVisible(False)
        risk_layout.addWidget(self._drift_split_suggestion)
        layout.addWidget(risk_group)

        # ── Camera Move ──
        move_group = QGroupBox("Camera Move")
        move_layout = QVBoxLayout(move_group)
        self._planner_camera = QComboBox()
        for label, _, _ in _PLANNER_CAMERA_MOVES:
            self._planner_camera.addItem(label)
        self._planner_camera.currentIndexChanged.connect(self._on_planner_changed)
        move_layout.addWidget(self._planner_camera)
        layout.addWidget(move_group)

        # ── Subject Change ──
        subj_group = QGroupBox("Subject Change")
        subj_layout = QVBoxLayout(subj_group)
        self._planner_subject = QComboBox()
        for label, _, _ in _PLANNER_SUBJECT_CHANGES:
            self._planner_subject.addItem(label)
        self._planner_subject.currentIndexChanged.connect(self._on_planner_changed)
        subj_layout.addWidget(self._planner_subject)
        layout.addWidget(subj_group)

        # ── Template selector ──
        tmpl_group = QGroupBox("Transformation Template")
        tmpl_layout = QHBoxLayout(tmpl_group)
        self._template_combo = QComboBox()
        self._template_combo.setToolTip(
            "Pipeline templates define what transformations are available.\n"
            "Each template has its own stages and endings.\n"
            "Import custom templates or create your own JSON files."
        )
        for t in self._templates:
            self._template_combo.addItem(t.name)
        self._template_combo.currentIndexChanged.connect(self._on_template_changed)
        tmpl_layout.addWidget(self._template_combo, 1)

        tmpl_import_btn = QPushButton("Import")
        tmpl_import_btn.setToolTip("Import a template JSON file")
        tmpl_import_btn.clicked.connect(self._on_import_template)
        tmpl_layout.addWidget(tmpl_import_btn)

        tmpl_export_btn = QPushButton("Export")
        tmpl_export_btn.setToolTip("Export the current template to share with others")
        tmpl_export_btn.clicked.connect(self._on_export_template)
        tmpl_layout.addWidget(tmpl_export_btn)
        layout.addWidget(tmpl_group)

        # ── Transformation controls (visible when "Body Transformation" selected) ──
        self._transform_group = QGroupBox("Transformation")
        self._transform_group.setVisible(False)
        tl = QVBoxLayout(self._transform_group)

        # Type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._planner_transform_type = QComboBox()
        for label, _ in self._get_transformation_types():
            self._planner_transform_type.addItem(label)
        self._planner_transform_type.currentIndexChanged.connect(
            self._on_transform_type_changed
        )
        type_row.addWidget(self._planner_transform_type, 1)
        tl.addLayout(type_row)

        # Stage
        stage_row = QHBoxLayout()
        stage_row.addWidget(QLabel("Stage:"))
        self._planner_stage = QComboBox()
        self._planner_stage.currentIndexChanged.connect(self._on_planner_changed)
        stage_row.addWidget(self._planner_stage, 1)
        tl.addLayout(stage_row)

        # Ending variant
        ending_row = QHBoxLayout()
        ending_row.addWidget(QLabel("Ending:"))
        self._planner_ending = QComboBox()
        self._planner_ending.addItem("None — stop at this stage")
        self._planner_ending.currentIndexChanged.connect(self._on_planner_changed)
        ending_row.addWidget(self._planner_ending, 1)
        tl.addLayout(ending_row)

        # Custom prompt (for custom transform type)
        self._planner_custom_prompt = QPlainTextEdit()
        self._planner_custom_prompt.setMaximumHeight(60)
        self._planner_custom_prompt.setPlaceholderText(
            "Describe the transformation for the end frame..."
        )
        self._planner_custom_prompt.setVisible(False)
        tl.addWidget(self._planner_custom_prompt)

        layout.addWidget(self._transform_group)

        # ── Composed end-frame prompt ──
        layout.addWidget(QLabel("End-frame prompt (editable):"))
        self._planner_end_prompt = QPlainTextEdit()
        self._planner_end_prompt.setMaximumHeight(80)
        self._planner_end_prompt.setPlaceholderText(
            "Auto-composed from your selections. Edit freely."
        )
        layout.addWidget(self._planner_end_prompt)

        # ── Clip chain preview ──
        chain_group = QGroupBox("Clip Chain Plan")
        chain_layout = QVBoxLayout(chain_group)
        self._clip_chain_label = QLabel(
            "Your selections will generate a clip chain plan here.\n"
            "Each row = one 2-3 second clip. The tool enforces one change per clip."
        )
        self._clip_chain_label.setWordWrap(True)
        self._clip_chain_label.setStyleSheet("color: #aaa; font-size: 11px;")
        chain_layout.addWidget(self._clip_chain_label)
        layout.addWidget(chain_group)

        # ── Action buttons ──
        btn_row = QHBoxLayout()

        self._planner_compose_btn = QPushButton("Compose End Prompt")
        self._planner_compose_btn.setStyleSheet("padding: 6px 16px;")
        self._planner_compose_btn.clicked.connect(self._on_planner_compose)
        btn_row.addWidget(self._planner_compose_btn)

        self._planner_enhance_btn = QPushButton("Enhance with AI")
        self._planner_enhance_btn.setToolTip(
            "Use Qwen AI to expand the end-frame prompt into richer detail.\n"
            "Adds vivid descriptions of the transformation, mood, and visual style.\n"
            "Tip: Compose first, then Enhance."
        )
        self._planner_enhance_btn.setStyleSheet(
            "padding: 6px 16px; background: #6a1b9a; color: white; font-weight: bold;"
        )
        self._planner_enhance_btn.clicked.connect(lambda: self._on_enhance_prompt("planner"))
        btn_row.addWidget(self._planner_enhance_btn)

        self._planner_generate_btn = QPushButton("Generate End Frame Preview")
        self._planner_generate_btn.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px 20px; "
            "background: #1565c0; color: white;"
        )
        self._planner_generate_btn.clicked.connect(self._on_planner_generate_end)
        btn_row.addWidget(self._planner_generate_btn)
        layout.addLayout(btn_row)

        lock_row = QHBoxLayout()
        lock_row.addStretch()
        self._planner_lock_btn = QPushButton("Lock Pair — Save Start + End to Scene Graph")
        self._planner_lock_btn.setStyleSheet(
            "font-weight: bold; padding: 8px 20px; "
            "background: #2e7d32; color: white;"
        )
        self._planner_lock_btn.setEnabled(False)
        self._planner_lock_btn.clicked.connect(self._on_planner_lock_pair)
        lock_row.addWidget(self._planner_lock_btn)
        lock_row.addStretch()
        layout.addLayout(lock_row)

        # ── Run Clip Chain (auto-executor) ──
        chain_exec_group = QGroupBox("Auto-Execute Clip Chain")
        chain_exec_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #ff9800; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; }"
        )
        cel = QVBoxLayout(chain_exec_group)
        cel.addWidget(QLabel(
            "Lock a pair above, then hit Run. The system will:\n"
            "1. Run realism pass on the start frame\n"
            "2. Generate each clip in the chain (one change per clip)\n"
            "3. Extract the best frame between clips\n"
            "4. Use that frame as the start of the next clip\n"
            "5. Add everything to the scene graph automatically"
        ))

        chain_btn_row = QHBoxLayout()
        chain_btn_row.addStretch()
        self._run_chain_btn = QPushButton("Run Clip Chain (Full Auto)")
        self._run_chain_btn.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 10px 28px; "
            "background: #e65100; color: white;"
        )
        self._run_chain_btn.setToolTip(
            "Executes the entire clip chain hands-free.\n"
            "Generates each clip, extracts best frame, chains to next.\n"
            "Walk away and come back to a finished scene."
        )
        self._run_chain_btn.setEnabled(False)
        self._run_chain_btn.clicked.connect(self._on_run_clip_chain)
        chain_btn_row.addWidget(self._run_chain_btn)
        chain_btn_row.addStretch()
        cel.addLayout(chain_btn_row)

        # Chain progress
        self._chain_progress_label = QLabel("")
        self._chain_progress_label.setWordWrap(True)
        self._chain_progress_label.setStyleSheet("color: #aaa; font-size: 11px;")
        cel.addWidget(self._chain_progress_label)

        layout.addWidget(chain_exec_group)

        layout.addStretch()
        scroll.setWidget(scroll_inner)
        outer.addWidget(scroll)
        self._stack.addWidget(panel)

        # Track generated end frame
        self._planner_end_path: str | None = None

        # Populate initial transform type
        self._on_transform_type_changed(0)

    # ── Step 3: Realism Pass (was Step 2) ───────────────────────────────

    def _build_realism_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(
            "Run img2img at different denoise strengths to find the sweet spot.\n"
            "Lower = preserves more original detail. Higher = more creative freedom."
        ))

        # Source preview
        self._realism_source = QLabel("Select a node from the tree first")
        self._realism_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._realism_source.setMinimumHeight(150)
        self._realism_source.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._realism_source)

        # Denoise strength
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Denoise:"))
        self._realism_preset = QComboBox()
        for label, _ in _REALISM_PRESETS:
            self._realism_preset.addItem(label)
        self._realism_preset.setCurrentIndex(2)
        strength_row.addWidget(self._realism_preset)

        strength_row.addWidget(QLabel("Prompt:"))
        self._realism_prompt = QComboBox()
        self._realism_prompt.addItem("photorealistic, highly detailed, 8k")
        self._realism_prompt.addItem("photorealistic, cinematic lighting, film grain")
        self._realism_prompt.addItem("hyperrealistic photograph, natural lighting")
        self._realism_prompt.setEditable(True)
        self._realism_prompt.setMinimumWidth(200)
        strength_row.addWidget(self._realism_prompt, 1)
        layout.addLayout(strength_row)

        # Batch run
        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch:"))
        self._realism_batch = QSpinBox()
        self._realism_batch.setRange(1, 6)
        self._realism_batch.setValue(3)
        self._realism_batch.setToolTip(
            "Run multiple passes at different strengths.\n"
            "Results appear as children in the scene graph."
        )
        batch_row.addWidget(self._realism_batch)
        batch_row.addStretch()
        self._realism_run = QPushButton("Run Realism Pass")
        self._realism_run.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self._realism_run.clicked.connect(self._on_realism_run)
        batch_row.addWidget(self._realism_run)
        layout.addLayout(batch_row)

        # Auto Realism — one button, all strengths, auto-pick best
        auto_row = QHBoxLayout()
        auto_row.addStretch()
        self._realism_auto = QPushButton("Auto Realism (All Strengths + Best Pick)")
        self._realism_auto.setStyleSheet(
            "font-weight: bold; padding: 8px 20px; "
            "background: #2e7d32; color: white;"
        )
        self._realism_auto.setToolTip(
            "Runs img2img at all 6 denoise strengths automatically,\n"
            "scores the results, and stars the best one.\n"
            "One button — walk away and come back to the best result."
        )
        self._realism_auto.clicked.connect(self._on_auto_realism)
        auto_row.addWidget(self._realism_auto)
        auto_row.addStretch()
        layout.addLayout(auto_row)

        layout.addStretch()
        self._stack.addWidget(panel)

    # ── Step 3: Inpaint ──────────────────────────────────────────────────

    def _build_inpaint_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(
            "Select a node from the tree to inpaint.\n"
            "This will open the image in the Image tab's inpaint mode.\n"
            "(Coming soon: inline brush tool)"
        ))

        self._inpaint_source = QLabel("Select a node")
        self._inpaint_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._inpaint_source.setMinimumHeight(200)
        self._inpaint_source.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._inpaint_source, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._inpaint_open = QPushButton("Open in Image Tab (Inpaint)")
        self._inpaint_open.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self._inpaint_open.clicked.connect(self._on_inpaint_open)
        btn_row.addWidget(self._inpaint_open)

        self._inpaint_skip = QPushButton("Skip (Looks Good)")
        self._inpaint_skip.clicked.connect(self._next_step)
        btn_row.addWidget(self._inpaint_skip)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        self._stack.addWidget(panel)

    # ── Step 4: Orbit Clip ───────────────────────────────────────────────

    def _build_orbit_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        warning = QLabel(
            "IMPORTANT: Keep orbit clips to 2-3 seconds MAX.\n"
            "5 seconds = drift. 2 seconds = consistency.\n"
            "Characters should be STILL — only the camera moves."
        )
        warning.setStyleSheet(
            "color: #ffb74d; font-weight: bold; padding: 6px; "
            "background: #3e2723; border-radius: 4px;"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)

        # Source
        self._orbit_source = QLabel("Select your best realism-passed image from the tree")
        self._orbit_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orbit_source.setMinimumHeight(150)
        self._orbit_source.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._orbit_source)

        # Prompt template
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Camera:"))
        self._orbit_prompt = QComboBox()
        for label, _ in _ORBIT_PROMPTS:
            self._orbit_prompt.addItem(label)
        self._orbit_prompt.setCurrentIndex(0)
        prompt_row.addWidget(self._orbit_prompt, 1)
        layout.addLayout(prompt_row)

        # Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration:"))
        self._orbit_duration = QComboBox()
        self._orbit_duration.addItem("2 seconds (33 frames) — Safe", 33)
        self._orbit_duration.addItem("3 seconds (49 frames) — Max recommended", 49)
        self._orbit_duration.addItem("5 seconds (81 frames) — RISKY: may drift!", 81)
        self._orbit_duration.setCurrentIndex(1)
        dur_row.addWidget(self._orbit_duration, 1)
        layout.addLayout(dur_row)

        # Generate
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._orbit_gen = QPushButton("Generate Orbit Clip")
        self._orbit_gen.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px 20px;"
        )
        self._orbit_gen.clicked.connect(self._on_orbit_generate)
        btn_row.addWidget(self._orbit_gen)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Preview
        self._orbit_preview = VideoPlayerWidget("Orbit Preview")
        layout.addWidget(self._orbit_preview, 1)

        self._stack.addWidget(panel)

    # ── Step 5: Extract Stills ───────────────────────────────────────────

    def _build_extract_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(
            "Extract stills from your orbit clip.\n"
            "Each still faces a different direction — these are your angle library.\n"
            "Star the best ones to use as starting points for scene shots."
        ))

        # Count
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Extract:"))
        self._extract_count = QSpinBox()
        self._extract_count.setRange(3, 30)
        self._extract_count.setValue(12)
        self._extract_count.setToolTip("Number of evenly-spaced stills to extract")
        count_row.addWidget(self._extract_count)
        count_row.addWidget(QLabel("stills"))
        count_row.addStretch()

        self._extract_btn = QPushButton("Extract Stills")
        self._extract_btn.setStyleSheet("font-weight: bold; padding: 8px 16px;")
        self._extract_btn.clicked.connect(self._on_extract_stills)
        count_row.addWidget(self._extract_btn)
        layout.addLayout(count_row)

        # Info
        self._extract_info = QLabel(
            "After extraction, stills appear as children in the scene graph.\n"
            "Right-click to star/unstar. Starred stills become your scene starting points."
        )
        self._extract_info.setWordWrap(True)
        self._extract_info.setStyleSheet("color: #aaa;")
        layout.addWidget(self._extract_info)

        layout.addStretch()
        self._stack.addWidget(panel)

    # ── Step 6: Scene Shot ───────────────────────────────────────────────

    # ── Step 5: Face Restoration ──────────────────────────────────────

    def _build_face_restore_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel(
            "Enhance faces using AI face restoration.\n"
            "This fixes blurry/distorted faces from the generation process.\n"
            "Select the best image from the tree, then run face restore."
        ))

        # Source preview
        self._face_restore_source = QLabel("Select an image from the tree")
        self._face_restore_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_restore_source.setMinimumHeight(180)
        self._face_restore_source.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._face_restore_source)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Face Restore Model:"))
        self._face_restore_model = QComboBox()
        self._face_restore_model.setMinimumWidth(200)
        self._face_restore_model.setToolTip(
            "Face restoration model:\n"
            "  GFPGAN — Best for general face cleanup\n"
            "  RestoreFormer — Good for heavily degraded faces\n"
            "  CodeFormer — High fidelity face reconstruction"
        )
        model_row.addWidget(self._face_restore_model, 1)
        model_row.addWidget(QLabel("Strength:"))
        self._face_restore_strength = QDoubleSpinBox()
        self._face_restore_strength.setRange(0.0, 1.0)
        self._face_restore_strength.setSingleStep(0.1)
        self._face_restore_strength.setValue(0.7)
        self._face_restore_strength.setToolTip(
            "Restoration strength:\n"
            "  0.3 = subtle cleanup\n"
            "  0.7 = recommended balance\n"
            "  1.0 = maximum restoration (may look artificial)"
        )
        model_row.addWidget(self._face_restore_strength)
        layout.addLayout(model_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self._face_restore_run = QPushButton("Run Face Restore")
        self._face_restore_run.setStyleSheet(
            "font-weight: bold; padding: 8px 20px; "
            "background: #2e7d32; color: white;"
        )
        self._face_restore_run.clicked.connect(self._on_face_restore_run)
        btn_row.addWidget(self._face_restore_run)
        self._face_restore_skip = QPushButton("Skip (Faces Look Good)")
        self._face_restore_skip.setStyleSheet("padding: 8px 16px;")
        self._face_restore_skip.clicked.connect(self._next_step)
        btn_row.addWidget(self._face_restore_skip)
        layout.addLayout(btn_row)

        # Result preview
        self._face_restore_result = QLabel("Result will appear here")
        self._face_restore_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_restore_result.setMinimumHeight(180)
        self._face_restore_result.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._face_restore_result)

        layout.addStretch()
        self._stack.addWidget(panel)

        # Scan for face restore models on startup
        self._scan_face_restore_models()

    def _scan_face_restore_models(self) -> None:
        """Populate face restore model dropdown."""
        try:
            mp = self.state.global_config.model_paths
            models_dir = mp.get("upscaler_dir", "")
            if models_dir:
                from supremediffusion.postprocessing.ai_upscale import scan_models
                models = scan_models(models_dir)
                # Filter for likely face restore models
                face_models = [m for m in models if any(
                    kw in m.lower() for kw in
                    ["gfpgan", "restoreformer", "codeformer", "face"]
                )]
                if face_models:
                    self._face_restore_model.clear()
                    self._face_restore_model.addItems(face_models)
                    return
            # Fallback defaults
            self._face_restore_model.clear()
            self._face_restore_model.addItem("GFPGANv1.4.pth")
            self._face_restore_model.addItem("RestoreFormer.pth")
        except Exception as exc:
            logger.warning("Failed to scan face restore models: %s", exc)
            self._face_restore_model.addItem("GFPGANv1.4.pth")

    # ── Step 6: Face Swap (Optional) ───────────────────────────────────

    def _build_face_swap_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel(
            "OPTIONAL: Swap a face onto your generated character.\n"
            "Drop in a clear, front-facing reference photo of the face you want.\n"
            "Skip this step if you don't need face swapping."
        ))

        # Source preview (target image from tree)
        self._face_swap_target = QLabel("Select an image from the tree (target)")
        self._face_swap_target.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_swap_target.setMinimumHeight(120)
        self._face_swap_target.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._face_swap_target)

        # Reference face drop zone
        self._face_swap_ref = ImageDropWidget("Drop Reference Face Here", thumb_height=120)
        self._face_swap_ref.setToolTip(
            "Drop a clear, front-facing photo of the face you want.\n"
            "Best results with good lighting and straight-on angle."
        )
        layout.addWidget(self._face_swap_ref)

        # Settings row
        settings_row = QHBoxLayout()
        settings_row.addWidget(QLabel("Swap Model:"))
        self._face_swap_model = QComboBox()
        self._face_swap_model.addItem("inswapper_128", "inswapper_128")
        self._face_swap_model.addItem("reswapper_128 (better)", "reswapper_128")
        self._face_swap_model.addItem("reswapper_256 (best)", "reswapper_256")
        self._face_swap_model.setToolTip(
            "Swap model quality:\n"
            "  inswapper_128 — fast, decent quality\n"
            "  reswapper_128 — improved consistency\n"
            "  reswapper_256 — best quality, slower"
        )
        settings_row.addWidget(self._face_swap_model)
        settings_row.addWidget(QLabel("Enhancer:"))
        self._face_swap_enhancer = QComboBox()
        self._face_swap_enhancer.addItem("None", "")
        self._face_swap_enhancer.addItem("GFPGAN", "gfpgan")
        self._face_swap_enhancer.addItem("CodeFormer", "codeformer")
        self._face_swap_enhancer.addItem("GPEN", "gpen")
        self._face_swap_enhancer.addItem("RestoreFormer++", "restoreformer")
        self._face_swap_enhancer.setCurrentIndex(1)
        self._face_swap_enhancer.setToolTip("Post-swap face enhancement (recommended: GFPGAN)")
        settings_row.addWidget(self._face_swap_enhancer)
        layout.addLayout(settings_row)

        # Action buttons
        btn_row = QHBoxLayout()
        self._face_swap_run = QPushButton("Run Face Swap")
        self._face_swap_run.setStyleSheet(
            "font-weight: bold; padding: 8px 20px; "
            "background: #1565c0; color: white;"
        )
        self._face_swap_run.clicked.connect(self._on_face_swap_run)
        btn_row.addWidget(self._face_swap_run)
        self._face_swap_skip = QPushButton("Skip (No Face Swap)")
        self._face_swap_skip.setStyleSheet("padding: 8px 16px;")
        self._face_swap_skip.clicked.connect(self._next_step)
        btn_row.addWidget(self._face_swap_skip)
        layout.addLayout(btn_row)

        # Result preview
        self._face_swap_result = QLabel("Result will appear here")
        self._face_swap_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_swap_result.setMinimumHeight(180)
        self._face_swap_result.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._face_swap_result)

        layout.addStretch()
        self._stack.addWidget(panel)

    # ── Step 9: Scene Shot ─────────────────────────────────────────────

    def _build_scene_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(
            "Pick a starred still from the tree and describe your scene.\n"
            "Tip: 'camera zooms in on the blonde girl' works well.\n"
            "Keep it to 2-3 seconds, then extract the best frame and iterate."
        ))

        # Source
        self._scene_source = QLabel("Select a starred still from the tree")
        self._scene_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scene_source.setMinimumHeight(120)
        self._scene_source.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._scene_source)

        # Prompt
        layout.addWidget(QLabel("Scene prompt:"))
        self._scene_prompt = QPlainTextEdit()
        self._scene_prompt.setMaximumHeight(80)
        self._scene_prompt.setPlaceholderText(
            "e.g. camera zooms in on the blonde girl sitting on the couch, "
            "she turns to look at the camera"
        )
        layout.addWidget(self._scene_prompt)

        # Enhance button for scene prompt
        scene_enhance_row = QHBoxLayout()
        scene_enhance_row.addStretch()
        self._scene_enhance_btn = QPushButton("Enhance with AI")
        self._scene_enhance_btn.setToolTip(
            "Use Qwen AI to expand your scene description into richer detail.\n"
            "Adds camera movement, mood, and visual specifics."
        )
        self._scene_enhance_btn.setStyleSheet(
            "padding: 4px 12px; background: #6a1b9a; color: white; font-weight: bold;"
        )
        self._scene_enhance_btn.clicked.connect(lambda: self._on_enhance_prompt("scene"))
        scene_enhance_row.addWidget(self._scene_enhance_btn)
        layout.addLayout(scene_enhance_row)

        # Duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Duration:"))
        self._scene_duration = QComboBox()
        self._scene_duration.addItem("2 seconds (33 frames) — Safe", 33)
        self._scene_duration.addItem("3 seconds (49 frames) — Max", 49)
        self._scene_duration.setCurrentIndex(0)
        dur_row.addWidget(self._scene_duration)
        dur_row.addStretch()

        self._scene_gen = QPushButton("Generate Scene Clip")
        self._scene_gen.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px 20px;"
        )
        self._scene_gen.clicked.connect(self._on_scene_generate)
        dur_row.addWidget(self._scene_gen)
        layout.addLayout(dur_row)

        # Preview
        self._scene_preview = VideoPlayerWidget("Scene Preview")
        layout.addWidget(self._scene_preview, 1)

        # Extract best frame
        extract_row = QHBoxLayout()
        extract_row.addStretch()
        self._scene_extract = QPushButton("Extract Best Frame as New Starting Point")
        self._scene_extract.setStyleSheet("padding: 6px 16px;")
        self._scene_extract.clicked.connect(self._on_scene_extract)
        extract_row.addWidget(self._scene_extract)
        extract_row.addStretch()
        layout.addLayout(extract_row)

        self._stack.addWidget(panel)

    # ── Step 10: Final Polish (Inpaint Seams + Hires Fix) ──────────────

    def _build_final_polish_panel(self) -> None:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        scroll_inner = QWidget()
        layout = QVBoxLayout(scroll_inner)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        # ── Section A: Inpaint Clip Seams ──
        seam_group = QGroupBox("Inpaint Clip Seams")
        seam_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #e65100; "
            "border-radius: 4px; margin-top: 6px; padding-top: 14px; }"
        )
        sl = QVBoxLayout(seam_group)
        sl.addWidget(QLabel(
            "Adjacent clips will have slight inconsistencies at the transition.\n"
            "Use inpainting to fix the seam frames where clips meet.\n"
            "This is normal — even at low CFG, clip boundaries need cleanup."
        ))
        seam_btn_row = QHBoxLayout()
        self._seam_inpaint_btn = QPushButton("Open Seam Frames in Inpaint")
        self._seam_inpaint_btn.setStyleSheet("padding: 6px 16px;")
        self._seam_inpaint_btn.clicked.connect(self._on_inpaint_seams)
        seam_btn_row.addWidget(self._seam_inpaint_btn)
        self._seam_skip_btn = QPushButton("Skip Seams")
        self._seam_skip_btn.setStyleSheet("padding: 6px 16px;")
        seam_btn_row.addWidget(self._seam_skip_btn)
        sl.addLayout(seam_btn_row)
        layout.addWidget(seam_group)

        # ── Section B: Hires Fix ──
        hires_group = QGroupBox("Hires Fix — Upscale Final Output")
        hires_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #2e7d32; "
            "border-radius: 4px; margin-top: 6px; padding-top: 14px; }"
        )
        hl = QVBoxLayout(hires_group)

        # Warning
        warn_lbl = QLabel(
            "IMPORTANT: Hires Fix processes the FINAL stitched output.\n"
            "Do NOT run this per-clip — it causes progressive degradation\n"
            "when clips are concatenated. Run it ONCE at the end."
        )
        warn_lbl.setStyleSheet(
            "color: #FFD54F; background: #433000; border: 1px solid #665200; "
            "border-radius: 4px; padding: 6px; font-weight: bold;"
        )
        warn_lbl.setWordWrap(True)
        hl.addWidget(warn_lbl)

        # Upscale model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Upscale Model:"))
        self._hires_model = QComboBox()
        self._hires_model.setMinimumWidth(200)
        self._hires_model.setToolTip(
            "Upscale model:\n"
            "  RealESRGAN x4 — Best for photorealistic (4x native)\n"
            "  RealESRGAN x2 — Moderate upscale, good for sharpening\n"
            "  SwinIR — Alternative high-quality upscaler"
        )
        model_row.addWidget(self._hires_model, 1)
        hl.addLayout(model_row)

        # Scale slider
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self._hires_scale = QDoubleSpinBox()
        self._hires_scale.setRange(1.0, 4.0)
        self._hires_scale.setSingleStep(0.5)
        self._hires_scale.setValue(2.0)
        self._hires_scale.setToolTip(
            "Upscale factor:\n"
            "  1.5x — Subtle sharpening\n"
            "  2.0x — Recommended (doubles resolution)\n"
            "  4.0x — Maximum (quadruples, very slow)"
        )
        scale_row.addWidget(self._hires_scale)
        self._hires_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._hires_scale_slider.setRange(10, 40)  # 1.0 to 4.0
        self._hires_scale_slider.setValue(20)
        self._hires_scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._hires_scale_slider.setTickInterval(5)
        # Sync slider and spinbox
        self._hires_scale_slider.valueChanged.connect(
            lambda v: self._hires_scale.setValue(v / 10.0)
        )
        self._hires_scale.valueChanged.connect(
            lambda v: self._hires_scale_slider.setValue(int(v * 10))
        )
        scale_row.addWidget(self._hires_scale_slider, 1)
        hl.addLayout(scale_row)

        # Method combo
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._hires_method = QComboBox()
        self._hires_method.addItem("Pixel (ESRGAN/SwinIR)", "pixel")
        self._hires_method.addItem("Lanczos", "lanczos")
        self._hires_method.setToolTip(
            "Pixel: Uses AI upscale model (best quality)\n"
            "Lanczos: Fast bicubic resize (no AI, for testing)"
        )
        method_row.addWidget(self._hires_method)
        method_row.addStretch()
        hl.addLayout(method_row)

        # Second pass (img2img) options
        self._hires_second_pass = QCheckBox("Run img2img second pass after upscale")
        self._hires_second_pass.setToolTip(
            "After upscaling, run a low-denoise img2img pass to add detail.\n"
            "Recommended: 0.3-0.5 denoise strength.\n"
            "This adds fine detail that upscaling alone can't create."
        )
        hl.addWidget(self._hires_second_pass)

        second_row = QHBoxLayout()
        second_row.addWidget(QLabel("Denoise:"))
        self._hires_denoise = QDoubleSpinBox()
        self._hires_denoise.setRange(0.1, 0.7)
        self._hires_denoise.setSingleStep(0.05)
        self._hires_denoise.setValue(0.3)
        self._hires_denoise.setToolTip(
            "Denoise strength for second pass:\n"
            "  0.2-0.3 = subtle refinement (recommended)\n"
            "  0.4-0.5 = noticeable detail addition\n"
            "  0.6-0.7 = heavy redraw (may change content)"
        )
        second_row.addWidget(self._hires_denoise)
        second_row.addWidget(QLabel("Steps:"))
        self._hires_steps = QSpinBox()
        self._hires_steps.setRange(1, 100)
        self._hires_steps.setValue(20)
        self._hires_steps.setToolTip("Inference steps for img2img second pass. 15-25 is good.")
        second_row.addWidget(self._hires_steps)
        second_row.addStretch()
        hl.addLayout(second_row)

        # Run button
        hires_btn_row = QHBoxLayout()
        self._hires_run = QPushButton("Run Hires Fix")
        self._hires_run.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px 20px; "
            "background: #2e7d32; color: white;"
        )
        self._hires_run.clicked.connect(self._on_hires_fix_run)
        hires_btn_row.addWidget(self._hires_run)
        self._hires_skip = QPushButton("Skip Hires Fix")
        self._hires_skip.setStyleSheet("padding: 8px 16px;")
        hires_btn_row.addWidget(self._hires_skip)
        hl.addLayout(hires_btn_row)

        # Result preview
        self._hires_result = QLabel("Hires result will appear here")
        self._hires_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hires_result.setMinimumHeight(200)
        self._hires_result.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        hl.addWidget(self._hires_result)

        layout.addWidget(hires_group)

        layout.addStretch()
        scroll.setWidget(scroll_inner)
        outer.addWidget(scroll)
        self._stack.addWidget(panel)

        # Scan upscale models
        self._scan_upscale_models()

    def _scan_upscale_models(self) -> None:
        """Populate hires fix upscale model dropdown."""
        try:
            mp = self.state.global_config.model_paths
            models_dir = mp.get("upscaler_dir", "")
            if models_dir:
                from supremediffusion.postprocessing.ai_upscale import scan_models
                models = scan_models(models_dir)
                if models:
                    self._hires_model.clear()
                    self._hires_model.addItems(models)
                    return
            # Fallback defaults
            self._hires_model.clear()
            self._hires_model.addItem("RealESRGAN_x4plus.pth")
            self._hires_model.addItem("RealESRGAN_x2plus.pth")
        except Exception as exc:
            logger.warning("Failed to scan upscale models: %s", exc)
            self._hires_model.addItem("RealESRGAN_x4plus.pth")

    # ── Step navigation ──────────────────────────────────────────────────

    def _update_step_ui(self) -> None:
        step = self._current_step
        name, desc = _STEPS[step]
        self._step_label.setText(f"{name}  ({step + 1} of {len(_STEPS)})")
        self._step_desc.setText(desc)
        self._stack.setCurrentIndex(step)
        self._btn_prev.setEnabled(step > 0)
        self._btn_next.setEnabled(step < len(_STEPS) - 1)

    @Slot()
    def _prev_step(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._update_step_ui()

    @Slot()
    def _next_step(self) -> None:
        if self._current_step < len(_STEPS) - 1:
            self._current_step += 1
            self._update_step_ui()

    # ── Graph persistence ────────────────────────────────────────────────

    def _graph_path(self) -> Path:
        return self.project_path / "scene_graph.json"

    def _load_graph(self) -> None:
        path = self._graph_path()
        self._graph = SceneGraph.load(path)
        self._scene_tree.set_graph(self._graph)

    def _save_graph(self) -> None:
        self._graph.save(self._graph_path())
        self._scene_tree.refresh()

    # ── Worker helpers ──────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        """Disable/enable generation buttons and show/hide progress UI."""
        self._progress_bar.setVisible(busy)
        self._abort_btn.setVisible(busy)
        if not busy:
            self._progress_bar.setValue(0)
            self.state.release_generation("Pipeline Wizard")
        # Disable all generate buttons while busy
        for btn in (self._realism_run, self._realism_auto, self._orbit_gen,
                    self._extract_btn, self._scene_gen, self._import_btn,
                    self._comp_generate_btn, self._planner_generate_btn,
                    self._planner_lock_btn, self._run_chain_btn,
                    self._face_restore_run, self._face_swap_run,
                    self._hires_run):
            btn.setEnabled(not busy)

    def _get_video_pipeline(self):
        """Load video pipeline if needed and return it, or None on failure."""
        pipeline = self.state.pipeline
        if pipeline is None:
            self._show_status("Loading video pipeline... (this may take a moment)")
            try:
                self.state.load_pipelines()
                pipeline = self.state.pipeline
            except Exception as exc:
                self._show_status(f"Failed to load video pipeline: {exc}")
                return None
        if pipeline is None:
            self._show_status("Video pipeline not available. Check Settings tab.")
        return pipeline

    def _get_img_pipeline(self):
        """Load image (SDXL) pipeline if needed and return it, or None."""
        pipeline = self._active_strategy.get_pipeline(self.state)
        if pipeline is None:
            self._show_status("Loading image pipeline... (this may take a moment)")
            try:
                load_fn = getattr(self.state, self._active_strategy.load_method)
                load_fn()
                pipeline = self._active_strategy.get_pipeline(self.state)
            except Exception as exc:
                self._show_status(f"Failed to load image pipeline: {exc}")
                return None
        if pipeline is None:
            self._show_status("Image pipeline not available. Check Settings tab.")
        return pipeline

    def _apply_quality_defaults(self, cfg) -> None:
        """Apply generation settings from the wizard's params widget.

        Uses whatever the user has configured in the Model Setup
        section (checkpoint, sampler, steps, CFG, etc.) rather than hardcoded values.
        Falls back to safe defaults if the widget hasn't been populated.
        """
        # Sync visible wizard controls into hidden ImageParamsWidget
        self._sync_wizard_to_params()
        # Let the ImageParamsWidget write all its settings to the config
        self._wizard_params.collect_to_config(cfg, strategy=self._active_strategy)

        # Always apply a good negative prompt
        self._active_strategy.set_config(cfg, "negative_prompt",
                                 "blurry, low quality, low resolution, pixelated, "
                                 "cartoon, anime, drawing, sketch, painting, illustration, "
                                 "3d render, cgi, plastic, doll, mannequin, "
                                 "text, watermark, signature, logo, "
                                 "ugly, deformed, disfigured, mutated, bad anatomy, "
                                 "bad hands, extra fingers, missing fingers, "
                                 "extra limbs, fused fingers, too many fingers")

    def _connect_worker(self, worker) -> None:
        """Wire up a worker's signals and start it."""
        if not self.state.acquire_generation("Pipeline Wizard"):
            owner = self.state.generation_owner
            self._show_status(
                f"GPU is busy — {owner} is generating. Wait for it to finish."
            )
            worker.deleteLater()
            return
        self._worker = worker
        worker.progress.connect(self._on_worker_progress)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(worker.deleteLater)
        self._set_busy(True)
        worker.start()

    @Slot(float, str)
    def _on_worker_progress(self, frac: float, desc: str) -> None:
        self._progress_bar.setValue(int(frac * 100))
        self._show_status(desc)

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        self._set_busy(False)
        self._worker = None
        self.state.release_generation("Pipeline Wizard")
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()
            self._show_status("Aborting...")

    # ── Mode toggle ─────────────────────────────────────────────────────

    @Slot(int)
    def _on_mode_changed(self, idx: int) -> None:
        self._mode = self._mode_combo.currentData()
        is_guided = self._mode == "guided"
        # Show/hide step navigation in guided mode
        self._btn_prev.setVisible(is_guided)
        self._btn_next.setVisible(is_guided)
        self._step_desc.setVisible(is_guided)
        if is_guided:
            self._update_step_ui()
        else:
            self._step_label.setText(
                "Freestyle — right-click nodes to generate"
                if self._mode == "freestyle"
                else "Power — batch queue (coming soon)"
            )

    # ── Node selection ───────────────────────────────────────────────────

    def _get_selected_image_node(self):
        """Return the currently selected ShotNode, or None if nothing usable is selected."""
        if not self._selected_node_id:
            return None
        node = self._graph.get(self._selected_node_id)
        if node and node.file_path and Path(node.file_path).is_file():
            return node
        return None

    @Slot(str)
    def _on_node_selected(self, node_id: str) -> None:
        self._selected_node_id = node_id
        node = self._graph.get(node_id)
        if not node:
            return

        # Update source previews on relevant panels
        if node.file_path and node.node_type == NodeType.IMAGE:
            pixmap = __import__("PySide6.QtGui", fromlist=["QPixmap"]).QPixmap(node.file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    400, 250,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._realism_source.setPixmap(scaled)
                self._inpaint_source.setPixmap(scaled)
                self._face_restore_source.setPixmap(scaled)
                self._face_swap_target.setPixmap(scaled)
                self._orbit_source.setPixmap(scaled)
                self._scene_source.setPixmap(scaled)

                # Also update planner start frame
                planner_scaled = pixmap.scaled(
                    280, 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._planner_start_img.setPixmap(planner_scaled)
                self._planner_start_img.setStyleSheet(
                    "background: #1a1a1a; border: 2px solid #2e7d32; border-radius: 4px;"
                )

    def _on_planner_node_selected(self, node_id: str) -> None:
        """Update planner when a node is selected via plan_shot action."""
        node = self._graph.get(node_id)
        if node and node.file_path and node.node_type == NodeType.IMAGE:
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(node.file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    280, 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._planner_start_img.setPixmap(scaled)

    # ── Node actions (from context menu) ─────────────────────────────────

    @Slot(str, str)
    def _on_node_action(self, node_id: str, action: str) -> None:
        node = self._graph.get(node_id)
        if not node:
            return

        if action == "orbit_clip":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 6  # Orbit Clip
                self._update_step_ui()
            self._scene_tree.select_node(node_id)
            if self._mode == "freestyle":
                self._on_orbit_generate()

        elif action == "zoom_clip":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 8  # Scene Shot
                self._update_step_ui()
            self._scene_tree.select_node(node_id)

        elif action == "plan_shot":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 1  # Plan Your Shot
                self._update_step_ui()
            self._scene_tree.select_node(node_id)
            self._on_planner_node_selected(node_id)

        elif action == "realism_pass":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 2  # Realism Pass
                self._update_step_ui()
            self._scene_tree.select_node(node_id)
            if self._mode == "freestyle":
                self._on_realism_run()

        elif action == "inpaint":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 3  # Inpaint
                self._update_step_ui()
            self._scene_tree.select_node(node_id)
            if self._mode == "freestyle":
                self._on_inpaint_open()

        elif action == "extract_stills":
            self._selected_node_id = node_id
            if self._mode == "guided":
                self._current_step = 7  # Extract Stills
                self._update_step_ui()
            self._scene_tree.select_node(node_id)
            if self._mode == "freestyle":
                self._on_extract_stills()

        elif action == "upscale":
            self._show_status("Upscale pipeline coming soon")

        elif action == "extend":
            self._show_status("Video extend integration coming soon")

        elif action == "use_as_source":
            # Set this image as the source in the main Video tab
            if node.file_path:
                self.state.set_image_path(node.file_path)
                self._show_status(f"Set as source: {Path(node.file_path).name}")

        elif action == "rename":
            from PySide6.QtWidgets import QInputDialog
            new_name, ok = QInputDialog.getText(
                self, "Rename Node", "Name:", text=node.name,
            )
            if ok and new_name:
                node.name = new_name
                self._save_graph()

        elif action == "add_note":
            from PySide6.QtWidgets import QInputDialog
            note, ok = QInputDialog.getMultiLineText(
                self, "Add Note", "Note:", node.notes,
            )
            if ok:
                node.notes = note
                self._save_graph()

        elif action == "remove":
            self._graph.remove_node(node_id, recursive=False)
            self._save_graph()

        elif action == "remove_tree":
            self._graph.remove_node(node_id, recursive=True)
            self._save_graph()

        elif action == "open_stillgrabber":
            self._show_status("Opening in StillGrabber... (integrate with tab switch)")

    # ── Step handlers ────────────────────────────────────────────────────

    @Slot()
    def _on_compose_prompt(self) -> None:
        """Build a prompt from the composer dropdowns."""
        parts = []

        # Scene
        scene_idx = self._comp_scene.currentIndex()
        _, scene_desc = _SCENES[scene_idx]
        parts.append(scene_desc)

        # Characters
        char_idx = self._comp_chars.currentIndex()
        _, char_desc = _CHARACTERS[char_idx]
        if char_desc:
            parts.append(char_desc)

            # Body, hair, clothing only if characters present
            body_idx = self._comp_body.currentIndex()
            _, body_desc = _BODY_TYPES[body_idx]
            parts.append(body_desc)

            hair_idx = self._comp_hair.currentIndex()
            _, hair_desc = _HAIR_OPTIONS[hair_idx]
            if hair_desc:
                parts.append(hair_desc)

            cloth_idx = self._comp_clothing.currentIndex()
            _, cloth_desc = _CLOTHING[cloth_idx]
            parts.append(cloth_desc)

        # Lighting
        light_idx = self._comp_lighting.currentIndex()
        _, light_desc = _LIGHTING[light_idx]
        parts.append(light_desc)

        # Camera
        cam_idx = self._comp_camera.currentIndex()
        _, cam_desc = _CAMERA_ANGLES[cam_idx]
        parts.append(cam_desc)

        # Style
        style_idx = self._comp_style.currentIndex()
        _, style_desc = _STYLES[style_idx]
        parts.append(style_desc)

        prompt = ", ".join(p for p in parts if p)
        self._comp_prompt.setPlainText(prompt)
        self._show_status("Prompt composed — edit if you like, then hit Generate Base Image.")

    # ── Qwen Prompt Enhancement ─────────────────────────────────────────

    def _on_enhance_prompt(self, target: str) -> None:
        """Launch Qwen enhancement for one of the prompt fields.

        *target* is ``"comp"`` (Scene Composer), ``"planner"`` (Shot Planner
        end-frame), or ``"scene"`` (Scene Shot).
        """
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        from sdqt.deps import check_and_install
        if not check_and_install("transformers", self):
            return

        # Get the right prompt widget
        widget = self._enhance_widget_for(target)
        if widget is None:
            return
        current = widget.toPlainText().strip()
        if not current:
            self._show_status("Write or compose a prompt first, then Enhance.")
            return

        # Strip LoRA tags before sending to Qwen
        clean_text, lora_tags = extract_lora_tags(current)

        # Determine style — manual override takes priority
        manual = self._enhance_style_combo.currentData()
        if manual and manual != "auto":
            style = manual
        else:
            style = self._detect_enhance_style()
        logger.info("Enhance style: %s (manual=%s, checkpoint=%s)",
                     style, manual, self._wizard_params.checkpoint.currentText())

        # Save state for callback
        self._enhance_target = target
        self._enhance_lora_tags = lora_tags

        # Disable all enhance buttons
        self._set_enhance_buttons_enabled(False)
        style_labels = {
            "sd15": "SD 1.5", "sdxl": "SDXL", "flux": "FLUX",
            "pony_real": "Pony (Realistic)", "pony_anime": "Pony (Anime)",
            "illustrious": "Illustrious", "noobai": "NoobAI",
            "sd3": "SD3", "wan": "Wan Video",
        }
        style_label = style_labels.get(style, style)
        self._show_status(f"Enhancing prompt for {style_label} style...")

        worker = PromptEnhanceWorker(clean_text, style, "pos", self.state, parent=self)
        worker.finished_ok.connect(self._on_enhance_done)
        worker.error.connect(self._on_enhance_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    # ── Checkpoint family change → strategy switch + quality warning ───

    def _on_checkpoint_family_changed(self, info) -> None:
        """Called when the user picks a different checkpoint in the grouped combo."""
        from supremediffusion.models.sd_models import CheckpointInfo
        if not isinstance(info, CheckpointInfo):
            return
        # Update active strategy
        new_strategy = STRATEGY_MAP.get(info.model_type, SDXL_STRATEGY)
        self._active_strategy = new_strategy

        # Sync to hidden ImageParamsWidget
        idx = self._wizard_params.checkpoint.findText(info.name)
        if idx >= 0:
            self._wizard_params.checkpoint.setCurrentIndex(idx)

        # Show/hide controls based on strategy
        has_vae = new_strategy.has_vae_selector
        self._lbl_vae_row.setVisible(has_vae)
        self._wizard_vae.setVisible(has_vae)

        has_sampler = new_strategy.has_sampler
        self._lbl_sampler_row.setVisible(has_sampler)
        self._wizard_sampler.setVisible(has_sampler)
        self._wizard_scheduler.setVisible(has_sampler)

        has_clip = getattr(new_strategy, "has_clip_skip", True)
        self._lbl_clip_skip.setVisible(has_clip)
        self._wizard_clip_skip.setVisible(has_clip)

        # Update quality warning
        self._update_quality_warning()

        self._show_status(f"Checkpoint: {info.name} ({info.model_type.upper()})")

    def _update_quality_warning(self) -> None:
        """Check LoRA/checkpoint compatibility and show/hide the warning banner."""
        ckpt_family = self._active_strategy.family if self._active_strategy else "sdxl"

        # Collect base-model groups of all active LoRAs from the picker
        lora_groups = []
        if hasattr(self, "_wizard_lora_picker"):
            from sdqt.widgets.lora_picker import _detect_lora_base_model
            activated = self._wizard_lora_picker.get_activated()
            lora_dir = self.state.global_config.model_paths.get("sd_lora_dir", "")
            if lora_dir and activated:
                for fname in activated:
                    lora_path = Path(lora_dir) / fname
                    if not lora_path.exists():
                        # Try recursive search
                        matches = list(Path(lora_dir).rglob(fname))
                        lora_path = matches[0] if matches else lora_path
                    group = _detect_lora_base_model(lora_path)
                    lora_groups.append(group)

        if hasattr(self, "_quality_warning"):
            self._quality_warning.update_warning(ckpt_family, lora_groups)

    def _sync_wizard_to_params(self) -> None:
        """Sync visible wizard controls into the hidden ImageParamsWidget before generation."""
        p = self._wizard_params
        p.steps.setValue(self._wizard_steps.value())
        p.cfg_scale.setValue(self._wizard_cfg.value())
        p.seed.setValue(self._wizard_seed.value())
        p.clip_skip.setValue(self._wizard_clip_skip.value())
        p.width.setValue(self._wizard_width.value())
        p.height.setValue(self._wizard_height.value())
        p.batch_count.setValue(self._wizard_batch_count.value())
        p.batch_size.setValue(1)

        # Sync VAE
        if self._wizard_vae.currentText() != "Automatic":
            idx = p.vae.findText(self._wizard_vae.currentText())
            if idx >= 0:
                p.vae.setCurrentIndex(idx)

        # Sync sampler/scheduler
        idx = p.sampler.findText(self._wizard_sampler.currentText())
        if idx >= 0:
            p.sampler.setCurrentIndex(idx)
        idx = p.scheduler.findText(self._wizard_scheduler.currentText())
        if idx >= 0:
            p.scheduler.setCurrentIndex(idx)

        # Sync LoRA weights
        p.lora_weights.setText(self._wizard_lora_weights.text())

    # ── LoRA toggle → prompt tag insertion ─────────────────────────────

    @Slot(str, bool)
    def _on_wizard_lora_toggled(self, tag: str, checked: bool) -> None:
        """Insert or remove a LoRA prompt tag from the currently active prompt."""
        # Determine which prompt widget is active based on current step
        step = self._current_step
        if step == 0:
            widget = self._comp_prompt
        elif step == 1:
            widget = self._planner_end_prompt
        elif step == 8:
            widget = self._scene_prompt
        else:
            # Steps 2-5 don't have user-editable prompts in the wizard
            return

        current = widget.toPlainText()
        if checked:
            if tag not in current:
                sep = ", " if current.strip() else ""
                widget.setPlainText(f"{current}{sep}{tag}")
                self._show_status(f"LoRA added to prompt: {tag}")
        else:
            # Remove the tag (and optional trailing trigger words)
            stem_match = re.search(r"<lora:([^:>]+):", tag)
            if stem_match:
                stem = stem_match.group(1)
                pattern = rf",?\s*<lora:{re.escape(stem)}:[^>]+>(?:\s+[^<,\n]+)?"
                cleaned = re.sub(pattern, "", current).strip(", ")
                widget.setPlainText(cleaned)
                self._show_status(f"LoRA removed from prompt.")

        # Update quality warning when LoRA selection changes
        self._update_quality_warning()

    def _enhance_widget_for(self, target: str):
        """Return the QPlainTextEdit for the given enhance target."""
        if target == "comp":
            return self._comp_prompt
        elif target == "planner":
            return self._planner_end_prompt
        elif target == "scene":
            return self._scene_prompt
        return None

    def _detect_enhance_style(self) -> str:
        """Pick the Qwen enhance style based on the selected checkpoint.

        Maps checkpoint model_type + name heuristics to the style keys
        understood by PromptEnhanceWorker: sd15, sdxl, flux, pony_real,
        pony_anime, illustrious, noobai, sd3.
        """
        selected = self._wizard_params.checkpoint.currentText()
        if not selected:
            return "sdxl"

        # Find the CheckpointInfo for the selected name
        info = None
        for ci in self._checkpoint_infos:
            if ci.name == selected:
                info = ci
                break

        low = selected.lower()

        # Pony models — check name heuristic first (most specific)
        if "pony" in low:
            # Distinguish anime vs realistic pony
            if any(kw in low for kw in ("anime", "illust", "cartoon", "toon")):
                return "pony_anime"
            return "pony_real"

        # Illustrious / NoobAI
        if "noobai" in low or "noob" in low:
            return "noobai"
        if "illustrious" in low:
            return "illustrious"

        # SD3 / SD3.5
        if "sd3" in low or "sd_3" in low or "stable-diffusion-3" in low:
            return "sd3"

        # FLUX
        if info and info.model_type == "flux":
            return "flux"
        if "flux" in low:
            return "flux"

        # Fall back to model_type from header inspection
        if info:
            return info.model_type  # "sd15" or "sdxl"

        return "sdxl"

    @Slot(str)
    def _on_enhance_done(self, result: str) -> None:
        target = self._enhance_target
        result = reinsert_lora_tags(result, self._enhance_lora_tags)

        widget = self._enhance_widget_for(target)
        if widget is not None:
            widget.setPlainText(result)

        self._set_enhance_buttons_enabled(True)
        self._show_status("Prompt enhanced by Qwen AI. Review and edit if needed.")

    @Slot(str)
    def _on_enhance_error(self, msg: str) -> None:
        self._set_enhance_buttons_enabled(True)
        self._show_status(f"Enhance error: {msg}")

    def _set_enhance_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self._comp_enhance_btn, self._planner_enhance_btn, self._scene_enhance_btn):
            btn.setEnabled(enabled)

    @Slot()
    def _on_generate_base(self) -> None:
        """Generate a base establishing shot via txt2img."""
        prompt = self._comp_prompt.toPlainText().strip()
        if not prompt:
            self._on_compose_prompt()
            prompt = self._comp_prompt.toPlainText().strip()
        if not prompt:
            self._show_status("Compose a prompt first.")
            return

        pipeline = self._get_img_pipeline()
        if not pipeline:
            return

        cfg = ProjectConfig.load(self.project_path)
        self._active_strategy.set_config(cfg, "prompt", prompt)
        self._apply_quality_defaults(cfg)
        cfg.save(self.project_path)

        self._show_status("Generating base image...")

        WorkerCls = self._active_strategy.resolve_worker("txt2img")
        worker = WorkerCls(pipeline, self.project_name, cfg, parent=self)
        worker.finished_ok.connect(self._on_generate_base_done)
        self._connect_worker(worker)

    def _on_generate_base_done(self, paths: list[str]) -> None:
        self._set_busy(False)
        self._worker = None
        if not paths:
            self._show_status("Generation produced no output.")
            return

        self._generated_base_path = paths[0]
        # Show preview
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(self._generated_base_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                500, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._comp_preview.setPixmap(scaled)
        self._comp_add_btn.setEnabled(True)
        self._show_status(
            f"Base image generated! Click 'Add Generated Image to Scene Graph' to continue."
        )

    @Slot()
    def _on_add_generated_base(self) -> None:
        """Add the generated base image as a root node."""
        if not self._generated_base_path:
            self._show_status("Generate an image first.")
            return

        persisted = self.persist_file(self._generated_base_path, subdir="scenes/base")
        prompt = self._comp_prompt.toPlainText().strip()

        node = SceneGraph.create_image_node(
            file_path=persisted or self._generated_base_path,
            name="Establishing Shot",
            edge_type=EdgeType.IMPORT,
            params=GenerationParams(prompt=prompt, model_type="sdxl"),
        )
        self._graph.add_root(node)
        self._save_graph()
        self._scene_tree.select_node(node.id)
        self._generated_base_path = None
        self._comp_add_btn.setEnabled(False)
        self._show_status(f"Added root: {node.name} — hit Next to Plan Your Shot.")

    @Slot()
    def _on_import_base(self) -> None:
        """Import the dropped image as a root node."""
        img_path = self._import_drop.image_path
        if not img_path:
            self._show_status("Drop an image first.")
            return

        # Persist into project
        persisted = self.persist_file(img_path, subdir="scenes/base")

        node = SceneGraph.create_image_node(
            file_path=persisted or img_path,
            name="Establishing Shot",
            edge_type=EdgeType.IMPORT,
        )
        self._graph.add_root(node)
        self._save_graph()
        self._scene_tree.select_node(node.id)
        self._show_status(f"Added root: {node.name}")

    # ── 3D Viewport Render ──────────────────────────────────────────────

    @Slot()
    def _on_render_from_viewport(self) -> None:
        """Request a render from the Daz2Supreme tab's GL viewport."""
        res_data = self._render_res.currentData()
        if not res_data:
            res_data = (1024, 1024)
        width, height = res_data

        # Try to find the Daz2Supreme tab's viewport
        viewport = self._find_gl_viewport()
        if not viewport:
            self._show_status(
                "No 3D viewport found. Open the Daz2Supreme tab and load a scene first."
            )
            return

        self._show_status(f"Rendering {width}x{height} from 3D viewport...")
        img = viewport.render_to_image(width, height)
        if img is None:
            self._show_status(
                "Render failed — OpenGL not available. Try: pip install PyOpenGL"
            )
            return

        # Save to project
        import time
        out_dir = self.project_path / "scenes" / "base"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"viewport_render_{int(time.time())}.png"
        img.save(str(out_path))

        # Show preview (reuse the generated preview label)
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(str(out_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                500, 350,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._comp_preview.setPixmap(scaled)

        self._generated_base_path = str(out_path)
        self._comp_add_btn.setEnabled(True)
        self._show_status(
            f"Viewport render saved! Click 'Add Generated Image to Scene Graph' to use it."
        )

    def _find_gl_viewport(self):
        """Walk up to MainWindow and find the Daz2Supreme tab's viewport."""
        # Walk parent chain to find main window
        widget = self.parent()
        while widget is not None:
            if hasattr(widget, "_daz_tab"):
                daz_tab = widget._daz_tab
                if hasattr(daz_tab, "_viewport"):
                    return daz_tab._viewport
                break
            widget = widget.parent() if hasattr(widget, "parent") else None
        return None

    # ── Shot Planner handlers ───────────────────────────────────────────

    @Slot(int)
    def _on_planner_changed(self, _idx: int = 0) -> None:
        """Re-calculate drift risk whenever any planner dropdown changes.

        Drift risk measures the WORST SINGLE CLIP, not the total.
        When auto-split is active, each clip only does one thing,
        so the per-clip cost is what matters.
        """
        cam_idx = self._planner_camera.currentIndex()
        subj_idx = self._planner_subject.currentIndex()
        _, _, cam_cost = _PLANNER_CAMERA_MOVES[cam_idx]
        _, _, subj_cost = _PLANNER_SUBJECT_CHANGES[subj_idx]

        # Show/hide transformation controls
        is_transform = (subj_idx == len(_PLANNER_SUBJECT_CHANGES) - 1)
        self._transform_group.setVisible(is_transform)

        # Calculate per-clip cost (worst single clip in the chain)
        # When auto-split is active, camera and subject are separate clips
        will_split = (cam_cost > 0 and (subj_cost > 0 or is_transform))

        if will_split:
            # Each clip does ONE thing — per-clip cost is the max of any single clip
            clip_costs = []
            if cam_cost > 0:
                clip_costs.append(cam_cost)  # camera-only clip

            if is_transform:
                # Each stage is its own clip — cost 1 per stage (one incremental change)
                clip_costs.append(1)  # each stage clip = 1 incremental step

                ending_idx = self._planner_ending.currentIndex()
                if ending_idx > 0:
                    clip_costs.append(2)  # pop/aftermath is dramatic but static camera = manageable
            elif subj_cost > 0:
                clip_costs.append(subj_cost)

            per_clip_cost = max(clip_costs) if clip_costs else 0
        else:
            # Single clip — everything happens at once
            per_clip_cost = cam_cost + subj_cost
            if is_transform:
                ttype_key = self._get_transform_key()
                stage_idx = self._planner_stage.currentIndex()
                stages = self._get_transformation_stages(ttype_key)
                if stages and 0 <= stage_idx < len(stages):
                    _, _, clips = stages[stage_idx]
                    per_clip_cost += min(clips, 3)
                ending_idx = self._planner_ending.currentIndex()
                if ending_idx > 0:
                    per_clip_cost += 1

        self._update_drift_meter(per_clip_cost, cam_cost, subj_cost,
                                 per_clip_cost - cam_cost if will_split else 0)
        self._update_clip_chain()

    def _get_transform_key(self) -> str:
        """Return the current transformation type key."""
        idx = self._planner_transform_type.currentIndex()
        types = self._get_transformation_types()
        if 0 <= idx < len(types):
            return types[idx][1]
        return "custom"

    @Slot(int)
    def _on_transform_type_changed(self, _idx: int) -> None:
        """Populate stage and ending dropdowns for the selected transform type."""
        key = self._get_transform_key()

        # Stages
        self._planner_stage.blockSignals(True)
        self._planner_stage.clear()
        stages = self._get_transformation_stages(key)
        for label, _, _ in stages:
            self._planner_stage.addItem(label)
        if not stages:
            self._planner_stage.addItem("(define stages in custom prompt)")
        self._planner_stage.blockSignals(False)

        # Endings
        self._planner_ending.blockSignals(True)
        self._planner_ending.clear()
        self._planner_ending.addItem("None — stop at this stage")
        endings = self._get_ending_variants(key)
        for label, _, phase in endings:
            prefix = {"pop": "💥 ", "climax": "⚡ ", "aftermath": "🔚 "}.get(phase, "")
            self._planner_ending.addItem(f"{prefix}{label}")
        self._planner_ending.blockSignals(False)

        # Custom prompt visibility
        self._planner_custom_prompt.setVisible(key == "custom")

        self._on_planner_changed()

    def _update_drift_meter(self, per_clip: int, cam: int, subj: int, transform: int) -> None:
        """Update the drift risk bar and label.

        per_clip = the worst single clip cost (not total across all clips).
        """
        self._drift_bar.setValue(min(per_clip, 6))

        if per_clip <= _DRIFT_GREEN:
            color = "#4caf50"
            level = "GREEN — Safe"
            tip = "Each clip has only one change. This will hold steady."
        elif per_clip <= _DRIFT_YELLOW:
            color = "#ff9800"
            level = "YELLOW — Moderate Risk"
            tip = "One of the clips has a big change. It might drift."
        else:
            color = "#f44336"
            level = "RED — High Drift Risk"
            tip = "A single clip is trying to do too much. Consider simpler changes."

        self._drift_bar.setStyleSheet(
            f"QProgressBar {{ background: #333; border-radius: 7px; }}"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 7px; }}"
        )
        self._drift_label.setText(f"{level} (per-clip cost: {per_clip})\n{tip}")

        # Auto-split suggestion
        if cam > 0 and (subj > 0 or transform > 0):
            self._drift_split_suggestion.setText(
                "Auto-split active: camera move and subject change are in separate clips. "
                "Each clip does one thing — drift risk is measured per clip, not total."
            )
            self._drift_split_suggestion.setVisible(True)
        else:
            self._drift_split_suggestion.setVisible(False)

    def _update_clip_chain(self) -> None:
        """Build a text summary of the planned clip chain."""
        cam_idx = self._planner_camera.currentIndex()
        subj_idx = self._planner_subject.currentIndex()
        cam_label, cam_prompt, cam_cost = _PLANNER_CAMERA_MOVES[cam_idx]
        subj_label, subj_prompt, subj_cost = _PLANNER_SUBJECT_CHANGES[subj_idx]

        is_transform = (subj_idx == len(_PLANNER_SUBJECT_CHANGES) - 1)
        clips = []

        # Camera-only clip (if camera move selected and subject also changes)
        needs_split = (cam_cost > 0 and (subj_cost > 0 or is_transform))
        total_cost = cam_cost + subj_cost
        if is_transform:
            tkey = self._get_transform_key()
            stage_idx = self._planner_stage.currentIndex()
            stages = self._get_transformation_stages(tkey)
            if stages and 0 <= stage_idx < len(stages):
                total_cost += min(stages[stage_idx][2], 3)

        if needs_split or total_cost > _DRIFT_GREEN:
            # Split: camera clip first
            if cam_cost > 0:
                clips.append(f"Clip A — {cam_label} (static subject, 2 sec)")

            if is_transform:
                tkey = self._get_transform_key()
                stage_idx = self._planner_stage.currentIndex()
                stages = self._get_transformation_stages(tkey)
                # Add one clip per stage up to selected
                for i in range(stage_idx + 1):
                    if i < len(stages):
                        slabel, _, sclips = stages[i]
                        clip_letter = chr(ord('B') + i) if cam_cost > 0 else chr(ord('A') + i)
                        clips.append(
                            f"Clip {clip_letter} — {slabel} (static camera, 2 sec)"
                        )

                # Ending
                ending_idx = self._planner_ending.currentIndex()
                if ending_idx > 0:
                    endings = self._get_ending_variants(tkey)
                    if 0 < ending_idx <= len(endings):
                        elabel, _, phase = endings[ending_idx - 1]
                        clip_letter = chr(ord('B') + stage_idx + 1) if cam_cost > 0 else chr(ord('A') + stage_idx + 1)
                        phase_note = "STATIC CAMERA" if phase == "pop" else "static camera"
                        clips.append(
                            f"Clip {clip_letter} — {elabel} ({phase_note}, 2 sec)"
                        )
            elif subj_cost > 0:
                clip_letter = "B" if cam_cost > 0 else "A"
                clips.append(
                    f"Clip {clip_letter} — {subj_label} (static camera, 2 sec)"
                )
        else:
            # Single clip
            parts = []
            if cam_label != "Static (No Camera Move)":
                parts.append(cam_label)
            if subj_label != "No Change (Static Subject)":
                parts.append(subj_label)
            desc = " + ".join(parts) if parts else "Static hold"
            clips.append(f"Clip A — {desc} (2 sec)")

        # Format
        total = len(clips)
        lines = [f"  {c}" for c in clips]
        chain_text = (
            f"Clip Chain: {total} clip{'s' if total != 1 else ''}, "
            f"~{total * 2} seconds total\n" + "\n".join(lines)
        )
        self._clip_chain_label.setText(chain_text)

    @Slot()
    def _on_planner_compose(self) -> None:
        """Build the end-frame prompt from planner selections."""
        # Start with the original base prompt as context
        start_prompt = ""
        if self._selected_node_id:
            node = self._graph.get(self._selected_node_id)
            if node and node.params and node.params.prompt:
                start_prompt = node.params.prompt

        cam_idx = self._planner_camera.currentIndex()
        subj_idx = self._planner_subject.currentIndex()
        _, cam_fragment, _ = _PLANNER_CAMERA_MOVES[cam_idx]
        _, subj_fragment, _ = _PLANNER_SUBJECT_CHANGES[subj_idx]

        parts = []

        # Keep scene/character description from start prompt
        if start_prompt:
            # Strip style suffixes to keep just the scene description
            for suffix in ("photorealistic", "8k", "cinematic", "film grain",
                           "hyperrealistic", "professional photography"):
                start_prompt = start_prompt.replace(suffix, "")
            parts.append(start_prompt.strip().rstrip(",").strip())

        # Transformation prompt
        is_transform = (subj_idx == len(_PLANNER_SUBJECT_CHANGES) - 1)
        if is_transform:
            tkey = self._get_transform_key()
            if tkey == "custom":
                custom = self._planner_custom_prompt.toPlainText().strip()
                if custom:
                    parts.append(custom)
            else:
                stage_idx = self._planner_stage.currentIndex()
                stages = self._get_transformation_stages(tkey)
                if stages and 0 <= stage_idx < len(stages):
                    _, stage_fragment, _ = stages[stage_idx]
                    parts.append(stage_fragment)

                # Ending
                ending_idx = self._planner_ending.currentIndex()
                if ending_idx > 0:
                    endings = self._get_ending_variants(tkey)
                    if 0 < ending_idx <= len(endings):
                        _, ending_fragment, _ = endings[ending_idx - 1]
                        parts.append(ending_fragment)
        elif subj_fragment:
            parts.append(subj_fragment)

        # Style
        parts.append("photorealistic, highly detailed, 8k")

        prompt = ", ".join(p for p in parts if p)
        self._planner_end_prompt.setPlainText(prompt)
        self._show_status("End-frame prompt composed — edit if needed, then Generate.")

    @Slot()
    def _on_planner_generate_end(self) -> None:
        """Generate end-frame preview via img2img on the start frame."""
        if not self._selected_node_id:
            self._show_status("Select a base image from the tree first (Step 1).")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.IMAGE:
            self._show_status("Select an IMAGE node as your start frame.")
            return

        prompt = self._planner_end_prompt.toPlainText().strip()
        if not prompt:
            self._on_planner_compose()
            prompt = self._planner_end_prompt.toPlainText().strip()
        if not prompt:
            self._show_status("Compose an end-frame prompt first.")
            return

        pipeline = self._get_img_pipeline()
        if not pipeline:
            return

        cfg = ProjectConfig.load(self.project_path)
        self._active_strategy.set_config(cfg, "prompt", prompt)
        self._apply_quality_defaults(cfg)

        # Scale denoise strength with transformation severity:
        #   No change / subtle move  → 0.45  (keep most of the start frame)
        #   Stage 1-2 transformation → 0.65  (moderate change)
        #   Stage 3-4 transformation → 0.80  (major change)
        #   Ending (pop/aftermath)   → 0.90  (near-complete reimagining)
        denoise = 0.45
        subj_idx = self._planner_subject.currentIndex()
        is_transform = (subj_idx == len(_PLANNER_SUBJECT_CHANGES) - 1)
        if is_transform:
            stage_idx = self._planner_stage.currentIndex()
            ending_idx = self._planner_ending.currentIndex()
            if ending_idx > 0:
                denoise = 0.90  # pop/aftermath = drastic visual change
            elif stage_idx >= 2:
                denoise = 0.80  # stage 3-4
            elif stage_idx >= 0:
                denoise = 0.65  # stage 1-2
        elif subj_idx > 0:
            denoise = 0.50  # subtle subject change (head turn, etc.)
        self._active_strategy.set_config(cfg, "denoising_strength", denoise)
        logger.info("End-frame denoise: %.2f (subject=%d, transform=%s)", denoise, subj_idx, is_transform)

        cfg.img2img_source_path = node.file_path
        cfg.img_batch_count = 1
        cfg.img_batch_size = 1
        cfg.save(self.project_path)

        self._show_status("Generating end-frame preview...")

        WorkerCls = self._active_strategy.resolve_worker("img2img")
        worker = WorkerCls(pipeline, self.project_name, node.file_path, cfg, parent=self)
        worker.finished_ok.connect(self._on_planner_end_done)
        self._connect_worker(worker)

    def _on_planner_end_done(self, paths: list[str]) -> None:
        self._set_busy(False)
        self._worker = None
        if not paths:
            self._show_status("End-frame generation produced no output.")
            return

        self._planner_end_path = paths[0]
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(self._planner_end_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                280, 180,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._planner_end_img.setPixmap(scaled)
            self._planner_end_img.setStyleSheet(
                "background: #1a1a1a; border: 2px solid #1565c0; border-radius: 4px;"
            )
        self._planner_lock_btn.setEnabled(True)
        self._show_status(
            "End-frame preview generated! Adjust and re-generate, or Lock Pair."
        )

    @Slot()
    def _on_planner_lock_pair(self) -> None:
        """Save the start + end frame pair to the scene graph."""
        if not self._selected_node_id or not self._planner_end_path:
            self._show_status("Generate an end-frame first.")
            return

        start_node = self._graph.get(self._selected_node_id)
        if not start_node:
            return

        # Persist end frame
        persisted = self.persist_file(self._planner_end_path, subdir="scenes/end_frames")
        end_prompt = self._planner_end_prompt.toPlainText().strip()

        # Create the end frame as a child of the start frame with a new edge type
        child = SceneGraph.create_image_node(
            file_path=persisted or self._planner_end_path,
            name="END FRAME Target",
            edge_type=EdgeType.IMG2IMG,
            params=GenerationParams(
                prompt=end_prompt,
                denoise_strength=0.55,
                model_type="sdxl",
            ),
            starred=True,
        )
        self._graph.add_child(start_node.id, child)

        # Store clip chain info in the node's notes
        chain_text = self._clip_chain_label.text()
        child.notes = f"Shot Plan:\n{chain_text}"

        self._save_graph()
        self._scene_tree.select_node(child.id)
        self._planner_end_path = None
        self._planner_lock_btn.setEnabled(False)
        self._run_chain_btn.setEnabled(True)
        # Store the locked pair IDs for chain execution
        self._chain_start_node_id = start_node.id
        self._chain_end_node_id = child.id
        self._show_status(
            "Pair locked! Start + End frames saved. "
            "Hit 'Run Clip Chain' for full auto, or Next for manual steps."
        )

    # ── Auto Clip Chain Executor ──────────────────────────────────────

    @Slot()
    def _on_run_clip_chain(self) -> None:
        """Build and execute the full clip chain automatically."""
        start_id = getattr(self, "_chain_start_node_id", None)
        if not start_id:
            self._show_status("Lock a start/end pair first.")
            return
        start_node = self._graph.get(start_id)
        if not start_node or not start_node.file_path:
            self._show_status("Start node missing or has no file.")
            return

        # Build the clip queue from current planner settings
        chain = self._build_chain_queue()
        if not chain:
            self._show_status("No clips to execute — check your planner settings.")
            return

        self._chain_queue = chain
        self._chain_idx = 0
        self._chain_current_source_id = start_id
        self._chain_current_source_path = start_node.file_path
        self._chain_total = len(chain)

        self._chain_progress_label.setText(
            f"Starting clip chain: {self._chain_total} clips to generate..."
        )
        self._show_status(
            f"Clip chain started — {self._chain_total} clips. "
            "This will take a while. You can Abort at any time."
        )

        # Step 1: Run realism pass on start frame first
        self._chain_phase = "realism"
        self._run_chain_realism()

    def _build_chain_queue(self) -> list[dict]:
        """Build ordered list of clip specs from planner state."""
        cam_idx = self._planner_camera.currentIndex()
        subj_idx = self._planner_subject.currentIndex()
        cam_label, cam_prompt, cam_cost = _PLANNER_CAMERA_MOVES[cam_idx]
        subj_label, subj_prompt, subj_cost = _PLANNER_SUBJECT_CHANGES[subj_idx]

        is_transform = (subj_idx == len(_PLANNER_SUBJECT_CHANGES) - 1)
        clips = []

        needs_split = (cam_cost > 0 and (subj_cost > 0 or is_transform))
        total_cost = cam_cost + subj_cost
        if is_transform:
            tkey = self._get_transform_key()
            stage_idx = self._planner_stage.currentIndex()
            stages = self._get_transformation_stages(tkey)
            if stages and 0 <= stage_idx < len(stages):
                total_cost += min(stages[stage_idx][2], 3)

        # Get the base scene prompt from start node
        start_node = self._graph.get(self._chain_start_node_id)
        base_prompt = ""
        if start_node and start_node.params and start_node.params.prompt:
            base_prompt = start_node.params.prompt
            # Strip quality suffixes for reuse
            for suffix in ("photorealistic", "8k", "cinematic", "film grain",
                           "hyperrealistic", "professional photography"):
                base_prompt = base_prompt.replace(suffix, "")
            base_prompt = base_prompt.strip().rstrip(",").strip()

        if needs_split or total_cost > _DRIFT_GREEN:
            # Camera clip first
            if cam_cost > 0:
                clips.append({
                    "name": f"Camera — {cam_label}",
                    "prompt": (
                        f"{base_prompt}, {cam_prompt}, "
                        "all characters are standing still, "
                        "smooth steady motion, photorealistic, 8k"
                    ),
                    "duration": 33,  # 2 seconds
                })

            if is_transform:
                tkey = self._get_transform_key()
                stage_idx = self._planner_stage.currentIndex()
                stages = self._get_transformation_stages(tkey)
                for i in range(stage_idx + 1):
                    if i < len(stages):
                        slabel, sfragment, _ = stages[i]
                        clips.append({
                            "name": slabel,
                            "prompt": (
                                f"{base_prompt}, the camera is static and locked off, "
                                f"{sfragment}, photorealistic, 8k"
                            ),
                            "duration": 33,
                        })

                # Ending variant
                ending_idx = self._planner_ending.currentIndex()
                if ending_idx > 0:
                    endings = self._get_ending_variants(tkey)
                    if 0 < ending_idx <= len(endings):
                        elabel, efragment, phase = endings[ending_idx - 1]
                        clips.append({
                            "name": elabel,
                            "prompt": (
                                f"{base_prompt}, the camera is static and locked off, "
                                f"{efragment}, photorealistic, 8k"
                            ),
                            "duration": 33,
                        })
            elif subj_cost > 0:
                clips.append({
                    "name": subj_label,
                    "prompt": (
                        f"{base_prompt}, the camera is static and locked off, "
                        f"{subj_prompt}, photorealistic, 8k"
                    ),
                    "duration": 33,
                })
        else:
            # Single clip
            prompt_parts = [base_prompt]
            if cam_prompt:
                prompt_parts.append(cam_prompt)
            if subj_prompt:
                prompt_parts.append(subj_prompt)
            prompt_parts.append("photorealistic, 8k")
            clips.append({
                "name": f"{cam_label} + {subj_label}" if subj_prompt else cam_label,
                "prompt": ", ".join(prompt_parts),
                "duration": 33,
            })

        return clips

    def _run_chain_realism(self) -> None:
        """Run a quick realism pass on the start frame before generating clips."""
        pipeline = self._get_img_pipeline()
        if not pipeline:
            self._chain_progress_label.setText("Failed — image pipeline not available.")
            return

        cfg = ProjectConfig.load(self.project_path)
        self._active_strategy.set_config(cfg, "prompt",
                                 "photorealistic, highly detailed, 8k, natural lighting")
        self._apply_quality_defaults(cfg)
        self._active_strategy.set_config(cfg, "denoising_strength", 0.45)
        cfg.img2img_source_path = self._chain_current_source_path
        cfg.img_batch_count = 1
        cfg.img_batch_size = 1
        cfg.save(self.project_path)

        self._chain_progress_label.setText(
            "Phase 1: Running realism pass on start frame..."
        )

        WorkerCls = self._active_strategy.resolve_worker("img2img")
        worker = WorkerCls(pipeline, self.project_name,
                           self._chain_current_source_path, cfg, parent=self)
        worker.finished_ok.connect(self._on_chain_realism_done)
        self._connect_worker(worker)

    def _on_chain_realism_done(self, paths: list[str]) -> None:
        self._worker = None
        if not paths:
            self._set_busy(False)
            self._chain_progress_label.setText("Realism pass produced no output.")
            return

        # Persist and add to graph
        persisted = self.persist_file(paths[0], subdir="scenes/realism")
        path = persisted or paths[0]
        child = SceneGraph.create_image_node(
            file_path=path,
            name="Chain — Realism Base",
            edge_type=EdgeType.IMG2IMG,
            params=GenerationParams(
                prompt="photorealistic, highly detailed, 8k",
                denoise_strength=0.45,
                model_type="sdxl",
            ),
        )
        self._graph.add_child(self._chain_current_source_id, child)
        self._save_graph()

        # Use the realism result as our chain starting point
        self._chain_current_source_id = child.id
        self._chain_current_source_path = path

        # Free VRAM before video generation
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Now start generating clips
        self._chain_phase = "clips"
        self._chain_idx = 0
        self._run_next_chain_clip()

    def _run_next_chain_clip(self) -> None:
        """Generate the next clip in the chain."""
        if self._chain_idx >= len(self._chain_queue):
            self._finish_chain()
            return

        clip = self._chain_queue[self._chain_idx]
        clip_num = self._chain_idx + 1
        total = self._chain_total

        self._chain_progress_label.setText(
            f"Clip {clip_num}/{total}: {clip['name']}\n"
            f"Prompt: {clip['prompt'][:80]}..."
        )
        self._show_status(
            f"Chain clip {clip_num}/{total}: {clip['name']}..."
        )

        pipeline = self._get_video_pipeline()
        if not pipeline:
            self._set_busy(False)
            self._chain_progress_label.setText("Failed — video pipeline not available.")
            return

        cfg = ProjectConfig.load(self.project_path)
        cfg.prompt = clip["prompt"]
        cfg.num_frames = clip["duration"]
        cfg.save(self.project_path)

        worker = InferenceWorker(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=1,
            image_path=self._chain_current_source_path,
            global_config=self.state.global_config,
            parent=self,
        )

        worker.finished_ok.connect(
            lambda path, c=clip: self._on_chain_clip_done(path, c)
        )
        self._connect_worker(worker)

    def _on_chain_clip_done(self, result_path: str, clip: dict) -> None:
        self._worker = None
        if not result_path:
            self._set_busy(False)
            self._chain_progress_label.setText(
                f"Clip '{clip['name']}' produced no output. Chain stopped."
            )
            return

        # Persist clip
        persisted = self.persist_file(result_path, subdir="scenes/chain_clips")
        clip_path = persisted or result_path

        # Add clip to graph
        clip_node = SceneGraph.create_clip_node(
            file_path=clip_path,
            name=f"Chain — {clip['name']}",
            edge_type=EdgeType.I2V,
            params=GenerationParams(
                prompt=clip["prompt"],
                duration_frames=clip["duration"],
                model_type="wan_i2v",
            ),
        )
        self._graph.add_child(self._chain_current_source_id, clip_node)
        self._save_graph()

        # Now extract best frame from this clip to use as next source
        self._chain_clip_node_id = clip_node.id
        self._chain_extract_clip_path = clip_path
        self._run_chain_extract(clip_path, clip_node.id)

    def _run_chain_extract(self, clip_path: str, clip_id: str) -> None:
        """Extract best frame from the just-generated clip."""
        output_dir = str(
            self.project_path / "scenes" / "chain_stills" / Path(clip_path).stem
        )

        worker = _StillGrabWorker(
            video_path=clip_path,
            count=5,
            output_dir=output_dir,
            parent=self,
        )
        worker.finished_ok.connect(
            lambda paths, cid=clip_id: self._on_chain_extract_done(paths, cid)
        )
        self._connect_worker(worker)

    def _on_chain_extract_done(self, paths: list[str], clip_id: str) -> None:
        self._worker = None
        if not paths:
            self._set_busy(False)
            self._chain_progress_label.setText("Frame extraction failed. Chain stopped.")
            return

        # Score and pick best
        from sdqt.models.still_scorer import score_frames, pick_best
        scores = score_frames(paths)
        best = pick_best(scores, top_n=1)

        best_path = best[0].path if best else paths[0]

        # Add best frame to graph
        child = SceneGraph.create_image_node(
            file_path=best_path,
            name=f"Chain — Best Frame (Clip {self._chain_idx + 1})",
            edge_type=EdgeType.STILL_EXTRACT,
            starred=True,
            sharpness_score=best[0].sharpness if best else 0,
        )
        self._graph.add_child(clip_id, child)
        self._save_graph()

        # This frame becomes the source for the next clip
        self._chain_current_source_id = child.id
        self._chain_current_source_path = best_path

        # Free VRAM between clips
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Advance to next clip
        self._chain_idx += 1
        self._run_next_chain_clip()

    def _finish_chain(self) -> None:
        """All clips in the chain are done."""
        self._set_busy(False)
        total = self._chain_total
        self._chain_progress_label.setText(
            f"Clip chain complete! {total} clips generated.\n"
            "All clips and best frames are in the scene graph. "
            "Review the tree, star your favorites, and iterate."
        )
        self._show_status(
            f"Clip chain complete — {total} clips, all best frames extracted. "
            "Check the scene graph tree."
        )
        self._run_chain_btn.setEnabled(False)

    @Slot()
    def _on_realism_run(self) -> None:
        """Run img2img realism pass on selected node."""
        if not self._selected_node_id:
            self._show_status("Select a node from the tree first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.IMAGE:
            self._show_status("Select an IMAGE node to run realism pass on.")
            return

        preset_idx = self._realism_preset.currentIndex()
        _, denoise = _REALISM_PRESETS[preset_idx]
        prompt = self._realism_prompt.currentText()

        pipeline = self._get_img_pipeline()
        if not pipeline:
            return

        cfg = ProjectConfig.load(self.project_path)
        self._active_strategy.set_config(cfg, "prompt", prompt)
        self._apply_quality_defaults(cfg)
        self._active_strategy.set_config(cfg, "denoising_strength", denoise)
        cfg.img2img_source_path = node.file_path
        cfg.save(self.project_path)

        self._show_status(f"Running realism pass (denoise={denoise:.2f})...")

        WorkerCls = self._active_strategy.resolve_worker("img2img")
        worker = WorkerCls(pipeline, self.project_name, node.file_path, cfg, parent=self)

        parent_id = node.id
        worker.finished_ok.connect(
            lambda paths, pid=parent_id, d=denoise, p=prompt:
                self._on_realism_done(paths, pid, d, p)
        )
        self._connect_worker(worker)

    def _on_realism_done(self, paths: list[str], parent_id: str, denoise: float, prompt: str) -> None:
        self._set_busy(False)
        self._worker = None
        if not paths:
            self._show_status("Realism pass produced no output.")
            return

        for i, path in enumerate(paths):
            persisted = self.persist_file(path, subdir="scenes/realism")
            child = SceneGraph.create_image_node(
                file_path=persisted or path,
                name=f"Realism d={denoise:.2f} #{i+1}",
                edge_type=EdgeType.IMG2IMG,
                params=GenerationParams(
                    prompt=prompt,
                    denoise_strength=denoise,
                    model_type="sdxl",
                ),
            )
            self._graph.add_child(parent_id, child)

        self._save_graph()
        self._show_status(f"Realism pass done — {len(paths)} result(s) added to tree.")

    # ── Auto Realism Pipeline ───────────────────────────────────────────

    @Slot()
    def _on_auto_realism(self) -> None:
        """Run img2img at all denoise strengths, score results, star the best."""
        if not self._selected_node_id:
            self._show_status("Select a node from the tree first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.IMAGE:
            self._show_status("Select an IMAGE node to run auto-realism on.")
            return

        pipeline = self._get_img_pipeline()
        if not pipeline:
            return

        prompt = self._realism_prompt.currentText()

        # Queue all denoise strengths
        self._auto_realism_queue = [d for _, d in _REALISM_PRESETS]
        self._auto_realism_parent_id = node.id
        self._auto_realism_source = node.file_path
        self._auto_realism_prompt = prompt
        self._auto_realism_results: list[tuple[str, float]] = []  # (path, denoise)
        self._auto_realism_total = len(self._auto_realism_queue)

        self._show_status(
            f"Auto Realism: running {self._auto_realism_total} passes "
            f"(denoise {self._auto_realism_queue[0]:.2f} → "
            f"{self._auto_realism_queue[-1]:.2f})..."
        )
        self._run_next_auto_realism()

    def _run_next_auto_realism(self) -> None:
        """Run the next denoise strength in the auto-realism queue."""
        if not self._auto_realism_queue:
            self._finish_auto_realism()
            return

        denoise = self._auto_realism_queue.pop(0)
        done = self._auto_realism_total - len(self._auto_realism_queue)
        self._show_status(
            f"Auto Realism: pass {done}/{self._auto_realism_total} "
            f"(denoise={denoise:.2f})..."
        )

        pipeline = self._active_strategy.get_pipeline(self.state)
        if not pipeline:
            self._show_status("Pipeline lost during auto-realism batch.")
            self._set_busy(False)
            return

        cfg = ProjectConfig.load(self.project_path)
        self._active_strategy.set_config(cfg, "prompt", self._auto_realism_prompt)
        self._apply_quality_defaults(cfg)
        self._active_strategy.set_config(cfg, "denoising_strength", denoise)
        cfg.img2img_source_path = self._auto_realism_source
        # Force batch=1 for auto-realism — we vary denoise, not batch
        cfg.img_batch_count = 1
        cfg.img_batch_size = 1
        cfg.save(self.project_path)

        WorkerCls = self._active_strategy.resolve_worker("img2img")
        worker = WorkerCls(pipeline, self.project_name,
                           self._auto_realism_source, cfg, parent=self)

        d = denoise  # capture for lambda
        worker.finished_ok.connect(
            lambda paths, den=d: self._on_auto_realism_step_done(paths, den)
        )
        self._connect_worker(worker)

    def _on_auto_realism_step_done(self, paths: list[str], denoise: float) -> None:
        """Handle one auto-realism step completing, then run the next."""
        self._worker = None
        if paths:
            for path in paths:
                persisted = self.persist_file(path, subdir="scenes/realism")
                self._auto_realism_results.append((persisted or path, denoise))

        # Free VRAM between passes to prevent OOM
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Don't call _set_busy(False) yet — chain to the next pass
        if self._auto_realism_queue:
            self._run_next_auto_realism()
        else:
            self._finish_auto_realism()

    def _finish_auto_realism(self) -> None:
        """All passes done — score results, add to graph, star the best."""
        self._set_busy(False)
        results = self._auto_realism_results
        if not results:
            self._show_status("Auto Realism produced no results.")
            return

        # Score all results
        self._show_status("Scoring realism results...")
        from sdqt.models.still_scorer import score_frames, pick_best
        paths = [r[0] for r in results]
        scores = score_frames(paths)
        best = pick_best(scores, top_n=1)
        best_paths = {s.path for s in best}

        # Build a path→denoise lookup
        path_to_denoise = {p: d for p, d in results}

        parent_id = self._auto_realism_parent_id
        prompt = self._auto_realism_prompt

        for fs in scores:
            denoise = path_to_denoise.get(fs.path, 0.0)
            is_best = fs.path in best_paths
            child = SceneGraph.create_image_node(
                file_path=fs.path,
                name=f"{'BEST ' if is_best else ''}Realism d={denoise:.2f}",
                edge_type=EdgeType.IMG2IMG,
                params=GenerationParams(
                    prompt=prompt,
                    denoise_strength=denoise,
                    model_type="sdxl",
                ),
                starred=is_best,
                sharpness_score=fs.sharpness,
                face_count=fs.face_count,
                composition_score=fs.composition,
            )
            self._graph.add_child(parent_id, child)

        self._save_graph()
        best_denoise = path_to_denoise.get(best[0].path, 0) if best else "?"
        self._show_status(
            f"Auto Realism complete — {len(results)} results, "
            f"best at denoise={best_denoise:.2f} (auto-starred). "
            "Hit Next to continue to Inpaint or skip to Orbit Clip."
        )

        # Clean up state
        self._auto_realism_queue = []
        self._auto_realism_results = []

    @Slot()
    def _on_inpaint_open(self) -> None:
        """Open selected image in the Image tab's inpaint mode."""
        if not self._selected_node_id:
            self._show_status("Select a node first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or not node.file_path:
            return
        # Set as source and switch to image tab
        self.state.set_image_path(node.file_path)
        self._show_status(f"Set source to {Path(node.file_path).name}. Switch to Image tab → Inpaint.")

    @Slot()
    def _on_orbit_generate(self) -> None:
        """Generate orbit clip from selected image."""
        if not self._selected_node_id:
            self._show_status("Select a source image from the tree first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.IMAGE:
            self._show_status("Select an IMAGE node as the source for orbit clip.")
            return

        prompt_idx = self._orbit_prompt.currentIndex()
        _, prompt = _ORBIT_PROMPTS[prompt_idx]
        duration = self._orbit_duration.currentData()
        dur_sec = (duration - 1) / 16

        if dur_sec > 3:
            self._show_status(
                f"WARNING: {dur_sec:.1f}s is risky! The scene may drift. "
                "Consider 2-3 seconds for consistency."
            )

        pipeline = self._get_video_pipeline()
        if not pipeline:
            return

        cfg = ProjectConfig.load(self.project_path)
        cfg.prompt = prompt
        cfg.num_frames = duration
        cfg.save(self.project_path)

        self._show_status(f"Generating orbit clip ({dur_sec:.1f}s, {duration} frames)...")

        worker = InferenceWorker(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=1,
            image_path=node.file_path,
            global_config=self.state.global_config,
            parent=self,
        )

        parent_id = node.id
        worker.finished_ok.connect(
            lambda path, pid=parent_id, p=prompt, d=duration:
                self._on_orbit_done(path, pid, p, d)
        )
        self._connect_worker(worker)

    def _on_orbit_done(self, result_path: str, parent_id: str, prompt: str, duration: int) -> None:
        self._set_busy(False)
        self._worker = None
        if not result_path:
            self._show_status("Orbit clip generation produced no output.")
            return

        persisted = self.persist_file(result_path, subdir="scenes/orbits")
        child = SceneGraph.create_clip_node(
            file_path=persisted or result_path,
            name=f"Orbit {duration}f",
            edge_type=EdgeType.I2V,
            params=GenerationParams(
                prompt=prompt,
                duration_frames=duration,
                model_type="wan_i2v",
            ),
        )
        self._graph.add_child(parent_id, child)
        self._save_graph()
        self._scene_tree.select_node(child.id)

        # Play preview
        self._orbit_preview.load_video(persisted or result_path, auto_play=True)
        self._show_status(f"Orbit clip done — added to tree. Select it and go to Extract Stills.")

    @Slot()
    def _on_extract_stills(self) -> None:
        """Extract stills from selected clip."""
        if not self._selected_node_id:
            self._show_status("Select a clip from the tree first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.CLIP:
            self._show_status("Select a CLIP node to extract stills from.")
            return

        count = self._extract_count.value()
        output_dir = str(self.project_path / "scenes" / "stills" / Path(node.file_path).stem)
        self._show_status(f"Extracting {count} stills from {Path(node.file_path).name}...")

        worker = _StillGrabWorker(
            video_path=node.file_path,
            count=count,
            output_dir=output_dir,
            parent=self,
        )

        parent_id = node.id
        worker.finished_ok.connect(
            lambda paths, pid=parent_id: self._on_extract_done(paths, pid)
        )
        self._connect_worker(worker)

    def _on_extract_done(self, paths: list[str], parent_id: str) -> None:
        self._set_busy(False)
        self._worker = None
        if not paths:
            self._show_status("Still extraction produced no frames.")
            return

        # Score frames for quality
        self._show_status("Scoring frames for quality...")
        from sdqt.models.still_scorer import score_frames, pick_best
        scores = score_frames(paths)
        best = pick_best(scores, top_n=3)
        best_paths = {s.path for s in best}

        count = len(scores)
        for i, fs in enumerate(scores):
            ts = (i + 1) / (count + 1)
            is_pick = fs.path in best_paths
            child = SceneGraph.create_image_node(
                file_path=fs.path,
                name=f"{'PICK ' if is_pick else ''}Still #{i+1:02d}",
                edge_type=EdgeType.STILL_EXTRACT,
                extract_timestamp=ts,
                starred=is_pick,
                sharpness_score=fs.sharpness,
                face_count=fs.face_count,
                composition_score=fs.composition,
            )
            self._graph.add_child(parent_id, child)

        self._save_graph()
        self._show_status(
            f"Extracted {count} stills — top {len(best)} auto-starred as PICK. "
            "Use these as scene starting points, or star/unstar your own picks."
        )

    @Slot()
    def _on_scene_generate(self) -> None:
        """Generate scene clip from selected still."""
        if not self._selected_node_id:
            self._show_status("Select a still from the tree first.")
            return
        node = self._graph.get(self._selected_node_id)
        if not node or node.node_type != NodeType.IMAGE:
            self._show_status("Select an IMAGE node as your scene starting point.")
            return

        prompt = self._scene_prompt.toPlainText().strip()
        if not prompt:
            self._show_status("Write a scene prompt first.")
            return

        duration = self._scene_duration.currentData()
        dur_sec = (duration - 1) / 16

        pipeline = self._get_video_pipeline()
        if not pipeline:
            return

        cfg = ProjectConfig.load(self.project_path)
        cfg.prompt = prompt
        cfg.num_frames = duration
        cfg.save(self.project_path)

        self._show_status(f"Generating scene clip ({dur_sec:.1f}s)...")

        worker = InferenceWorker(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            mode=1,
            image_path=node.file_path,
            global_config=self.state.global_config,
            parent=self,
        )

        parent_id = node.id
        worker.finished_ok.connect(
            lambda path, pid=parent_id, p=prompt, d=duration:
                self._on_scene_done(path, pid, p, d)
        )
        self._connect_worker(worker)

    def _on_scene_done(self, result_path: str, parent_id: str, prompt: str, duration: int) -> None:
        self._set_busy(False)
        self._worker = None
        if not result_path:
            self._show_status("Scene clip generation produced no output.")
            return

        persisted = self.persist_file(result_path, subdir="scenes/shots")
        child = SceneGraph.create_clip_node(
            file_path=persisted or result_path,
            name=f"Scene {Path(result_path).stem}",
            edge_type=EdgeType.I2V,
            params=GenerationParams(
                prompt=prompt,
                duration_frames=duration,
                model_type="wan_i2v",
            ),
        )
        self._graph.add_child(parent_id, child)
        self._save_graph()
        self._scene_tree.select_node(child.id)
        self._scene_preview.load_video(persisted or result_path, auto_play=True)
        self._show_status("Scene clip done — use 'Extract Best Frame' to continue iterating.")
        # Store the last scene clip ID for extract
        self._last_scene_clip_id = child.id

    @Slot()
    def _on_scene_extract(self) -> None:
        """Extract best frame from last generated scene clip."""
        clip_id = getattr(self, "_last_scene_clip_id", None)
        if not clip_id:
            # Fall back to selected node if it's a clip
            if self._selected_node_id:
                node = self._graph.get(self._selected_node_id)
                if node and node.node_type == NodeType.CLIP:
                    clip_id = node.id
        if not clip_id:
            self._show_status("No scene clip to extract from. Generate a scene clip first.")
            return

        clip_node = self._graph.get(clip_id)
        if not clip_node or not clip_node.file_path:
            self._show_status("Clip node has no file path.")
            return

        output_dir = str(self.project_path / "scenes" / "stills" / Path(clip_node.file_path).stem)
        self._show_status("Extracting best frames from scene clip...")

        worker = _StillGrabWorker(
            video_path=clip_node.file_path,
            count=5,  # Extract 5 candidates, user stars the best
            output_dir=output_dir,
            parent=self,
        )

        worker.finished_ok.connect(
            lambda paths, pid=clip_id: self._on_scene_extract_done(paths, pid)
        )
        self._connect_worker(worker)

    def _on_scene_extract_done(self, paths: list[str], parent_id: str) -> None:
        self._set_busy(False)
        self._worker = None
        if not paths:
            self._show_status("Frame extraction produced no results.")
            return

        # Score and auto-pick the best frame
        from sdqt.models.still_scorer import score_frames, pick_best
        scores = score_frames(paths)
        best = pick_best(scores, top_n=1)
        best_paths = {s.path for s in best}

        for i, fs in enumerate(scores):
            ts = (i + 1) / (len(scores) + 1)
            is_pick = fs.path in best_paths
            child = SceneGraph.create_image_node(
                file_path=fs.path,
                name=f"{'BEST ' if is_pick else ''}Scene Still #{i+1:02d}",
                edge_type=EdgeType.STILL_EXTRACT,
                extract_timestamp=ts,
                starred=is_pick,
                sharpness_score=fs.sharpness,
                face_count=fs.face_count,
                composition_score=fs.composition,
            )
            self._graph.add_child(parent_id, child)

        self._save_graph()
        self._show_status(
            f"Extracted {len(scores)} frames — best frame auto-starred. "
            "Use it as a new starting point (Scene Shot or Orbit Clip)."
        )

    # ── Face Restore handler ─────────────────────────────────────────

    def _on_face_restore_run(self) -> None:
        """Run face restoration on the selected image."""
        node = self._get_selected_image_node()
        if not node:
            self._show_status("Select an image from the tree first.")
            return

        source_path = node.file_path
        if not source_path or not Path(source_path).is_file():
            self._show_status("Selected node has no valid image file.")
            return

        model_name = self._face_restore_model.currentText()
        mp = self.state.global_config.model_paths
        models_dir = mp.get("upscaler_dir", "")
        model_path = str(Path(models_dir) / model_name) if models_dir else model_name

        output_dir = str(Path(self.project_path) / "face_restore")

        from sdqt.workers.face_restore_worker import FaceRestoreWorker
        worker = FaceRestoreWorker(
            source_path=source_path,
            model_path=model_path,
            output_dir=output_dir,
            parent=self,
        )
        worker.finished_ok.connect(self._on_face_restore_done)
        self._connect_worker(worker)

    def _on_face_restore_done(self, result_path: str) -> None:
        """Handle face restoration result."""
        self._set_busy(False)
        if not result_path or not Path(result_path).is_file():
            self._show_status("Face restore produced no output.")
            return

        # Show preview
        from PySide6.QtGui import QPixmap
        pix = QPixmap(result_path)
        if not pix.isNull():
            self._face_restore_result.setPixmap(
                pix.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )

        # Add to scene graph
        if self._selected_node_id:
            child = SceneGraph.create_image_node(
                file_path=result_path,
                name=f"Face Restored",
                edge_type=EdgeType.IMG2IMG,
            )
            self._graph.add_child(self._selected_node_id, child)
            self._save_graph()

        self._show_status(f"Face restoration complete: {Path(result_path).name}")

    # ── Face Swap handler ────────────────────────────────────────────

    def _on_face_swap_run(self) -> None:
        """Run face swap on the selected image."""
        node = self._get_selected_image_node()
        if not node:
            self._show_status("Select an image from the tree first.")
            return

        target_path = node.file_path
        if not target_path or not Path(target_path).is_file():
            self._show_status("Selected node has no valid image file.")
            return

        ref_path = self._face_swap_ref.path()
        if not ref_path or not Path(ref_path).is_file():
            self._show_status("Drop a reference face image first.")
            return

        swap_model = self._face_swap_model.currentData() or "inswapper_128"
        enhancer = self._face_swap_enhancer.currentData() or None
        if enhancer == "":
            enhancer = None

        mp = self.state.global_config.model_paths
        models_dir = mp.get("face_swap_models_dir", mp.get("models_root", ""))
        output_dir = str(Path(self.project_path) / "face_swap")

        from sdqt.workers.face_swap_worker import FaceSwapWorker
        worker = FaceSwapWorker(
            source_face_path=ref_path,
            target_path=target_path,
            models_dir=models_dir,
            output_dir=output_dir,
            swap_model=swap_model,
            enhancer=enhancer,
            parent=self,
        )
        worker.finished_ok.connect(self._on_face_swap_done)
        self._connect_worker(worker)

    def _on_face_swap_done(self, result_path: str) -> None:
        """Handle face swap result."""
        self._set_busy(False)
        if not result_path or not Path(result_path).is_file():
            self._show_status("Face swap produced no output.")
            return

        from PySide6.QtGui import QPixmap
        pix = QPixmap(result_path)
        if not pix.isNull():
            self._face_swap_result.setPixmap(
                pix.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )

        if self._selected_node_id:
            child = SceneGraph.create_image_node(
                file_path=result_path,
                name=f"Face Swapped",
                edge_type=EdgeType.IMG2IMG,
            )
            self._graph.add_child(self._selected_node_id, child)
            self._save_graph()

        self._show_status(f"Face swap complete: {Path(result_path).name}")

    # ── Hires Fix handler ────────────────────────────────────────────

    def _on_hires_fix_run(self) -> None:
        """Run Hires Fix (upscale + optional img2img second pass)."""
        node = self._get_selected_image_node()
        if not node:
            self._show_status("Select an image from the tree first.")
            return

        source_path = node.file_path
        if not source_path or not Path(source_path).is_file():
            self._show_status("Selected node has no valid image file.")
            return

        model_name = self._hires_model.currentText()
        mp = self.state.global_config.model_paths
        models_dir = mp.get("upscaler_dir", "")
        model_path = str(Path(models_dir) / model_name) if models_dir else model_name

        scale = self._hires_scale.value()
        do_second = self._hires_second_pass.isChecked()
        denoise = self._hires_denoise.value()
        steps = self._hires_steps.value()

        output_dir = str(Path(self.project_path) / "hires_fix")

        # Get img pipeline for second pass if needed
        img_pipeline = None
        prompt = ""
        neg_prompt = ""
        if do_second:
            img_pipeline = self._get_img_pipeline()
            cfg = ProjectConfig.load(self.project_path)
            prompt = self._active_strategy.get_config(cfg, "prompt") or ""
            neg_prompt = self._active_strategy.get_config(cfg, "negative_prompt") or ""

        from sdqt.workers.hires_fix_worker import HiresFixWorker
        worker = HiresFixWorker(
            source_path=source_path,
            upscale_model_path=model_path,
            output_dir=output_dir,
            scale=scale,
            do_second_pass=do_second,
            denoise_strength=denoise,
            img_pipeline=img_pipeline,
            prompt=prompt,
            negative_prompt=neg_prompt,
            steps=steps,
            cfg_scale=self._wizard_cfg.value(),
            sampler=self._wizard_sampler.currentText(),
            scheduler=self._wizard_scheduler.currentText(),
            parent=self,
        )
        worker.finished_ok.connect(self._on_hires_fix_done)
        self._connect_worker(worker)

    def _on_hires_fix_done(self, result_path: str) -> None:
        """Handle hires fix result."""
        self._set_busy(False)
        if not result_path or not Path(result_path).is_file():
            self._show_status("Hires fix produced no output.")
            return

        from PySide6.QtGui import QPixmap
        pix = QPixmap(result_path)
        if not pix.isNull():
            self._hires_result.setPixmap(
                pix.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )

        if self._selected_node_id:
            child = SceneGraph.create_image_node(
                file_path=result_path,
                name=f"Hires {self._hires_scale.value():.1f}x",
                edge_type=EdgeType.IMG2IMG,
            )
            self._graph.add_child(self._selected_node_id, child)
            self._save_graph()

        self._show_status(f"Hires fix complete: {Path(result_path).name}")

    # ── Inpaint seams handler ────────────────────────────────────────

    def _on_inpaint_seams(self) -> None:
        """Open the inpaint tab for fixing clip-to-clip seam frames."""
        node = self._get_selected_image_node()
        if node and node.file_path:
            # Switch to inpaint mode in the image tab
            self._show_status(
                "Open the Image tab → Inpaint to fix seam frames. "
                "Paint over the transition area and regenerate."
            )
        else:
            self._show_status("Select a seam frame from the tree first.")
