"""Create a Character — 6-page wizard sequence."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QEventLoop, Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import re

from supremediffusion.config.project_config import ProjectConfig

from sdqt import deps as _deps
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.lora_picker import LoRAPickerWidget
from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage
from sdqt.workers.base import BaseWorker
from sdqt.workers.interrogate import (
    BLIPInterrogateWorker,
    QwenCaptionWorker,
    WD14TaggerWorker,
)
from sdqt.workers.pipeline_load import PipelineLoadWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

from sdqt.utils.lora_tags import LORA_TAG_CAPTURE_RE as _LORA_TAG_RE, strip_lora_tags as _strip_lora_tags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pose descriptions for Page 5
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoseSpec:
    """A generated pose, tagged with its framing so datasets can be batched to
    the 3-distance rule and captioned with distance tags."""
    description: str
    distance: str                    # "close" | "medium" | "full"
    angle: str                       # "front" | "three_quarter" | "side" | "back"
    orientation: str = "portrait"    # "portrait" | "landscape" (AR default)


# Generation set is weighted toward FULL + MEDIUM (varied angles); the close-up
# majority the 512px formula wants is manufactured by cropping in the dataset
# step (see _build_pose_pools). ``back`` angle is full-body only.
POSES: list[PoseSpec] = [
    # Full body — angle coverage (front / three-quarter / side / back)
    PoseSpec("full body shot, standing front view, arms at sides, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, standing three-quarter view, slight turn, feet visible, head to toe, wide shot", "full", "three_quarter"),
    PoseSpec("full body shot, standing side profile view, feet visible, head to toe, wide shot", "full", "side"),
    PoseSpec("full body shot, standing back view from behind, feet visible, head to toe, wide shot", "full", "back"),
    PoseSpec("full body shot, standing, hands on hips, confident, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, walking forward dynamic stride, feet visible, head to toe, wide shot", "full", "front"),
    PoseSpec("full body shot, sitting in a chair front view, hands on lap, legs visible, wide framing", "full", "front"),
    PoseSpec("full body shot, sitting in a chair three-quarter view, legs crossed, whole body visible, wide framing", "full", "three_quarter"),
    PoseSpec("full body shot, sitting cross-legged on the ground, casual pose, whole body visible, wide framing", "full", "front"),
    PoseSpec("full body shot, kneeling, upright posture, whole body visible, wide framing", "full", "three_quarter"),
    PoseSpec("full body shot, reclining on a sofa, leaning back, one arm resting, entire body in frame", "full", "side", "landscape"),
    PoseSpec("full body shot, laying on side, head resting on arm, peaceful, entire body in frame", "full", "side", "landscape"),
    # Medium / waist-up — varied angles
    PoseSpec("medium shot, waist up, front view, relaxed expression, neutral background", "medium", "front"),
    PoseSpec("medium shot, waist up, three-quarter view, soft smile, neutral background", "medium", "three_quarter"),
    PoseSpec("medium shot, waist up, side profile view, neutral background", "medium", "side"),
    PoseSpec("medium shot, upper body, hands on hips, front view, neutral background", "medium", "front"),
    PoseSpec("medium shot, upper body, arms crossed, three-quarter view, neutral background", "medium", "three_quarter"),
    # Close-up portraits (a few genuine ones for expression variety; the rest
    # are cropped from full/medium poses to hit the 60% close-up target)
    PoseSpec("close-up portrait, head and shoulders, front view, neutral expression, sharp focus on face", "close", "front"),
    PoseSpec("close-up portrait, head and shoulders, three-quarter view, gentle smile, sharp focus on face", "close", "three_quarter"),
    PoseSpec("close-up portrait, head and shoulders, side profile view, sharp focus on face", "close", "side"),
    PoseSpec("close-up portrait, head and shoulders, front view, soft smile, sharp focus on face", "close", "front"),
]

# Extra negative terms appended to user negative for pose generation
# to prevent close-up / portrait-only framing
_POSE_NEGATIVE_EXTRA = (
    "close-up, closeup, portrait, headshot, cropped, upper body only, "
    "face only, bust shot, shoulders up, head and shoulders, "
    "cut off, out of frame, partial body"
)

# Framing guarantee prepended to the BASE image prompt. The base becomes the
# canonical reference for every pose (and for any face/body swap), so it must be
# a clean full-body, front-facing, single-subject shot — not a headshot.
_BASE_FRAMING = (
    "solo, one person, full body photo, facing the viewer, standing front view, "
    "head to toe, entire body visible, wide shot"
)

# Distance + angle caption tags (trigger word leads, then these). Distance tags
# teach the model that blur belongs to wide shots, not the character.
_DISTANCE_TAG = {"close": "close-up shot", "medium": "medium shot", "full": "full body shot"}
_ANGLE_TAG = {
    "front": "front view", "three_quarter": "three-quarter view",
    "side": "side profile view", "back": "back view",
}

# 3-distance composition targets. Close-heavy for 512px video / low-VRAM (forces
# facial detail); balanced for high-res image LoRAs (full body still has detail).
RATIO_VIDEO = {"close": 0.60, "medium": 0.30, "full": 0.10}
RATIO_HIGHRES = {"close": 0.40, "medium": 0.30, "full": 0.30}


def _caption_for(trigger: str, distance: str, angle: str, desc: str) -> str:
    """Trigger-first caption with distance + angle tags, then the description."""
    parts = [trigger, _DISTANCE_TAG.get(distance, "full body shot"),
             _ANGLE_TAG.get(angle, ""), desc]
    return ", ".join(p for p in parts if p).rstrip(", ")


def _pose_filename(idx: int, distance: str, angle: str) -> str:
    """Encode distance + angle in the saved pose filename (``__`` separates
    fields so single-underscore values like ``three_quarter`` survive)."""
    return f"pose_{idx + 1:03d}__{distance}__{angle}.png"


def _parse_pose_distance_angle(path) -> tuple[str, str]:
    """Recover (distance, angle) from a saved pose filename; legacy → full/front."""
    parts = Path(path).stem.split("__")
    if len(parts) >= 3 and parts[1] in ("close", "medium", "full"):
        return parts[1], parts[2]
    return "full", "front"


def _pose_negative_for(distance: str, base_neg: str) -> str:
    """Distance-conditional negative. Full body keeps the anti-crop bans; medium
    and close MUST NOT ban close-ups or they never frame tight."""
    base = (base_neg or "").strip().rstrip(",")
    if distance == "full":
        extra = _POSE_NEGATIVE_EXTRA
    elif distance == "medium":
        extra = "cut off, out of frame, partial body, deformed, extra limbs"
    else:  # close
        extra = "full body, wide shot, cropped face, out of frame, deformed"
    return f"{base}, {extra}" if base else extra

# Model-family native resolution buckets (÷64, at the family's megapixel target).
# Off-spec resolutions (e.g. SDXL at 512) cause duplication/mutations, so each
# pose picks from the SELECTED model's bucket set. The 1-MP ÷64 dims used for
# sdxl/flux/zimage are also ÷16 and ÷32, so they are valid for every family.
_SDXL_BUCKETS = [
    ("Portrait 832×1216", 832, 1216),
    ("Portrait 896×1152", 896, 1152),
    ("Portrait 768×1344", 768, 1344),
    ("Square 1024×1024", 1024, 1024),
    ("Landscape 1216×832", 1216, 832),
    ("Landscape 1152×896", 1152, 896),
    ("Landscape 1344×768", 1344, 768),
]
# FLUX is flexible and handles higher pixel counts well — offer the 1-MP set
# plus a couple of larger full-body options.
_FLUX_BUCKETS = _SDXL_BUCKETS + [
    ("Portrait 1024×1536", 1024, 1536),
    ("Landscape 1536×1024", 1536, 1024),
]
_AR_BUCKETS = {
    "sd15": [
        ("Portrait 512×768", 512, 768),
        ("Portrait 512×704", 512, 704),
        ("Square 512×512", 512, 512),
        ("Landscape 768×512", 768, 512),
        ("Landscape 704×512", 704, 512),
    ],
    "sdxl": _SDXL_BUCKETS,
    "flux": _FLUX_BUCKETS,
    "zimage": _SDXL_BUCKETS,
}
# Default (w, h) per family + orientation — sensible full-body sweet spots.
_AR_DEFAULTS = {
    "sd15": {"portrait": (512, 768), "landscape": (768, 512)},
    "sdxl": {"portrait": (832, 1216), "landscape": (1216, 832)},
    "flux": {"portrait": (832, 1216), "landscape": (1216, 832)},
    "zimage": {"portrait": (832, 1216), "landscape": (1216, 832)},
}


def _buckets_for_model(model_type: str) -> list:
    """Valid resolution buckets for the selected model family."""
    return _AR_BUCKETS.get((model_type or "").lower(), _SDXL_BUCKETS)


def _ar_defaults_for_model(model_type: str) -> dict:
    return _AR_DEFAULTS.get((model_type or "").lower(), _AR_DEFAULTS["sdxl"])


# Recommended CFG / steps / sampler / scheduler per model family, loaded into the
# wizard's settings when a model is selected. Sampler/scheduler names match
# sd_samplers.list_samplers()/list_schedulers(); flux/zimage ignore sampler.
_FAMILY_RECOMMENDED = {
    "sd15":   {"cfg": 7.0, "steps": 25, "sampler": "DPM++ 2M", "scheduler": "Karras"},
    "sdxl":   {"cfg": 4.0, "steps": 30, "sampler": "DPM++ 2M", "scheduler": "Karras"},
    "flux":   {"cfg": 3.5, "steps": 35, "sampler": "Euler",    "scheduler": "Automatic"},
    "zimage": {"cfg": 0.0, "steps": 9,  "sampler": "Euler",    "scheduler": "Automatic"},
}


# Medium token prepended to the prompt based on the style chosen on page 1.
_STYLE_MEDIUM = {"realism": "photo", "cartoon": "cartoon", "anime": "anime"}


def _strategy_for(model_family: str):
    """Return the ModelFamilyStrategy for a family (defaults to SDXL)."""
    from sdqt.tabs.image_tabs.model_family import STRATEGY_MAP, SDXL_STRATEGY
    return STRATEGY_MAP.get((model_family or "sdxl").lower(), SDXL_STRATEGY)


def _apply_family_gen_cfg(cfg, strategy, cs, *, prompt, negative, width, height,
                          batch_count, seed=None) -> None:
    """Write generation params onto the family's prefixed config fields.

    Common fields go through the strategy's {prefix}_{field} convention so each
    pipeline reads its own keys. SD/SDXL-only extras (checkpoint, sampler,
    scheduler, LoRAs) are set only for those families.
    """
    strategy.set_config(cfg, "prompt", prompt)
    strategy.set_config(cfg, "negative_prompt", negative)
    strategy.set_config(cfg, "steps", int(cs.steps))
    strategy.set_config(cfg, "cfg_scale", float(cs.cfg_scale))
    strategy.set_config(cfg, "width", int(width))
    strategy.set_config(cfg, "height", int(height))
    strategy.set_config(cfg, "seed", int(seed if seed is not None else cs.seed))
    strategy.set_config(cfg, "batch_count", int(batch_count))
    if strategy.family in ("sd15", "sdxl"):
        cfg.img_batch_size = 1
        if cs.checkpoint:
            cfg.img_checkpoint = cs.checkpoint
        if cs.sampler:
            cfg.img_sampler = cs.sampler
        if cs.scheduler:
            cfg.img_scheduler = cs.scheduler
        cfg.img_loras = cs.selected_loras
        cfg.img_lora_multipliers = cs.lora_multipliers
        # ADetailer face restore (Txt2ImgWorker runs it when enabled). SD-only;
        # flux/zimage workers ignore this flag.
        cfg.adetailer_enabled = bool(getattr(cs, "adetailer", False))


def _run_family_txt2img(wizard, app_state, cfg, strategy, on_done, on_error) -> None:
    """Dispatch a txt2img run to the right family pipeline + worker, loading the
    pipeline first if needed."""
    pipeline = strategy.get_pipeline(app_state)
    if pipeline is None:
        loader = PipelineLoadWorker(app_state, strategy.load_method, parent=wizard)
        wizard._run_worker(
            loader,
            on_done=lambda _: _run_family_txt2img(
                wizard, app_state, cfg, strategy, on_done, on_error),
            on_error=on_error,
        )
        return
    worker_cls = strategy.resolve_worker("txt2img")
    worker = worker_cls(
        pipeline=pipeline,
        project_name=_project_name(app_state),
        project_config=cfg,
        parent=wizard,
    )
    wizard._run_worker(worker, on_done=on_done, on_error=on_error)


class GenerationSettingsBar(QWidget):
    """Persistent generation-settings bar shown at the bottom of every wizard
    page: model + sampler/scheduler + steps/CFG/seed + base width/height.

    Selecting a model loads that family's recommended CFG / steps / sampler and
    a native portrait base resolution. All controls write live to
    CharacterState so every page reads the current settings.
    """

    def __init__(self, char_state, app_state, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._loading = False
        self._build_ui()

    def _build_ui(self) -> None:
        from supremediffusion.models.sd_samplers import list_samplers, list_schedulers

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        grp = QGroupBox("Generation Settings")
        outer.addWidget(grp)
        v = QVBoxLayout(grp)
        v.setSpacing(4)

        r0 = QHBoxLayout()
        r0.addWidget(QLabel("Model:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(280)
        try:
            from supremediffusion.models.sd_models import scan_all_image_models
            for m in scan_all_image_models(self._app_state.global_config):
                self._ckpt_combo.addItem(
                    f"[{m.model_type.upper()}] {m.name}", (m.name, m.model_type))
        except Exception:
            logger.debug("model scan failed", exc_info=True)
        self._ckpt_combo.currentIndexChanged.connect(self._on_model_changed)
        r0.addWidget(self._ckpt_combo, 1)
        v.addLayout(r0)

        r1 = QHBoxLayout()
        r1.setSpacing(6)
        r1.addWidget(QLabel("Sampler:"))
        self._sampler_combo = QComboBox()
        try:
            self._sampler_combo.addItems(list_samplers())
        except Exception:
            pass
        self._sampler_combo.currentTextChanged.connect(lambda _t: self._push())
        r1.addWidget(self._sampler_combo)
        r1.addWidget(QLabel("Scheduler:"))
        self._sched_combo = QComboBox()
        try:
            self._sched_combo.addItems(list_schedulers())
        except Exception:
            pass
        self._sched_combo.currentTextChanged.connect(lambda _t: self._push())
        r1.addWidget(self._sched_combo)
        r1.addStretch()
        v.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(6)
        r2.addWidget(QLabel("Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 150)
        self._steps_spin.valueChanged.connect(lambda _v: self._push())
        r2.addWidget(self._steps_spin)
        r2.addWidget(QLabel("CFG:"))
        self._cfg_spin = QDoubleSpinBox()
        self._cfg_spin.setRange(0.0, 30.0)
        self._cfg_spin.setSingleStep(0.5)
        self._cfg_spin.setDecimals(1)
        self._cfg_spin.valueChanged.connect(lambda _v: self._push())
        r2.addWidget(self._cfg_spin)
        r2.addWidget(QLabel("Seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 2147483647)
        self._seed_spin.valueChanged.connect(lambda _v: self._push())
        r2.addWidget(self._seed_spin)
        r2.addWidget(QLabel("W:"))
        self._w_spin = QSpinBox()
        self._w_spin.setRange(256, 2048)
        self._w_spin.setSingleStep(64)
        self._w_spin.valueChanged.connect(lambda _v: self._push())
        r2.addWidget(self._w_spin)
        r2.addWidget(QLabel("H:"))
        self._h_spin = QSpinBox()
        self._h_spin.setRange(256, 2048)
        self._h_spin.setSingleStep(64)
        self._h_spin.valueChanged.connect(lambda _v: self._push())
        r2.addWidget(self._h_spin)
        r2.addStretch()
        v.addLayout(r2)

        r3 = QHBoxLayout()
        self._adetailer_cb = QCheckBox("ADetailer face restore (SD/SDXL base + poses)")
        self._adetailer_cb.setChecked(bool(getattr(self._cs, "adetailer", True)))
        self._adetailer_cb.setToolTip(
            "Run ADetailer (face detect + high-detail inpaint) on each SD/SDXL "
            "generated image for crisper faces. Flux/Z-Image generations skip it."
        )
        self._adetailer_cb.toggled.connect(lambda _v: self._push())
        r3.addWidget(self._adetailer_cb)
        r3.addStretch()
        v.addLayout(r3)

    def _ckpt_index_for_name(self, name: str) -> int:
        for i in range(self._ckpt_combo.count()):
            d = self._ckpt_combo.itemData(i)
            if d and d[0] == name:
                return i
        return -1

    def _selected_model(self) -> tuple:
        d = self._ckpt_combo.currentData()
        if d:
            return d[0], d[1]
        return self._ckpt_combo.currentText(), self._cs.model_family or "sdxl"

    @Slot()
    def _on_model_changed(self) -> None:
        if self._loading:
            return
        name, family = self._selected_model()
        self._cs.checkpoint = name
        self._cs.model_family = family
        rec = _FAMILY_RECOMMENDED.get(family, _FAMILY_RECOMMENDED["sdxl"])
        self._loading = True
        self._steps_spin.setValue(int(rec["steps"]))
        self._cfg_spin.setValue(float(rec["cfg"]))
        si = self._sampler_combo.findText(rec["sampler"])
        if si >= 0:
            self._sampler_combo.setCurrentIndex(si)
        ci = self._sched_combo.findText(rec["scheduler"])
        if ci >= 0:
            self._sched_combo.setCurrentIndex(ci)
        pw, ph = _ar_defaults_for_model(family)["portrait"]
        self._w_spin.setValue(int(pw))
        self._h_spin.setValue(int(ph))
        self._loading = False
        self._push()

    def _push(self) -> None:
        """Write all controls to CharacterState (live)."""
        if self._loading:
            return
        name, family = self._selected_model()
        if name:
            self._cs.checkpoint = name
            self._cs.model_family = family
        self._cs.sampler = self._sampler_combo.currentText()
        self._cs.scheduler = self._sched_combo.currentText()
        self._cs.steps = self._steps_spin.value()
        self._cs.cfg_scale = self._cfg_spin.value()
        self._cs.seed = self._seed_spin.value()
        self._cs.width = self._w_spin.value()
        self._cs.height = self._h_spin.value()
        self._cs.adetailer = self._adetailer_cb.isChecked()

    def load_from_state(self) -> None:
        """Populate the controls from CharacterState (call once after init)."""
        self._loading = True
        i = self._ckpt_index_for_name(self._cs.checkpoint) if self._cs.checkpoint else -1
        if i < 0 and self._cs.checkpoint:
            self._ckpt_combo.insertItem(
                0, f"[{(self._cs.model_family or 'SDXL').upper()}] {self._cs.checkpoint}",
                (self._cs.checkpoint, self._cs.model_family or "sdxl"))
            i = 0
        if i >= 0:
            self._ckpt_combo.setCurrentIndex(i)
        if self._cs.sampler:
            si = self._sampler_combo.findText(self._cs.sampler)
            if si >= 0:
                self._sampler_combo.setCurrentIndex(si)
        if self._cs.scheduler:
            ci = self._sched_combo.findText(self._cs.scheduler)
            if ci >= 0:
                self._sched_combo.setCurrentIndex(ci)
        fam = self._cs.model_family or self._selected_model()[1]
        pw, ph = _ar_defaults_for_model(fam)["portrait"]
        rec = _FAMILY_RECOMMENDED.get(fam, _FAMILY_RECOMMENDED["sdxl"])
        self._steps_spin.setValue(int(self._cs.steps or rec["steps"]))
        self._cfg_spin.setValue(float(self._cs.cfg_scale if self._cs.cfg_scale else rec["cfg"]))
        self._seed_spin.setValue(int(self._cs.seed if self._cs.seed is not None else -1))
        self._w_spin.setValue(int(self._cs.width or pw))
        self._h_spin.setValue(int(self._cs.height or ph))
        self._adetailer_cb.setChecked(bool(getattr(self._cs, "adetailer", True)))
        self._loading = False
        self._push()


class PortraitCropDialog(QDialog):
    """Crop a supplied image to a target (portrait) aspect with a draggable,
    aspect-locked rectangle. Returns the cropped image path via ``result_path``.
    """

    def __init__(self, image_path: str, aspect: float, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crop to Portrait")
        self.setMinimumSize(560, 640)
        self._image_path = image_path
        self._result_path: str | None = None

        from sdqt.widgets.crop_editor import _CropCanvas

        v = QVBoxLayout(self)
        note = QLabel(
            "Drag / resize the rectangle to choose the area to keep. It's locked "
            "to the model's portrait aspect so the base is full-body, not cropped odd."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaa; font-size: 13px;")
        v.addWidget(note)

        self._canvas = _CropCanvas()
        pm = QPixmap(image_path)
        self._canvas.set_image(pm)
        # Seed a centered portrait crop, then lock the aspect.
        w_norm = aspect if aspect <= 1.0 else 1.0
        self._canvas.set_crop_rect((1.0 - w_norm) / 2.0, 0.0, w_norm, 1.0)
        self._canvas.set_aspect(aspect)
        v.addWidget(self._canvas, 1)

        from PySide6.QtWidgets import QDialogButtonBox
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _accept(self) -> None:
        from PIL import Image
        import tempfile
        x, y, w, h = self._canvas.crop_rect()
        try:
            img = Image.open(self._image_path).convert("RGB")
            W, H = img.size
            left, top = max(0, int(x * W)), max(0, int(y * H))
            right, bottom = min(W, int((x + w) * W)), min(H, int((y + h) * H))
            if right - left < 8 or bottom - top < 8:
                self.reject()
                return
            crop = img.crop((left, top, right, bottom))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="char_crop_", delete=False)
            tmp.close()
            crop.save(tmp.name)
            self._result_path = tmp.name
            self.accept()
        except Exception:
            logger.warning("Portrait crop failed", exc_info=True)
            self.reject()

    @property
    def result_path(self) -> str | None:
        return self._result_path


def _pose_is_landscape(pose_text: str) -> bool:
    low = pose_text.lower()
    return "laying" in low or "reclining" in low

# ---------------------------------------------------------------------------
# Shared character state
# ---------------------------------------------------------------------------


@dataclass
class CharacterState:
    name: str = ""
    description: str = ""
    style: str = "realism"
    selected_loras: list = field(default_factory=list)
    lora_multipliers: str = ""
    lora_prompt_tags: str = ""      # "<lora:name:w>, ..." for prompt injection
    lora_trigger_words: str = ""    # "trigger1, trigger2, ..." from metadata
    positive_prompt: str = ""
    negative_prompt: str = ""
    reference_image: str = ""        # optional user-supplied character reference
    base_images: list[str] = field(default_factory=list)
    selected_base: str = ""
    face_swap_enabled: bool = False
    face_source_path: str = ""
    face_result_path: str = ""
    face_enhancer: str = ""
    face_enhancer_strength: float = 0.5
    face_blend_ratio: float = 0.5    # face-swap enhancer blend (0=raw swap, 1=full enhance)
    body_swap_enabled: bool = False
    body_source_path: str = ""
    body_result_path: str = ""
    body_ip_scale: float = 0.8       # body-swap IP-Adapter identity strength
    body_denoise: float = 0.75       # body-swap inpaint denoise
    body_cfg: float = 7.0            # body-swap CFG
    body_cn_strength: float = 0.7    # body-swap ControlNet (pose) strength
    pose_images: list[str] = field(default_factory=list)
    approved_poses: list[str] = field(default_factory=list)
    # Parallel to approved_poses: {distance, angle, orientation, pose_index}.
    # Runtime-only; the durable source of truth on reload is the saved filename.
    approved_pose_specs: list = field(default_factory=list)
    checkpoint: str = ""
    model_family: str = ""           # "sd15" / "sdxl" / "flux" / "zimage"
    sampler: str = ""
    scheduler: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    seed: int = -1
    ref_look_strength: float = 0.7   # IP-Adapter scale: canonical look → poses
    apply_body_to_poses: bool = True  # body-swap canonical body onto every pose
    adetailer: bool = True            # ADetailer face restore on SD/SDXL base+poses

    # Scalar fields persisted to character.json (image paths are saved as files
    # and reconstructed on load — see load_character).
    _META_FIELDS = (
        "name", "description", "style", "selected_loras", "lora_multipliers",
        "lora_prompt_tags", "lora_trigger_words", "positive_prompt", "negative_prompt",
        "face_swap_enabled", "face_enhancer", "face_enhancer_strength", "face_blend_ratio",
        "body_swap_enabled", "body_ip_scale", "body_denoise", "body_cfg", "body_cn_strength",
        "checkpoint", "model_family", "sampler", "scheduler", "steps", "cfg_scale",
        "width", "height", "seed", "ref_look_strength", "apply_body_to_poses", "adetailer",
    )

    def to_meta(self) -> dict:
        return {k: getattr(self, k) for k in self._META_FIELDS}

    def apply_meta(self, d: dict) -> None:
        for k in self._META_FIELDS:
            if k in d:
                setattr(self, k, d[k])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _characters_dir(app_state) -> Path:
    """Resolve the saved-characters directory."""
    cd = app_state.global_config.characters_dir
    if not cd:
        from supremediffusion.config.defaults import APP_ROOT
        cd = str(APP_ROOT / "library" / "characters")
    return Path(cd)


def load_character(char_dir) -> CharacterState:
    """Reconstruct a CharacterState from a saved character directory."""
    char_dir = Path(char_dir)
    cs = CharacterState()
    meta_path = char_dir / "character.json"
    if meta_path.is_file():
        try:
            with open(meta_path) as f:
                cs.apply_meta(json.load(f))
        except Exception:
            logger.warning("Failed to read %s", meta_path, exc_info=True)
    # Reconstruct image paths from the saved files.
    base = char_dir / "base.png"
    if base.is_file():
        cs.selected_base = str(base)
    ref = char_dir / "reference_ref.png"
    if ref.is_file():
        cs.reference_image = str(ref)
    face = char_dir / "face_ref.png"
    if face.is_file():
        cs.face_source_path = str(face)
    body = char_dir / "body_ref.png"
    if body.is_file():
        cs.body_source_path = str(body)
    poses_dir = char_dir / "poses"
    if poses_dir.is_dir():
        cs.approved_poses = sorted(str(p) for p in poses_dir.glob("*.png"))
        cs.pose_images = list(cs.approved_poses)
        # Recover framing from filenames (legacy files → full/front).
        cs.approved_pose_specs = [
            {"distance": d, "angle": a, "orientation":
                "landscape" if d == "full" and a == "side" else "portrait"}
            for d, a in (_parse_pose_distance_angle(p) for p in cs.approved_poses)
        ]
    return cs


def _crop_head(image_path: str, out_path: str, pipe) -> bool:
    """Crop the head/face region (with padding for hair) from an image.

    Uses InsightFace detection via the FaceSwapPipeline. Returns True on success.
    """
    from PIL import Image
    try:
        faces = pipe.detect_faces(image_path)
        if not faces:
            return False
        # Largest face
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                   reverse=True)
        x1, y1, x2, y2 = faces[0]["bbox"]
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        bw, bh = x2 - x1, y2 - y1
        # Expand to include hair/neck (60% pad, a bit more on top).
        px, py = int(bw * 0.6), int(bh * 0.6)
        left = max(0, x1 - px)
        top = max(0, y1 - int(py * 1.3))
        right = min(W, x2 + px)
        bottom = min(H, y2 + py)
        img.crop((left, top, right, bottom)).save(out_path)
        return True
    except Exception:
        logger.debug("head crop failed for %s", image_path, exc_info=True)
        return False


def _crop_upper_body(image_path: str, out_path: str, pipe) -> bool:
    """Crop a waist-up (medium) region using the detected face as an anchor."""
    from PIL import Image
    try:
        faces = pipe.detect_faces(image_path)
        if not faces:
            return False
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                   reverse=True)
        x1, y1, x2, y2 = faces[0]["bbox"]
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        bw, bh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2.0
        # Waist is ~4.5 face-heights below the chin; widen to ~3x face width.
        top = max(0, int(y1 - 0.6 * bh))
        bottom = min(H, int(y2 + 4.5 * bh))
        half_w = 1.6 * bw
        left = max(0, int(cx - half_w))
        right = min(W, int(cx + half_w))
        if right - left < 16 or bottom - top < 16:
            return False
        img.crop((left, top, right, bottom)).save(out_path)
        return True
    except Exception:
        logger.debug("upper-body crop failed for %s", image_path, exc_info=True)
        return False


def _compose_dataset(pools: dict, out_dir: Path, ratio: dict, *, total: int | None = None) -> str:
    """Assemble a FLAT folder of NNN.png + NNN.txt sampled from per-distance
    pools to hit ``ratio`` (e.g. {"close":0.6,"medium":0.3,"full":0.1}).
    Oversamples (duplicates) a short pool, random-subsets a long one."""
    import random
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()
    avail = {k: list(v) for k, v in pools.items() if k in ratio}
    if not any(avail.values()):
        return str(out_dir)
    if total is None:
        total = sum(len(v) for v in avail.values())
    rng = random.Random(42)
    n = 0
    for dist, frac in ratio.items():
        pool = avail.get(dist) or []
        if not pool:
            continue
        want = max(1, round(frac * total))
        if len(pool) >= want:
            chosen = rng.sample(pool, want)
        else:  # oversample with duplication
            chosen = list(pool) + [rng.choice(pool) for _ in range(want - len(pool))]
            logger.info("compose: %s pool short (%d/%d) — duplicated to balance",
                        dist, len(pool), want)
        for img, cap in chosen:
            n += 1
            dst = out_dir / f"{n:03d}.png"
            try:
                from PIL import Image
                Image.open(img).convert("RGB").save(dst)
            except Exception:
                import shutil as _sh
                _sh.copy2(img, dst)
            dst.with_suffix(".txt").write_text(cap)
    return str(out_dir)


def _build_character_datasets(char_dir: Path, cs: CharacterState, app_state) -> dict:
    """Build per-distance pools from the approved poses (+ head/upper-body crops),
    then compose the 3-distance datasets the trainers consume.

    Returns {"trigger", "video512" (60/30/10), "highres" (40/30/30),
    "full", "face"} — the last two kept for back-compat.
    """
    import shutil as _sh

    safe = "".join(c if c.isalnum() else "_" for c in (cs.name or "char")).strip("_").lower()
    trigger = safe or "character"
    desc = (cs.description or cs.positive_prompt or "").strip()
    poses = [p for p in cs.approved_poses if p and Path(p).is_file()]
    specs = cs.approved_pose_specs or []

    pool_root = char_dir / "datasets" / "_pool"
    for sub in ("close", "medium", "full"):
        (pool_root / sub).mkdir(parents=True, exist_ok=True)
        for old in (pool_root / sub).glob("*"):
            old.unlink()
    pools: dict = {"close": [], "medium": [], "full": []}

    def _add(dist: str, src_img: str, angle: str) -> None:
        idx = len(pools[dist]) + 1
        dst = pool_root / dist / f"{idx:03d}.png"
        try:
            from PIL import Image
            Image.open(src_img).convert("RGB").save(dst)
        except Exception:
            _sh.copy2(src_img, dst)
        pools[dist].append((str(dst), _caption_for(trigger, dist, angle, desc)))

    # 1) Each generated pose → its own distance pool.
    for i, pose in enumerate(poses):
        sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
        _add(sp.get("distance", "full"), pose, sp.get("angle", "front"))

    # 2) Crop close (head) + medium (waist-up) from full/medium poses (crop-dominant).
    face_pipe = None
    try:
        from supremediffusion.models.face_swap import FaceSwapPipeline
        models_dir = app_state.global_config.model_paths.get("face_models_dir", "")
        face_pipe = FaceSwapPipeline(models_dir) if models_dir else FaceSwapPipeline()
        crop_dir = pool_root / "_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        for old in crop_dir.glob("*"):
            old.unlink()
        cn = 0
        for i, pose in enumerate(poses):
            sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
            dist, angle = sp.get("distance", "full"), sp.get("angle", "front")
            if dist in ("full", "medium"):
                cn += 1
                head = crop_dir / f"head_{cn:03d}.png"
                if _crop_head(pose, str(head), face_pipe):
                    _add("close", str(head), angle)
            if dist == "full":
                cn += 1
                up = crop_dir / f"upper_{cn:03d}.png"
                if _crop_upper_body(pose, str(up), face_pipe):
                    _add("medium", str(up), angle)
    except Exception:
        logger.warning("Crop pool build failed (InsightFace unavailable?)", exc_info=True)
    finally:
        if face_pipe is not None:
            try:
                face_pipe.release()
            except Exception:
                pass

    datasets_root = char_dir / "datasets"
    video512 = _compose_dataset(pools, datasets_root / "video512", RATIO_VIDEO)
    highres = _compose_dataset(pools, datasets_root / "highres", RATIO_HIGHRES)
    # Back-compat single-distance variants.
    full_only = _compose_dataset({"full": pools["full"]}, datasets_root / "full", {"full": 1.0})
    face_only = (_compose_dataset({"close": pools["close"]}, datasets_root / "face", {"close": 1.0})
                 if pools["close"] else None)

    logger.info("datasets: pools close=%d medium=%d full=%d", len(pools["close"]),
                len(pools["medium"]), len(pools["full"]))
    return {
        "trigger": trigger,
        "video512": video512,
        "highres": highres,
        "full": full_only,
        "face": face_only,
    }


def _project_path(app_state) -> Path:
    """Return the current project directory path."""
    name = app_state.current_project or "_default"
    return app_state.project_manager.get_project_path(name)


def _load_project_config(app_state) -> ProjectConfig:
    """Load the current project's ProjectConfig."""
    return ProjectConfig.load(_project_path(app_state))


def _project_name(app_state) -> str:
    return app_state.current_project or "_default"


def _detect_prompt_style(checkpoint_name: str, app_state) -> str:
    """Infer the correct prompt style key from the checkpoint name/path.

    Returns a key that matches :data:`prompt_enhance._SYSTEM_PROMPTS`:
    ``"pony_real"``, ``"pony_anime"``, ``"illustrious"``, ``"noobai"``,
    ``"sd15"``, ``"sdxl"``, ``"sd3"``, ``"flux"``.
    """
    low = checkpoint_name.lower()

    # Pony-based (SDXL finetune but needs score tags + booru style)
    if "pony" in low:
        # Anime-leaning pony vs realistic pony
        if any(kw in low for kw in ("anime", "toon", "cartoon", "illustration")):
            return "pony_anime"
        return "pony_real"

    # NoobAI (Illustrious variant)
    if "noob" in low:
        return "noobai"

    # Illustrious
    if "illustrious" in low:
        return "illustrious"

    # SD3 / SD3.5
    if "sd3" in low or "sd_3" in low or "stable-diffusion-3" in low:
        return "sd3"

    # FLUX
    if "flux" in low or "chroma" in low:
        return "flux"

    # Fall back to architecture detection
    if checkpoint_name:
        try:
            from supremediffusion.models.sd_models import detect_model_family
            family = detect_model_family(checkpoint_name)
            if family in ("sd15", "sdxl", "flux"):
                return family
        except Exception:
            pass

    return "sdxl"


_FACE_IMAGE_EXTS = {"*.png", "*.jpg", "*.jpeg", "*.webp"}


def _collect_face_library(app_state) -> list[Path]:
    """Collect face images from both the config face_library_dir and the
    FaceLibraryTab's computed path (projects_root parent / library / faces)."""
    seen = set()
    results = []
    dirs = []
    # Config-specified face library dir
    face_dir = getattr(app_state.global_config, "face_library_dir", "")
    if face_dir:
        dirs.append(Path(face_dir))
    # FaceLibraryTab computed dir (projects_root parent / library / faces)
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


def _browse_character_image(app_state, parent_widget) -> str | None:
    """Show a dialog to pick an image from the character library.

    Returns the selected image path, or None if cancelled/empty.
    Scans characters_dir for character.json subdirs, shows a two-level
    picker: character list → image grid (base + poses).
    """
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QHBoxLayout,
        QListWidget,
        QListWidgetItem,
        QSplitter,
    )
    from PySide6.QtCore import QSize

    chars_dir = getattr(app_state.global_config, "characters_dir", "")
    if not chars_dir or not Path(chars_dir).is_dir():
        QMessageBox.information(parent_widget, "Character Library", "No characters saved yet.")
        return None

    # Scan for characters
    characters: list[tuple[str, Path]] = []  # (name, dir)
    for d in sorted(Path(chars_dir).iterdir()):
        if d.is_dir() and (d / "character.json").is_file():
            try:
                import json as _json
                meta = _json.loads((d / "character.json").read_text())
                name = meta.get("name", d.name)
            except Exception:
                name = d.name
            characters.append((name, d))

    if not characters:
        QMessageBox.information(parent_widget, "Character Library", "No characters saved yet.")
        return None

    # Build dialog
    dlg = QDialog(parent_widget)
    dlg.setWindowTitle("Select from Character Library")
    dlg.setMinimumSize(600, 400)

    root = QHBoxLayout(dlg)
    splitter = QSplitter()

    # Left: character list
    char_list = QListWidget()
    for name, d in characters:
        item = QListWidgetItem(name)
        base = d / "base.png"
        if base.is_file():
            pm = QPixmap(str(base))
            if not pm.isNull():
                item.setIcon(QIcon(pm.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
        item.setData(Qt.UserRole, str(d))
        char_list.addItem(item)
    splitter.addWidget(char_list)

    # Right: image grid
    img_list = QListWidget()
    img_list.setViewMode(QListWidget.IconMode)
    img_list.setIconSize(QSize(100, 100))
    img_list.setResizeMode(QListWidget.Adjust)
    img_list.setSpacing(6)
    img_list.setWrapping(True)
    splitter.addWidget(img_list)

    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)
    root.addWidget(splitter)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    root.addWidget(buttons)

    selected_path: list[str] = []  # mutable container for closure

    def _on_char_changed(current, _prev):
        img_list.clear()
        if not current:
            return
        char_dir = Path(current.data(Qt.UserRole))
        _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
        # Add base image first
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                pm = QPixmap(str(f))
                if pm.isNull():
                    continue
                item = QListWidgetItem()
                item.setIcon(QIcon(pm.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                item.setText(f.name)
                item.setData(Qt.UserRole, str(f))
                item.setToolTip(str(f))
                img_list.addItem(item)
        # Add poses
        poses_dir = char_dir / "poses"
        if poses_dir.is_dir():
            for f in sorted(poses_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                    pm = QPixmap(str(f))
                    if pm.isNull():
                        continue
                    item = QListWidgetItem()
                    item.setIcon(QIcon(pm.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
                    item.setText(f"pose: {f.name}")
                    item.setData(Qt.UserRole, str(f))
                    item.setToolTip(str(f))
                    img_list.addItem(item)

    char_list.currentItemChanged.connect(_on_char_changed)

    def _on_img_double_click(item):
        selected_path.clear()
        selected_path.append(item.data(Qt.UserRole))
        dlg.accept()

    img_list.itemDoubleClicked.connect(_on_img_double_click)

    def _on_accept():
        item = img_list.currentItem()
        if item:
            selected_path.clear()
            selected_path.append(item.data(Qt.UserRole))
        dlg.accept()

    buttons.accepted.connect(_on_accept)
    buttons.rejected.connect(dlg.reject)

    # Select first character
    if char_list.count() > 0:
        char_list.setCurrentRow(0)

    from PySide6.QtWidgets import QDialog as _QDlg
    if dlg.exec() == _QDlg.Accepted and selected_path:
        return selected_path[0]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 — Character Info
# ═══════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Prerequisites — collapsible model / package / toolkit downloader (Page 1)
# ---------------------------------------------------------------------------


class _GitCloneWorker(BaseWorker):
    """Shallow-clone a git repo into a destination dir on a background thread."""

    def __init__(self, url: str, dest: Path, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._dest = dest

    def do_work(self) -> str:
        self.status.emit(f"Cloning {self._url} …")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", self._url, str(self._dest)],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()
            raise RuntimeError(err[-500:] or "git clone failed")
        return str(self._dest)


def _prq_help(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color: #aaa; font-size: 12px;")
    return lbl


def _prq_subhdr(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #ddd; font-weight: bold; font-size: 12px; margin-top: 4px;")
    return lbl


class PrerequisitesSection(CollapsibleSection):
    """Collapsible panel that detects and downloads/installs everything the
    Character Creator needs: face-swap models, the ADetailer YOLOv8 face model,
    the Qwen-VL captioner, their Python packages, and the kohya / ai-toolkit
    LoRA-training repos. Each row shows a status (green ready / red required /
    amber optional) and a one-click action when missing.
    """

    def __init__(self, app_state, parent=None) -> None:
        super().__init__("Prerequisites", collapsed=True, parent=parent)
        self._app = app_state
        self._rows: dict[str, dict] = {}

        self.add_widget(_prq_help(
            "Download the models, Python packages, and training tools the "
            "Character Creator uses. <b>Green ✓</b> = ready, <b>red ✗</b> = "
            "required and missing, <b>amber ○</b> = optional (falls back "
            "gracefully). Large model downloads ask for confirmation first."
        ))

        # --- Models (registry-backed) -------------------------------------
        self.add_widget(_prq_subhdr("Models"))
        self._add_row(
            "face_swap_model", "Face / Body Swap (InsightFace + inswapper)",
            "Required for the face/body-swap step and the InsightFace ADetailer "
            "detector (~750 MB).",
            "Download", required=True,
            check=self._face_models_present,
            action=lambda: self._download_models("face_swap"),
        )
        self._add_row(
            "adetailer_model", "ADetailer face model (YOLOv8s)",
            "Sharper face detection for ADetailer face-restore (~22 MB). Falls "
            "back to InsightFace if absent. Downloading also offers to install "
            "the ultralytics library it needs.",
            "Download", required=False,
            check=lambda: not self._reg().get_missing_models("adetailer"),
            action=self._download_adetailer,
        )
        self._add_row(
            "qwen_vl_model", "Image captioning model (Qwen-VL)",
            "Used by the reference-image analyzer to auto-describe poses/outfits.",
            "Download", required=False,
            check=lambda: not self._reg().get_missing_models("qwen_vl"),
            action=lambda: self._download_models("qwen_vl"),
        )

        # --- Python packages (pip into the current venv) ------------------
        self.add_widget(_prq_subhdr("Python packages"))
        self._add_row(
            "face_swap_pip", "Face-swap libraries (insightface, gfpgan, …)",
            "insightface + onnxruntime-gpu + gfpgan + realesrgan.",
            "Install", required=True,
            check=lambda: _deps.is_installed("face_swap"),
            action=lambda: _deps.check_and_install("face_swap", self._dlg_parent()),
        )
        self._add_row(
            "yolo_pip", "YOLOv8 detector library (ultralytics)",
            "Only needed for the YOLOv8 ADetailer detector.",
            "Install", required=False,
            check=lambda: _deps.is_installed("adetailer_yolo"),
            action=lambda: _deps.check_and_install("adetailer_yolo", self._dlg_parent()),
        )
        self._add_row(
            "qwen_vl_pip", "Captioning libraries (qwen-vl-utils, bitsandbytes)",
            "Needed for the Qwen-VL reference analyzer.",
            "Install", required=False,
            check=lambda: _deps.is_installed("qwen_vl"),
            action=lambda: _deps.check_and_install("qwen_vl", self._dlg_parent()),
        )

        # --- LoRA training toolkits (git repos) ---------------------------
        self.add_widget(_prq_subhdr("LoRA training toolkits"))
        self.add_widget(_prq_help(
            "Cloning fetches the repo only. Each toolkit needs its own Python "
            "venv with its requirements installed (see its README) before LoRA "
            "training will run."
        ))
        self._add_row(
            "kohya", "kohya sd-scripts (SD 1.5 / SDXL / Pony / Illustrious)",
            "Cloned into third_party/sd-scripts. You must still create its "
            "Python venv and install its requirements (see its README) before "
            "image-LoRA training.",
            "Clone", required=False,
            check=self._kohya_present,
            action=lambda: self._clone(
                "https://github.com/kohya-ss/sd-scripts.git", "sd-scripts"),
        )
        self._add_row(
            "aitoolkit", "Ostris ai-toolkit (Flux / Z-Image / Wan / LTX)",
            "Cloned into third_party/ai-toolkit. You must still create its "
            "Python venv and install its requirements (see its README) before "
            "video / Flux / Z-Image LoRA training.",
            "Clone", required=False,
            check=self._aitoolkit_present,
            action=lambda: self._clone(
                "https://github.com/ostris/ai-toolkit.git", "ai-toolkit"),
        )

        # --- Re-check --------------------------------------------------------
        recheck_row = QHBoxLayout()
        recheck_row.addStretch()
        recheck = QPushButton("Re-check")
        recheck.setFixedWidth(110)
        recheck.clicked.connect(self.refresh)
        recheck_row.addWidget(recheck)
        self.add_layout(recheck_row)

        self.refresh()

    # -- helpers ----------------------------------------------------------

    def _reg(self):
        return self._app.model_registry

    def _dlg_parent(self) -> QWidget:
        return self.window() or self

    def _face_models_present(self) -> bool:
        """Accurate presence check for the face-swap models.

        The registry's ``config_key`` check returns True as soon as
        ``face_models_dir`` is a non-empty directory, which false-positives
        when (say) inswapper is present but buffalo_l is not. Probe the actual
        files instead — buffalo_l may be at ``face/buffalo_l`` or the
        InsightFace-native ``face/models/buffalo_l``.
        """
        cfg = self._app.global_config
        fmd = cfg.model_paths.get("face_models_dir", "") or str(
            Path(cfg.models_root) / "face")
        base = Path(fmd)
        buffalo = any(
            d.is_dir() and any(d.glob("*.onnx"))
            for d in (base / "buffalo_l", base / "models" / "buffalo_l")
        )
        inswapper = (base / "inswapper_128.onnx").is_file()
        return buffalo and inswapper

    def _download_adetailer(self) -> None:
        """Download the YOLOv8 model, then offer its required library —
        the model is useless without ultralytics."""
        self._download_models("adetailer")
        if not _deps.is_installed("adetailer_yolo"):
            _deps.check_and_install("adetailer_yolo", self._dlg_parent())

    @staticmethod
    def _kohya_present() -> bool:
        from sdqt.workers.lora_train import find_kohya_script
        return find_kohya_script() is not None

    @staticmethod
    def _aitoolkit_present() -> bool:
        from sdqt.workers.video_lora_train import find_ai_toolkit
        return find_ai_toolkit() is not None

    def _add_row(self, key, label, detail, btn_text, *, required, check, action):
        row = QHBoxLayout()
        row.setSpacing(6)
        status = QLabel("…")
        status.setFixedWidth(18)
        status.setAlignment(Qt.AlignCenter)
        name = QLabel(label)
        name.setToolTip(detail)
        btn = QPushButton(btn_text)
        btn.setFixedWidth(110)
        btn.setToolTip(detail)
        btn.clicked.connect(lambda _=False, k=key: self._do_action(k))
        row.addWidget(status)
        row.addWidget(name, 1)
        row.addWidget(btn)
        self.add_layout(row)
        self._rows[key] = {
            "status": status, "btn": btn, "check": check,
            "action": action, "required": required, "text": btn_text,
        }

    def _do_action(self, key: str) -> None:
        rec = self._rows[key]
        rec["btn"].setEnabled(False)
        try:
            rec["action"]()
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            logger.exception("Prerequisite action failed: %s", key)
            QMessageBox.warning(self._dlg_parent(), "Prerequisite", f"Failed:\n{exc}")
        self.refresh()

    def _download_models(self, feature: str) -> None:
        from sdqt.models.manager import check_and_prompt_download
        ok = check_and_prompt_download(feature, self._reg(), self._dlg_parent())
        if ok:
            try:
                self._app.sync_config_after_download()
            except Exception:
                logger.debug("config sync after download failed", exc_info=True)

    def _clone(self, url: str, name: str) -> None:
        # project root == sdqt/sequences/create_character.py → parent×3
        root = Path(__file__).resolve().parent.parent.parent
        dest = root / "third_party" / name
        if dest.exists():
            QMessageBox.information(
                self._dlg_parent(), "Already present",
                f"{name} already exists at:\n{dest}")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)

        dlg = QProgressDialog(f"Cloning {name}…", None, 0, 0, self._dlg_parent())
        dlg.setWindowTitle("Cloning repository")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()

        worker = _GitCloneWorker(url, dest, parent=self)
        holder = {"err": ""}
        worker.error.connect(lambda m: holder.__setitem__("err", m))
        loop = QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec()
        dlg.close()

        if holder["err"]:
            QMessageBox.warning(
                self._dlg_parent(), "Clone failed",
                f"Could not clone {name}:\n{holder['err']}\n\n"
                f"You can clone it manually into:\n{dest}")
        else:
            QMessageBox.information(
                self._dlg_parent(), "Cloned",
                f"Cloned {name} into third_party/.\n\n"
                "Next: create its Python venv and install its requirements "
                "(see the repo README) before training.")

    def refresh(self) -> None:
        for rec in self._rows.values():
            try:
                present = bool(rec["check"]())
            except Exception:
                logger.debug("prerequisite check failed", exc_info=True)
                present = False
            status, btn = rec["status"], rec["btn"]
            if present:
                status.setText("✓")
                status.setStyleSheet("color: #50e850; font-weight: bold;")
                btn.setEnabled(False)
                btn.setText("Installed")
            else:
                if rec["required"]:
                    status.setText("✗")
                    status.setStyleSheet("color: #e85050; font-weight: bold;")
                else:
                    status.setText("○")
                    status.setStyleSheet("color: #e8a33d; font-weight: bold;")
                btn.setEnabled(True)
                btn.setText(rec["text"])


class CharInfoPage(WizardPage):
    """Name, description, style, LoRA selection."""

    def __init__(self, char_state: CharacterState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = parent  # the wizard (passed as parent before stack reparenting)
        self._interrogate_worker = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Load an existing saved character to edit (any stage), then Save / Save as Copy.
        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load Saved Character…")
        self._load_btn.clicked.connect(self._on_load_character)
        load_row.addWidget(self._load_btn)
        load_row.addStretch()
        layout.addLayout(load_row)

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
        self._desc.textChanged.connect(self._on_desc_changed)
        self._lora_parse_pending: set[str] = set()  # stems already auto-activated
        layout.addWidget(self._desc)

        # Optional reference image + analyzers (same as the image modules).
        # When supplied, the analyzers append a caption/tags to the description,
        # and the reference can be used directly as the base on the next page.
        ref_group = QGroupBox("Reference Image (optional)")
        ref_layout = QVBoxLayout(ref_group)
        ref_hint = QLabel(
            "Supplying a reference makes it the base (the base-generation step is "
            "skipped). For best results use a clear, <b>full-body, front-facing</b> "
            "image in your chosen <b>style</b> and a neutral <b>pose</b>, then crop "
            "it to portrait. Use Analyze to read its details into the description."
        )
        ref_hint.setWordWrap(True)
        ref_hint.setStyleSheet("color: #aaa; font-size: 13px;")
        ref_layout.addWidget(ref_hint)
        self._ref_image = ImageDropWidget("Reference", thumb_height=160)
        self._ref_image.image_loaded.connect(self._on_ref_loaded)
        self._ref_image.image_cleared.connect(self._on_ref_cleared)
        ref_layout.addWidget(self._ref_image)
        crop_row = QHBoxLayout()
        crop_row.addStretch()
        self._crop_btn = QPushButton("Crop to Portrait")
        self._crop_btn.setToolTip("Crop the supplied image to the model's portrait aspect")
        self._crop_btn.clicked.connect(self._on_crop_reference)
        self._crop_btn.setEnabled(False)
        crop_row.addWidget(self._crop_btn)
        ref_layout.addLayout(crop_row)
        analyze_row = QHBoxLayout()
        analyze_row.addWidget(QLabel("Analyze:"))
        self._clip_btn = QPushButton("CLIP")
        self._clip_btn.setToolTip("Caption the reference image (BLIP) into the description")
        self._clip_btn.clicked.connect(lambda: self._analyze(BLIPInterrogateWorker))
        analyze_row.addWidget(self._clip_btn)
        self._wd14_btn = QPushButton("WD14")
        self._wd14_btn.setToolTip("Danbooru-style tags from the reference image")
        self._wd14_btn.clicked.connect(lambda: self._analyze(WD14TaggerWorker))
        analyze_row.addWidget(self._wd14_btn)
        self._qwen_btn = QPushButton("Qwen-VL")
        self._qwen_btn.setToolTip("Detailed natural-language caption (Qwen 2.5 VL)")
        self._qwen_btn.clicked.connect(lambda: self._analyze(QwenCaptionWorker))
        analyze_row.addWidget(self._qwen_btn)
        analyze_row.addStretch()
        ref_layout.addLayout(analyze_row)
        self._set_analyze_enabled(False)
        layout.addWidget(ref_group)

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

        # Prerequisites — download/install models, packages, training toolkits
        self._prereqs = PrerequisitesSection(self._app_state, parent=self)
        layout.addWidget(self._prereqs)

        layout.addStretch()

    def _on_load_character(self) -> None:
        """Load a saved character into the wizard for editing."""
        from PySide6.QtWidgets import QInputDialog
        base = _characters_dir(self._app_state)
        dirs = []
        if base.is_dir():
            dirs = [d for d in sorted(base.iterdir())
                    if d.is_dir() and (d / "character.json").is_file()]
        if not dirs:
            QMessageBox.information(self, "No characters", "No saved characters found.")
            return
        names = [d.name for d in dirs]
        name, ok = QInputDialog.getItem(
            self, "Load Character", "Choose a saved character to edit:", names, 0, False)
        if not ok or not name:
            return
        chosen = next(d for d in dirs if d.name == name)
        loaded = load_character(chosen)
        # Apply onto the SHARED state in place so every page sees it.
        self._cs.apply_meta(loaded.to_meta())
        for attr in ("selected_base", "reference_image", "face_source_path",
                     "body_source_path", "approved_poses", "pose_images"):
            setattr(self._cs, attr, getattr(loaded, attr))
        if self._wizard is not None:
            self._wizard._loaded_char_dir = str(chosen)
            bar = getattr(self._wizard, "_settings_bar", None)
            if bar is not None:
                bar.load_from_state()
            status = getattr(self._wizard, "_status", None)
            if status is not None:
                status.setText(
                    f"Loaded '{name}'. Edit on any page, then Save (update) or Save as Copy.")
        self.on_enter()  # refresh page-1 fields from the loaded state

    def _on_lora_insert(self, text: str) -> None:
        """Insert LoRA tags/triggers into the description field."""
        current = self._desc.toPlainText().strip()
        if current:
            self._desc.setPlainText(f"{current}, {text}")
        else:
            self._desc.setPlainText(text)

    # -- Reference image + analyzers --------------------------------------

    def _set_analyze_enabled(self, enabled: bool) -> None:
        for b in (self._clip_btn, self._wd14_btn, self._qwen_btn):
            b.setEnabled(enabled)
        self._crop_btn.setEnabled(enabled)

    @Slot(str)
    def _on_ref_loaded(self, path: str) -> None:
        self._cs.reference_image = path
        self._set_analyze_enabled(True)

    @Slot()
    def _on_ref_cleared(self) -> None:
        self._cs.reference_image = ""
        self._set_analyze_enabled(False)

    def _on_crop_reference(self) -> None:
        """Open the portrait crop dialog on the supplied reference image."""
        path = self._ref_image.image_path
        if not path or not Path(path).is_file():
            return
        pw, ph = _ar_defaults_for_model(self._cs.model_family or "sdxl")["portrait"]
        aspect = pw / ph if ph else 0.6667
        dlg = PortraitCropDialog(path, aspect, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.result_path:
            self._ref_image.load_image(dlg.result_path)  # emits image_loaded → updates state

    def _analyze(self, worker_cls) -> None:
        """Run an image-module interrogator on the reference image and append
        the result to the description."""
        path = self._ref_image.image_path
        if not path:
            return
        self._set_analyze_enabled(False)
        if worker_cls is QwenCaptionWorker:
            worker = worker_cls(path, self._app_state, parent=self)
        else:
            worker = worker_cls(path, parent=self)
        worker.finished_ok.connect(self._on_analyze_done)
        worker.error.connect(self._on_analyze_error)
        self._interrogate_worker = worker
        worker.start()

    @Slot(object)
    def _on_analyze_done(self, caption) -> None:
        self._set_analyze_enabled(True)
        text = (str(caption) if caption else "").strip()
        if not text:
            return
        cur = self._desc.toPlainText().strip()
        self._desc.setPlainText(f"{cur}, {text}" if cur else text)

    @Slot(str)
    def _on_analyze_error(self, msg: str) -> None:
        self._set_analyze_enabled(True)
        QMessageBox.warning(self, "Analyze failed", msg)

    @Slot()
    def _on_desc_changed(self) -> None:
        """Auto-activate LoRAs when user pastes ``<lora:name:weight>`` tags."""
        text = self._desc.toPlainText()
        for match in _LORA_TAG_RE.finditer(text):
            stem = match.group(1)
            if stem in self._lora_parse_pending:
                continue
            try:
                weight = float(match.group(2))
            except ValueError:
                weight = 1.0
            if self._lora_picker.activate_by_stem(stem, weight):
                self._lora_parse_pending.add(stem)
                logger.info("Auto-activated LoRA '%s' at weight %.2f from pasted prompt", stem, weight)

    def on_enter(self) -> None:
        self._name.setText(self._cs.name)
        self._desc.setPlainText(self._cs.description)
        if self._cs.reference_image and Path(self._cs.reference_image).is_file():
            self._ref_image.load_image(self._cs.reference_image)

    def on_leave(self) -> None:
        self._cs.name = self._name.text().strip()
        self._cs.description = self._desc.toPlainText().strip()
        self._cs.reference_image = self._ref_image.image_path or ""
        # A supplied reference IS the base — set it now so the (skipped) base
        # page doesn't strand the downstream pages without a base image.
        if self._cs.reference_image and Path(self._cs.reference_image).is_file():
            self._cs.selected_base = self._cs.reference_image
        styles = ["realism", "cartoon", "anime"]
        btn_id = self._style_group.checkedId()
        self._cs.style = styles[btn_id] if 0 <= btn_id < len(styles) else "realism"
        self._cs.selected_loras = self._lora_picker.get_activated()
        self._cs.lora_multipliers = self._lora_picker.get_multipliers()
        # Capture LoRA tags and trigger words for prompt injection later
        lora_tags, trigger_text = self._lora_picker.get_prompt_additions()
        self._cs.lora_prompt_tags = lora_tags
        self._cs.lora_trigger_words = trigger_text

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
    """Auto-generate positive + negative prompts using Qwen."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = wizard
        self._generated = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Explanation
        info = QLabel(
            "The prompt is seeded with the base-image requirements (style + full "
            "body, facing the viewer) plus your description. Edit it directly, or "
            "optionally click Enhance to expand it with Qwen."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #999; font-size: 12px; padding: 4px 0 8px 0;")
        layout.addWidget(info)

        # Source summary (what we're working from)
        self._source_label = QLabel()
        self._source_label.setWordWrap(True)
        self._source_label.setStyleSheet(
            "color: #8bf; font-size: 11px; background: #1a2a3a; "
            "border: 1px solid #2a4a6a; border-radius: 4px; padding: 6px;"
        )
        layout.addWidget(self._source_label)

        layout.addWidget(QLabel("<b>Positive Prompt</b>"))
        self._pos_edit = QPlainTextEdit()
        self._pos_edit.setMaximumHeight(140)
        self._pos_edit.setPlaceholderText("Generating from your description...")
        layout.addWidget(self._pos_edit)

        pos_row = QHBoxLayout()
        pos_row.addStretch()
        self._reroll_pos = QPushButton("Enhance Positive (Qwen)")
        self._reroll_pos.setFixedWidth(170)
        self._reroll_pos.clicked.connect(lambda: self._enhance("pos"))
        pos_row.addWidget(self._reroll_pos)
        layout.addLayout(pos_row)

        layout.addWidget(QLabel("<b>Negative Prompt</b>"))
        self._neg_edit = QPlainTextEdit()
        self._neg_edit.setMaximumHeight(100)
        self._neg_edit.setPlaceholderText("Will auto-generate after positive prompt...")
        layout.addWidget(self._neg_edit)

        neg_row = QHBoxLayout()
        neg_row.addStretch()
        self._reroll_neg = QPushButton("Enhance Negative (Qwen)")
        self._reroll_neg.setFixedWidth(170)
        self._reroll_neg.clicked.connect(lambda: self._enhance("neg"))
        neg_row.addWidget(self._reroll_neg)
        layout.addLayout(neg_row)

        layout.addStretch()

    def on_enter(self) -> None:
        # Show what we're building from
        desc = self._cs.description or "(no description)"
        loras = ", ".join(self._cs.selected_loras) if self._cs.selected_loras else "none"
        self._source_label.setText(
            f"Building from:  {desc[:120]}{'...' if len(desc) > 120 else ''}"
            f"\nLoRAs: {loras}  |  Style: {self._cs.style}"
        )

        # Seed the positive prompt with the base-image requirements prepended:
        # style medium (photo/anime/cartoon, from the page-1 style choice) +
        # full-body, facing-viewer framing, then the description. Qwen is NOT
        # auto-run — it's the optional Enhance button.
        if not self._pos_edit.toPlainText().strip():
            medium = _STYLE_MEDIUM.get(self._cs.style, "photo")
            prefix = f"{medium}, solo, full body, facing the viewer, head to toe"
            base = (self._cs.positive_prompt or self._cs.description or "").strip()
            self._pos_edit.setPlainText(f"{prefix}, {base}" if base else prefix)

    def _enhance(self, which: str) -> None:
        desc = self._cs.description
        if not desc:
            self._wizard._status.setText("No description to enhance — type one on the previous page")
            return
        # Include LoRA and style context in the enhancement request
        enhance_text = desc
        if self._cs.selected_loras:
            lora_names = ", ".join(Path(l).stem for l in self._cs.selected_loras)
            enhance_text += f"\nActive LoRAs: {lora_names}"
        if self._cs.style:
            enhance_text += f"\nStyle: {self._cs.style}"

        # Detect correct prompt style from the selected checkpoint
        prompt_style = _detect_prompt_style(self._cs.checkpoint, self._app_state)
        logger.info("Prompt style detected as '%s' for checkpoint '%s'",
                     prompt_style, self._cs.checkpoint)

        worker = PromptEnhanceWorker(
            text=enhance_text,
            style=prompt_style,
            which=which,
            state=self._app_state,
            parent=self,
        )
        if which == "pos":
            self._wizard._run_worker(worker, on_done=self._on_pos_done, on_error=self._on_error)
        else:
            self._wizard._run_worker(worker, on_done=self._on_neg_done, on_error=self._on_error)

    @Slot(object)
    def _on_pos_done(self, text) -> None:
        self._pos_edit.setPlainText(str(text))
        self._cs.positive_prompt = str(text)
        self._generated = True
        # Auto-generate negative after positive
        if not self._neg_edit.toPlainText().strip():
            self._enhance("neg")

    @Slot(object)
    def _on_neg_done(self, text) -> None:
        self._neg_edit.setPlainText(str(text))
        self._cs.negative_prompt = str(text)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Qwen enhance failed: {msg}")
        # If positive prompt is still empty, keep the raw description as fallback
        if not self._pos_edit.toPlainText().strip() and self._cs.description:
            self._pos_edit.setPlainText(self._cs.description)

    def on_leave(self) -> None:
        self._cs.positive_prompt = self._pos_edit.toPlainText().strip()
        self._cs.negative_prompt = self._neg_edit.toPlainText().strip()
        # Free Qwen VRAM
        self._app_state.unload_qwen()

    def validate(self) -> bool:
        # Save fields without unloading (on_leave called by wizard after validate)
        self._cs.positive_prompt = self._pos_edit.toPlainText().strip()
        self._cs.negative_prompt = self._neg_edit.toPlainText().strip()
        if not self._cs.positive_prompt:
            QMessageBox.warning(self, "Missing", "Please generate or enter a positive prompt.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Base Image Generation
# ═══════════════════════════════════════════════════════════════════════════


class BaseGenPage(WizardPage):
    """Generate 4 candidate base images; user picks one."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = wizard
        self._has_generated = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Generation settings now live in the wizard's persistent footer
        # (GenerationSettingsBar), shared by every page.
        self._gen_btn = QPushButton("Generate Base Images")
        self._gen_btn.setFixedWidth(200)
        self._gen_btn.clicked.connect(self._generate)
        layout.addWidget(self._gen_btn)

        self._hint = QLabel(
            "Pick a base image. If you supplied a reference image it appears "
            "first — select it to build the character from your reference, or "
            "generate new candidates."
        )
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #aaa; font-size: 13px;")
        layout.addWidget(self._hint)

        self._gallery = ImageGalleryWidget("Base Images")
        self._gallery.image_selected.connect(self._on_selected)
        layout.addWidget(self._gallery, 1)

    def can_skip(self) -> bool:
        # When a reference image is supplied it IS the base — skip generation
        # and jump straight to the optional Face/Body swap page.
        ref = self._cs.reference_image
        return bool(ref and Path(ref).is_file())

    def on_enter(self) -> None:
        # Show the optional reference image first (so it can be used directly as
        # the base), then any candidates from a previous generation.
        display: list[str] = []
        ref = self._cs.reference_image
        if ref and Path(ref).is_file():
            display.append(ref)
        display.extend(p for p in self._cs.base_images if p not in display)
        if display:
            self._gallery.load_images(display)
            if not self._cs.selected_base and ref and Path(ref).is_file():
                self._cs.selected_base = ref

    def _generate(self) -> None:
        self._gen_btn.setText("Regenerate")
        self._has_generated = True
        strategy = _strategy_for(self._cs.model_family)

        cfg = _load_project_config(self._app_state)
        # The positive prompt (seeded on page 2) already carries the base-image
        # framing (style + full body, facing the viewer). Append LoRA trigger
        # words (real tokens); skip <lora:> tags (diffusers ignores them).
        prompt_parts = [_strip_lora_tags(self._cs.positive_prompt)]
        if self._cs.lora_trigger_words:
            prompt_parts.append(self._cs.lora_trigger_words)
        prompt = ", ".join(p for p in prompt_parts if p)
        base_neg = self._cs.negative_prompt or ""
        negative = f"{base_neg}, {_POSE_NEGATIVE_EXTRA}" if base_neg else _POSE_NEGATIVE_EXTRA

        _apply_family_gen_cfg(
            cfg, strategy, self._cs,
            prompt=prompt, negative=negative,
            width=self._cs.width, height=self._cs.height, batch_count=4,
        )
        cfg.save(_project_path(self._app_state))
        _run_family_txt2img(
            self._wizard, self._app_state, cfg, strategy,
            on_done=self._on_images, on_error=self._on_error,
        )

    @Slot(object)
    def _on_images(self, paths) -> None:
        self._cs.base_images = list(paths) if paths else []
        self._gallery.load_images(self._cs.base_images)
        if self._cs.base_images:
            self._cs.selected_base = self._cs.base_images[0]

    @Slot(str)
    def _on_selected(self, path: str) -> None:
        self._cs.selected_base = path

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Generation error: {msg}")

    def validate(self) -> bool:
        if not self._cs.selected_base or not Path(self._cs.selected_base).is_file():
            QMessageBox.warning(self, "Missing", "Please generate and select a base image.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Face / Body Swap (optional, never auto-skipped)
# ═══════════════════════════════════════════════════════════════════════════


class FaceBodySwapPage(WizardPage):
    """Optional face and body swap on the selected base image."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
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

        # Face swap
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
        from PySide6.QtWidgets import QDoubleSpinBox
        self._enhancer_strength = QDoubleSpinBox()
        self._enhancer_strength.setRange(0.0, 1.0)
        self._enhancer_strength.setValue(0.5)
        self._enhancer_strength.setSingleStep(0.1)
        self._enhancer_strength.setFixedWidth(80)
        self._enhancer_strength.setEnabled(False)
        enhance_row.addWidget(self._enhancer_strength)
        enhance_row.addWidget(QLabel("Blend:"))
        self._face_blend = QDoubleSpinBox()
        self._face_blend.setRange(0.0, 1.0)
        self._face_blend.setSingleStep(0.05)
        self._face_blend.setDecimals(2)
        self._face_blend.setValue(float(self._cs.face_blend_ratio))
        self._face_blend.setFixedWidth(80)
        self._face_blend.setToolTip(
            "How much of the restored/enhanced face to blend over the raw swap. "
            "0 = raw swap, 1 = full enhancement."
        )
        enhance_row.addWidget(self._face_blend)
        enhance_row.addStretch()
        face_layout.addLayout(enhance_row)

        self._run_face_btn = QPushButton("Run Face Swap")
        self._run_face_btn.clicked.connect(self._on_run_face)
        face_layout.addWidget(self._run_face_btn)

        layout.addWidget(face_group)

        # Body swap
        body_group = QGroupBox("Body Swap")
        body_layout = QVBoxLayout(body_group)
        self._body_cb = QCheckBox("Apply Body Swap (ControlNet OpenPose + IP-Adapter)")
        self._body_cb.toggled.connect(self._on_body_toggled)
        body_layout.addWidget(self._body_cb)

        body_row = QHBoxLayout()
        body_row.addWidget(QLabel("Source Body:"))
        self._body_path = QLineEdit()
        self._body_path.setReadOnly(True)
        body_row.addWidget(self._body_path, 1)
        self._body_browse = QPushButton("Browse")
        self._body_browse.setFixedWidth(75)
        self._body_browse.clicked.connect(self._browse_body)
        body_row.addWidget(self._body_browse)
        self._body_char_btn = QPushButton("Character Library")
        self._body_char_btn.setFixedWidth(120)
        self._body_char_btn.clicked.connect(self._browse_body_from_characters)
        body_row.addWidget(self._body_char_btn)
        body_layout.addLayout(body_row)

        # Body-swap fine-tune controls
        bf1 = QHBoxLayout()
        bf1.setSpacing(6)
        bf1.addWidget(QLabel("Identity:"))
        self._body_ip = QDoubleSpinBox()
        self._body_ip.setRange(0.0, 1.0)
        self._body_ip.setSingleStep(0.05)
        self._body_ip.setDecimals(2)
        self._body_ip.setValue(self._cs.body_ip_scale)
        self._body_ip.setToolTip("IP-Adapter strength — how strongly the body source's identity is applied.")
        bf1.addWidget(self._body_ip)
        bf1.addWidget(QLabel("Denoise:"))
        self._body_denoise = QDoubleSpinBox()
        self._body_denoise.setRange(0.0, 1.0)
        self._body_denoise.setSingleStep(0.05)
        self._body_denoise.setDecimals(2)
        self._body_denoise.setValue(self._cs.body_denoise)
        self._body_denoise.setToolTip("Inpaint denoise — higher repaints more of the body region.")
        bf1.addWidget(self._body_denoise)
        bf1.addStretch()
        body_layout.addLayout(bf1)

        bf2 = QHBoxLayout()
        bf2.setSpacing(6)
        bf2.addWidget(QLabel("CFG:"))
        self._body_cfg = QDoubleSpinBox()
        self._body_cfg.setRange(1.0, 30.0)
        self._body_cfg.setSingleStep(0.5)
        self._body_cfg.setDecimals(1)
        self._body_cfg.setValue(self._cs.body_cfg)
        bf2.addWidget(self._body_cfg)
        bf2.addWidget(QLabel("Pose strength:"))
        self._body_cn = QDoubleSpinBox()
        self._body_cn.setRange(0.0, 2.0)
        self._body_cn.setSingleStep(0.05)
        self._body_cn.setDecimals(2)
        self._body_cn.setValue(self._cs.body_cn_strength)
        self._body_cn.setToolTip("ControlNet OpenPose strength — how tightly the original pose is held.")
        bf2.addWidget(self._body_cn)
        bf2.addStretch()
        body_layout.addLayout(bf2)

        self._run_body_btn = QPushButton("Run Body Swap")
        self._run_body_btn.clicked.connect(self._on_run_body)
        body_layout.addWidget(self._run_body_btn)

        layout.addWidget(body_group)

        layout.addStretch()

    def on_enter(self) -> None:
        # Reset swap-completed flag when user revisits this page
        self._swap_completed = False
        self._before_path = ""
        if self._cs.selected_base and Path(self._cs.selected_base).is_file():
            pm = QPixmap(self._cs.selected_base).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._preview.setPixmap(pm)
        self._face_cb.setChecked(self._cs.face_swap_enabled)
        self._body_cb.setChecked(self._cs.body_swap_enabled)
        if self._cs.face_source_path:
            self._face_path.setText(self._cs.face_source_path)
        if self._cs.body_source_path:
            self._body_path.setText(self._cs.body_source_path)
        # Controls are usable independent of the checkboxes — the Run buttons
        # trigger each swap individually.
        for w in (
            self._face_browse, self._face_lib_btn, self._face_char_btn,
            self._enhancer_combo, self._enhancer_strength,
            self._body_browse, self._body_char_btn,
        ):
            w.setEnabled(True)
        # Restore body-swap fine-tune values + face blend
        self._body_ip.setValue(float(self._cs.body_ip_scale))
        self._body_denoise.setValue(float(self._cs.body_denoise))
        self._body_cfg.setValue(float(self._cs.body_cfg))
        self._body_cn.setValue(float(self._cs.body_cn_strength))
        self._face_blend.setValue(float(self._cs.face_blend_ratio))

    def on_leave(self) -> None:
        # Persist swap tuning so it survives Back/Next navigation.
        self._apply_body_controls_to_state()
        self._cs.face_blend_ratio = self._face_blend.value()
        enhancer = self._enhancer_combo.currentText()
        self._cs.face_enhancer = enhancer if enhancer != "None" else ""
        self._cs.face_enhancer_strength = self._enhancer_strength.value()

    def _on_face_toggled(self, checked: bool) -> None:
        self._cs.face_swap_enabled = checked
        self._swap_completed = False

    def _on_body_toggled(self, checked: bool) -> None:
        self._cs.body_swap_enabled = checked
        self._swap_completed = False

    def _apply_body_controls_to_state(self) -> None:
        self._cs.body_ip_scale = self._body_ip.value()
        self._cs.body_denoise = self._body_denoise.value()
        self._cs.body_cfg = self._body_cfg.value()
        self._cs.body_cn_strength = self._body_cn.value()

    def _browse_face(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Face Source", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._face_path.setText(path)
            self._cs.face_source_path = path

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
            self._cs.face_source_path = dialog.selected_path

    def _browse_face_from_characters(self) -> None:
        path = _browse_character_image(self._app_state, self)
        if path:
            self._face_path.setText(path)
            self._cs.face_source_path = path

    def _browse_body(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Body Source", "", "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if path:
            self._body_path.setText(path)
            self._cs.body_source_path = path

    def _browse_body_from_characters(self) -> None:
        path = _browse_character_image(self._app_state, self)
        if path:
            self._body_path.setText(path)
            self._cs.body_source_path = path

    def validate(self) -> bool:
        # Swaps are now triggered individually via the Run buttons and applied
        # through the before/after popup, so advancing is always allowed.
        self._apply_body_controls_to_state()
        return True

    # -- Face swap (individual) -------------------------------------------

    def _on_run_face(self) -> None:
        if not self._cs.face_source_path:
            QMessageBox.warning(self, "Missing", "Select a face source image first.")
            return
        self._before_path = self._cs.selected_base
        self._run_face_swap()

    def _run_face_swap(self) -> None:
        from sdqt.workers.image import FaceSwapWorker

        cfg = _load_project_config(self._app_state)
        enhancer = self._enhancer_combo.currentText()
        self._cs.face_enhancer = enhancer if enhancer != "None" else ""
        self._cs.face_enhancer_strength = self._enhancer_strength.value()
        self._cs.face_blend_ratio = self._face_blend.value()
        cfg.faceswap_enhancer = self._cs.face_enhancer
        cfg.faceswap_enhancer_strength = self._cs.face_enhancer_strength
        cfg.faceswap_blend_ratio = self._cs.face_blend_ratio
        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")

        target = self._before_path or self._cs.selected_base
        self._run_face_btn.setEnabled(False)
        self._run_face_btn.setText("Swapping…")
        worker = FaceSwapWorker(
            source_path=self._cs.face_source_path,
            target_path=target,
            project_config=cfg,
            models_dir=models_dir,
            enhancer_strength=self._cs.face_enhancer_strength,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_face_done, on_error=self._on_swap_error)

    @Slot(object)
    def _on_face_done(self, result_path) -> None:
        self._run_face_btn.setEnabled(True)
        self._run_face_btn.setText("Run Face Swap")
        self._cs.face_result_path = str(result_path)
        self._show_before_after(self._before_path, str(result_path), "face")

    # -- Body swap (individual, real ControlNet + IP-Adapter) -------------

    def _on_run_body(self) -> None:
        if not self._cs.body_source_path:
            QMessageBox.warning(self, "Missing", "Select a body source image first.")
            return
        self._apply_body_controls_to_state()
        self._before_path = self._cs.selected_base
        self._run_body_swap()

    def _run_body_swap(self) -> None:
        if self._app_state.img_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_sd_pipelines", parent=self)
            self._wizard._run_worker(loader, on_done=lambda _: self._segment_then_body())
        else:
            self._segment_then_body()

    def _segment_then_body(self) -> None:
        """Segment the person in the target, then run the body double."""
        from sdqt.workers.image import AutoSegmentWorker

        target = self._before_path or self._cs.selected_base
        if not target or not Path(target).is_file():
            self._on_swap_error("No base image to swap onto.")
            return
        self._run_body_btn.setEnabled(False)
        self._run_body_btn.setText("Segmenting…")
        models_root = self._app_state.global_config.models_root
        worker = AutoSegmentWorker(target, models_root, parent=self)
        self._wizard._run_worker(
            worker,
            on_done=lambda mask: self._do_body_swap(target, str(mask)),
            on_error=self._on_swap_error,
        )

    def _do_body_swap(self, target: str, mask: str) -> None:
        from sdqt.workers.image import BodyDoubleWorker

        self._run_body_btn.setText("Swapping…")
        cfg = _load_project_config(self._app_state)
        if self._cs.checkpoint:
            cfg.bodydouble_checkpoint = self._cs.checkpoint
        cfg.bodydouble_controlnet_type = "openpose"
        cfg.bodydouble_controlnet_strength = float(self._cs.body_cn_strength)
        cfg.bodydouble_ip_adapter_variant = "faceid_plus"
        cfg.bodydouble_ip_adapter_scale = float(self._cs.body_ip_scale)
        cfg.bodydouble_denoising_strength = float(self._cs.body_denoise)
        cfg.bodydouble_cfg_scale = float(self._cs.body_cfg)
        cfg.bodydouble_steps = int(self._cs.steps)
        if self._cs.sampler:
            cfg.bodydouble_sampler = self._cs.sampler
        if self._cs.scheduler:
            cfg.bodydouble_scheduler = self._cs.scheduler
        cfg.bodydouble_prompt = self._cs.positive_prompt
        cfg.bodydouble_negative_prompt = self._cs.negative_prompt
        cfg.bodydouble_seed = int(self._cs.seed)

        worker = BodyDoubleWorker(
            pipeline=self._app_state.img_pipeline,
            project_name=_project_name(self._app_state),
            target_image=target,
            mask=mask,
            source_person=self._cs.body_source_path,
            project_config=cfg,
            parent=self,
        )
        self._wizard._run_worker(
            worker,
            on_done=lambda paths: self._on_body_done(paths, mask),
            on_error=self._on_swap_error,
        )

    @Slot(object)
    def _on_body_done(self, paths, mask: str = "") -> None:
        self._run_body_btn.setEnabled(True)
        self._run_body_btn.setText("Run Body Swap")
        # Clean up the temp segmentation mask
        if mask:
            try:
                os.unlink(mask)
            except OSError:
                pass
        result = ""
        if paths:
            result = str(paths[0]) if isinstance(paths, list) else str(paths)
            self._cs.body_result_path = result
        self._show_before_after(self._before_path, result, "body")

    # -- Before/After preview --------------------------------------------

    def _show_before_after(self, before: str, after: str, swap_type: str) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{swap_type.title()} Swap — Before / After")
        dlg.setMinimumSize(740, 480)
        v = QVBoxLayout(dlg)

        row = QHBoxLayout()
        for title, path in (("Before", before), ("After", after)):
            col = QVBoxLayout()
            cap = QLabel(title)
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("font-weight: bold;")
            col.addWidget(cap)
            img = QLabel()
            img.setAlignment(Qt.AlignCenter)
            img.setMinimumSize(340, 400)
            img.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
            if path and Path(path).is_file():
                img.setPixmap(QPixmap(path).scaled(
                    340, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img.setText("(none)")
            col.addWidget(img, 1)
            row.addLayout(col)
        v.addLayout(row, 1)

        btn_box = QDialogButtonBox()
        accept_btn = btn_box.addButton("Accept", QDialogButtonBox.AcceptRole)
        retry_btn = btn_box.addButton("Retry", QDialogButtonBox.ActionRole)
        btn_box.addButton("Discard", QDialogButtonBox.RejectRole)
        v.addWidget(btn_box)

        choice = {"action": "discard"}
        accept_btn.clicked.connect(lambda: (choice.update(action="accept"), dlg.accept()))
        retry_btn.clicked.connect(lambda: (choice.update(action="retry"), dlg.accept()))
        btn_box.rejected.connect(dlg.reject)
        dlg.exec()

        if choice["action"] == "accept" and after and Path(after).is_file():
            self._cs.selected_base = after
            self._swap_completed = True
            self._preview.setPixmap(QPixmap(after).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif choice["action"] == "retry":
            if swap_type == "face":
                self._run_face_swap()
            else:
                self._run_body_swap()
        # discard = keep the current base, stay on page

    @Slot(str)
    def _on_swap_error(self, msg: str) -> None:
        self._run_face_btn.setEnabled(True)
        self._run_face_btn.setText("Run Face Swap")
        self._run_body_btn.setEnabled(True)
        self._run_body_btn.setText("Run Body Swap")
        QMessageBox.warning(self, "Swap Error", msg)


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Pose Variants
# ═══════════════════════════════════════════════════════════════════════════


class PoseVariantsPage(WizardPage):
    """Generate 8 pose variants and let the user approve/reroll them."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = wizard
        self._pose_paths: list[str] = [""] * len(POSES)
        self._pose_checks: list[QCheckBox] = []
        self._generating = False
        self._current_pose_idx = 0
        self._reroll_indices: list[int] = []
        self._current_reroll: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        btn_row = QHBoxLayout()
        self._begin_btn = QPushButton("Generate Pose Variants")
        self._begin_btn.setFixedWidth(200)
        self._begin_btn.clicked.connect(self._begin_generation)
        btn_row.addWidget(self._begin_btn)

        self._reroll_btn = QPushButton("Reroll Unchecked")
        self._reroll_btn.setFixedWidth(140)
        self._reroll_btn.clicked.connect(self._reroll_unchecked)
        self._reroll_btn.setEnabled(False)
        btn_row.addWidget(self._reroll_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Apply the canonical LOOK (the selected base — a generated base OR the
        # supplied reference image) to every generated pose via IP-Adapter.
        look_row = QHBoxLayout()
        look_row.addWidget(QLabel("Apply canonical look to poses:"))
        self._look_strength = QDoubleSpinBox()
        self._look_strength.setRange(0.0, 1.0)
        self._look_strength.setSingleStep(0.05)
        self._look_strength.setDecimals(2)
        self._look_strength.setValue(float(self._cs.ref_look_strength))
        self._look_strength.setToolTip(
            "IP-Adapter strength applying the selected base / reference image's "
            "look to each pose. 0 = off. Combines with the face-swap step."
        )
        look_row.addWidget(self._look_strength)
        look_row.addStretch()
        layout.addLayout(look_row)

        # Apply the canonical BODY to each pose (model-agnostic post-process:
        # segment + ControlNet pose + IP-Adapter). Default ON — this is how
        # flux/z-image poses get a consistent body, not just SD/SDXL.
        body_row = QHBoxLayout()
        self._apply_body_cb = QCheckBox("Apply canonical body to every pose (recommended)")
        self._apply_body_cb.setChecked(bool(self._cs.apply_body_to_poses))
        self._apply_body_cb.setToolTip(
            "Body-swap the canonical body onto each pose so the build/identity "
            "stays consistent across poses for ANY model family. Adds a pass per "
            "pose (slower)."
        )
        self._apply_body_cb.toggled.connect(
            lambda v: setattr(self._cs, "apply_body_to_poses", bool(v)))
        body_row.addWidget(self._apply_body_cb)
        body_row.addStretch()
        layout.addLayout(body_row)

        # Grid of thumbnails with checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._grid = QGridLayout(container)
        self._grid.setSpacing(8)

        self._thumb_labels: list[QLabel] = []
        self._pose_checks = []
        self._pose_ar_combos: list[QComboBox] = []
        for i, spec in enumerate(POSES):
            col = i % 4
            row = (i // 4) * 3  # 3 rows per visual row (thumb + checkbox + AR)

            thumb = QLabel()
            thumb.setFixedSize(160, 160)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("background: #2a2a2a; border: 1px solid #444;")
            thumb.setCursor(Qt.CursorShape.PointingHandCursor)
            thumb.setToolTip(spec.description)
            thumb.setProperty("pose_idx", i)
            thumb.installEventFilter(self)
            self._thumb_labels.append(thumb)
            self._grid.addWidget(thumb, row, col)

            cb = QCheckBox(f"{_DISTANCE_TAG.get(spec.distance, '')[:10]}: {spec.description[:24]}")
            cb.setChecked(True)
            self._pose_checks.append(cb)
            self._grid.addWidget(cb, row + 1, col)

            # Per-pose aspect-ratio dropdown — default chosen by pose type, but
            # user-overridable before generating. Populated per model family.
            ar = QComboBox()
            ar.setToolTip("Aspect ratio for this pose (model-native resolution buckets)")
            self._pose_ar_combos.append(ar)
            self._grid.addWidget(ar, row + 2, col)

        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

    def _populate_ar_combos(self) -> None:
        """Fill each pose's AR dropdown with the selected model family's buckets,
        defaulting by pose orientation (laying/reclining → landscape)."""
        family = self._cs.model_family or "sdxl"
        buckets = _buckets_for_model(family)
        defaults = _ar_defaults_for_model(family)
        for i, ar in enumerate(self._pose_ar_combos):
            prev = ar.currentData()  # preserve a prior user choice if still valid
            ar.blockSignals(True)
            ar.clear()
            for label, w, h in buckets:
                ar.addItem(label, (w, h))
            want = defaults["landscape" if POSES[i].orientation == "landscape" else "portrait"]
            target = prev if prev in [(w, h) for _, w, h in buckets] else want
            idx = next((k for k in range(ar.count()) if ar.itemData(k) == target), 0)
            ar.setCurrentIndex(idx)
            ar.blockSignals(False)

    def eventFilter(self, obj, event) -> bool:
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.MouseButtonDblClick:
            idx = obj.property("pose_idx")
            if idx is not None and 0 <= idx < len(self._pose_paths):
                path = self._pose_paths[idx]
                if path and Path(path).is_file():
                    from sdqt.widgets.image_lightbox import ImageLightbox
                    valid = [p for p in self._pose_paths if p and Path(p).is_file()]
                    lightbox = ImageLightbox(path, image_paths=valid, parent=self.window())
                    lightbox.exec()
                    return True
        return super().eventFilter(obj, event)

    def on_enter(self) -> None:
        # Populate per-pose AR dropdowns for the selected model family.
        self._populate_ar_combos()
        # Restore previous results if any
        if self._cs.pose_images:
            self._pose_paths = list(self._cs.pose_images)
            for i, p in enumerate(self._pose_paths):
                if p and Path(p).is_file() and i < len(self._thumb_labels):
                    pm = QPixmap(p).scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self._thumb_labels[i].setPixmap(pm)

    def _begin_generation(self) -> None:
        self._begin_btn.setText("Regenerating...")
        self._begin_btn.setEnabled(False)
        self._reroll_btn.setEnabled(False)
        self._current_pose_idx = 0
        self._generate_next_pose()

    def _reroll_unchecked(self) -> None:
        # Find unchecked poses to regenerate
        self._reroll_indices = [
            i for i, cb in enumerate(self._pose_checks) if not cb.isChecked()
        ]
        if not self._reroll_indices:
            return
        self._reroll_btn.setEnabled(False)
        self._current_reroll = 0
        self._reroll_next()

    def _reroll_next(self) -> None:
        if self._current_reroll >= len(self._reroll_indices):
            self._reroll_btn.setEnabled(True)
            return
        idx = self._reroll_indices[self._current_reroll]
        self._generate_pose(idx, callback=self._on_reroll_done)

    def _on_reroll_done(self, paths, idx) -> None:
        if paths:
            p = paths[0] if isinstance(paths, list) else str(paths)
            self._pose_paths[idx] = p
            self._postprocess_pose(idx, self._finish_reroll_pose)
            return
        self._current_reroll += 1
        self._reroll_next()

    def _finish_reroll_pose(self, result_path, idx) -> None:
        """Update thumbnail after face swap on a rerolled pose."""
        if result_path and Path(result_path).is_file():
            self._pose_paths[idx] = result_path
            pm = QPixmap(result_path).scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._thumb_labels[idx].setPixmap(pm)
        self._pose_checks[idx].setChecked(True)
        self._current_reroll += 1
        self._reroll_next()

    def _generate_next_pose(self) -> None:
        if self._current_pose_idx >= len(POSES):
            self._begin_btn.setText("Regenerate All")
            self._begin_btn.setEnabled(True)
            self._reroll_btn.setEnabled(True)
            self._cs.pose_images = list(self._pose_paths)
            return
        self._generate_pose(self._current_pose_idx, callback=self._on_seq_done)

    def _generate_pose(self, idx: int, callback=None) -> None:
        strategy = _strategy_for(self._cs.model_family)
        pipeline = strategy.get_pipeline(self._app_state)
        if pipeline is None:
            loader = PipelineLoadWorker(self._app_state, strategy.load_method, parent=self)
            self._wizard._run_worker(
                loader,
                on_done=lambda _: self._generate_pose(idx, callback),
                on_error=self._on_pose_error,
            )
            return

        spec = POSES[idx]
        cfg = _load_project_config(self._app_state)
        # Pose framing leads (distance + angle tag + the pose text), then the
        # character prompt + LoRA trigger words.
        parts = [
            _DISTANCE_TAG.get(spec.distance, ""), spec.description,
            _strip_lora_tags(self._cs.positive_prompt),
        ]
        if self._cs.lora_trigger_words:
            parts.append(self._cs.lora_trigger_words)
        pose_prompt = ", ".join(p for p in parts if p)
        # Distance-conditional negative: don't ban close-ups on close/medium poses.
        negative = _pose_negative_for(spec.distance, self._cs.negative_prompt)

        # Per-pose aspect ratio from its dropdown (model-native bucket).
        ar = self._pose_ar_combos[idx].currentData() if idx < len(self._pose_ar_combos) else None
        pw, ph = ar if ar else (self._cs.width, self._cs.height)

        _apply_family_gen_cfg(
            cfg, strategy, self._cs,
            prompt=pose_prompt, negative=negative,
            width=int(pw), height=int(ph), batch_count=1, seed=-1,  # random per pose
        )

        # Apply the canonical look via IP-Adapter (SD/SDXL only — the flux/zimage
        # txt2img workers don't consume it). Pose comes from the prompt; the look
        # comes from the canonical base. 0 = off.
        look = float(self._look_strength.value())
        self._cs.ref_look_strength = look
        canonical = self._cs.selected_base
        if (strategy.family in ("sd15", "sdxl") and canonical
                and Path(canonical).is_file() and look > 0):
            cfg.img_ipadapter_path = canonical
            cfg.img_ipadapter_scale = look
            cfg.img_ipadapter_variant = "plus"

        captured_idx = idx

        def on_done(paths):
            if callback:
                callback(paths, captured_idx)

        worker_cls = strategy.resolve_worker("txt2img")
        worker = worker_cls(
            pipeline=pipeline,
            project_name=_project_name(self._app_state),
            project_config=cfg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=on_done, on_error=self._on_pose_error)

    def _postprocess_pose(self, idx: int, finish_cb) -> None:
        """Apply the canonical body (if enabled) then face (if enabled) to a
        pose, then call ``finish_cb(result_path, idx)``. Order: body → face so
        the face lands on the swapped body."""
        def do_face(_r=None, _i=None):
            if self._cs.face_swap_enabled and (self._cs.selected_base or self._cs.face_source_path):
                self._run_pose_face_swap(idx, callback=finish_cb)
            else:
                finish_cb(self._pose_paths[idx], idx)
        if (self._cs.apply_body_to_poses and self._cs.selected_base
                and Path(self._cs.selected_base).is_file()):
            self._run_pose_body_swap(idx, do_face)
        else:
            do_face()

    def _run_pose_body_swap(self, idx: int, callback) -> None:
        """Segment the pose, then body-double the canonical body onto it."""
        from sdqt.workers.image import AutoSegmentWorker
        target = self._pose_paths[idx]
        if not target or not Path(target).is_file():
            callback(target, idx)
            return
        models_root = self._app_state.global_config.models_root
        seg = AutoSegmentWorker(target, models_root, parent=self)
        self._wizard._run_worker(
            seg,
            on_done=lambda mask: self._do_pose_body_swap(idx, target, str(mask), callback),
            on_error=lambda _m: callback(self._pose_paths[idx], idx),
        )

    def _do_pose_body_swap(self, idx: int, target: str, mask: str, callback) -> None:
        from sdqt.workers.image import BodyDoubleWorker
        # body_double is an SD ControlNet + IP-Adapter inpaint — needs the SD
        # pipeline (model-agnostic w.r.t. how the pose was generated).
        if self._app_state.img_pipeline is None:
            loader = PipelineLoadWorker(self._app_state, "load_sd_pipelines", parent=self)
            self._wizard._run_worker(
                loader,
                on_done=lambda _: self._do_pose_body_swap(idx, target, mask, callback),
                on_error=lambda _m: callback(self._pose_paths[idx], idx),
            )
            return
        cfg = _load_project_config(self._app_state)
        # Only force the character checkpoint for SD/SDXL families; flux/zimage
        # fall back to the project's img_checkpoint (body_double is SD-based).
        if (self._cs.model_family or "sdxl") in ("sd15", "sdxl") and self._cs.checkpoint:
            cfg.bodydouble_checkpoint = self._cs.checkpoint
        cfg.bodydouble_controlnet_type = "openpose"
        cfg.bodydouble_controlnet_strength = float(self._cs.body_cn_strength)
        cfg.bodydouble_ip_adapter_variant = "faceid_plus"
        cfg.bodydouble_ip_adapter_scale = float(self._cs.body_ip_scale)
        cfg.bodydouble_denoising_strength = float(self._cs.body_denoise)
        cfg.bodydouble_cfg_scale = float(self._cs.body_cfg)
        cfg.bodydouble_steps = int(self._cs.steps)
        cfg.bodydouble_prompt = self._cs.positive_prompt
        cfg.bodydouble_negative_prompt = self._cs.negative_prompt
        cfg.bodydouble_seed = -1
        cap = idx

        def _cleanup():
            try:
                os.unlink(mask)
            except OSError:
                pass

        def on_done(paths):
            _cleanup()
            res = (paths[0] if isinstance(paths, list) else str(paths)) if paths else self._pose_paths[cap]
            if res and Path(res).is_file():
                self._pose_paths[cap] = res
            callback(self._pose_paths[cap], cap)

        def on_err(msg):
            _cleanup()
            logger.warning("Pose body-swap failed for %d: %s — keeping original", cap, msg)
            callback(self._pose_paths[cap], cap)

        worker = BodyDoubleWorker(
            pipeline=self._app_state.img_pipeline,
            project_name=_project_name(self._app_state),
            target_image=target, mask=mask, source_person=self._cs.selected_base,
            project_config=cfg, parent=self,
        )
        self._wizard._run_worker(worker, on_done=on_done, on_error=on_err)

    def _on_seq_done(self, paths, idx) -> None:
        if paths:
            p = paths[0] if isinstance(paths, list) else str(paths)
            self._pose_paths[idx] = p
            self._postprocess_pose(idx, self._finish_seq_pose)
            return
        # Mark failed pose visually
        self._thumb_labels[idx].setText("Failed")
        self._thumb_labels[idx].setStyleSheet(
            "background: #3a2020; border: 1px solid #644; color: #c88;"
        )
        self._current_pose_idx = idx + 1
        self._generate_next_pose()

    def _finish_seq_pose(self, result_path, idx) -> None:
        """Update thumbnail after face swap on a sequential pose, then continue."""
        if result_path and Path(result_path).is_file():
            self._pose_paths[idx] = result_path
            pm = QPixmap(result_path).scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._thumb_labels[idx].setPixmap(pm)
        self._current_pose_idx = idx + 1
        self._generate_next_pose()

    def _run_pose_face_swap(self, idx: int, callback=None) -> None:
        """Run face swap on a generated pose to ensure face consistency.

        The face SOURCE is the canonical base (``selected_base``) — i.e. the
        post-swap base if a face swap was applied — so every pose references the
        NEW face, not the originally-supplied face source.
        """
        from sdqt.workers.image import FaceSwapWorker

        target_path = self._pose_paths[idx]
        if not target_path or not Path(target_path).is_file():
            if callback:
                callback(None, idx)
            return

        cfg = _load_project_config(self._app_state)
        # Apply enhancer + blend settings from shared character state
        cfg.faceswap_enhancer = self._cs.face_enhancer
        cfg.faceswap_enhancer_strength = self._cs.face_enhancer_strength
        cfg.faceswap_blend_ratio = self._cs.face_blend_ratio
        models_dir = self._app_state.global_config.model_paths.get("face_models_dir", "")

        face_source = self._cs.selected_base or self._cs.face_source_path
        worker = FaceSwapWorker(
            source_path=face_source,
            target_path=target_path,
            project_config=cfg,
            models_dir=models_dir,
            enhancer_strength=self._cs.face_enhancer_strength,
            parent=self,
        )

        captured_idx = idx

        def on_done(result_path):
            if callback:
                callback(str(result_path), captured_idx)

        def on_error(msg):
            logger.warning("Face swap failed for pose %d: %s — keeping original", captured_idx, msg)
            # Keep the original un-swapped pose and continue
            if callback:
                callback(target_path, captured_idx)

        self._wizard._run_worker(worker, on_done=on_done, on_error=on_error)

    @Slot(str)
    def _on_pose_error(self, msg: str) -> None:
        logger.warning("Pose %d error: %s", self._current_pose_idx, msg)
        self._wizard._status.setText(f"Pose generation error: {msg}")
        # Mark failed pose visually
        idx = self._current_pose_idx
        if 0 <= idx < len(self._thumb_labels):
            self._thumb_labels[idx].setText("Failed")
            self._thumb_labels[idx].setStyleSheet(
                "background: #3a2020; border: 1px solid #644; color: #c88;"
            )
            self._pose_checks[idx].setChecked(False)
        # Continue to next pose
        self._current_pose_idx += 1
        self._generate_next_pose()

    def _collect_approved(self) -> None:
        """Set approved_poses + approved_pose_specs (parallel, framing preserved)."""
        self._cs.pose_images = list(self._pose_paths)
        kept = [
            i for i, cb in enumerate(self._pose_checks)
            if cb.isChecked() and self._pose_paths[i] and Path(self._pose_paths[i]).is_file()
        ]
        self._cs.approved_poses = [self._pose_paths[i] for i in kept]
        self._cs.approved_pose_specs = [
            {"distance": POSES[i].distance, "angle": POSES[i].angle,
             "orientation": POSES[i].orientation, "pose_index": i}
            for i in kept
        ]

    def on_leave(self) -> None:
        self._collect_approved()
        # Free SD VRAM — last page using image generation
        self._app_state.unload_sd_pipelines()

    def validate(self) -> bool:
        self._collect_approved()
        if not self._cs.approved_poses:
            QMessageBox.warning(self, "Missing", "Please approve at least one pose variant.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 — Review & Save
# ═══════════════════════════════════════════════════════════════════════════


class ReviewSavePage(WizardPage):
    """Review all results and save the character to the library."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = wizard
        self._saved = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        inner = QVBoxLayout(content)

        # Base image preview
        self._base_preview = QLabel()
        self._base_preview.setAlignment(Qt.AlignCenter)
        self._base_preview.setFixedHeight(200)
        inner.addWidget(self._base_preview)

        # Info summary
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.RichText)
        inner.addWidget(self._info)

        # Pose gallery
        inner.addWidget(QLabel("<b>Approved Poses</b>"))
        self._pose_gallery = ImageGalleryWidget("Poses")
        inner.addWidget(self._pose_gallery)

        inner.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self._save_btn = QPushButton("Save Character")
        self._save_btn.setFixedWidth(160)
        self._save_btn.clicked.connect(self._save)
        save_row.addWidget(self._save_btn)
        self._save_copy_btn = QPushButton("Save as Copy…")
        self._save_copy_btn.setFixedWidth(130)
        self._save_copy_btn.setToolTip("Save under a new name without overwriting the loaded character")
        self._save_copy_btn.clicked.connect(lambda: self._save(as_copy=True))
        save_row.addWidget(self._save_copy_btn)
        save_row.addStretch()
        layout.addLayout(save_row)

        self._save_note = QLabel(
            "Saving builds the training datasets (full-body + face-only) and "
            "unlocks the next page, where you can train the character LoRA."
        )
        self._save_note.setWordWrap(True)
        self._save_note.setStyleSheet("color: #aaa; font-size: 13px;")
        layout.addWidget(self._save_note)

    def on_enter(self) -> None:
        self._saved = False
        self._save_btn.setEnabled(True)
        self._save_copy_btn.setEnabled(bool(getattr(self._wizard, "_loaded_char_dir", None)))

        # Base preview
        if self._cs.selected_base and Path(self._cs.selected_base).is_file():
            pm = QPixmap(self._cs.selected_base).scaled(
                300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._base_preview.setPixmap(pm)

        # Info
        lora_str = ", ".join(self._cs.selected_loras) if self._cs.selected_loras else "None"
        pos_text = self._cs.positive_prompt[:200]
        neg_text = self._cs.negative_prompt[:100]
        self._info.setText(
            f"<b>Name:</b> {self._cs.name}<br>"
            f"<b>Style:</b> {self._cs.style}<br>"
            f"<b>Checkpoint:</b> {self._cs.checkpoint or '(project default)'}<br>"
            f"<b>LoRAs:</b> {lora_str}<br>"
            f"<b>Positive:</b> {pos_text}<br>"
            f"<b>Negative:</b> {neg_text}"
        )

        # Pose gallery
        if self._cs.approved_poses:
            self._pose_gallery.load_images(self._cs.approved_poses)

    def _resolve_char_dir(self, as_copy: bool) -> Path:
        """Pick the save directory, honoring update-vs-copy and avoiding clobber."""
        base = _characters_dir(self._app_state)
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in self._cs.name).strip()
        safe = safe or "unnamed"
        loaded = getattr(self._wizard, "_loaded_char_dir", None)
        # Update-in-place only when not a copy AND the name matches the loaded dir.
        if not as_copy and loaded and Path(loaded).name == safe:
            return Path(loaded)
        target = base / safe
        if as_copy or (loaded and Path(loaded).name == safe):
            n = 1
            while (base / f"{safe}_copy_{n}").exists():
                n += 1
            target = base / f"{safe}_copy_{n}"
        return target

    def _save(self, as_copy: bool = False) -> None:
        char_dir = self._resolve_char_dir(as_copy)
        char_dir.mkdir(parents=True, exist_ok=True)
        poses_dir = char_dir / "poses"
        poses_dir.mkdir(exist_ok=True)

        # Save image assets (reconstructed by load_character from these names).
        if self._cs.selected_base and Path(self._cs.selected_base).is_file():
            shutil.copy2(self._cs.selected_base, char_dir / "base.png")
        if self._cs.reference_image and Path(self._cs.reference_image).is_file():
            shutil.copy2(self._cs.reference_image, char_dir / "reference_ref.png")
        if self._cs.face_source_path and Path(self._cs.face_source_path).is_file():
            shutil.copy2(self._cs.face_source_path, char_dir / "face_ref.png")
        if self._cs.body_source_path and Path(self._cs.body_source_path).is_file():
            shutil.copy2(self._cs.body_source_path, char_dir / "body_ref.png")
        specs = self._cs.approved_pose_specs or []
        for i, pose_path in enumerate(self._cs.approved_poses):
            if pose_path and Path(pose_path).is_file():
                sp = specs[i] if i < len(specs) else {"distance": "full", "angle": "front"}
                shutil.copy2(pose_path,
                             poses_dir / _pose_filename(i, sp.get("distance", "full"),
                                                        sp.get("angle", "front")))

        # Full-state metadata (round-trips for load/edit).
        meta = self._cs.to_meta()
        meta["pose_count"] = len(self._cs.approved_poses)
        with open(char_dir / "character.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Re-point the approved poses at the saved copies so the datasets build
        # from durable files; rebuild specs from the (re-globbed) filenames so
        # ordering stays aligned with the framing.
        saved_poses = sorted(str(p) for p in poses_dir.glob("*.png"))
        if saved_poses:
            self._cs.approved_poses = saved_poses
            self._cs.approved_pose_specs = [
                {"distance": d, "angle": a, "orientation":
                    "landscape" if d == "full" and a == "side" else "portrait"}
                for d, a in (_parse_pose_distance_angle(p) for p in saved_poses)
            ]
        self._wizard._datasets = _build_character_datasets(char_dir, self._cs, self._app_state)
        self._wizard._char_dir = str(char_dir)

        self._wizard._loaded_char_dir = str(char_dir)
        self._saved = True
        self._save_copy_btn.setEnabled(True)
        self._wizard._status.setText(f"Character saved to {char_dir}")
        ds = self._wizard._datasets
        face_note = "full-body + face datasets" if ds.get("face") else "full-body dataset (face crops unavailable)"
        QMessageBox.information(
            self, "Saved",
            f"Character '{self._cs.name}' saved to:\n{char_dir}\n\n"
            f"Training {face_note} built. Go to the next page to train the LoRA.",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Page 7 — Train LoRA (in-wizard)
# ═══════════════════════════════════════════════════════════════════════════


class TrainLoRAPage(WizardPage):
    """Train the character LoRA in-wizard from the auto-built datasets."""

    def __init__(self, char_state: CharacterState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._cs = char_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>Train Character LoRA</b>"))
        self._status_lbl = QLabel("Save the character on the previous page to build datasets.")
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color:#aaa; font-size:13px;")
        layout.addWidget(self._status_lbl)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Target:"))
        self._target = QComboBox()
        self._target.addItem("Image LoRA (SD/SDXL/Pony/Illustrious/Flux/Z-Image)", "image")
        self._target.addItem("Wan 2.2 video LoRA (ai-toolkit)", "wan")
        self._target.addItem("LTX 2.3 video LoRA (ai-toolkit)", "ltx")
        self._target.currentIndexChanged.connect(self._on_target_changed)
        r1.addWidget(self._target, 1)
        layout.addLayout(r1)

        # Dataset is auto-selected by target + model + Low-VRAM (no manual pick):
        # video/low-VRAM 512 → close-heavy 60/30/10; high-res image → 40/30/30.
        self._dataset_note = QLabel("Dataset: auto (composed on Save).")
        self._dataset_note.setStyleSheet("color:#8bf; font-size:12px;")
        layout.addWidget(self._dataset_note)

        # Download composed datasets as .zip.
        dl = QHBoxLayout()
        dl.addWidget(QLabel("Download:"))
        self._dl_video_btn = QPushButton("Video 60/30/10 (.zip)")
        self._dl_video_btn.clicked.connect(lambda: self._download_dataset("video512"))
        dl.addWidget(self._dl_video_btn)
        self._dl_highres_btn = QPushButton("High-res (.zip)")
        self._dl_highres_btn.clicked.connect(lambda: self._download_dataset("highres"))
        dl.addWidget(self._dl_highres_btn)
        self._dl_face_btn = QPushButton("Face (.zip)")
        self._dl_face_btn.clicked.connect(lambda: self._download_dataset("face"))
        dl.addWidget(self._dl_face_btn)
        dl.addStretch()
        layout.addLayout(dl)

        r3 = QHBoxLayout()
        r3.setSpacing(6)
        r3.addWidget(QLabel("Rank:"))
        self._rank = QSpinBox(); self._rank.setRange(4, 256); self._rank.setValue(32)
        r3.addWidget(self._rank)
        r3.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox(); self._steps.setRange(100, 20000); self._steps.setSingleStep(100); self._steps.setValue(1500)
        r3.addWidget(self._steps)
        r3.addWidget(QLabel("LR:"))
        self._lr = QLineEdit("1e-4"); self._lr.setFixedWidth(80)
        r3.addWidget(self._lr)
        r3.addStretch()
        layout.addLayout(r3)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("Output name:"))
        self._out_name = QLineEdit()
        r4.addWidget(self._out_name, 1)
        layout.addLayout(r4)

        # Low-VRAM toggle — seeded from the global Memory Profile (1=min VRAM …
        # 5=max speed; ≤3 ⇒ low-VRAM). ON applies the 12 GB preset (float8 quant,
        # offload, 512px dataset); OFF trains at full precision (needs more VRAM).
        r5 = QHBoxLayout()
        self._low_vram = QCheckBox("Low VRAM (12 GB preset)")
        prof = int(getattr(self._app_state.global_config, "memory_profile", 3) or 3)
        self._low_vram.setChecked(prof <= 3)
        self._low_vram.setToolTip(
            "Float8 quantization + offload + 512px dataset to fit ~12 GB. "
            "Off = full precision (faster, more VRAM). Default from your Memory Profile."
        )
        r5.addWidget(self._low_vram)
        r5.addStretch()
        layout.addLayout(r5)

        br = QHBoxLayout()
        self._train_btn = QPushButton("Start Training")
        self._train_btn.clicked.connect(self._start)
        br.addWidget(self._train_btn)
        br.addStretch()
        layout.addLayout(br)
        layout.addStretch()

    def on_enter(self) -> None:
        ds = getattr(self._wizard, "_datasets", None) or {}
        if not self._out_name.text():
            safe = "".join(c if c.isalnum() else "_" for c in (self._cs.name or "character")).strip("_").lower()
            self._out_name.setText(f"{safe}_lora")
        has = bool(ds)
        self._dl_video_btn.setEnabled(has and bool(ds.get("video512")))
        self._dl_highres_btn.setEnabled(has and bool(ds.get("highres")))
        self._dl_face_btn.setEnabled(has and bool(ds.get("face")))
        if has:
            self._status_lbl.setText("Datasets ready (composed on Save). Pick a target and train.")
        else:
            self._status_lbl.setText("Save the character on the previous page to build datasets.")
        self._on_target_changed()

    def _on_target_changed(self) -> None:
        ds = getattr(self._wizard, "_datasets", None) or {}
        self._train_btn.setEnabled(bool(ds))
        chosen = self._dataset_for_training()
        which = "60/30/10 close-heavy (video512)" if (chosen and "video512" in chosen) else "balanced (highres)"
        self._dataset_note.setText(f"Dataset (auto): {which}" if chosen else
                                   "Dataset: auto (composed on Save).")

    def _dataset_for_training(self) -> str | None:
        """Auto-select the composed dataset by target + model family + Low-VRAM."""
        ds = getattr(self._wizard, "_datasets", None) or {}
        if not ds:
            return None
        kind = self._target.currentData()
        family = (self._cs.model_family or "sdxl").lower()
        low_vram = self._low_vram.isChecked()
        # Video (Wan/LTX), SD 1.5, and any low-VRAM image → close-heavy 60/30/10 @512.
        if kind in ("wan", "ltx") or family == "sd15" or low_vram:
            return ds.get("video512") or ds.get("highres")
        return ds.get("highres") or ds.get("video512")

    def _download_dataset(self, key: str) -> None:
        ds = getattr(self._wizard, "_datasets", None) or {}
        src = ds.get(key)
        if not src or not Path(src).is_dir():
            QMessageBox.warning(self, "No dataset", "That dataset hasn't been built (Save first).")
            return
        from sdqt.utils.file_dialog import get_save_filename
        name = (self._out_name.text() or "character").strip()
        path, _ = get_save_filename(self, "Save Dataset Zip", f"{name}_{key}.zip", "Zip (*.zip)")
        if not path:
            return
        if path.lower().endswith(".zip"):
            path = path[:-4]
        try:
            archive = shutil.make_archive(path, "zip", src)
            self._status_lbl.setText(f"Dataset zipped → {archive}")
            QMessageBox.information(self, "Downloaded", f"Saved:\n{archive}")
        except Exception as exc:
            QMessageBox.warning(self, "Zip failed", str(exc))

    def _start(self) -> None:
        ds = getattr(self._wizard, "_datasets", None) or {}
        if not ds:
            QMessageBox.information(self, "Save first",
                                   "Save the character first (previous page) to build datasets.")
            return
        dataset_dir = self._dataset_for_training()
        if not dataset_dir or not Path(dataset_dir).is_dir():
            QMessageBox.warning(self, "No dataset", "The composed dataset isn't available.")
            return
        kind = self._target.currentData()
        if kind == "image":
            self._start_image(dataset_dir)
        elif kind == "wan":
            self._start_wan(dataset_dir)
        elif kind == "ltx":
            self._start_ltx(dataset_dir)

    def _resolve_checkpoint_path(self) -> str | None:
        try:
            from supremediffusion.models.sd_models import scan_all_image_models
            for m in scan_all_image_models(self._app_state.global_config):
                if m.name == self._cs.checkpoint:
                    return m.filename
        except Exception:
            logger.debug("checkpoint resolve failed", exc_info=True)
        return None

    def _write_kohya_config(self, dataset_dir: str, ckpt_path: str, out_dir: str,
                            is_sdxl: bool, low_vram: bool = True):
        import tempfile
        # Low VRAM caps resolution at 768 + 8-bit optimizer + grad checkpointing;
        # otherwise allow 1024 + plain AdamW for speed.
        res = (768 if low_vram else 1024) if is_sdxl else (512 if low_vram else 768)
        cfg = {
            "pretrained_model_name_or_path": ckpt_path,
            "train_data_dir": dataset_dir,
            "output_dir": out_dir,
            "output_name": self._out_name.text() or "lora",
            "network_module": "networks.lora",
            "network_dim": self._rank.value(),
            "network_alpha": max(1, self._rank.value() // 2),
            "resolution": f"{res},{res}",
            "max_train_steps": self._steps.value(),
            "learning_rate": float(self._lr.text() or "1e-4"),
            "lr_scheduler": "cosine",
            "optimizer_type": "AdamW8bit" if low_vram else "AdamW",
            "mixed_precision": "bf16" if is_sdxl else "fp16",
            "gradient_checkpointing": low_vram,
            "cache_latents": True,
            "enable_bucket": True,
            "seed": 42,
            "train_batch_size": 1,
            "save_every_n_steps": max(self._steps.value() // 3, 200),
        }
        if is_sdxl:
            cfg["sdxl"] = True
        try:
            import toml
            p = Path(tempfile.mktemp(suffix=".toml", prefix="char_lora_"))
            with open(p, "w") as f:
                toml.dump(cfg, f)
        except ImportError:
            p = Path(tempfile.mktemp(suffix=".json", prefix="char_lora_"))
            p.write_text(json.dumps(cfg, indent=2))
        return p

    def _resolve_family_model_path(self, family: str) -> str:
        """Resolve the base model path for ai-toolkit (flux / z-image / ltx)."""
        mp = self._app_state.global_config.model_paths
        if family == "flux":
            for k in ("flux_transformer", "flux_dir", "flux_model", "flux"):
                if mp.get(k):
                    return mp[k]
            return "black-forest-labs/FLUX.1-dev"
        if family == "ltx":
            for k in ("ltx_transformer", "ltx_dir", "ltx_model", "ltx"):
                if mp.get(k):
                    return mp[k]
            return "Lightricks/LTX-Video"
        for k in ("zimage_model", "zimage_dir", "zimage", "zimage_transformer"):
            if mp.get(k):
                return mp[k]
        return "alibaba-pai/Z-Image-Turbo"

    def _start_ltx(self, dataset_dir: str) -> None:
        from sdqt.workers.video_lora_train import ImageLoRAAIToolkitWorker, find_ai_toolkit
        toolkit = find_ai_toolkit()
        if not toolkit:
            QMessageBox.warning(self, "ai-toolkit missing",
                                "ai-toolkit not found — install it to train LTX 2.3 LoRAs.")
            return
        out_dir = _characters_dir(self._app_state) / "_loras"
        out_dir.mkdir(parents=True, exist_ok=True)
        worker = ImageLoRAAIToolkitWorker(
            ai_toolkit_dir=toolkit, name=self._out_name.text() or "char_lora",
            family="ltx", model_path=self._resolve_family_model_path("ltx"),
            dataset_dir=dataset_dir, output_dir=str(out_dir),
            rank=self._rank.value(), lr=float(self._lr.text() or "1e-4"),
            steps=self._steps.value(), low_vram=self._low_vram.isChecked(),
            parent=self._wizard,
        )
        self._train_btn.setEnabled(False)
        self._status_lbl.setText("Training LTX 2.3 LoRA via ai-toolkit…")
        self._wizard._run_worker(worker, on_done=self._on_done, on_error=self._on_err)

    def _start_image(self, dataset_dir: str) -> None:
        family = (self._cs.model_family or "sdxl").lower()
        low_vram = self._low_vram.isChecked()
        out_dir = _characters_dir(self._app_state) / "_loras"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Flux + Z-Image Turbo train via ai-toolkit (Ostris); SD1.5/SDXL/Pony/
        # Illustrious via kohya (Pony & Illustrious ARE SDXL).
        if family in ("flux", "zimage"):
            from sdqt.workers.video_lora_train import ImageLoRAAIToolkitWorker, find_ai_toolkit
            toolkit = find_ai_toolkit()
            if not toolkit:
                QMessageBox.warning(self, "ai-toolkit missing",
                                    f"ai-toolkit not found — install it to train {family.upper()} LoRAs.")
                return
            worker = ImageLoRAAIToolkitWorker(
                ai_toolkit_dir=toolkit, name=self._out_name.text() or "char_lora",
                family=family, model_path=self._resolve_family_model_path(family),
                dataset_dir=dataset_dir, output_dir=str(out_dir),
                rank=self._rank.value(), lr=float(self._lr.text() or "1e-4"),
                steps=self._steps.value(), low_vram=low_vram, parent=self._wizard,
            )
            self._train_btn.setEnabled(False)
            self._status_lbl.setText(f"Training {family.upper()} LoRA via ai-toolkit…")
            self._wizard._run_worker(worker, on_done=self._on_done, on_error=self._on_err)
            return

        from sdqt.workers.lora_train import LoRATrainWorker, find_kohya_script
        script = find_kohya_script()
        if not script:
            QMessageBox.warning(self, "kohya missing",
                                "kohya sd-scripts not found — install it to train image LoRAs.")
            return
        ckpt = self._resolve_checkpoint_path()
        if not ckpt:
            QMessageBox.warning(self, "No checkpoint",
                                "Couldn't resolve the base checkpoint. Select an SD 1.5 / SDXL / "
                                "Pony / Illustrious model in the footer.")
            return
        is_sdxl = family == "sdxl"
        config_path = self._write_kohya_config(dataset_dir, ckpt, str(out_dir), is_sdxl, low_vram)
        worker = LoRATrainWorker(
            config_path=str(config_path), kohya_script=script,
            output_path=str(out_dir / f"{self._out_name.text() or 'lora'}.safetensors"),
            is_sdxl=is_sdxl, parent=self._wizard,
        )
        self._train_btn.setEnabled(False)
        self._status_lbl.setText("Training image LoRA…")
        self._wizard._run_worker(worker, on_done=self._on_done, on_error=self._on_err)

    def _start_wan(self, dataset_dir: str) -> None:
        from sdqt.workers.video_lora_train import VideoLoRATrainWorker, find_ai_toolkit
        toolkit = find_ai_toolkit()
        if not toolkit:
            QMessageBox.warning(self, "ai-toolkit missing",
                                "ai-toolkit not found — install it to train Wan 2.2 LoRAs.")
            return
        mp = (self._app_state.global_config.model_paths.get("wan_model_dir", "")
              or self._app_state.global_config.model_paths.get("transformer", ""))
        if not mp:
            QMessageBox.warning(self, "No Wan model",
                                "Set the Wan model path in Settings to train Wan 2.2 LoRAs.")
            return
        out_dir = _characters_dir(self._app_state) / "_loras"
        out_dir.mkdir(parents=True, exist_ok=True)
        lr = float(self._lr.text() or "1e-4")
        ds = getattr(self._wizard, "_datasets", {}) or {}
        worker = VideoLoRATrainWorker(
            ai_toolkit_dir=toolkit, name=self._out_name.text() or "char_lora",
            model_path=mp, dataset_dir=dataset_dir, output_dir=str(out_dir),
            hn_rank=self._rank.value(), ln_rank=self._rank.value(),
            hn_lr=lr, ln_lr=lr, hn_steps=self._steps.value(), ln_steps=self._steps.value(),
            num_frames=1,  # character dataset is still poses, not video clips
            trigger_token=ds.get("trigger", ""), low_vram=self._low_vram.isChecked(),
            parent=self._wizard,
        )
        self._train_btn.setEnabled(False)
        self._status_lbl.setText("Training Wan 2.2 LoRA…")
        self._wizard._run_worker(worker, on_done=self._on_done, on_error=self._on_err)

    @Slot(object)
    def _on_done(self, result) -> None:
        self._train_btn.setEnabled(True)
        self._status_lbl.setText("Training complete.")
        QMessageBox.information(self, "Training complete", f"LoRA training finished.\n{result}")

    @Slot(str)
    def _on_err(self, msg: str) -> None:
        self._train_btn.setEnabled(True)
        self._status_lbl.setText(f"Training error: {msg}")
        QMessageBox.warning(self, "Training error", msg)

    def validate(self) -> bool:
        return True  # training is optional


# ═══════════════════════════════════════════════════════════════════════════
# CreateCharacterWizard — ties all pages together
# ═══════════════════════════════════════════════════════════════════════════


class CreateCharacterWizard(SequenceWizard):
    """7-page wizard: Info → Prompt → Base Gen → Face/Body → Poses → Save → Train."""

    def __init__(self, state, *, img_params: dict | None = None, parent=None) -> None:
        self._cs = CharacterState()

        # Seed CharacterState from the Sequences-tab params (or project config)
        # BEFORE building the footer settings bar, which loads from it.
        if img_params:
            self._cs.checkpoint = img_params.get("checkpoint", "")
            self._cs.sampler = img_params.get("sampler", "")
            self._cs.steps = img_params.get("steps", 20)
            self._cs.cfg_scale = img_params.get("cfg_scale", 7.0)
            self._cs.width = img_params.get("width", 512)
            self._cs.height = img_params.get("height", 512)
            self._cs.seed = img_params.get("seed", -1)
        else:
            cfg = ProjectConfig.load(
                state.project_manager.get_project_path(state.current_project or "_default")
            )
            self._cs.checkpoint = cfg.img_checkpoint
            self._cs.sampler = cfg.img_sampler
            self._cs.steps = cfg.img_steps
            self._cs.cfg_scale = cfg.img_cfg_scale
            self._cs.width = cfg.img_width
            self._cs.height = cfg.img_height
            self._cs.seed = cfg.img_seed

        # Cross-page state for save/train.
        self._datasets: dict = {}
        self._loaded_char_dir: str | None = None
        self._char_dir: str | None = None

        # Persistent footer: full generation settings (model + CFG/steps/
        # sampler/scheduler/resolution), available on every page.
        self._settings_bar = GenerationSettingsBar(self._cs, state)

        super().__init__(
            "Create a Character",
            state,
            footer_widget=self._settings_bar,
            parent=parent,
        )

        # Populate the bar's controls from the seeded state.
        self._settings_bar.load_from_state()

        # Build pages
        self._pages = [
            CharInfoPage(self._cs, state, self),
            PromptGenPage(self._cs, state, self, self),
            BaseGenPage(self._cs, state, self, self),
            FaceBodySwapPage(self._cs, state, self, self),
            PoseVariantsPage(self._cs, state, self, self),
            ReviewSavePage(self._cs, state, self, self),
            TrainLoRAPage(self._cs, state, self, self),
        ]
        self._finish_setup()

    def _on_finish(self) -> None:
        # Page 6 handles saving via its own button
        self.accept()
