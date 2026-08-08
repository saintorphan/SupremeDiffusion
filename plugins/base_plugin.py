"""Base class for SupremeDiffusion plugins + ContentPack loader."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContentPack — loaded from a per-type JSON file
# ---------------------------------------------------------------------------

@dataclass
class ContentPack:
    """A single content JSON loaded from a plugin's content/ directory.

    Every field except *name* and *key* is optional — a minimal content pack
    only needs stages or prompts to be useful.
    """

    name: str
    key: str
    description: str = ""
    author: str = ""
    file_path: str = ""

    # Stage progression
    stages: list[dict] = field(default_factory=list)
    endings: list[dict] = field(default_factory=list)

    # Quick-gen builder options  (label, prompt) tuples
    test_prompts: list[dict] = field(default_factory=list)
    video_prompts: list[tuple[str, str]] = field(default_factory=list)
    image_prompts: list[tuple[str, str]] = field(default_factory=list)
    clothing: list[tuple[str, str]] = field(default_factory=list)
    poses: list[tuple[str, str]] = field(default_factory=list)
    insertions: list[tuple[str, str]] = field(default_factory=list)
    expressions: list[tuple[str, str]] = field(default_factory=list)
    settings: list[tuple[str, str]] = field(default_factory=list)
    camera_angles: list[tuple[str, str]] = field(default_factory=list)
    atmospheres: list[tuple[str, str]] = field(default_factory=list)

    # Sequence-specific (substances, speeds, ending variants)
    substances: list[dict] = field(default_factory=list)
    speeds: list[dict] = field(default_factory=list)
    ending_variants_by_substance: dict[str, list[dict]] = field(default_factory=dict)

    # LoRA suggestions
    lora_suggestions: dict = field(default_factory=dict)

    # Auto Direct system prompt override
    auto_direct_system_prompt: str = ""

    # Preferred quality tier (draft/standard/production/ultra)
    # If set, the system uses this as the default tier for generation.
    # Hardware auto-detect may downgrade if the GPU can't handle it.
    quality_tier: str = ""


def _load_label_prompt_list(raw: list[dict]) -> list[tuple[str, str]]:
    """Convert ``[{"label": "...", "prompt": "..."}]`` → ``[(label, prompt)]``."""
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict) and "label" in item:
            out.append((item["label"], item.get("prompt", "")))
    return out


class ContentValidationError(Exception):
    """Raised when a content pack JSON fails validation."""


def _validate_content(data: dict, path: Path) -> list[str]:
    """Validate content pack data against the schema.

    Returns a list of warning strings (non-fatal issues).
    Raises ContentValidationError for fatal problems.
    """
    fname = path.name
    warnings: list[str] = []

    # Must be a dict
    if not isinstance(data, dict):
        raise ContentValidationError(
            f"{fname}: Content pack must be a JSON object, got {type(data).__name__}"
        )

    # Required fields
    if "name" not in data or not isinstance(data["name"], str) or not data["name"].strip():
        raise ContentValidationError(
            f"{fname}: Missing or empty required field 'name'"
        )
    if "key" not in data or not isinstance(data["key"], str) or not data["key"].strip():
        raise ContentValidationError(
            f"{fname}: Missing or empty required field 'key'"
        )

    # Validate list fields — each must be an array of objects with 'label'
    _LIST_FIELDS = [
        "stages", "endings", "test_prompts", "video_prompts", "image_prompts",
        "clothing", "poses", "insertions", "expressions",
        "settings", "camera_angles", "atmospheres", "substances", "speeds",
    ]
    for field_name in _LIST_FIELDS:
        value = data.get(field_name)
        if value is None:
            continue  # Optional — skip
        if not isinstance(value, list):
            raise ContentValidationError(
                f"{fname}: '{field_name}' must be an array, got {type(value).__name__}"
            )
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ContentValidationError(
                    f"{fname}: '{field_name}[{i}]' must be an object, "
                    f"got {type(item).__name__}"
                )
            if "label" not in item:
                warnings.append(
                    f"{fname}: '{field_name}[{i}]' missing 'label' field — "
                    f"item will be skipped"
                )

    # Validate dict fields
    for field_name in ("lora_suggestions", "ending_variants_by_substance"):
        value = data.get(field_name)
        if value is not None and not isinstance(value, dict):
            raise ContentValidationError(
                f"{fname}: '{field_name}' must be an object, got {type(value).__name__}"
            )

    # Validate string fields
    for field_name in ("auto_direct_system_prompt", "description", "author"):
        value = data.get(field_name)
        if value is not None and not isinstance(value, str):
            warnings.append(
                f"{fname}: '{field_name}' should be a string, got {type(value).__name__}"
            )

    return warnings


def load_content_pack(path: Path) -> ContentPack:
    """Load and validate a single content JSON file. Returns a ContentPack.

    Raises ContentValidationError for fatal schema violations.
    Logs warnings for non-fatal issues.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate before building the ContentPack
    warnings = _validate_content(data, path)
    for w in warnings:
        logger.warning("Content pack: %s", w)

    pack = ContentPack(
        name=data.get("name", path.stem),
        key=data.get("key", path.stem),
        description=data.get("description", ""),
        author=data.get("author", ""),
        file_path=str(path),
        stages=data.get("stages", []),
        endings=data.get("endings", []),
        test_prompts=data.get("test_prompts", []),
        video_prompts=_load_label_prompt_list(data.get("video_prompts", [])),
        image_prompts=_load_label_prompt_list(data.get("image_prompts", [])),
        clothing=_load_label_prompt_list(data.get("clothing", [])),
        poses=_load_label_prompt_list(data.get("poses", [])),
        insertions=_load_label_prompt_list(data.get("insertions", [])),
        expressions=_load_label_prompt_list(data.get("expressions", [])),
        settings=_load_label_prompt_list(data.get("settings", [])),
        camera_angles=_load_label_prompt_list(data.get("camera_angles", [])),
        atmospheres=_load_label_prompt_list(data.get("atmospheres", [])),
        substances=data.get("substances", []),
        speeds=data.get("speeds", []),
        ending_variants_by_substance=data.get("ending_variants_by_substance", {}),
        lora_suggestions=data.get("lora_suggestions", {}),
        auto_direct_system_prompt=data.get("auto_direct_system_prompt", ""),
        quality_tier=data.get("quality_tier", ""),
    )
    return pack


def load_shared_content(path: Path) -> ContentPack:
    """Load _shared.json as a base ContentPack that other packs inherit from."""
    return load_content_pack(path)


def merge_with_shared(pack: ContentPack, shared: ContentPack) -> ContentPack:
    """Fill empty fields in *pack* with values from *shared*."""
    list_fields = [
        "video_prompts", "image_prompts", "clothing", "poses",
        "insertions", "expressions", "settings", "camera_angles",
        "atmospheres", "substances", "speeds",
    ]
    for fname in list_fields:
        pack_val = getattr(pack, fname)
        shared_val = getattr(shared, fname)
        if not pack_val and shared_val:
            setattr(pack, fname, list(shared_val))
        elif pack_val and shared_val:
            # Pack-specific items first, then shared as fallback
            setattr(pack, fname, list(pack_val) + list(shared_val))

    # Merge dicts
    if not pack.lora_suggestions and shared.lora_suggestions:
        pack.lora_suggestions = dict(shared.lora_suggestions)
    elif shared.lora_suggestions:
        merged = dict(shared.lora_suggestions)
        merged.update(pack.lora_suggestions)
        pack.lora_suggestions = merged

    if not pack.auto_direct_system_prompt and shared.auto_direct_system_prompt:
        pack.auto_direct_system_prompt = shared.auto_direct_system_prompt

    if not pack.ending_variants_by_substance and shared.ending_variants_by_substance:
        pack.ending_variants_by_substance = dict(shared.ending_variants_by_substance)

    return pack


# ---------------------------------------------------------------------------
# PluginBase — all plugins extend this
# ---------------------------------------------------------------------------

def generate_content_template(key: str = "my_content", name: str = "My Content Pack") -> dict:
    """Return a skeleton content pack dict with all fields and example values.

    Plugin authors and users can call this to create a new content JSON
    that conforms to the schema and can be edited by hand or in-app.
    """
    return {
        "content_version": 1,
        "name": name,
        "key": key,
        "description": "Describe this content pack in one line.",
        "author": "",
        "stages": [
            {"label": "Stage 1", "prompt": "description of stage 1", "drift": 2},
        ],
        "endings": [
            {"label": "Default Ending", "prompt": "ending description", "type": ""},
        ],
        "video_prompts": [
            {"label": "Example Video Prompt", "prompt": "a detailed video prompt"},
        ],
        "image_prompts": [
            {"label": "Example Image Prompt", "prompt": "a detailed image prompt"},
        ],
        "clothing": [
            {"label": "Default Outfit", "prompt": "wearing casual clothes"},
        ],
        "poses": [
            {"label": "Standing", "prompt": "standing in a neutral pose"},
        ],
        "insertions": [],
        "expressions": [
            {"label": "Neutral", "prompt": "neutral expression"},
        ],
        "settings": [
            {"label": "Studio", "prompt": "in a photography studio, white background"},
        ],
        "camera_angles": [
            {"label": "Medium Shot", "prompt": "medium shot, waist up"},
        ],
        "atmospheres": [
            {"label": "Neutral", "prompt": "neutral lighting, clean atmosphere"},
        ],
        "substances": [],
        "speeds": [],
        "ending_variants_by_substance": {},
        "lora_suggestions": {},
        "auto_direct_system_prompt": "",
    }


def add_item_to_content_pack(pack_path: Path, field_name: str, item: dict) -> None:
    """Append an item to a list field in a content pack JSON and save.

    Parameters
    ----------
    pack_path : Path
        Path to the content pack JSON file.
    field_name : str
        The list field to append to (e.g. "clothing", "poses").
    item : dict
        The item to append (must have at least "label").
    """
    with open(pack_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if field_name not in data:
        data[field_name] = []
    if not isinstance(data[field_name], list):
        raise ValueError(f"'{field_name}' is not a list in {pack_path.name}")

    data[field_name].append(item)

    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info("Added '%s' to %s in %s", item.get("label", "?"), field_name, pack_path.name)


class PluginBase(BaseTab):
    """Base class for all SupremeDiffusion plugins.

    Subclasses must set the ``PLUGIN_*`` class attributes and override
    ``_build_ui()`` to create their interface.
    """

    # Metadata — subclass MUST override these
    PLUGIN_NAME: str = "Unnamed Plugin"
    PLUGIN_VERSION: str = "1.0"
    PLUGIN_DESCRIPTION: str = ""
    PLUGIN_AUTHOR: str = ""
    PLUGIN_ICON: str = ""
    TAB_LABEL: str = ""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._content_packs: list[ContentPack] = []
        self._shared_content: ContentPack | None = None
        self._active_content: ContentPack | None = None

    # -- Content discovery ---------------------------------------------------

    def _content_dir(self) -> Path:
        """Return the path to this plugin's content/ directory.

        Override in subclasses if the content dir is in a non-standard location.
        Default: ``<module_file_dir>/content/``.
        """
        import sys
        mod = sys.modules.get(self.__class__.__module__)
        if mod and hasattr(mod, "__file__") and mod.__file__:
            return Path(mod.__file__).parent / "content"
        # Fallback: relative to working directory
        return Path("plugins") / "content"

    def discover_content(self) -> list[ContentPack]:
        """Scan the plugin's content/ directory for JSON packs.

        Loads ``_shared.json`` first (if present) and merges its fields
        into every subsequent pack as defaults.
        """
        content_dir = self._content_dir()
        if not content_dir.is_dir():
            logger.warning("Plugin %s: no content/ directory at %s",
                           self.PLUGIN_NAME, content_dir)
            return []

        # Load shared base if present
        shared_path = content_dir / "_shared.json"
        if shared_path.is_file():
            try:
                self._shared_content = load_shared_content(shared_path)
            except Exception:
                logger.exception("Failed to load _shared.json for %s", self.PLUGIN_NAME)

        packs: list[ContentPack] = []
        for json_file in sorted(content_dir.glob("*.json")):
            if json_file.name.startswith("_"):
                continue
            try:
                pack = load_content_pack(json_file)
            except ContentValidationError as exc:
                logger.warning("Content pack rejected: %s", exc)
                continue
            except json.JSONDecodeError as exc:
                logger.warning("Invalid JSON in %s: %s", json_file.name, exc)
                continue
            try:
                if self._shared_content:
                    pack = merge_with_shared(pack, self._shared_content)
                packs.append(pack)
                logger.info("Loaded content: %s (%s)", pack.name, json_file.name)
            except Exception:
                logger.exception("Failed to load content pack %s", json_file.name)

        self._content_packs = packs
        return packs

    @classmethod
    def get_metadata(cls) -> dict:
        """Return plugin metadata for registration / display."""
        return {
            "name": cls.PLUGIN_NAME,
            "version": cls.PLUGIN_VERSION,
            "description": cls.PLUGIN_DESCRIPTION,
            "author": cls.PLUGIN_AUTHOR,
            "icon": cls.PLUGIN_ICON,
            "tab_label": cls.TAB_LABEL or cls.PLUGIN_NAME,
        }
