"""Pipeline template / content pack loader.

Templates define transformation presets for the Pipeline Wizard AND optional
Quick Gen content packs (prompt presets, subject dropdowns, LoRA keywords,
Auto Direct system prompts).  Users can create, import, and export packs.

Template format (JSON) — Quick Gen keys are optional:
{
  "template_version": 1,
  "name": "Display Name",
  "description": "What this template contains",
  "author": "Author Name",

  // ── Pipeline Wizard ──
  "transformation_types": [ {"label": "...", "key": "..."} ],
  "transformation_stages": { "key": [ {"label":"...","prompt":"...","drift":1} ] },
  "ending_variants":       { "key": [ {"label":"...","prompt":"...","type":"climax|aftermath"} ] },

  // ── Quick Gen content pack (all optional) ──
  "quick_gen_video_prompts":  [ {"label":"...","prompt":"..."} ],
  "quick_gen_image_prompts":  [ {"label":"...","prompt":"..."} ],
  "quick_gen_clothing":       [ {"label":"...","prompt":"..."} ],
  "quick_gen_poses":          [ {"label":"...","prompt":"..."} ],
  "quick_gen_insertions":     [ {"label":"...","prompt":"..."} ],
  "quick_gen_expressions":    [ {"label":"...","prompt":"..."} ],
  "quick_gen_lora_keywords":  { "keyword": ["lora_stem", ...] },
  "auto_direct_system_prompt": "system prompt string"
}
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_VERSION = 1


@dataclass
class PipelineTemplate:
    """A loaded pipeline template / content pack."""

    name: str
    description: str
    author: str
    file_path: str  # where it was loaded from

    # ── Pipeline Wizard fields ──
    # (label, key) pairs
    transformation_types: list[tuple[str, str]] = field(default_factory=list)

    # key -> [(label, prompt, drift_cost)]
    transformation_stages: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)

    # key -> [(label, prompt, ending_type)]
    ending_variants: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)

    # ── Quick Gen content pack fields (all optional) ──
    # [(label, prompt)]
    quick_gen_video_prompts: list[tuple[str, str]] = field(default_factory=list)
    quick_gen_image_prompts: list[tuple[str, str]] = field(default_factory=list)
    quick_gen_clothing: list[tuple[str, str]] = field(default_factory=list)
    quick_gen_poses: list[tuple[str, str]] = field(default_factory=list)
    quick_gen_insertions: list[tuple[str, str]] = field(default_factory=list)
    quick_gen_expressions: list[tuple[str, str]] = field(default_factory=list)

    # keyword -> [lora_stem, ...]
    quick_gen_lora_keywords: dict[str, list[str]] = field(default_factory=dict)

    # Auto Direct system prompt override
    auto_direct_system_prompt: str = ""

    @property
    def has_quick_gen(self) -> bool:
        """True if this template provides any Quick Gen content."""
        return bool(
            self.quick_gen_video_prompts
            or self.quick_gen_image_prompts
            or self.quick_gen_clothing
            or self.quick_gen_poses
            or self.quick_gen_insertions
            or self.quick_gen_expressions
            or self.quick_gen_lora_keywords
            or self.auto_direct_system_prompt
        )


def _templates_dir(app_root: str | Path) -> Path:
    """Return the pipeline_templates directory, creating it if needed."""
    d = Path(app_root) / "pipeline_templates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_template(path: str | Path) -> PipelineTemplate | None:
    """Load a single template from a JSON file."""
    path = Path(path)
    if not path.is_file():
        logger.warning("Template file not found: %s", path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse template %s: %s", path, exc)
        return None

    version = data.get("template_version", 0)
    if version != TEMPLATE_VERSION:
        logger.warning("Template %s has version %s, expected %s", path, version, TEMPLATE_VERSION)

    # Parse transformation types
    types = []
    for entry in data.get("transformation_types", []):
        types.append((entry["label"], entry["key"]))

    # Ensure "Custom" is always available
    custom_keys = [k for _, k in types if k == "custom"]
    if not custom_keys:
        types.append(("Custom (Write Your Own)", "custom"))

    # Parse stages
    stages: dict[str, list[tuple[str, str, int]]] = {}
    for key, stage_list in data.get("transformation_stages", {}).items():
        parsed = []
        for s in stage_list:
            parsed.append((s["label"], s["prompt"], s.get("drift", 1)))
        stages[key] = parsed

    # Parse endings
    endings: dict[str, list[tuple[str, str, str]]] = {}
    for key, end_list in data.get("ending_variants", {}).items():
        parsed = []
        for e in end_list:
            parsed.append((e["label"], e["prompt"], e.get("type", "climax")))
        endings[key] = parsed

    # ── Parse Quick Gen content pack fields ──
    def _parse_label_prompt(items: list) -> list[tuple[str, str]]:
        return [(e["label"], e.get("prompt", "")) for e in items]

    qg_video = _parse_label_prompt(data.get("quick_gen_video_prompts", []))
    qg_image = _parse_label_prompt(data.get("quick_gen_image_prompts", []))
    qg_clothing = _parse_label_prompt(data.get("quick_gen_clothing", []))
    qg_poses = _parse_label_prompt(data.get("quick_gen_poses", []))
    qg_insertions = _parse_label_prompt(data.get("quick_gen_insertions", []))
    qg_expressions = _parse_label_prompt(data.get("quick_gen_expressions", []))
    qg_lora_kw: dict[str, list[str]] = data.get("quick_gen_lora_keywords", {})
    qg_auto_direct: str = data.get("auto_direct_system_prompt", "")

    return PipelineTemplate(
        name=data.get("name", path.stem),
        description=data.get("description", ""),
        author=data.get("author", ""),
        file_path=str(path),
        transformation_types=types,
        transformation_stages=stages,
        ending_variants=endings,
        quick_gen_video_prompts=qg_video,
        quick_gen_image_prompts=qg_image,
        quick_gen_clothing=qg_clothing,
        quick_gen_poses=qg_poses,
        quick_gen_insertions=qg_insertions,
        quick_gen_expressions=qg_expressions,
        quick_gen_lora_keywords=qg_lora_kw,
        auto_direct_system_prompt=qg_auto_direct,
    )


def scan_templates(app_root: str | Path) -> list[PipelineTemplate]:
    """Scan the pipeline_templates directory and return all valid templates."""
    d = _templates_dir(app_root)
    templates = []
    for f in sorted(d.glob("*.json")):
        t = load_template(f)
        if t is not None:
            templates.append(t)
    return templates


def import_template(source_path: str | Path, app_root: str | Path) -> PipelineTemplate | None:
    """Copy a template JSON file into the templates directory and load it."""
    source = Path(source_path)
    if not source.is_file():
        logger.warning("Import source not found: %s", source)
        return None

    dest = _templates_dir(app_root) / source.name
    # Avoid overwriting — add suffix if needed
    counter = 1
    while dest.exists():
        dest = _templates_dir(app_root) / f"{source.stem}_{counter}{source.suffix}"
        counter += 1

    shutil.copy2(source, dest)
    logger.info("Imported template: %s -> %s", source, dest)
    return load_template(dest)


def export_template(template: PipelineTemplate, dest_path: str | Path) -> bool:
    """Export a loaded template to a destination path."""
    try:
        shutil.copy2(template.file_path, dest_path)
        logger.info("Exported template: %s -> %s", template.file_path, dest_path)
        return True
    except Exception as exc:
        logger.warning("Failed to export template: %s", exc)
        return False


def save_template(template: PipelineTemplate, path: str | Path | None = None) -> bool:
    """Save a PipelineTemplate back to JSON."""
    path = Path(path or template.file_path)

    def _ser_lp(items: list[tuple[str, str]]) -> list[dict]:
        return [{"label": l, "prompt": p} for l, p in items]

    data: dict = {
        "template_version": TEMPLATE_VERSION,
        "name": template.name,
        "description": template.description,
        "author": template.author,
        "transformation_types": [
            {"label": label, "key": key}
            for label, key in template.transformation_types
        ],
        "transformation_stages": {
            key: [
                {"label": label, "prompt": prompt, "drift": drift}
                for label, prompt, drift in stages
            ]
            for key, stages in template.transformation_stages.items()
        },
        "ending_variants": {
            key: [
                {"label": label, "prompt": prompt, "type": etype}
                for label, prompt, etype in endings
            ]
            for key, endings in template.ending_variants.items()
        },
    }
    # Quick Gen content pack — only include if present
    if template.has_quick_gen:
        if template.quick_gen_video_prompts:
            data["quick_gen_video_prompts"] = _ser_lp(template.quick_gen_video_prompts)
        if template.quick_gen_image_prompts:
            data["quick_gen_image_prompts"] = _ser_lp(template.quick_gen_image_prompts)
        if template.quick_gen_clothing:
            data["quick_gen_clothing"] = _ser_lp(template.quick_gen_clothing)
        if template.quick_gen_poses:
            data["quick_gen_poses"] = _ser_lp(template.quick_gen_poses)
        if template.quick_gen_insertions:
            data["quick_gen_insertions"] = _ser_lp(template.quick_gen_insertions)
        if template.quick_gen_expressions:
            data["quick_gen_expressions"] = _ser_lp(template.quick_gen_expressions)
        if template.quick_gen_lora_keywords:
            data["quick_gen_lora_keywords"] = template.quick_gen_lora_keywords
        if template.auto_direct_system_prompt:
            data["auto_direct_system_prompt"] = template.auto_direct_system_prompt
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        template.file_path = str(path)
        logger.info("Saved template: %s", path)
        return True
    except Exception as exc:
        logger.warning("Failed to save template: %s", exc)
        return False
