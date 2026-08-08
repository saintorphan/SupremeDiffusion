"""LoRA Auto-Prompter — auto-parse .civitai.info and .json sidecars for trigger
words, recommended weights, and example prompts.

Ported from the Neo Forge lora-auto-prompter extension, adapted for
SupremeDiffusion's Quick tab.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML stripping helper
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def _strip_html(html_str: str) -> str:
    if not html_str:
        return ""
    s = _HTMLStripper()
    try:
        s.feed(html_str)
        return s.get_text()
    except Exception:
        return re.sub(r"<[^>]+>", " ", html_str).strip()


def _safe_json_load(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Sidecar file discovery
# ---------------------------------------------------------------------------

def _find_sidecar_files(lora_path: str) -> dict[str, str | None]:
    """Find .civitai.info, .json, and preview sidecar files."""
    base = os.path.splitext(lora_path)[0]
    result: dict[str, str | None] = {"civitai_info": None, "json": None, "preview": None}

    for pattern in [f"{base}.civitai.info", f"{base}_civitai.info"]:
        if os.path.isfile(pattern):
            result["civitai_info"] = pattern
            break

    json_path = f"{base}.json"
    if os.path.isfile(json_path):
        result["json"] = json_path

    for sep in [".", "_"]:
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            p = f"{base}{sep}preview{ext}"
            if os.path.isfile(p):
                result["preview"] = p
                break
        if result["preview"]:
            break

    return result


# ---------------------------------------------------------------------------
# Civitai info parser
# ---------------------------------------------------------------------------

def _parse_civitai_info(filepath: str) -> dict:
    data = _safe_json_load(filepath)
    if not data:
        return {}

    model_info = data.get("model", {}) or {}
    images = data.get("images", []) or []
    stats = data.get("stats", {}) or {}

    trained_words = data.get("trainedWords", []) or []
    trigger_words = [w.strip() for w in trained_words if w and w.strip()]

    example_prompts: list[str] = []
    negative_prompts: list[str] = []
    example_settings: dict = {}

    for img in images:
        meta = img.get("meta", {}) or {}
        prompt = meta.get("prompt", "")
        neg = meta.get("negativePrompt", "")
        if prompt and prompt not in example_prompts:
            example_prompts.append(prompt)
        if neg and neg not in negative_prompts:
            negative_prompts.append(neg)
        if not example_settings and meta:
            settings: dict = {}
            if meta.get("sampler"):
                settings["sampler"] = meta["sampler"]
            if meta.get("steps"):
                settings["steps"] = int(meta["steps"])
            if meta.get("cfgScale"):
                settings["cfg_scale"] = float(meta["cfgScale"])
            if settings:
                example_settings = settings

    tags = model_info.get("tags", []) or []

    return {
        "display_name": model_info.get("name", ""),
        "base_model": data.get("baseModel", ""),
        "trigger_words": trigger_words,
        "example_prompts": example_prompts,
        "negative_prompts": negative_prompts,
        "description": _strip_html(
            model_info.get("description", "") or data.get("description", "")
        ),
        "tags": tags,
        "stats": {
            "downloads": stats.get("downloadCount", 0),
            "thumbs_up": stats.get("thumbsUpCount", 0),
        },
        "example_settings": example_settings,
    }


# ---------------------------------------------------------------------------
# JSON sidecar parser (sd_civitai_helper format)
# ---------------------------------------------------------------------------

def _parse_json_sidecar(filepath: str) -> dict:
    data = _safe_json_load(filepath)
    if not data:
        return {}

    ext_info = data.get("extensions", {}).get("sd_civitai_helper", {})
    if ext_info.get("skeleton_file", False):
        return {}

    activation = data.get("activation text", "")
    trigger_words = [w.strip() for w in activation.split(",") if w.strip()] if activation else []

    weight = data.get("preferred weight", 0)
    if not isinstance(weight, (int, float)):
        weight = 0

    return {
        "description": _strip_html(data.get("description", "")),
        "trigger_words": trigger_words,
        "preferred_weight": float(weight) if weight > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Weight mining from metadata
# ---------------------------------------------------------------------------

_LORA_REF_RE = re.compile(r"<lora:[^:]+:([\d.]+)>")
_DESC_WEIGHT_RE = re.compile(
    r"(?:recommend(?:ed)?|suggest(?:ed)?|use|set|try|optimal|best)"
    r"[\s:]*(?:weight|strength|value)?[\s:]*(?:at|of|to|around|~|is)?[\s:]*"
    r"(\d+\.?\d*(?:\s*[-\u2013~]+\s*\d+\.?\d*)?)",
    re.IGNORECASE,
)
_SIMPLE_WEIGHT_RE = re.compile(
    r"(?:weight|strength)[\s:]+(\d+\.?\d*(?:\s*[-\u2013~]\s*\d+\.?\d*)?)",
    re.IGNORECASE,
)


def _extract_weight_from_metadata(
    example_prompts: list[str],
    description: str,
    trigger_words: list[str],
    default: float,
) -> float:
    weights: list[float] = []

    for prompt in example_prompts or []:
        for m in _LORA_REF_RE.finditer(prompt):
            try:
                w = float(m.group(1))
                if 0.1 <= w <= 2.0:
                    weights.append(w)
            except ValueError:
                pass

    for tw in trigger_words or []:
        for m in _LORA_REF_RE.finditer(tw):
            try:
                w = float(m.group(1))
                if 0.1 <= w <= 2.0:
                    weights.append(w)
            except ValueError:
                pass

    desc = description or ""
    for pattern in [_DESC_WEIGHT_RE, _SIMPLE_WEIGHT_RE]:
        for m in pattern.finditer(desc):
            raw = m.group(1)
            parts = re.split(r"[-\u2013~]|to", raw)
            for p in parts:
                try:
                    w = float(p.strip())
                    if 0.1 <= w <= 2.0:
                        weights.append(w)
                except ValueError:
                    pass

    if not weights:
        return default
    weights.sort()
    return weights[len(weights) // 2]


# ---------------------------------------------------------------------------
# Trigger word analysis — intensity stages & classification
# ---------------------------------------------------------------------------

INTENSITY_KEYWORDS = [
    (["partial", "slight", "mild", "light", "early", "beginning", "small", "mini"], 0.2),
    (["moderate", "medium", "normal", "standard", "regular"], 0.4),
    (["full", "complete", "large", "big", "heavy", "deep", "strong"], 0.6),
    (["hyper", "extreme", "massive", "huge", "mega", "ultra", "enormous", "giant"], 0.8),
    (["maximum", "absolute", "peak", "overwhelming", "transcendent", "impossible"], 1.0),
]

MODIFIER_PATTERNS = [
    "skin", "color", "colored", "multicolored",
    "through clothes", "inward", "glow", "shimmer",
    "wet", "dripping", "torn", "tearing",
    "shiny", "matte", "translucent", "opaque",
]

_BODY_WORDS = {"breast", "thigh", "hip", "hair", "eyes", "skin"}
_CLOTHING_WORDS = {
    "dress", "shirt", "pants", "skirt", "outfit", "uniform",
    "bikini", "lingerie", "stockings", "boots", "shoes",
    "hat", "glasses", "armor", "costume", "underwear",
    "bra", "panties", "fishnet", "latex", "leather",
    "corset", "bodysuit", "leotard", "swimsuit",
}
_GENERIC_TAGS = {
    "1girl", "2girls", "1boy", "solo",
    "standing", "sitting", "lying", "kneeling",
    "smile", "blush", "looking at viewer",
    "indoors", "outdoors", "simple background",
    "masterpiece", "best quality", "highres",
}


def _is_generic_tag(tag: str) -> bool:
    t = tag.lower().strip()
    if t in _GENERIC_TAGS:
        return True
    for w in _BODY_WORDS:
        if w in t:
            return True
    return False


def analyze_triggers(trigger_words: list[str]) -> dict:
    """Classify trigger words into core, stages, modifiers, and descriptors."""
    if not trigger_words:
        return {"core": None, "stages": [], "modifiers": [], "descriptors": [],
                "has_stages": False}

    clean: list[str] = []
    for tw in trigger_words:
        for sub in tw.split(","):
            sub = sub.strip().strip("|").strip()
            if sub and len(sub) > 1 and sub not in ("()", ">", "||"):
                if not sub.startswith("<lora:"):
                    clean.append(sub)

    seen: set[str] = set()
    unique: list[str] = []
    for t in clean:
        key = t.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(t)

    if not unique:
        return {"core": None, "stages": [], "modifiers": [], "descriptors": [],
                "has_stages": False}

    stages: list[tuple[str, float]] = []
    modifiers: list[str] = []
    descriptors: list[str] = []
    core_candidates: list[str] = []

    for trig in unique:
        trig_lower = trig.lower()

        is_modifier = any(mp in trig_lower for mp in MODIFIER_PATTERNS)
        if is_modifier:
            modifiers.append(trig)
            continue

        matched_intensity = None
        for keywords, level in INTENSITY_KEYWORDS:
            if any(kw in trig_lower for kw in keywords):
                matched_intensity = level
                break

        if matched_intensity is not None:
            stages.append((trig, matched_intensity))
        elif _is_generic_tag(trig):
            descriptors.append(trig)
        else:
            core_candidates.append(trig)

    stages.sort(key=lambda x: x[1])
    core = core_candidates[0] if core_candidates else None

    return {
        "core": core,
        "stages": stages,
        "modifiers": modifiers,
        "descriptors": descriptors,
        "has_stages": len(stages) >= 2,
    }


def select_triggers_for_intensity(
    trigger_analysis: dict,
    intensity: float = 0.5,
    include_modifiers: bool = True,
    base_weight: float | None = None,
) -> tuple[list[str], float]:
    """Pick triggers and LoRA weight for a desired intensity (0.0-1.0)."""
    selected: list[str] = []

    if trigger_analysis["core"]:
        selected.append(trigger_analysis["core"])

    stages = trigger_analysis["stages"]
    if stages:
        best = min(stages, key=lambda s: abs(s[1] - intensity))
        selected.append(best[0])
        if intensity >= 0.6:
            for stage_trigger, stage_level in stages:
                if stage_level <= intensity and stage_trigger not in selected:
                    selected.append(stage_trigger)

    for desc in trigger_analysis["descriptors"]:
        selected.append(desc)

    if include_modifiers:
        for mod in trigger_analysis["modifiers"]:
            selected.append(mod)

    if base_weight is None or base_weight <= 0:
        base_weight = 0.7
    scale = 0.6 + (intensity * 0.7)
    weight = round(min(1.5, max(0.2, base_weight * scale)), 2)

    return selected, weight


def get_intensity_label(intensity: float) -> str:
    if intensity <= 0.15:
        return "Subtle"
    if intensity <= 0.35:
        return "Mild"
    if intensity <= 0.55:
        return "Moderate"
    if intensity <= 0.75:
        return "Strong"
    if intensity <= 0.9:
        return "Extreme"
    return "Maximum"


# ---------------------------------------------------------------------------
# Unified LoRA metadata parser
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def parse_lora_metadata(lora_path: str) -> dict:
    """Parse all available sidecar metadata for a LoRA file.

    Returns a dict with: display_name, trigger_words, example_prompts,
    negative_prompts, recommended_weight, description, tags, has_metadata,
    preview_path, etc.
    """
    filename = os.path.basename(lora_path)
    sidecars = _find_sidecar_files(lora_path)

    civitai_data: dict = {}
    json_data: dict = {}
    sources: list[str] = []

    if sidecars["civitai_info"]:
        civitai_data = _parse_civitai_info(sidecars["civitai_info"])
        if civitai_data:
            sources.append("civitai")

    if sidecars["json"]:
        json_data = _parse_json_sidecar(sidecars["json"])
        if json_data:
            sources.append("json")

    # Merge trigger words (deduplicated, case-insensitive)
    all_triggers = civitai_data.get("trigger_words", []) + json_data.get("trigger_words", [])
    seen_lower: set[str] = set()
    trigger_words: list[str] = []
    for t in all_triggers:
        low = t.lower().strip()
        if low and low not in seen_lower:
            seen_lower.add(low)
            trigger_words.append(t.strip())

    # Display name
    display_name = civitai_data.get("display_name", "") or Path(filename).stem

    # Weight — mine from all sources
    json_weight = json_data.get("preferred_weight", 0)
    if json_weight > 0:
        recommended_weight = json_weight
    else:
        recommended_weight = _extract_weight_from_metadata(
            civitai_data.get("example_prompts", []),
            civitai_data.get("description", ""),
            trigger_words,
            0.7,
        )

    return {
        "filename": filename,
        "full_path": lora_path,
        "display_name": display_name,
        "trigger_words": trigger_words,
        "example_prompts": civitai_data.get("example_prompts", []),
        "negative_prompts": civitai_data.get("negative_prompts", []),
        "base_model": civitai_data.get("base_model", ""),
        "recommended_weight": recommended_weight,
        "description": civitai_data.get("description", "") or json_data.get("description", ""),
        "tags": civitai_data.get("tags", []),
        "has_metadata": len(sources) > 0,
        "metadata_sources": sources,
        "stats": civitai_data.get("stats", {}),
        "example_settings": civitai_data.get("example_settings", {}),
        "preview_path": sidecars.get("preview"),
    }


def auto_lora_prompt_tag(
    lora_path: str,
    intensity: float = 0.5,
) -> tuple[str, dict]:
    """Build a smart prompt tag for a LoRA using parsed sidecar metadata.

    Returns (tag_string, metadata_dict).
    The tag_string is e.g. ``<lora:name:0.7> core_trigger stage_trigger``
    """
    meta = parse_lora_metadata(lora_path)
    stem = Path(lora_path).stem

    if not meta["has_metadata"] or not meta["trigger_words"]:
        # Fallback: basic tag with no triggers
        weight = meta["recommended_weight"]
        return f"<lora:{stem}:{weight}>", meta

    analysis = analyze_triggers(meta["trigger_words"])
    base_weight = meta["recommended_weight"]

    if analysis["has_stages"]:
        triggers, weight = select_triggers_for_intensity(
            analysis, intensity,
            include_modifiers=(intensity >= 0.4),
            base_weight=base_weight,
        )
    else:
        # No stages — use all triggers, recommended weight scaled by intensity
        triggers = []
        if analysis["core"]:
            triggers.append(analysis["core"])
        triggers.extend(t for t, _ in analysis["stages"])
        triggers.extend(analysis["descriptors"])
        if intensity >= 0.4:
            triggers.extend(analysis["modifiers"])
        # Scale weight by intensity
        scale = 0.6 + (intensity * 0.7)
        weight = round(min(1.5, max(0.2, base_weight * scale)), 2)

    trigger_text = ", ".join(triggers)
    tag = f"<lora:{stem}:{weight}>"
    if trigger_text:
        tag += f" {trigger_text}"

    return tag, meta
