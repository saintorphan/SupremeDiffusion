"""Effect registry, helpers, and legacy migration for timeline clip effects."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Effect type registry
# ---------------------------------------------------------------------------

# Each entry: (label, {param_name: (min, max, default, decimals, step, suffix)})
EFFECT_REGISTRY: dict[str, tuple[str, dict]] = {
    "hue_sat": (
        "Hue / Saturation",
        {
            "hue_shift": (-180.0, 180.0, 0.0, 1, 5.0, "°"),
            "saturation": (0.0, 3.0, 1.0, 2, 0.05, ""),
        },
    ),
    "brightness_contrast": (
        "Brightness / Contrast",
        {
            "brightness": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "contrast": (0.0, 3.0, 1.0, 2, 0.05, ""),
        },
    ),
    "levels": (
        "Levels",
        {
            "black_point": (0.0, 255.0, 0.0, 0, 1.0, ""),
            "white_point": (0.0, 255.0, 255.0, 0, 1.0, ""),
            "gamma": (0.1, 3.0, 1.0, 2, 0.05, ""),
        },
    ),
    "shadows_highlights": (
        "Shadows / Highlights",
        {
            "shadows": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "highlights": (-1.0, 1.0, 0.0, 2, 0.05, ""),
        },
    ),
    "color_balance": (
        "Color Balance",
        {
            "shadow_r": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "shadow_g": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "shadow_b": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "midtone_r": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "midtone_g": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "midtone_b": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "highlight_r": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "highlight_g": (-1.0, 1.0, 0.0, 2, 0.05, ""),
            "highlight_b": (-1.0, 1.0, 0.0, 2, 0.05, ""),
        },
    ),
    "color_temperature": (
        "Color Temperature",
        {
            "temperature": (-100.0, 100.0, 0.0, 0, 5.0, ""),
            "tint": (-100.0, 100.0, 0.0, 0, 5.0, ""),
        },
    ),
    "channel_mixer": (
        "Channel Mixer",
        {
            "rr": (0.0, 3.0, 1.0, 3, 0.01, ""),
            "gg": (0.0, 3.0, 1.0, 3, 0.01, ""),
            "bb": (0.0, 3.0, 1.0, 3, 0.01, ""),
        },
    ),
    "vibrance": (
        "Vibrance",
        {
            "vibrance": (0.0, 2.0, 1.0, 2, 0.05, ""),
        },
    ),
    "sharpen": (
        "Sharpen",
        {
            "amount": (0.0, 10.0, 0.0, 1, 0.5, ""),
        },
    ),
    "blur": (
        "Blur",
        {
            "amount": (0.0, 10.0, 0.0, 1, 0.5, ""),
        },
    ),
    "denoise": (
        "Denoise",
        {
            "strength": (0.0, 10.0, 0.0, 1, 0.5, ""),
        },
    ),
    "vignette": (
        "Vignette",
        {
            "intensity": (0.0, 1.0, 0.0, 2, 0.05, ""),
        },
    ),
    "film_grain": (
        "Film Grain",
        {
            "intensity": (0.0, 1.0, 0.0, 2, 0.05, ""),
        },
    ),
    "crop_zoom": (
        "Crop / Zoom / Fit",
        {
            # Crop rect as fractions of source (0.0–1.0)
            "x": (0.0, 1.0, 0.0, 4, 0.01, ""),
            "y": (0.0, 1.0, 0.0, 4, 0.01, ""),
            "w": (0.01, 1.0, 1.0, 4, 0.01, ""),
            "h": (0.01, 1.0, 1.0, 4, 0.01, ""),
        },
    ),
    "speed": (
        "Speed",
        {
            "factor": (0.25, 4.0, 1.0, 2, 0.25, "×"),
        },
    ),
    "compress": (
        "Compress",
        {
            "crf": (0.0, 51.0, 23.0, 0, 1.0, ""),
            "scale": (0.1, 1.0, 1.0, 2, 0.1, ""),
        },
    ),
    "ai_enhance": (
        "AI Enhance",
        {
            "tile_size": (0.0, 1024.0, 512.0, 0, 128.0, "px"),
        },
    ),
    "lut3d": (
        "3D LUT",
        {},  # no numeric params — lut_file stored directly in params
    ),
}

# ---------------------------------------------------------------------------
# Effect categories for organised menus
# ---------------------------------------------------------------------------

EFFECT_CATEGORIES: dict[str, list[str]] = {
    "Color": [
        "hue_sat", "brightness_contrast", "levels", "shadows_highlights",
        "color_balance", "channel_mixer", "color_temperature", "vibrance",
        "lut3d",
    ],
    "Filter": ["sharpen", "blur", "denoise", "film_grain", "vignette"],
    "Transform": ["speed", "crop_zoom"],
    "Enhance": ["ai_enhance"],
    "Encode": ["compress"],
}

# Ordered list of type keys (controls "Add Effect" menu order)
EFFECT_TYPES = list(EFFECT_REGISTRY.keys())


def make_effect(effect_type: str, **overrides) -> dict:
    """Create a new effect dict with default params."""
    if effect_type not in EFFECT_REGISTRY:
        raise ValueError(f"Unknown effect type: {effect_type}")
    _, param_defs = EFFECT_REGISTRY[effect_type]
    params = {k: v[2] for k, v in param_defs.items()}  # index 2 = default
    params.update(overrides)
    # lut3d stores a file path, not numeric params
    if effect_type == "lut3d" and "lut_file" not in params:
        params["lut_file"] = ""
    return {"type": effect_type, "enabled": True, "params": params}


def effect_label(effect_type: str) -> str:
    """Return the human-readable label for an effect type."""
    entry = EFFECT_REGISTRY.get(effect_type)
    return entry[0] if entry else effect_type


def effect_is_identity(fx: dict) -> bool:
    """Return True if this effect is at default values (no visual change)."""
    etype = fx.get("type", "")
    if etype == "lut3d":
        return not fx.get("params", {}).get("lut_file", "")
    entry = EFFECT_REGISTRY.get(etype)
    if not entry:
        return True
    _, param_defs = entry
    params = fx.get("params", {})
    for k, meta in param_defs.items():
        default = meta[2]
        if abs(params.get(k, default) - default) > 1e-6:
            return False
    return True


def migrate_legacy_effects(effects_data) -> list[dict]:
    """Convert old flat dict format to new effect stack list.

    Old format: {"hue_shift": 15, "saturation": 1.2, "brightness": 0.1, "contrast": 1.0}
    New format: [{"type": "hue_sat", "enabled": true, "params": {...}}, ...]
    """
    # Already new format
    if isinstance(effects_data, list):
        return effects_data

    if not isinstance(effects_data, dict) or not effects_data:
        return []

    result = []

    # Migrate hue/saturation
    hue_shift = effects_data.get("hue_shift", 0.0)
    saturation = effects_data.get("saturation", 1.0)
    if hue_shift != 0.0 or saturation != 1.0:
        result.append(make_effect("hue_sat", hue_shift=hue_shift, saturation=saturation))

    # Migrate brightness/contrast
    brightness = effects_data.get("brightness", 0.0)
    contrast = effects_data.get("contrast", 1.0)
    if brightness != 0.0 or contrast != 1.0:
        result.append(make_effect("brightness_contrast", brightness=brightness, contrast=contrast))

    return result
