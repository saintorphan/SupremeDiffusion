"""Wan 2.2 LoRA preset discovery and auto-configuration.

When the user picks "Lightning 6-step" or "SVI Pro" in the UI, this module:
- Scans the configured ``lora_dir`` (recursive) for files matching the preset's
  filename patterns.
- Returns the matching HIGH + LOW filenames + recommended multipliers.
- Recommends sampler / step-count / CFG adjustments compatible with the preset.

The actual LoRA loading is handled by the existing ``LoRAManager.apply_loras()``
in ``models/lora.py`` — diffusers' PEFT integration automatically routes HIGH
LoRAs to ``pipe.transformer`` and LOW LoRAs to ``pipe.transformer_2`` via key
matching. No expert-routing code is needed here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# UI preset values — keep in sync with generate.py dropdowns.
ACCELERATION_PRESETS = ("off", "lightning_4step", "lightning_6step")
ANTIDRIFT_PRESETS = ("off", "svi_pro")

ACCELERATION_LABELS = {
    "off": "Off (default — 25-step)",
    "lightning_4step": "Lightning 4-step (fastest)",
    "lightning_6step": "Lightning 6-step (recommended)",
}
ANTIDRIFT_LABELS = {
    "off": "Off",
    "svi_pro": "SVI Pro (anti-drift)",
}


def _all_loras(lora_dir: str) -> list[Path]:
    """Return all *.safetensors files under lora_dir (recursive)."""
    p = Path(lora_dir)
    if not p.is_dir():
        return []
    return sorted(p.rglob("*.safetensors"))


def _best_match(files: list[Path], required: tuple[str, ...]) -> Optional[Path]:
    """Pick the file whose lowercase name contains every hint in ``required``.

    Returns ``None`` if no file matches all hints.

    Tie-break (multiple matches): a bare "shortest filename" heuristic can pick
    the wrong HIGH/LOW when an unrelated short file also happens to contain the
    hints. Instead we rank by:
      1. token-boundary hits — hints flanked by non-alphanumeric separators
         (``_``/``-``/``.``) are much stronger signals than mid-word substrings,
      2. then shortest filename as the final, deterministic tie-break.
    """
    matches = []
    for f in files:
        name = f.name.lower()
        if all(h.lower() in name for h in required):
            matches.append(f)
    if not matches:
        return None

    def _boundary_hits(name: str) -> int:
        score = 0
        for h in required:
            hl = h.lower()
            start = 0
            while True:
                idx = name.find(hl, start)
                if idx < 0:
                    break
                before = name[idx - 1] if idx > 0 else "_"
                after_i = idx + len(hl)
                after = name[after_i] if after_i < len(name) else "_"
                if not before.isalnum() and not after.isalnum():
                    score += 1
                    break
                start = idx + 1
        return score

    # Highest boundary score wins; ties broken by shortest filename.
    return max(
        matches,
        key=lambda f: (_boundary_hits(f.name.lower()), -len(f.name)),
    )


def find_lightning_loras(lora_dir: str) -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(high_path, low_path)`` for Wan 2.2 Lightning LoRAs in lora_dir.

    Either may be None if not found.
    """
    files = _all_loras(lora_dir)
    high = _best_match(files, ("lightning", "high"))
    low = _best_match(files, ("lightning", "low"))
    return high, low


def find_svi_pro_loras(lora_dir: str) -> tuple[Optional[Path], Optional[Path]]:
    """Return ``(high_path, low_path)`` for SVI Pro LoRAs in lora_dir.

    Prefers v2 PRO rank-128 fp16 variants when multiple SVI files match, then
    falls back to plain ``SVI ... HIGH/LOW`` naming — many distributions ship
    the anti-drift LoRAs as e.g. ``SVI_HIGH.safetensors`` without a literal
    ``pro`` token, which the PRO-only patterns would silently skip.
    """
    files = _all_loras(lora_dir)
    # Most-specific → least-specific match attempts, first hit wins per phase.
    high = (
        _best_match(files, ("svi", "pro", "high", "rank_128"))
        or _best_match(files, ("svi", "pro", "high"))
        or _best_match(files, ("svi", "high"))
    )
    low = (
        _best_match(files, ("svi", "pro", "low", "rank_128"))
        or _best_match(files, ("svi", "pro", "low"))
        or _best_match(files, ("svi", "low"))
    )
    return high, low


# ---------------------------------------------------------------------------
# Preset → (lora list, multipliers, sampler hints) resolver
# ---------------------------------------------------------------------------


def resolve_preset(
    *,
    acceleration: str = "off",
    antidrift: str = "off",
    lora_dir: str,
    user_loras: list[str] | None = None,
    user_multipliers: str = "",
) -> dict:
    """Combine acceleration + antidrift presets with the user's existing LoRA
    selections.

    Returns a dict::

        {
            "loras":         ["fileA.safetensors", "fileB...", ...],
            "multipliers":   "0.5,1.0,1.0,1.0,...",   # comma-separated
            "num_steps":     int or None,   # None = leave unchanged
            "guidance_scale": float or None,
            "sampler":       str or None,   # recommended sampler
            "warnings":      list[str],     # things the user should know
        }

    The lora_dir is scanned ONCE; missing files produce warnings, not errors.
    """
    user_loras = list(user_loras or [])
    out_loras: list[str] = []
    out_mults: list[float] = []
    warnings: list[str] = []

    # Per-LoRA strength conventions matched to upstream SVI workflow.
    LIGHTNING_HIGH_STRENGTH = 0.5
    LIGHTNING_LOW_STRENGTH = 1.0
    SVI_HIGH_STRENGTH = 1.0
    SVI_LOW_STRENGTH = 1.0

    # --- Acceleration (Lightning) -------------------------------------
    if acceleration in ("lightning_4step", "lightning_6step"):
        high, low = find_lightning_loras(lora_dir)
        if high is None or low is None:
            warnings.append(
                f"Lightning LoRA(s) not found in {lora_dir} — preset disabled. "
                "Expected files matching 'Lightning ... HIGH/LOW'."
            )
        else:
            out_loras += [high.name, low.name]
            out_mults += [LIGHTNING_HIGH_STRENGTH, LIGHTNING_LOW_STRENGTH]

    # --- Anti-drift (SVI Pro) -----------------------------------------
    if antidrift == "svi_pro":
        high, low = find_svi_pro_loras(lora_dir)
        if high is None or low is None:
            warnings.append(
                f"SVI Pro LoRA(s) not found in {lora_dir} — preset disabled. "
                "Expected files matching 'SVI ... HIGH/LOW' "
                "(PRO / rank_128 variants preferred when present)."
            )
        else:
            out_loras += [high.name, low.name]
            out_mults += [SVI_HIGH_STRENGTH, SVI_LOW_STRENGTH]

    # --- Merge with user's existing selections ------------------------
    # Parse the user's existing multiplier string (comma-separated floats,
    # possibly with phase syntax like "1;0 0;1" which we preserve as-is).
    user_mult_list: list[str] = []
    has_phase = ";" in (user_multipliers or "")
    if user_multipliers:
        # If phase syntax, just split on commas of the top-level string —
        # individual entries may contain semicolons.
        user_mult_list = [m.strip() for m in user_multipliers.split(",") if m.strip()]

    # Pad user_mult_list with "1" if it's shorter than user_loras.
    while len(user_mult_list) < len(user_loras):
        user_mult_list.append("1")

    # De-duplicate: skip user_loras that the preset already added.
    preset_set = {n.lower() for n in out_loras}
    for ulora, umult in zip(user_loras, user_mult_list):
        if ulora.lower() in preset_set:
            continue
        out_loras.append(ulora)
        try:
            out_mults.append(float(umult))
        except ValueError:
            # Preserve phase syntax — drop to string later
            out_mults.append(umult)  # type: ignore[arg-type]

    # Build multiplier string. If any entry is non-numeric (phase), keep raw.
    mult_parts: list[str] = []
    for m in out_mults:
        if isinstance(m, (int, float)):
            mult_parts.append(f"{m:g}")
        else:
            mult_parts.append(str(m))
    mult_str = ",".join(mult_parts)

    # --- Sampler / step / CFG suggestions -----------------------------
    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    sampler: Optional[str] = None

    if acceleration == "lightning_4step":
        num_steps = 4
        guidance_scale = 1.5
        sampler = "euler/beta"
    elif acceleration == "lightning_6step":
        num_steps = 6
        guidance_scale = 1.5
        sampler = "euler/beta"

    return {
        "loras": out_loras,
        "multipliers": mult_str,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "sampler": sampler,
        "warnings": warnings,
    }
