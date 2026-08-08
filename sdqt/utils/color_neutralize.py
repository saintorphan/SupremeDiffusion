"""Color drift neutralization for frames extracted from prior generations.

LTX 2.3 / Wan2.2 in mixed-precision modes produce a slight per-frame warm
bias. When you extract a frame from a generated clip and feed it back as
the conditioning input for the next clip, that bias is part of the
conditioning — and the next generation amplifies it. After N chunks the
output is visibly orange.

This module breaks the compound chain by recolor-balancing extracted
frames before they become conditioning inputs.

Two strategies, picked by whether a reference frame is available:

* **Anchor-to-reference**: compute per-channel mean ratio between the
  current frame and a stored reference, apply as gain. Preserves
  intentional color (sunset stays orange) — only corrects when the
  current frame's channel-mean has *grown beyond* the reference's.
* **Gray-world**: assume the scene should average to neutral grey. Scale
  channels so their means equalize. Used as fallback when no reference
  is configured. Heavier-handed; can over-correct intentionally warm
  scenes.

Hooked into all "extract frame to feed forward" call sites
(``_capture_last_frame_path``, ``capture_frame``,
``_extract_and_send_final_frame``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _channel_means(arr: np.ndarray) -> np.ndarray:
    """Per-channel mean of an HxWx3 float array in [0, 1]. Guards near-black."""
    means = arr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    return np.maximum(means, 1e-3)


def neutralize_image(
    img: Image.Image,
    reference_path: str | None = None,
    strength: float = 1.0,
    *,
    gain_clip: tuple[float, float] = (0.7, 1.5),
) -> Image.Image:
    """Return a color-neutralized copy of *img*.

    Args:
        img: Source frame (any mode; converted to RGB internally).
        reference_path: Optional path to a reference image. When valid, runs
            anchor-to-reference. When ``None`` / missing / unreadable, falls
            back to gray-world.
        strength: 0..1 blend between original (0) and neutralized (1).
        gain_clip: Per-channel gain bounds. Prevents extreme corrections on
            near-monochromatic frames (e.g. a deep red sunset shouldn't get
            forced to neutral). Defaults to ``(0.7, 1.5)``.

    Returns:
        New PIL Image in RGB mode. Original is not modified.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return img.convert("RGB") if img.mode != "RGB" else img.copy()

    src = img.convert("RGB") if img.mode != "RGB" else img
    arr = np.asarray(src, dtype=np.float32) / 255.0  # H×W×3, [0, 1]
    cur_means = _channel_means(arr)

    ref_means: np.ndarray | None = None
    if reference_path:
        ref_path = Path(reference_path)
        if ref_path.is_file():
            try:
                with Image.open(ref_path) as ref_img:
                    ref_arr = np.asarray(
                        ref_img.convert("RGB"), dtype=np.float32
                    ) / 255.0
                ref_means = _channel_means(ref_arr)
            except Exception as exc:
                logger.warning(
                    "Neutralize: failed to read reference %s (%s); "
                    "falling back to gray-world.",
                    ref_path, exc,
                )

    if ref_means is None:
        # Gray-world: target each channel to the average of all three.
        target = float(cur_means.mean())
        ref_means = np.array([target, target, target], dtype=np.float32)

    gain = ref_means / cur_means
    gain = np.clip(gain, gain_clip[0], gain_clip[1])

    neutralized = np.clip(arr * gain, 0.0, 1.0)
    if strength < 1.0:
        neutralized = arr * (1.0 - strength) + neutralized * strength

    out_u8 = (neutralized * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(out_u8, mode="RGB")


def neutralize_image_path(
    src_path: str,
    out_path: str | None = None,
    *,
    reference_path: str | None = None,
    strength: float = 1.0,
) -> str:
    """Read *src_path*, neutralize, write back (or to *out_path*). Returns the path written.

    Convenience wrapper for call sites that already have a temp file path
    rather than a PIL Image. If *out_path* is omitted, overwrites *src_path*.
    """
    target = out_path or src_path
    with Image.open(src_path) as img:
        result = neutralize_image(
            img, reference_path=reference_path, strength=strength,
        )
    result.save(target)
    return target


def save_as_neutralizer_reference(
    src: Image.Image | str,
    global_config,
) -> str:
    """Save *src* as the project's neutralizer reference image and persist.

    Used by the "Set as Neutralizer Reference" context menu entries on
    timeline clips and the video player. Writes a stable PNG to
    ``~/.supremediffusion/neutralizer_reference.png`` (overwriting any
    previous reference), updates ``global_config.neutralize_reference_path``,
    and calls ``global_config.save()`` so it survives a restart.

    Args:
        src: Either a PIL ``Image`` or a path to an existing image file.
        global_config: ``GlobalConfig`` instance to update + save.

    Returns:
        Absolute path to the written reference image.
    """
    ref_dir = Path.home() / ".supremediffusion"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / "neutralizer_reference.png"

    if isinstance(src, (str, Path)):
        with Image.open(str(src)) as opened:
            opened.convert("RGB").save(ref_path)
    else:
        src.convert("RGB").save(ref_path)

    global_config.neutralize_reference_path = str(ref_path)
    try:
        global_config.save()
    except Exception:
        logger.warning("Failed to persist neutralizer reference path to config",
                       exc_info=True)
    return str(ref_path)


def neutralize_if_enabled(
    img: Image.Image,
    global_config,
) -> Image.Image:
    """Apply neutralization if it's enabled in global config; otherwise pass through.

    Reads ``neutralize_frames_enabled``, ``neutralize_reference_path``, and
    ``neutralize_strength`` from *global_config*. Safe to call with any
    config object — missing attributes are treated as disabled / defaults.
    """
    if not getattr(global_config, "neutralize_frames_enabled", False):
        return img
    ref_path = getattr(global_config, "neutralize_reference_path", "") or ""
    strength = float(getattr(global_config, "neutralize_strength", 1.0) or 1.0)
    return neutralize_image(img, reference_path=ref_path or None, strength=strength)
