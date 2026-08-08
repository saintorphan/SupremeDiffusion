"""Unified color anchor post-processing.

Wraps :func:`supremediffusion.utils.color_correct.apply_color_match` with:

- **Multi-mode reference selection** — pick the anchor reference per-clip from
  the original source, a fixed user-supplied path, a per-window rotating list,
  or fall back to the previous chunk's last frame (legacy behavior).
- **Configurable persistence** — control how much of the chunk gets corrected
  instead of fading to a baseline within ~1 second (the legacy behavior, which
  let the VAE warm bias re-accumulate).

The actual color math (LAB mean-shift, HM-MVGD-HM, etc.) is unchanged — those
algorithms are correctly implemented in ``color_correct.py``. This module fixes
the **wiring** that was broken: the post-anchor was only firing on Video
Extender, never on Mode 1/2/3, and the dead ``anchor_image_mode`` /
``anchor_image_path`` config keys had no backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# Anchor modes — keep in sync with PROJECT_DEFAULTS and the Generate tab UI.
ANCHOR_MODES = (
    "off",            # No color anchoring
    "start",          # Anchor every chunk to the original source frame
    "single",         # Anchor every chunk to anchor_image_path
    "per_window",     # Rotate through ';'-separated paths in anchor_image_path
    "previous_last",  # Legacy: anchor to the previous chunk's last frame
)


def resolve_reference(
    *,
    mode: str,
    source_image: Optional[Image.Image] = None,
    anchor_image_path: str = "",
    chunk_index: int = 0,
    previous_last_frame: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """Pick the reference image for this chunk based on the configured mode.

    Returns ``None`` to mean "no anchoring" — caller should pass-through.
    """
    if mode == "off" or not mode:
        return None
    if mode == "start":
        return source_image
    if mode == "single":
        if anchor_image_path and Path(anchor_image_path).is_file():
            try:
                return Image.open(anchor_image_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("anchor_image_path %s unreadable: %s", anchor_image_path, exc)
                return source_image
        return source_image
    if mode == "per_window":
        if not anchor_image_path:
            return source_image
        paths = [p.strip() for p in anchor_image_path.split(";") if p.strip()]
        if not paths:
            return source_image
        chosen = paths[chunk_index % len(paths)]
        if Path(chosen).is_file():
            try:
                return Image.open(chosen).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("per_window path %s unreadable: %s", chosen, exc)
        return source_image
    if mode == "previous_last":
        return previous_last_frame if previous_last_frame is not None else source_image
    logger.warning("Unknown anchor mode %r — defaulting to 'start'", mode)
    return source_image


def apply_color_anchor(
    frames: Sequence[Image.Image],
    reference: Optional[Image.Image],
    *,
    method: str = "mean-only-lab",
    strength: float = 0.8,
    persistence: float = 0.6,
    fade_frames: int = 16,
) -> List[Image.Image]:
    """Apply color anchor to ``frames`` against ``reference``.

    Args:
        frames: input frames (RGB PIL.Image, any size).
        reference: the anchor frame. ``None`` = pass-through.
        method: color-match algorithm — see :data:`COLOR_MATCH_METHODS` in
            ``color_correct.py``. Default ``mean-only-lab`` is the drift-
            resistant default; ``hm-mvgd-hm`` for strong single-shot match.
        strength: peak correction strength (0–1). Applied to frame 0.
        persistence: baseline strength held across all frames after the fade
            zone. The legacy ``_seam_anchor_extension`` used 0.3 here and the
            VAE warm bias would re-accumulate by frame ~24. Default 0.6 holds
            stronger correction throughout the clip.
        fade_frames: number of frames to fade from ``strength`` (frame 0) down
            to ``persistence * strength`` (frame ``fade_frames``).

    Returns:
        New list of PIL.Image frames, same length as input.
    """
    if not frames:
        return list(frames)
    if reference is None:
        return list(frames)
    if strength <= 0.0:
        return list(frames)

    from supremediffusion.utils.color_correct import apply_color_match

    ref_np = np.asarray(reference.convert("RGB")).astype(np.float32) / 255.0
    out: List[Image.Image] = []
    fade_n = max(int(fade_frames), 1)
    floor_strength = strength * float(persistence)

    for i, frame in enumerate(frames):
        if i < fade_n:
            # Linear fade from `strength` (i=0) to `floor_strength` (i=fade_n).
            t = i / fade_n
            s = strength * (1.0 - t) + floor_strength * t
        else:
            s = floor_strength
        if s <= 0.0:
            out.append(frame)
            continue
        tgt_np = np.asarray(frame.convert("RGB")).astype(np.float32) / 255.0
        try:
            matched = apply_color_match(tgt_np, ref_np, method=method, strength=s)
        except Exception as exc:  # noqa: BLE001
            logger.warning("color anchor frame %d failed: %s — pass-through", i, exc)
            out.append(frame)
            continue
        arr = np.clip(matched * 255.0, 0, 255).astype(np.uint8)
        out.append(Image.fromarray(arr))

    logger.info(
        "color anchor: %d frames (method=%s strength=%.2f persistence=%.2f fade=%d)",
        len(out), method, strength, persistence, fade_n,
    )
    return out


def anchor_frames_for_chunk(
    frames: Sequence[Image.Image],
    *,
    config,
    source_image: Optional[Image.Image] = None,
    chunk_index: int = 0,
    previous_last_frame: Optional[Image.Image] = None,
) -> List[Image.Image]:
    """High-level helper — reads anchor mode/strength/persistence/method from
    ``config`` (ProjectConfig) and dispatches to :func:`apply_color_anchor`.

    This is what pipeline.py call sites should call after decode. One line.
    """
    mode = (getattr(config, "anchor_image_mode", "start") or "start").strip().lower()
    if mode == "off":
        return list(frames)

    reference = resolve_reference(
        mode=mode,
        source_image=source_image,
        anchor_image_path=getattr(config, "anchor_image_path", "") or "",
        chunk_index=chunk_index,
        previous_last_frame=previous_last_frame,
    )
    if reference is None:
        return list(frames)

    method = getattr(config, "color_correction_method", "mean-only-lab") or "mean-only-lab"
    # Two strength sources live in the config — color_correction_strength is
    # the per-chunk post-anchor (default 0.3 in the legacy code). The new
    # color_anchor_persistence controls how much of that strength holds across
    # the whole chunk.
    #
    # No ``or 0.8`` truthiness fallback here — callers gate on
    # ``color_correction_strength > 0`` before invoking this helper, and the
    # SVI path threads the slider value through (set on win_cfg). An explicit
    # 0.0 must mean 0.0, not silently snap back to the default.
    base_strength = float(getattr(config, "color_correction_strength", 0.8))
    persistence = float(getattr(config, "color_anchor_persistence", 0.6) or 0.6)

    return apply_color_anchor(
        frames, reference,
        method=method, strength=base_strength, persistence=persistence,
    )
