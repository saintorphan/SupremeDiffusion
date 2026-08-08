"""Spatial upsampling (Lanczos) for Supreme Diffusion.

Operates on (N, H, W, 3) uint8 numpy arrays.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def spatial_upsample(
    frames: np.ndarray,
    mode: str,
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Upscale frames using Lanczos resampling.

    Parameters
    ----------
    frames:
        (N, H, W, 3) uint8 numpy array.
    mode:
        ``"lanczos1.5"`` for 1.5x or ``"lanczos2"`` for 2x.
    progress_cb:
        Optional callback(fraction, description).

    Returns
    -------
    Upscaled (N, H', W', 3) uint8 numpy array.
    """
    if mode == "lanczos1.5":
        scale = 1.5
    elif mode in ("lanczos2", "lanczos"):
        scale = 2.0
    else:
        logger.warning("Unknown spatial upsampling mode '%s', skipping.", mode)
        return frames

    n, h, w, c = frames.shape
    new_h = round(h * scale / 16) * 16
    new_w = round(w * scale / 16) * 16
    logger.info("Spatial upsample %s: %dx%d -> %dx%d (%d frames)", mode, w, h, new_w, new_h, n)

    out = np.empty((n, new_h, new_w, c), dtype=np.uint8)
    for i in range(n):
        img = Image.fromarray(frames[i])
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        out[i] = np.array(img)
        if progress_cb and (i % 10 == 0 or i == n - 1):
            progress_cb((i + 1) / n, f"Upscaling frame {i + 1}/{n}")

    return out
