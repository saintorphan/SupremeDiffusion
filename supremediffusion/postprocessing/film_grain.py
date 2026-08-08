"""Film grain effect for Supreme Diffusion.

Adapted from wan2gp (via Lightricks/ComfyUI-LTXVideo).
Operates on (N, H, W, 3) uint8 numpy arrays.
"""

from __future__ import annotations

import numpy as np
import torch


def add_film_grain(
    frames: np.ndarray,
    intensity: float,
    saturation: float = 0.5,
) -> np.ndarray:
    """Apply synthetic film grain to frames.

    Parameters
    ----------
    frames:
        (N, H, W, 3) uint8 numpy array.
    intensity:
        Grain strength (0 = none, typical range 0.01–0.15).
    saturation:
        Color vs monochrome grain (0 = grayscale, 1 = full color).

    Returns
    -------
    (N, H, W, 3) uint8 numpy array with grain applied.
    """
    if intensity <= 0:
        return frames

    # Work in float [0, 1] with NHWC layout
    t = torch.from_numpy(frames).float().div_(255.0)

    grain = torch.randn_like(t)
    # Channel-specific weighting: R=2x, G=1x, B=3x (film-like color cast)
    grain[:, :, :, 0] *= 2
    grain[:, :, :, 2] *= 3
    # Blend colored and grayscale grain
    gray_grain = grain[:, :, :, 1].unsqueeze(3).expand_as(grain)
    grain = grain * saturation + gray_grain * (1 - saturation)

    t = (t + intensity * grain).clamp_(0, 1)
    return t.mul_(255).to(torch.uint8).numpy()
