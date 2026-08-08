"""Laplacian pyramid blending for inpaint paste-back.

Multi-level frequency-domain blending eliminates the visible seam where an
inpainted region meets the surrounding pixels. Same technique Photoshop's
"healing brush" uses for panoramas.

Pure-torch implementation ported from Code2Collapse's ComfyUI-CustomNodePacks
inpaint_suite.py (Apache-2.0). The math is unchanged; we removed the ComfyUI-
specific node wrappers and stripped to a single ``laplacian_pyramid_blend``
function that takes PIL.Image / numpy / torch.Tensor input.

Usage::

    from PIL import Image
    from supremediffusion.utils.laplacian_blend import laplacian_pyramid_blend

    # background (original), foreground (inpainted result), and a mask
    result = laplacian_pyramid_blend(background, foreground, mask, levels=5)

Where:
- background, foreground: PIL.Image RGB at same size
- mask: PIL.Image L — 0=use background, 255=use foreground

Returns a new PIL.Image at the same size.
"""

from __future__ import annotations

import math
from typing import List

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # noqa
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # noqa
    np = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Gaussian kernel helpers
# ---------------------------------------------------------------------------


def _gauss_kernel_1d(sigma: float, device, dtype=None):
    """Normalized 1D Gaussian kernel."""
    if dtype is None:
        dtype = torch.float32
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    size = 2 * radius + 1
    x = torch.arange(size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur_2d(tensor, sigma: float):
    """Separable 2D Gaussian blur on (B, C, H, W). Pure torch — no cv2."""
    if sigma <= 0:
        return tensor
    device = tensor.device
    k1d = _gauss_kernel_1d(sigma, device, tensor.dtype)
    pad = len(k1d) // 2
    c = tensor.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    out = F.conv2d(F.pad(tensor, (pad, pad, 0, 0), mode="replicate"), kh, groups=c)
    kv = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    out = F.conv2d(F.pad(out, (0, 0, pad, pad), mode="replicate"), kv, groups=c)
    return out


# ---------------------------------------------------------------------------
# Laplacian pyramid
# ---------------------------------------------------------------------------


def _build_laplacian_pyramid(img, levels: int) -> List:
    """Build a Laplacian pyramid from (B, C, H, W) tensor.

    Returns list of tensors: ``levels`` Laplacian layers + 1 residual.
    Each Laplacian = current − upsample(downsample(current)).
    """
    pyramid: List = []
    current = img
    for _ in range(levels):
        h, w = current.shape[2], current.shape[3]
        down = _gaussian_blur_2d(current, sigma=1.0)
        down = F.interpolate(
            down, size=(max(1, h // 2), max(1, w // 2)),
            mode="bilinear", align_corners=False,
        )
        up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
        pyramid.append(current - up)
        current = down
    pyramid.append(current)  # residual
    return pyramid


def _reconstruct_from_pyramid(pyramid: List):
    """Collapse a Laplacian pyramid back to a single image tensor."""
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[2], pyramid[i].shape[3]
        up = F.interpolate(current, size=(h, w), mode="bilinear", align_corners=False)
        current = up + pyramid[i]
    return current


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def laplacian_pyramid_blend(
    background,  # PIL.Image | np.ndarray | torch.Tensor
    foreground,
    mask,
    *,
    levels: int = 5,
):
    """Blend ``foreground`` into ``background`` using ``mask``.

    Multi-level Laplacian pyramid decomposition + per-level blend +
    reconstruction. Hides the seam between foreground and background by
    blending each frequency band separately.

    Args:
        background: original image (kept where mask is 0).
        foreground: replacement content (used where mask is 1).
        mask: 0..1 / 0..255 grayscale — 0=keep bg, 1=use fg.
        levels: pyramid depth. 5 is the standard. Clamped to image size.

    Returns the blended image in the same format as the input (PIL.Image
    if any input was PIL.Image, otherwise the input format of background).
    """
    if torch is None:
        raise RuntimeError("torch required for Laplacian pyramid blending.")

    from PIL import Image

    return_pil = isinstance(background, Image.Image) or isinstance(foreground, Image.Image)

    bg_t = _to_tensor_4d(background)
    fg_t = _to_tensor_4d(foreground)
    mask_t = _to_mask_4d(mask)

    # Match sizes — bilinear-upscale mask if needed, error if bg/fg shapes differ
    if bg_t.shape != fg_t.shape:
        raise ValueError(
            f"background and foreground shapes differ: {bg_t.shape} vs {fg_t.shape}"
        )
    if mask_t.shape[2:] != bg_t.shape[2:]:
        mask_t = F.interpolate(mask_t, size=bg_t.shape[2:], mode="bilinear", align_corners=False)

    # Clamp levels by image size
    min_dim = min(bg_t.shape[2], bg_t.shape[3])
    max_levels = max(1, int(math.log2(max(min_dim, 1))))
    levels = max(1, min(int(levels), max_levels))

    pyr_bg = _build_laplacian_pyramid(bg_t, levels)
    pyr_fg = _build_laplacian_pyramid(fg_t, levels)

    # Gaussian pyramid for mask (smooth blend across bands)
    mask_pyr: List = []
    current_mask = mask_t
    for _ in range(levels):
        mask_pyr.append(current_mask)
        h, w = current_mask.shape[2], current_mask.shape[3]
        current_mask = F.interpolate(
            current_mask, size=(max(1, h // 2), max(1, w // 2)),
            mode="bilinear", align_corners=False,
        )
    mask_pyr.append(current_mask)

    blended_pyr: List = [
        (1.0 - mask_pyr[i]) * pyr_bg[i] + mask_pyr[i] * pyr_fg[i]
        for i in range(len(pyr_bg))
    ]

    result_t = _reconstruct_from_pyramid(blended_pyr).clamp(0, 1)
    return _tensor_to_image(result_t) if return_pil else _tensor_to_numpy(result_t)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _to_tensor_4d(img):
    """Convert PIL/numpy/tensor input to (1, 3, H, W) float32 [0,1] tensor."""
    from PIL import Image
    if isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None].repeat(3, axis=2)
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
    # Already a tensor
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if img.dtype != torch.float32:
        img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    return img


def _to_mask_4d(mask):
    """Convert PIL.L / numpy / tensor mask to (1, 1, H, W) float32 [0,1]."""
    from PIL import Image
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)
    if isinstance(mask, np.ndarray):
        arr = mask.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.ndim == 2:
            return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            arr = arr[:, :, 0]
            return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)
    # Tensor
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0) if mask.shape[0] == 1 else mask.mean(0, keepdim=True).unsqueeze(0)
    if mask.dtype != torch.float32:
        mask = mask.float()
    if mask.max() > 1.5:
        mask = mask / 255.0
    return mask


def _tensor_to_image(t):
    from PIL import Image
    arr = (t.squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    return Image.fromarray(arr)


def _tensor_to_numpy(t):
    return (t.squeeze(0).clamp(0, 1).permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
