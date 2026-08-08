"""AI video enhancement via spandrel — upscale, denoise, sharpen, face restore.

All models loaded via spandrel's universal loader. Operates frame-by-frame.
Model files (.pth, .safetensors) auto-detected from configured directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Model URLs for auto-download ─────────────────────────────────────────────

_MODEL_URLS = {
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    ),
    "SCUNet_color_real_psnr.pth": (
        "https://github.com/cszn/KAIR/releases/download/"
        "v1.0/scunet_color_real_psnr.pth"
    ),
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.2.1/RealESRGAN_x2plus.pth"
    ),
}

# Model file extensions
_MODEL_EXTS = {".pth", ".safetensors", ".ckpt"}

# ── Default model names per task ─────────────────────────────────────────────

DEFAULT_UPSCALER = "RealESRGAN_x4plus.pth"
DEFAULT_DENOISER = "SCUNet_color_real_psnr.pth"
DEFAULT_SHARPENER = "RealESRGAN_x2plus.pth"  # 2x upscale then downscale = sharpen


def scan_models(models_dir: str) -> list[str]:
    """Return list of available model filenames in the directory."""
    d = Path(models_dir)
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.suffix.lower() in _MODEL_EXTS and p.is_file()
    )


# Keep old name for backward compat
scan_upscaler_models = scan_models


def download_model(models_dir: str, filename: str, progress_cb=None) -> str:
    """Download a model if not present. Returns the path."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / filename
    if dest.is_file():
        return str(dest)

    url = _MODEL_URLS.get(filename)
    if not url:
        raise FileNotFoundError(f"No download URL for {filename}")

    if progress_cb:
        progress_cb(0.0, f"Downloading {filename}...")

    import urllib.request
    urllib.request.urlretrieve(url, str(dest))

    if progress_cb:
        progress_cb(1.0, f"Downloaded {filename}")

    return str(dest)


# Keep old name for backward compat
def download_default_model(models_dir: str, progress_cb=None) -> str:
    return download_model(models_dir, DEFAULT_UPSCALER, progress_cb)


def _load_model(model_path: str, device: torch.device, half: bool = True):
    """Load any model via spandrel."""
    import spandrel
    model = spandrel.ModelLoader(device=device).load_from_file(model_path)
    model = model.eval()
    if half and device.type == "cuda" and getattr(model, "supports_half", False):
        model = model.half()
    return model


def _resolve_model(
    models_dir: str, default_name: str, progress_cb=None,
) -> str:
    """Find or download a model. Returns path."""
    models = scan_models(models_dir)
    # Prefer exact default name
    if default_name in models:
        return str(Path(models_dir) / default_name)
    # Otherwise use first available
    if models:
        return str(Path(models_dir) / models[0])
    # Download default
    return download_model(models_dir, default_name, progress_cb)


# ── Frame-by-frame processing ────────────────────────────────────────────────

@torch.inference_mode()
def upscale_frames(
    frames: np.ndarray,
    model_path: str,
    tile_size: int = 512,
    half: bool = True,
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Upscale frames using a spandrel model (ESRGAN, SwinIR, HAT, etc).

    Returns upscaled (N, H*scale, W*scale, 3) uint8 array.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device, half=half and device.type == "cuda")
    scale = model.scale

    n, h, w, c = frames.shape
    new_h, new_w = h * scale, w * scale
    logger.info("AI upscale: %dx%d → %dx%d (%d frames, scale=%dx, tile=%d)",
                w, h, new_w, new_h, n, scale, tile_size)

    out = np.empty((n, new_h, new_w, c), dtype=np.uint8)

    for i in range(n):
        img = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        if half and device.type == "cuda" and getattr(model, "supports_half", False):
            img = img.half()

        if tile_size > 0 and (h > tile_size or w > tile_size):
            result = _tile_process(model, img, tile_size, scale)
        else:
            result = model(img)

        result = result.squeeze(0).clamp(0, 1).float().cpu()
        result = (result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        out[i] = result

        if progress_cb and (i % 5 == 0 or i == n - 1):
            progress_cb((i + 1) / n, f"Upscaling frame {i + 1}/{n}")

    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def denoise_frames(
    frames: np.ndarray,
    model_path: str,
    tile_size: int = 512,
    half: bool = True,
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Denoise frames using a spandrel model (SCUNet, NAFNet, DRUNet, etc).

    Returns denoised (N, H, W, 3) uint8 array (same dimensions).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device, half=half and device.type == "cuda")
    scale = getattr(model, "scale", 1)

    n, h, w, c = frames.shape
    logger.info("AI denoise: %dx%d (%d frames, tile=%d)", w, h, n, tile_size)

    out = np.empty_like(frames)

    for i in range(n):
        img = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        if half and device.type == "cuda" and getattr(model, "supports_half", False):
            img = img.half()

        if tile_size > 0 and (h > tile_size or w > tile_size):
            result = _tile_process(model, img, tile_size, scale)
        else:
            result = model(img)

        result = result.squeeze(0).clamp(0, 1).float().cpu()
        result = (result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        # If model upscaled, resize back to original
        if result.shape[0] != h or result.shape[1] != w:
            from PIL import Image
            result = np.array(Image.fromarray(result).resize((w, h), Image.LANCZOS))

        out[i] = result

        if progress_cb and (i % 5 == 0 or i == n - 1):
            progress_cb((i + 1) / n, f"Denoising frame {i + 1}/{n}")

    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def sharpen_frames(
    frames: np.ndarray,
    model_path: str,
    tile_size: int = 512,
    half: bool = True,
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Sharpen frames by upscaling 2x then downscaling back.

    Uses RealESRGAN x2plus (or any 2x model) to add real detail,
    then Lanczos downscale to original dimensions = net sharpening.
    Returns sharpened (N, H, W, 3) uint8 array (same dimensions).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device, half=half and device.type == "cuda")
    scale = model.scale

    n, h, w, c = frames.shape
    logger.info("AI sharpen: %dx%d (%d frames, scale=%dx then downscale, tile=%d)",
                w, h, n, scale, tile_size)

    from PIL import Image
    out = np.empty_like(frames)

    for i in range(n):
        img = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        if half and device.type == "cuda" and getattr(model, "supports_half", False):
            img = img.half()

        if tile_size > 0 and (h > tile_size or w > tile_size):
            result = _tile_process(model, img, tile_size, scale)
        else:
            result = model(img)

        result = result.squeeze(0).clamp(0, 1).float().cpu()
        result = (result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        # Downscale back to original dimensions
        result = np.array(Image.fromarray(result).resize((w, h), Image.LANCZOS))
        out[i] = result

        if progress_cb and (i % 5 == 0 or i == n - 1):
            progress_cb((i + 1) / n, f"Sharpening frame {i + 1}/{n}")

    del model
    torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def face_restore_frames(
    frames: np.ndarray,
    model_path: str,
    tile_size: int = 0,
    half: bool = True,
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Restore faces using GFPGAN/RestoreFormer via spandrel.

    Returns restored (N, H, W, 3) uint8 array (same dimensions).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device, half=half and device.type == "cuda")
    scale = getattr(model, "scale", 1)

    n, h, w, c = frames.shape
    logger.info("Face restore: %dx%d (%d frames)", w, h, n)

    out = np.empty_like(frames)

    for i in range(n):
        img = torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        if half and device.type == "cuda" and getattr(model, "supports_half", False):
            img = img.half()

        try:
            result = model(img)
            result = result.squeeze(0).clamp(0, 1).float().cpu()
            result = (result.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

            if result.shape[0] != h or result.shape[1] != w:
                from PIL import Image
                result = np.array(Image.fromarray(result).resize((w, h), Image.LANCZOS))

            out[i] = result
        except Exception:
            # Face restore can fail on frames without faces — pass through
            out[i] = frames[i]

        if progress_cb and (i % 5 == 0 or i == n - 1):
            progress_cb((i + 1) / n, f"Restoring faces {i + 1}/{n}")

    del model
    torch.cuda.empty_cache()
    return out


# ── Tiled processing ─────────────────────────────────────────────────────────

def _tile_process(model, img: torch.Tensor, tile_size: int, scale: int) -> torch.Tensor:
    """Process an image in overlapping tiles to fit in VRAM."""
    overlap = 32
    _, _, h, w = img.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, device=img.device, dtype=img.dtype)
    count = torch.zeros(1, 1, out_h, out_w, device=img.device, dtype=img.dtype)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = img[:, :, y_start:y_end, x_start:x_end]
            tile_out = model(tile)

            oy = y_start * scale
            ox = x_start * scale
            oh = tile_out.shape[2]
            ow = tile_out.shape[3]

            output[:, :, oy:oy + oh, ox:ox + ow] += tile_out
            count[:, :, oy:oy + oh, ox:ox + ow] += 1

    output /= count.clamp(min=1)
    return output
