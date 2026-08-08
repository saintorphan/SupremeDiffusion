"""ESRGAN-family upscaler pipeline wrapper.

Loads any ESRGAN / RealESRGAN / UltraSharp / NMKD upscaler model via the
``spandrel`` library, which auto-detects the architecture from the state dict.
Tile-based inference keeps VRAM bounded.

Lifecycle mirrors the other pipeline wrappers (load / unload / is_loaded /
generate) so AppState can manage it the same way.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


class UpscalerPipeline:
    """Wraps spandrel-loaded ESRGAN models for tiled image/frame upscaling."""

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self._model: Any = None
        self._model_path: Optional[str] = None
        self._scale: int = 1
        self._loaded: bool = False
        self._device: Any = None

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def scale(self) -> int:
        return self._scale

    # ------------------------------------------------------------------
    def load(self, model_path: str) -> None:
        """Load a .pth / .safetensors upscaler model via spandrel."""
        if torch is None:
            raise RuntimeError("PyTorch required for upscaling.")
        from spandrel import ModelLoader

        if self._loaded and self._model_path == model_path:
            return

        if self._loaded:
            self.unload()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading upscaler: %s", Path(model_path).name)
        loader = ModelLoader()
        desc = loader.load_from_file(model_path)
        self._model = desc.eval().to(device)
        self._scale = int(desc.scale)
        self._model_path = model_path
        self._device = device
        self._loaded = True
        logger.info("Upscaler loaded: %sx scale, %s arch", self._scale,
                    type(desc.model).__name__)

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:  # noqa: BLE001
                pass
            del self._model
            self._model = None
        self._model_path = None
        self._loaded = False
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def upscale(
        self,
        image: Any,  # PIL.Image or np.ndarray
        *,
        tile_size: int = 512,
        tile_overlap: int = 32,
    ) -> Any:
        """Upscale a single image. Returns a PIL.Image.

        Args:
            image: PIL.Image (any mode → converted to RGB) or HxWx3 uint8 np.array.
            tile_size: spatial tile size — lower = less VRAM, more overhead.
            tile_overlap: feather between tiles to hide seams.
        """
        if not self._loaded:
            raise RuntimeError("Upscaler not loaded — call load(model_path) first.")
        if torch is None or np is None:
            raise RuntimeError("torch + numpy required.")

        from PIL import Image as _Image
        if isinstance(image, _Image.Image):
            arr = np.asarray(image.convert("RGB"))
        else:
            arr = np.asarray(image)
        h, w = arr.shape[:2]
        tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self._device)

        out = self._tiled_inference(tensor, tile_size=tile_size, tile_overlap=tile_overlap)
        out = (out.clamp(0, 1) * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return _Image.fromarray(out)

    def upscale_frames(
        self,
        frames: list,
        *,
        tile_size: int = 512,
        tile_overlap: int = 32,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list:
        """Upscale a list of PIL.Image frames. Returns new list."""
        out = []
        n = len(frames)
        for i, f in enumerate(frames):
            out.append(self.upscale(f, tile_size=tile_size, tile_overlap=tile_overlap))
            if progress_cb:
                progress_cb((i + 1) / max(1, n), f"upscale {i+1}/{n}")
        return out

    # ------------------------------------------------------------------
    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore[misc]
    def _tiled_inference(
        self,
        x: Any,  # [B, C, H, W] float32 tensor
        *,
        tile_size: int,
        tile_overlap: int,
    ) -> Any:
        """Run model on overlapping tiles, blend with Hann window."""
        if torch is None:
            raise RuntimeError("torch required.")

        b, c, h, w = x.shape
        s = self._scale

        # If the image is smaller than one tile, run in one pass.
        if h <= tile_size and w <= tile_size:
            return self._model(x)

        out = torch.zeros((b, c, h * s, w * s), device=x.device, dtype=x.dtype)
        weight = torch.zeros((1, 1, h * s, w * s), device=x.device, dtype=x.dtype)

        stride = tile_size - tile_overlap
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                y0c = max(0, y1 - tile_size)
                x0c = max(0, x1 - tile_size)
                tile_in = x[:, :, y0c:y1, x0c:x1]
                tile_out = self._model(tile_in)

                # Build a Hann window for soft blending at the tile edges.
                th, tw = tile_out.shape[-2:]
                win = _hann_window(th, tw, device=x.device, dtype=x.dtype)

                oy0, ox0 = y0c * s, x0c * s
                oy1, ox1 = oy0 + th, ox0 + tw
                out[:, :, oy0:oy1, ox0:ox1] += tile_out * win
                weight[:, :, oy0:oy1, ox0:ox1] += win

                if x1 >= w:
                    break
            if y1 >= h:
                break

        return out / (weight + 1e-8)


# ----------------------------------------------------------------------
# Module-level cache
#
# Repeated runs (batch image jobs, per-frame timeline upscales, the Phase 3
# PostProcessPipeline that builds a fresh worker per clip) otherwise reload the
# same spandrel weights — a costly disk read + VRAM allocation each time. We
# keep one shared pipeline instance and only re-run ``load()`` when the
# requested ``model_path`` differs from what's resident; ``load()`` itself
# already short-circuits on an identical path, so back-to-back calls with the
# same model are a no-op.
_CACHED_UPSCALER: Optional["UpscalerPipeline"] = None


def get_upscaler(model_path: str, global_config: Any = None) -> "UpscalerPipeline":
    """Return a shared :class:`UpscalerPipeline` with ``model_path`` loaded.

    The instance is cached at module scope and reused across calls; switching
    ``model_path`` swaps the resident weights in place (``load()`` unloads the
    previous model first). Callers should NOT ``unload()`` the returned
    pipeline if they want to benefit from caching — let it stay resident for
    the next run. To free VRAM explicitly, call :func:`release_upscaler`.
    """
    global _CACHED_UPSCALER
    if _CACHED_UPSCALER is None:
        _CACHED_UPSCALER = UpscalerPipeline(global_config)
    elif global_config is not None:
        _CACHED_UPSCALER.config = global_config
    _CACHED_UPSCALER.load(model_path)
    return _CACHED_UPSCALER


def release_upscaler() -> None:
    """Unload and drop the cached upscaler, freeing its VRAM."""
    global _CACHED_UPSCALER
    if _CACHED_UPSCALER is not None:
        _CACHED_UPSCALER.unload()
        _CACHED_UPSCALER = None


def _hann_window(h: int, w: int, *, device: Any, dtype: Any) -> Any:
    """2D Hann window as a [1, 1, h, w] tensor for tile-blending.

    Uses a *symmetric* Hann window (``periodic=False``) — its prior form,
    ``0.5 - 0.5*cos(linspace(0, π, N))``, was a one-sided 0→1 ramp that gave
    weight 0.0 at index 0. The tile covering output index 0 then contributed
    nothing, leaving a near-black 1px top row / left column on multi-tile
    images. The symmetric window peaks at the centre and tapers to both
    edges; the epsilon clamp keeps the extreme edge rows/cols non-zero so a
    single tile can still cover them (the run() divisor adds its own 1e-8).
    """
    yh = torch.hann_window(h, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    xw = torch.hann_window(w, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    win = yh.unsqueeze(1) * xw.unsqueeze(0)
    return win.unsqueeze(0).unsqueeze(0)
