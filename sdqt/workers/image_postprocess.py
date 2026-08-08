"""Worker for post-processing still images (AI enhancement + spatial upscale)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import BaseWorker

logger = logging.getLogger(__name__)


class ImagePostProcessWorker(BaseWorker):
    """Apply AI enhancement and spatial upscaling to a single image.

    Emits ``finished_ok`` with the output file path.
    """

    def __init__(
        self,
        source_path: str,
        *,
        ai_denoise: bool = False,
        ai_sharpen: bool = False,
        ai_enhance: bool = False,
        ai_face_restore: bool = False,
        ai_tile: int = 512,
        spatial: str = "",
        upscaler_dir: str = "",
        face_models_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._ai_denoise = ai_denoise
        self._ai_sharpen = ai_sharpen
        self._ai_enhance = ai_enhance
        self._ai_face_restore = ai_face_restore
        self._ai_tile = ai_tile
        self._spatial = spatial
        self._upscaler_dir = upscaler_dir
        self._face_models_dir = face_models_dir

    def do_work(self) -> str:
        from PIL import Image
        from supremediffusion.postprocessing.ai_upscale import (
            upscale_frames, denoise_frames, sharpen_frames,
            face_restore_frames, _resolve_model,
            DEFAULT_UPSCALER, DEFAULT_DENOISER, DEFAULT_SHARPENER,
        )

        self.progress.emit(0.0, "Loading image...")
        if self.is_aborted:
            raise InterruptedError("Aborted by user")
        img = Image.open(self._source).convert("RGB")
        frames = np.array(img)[np.newaxis]  # (1, H, W, 3)

        upscaler_dir = self._upscaler_dir
        if not upscaler_dir:
            upscaler_dir = str(Path.home() / ".supremediffusion" / "models" / "upscalers")

        ai_steps = sum([self._ai_denoise, self._ai_sharpen,
                        self._ai_enhance, self._ai_face_restore])
        ai_done = 0
        ai_frac = 0.8 if ai_steps > 0 else 0.0

        def _ai_progress(f, d, step=0):
            # Abort-aware: raised inside the tiled inference loop so long AI
            # passes can be cancelled mid-image, not just between images.
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            base = (step / ai_steps) * ai_frac if ai_steps else 0
            span = ai_frac / ai_steps if ai_steps else 0
            self.progress.emit(base + f * span, d)

        if self._ai_denoise:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Denoise...")
            model_path = _resolve_model(
                upscaler_dir, DEFAULT_DENOISER,
                progress_cb=lambda f, d: self.progress.emit(0, d),
            )
            step = ai_done
            frames = denoise_frames(
                frames, model_path, tile_size=self._ai_tile,
                progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
            )
            ai_done += 1

        if self._ai_sharpen:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Sharpen...")
            model_path = _resolve_model(
                upscaler_dir, DEFAULT_SHARPENER,
                progress_cb=lambda f, d: self.progress.emit(0, d),
            )
            step = ai_done
            frames = sharpen_frames(
                frames, model_path, tile_size=self._ai_tile,
                progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
            )
            ai_done += 1

        if self._ai_enhance:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Upscale 4x...")
            model_path = _resolve_model(
                upscaler_dir, DEFAULT_UPSCALER,
                progress_cb=lambda f, d: self.progress.emit(0, d),
            )
            step = ai_done
            frames = upscale_frames(
                frames, model_path, tile_size=self._ai_tile,
                progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
            )
            ai_done += 1

        if self._ai_face_restore:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "Face Restore...")
            face_dir = self._face_models_dir
            gfpgan_path = ""
            if face_dir:
                gfpgan = Path(face_dir) / "GFPGANv1.4.pth"
                if gfpgan.is_file():
                    gfpgan_path = str(gfpgan)
            if not gfpgan_path:
                gfpgan_path = _resolve_model(
                    upscaler_dir, "GFPGANv1.4.pth",
                    progress_cb=lambda f, d: self.progress.emit(0, d),
                )
            step = ai_done
            frames = face_restore_frames(
                frames, gfpgan_path, tile_size=0,
                progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
            )
            ai_done += 1

        # Spatial (Lanczos) upscale
        if self._spatial:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(0.85, "Lanczos upscale...")
            from supremediffusion.postprocessing.spatial import spatial_upsample
            frames = spatial_upsample(frames, self._spatial)

        # Save result
        self.progress.emit(0.95, "Saving...")
        result = Image.fromarray(frames[0])

        src = Path(self._source)
        suffix = ""
        if self._ai_denoise:
            suffix += "_dn"
        if self._ai_sharpen:
            suffix += "_sharp"
        if self._ai_enhance:
            suffix += "_4x"
        if self._ai_face_restore:
            suffix += "_face"
        if self._spatial:
            suffix += f"_{self._spatial.replace('.', '')}"
        out_path = str(src.parent / f"{src.stem}{suffix}{src.suffix}")

        # Avoid overwriting source directly here — caller handles that
        if out_path == self._source:
            out_path = str(src.parent / f"{src.stem}{suffix}_pp{src.suffix}")

        result.save(out_path)
        self.progress.emit(1.0, "Done")
        logger.info("Image post-processed: %s -> %s", self._source, out_path)
        return out_path
