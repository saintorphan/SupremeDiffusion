"""Post-processing orchestrator for Supreme Diffusion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default RIFE v4 model path (inside app install directory)
_APP_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RIFE_PATH = _APP_ROOT / "models" / "rife" / "rife4.26.pkl"


class PostProcessor:
    """Applies temporal upsampling, spatial upsampling, and film grain to frames."""

    def __init__(self, rife_model_path: Optional[str] = None):
        self.rife_model_path = rife_model_path or str(_DEFAULT_RIFE_PATH)

    def process(
        self,
        frames: np.ndarray,
        fps: int,
        temporal_upsampling: str = "",
        spatial_upsampling: str = "",
        film_grain_intensity: float = 0.0,
        film_grain_saturation: float = 0.5,
        progress_cb: Optional[callable] = None,
    ) -> tuple[np.ndarray, int]:
        """Run all enabled post-processing steps.

        Parameters
        ----------
        frames:
            (N, H, W, 3) uint8 numpy array.
        fps:
            Input frame rate.
        temporal_upsampling:
            ``"rife2"`` (2x), ``"rife4"`` (4x), or ``""`` to skip.
        spatial_upsampling:
            ``"lanczos1.5"`` (1.5x), ``"lanczos2"`` (2x), or ``""`` to skip.
        film_grain_intensity:
            Grain strength (0 = disabled).
        film_grain_saturation:
            Color vs mono grain blend.
        progress_cb:
            Optional callback(fraction, description) for overall progress.

        Returns
        -------
        (processed_frames, output_fps) tuple.
        """
        output_fps = fps

        # --- Temporal upsampling (RIFE) ---
        if temporal_upsampling in ("rife2", "rife4"):
            exp = 1 if temporal_upsampling == "rife2" else 2
            rife_path = Path(self.rife_model_path)
            if not rife_path.is_file():
                logger.warning("RIFE model not found at %s, skipping temporal upsampling.", rife_path)
            else:
                from supremediffusion.postprocessing.rife.interpolate import temporal_interpolation

                def _rife_progress(frac, desc):
                    if progress_cb:
                        progress_cb(frac, desc)

                frames = temporal_interpolation(
                    frames, exp=exp, model_path=str(rife_path),
                    device="cuda", progress_cb=_rife_progress,
                )
                output_fps = fps * (2 ** exp)
                logger.info("Temporal upsampling done: fps %d -> %d", fps, output_fps)

        # --- Spatial upsampling (Lanczos) ---
        if spatial_upsampling and spatial_upsampling.startswith("lanczos"):
            from supremediffusion.postprocessing.spatial import spatial_upsample

            def _spatial_progress(frac, desc):
                if progress_cb:
                    progress_cb(frac, desc)

            frames = spatial_upsample(frames, mode=spatial_upsampling, progress_cb=_spatial_progress)

        # --- Film grain ---
        if film_grain_intensity > 0:
            from supremediffusion.postprocessing.film_grain import add_film_grain
            if progress_cb:
                progress_cb(0, "Applying film grain...")
            frames = add_film_grain(frames, intensity=film_grain_intensity, saturation=film_grain_saturation)
            if progress_cb:
                progress_cb(1.0, "Film grain applied.")

        return frames, output_fps
