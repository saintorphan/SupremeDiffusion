"""Worker for iterative texture refinement using SD img2img + UV projection."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

# 8 camera angles: 4 cardinal + 4 diagonal, two elevation tiers
DEFAULT_ANGLES: list[tuple[float, float]] = [
    (-20, 0), (-20, 90), (-20, 180), (-20, 270),       # cardinal, high
    (-20, 45), (-20, 135), (-20, 225), (-20, 315),      # diagonal, high
]


class TextureRefineWorker(BaseWorker):
    """Render multi-angle views, enhance with SD img2img, project back to UV texture.

    Args:
        mesh_path: Path to the mesh file (must have UV coordinates).
        render_func: Callable(rot_x, rot_y, width, height) -> QImage.
        app_state: AppState for SD pipeline access.
        prompt: Positive prompt for img2img enhancement.
        negative_prompt: Negative prompt.
        denoise_strength: Denoising strength for img2img (0.2-0.5 recommended).
        tex_size: Output texture resolution.
        render_size: Resolution to render each view at.
        angles: Camera angles to render from.
        checkpoint: SD checkpoint name (or "" for current).
    """

    def __init__(
        self,
        mesh_path: str,
        render_func,
        app_state,
        prompt: str,
        negative_prompt: str = "",
        denoise_strength: float = 0.35,
        tex_size: int = 1024,
        render_size: int = 512,
        angles: list[tuple[float, float]] | None = None,
        checkpoint: str = "",
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._mesh_path = mesh_path
        self._render_func = render_func
        self._state = app_state
        self._prompt = prompt
        self._neg_prompt = negative_prompt
        self._denoise = denoise_strength
        self._tex_size = tex_size
        self._render_size = render_size
        self._angles = angles or DEFAULT_ANGLES
        self._checkpoint = checkpoint

    def do_work(self) -> str:
        """Run the full texture refinement pipeline.

        Returns path to the saved albedo texture.
        """
        import numpy as np

        n_angles = len(self._angles)

        # Step 1: Ensure SD pipeline loaded
        self.status.emit("Loading SD pipeline...")
        self.progress.emit(0.0, "Loading SD pipeline...")
        if self._state.img_pipeline is None:
            self._state.load_sd_pipelines()
        if self.is_aborted:
            raise InterruptedError

        # Step 2: Initialize refiner
        self.status.emit("Initializing texture refiner...")
        self.progress.emit(0.05, "Loading mesh UVs...")
        from supremediffusion.models.texture_refiner import TextureRefiner
        refiner = TextureRefiner(self._mesh_path, tex_size=self._tex_size)

        # Step 3: For each angle, render → enhance → project
        for i, (rot_x, rot_y) in enumerate(self._angles):
            if self.is_aborted:
                raise InterruptedError

            base_frac = 0.1 + (i / n_angles) * 0.8
            angle_label = f"View {i + 1}/{n_angles} (el={rot_x:.0f}° az={rot_y:.0f}°)"

            # Render raw view
            self.status.emit(f"Rendering {angle_label}...")
            self.progress.emit(base_frac, f"Rendering {angle_label}...")
            qimg = self._render_func(rot_x, rot_y, self._render_size, self._render_size)

            # Convert QImage to file for img2img
            tmp_render = Path(tempfile.mktemp(suffix=".png", prefix="tex_render_"))
            qimg.save(str(tmp_render), "PNG")

            if self.is_aborted:
                tmp_render.unlink(missing_ok=True)
                raise InterruptedError

            # Enhance with SD img2img
            self.status.emit(f"Enhancing {angle_label}...")
            self.progress.emit(base_frac + 0.3 / n_angles, f"Enhancing {angle_label}...")
            enhanced_path = self._run_img2img(str(tmp_render))
            tmp_render.unlink(missing_ok=True)

            if enhanced_path is None:
                logger.warning("img2img failed for angle %s, using raw render", angle_label)
                # Fall back to raw render — re-save since we deleted it
                qimg.save(str(tmp_render), "PNG")
                enhanced_path = str(tmp_render)

            if self.is_aborted:
                raise InterruptedError

            # Load enhanced image as numpy array
            from PIL import Image
            enhanced_img = np.array(Image.open(enhanced_path).convert("RGB"))

            # Project onto UV texture
            self.status.emit(f"Projecting {angle_label}...")
            self.progress.emit(base_frac + 0.6 / n_angles, f"Projecting {angle_label}...")
            texels = refiner.project_view(
                enhanced_img,
                rot_x=rot_x,
                rot_y=rot_y,
                zoom=3.0,
                aspect=1.0,
                fov=45.0,
            )
            logger.info("View %d: %d texels projected (coverage %.1f%%)",
                        i + 1, texels, refiner.coverage * 100)

        # Step 4: Fill gaps and save
        self.status.emit("Filling texture gaps...")
        self.progress.emit(0.92, "Filling gaps...")
        tex_path = refiner.save()

        self.status.emit(f"Texture saved — {refiner.coverage * 100:.0f}% coverage")
        self.progress.emit(1.0, "Complete")
        return tex_path

    def _run_img2img(self, source_path: str) -> str | None:
        """Run a single img2img pass and return the output path."""
        from supremediffusion.config.project_config import ProjectConfig

        try:
            project_name = self._state.current_project or "_default"
            project_path = self._state.project_manager.get_project_path(project_name)
            cfg = ProjectConfig.load(project_path)

            cfg.img_prompt = self._prompt
            cfg.img_negative_prompt = self._neg_prompt
            cfg.img_denoising_strength = self._denoise
            cfg.img_batch_count = 1
            cfg.img_batch_size = 1
            cfg.img_width = self._render_size
            cfg.img_height = self._render_size
            cfg.img_seed = -1
            cfg.img2img_source_path = source_path
            if self._checkpoint:
                cfg.img_checkpoint = self._checkpoint

            pipeline = self._state.img_pipeline
            result = pipeline.run_img2img(
                project_name=project_name,
                config=cfg,
                callback=None,
            )

            if result and isinstance(result, list) and result[0]:
                return result[0]
            return str(result) if result else None

        except Exception as exc:
            logger.warning("img2img enhancement failed: %s", exc)
            return None
