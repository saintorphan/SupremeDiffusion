"""Orchestrator for FLUX/Chroma image generation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from supremediffusion.config.project_config import ProjectConfig
from supremediffusion.config.global_config import GlobalConfig

logger = logging.getLogger(__name__)


class FluxGenerationPipeline:
    """High-level orchestrator that ties FLUX model loading and image
    generation together — parallel to ImageGenerationPipeline for SD.
    """

    def __init__(self, flux_pipeline: Any, project_manager: Any) -> None:
        self.flux = flux_pipeline
        self.project_manager = project_manager
        self._global_config = GlobalConfig.load()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_project_path(self, project_name: str) -> Path:
        return self.project_manager.get_project_path(project_name)

    def _apply_loras(self, config: ProjectConfig) -> None:
        """Apply LoRAs from project config to the FLUX pipeline."""
        lora_dir = self._global_config.model_paths.get("flux_lora_dir", "")
        lora_names = getattr(config, "flux_activated_loras", []) or []
        multipliers_str = getattr(config, "flux_lora_multipliers", "") or ""
        multipliers = []
        if multipliers_str:
            for s in multipliers_str.split(","):
                try:
                    multipliers.append(float(s.strip()))
                except ValueError:
                    multipliers.append(1.0)
        self.flux.apply_loras(lora_dir, lora_names, multipliers or None)

    def _save_images_to_temp(self, images: list) -> list[str]:
        """Save PIL images to temp files, returning their paths.

        Images are NOT auto-saved to the project — the user must
        explicitly click 'Save to Project' in the UI.
        """
        paths = []
        for i, img in enumerate(images):
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix=f"flux_img_{i:02d}_", delete=False,
            )
            img.save(tmp.name)
            tmp.close()
            paths.append(tmp.name)
            logger.info("Generated image: %s", tmp.name)
        return paths

    @staticmethod
    def _remove_backgrounds(paths: list[str]) -> list[str]:
        """Remove backgrounds from generated images using rembg.

        Replaces each image file in-place and returns the same paths.
        Falls back silently if rembg is not installed.
        """
        try:
            from rembg import remove as rembg_remove
            from PIL import Image

            for path in paths:
                img = Image.open(path)
                result = rembg_remove(img)
                result.save(path)
                logger.info("Background removed: %s", path)

        except ImportError:
            logger.warning(
                "rembg is not installed — skipping background removal. "
                "Install with: pip install rembg"
            )
        except Exception:
            logger.exception("Background removal failed")

        return paths

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def run_txt2img(
        self,
        project_name: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run FLUX text-to-image generation.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("FLUX Txt2Img: project=%s", project_name)

        try:
            sched_name = getattr(config, "flux_scheduler", "Euler") or "Euler"
            self.flux.set_scheduler(sched_name)
            self._apply_loras(config)

            all_images = []
            for batch_idx in range(config.flux_batch_count):
                seed = config.flux_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.flux.generate_txt2img(
                    prompt=config.flux_prompt,
                    negative_prompt=config.flux_negative_prompt,
                    width=config.flux_width,
                    height=config.flux_height,
                    steps=config.flux_steps,
                    guidance_scale=config.flux_cfg_scale,
                    seed=seed,
                    max_seq_len=config.flux_max_seq_len,
                    callback=callback,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)
            if getattr(config, "flux_remove_bg", False):
                paths = self._remove_backgrounds(paths)
            return paths

        except Exception:
            logger.exception("FLUX Txt2Img generation failed")
            raise

    def run_img2img(
        self,
        project_name: str,
        source_image: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run FLUX image-to-image generation.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("FLUX Img2Img: project=%s source=%s", project_name, source_image)

        try:
            sched_name = getattr(config, "flux_scheduler", "Euler") or "Euler"
            self.flux.set_scheduler(sched_name)
            self._apply_loras(config)

            all_images = []
            for batch_idx in range(config.flux_batch_count):
                seed = config.flux_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.flux.generate_img2img(
                    image=source_image,
                    prompt=config.flux_prompt,
                    negative_prompt=config.flux_negative_prompt,
                    strength=config.flux_denoising_strength,
                    width=config.flux_width,
                    height=config.flux_height,
                    steps=config.flux_steps,
                    guidance_scale=config.flux_cfg_scale,
                    seed=seed,
                    max_seq_len=config.flux_max_seq_len,
                    callback=callback,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)
            if getattr(config, "flux_remove_bg", False):
                paths = self._remove_backgrounds(paths)
            return paths

        except Exception:
            logger.exception("FLUX Img2Img generation failed")
            raise

    def run_fill(
        self,
        project_name: str,
        source_image: str,
        mask: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run FLUX Fill inpainting generation.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("FLUX Fill: project=%s source=%s", project_name, source_image)

        try:
            sched_name = getattr(config, "flux_scheduler", "Euler") or "Euler"
            self.flux.set_scheduler(sched_name)
            self._apply_loras(config)

            # The inpaint UI writes the base flux_* fields (via the shared
            # ImageParamsWidget / strategy.set_config), not flux_fill_*.
            all_images = []
            for batch_idx in range(config.flux_batch_count):
                seed = config.flux_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.flux.generate_fill(
                    image=source_image,
                    mask=mask,
                    prompt=config.flux_prompt,
                    negative_prompt=config.flux_negative_prompt,
                    strength=config.flux_denoising_strength,
                    width=config.flux_width,
                    height=config.flux_height,
                    steps=config.flux_steps,
                    guidance_scale=config.flux_cfg_scale,
                    seed=seed,
                    max_seq_len=config.flux_max_seq_len,
                    callback=callback,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)
            if getattr(config, "flux_remove_bg", False):
                paths = self._remove_backgrounds(paths)
            return paths

        except Exception:
            logger.exception("FLUX Fill generation failed")
            raise
