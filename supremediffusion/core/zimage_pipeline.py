"""Orchestrator for Z-Image Turbo image generation."""

from __future__ import annotations

import inspect
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from supremediffusion.config.project_config import ProjectConfig

logger = logging.getLogger(__name__)


class ZImageGenerationPipeline:
    """High-level orchestrator that ties Z-Image Turbo model loading and
    image generation together -- parallel to FluxGenerationPipeline for FLUX.
    """

    def __init__(self, zimage_pipeline: Any, project_manager: Any) -> None:
        self.zimage = zimage_pipeline
        self.project_manager = project_manager

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_project_path(self, project_name: str) -> Path:
        return self.project_manager.get_project_path(project_name)

    def _apply_loras(self, config: ProjectConfig) -> None:
        """Apply LoRAs from config if any are activated."""
        lora_dir = self.zimage.config.model_paths.get("zimage_lora_dir", "")
        activated = getattr(config, "zimage_activated_loras", []) or []
        multipliers_str = getattr(config, "zimage_lora_multipliers", "") or ""

        multipliers = None
        if multipliers_str.strip():
            try:
                multipliers = [float(x.strip()) for x in multipliers_str.split(",") if x.strip()]
            except ValueError:
                logger.warning("Invalid LoRA multipliers: %s", multipliers_str)

        self.zimage.apply_loras(lora_dir, activated, multipliers)

    def _extra_gen_kwargs(self, method: Callable, config: ProjectConfig) -> dict:
        """Resolve negative-prompt and CFG-normalization kwargs.

        Z-Image Turbo natively supports a negative prompt plus CFG
        normalization / truncation. The model wrapper only gains these
        parameters incrementally, so each one is forwarded only if *method*
        actually accepts it (keeps older wrappers working).
        """
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return {}

        extras: dict[str, Any] = {}
        if "negative_prompt" in params:
            extras["negative_prompt"] = getattr(config, "zimage_negative_prompt", "") or ""
        if "cfg_normalization" in params:
            extras["cfg_normalization"] = getattr(config, "zimage_cfg_normalization", True)
        if "cfg_truncation" in params:
            extras["cfg_truncation"] = getattr(config, "zimage_cfg_truncation", 0.0)
        return extras

    def _save_images_to_temp(self, images: list) -> list[str]:
        """Save PIL images to temp files, returning their paths.

        Images are NOT auto-saved to the project -- the user must
        explicitly click 'Save to Project' in the UI.
        """
        paths = []
        for i, img in enumerate(images):
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix=f"zimage_img_{i:02d}_", delete=False,
            )
            img.save(tmp.name)
            tmp.close()
            paths.append(tmp.name)
            logger.info("Generated image: %s", tmp.name)
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
        """Run Z-Image Turbo text-to-image generation.

        Reads config fields: zimage_prompt, zimage_steps, zimage_cfg_scale,
        zimage_width, zimage_height, zimage_seed, zimage_batch_count,
        zimage_remove_bg.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("Z-Image Txt2Img: project=%s", project_name)

        try:
            self._apply_loras(config)

            extra_kwargs = self._extra_gen_kwargs(self.zimage.run_txt2img, config)

            all_images = []
            for batch_idx in range(config.zimage_batch_count):
                seed = config.zimage_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.zimage.run_txt2img(
                    prompt=config.zimage_prompt,
                    width=config.zimage_width,
                    height=config.zimage_height,
                    steps=config.zimage_steps,
                    guidance_scale=config.zimage_cfg_scale,
                    seed=seed,
                    callback=callback,
                    **extra_kwargs,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)

            if getattr(config, "zimage_remove_bg", False):
                paths = self._remove_backgrounds(paths)

            return paths

        except Exception:
            logger.exception("Z-Image Txt2Img generation failed")
            raise

    def run_img2img(
        self,
        project_name: str,
        source_image: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run Z-Image Turbo image-to-image generation.

        Reads config fields: zimage_prompt, zimage_steps, zimage_cfg_scale,
        zimage_width, zimage_height, zimage_seed, zimage_batch_count,
        zimage_denoising_strength, zimage_remove_bg.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("Z-Image Img2Img: project=%s source=%s", project_name, source_image)

        try:
            self._apply_loras(config)

            extra_kwargs = self._extra_gen_kwargs(self.zimage.run_img2img, config)

            all_images = []
            for batch_idx in range(config.zimage_batch_count):
                seed = config.zimage_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.zimage.run_img2img(
                    image=source_image,
                    prompt=config.zimage_prompt,
                    strength=config.zimage_denoising_strength,
                    width=config.zimage_width,
                    height=config.zimage_height,
                    steps=config.zimage_steps,
                    guidance_scale=config.zimage_cfg_scale,
                    seed=seed,
                    callback=callback,
                    **extra_kwargs,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)

            if getattr(config, "zimage_remove_bg", False):
                paths = self._remove_backgrounds(paths)

            return paths

        except Exception:
            logger.exception("Z-Image Img2Img generation failed")
            raise

    def run_inpaint(
        self,
        project_name: str,
        source_image: str,
        mask: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run Z-Image Turbo inpainting generation.

        Reads the base zimage_* fields (zimage_prompt, zimage_steps,
        zimage_cfg_scale, zimage_denoising_strength, zimage_width,
        zimage_height, zimage_seed, zimage_remove_bg) — the inpaint UI
        writes these via the shared ImageParamsWidget / strategy.set_config,
        not zimage_inpaint_*.

        Returns list of saved image paths.
        """
        self._resolve_project_path(project_name)
        logger.info("Z-Image Inpaint: project=%s source=%s", project_name, source_image)

        try:
            self._apply_loras(config)

            extra_kwargs = self._extra_gen_kwargs(self.zimage.run_inpaint, config)

            all_images = []
            for batch_idx in range(config.zimage_batch_count):
                seed = config.zimage_seed
                if seed >= 0:
                    seed = seed + batch_idx

                images = self.zimage.run_inpaint(
                    image=source_image,
                    mask=mask,
                    prompt=config.zimage_prompt,
                    strength=config.zimage_denoising_strength,
                    width=config.zimage_width,
                    height=config.zimage_height,
                    steps=config.zimage_steps,
                    guidance_scale=config.zimage_cfg_scale,
                    seed=seed,
                    callback=callback,
                    **extra_kwargs,
                )
                all_images.extend(images)

            paths = self._save_images_to_temp(all_images)

            if getattr(config, "zimage_remove_bg", False):
                paths = self._remove_backgrounds(paths)

            return paths

        except Exception:
            logger.exception("Z-Image Inpaint generation failed")
            raise

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

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
                "rembg is not installed -- skipping background removal. "
                "Install with: pip install rembg"
            )
        except Exception:
            logger.exception("Background removal failed")

        return paths
