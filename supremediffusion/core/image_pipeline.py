"""Orchestrator for Stable Diffusion image generation."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from supremediffusion.config.project_config import ProjectConfig

logger = logging.getLogger(__name__)


class ImageGenerationPipeline:
    """High-level orchestrator that ties checkpoint loading, LoRA management,
    and image generation together — parallel to GenerationPipeline for video.
    """

    def __init__(self, sd_pipeline: Any, project_manager: Any) -> None:
        self.sd = sd_pipeline
        self.project_manager = project_manager

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_project_path(self, project_name: str) -> Path:
        return self.project_manager.get_project_path(project_name)

    def _save_images_to_temp(self, images: list) -> list[str]:
        """Save PIL images to temp files, returning their paths."""
        import tempfile

        paths = []
        for i, img in enumerate(images):
            tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix=f"sd_img_{i:02d}_", delete=False,
            )
            img.save(tmp.name)
            tmp.close()
            paths.append(tmp.name)
            logger.info("Generated image: %s", tmp.name)
        return paths

    def _save_images_to_project(
        self, images: list, project_path, subdir: str, config, method: str,
    ) -> list[str]:
        """Save PIL images to project subdirectory with embedded metadata.

        Args:
            images: List of PIL Images.
            project_path: Path to the project root.
            subdir: Subdirectory under images/ (e.g. 'txt2img', 'img2img').
            config: ProjectConfig with generation parameters.
            method: Generation method name for metadata.

        Returns:
            List of saved file paths.
        """
        from datetime import datetime
        from pathlib import Path
        from PIL.PngImagePlugin import PngInfo

        out_dir = Path(project_path) / "images" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build A1111-style parameters string
        params_parts = []
        if getattr(config, "img_prompt", ""):
            params_parts.append(config.img_prompt)
        neg = getattr(config, "img_negative_prompt", "")
        if neg:
            params_parts.append(f"Negative prompt: {neg}")
        settings = []
        if getattr(config, "img_steps", 0):
            settings.append(f"Steps: {config.img_steps}")
        if getattr(config, "img_sampler", ""):
            settings.append(f"Sampler: {config.img_sampler}")
        if getattr(config, "img_cfg_scale", 0):
            settings.append(f"CFG scale: {config.img_cfg_scale}")
        if getattr(config, "img_seed", -1) >= 0:
            settings.append(f"Seed: {config.img_seed}")
        if getattr(config, "img_width", 0):
            settings.append(f"Size: {config.img_width}x{config.img_height}")
        if getattr(config, "img_checkpoint", ""):
            settings.append(f"Model: {config.img_checkpoint}")
        if getattr(config, "img_denoising_strength", 0):
            settings.append(f"Denoising strength: {config.img_denoising_strength}")
        settings.append(f"Method: {method}")
        if settings:
            params_parts.append(", ".join(settings))
        params_str = "\n".join(params_parts)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = []
        for i, img in enumerate(images):
            fname = f"{method}_{ts}_{i:02d}.png"
            out_path = out_dir / fname
            # Embed metadata
            meta = PngInfo()
            meta.add_text("parameters", params_str)
            img.save(str(out_path), pnginfo=meta)
            paths.append(str(out_path))
            logger.info("Saved to project: %s", out_path)
        return paths

    def _ensure_checkpoint(self, config: ProjectConfig) -> None:
        """Ensure the SD pipeline has the requested checkpoint loaded."""
        checkpoint = config.img_checkpoint
        if not checkpoint:
            raise ValueError("No checkpoint selected. Please choose a model.")

        # Check if checkpoint path is absolute or needs resolution
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_dir = self.sd.config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                # First try the path as-is (already has extension)
                candidate = Path(ckpt_dir) / checkpoint
                if candidate.exists():
                    checkpoint = str(candidate)
                else:
                    # Try appending extensions
                    for ext in (".safetensors", ".ckpt"):
                        candidate = Path(ckpt_dir) / f"{checkpoint}{ext}"
                        if candidate.exists():
                            checkpoint = str(candidate)
                            break

        vae_name = config.img_vae or "Automatic"
        self.sd.load(
            checkpoint,
            vae_name=vae_name,
            vae_precision=getattr(config, "sd_vae_precision", "16"),
        )

    def _resolve_lora_list(self, config: ProjectConfig) -> list[dict]:
        """Normalise the shared image LoRA selection into backend dicts.

        The UI stores plain LoRA names in ``img_loras`` and a comma-separated
        weight string in ``img_lora_multipliers``; the backend expects
        ``[{"name", "weight"}, ...]``.
        """
        lora_list = getattr(config, "img_loras", None) or []
        if not lora_list:
            return []
        # Parse per-LoRA weights from comma-separated multiplier string
        multiplier_str = getattr(config, "img_lora_multipliers", "") or ""
        weights = []
        if multiplier_str.strip():
            for part in multiplier_str.split(","):
                try:
                    weights.append(float(part.strip()))
                except ValueError:
                    weights.append(1.0)

        # Normalise: UI stores plain strings, backend expects dicts
        normalised: list[dict] = []
        for i, item in enumerate(lora_list):
            if isinstance(item, str):
                w = weights[i] if i < len(weights) else 1.0
                normalised.append({"name": item, "weight": w})
            else:
                normalised.append(item)
        return normalised

    def _apply_loras(self, config: ProjectConfig) -> None:
        """Apply LoRAs from config to the main SD pipe."""
        normalised = self._resolve_lora_list(config)
        if normalised:
            self.sd.apply_loras(normalised)

    def _remove_loras(self) -> None:
        """Remove any active LoRAs."""
        self.sd.remove_loras()

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def run_txt2img(
        self,
        project_name: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run text-to-image generation.

        Returns list of saved image paths.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info("Txt2Img: project=%s checkpoint=%s", project_name, config.img_checkpoint)

        try:
            self._ensure_checkpoint(config)
            self._apply_loras(config)

            all_images = []
            try:
                for batch_idx in range(config.img_batch_count):
                    seed = config.img_seed
                    if seed >= 0:
                        seed = seed + batch_idx

                    # Refiner params (SDXL only, optional)
                    refiner_kwargs = {}
                    if getattr(config, "img_refiner_enabled", False):
                        refiner_ckpt = getattr(config, "img_refiner_checkpoint", "")
                        if refiner_ckpt:
                            refiner_kwargs = {
                                "refiner_checkpoint": refiner_ckpt,
                                "refiner_switch_at": getattr(config, "img_refiner_switch_at", 0.8),
                                "refiner_steps": getattr(config, "img_refiner_steps", 10),
                                "refiner_cfg_scale": getattr(config, "img_refiner_cfg_scale", 7.0),
                            }

                    # Optional IP-Adapter reference-look (e.g. character creator
                    # applying a canonical look to generated poses).
                    ip_kwargs: dict = {}
                    ip_path = getattr(config, "img_ipadapter_path", "") or ""
                    ip_scale = float(getattr(config, "img_ipadapter_scale", 0.0) or 0.0)
                    if ip_path and ip_scale > 0.0 and Path(ip_path).is_file():
                        ip_kwargs = {
                            "ip_adapter_image": ip_path,
                            "ip_adapter_variant": getattr(config, "img_ipadapter_variant", "plus") or "plus",
                            "ip_adapter_scale": ip_scale,
                        }

                    images = self.sd.generate_txt2img(
                        prompt=config.img_prompt,
                        negative_prompt=config.img_negative_prompt,
                        width=config.img_width,
                        height=config.img_height,
                        steps=config.img_steps,
                        cfg_scale=config.img_cfg_scale,
                        seed=seed,
                        sampler=config.img_sampler,
                        scheduler=config.img_scheduler,
                        batch_size=config.img_batch_size,
                        clip_skip=config.img_clip_skip,
                        callback=callback,
                        **refiner_kwargs,
                        **ip_kwargs,
                    )
                    all_images.extend(images)
            finally:
                self._remove_loras()

            # Save to project with metadata; the saved paths feed the gallery
            # directly so we don't write each image a second time to /tmp.
            return self._save_images_to_project(
                all_images, project_path, "txt2img", config, "txt2img",
            )

        except Exception:
            logger.exception("Txt2Img generation failed")
            raise

    def run_img2img(
        self,
        project_name: str,
        source_image: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run image-to-image generation.

        Returns list of saved image paths.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info("Img2Img: project=%s source=%s", project_name, source_image)

        try:
            self._ensure_checkpoint(config)
            self._apply_loras(config)

            all_images = []
            try:
                for batch_idx in range(config.img_batch_count):
                    seed = config.img_seed
                    if seed >= 0:
                        seed = seed + batch_idx

                    images = self.sd.generate_img2img(
                        image=source_image,
                        prompt=config.img_prompt,
                        negative_prompt=config.img_negative_prompt,
                        denoising_strength=config.img_denoising_strength,
                        width=config.img_width,
                        height=config.img_height,
                        steps=config.img_steps,
                        cfg_scale=config.img_cfg_scale,
                        seed=seed,
                        sampler=config.img_sampler,
                        scheduler=config.img_scheduler,
                        resize_mode=config.img_resize_mode,
                        batch_size=config.img_batch_size,
                        clip_skip=config.img_clip_skip,
                        callback=callback,
                    )
                    all_images.extend(images)
            finally:
                self._remove_loras()

            return self._save_images_to_project(
                all_images, project_path, "img2img", config, "img2img",
            )

        except Exception:
            logger.exception("Img2Img generation failed")
            raise

    def run_inpaint(
        self,
        project_name: str,
        source_image: str,
        mask: str,
        config: ProjectConfig,
        callback: Optional[Callable] = None,
    ) -> list[str]:
        """Run inpainting generation.

        Returns list of saved image paths.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info("Inpaint: project=%s source=%s", project_name, source_image)

        try:
            self._ensure_checkpoint(config)
            self._apply_loras(config)

            all_images = []
            try:
                for batch_idx in range(config.img_batch_count):
                    seed = config.img_seed
                    if seed >= 0:
                        seed = seed + batch_idx

                    # Invert mask if requested (inpaint not masked)
                    effective_mask = mask
                    if config.img_mask_invert:
                        from PIL import Image as _PILImage, ImageOps
                        _m = _PILImage.open(mask).convert("L") if isinstance(mask, str) else mask
                        effective_mask = ImageOps.invert(_m)

                    # Phase 4b: Laplacian pyramid blend (opt-in via config).
                    self.sd._use_laplacian_blend = bool(
                        getattr(config, "img_inpaint_laplacian_blend", False)
                    )

                    images = self.sd.generate_inpaint(
                        image=source_image,
                        mask=effective_mask,
                        prompt=config.img_prompt,
                        negative_prompt=config.img_negative_prompt,
                        denoising_strength=config.img_denoising_strength,
                        width=config.img_width,
                        height=config.img_height,
                        steps=config.img_steps,
                        cfg_scale=config.img_cfg_scale,
                        seed=seed,
                        sampler=config.img_sampler,
                        scheduler=config.img_scheduler,
                        mask_blur=config.img_mask_blur,
                        inpainting_fill=config.img_inpainting_fill,
                        full_res=config.img_inpaint_full_res,
                        padding=config.img_inpaint_full_res_padding,
                        batch_size=config.img_batch_size,
                        clip_skip=config.img_clip_skip,
                        callback=callback,
                    )
                    all_images.extend(images)
            finally:
                self._remove_loras()

            return self._save_images_to_project(
                all_images, project_path, "inpaint", config, "inpaint",
            )

        except Exception:
            logger.exception("Inpaint generation failed")
            raise

    def _resolve_checkpoint(self, checkpoint: str) -> str:
        """Resolve a checkpoint name to its full path."""
        if not checkpoint:
            raise ValueError("No checkpoint selected.")
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_dir = self.sd.config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                for ext in (".safetensors", ".ckpt"):
                    candidate = Path(ckpt_dir) / f"{checkpoint}{ext}"
                    if candidate.exists():
                        return str(candidate)
        return checkpoint

    def run_body_double(
        self,
        project_name: str,
        target_image: str,
        mask: str,
        source_person: str,
        control_images: list[Any] | Any = None,
        config: "ProjectConfig" = None,
        num_images: int = 1,
        callback: Optional[Callable] = None,
        # Legacy parameter name
        pose_image: Any = None,
    ) -> list[str]:
        """Run body double generation (ControlNet + IP-Adapter appearance).

        Returns list of saved image paths.
        """
        logger.info("BodyDouble: project=%s target=%s", project_name, target_image)
        project_path = self._resolve_project_path(project_name)

        # Backward compatibility: pose_image → control_images
        if control_images is None and pose_image is not None:
            control_images = [pose_image]

        try:
            checkpoint = self._resolve_checkpoint(
                config.bodydouble_checkpoint or config.img_checkpoint
            )

            # Build ControlNet type and strength lists from config
            cn1_type = getattr(config, "bodydouble_controlnet_type", "") or ""
            controlnet_types = []
            controlnet_strengths = []
            if cn1_type:
                controlnet_types.append(cn1_type)
                controlnet_strengths.append(config.bodydouble_controlnet_strength)
                cn2_type = getattr(config, "bodydouble_controlnet2_type", "")
                if cn2_type:
                    controlnet_types.append(cn2_type)
                    controlnet_strengths.append(
                        getattr(config, "bodydouble_controlnet2_strength", 0.5)
                    )
            control_guidance_start = getattr(config, "bodydouble_guidance_start", 0.0)
            control_guidance_end = getattr(config, "bodydouble_guidance_end", 1.0)
            clip_skip = getattr(config, "bodydouble_clip_skip", 1)
            style_preset = getattr(config, "bodydouble_style_preset", "balanced")

            ip2_variant = getattr(config, "bodydouble_ip_adapter2_variant", "") or ""
            ip2_scale = getattr(config, "bodydouble_ip_adapter2_scale", 0.5)
            style_ref = getattr(config, "bodydouble_style_ref_path", "") or ""

            images = self.sd.generate_body_double(
                image=target_image,
                mask=mask,
                source_person=source_person,
                control_images=control_images,
                prompt=config.bodydouble_prompt or "",
                negative_prompt=config.bodydouble_negative_prompt or "",
                denoising_strength=config.bodydouble_denoising_strength,
                steps=config.bodydouble_steps,
                cfg_scale=config.bodydouble_cfg_scale,
                seed=config.bodydouble_seed,
                sampler=config.bodydouble_sampler or config.img_sampler,
                scheduler=config.bodydouble_scheduler or config.img_scheduler,
                checkpoint_path=checkpoint,
                controlnet_types=controlnet_types,
                controlnet_strengths=controlnet_strengths,
                ip_adapter_variant=getattr(config, "bodydouble_ip_adapter_variant", "plus") or "plus",
                ip_adapter_scale=config.bodydouble_ip_adapter_scale,
                mask_blur=config.bodydouble_mask_blur,
                width=getattr(config, "bodydouble_width", 0),
                height=getattr(config, "bodydouble_height", 0),
                num_images_per_prompt=num_images,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                clip_skip=clip_skip,
                ip_adapter_style_preset=style_preset,
                ip_adapter2_variant=ip2_variant,
                ip_adapter2_scale=ip2_scale,
                style_ref_image=style_ref if style_ref else None,
                loras=self._resolve_lora_list(config),
                callback=callback,
            )

            return self._save_images_to_project(
                images, project_path, "bodydouble", config, "bodydouble",
            )

        except Exception:
            logger.exception("Body double generation failed")
            raise

    def run_controlnet(
        self,
        project_name: str,
        control_images: list[Any],
        source_image: Optional[str],
        config: "ProjectConfig",
        num_images: int = 1,
        callback: Optional[Callable] = None,
        mask_image: Optional[str] = None,
    ) -> list[str]:
        """Run ControlNet generation with optional inpaint mask.

        Returns list of saved image paths.
        """
        logger.info("ControlNet: project=%s mode=%s", project_name, config.controlnet_mode)
        project_path = self._resolve_project_path(project_name)

        try:
            checkpoint = self._resolve_checkpoint(
                config.controlnet_checkpoint or config.img_checkpoint
            )

            controlnet_types = [config.controlnet_type or "canny"]
            controlnet_strengths = [config.controlnet_strength]
            if config.controlnet_type2:
                controlnet_types.append(config.controlnet_type2)
                controlnet_strengths.append(config.controlnet_strength2)

            control_guidance_start = getattr(config, "controlnet_guidance_start", 0.0)
            control_guidance_end = getattr(config, "controlnet_guidance_end", 1.0)
            guess_mode = getattr(config, "controlnet_guess_mode", False)
            clip_skip = getattr(config, "controlnet_clip_skip", 1)

            src = source_image if config.controlnet_mode in ("img2img", "inpaint") else None

            images = self.sd.generate_controlnet(
                control_images=control_images,
                source_image=src,
                prompt=config.controlnet_prompt or "",
                negative_prompt=config.controlnet_negative_prompt or "",
                steps=config.controlnet_steps,
                cfg_scale=config.controlnet_cfg_scale,
                seed=config.controlnet_seed,
                sampler=config.controlnet_sampler or config.img_sampler,
                scheduler=config.controlnet_scheduler or config.img_scheduler,
                checkpoint_path=checkpoint,
                controlnet_types=controlnet_types,
                controlnet_strengths=controlnet_strengths,
                width=config.controlnet_width,
                height=config.controlnet_height,
                denoising_strength=config.controlnet_denoising_strength,
                num_images_per_prompt=num_images,
                callback=callback,
                control_guidance_start=control_guidance_start,
                control_guidance_end=control_guidance_end,
                guess_mode=guess_mode,
                clip_skip=clip_skip,
                mask_image=mask_image if config.controlnet_mode == "inpaint" else None,
                loras=self._resolve_lora_list(config),
            )

            return self._save_images_to_project(
                images, project_path, "controlnet", config, "controlnet",
            )

        except Exception:
            logger.exception("ControlNet generation failed")
            raise
