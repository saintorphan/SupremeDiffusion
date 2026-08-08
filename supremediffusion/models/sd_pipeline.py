"""Core Stable Diffusion pipeline for SD 1.5 / SDXL generation."""

from __future__ import annotations

import gc as gc_module
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from supremediffusion.models.sd_models import detect_model_type
from supremediffusion.models.sd_samplers import create_scheduler

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class SDImagePipeline:
    """High-level wrapper for Stable Diffusion image generation.

    Follows the same lazy-load / explicit-unload pattern as WanI2VPipeline.
    Supports SD 1.5 and SDXL checkpoints loaded from single .safetensors/.ckpt files.
    """

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self.pipe: Any = None
        self._img2img_pipe: Any = None
        self._inpaint_pipe: Any = None
        self._refiner_pipe: Any = None
        self._refiner_checkpoint: str = ""
        self._loaded: bool = False
        self._current_checkpoint: str = ""
        self._model_type: str = ""  # "sd15" or "sdxl"
        self._current_vae: str = ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: str,
        vae_name: str = "Automatic",
        vae_precision: str = "16",
    ) -> None:
        """Load a checkpoint from a .safetensors/.ckpt file.

        Uses from_single_file() for local checkpoint files.
        Caches the loaded checkpoint — only reloads on change.

        *vae_precision* threads the ``sd_vae_precision`` project setting
        ("16"/"32") through to a custom VAE load.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required but not installed.")

        # Skip reload if same checkpoint already loaded
        if self._loaded and self._current_checkpoint == checkpoint_path:
            # But still swap VAE if needed
            if vae_name != self._current_vae:
                self._load_vae(vae_name, vae_precision)
            return

        # Unload existing pipeline
        if self._loaded:
            self.unload()

        logger.info("Loading SD checkpoint: %s", checkpoint_path)
        model_type = detect_model_type(checkpoint_path)
        self._model_type = model_type

        try:
            if model_type == "sdxl":
                from diffusers import StableDiffusionXLPipeline
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
            else:
                from diffusers import StableDiffusionPipeline
                self.pipe = StableDiffusionPipeline.from_single_file(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
        except Exception as exc:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_path, exc)
            raise

        # Move to GPU
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")

        # Try to enable xformers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers attention for SD pipeline.")
        except Exception:
            logger.debug("xformers not available, using default attention.")

        # Enable VAE slicing and tiling to reduce peak VRAM during encode/decode
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            # Diffusers sets tile_latent_min_size = sample_size/8 (128 for SDXL),
            # but the tiling guard is `>` not `>=`, so tiling never activates at
            # native resolution.  Lower the threshold so it actually kicks in.
            self.pipe.vae.tile_latent_min_size = 32
            self.pipe.vae.tile_sample_min_size = 256
            logger.info("Enabled VAE slicing and tiling for SD pipeline.")

        # Load custom VAE if specified
        if vae_name != "Automatic":
            self._load_vae(vae_name, vae_precision)
        self._current_vae = vae_name

        self._current_checkpoint = checkpoint_path
        self._loaded = True

        # Free CPU-side loading artifacts before first generation
        gc_module.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("SD pipeline loaded (%s): %s", model_type, checkpoint_path)

    def _load_vae(self, vae_name: str, vae_precision: str = "16") -> None:
        """Load and swap in a custom VAE.

        *vae_precision* is the ``sd_vae_precision`` project setting ("16" or
        "32"); "32" maps to fp32 to escape the classic SDXL fp16-VAE
        black-image NaN, otherwise fp16.
        """
        if vae_name == "Automatic" or not vae_name:
            self._current_vae = "Automatic"
            return

        vae_dir = self.config.model_paths.get("sd_vae_dir", "")
        if not vae_dir:
            logger.warning("No sd_vae_dir configured, cannot load VAE '%s'", vae_name)
            return

        vae_dir = Path(vae_dir)
        # Find the VAE file
        vae_file = None
        for ext in (".safetensors", ".ckpt", ".pt"):
            candidate = vae_dir / f"{vae_name}{ext}"
            if candidate.exists():
                vae_file = candidate
                break

        if vae_file is None:
            logger.warning("VAE '%s' not found in %s", vae_name, vae_dir)
            return

        # fp32 escape hatch for the classic SDXL fp16-VAE black-image NaN.
        vae_dtype = torch.float32 if str(vae_precision) == "32" else torch.float16

        try:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_single_file(
                str(vae_file),
                torch_dtype=vae_dtype,
            )
            if torch.cuda.is_available():
                vae = vae.to("cuda")
            self.pipe.vae = vae
            # Invalidate cached variant pipes so they pick up the new VAE
            self._img2img_pipe = None
            self._inpaint_pipe = None
            self._current_vae = vae_name
            logger.info("Custom VAE loaded: %s", vae_file)
        except Exception as exc:
            logger.warning("Failed to load VAE %s: %s", vae_file, exc)

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Free the pipeline and release GPU memory."""
        if self._body_double_pipe is not None:
            self._teardown_body_double()
        if self._controlnet_pipe is not None:
            self._teardown_controlnet()
        if self._refiner_pipe is not None:
            del self._refiner_pipe
            self._refiner_pipe = None
            self._refiner_checkpoint = ""
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        if self.pipe is not None:
            # Move all components to CPU before delete
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
            del self.pipe
            self.pipe = None

        self._loaded = False
        self._current_checkpoint = ""
        self._model_type = ""
        self._current_vae = ""
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc_module.collect()

        logger.info("SD pipeline unloaded and CUDA cache cleared.")

    # ------------------------------------------------------------------
    # Refiner
    # ------------------------------------------------------------------

    def _load_refiner(self, checkpoint_path: str) -> Any:
        """Load an SDXL refiner checkpoint (img2img pipeline).

        Caches the loaded refiner — only reloads on checkpoint change.
        Only SDXL checkpoints can be used as refiners.
        """
        if self._refiner_pipe is not None and self._refiner_checkpoint == checkpoint_path:
            return self._refiner_pipe

        # Unload previous refiner
        if self._refiner_pipe is not None:
            del self._refiner_pipe
            self._refiner_pipe = None

        logger.info("Loading SDXL refiner: %s", checkpoint_path)
        from diffusers import StableDiffusionXLImg2ImgPipeline
        self._refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=checkpoint_path.endswith(".safetensors"),
        )
        if torch.cuda.is_available():
            self._refiner_pipe = self._refiner_pipe.to("cuda")
        self._refiner_checkpoint = checkpoint_path
        logger.info("SDXL refiner loaded.")
        return self._refiner_pipe

    def _run_refiner(
        self,
        images: list,
        prompt: str,
        negative_prompt: str,
        refiner_checkpoint: str,
        refiner_steps: int = 10,
        refiner_cfg_scale: float = 7.0,
        seed: int = -1,
        refiner_switch_at: float = 0.8,
        base_steps: int = 0,
    ) -> list:
        """Run the SDXL refiner pass on latent images from the base pass.

        Follows the diffusers ensemble-of-experts recipe: the base stops at
        ``denoising_end=refiner_switch_at`` and emits latents; the refiner picks
        up at ``denoising_start=refiner_switch_at`` so it *continues* the same
        schedule instead of re-noising with a fresh ``strength``. Both stages
        share the same ``num_inference_steps`` so the timestep grids align.

        Returns refined PIL images.
        """
        if self._model_type != "sdxl":
            logger.info("Refiner skipped — only supported for SDXL checkpoints.")
            return images

        refiner_path = self._resolve_refiner_path(refiner_checkpoint)
        if not refiner_path:
            logger.warning("Refiner checkpoint not found: %s", refiner_checkpoint)
            return images

        refiner = self._load_refiner(refiner_path)
        generator = self._make_generator(seed)

        # The refiner must share the base's total step grid for denoising_start
        # to land on a real timestep. Fall back to refiner_steps only when the
        # base count is unknown.
        total_steps = base_steps if base_steps > 0 else refiner_steps

        refined = []
        for img in images:
            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": img,
                "num_inference_steps": total_steps,
                "denoising_start": refiner_switch_at,
                "guidance_scale": refiner_cfg_scale,
            }
            if generator is not None:
                kwargs["generator"] = generator

            with torch.inference_mode():
                output = refiner(**kwargs)
            refined.extend(output.images)
            del output

        return refined

    def _resolve_refiner_path(self, name_or_path: str) -> str | None:
        """Resolve a refiner checkpoint name to a full path."""
        p = Path(name_or_path)
        if p.is_file():
            return str(p)
        # Check refiner directory
        refiner_dir = self.config.model_paths.get("sd_refiner_dir", "")
        if refiner_dir:
            candidate = Path(refiner_dir) / name_or_path
            if candidate.is_file():
                return str(candidate)
        # Check main checkpoint directory as fallback
        ckpt_dir = self.config.model_paths.get("sd_checkpoint_dir", "")
        if ckpt_dir:
            candidate = Path(ckpt_dir) / name_or_path
            if candidate.is_file():
                return str(candidate)
        return None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_txt2img(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler: str,
        scheduler: str,
        batch_size: int = 1,
        clip_skip: int = 1,
        callback: Optional[Callable] = None,
        refiner_checkpoint: str = "",
        refiner_switch_at: float = 0.8,
        refiner_steps: int = 10,
        refiner_cfg_scale: float = 7.0,
        ip_adapter_image: Any = None,
        ip_adapter_variant: str = "plus",
        ip_adapter_scale: float = 0.0,
    ) -> list:
        """Generate images from text prompt, with optional SDXL refiner pass.

        When ``ip_adapter_image`` + ``ip_adapter_scale`` > 0 are supplied, a
        CLIP-based IP-Adapter (``plus``/``plus_face``/``base``) is loaded onto
        the pipe to apply the reference image's *look* to the generation, then
        unloaded afterwards. FaceID variants need embeddings and are not
        supported on this path.
        """
        self._ensure_loaded()
        self._free_vram_before_gen()
        self._set_scheduler(sampler, scheduler)

        generator = self._make_generator(seed)
        use_refiner = bool(refiner_checkpoint) and self._model_type == "sdxl"

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "num_images_per_prompt": batch_size,
        }
        self._apply_clip_skip_kwarg(kwargs, clip_skip)

        # Optional IP-Adapter reference-look conditioning (CLIP variants only).
        ip_loaded = False
        if (
            ip_adapter_image is not None
            and ip_adapter_scale
            and float(ip_adapter_scale) > 0.0
            and not str(ip_adapter_variant).startswith("faceid")
        ):
            try:
                from PIL import Image as _Img
                from supremediffusion.models.controlnet_types import get_ip_adapter_config
                is_sdxl = self._model_type == "sdxl"
                subfolder, weight = get_ip_adapter_config(ip_adapter_variant, is_sdxl)
                load_kwargs = {"weight_name": weight}
                if subfolder is not None:
                    load_kwargs["subfolder"] = subfolder
                self.pipe.load_ip_adapter("h94/IP-Adapter", **load_kwargs)
                self.pipe.set_ip_adapter_scale(float(ip_adapter_scale))
                ref = ip_adapter_image
                if isinstance(ref, str):
                    ref = _Img.open(ref).convert("RGB")
                kwargs["ip_adapter_image"] = ref
                ip_loaded = True
                logger.info(
                    "txt2img IP-Adapter '%s' applied (scale %.2f)",
                    ip_adapter_variant, float(ip_adapter_scale),
                )
            except Exception as exc:
                logger.warning("IP-Adapter load failed (%s) — generating without it", exc)
                ip_loaded = False
        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb
        if use_refiner:
            # Output latents for refiner input
            kwargs["output_type"] = "latent"
            kwargs["denoising_end"] = refiner_switch_at

        try:
            with torch.inference_mode():
                output = self.pipe(**kwargs)
            images = output.images
            del output
        finally:
            if ip_loaded:
                try:
                    self.pipe.unload_ip_adapter()
                except Exception:
                    logger.debug("IP-Adapter unload failed", exc_info=True)

        if use_refiner:
            images = self._run_refiner(
                images, prompt, negative_prompt,
                refiner_checkpoint, refiner_steps, refiner_cfg_scale, seed,
                refiner_switch_at=refiner_switch_at, base_steps=steps,
            )

        self._cleanup_after_gen()
        return images

    def generate_img2img(
        self,
        image: Any,
        prompt: str,
        negative_prompt: str,
        denoising_strength: float,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler: str,
        scheduler: str,
        resize_mode: int = 0,
        batch_size: int = 1,
        clip_skip: int = 1,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate images from an input image + prompt."""
        self._ensure_loaded()
        self._set_scheduler(sampler, scheduler)

        # denoise=0 = passthrough (lets ADetailer post-step run on the source
        # without an img2img round-trip). Same rationale as generate_inpaint.
        # Still honor the requested w/h so a pure-resize request isn't ignored.
        if denoising_strength is not None and float(denoising_strength) <= 0.001:
            if Image is not None:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                if isinstance(image, Image.Image):
                    image = self._resize_image(image, width, height, resize_mode)
            return [image] * max(1, batch_size)

        # Ensure we have the img2img pipeline variant
        pipe = self._get_img2img_pipe()
        self._free_vram_before_gen()

        # Resize input image
        if Image is not None and isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if Image is not None and isinstance(image, Image.Image):
            image = self._resize_image(image, width, height, resize_mode)

        generator = self._make_generator(seed)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "strength": denoising_strength,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "num_images_per_prompt": batch_size,
        }
        self._apply_clip_skip_kwarg(kwargs, clip_skip)
        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb

        with torch.inference_mode():
            output = pipe(**kwargs)
        images = output.images
        del output
        self._cleanup_after_gen()
        return images

    def generate_inpaint(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        negative_prompt: str,
        denoising_strength: float,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler: str,
        scheduler: str,
        mask_blur: int = 4,
        inpainting_fill: int = 1,
        full_res: bool = False,
        padding: int = 32,
        batch_size: int = 1,
        clip_skip: int = 1,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate inpainted images.

        When *full_res* is True, only the masked region (plus *padding*) is
        sent through the diffusion pipeline and composited back onto the
        original image at full resolution, preventing progressive degradation
        of non-masked areas.

        When *full_res* is False, the pipeline still composites the original
        non-masked pixels back onto the result to avoid VAE round-trip
        degradation outside the mask.
        """
        self._ensure_loaded()
        self._set_scheduler(sampler, scheduler)

        # SDXL inpaint computes effective_steps = int(steps * strength) and
        # raises if the result is 0. Treat denoise=0 as "passthrough" so the
        # ADetailer post-step (or other downstream work) can run on the
        # untouched source. Return the original image without invoking the
        # diffusion pipeline (still honoring the requested w/h).
        if denoising_strength is not None and float(denoising_strength) <= 0.001:
            if Image is not None:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                if isinstance(image, Image.Image) and width > 0 and height > 0:
                    image = image.resize((width, height), Image.LANCZOS)
            return [image] * max(1, batch_size)

        pipe = self._get_inpaint_pipe()
        self._free_vram_before_gen()

        # Load images if paths
        if Image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            if isinstance(mask, str):
                mask = Image.open(mask).convert("L")

        # Ensure mask matches image size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Apply mask blur
        raw_mask = mask  # keep un-blurred copy for compositing
        if mask_blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # A1111-style masked-region pre-fill (inpainting_fill):
        #   0 = fill (neutral-color the region), 1 = original (default, no-op),
        #   2 = latent noise (RGB noise), 3 = latent nothing (mid-gray).
        # The pipeline still denoises this region; pre-filling just changes the
        # content the diffusion process starts from inside the mask.
        image = self._apply_inpaint_fill(image, raw_mask, inpainting_fill, seed)

        generator = self._make_generator(seed)

        if full_res:
            return self._inpaint_full_res(
                pipe, image, mask, raw_mask, prompt, negative_prompt,
                denoising_strength, width, height, steps, cfg_scale,
                padding, batch_size, generator, callback, clip_skip,
            )

        # --- Whole-image inpainting with paste-back -----------------------
        original = image.copy()
        orig_w, orig_h = original.size

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "strength": denoising_strength,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "num_images_per_prompt": batch_size,
        }
        self._apply_clip_skip_kwarg(kwargs, clip_skip)
        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb

        with torch.inference_mode():
            output = pipe(**kwargs)
        gen_images = output.images
        del output
        self._cleanup_after_gen()

        # Composite: paste original pixels back onto non-masked areas.
        # Use the larger of source vs target resolution so inpainting
        # at a higher resolution than the source produces upscaled output.
        out_w = max(orig_w, width)
        out_h = max(orig_h, height)
        comp_mask = mask.resize((out_w, out_h), Image.LANCZOS)
        if original.size != (out_w, out_h):
            original = original.resize((out_w, out_h), Image.LANCZOS)

        # Phase 4b: Optional Laplacian pyramid blending for seam-free composite.
        # Pre-existing behavior (PIL.Image.composite alpha blend) is kept as
        # the default; opt-in via a project config flag so existing workflows
        # don't change. The Laplacian path hides the inpaint mask boundary at
        # every frequency band (Photoshop healing-brush style) instead of just
        # alpha-blurring the mask edge.
        use_lap_blend = bool(getattr(self, "_use_laplacian_blend", False))
        results = []
        for gen_img in gen_images:
            if gen_img.size != (out_w, out_h):
                gen_img = gen_img.resize((out_w, out_h), Image.LANCZOS)
            if use_lap_blend:
                try:
                    from supremediffusion.utils.laplacian_blend import laplacian_pyramid_blend
                    result = laplacian_pyramid_blend(
                        background=original, foreground=gen_img, mask=comp_mask,
                        levels=5,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Laplacian blend failed (%s) — falling back to alpha composite", exc)
                    result = Image.composite(gen_img, original, comp_mask)
            else:
                result = Image.composite(gen_img, original, comp_mask)
            results.append(result)
        del gen_images, original, image, mask, raw_mask
        return results

    def _inpaint_full_res(
        self,
        pipe,
        image,
        mask,
        raw_mask,
        prompt: str,
        negative_prompt: str,
        denoising_strength: float,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        padding: int,
        batch_size: int,
        generator,
        callback,
        clip_skip: int = 1,
    ) -> list:
        """Crop to masked region, inpaint, paste back at full resolution."""
        import numpy as np

        orig_w, orig_h = image.size
        original = image.copy()

        # Find bounding box of the mask
        mask_np = np.array(raw_mask)
        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)
        if not rows.any() or not cols.any():
            # Empty mask — return original
            return [original.copy()] * batch_size

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        rmin = max(0, rmin - padding)
        rmax = min(orig_h - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(orig_w - 1, cmax + padding)

        # Crop image and mask to bounding box
        crop_box = (cmin, rmin, cmax + 1, rmax + 1)
        crop_img = image.crop(crop_box)
        crop_mask = mask.crop(crop_box)
        crop_raw_mask = raw_mask.crop(crop_box)

        # Run inpainting on the crop at the requested resolution
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": crop_img.resize((width, height), Image.LANCZOS),
            "mask_image": crop_mask.resize((width, height), Image.LANCZOS),
            "strength": denoising_strength,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "num_images_per_prompt": batch_size,
        }
        self._apply_clip_skip_kwarg(kwargs, clip_skip)
        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb

        with torch.inference_mode():
            output = pipe(**kwargs)
        gen_images = output.images
        del output
        self._cleanup_after_gen()

        # Paste each result back — use larger of source vs target resolution
        out_w = max(orig_w, width)
        out_h = max(orig_h, height)
        scale_x = out_w / orig_w
        scale_y = out_h / orig_h
        paste_x = int(cmin * scale_x)
        paste_y = int(rmin * scale_y)
        paste_w = int((cmax + 1 - cmin) * scale_x)
        paste_h = int((rmax + 1 - rmin) * scale_y)

        from PIL import ImageFilter
        comp_mask = crop_raw_mask.resize((paste_w, paste_h), Image.LANCZOS)
        comp_mask = comp_mask.filter(ImageFilter.GaussianBlur(radius=2))
        if original.size != (out_w, out_h):
            original = original.resize((out_w, out_h), Image.LANCZOS)
        crop_img_scaled = crop_img.resize((paste_w, paste_h), Image.LANCZOS)
        results = []
        for gen_img in gen_images:
            gen_crop = gen_img.resize((paste_w, paste_h), Image.LANCZOS)
            merged_crop = Image.composite(gen_crop, crop_img_scaled, comp_mask)
            result = original.copy()
            result.paste(merged_crop, (paste_x, paste_y))
            results.append(result)
        del gen_images, original, image, mask, raw_mask, crop_img, crop_mask, crop_raw_mask
        return results

    # ------------------------------------------------------------------
    # Body Double (ControlNet + IP-Adapter inpainting)
    # ------------------------------------------------------------------

    _body_double_pipe: Any = None
    _bd_cache_key: tuple | None = None
    _bd_ip_embed_cache: dict = {}  # {image_path_hash: embed_tensors}
    _face_app: Any = None  # Cached InsightFace FaceAnalysis instance

    def _get_body_double_pipe(
        self,
        checkpoint_path: str,
        controlnet_types: list[str] | None = None,
        ip_adapter_variants: list[str] | None = None,
    ):
        """Build a standalone ControlNet+Inpaint pipeline for body double.

        Loads from checkpoint independently (no shared components with main pipe)
        and uses sequential CPU offload to keep peak VRAM under ~6 GB.

        Args:
            controlnet_types: List of CN type keys (e.g. ["openpose", "depth"]).
                              Defaults to ["openpose"].
            ip_adapter_variants: List of IP-Adapter variant keys. First is primary
                                 (identity), optional second is style adapter.
                                 Defaults to ["plus"].
        """
        from supremediffusion.models.controlnet_types import (
            get_controlnet_model, get_ip_adapter_config,
        )

        if controlnet_types is None:
            controlnet_types = ["openpose"]
        if ip_adapter_variants is None:
            ip_adapter_variants = ["plus"]

        cache_key = (
            checkpoint_path,
            tuple(controlnet_types),
            tuple(ip_adapter_variants),
        )
        if self._body_double_pipe is not None and self._bd_cache_key == cache_key:
            return self._body_double_pipe
        if self._body_double_pipe is not None:
            self._teardown_body_double()

        # Fully unload main pipe to free all VRAM
        logger.info("Unloading main pipeline for body double...")
        main_was_loaded = self._loaded
        main_checkpoint = self._current_checkpoint
        main_vae = self._current_vae
        self.unload()

        model_type = detect_model_type(checkpoint_path)
        is_sdxl = model_type == "sdxl"

        # Load ControlNet model(s)
        logger.info("Loading ControlNet %s for body double...", controlnet_types)
        cn_models = [get_controlnet_model(ct, is_sdxl) for ct in controlnet_types]
        if len(cn_models) == 1:
            controlnet = cn_models[0]
        else:
            from diffusers import MultiControlNetModel
            controlnet = MultiControlNetModel(cn_models)

        # Resolve IP-Adapter config for each variant
        primary_variant = ip_adapter_variants[0]
        is_faceid = primary_variant.startswith("faceid")

        if is_faceid:
            # FaceID variants use InsightFace embeddings, not CLIP ViT-H
            # FaceID Plus also needs the LAION CLIP encoder for dual guidance
            if primary_variant == "faceid_plus":
                from transformers import CLIPVisionModelWithProjection
                logger.info("Loading LAION CLIP ViT-H for FaceID Plus...")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    torch_dtype=torch.float16,
                )
            else:
                image_encoder = None
        else:
            # Standard CLIP-based IP-Adapter
            from transformers import CLIPVisionModelWithProjection
            logger.info("Loading CLIP ViT-H image encoder for IP-Adapter...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                torch_dtype=torch.float16,
            )

        if is_sdxl:
            from diffusers import StableDiffusionXLControlNetInpaintPipeline
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                checkpoint_path,
                controlnet=controlnet,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
                use_safetensors=checkpoint_path.endswith(".safetensors"),
            )
        else:
            from diffusers import StableDiffusionControlNetInpaintPipeline
            pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                checkpoint_path,
                controlnet=controlnet,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
                use_safetensors=checkpoint_path.endswith(".safetensors"),
            )

        # Load IP-Adapter(s) — single or dual
        ip_repos = []
        ip_weights = []
        ip_subfolders = []
        for var in ip_adapter_variants:
            is_var_faceid = var.startswith("faceid")
            subfolder, weight = get_ip_adapter_config(var, is_sdxl)
            ip_repos.append("h94/IP-Adapter-FaceID" if is_var_faceid else "h94/IP-Adapter")
            ip_weights.append(weight)
            ip_subfolders.append(subfolder)

        if len(ip_adapter_variants) == 1:
            logger.info("Loading IP-Adapter %s...", ip_adapter_variants[0])
            load_kwargs = {"weight_name": ip_weights[0]}
            if ip_subfolders[0] is not None:
                load_kwargs["subfolder"] = ip_subfolders[0]
            if is_faceid and primary_variant != "faceid_plus":
                load_kwargs["image_encoder_folder"] = None
            pipe.load_ip_adapter(ip_repos[0], **load_kwargs)
        else:
            # Dual IP-Adapter: load both adapters at once
            logger.info("Loading dual IP-Adapters: %s...", ip_adapter_variants)
            load_kwargs: dict = {
                "weight_name": ip_weights,
            }
            # Subfolder list — use "" for entries with None (FaceID)
            subfolder_list = [s if s is not None else "" for s in ip_subfolders]
            if any(s for s in subfolder_list):
                load_kwargs["subfolder"] = subfolder_list
            if is_faceid and primary_variant != "faceid_plus":
                load_kwargs["image_encoder_folder"] = None
            pipe.load_ip_adapter(ip_repos, **load_kwargs)

        pipe.enable_sequential_cpu_offload()

        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        self._body_double_pipe = pipe
        self._bd_cache_key = cache_key
        # Stash info so we can reload the main pipe after teardown
        self._bd_main_was_loaded = main_was_loaded
        self._bd_main_checkpoint = main_checkpoint
        self._bd_main_vae = main_vae
        logger.info("Body double pipeline ready (sequential offload).")
        return pipe

    def generate_body_double(
        self,
        image: Any,
        mask: Any,
        source_person: Any,
        control_images: list[Any] | Any = None,
        prompt: str = "",
        negative_prompt: str = "",
        denoising_strength: float = 0.85,
        steps: int = 30,
        cfg_scale: float = 7.5,
        seed: int = -1,
        sampler: str = "DPM++ 3M SDE",
        scheduler: str = "Karras",
        checkpoint_path: str = "",
        controlnet_types: list[str] | None = None,
        controlnet_strengths: list[float] | None = None,
        ip_adapter_variant: str = "plus",
        ip_adapter_scale: float = 0.6,
        ip_adapter_style_preset: str = "balanced",
        mask_blur: int = 8,
        width: int = 0,
        height: int = 0,
        num_images_per_prompt: int = 1,
        callback: Optional[Callable] = None,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
        clip_skip: int = 1,
        ip_adapter2_variant: str = "",
        ip_adapter2_scale: float = 0.5,
        style_ref_image: Any = None,
        loras: list[dict] | None = None,
        # Legacy parameter name — maps to control_images if provided
        pose_image: Any = None,
    ) -> list:
        """Generate body double: replace a person using ControlNet + IP-Adapter appearance.

        Args:
            control_images: List of condition images (one per ControlNet type).
                            Also accepts a single image for backward compatibility.
            controlnet_types: List of CN type keys. Defaults to ["openpose"].
            controlnet_strengths: List of conditioning scales. Defaults to [0.7].
            ip_adapter_variant: Primary IP-Adapter weight variant key.
            ip_adapter2_variant: Optional second IP-Adapter for style transfer.
            ip_adapter2_scale: Scale for the second IP-Adapter.
            style_ref_image: Reference image for the second IP-Adapter.
            width/height: Override dimensions. 0 = use source image dimensions.
        """
        if not checkpoint_path:
            raise ValueError("No checkpoint specified for body double.")

        if controlnet_types is None:
            controlnet_types = ["openpose"]
        if controlnet_strengths is None:
            controlnet_strengths = [0.7]

        # Backward compatibility: pose_image → control_images
        if control_images is None and pose_image is not None:
            control_images = [pose_image]
        elif control_images is not None and not isinstance(control_images, list):
            control_images = [control_images]
        if control_images is None:
            raise ValueError("No control images provided.")

        # Build IP-Adapter variant list for pipe caching
        ip_variants = [ip_adapter_variant]
        has_dual_ip = bool(ip_adapter2_variant and style_ref_image)
        if has_dual_ip:
            ip_variants.append(ip_adapter2_variant)

        pipe = self._get_body_double_pipe(
            checkpoint_path,
            controlnet_types=controlnet_types,
            ip_adapter_variants=ip_variants,
        )

        # Set scheduler on the body double pipe directly
        sched_config = dict(pipe.scheduler.config)
        pipe.scheduler = create_scheduler(sampler, scheduler, sched_config)

        # Clip skip is applied per-call below (passed into the diffusers
        # __call__ as ``clip_skip``) — mutating text_encoder.config has no
        # effect because the encoder layers are built once at load time.

        # Set IP-Adapter scale(s)
        if has_dual_ip:
            pipe.set_ip_adapter_scale([ip_adapter_scale, ip_adapter2_scale])
        else:
            pipe.set_ip_adapter_scale(ip_adapter_scale)
            # Apply style preset (InstantStyle layer targeting) — single adapter only
            if ip_adapter_style_preset == "style_only":
                pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, ip_adapter_scale, 0.0]}})
            elif ip_adapter_style_preset == "layout_style":
                pipe.set_ip_adapter_scale({
                    "down": {"block_2": [0.0, ip_adapter_scale]},
                    "up": {"block_0": [0.0, ip_adapter_scale, 0.0]},
                })

        # Load images if paths
        if Image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            if isinstance(mask, str):
                mask = Image.open(mask).convert("L")
            if isinstance(source_person, str):
                source_person = Image.open(source_person).convert("RGB")
            if isinstance(style_ref_image, str):
                style_ref_image = Image.open(style_ref_image).convert("RGB")
            for i, ci in enumerate(control_images):
                if isinstance(ci, str):
                    control_images[i] = Image.open(ci).convert("RGB")

        # Ensure mask matches image size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        # Apply mask blur
        original = image.copy()
        raw_mask = mask
        if mask_blur > 0:
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

        # Resize control images to match
        for i, ci in enumerate(control_images):
            if ci.size != image.size:
                control_images[i] = ci.resize(image.size, Image.LANCZOS)

        generator = self._make_generator(seed)
        orig_w, orig_h = image.size

        # Use override dimensions if provided, else source dimensions
        if width > 0 and height > 0:
            gen_w = (width // 8) * 8
            gen_h = (height // 8) * 8
        else:
            gen_w = (orig_w // 8) * 8
            gen_h = (orig_h // 8) * 8

        # Resize all inputs to generation size if needed
        if (gen_w, gen_h) != (orig_w, orig_h):
            image = image.resize((gen_w, gen_h), Image.LANCZOS)
            mask = mask.resize((gen_w, gen_h), Image.NEAREST)
            raw_mask = raw_mask.resize((gen_w, gen_h), Image.NEAREST)
            for i, ci in enumerate(control_images):
                control_images[i] = ci.resize((gen_w, gen_h), Image.LANCZOS)

        # Prepare control_image: single image or list for MultiControlNet
        ctrl_img = control_images[0] if len(control_images) == 1 else control_images
        # Prepare controlnet_conditioning_scale
        cn_scale = controlnet_strengths[0] if len(controlnet_strengths) == 1 else controlnet_strengths

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask,
            "control_image": ctrl_img,
            "strength": denoising_strength,
            "width": gen_w,
            "height": gen_h,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "controlnet_conditioning_scale": cn_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
        }
        self._apply_clip_skip_kwarg(kwargs, clip_skip)

        # Prepare IP-Adapter image embeddings
        _ip_cache_key = id(source_person) if not isinstance(source_person, str) else source_person

        if ip_adapter_variant.startswith("faceid"):
            # FaceID: use InsightFace face embeddings
            cached = self._bd_ip_embed_cache.get(("faceid", _ip_cache_key))
            if cached is not None:
                face_embeds = cached
            else:
                face_embeds = self._extract_face_embeddings(source_person)
                self._bd_ip_embed_cache[("faceid", _ip_cache_key)] = face_embeds
            if has_dual_ip:
                # Dual IP: FaceID embeds for adapter 0, CLIP embeds for adapter 1
                style_embeds = pipe.prepare_ip_adapter_image_embeds(
                    [style_ref_image], None, torch.device("cpu"),
                    num_images_per_prompt, True,
                )
                # Interleave: [adapter0_embeds, adapter1_embeds]
                kwargs["ip_adapter_image_embeds"] = face_embeds + style_embeds
            else:
                kwargs["ip_adapter_image_embeds"] = face_embeds
            if ip_adapter_variant == "faceid_plus":
                clip_embeds = pipe.prepare_ip_adapter_image_embeds(
                    [source_person], None, torch.device("cpu"),
                    num_images_per_prompt, True,
                )[0]
                pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = (
                    clip_embeds.to(dtype=torch.float16)
                )
                pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False
        elif has_dual_ip:
            # Dual CLIP IP-Adapters: prepare embeds for each adapter separately
            person_embeds = pipe.prepare_ip_adapter_image_embeds(
                [source_person], None, torch.device("cpu"),
                num_images_per_prompt, True,
            )
            style_embeds = pipe.prepare_ip_adapter_image_embeds(
                [style_ref_image], None, torch.device("cpu"),
                num_images_per_prompt, True,
            )
            kwargs["ip_adapter_image_embeds"] = person_embeds + style_embeds
        else:
            # Single CLIP IP-Adapter — cache embeddings for reuse
            cached = self._bd_ip_embed_cache.get(("clip", _ip_cache_key))
            if cached is not None:
                kwargs["ip_adapter_image_embeds"] = cached
            else:
                embeds = pipe.prepare_ip_adapter_image_embeds(
                    [source_person], None, torch.device("cpu"),
                    num_images_per_prompt, True,
                )
                self._bd_ip_embed_cache[("clip", _ip_cache_key)] = embeds
                kwargs["ip_adapter_image_embeds"] = embeds

        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb

        self._apply_loras_to_pipe(pipe, loras or [])
        try:
            with torch.inference_mode():
                output = pipe(**kwargs)
            gen_images = output.images
            del output
        finally:
            if loras:
                self._remove_loras_from_pipe(pipe)

        # Composite: paste original pixels back onto non-masked areas
        orig_w, orig_h = original.size
        comp_mask = raw_mask.resize((orig_w, orig_h), Image.LANCZOS)
        if mask_blur > 0:
            from PIL import ImageFilter
            comp_mask = comp_mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        results = []
        for gen_img in gen_images:
            if gen_img.size != (orig_w, orig_h):
                gen_img = gen_img.resize((orig_w, orig_h), Image.LANCZOS)
            result = Image.composite(gen_img, original, comp_mask)
            results.append(result)
        del gen_images, original, image, mask, raw_mask, comp_mask
        del source_person, control_images

        return results

    @classmethod
    def _get_face_app(cls):
        """Return a cached InsightFace FaceAnalysis instance."""
        if cls._face_app is None:
            from insightface.app import FaceAnalysis
            logger.info("Loading InsightFace buffalo_l model...")
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            cls._face_app = app
        return cls._face_app

    @classmethod
    def _extract_face_embeddings(cls, image) -> list:
        """Extract face embeddings from an image using InsightFace.

        Returns a list of embedding tensors suitable for ip_adapter_image_embeds.
        """
        import numpy as np
        import cv2

        app = cls._get_face_app()

        # Convert PIL to cv2
        img_array = np.array(image)
        if img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        faces = app.get(img_array)

        if not faces:
            raise RuntimeError(
                "No face detected in the appearance image. "
                "FaceID requires a clearly visible face — try a different image "
                "or switch to a non-FaceID IP-Adapter variant."
            )

        face_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        # Format: [neg_embeds, pos_embeds] for classifier-free guidance
        neg_embeds = torch.zeros_like(face_embed)
        id_embeds = torch.cat([neg_embeds, face_embed]).unsqueeze(0).to(dtype=torch.float16)
        return [id_embeds]

    def _teardown_body_double(self) -> None:
        """Destroy the body double pipeline and free all VRAM."""
        pipe = self._body_double_pipe
        if pipe is not None:
            # 1. Remove accelerate sequential-offload hooks
            try:
                pipe.remove_all_hooks()
            except Exception:
                pass

            # 2. Move every submodule to CPU to release CUDA tensors
            for attr in ("unet", "vae", "text_encoder", "text_encoder_2",
                         "controlnet", "image_encoder", "safety_checker"):
                component = getattr(pipe, attr, None)
                if component is not None:
                    try:
                        component.to("cpu")
                    except Exception:
                        pass
                    try:
                        setattr(pipe, attr, None)
                    except Exception:
                        pass

            # 3. Clear IP-Adapter state
            if hasattr(pipe, "unet") and pipe.unet is not None:
                try:
                    pipe.unet.encoder_hid_proj = None
                except Exception:
                    pass

            self._body_double_pipe = None
            self._bd_cache_key = None
            self._bd_ip_embed_cache = {}
            del pipe

            # Release cached InsightFace app
            if SDImagePipeline._face_app is not None:
                SDImagePipeline._face_app = None

        # 4. Aggressive cleanup — multiple passes
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Body double cleaned up, VRAM freed.")

        # 5. Restore the main pipeline that was unloaded for body double
        if getattr(self, "_bd_main_was_loaded", False):
            ckpt = getattr(self, "_bd_main_checkpoint", None)
            vae = getattr(self, "_bd_main_vae", None) or "Automatic"
            if ckpt:
                try:
                    logger.info("Restoring main SD pipeline: %s", ckpt)
                    self.load(ckpt, vae)
                except Exception:
                    logger.warning("Failed to restore main SD pipeline after body double", exc_info=True)
        self._bd_main_was_loaded = False
        self._bd_main_checkpoint = None
        self._bd_main_vae = None

    def unload_body_double(self) -> None:
        """Free the body double pipeline."""
        self._teardown_body_double()

    # ------------------------------------------------------------------
    # ControlNet generation (no IP-Adapter)
    # ------------------------------------------------------------------

    _controlnet_pipe: Any = None
    _cn_cache_key: tuple | None = None

    def _get_controlnet_pipe(
        self,
        checkpoint_path: str,
        controlnet_types: list[str],
        mode: str = "txt2img",
    ):
        """Build a standalone ControlNet pipeline for general generation.

        Args:
            mode: "txt2img", "img2img", or "inpaint"
        """
        cache_key = (checkpoint_path, tuple(controlnet_types), mode)
        if self._controlnet_pipe is not None and self._cn_cache_key == cache_key:
            return self._controlnet_pipe
        # Otherwise teardown existing and rebuild
        if self._controlnet_pipe is not None:
            self._teardown_controlnet()

        from supremediffusion.models.controlnet_types import get_controlnet_model

        # Fully unload main pipe to free all VRAM
        logger.info("Unloading main pipeline for ControlNet...")
        main_was_loaded = self._loaded
        main_checkpoint = self._current_checkpoint
        main_vae = self._current_vae
        self.unload()

        model_type = detect_model_type(checkpoint_path)
        is_sdxl = model_type == "sdxl"

        # Load ControlNet model(s)
        logger.info("Loading ControlNet %s...", controlnet_types)
        cn_models = [get_controlnet_model(ct, is_sdxl) for ct in controlnet_types]
        if len(cn_models) == 1:
            controlnet = cn_models[0]
        else:
            from diffusers import MultiControlNetModel
            controlnet = MultiControlNetModel(cn_models)

        if is_sdxl:
            if mode == "inpaint":
                from diffusers import StableDiffusionXLControlNetInpaintPipeline
                pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
            elif mode == "img2img":
                from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
            else:
                from diffusers import StableDiffusionXLControlNetPipeline
                pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
        else:
            if mode == "inpaint":
                from diffusers import StableDiffusionControlNetInpaintPipeline
                pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
            elif mode == "img2img":
                from diffusers import StableDiffusionControlNetImg2ImgPipeline
                pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )
            else:
                from diffusers import StableDiffusionControlNetPipeline
                pipe = StableDiffusionControlNetPipeline.from_single_file(
                    checkpoint_path,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=checkpoint_path.endswith(".safetensors"),
                )

        pipe.enable_sequential_cpu_offload()

        if hasattr(pipe, "vae"):
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        self._controlnet_pipe = pipe
        self._cn_cache_key = cache_key
        self._cn_main_was_loaded = main_was_loaded
        self._cn_main_checkpoint = main_checkpoint
        self._cn_main_vae = main_vae
        logger.info("ControlNet pipeline ready (sequential offload).")
        return pipe

    def generate_controlnet(
        self,
        control_images: list[Any],
        source_image: Any | None,
        prompt: str,
        negative_prompt: str,
        steps: int,
        cfg_scale: float,
        seed: int,
        sampler: str,
        scheduler: str,
        checkpoint_path: str,
        controlnet_types: list[str],
        controlnet_strengths: list[float],
        width: int,
        height: int,
        denoising_strength: float = 0.75,
        num_images_per_prompt: int = 1,
        callback: Optional[Callable] = None,
        control_guidance_start: float | list[float] = 0.0,
        control_guidance_end: float | list[float] = 1.0,
        guess_mode: bool = False,
        clip_skip: int = 1,
        mask_image: Any | None = None,
        loras: list[dict] | None = None,
    ) -> list:
        """Generate images with ControlNet conditioning (no IP-Adapter).

        Supports txt2img, img2img, and inpaint modes.

        *loras* is an optional ``[{"name", "weight"}]`` list applied to the
        specialized ControlNet pipe for this call and removed afterwards.
        """
        if not checkpoint_path:
            raise ValueError("No checkpoint specified for ControlNet generation.")

        if mask_image is not None:
            mode = "inpaint"
        elif source_image is not None:
            mode = "img2img"
        else:
            mode = "txt2img"
        pipe = self._get_controlnet_pipe(checkpoint_path, controlnet_types, mode)

        # Set scheduler
        sched_config = dict(pipe.scheduler.config)
        pipe.scheduler = create_scheduler(sampler, scheduler, sched_config)

        # Load images if paths
        for i, ci in enumerate(control_images):
            if isinstance(ci, str):
                control_images[i] = Image.open(ci).convert("RGB")
        if source_image is not None and isinstance(source_image, str):
            source_image = Image.open(source_image).convert("RGB")

        generator = self._make_generator(seed)
        gen_w = (width // 8) * 8
        gen_h = (height // 8) * 8

        # Resize control images
        for i, ci in enumerate(control_images):
            if ci.size != (gen_w, gen_h):
                control_images[i] = ci.resize((gen_w, gen_h), Image.LANCZOS)

        ctrl_img = control_images[0] if len(control_images) == 1 else control_images
        cn_scale = controlnet_strengths[0] if len(controlnet_strengths) == 1 else controlnet_strengths

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": ctrl_img if mode == "txt2img" else source_image,
            "width": gen_w,
            "height": gen_h,
            "num_inference_steps": steps,
            "guidance_scale": cfg_scale,
            "controlnet_conditioning_scale": cn_scale,
            "num_images_per_prompt": num_images_per_prompt,
        }
        kwargs["control_guidance_start"] = control_guidance_start
        kwargs["control_guidance_end"] = control_guidance_end
        kwargs["guess_mode"] = guess_mode
        self._apply_clip_skip_kwarg(kwargs, clip_skip)

        if mode == "txt2img":
            # For txt2img ControlNet, the control image goes as "image"
            pass
        elif mode == "inpaint":
            if isinstance(mask_image, str):
                mask_image = Image.open(mask_image).convert("L")
            kwargs["image"] = source_image.resize((gen_w, gen_h), Image.LANCZOS)
            kwargs["mask_image"] = mask_image.resize((gen_w, gen_h), Image.LANCZOS)
            kwargs["control_image"] = ctrl_img
            kwargs["strength"] = denoising_strength
        else:
            # For img2img, source goes as "image", control as "control_image"
            kwargs["image"] = source_image.resize((gen_w, gen_h), Image.LANCZOS)
            kwargs["control_image"] = ctrl_img
            kwargs["strength"] = denoising_strength

        if generator is not None:
            kwargs["generator"] = generator
        if callback is not None:
            _user_cb = callback
            def _safe_cb(pipe, step, timestep, cb_kwargs):
                return _user_cb(pipe, step, timestep, cb_kwargs)
            kwargs["callback_on_step_end"] = _safe_cb

        self._apply_loras_to_pipe(pipe, loras or [])
        try:
            with torch.inference_mode():
                output = pipe(**kwargs)
            results = output.images
            del output
        finally:
            if loras:
                self._remove_loras_from_pipe(pipe)

        return list(results)

    def _teardown_controlnet(self) -> None:
        """Destroy the ControlNet pipeline and free all VRAM."""
        pipe = self._controlnet_pipe
        if pipe is not None:
            try:
                pipe.remove_all_hooks()
            except Exception:
                pass
            for attr in ("unet", "vae", "text_encoder", "text_encoder_2",
                         "controlnet", "safety_checker"):
                component = getattr(pipe, attr, None)
                if component is not None:
                    try:
                        component.to("cpu")
                    except Exception:
                        pass
                    try:
                        setattr(pipe, attr, None)
                    except Exception:
                        pass
            self._controlnet_pipe = None
            self._cn_cache_key = None
            del pipe

        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("ControlNet pipeline cleaned up, VRAM freed.")

    def unload_controlnet(self) -> None:
        """Free the ControlNet pipeline."""
        self._teardown_controlnet()

    # ------------------------------------------------------------------
    # LoRA support
    # ------------------------------------------------------------------

    def apply_loras(self, lora_list: list[dict]) -> None:
        """Load and fuse LoRA weights onto the main txt2img pipe.

        lora_list: [{"name": "lora_stem", "weight": 0.8}, ...]
        """
        if not self._loaded or self.pipe is None:
            return
        self._apply_loras_to_pipe(self.pipe, lora_list)

    def _apply_loras_to_pipe(self, pipe: Any, lora_list: list[dict]) -> None:
        """Load and set LoRA adapters on an explicit *pipe*.

        Shared by the main txt2img pipe and the specialized ControlNet /
        body-double pipes. ``lora_list`` is the same ``[{"name", "weight"}]``
        shape used by :meth:`apply_loras`.
        """
        if pipe is None or not lora_list:
            return

        lora_dir = self.config.model_paths.get("sd_lora_dir", "")
        if not lora_dir:
            logger.warning("No sd_lora_dir configured, cannot load LoRAs.")
            return

        lora_dir = Path(lora_dir)
        adapter_names = []
        adapter_weights = []

        for i, lora_info in enumerate(lora_list):
            name = lora_info.get("name", "")
            weight = lora_info.get("weight", 1.0)
            if not name:
                continue

            # Find the file — check root first, then search subdirectories.
            # ``name`` may be a stem ("my_lora") or a filename
            # ("my_lora.safetensors") depending on the caller.
            lora_file = None

            # Build candidate names: exact name + stem+ext variants
            candidates = [name]
            stem = Path(name).stem
            if stem != name:  # name already had an extension
                candidates.append(stem)

            for cand in candidates:
                # Flat check
                direct = lora_dir / cand
                if direct.is_file():
                    lora_file = direct
                    break
                # Flat check with extensions
                for ext in (".safetensors", ".ckpt", ".pt"):
                    flat = lora_dir / f"{cand}{ext}"
                    if flat.is_file():
                        lora_file = flat
                        break
                if lora_file:
                    break
                # Recursive search
                for ext in ("", ".safetensors", ".ckpt", ".pt"):
                    pattern = f"{cand}{ext}" if ext else cand
                    for match in lora_dir.rglob(pattern):
                        if match.is_file():
                            lora_file = match
                            break
                    if lora_file:
                        break
                if lora_file:
                    break

            if lora_file is None:
                logger.warning("LoRA '%s' not found in %s", name, lora_dir)
                continue

            adapter_name = f"lora_{i}"
            try:
                pipe.load_lora_weights(
                    str(lora_file),
                    adapter_name=adapter_name,
                )
                adapter_names.append(adapter_name)
                adapter_weights.append(weight)
                logger.info("Loaded LoRA: %s (weight=%.2f)", name, weight)
            except Exception as exc:
                logger.warning("Failed to load LoRA %s: %s", name, exc)

        if adapter_names:
            try:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                logger.info("Applied %d LoRA(s)", len(adapter_names))
            except Exception as exc:
                logger.warning("Failed to set LoRA adapters: %s", exc)

    def remove_loras(self) -> None:
        """Remove all loaded LoRAs from the main txt2img pipe."""
        self._remove_loras_from_pipe(self.pipe)

    @staticmethod
    def _remove_loras_from_pipe(pipe: Any) -> None:
        """Unload LoRA weights from an explicit *pipe* (best-effort)."""
        if pipe is None:
            return
        try:
            pipe.unload_lora_weights()
            logger.info("LoRA weights unloaded.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def current_checkpoint(self) -> str:
        return self._current_checkpoint

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cleanup_after_gen() -> None:
        """Free intermediate CUDA tensors after generation.

        Called after `del output` so gc.collect() can reclaim the
        CUDA tensors held by the now-unreachable diffusers output object.
        """
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if not self._loaded or self.pipe is None:
            raise RuntimeError(
                "SD pipeline not loaded. Select a checkpoint first."
            )

    def _free_vram_before_gen(self) -> None:
        """Free VRAM before generation to prevent OOM on tight GPUs."""
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _set_scheduler(self, sampler: str, scheduler: str) -> None:
        """Configure the pipeline's scheduler from sampler/scheduler names."""
        config = dict(self.pipe.scheduler.config)
        self.pipe.scheduler = create_scheduler(sampler, scheduler, config)

    @staticmethod
    def _apply_clip_skip_kwarg(kwargs: dict, clip_skip: int) -> None:
        """Add diffusers' ``clip_skip`` to a pipeline call.

        Mutating ``text_encoder.config.num_hidden_layers`` is a no-op because
        the CLIP encoder layers are built once at load and the forward never
        re-reads config. diffusers' ``__call__`` instead accepts a ``clip_skip``
        arg and slices the prompt embeddings accordingly. The A1111 convention
        used by this app treats ``clip_skip=1`` as "no skip" and ``clip_skip=2``
        as "penultimate layer", which maps to diffusers' ``clip_skip = n - 1``.
        """
        try:
            n = int(clip_skip)
        except (TypeError, ValueError):
            return
        if n > 1:
            kwargs["clip_skip"] = n - 1

    def _apply_inpaint_fill(self, image, raw_mask, inpainting_fill: int, seed: int):
        """A1111-style masked-region pre-fill.

        ``inpainting_fill`` modes (matching the UI dropdown):
            0 = fill          → masked region set to the mean of unmasked pixels
            1 = original      → leave the masked content untouched (default)
            2 = latent noise  → masked region set to random RGB noise
            3 = latent nothing→ masked region set to neutral mid-gray

        Only pixels where ``raw_mask`` > 0 are modified; everything else is
        preserved so the downstream paste-back composite stays correct. We work
        in image space (not latent space) so this is backend-agnostic; the
        diffusion pass still denoises the region from this starting content.
        """
        try:
            mode = int(inpainting_fill)
        except (TypeError, ValueError):
            return image
        if mode == 1:  # original — no change
            return image
        if Image is None:
            return image

        import numpy as np

        arr = np.array(image.convert("RGB"), dtype=np.float32)
        mask_np = np.array(raw_mask.convert("L"))
        sel = mask_np > 0
        if not sel.any():
            return image

        if mode == 0:  # fill: average color of the unmasked region
            unmasked = ~sel
            if unmasked.any():
                fill_color = arr[unmasked].mean(axis=0)
            else:
                fill_color = arr.reshape(-1, 3).mean(axis=0)
            arr[sel] = fill_color
        elif mode == 2:  # latent noise → image-space RGB noise
            rng = np.random.default_rng(None if seed < 0 else seed)
            noise = rng.integers(0, 256, size=(int(sel.sum()), 3)).astype(np.float32)
            arr[sel] = noise
        elif mode == 3:  # latent nothing → neutral mid-gray
            arr[sel] = 127.0
        else:
            return image

        return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")

    def _make_generator(self, seed: int):
        """Create a torch Generator for reproducible results."""
        if seed < 0 or torch is None:
            return None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.Generator(device=device).manual_seed(seed)

    def _get_img2img_pipe(self):
        """Get or create a cached img2img pipeline variant sharing components."""
        if self._img2img_pipe is not None:
            # Keep scheduler in sync with the main pipe
            self._img2img_pipe.scheduler = self.pipe.scheduler
            return self._img2img_pipe

        if self._model_type == "sdxl":
            from diffusers import StableDiffusionXLImg2ImgPipeline
            self._img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
            )
        else:
            from diffusers import StableDiffusionImg2ImgPipeline
            self._img2img_pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

        # Re-enable memory optimisations — pipeline construction can reset them
        self._img2img_pipe.enable_vae_tiling()
        self._img2img_pipe.enable_vae_slicing()
        try:
            self._img2img_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        return self._img2img_pipe

    def _get_inpaint_pipe(self):
        """Get or create a cached inpainting pipeline variant sharing components."""
        if self._inpaint_pipe is not None:
            # Keep scheduler in sync with the main pipe
            self._inpaint_pipe.scheduler = self.pipe.scheduler
            return self._inpaint_pipe

        if self._model_type == "sdxl":
            from diffusers import StableDiffusionXLInpaintPipeline
            self._inpaint_pipe = StableDiffusionXLInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
            )
        else:
            from diffusers import StableDiffusionInpaintPipeline
            self._inpaint_pipe = StableDiffusionInpaintPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

        # Re-enable memory optimisations — pipeline construction can reset them
        self._inpaint_pipe.enable_vae_tiling()
        self._inpaint_pipe.enable_vae_slicing()
        try:
            self._inpaint_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        return self._inpaint_pipe

    @staticmethod
    def _resize_image(image, width: int, height: int, resize_mode: int = 0):
        """Resize image according to resize_mode.

        0 = Just resize (stretch)
        1 = Crop and resize (center crop to aspect ratio, then resize)
        2 = Resize and fill (letterbox/pillarbox)
        """
        if image.width == width and image.height == height:
            return image

        if resize_mode == 0:
            return image.resize((width, height), Image.LANCZOS)
        elif resize_mode == 1:
            # Center crop to target aspect ratio, then resize
            target_ratio = width / height
            img_ratio = image.width / image.height
            if img_ratio > target_ratio:
                new_w = int(image.height * target_ratio)
                left = (image.width - new_w) // 2
                image = image.crop((left, 0, left + new_w, image.height))
            else:
                new_h = int(image.width / target_ratio)
                top = (image.height - new_h) // 2
                image = image.crop((0, top, image.width, top + new_h))
            return image.resize((width, height), Image.LANCZOS)
        elif resize_mode == 2:
            # Resize to fit, then pad
            img_ratio = image.width / image.height
            target_ratio = width / height
            if img_ratio > target_ratio:
                new_w = width
                new_h = int(width / img_ratio)
            else:
                new_h = height
                new_w = int(height * img_ratio)
            resized = image.resize((new_w, new_h), Image.LANCZOS)
            result = Image.new("RGB", (width, height), (0, 0, 0))
            result.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
            return result
        else:
            return image.resize((width, height), Image.LANCZOS)
