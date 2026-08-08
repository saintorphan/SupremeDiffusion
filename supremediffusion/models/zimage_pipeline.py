"""Core Z-Image Turbo pipeline for image generation."""

from __future__ import annotations

import gc as gc_module
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class ZImageModelPipeline:
    """Wrapper for Z-Image Turbo diffusers pipelines (alibaba-pai/Z-Image-Turbo).

    Supports three generation modes:
    - txt2img via ZImagePipeline
    - img2img via ZImageImg2ImgPipeline
    - inpainting via ZImageInpaintPipeline

    The model is 6B params and uses bfloat16 precision.
    Default: 9 steps, guidance_scale=0.0.

    Single from_pretrained call loads everything (text encoder, VAE,
    transformer all bundled). img2img and inpaint variants are created
    via from_pipe() to share components.
    """

    DEFAULT_STEPS = 9
    DEFAULT_GUIDANCE = 0.0

    def __init__(self, global_config: Any, model_dir: Optional[str] = None) -> None:
        self.config = global_config
        self._txt2img_pipe: Any = None
        self._img2img_pipe: Any = None
        self._inpaint_pipe: Any = None
        self._loaded: bool = False
        self._model_path: str = ""

        if model_dir is not None:
            self.load(model_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_dir: str) -> None:
        """Load Z-Image Turbo pipeline from a local path.

        Accepts either:
        - A single .safetensors file (fp8/bf16 AIO checkpoint)
        - A directory in diffusers split format

        Also creates img2img and inpaint variants via from_pipe() for
        component sharing. Caches the loaded model -- only reloads on
        path change.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required but not installed.")

        if self._loaded and self._model_path == model_dir:
            return

        # Unload existing pipelines
        if self._loaded:
            self.unload()

        logger.info("Loading Z-Image Turbo pipeline from: %s", model_dir)
        model_path = Path(model_dir)
        is_single_file = model_path.is_file() and model_path.suffix == ".safetensors"

        try:
            from diffusers import ZImagePipeline

            if is_single_file:
                # Single-file checkpoints don't include the Qwen3 text
                # encoder or VAE.  Load them separately.
                from diffusers import AutoencoderKL

                text_encoder = self._load_quantized_text_encoder()

                logger.info("Loading VAE from %s...", self._HF_REPO)
                vae = AutoencoderKL.from_pretrained(
                    self._HF_REPO,
                    subfolder="vae",
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )

                self._txt2img_pipe = ZImagePipeline.from_single_file(
                    str(model_path),
                    text_encoder=text_encoder,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                )
                # Move everything to CPU so enable_model_cpu_offload can
                # shuttle components to GPU one at a time during inference.
                self._txt2img_pipe.to("cpu")
                logger.info("Z-Image txt2img pipeline loaded from single file.")
            else:
                self._txt2img_pipe = ZImagePipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.bfloat16,
                )
                logger.info("Z-Image txt2img pipeline loaded from directory.")

            self._txt2img_pipe.enable_model_cpu_offload()

            # Create img2img variant sharing components
            from diffusers import ZImageImg2ImgPipeline

            self._img2img_pipe = ZImageImg2ImgPipeline.from_pipe(
                self._txt2img_pipe,
            )
            logger.info("Z-Image img2img pipeline created from pipe.")

            # Create inpaint variant sharing components
            from diffusers import ZImageInpaintPipeline

            self._inpaint_pipe = ZImageInpaintPipeline.from_pipe(
                self._txt2img_pipe,
            )
            logger.info("Z-Image inpaint pipeline created from pipe.")

        except Exception as exc:
            logger.error("Failed to load Z-Image pipeline from %s: %s", model_dir, exc)
            self.unload()
            raise

        self._model_path = model_dir
        self._loaded = True

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Free all pipelines and release GPU memory."""
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        if self._txt2img_pipe is not None:
            del self._txt2img_pipe
            self._txt2img_pipe = None
        self._loaded = False
        self._model_path = ""
        self._gc_cleanup()
        logger.info("Z-Image pipelines unloaded and CUDA cache cleared.")

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def apply_loras(
        self,
        lora_dir: str,
        lora_names: list[str],
        multipliers: list[float] | None = None,
    ) -> None:
        """Load and fuse LoRA weights into the pipeline.

        Args:
            lora_dir: Directory containing .safetensors LoRA files.
            lora_names: List of filenames to load.
            multipliers: Per-LoRA scale factors (defaults to 1.0 each).
        """
        self._ensure_loaded()

        # Unfuse any previously applied LoRAs
        try:
            self._txt2img_pipe.unfuse_lora()
            self._txt2img_pipe.unload_lora_weights()
        except Exception:
            pass

        if not lora_names:
            return

        lora_path = Path(lora_dir)
        if multipliers is None:
            multipliers = [1.0] * len(lora_names)
        elif len(multipliers) < len(lora_names):
            multipliers = multipliers + [1.0] * (len(lora_names) - len(multipliers))

        adapter_names = []
        for i, name in enumerate(lora_names):
            fp = lora_path / name
            if not fp.is_file():
                logger.warning("LoRA file not found: %s", fp)
                continue
            adapter_name = f"lora_{i}"
            try:
                self._txt2img_pipe.load_lora_weights(
                    str(fp), adapter_name=adapter_name,
                )
                adapter_names.append(adapter_name)
                logger.info("Loaded LoRA: %s (scale=%.2f)", name, multipliers[i])
            except (KeyError, RuntimeError) as exc:
                logger.warning("Skipping incompatible LoRA %s: %s", name, exc)

        if adapter_names:
            scales = multipliers[: len(adapter_names)]
            self._txt2img_pipe.set_adapters(adapter_names, adapter_weights=scales)
            self._txt2img_pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            logger.info("Fused %d LoRA(s).", len(adapter_names))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _extra_call_kwargs(
        self,
        pipe: Any,
        *,
        negative_prompt: str = "",
        cfg_normalization: bool = True,
        cfg_truncation: float = 0.0,
    ) -> dict:
        """Forward negative_prompt / cfg_* to the underlying ZImage pipeline.

        Z-Image Turbo variants differ in which guidance knobs their
        ``__call__`` exposes, so we signature-filter: anything the pipeline
        does not accept is dropped (no TypeError), and the arg self-activates
        once a variant supports it.
        """
        cand: dict[str, Any] = {}
        if negative_prompt:
            cand["negative_prompt"] = negative_prompt
        cand["cfg_normalization"] = bool(cfg_normalization)
        cand["cfg_truncation"] = float(cfg_truncation)
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError, AttributeError):
            return {}
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return cand
        return {k: v for k, v in cand.items() if k in params}

    def run_txt2img(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = -1,
        negative_prompt: str = "",
        cfg_normalization: bool = True,
        cfg_truncation: float = 0.0,
        batch_size: int = 1,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate images from text prompt using ZImagePipeline.

        Generates one image at a time to conserve VRAM, looping batch_size
        times with incrementing seeds.
        """
        self._ensure_loaded()

        images = []
        for i in range(batch_size):
            current_seed = seed + i if seed >= 0 else seed
            generator = self._make_generator(current_seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback
            kwargs.update(self._extra_call_kwargs(
                self._txt2img_pipe, negative_prompt=negative_prompt,
                cfg_normalization=cfg_normalization, cfg_truncation=cfg_truncation,
            ))

            with torch.inference_mode():
                output = self._txt2img_pipe(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    def run_img2img(
        self,
        image: Any,
        prompt: str,
        strength: float = 0.7,
        width: int = 1024,
        height: int = 1024,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = -1,
        negative_prompt: str = "",
        cfg_normalization: bool = True,
        cfg_truncation: float = 0.0,
        batch_size: int = 1,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate images from an input image + prompt using ZImageImg2ImgPipeline."""
        self._ensure_loaded()

        # Load image from path if string
        if Image is not None and isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Resize input to target dimensions
        if Image is not None and isinstance(image, Image.Image):
            if image.width != width or image.height != height:
                image = image.resize((width, height), Image.LANCZOS)

        images = []
        for i in range(batch_size):
            current_seed = seed + i if seed >= 0 else seed
            generator = self._make_generator(current_seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "image": image,
                "strength": strength,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback
            kwargs.update(self._extra_call_kwargs(
                self._img2img_pipe, negative_prompt=negative_prompt,
                cfg_normalization=cfg_normalization, cfg_truncation=cfg_truncation,
            ))

            with torch.inference_mode():
                output = self._img2img_pipe(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    def run_inpaint(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        strength: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        seed: int = -1,
        negative_prompt: str = "",
        cfg_normalization: bool = True,
        cfg_truncation: float = 0.0,
        batch_size: int = 1,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate inpainted images using ZImageInpaintPipeline."""
        self._ensure_loaded()

        # Load image and mask from paths if strings
        if Image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            if isinstance(mask, str):
                mask = Image.open(mask).convert("L")

        # Ensure mask matches image size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        images = []
        for i in range(batch_size):
            current_seed = seed + i if seed >= 0 else seed
            generator = self._make_generator(current_seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "image": image,
                "mask_image": mask,
                "height": height,
                "width": width,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
            }
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback
            kwargs.update(self._extra_call_kwargs(
                self._inpaint_pipe, negative_prompt=negative_prompt,
                cfg_normalization=cfg_normalization, cfg_truncation=cfg_truncation,
            ))

            with torch.inference_mode():
                output = self._inpaint_pipe(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> str:
        return self._model_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # HuggingFace repo used only for the model config (no weight download)
    _HF_REPO = "Tongyi-MAI/Z-Image-Turbo"

    def _load_quantized_text_encoder(self):
        """Load the Qwen3 text encoder in bf16 on CPU.

        Kept on CPU so that ``enable_model_cpu_offload()`` can shuttle it
        to GPU only during the encoding step, then free VRAM for the
        transformer.  Uses HF cache (local_files_only) — no network
        download is attempted.
        """
        logger.info("Loading Qwen3 text encoder (bf16, CPU)...")
        from transformers import AutoModel

        text_encoder = AutoModel.from_pretrained(
            self._HF_REPO,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        logger.info("Qwen3 text encoder loaded (bf16 on CPU).")
        return text_encoder

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._txt2img_pipe is None:
            raise RuntimeError(
                "Z-Image pipeline not loaded. Load a Z-Image model first."
            )

    def _make_generator(self, seed: int):
        """Create a torch Generator for reproducible results."""
        if seed < 0 or torch is None:
            return None
        device = "cpu"  # CPU generator works with model_cpu_offload
        return torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _cleanup_after_gen() -> None:
        """Free intermediate CUDA tensors after generation."""
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _gc_cleanup() -> None:
        """Full garbage collection and CUDA cache clear."""
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc_module.collect()
