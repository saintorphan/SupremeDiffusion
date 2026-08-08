"""Core FLUX/Chroma pipeline for image generation (GGUF single-file loading).

All shared components (VAE, T5, CLIP-L) are loaded from local single-file
safetensors in the flux_gguf_dir.  Only the transformer comes from GGUF.
"""

from __future__ import annotations

import gc as gc_module
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


class FluxImagePipeline:
    """Wrapper for FLUX/Chroma diffusers pipelines.

    Supports three generation modes:
    - txt2img via ChromaPipeline
    - img2img via ChromaImg2ImgPipeline
    - inpainting (fill) via FluxFillPipeline

    Components loaded from local files in flux_gguf_dir:
    - ae.safetensors (VAE)
    - t5xxl_fp8_e4m3fn.safetensors (T5 text encoder, fp8)
    - clip_l.safetensors (CLIP-L text encoder)
    - chroma/*.gguf or fill/*.gguf (transformer)
    """

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self._chroma_pipe: Any = None
        self._chroma_img2img: Any = None
        self._fill_pipe: Any = None
        self._loaded_chroma: bool = False
        self._loaded_fill: bool = False
        self._chroma_model_path: str = ""
        self._fill_model_path: str = ""

        # Shared components loaded once, reused across pipelines
        self._vae: Any = None
        self._t5_encoder: Any = None
        self._t5_tokenizer: Any = None
        self._clip_encoder: Any = None
        self._clip_tokenizer: Any = None
        self._scheduler: Any = None

    # ------------------------------------------------------------------
    # Shared component loading
    # ------------------------------------------------------------------

    def _get_base_dir(self) -> Path:
        """Return the flux_gguf_dir path."""
        return Path(self.config.model_paths.get("flux_gguf_dir", ""))

    def _load_shared_components(self, need_clip: bool = False) -> None:
        """Load VAE, T5, scheduler, and optionally CLIP from local files.

        Components are cached so they're only loaded once even if both
        Chroma and Fill pipelines are created.
        """
        base = self._get_base_dir()

        if self._vae is None:
            self._load_vae(base / "ae.safetensors")

        if self._t5_encoder is None:
            self._load_t5(base / "t5xxl_fp8_e4m3fn.safetensors")

        if self._scheduler is None:
            self._load_scheduler()

        if need_clip and self._clip_encoder is None:
            self._load_clip(base / "clip_l.safetensors")

    def _load_vae(self, path: Path) -> None:
        """Load FLUX VAE from ae.safetensors.

        Uses the Chroma HF-cache VAE config (16-channel latent space).
        """
        from diffusers import AutoencoderKL

        logger.info("Loading FLUX VAE from %s...", path)
        self._vae = AutoencoderKL.from_single_file(
            str(path),
            config="lodestones/Chroma",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        logger.info("FLUX VAE loaded (%.0f MB).", path.stat().st_size / 1024**2)

    def _load_t5(self, path: Path) -> None:
        """Load T5-XXL text encoder from fp8 safetensors.

        Creates an empty model from the Chroma HF-cache config, then loads
        the fp8 weights from the local single-file safetensors (no network).
        """
        from transformers import AutoConfig, T5EncoderModel, T5Tokenizer
        from safetensors.torch import load_file

        logger.info("Loading T5-XXL fp8 from %s...", path)

        # Config from Chroma HF cache (already downloaded, ~1 KB)
        config = AutoConfig.from_pretrained(
            "lodestones/Chroma", subfolder="text_encoder",
            local_files_only=True,
        )

        # Build empty model on meta device (zero RAM)
        with torch.device("meta"):
            self._t5_encoder = T5EncoderModel(config)

        # Load fp8 weights from local file (~4.6 GB) and upcast to bf16
        # fp8 is a storage format only — compute requires bf16
        state_dict = load_file(str(path))
        state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        self._t5_encoder.load_state_dict(state_dict, assign=True)
        del state_dict
        logger.info("T5-XXL text encoder loaded (fp8→bf16, %.0f MB on disk).", path.stat().st_size / 1024**2)

        # Tokenizer from Chroma HF cache
        self._t5_tokenizer = T5Tokenizer.from_pretrained(
            "lodestones/Chroma", subfolder="tokenizer",
            local_files_only=True,
        )

    def _load_clip(self, path: Path) -> None:
        """Load CLIP-L text encoder from local safetensors.

        Creates an empty model from the HF-cache config, then loads
        weights from the local clip_l.safetensors (fp16, ~235 MB).
        """
        from transformers import AutoConfig, CLIPTextModel, CLIPTokenizer
        from safetensors.torch import load_file

        logger.info("Loading CLIP-L from %s...", path)

        # Config from HF cache (already downloaded)
        full_config = AutoConfig.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=True,
        )

        # Build empty CLIPTextModel on meta device
        with torch.device("meta"):
            self._clip_encoder = CLIPTextModel(full_config.text_config)

        # Load fp16 weights from local file
        state_dict = load_file(str(path))
        self._clip_encoder.load_state_dict(state_dict, assign=True)
        del state_dict
        logger.info("CLIP-L text encoder loaded (%.0f MB).", path.stat().st_size / 1024**2)

        # Tokenizer from HF cache
        self._clip_tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            local_files_only=True,
        )

    def _load_scheduler(self) -> None:
        """Load FlowMatch scheduler from Chroma HF cache."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "lodestones/Chroma", subfolder="scheduler",
        )

    # ------------------------------------------------------------------
    # Pipeline loading
    # ------------------------------------------------------------------

    def load_chroma(self, gguf_path: str) -> None:
        """Load Chroma pipeline from a GGUF transformer file.

        Uses local ae.safetensors for VAE and t5xxl_fp8 for text encoder.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required but not installed.")

        if self._loaded_chroma and self._chroma_model_path == gguf_path:
            return

        if self._loaded_chroma:
            self._unload_chroma()

        logger.info("Loading Chroma transformer from GGUF: %s", gguf_path)

        try:
            from diffusers import (
                ChromaPipeline,
                ChromaImg2ImgPipeline,
                ChromaTransformer2DModel,
                GGUFQuantizationConfig,
            )

            quantization_config = GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16,
            )

            transformer = ChromaTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                config="lodestones/Chroma",
                subfolder="transformer",
            )

            # Load shared components (VAE, T5, scheduler) from local files
            self._load_shared_components(need_clip=False)

            # Assemble pipeline from individual components
            self._chroma_pipe = ChromaPipeline(
                transformer=transformer,
                text_encoder=self._t5_encoder,
                tokenizer=self._t5_tokenizer,
                vae=self._vae,
                scheduler=self._scheduler,
            )
            self._chroma_pipe.enable_model_cpu_offload()
            logger.info("Chroma txt2img pipeline loaded (GGUF).")

            # Build img2img sharing all components
            self._chroma_img2img = ChromaImg2ImgPipeline(
                transformer=transformer,
                text_encoder=self._t5_encoder,
                tokenizer=self._t5_tokenizer,
                vae=self._vae,
                scheduler=self._scheduler,
            )
            self._chroma_img2img.enable_model_cpu_offload()
            logger.info("Chroma img2img pipeline created.")

        except Exception as exc:
            logger.error("Failed to load Chroma from GGUF %s: %s", gguf_path, exc)
            self._unload_chroma()
            raise

        self._chroma_model_path = gguf_path
        self._loaded_chroma = True

    def load_fill(self, gguf_path: str) -> None:
        """Load FluxFillPipeline from a GGUF transformer file.

        Uses local ae.safetensors, t5xxl_fp8, and clip_l.safetensors.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required but not installed.")

        if self._loaded_fill and self._fill_model_path == gguf_path:
            return

        if self._loaded_fill:
            self._unload_fill()

        logger.info("Loading FluxFill transformer from GGUF: %s", gguf_path)

        try:
            from diffusers import (
                FluxFillPipeline,
                FluxTransformer2DModel,
                GGUFQuantizationConfig,
            )

            quantization_config = GGUFQuantizationConfig(
                compute_dtype=torch.bfloat16,
            )

            # Use local config (FLUX.1-Fill-dev is a gated repo)
            fill_config_dir = str(Path(gguf_path).parent)
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                config=fill_config_dir,
                subfolder="transformer",
            )

            # Load shared components including CLIP
            self._load_shared_components(need_clip=True)

            self._fill_pipe = FluxFillPipeline(
                transformer=transformer,
                text_encoder=self._clip_encoder,
                tokenizer=self._clip_tokenizer,
                text_encoder_2=self._t5_encoder,
                tokenizer_2=self._t5_tokenizer,
                vae=self._vae,
                scheduler=self._scheduler,
            )
            self._fill_pipe.enable_model_cpu_offload()
            logger.info("FluxFill pipeline loaded (GGUF).")

        except Exception as exc:
            logger.error("Failed to load FluxFill from GGUF %s: %s", gguf_path, exc)
            self._unload_fill()
            raise

        self._fill_model_path = gguf_path
        self._loaded_fill = True

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def _unload_chroma(self) -> None:
        """Free Chroma pipelines."""
        if self._chroma_img2img is not None:
            del self._chroma_img2img
            self._chroma_img2img = None
        if self._chroma_pipe is not None:
            del self._chroma_pipe
            self._chroma_pipe = None
        self._loaded_chroma = False
        self._chroma_model_path = ""
        self._gc_cleanup()

    def _unload_fill(self) -> None:
        """Free FluxFill pipeline."""
        if self._fill_pipe is not None:
            del self._fill_pipe
            self._fill_pipe = None
        self._loaded_fill = False
        self._fill_model_path = ""
        self._gc_cleanup()

    def unload(self) -> None:
        """Free all pipelines and shared components."""
        self._unload_chroma()
        self._unload_fill()
        # Free shared components
        self._vae = None
        self._t5_encoder = None
        self._t5_tokenizer = None
        self._clip_encoder = None
        self._clip_tokenizer = None
        self._scheduler = None
        self._gc_cleanup()
        logger.info("FLUX pipelines unloaded and CUDA cache cleared.")

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def apply_loras(
        self,
        lora_dir: str,
        lora_names: list[str],
        multipliers: list[float] | None = None,
    ) -> None:
        """Load and fuse LoRA weights into every loaded FLUX pipeline.

        Applies to the Chroma txt2img pipe (shared transformer also covers
        img2img) and, when loaded, the FluxFill pipe — the fill pipe has its
        own transformer, so it must be addressed separately or it would
        otherwise run with no LoRAs.

        Args:
            lora_dir: Directory containing .safetensors LoRA files.
            lora_names: List of filenames to load.
            multipliers: Per-LoRA scale factors (defaults to 1.0 each).
        """
        # chroma_pipe and chroma_img2img share a transformer; fill_pipe has
        # its own, so apply to each distinct pipeline that's loaded.
        targets = [
            (label, pipe)
            for label, pipe in (
                ("Chroma", self._chroma_pipe),
                ("FluxFill", self._fill_pipe),
            )
            if pipe is not None
        ]
        if not targets:
            logger.warning("No FLUX pipeline loaded — cannot apply LoRAs.")
            return

        for label, pipe in targets:
            self._apply_loras_to_pipe(label, pipe, lora_dir, lora_names, multipliers)

    def _apply_loras_to_pipe(
        self,
        label: str,
        pipe: Any,
        lora_dir: str,
        lora_names: list[str],
        multipliers: list[float] | None = None,
    ) -> None:
        """Load LoRAs onto a single diffusers pipeline (unfused)."""
        # Unload any previously loaded LoRAs
        try:
            pipe.unload_lora_weights()
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
                pipe.load_lora_weights(
                    str(fp), adapter_name=adapter_name,
                )
                adapter_names.append(adapter_name)
                logger.info("Loaded LoRA: %s (scale=%.2f)", name, multipliers[i])
            except Exception as exc:
                logger.warning("Skipping incompatible LoRA %s: %s", name, exc)

        if adapter_names:
            scales = multipliers[: len(adapter_names)]
            pipe.set_adapters(adapter_names, adapter_weights=scales)
            # Don't fuse — GGUF quantized weights can't be modified in-place.
            # set_adapters applies LoRA at inference time instead.
            logger.info(
                "Applied %d LoRA(s) to %s pipeline (unfused).",
                len(adapter_names), label,
            )

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
        guidance_scale: float,
        seed: int,
        batch_size: int = 1,
        max_seq_len: int = 512,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate images from text prompt using ChromaPipeline."""
        self._ensure_chroma_loaded()

        images = []
        for i in range(batch_size):
            current_seed = seed + i if seed >= 0 else seed
            generator = self._make_generator(current_seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
                "max_sequence_length": max_seq_len,
            }
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback

            with torch.inference_mode():
                output = self._chroma_pipe(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    def generate_img2img(
        self,
        image: Any,
        prompt: str,
        negative_prompt: str,
        strength: float,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        batch_size: int = 1,
        max_seq_len: int = 512,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate images from an input image + prompt using ChromaImg2ImgPipeline."""
        self._ensure_chroma_loaded()

        if Image is not None and isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if Image is not None and isinstance(image, Image.Image):
            if image.width != width or image.height != height:
                image = image.resize((width, height), Image.LANCZOS)

        images = []
        for i in range(batch_size):
            current_seed = seed + i if seed >= 0 else seed
            generator = self._make_generator(current_seed)

            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "strength": strength,
                "height": height,
                "width": width,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": 1,
                "max_sequence_length": max_seq_len,
            }
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback

            with torch.inference_mode():
                output = self._chroma_img2img(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    def generate_fill(
        self,
        image: Any,
        mask: Any,
        prompt: str,
        strength: float,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str = "",
        batch_size: int = 1,
        max_seq_len: int = 512,
        callback: Optional[Callable] = None,
    ) -> list:
        """Generate inpainted images using FluxFillPipeline."""
        self._ensure_fill_loaded()

        if Image is not None:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            if isinstance(mask, str):
                mask = Image.open(mask).convert("L")

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
                "max_sequence_length": max_seq_len,
            }
            # NOTE: this diffusers FluxFillPipeline.__call__ has no
            # negative_prompt/true_cfg_scale (distilled-guidance only), so the
            # negative prompt is accepted for API parity but cannot be applied.
            if generator is not None:
                kwargs["generator"] = generator
            if callback is not None:
                kwargs["callback_on_step_end"] = callback

            with torch.inference_mode():
                output = self._fill_pipe(**kwargs)
            images.extend(output.images)
            del output
            self._cleanup_after_gen()

        return images

    # ------------------------------------------------------------------
    # Scheduler selection
    # ------------------------------------------------------------------

    # Friendly name → diffusers class name
    SCHEDULERS: dict[str, str] = {
        "Euler": "FlowMatchEulerDiscreteScheduler",
        "Heun": "FlowMatchHeunDiscreteScheduler",
        "DPM++ 2M": "DPMSolverMultistepScheduler",
        "DEIS 2M": "DEISMultistepScheduler",
        "UniPC": "UniPCMultistepScheduler",
    }

    def set_scheduler(self, name: str) -> None:
        """Swap the scheduler on all loaded pipelines.

        *name* must be one of the keys in ``SCHEDULERS``.
        """
        cls_name = self.SCHEDULERS.get(name)
        if cls_name is None:
            logger.warning("Unknown scheduler %r, keeping current.", name)
            return

        import diffusers.schedulers as sched_mod
        cls = getattr(sched_mod, cls_name, None)
        if cls is None:
            logger.warning("Scheduler class %s not found in diffusers.", cls_name)
            return

        new_sched = cls.from_config(self._scheduler.config)
        self._scheduler = new_sched

        for pipe in (self._chroma_pipe, self._chroma_img2img, self._fill_pipe):
            if pipe is not None:
                pipe.scheduler = new_sched

        logger.info("Scheduler set to %s (%s).", name, cls_name)

    @classmethod
    def list_schedulers(cls) -> list[str]:
        """Return list of available scheduler names for UI dropdowns."""
        return list(cls.SCHEDULERS.keys())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_chroma_loaded(self) -> bool:
        return self._loaded_chroma

    @property
    def is_fill_loaded(self) -> bool:
        return self._loaded_fill

    @property
    def chroma_model_path(self) -> str:
        return self._chroma_model_path

    @property
    def fill_model_path(self) -> str:
        return self._fill_model_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_chroma_loaded(self) -> None:
        if not self._loaded_chroma or self._chroma_pipe is None:
            raise RuntimeError(
                "Chroma pipeline not loaded. Load a Chroma model first."
            )

    def _ensure_fill_loaded(self) -> None:
        if not self._loaded_fill or self._fill_pipe is None:
            raise RuntimeError(
                "FluxFill pipeline not loaded. Load a FLUX Fill model first."
            )

    def _make_generator(self, seed: int):
        """Create a torch Generator for reproducible results."""
        if seed < 0 or torch is None:
            return None
        device = "cpu"
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
