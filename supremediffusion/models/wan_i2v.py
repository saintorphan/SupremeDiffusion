"""Wan Image-to-Video inference pipeline wrapper for Supreme Diffusion."""

from __future__ import annotations

import gc
import math
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch is not installed. Inference will be unavailable.")

try:
    from diffusers import WanImageToVideoPipeline
except ImportError:
    WanImageToVideoPipeline = None  # type: ignore[assignment]
    logger.warning(
        "diffusers is not installed or missing WanImageToVideoPipeline. "
        "Inference will be unavailable."
    )

try:
    from mmgp.offload import profile as mmgp_profile  # type: ignore[import-untyped]
except ImportError:
    mmgp_profile = None  # type: ignore[assignment]

# Guider classes for wan2gp custom denoising loop
try:
    from diffusers.guiders import (
        SkipLayerGuidance,
        AdaptiveProjectedGuidance,
        ClassifierFreeZeroStarGuidance,
    )
except ImportError:
    SkipLayerGuidance = None  # type: ignore[assignment,misc]
    AdaptiveProjectedGuidance = None  # type: ignore[assignment,misc]
    ClassifierFreeZeroStarGuidance = None  # type: ignore[assignment,misc]

try:
    from diffusers.hooks import TaylorSeerCacheConfig
except ImportError:
    TaylorSeerCacheConfig = None  # type: ignore[assignment,misc]

from supremediffusion.models import loader


class WanI2VPipeline:
    """High-level wrapper around ``WanImageToVideoPipeline``.

    Supports dual-transformer Lightning models (HIGH noise + LOW noise
    distilled transformers) via diffusers' native ``transformer_2`` /
    ``boundary_ratio`` mechanism.
    """

    def __init__(self, global_config: Any) -> None:
        self.pipe: Any = None
        self.config = global_config
        self._loaded: bool = False
        self._has_dual_transformer: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all model components and assemble the diffusers pipeline."""
        if torch is None:
            raise RuntimeError("PyTorch is required for inference but is not installed.")
        if WanImageToVideoPipeline is None:
            raise RuntimeError(
                "diffusers.WanImageToVideoPipeline is required but is not available."
            )

        # Free any existing pipeline before loading new one
        if self._loaded:
            logger.info("Unloading existing pipeline before reload.")
            self.unload()

        model_paths = self.config.model_paths

        # Force CPU for model loading — mmgp.profile() handles GPU placement later
        prev_device = None
        try:
            prev_device = torch.get_default_device()
        except Exception:
            pass
        torch.set_default_device("cpu")

        # The Settings "Transformer Quantization" dropdown selects the desired
        # load format. Previously precision was derived purely from
        # detect_format() on the filename, so the dropdown was a dead control.
        # Resolve it to a loader format override (None = keep auto-detect).
        quant_format = self._resolve_quant_format(
            getattr(self.config, "transformer_quantization", "")
        )

        def _resolve_fmt(path: str) -> str:
            # On-disk format ALWAYS wins for files that self-declare their
            # quantization (fp8 / gguf / quanto_int8). The Settings quantization
            # dropdown only applies to a generic, unmarked diffusers checkpoint
            # — it must NEVER override a real on-disk format. (Overriding an FP8
            # single-file safetensors with "diffusers" made the loader try to
            # read it as a config.json and crash.)
            try:
                detected = loader.detect_format(path)
            except ValueError:
                detected = None
            if detected and detected != "diffusers":
                return detected
            return quant_format or detected or "diffusers"

        # --- Transformer (HIGH noise / primary) ---
        transformer_path = model_paths.get("transformer", "")
        if not transformer_path:
            raise ValueError("No transformer path configured in global config.")
        logger.info("Loading transformer (HIGH noise) from: %s", transformer_path)
        fmt = _resolve_fmt(transformer_path)
        transformer = loader.load_transformer(transformer_path, format=fmt)

        # --- Transformer 2 (LOW noise — optional, for Lightning) ---
        transformer2_path = model_paths.get("transformer_2", "")
        transformer_2 = None
        if transformer2_path:
            logger.info("Loading transformer_2 (LOW noise) from: %s", transformer2_path)
            fmt2 = _resolve_fmt(transformer2_path)
            transformer_2 = loader.load_transformer(transformer2_path, format=fmt2)
            self._has_dual_transformer = True
            logger.info("Dual-transformer Lightning mode enabled.")
        else:
            self._has_dual_transformer = False

        # --- Text encoder ---
        text_encoder_path = model_paths.get("text_encoder", "")
        if not text_encoder_path:
            raise ValueError("No text_encoder path configured in global config.")
        logger.info("Loading text encoder from: %s", text_encoder_path)
        text_encoder = loader.load_text_encoder(
            text_encoder_path,
            quantization=getattr(self.config, "text_encoder_quantization", "int8"),
        )

        # --- VAE ---
        vae_path = model_paths.get("vae", "")
        if not vae_path:
            raise ValueError("No vae path configured in global config.")
        logger.info("Loading VAE from: %s", vae_path)
        vae = loader.load_vae(
            vae_path,
            precision=getattr(self.config, "vae_precision", "16"),
        )

        # --- CLIP Image Encoder (required for I2V) ---
        # If a local path is configured, load from there.
        # Otherwise let diffusers resolve it automatically (HF cache / download).
        image_encoder_path = model_paths.get("image_encoder", "")
        image_encoder = None
        image_processor = None
        if image_encoder_path:
            logger.info("Loading CLIP image encoder from: %s", image_encoder_path)
            try:
                from transformers import CLIPVisionModel, CLIPImageProcessor
                image_encoder = CLIPVisionModel.from_pretrained(
                    image_encoder_path, torch_dtype=torch.float32,
                )
                image_processor = CLIPImageProcessor.from_pretrained(image_encoder_path)
                logger.info("CLIP image encoder + processor loaded from local path.")
            except Exception as exc:
                logger.warning("Failed to load CLIP from local path: %s — will let diffusers resolve it", exc)
        else:
            logger.info(
                "No image_encoder path configured — diffusers will resolve automatically. "
                "Set a local path in Settings → Video Model Paths to avoid downloads."
            )

        # --- Tokenizer ---
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            logger.info("Tokenizer loaded from %s", text_encoder_path)
        except Exception as exc:
            logger.warning("Could not load tokenizer from %s: %s", text_encoder_path, exc)
            tokenizer = None

        # --- Scheduler ---
        flow_shift = getattr(self.config, "flow_shift", 5.0) or 5.0
        from supremediffusion.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
        scheduler = FlowUniPCMultistepScheduler(shift=float(flow_shift))
        logger.info("Using FlowUniPCMultistepScheduler with shift=%.1f", flow_shift)

        # --- Compute boundary_ratio for dual-transformer ---
        # boundary_ratio = switch_threshold / num_train_timesteps
        # wan2gp default: switch_threshold=900, num_train_timesteps=1000 → 0.9
        boundary_ratio = None
        if self._has_dual_transformer:
            switch_threshold = getattr(self.config, "switch_threshold", 900) or 900
            num_train_timesteps = 1000  # standard for Wan models
            boundary_ratio = switch_threshold / num_train_timesteps
            logger.info(
                "Dual-transformer boundary_ratio=%.3f (switch_threshold=%d/%d)",
                boundary_ratio, switch_threshold, num_train_timesteps,
            )

        # --- Sync image_embedder for dual-transformer setups --------
        #
        # If one transformer has image_embedder (image_dim set in config)
        # and the other doesn't, the pipeline will crash when it passes
        # image_embeds to the transformer without the embedder.
        # Fix: deep-copy the module so both can process image_embeds.
        # Must be a copy (not shared ref) or mmgp sees aliased params
        # and fails its assertion checks during profiling.
        if self._has_dual_transformer:
            import copy

            t1_emb = getattr(
                getattr(transformer, "condition_embedder", None),
                "image_embedder", None,
            )
            t2_emb = getattr(
                getattr(transformer_2, "condition_embedder", None),
                "image_embedder", None,
            )
            if t1_emb is not None and t2_emb is None:
                transformer_2.condition_embedder.image_embedder = copy.deepcopy(t1_emb)
                logger.warning(
                    "transformer_2 missing image_embedder — copied from primary. "
                    "If quality is poor, check model compatibility."
                )
            elif t2_emb is not None and t1_emb is None:
                transformer.condition_embedder.image_embedder = copy.deepcopy(t2_emb)
                logger.warning(
                    "Primary transformer missing image_embedder — copied from transformer_2. "
                    "If quality is poor, check model compatibility."
                )

        # --- Assemble pipeline ---
        pipe_kwargs = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "vae": vae,
            "transformer": transformer,
            "scheduler": scheduler,
        }
        # Only pass CLIP components if explicitly loaded — otherwise
        # let diffusers resolve them automatically from its defaults.
        if image_encoder is not None:
            pipe_kwargs["image_encoder"] = image_encoder
        if image_processor is not None:
            pipe_kwargs["image_processor"] = image_processor
        if self._has_dual_transformer:
            pipe_kwargs["transformer_2"] = transformer_2
            pipe_kwargs["boundary_ratio"] = boundary_ratio

        self.pipe = WanImageToVideoPipeline(**pipe_kwargs)

        # --- Memory management / device placement ---
        memory_profile = getattr(self.config, "memory_profile", None)
        if memory_profile is not None and mmgp_profile is not None:
            try:
                mmgp_profile(self.pipe, profile_no=int(memory_profile), verboseLevel=1)
                logger.info("Applied mmgp memory profile %s.", memory_profile)
            except Exception as exc:
                logger.warning("Failed to apply mmgp profile %s: %s — falling back to .to(cuda)", memory_profile, exc)
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
        elif torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            logger.info("Pipeline moved to CUDA.")

        # Restore default device after profiling
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        elif prev_device is not None:
            torch.set_default_device(prev_device)

        # --- Attention mode ---
        self._apply_attention_mode(
            getattr(self.config, "attention_mode", "sdpa")
        )

        self._loaded = True
        logger.info(
            "WanI2VPipeline fully loaded (dual_transformer=%s).",
            self._has_dual_transformer,
        )

    # ------------------------------------------------------------------
    # Unloading
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Free the pipeline and release GPU memory."""
        if self.pipe is not None:
            # Explicitly delete sub-components to break circular refs.
            # image_encoder/image_processor are included so the CLIP vision
            # tower is moved off the GPU too (previously left resident).
            for attr in (
                "transformer", "transformer_2", "text_encoder", "vae",
                "tokenizer", "image_encoder", "image_processor",
            ):
                component = getattr(self.pipe, attr, None)
                if component is not None:
                    try:
                        component.to("cpu")
                    except Exception:
                        pass
                    try:
                        setattr(self.pipe, attr, None)
                    except Exception:
                        pass
                    del component
            del self.pipe
            self.pipe = None

        self._loaded = False
        self._has_dual_transformer = False

        # GC first to release Python refs, then clear CUDA caches
        gc.collect()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()
        logger.info("WanI2VPipeline unloaded and CUDA cache cleared.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Any,
        prompt: str,
        negative_prompt: str,
        project_config: Any,
        last_frame: Optional[Any] = None,
        guidance_video: Optional[Any] = None,
        callback: Optional[Callable[..., None]] = None,
        lora_step_callback: Optional[Callable[..., Any]] = None,
        overlap_frames: Optional[list] = None,
        overlap_noise: float = 0.0,
    ) -> Any:
        """Run inference via pipe(**kwargs).

        For dual-transformer Lightning models, the pipeline handles phase
        switching natively via boundary_ratio. For single-transformer models
        with Lightning LoRAs, phase switching is handled via callbacks.
        """
        if not self._loaded or self.pipe is None:
            logger.info("Pipeline not loaded yet — auto-loading now.")
            self.load()

        pipe = self.pipe
        cfg = project_config

        # -- Read all config values ----------------------------------------

        num_steps = getattr(cfg, "num_inference_steps", 4)
        guidance_scale = getattr(cfg, "guidance_scale", 1.0)
        guidance2_scale = getattr(cfg, "guidance2_scale", 1.0)
        num_frames = getattr(cfg, "video_length", 81)
        flow_shift = getattr(cfg, "flow_shift", 5.0) or 5.0
        seed = getattr(cfg, "seed", -1)
        guidance_phases = getattr(cfg, "guidance_phases", 1) or 1
        switch_threshold = getattr(cfg, "switch_threshold", 900) or 900

        resolution = getattr(cfg, "resolution", "832x480")
        width, height = 832, 480
        if resolution and "x" in resolution:
            w_s, h_s = resolution.split("x", 1)
            # Strip any trailing text like " (1:1)" from resolution strings
            w_s = w_s.strip().split()[0]
            h_s = h_s.strip().split()[0]
            width, height = int(w_s), int(h_s)

        # Center-crop the source to the target aspect BEFORE the pipeline sees
        # it. Otherwise diffusers' default preprocess STRETCHES a mismatched
        # source to WxH (e.g. a 16:9 frame squished into 832x480). resize_image
        # = scale-to-cover + center-crop, matching LTX's resize_and_center_crop,
        # so Wan crops (zooms) like LTX instead of distorting the aspect.
        try:
            from supremediffusion.core.input_handler import InputHandler
            if image is not None and hasattr(image, "resize"):
                image = InputHandler.resize_image(image, width, height)
            if last_frame is not None and hasattr(last_frame, "resize"):
                last_frame = InputHandler.resize_image(last_frame, width, height)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wan source center-crop failed (%s) — passing through", exc)

        # -- Reconfigure scheduler -----------------------------------------
        # Honor project_config.sample_solver — previously hardcoded UniPC, so
        # the UI dropdown choice was silently ignored. Now dispatched through
        # the unified wan_scheduler module which supports euler/beta + euler_a/beta
        # (Lightning-recommended) plus heun, lcm, sa_solver, etc.
        sample_solver = getattr(project_config, "sample_solver", "unipc") or "unipc"
        from supremediffusion.models.wan_scheduler import (
            build_scheduler,
            LIGHTNING_RECOMMENDED,
        )

        # -- Lightning sampler guard ---------------------------------------
        # Lightning LoRAs + a 4/6-step config require a Lightning-compatible
        # sampler (euler/beta, euler_a/beta, ...). Nothing stops the user
        # switching sample_solver back to unipc/heun while the acceleration
        # preset stays active → badly under-denoised output. Coerce to the
        # recommended sampler with a warning rather than fail silently.
        accel = (getattr(project_config, "acceleration_preset", "off") or "off").lower()
        lightning_active = accel.startswith("lightning")
        if lightning_active and sample_solver.strip().lower() not in LIGHTNING_RECOMMENDED:
            recommended = LIGHTNING_RECOMMENDED[0]
            logger.warning(
                "Lightning preset '%s' active but sampler '%s' is not "
                "Lightning-compatible — coercing to '%s' to avoid "
                "under-denoised output.",
                accel, sample_solver, recommended,
            )
            sample_solver = recommended

        pipe.scheduler = build_scheduler(sample_solver, flow_shift=float(flow_shift))
        logger.info("Scheduler = %s (shift=%.1f)", sample_solver, flow_shift)

        # -- Update boundary_ratio if dual-transformer ---------------------
        # The user may change switch_threshold per-project, so recompute.
        if self._has_dual_transformer:
            new_ratio = switch_threshold / 1000.0
            pipe.config.boundary_ratio = new_ratio
            logger.info("Updated boundary_ratio=%.3f for this generation.", new_ratio)

        logger.info(
            "Generating: steps=%d, guidance=%.2f/%.2f, shift=%.1f, frames=%d, "
            "res=%dx%d, seed=%s, phases=%d, dual_transformer=%s",
            num_steps, guidance_scale, guidance2_scale, flow_shift, num_frames,
            width, height, seed, guidance_phases, self._has_dual_transformer,
        )

        # -- Build pipeline kwargs -----------------------------------------

        kwargs: dict[str, Any] = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }

        if seed >= 0:
            kwargs["generator"] = torch.Generator(
                device=pipe._execution_device
            ).manual_seed(seed)

        if last_frame is not None:
            kwargs["last_image"] = last_frame

        # guidance_scale_2 is used by the pipeline for the low-noise stage
        # when boundary_ratio is set (dual-transformer mode)
        if self._has_dual_transformer:
            kwargs["guidance_scale_2"] = guidance2_scale

        # -- Guidance video: encode source latents (wan2gp-style V2V) ---------
        #
        # VAE-encode the guide video into source_latents, then during
        # denoising re-inject `randn * sigma + (1 - sigma) * source`
        # at each step up to injection_step.
        #
        # Critical: we pre-generate the initial noise and pass it to the
        # pipeline as `latents` so the callback can reuse the EXACT same
        # noise tensor.  wan2gp does `randn = latents` before denoising —
        # using different noise causes mosaic artifacts.

        denoising_strength = getattr(cfg, "denoising_strength", 1.0)

        # When guidance is provided, denoising must be < 1.0 or the
        # entire guidance injection block is skipped.
        if guidance_video is not None:
            if denoising_strength >= 1.0:
                denoising_strength = 0.7
                logger.info("Guidance active: auto-lowered denoising_strength to %.2f", denoising_strength)

        v2v_skip_steps = 0  # how many timesteps to skip (wan2gp-style)

        # Per-step SVI overlap re-injection state (populated by the prefill
        # block below, consumed by _step_callback). Holds the *clean* encoded
        # overlap latents so we can re-anchor the overlap band to the matching
        # noise level after every scheduler step.
        svi_overlap: Optional[dict[str, Any]] = None

        # -- Sliding-window overlap prefill (SVI 2 Pro) --------------------
        #
        # When the orchestrator passes overlap_frames (last N frames of the
        # previous window), we VAE-encode those frames and re-inject them into
        # the FIRST K latent positions at every denoising step, anchored to
        # that step's noise level. The remaining latent positions denoise as
        # pure new content.
        #
        # This is the "right way" — the model's temporal attention bridges
        # overlap and new frames during joint denoising, producing genuinely
        # seamless transitions instead of orchestrator-level cuts. A one-shot
        # prefill at the starting sigma was a near no-op for flow-matching
        # schedulers (first_sigma ≈ 1.0 → overlap weighted ~0 → pure noise);
        # per-step re-injection keeps the band locked across the trajectory.
        #
        # overlap_noise (0.0 → 1.0): how much noise to add to the encoded
        # overlap latents on top of the schedule sigma. 0.0 = clean
        # source-anchored (strict continuity, may force motion); higher values
        # give the model more freedom to deviate from the prior window.
        # Wan2GP defaults to 0.
        if overlap_frames is not None and len(overlap_frames) > 0 and v2v_skip_steps == 0:
            try:
                from PIL import Image as _PILImage

                # Build a (N, 3, H, W) tensor in [0, 1] from PIL images
                ov_arrs = []
                for img in overlap_frames:
                    if isinstance(img, _PILImage.Image):
                        arr = torch.from_numpy(
                            __import__("numpy").asarray(img.convert("RGB"))
                        ).float() / 255.0
                        ov_arrs.append(arr.permute(2, 0, 1))  # (3, H, W)
                    else:
                        # Already a tensor
                        ov_arrs.append(img.float() / 255.0 if img.max() > 1.5 else img.float())
                ov_tensor = torch.stack(ov_arrs, dim=0)  # (N, 3, H, W)

                # Resize to generation resolution if needed
                ov_h, ov_w = ov_tensor.shape[2], ov_tensor.shape[3]
                if ov_h != height or ov_w != width:
                    ov_tensor = torch.nn.functional.interpolate(
                        ov_tensor, size=(height, width),
                        mode="bilinear", align_corners=False,
                    )

                # VAE expects (B, C, N, H, W) in [-1, 1]
                ov_in = ov_tensor.permute(1, 0, 2, 3).unsqueeze(0)
                ov_in = (ov_in * 2.0 - 1.0).to(
                    device=pipe._execution_device, dtype=pipe.vae.dtype,
                )
                overlap_latents = pipe.vae.encode(ov_in).latent_dist.mode()
                # Normalize to diffusers latent space
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(overlap_latents.device, overlap_latents.dtype)
                )
                latents_std = (
                    1.0 / torch.tensor(pipe.vae.config.latents_std)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(overlap_latents.device, overlap_latents.dtype)
                )
                overlap_latents = (overlap_latents - latents_mean) * latents_std

                # Defensive num_frames alignment — mirror the %temporal==1
                # rounding the pipeline (and _wan2gp_call) apply, so the noise
                # tensor we pre-generate here matches the latent count the
                # pipeline will actually expect. Without this, a window size that
                # isn't already %4==1-aligned yields a noise_shape with the wrong
                # temporal extent and the kwargs["latents"] injection mismatches.
                temporal = pipe.vae_scale_factor_temporal
                if num_frames % temporal != 1:
                    num_frames = num_frames // temporal * temporal + 1
                num_frames = max(num_frames, 1)

                # Pre-generate the full noise tensor for this window
                lat_frames = (num_frames - 1) // temporal + 1
                latent_h = height // pipe.vae_scale_factor_spatial
                latent_w = width // pipe.vae_scale_factor_spatial
                noise_shape = (1, pipe.vae.config.z_dim, lat_frames, latent_h, latent_w)
                if seed >= 0:
                    noise_gen = torch.Generator(device=pipe._execution_device).manual_seed(seed)
                    initial_noise = torch.randn(
                        noise_shape, generator=noise_gen,
                        device=pipe._execution_device, dtype=torch.float32,
                    )
                else:
                    initial_noise = torch.randn(
                        noise_shape, device=pipe._execution_device, dtype=torch.float32,
                    )

                # The scheduler's first sigma — at the START of denoising the
                # whole latent volume sits at this noise level. We seed the
                # overlap band so step 0 is well-formed; the per-step callback
                # below then re-anchors it to each subsequent step's sigma.
                # Restore the scheduler's prior timesteps/sigmas afterwards so
                # this read-only probe doesn't mutate shared scheduler state
                # before the pipeline runs its own set_timesteps() setup.
                _prev_timesteps = getattr(pipe.scheduler, "timesteps", None)
                _prev_sigmas = getattr(pipe.scheduler, "sigmas", None)
                pipe.scheduler.set_timesteps(num_steps, device=pipe._execution_device)
                first_sigma = pipe.scheduler.timesteps[0].item() / 1000.0
                if _prev_timesteps is not None:
                    pipe.scheduler.timesteps = _prev_timesteps
                if _prev_sigmas is not None:
                    pipe.scheduler.sigmas = _prev_sigmas
                # User parameter is in [0, 1]; treat it as extra freedom added
                # on top of the schedule sigma. 0 = strict continuity.
                sigma_eff = max(float(overlap_noise), 0.0)
                if sigma_eff > 1.0:
                    sigma_eff = 1.0

                k = min(overlap_latents.shape[2], initial_noise.shape[2])
                overlap_clean = overlap_latents[:, :, :k].to(initial_noise.dtype)

                # Seed the initial latents: overlap noised to the first sigma
                # (so the first model forward sees a well-formed trajectory).
                seed_sigma = min(max(sigma_eff, first_sigma), 1.0)
                initial_noise[:, :, :k] = (
                    overlap_clean * (1.0 - seed_sigma)
                    + initial_noise[:, :, :k] * seed_sigma
                )
                kwargs["latents"] = initial_noise
                # Keep the pipeline's frame count in sync with the (possibly
                # rounded) value used to size the noise tensor above.
                kwargs["num_frames"] = num_frames

                # Stash clean overlap for per-step re-injection. The callback
                # re-anchors latents[:, :, :k] to overlap_clean at each step's
                # sigma so the band tracks the denoising trajectory as known
                # temporal context instead of collapsing into pure noise.
                svi_overlap = {
                    "clean": overlap_clean,
                    "k": k,
                    "extra_noise": sigma_eff,
                    "noise": initial_noise[:, :, :k].clone(),
                }

                logger.info(
                    "SVI overlap prefill: %d/%d latent frames anchored "
                    "(overlap_noise=%.2f, seed_sigma=%.3f, per-step re-injection, "
                    "%d video frames)",
                    k, lat_frames, sigma_eff, seed_sigma, len(overlap_frames),
                )

                # Defuse the V2V branch below — we've already populated
                # kwargs["latents"] and we want full denoising (no step skip).
                guidance_video = None

                del overlap_latents

            except Exception as exc:
                logger.warning(
                    "SVI overlap prefill failed: %s — falling back to plain I2V "
                    "(window seam may be visible)", exc, exc_info=True,
                )

        if guidance_video is not None and denoising_strength < 1.0:
            try:
                # guidance_video is a tensor (N, 3, H, W) in [0, 1]
                # Resize to generation resolution if dimensions don't match
                gv_h, gv_w = guidance_video.shape[2], guidance_video.shape[3]
                if gv_h != height or gv_w != width:
                    logger.info(
                        "Resizing guidance video from %dx%d to %dx%d to match generation resolution",
                        gv_w, gv_h, width, height,
                    )
                    guidance_video = torch.nn.functional.interpolate(
                        guidance_video, size=(height, width), mode="bilinear", align_corners=False,
                    )
                # VAE expects (B, C, N, H, W) in [-1, 1]
                gv = guidance_video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, N, H, W)
                gv = (gv * 2.0 - 1.0).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
                source_latents = pipe.vae.encode(gv).latent_dist.mode()
                # Normalize to match diffusers' latent space
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(source_latents.device, source_latents.dtype)
                )
                latents_std = (
                    1.0 / torch.tensor(pipe.vae.config.latents_std)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(source_latents.device, source_latents.dtype)
                )
                source_latents = (source_latents - latents_mean) * latents_std

                lat_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

                # Stretch guidance latents temporally if shorter than generation
                if source_latents.shape[2] < lat_frames:
                    logger.info(
                        "Stretching guidance latents from %d to %d temporal frames",
                        source_latents.shape[2], lat_frames,
                    )
                    source_latents = torch.nn.functional.interpolate(
                        source_latents,
                        size=(lat_frames, source_latents.shape[3], source_latents.shape[4]),
                        mode="trilinear",
                        align_corners=False,
                    )

                # wan2gp approach for full-length source: skip early timesteps
                # and start from noised source.  No per-step injection.
                #
                # 1. Compute how many timesteps to skip
                # 2. Pre-generate noise
                # 3. Set initial latents = noise * sigma_start + (1-sigma_start) * source
                # 4. Patch scheduler to skip those timesteps
                # 5. Denoiser runs normally from there — no callback injection
                v2v_skip_steps = int(round(num_steps * (1.0 - denoising_strength)))

                # Pre-generate noise
                latent_h = height // pipe.vae_scale_factor_spatial
                latent_w = width // pipe.vae_scale_factor_spatial
                noise_shape = (1, pipe.vae.config.z_dim, lat_frames, latent_h, latent_w)
                if seed >= 0:
                    noise_gen = torch.Generator(device=pipe._execution_device).manual_seed(seed)
                    initial_noise = torch.randn(
                        noise_shape, generator=noise_gen,
                        device=pipe._execution_device, dtype=torch.float32,
                    )
                else:
                    initial_noise = torch.randn(
                        noise_shape, device=pipe._execution_device, dtype=torch.float32,
                    )

                # Compute the starting sigma after skipping steps.
                # We need to query the scheduler for the actual timestep values.
                pipe.scheduler.set_timesteps(num_steps, device=pipe._execution_device)
                full_timesteps = pipe.scheduler.timesteps
                if v2v_skip_steps > 0 and v2v_skip_steps < len(full_timesteps):
                    start_sigma = full_timesteps[v2v_skip_steps].item() / 1000.0
                else:
                    start_sigma = full_timesteps[0].item() / 1000.0

                # Blend: noised source at the starting noise level
                n_guide = min(source_latents.shape[2], initial_noise.shape[2])
                initial_noise[:, :, :n_guide] = (
                    initial_noise[:, :, :n_guide] * start_sigma
                    + (1.0 - start_sigma) * source_latents[:, :, :n_guide]
                )
                kwargs["latents"] = initial_noise

                logger.info(
                    "V2V guidance: skip %d/%d steps, start_sigma=%.3f, "
                    "denoising=%.2f, %d/%d latent frames",
                    v2v_skip_steps, num_steps, start_sigma,
                    denoising_strength, source_latents.shape[2], lat_frames,
                )

                # Clean up source_latents — no longer needed after initial blend
                del source_latents
                source_latents = None

            except Exception as exc:
                logger.warning("Failed to encode guidance video: %s — ignoring", exc)
                v2v_skip_steps = 0

        # Patch scheduler to skip early timesteps for V2V.
        # The pipeline calls set_timesteps() internally, so we monkey-patch
        # to truncate the schedule after it's (re)computed.
        if v2v_skip_steps > 0:
            _orig_set_timesteps = pipe.scheduler.set_timesteps
            _skip = v2v_skip_steps

            def _patched_set_timesteps(n, **kw):
                _orig_set_timesteps(n, **kw)
                pipe.scheduler.timesteps = pipe.scheduler.timesteps[_skip:]
                if hasattr(pipe.scheduler, "sigmas") and pipe.scheduler.sigmas is not None:
                    pipe.scheduler.sigmas = pipe.scheduler.sigmas[_skip:]

            pipe.scheduler.set_timesteps = _patched_set_timesteps

        # -- Build combined step callback ----------------------------------

        has_dual = self._has_dual_transformer
        phase_state = {
            "label": "Phase 1 (High Noise)" if (guidance_phases >= 2 or has_dual) else "",
            "switched": False,
        }

        def _step_callback(pipe_ref, step_index, timestep, cb_kwargs):
            latents = cb_kwargs.get("latents")

            # SVI overlap re-injection: re-anchor the overlap band to the
            # *next* step's noise level so it stays locked to the previous
            # window's content as known temporal context. Without this the
            # overlap region drifts and the inter-window seam reappears.
            if svi_overlap is not None and latents is not None:
                try:
                    sched = pipe_ref.scheduler
                    next_idx = step_index + 1
                    if (
                        getattr(sched, "timesteps", None) is not None
                        and next_idx < len(sched.timesteps)
                    ):
                        next_sigma = float(sched.timesteps[next_idx].item()) / 1000.0
                    else:
                        next_sigma = 0.0  # final step → fully clean
                    next_sigma = min(max(next_sigma, svi_overlap["extra_noise"]), 1.0)
                    k = svi_overlap["k"]
                    clean = svi_overlap["clean"].to(latents.dtype)
                    noise = svi_overlap["noise"].to(latents.dtype)
                    latents[:, :, :k] = (
                        clean * (1.0 - next_sigma) + noise * next_sigma
                    )
                    cb_kwargs["latents"] = latents
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SVI overlap re-injection skipped at step %d: %s",
                                   step_index, exc)

            # Phase label switching (actual model switching is handled by the
            # pipeline's boundary_ratio for dual-transformer, or by LoRA
            # callbacks for single-transformer)
            if not phase_state["switched"] and timestep <= switch_threshold:
                phase_state["switched"] = True
                phase_state["label"] = "Phase 2 (Low Noise)"
                logger.info(
                    "Phase switch at step %d (t=%s)%s",
                    step_index, timestep,
                    " — switched to transformer_2" if has_dual else "",
                )

            # LoRA phase switching (for single-transformer + Lightning LoRAs)
            if lora_step_callback is not None:
                lora_step_callback(pipe_ref, step_index, timestep, cb_kwargs)

            # Pass phase label for progress bar
            cb_kwargs["phase_label"] = phase_state["label"]

            # User callback (progress bar)
            if callback is not None:
                result = callback(pipe_ref, step_index, timestep, cb_kwargs)
                if result is not None:
                    cb_kwargs.update(result)

            return cb_kwargs

        kwargs["callback_on_step_end"] = _step_callback
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        # -- TEA cache (works with both standard and wan2gp loops) ---------

        denoising_loop = getattr(self.config, "denoising_loop", "standard")
        tea_setting = getattr(cfg, "tea_cache_setting", "off")
        tea_enabled = tea_setting != "off"
        if tea_enabled:
            self._enable_tea_cache(pipe, tea_setting, num_steps, cfg)

        # -- Run pipeline --------------------------------------------------

        try:
            use_wan2gp = (
                denoising_loop == "wan2gp"
                and bool(getattr(cfg, "quality_overrides_enabled", False))
            )
            if use_wan2gp:
                logger.info("Using wan2gp custom denoising loop.")
                output = self._wan2gp_call(pipe, kwargs, cfg, _step_callback)
            else:
                logger.info("Calling pipe(**kwargs) — denoising + VAE decode...")
                output = pipe(**kwargs)
                logger.info("pipe() returned, type=%s", type(output).__name__)
        finally:
            # Disable TEA cache if it was enabled
            if tea_enabled:
                self._disable_tea_cache(pipe)
            # Restore scheduler if patched
            if v2v_skip_steps > 0 and hasattr(pipe.scheduler, 'set_timesteps'):
                try:
                    pipe.scheduler.set_timesteps = _orig_set_timesteps
                except NameError:
                    pass
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Extract frames
        if hasattr(output, "frames"):
            frames = output.frames
            if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], list):
                frames = frames[0]
            return frames
        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the pipeline components are currently loaded."""
        return self._loaded

    @property
    def has_dual_transformer(self) -> bool:
        """Whether the pipeline has two transformers (Lightning mode)."""
        return self._has_dual_transformer

    # ------------------------------------------------------------------
    # TEA Cache
    # ------------------------------------------------------------------

    # Map UI dropdown → TaylorSeerCacheConfig parameters
    _TEA_PRESETS: dict[str, dict[str, int]] = {
        "light":  {"cache_interval": 9, "max_order": 1},
        "medium": {"cache_interval": 5, "max_order": 1},
        "heavy":  {"cache_interval": 3, "max_order": 2},
    }

    def _enable_tea_cache(
        self, pipe: Any, preset: str, num_steps: int, cfg: Any,
    ) -> None:
        """Enable TaylorSeer cache on the transformer(s)."""
        if TaylorSeerCacheConfig is None:
            logger.warning("TaylorSeerCacheConfig not available — skipping TEA cache.")
            return

        params = self._TEA_PRESETS.get(preset)
        if params is None:
            logger.warning("Unknown TEA cache preset '%s' — skipping.", preset)
            return

        start_perc = getattr(cfg, "tea_cache_start_step_perc", 0.0)
        disable_before = max(int(num_steps * start_perc / 100.0), 1)

        tea_cfg = TaylorSeerCacheConfig(
            cache_interval=params["cache_interval"],
            max_order=params["max_order"],
            disable_cache_before_step=disable_before,
        )
        logger.info("Enabling TEA cache: %s (disable_before_step=%d)", tea_cfg, disable_before)

        for name in ("transformer", "transformer_2"):
            t = getattr(pipe, name, None)
            if t is not None and hasattr(t, "enable_cache"):
                t.enable_cache(tea_cfg)

    @staticmethod
    def _disable_tea_cache(pipe: Any) -> None:
        """Disable TaylorSeer cache on the transformer(s)."""
        for name in ("transformer", "transformer_2"):
            t = getattr(pipe, name, None)
            if t is not None and hasattr(t, "disable_cache"):
                try:
                    t.disable_cache()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Guider construction (wan2gp mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _phase_cfg(cfg: Any, name: str, phase: int, default: Any) -> Any:
        """Read a guider config field, honoring an optional per-phase override.

        Phase 2 (low-noise) reads ``<name>_2`` when present, falling back to the
        phase-1 field so behavior is unchanged unless the override is set. This
        lets a run enable e.g. SLG only on the high-noise phase (``slg_switch=1``,
        ``slg_switch_2=0``) without affecting existing single-phase configs.
        """
        base = getattr(cfg, name, default)
        if phase >= 2:
            return getattr(cfg, name + "_2", base)
        return base

    @staticmethod
    def _phase_guider_differs(cfg: Any) -> bool:
        """Whether any guider switch has a phase-2 override that differs from phase 1."""
        for name, default in (
            ("slg_switch", 0),
            ("apg_switch", 0),
            ("cfg_zero_step", 0),
            ("cfg_star_switch", 0),
            ("slg_layers", None),
            ("slg_start_perc", 0),
            ("slg_end_perc", 100),
        ):
            base = getattr(cfg, name, default)
            if getattr(cfg, name + "_2", base) != base:
                return True
        return False

    @staticmethod
    def _build_guider(cfg: Any, guidance_scale: float, phase: int = 1) -> Any:
        """Construct a diffusers guider from project config, or return None for standard CFG.

        ``phase`` selects which set of SLG/APG/CFG switches to honor: phase 1
        (high-noise) reads the plain fields, phase 2 (low-noise) reads the
        ``_2``-suffixed overrides when present so the low-noise phase can run
        independent skip-layer / CFG settings.
        """
        _pc = WanI2VPipeline._phase_cfg
        slg_on = bool(_pc(cfg, "slg_switch", phase, 0)) and guidance_scale > 1.0
        apg_on = bool(_pc(cfg, "apg_switch", phase, 0)) and guidance_scale > 1.0
        cfg_zero_step = _pc(cfg, "cfg_zero_step", phase, 0)
        cfg_star = _pc(cfg, "cfg_star_switch", phase, 0)

        if slg_on and SkipLayerGuidance is not None:
            layers = _pc(cfg, "slg_layers", phase, None)
            if not layers:
                layers = list(range(7, 20))  # wan2gp default: 7-19
            return SkipLayerGuidance(
                guidance_scale=guidance_scale,
                skip_layer_guidance_scale=2.8,
                skip_layer_guidance_start=_pc(cfg, "slg_start_perc", phase, 0) / 100.0,
                skip_layer_guidance_stop=_pc(cfg, "slg_end_perc", phase, 100) / 100.0,
                skip_layer_guidance_layers=layers,
            )
        elif apg_on and AdaptiveProjectedGuidance is not None:
            return AdaptiveProjectedGuidance(
                guidance_scale=guidance_scale,
            )
        elif (cfg_star > 0 or cfg_zero_step > 0) and guidance_scale > 1.0:
            if ClassifierFreeZeroStarGuidance is not None:
                zero_steps = max(cfg_zero_step, 0)
                return ClassifierFreeZeroStarGuidance(
                    guidance_scale=guidance_scale,
                    zero_init_steps=zero_steps,
                )

        return None  # standard CFG

    # ------------------------------------------------------------------
    # Custom denoising loop (wan2gp mode)
    # ------------------------------------------------------------------

    def _wan2gp_call(
        self,
        pipe: Any,
        kwargs: dict[str, Any],
        cfg: Any,
        step_callback: Optional[Callable] = None,
    ) -> Any:
        """Run generation with a custom denoising loop supporting SLG/APG/CFG Zero*.

        This is a forked version of ``WanImageToVideoPipeline.__call__()`` that
        replaces only the denoising loop (lines 741-810 of the diffusers source).
        All setup (encoding, latent preparation) and teardown (VAE decoding) are
        delegated to the pipeline's own methods.
        """
        # --- Resolve all parameters from kwargs ---------------------------
        image = kwargs["image"]
        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        num_steps = kwargs.get("num_inference_steps", 4)
        guidance_scale = kwargs.get("guidance_scale", 1.0)
        guidance_scale_2 = kwargs.get("guidance_scale_2")
        num_frames = kwargs.get("num_frames", 81)
        height = kwargs.get("height", 480)
        width = kwargs.get("width", 832)
        generator = kwargs.get("generator")
        latents = kwargs.get("latents")
        last_image = kwargs.get("last_image")

        device = pipe._execution_device
        transformer_dtype = (
            pipe.transformer.dtype if pipe.transformer is not None else pipe.transformer_2.dtype
        )

        # -- Motion amplitude scaling (applied to initial noise) -----------
        motion_amp = getattr(cfg, "motion_amplitude", 1.0)

        # -- Build guider(s) -----------------------------------------------
        # For dual-transformer / dual-phase runs the high-noise and low-noise
        # phases can differ in guidance scale AND in SLG/APG/CFG settings. We
        # build a phase-1 (high-noise) guider and an independent phase-2
        # (low-noise) guider — the latter reads the low-noise scale plus any
        # `_2`-suffixed switch overrides — and switch between them at the
        # boundary. Phase 2 reuses the phase-1 guider only when nothing differs.
        guider = self._build_guider(cfg, guidance_scale, phase=1)
        gs2 = guidance_scale_2 if guidance_scale_2 is not None else guidance_scale
        guider_2 = self._build_guider(cfg, gs2, phase=2)
        # Collapse to a single guider when phase 2 is identical so we don't pay
        # for a redundant object / set_state pair every step.
        if (
            type(guider_2) is type(guider)
            and gs2 == guidance_scale
            and not self._phase_guider_differs(cfg)
        ):
            guider_2 = guider
        if guider is not None or guider_2 is not None:
            g1_name = type(guider).__name__ if guider is not None else "CFG"
            g2_name = type(guider_2).__name__ if guider_2 is not None else "CFG"
            if guider_2 is not guider:
                logger.info(
                    "wan2gp guiders: high=%s(%.2f) / low=%s(%.2f)",
                    g1_name, guidance_scale, g2_name, gs2,
                )
            else:
                logger.info("wan2gp guider: %s", g1_name)
        else:
            logger.info("wan2gp mode: no guider active, using standard CFG formula.")

        do_cfg = guidance_scale > 1.0

        # --- Run standard pipeline setup (encoding, latent prep) ----------
        # We use the pipeline's own methods for encoding and preparation,
        # then take over the denoising loop.

        # num_frames rounding
        if num_frames % pipe.vae_scale_factor_temporal != 1:
            num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        # Resolution alignment
        patch_size = (
            pipe.transformer.config.patch_size
            if pipe.transformer is not None
            else pipe.transformer_2.config.patch_size
        )
        h_mult = pipe.vae_scale_factor_spatial * patch_size[1]
        w_mult = pipe.vae_scale_factor_spatial * patch_size[2]
        height = height // h_mult * h_mult
        width = width // w_mult * w_mult

        if pipe.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        pipe._guidance_scale = guidance_scale
        pipe._guidance_scale_2 = guidance_scale_2
        pipe._attention_kwargs = None
        pipe._current_timestep = None
        pipe._interrupt = False

        batch_size = 1

        # Encode prompt
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Encode image embedding (requires CLIP image_encoder)
        image_embeds = None
        if (
            pipe.transformer is not None
            and pipe.transformer.config.image_dim is not None
            and getattr(pipe, "image_encoder", None) is not None
        ):
            if last_image is None:
                image_embeds = pipe.encode_image(image, device)
            else:
                image_embeds = pipe.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1).to(transformer_dtype)

        # Prepare timesteps
        pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = pipe.scheduler.timesteps

        # Prepare latents
        num_channels_latents = pipe.vae.config.z_dim
        image_proc = pipe.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float32
        )
        last_image_proc = None
        if last_image is not None:
            last_image_proc = pipe.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = pipe.prepare_latents(
            image_proc,
            batch_size,
            num_channels_latents,
            height, width, num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image_proc,
        )
        if pipe.config.expand_timesteps:
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs

        # Apply motion amplitude to initial latents
        if motion_amp != 1.0:
            logger.info("Applying motion amplitude scaling: %.2f", motion_amp)
            latents = latents * motion_amp

        # --- Custom denoising loop ----------------------------------------
        num_warmup_steps = len(timesteps) - num_steps * pipe.scheduler.order
        pipe._num_timesteps = len(timesteps)

        boundary_timestep = None
        if pipe.config.boundary_ratio is not None:
            boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps

        with pipe.progress_bar(total=num_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    # Stop denoising on interrupt — `continue` here would busy-spin
                    # through the remaining timesteps doing no work.
                    break

                pipe._current_timestep = t

                # Select transformer + guider for this step (dual-phase)
                if boundary_timestep is None or t >= boundary_timestep:
                    current_model = pipe.transformer
                    current_guidance_scale = guidance_scale
                    current_guider = guider
                else:
                    current_model = pipe.transformer_2
                    current_guidance_scale = guidance_scale_2
                    current_guider = guider_2

                # Prepare latent input
                if pipe.config.expand_timesteps:
                    latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
                    timestep = t.expand(latents.shape[0])

                # Set guider state for this step
                if current_guider is not None:
                    current_guider.set_state(step=i, num_inference_steps=num_steps, timestep=t)

                # Guard: only pass image_embeds to models with image_embedder
                current_image_embeds = (
                    image_embeds
                    if getattr(current_model.config, "image_dim", None) is not None
                    else None
                )

                # -- Forward pass 1: conditional ---
                with current_model.cache_context("cond"):
                    if current_guider is not None:
                        current_guider.prepare_models(current_model)
                    noise_pred_cond = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=current_image_embeds,
                        return_dict=False,
                    )[0]
                    if current_guider is not None:
                        current_guider.cleanup_models(current_model)

                # -- Forward pass 2: unconditional (if CFG active) ---
                noise_pred_uncond = None
                if do_cfg:
                    with current_model.cache_context("uncond"):
                        if current_guider is not None:
                            current_guider.prepare_models(current_model)
                        noise_pred_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=current_image_embeds,
                            return_dict=False,
                        )[0]
                        if current_guider is not None:
                            current_guider.cleanup_models(current_model)

                # -- Forward pass 3: SLG skip pass (conditional with skipped layers) ---
                noise_pred_skip = None
                if (
                    current_guider is not None
                    and SkipLayerGuidance is not None
                    and isinstance(current_guider, SkipLayerGuidance)
                    and current_guider._is_slg_enabled()
                ):
                    # Third pass: conditional prompt but with layer-skip hooks
                    current_guider.prepare_models(current_model)  # applies skip hooks
                    noise_pred_skip = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=current_image_embeds,
                        return_dict=False,
                    )[0]
                    current_guider.cleanup_models(current_model)  # removes skip hooks

                # -- Combine predictions ---
                if current_guider is not None:
                    if (
                        SkipLayerGuidance is not None
                        and isinstance(current_guider, SkipLayerGuidance)
                    ):
                        result = current_guider.forward(
                            noise_pred_cond, noise_pred_uncond, noise_pred_skip,
                        )
                    else:
                        result = current_guider.forward(noise_pred_cond, noise_pred_uncond)
                    noise_pred = result.pred
                elif do_cfg:
                    # Standard CFG formula (fallback when no guider)
                    noise_pred = noise_pred_uncond + current_guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = noise_pred_cond

                # -- Scheduler step ---
                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # -- Callback ---
                if step_callback is not None:
                    cb_kwargs = {"latents": latents}
                    cb_out = step_callback(pipe, i, t, cb_kwargs)
                    latents = cb_out.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0
                ):
                    progress_bar.update()

        pipe._current_timestep = None

        # --- Post-denoising (same as standard pipeline) ---
        if pipe.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        latents = latents.to(pipe.vae.dtype)
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0 / torch.tensor(pipe.vae.config.latents_std)
            .view(1, pipe.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type="pil")

        pipe.maybe_free_model_hooks()

        from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput
        return WanPipelineOutput(frames=video)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_quant_format(quantization: str) -> Optional[str]:
        """Map the Settings transformer-quantization choice to a loader format.

        The dropdown offers ``int8``/``fp8``/``bf16``/``none``; ``loader.load_transformer``
        accepts a ``format`` of ``quanto_int8``/``gguf``/``fp8``/``diffusers``.
        Returning ``None`` means "no override" so the filename-based
        ``detect_format()`` auto-detection still applies (e.g. for GGUF files).
        """
        key = (quantization or "").strip().lower()
        mapping = {
            "int8": "quanto_int8",
            "fp8": "fp8",
            "bf16": "diffusers",
            "none": "diffusers",
        }
        return mapping.get(key)

    def _apply_attention_mode(self, mode: str) -> None:
        """Configure the attention processor on the pipeline's transformer(s).

        The Settings dropdown exposes the upstream attention-mode names
        (sdpa/auto/flash/xformers/sage/sage2/sage3/radial). This method maps
        each of them onto something the Wan transformers can actually use:
        xformers + the sage family wire real processors when their package is
        importable, everything else (and any unavailable kernel) falls back to
        SDPA with an accurate log line — never a silent mislabel.
        """
        mode = (mode or "sdpa").lower()
        # Tolerate legacy/aliased names so a stale config never hits the
        # "unknown mode" path.
        aliases = {
            "sage_attn": "sage",
            "flash_attn": "flash",
            "sageattention": "sage",
        }
        mode = aliases.get(mode, mode)

        # "auto": pick the best implemented backend that's actually installed.
        if mode == "auto":
            if self._sage_available():
                mode = "sage"
            else:
                mode = "sdpa"
            logger.info("Attention mode 'auto' resolved to '%s'.", mode)

        if mode == "xformers":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory-efficient attention.")
            except Exception as exc:
                logger.warning(
                    "Could not enable xformers attention: %s. Falling back to sdpa.", exc
                )
                self._apply_attention_mode("sdpa")

        elif mode == "sdpa":
            try:
                from diffusers.models.transformers.transformer_wan import WanAttnProcessor
                for name in ("transformer", "transformer_2"):
                    t = getattr(self.pipe, name, None)
                    if t is not None:
                        t.set_attn_processor(WanAttnProcessor())
                logger.info("Using SDPA (PyTorch 2.0) attention via WanAttnProcessor.")
            except ImportError:
                # Older diffusers without WanAttnProcessor — leave the default
                logger.info("Using default Wan attention processor (SDPA).")
            except Exception as exc:
                logger.warning("Could not set SDPA attention processor: %s", exc)

        elif mode in ("sage", "sage2", "sage3"):
            # All sage variants share one importable kernel (`sageattn`); the
            # 2/3 suffixes select the kernel generation at runtime inside the
            # package. Wire a real processor instead of just importing it.
            proc = self._make_sage_processor(mode)
            if proc is None:
                logger.warning(
                    "sageattention (%s) unavailable — falling back to sdpa.", mode
                )
                self._apply_attention_mode("sdpa")
            else:
                applied = False
                for name in ("transformer", "transformer_2"):
                    t = getattr(self.pipe, name, None)
                    if t is not None:
                        t.set_attn_processor(proc)
                        applied = True
                if applied:
                    logger.info("SageAttention (%s) processor installed.", mode)
                else:
                    logger.warning(
                        "No transformer to attach SageAttention to — falling back to sdpa."
                    )
                    self._apply_attention_mode("sdpa")

        elif mode == "flash":
            # FlashAttention on Wan is exposed through PyTorch SDPA's flash
            # backend; there is no separate diffusers processor to install.
            # If flash_attn isn't even importable, SDPA still routes to its
            # built-in flash kernel where the hardware supports it.
            try:
                import flash_attn  # type: ignore[import-untyped]  # noqa: F401
                logger.info(
                    "FlashAttention available — routing through SDPA flash backend."
                )
            except ImportError:
                logger.info(
                    "flash_attn not installed — using SDPA (flash backend used "
                    "automatically when supported)."
                )
            self._apply_attention_mode("sdpa")

        elif mode == "radial":
            # Radial attention is implemented for the LTX backend only; there
            # is no Wan processor for it. Be explicit rather than silent.
            logger.warning(
                "Radial attention is not implemented for the Wan backend — "
                "using sdpa."
            )
            self._apply_attention_mode("sdpa")

        else:
            logger.warning("Unknown attention mode '%s'; defaulting to sdpa.", mode)
            self._apply_attention_mode("sdpa")

    @staticmethod
    def _sage_available() -> bool:
        """Whether the sageattention kernel can be imported."""
        try:
            import sageattention  # type: ignore[import-untyped]  # noqa: F401
            return True
        except Exception:
            return False

    def _make_sage_processor(self, variant: str) -> Any:
        """Build a Wan attention processor backed by sageattn, or None.

        Subclasses the diffusers ``WanAttnProcessor`` and replaces the scaled
        dot-product call with sageattn so the transformer actually runs the
        Sage kernel instead of falling back to a no-op import.
        """
        try:
            from sageattention import sageattn  # type: ignore[import-untyped]
        except Exception:
            return None
        try:
            from diffusers.models.transformers.transformer_wan import WanAttnProcessor
        except Exception:
            return None

        # Monkey-patch torch.scaled_dot_product_attention only inside the
        # processor call so we don't disturb the rest of the pipeline.
        _sage = sageattn

        class _SageWanAttnProcessor(WanAttnProcessor):  # type: ignore[misc, valid-type]
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                if torch is None:
                    return super().__call__(*args, **kwargs)
                orig_sdpa = torch.nn.functional.scaled_dot_product_attention

                def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                          is_causal=False, scale=None, **_kw):
                    if attn_mask is not None or is_causal:
                        # sageattn doesn't support masks/causal here — defer.
                        return orig_sdpa(
                            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale,
                        )
                    return _sage(q, k, v)

                torch.nn.functional.scaled_dot_product_attention = _sdpa
                try:
                    return super().__call__(*args, **kwargs)
                finally:
                    torch.nn.functional.scaled_dot_product_attention = orig_sdpa

        logger.debug("Built Sage Wan attention processor (variant=%s).", variant)
        return _SageWanAttnProcessor()
