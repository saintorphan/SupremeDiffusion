"""Native LatentSync lip sync pipeline — loads/unloads like other pipelines."""

from __future__ import annotations

import gc
import logging
import math
import tempfile
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_LATENTSYNC_DIR = Path("~/LatentSync").expanduser()
_CKPT_PATH = _LATENTSYNC_DIR / "checkpoints" / "latentsync_unet.pt"
_WHISPER_PATH = _LATENTSYNC_DIR / "checkpoints" / "whisper" / "tiny.pt"
_CONFIG_PATH = _LATENTSYNC_DIR / "configs" / "unet" / "stage2_512.yaml"
_SCHEDULER_DIR = _LATENTSYNC_DIR / "configs"
_MASK_PATH = _LATENTSYNC_DIR / "latentsync" / "utils" / "mask.png"


class NativeLipSyncPipeline:
    """Manages LatentSync model lifecycle and inference.

    Follows the same load/unload pattern as WanI2VPipeline, SDImagePipeline, etc.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the LatentSync pipeline into GPU memory."""
        if self._loaded:
            return

        logger.info("Loading LatentSync pipeline...")

        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.whisper.audio2feature import Audio2Feature

        config = OmegaConf.load(str(_CONFIG_PATH))

        is_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16 else torch.float32

        scheduler = DDIMScheduler.from_pretrained(str(_SCHEDULER_DIR))

        whisper_path = str(_WHISPER_PATH)
        audio_encoder = Audio2Feature(
            model_path=whisper_path,
            device="cpu",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=dtype,
        )
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            str(_CKPT_PATH),
            device="cpu",
        )
        unet = unet.to(dtype=dtype)

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        pipeline.enable_sequential_cpu_offload()

        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()

        # DeepCache for faster inference
        try:
            from DeepCache import DeepCacheSDHelper
            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        except ImportError:
            logger.warning("DeepCache not available, skipping")

        self._pipeline = pipeline
        self._config = config
        self._dtype = dtype
        self._loaded = True
        logger.info("LatentSync pipeline ready.")

    def unload(self) -> None:
        """Free all GPU memory."""
        if self._pipeline is not None:
            # Move components to CPU and delete
            for attr in ("vae", "unet"):
                component = getattr(self._pipeline, attr, None)
                if component is not None:
                    try:
                        component.to("cpu")
                    except Exception:
                        pass
            del self._pipeline
            self._pipeline = None

        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("LatentSync pipeline unloaded.")

    def run(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        inference_steps: int = 20,
        guidance_scale: float = 1.5,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """Run lip sync inference.

        Args:
            video_path: Path to 512x512 cropped face video.
            audio_path: Path to audio file.
            output_path: Where to write the synced video.
            inference_steps: DDIM denoising steps.
            guidance_scale: Classifier-free guidance strength.
            progress_callback: Optional (fraction, description) callback.

        Returns:
            Path to output video.
        """
        if not self._loaded:
            self.load()

        if progress_callback:
            progress_callback(0.05, "Preparing audio & video...")

        config = self._config

        # Estimate total batches for progress tracking from video length
        import math
        try:
            from latentsync.utils.util import read_video
            video_frames = read_video(video_path, use_decord=False)
            num_frames_cfg = config.data.num_frames
            est_batches = max(1, math.ceil(len(video_frames) / num_frames_cfg))
            del video_frames  # free memory, pipeline will re-read
        except Exception:
            est_batches = 1

        # Track progress through the callback
        # callback(step, timestep, latents) is called per denoising step
        self._batch_count = 0
        self._est_batches = est_batches

        def _step_callback(step: int, timestep, latents) -> None:
            if not progress_callback:
                return
            # Map to overall progress: each batch does inference_steps
            batch_frac = self._batch_count / self._est_batches
            step_frac = (step + 1) / inference_steps
            # Inference portion is 0.10 to 0.80
            overall = 0.10 + 0.70 * (batch_frac + step_frac / self._est_batches)
            overall = min(overall, 0.80)
            progress_callback(
                overall,
                f"Batch {self._batch_count + 1}/{self._est_batches}, "
                f"step {step + 1}/{inference_steps}",
            )

        # Monkey-patch tqdm in the pipeline to track batch progress
        import tqdm as _tqdm_mod
        _orig_tqdm = _tqdm_mod.tqdm
        pipeline_self = self

        class _TrackingTqdm(_orig_tqdm):
            def __init__(self, *args, **kwargs):
                desc = kwargs.get("desc", "")
                super().__init__(*args, **kwargs)
                self._is_inference = "inference" in desc.lower() if desc else False

            def update(self, n=1):
                super().update(n)
                if self._is_inference and progress_callback:
                    batch = pipeline_self._batch_count
                    total_b = pipeline_self._est_batches
                    frac = 0.10 + 0.70 * ((batch + 1) / total_b)
                    progress_callback(min(frac, 0.80),
                                      f"Batch {batch + 1}/{total_b} complete")
                    pipeline_self._batch_count = batch + 1

        _tqdm_mod.tqdm = _TrackingTqdm

        try:
            if progress_callback:
                progress_callback(0.08, "Running LatentSync inference...")

            self._pipeline(
                video_path=video_path,
                audio_path=audio_path,
                video_out_path=output_path,
                num_frames=config.data.num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                weight_dtype=self._dtype,
                width=config.data.resolution,
                height=config.data.resolution,
                mask_image_path=str(_MASK_PATH),
                callback=_step_callback,
                callback_steps=1,
            )
        finally:
            _tqdm_mod.tqdm = _orig_tqdm

        if progress_callback:
            progress_callback(0.85, "LatentSync inference complete")

        return output_path
