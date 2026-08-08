"""Worker for LTX Audio-to-Video generation (image+audio and control video+audio)."""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any

from .base import BaseWorker

logger = logging.getLogger(__name__)


class LTXA2VidWorker(BaseWorker):
    """Run LTX video generation with audio conditioning on a background thread.

    Two modes:
    - **Image mode** (control_video=None): TI2VidTwoStagesPipeline with
      image conditioning + audio conditioning.
    - **Control Video mode** (control_video set): ICLoraPipeline with
      source video as reference conditioning + audio conditioning.

    Uses Wan2GP's unified pipelines (copied into supremediffusion/models/ltx2/)
    which accept both ``audio_conditionings`` and ``video_conditioning``.

    Emits ``finished_ok`` with the output video path.
    """

    def __init__(
        self,
        source_image_path: str,
        audio_path: str,
        output_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        *,
        width: int = 1024,
        height: int = 768,
        num_frames: int = 121,
        fps: float = 24.0,
        steps: int = 30,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        a2v_scale: float = 3.0,
        seed: int = -1,
        audio_start: float = 0.0,
        control_video: str | None = None,
        control_strength: float = 0.85,
        loop_silence_pad: bool = False,
        source_width: int = 0,
        source_height: int = 0,
        source_fps: float = 0,
        state: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_path = source_image_path
        self._audio_path = audio_path
        self._output_path = output_path
        self._prompt = prompt
        self._negative = negative_prompt
        self._width = width
        self._height = height
        self._num_frames = num_frames
        self._fps = fps
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._stg_scale = stg_scale
        self._a2v_scale = a2v_scale
        self._seed = seed
        self._audio_start = audio_start
        self._control_video = control_video
        self._control_strength = control_strength
        self._loop_silence_pad = loop_silence_pad
        self._source_width = source_width
        self._source_height = source_height
        self._source_fps = source_fps
        self._state = state
        # Temp WAV conditioning files we create (extract/pad) so they can be
        # cleaned up after the muxed output is written.
        self._temp_audio_files: list[str] = []
        # Live pipeline handle so abort() can flip its interrupt flag in
        # addition to the interrupt_check callback the loops poll.
        self._pipe: Any = None

    def abort(self) -> None:
        """Request cancellation.

        In addition to the standard ``is_aborted`` flag (polled by the
        ``interrupt_check`` callback we hand to the pipeline), push the
        LTX backend interrupt flag directly. The two-stage / IC-LoRA
        pipelines obtain a fresh transformer per call, so the callback is
        the canonical path; setting ``_interrupt`` here mirrors the
        ltx_backend setter (see ``ltx2.py`` ``_interrupt`` property) for
        any transformer/wrapper that exposes it, and is a harmless no-op
        otherwise.
        """
        super().abort()
        pipe = self._pipe
        if pipe is None:
            return
        for target in (
            pipe,
            getattr(pipe, "transformer", None),
            getattr(pipe, "velocity_model", None),
            getattr(pipe, "_ltx2", None),
        ):
            if target is None:
                continue
            try:
                if hasattr(target, "_interrupt"):
                    target._interrupt = True
            except Exception:  # pragma: no cover - defensive
                pass

    def do_work(self) -> str:
        if self._control_video:
            return self._run_control_video()
        return self._run_image_to_video()

    def _get_model_paths(self) -> dict[str, str]:
        cfg = self._state.global_config if self._state else None
        return cfg.model_paths if cfg else {}

    def _resolve_seed(self) -> int:
        seed = self._seed
        if seed < 0:
            import random
            seed = random.randint(0, 2**32 - 1)
        return seed

    def _unload_other_pipelines(self) -> None:
        if self._state is not None:
            self._state.unload_video_pipelines()
            self._state.unload_sd_pipelines()
            self._state.unload_flux_pipelines()
            self._state._force_gc()

    def _prepare_audio_path(self) -> str:
        """Resolve audio: extract from video if needed, pad for loop silence."""
        audio_path = self._audio_path
        audio_is_video = Path(audio_path).suffix.lower() in {
            ".mp4", ".mov", ".avi", ".mkv", ".webm",
        }
        if audio_is_video:
            audio_path = self._extract_audio_from_video(audio_path)
        if self._loop_silence_pad:
            audio_path = self._pad_audio_start(audio_path)
        return audio_path

    def _load_and_encode_audio(self, audio_path: str, pipe: Any) -> list:
        """Load audio file, encode via the pipeline's audio VAE, return [AudioConditionByLatent]."""
        import torch
        import torchaudio

        from supremediffusion.models.ltx2.ltx_core.conditioning import AudioConditionByLatent
        from supremediffusion.models.ltx2.ltx_core.model.audio_vae import AudioProcessor

        # Load audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        # Handle start offset
        if self._audio_start > 0:
            skip = int(self._audio_start * sample_rate)
            waveform = waveform[:, skip:]
        # Trim to video duration
        max_samples = int(self._num_frames / max(self._fps, 1) * sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]

        waveform = waveform.unsqueeze(0)  # (1, C, T)

        # Get audio encoder from the pipeline's model ledger
        audio_encoder = pipe.stage_1_model_ledger.audio_encoder()

        # Convert waveform → mel → latent
        audio_processor = AudioProcessor(
            sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        )
        waveform_cpu = waveform.to(device="cpu", dtype=torch.float32)
        audio_processor = audio_processor.to("cpu")
        mel = audio_processor.waveform_to_mel(waveform_cpu, sample_rate)

        audio_params = next(audio_encoder.parameters(), None)
        device = audio_params.device if audio_params is not None else torch.device("cuda")
        dtype = audio_params.dtype if audio_params is not None else torch.bfloat16
        mel = mel.to(device=device, dtype=dtype)

        with torch.inference_mode():
            audio_latent = audio_encoder(mel)

        audio_latent = audio_latent.to(device="cuda", dtype=torch.bfloat16)

        return [AudioConditionByLatent(audio_latent, 1.0)]

    def _pad_audio_start(self, audio_path: str, pad_seconds: float = 0.5) -> str:
        """Prepend silence to audio so mouth is idle at loop seam."""
        import subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="ltx_padded_", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=24000:cl=mono",
            "-t", str(pad_seconds),
            "-i", audio_path,
            "-filter_complex", "[0][1]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-acodec", "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            logger.warning("Audio padding failed, using original")
            Path(tmp.name).unlink(missing_ok=True)
            return audio_path
        logger.info("Padded %.1fs silence at start of audio", pad_seconds)
        self._temp_audio_files.append(tmp.name)
        return tmp.name

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video to a temp WAV file."""
        import subprocess
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="ltx_src_audio_", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            Path(tmp.name).unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to extract audio: "
                + result.stderr.decode(errors="replace")[:200]
            )
        self._temp_audio_files.append(tmp.name)
        return tmp.name

    # ------------------------------------------------------------------
    # Mode 1: Image + Audio → Video
    # ------------------------------------------------------------------

    def _run_image_to_video(self) -> str:
        import torch

        self.status.emit("Loading LTX pipeline...")
        self.progress.emit(0.0, "Loading pipeline...")
        self._unload_other_pipelines()

        from supremediffusion.models.ltx2.ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        paths = self._get_model_paths()
        checkpoint = paths.get("ltx_transformer", "")
        gemma_root = paths.get("ltx_text_encoder", "")
        spatial = paths.get("ltx_spatial_upsampler", "")

        if not checkpoint:
            raise ValueError("No LTX transformer path configured.")
        if not gemma_root:
            raise ValueError("No LTX text encoder path configured.")

        pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint,
            spatial_upsampler_path=spatial or "",
            gemma_root=gemma_root,
            loras=[],
        )
        self._pipe = pipe

        self.status.emit("Encoding audio...")
        self.progress.emit(0.05, "Encoding audio...")

        audio_path = self._prepare_audio_path()
        audio_conds = self._load_and_encode_audio(audio_path, pipe)

        self.status.emit("Generating video with audio...")
        self.progress.emit(0.1, "Generating...")

        seed = self._resolve_seed()

        video_iter, raw_audio = pipe(
            prompt=self._prompt,
            negative_prompt=self._negative or "",
            seed=seed,
            height=self._height,
            width=self._width,
            num_frames=self._num_frames,
            frame_rate=self._fps,
            num_inference_steps=int(self._steps),
            cfg_guidance_scale=float(self._cfg_scale),
            audio_cfg_guidance_scale=float(self._a2v_scale),
            images=[(self._image_path, 0, 1.0)],
            audio_conditionings=audio_conds,
            interrupt_check=lambda: self.is_aborted,
        )

        self._save_result(video_iter, audio_path, pipe=pipe)
        return self._output_path

    # ------------------------------------------------------------------
    # Mode 2: Control Video + Audio → Video (IC-LoRA)
    # ------------------------------------------------------------------

    def _run_control_video(self) -> str:
        import torch

        self.status.emit("Loading LTX IC-LoRA pipeline...")
        self.progress.emit(0.0, "Loading pipeline...")
        self._unload_other_pipelines()

        from supremediffusion.models.ltx2.ltx_pipelines.ic_lora import ICLoraPipeline
        from supremediffusion.models.ltx2.ltx_core.loader import (
            LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP,
        )

        paths = self._get_model_paths()
        checkpoint = paths.get("ltx_transformer", "")
        gemma_root = paths.get("ltx_text_encoder", "")
        spatial = paths.get("ltx_spatial_upsampler", "")

        if not checkpoint:
            raise ValueError("No LTX transformer path configured.")
        if not gemma_root:
            raise ValueError("No LTX text encoder path configured.")

        # Find IC-LoRA in lora dir
        loras = []
        lora_dir = paths.get("ltx_lora_dir", "")
        if lora_dir:
            lora_path = Path(lora_dir)
            if lora_path.is_dir():
                for f in lora_path.iterdir():
                    if "ic-lora" in f.name.lower() and f.suffix.lower() == ".safetensors":
                        loras.append(LoraPathStrengthAndSDOps(
                            str(f), 1.0, LTXV_LORA_COMFY_RENAMING_MAP,
                        ))
                        logger.info("Using IC-LoRA: %s", f.name)
                        break

        pipe = ICLoraPipeline(
            checkpoint_path=checkpoint,
            spatial_upsampler_path=spatial or "",
            gemma_root=gemma_root,
            loras=loras,
        )
        self._pipe = pipe

        self.status.emit("Encoding audio...")
        self.progress.emit(0.05, "Encoding audio...")

        audio_path = self._prepare_audio_path()
        audio_conds = self._load_and_encode_audio(audio_path, pipe)

        self.status.emit("Generating with control video + audio...")
        self.progress.emit(0.1, "Generating...")

        seed = self._resolve_seed()

        # NOTE: ICLoraPipeline is distilled (fixed DISTILLED_SIGMA_VALUES) and
        # forces alt_guidance_scale=1.0 internally, so it has no
        # num_inference_steps / cfg_guidance_scale / audio_cfg_guidance_scale
        # knobs — steps/cfg/a2v are inherently inert on the control-video path.
        video_iter, raw_audio = pipe(
            prompt=self._prompt,
            seed=seed,
            height=self._height,
            width=self._width,
            num_frames=self._num_frames,
            frame_rate=self._fps,
            images=[(self._image_path, 0, 1.0)],
            video_conditioning=[(self._control_video, self._control_strength)],
            audio_conditionings=audio_conds,
            interrupt_check=lambda: self.is_aborted,
        )

        self._save_result(video_iter, audio_path, pipe=pipe)
        return self._output_path

    # ------------------------------------------------------------------
    # Shared output
    # ------------------------------------------------------------------

    def _save_result(self, video_iter, audio_wav_path: str, *, pipe=None) -> None:
        import torch

        self.status.emit("Encoding video...")
        self.progress.emit(0.8, "Encoding...")

        all_chunks = []
        for chunk in video_iter:
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            if isinstance(chunk, torch.Tensor):
                all_chunks.append(chunk.detach().cpu())

        if not all_chunks:
            raise RuntimeError("Pipeline returned no frames.")

        video_tensor = torch.cat(all_chunks, dim=0)
        if video_tensor.dim() == 4 and video_tensor.shape[1] in (1, 3):
            video_tensor = video_tensor.permute(0, 2, 3, 1)

        video_np = (video_tensor.float().clamp(0, 1) * 255).byte().numpy()

        # Preserve fractional rates (23.976/29.97) — truncating to int causes
        # A/V drift over the clip length.
        output_fps = float(self._fps)

        # Honor the user's configured encoder (nvenc-first) instead of the
        # OutputHandler libx264 default.
        from sdqt.utils.codec import configured_codec_args
        codec_args = configured_codec_args()

        # Check if we need to scale to match source resolution
        need_scale = (
            self._source_width > 0 and self._source_height > 0
            and (video_np.shape[2] != self._source_width
                 or video_np.shape[1] != self._source_height)
        )
        need_fps_match = self._source_fps > 0 and abs(self._source_fps - self._fps) > 0.5

        from supremediffusion.core.output_handler import OutputHandler

        if need_scale or need_fps_match:
            # Save to temp first, then ffmpeg scale/fps to final output
            tmp_path = self._output_path + ".tmp.mp4"
            OutputHandler.save_generation(
                video_np, tmp_path, fps=output_fps, codec=codec_args,
                audio_path=audio_wav_path,
            )
            self.status.emit("Scaling to match source...")
            self.progress.emit(0.9, "Scaling to match source...")
            self._scale_to_source(tmp_path, self._output_path)
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
        else:
            OutputHandler.save_generation(
                video_np, self._output_path, fps=output_fps, codec=codec_args,
                audio_path=audio_wav_path,
            )

        self._pipe = None
        del pipe, video_tensor, video_np, all_chunks
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except ImportError:
            pass

        # Prune temp WAV conditioning files now that audio has been muxed.
        for tmp in self._temp_audio_files:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass
        self._temp_audio_files.clear()

        self.progress.emit(1.0, "Complete")

    def _scale_to_source(self, input_path: str, output_path: str) -> None:
        """Scale video to match source resolution and FPS for seamless stitching."""
        import subprocess

        from sdqt.utils.codec import (
            configured_codec_args,
            configured_encoder_name,
            pix_fmt_args,
        )

        target_w = self._source_width
        target_h = self._source_height
        target_fps = self._source_fps if self._source_fps > 0 else self._fps

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", f"scale={target_w}:{target_h}:flags=lanczos,fps={target_fps}",
            *configured_codec_args(),
            *pix_fmt_args(configured_encoder_name()),
        ]
        # Copy audio if present
        cmd.extend(["-c:a", "copy"])
        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.warning(
                "Scale failed, using unscaled output: %s",
                result.stderr.decode(errors="replace")[:200],
            )
            import shutil
            shutil.copy2(input_path, output_path)
