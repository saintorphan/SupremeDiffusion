"""Sequence generation worker — chains image gen → video gen → stitch.

This worker runs a full stage-based sequence on a background thread:
  1. Generate (or accept) a starting image via txt2img
  2. For each subsequent stage: img2img the previous frame with the stage prompt
  3. Generate a video clip from each stage image via img2vid (Mode 1)
  4. Optionally generate an ending clip
  5. Stitch all clips into one final video

Pipeline swapping: SD image and Wan video pipelines are mutually exclusive
(they share VRAM). The worker automatically loads/unloads the right pipeline
before each operation — no manual pipeline management needed.
"""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Signal

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — pipeline swap
# ---------------------------------------------------------------------------

def _ensure_sd(state: Any) -> tuple[Any, Any]:
    """Make sure SD image pipeline is loaded. Returns (sd_pipeline, img_pipeline).

    Loads SD if not already loaded (which auto-unloads video/FLUX/etc).
    """
    if not state.sd_pipelines_loaded:
        logger.info("Sequence worker: loading SD image pipeline...")
        state.load_sd_pipelines()
    return state.sd_pipeline, state.img_pipeline


def _ensure_video(state: Any) -> tuple[Any, Any]:
    """Make sure Wan video pipeline is loaded. Returns (wan_pipeline, gen_pipeline).

    Loads video if not already loaded (which auto-unloads SD/FLUX/etc).
    """
    if not state.pipelines_loaded:
        logger.info("Sequence worker: loading video pipeline...")
        state.load_pipelines()
    return state.wan_pipeline, state.pipeline


class SingleStageWorker(BaseWorker):
    """Generate a single stage: image (txt2img or img2img) + video clip.

    Used by Guided mode to run one stage at a time so the user can
    approve, regenerate, or tweak the prompt between stages.

    Signals:
        stage_image_ready(int, str): (stage_index, image_path)
        stage_clip_ready(int, str): (stage_index, clip_path)
    """

    stage_image_ready = Signal(int, str)
    stage_clip_ready = Signal(int, str)

    def __init__(
        self,
        *,
        state: Any,
        project_name: str,
        img_config: Any,
        vid_config: Any,
        output_dir: str,
        subject_prompt: str,
        negative_prompt: str,
        stage: dict,
        stage_index: int,
        prev_image_path: Optional[str] = None,
        generate_clip: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._project_name = project_name
        self._img_config = img_config
        self._vid_config = vid_config
        self._output_dir = Path(output_dir)
        self._subject_prompt = subject_prompt
        self._negative_prompt = negative_prompt
        self._stage = stage
        self._stage_index = stage_index
        self._prev_image = prev_image_path
        self._generate_clip = generate_clip

    def do_work(self) -> dict:
        """Generate one stage. Returns dict with image_path, clip_path, last_frame_path."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        idx = self._stage_index
        stage_prompt = self._build_prompt(self._stage["prompt"])
        drift = self._stage.get("drift", 2)

        # ── 1. Generate stage image (needs SD pipeline) ──
        _sd, img_pipeline = _ensure_sd(self._state)

        if self._prev_image is None:
            self.progress.emit(0.0, f"Stage {idx + 1} — Generating image (txt2img)...")
            img_path = self._run_txt2img(img_pipeline, stage_prompt)
        else:
            denoise = min(0.15 * drift, 0.75)
            self.progress.emit(0.0, f"Stage {idx + 1} — Generating image (img2img, denoise {denoise:.2f})...")
            img_path = self._run_img2img(img_pipeline, self._prev_image, stage_prompt, denoise)

        stage_img_path = str(self._output_dir / f"stage_{idx + 1}.png")
        shutil.copy2(img_path, stage_img_path)
        self.stage_image_ready.emit(idx, stage_img_path)

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        result = {"image_path": stage_img_path, "clip_path": None, "last_frame_path": stage_img_path}

        # ── 2. Generate video clip (needs Wan pipeline) ──
        if self._generate_clip:
            _wan, vid_pipeline = _ensure_video(self._state)
            self.progress.emit(0.5, f"Stage {idx + 1} — Generating video clip...")
            clip_path = self._run_img2vid(vid_pipeline, stage_img_path, stage_prompt, idx)
            self.stage_clip_ready.emit(idx, clip_path)

            # Extract last frame for next stage
            last_frame = self._extract_last_frame(clip_path, idx)
            result["clip_path"] = clip_path
            result["last_frame_path"] = last_frame

        self.progress.emit(1.0, f"Stage {idx + 1} complete")
        return result

    def _build_prompt(self, stage_prompt: str) -> str:
        parts = [p for p in [self._subject_prompt, stage_prompt] if p.strip()]
        return ", ".join(parts)

    def _run_txt2img(self, img_pipeline: Any, prompt: str) -> str:
        cfg = copy.copy(self._img_config)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = self._negative_prompt
        cfg.img_batch_size = 1
        total_steps = cfg.img_steps or 20
        callback = self.make_step_callback(total_steps, "txt2img")
        results = img_pipeline.run_txt2img(
            project_name=self._project_name, config=cfg, callback=callback,
        )
        if not results:
            raise RuntimeError("txt2img returned no images")
        return results[0]

    def _run_img2img(self, img_pipeline: Any, source_path: str, prompt: str, denoise: float) -> str:
        cfg = copy.copy(self._img_config)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = self._negative_prompt
        cfg.img_batch_size = 1
        cfg.img_denoising_strength = denoise
        total_steps = cfg.img_steps or 20
        callback = self.make_step_callback(total_steps, "img2img")
        results = img_pipeline.run_img2img(
            project_name=self._project_name, image_path=source_path,
            config=cfg, callback=callback,
        )
        if not results:
            raise RuntimeError("img2img returned no images")
        return results[0]

    def _run_img2vid(self, vid_pipeline: Any, image_path: str, prompt: str, stage_idx: int) -> str:
        cfg = self._vid_config
        if hasattr(cfg, "prompt"):
            cfg.prompt = prompt
        total_steps = cfg.num_inference_steps or 4
        callback = self.make_step_callback(total_steps, f"Stage {stage_idx + 1} video")
        result = vid_pipeline.run_mode1(
            project_name=self._project_name, image_path=image_path,
            project_config=cfg, generate_guidance=False, callback=callback,
        )
        if isinstance(result, tuple):
            _, video_path = result
        else:
            video_path = result
        return str(video_path)

    def _extract_last_frame(self, video_path: str, stage_idx: int) -> str:
        from supremediffusion.utils.video import extract_single_frame, probe_video
        info = probe_video(video_path)
        num_frames = info.get("num_frames", 1)
        img = extract_single_frame(video_path, max(0, num_frames - 1))
        out_path = str(self._output_dir / f"stage_{stage_idx + 1}_lastframe.png")
        img.save(out_path)
        return out_path


class StitchWorker(BaseWorker):
    """Stitch a list of clips into a single video. Lightweight final step."""

    def __init__(self, *, clips: list[str], output_path: str, parent=None) -> None:
        super().__init__(parent)
        self._clips = clips
        self._output_path = output_path

    def do_work(self) -> str:
        from supremediffusion.utils.video import concat_videos_multi
        if not self._clips:
            raise RuntimeError("No clips to stitch")
        self.progress.emit(0.5, "Stitching clips...")
        concat_videos_multi(self._clips, self._output_path)
        self.progress.emit(1.0, "Stitch complete")
        return self._output_path


class SequenceGenerationWorker(BaseWorker):
    """Chain image → video → stitch for a full transformation sequence.

    Automatically swaps between SD and Wan pipelines as needed — no manual
    pipeline loading required.

    Signals:
        stage_image_ready(int, str): (stage_index, image_path) — emitted when
            a stage image is generated so the UI can show thumbnails.
        stage_clip_ready(int, str): (stage_index, clip_path) — emitted when
            a video clip for a stage is rendered.
    """

    stage_image_ready = Signal(int, str)
    stage_clip_ready = Signal(int, str)

    def __init__(
        self,
        *,
        state: Any,
        project_name: str,
        img_config: Any,
        vid_config: Any,
        output_dir: str,
        subject_prompt: str,
        negative_prompt: str,
        stages: list[dict],
        ending: Optional[dict] = None,
        start_image_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._project_name = project_name
        self._img_config = img_config
        self._vid_config = vid_config
        self._output_dir = Path(output_dir)
        self._subject_prompt = subject_prompt
        self._negative_prompt = negative_prompt
        self._stages = stages
        self._ending = ending
        self._start_image = start_image_path

    # -- Main work loop -------------------------------------------------------

    def do_work(self) -> str:
        """Run the full sequence. Returns path to the final stitched video."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        total = len(self._stages) + (1 if self._ending else 0)
        clips: list[str] = []
        prev_image: str | None = self._start_image

        for i, stage in enumerate(self._stages):
            if self.is_aborted:
                raise InterruptedError("Aborted by user")

            phase = f"Stage {i + 1}/{total}"
            overall_frac = i / total

            # ── 1. Generate stage image (swap to SD) ──
            stage_prompt = self._build_prompt(stage["prompt"])
            drift = stage.get("drift", 2)

            _sd, img_pipeline = _ensure_sd(self._state)

            if prev_image is None:
                self.progress.emit(overall_frac, f"{phase} — Generating image...")
                prev_image = self._run_txt2img(img_pipeline, stage_prompt)
            else:
                self.progress.emit(overall_frac, f"{phase} — Generating stage image...")
                denoise = min(0.15 * drift, 0.75)
                prev_image = self._run_img2img(img_pipeline, prev_image, stage_prompt, denoise)

            # Save stage image
            stage_img_path = str(self._output_dir / f"stage_{i + 1}.png")
            shutil.copy2(prev_image, stage_img_path)
            self.stage_image_ready.emit(i, stage_img_path)
            logger.info("Stage %d image: %s", i + 1, stage_img_path)

            if self.is_aborted:
                raise InterruptedError("Aborted by user")

            # ── 2. Generate video clip (swap to Wan) ──
            _wan, vid_pipeline = _ensure_video(self._state)
            self.progress.emit(overall_frac + 0.5 / total, f"{phase} — Generating video clip...")
            clip_path = self._run_img2vid(vid_pipeline, prev_image, stage_prompt, i)
            clips.append(clip_path)
            self.stage_clip_ready.emit(i, clip_path)
            logger.info("Stage %d clip: %s", i + 1, clip_path)

            # Extract last frame for next stage's img2img
            prev_image = self._extract_last_frame(clip_path, i)

        # ── 3. Optional ending ──
        if self._ending and not self.is_aborted:
            ending_idx = len(self._stages)
            self.progress.emit((total - 1) / total, "Generating ending...")

            _sd, img_pipeline = _ensure_sd(self._state)
            ending_prompt = self._build_prompt(self._ending["prompt"])
            ending_img = self._run_img2img(img_pipeline, prev_image, ending_prompt, 0.65)
            ending_img_path = str(self._output_dir / "ending.png")
            shutil.copy2(ending_img, ending_img_path)
            self.stage_image_ready.emit(ending_idx, ending_img_path)

            if not self.is_aborted:
                _wan, vid_pipeline = _ensure_video(self._state)
                ending_clip = self._run_img2vid(vid_pipeline, ending_img, ending_prompt, ending_idx)
                clips.append(ending_clip)
                self.stage_clip_ready.emit(ending_idx, ending_clip)

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # ── 4. Stitch all clips ──
        self.progress.emit(0.95, "Stitching clips...")
        final_path = str(self._output_dir / "sequence_final.mp4")
        self._stitch(clips, final_path)

        self.progress.emit(1.0, "Sequence complete")
        logger.info("Sequence complete: %s (%d clips)", final_path, len(clips))
        return final_path

    # -- Internal helpers -----------------------------------------------------

    def _build_prompt(self, stage_prompt: str) -> str:
        """Combine subject description with stage-specific prompt."""
        parts = [p for p in [self._subject_prompt, stage_prompt] if p.strip()]
        return ", ".join(parts)

    def _run_txt2img(self, img_pipeline: Any, prompt: str) -> str:
        """Generate a single image via txt2img. Returns image path."""
        cfg = copy.copy(self._img_config)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = self._negative_prompt
        cfg.img_batch_size = 1

        total_steps = cfg.img_steps or 20
        callback = self.make_step_callback(total_steps, "txt2img")

        results = img_pipeline.run_txt2img(
            project_name=self._project_name,
            config=cfg,
            callback=callback,
        )
        if not results:
            raise RuntimeError("txt2img returned no images")
        return results[0]

    def _run_img2img(self, img_pipeline: Any, source_path: str, prompt: str, denoise: float) -> str:
        """Run img2img on source with given prompt and denoise strength."""
        cfg = copy.copy(self._img_config)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = self._negative_prompt
        cfg.img_batch_size = 1
        cfg.img_denoising_strength = denoise

        total_steps = cfg.img_steps or 20
        callback = self.make_step_callback(total_steps, "img2img")

        results = img_pipeline.run_img2img(
            project_name=self._project_name,
            image_path=source_path,
            config=cfg,
            callback=callback,
        )
        if not results:
            raise RuntimeError("img2img returned no images")
        return results[0]

    def _run_img2vid(self, vid_pipeline: Any, image_path: str, prompt: str, stage_idx: int) -> str:
        """Generate a video clip from an image via Mode 1."""
        cfg = self._vid_config
        if hasattr(cfg, "prompt"):
            cfg.prompt = prompt

        total_steps = cfg.num_inference_steps or 4
        callback = self.make_step_callback(total_steps, f"Stage {stage_idx + 1} video")

        result = vid_pipeline.run_mode1(
            project_name=self._project_name,
            image_path=image_path,
            project_config=cfg,
            generate_guidance=False,
            callback=callback,
        )

        if isinstance(result, tuple):
            _, video_path = result
        else:
            video_path = result

        return str(video_path)

    def _extract_last_frame(self, video_path: str, stage_idx: int) -> str:
        """Extract the last frame of a video as PNG for the next stage."""
        from supremediffusion.utils.video import extract_single_frame, probe_video

        info = probe_video(video_path)
        num_frames = info.get("num_frames", 1)
        last_idx = max(0, num_frames - 1)

        img = extract_single_frame(video_path, last_idx)
        out_path = str(self._output_dir / f"stage_{stage_idx + 1}_lastframe.png")
        img.save(out_path)
        return out_path

    def _stitch(self, clips: list[str], output_path: str) -> None:
        """Concatenate all clips into one video."""
        from supremediffusion.utils.video import concat_videos_multi

        if not clips:
            raise RuntimeError("No clips to stitch")
        concat_videos_multi(clips, output_path)
