"""Worker threads for pipeline inference."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
from supremediffusion.config.project_config import ProjectConfig

from sdqt.utils.codec import configured_codec_args

from .base import BaseWorker

logger = logging.getLogger(__name__)

# Map UI combo text → PostProcessor parameter values
_TEMPORAL_MAP = {"RIFE x2": "rife2", "RIFE x4": "rife4"}
_SPATIAL_MAP = {"Lanczos 1.5x": "lanczos1.5", "Lanczos 2.0x": "lanczos2"}


class GenResult(str):
    """``finished_ok`` payload for :class:`InferenceWorker`.

    Behaves as the output video path string (so ``str(result)`` /
    ``Path(result)`` keep working for callers that only want the path),
    while also carrying the actually-used ``resolved_seed`` (after any -1
    randomization) for the SEED-REUSE contract.
    """

    resolved_seed: int = -1

    def __new__(cls, video_path: str, resolved_seed: int = -1):
        obj = super().__new__(cls, video_path)
        obj.resolved_seed = resolved_seed
        return obj


def _needs_postprocessing(cfg: ProjectConfig) -> bool:
    """Check if any post-processing is enabled in the config."""
    temporal = getattr(cfg, "temporal_upsampling", "Disabled")
    spatial = getattr(cfg, "spatial_upsampling", "Disabled")
    grain = getattr(cfg, "film_grain_intensity", 0.0)
    return (
        temporal in _TEMPORAL_MAP
        or spatial in _SPATIAL_MAP
        or (grain and grain > 0)
    )


def _run_postprocessing(
    frames: np.ndarray,
    fps: int,
    cfg: ProjectConfig,
    progress_cb=None,
) -> tuple[np.ndarray, int]:
    """Run PostProcessor on raw frames and return (processed_frames, output_fps)."""
    from supremediffusion.postprocessing import PostProcessor

    temporal = _TEMPORAL_MAP.get(getattr(cfg, "temporal_upsampling", ""), "")
    spatial = _SPATIAL_MAP.get(getattr(cfg, "spatial_upsampling", ""), "")
    grain_intensity = getattr(cfg, "film_grain_intensity", 0.0) or 0.0
    grain_saturation = getattr(cfg, "film_grain_saturation", 0.5) or 0.5

    pp = PostProcessor()
    return pp.process(
        frames=frames,
        fps=fps,
        temporal_upsampling=temporal,
        spatial_upsampling=spatial,
        film_grain_intensity=grain_intensity,
        film_grain_saturation=grain_saturation,
        progress_cb=progress_cb,
    )


class InferenceWorker(BaseWorker):
    """Run a generation pipeline call on a background thread.

    Supports Mode 1, 2, and 3, with optional post-processing.

    Usage::

        worker = InferenceWorker(pipeline, project_name, mode=2,
                                 first_frame_path=a, last_frame_path=b,
                                 project_config=cfg)
        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_done)      # receives GenResult(path; .resolved_seed)
        worker.error.connect(on_error)
        worker.start()
    """

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        project_config: ProjectConfig,
        *,
        mode: int = 1,
        image_path: Optional[str] = None,
        first_frame_path: Optional[str] = None,
        last_frame_path: Optional[str] = None,
        video_path: Optional[str] = None,
        start_frame: int = 0,
        end_frame: int = 0,
        generate_guidance: bool = False,
        guidance_frame: str = "first",
        guidance_tensor: Any = None,
        guidance_from_source: bool = False,
        global_config: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._config = project_config
        self._mode = mode
        self._image_path = image_path
        self._first_frame = first_frame_path
        self._last_frame = last_frame_path
        self._video_path = video_path
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._gen_guidance = generate_guidance
        self._guidance_frame = guidance_frame
        self._guidance_tensor = guidance_tensor
        self._guidance_from_source = guidance_from_source
        self._global_config = global_config

    @property
    def global_config(self) -> Any:
        """Expose the GlobalConfig so PostProcessPipeline (passed ``state=self``)
        can resolve ``self.state.global_config`` for upscale + RIFE steps."""
        return self._global_config

    def do_work(self) -> "GenResult":
        total_steps = self._config.num_inference_steps or 4
        callback = self.make_step_callback(total_steps)

        self.status.emit("Loading model...")
        self.progress.emit(0.0, "Starting...")

        # Route pre-generation progress (e.g. depth/pose control-map extraction,
        # which runs before the denoising step callback fires) to the same
        # status line. The LTX pipeline reads this optional (frac, text) sink.
        _active = getattr(self._pipeline, "wan", None)
        if _active is not None:
            try:
                _active._status_cb = self.progress.emit
            except Exception:  # noqa: BLE001
                pass

        # Resolve a concrete seed for the SEED-REUSE contract. When the UI
        # leaves seed at -1 (random) the backend never records the actual
        # value, so "Reuse seed" would have nothing to write back. Pick a
        # real seed here and stamp it onto the config before generation so
        # the run is reproducible and the resolved value can be reported.
        resolved_seed = getattr(self._config, "seed", -1)
        if resolved_seed is None or resolved_seed < 0:
            import random
            resolved_seed = random.randint(0, 999_999_999)
            try:
                self._config.seed = resolved_seed
            except Exception:  # noqa: BLE001
                pass

        postprocess = _needs_postprocessing(self._config) and self._mode in (1, 2)

        if self._mode == 1:
            result = self._pipeline.run_mode1(
                project_name=self._project_name,
                image_path=self._image_path,
                project_config=self._config,
                generate_guidance=self._gen_guidance,
                callback=callback,
                guidance_tensor=self._guidance_tensor,
                guidance_from_source=self._guidance_from_source,
                return_frames=postprocess,
            )
        elif self._mode == 2:
            result = self._pipeline.run_mode2(
                project_name=self._project_name,
                first_frame_path=self._first_frame,
                last_frame_path=self._last_frame,
                project_config=self._config,
                generate_guidance=self._gen_guidance,
                guidance_frame=self._guidance_frame,
                callback=callback,
                guidance_tensor=self._guidance_tensor,
                guidance_from_source=self._guidance_from_source,
                return_frames=postprocess,
            )
        elif self._mode == 3:
            result = self._pipeline.run_mode3(
                project_name=self._project_name,
                video_path=self._video_path,
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                project_config=self._config,
                generate_guidance=self._gen_guidance,
                callback=callback,
            )
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

        logger.info("Pipeline returned, type=%s", type(result).__name__)

        if postprocess:
            raw_frames, preview_path = result
            self.status.emit("Post-processing...")
            self.progress.emit(0.9, "Post-processing...")
            from supremediffusion.core.output_handler import OutputHandler
            np_frames = OutputHandler.frames_to_numpy(raw_frames)
            fps = self._config.fps if hasattr(self._config, "fps") else 16
            processed, output_fps = _run_postprocessing(np_frames, fps, self._config)
            # Preserve audio from LTX pipeline through post-processing
            audio_path = None
            wan = getattr(self._pipeline, "wan", None)
            if wan is not None:
                audio_path = getattr(wan, "last_audio_path", None)
            OutputHandler.save_generation(
                processed, preview_path, fps=output_fps, audio_path=audio_path,
                codec=configured_codec_args(),
            )
            result = preview_path

        # ── Phase 3 unified post-processing pipeline ──────────────────
        # Pass ``state=self`` so PostProcessPipeline's upscale/RIFE steps can
        # read ``self.state.global_config`` via this worker's ``global_config``
        # property. Previously ``getattr(self._pipeline, "_state", None) or self``
        # resolved to the worker but the worker had no ``global_config``, so
        # every quality/master upscale + RIFE threw and was silently swallowed.
        from sdqt.workers._post_phase3 import apply_phase3_postprocessing
        result = apply_phase3_postprocessing(
            result=result,
            config=self._config,
            global_config=self._global_config,
            project_name=self._project_name,
            state=self,
            progress_emit=self.progress.emit,
            status_emit=self.status.emit,
            is_aborted=lambda: self.is_aborted,
        )

        logger.info("Generation complete, result=%s", result)
        self.progress.emit(1.0, "Complete")
        return GenResult(result, resolved_seed)


class VideoExtendWorker(BaseWorker):
    """Run video extension (continuation) on a background thread.

    Emits ``finished_ok`` with a dict containing:
    - ``"video_path"``: path to the encoded extension video
    - ``"last_frame_path"``: lossless PNG of the last generated frame,
      for use as ``start_image_path`` on the next continuation to avoid
      detail loss from re-decoding a compressed video.
    """

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        project_config: ProjectConfig,
        video_path: str,
        *,
        final_frame_path: Optional[str] = None,
        guidance_video_path: Optional[str] = None,
        guidance_image: Any = None,
        start_image_path: Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._config = project_config
        self._video_path = video_path
        self._final_frame_path = final_frame_path
        self._guidance_video_path = guidance_video_path
        self._guidance_image = guidance_image
        self._start_image_path = start_image_path

    def do_work(self) -> dict:
        total_steps = self._config.num_inference_steps or 4
        callback = self.make_step_callback(total_steps)

        self.status.emit("Loading model...")
        self.progress.emit(0.0, "Starting extension...")

        # If a static guidance PIL image was supplied, build the guidance
        # tensor in-memory at the resolution requested by the config — this
        # bypasses the libx264/yuv420p RGB↔YUV roundtrip that ffmpeg would
        # otherwise apply when encoding then decoding a guidance .mp4.
        guidance_tensor = None
        if self._guidance_image is not None:
            try:
                from supremediffusion.core.guidance import GuidanceGenerator
                # ProjectConfig stores frame count as ``video_length`` and the
                # resolution as a "WxH" string — there are no num_frames/width/
                # height fields, so the previous getattr() reads always fell back
                # to defaults (81 frames, the source image's own dimensions).
                num_frames = getattr(self._config, "video_length", 0) or 81
                width, height = self._guidance_image.width, self._guidance_image.height
                resolution = getattr(self._config, "resolution", "") or ""
                if "x" in resolution:
                    try:
                        w_s, h_s = resolution.split("x", 1)
                        width = int(w_s.strip().split()[0])
                        height = int(h_s.strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                guidance_tensor = GuidanceGenerator.build_static_tensor(
                    self._guidance_image, num_frames=num_frames,
                    width=width, height=height,
                )
                logger.info(
                    "Built static guidance tensor in-memory (%d frames @ %dx%d) — no codec roundtrip",
                    num_frames, width, height,
                )
            except Exception:
                logger.warning("Failed to build in-memory guidance tensor; falling back to file path",
                               exc_info=True)

        # Always request raw frames so we can save the last frame losslessly
        result = self._pipeline.run_extend(
            project_name=self._project_name,
            video_path=self._video_path,
            project_config=self._config,
            final_frame_path=self._final_frame_path,
            guidance_video_path=self._guidance_video_path,
            guidance_tensor=guidance_tensor,
            start_image_path=self._start_image_path,
            callback=callback,
            return_frames=True,
        )

        raw_frames, preview_path = result

        # Save the last generated frame as lossless PNG before any encoding.
        # This preserves full detail for the next continuation's start image.
        # Use frames_to_numpy to normalise any tensor/ndarray shape and dtype,
        # then grab the final frame as a standard (H, W, 3) uint8 array.
        last_frame_path = str(Path(preview_path).with_suffix(".last_frame.png"))
        from PIL import Image as _PILImage
        from supremediffusion.core.output_handler import OutputHandler
        try:
            np_all = OutputHandler.frames_to_numpy(raw_frames)
            _PILImage.fromarray(np_all[-1]).save(last_frame_path)
        except Exception:
            logger.warning("Could not save lossless last frame", exc_info=True)
            last_frame_path = None

        # Encode raw frames at native resolution — no post-processing.
        # RIFE / spatial upsampling / grain are deferred to Save Clip so
        # that continuations stitch cleanly (same res/fps) and the model's
        # start frame is never extracted from a post-processed video.
        fps = self._config.fps if hasattr(self._config, "fps") else 16
        OutputHandler.save_generation(
            raw_frames, preview_path, fps=fps, codec=configured_codec_args(),
        )

        self.progress.emit(1.0, "Extension complete")
        return {"video_path": preview_path, "last_frame_path": last_frame_path}


class PostProcessWorker(BaseWorker):
    """Run post-processing (RIFE, spatial upsampling, film grain) on a video.

    Decodes the input video to frames, applies the configured post-processing
    pipeline, and encodes the result to *output_path*.
    Emits ``finished_ok`` with the output path string.
    """

    def __init__(
        self,
        source_path: str,
        output_path: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._output = output_path
        self._config = project_config

    def do_work(self) -> str:
        from supremediffusion.utils.video import extract_frames, probe_video
        from supremediffusion.core.output_handler import OutputHandler
        import tempfile

        info = probe_video(self._source)
        fps = info.get("fps", 16)

        self.progress.emit(0.0, "Extracting frames...")
        with tempfile.TemporaryDirectory(prefix="sd_pp_") as tmpdir:
            frame_paths = extract_frames(self._source, tmpdir)
            from PIL import Image as _PILImage
            frames = np.stack(
                [np.array(_PILImage.open(p).convert("RGB")) for p in frame_paths],
                axis=0,
            )

        self.progress.emit(0.2, "Post-processing...")

        def _pp_progress(frac, desc):
            self.progress.emit(0.2 + frac * 0.6, desc)

        processed, output_fps = _run_postprocessing(
            frames, fps, self._config, progress_cb=_pp_progress,
        )

        self.progress.emit(0.8, "Encoding...")
        OutputHandler.save_generation(
            processed, self._output, fps=output_fps, codec=configured_codec_args(),
        )

        # ── Phase 3 unified pipeline (Quality preset) ──────────────────
        # Applied to the Save Clip output. Color anchor was already done
        # during generation; this adds upscale + RIFE based on quality preset.
        # Resolve a GlobalConfig so the upscale step can locate the model dir;
        # without it the Quality/Master upscale step would have no upscaler_dir
        # to resolve against and would silently no-op.
        try:
            global_config = None
            try:
                from supremediffusion.config.global_config import GlobalConfig
                global_config = GlobalConfig.load()
            except Exception:  # noqa: BLE001
                logger.warning("Phase 3 (Save Clip): could not load GlobalConfig")
            from sdqt.workers._post_phase3 import apply_phase3_postprocessing
            result = apply_phase3_postprocessing(
                result=self._output,
                config=self._config,
                global_config=global_config,
                project_name="",
                state=None,
                progress_emit=lambda f, d: self.progress.emit(0.95 + 0.05 * f, d),
                status_emit=self.status.emit,
                is_aborted=lambda: self.is_aborted,
            )
            if isinstance(result, str) and result != self._output:
                # Phase 3 created a new file — replace the output
                import shutil
                shutil.move(result, self._output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Phase 3 (Save Clip) failed: %s", exc)

        self.progress.emit(1.0, "Post-processing complete")
        return self._output


class LongshotRenderWorker(BaseWorker):
    """Render all pending gapfills for a Longshot session sequentially.

    Emits ``progress`` per gapfill and ``finished_ok`` with the updated
    gapfills list when done.
    """

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        project_config: ProjectConfig,
        gapfills: list[dict],
        ls_dir: Path,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._config = project_config
        self._gapfills = gapfills
        self._ls_dir = ls_dir

    def do_work(self) -> list[dict]:
        pending = [i for i, gf in enumerate(self._gapfills) if gf["status"] == "pending"]
        total = len(pending)
        if total == 0:
            return self._gapfills

        for step, idx in enumerate(pending):
            if self.is_aborted:
                self.status.emit(f"Aborted after {step}/{total} gapfills.")
                break

            gf = self._gapfills[idx]
            gf["status"] = "rendering"
            self.progress.emit(step / total, f"Rendering gapfill {step + 1}/{total}")

            total_steps = self._config.num_inference_steps or 4
            callback = self.make_step_callback(total_steps, f"Gapfill {step + 1}/{total}")

            try:
                postprocess = _needs_postprocessing(self._config)
                result_path = self._pipeline.run_mode2(
                    project_name=self._project_name,
                    first_frame_path=gf["frame_a_path"],
                    last_frame_path=gf["frame_b_path"],
                    project_config=self._config,
                    callback=callback,
                    return_frames=postprocess,
                )
                render_dest = str(self._ls_dir / f"gapfill_{idx:03d}_render.mp4")
                if postprocess:
                    from supremediffusion.core.output_handler import OutputHandler
                    raw_frames, preview_path = result_path
                    self.status.emit(f"Post-processing gapfill {step + 1}/{total}...")
                    np_frames = OutputHandler.frames_to_numpy(raw_frames)
                    fps = self._config.fps if hasattr(self._config, "fps") else 16
                    processed, output_fps = _run_postprocessing(
                        np_frames, fps, self._config,
                    )
                    OutputHandler.save_generation(
                        processed, render_dest, fps=output_fps,
                        codec=configured_codec_args(),
                    )
                elif result_path != render_dest:
                    shutil.copy2(result_path, render_dest)
                gf["render_path"] = render_dest
                gf["status"] = "done"
            except InterruptedError:
                gf["status"] = "pending"
                raise
            except Exception as exc:
                logger.error("Gapfill %d failed: %s", idx, exc, exc_info=True)
                gf["status"] = "error"

        self.progress.emit(1.0, "Render complete")
        return self._gapfills
