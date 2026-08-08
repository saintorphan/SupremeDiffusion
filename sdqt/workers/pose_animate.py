"""Pose-driven body animation worker (MimicMotion)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sdqt.workers.base import BaseWorker
from sdqt.utils.codec import configured_codec_args

logger = logging.getLogger(__name__)


class PoseAnimateWorker(BaseWorker):
    """Runs MimicMotion: extract pose from driver video, render frames from ref image."""

    def __init__(
        self,
        state: Any,
        project_name: str,
        reference_image_path: str,
        driving_video_path: str,
        *,
        audio_path: str | None = None,
        resolution: int = 576,
        num_inference_steps: int = 25,
        min_guidance: float = 1.0,
        max_guidance: float = 3.0,
        noise_aug_strength: float = 0.02,
        seed: int = -1,
        fps: int = 24,
        output_path: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._project_name = project_name
        self._reference = reference_image_path
        self._driving = driving_video_path
        self._audio = audio_path
        self._resolution = int(resolution)
        self._steps = int(num_inference_steps)
        self._min_g = float(min_guidance)
        self._max_g = float(max_guidance)
        self._noise_aug = float(noise_aug_strength)
        self._seed = int(seed)
        self._fps = int(fps)
        self._output_path = output_path

    # ------------------------------------------------------------------
    def _make_callback(self):
        """Diffusers callback_on_step_end signature: (pipe, step, timestep, kwargs)."""
        total = self._steps

        def _cb(_pipe, step_index, _timestep, cb_kwargs):
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            frac = (step_index + 1) / max(1, total)
            self.progress.emit(frac, f"Step {step_index + 1}/{total}")
            return cb_kwargs

        return _cb

    # ------------------------------------------------------------------
    def do_work(self) -> str:
        if not self._reference or not os.path.isfile(self._reference):
            raise RuntimeError(f"Reference image not found: {self._reference}")
        if not self._driving or not os.path.isfile(self._driving):
            raise RuntimeError(f"Driving video not found: {self._driving}")

        # Load pipeline (auto-unloads other GPU pipelines).
        self.status.emit("Loading MimicMotion pipeline...")
        self.progress.emit(0.0, "Loading models")
        if self._state.mimicmotion_pipeline is None:
            self._state.load_mimicmotion_pipelines()
        pipeline = self._state.mimicmotion_pipeline
        if pipeline is None:
            raise RuntimeError("MimicMotion pipeline failed to initialise.")

        # First call also runs preprocess (DWPose extraction) — emit a status
        # so the user knows the long pause isn't a hang.
        self.status.emit("Extracting DWPose skeleton from driver video...")

        t0 = time.time()
        frames = pipeline.generate(
            reference_image_path=self._reference,
            driving_video_path=self._driving,
            resolution=self._resolution,
            num_inference_steps=self._steps,
            min_guidance=self._min_g,
            max_guidance=self._max_g,
            noise_aug_strength=self._noise_aug,
            seed=self._seed,
            fps=7,  # SVD's training fps — for the model; mp4 fps is separate
            callback=self._make_callback(),
        )

        elapsed = time.time() - t0
        self.status.emit(f"Inference done in {elapsed:.1f}s, encoding mp4...")
        self.progress.emit(0.95, "Encoding mp4")

        # Two-step save: torchvision libx264 → re-encode with the project's
        # configured codec so timeline/preview plays in the same colorspace.
        out_path = self._output_path
        if not out_path:
            project_root = self._state.project_manager.get_project_path(self._project_name)
            outputs = Path(project_root) / "outputs"
            outputs.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = str(outputs / f"mimicmotion_{ts}.mp4")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            from supremediffusion.models.mimicmotion_pipeline import MimicMotionPipeline
            MimicMotionPipeline.save_mp4(frames, tmp_path, fps=self._fps)

            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", tmp_path,
                *configured_codec_args(),
                "-pix_fmt", "yuv420p",
                out_path,
            ]
            self._run_ffmpeg(cmd)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # If no audio supplied, MimicMotion render is the final output.
        if not self._audio or not os.path.isfile(self._audio):
            self.progress.emit(1.0, "Complete")
            return out_path

        # ── Lip sync post-step: face-detect → LatentSync → composite → mux ─
        self.status.emit("Lip sync: detecting face in pose-animation output...")
        self.progress.emit(0.50, "Face detect for lip sync")

        bbox = self._detect_face_bbox(out_path)
        if bbox is None:
            # No face found — skip lip sync, return body-only output.
            self.status.emit(
                "No face detected in MimicMotion output — skipping lip sync, returning body-only render."
            )
            self.progress.emit(1.0, "Complete (lip sync skipped)")
            return out_path

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # Build the lip-synced output path alongside the body-only render.
        ls_out = out_path[:-4] + "_lipsync.mp4"
        self.status.emit("Lip sync: running LatentSync (loads pipeline, unloads MimicMotion)...")
        self.progress.emit(0.55, "Loading LatentSync")

        # Re-use the existing LipSyncWorker logic by instantiating it inline and
        # driving its do_work() in this thread. AppState handles the
        # pipeline swap (MimicMotion → LatentSync) so VRAM stays inside 12 GB.
        from sdqt.workers.lip_sync import LipSyncWorker
        inner = LipSyncWorker(
            video_path=out_path,
            audio_path=self._audio,
            face_bbox=bbox,
            output_path=ls_out,
            state=self._state,
            inference_steps=20,
            guidance_scale=1.5,
        )
        # Forward inner progress into our own outer band [0.55, 1.0]
        def _forward(f: float, d: str) -> None:
            mapped = 0.55 + f * 0.45
            self.progress.emit(mapped, d)
        inner.progress.connect(_forward)
        inner.status.connect(self.status.emit)
        # Run synchronously on this thread (we ARE the worker thread).
        result_path = inner.do_work()
        self.progress.emit(1.0, "Complete with lip sync")
        return result_path

    # ------------------------------------------------------------------
    def _detect_face_bbox(self, video_path: str) -> list[int] | None:
        """Detect the largest face in frame 0 of a video. Returns [x1,y1,x2,y2] or None."""
        import io
        import sys
        from supremediffusion.models.face_swap import FaceSwapPipeline
        # Extract first frame
        frame_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                 "-vframes", "1", "-q:v", "2", frame_path],
                check=True,
            )
            models_dir = self._state.global_config.model_paths.get("face_models_dir", "")
            _orig = sys.stdout
            sys.stdout = io.StringIO()
            try:
                pipe = FaceSwapPipeline(models_dir)
                faces = pipe.detect_faces(frame_path)
            finally:
                sys.stdout = _orig
                try:
                    pipe.release()
                except Exception:  # noqa: BLE001
                    pass
            if not faces:
                return None
            # Pick the largest detected face.
            best = None
            best_area = 0
            for f in faces:
                bbox = f.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best = [int(x1), int(y1), int(x2), int(y2)]
            return best
        finally:
            try:
                os.unlink(frame_path)
            except OSError:
                pass
