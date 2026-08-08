"""Worker threads for lip sync face detection and LatentSync inference."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from .base import BaseWorker
from sdqt.utils.codec import pix_fmt_args, configured_codec_args as _codec_args

logger = logging.getLogger(__name__)

# Padding ratio around detected face bbox for crop
_FACE_PAD_RATIO = 0.35


class VideoFaceDetectWorker(BaseWorker):
    """Detect all faces in the first frame of a video (or an image).

    Returns list of ``{"bbox": [x1,y1,x2,y2], "thumbnail": PIL.Image}``.
    """

    def __init__(
        self,
        source_path: str,
        models_dir: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._models_dir = models_dir

    def do_work(self) -> list[dict]:
        import io
        import sys

        from PIL import Image

        self.status.emit("Extracting first frame...")
        self.progress.emit(0.0, "Extracting first frame...")

        # Extract first frame via ffmpeg
        frame_path = self._extract_first_frame(self._source)

        self.status.emit("Detecting faces...")
        self.progress.emit(0.3, "Loading face analyser...")

        from supremediffusion.models.face_swap import FaceSwapPipeline

        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipe = FaceSwapPipeline(self._models_dir)
        finally:
            sys.stdout = _orig_stdout
        try:
            raw_faces = pipe.detect_faces(frame_path)
        finally:
            pipe.release()

        self.progress.emit(0.8, f"Found {len(raw_faces)} face(s)")

        # Build result with bbox and thumbnail
        frame_img = Image.open(frame_path)
        results = []
        for face in raw_faces:
            crop_img = face.get("crop_image")
            bbox = face.get("bbox")
            if bbox is None and crop_img is not None:
                continue
            results.append({
                "index": face.get("index", len(results)),
                "bbox": [int(c) for c in bbox] if bbox else None,
                "thumbnail": crop_img,
            })

        self.progress.emit(1.0, f"Found {len(results)} face(s)")
        return results

    @staticmethod
    def _extract_first_frame(video_path: str) -> str:
        """Extract the first frame from a video as a PNG temp file."""
        p = Path(video_path)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            return video_path

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vframes", "1", "-q:v", "2",
                tmp.name,
            ],
            capture_output=True,
            check=True,
        )
        return tmp.name


class LipSyncWorker(BaseWorker):
    """Run LatentSync lip sync natively with face crop/composite pipeline.

    Steps:
    1. Load LatentSync pipeline (unloads other pipelines to free VRAM)
    2. Crop video to padded face region
    3. Run LatentSync inference on cropped video + audio
    4. Composite synced crop back onto original video
    5. Mux audio onto composited video
    """

    def __init__(
        self,
        video_path: str,
        audio_path: str,
        face_bbox: list[int],
        output_path: str,
        state,
        *,
        inference_steps: int = 20,
        guidance_scale: float = 1.5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_path
        self._audio = audio_path
        self._bbox = face_bbox
        self._output = output_path
        self._state = state
        self._steps = inference_steps
        self._guidance = guidance_scale

    def do_work(self) -> str:
        tmp_dir = Path(tempfile.mkdtemp(prefix="lipsync_"))

        # -- 1. Load LatentSync (unloads video/SD/FLUX pipelines) ----------
        self.status.emit("Loading LatentSync pipeline...")
        self.progress.emit(0.0, "Loading LatentSync pipeline...")

        pipeline = self._state.load_lipsync_pipeline()

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # -- 2. Compute padded crop region --------------------------------
        self.status.emit("Cropping video to face region...")
        self.progress.emit(0.05, "Cropping video to face region...")

        crop_rect, video_w, video_h = self._compute_crop(self._video, self._bbox)
        cx, cy, cw, ch = crop_rect

        _LS_SIZE = 512
        cropped_video = str(tmp_dir / "cropped.mkv")
        self._run_ffmpeg([
            "ffmpeg", "-y", "-i", self._video,
            "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={_LS_SIZE}:{_LS_SIZE}:flags=lanczos",
            "-c:v", "ffv1", "-pix_fmt", "bgr0",
            "-an", cropped_video,
        ])
        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # -- 3. Run LatentSync natively -----------------------------------
        self.status.emit("Running LatentSync...")
        self.progress.emit(0.15, "Running LatentSync...")

        synced_crop = str(tmp_dir / "synced_crop.mp4")

        def _progress_cb(frac: float, desc: str) -> None:
            # Map LatentSync progress (0-1) to our range (0.15-0.85)
            mapped = 0.15 + frac * 0.70
            self.progress.emit(mapped, desc)

        pipeline.run(
            video_path=cropped_video,
            audio_path=self._audio,
            output_path=synced_crop,
            inference_steps=self._steps,
            guidance_scale=self._guidance,
            progress_callback=_progress_cb,
        )

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        if not Path(synced_crop).is_file():
            raise RuntimeError("LatentSync produced no output")

        # -- 4. Composite synced crop back onto original ------------------
        self.status.emit("Compositing result...")
        self.progress.emit(0.85, "Compositing result...")

        composited = str(tmp_dir / "composited.mp4")
        self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", self._video,
            "-i", synced_crop,
            "-filter_complex",
            (f"[1:v]scale={cw}:{ch}:flags=lanczos,"
             f"scale=in_range=full:out_range=full[crop];"
             f"[0:v]scale=in_range=full:out_range=full[bg];"
             f"[bg][crop]overlay={cx}:{cy}"),
            *_codec_args(),
            *pix_fmt_args(),
            "-an", composited,
        ])
        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # -- 5. Mux audio onto composited video ---------------------------
        self.status.emit("Adding audio...")
        self.progress.emit(0.92, "Adding audio...")

        dur_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", composited],
            capture_output=True, text=True,
        )
        vid_dur = dur_result.stdout.strip()

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", composited,
            "-i", self._audio,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        ]
        if vid_dur:
            mux_cmd += ["-t", vid_dur]
        else:
            mux_cmd += ["-shortest"]
        mux_cmd.append(self._output)
        self._run_ffmpeg(mux_cmd)

        self.progress.emit(1.0, "Done")
        return self._output

    def _compute_crop(
        self, video_path: str, bbox: list[int],
    ) -> tuple[tuple[int, int, int, int], int, int]:
        """Compute a padded crop rectangle that fits within the video frame."""
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        dims = result.stdout.strip().split("x")
        video_w, video_h = int(dims[0]), int(dims[1])

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * _FACE_PAD_RATIO)
        pad_y = int(bh * _FACE_PAD_RATIO)

        cx = max(0, x1 - pad_x)
        cy = max(0, y1 - pad_y)
        cx2 = min(video_w, x2 + pad_x)
        cy2 = min(video_h, y2 + pad_y)

        cw = (cx2 - cx) & ~1
        ch = (cy2 - cy) & ~1

        return (cx, cy, cw, ch), video_w, video_h

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run an ffmpeg command, checking for abort."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        _, stderr = proc.communicate()
        if self.is_aborted:
            return
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='replace')[-500:]}")
