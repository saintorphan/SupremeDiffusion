"""Worker thread for Video-Retalking lip sync inference."""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

from .base import BaseWorker
from sdqt.utils.codec import pix_fmt_args, configured_codec_args as _codec_args

logger = logging.getLogger(__name__)

_RETALKING_DIR = Path("~/video-retalking").expanduser()
_FACE_PAD_RATIO = 0.35


class VideoRetalkingWorker(BaseWorker):
    """Run Video-Retalking lip sync as a subprocess.

    Video-Retalking runs in its own environment but uses much less VRAM
    than LatentSync (~4-5 GB) and handles non-frontal faces better.

    Steps:
    1. Crop video to padded face region
    2. Run Video-Retalking on cropped video + audio
    3. Composite synced crop back onto original video
    4. Mux audio
    """

    def __init__(
        self,
        video_path: str,
        audio_path: str,
        face_bbox: list[int],
        output_path: str,
        *,
        expression: str = "neutral",
        up_face: str = "original",
        img_size: int = 384,
        pads: list[int] | None = None,
        nosmooth: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_path
        self._audio = audio_path
        self._bbox = face_bbox
        self._output = output_path
        self._expression = expression
        self._up_face = up_face
        self._img_size = img_size
        self._pads = pads or [0, 20, 0, 0]
        self._nosmooth = nosmooth
        self._proc: subprocess.Popen | None = None

    def abort(self) -> None:
        super().abort()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def do_work(self) -> str:
        if not _RETALKING_DIR.is_dir():
            raise RuntimeError(
                "Video-Retalking not found. Clone to ~/video-retalking:\n"
                "git clone https://github.com/OpenTalker/video-retalking ~/video-retalking"
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="retalking_"))

        # 1. Compute crop
        self.progress.emit(0.0, "Cropping video to face region...")
        crop_rect, video_w, video_h = self._compute_crop(self._video, self._bbox)
        cx, cy, cw, ch = crop_rect

        cropped_video = str(tmp_dir / "cropped.mp4")
        self._run_ffmpeg([
            "ffmpeg", "-y", "-i", self._video,
            "-vf", f"scale=in_range=full:out_range=full,crop={cw}:{ch}:{cx}:{cy}",
            *_codec_args(),
            *pix_fmt_args(),
            "-an", cropped_video,
        ])
        if self.is_aborted:
            raise InterruptedError("Aborted")

        # 2. Run Video-Retalking
        self.progress.emit(0.10, "Running Video-Retalking...")
        synced_crop = str(tmp_dir / "synced_crop.mp4")
        self._run_retalking(cropped_video, self._audio, synced_crop)
        if self.is_aborted:
            raise InterruptedError("Aborted")

        if not Path(synced_crop).is_file():
            raise RuntimeError("Video-Retalking produced no output")

        # 3. Composite back
        self.progress.emit(0.85, "Compositing result...")
        composited = str(tmp_dir / "composited.mp4")
        self._run_ffmpeg([
            "ffmpeg", "-y",
            "-i", self._video,
            "-i", synced_crop,
            "-filter_complex",
            f"[1:v]scale={cw}:{ch}:flags=lanczos[crop];[0:v][crop]overlay={cx}:{cy}",
            *_codec_args(),
            "-an", composited,
        ])
        if self.is_aborted:
            raise InterruptedError("Aborted")

        # 4. Mux audio
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
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", video_path],
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
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._proc = proc
        _, stderr = proc.communicate()
        self._proc = None
        if self.is_aborted:
            return
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='replace')[-500:]}")

    def _run_retalking(self, video_path: str, audio_path: str, output_path: str) -> None:
        """Run Video-Retalking inference."""
        # Find python in the retalking venv, or use system python
        venv_python = _RETALKING_DIR / "venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = _RETALKING_DIR / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = Path("python3")

        inference_script = _RETALKING_DIR / "inference.py"
        if not inference_script.is_file():
            raise RuntimeError(
                f"inference.py not found in {_RETALKING_DIR}. "
                "Ensure Video-Retalking is properly installed."
            )

        # Ensure temp dir exists (Video-Retalking uses relative temp/ paths)
        (_RETALKING_DIR / "temp").mkdir(exist_ok=True)

        # Pre-convert audio to wav in a safe path (no spaces/special chars)
        safe_audio = str(Path(tempfile.mkdtemp(prefix="vr_")) / "audio.wav")
        self._run_ffmpeg([
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", safe_audio,
        ])

        cmd = [
            str(venv_python),
            str(inference_script),
            "--face", video_path,
            "--audio", safe_audio,
            "--outfile", output_path,
            "--exp_img", self._expression,
            "--up_face", self._up_face,
            "--img_size", str(self._img_size),
            "--pads", *[str(p) for p in self._pads],
        ]
        if self._nosmooth:
            cmd.append("--nosmooth")

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_RETALKING_DIR),
            env=env,
            text=True,
        )
        self._proc = proc

        output_lines: list[str] = []
        step_re = re.compile(r"(\d+)/(\d+)")
        for line in proc.stdout:
            output_lines.append(line.rstrip())
            logger.debug("VideoRetalking: %s", line.rstrip())
            if self.is_aborted:
                proc.terminate()
                proc.wait()
                self._proc = None
                return
            m = step_re.search(line)
            if m:
                current, total = int(m.group(1)), int(m.group(2))
                frac = 0.10 + 0.75 * (current / max(total, 1))
                self.progress.emit(frac, f"Video-Retalking {current}/{total}")

        proc.wait()
        self._proc = None

        if self.is_aborted:
            return
        if proc.returncode != 0:
            tail = "\n".join(output_lines[-15:])
            logger.error("Video-Retalking failed (rc=%d):\n%s", proc.returncode, tail)
            error_msg = "Video-Retalking inference failed"
            for line in reversed(output_lines):
                if "Error" in line or "error" in line or "Exception" in line:
                    error_msg = line.strip()[:200]
                    break
            raise RuntimeError(error_msg)
