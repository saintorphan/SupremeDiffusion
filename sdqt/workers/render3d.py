"""Worker for rendering 3D animation frame sequences + ffmpeg assembly."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter

from sdqt.workers.base import BaseWorker
from sdqt.utils.codec import (
    configured_codec_args as _codec_args,
    configured_encoder_name as _enc_name,
    pix_fmt_args as _pix_fmt,
)

logger = logging.getLogger(__name__)


class FrameSequenceWorker(BaseWorker):
    """Render an animation to a sequence of PNG frames, then assemble to video.

    Runs on a background thread. The OpenGL rendering must happen on the main
    thread (Qt requirement), so this worker receives pre-rendered QImages via
    a callback pattern: the main thread evaluates each frame and passes the
    image to us for compositing + saving.

    Actually, since we can't call OpenGL from a thread, this worker does the
    compositing and ffmpeg assembly. Frame rendering is driven by a QTimer on
    the main thread that feeds frames into the output directory.
    """

    def __init__(
        self,
        frames_dir: str,
        output_path: str,
        total_frames: int,
        fps: int = 24,
        background_path: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._frames_dir = Path(frames_dir)
        self._output_path = output_path
        self._total_frames = total_frames
        self._fps = fps
        self._bg_path = background_path

    def do_work(self) -> str:
        """Wait for all frames to be rendered, then assemble with ffmpeg."""
        # Frames are rendered by the main thread timer.
        # We just assemble the final video once all frames exist.
        self.status.emit("Waiting for frame rendering to complete...")

        # Poll for frames (they're written by the main thread)
        import time
        while True:
            if self.is_aborted:
                raise InterruptedError("Aborted")
            existing = sorted(self._frames_dir.glob("frame_*.png"))
            if len(existing) >= self._total_frames:
                break
            self.progress.emit(
                len(existing) / max(1, self._total_frames),
                f"Rendered {len(existing)}/{self._total_frames} frames",
            )
            time.sleep(0.1)

        # Assemble to video with ffmpeg
        self.status.emit("Assembling video with ffmpeg...")
        self.progress.emit(0.95, "Assembling video...")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self._fps),
            "-i", str(self._frames_dir / "frame_%05d.png"),
            *_codec_args(),
            *_pix_fmt(_enc_name()),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=full",
            self._output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

        self.progress.emit(1.0, "Complete")
        return self._output_path


def composite_frame(
    render: QImage,
    background_path: str | None,
) -> QImage:
    """Composite a 3D render (with alpha) over a background image.

    Call this on the main thread for each rendered frame before saving.
    """
    if not background_path or not Path(background_path).is_file():
        return render

    bg = QImage(background_path)
    if bg.isNull():
        return render

    bg = bg.scaled(render.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                   Qt.TransformationMode.SmoothTransformation)
    painter = QPainter(bg)
    painter.drawImage(0, 0, render)
    painter.end()
    return bg
