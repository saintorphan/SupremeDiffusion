"""GL-based timeline export — renders each frame through the OpenGL pipeline.

Produces pixel-identical output to the GL preview. Runs on the main thread
(OpenGL requirement) using a QTimer to step frame-by-frame, piping raw RGB
to an ffmpeg subprocess for encoding.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from sdqt.utils.codec import (
    get_codec_args, configured_codec_args, pix_fmt_args, full_range_filter,
)

logger = logging.getLogger(__name__)


class GLExportController(QObject):
    """Drives GL frame-by-frame export on the main thread.

    Signals:
        progress(float, str)  — 0-1 fraction + description
        finished(str)         — output path on success
        error(str)            — error message
    """

    progress = Signal(float, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        gl_preview,
        output: str,
        width: int,
        height: int,
        fps: float = 25.0,
        codec: str = "",
        rife_mode: str = "",
        audio_tracks: list | None = None,
        godot_mode: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._gl = gl_preview
        self._output = output
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._rife_mode = rife_mode
        self._audio_tracks = audio_tracks or []
        self._godot_mode = godot_mode

        self._frame_idx = 0
        self._total_frames = 0
        self._duration = 0.0
        self._proc: subprocess.Popen | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(0)  # as fast as possible
        self._timer.timeout.connect(self._render_next_frame)
        self._aborted = False
        self._encode_path = ""  # path ffmpeg writes to (may be temp if RIFE)

    def start(self) -> None:
        """Begin the export."""
        self._duration = self._gl.duration
        if self._duration <= 0:
            self.error.emit("No timeline content to export")
            return

        self._total_frames = int(self._duration * self._fps)
        self._frame_idx = 0

        # If RIFE, encode to temp first
        if self._rife_mode:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="gl_exp_")
            tmp.close()
            self._encode_path = tmp.name
        else:
            self._encode_path = self._output

        Path(self._encode_path).parent.mkdir(parents=True, exist_ok=True)

        # Start ffmpeg with raw RGB input on stdin
        codec_args = get_codec_args(self._codec) if self._codec else configured_codec_args()
        # Extract encoder name for pix_fmt selection
        enc_name = ""
        for i, a in enumerate(codec_args):
            if a == "-c:v" and i + 1 < len(codec_args):
                enc_name = codec_args[i + 1]
                break
        if self._godot_mode:
            # Force the bt709_limited / Godot profile for this run regardless
            # of the project's setting — same as ClipPostProcessWorker handles
            # the legacy godot_mode flag.
            from supremediffusion.config.color_profile import get_profile
            color_args = get_profile("bt709_limited").encoder_args(enc_name)
        else:
            color_args = pix_fmt_args(enc_name)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-vf", full_range_filter(),
            *codec_args,
            *color_args,
            self._encode_path,
        ]

        logger.info("GL export cmd: %s", " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception as exc:
            self.error.emit(f"Failed to start ffmpeg: {exc}")
            return

        self._timer.start()

    def abort(self) -> None:
        if self._aborted:
            return
        self._aborted = True
        self._timer.stop()
        self._cleanup_proc()
        logger.warning("GL export aborted at frame %d/%d", self._frame_idx, self._total_frames)
        self.error.emit("Export aborted")

    def _render_next_frame(self) -> None:
        if self._aborted or self._proc is None:
            self._timer.stop()
            return

        if self._frame_idx >= self._total_frames:
            self._timer.stop()
            self._finish()
            return

        position = self._frame_idx / self._fps

        # Render frame through GL pipeline
        frame_data = self._gl.render_frame_for_export(
            position, self._width, self._height,
        )

        if frame_data is None:
            # Past the end of content — stop instead of writing black
            self._total_frames = self._frame_idx
            self._timer.stop()
            self._finish()
            return

        try:
            self._proc.stdin.write(frame_data)
        except (BrokenPipeError, OSError) as exc:
            self._timer.stop()
            stderr = self._read_stderr()
            self.error.emit(f"ffmpeg pipe error: {exc}\n{stderr}")
            self._cleanup_proc()
            return

        self._frame_idx += 1
        if self._frame_idx % 10 == 0 or self._frame_idx >= self._total_frames:
            frac = self._frame_idx / self._total_frames
            self.progress.emit(
                frac * (0.9 if self._rife_mode else 1.0),
                f"GL render: frame {self._frame_idx}/{self._total_frames}",
            )

    def _finish(self) -> None:
        """Close ffmpeg pipe and handle RIFE post-processing."""
        if self._proc is None:
            return

        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=120)
        except Exception as exc:
            stderr = self._read_stderr()
            self.error.emit(f"ffmpeg finalize error: {exc}\n{stderr}")
            self._cleanup_proc()
            return

        if self._proc.returncode != 0:
            stderr = self._read_stderr()
            self.error.emit(f"ffmpeg failed (rc={self._proc.returncode})\n{stderr[-500:]}")
            self._cleanup_proc()
            return

        self._proc = None

        # Audio mux: combine audio from source clips into the rendered video
        if self._audio_tracks:
            self._mux_audio()

        # RIFE post-processing
        if self._rife_mode:
            self.progress.emit(0.9, f"RIFE {self._rife_mode} upsampling...")
            try:
                from sdqt.workers.timeline import TimelineExportWorker
                worker = TimelineExportWorker.__new__(TimelineExportWorker)
                worker._rife_mode = self._rife_mode
                worker._output = self._output
                rife_out = worker._apply_rife(self._encode_path)
                if rife_out != self._output:
                    import shutil
                    shutil.move(rife_out, self._output)
            except Exception as exc:
                self.error.emit(f"RIFE failed: {exc}")
                return

        self.progress.emit(1.0, "Export complete")
        self.finished.emit(self._output)

    def _mux_audio(self) -> None:
        """Mux audio from source clips into the rendered video."""
        # Find the first clip with audio
        audio_source = None
        audio_offset = 0.0
        for track_info in self._audio_tracks:
            path = track_info.get("path", "")
            if path and Path(path).is_file():
                audio_source = path
                audio_offset = track_info.get("media_offset", 0.0)
                break

        if not audio_source:
            return

        import shutil
        tmp_muxed = self._encode_path + ".muxed.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", self._encode_path,
            "-i", audio_source,
        ]
        if audio_offset > 0:
            cmd.extend(["-ss", str(audio_offset)])
        cmd.extend([
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            tmp_muxed,
        ])
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                shutil.move(tmp_muxed, self._encode_path)
                logger.info("Audio muxed from %s into export", audio_source)
            else:
                logger.warning("Audio mux failed: %s", result.stderr.decode(errors="replace")[:200])
                Path(tmp_muxed).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Audio mux error: %s", exc)
            Path(tmp_muxed).unlink(missing_ok=True)

    def _read_stderr(self) -> str:
        if self._proc and self._proc.stderr:
            try:
                return self._proc.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                pass
        return ""

    def _cleanup_proc(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
