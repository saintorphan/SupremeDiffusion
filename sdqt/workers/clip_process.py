"""Background worker for per-clip ffmpeg operations (speed, reverse, rotate)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .base import BaseWorker

logger = logging.getLogger(__name__)


class ClipProcessWorker(BaseWorker):
    """Run an ffmpeg command on a clip in the background.

    Returns (output_path, duration) on success.
    """

    def __init__(
        self,
        cmd: list[str],
        output_path: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._cmd = cmd
        self._output_path = output_path

    def do_work(self) -> tuple[str, float]:
        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.status.emit("Processing clip...")

        import time
        import tempfile
        # stderr goes to a real file, not a pipe — nothing drains a pipe while
        # we poll, so ffmpeg would block mid-write once the 64KB buffer fills
        # and poll() would spin forever (hang on long clips).
        with tempfile.TemporaryFile() as stderr_buf:
            proc = subprocess.Popen(
                self._cmd, stdout=subprocess.DEVNULL, stderr=stderr_buf,
            )
            while proc.poll() is None:
                if self.is_aborted:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise InterruptedError("Aborted")
                time.sleep(0.3)

            if proc.returncode != 0:
                stderr_buf.seek(0)
                stderr = stderr_buf.read().decode("utf-8", errors="replace")[-500:]
                raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr}")

        out = Path(self._output_path)
        if not out.is_file() or out.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no output")

        # Probe duration
        dur = self._probe_duration(self._output_path)
        return (self._output_path, dur)

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries",
                 "format=duration", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 5.0
