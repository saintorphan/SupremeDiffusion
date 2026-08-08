"""FPS-match indicator + fixer for a guidance / control video.

Bound to a :class:`VideoDropWidget`, this compact status bar shows whether the
loaded video's frame rate matches the render target fps:

* green ✓ when the rates align,
* red ✗ + a **Match FPS** button when they don't,
* amber ⚠ if the fps can't be probed.

**Match FPS** re-encodes the video to the target rate (frame duplication / drop
via the ffmpeg ``fps`` filter) on a background thread, then reloads the result
into the source widget — which re-triggers the check and flips it green.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from sdqt.widgets.video_drop import VideoDropWidget
from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class _FpsMatchWorker(BaseWorker):
    """Re-encode *src* to *target_fps* and return the new file path."""

    def __init__(self, src: str, target_fps: int, parent=None) -> None:
        super().__init__(parent)
        self._src = src
        self._target_fps = int(target_fps)

    def do_work(self) -> object:
        from sdqt.utils.codec import configured_codec_args

        src = Path(self._src)
        out = src.with_name(f"{src.stem}_{self._target_fps}fps.mp4")
        # If the source dir isn't writable, drop the result in a temp dir.
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            probe = out.with_suffix(out.suffix + ".tmpcheck")
            probe.touch()
            probe.unlink(missing_ok=True)
        except OSError:
            out = Path(tempfile.gettempdir()) / out.name

        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"fps={self._target_fps}",
            "-r", str(self._target_fps),
            *configured_codec_args(),
            "-an",
            str(out),
        ]
        self.status.emit(f"Re-encoding guidance video to {self._target_fps} fps…")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-600:]
            raise RuntimeError(f"ffmpeg fps match failed: {tail}")
        return str(out)


class GuidanceFpsCheck(QWidget):
    """Compact fps-match status bar bound to a :class:`VideoDropWidget`."""

    fps_matched = Signal(str)  # emitted with the new path after a successful match

    def __init__(
        self,
        video_widget: VideoDropWidget,
        target_fps_getter: Callable[[], int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_widget
        self._target_fps_getter = target_fps_getter
        self._source_fps: Optional[float] = None
        self._worker: Optional[_FpsMatchWorker] = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._status = QLabel("")
        row.addWidget(self._status)
        self._match_btn = QPushButton("Match FPS")
        self._match_btn.setToolTip(
            "Re-encode the guidance video to the render frame rate so its motion "
            "lines up frame-for-frame with the generated output."
        )
        self._match_btn.clicked.connect(self._on_match_clicked)
        self._match_btn.setVisible(False)
        row.addWidget(self._match_btn)
        row.addStretch()

        self._video.file_loaded.connect(self._on_video_loaded)
        self._video.file_cleared.connect(self._on_cleared)
        self.setVisible(False)

    # -- Public API --------------------------------------------------------

    def recheck(self) -> None:
        """Re-probe the current video and refresh the indicator."""
        path = self._video.file_path
        if not path or not Path(path).is_file():
            self._on_cleared()
            return
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            fps = float(info.get("fps", 0) or 0)
            self._source_fps = fps if fps > 0 else None
        except Exception:
            logger.debug("FPS probe failed for %s", path, exc_info=True)
            self._source_fps = None
        self._update_display()

    # -- Signal handlers ---------------------------------------------------

    def _on_video_loaded(self, _path: str) -> None:
        self.recheck()

    def _on_cleared(self) -> None:
        self._source_fps = None
        self.setVisible(False)

    def _update_display(self) -> None:
        target = int(self._target_fps_getter() or 0)
        self.setVisible(True)
        if not self._source_fps or self._source_fps <= 0:
            self._status.setText(
                "<span style='color:#e0a030;'>&#9888; guidance fps unknown</span>"
            )
            self._match_btn.setVisible(False)
            return
        src = round(self._source_fps, 2)
        if target > 0 and abs(self._source_fps - target) < 0.05:
            self._status.setText(
                f"<span style='color:#46c46e;'>&#10003; {src:g} fps matches render</span>"
            )
            self._match_btn.setVisible(False)
        else:
            self._status.setText(
                f"<span style='color:#e05050;'>&#10007; {src:g} fps &#8800; "
                f"{target} fps render</span>"
            )
            self._match_btn.setVisible(True)

    def _on_match_clicked(self) -> None:
        path = self._video.file_path
        if not path or not Path(path).is_file():
            return
        target = int(self._target_fps_getter() or 0)
        if target <= 0:
            return
        self._match_btn.setEnabled(False)
        self._match_btn.setText("Matching…")
        self._worker = _FpsMatchWorker(path, target, self)
        self._worker.finished_ok.connect(self._on_match_done)
        self._worker.error.connect(self._on_match_error)
        self._worker.start()

    def _on_match_done(self, out_path) -> None:
        self._match_btn.setEnabled(True)
        self._match_btn.setText("Match FPS")
        if out_path and Path(str(out_path)).is_file():
            # load_file emits file_loaded → recheck() → flips the indicator green
            self._video.load_file(str(out_path))
            self.fps_matched.emit(str(out_path))

    def _on_match_error(self, msg: str) -> None:
        self._match_btn.setEnabled(True)
        self._match_btn.setText("Match FPS")
        self._status.setText(
            "<span style='color:#e05050;'>&#10007; fps match failed (see log)</span>"
        )
        logger.warning("Match FPS failed: %s", msg)
