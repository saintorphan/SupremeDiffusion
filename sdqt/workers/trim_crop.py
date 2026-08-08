"""Worker threads for video trim and crop operations."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseWorker
from sdqt.utils.codec import pix_fmt_args, configured_codec_args as _codec_args

logger = logging.getLogger(__name__)


class TrimWorker(BaseWorker):
    """Trim a video to a time range."""

    def __init__(
        self,
        source: str,
        output: str,
        start_sec: float,
        end_sec: float,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._start = start_sec
        self._end = end_sec

    def do_work(self) -> str:
        from supremediffusion.utils.video import trim_video_precise

        self.progress.emit(0.0, "Trimming...")
        trim_video_precise(self._source, self._output, self._start, self._end)
        self.progress.emit(1.0, "Trim complete")
        return self._output


class ColorCorrectWorker(BaseWorker):
    """Apply color match to a video using a reference image.

    When ``method`` is set, dispatches through the unified color-matcher
    registry (mkl/hm/reinhard/mvgd/hm-mvgd-hm/hm-mkl-hm) — same code path
    as the in-pipeline auto-match. ``histogram_match`` is the legacy flag
    used when ``method`` is None.
    """

    def __init__(
        self,
        source: str,
        reference: str,
        output: str,
        strength: float = 1.0,
        histogram_match: bool = False,
        method: str | None = None,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._reference = reference
        self._output = output
        self._strength = strength
        self._histogram_match = histogram_match
        self._method = method

    def do_work(self) -> str:
        from supremediffusion.utils.color_correct import color_correct_video

        def _cb(frac, desc):
            self.progress.emit(frac, f"Color correcting... {desc}")

        self.progress.emit(0.0, "Color correcting...")
        color_correct_video(
            self._source, self._reference, self._output,
            self._strength, _cb,
            histogram_match=self._histogram_match,
            method=self._method,
        )
        self.progress.emit(1.0, "Color correction complete")
        return self._output


class BrightnessContrastWorker(BaseWorker):
    """Adjust brightness/contrast — manual sliders or reference matching."""

    def __init__(
        self,
        source: str,
        output: str,
        *,
        brightness: float = 1.0,
        contrast: float = 1.0,
        reference: str | None = None,
        strength: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._brightness = brightness
        self._contrast = contrast
        self._reference = reference
        self._strength = strength

    def do_work(self) -> str:
        from supremediffusion.utils.brightness_contrast import brightness_contrast_video

        def _cb(frac, desc):
            self.progress.emit(frac, f"Brightness/contrast... {desc}")

        self.progress.emit(0.0, "Correcting brightness/contrast...")
        brightness_contrast_video(
            self._source, self._output,
            brightness=self._brightness,
            contrast=self._contrast,
            reference_path=self._reference,
            strength=self._strength,
            progress_callback=_cb,
        )
        self.progress.emit(1.0, "Brightness/contrast correction complete")
        return self._output


class HueSaturationWorker(BaseWorker):
    """Adjust hue/saturation — manual sliders or reference matching."""

    def __init__(
        self,
        source: str,
        output: str,
        *,
        hue_shift: float = 0.0,
        saturation: float = 1.0,
        reference: str | None = None,
        strength: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._hue_shift = hue_shift
        self._saturation = saturation
        self._reference = reference
        self._strength = strength

    def do_work(self) -> str:
        from supremediffusion.utils.hue_saturation import hue_saturation_video

        def _cb(frac, desc):
            self.progress.emit(frac, f"Hue/saturation... {desc}")

        self.progress.emit(0.0, "Adjusting hue/saturation...")
        hue_saturation_video(
            self._source, self._output,
            hue_shift=self._hue_shift,
            saturation=self._saturation,
            reference_path=self._reference,
            strength=self._strength,
            progress_callback=_cb,
        )
        self.progress.emit(1.0, "Hue/saturation adjustment complete")
        return self._output


class SpeedWorker(BaseWorker):
    """Change playback speed or reverse a video via FFmpeg."""

    def __init__(
        self,
        source: str,
        output: str,
        speed: float,
        *,
        reverse: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._speed = speed
        self._reverse = reverse

    def do_work(self) -> str:
        import subprocess

        if self._reverse:
            return self._do_reverse()
        return self._do_speed()

    def _do_reverse(self) -> str:
        import subprocess

        self.progress.emit(0.0, "Reversing...")
        cmd = [
            "ffmpeg", "-y", "-i", self._source,
            "-vf", "scale=in_range=full:out_range=full,reverse,setpts=PTS-STARTPTS",
            "-af", "areverse,asetpts=PTS-STARTPTS",
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            self._output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:] if result.stderr else "ffmpeg reverse failed")
        self.progress.emit(1.0, "Reverse complete")
        return self._output

    def _do_speed(self) -> str:
        import subprocess

        self.progress.emit(0.0, f"Applying {self._speed:.2f}x speed...")
        pts_factor = 1.0 / self._speed

        # Build atempo chain (ffmpeg atempo range is 0.5–100.0, chain for < 0.5)
        atempo_filters: list[str] = []
        remaining = self._speed
        while remaining < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining /= 0.5
        while remaining > 100.0:
            atempo_filters.append("atempo=100.0")
            remaining /= 100.0
        atempo_filters.append(f"atempo={remaining:.6f}")

        vf = f"scale=in_range=full:out_range=full,setpts={pts_factor:.6f}*PTS"
        af = ",".join(atempo_filters)

        cmd = [
            "ffmpeg", "-y", "-i", self._source,
            "-vf", vf, "-af", af,
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            self._output,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:] if result.stderr else "ffmpeg failed")

        self.progress.emit(1.0, "Speed change complete")
        return self._output


class CropWorker(BaseWorker):
    """Crop a video to specified dimensions."""

    def __init__(
        self,
        source: str,
        output: str,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def do_work(self) -> str:
        from supremediffusion.utils.video import crop_video

        self.progress.emit(0.0, "Cropping...")
        crop_video(self._source, self._output, self._x, self._y, self._w, self._h)
        self.progress.emit(1.0, "Crop complete")
        return self._output
