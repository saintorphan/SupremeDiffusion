"""Clip Color Correct worker — match-grade a clip against a reference image/video."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from .base import BaseWorker
from sdqt.utils.codec import (
    configured_codec_args as _codec_args,
    configured_encoder_name as _enc_name,
    pix_fmt_args as _pix_fmt,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ClipColorCorrectWorker(BaseWorker):
    """Generate a 3D LUT from a reference image/video and apply it to a source clip.

    - Extracts one representative frame from the reference (first frame of a
      video, or the image itself if the reference is a still).
    - Extracts a representative frame from the source clip (near the middle).
    - Runs ``generate_lut3d(ref_frame, target_frame)`` → .cube file.
    - Applies the LUT via ffmpeg's ``lut3d`` filter, preserving audio.
    """

    def __init__(
        self,
        source: str,
        output: str,
        *,
        reference: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._reference = reference

    # ------------------------------------------------------------------

    def do_work(self) -> str:
        if not Path(self._source).is_file():
            raise RuntimeError(f"Source not found: {self._source}")
        if not self._reference or not Path(self._reference).is_file():
            raise RuntimeError("Color-correct reference missing")

        self.progress.emit(0.10, "Extracting reference frame…")
        ref_frame = self._load_reference_frame()
        if ref_frame is None:
            raise RuntimeError("Could not extract reference frame")

        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.progress.emit(0.25, "Extracting target frame…")
        tgt_frame = self._load_source_frame()
        if tgt_frame is None:
            raise RuntimeError("Could not extract target frame")

        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.progress.emit(0.45, "Generating 3D LUT…")
        from sdqt.utils.grade_compute import generate_lut3d

        tmpdir = Path(tempfile.mkdtemp(prefix="ccorrect_"))
        lut_path = str(tmpdir / "grade.cube")
        try:
            generate_lut3d(ref_frame, tgt_frame, lut_path)
        except Exception as exc:
            logger.warning("generate_lut3d failed: %s", exc, exc_info=True)
            raise RuntimeError(f"LUT generation failed: {exc}")

        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.progress.emit(0.70, "Applying LUT…")
        self._apply_lut(self._source, lut_path, self._output)

        self.progress.emit(1.0, "Color correct complete")
        return self._output

    # ------------------------------------------------------------------

    @staticmethod
    def _probe_color_meta(path: str) -> dict:
        """Probe a video's colour metadata so we can preserve it on re-encode."""
        out: dict = {}
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries",
                 "stream=color_range,color_space,color_primaries,color_transfer",
                 "-of", "default=nw=1:nk=0", path],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if v and v.lower() not in ("unknown", "n/a", "und"):
                    out[k.strip()] = v
        except Exception as exc:
            logger.warning("probe_color_meta failed: %s", exc)
        return out

    # ------------------------------------------------------------------

    def _load_reference_frame(self):
        """Return uint8 RGB (H,W,3) frame from an image or the first frame of a video."""
        suffix = Path(self._reference).suffix.lower()

        if suffix in _IMAGE_EXTS:
            try:
                import numpy as np
                from PIL import Image
                img = Image.open(self._reference).convert("RGB")
                return np.array(img)
            except Exception as exc:
                logger.warning("Failed to read reference image: %s", exc)
                return None

        # Video reference: use the same helper Match Grade uses so the
        # same extraction path is exercised.
        from sdqt.utils.grade_compute import render_frame_with_effects
        return render_frame_with_effects(self._reference, [], media_offset=0.0)

    def _load_source_frame(self):
        """Grab a representative (mid-ish) frame from the generated source clip.

        Uses the same ``render_frame_with_effects`` helper the timeline Match
        Grade → / ← feature uses, so this runs through code that's already
        known to work for LUT generation.
        """
        from sdqt.utils.grade_compute import render_frame_with_effects

        dur = self._probe_duration(self._source)
        offset = max(dur / 2.0, 0.1)
        return render_frame_with_effects(self._source, [], media_offset=offset)

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip() or 0.0)
        except Exception:
            return 0.0

    def _apply_lut(self, src: str, lut_path: str, out: str) -> None:
        """Apply the LUT via ffmpeg. Filter chain + output flags pull from
        the active project's color profile (via the thread-local set by
        BaseWorker.run()), so the LUT result lands in the same color space
        the rest of the project uses.
        """
        from sdqt.utils.codec import project_input_filter

        # Escape : in the LUT path for ffmpeg filter graph
        lut_filt = lut_path.replace("\\", "/").replace(":", "\\:")
        # Probe-aware range/matrix conversion — full_range_filter() assumed
        # full-range input and washed out tv-range clips before the LUT.
        in_filt = project_input_filter(source_path=src, add_setsar=False,
                                       add_matrix=True)
        vf = f"{in_filt},lut3d=file='{lut_filt}'"

        base = [
            "ffmpeg", "-y", "-v", "error",
            "-i", src,
            "-vf", vf,
            *_codec_args(),
            *_pix_fmt(_enc_name()),
        ]
        cmd = base + ["-c:a", "copy", out]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Fall back without audio copy (some generated clips have no audio)
            cmd2 = base + ["-an", out]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"lut3d apply failed: {result2.stderr[-400:]}"
                )
