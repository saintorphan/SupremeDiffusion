"""Batch color correction workers.

Two workers live here:

- :class:`ColorCorrectionBatchWorker` — applies computed grade *effects*
  (ffmpeg ``eq``/``colorbalance``/``lut3d`` filter chain) to a batch of files.
- :class:`ClipColorMatchBatchWorker` — true **per-frame** reference color
  match (the real "match this clip to this source image" tool). Runs the same
  color-matcher transfer the Generate-tab Mode-2 "match frame" LF/RF buttons
  use, applied to *every* frame of the clip rather than baking one static LUT.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from sdqt.utils.codec import configured_codec_args
from sdqt.utils.naming import cap_stem
from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".ts", ".m4v"}


class ColorCorrectionBatchWorker(BaseWorker):
    """Apply computed color correction effects to a batch of files.

    For videos: uses ffmpeg filter chain.
    For images: uses ffmpeg single-frame processing.
    """

    def __init__(
        self,
        effects: list[dict],
        files: list[str],
        output_dir: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._effects = effects
        self._files = files
        self._output_dir = output_dir

    def do_work(self) -> list[str]:
        from sdqt.workers.video_effects import _build_video_filters

        vf = _build_video_filters(self._effects)
        if not vf:
            return list(self._files)  # nothing to do

        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[str] = []
        total = len(self._files)

        for i, src in enumerate(self._files):
            if self.is_aborted:
                break

            src_path = Path(src)
            self.progress.emit(
                (i + 0.5) / total,
                f"Color correcting {src_path.name} ({i + 1}/{total})",
            )

            suffix = src_path.suffix.lower()
            stem = src_path.stem
            # Avoid stacking _cc suffixes from multi-pass refinement
            while stem.endswith("_cc"):
                stem = stem[:-3]
            out_name = f"{cap_stem(stem)}_cc{src_path.suffix}"
            out_path = str(out_dir / out_name)

            # If input and output resolve to the same file (refinement pass),
            # write to a temp file first then replace
            same_file = Path(src).resolve() == Path(out_path).resolve()

            try:
                write_path = out_path
                if same_file:
                    fd, write_path = tempfile.mkstemp(suffix=src_path.suffix,
                                                      dir=str(out_dir))
                    os.close(fd)

                if suffix in _IMAGE_EXTS:
                    self._process_image(src, write_path, vf)
                elif suffix in _VIDEO_EXTS:
                    self._process_video(src, write_path, vf)
                else:
                    logger.warning("Unsupported file type: %s", suffix)
                    if same_file:
                        Path(write_path).unlink(missing_ok=True)
                    continue

                if same_file:
                    os.replace(write_path, out_path)

                results.append(out_path)
            except Exception as exc:
                logger.error("Failed to process %s: %s", src, exc)
                if same_file:
                    Path(write_path).unlink(missing_ok=True)

            self.progress.emit((i + 1) / total, f"Done {i + 1}/{total}")

        return results

    def _process_video(self, src: str, out: str, vf: str) -> None:
        from sdqt.utils.codec import project_input_filter, pix_fmt_args
        # Probe-aware range/matrix conversion — the old full_range_filter()
        # assumed full-range input and washed out tv-range clips.
        base = project_input_filter(source_path=src, add_setsar=False,
                                    add_matrix=True)
        full_vf = f"{base},{vf}" if vf else base
        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vf", full_vf,
            *configured_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            out,
        ]
        logger.info("CC video cmd: %s", " ".join(cmd))
        self._run_ffmpeg(cmd, label="color correct video")

    def _process_image(self, src: str, out: str, vf: str) -> None:
        # Images are full-range RGB in AND out — no range conversion belongs
        # here. The old full_range_filter() squeezed the output into tv
        # levels (16–235) inside a PNG/JPEG, washing every corrected image.
        cmd = [
            "ffmpeg", "-y", "-i", src,
        ]
        if vf:
            cmd += ["-vf", vf]
        cmd += [
            "-frames:v", "1",
            "-update", "1",
            out,
        ]
        logger.info("CC image cmd: %s", " ".join(cmd))
        self._run_ffmpeg(cmd, label="color correct image")


class ClipColorMatchBatchWorker(BaseWorker):
    """Per-frame reference color match for a batch of clips / images.

    For each source file, matches **every frame** to the reference image
    using the color-matcher registry (default ``mkl`` — the exact transfer
    the Generate-tab Mode-2 "match frame" LF/RF buttons use). This is a true
    per-frame match, not a single static LUT baked from one arbitrary frame,
    so the look holds across the whole clip.

    Videos: matched to a temp video, then the source audio (if any) is muxed
    back so audio survives. Images: single-shot match.
    """

    def __init__(
        self,
        reference: str,
        files: list[str],
        output_dir: str,
        *,
        method: str = "mkl",
        strength: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reference = reference
        self._files = files
        self._output_dir = output_dir
        self._method = method
        self._strength = strength

    def do_work(self) -> list[str]:
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[str] = []
        total = max(len(self._files), 1)

        for i, src in enumerate(self._files):
            if self.is_aborted:
                break

            src_path = Path(src)
            stem = src_path.stem
            # Avoid stacking _matched suffixes on repeated passes
            while stem.endswith("_matched"):
                stem = stem[: -len("_matched")]
            suffix = src_path.suffix.lower()

            try:
                if suffix in _IMAGE_EXTS:
                    out_path = str(out_dir / f"{cap_stem(stem)}_matched{src_path.suffix}")
                    self._match_image(src, out_path)
                else:
                    out_path = str(out_dir / f"{cap_stem(stem)}_matched.mp4")
                    self._match_video(src, out_path, i, total)
                results.append(out_path)
            except Exception as exc:
                logger.error("Color match failed for %s: %s", src, exc)

            self.progress.emit((i + 1) / total, f"Done {i + 1}/{total}")

        return results

    def _match_video(self, src: str, out_path: str, i: int, total: int) -> None:
        from supremediffusion.utils.color_correct import color_correct_video

        src_name = Path(src).name

        def _cb(frac: float, desc: str) -> None:
            # color_correct_video ignores the return value, so abort can only
            # take effect between files — good enough; a single clip finishes.
            self.progress.emit(
                (i + float(frac)) / total,
                f"{src_name} ({i + 1}/{total}) — {desc}",
            )

        fd, tmp_v = tempfile.mkstemp(suffix=".mp4", prefix="ccmatch_")
        os.close(fd)
        try:
            color_correct_video(
                src, self._reference, tmp_v,
                self._strength, _cb, method=self._method,
            )
            self._mux_audio(tmp_v, src, out_path)
        finally:
            Path(tmp_v).unlink(missing_ok=True)

    def _mux_audio(self, video_only: str, audio_src: str, out_path: str) -> None:
        """Copy the matched video + the source's audio (if any) into out_path."""
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", video_only,
            "-i", audio_src,
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "copy",
            "-shortest", out_path,
        ]
        try:
            self._run_ffmpeg(cmd, label="mux matched audio")
        except Exception as exc:
            logger.warning("audio mux failed (%s) — writing video without audio", exc)
            shutil.copyfile(video_only, out_path)

    def _match_image(self, src: str, out_path: str) -> None:
        import numpy as np
        from PIL import Image
        from supremediffusion.utils.color_correct import apply_color_match

        tgt = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255.0
        ref = np.asarray(
            Image.open(self._reference).convert("RGB")
        ).astype(np.float32) / 255.0
        matched = apply_color_match(
            tgt, ref, method=self._method, strength=self._strength,
        )
        arr = np.clip(matched * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_path)
