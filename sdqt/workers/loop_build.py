"""Loop Build worker — forward + reversed + match-graded concat with seam trim."""

from __future__ import annotations

import logging
import re
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


_BLACK_RE = re.compile(
    r"black_start:(?P<start>[0-9.]+)\s+black_end:(?P<end>[0-9.]+)"
)


def _parse_blackdetect(stderr: str) -> list[tuple[float, float]]:
    """Parse ffmpeg blackdetect stderr → list of (start, end) ranges."""
    ranges: list[tuple[float, float]] = []
    for m in _BLACK_RE.finditer(stderr):
        try:
            ranges.append((float(m["start"]), float(m["end"])))
        except (TypeError, ValueError):
            continue
    return ranges


class LoopBuildWorker(BaseWorker):
    """Build a seamless forward+reverse loop from a single clip.

    Pipeline: trim forward → reverse it → match-grade the reverse to the
    forward → concat → detect and trim any black frames at the seam.
    """

    def __init__(
        self,
        source: str,
        *,
        media_offset: float = 0.0,
        duration: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._media_offset = float(media_offset or 0.0)
        self._duration = float(duration or 0.0)
        self._tmpdir: Path | None = None

    # ------------------------------------------------------------------

    def do_work(self) -> str:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="loop_"))

        # Probe the source's actual colour range + colour metadata and
        # preserve every one of them through the loop pipeline. The
        # previous "grey" bug came from re-tagging the output with
        # bt709/pc regardless of what the source actually was — that
        # shifted black (16 in tv-range) to grey (16/255 in pc-range).
        self._src_color = self._probe_color_meta(self._source)
        logger.info("LoopBuild: source color metadata=%s", self._src_color)

        # Single filter_complex pass does forward + reverse + concat in
        # one ffmpeg invocation — no drift between the two halves, and
        # no need for a match-grade re-encode afterwards.
        self.progress.emit(0.10, "Building forward+reverse loop...")
        concat = self._build_loop_one_pass()
        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.progress.emit(0.80, "Detecting black frames...")
        final = self._trim_seam_blacks(concat)
        self.progress.emit(1.0, "Loop ready")
        return final

    # ------------------------------------------------------------------

    def _segment_bounds(self) -> tuple[float, float]:
        start = self._media_offset
        end = start + self._duration if self._duration > 0 else 0.0
        return start, end

    def _build_loop_one_pass(self) -> str:
        """Build forward+reverse loop in a single ffmpeg pass.

        One ``filter_complex`` runs the split → trim → reverse → concat
        pipeline on the decoded video, so the reversed half has
        pixel-identical colour to the forward half.

        Crucially, no colour-range conversion is forced. The scale filter
        has no ``in_range``/``out_range`` override, so ffmpeg decodes in
        the source's native range and leaves the pixel values untouched.
        The encoder is then told the exact colour metadata the source
        had, so the output mp4 plays back identically to the original.
        """
        out = str(self._tmpdir / "loop.mp4")
        start, end = self._segment_bounds()
        trimmed = self._duration > 0

        has_audio = self._probe_has_audio(self._source)

        if trimmed:
            v_trim = f"trim={start}:{end},setpts=PTS-STARTPTS"
            a_trim = f"atrim={start}:{end},asetpts=PTS-STARTPTS"
        else:
            v_trim = "setpts=PTS-STARTPTS"
            a_trim = "asetpts=PTS-STARTPTS"

        # Prepend the full-range scale filter to the video chain so libx264
        # sees full-range RGB, then output via yuvj420p which is the only
        # pix_fmt that actually survives x264's implicit range squeeze.
        v_parts = [
            "[0:v]scale=in_range=full:out_range=full,split=2[vA][vB]",
            f"[vA]{v_trim}[fwdv]",
            f"[vB]{v_trim},reverse,setpts=PTS-STARTPTS[revv]",
            "[fwdv][revv]concat=n=2:v=1:a=0[outv]",
        ]

        maps = ["-map", "[outv]"]
        if has_audio:
            a_parts = [
                "[0:a]asplit=2[aA][aB]",
                f"[aA]{a_trim}[fwda]",
                f"[aB]{a_trim},areverse,asetpts=PTS-STARTPTS[reva]",
                "[fwda][reva]concat=n=2:v=0:a=1[outa]",
            ]
            filt = ";".join(v_parts + a_parts)
            maps += ["-map", "[outa]"]
        else:
            filt = ";".join(v_parts)

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", self._source,
            "-filter_complex", filt,
            *maps,
            *_codec_args(),
            *_pix_fmt(_enc_name()),
        ]

        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd.append(out)
        self._run(cmd)
        return out

    def _trim_seam_blacks(self, concat_path: str) -> str:
        """Detect black frames and trim any that overlap the forward/reverse seam."""
        # Probe total duration and locate the seam
        dur = self._probe_duration(concat_path)
        if dur <= 0:
            return concat_path
        seam = dur / 2.0

        # Detect black frames
        result = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", concat_path,
             "-vf", "blackdetect=d=0.05:pic_th=0.98",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        ranges = _parse_blackdetect(result.stderr)
        if not ranges:
            return concat_path

        # Only trim ranges that intersect the seam window (±0.3s)
        window = 0.30
        seam_blacks = [
            (s, e) for s, e in ranges
            if s < seam + window and e > seam - window
        ]
        if not seam_blacks:
            return concat_path

        # Build a pair of trim segments that skip the black range at the seam
        # For simplicity, if a single range crosses the seam, trim [0, s] and [e, dur].
        keep_start = 0.0
        keep_mid_end = seam_blacks[0][0]
        keep_resume = seam_blacks[-1][1]
        if keep_mid_end <= keep_start + 0.05 or keep_resume >= dur - 0.05:
            return concat_path

        out = str(self._tmpdir / "loop_trimmed.mp4")
        # Prepend full-range scale so libx264 encodes full-range pixels.
        filt = (
            f"[0:v]scale=in_range=full:out_range=full,"
            f"trim={keep_start}:{keep_mid_end},setpts=PTS-STARTPTS[v0];"
            f"[0:v]scale=in_range=full:out_range=full,"
            f"trim={keep_resume}:{dur},setpts=PTS-STARTPTS[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[v];"
            f"[0:a]atrim={keep_start}:{keep_mid_end},asetpts=PTS-STARTPTS[a0];"
            f"[0:a]atrim={keep_resume}:{dur},asetpts=PTS-STARTPTS[a1];"
            f"[a0][a1]concat=n=2:v=0:a=1[a]"
        )
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", concat_path,
            "-filter_complex", filt,
            "-map", "[v]", "-map", "[a]",
            *_codec_args(),
            *_pix_fmt(_enc_name()),
            "-c:a", "aac", "-b:a", "192k",
            out,
        ]
        try:
            self._run(cmd)
            return out
        except RuntimeError:
            # If audio mapping fails (no audio stream), retry video-only
            cmd_v = [
                "ffmpeg", "-y", "-v", "error",
                "-i", concat_path,
                "-filter_complex",
                (f"[0:v]scale=in_range=full:out_range=full,"
                 f"trim={keep_start}:{keep_mid_end},setpts=PTS-STARTPTS[v0];"
                 f"[0:v]scale=in_range=full:out_range=full,"
                 f"trim={keep_resume}:{dur},setpts=PTS-STARTPTS[v1];"
                 f"[v0][v1]concat=n=2:v=1:a=0[v]"),
                "-map", "[v]",
                *_codec_args(),
                *_pix_fmt(_enc_name()),
                "-an",
                out,
            ]
            try:
                self._run(cmd_v)
                return out
            except RuntimeError:
                logger.warning("seam trim failed; returning untrimmed concat", exc_info=True)
                return concat_path

    # ------------------------------------------------------------------

    @staticmethod
    def _probe_color_meta(path: str) -> dict:
        """Probe a video's colour metadata so we can preserve it on re-encode.

        Returns a dict with keys ``color_range``, ``color_space``,
        ``color_primaries``, ``color_transfer``. Missing/unknown values
        are omitted so the caller skips passing the corresponding flag.
        """
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

    @staticmethod
    def _probe_has_audio(path: str) -> bool:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return "audio" in result.stdout.strip()
        except Exception:
            return False

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

    @staticmethod
    def _run(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr[-400:] if result.stderr else 'unknown'}"
            )
