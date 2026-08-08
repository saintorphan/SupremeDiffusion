"""Worker threads for timeline preview and export operations."""

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
    probe_color_range,
)

logger = logging.getLogger(__name__)


_probe_audio_cache: dict[str, bool] = {}


def _probe_has_audio(path: str) -> bool:
    """Check if a file actually has an audio stream. Results are cached."""
    if path in _probe_audio_cache:
        return _probe_audio_cache[path]
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=5,
        )
        has = bool(result.stdout.strip())
    except Exception:
        has = False
    _probe_audio_cache[path] = has
    return has


def clear_probe_cache() -> None:
    """Clear the audio probe cache (called on timeline data reload)."""
    _probe_audio_cache.clear()


def _build_effects_filter(effects) -> str:
    """Build ffmpeg filter string from ordered effect stack.

    Accepts either the new list[dict] format or legacy flat dict.
    Returns a filter fragment starting with a comma, or empty string.
    """
    from sdqt.models.effects import migrate_legacy_effects

    # Handle legacy dict format
    if isinstance(effects, dict):
        effects = migrate_legacy_effects(effects)

    if not effects:
        return ""

    parts = []
    for fx in effects:
        if not fx.get("enabled", True):
            continue
        builder = _FILTER_BUILDERS.get(fx.get("type", ""))
        if builder:
            frag = builder(fx.get("params", {}))
            if frag:
                parts.append(frag)
    return "," + ",".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Per-type ffmpeg filter builders
# ---------------------------------------------------------------------------

def _build_hue_sat(p: dict) -> str:
    hue_shift = p.get("hue_shift", 0.0)
    sat = p.get("saturation", 1.0)
    if hue_shift == 0.0 and sat == 1.0:
        return ""
    parts = []
    if hue_shift != 0.0:
        parts.append(f"h={hue_shift}")
    if sat != 1.0:
        parts.append(f"s={sat}")
    return f"hue={':'.join(parts)}"


def _build_brightness_contrast(p: dict) -> str:
    br = p.get("brightness", 0.0)
    ct = p.get("contrast", 1.0)
    if br == 0.0 and ct == 1.0:
        return ""
    parts = []
    if br != 0.0:
        parts.append(f"brightness={br}")
    if ct != 1.0:
        parts.append(f"contrast={ct}")
    return f"eq={':'.join(parts)}"


def _build_levels(p: dict) -> str:
    bp = p.get("black_point", 0.0) / 255.0
    wp = p.get("white_point", 255.0) / 255.0
    gamma = p.get("gamma", 1.0)
    if bp == 0.0 and wp == 1.0 and gamma == 1.0:
        return ""
    parts = []
    if bp != 0.0 or wp != 1.0:
        parts.append(
            f"colorlevels=rimin={bp}:gimin={bp}:bimin={bp}"
            f":rimax={wp}:gimax={wp}:bimax={wp}"
        )
    if gamma != 1.0:
        parts.append(f"eq=gamma={gamma}")
    return ",".join(parts)


def _build_shadows_highlights(p: dict) -> str:
    shadows = p.get("shadows", 0.0)
    highlights = p.get("highlights", 0.0)
    if shadows == 0.0 and highlights == 0.0:
        return ""
    # Map to curves: shadows affect quarter-tone, highlights affect three-quarter-tone
    s_val = max(0.0, min(1.0, 0.25 + shadows * 0.25))
    h_val = max(0.0, min(1.0, 0.75 + highlights * 0.25))
    return f"curves=master='0/0 0.25/{s_val:.3f} 0.75/{h_val:.3f} 1/1'"


def _build_color_balance(p: dict) -> str:
    rs = p.get("shadow_r", 0.0)
    gs = p.get("shadow_g", 0.0)
    bs = p.get("shadow_b", 0.0)
    rm = p.get("midtone_r", 0.0)
    gm = p.get("midtone_g", 0.0)
    bm = p.get("midtone_b", 0.0)
    rh = p.get("highlight_r", 0.0)
    gh = p.get("highlight_g", 0.0)
    bh = p.get("highlight_b", 0.0)
    if all(v == 0.0 for v in (rs, gs, bs, rm, gm, bm, rh, gh, bh)):
        return ""
    return (
        f"colorbalance=rs={rs}:gs={gs}:bs={bs}"
        f":rm={rm}:gm={gm}:bm={bm}"
        f":rh={rh}:gh={gh}:bh={bh}"
    )


def _build_color_temperature(p: dict) -> str:
    temp = p.get("temperature", 0.0)
    tint = p.get("tint", 0.0)
    if temp == 0.0 and tint == 0.0:
        return ""
    # Map temperature to warm (positive) / cool (negative) via color balance
    # Warm = more red shadows + less blue, Cool = opposite
    t = temp / 100.0  # -1..1
    ti = tint / 100.0  # -1..1
    rs = t * 0.3
    bs = -t * 0.3
    gm = ti * 0.3
    return f"colorbalance=rs={rs:.3f}:bs={bs:.3f}:gm={gm:.3f}"


def _build_channel_mixer(p: dict) -> str:
    rr = p.get("rr", 1.0)
    gg = p.get("gg", 1.0)
    bb = p.get("bb", 1.0)
    if rr == 1.0 and gg == 1.0 and bb == 1.0:
        return ""
    return f"colorchannelmixer=rr={rr:.4f}:gg={gg:.4f}:bb={bb:.4f}"


def _build_vibrance(p: dict) -> str:
    v = p.get("vibrance", 1.0)
    if v == 1.0:
        return ""
    return f"eq=saturation={v}"


def _build_sharpen(p: dict) -> str:
    amount = p.get("amount", 0.0)
    if amount <= 0.0:
        return ""
    half = amount / 2.0
    return f"unsharp=5:5:{amount}:5:5:{half}"


def _build_blur(p: dict) -> str:
    amount = p.get("amount", 0.0)
    if amount <= 0.0:
        return ""
    # boxblur requires integer-like radius; clamp minimum to 1
    r = max(1, int(round(amount)))
    return f"boxblur={r}:{r}"


def _build_denoise(p: dict) -> str:
    s = p.get("strength", 0.0)
    if s <= 0.0:
        return ""
    return f"hqdn3d={s}:{s}:{s}:{s}"


def _build_vignette(p: dict) -> str:
    intensity = p.get("intensity", 0.0)
    if intensity <= 0.0:
        return ""
    import math
    angle = math.pi / (2 + intensity * 3)
    return f"vignette={angle:.4f}"


def _build_film_grain(p: dict) -> str:
    intensity = p.get("intensity", 0.0)
    if intensity <= 0.0:
        return ""
    strength = int(round(intensity * 100))
    return f"noise=alls={strength}:allf=t"


def _build_crop_zoom(p: dict) -> str:
    x = p.get("x", 0.0)
    y = p.get("y", 0.0)
    w = p.get("w", 1.0)
    h = p.get("h", 1.0)
    if x == 0.0 and y == 0.0 and w >= 0.999 and h >= 0.999:
        return ""
    # Clamp the rect inside the frame so crop can't fail on x+w > 1.
    x = min(x, 1.0 - w) if w < 1.0 else 0.0
    y = min(y, 1.0 - h) if h < 1.0 else 0.0
    # ZOOM semantics, matching the GL preview shader: the selected region is
    # scaled back up to the ORIGINAL frame size, not left as a smaller frame.
    # A bare crop shrinks the output (e.g. 928->918), which the preview then
    # letterboxes — "the bake undid my zoom". Dims rounded even for yuv420p
    # encoders. Inside scale, iw/ih are the CROPPED dims, so dividing by the
    # fractions recovers the pre-crop size.
    return (
        f"crop=w=floor(iw*{w:.4f}/2)*2:h=floor(ih*{h:.4f}/2)*2:"
        f"x=iw*{x:.4f}:y=ih*{y:.4f},"
        f"scale=round(iw/{w:.4f}/2)*2:round(ih/{h:.4f}/2)*2:flags=lanczos"
    )


def _build_lut3d(p: dict) -> str:
    from pathlib import Path
    lut_file = p.get("lut_file", "")
    if not lut_file or not Path(lut_file).is_file():
        return ""
    escaped = lut_file.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
    return f"lut3d=file='{escaped}':interp=tetrahedral"


_FILTER_BUILDERS = {
    "crop_zoom": _build_crop_zoom,
    "hue_sat": _build_hue_sat,
    "brightness_contrast": _build_brightness_contrast,
    "levels": _build_levels,
    "shadows_highlights": _build_shadows_highlights,
    "color_balance": _build_color_balance,
    "channel_mixer": _build_channel_mixer,
    "color_temperature": _build_color_temperature,
    "vibrance": _build_vibrance,
    "sharpen": _build_sharpen,
    "blur": _build_blur,
    "denoise": _build_denoise,
    "vignette": _build_vignette,
    "film_grain": _build_film_grain,
    "lut3d": _build_lut3d,
}


def _build_normalize_vf(source_path: str, target_w, target_h) -> str:
    """Build a ``scale + pad`` filter for an input being normalized to the
    timeline's output resolution.

    Probes *source_path*'s actual color range so a limited-range source gets
    expanded to the active profile's output range instead of being silently
    retagged. Without this, dark colors get lifted and saturation flattens
    on every limited-range clip the timeline encodes.
    """
    from sdqt.utils.codec import probe_color_range, _resolve_profile
    in_range = probe_color_range(source_path)
    out_range = _resolve_profile().output_range()
    return (
        f"scale={target_w}:{target_h}:flags=lanczos:"
        f"force_original_aspect_ratio=decrease:"
        f"in_range={in_range}:out_range={out_range},"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )


def build_filter_complex(
    tracks: list, duration: float,
    width: int = 1280, height: int = 720, fps: int = 25, sample_rate: int = 44100,
) -> tuple[list[str], str, str, str]:
    """Build ffmpeg -filter_complex args for multi-track compositing.

    Args:
        tracks: list of TimelineTrack objects
        duration: total timeline duration in seconds
        width: output video width
        height: output video height
        fps: output frame rate
        sample_rate: output audio sample rate

    Returns:
        (input_args, filter_string, video_output_label, audio_output_label)
    """
    from sdqt.widgets.timeline_track import TrackType

    input_args: list[str] = []
    # Each clip gets its own input — even if two clips share a source file.
    # ffmpeg can't read the same stream twice in filter_complex without split,
    # and using separate -i entries is simpler and handles all cases.
    clip_input_map: list[tuple] = []  # (track, clip_idx, input_idx)
    input_idx = 0

    for track in tracks:
        if track.muted and not track.visible:
            continue
        for ci, clip in enumerate(track.clips):
            input_args.extend(["-i", clip.path])
            clip_input_map.append((track, ci, input_idx))
            input_idx += 1

    if input_idx == 0:
        return [], "", "", ""

    filters: list[str] = []
    audio_sources: list[str] = []
    stream_idx = 0

    # Group clips by track, sorted by start_time within each track
    track_clips: dict[str, list[tuple]] = {}  # track_id -> [(clip, inp_idx), ...]
    for track, ci, inp in clip_input_map:
        if ci >= len(track.clips):
            continue
        track_clips.setdefault(track.id, []).append((track.clips[ci], inp, track))

    for tid in track_clips:
        track_clips[tid].sort(key=lambda x: x[0].start_time)

    # Build per-track video+audio streams via paired concat.
    # Each clip gets a video segment and a matching audio segment (real or silence).
    # This keeps video and audio perfectly synced without adelay/amix drift.
    video_track_labels: list[str] = []
    audio_track_labels: list[str] = []
    video_track_index = 0  # tracks which video track layer we're on

    for track in tracks:
        clips_for_track = track_clips.get(track.id, [])
        if not clips_for_track:
            continue

        if track.track_type == TrackType.VIDEO and track.visible:
            is_upper = video_track_index > 0
            video_segments: list[str] = []
            audio_segments: list[str] = []
            has_any_audio = not track.muted
            cursor = 0.0  # current time position on track

            for clip, inp, _ in clips_for_track:
                # Insert gap fill + silence for any gap before this clip
                gap = clip.start_time - cursor
                if gap > 0.01:  # > 10ms gap
                    gtag_v = f"g{stream_idx}v"
                    if is_upper:
                        # Transparent gap for upper tracks so lower tracks show through
                        filters.append(
                            f"color=c=black@0.0:s={width}x{height}:d={gap}:r={fps},"
                            f"format=rgba,setsar=1,setpts=PTS-STARTPTS[{gtag_v}]"
                        )
                    else:
                        filters.append(
                            f"color=c=black:s={width}x{height}:d={gap}:r={fps},"
                            f"setsar=1,setpts=PTS-STARTPTS[{gtag_v}]"
                        )
                    video_segments.append(f"[{gtag_v}]")
                    if has_any_audio:
                        gtag_a = f"g{stream_idx}a"
                        filters.append(
                            f"anullsrc=r={sample_rate}:cl=stereo:d={gap},"
                            f"asetpts=PTS-STARTPTS[{gtag_a}]"
                        )
                        audio_segments.append(f"[{gtag_a}]")
                    stream_idx += 1

                mo = getattr(clip, "media_offset", 0.0) or 0.0
                trim_start = mo
                trim_end = mo + clip.duration
                vtag = f"s{stream_idx}v"
                atag = f"s{stream_idx}a"

                # Video: trim + scale + fps normalize + effects + optional fades
                fade_filter = ""
                clip_fade_in = getattr(clip, "fade_in", 0.0) or 0.0
                clip_fade_out = getattr(clip, "fade_out", 0.0) or 0.0
                if clip_fade_in > 0:
                    fade_filter += f",fade=t=in:st=0:d={clip_fade_in}"
                if clip_fade_out > 0:
                    fade_start = clip.duration - clip_fade_out
                    fade_filter += f",fade=t=out:st={fade_start}:d={clip_fade_out}"

                # Per-clip effects (hue/saturation/brightness/contrast)
                fx = _build_effects_filter(getattr(clip, "effects", []))

                # Per-clip rotation (metadata flag → ffmpeg transpose)
                rot = getattr(clip, "rotation", 0) % 360
                rot_filter = ""
                if rot == 90:
                    rot_filter = ",transpose=1"
                elif rot == 180:
                    rot_filter = ",transpose=1,transpose=1"
                elif rot == 270:
                    rot_filter = ",transpose=2"

                # Probe this clip's actual color range so a limited-range
                # source gets properly expanded to the active profile's
                # output range instead of being silently retagged. Same for
                # the YUV matrix: a BT.601-tagged import gets a real 601→709
                # conversion (dedicated colorspace filter — scale's matrix
                # options are no-ops for YUV→YUV) instead of a hue-shifting
                # retag. Untagged clips keep the no-conversion behavior.
                from sdqt.utils.codec import (
                    probe_color_range, matrix_convert_filter, _resolve_profile,
                )
                _profile = _resolve_profile()
                in_range = probe_color_range(clip.path)
                out_range = _profile.output_range()
                range_args = f"in_range={in_range}:out_range={out_range}"
                conv = matrix_convert_filter(clip.path, _profile)
                conv = f"{conv}," if conv else ""
                if is_upper:
                    # Upper tracks need rgba format so overlay preserves alpha
                    filters.append(
                        f"[{inp}:v]trim={trim_start}:{trim_end},setpts=PTS-STARTPTS{rot_filter},"
                        f"{conv}"
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                        f"{range_args},"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                        f"fps={fps},setsar=1{fx},format=rgba{fade_filter}[{vtag}]"
                    )
                else:
                    filters.append(
                        f"[{inp}:v]trim={trim_start}:{trim_end},setpts=PTS-STARTPTS{rot_filter},"
                        f"{conv}"
                        f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                        f"{range_args},"
                        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                        f"fps={fps},setsar=1{fx}{fade_filter}[{vtag}]"
                    )
                video_segments.append(f"[{vtag}]")

                # Audio: real audio or generated silence to match clip duration
                if has_any_audio:
                    real_audio = (clip.has_audio and not track.muted
                                  and not getattr(clip, "muted", False)
                                  and _probe_has_audio(clip.path))
                    if real_audio:
                        vol = clip.volume * track.volume
                        afade = ""
                        if clip_fade_in > 0:
                            afade += f",afade=t=in:st=0:d={clip_fade_in}"
                        if clip_fade_out > 0:
                            afade_start = clip.duration - clip_fade_out
                            afade += f",afade=t=out:st={afade_start}:d={clip_fade_out}"
                        filters.append(
                            f"[{inp}:a]aresample={sample_rate},atrim={mo}:{mo + clip.duration},"
                            f"asetpts=PTS-STARTPTS,volume={vol:.3f}{afade}[{atag}]"
                        )
                    else:
                        dur = clip.duration
                        filters.append(
                            f"anullsrc=r={sample_rate}:cl=stereo:d={dur},"
                            f"asetpts=PTS-STARTPTS[{atag}]"
                        )
                    audio_segments.append(f"[{atag}]")

                cursor = clip.start_time + clip.duration
                stream_idx += 1

            # Concat video segments
            if len(video_segments) == 1:
                video_track_labels.append(video_segments[0].strip("[]"))
            else:
                out_label = f"vt{track.id[:6]}"
                filters.append(
                    f"{''.join(video_segments)}concat=n={len(video_segments)}:v=1:a=0[{out_label}]"
                )
                video_track_labels.append(out_label)

            # Concat audio segments (paired with video)
            if audio_segments:
                if len(audio_segments) == 1:
                    audio_track_labels.append(audio_segments[0].strip("[]"))
                else:
                    aout_label = f"at{track.id[:6]}"
                    filters.append(
                        f"{''.join(audio_segments)}concat=n={len(audio_segments)}:v=0:a=1[{aout_label}]"
                    )
                    audio_track_labels.append(aout_label)

            video_track_index += 1

        elif track.track_type == TrackType.AUDIO and not track.muted:
            # Audio-only track: extract audio from each clip, concat with gap silence
            audio_segments: list[str] = []
            cursor = 0.0

            for clip, inp, _ in clips_for_track:
                gap = clip.start_time - cursor
                if gap > 0.01:
                    gtag_a = f"g{stream_idx}a"
                    filters.append(
                        f"anullsrc=r={sample_rate}:cl=stereo:d={gap},"
                        f"asetpts=PTS-STARTPTS[{gtag_a}]"
                    )
                    audio_segments.append(f"[{gtag_a}]")
                    stream_idx += 1

                mo = getattr(clip, "media_offset", 0.0) or 0.0
                atag = f"s{stream_idx}a"

                clip_fade_in = getattr(clip, "fade_in", 0.0) or 0.0
                clip_fade_out = getattr(clip, "fade_out", 0.0) or 0.0

                real_audio = (clip.has_audio and not getattr(clip, "muted", False)
                             and _probe_has_audio(clip.path))
                if real_audio:
                    vol = clip.volume * track.volume
                    afade = ""
                    if clip_fade_in > 0:
                        afade += f",afade=t=in:st=0:d={clip_fade_in}"
                    if clip_fade_out > 0:
                        afade_start = clip.duration - clip_fade_out
                        afade += f",afade=t=out:st={afade_start}:d={clip_fade_out}"
                    filters.append(
                        f"[{inp}:a]aresample={sample_rate},atrim={mo}:{mo + clip.duration},"
                        f"asetpts=PTS-STARTPTS,volume={vol:.3f}{afade}[{atag}]"
                    )
                else:
                    dur = clip.duration
                    filters.append(
                        f"anullsrc=r={sample_rate}:cl=stereo:d={dur},"
                        f"asetpts=PTS-STARTPTS[{atag}]"
                    )
                audio_segments.append(f"[{atag}]")

                cursor = clip.start_time + clip.duration
                stream_idx += 1

            # Concat audio segments for this audio track
            if audio_segments:
                if len(audio_segments) == 1:
                    audio_track_labels.append(audio_segments[0].strip("[]"))
                else:
                    aout_label = f"at{track.id[:6]}"
                    filters.append(
                        f"{''.join(audio_segments)}concat=n={len(audio_segments)}:v=0:a=1[{aout_label}]"
                    )
                    audio_track_labels.append(aout_label)

    # Combine video tracks via overlay (for multi-track compositing)
    if not video_track_labels:
        filters.append(f"color=c=black:s={width}x{height}:d={duration}:r={fps}[bg]")
        current_video = "bg"
    elif len(video_track_labels) == 1:
        current_video = video_track_labels[0]
    else:
        current_video = video_track_labels[0]
        for i, label in enumerate(video_track_labels[1:], 1):
            out = f"mov{i}"
            filters.append(
                f"[{current_video}][{label}]overlay=eof_action=pass:format=auto[{out}]"
            )
            current_video = out

    video_out = current_video
    audio_out = ""

    if audio_track_labels:
        if len(audio_track_labels) == 1:
            audio_out = audio_track_labels[0]
        else:
            amix_inputs = "".join(f"[{l}]" for l in audio_track_labels)
            filters.append(
                f"{amix_inputs}amix=inputs={len(audio_track_labels)}:"
                f"duration=longest:normalize=0[aout]"
            )
            audio_out = "aout"

    filter_string = ";".join(filters)
    return input_args, filter_string, video_out, audio_out


class TimelinePreviewWorker(BaseWorker):
    """Build a preview video from multi-track timeline via filter_complex."""

    def __init__(
        self,
        tracks: list | None = None,
        output: str = "",
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
        # Legacy support
        clip_paths: list[str] | None = None,
        project_config=None,
        global_config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._tracks = tracks
        self._output = output
        self._width = width
        self._height = height
        self._fps = fps
        self._clip_paths = clip_paths  # legacy fallback
        self._project_config = project_config
        self._global_config = global_config

    def do_work(self) -> str:
        # Legacy path: simple concat if no tracks provided
        if self._tracks is None and self._clip_paths:
            return self._legacy_concat()

        if not self._tracks:
            raise ValueError("No tracks to preview")

        self.progress.emit(0.0, "Building preview...")

        duration = max((t.total_duration() for t in self._tracks), default=0.0)
        if duration <= 0:
            raise ValueError("No clips on timeline")

        input_args, filter_str, video_out, audio_out = build_filter_complex(
            self._tracks, duration,
            width=self._width, height=self._height, fps=self._fps,
        )

        if not input_args:
            raise ValueError("No input files found")

        cmd = ["ffmpeg", "-y"] + input_args
        cmd += ["-filter_complex", filter_str]
        cmd += ["-map", f"[{video_out}]"]
        if audio_out:
            cmd += ["-map", f"[{audio_out}]"]
        cmd += [
            *_codec_args(),
            "-c:a", "aac", "-b:a", "128k",
            *_pix_fmt(_enc_name()),
            self._output,
        ]

        self.progress.emit(0.3, "Encoding preview...")
        logger.info("Timeline preview cmd: %s", " ".join(cmd))

        returncode, stderr = self._run_ffmpeg_abortable(cmd)
        if returncode != 0:
            stderr_tail = stderr[-800:] if stderr else "ffmpeg failed"
            logger.warning("filter_complex preview failed (rc=%d), trying fallback: %s", returncode, stderr_tail)
            return self._fallback_concat()

        self.progress.emit(1.0, "Preview ready")
        return self._output

    def _run_ffmpeg_abortable(self, cmd: list[str], timeout: int = 300) -> tuple[int, str]:
        """Run ffmpeg as Popen, polling for abort every 0.5s.

        Returns (returncode, stderr). Kills the process if aborted.
        """
        import time
        import tempfile
        # stderr goes to a real file, not a pipe — nothing drains a pipe while
        # we poll, so ffmpeg would block mid-write once the 64KB buffer fills
        # and poll() would spin until the timeout (hang on long timelines).
        with tempfile.TemporaryFile() as stderr_buf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=stderr_buf,
            )
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if self.is_aborted:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise InterruptedError("Aborted")
                if time.monotonic() > deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    raise RuntimeError("FFmpeg timed out")
                time.sleep(0.5)
            stderr_buf.seek(0)
            stderr = stderr_buf.read().decode("utf-8", errors="replace")
        return proc.returncode, stderr

    def _fallback_concat(self) -> str:
        """Fallback: concat all clips across all tracks sequentially."""
        all_paths = []
        if self._tracks:
            for t in self._tracks:
                for c in t.clips:
                    all_paths.append(c.path)
        if not all_paths:
            raise ValueError("No clips to concatenate")
        return self._do_concat(all_paths)

    def _legacy_concat(self) -> str:
        if not self._clip_paths:
            raise ValueError("No clips to concatenate")
        self.progress.emit(0.0, "Building preview...")
        return self._do_concat(self._clip_paths)

    def _do_concat(self, paths: list[str]) -> str:
        concat_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="tl_concat_"
        )
        for p in paths:
            escaped = p.replace("'", "'\\''")
            concat_file.write(f"file '{escaped}'\n")
        concat_file.close()

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file.name,
                "-c", "copy",
                self._output,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.progress.emit(0.3, "Re-encoding for concat...")
                cmd2 = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file.name,
                    *_codec_args(),
                    "-c:a", "aac", "-b:a", "192k",
                    *_pix_fmt(_enc_name()),
                    self._output,
                ]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)
                if result2.returncode != 0:
                    raise RuntimeError(
                        result2.stderr[-500:] if result2.stderr else "ffmpeg concat failed"
                    )
        finally:
            Path(concat_file.name).unlink(missing_ok=True)

        self.progress.emit(1.0, "Preview ready")
        return self._output


class TimelineExportWorker(BaseWorker):
    """Export timeline with multi-track compositing + optional RIFE."""

    def __init__(
        self,
        output: str,
        tracks: list | None = None,
        resolution: str = "",
        rife_mode: str = "",
        codec: str = "",
        *,
        # Legacy support
        clip_paths: list[str] | None = None,
        project_config=None,
        global_config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._tracks = tracks
        self._output = output
        self._resolution = resolution
        self._rife_mode = rife_mode
        self._codec = codec
        self._clip_paths = clip_paths
        self._project_config = project_config
        self._global_config = global_config

    def do_work(self) -> str:
        # Legacy path
        if self._tracks is None and self._clip_paths:
            return self._legacy_export()

        if not self._tracks:
            raise ValueError("No tracks to export")

        self.progress.emit(0.0, "Exporting...")

        duration = max((t.total_duration() for t in self._tracks), default=0.0)
        if duration <= 0:
            raise ValueError("No clips on timeline")

        # Parse resolution for filter_complex
        fc_w, fc_h = 1280, 720
        if self._resolution:
            try:
                fc_w, fc_h = (int(x) for x in self._resolution.split("x"))
            except ValueError:
                pass

        input_args, filter_str, video_out, audio_out = build_filter_complex(
            self._tracks, duration, width=fc_w, height=fc_h,
        )

        if not input_args:
            raise ValueError("No input files found")

        # Add resolution scale if requested
        if self._resolution:
            w, h = self._resolution.split("x")
            filter_str += f";[{video_out}]scale={w}:{h}:force_original_aspect_ratio=decrease," \
                          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[scaled]"
            video_out = "scaled"

        cmd = ["ffmpeg", "-y"] + input_args
        cmd += ["-filter_complex", filter_str]
        cmd += ["-map", f"[{video_out}]"]
        if audio_out:
            cmd += ["-map", f"[{audio_out}]"]
        from sdqt.utils.codec import get_codec_args
        codec_args = get_codec_args(self._codec) if self._codec else _codec_args()
        cmd += [
            *codec_args,
            "-c:a", "aac", "-b:a", "192k",
            *_pix_fmt(_enc_name()),
        ]

        # Output to temp if RIFE needed
        if self._rife_mode:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="tl_exp_")
            tmp.close()
            cmd.append(tmp.name)
        else:
            cmd.append(self._output)

        self.progress.emit(0.3, "Encoding...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr[-500:] if result.stderr else "ffmpeg export failed"
            raise RuntimeError(stderr)

        export_path = tmp.name if self._rife_mode else self._output

        # RIFE
        if self._rife_mode:
            self.progress.emit(0.7, f"RIFE {self._rife_mode} upsampling...")
            export_path = self._apply_rife(export_path)
            if export_path != self._output:
                import shutil
                shutil.move(export_path, self._output)
                export_path = self._output

        self.progress.emit(1.0, "Export complete")
        return self._output

    def _legacy_export(self) -> str:
        """Legacy single-track export path."""
        if not self._clip_paths:
            raise ValueError("No clips to export")

        total_steps = 1
        if self._resolution:
            total_steps += 1
        if self._rife_mode:
            total_steps += 1
        step = 0

        processed_clips = self._clip_paths
        if self._resolution:
            step += 1
            self.progress.emit(step / total_steps * 0.5, "Normalizing resolution...")
            processed_clips = self._normalize_clips()
            if self.is_aborted:
                raise InterruptedError("Aborted")

        step += 1
        self.progress.emit(step / total_steps * 0.7, "Concatenating clips...")
        concat_output = self._concat_clips(processed_clips)
        if self.is_aborted:
            raise InterruptedError("Aborted")

        if self._rife_mode:
            step += 1
            self.progress.emit(step / total_steps * 0.9, f"RIFE {self._rife_mode} upsampling...")
            concat_output = self._apply_rife(concat_output)
            if self.is_aborted:
                raise InterruptedError("Aborted")

        if concat_output != self._output:
            import shutil
            shutil.move(concat_output, self._output)

        self.progress.emit(1.0, "Export complete")
        return self._output

    def _normalize_clips(self) -> list[str]:
        w, h = self._resolution.split("x")
        normalized = []
        for i, path in enumerate(self._clip_paths):
            if self.is_aborted:
                raise InterruptedError("Aborted")
            out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="tl_norm_")
            out.close()
            _norm_codec = get_codec_args(self._codec) if self._codec else _codec_args()
            cmd = [
                "ffmpeg", "-y", "-i", path,
                "-vf", _build_normalize_vf(path, w, h),
                *_norm_codec,
                "-c:a", "aac", "-b:a", "192k",
                *_pix_fmt(_enc_name()),
                out.name,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Resolution normalize failed for {Path(path).name}")
            normalized.append(out.name)
            frac = (i + 1) / len(self._clip_paths)
            self.progress.emit(frac * 0.4, f"Normalizing {i + 1}/{len(self._clip_paths)}...")
        return normalized

    def _concat_clips(self, clips: list[str]) -> str:
        concat_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="tl_export_"
        )
        for p in clips:
            escaped = p.replace("'", "'\\''")
            concat_file.write(f"file '{escaped}'\n")
        concat_file.close()

        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="tl_cat_")
        out.close()

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file.name,
                "-c", "copy",
                out.name,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                _cat_codec = get_codec_args(self._codec) if self._codec else _codec_args()
                cmd2 = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file.name,
                    *_cat_codec,
                    "-c:a", "aac", "-b:a", "192k",
                    *_pix_fmt(_enc_name()),
                    out.name,
                ]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
                if result2.returncode != 0:
                    raise RuntimeError("Concat encoding failed")
        finally:
            Path(concat_file.name).unlink(missing_ok=True)

        return out.name

    def _apply_rife(self, input_path: str) -> str:
        multiplier = 2 if self._rife_mode == "x2" else 4
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="tl_rife_")
        out.close()
        try:
            from supremediffusion.postprocessing import PostProcessor
            pp = PostProcessor()
            pp.temporal_upscale(input_path, out.name, multiplier=multiplier)
            return out.name
        except ImportError:
            logger.warning("PostProcessor not available, skipping RIFE")
            return input_path
        except Exception:
            logger.warning("RIFE failed, returning un-upsampled video", exc_info=True)
            return input_path
        finally:
            # Explicitly free RIFE model from GPU
            try:
                del pp  # noqa: F821
            except NameError:
                pass
            self._cleanup_gpu()
