"""FFmpeg wrappers via subprocess for video I/O."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

def _color_args(color_profile=None, encoder: str = "libx264") -> list[str]:
    """Return ``-pix_fmt … -color_range … -colorspace …`` for *color_profile*.

    When *color_profile* is None, the global default (per
    ``GlobalConfig.default_color_profile``) is used, which itself defaults
    to ``bt709_limited`` — the same shape the old hardcoded ``_COLOR_ARGS``
    constant produced.
    """
    if color_profile is None:
        color_profile = _resolve_default_profile()
    return color_profile.encoder_args(encoder)


def _resolve_default_profile():
    """Resolve the global-default ColorProfile (hard fallback bt709_limited)."""
    try:
        from supremediffusion.config.global_config import GlobalConfig
        from supremediffusion.config.color_profile import get_profile
        cfg = GlobalConfig.load()
        return get_profile(getattr(cfg, "default_color_profile", "") or "")
    except Exception:
        from supremediffusion.config.color_profile import get_profile
        return get_profile(None)


def _rgb_encode_filter(color_profile=None) -> str:
    """Return the ``-vf`` filter for encoding full-range RGB input to YUV.

    Pins BOTH the range and the YUV matrix of the RGB→YUV conversion to the
    profile. Without ``out_color_matrix``, swscale converts with its own
    default (BT.601, even for HD) while the encoder stamps the profile's
    colorspace tag — a real hue shift on playback (measured |Δ|≈8/255 on
    saturated colors). Every rawvideo/PNG→video encode must use this.
    """
    if color_profile is None:
        color_profile = _resolve_default_profile()
    out_r = "full" if color_profile.color_range == "pc" else "tv"
    return (
        f"scale=in_range=full:out_range={out_r}"
        f":out_color_matrix={color_profile.colorspace}"
    )


def probe_video(path: str) -> dict:
    """Probe video file and return metadata dict with duration, fps, width, height, num_frames."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # Parse fps from r_frame_rate (e.g. "24000/1001")
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den if den else 0.0

    duration = float(data.get("format", {}).get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # nb_frames may not always be present
    nb_frames = video_stream.get("nb_frames")
    if nb_frames is not None:
        num_frames = int(nb_frames)
    else:
        num_frames = int(duration * fps) if fps > 0 else 0

    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }


def get_video_duration(path: str) -> float:
    """Return video duration in seconds."""
    return probe_video(path)["duration"]


def probe_color_range(path: str) -> str:
    """Return ``"full"`` or ``"tv"`` for *path*.

    Reads ``color_range`` and ``pix_fmt`` from ffprobe. Mirrors the heuristic
    in :func:`sdqt.utils.codec.probe_color_range` — kept here so the backend
    isn't reaching across packages. Defaults to ``"tv"`` (the common case for
    H.264/HEVC encodes) when the input is unlabeled.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_range,pix_fmt",
             "-of", "default=nw=1", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        # Parse by key. ffprobe emits fields in stream order (pix_fmt before
        # color_range), NOT the order requested, so positional parsing would
        # swap them — silently misdetecting every full-range (pc) clip as tv.
        fields: dict[str, str] = {}
        for ln in r.stdout.splitlines():
            k, sep, v = ln.strip().lower().partition("=")
            if sep:
                fields[k] = v
        color_range = fields.get("color_range", "")
        pix_fmt = fields.get("pix_fmt", "")
        if color_range in ("pc", "full"):
            return "full"
        if color_range in ("tv", "limited"):
            return "tv"
        if pix_fmt.startswith("yuvj") or pix_fmt.startswith("rgb"):
            return "full"
        return "tv"
    except Exception:
        return "tv"


def _png_export_filter(video_path: str) -> str:
    """Build a scale filter that converts the video's range to full-range PNG.

    PNGs are always full-range (0–255). Most videos this app produces are
    BT.709 *limited* (16–235), so without a conversion the captured PNG looks
    washed out — blacks at ~0.06, whites at ~0.92 instead of true 0/1.

    Returns a filter string to use as ``-vf`` for PNG export.
    """
    in_range = probe_color_range(video_path)
    return f"scale=in_range={in_range}:out_range=pc:flags=lanczos"


def extract_frames(
    video_path: str,
    output_dir: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> list[str]:
    """Extract frames from video to output_dir. Returns list of frame file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(video_path)
    src_fps = info["fps"]

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
    ]

    # Time-based seeking for start/end frame
    if start_frame > 0 and src_fps > 0:
        cmd.extend(["-ss", str(start_frame / src_fps)])
    if end_frame is not None and src_fps > 0:
        duration = (end_frame - start_frame) / src_fps
        cmd.extend(["-t", str(duration)])

    # Video filters: range conversion (TV→PC) is mandatory so PNG output
    # matches the video visually. Optional fps stacks on top.
    pix_filter = _png_export_filter(str(video_path))
    if fps is not None:
        cmd.extend(["-vf", f"fps={fps},{pix_filter}"])
    else:
        cmd.extend(["-vf", pix_filter])

    output_pattern = str(output_dir / "frame_%06d.png")
    cmd.append(output_pattern)

    subprocess.run(cmd, capture_output=True, check=True)

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    return [str(p) for p in frame_paths]


def extract_single_frame(video_path: str, frame_number: int) -> Image.Image:
    """Extract a single frame by number and return as PIL Image.

    Uses PNG pipe output to avoid color shifts from raw RGB conversion.
    Seeks before input (-ss before -i) for fast keyframe-based seeking.
    """
    import io

    info = probe_video(video_path)
    fps = info["fps"]
    if fps <= 0:
        raise ValueError(f"Cannot determine fps for {video_path}")

    num_frames = info.get("num_frames", frame_number + 1)

    # PNG is full-range (0–255). Convert from the video's actual range so
    # the captured frame matches what the user sees in the video player.
    pix_filter = _png_export_filter(str(video_path))

    for attempt_frame in range(frame_number, max(frame_number - 10, -1), -1):
        timestamp = attempt_frame / fps
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", pix_filter,
            "-f", "image2pipe", "-vcodec", "png", "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and len(result.stdout) > 100:
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")

    # Last resort: grab the very first frame
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", pix_filter,
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    if len(result.stdout) < 100:
        raise ValueError(
            f"Frame extraction failed for all attempts on {video_path} "
            f"(requested frame {frame_number}, total: {num_frames})"
        )
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def frames_to_video(
    frame_dir_or_list: Union[str, list[str]],
    output_path: str,
    fps: int = 16,
    codec: str = "libx264",
    color_profile=None,
) -> None:
    """Encode frames (directory or list of paths) into a video file.

    The output's pix_fmt / color_range / colorspace come from *color_profile*
    (defaults to the global default profile, which defaults to bt709_limited).
    """
    color_args = _color_args(color_profile, encoder=codec)
    rgb_vf = _rgb_encode_filter(color_profile)
    if isinstance(frame_dir_or_list, (str, Path)):
        pattern = str(Path(frame_dir_or_list) / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", pattern,
            "-vf", rgb_vf,
            "-c:v", codec, "-crf", "16", "-preset", "slow",
            *color_args,
            str(output_path),
        ]
    else:
        # Write a concat file for arbitrary frame paths
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for frame_path in frame_dir_or_list:
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {1.0 / fps}\n")
            concat_path = f.name

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-vf", rgb_vf,
            "-c:v", codec, "-crf", "16", "-preset", "slow",
            *color_args,
            "-vsync", "vfr",
            str(output_path),
        ]

    subprocess.run(cmd, capture_output=True, check=True)


def strip_black_frames(frames: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Remove leading and trailing near-black frames from a (N, H, W, 3) array.

    A frame is considered black if its mean pixel value is below *threshold*.
    Always keeps at least one frame.
    """
    if frames.shape[0] <= 1:
        return frames
    means = frames.mean(axis=(1, 2, 3)) if frames.ndim == 4 else frames.reshape(frames.shape[0], -1).mean(axis=1)
    start = 0
    while start < len(means) - 1 and means[start] < threshold:
        start += 1
    end = len(means)
    while end > start + 1 and means[end - 1] < threshold:
        end -= 1
    if start > 0 or end < len(means):
        import logging
        logging.getLogger(__name__).info(
            "Stripped black frames: %d leading, %d trailing (kept %d/%d)",
            start, len(means) - end, end - start, len(means),
        )
    return frames[start:end]


def encode_video(
    frames: np.ndarray,
    output_path: str,
    fps: int = 16,
    codec: str = "libx264",
    color_profile=None,
) -> None:
    """Encode numpy array (N, H, W, 3) uint8 to video via ffmpeg pipe.

    Strips leading/trailing black frames before encoding.
    Tags the output per *color_profile* (defaults to global-default profile)
    so playback decoders interpret the stream consistently.
    """
    frames = strip_black_frames(frames)
    n, h, w, c = frames.shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-vf", _rgb_encode_filter(color_profile),
        "-c:v", codec, "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder=codec),
        str(output_path),
    ]
    import logging as _log
    result = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if result.returncode != 0:
        _log.getLogger(__name__).error("encode_video failed: %s", result.stderr.decode(errors="replace"))
        result.check_returncode()


def generate_static_guidance_video(
    frame_path: str,
    num_frames: int,
    fps: int,
    width: int = 832,
    height: int = 480,
) -> str:
    """Create a guidance video where every frame is the same image.

    Returns the path to a directory of PNG frames (one per frame, all
    identical). This avoids any video encoding colorspace conversion
    that would shift colors between the source frame and the guidance.

    If the caller expects a video file, the returned path is actually
    a lossless mp4 encoded via ffmpeg with PNG codec to preserve RGB.
    """
    import logging
    from PIL import Image

    _logger = logging.getLogger(__name__)

    # Load and resize the source frame with PIL (no colorspace conversion)
    img = Image.open(frame_path).convert("RGB")
    if img.width != width or img.height != height:
        img = img.resize((width, height), resample=Image.Resampling.LANCZOS)

    # Write the frame as a temp PNG — then use ffmpeg to loop it
    # with PNG codec (lossless, no YUV conversion)
    tmp_frame = tempfile.mktemp(suffix=".png", prefix="sd_guidance_frame_")
    img.save(tmp_frame, "PNG")

    duration = num_frames / max(fps, 1)
    out_path = tempfile.mktemp(suffix=".mp4", prefix="sd_guidance_")

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", tmp_frame,
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "png",
        "-an",
        out_path,
    ]

    _logger.info("Generating static guidance video (PNG codec): %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Clean up temp frame
    try:
        os.remove(tmp_frame)
    except OSError:
        pass

    if result.returncode != 0:
        _logger.error("ffmpeg failed: %s", result.stderr)
        raise RuntimeError(f"Failed to generate static guidance video: {result.stderr}")

    _logger.info("Static guidance video created: %s (%.1fs @ %dfps, PNG codec)", out_path, duration, fps)
    return out_path


def crop_video(
    input_path: str,
    output_path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    color_profile=None,
) -> None:
    """Crop video to region (x, y, w, h) using ffmpeg with color preservation."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def trim_video(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
) -> None:
    """Trim video between start_sec and end_sec using ffmpeg (stream copy, keyframe-aligned)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(input_path),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def trim_video_precise(
    input_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    color_profile=None,
) -> None:
    """Frame-accurate trim via re-encode (slower but exact)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def concat_videos_crossfade(
    video_a: str,
    video_b: str,
    output_path: str,
    overlap_frames: int = 6,
    color_profile=None,
) -> None:
    """Concatenate two videos with a short crossfade at the seam.

    *overlap_frames* is the number of frames used for the crossfade
    transition.  The last *overlap_frames* of video_a blend into the
    first *overlap_frames* of video_b, hiding any colour discontinuity.
    """
    info_a = probe_video(video_a)
    info_b = probe_video(video_b)
    fps = info_a.get("fps", 16)
    w = info_a.get("width", 832)
    h = info_a.get("height", 480)

    xfade_duration = overlap_frames / max(fps, 1)
    # offset = duration_a - crossfade_duration
    dur_a = info_a.get("duration", 5.0)
    offset = max(dur_a - xfade_duration, 0)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_a),
        "-i", str(video_b),
        "-filter_complex",
        f"[0:v]scale={w}:{h}:flags=lanczos,fps={fps},setsar=1[v0];"
        f"[1:v]scale={w}:{h}:flags=lanczos,fps={fps},setsar=1[v1];"
        f"[v0][v1]xfade=transition=fade:duration={xfade_duration:.4f}:offset={offset:.4f}[outv]",
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "a", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except Exception:
        return False


def _videos_compatible(info_a: dict, info_b: dict) -> bool:
    """Check if two videos can be concatenated without re-encoding."""
    return (
        info_a.get("width") == info_b.get("width")
        and info_a.get("height") == info_b.get("height")
        and abs(info_a.get("fps", 0) - info_b.get("fps", 0)) < 0.5
    )


def concat_videos(
    video_a: str,
    video_b: str,
    output_path: str,
    color_profile=None,
) -> None:
    """Concatenate two videos (a then b) with color-preserving encoding.

    When the videos share resolution and fps, uses the concat demuxer
    with stream copy (no re-encoding) to avoid any quality or color loss.
    Falls back to re-encoding with explicit colorspace preservation when
    the streams are incompatible.
    """
    info_a = probe_video(video_a)
    info_b = probe_video(video_b)

    if _videos_compatible(info_a, info_b):
        # Stream copy via concat demuxer — zero quality loss
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(f"file '{Path(video_a).resolve()}'\n")
            f.write(f"file '{Path(video_b).resolve()}'\n")
            list_path = f.name
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        Path(list_path).unlink(missing_ok=True)
    else:
        # Re-encode with color preservation (and audio if present)
        w, h = info_a.get("width", 832), info_a.get("height", 480)
        fps = info_a.get("fps", 16)
        has_audio = _has_audio_stream(video_a) or _has_audio_stream(video_b)
        audio_filter = (
            f"[0:a]aresample=async=1[a0];[1:a]aresample=async=1[a1];"
            f"[a0][a1]concat=n=2:v=0:a=1[outa]"
        ) if has_audio else ""
        video_filter = (
            f"[0:v]scale={w}:{h}:flags=lanczos,fps={fps},setsar=1[v0];"
            f"[1:v]scale={w}:{h}:flags=lanczos,fps={fps},setsar=1[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[outv]"
        )
        full_filter = f"{video_filter};{audio_filter}" if audio_filter else video_filter
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_a),
            "-i", str(video_b),
            "-filter_complex", full_filter,
            "-map", "[outv]",
        ]
        if has_audio:
            cmd.extend(["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"])
        cmd.extend([
            "-c:v", "libx264", "-crf", "16", "-preset", "slow",
            *_color_args(color_profile, encoder="libx264"),
            str(output_path),
        ])
        subprocess.run(cmd, capture_output=True, check=True)


def concat_videos_multi(paths: list[str], output_path: str, color_profile=None) -> None:
    """Concatenate multiple videos sequentially.

    Uses stream copy when all inputs are compatible, re-encodes otherwise.
    """
    if len(paths) < 2:
        if paths:
            import shutil
            shutil.copy2(paths[0], output_path)
        return

    # Check compatibility across all inputs
    infos = [probe_video(p) for p in paths]
    compatible = all(_videos_compatible(infos[0], info) for info in infos[1:])

    if compatible:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for p in paths:
                f.write(f"file '{Path(p).resolve()}'\n")
            list_path = f.name
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        Path(list_path).unlink(missing_ok=True)
    else:
        w, h = infos[0].get("width", 832), infos[0].get("height", 480)
        fps = infos[0].get("fps", 16)
        inputs = []
        filter_parts = []
        for i, p in enumerate(paths):
            inputs.extend(["-i", str(p)])
            filter_parts.append(
                f"[{i}:v]scale={w}:{h}:flags=lanczos,fps={fps},setsar=1[v{i}]"
            )
        v_labels = "".join(f"[v{i}]" for i in range(len(paths)))
        filter_parts.append(f"{v_labels}concat=n={len(paths)}:v=1:a=0[outv]")

        cmd = ["ffmpeg", "-y"] + inputs
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]",
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                *_color_args(color_profile, encoder="libx264"), str(output_path)]
        subprocess.run(cmd, capture_output=True, check=True)


def write_video_metadata(video_path: str, metadata: dict) -> bool:
    """Embed a JSON metadata dict into the MP4 comment field.

    Re-muxes without re-encoding. Returns True on success.
    """
    meta_json = json.dumps(metadata, ensure_ascii=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-metadata", f"comment={meta_json}",
            "-c", "copy", tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            import shutil
            shutil.move(tmp.name, video_path)
            return True
    except Exception:
        pass
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return False


def read_video_metadata(video_path: str) -> dict | None:
    """Read JSON metadata from the MP4 comment field. Returns dict or None."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        comment = data.get("format", {}).get("tags", {}).get("comment", "")
        if not comment:
            return None
        return json.loads(comment)
    except (json.JSONDecodeError, Exception):
        return None


def trim_boundary_frames(input_path: str, output_path: str, color_profile=None) -> None:
    """Remove the first and last frame from a video.

    Used by Longshot assembly to avoid duplicate boundary frames when
    gapfill videos share their first/last frame with adjacent chunks.
    """
    info = probe_video(input_path)
    fps = info.get("fps", 16)
    num_frames = info.get("num_frames", 0)
    if num_frames <= 2:
        import shutil
        shutil.copy2(input_path, output_path)
        return

    # Trim 1 frame from start and 1 from end
    start_time = 1.0 / fps
    end_time = (num_frames - 1) / fps
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", f"{start_time:.6f}",
        "-to", f"{end_time:.6f}",
        "-c:v", "libx264", "-crf", "16", "-preset", "fast",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
