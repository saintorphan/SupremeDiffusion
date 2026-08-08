"""Standalone video color correction using Lab color transfer.

Reads every frame of a video, matches its Lab statistics to a reference
image, and writes a corrected video.  Can be used as a library function
or run directly from the command line:

    python -m supremediffusion.utils.color_correct \\
        --video /path/to/stitched.mp4 \\
        --reference /path/to/first_frame.png \\
        --strength 1.0 \\
        --output /path/to/corrected.mp4
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color-match method registry
# ---------------------------------------------------------------------------
#
# Methods dispatched by ``apply_color_match()``. The registry lets the rest
# of the codebase (in-pipeline drift correction, post-hoc CC video, timeline
# Match Grade) share one implementation without each reimplementing the math.
#
# Methods marked (color-matcher) call the upstream library by KJ — same code
# path as ComfyUI-KJNodes/ColorMatch.

# Default: mean-only Lab (the SVI drift-resistant choice — adds shifts only,
# no std multiplier so error doesn't compound across chained continuations).
COLOR_MATCH_METHODS = (
    "mean-only-lab",      # ours — default for in-pipeline / SVI inter-window
    "reinhard",           # color-matcher — full mean+std Lab
    "mkl",                # color-matcher — Monge-Kantorovich Linearization
    "mvgd",               # color-matcher — Multi-Variate Gaussian Distribution
    "hm",                 # color-matcher — per-channel histogram matching
    "hm-mvgd-hm",         # color-matcher — compound (paper recommends as best general)
    "hm-mkl-hm",          # color-matcher — fast compound
)

DEFAULT_CC_METHOD = "mean-only-lab"


def _color_match_mean_only_lab(
    target: np.ndarray, reference: np.ndarray, strength: float,
) -> np.ndarray:
    """Mean-only Lab transfer (drift-resistant)."""
    from skimage import color as skcolor

    ref_lab = skcolor.rgb2lab(np.clip(reference, 0.0, 1.0))
    ref_means = [ref_lab[:, :, j].mean() for j in range(3)]
    tgt_lab = skcolor.rgb2lab(np.clip(target, 0.0, 1.0))
    out_lab = tgt_lab.copy()
    for j in range(3):
        out_lab[:, :, j] += ref_means[j] - tgt_lab[:, :, j].mean()
    out_rgb = np.clip(skcolor.lab2rgb(out_lab), 0.0, 1.0)
    return (1.0 - strength) * target + strength * out_rgb


def _color_match_via_color_matcher(
    target: np.ndarray, reference: np.ndarray, method: str, strength: float,
) -> np.ndarray:
    """Dispatch to the upstream color-matcher library.

    Both inputs must be float arrays in [0, 1] of shape (H, W, 3) or
    (N, H, W, 3). Returns the same shape.
    """
    from color_matcher import ColorMatcher

    cm = ColorMatcher()
    if target.ndim == 3:
        out = cm.transfer(src=target, ref=reference, method=method)
        out = np.clip(out, 0.0, 1.0)
        return (1.0 - strength) * target + strength * out

    # Batch path — color-matcher operates per-frame
    out_frames = np.empty_like(target)
    for i in range(target.shape[0]):
        try:
            out = cm.transfer(src=target[i], ref=reference, method=method)
            out = np.clip(out, 0.0, 1.0)
            out_frames[i] = (1.0 - strength) * target[i] + strength * out
        except Exception as exc:
            logger.warning(
                "color-matcher %s failed on frame %d: %s — passing through",
                method, i, exc,
            )
            out_frames[i] = target[i]
    return out_frames


def apply_color_match(
    target: np.ndarray,
    reference: np.ndarray,
    method: str = DEFAULT_CC_METHOD,
    strength: float = 1.0,
) -> np.ndarray:
    """Apply the configured color-match method.

    Parameters
    ----------
    target:
        Target image(s) as float ``[0, 1]`` array — shape ``(H, W, 3)`` for
        a single frame or ``(N, H, W, 3)`` for a batch.
    reference:
        Reference image as float ``[0, 1]`` array of shape ``(H, W, 3)``.
    method:
        Method key from :data:`COLOR_MATCH_METHODS`. Falls back to mean-only
        Lab if an unknown method is passed.
    strength:
        Blend factor — ``0.0`` = pass-through, ``1.0`` = full correction.
    """
    if strength <= 0.0:
        return target
    strength = min(max(strength, 0.0), 1.0)

    if method == "mean-only-lab" or method not in COLOR_MATCH_METHODS:
        if method not in COLOR_MATCH_METHODS:
            logger.warning("Unknown CC method '%s' — using mean-only-lab", method)
        if target.ndim == 3:
            return _color_match_mean_only_lab(target, reference, strength)
        # Batch — broadcast mean-only across frames
        out = np.empty_like(target)
        for i in range(target.shape[0]):
            out[i] = _color_match_mean_only_lab(target[i], reference, strength)
        return out

    return _color_match_via_color_matcher(target, reference, method, strength)


def _probe(path: str) -> dict:
    """Minimal ffprobe wrapper."""
    import json

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fps_parts = vs.get("r_frame_rate", "16/1").split("/")
    fps = int(fps_parts[0]) / max(int(fps_parts[1]), 1) if len(fps_parts) == 2 else 16
    return {
        "width": int(vs.get("width", 0)),
        "height": int(vs.get("height", 0)),
        "fps": fps,
        "num_frames": int(vs.get("nb_frames", 0)),
        "duration": float(data.get("format", {}).get("duration", 0)),
    }


def _histogram_match_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match the histogram of *source* to *reference* (both 1-D float arrays)."""
    src_vals, src_unique_idx, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True,
    )
    ref_vals, ref_counts = np.unique(reference.ravel(), return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1]
    ref_cdf = np.cumsum(ref_counts).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # For each source CDF value, find the closest reference CDF value
    matched = np.interp(src_cdf, ref_cdf, ref_vals)
    return matched[src_unique_idx].reshape(source.shape)


def color_correct_video(
    video_path: str,
    reference_path: str,
    output_path: str,
    strength: float = 1.0,
    progress_callback=None,
    histogram_match: bool = False,
    color_profile=None,
    method: Optional[str] = None,
) -> str:
    """Apply Lab color correction to every frame of *video_path*.

    Parameters
    ----------
    video_path : str
        Input video to correct.
    reference_path : str
        Image whose colour palette should be matched (e.g. the original
        first frame of the very first generation).
    output_path : str
        Where to write the corrected video.
    strength : float
        Blend factor 0.0 (no correction) to 1.0 (full correction).
    progress_callback : callable, optional
        ``callback(fraction, description)`` called per frame.
    histogram_match : bool
        If True, use full histogram matching instead of mean/std transfer.
        Much more aggressive — forces the exact colour distribution to match.
        Ignored when ``method`` is supplied.
    method : str, optional
        Override the legacy ``histogram_match`` flag with an explicit method
        from :data:`COLOR_MATCH_METHODS`. When set, dispatches through the
        shared color-matcher registry.

    Returns the output path.
    """
    from skimage import color as skcolor

    # When method is supplied, route through the unified dispatcher and
    # bypass the legacy reinhard/HM branches below.
    if method is not None:
        return _color_correct_video_with_method(
            video_path, reference_path, output_path,
            method=method, strength=strength,
            progress_callback=progress_callback,
            color_profile=color_profile,
        )

    # --- Load reference ---
    ref_img = Image.open(reference_path).convert("RGB")
    ref_np = np.asarray(ref_img).astype(np.float64) / 255.0
    ref_np = np.clip(ref_np, 0.0, 1.0)
    ref_lab = skcolor.rgb2lab(ref_np)
    ref_stats = [(ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()) for j in range(3)]

    # --- Probe video ---
    info = _probe(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    expected_bytes = h * w * 3

    # --- Decode all frames via ffmpeg pipe ---
    decode_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-",
    ]
    # stderr must NOT be a pipe here: nothing drains it during the frame loop,
    # so once ffmpeg fills the 64KB pipe buffer it blocks mid-write and the
    # whole decode→match→encode chain deadlocks (hang partway through a clip).
    decode = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # --- Encode pipe ---
    from supremediffusion.utils.video import _color_args, _rgb_encode_filter
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        # Pin range + matrix of the RGB→YUV conversion to the stamped profile —
        # without this the matched frames pick up a BT.601-vs-709 hue shift.
        "-vf", _rgb_encode_filter(color_profile),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    encode = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_idx = 0
    total = info.get("num_frames", 0) or 1

    try:
        while True:
            raw = decode.stdout.read(expected_bytes)
            if len(raw) < expected_bytes:
                break

            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frame_f = frame_np.astype(np.float64) / 255.0

            try:
                frame_lab = skcolor.rgb2lab(frame_f)
                corrected_lab = frame_lab.copy()

                if histogram_match:
                    # Full histogram matching per Lab channel
                    for j in range(3):
                        corrected_lab[:, :, j] = _histogram_match_channel(
                            frame_lab[:, :, j], ref_lab[:, :, j],
                        )
                else:
                    # Mean/std statistical transfer
                    for j in range(3):
                        src_mean = frame_lab[:, :, j].mean()
                        src_std = frame_lab[:, :, j].std()
                        ref_mean, ref_std = ref_stats[j]
                        if src_std == 0:
                            corrected_lab[:, :, j] = ref_mean
                        else:
                            corrected_lab[:, :, j] = (
                                (corrected_lab[:, :, j] - src_mean) * (ref_std / src_std)
                                + ref_mean
                            )

                corrected_rgb = np.clip(skcolor.lab2rgb(corrected_lab), 0.0, 1.0)
                blended = (1.0 - strength) * frame_f + strength * corrected_rgb
                out_frame = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
            except Exception:
                out_frame = frame_np  # pass through on error

            encode.stdin.write(out_frame.tobytes())

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx / total, f"Frame {frame_idx}/{total}")

    finally:
        decode.stdout.close()
        decode.wait()
        encode.stdin.close()
        encode.wait()

    if progress_callback:
        progress_callback(1.0, f"Done — {frame_idx} frames")

    logger.info("Color-corrected %d frames → %s", frame_idx, output_path)
    return output_path


def _color_correct_video_with_method(
    video_path: str,
    reference_path: str,
    output_path: str,
    method: str,
    strength: float = 1.0,
    progress_callback=None,
    color_profile=None,
) -> str:
    """Method-driven version of :func:`color_correct_video` that dispatches
    through :func:`apply_color_match`.

    Wraps the same ffmpeg decode/encode pipe as the legacy path so output
    encoding stays consistent. Each frame is run through the registry —
    color-matcher's compound methods (HM-MVGD-HM etc.) handle bimodal
    palettes much better than pure mean+std.
    """
    # Load reference once
    ref_img = Image.open(reference_path).convert("RGB")
    ref_np = np.asarray(ref_img).astype(np.float32) / 255.0

    info = _probe(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    expected_bytes = h * w * 3

    decode_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet",
        "-",
    ]
    # stderr must NOT be a pipe here: nothing drains it during the frame loop,
    # so once ffmpeg fills the 64KB pipe buffer it blocks mid-write and the
    # whole decode→match→encode chain deadlocks (hang partway through a clip).
    decode = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    from supremediffusion.utils.video import _color_args, _rgb_encode_filter
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        # Pin range + matrix (see color_correct_video) — avoids 601/709 shift.
        "-vf", _rgb_encode_filter(color_profile),
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        *_color_args(color_profile, encoder="libx264"),
        str(output_path),
    ]
    encode = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Resize the reference to match if dimensions differ — color-matcher
    # tolerates mismatched sizes but the per-pixel methods don't need
    # alignment, only stats. Cheap PIL resize is fine.
    if ref_np.shape[0] != h or ref_np.shape[1] != w:
        ref_img_r = ref_img.resize((w, h), Image.LANCZOS)
        ref_np = np.asarray(ref_img_r).astype(np.float32) / 255.0

    frame_idx = 0
    total = info.get("num_frames", 0) or 1

    try:
        while True:
            raw = decode.stdout.read(expected_bytes)
            if len(raw) < expected_bytes:
                break

            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frame_f = frame_np.astype(np.float32) / 255.0
            try:
                out_f = apply_color_match(frame_f, ref_np, method=method, strength=strength)
                out_frame = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)
            except Exception as exc:
                logger.warning("CC method '%s' failed on frame %d: %s — pass-through",
                               method, frame_idx, exc)
                out_frame = frame_np

            encode.stdin.write(out_frame.tobytes())
            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx / total, f"Frame {frame_idx}/{total} ({method})")

    finally:
        decode.stdout.close()
        decode.wait()
        encode.stdin.close()
        encode.wait()

    if progress_callback:
        progress_callback(1.0, f"Done — {frame_idx} frames ({method})")
    logger.info("Color-corrected %d frames via %s → %s", frame_idx, method, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Color-correct a video using a reference frame")
    parser.add_argument("--video", "-v", required=True, help="Input video path")
    parser.add_argument("--reference", "-r", required=True, help="Reference image path")
    parser.add_argument("--output", "-o", help="Output path (default: <video>_cc.mp4)")
    parser.add_argument("--strength", "-s", type=float, default=1.0, help="Correction strength 0-1")
    args = parser.parse_args()

    output = args.output
    if not output:
        p = Path(args.video)
        output = str(p.with_stem(p.stem + "_cc"))

    def _progress(frac, desc):
        print(f"\r  {desc} ({frac*100:.0f}%)", end="", flush=True)

    print(f"Correcting: {args.video}")
    print(f"Reference:  {args.reference}")
    print(f"Strength:   {args.strength}")
    color_correct_video(args.video, args.reference, output, args.strength, _progress)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
