"""Video hue/saturation adjustment — manual sliders or reference-frame matching."""

from __future__ import annotations

import logging
import subprocess

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _probe(path: str) -> dict:
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


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float [0,1] array to HSV float (H in degrees, S/V in [0,1])."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    mask = diff > 0
    rm = mask & (maxc == r)
    gm = mask & (maxc == g) & ~rm
    bm = mask & ~rm & ~gm
    h[rm] = (60.0 * ((g[rm] - b[rm]) / diff[rm])) % 360.0
    h[gm] = (60.0 * ((b[gm] - r[gm]) / diff[gm]) + 120.0) % 360.0
    h[bm] = (60.0 * ((r[bm] - g[bm]) / diff[bm]) + 240.0) % 360.0

    s = np.where(maxc > 0, diff / maxc, 0.0)
    return np.stack([h, s, maxc], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV float (H degrees, S/V [0,1]) to RGB float [0,1]."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = h % 360.0
    c = v * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c

    hi = (h / 60.0).astype(np.int32) % 6
    rgb = np.zeros_like(hsv)
    for i, (r, g, b) in enumerate([(c, x, 0), (x, c, 0), (0, c, x),
                                    (0, x, c), (x, 0, c), (c, 0, x)]):
        mask = hi == i
        rgb[:, :, 0] += np.where(mask, r, 0)
        rgb[:, :, 1] += np.where(mask, g, 0)
        rgb[:, :, 2] += np.where(mask, b, 0)
    rgb += m[:, :, np.newaxis]
    return np.clip(rgb, 0.0, 1.0)


def preview_hue_sat(
    image_path: str,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    reference_path: str | None = None,
    strength: float = 1.0,
) -> Image.Image:
    """Apply hue/sat adjustment to a single image and return the result.

    If reference_path is given, matches hue/saturation stats to the reference
    (ignoring the manual hue_shift/saturation sliders).
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64) / 255.0
    hsv = _rgb_to_hsv(rgb)

    if reference_path:
        ref = Image.open(reference_path).convert("RGB")
        ref_rgb = np.array(ref, dtype=np.float64) / 255.0
        ref_hsv = _rgb_to_hsv(ref_rgb)
        hsv = _match_hsv(hsv, ref_hsv, strength)
    else:
        hsv = _apply_manual(hsv, hue_shift, saturation, strength)

    result_rgb = np.clip(_hsv_to_rgb(hsv) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result_rgb)


def _match_hsv(frame_hsv: np.ndarray, ref_hsv: np.ndarray, strength: float) -> np.ndarray:
    """Match frame's hue/saturation to reference using circular mean for hue."""
    out = frame_hsv.copy()

    # Circular hue mean
    frame_h_rad = np.deg2rad(frame_hsv[:, :, 0])
    ref_h_rad = np.deg2rad(ref_hsv[:, :, 0])
    frame_h_mean = np.arctan2(np.sin(frame_h_rad).mean(), np.cos(frame_h_rad).mean())
    ref_h_mean = np.arctan2(np.sin(ref_h_rad).mean(), np.cos(ref_h_rad).mean())
    h_shift = np.rad2deg(ref_h_mean - frame_h_mean)
    out[:, :, 0] = (frame_hsv[:, :, 0] + h_shift * strength) % 360.0

    # Saturation: mean/std transfer
    src_s_mean = frame_hsv[:, :, 1].mean()
    src_s_std = frame_hsv[:, :, 1].std()
    ref_s_mean = ref_hsv[:, :, 1].mean()
    ref_s_std = ref_hsv[:, :, 1].std()
    if src_s_std > 0:
        corrected_s = (frame_hsv[:, :, 1] - src_s_mean) * (ref_s_std / src_s_std) + ref_s_mean
    else:
        corrected_s = np.full_like(frame_hsv[:, :, 1], ref_s_mean)
    out[:, :, 1] = np.clip(
        frame_hsv[:, :, 1] * (1 - strength) + corrected_s * strength, 0, 1
    )
    return out


def _apply_manual(hsv: np.ndarray, hue_shift: float, saturation: float, strength: float) -> np.ndarray:
    out = hsv.copy()
    if hue_shift != 0:
        out[:, :, 0] = (hsv[:, :, 0] + hue_shift * strength) % 360.0
    if saturation != 1.0:
        adjusted = np.clip(hsv[:, :, 1] * saturation, 0, 1)
        out[:, :, 1] = hsv[:, :, 1] * (1 - strength) + adjusted * strength
    return out


def hue_saturation_video(
    video_path: str,
    output_path: str,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    reference_path: str | None = None,
    strength: float = 1.0,
    progress_callback=None,
    color_profile=None,
) -> str:
    """Adjust hue/saturation of every frame — manual or reference-matched."""
    info = _probe(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    expected_bytes = h * w * 3

    ref_hsv = None
    if reference_path:
        ref = Image.open(reference_path).convert("RGB")
        ref_rgb = np.array(ref, dtype=np.float64) / 255.0
        ref_hsv = _rgb_to_hsv(ref_rgb)

    # stderr must NOT be a pipe: nothing drains it during the frame loop, so
    # ffmpeg blocks once the 64KB pipe buffer fills and the chain deadlocks.
    decode = subprocess.Popen(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    from supremediffusion.utils.video import _color_args
    encode = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "slow",
         *_color_args(color_profile, encoder="libx264"),
         str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    total = info.get("num_frames", 0) or 1

    try:
        while True:
            raw = decode.stdout.read(expected_bytes)
            if len(raw) < expected_bytes:
                break

            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            rgb = frame_np.astype(np.float64) / 255.0
            hsv = _rgb_to_hsv(rgb)

            if ref_hsv is not None:
                hsv = _match_hsv(hsv, ref_hsv, strength)
            else:
                hsv = _apply_manual(hsv, hue_shift, saturation, strength)

            out = np.clip(_hsv_to_rgb(hsv) * 255, 0, 255).astype(np.uint8)
            encode.stdin.write(out.tobytes())

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

    logger.info("Hue/saturation adjusted %d frames → %s", frame_idx, output_path)
    return output_path
