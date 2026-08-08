"""Video brightness/contrast adjustment — manual sliders or reference-frame matching."""

from __future__ import annotations

import logging
import subprocess

import numpy as np
from PIL import Image, ImageEnhance

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


def _adjust_L(frame_lab, ref_L_mean, ref_L_std, strength):
    """Match L channel stats to reference."""
    corrected = frame_lab.copy()
    src_mean = frame_lab[:, :, 0].mean()
    src_std = frame_lab[:, :, 0].std()
    if src_std == 0:
        corrected[:, :, 0] = ref_L_mean
    else:
        corrected[:, :, 0] = (
            (frame_lab[:, :, 0] - src_mean) * (ref_L_std / src_std) + ref_L_mean
        )
    # Blend
    corrected[:, :, 0] = frame_lab[:, :, 0] * (1 - strength) + corrected[:, :, 0] * strength
    return corrected


def _apply_manual_bc(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    """Apply manual brightness/contrast using PIL enhancers.

    brightness: 1.0 = original, <1 darker, >1 brighter
    contrast: 1.0 = original, <1 less contrast, >1 more contrast
    """
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def preview_brightness_contrast(
    image_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    reference_path: str | None = None,
    strength: float = 1.0,
) -> Image.Image:
    """Preview brightness/contrast on a single image."""
    img = Image.open(image_path).convert("RGB")

    if reference_path:
        from skimage import color as skcolor
        ref = Image.open(reference_path).convert("RGB")
        ref_np = np.asarray(ref, dtype=np.float64) / 255.0
        ref_lab = skcolor.rgb2lab(ref_np)
        ref_L_mean = ref_lab[:, :, 0].mean()
        ref_L_std = ref_lab[:, :, 0].std()

        frame_np = np.asarray(img, dtype=np.float64) / 255.0
        frame_lab = skcolor.rgb2lab(frame_np)
        corrected_lab = _adjust_L(frame_lab, ref_L_mean, ref_L_std, strength)
        result_rgb = np.clip(skcolor.lab2rgb(corrected_lab) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_rgb)
    else:
        return _apply_manual_bc(img, brightness, contrast)


def brightness_contrast_video(
    video_path: str,
    output_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    reference_path: str | None = None,
    strength: float = 1.0,
    progress_callback=None,
    color_profile=None,
) -> str:
    """Adjust brightness/contrast — manual or reference-matched.

    Manual mode: brightness/contrast multipliers (1.0 = no change).
    Reference mode: matches L channel statistics to reference image.
    """
    info = _probe(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    expected_bytes = h * w * 3

    ref_L_mean = ref_L_std = None
    if reference_path:
        from skimage import color as skcolor
        ref = Image.open(reference_path).convert("RGB")
        ref_np = np.asarray(ref, dtype=np.float64) / 255.0
        ref_lab = skcolor.rgb2lab(ref_np)
        ref_L_mean = ref_lab[:, :, 0].mean()
        ref_L_std = ref_lab[:, :, 0].std()

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

            if reference_path and ref_L_mean is not None:
                from skimage import color as skcolor
                frame_f = frame_np.astype(np.float64) / 255.0
                frame_lab = skcolor.rgb2lab(frame_f)
                corrected_lab = _adjust_L(frame_lab, ref_L_mean, ref_L_std, strength)
                out = np.clip(skcolor.lab2rgb(corrected_lab) * 255, 0, 255).astype(np.uint8)
            else:
                img = Image.fromarray(frame_np, "RGB")
                img = _apply_manual_bc(img, brightness, contrast)
                out = np.array(img)

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

    logger.info("Brightness/contrast adjusted %d frames → %s", frame_idx, output_path)
    return output_path
