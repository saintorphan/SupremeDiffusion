"""Shared grade computation -- color analysis, correction scaling, and effect conversion.

Extracted from timeline_match_grade.py so both the timeline mixin and
the Color Correction tab can reuse the same analysis logic.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping: correction keys -> effect type
# ---------------------------------------------------------------------------

CORRECTION_CATEGORIES: dict[str, list[str]] = {
    "brightness_contrast": ["brightness", "contrast"],
    "color_balance": ["rgb_gain"],
    "color_temperature": ["temperature", "tint"],
    "hue_sat": ["saturation"],
    "levels": ["black_point", "white_point", "gamma"],
    "shadows_highlights": ["shadows", "highlights"],
}

# Human-readable labels for the categories
CATEGORY_LABELS: dict[str, str] = {
    "brightness_contrast": "Brightness / Contrast",
    "color_balance": "Color Balance",
    "color_temperature": "Color Temperature",
    "hue_sat": "Saturation",
    "levels": "Levels",
    "shadows_highlights": "Shadows / Highlights",
}

# Identity (neutral) values for each correction key
_IDENTITY: dict[str, float | list[float]] = {
    "brightness": 0.0,
    "contrast": 1.0,
    "rgb_gain": [1.0, 1.0, 1.0],
    "temperature": 0.0,
    "tint": 0.0,
    "saturation": 1.0,
    "black_point": 0.0,
    "white_point": 255.0,
    "gamma": 1.0,
    "shadows": 0.0,
    "highlights": 0.0,
}


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame_from_path(
    path: str, frame_number: int = 0,
) -> "np.ndarray | None":
    """Extract a single frame as a numpy RGB array from a media file.

    For images (.png, .jpg, etc.) *frame_number* is ignored.
    For videos the requested frame is extracted via ffmpeg.
    """
    import numpy as np

    p = Path(path)
    if not p.is_file():
        return None

    suffix = p.suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    if suffix in image_exts:
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return np.array(img)
        except Exception:
            return None

    # Video -- extract via ffmpeg
    try:
        from supremediffusion.utils.video import extract_single_frame
        img = extract_single_frame(path, frame_number)
        return np.array(img) if img is not None else None
    except Exception:
        return None


def extract_frame_from_clip(clip, which: str = "first") -> "np.ndarray | None":
    """Extract a frame from a timeline clip, with effects applied if any.

    If the clip has active effects, renders through ffmpeg so the returned
    frame matches what the user sees on screen.
    """
    import numpy as np
    try:
        from supremediffusion.utils.video import extract_single_frame, probe_video

        path = clip.path
        if not Path(path).is_file():
            return None

        info = probe_video(path)
        fps = info.get("fps", 16)
        num_frames = info.get("num_frames", 1)

        if which == "last":
            base_ts = clip.media_offset + clip.duration
            frame_num = int(base_ts * fps) - 1
        else:
            frame_num = int(clip.media_offset * fps)

        frame_num = max(0, min(frame_num, num_frames - 1))

        effects = getattr(clip, "effects", [])
        active = [fx for fx in effects if fx.get("enabled", True)]

        # Try up to 8 frames back/forward to skip black frames
        step = -1 if which == "last" else 1
        for offset in range(8):
            fn = frame_num + offset * step
            fn = max(0, min(fn, num_frames - 1))
            if active:
                ts = fn / max(fps, 1)
                frame = render_frame_with_effects(path, active, ts)
            else:
                img = extract_single_frame(path, fn)
                frame = np.array(img) if img is not None else None
            if frame is not None and frame.mean() > 5.0:
                return frame

        # All frames were black — return whatever we got
        if active:
            ts = frame_num / max(fps, 1)
            return render_frame_with_effects(path, active, ts)
        img = extract_single_frame(path, frame_num)
        return np.array(img) if img is not None else None
    except Exception:
        return None


def crop_region(
    frame: "np.ndarray", crop: tuple[float, float, float, float] | None,
) -> "np.ndarray":
    """Crop frame to normalized (x, y, w, h) rect. Returns full frame if None."""
    if crop is None:
        return frame
    x, y, w, h = crop
    fh, fw = frame.shape[:2]
    x0 = max(0, int(x * fw))
    y0 = max(0, int(y * fh))
    x1 = min(fw, int((x + w) * fw))
    y1 = min(fh, int((y + h) * fh))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame
    return frame[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Grade computation (Color Correction tab + Match Grade)
# ---------------------------------------------------------------------------

def compute_grade(ref: "np.ndarray", target: "np.ndarray") -> dict | None:
    """Compute color correction params to make *target* look like *ref*.

    Both inputs should be numpy uint8 RGB arrays (H, W, 3).
    Returns a dict of correction keys or None if no correction needed.
    """
    import numpy as np

    ref_f = ref.astype(np.float32) / 255.0
    tgt_f = target.astype(np.float32) / 255.0

    def _skin_chroma(img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask = (r > g) & (g > b) & (r > 0.25) & ((r - b) > 0.03)
        if mask.sum() < 200:
            return None, 0
        skin = img[mask]
        lum = np.clip(
            skin[:, 0] * 0.2126 + skin[:, 1] * 0.7152 + skin[:, 2] * 0.0722,
            0.01, None,
        )
        return np.array([
            (skin[:, 0] / lum).mean(),
            (skin[:, 1] / lum).mean(),
            (skin[:, 2] / lum).mean(),
        ]), mask.sum()

    ref_chroma, ref_cnt = _skin_chroma(ref_f)
    tgt_chroma, tgt_cnt = _skin_chroma(tgt_f)
    use_chroma = ref_chroma is not None and tgt_chroma is not None

    def _weighted_mean(img):
        lum = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.7152 + img[:, :, 2] * 0.0722
        w = np.clip((lum - 0.05) / 0.1, 0, 1) * np.clip((0.95 - lum) / 0.1, 0, 1)
        w_sum = w.sum()
        if w_sum < 1.0:
            return img.mean(axis=(0, 1))
        return np.array([
            (img[:, :, c] * w).sum() / w_sum for c in range(3)
        ])

    ref_mean = _weighted_mean(ref_f)
    tgt_mean = _weighted_mean(tgt_f)

    lum_weights = np.array([0.2126, 0.7152, 0.0722])
    ref_lum_mean = float((ref_mean * lum_weights).sum())
    tgt_lum_mean = float((tgt_mean * lum_weights).sum())
    ref_lum_std = float((ref_f * lum_weights).sum(axis=2).std())
    tgt_lum_std = float((tgt_f * lum_weights).sum(axis=2).std())

    corrections: dict = {}

    ref_lum = (ref_f * lum_weights).sum(axis=2).ravel()
    tgt_lum = (tgt_f * lum_weights).sum(axis=2).ravel()
    ref_bp = float(np.percentile(ref_lum, 2))
    ref_wp = float(np.percentile(ref_lum, 98))
    tgt_bp = float(np.percentile(tgt_lum, 2))
    tgt_wp = float(np.percentile(tgt_lum, 98))

    bp_diff = abs(ref_bp - tgt_bp) * 255
    wp_diff = abs(ref_wp - tgt_wp) * 255
    has_levels = bp_diff > 10 or wp_diff > 10

    if has_levels:
        corrections["black_point"] = float(np.clip(ref_bp * 255, 0, 80))
        corrections["white_point"] = float(np.clip(ref_wp * 255, 180, 255))

        ref_range = max(ref_wp - ref_bp, 0.01)
        tgt_range = max(tgt_wp - tgt_bp, 0.01)
        ref_mid = (ref_lum_mean - ref_bp) / ref_range
        tgt_mid = (tgt_lum_mean - tgt_bp) / tgt_range
        if tgt_mid > 0.01 and ref_mid > 0.01:
            gamma = math.log(max(ref_mid, 0.01)) / math.log(max(tgt_mid, 0.01))
            gamma = float(np.clip(gamma, 0.6, 1.6))
            if abs(gamma - 1.0) > 0.1:
                corrections["gamma"] = gamma

    brightness_delta = ref_lum_mean - tgt_lum_mean
    brightness_threshold = 0.08 if has_levels else 0.04
    if abs(brightness_delta) > brightness_threshold:
        dampening = 0.4 if has_levels else 0.7
        corrections["brightness"] = float(np.clip(brightness_delta * dampening, -0.3, 0.3))

    if tgt_lum_std > 0.01:
        contrast_ratio = ref_lum_std / tgt_lum_std
        if abs(contrast_ratio - 1.0) > 0.1:
            dampened = 1.0 + (contrast_ratio - 1.0) * 0.65
            corrections["contrast"] = float(np.clip(dampened, 0.6, 1.6))

    if use_chroma:
        r_gain = float(ref_chroma[0] / max(tgt_chroma[0], 0.01))
        g_gain = float(ref_chroma[1] / max(tgt_chroma[1], 0.01))
        b_gain = float(ref_chroma[2] / max(tgt_chroma[2], 0.01))
        logger.info(
            "Chroma gains (skin): R=%.4f G=%.4f B=%.4f (ref=%d tgt=%d px)",
            r_gain, g_gain, b_gain, ref_cnt, tgt_cnt,
        )
        # Skin-based chroma is already luminance-normalized, so use
        # gentler dampening to avoid overcorrecting the whole image
        # based on a skin-only sample.
        dampening = 0.5
    else:
        r_gain = float(ref_mean[0] / max(tgt_mean[0], 0.01))
        g_gain = float(ref_mean[1] / max(tgt_mean[1], 0.01))
        b_gain = float(ref_mean[2] / max(tgt_mean[2], 0.01))
        dampening = 0.65

    r_gain = 1.0 + (float(np.clip(r_gain, 0.7, 1.5)) - 1.0) * dampening
    g_gain = 1.0 + (float(np.clip(g_gain, 0.7, 1.5)) - 1.0) * dampening
    b_gain = 1.0 + (float(np.clip(b_gain, 0.7, 1.5)) - 1.0) * dampening
    max_dev = max(abs(r_gain - 1.0), abs(g_gain - 1.0), abs(b_gain - 1.0))
    has_rgb_gain = max_dev > 0.005
    if has_rgb_gain:
        corrections["rgb_gain"] = [r_gain, g_gain, b_gain]

    if not has_rgb_gain:
        r_delta = float(ref_mean[0] - tgt_mean[0])
        g_delta = float(ref_mean[1] - tgt_mean[1])
        b_delta = float(ref_mean[2] - tgt_mean[2])
        temp = float(np.clip((r_delta - b_delta) * 60, -30, 30))
        if abs(temp) > 2.0:
            corrections["temperature"] = temp
        tint = float(np.clip((g_delta - (r_delta + b_delta) / 2.0) * 60, -30, 30))
        if abs(tint) > 2.0:
            corrections["tint"] = tint

    ref_sat = float(ref_f.std(axis=2).mean())
    tgt_sat = float(tgt_f.std(axis=2).mean())
    if tgt_sat > 0.001:
        sat_ratio = ref_sat / tgt_sat
        if abs(sat_ratio - 1.0) > 0.05:
            dampened = 1.0 + (sat_ratio - 1.0) * 0.75
            corrections["saturation"] = float(np.clip(dampened, 0.3, 2.0))

    if not has_levels:
        ref_shadow_mean = float(ref_lum[ref_lum < 0.3].mean()) if (ref_lum < 0.3).any() else 0.15
        tgt_shadow_mean = float(tgt_lum[tgt_lum < 0.3].mean()) if (tgt_lum < 0.3).any() else 0.15
        ref_high_mean = float(ref_lum[ref_lum > 0.7].mean()) if (ref_lum > 0.7).any() else 0.85
        tgt_high_mean = float(tgt_lum[tgt_lum > 0.7].mean()) if (tgt_lum > 0.7).any() else 0.85

        shadow_delta = ref_shadow_mean - tgt_shadow_mean
        highlight_delta = ref_high_mean - tgt_high_mean

        if abs(shadow_delta) > 0.06:
            corrections["shadows"] = float(np.clip(shadow_delta * 0.5, -0.3, 0.3))
        if abs(highlight_delta) > 0.06:
            corrections["highlights"] = float(np.clip(highlight_delta * 0.5, -0.3, 0.3))

    if not corrections:
        return None
    return corrections


# ---------------------------------------------------------------------------
# Correction scaling
# ---------------------------------------------------------------------------

def scale_corrections(
    corrections: dict,
    category_mask: dict[str, bool],
    category_strengths: dict[str, float],
    overall_strength: float,
) -> dict:
    """Filter corrections by enabled categories and scale by strengths."""
    result: dict = {}

    for cat_name, keys in CORRECTION_CATEGORIES.items():
        if not category_mask.get(cat_name, True):
            continue
        cat_str = category_strengths.get(cat_name, 1.0) * overall_strength
        if cat_str < 0.001:
            continue

        for key in keys:
            if key not in corrections:
                continue
            val = corrections[key]
            identity = _IDENTITY[key]

            if isinstance(identity, list):
                result[key] = [
                    id_v + (v - id_v) * cat_str
                    for v, id_v in zip(val, identity)
                ]
            else:
                result[key] = identity + (val - identity) * cat_str

    return result


# ---------------------------------------------------------------------------
# Corrections -> effect stack
# ---------------------------------------------------------------------------

def corrections_to_effects(corrections: dict) -> list[dict]:
    """Convert a corrections dict to an effects list using ``make_effect``."""
    from sdqt.models.effects import make_effect

    effects: list[dict] = []

    if "brightness" in corrections or "contrast" in corrections:
        effects.append(make_effect(
            "brightness_contrast",
            brightness=corrections.get("brightness", 0.0),
            contrast=corrections.get("contrast", 1.0),
        ))

    if "temperature" in corrections or "tint" in corrections:
        effects.append(make_effect(
            "color_temperature",
            temperature=corrections.get("temperature", 0.0),
            tint=corrections.get("tint", 0.0),
        ))

    if "rgb_gain" in corrections:
        r, g, b = corrections["rgb_gain"]
        effects.append(make_effect(
            "channel_mixer",
            rr=r,
            gg=g,
            bb=b,
        ))

    if "saturation" in corrections:
        effects.append(make_effect(
            "hue_sat",
            saturation=corrections["saturation"],
        ))

    if "black_point" in corrections or "white_point" in corrections:
        effects.append(make_effect(
            "levels",
            black_point=corrections.get("black_point", 0.0),
            white_point=corrections.get("white_point", 255.0),
            gamma=corrections.get("gamma", 1.0),
        ))

    if "shadows" in corrections or "highlights" in corrections:
        effects.append(make_effect(
            "shadows_highlights",
            shadows=corrections.get("shadows", 0.0),
            highlights=corrections.get("highlights", 0.0),
        ))

    return effects


# ---------------------------------------------------------------------------
# color-matcher integration (Match All To)
# ---------------------------------------------------------------------------

def color_match_frame(
    ref: "np.ndarray", target: "np.ndarray", method: str = "mkl",
) -> "np.ndarray":
    """Apply color-matcher optimal transport to make *target* look like *ref*.

    Both inputs uint8 RGB (H, W, 3).  Returns uint8 RGB result.
    Uses Monge-Kantorovich Linearization by default -- matches the full 3D
    colour distribution including cross-channel covariance.
    """
    from color_matcher import ColorMatcher
    from color_matcher.normalizer import Normalizer

    cm = ColorMatcher()
    result = cm.transfer(src=target, ref=ref, method=method)
    return Normalizer(result).uint8_norm()


def derive_gains_from_transfer(
    target: "np.ndarray", transferred: "np.ndarray",
) -> dict:
    """Derive channel_mixer gains + saturation from a color-matched result.

    Compares the *transferred* (ideal output from color-matcher) against the
    *target* (original clip frame) to compute the best-fit linear gains.
    """
    import numpy as np

    tgt_f = target.astype(np.float32) / 255.0
    xfr_f = transferred.astype(np.float32) / 255.0

    corrections: dict = {}

    # Per-channel mean ratio -- best linear approximation of the transfer
    tgt_mean = tgt_f.mean(axis=(0, 1))
    xfr_mean = xfr_f.mean(axis=(0, 1))
    rr = float(xfr_mean[0] / max(tgt_mean[0], 0.001))
    gg = float(xfr_mean[1] / max(tgt_mean[1], 0.001))
    bb = float(xfr_mean[2] / max(tgt_mean[2], 0.001))
    rr = float(np.clip(rr, 0.3, 3.0))
    gg = float(np.clip(gg, 0.3, 3.0))
    bb = float(np.clip(bb, 0.3, 3.0))
    if max(abs(rr - 1), abs(gg - 1), abs(bb - 1)) > 0.002:
        corrections["rgb_gain"] = [rr, gg, bb]

    # Saturation ratio
    tgt_sat = float(tgt_f.std(axis=2).mean())
    xfr_sat = float(xfr_f.std(axis=2).mean())
    if tgt_sat > 0.001:
        sat = xfr_sat / tgt_sat
        sat = float(np.clip(sat, 0.2, 3.0))
        if abs(sat - 1.0) > 0.02:
            corrections["saturation"] = sat

    return corrections


def measure_frame(frame: "np.ndarray") -> dict:
    """Measure colour statistics from a uint8 RGB frame."""
    import numpy as np

    f = frame.astype(np.float32) / 255.0
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]

    return {
        "rgb_mean": [float(r.mean()), float(g.mean()), float(b.mean())],
        "sat_mean": float(f.std(axis=2).mean()),
    }


def refine_corrections(
    corrections: dict,
    ref_stats: dict,
    rendered_stats: dict,
) -> bool:
    """Adjust *corrections* in-place based on measured rendered output.

    For channel_mixer the adjustment is exact:
    ``new_rr = old_rr * ref_R / rendered_R``.
    """
    import numpy as np
    changed = False

    old_gain = corrections.get("rgb_gain", [1.0, 1.0, 1.0])
    new_gain = []
    for i in range(3):
        rendered_ch = max(rendered_stats["rgb_mean"][i], 0.001)
        ref_ch = ref_stats["rgb_mean"][i]
        adj = old_gain[i] * (ref_ch / rendered_ch)
        new_gain.append(float(np.clip(adj, 0.3, 3.0)))
    if any(abs(n - o) > 0.001 for n, o in zip(new_gain, old_gain)):
        corrections["rgb_gain"] = new_gain
        changed = True

    if rendered_stats["sat_mean"] > 0.001:
        sat_err = ref_stats["sat_mean"] / rendered_stats["sat_mean"]
        if abs(sat_err - 1.0) > 0.01:
            old_sat = corrections.get("saturation", 1.0)
            corrections["saturation"] = float(np.clip(old_sat * sat_err, 0.2, 3.0))
            changed = True

    return changed


def corrections_pass_threshold(ref_stats: dict, rendered_stats: dict) -> bool:
    """Check if rendered frame is close enough to reference."""
    for i in range(3):
        if abs(ref_stats["rgb_mean"][i] - rendered_stats["rgb_mean"][i]) > 0.008:
            return False
    if ref_stats["sat_mean"] > 0.005:
        sat_ratio = rendered_stats["sat_mean"] / ref_stats["sat_mean"]
        if abs(sat_ratio - 1.0) > 0.05:
            return False
    return True


def render_frame_with_effects(
    clip_path: str,
    effects: list[dict],
    media_offset: float = 0.0,
) -> "np.ndarray | None":
    """Render a single frame from *clip_path* with *effects* applied via ffmpeg.

    Returns a uint8 RGB numpy array, or None on failure.
    """
    import subprocess
    import numpy as np
    from supremediffusion.utils.video import probe_video

    try:
        info = probe_video(clip_path)
        w, h = info["width"], info["height"]
    except Exception:
        return None

    from sdqt.workers.video_effects import _build_video_filters
    from supremediffusion.utils.video import _png_export_filter

    vf = _build_video_filters(effects)

    # Append range conversion as the LAST filter so the rgb24 output is
    # full-range (matches the video player's display + the PNG path used by
    # extract_single_frame). Without this, tv-range clips come out washed.
    range_filt = _png_export_filter(clip_path)
    vf = f"{vf},{range_filt}" if vf else range_filt

    cmd = ["ffmpeg", "-y"]
    if media_offset > 0.01:
        cmd += ["-ss", f"{media_offset:.3f}"]
    cmd += ["-i", clip_path]
    cmd += ["-vf", vf]
    cmd += [
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    try:
        r = subprocess.run(
            cmd, capture_output=True, timeout=15,
        )
        if r.returncode != 0:
            logger.warning("render_frame_with_effects failed: %s", r.stderr[-200:])
            return None
        expected = h * w * 3
        if len(r.stdout) < expected:
            return None
        return np.frombuffer(r.stdout[:expected], dtype=np.uint8).reshape(h, w, 3)
    except Exception as exc:
        logger.warning("render_frame_with_effects error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Robust multi-frame sampling for match grade
# ---------------------------------------------------------------------------

# How many frames to sample across a clip when measuring its grade. One frame
# (the old behavior) makes the match depend on an arbitrary frame → "matches
# random". Sampling across the whole clip gives the clip's true overall color.
GRADE_SAMPLE_FRAMES = 5


def sample_clip_grade_frames(
    clip,
    n: int = GRADE_SAMPLE_FRAMES,
    crop: "tuple | None" = None,
    keep_match: bool = False,
) -> "np.ndarray | None":
    """Sample *n* frames spread across *clip* and stack them for robust color
    statistics — the input to :func:`generate_lut3d`.

    Measuring the whole clip (not one arbitrary frame) is what makes Match
    Grade match the clip's real grade instead of "matching random". Prior
    match effects (``_match_grade`` / ``_match_all_to`` / ``_match_similar``)
    are stripped by default so re-matching measures the clip's true source
    color and converges instead of compounding.

    Args:
        clip: timeline clip (needs ``.path``, ``.effects``, ``.media_offset``,
            ``.duration``).
        n: number of frames to sample across the clip.
        crop: optional normalized ``(x, y, w, h)`` region to measure.
        keep_match: keep ``_match_*`` effects — used for chain mode, where the
            already-graded appearance should propagate to the next clip.

    Returns a stacked uint8 RGB array ``(ΣH, W, 3)`` or ``None`` on failure.
    """
    import numpy as np

    def _is_match(fx: dict) -> bool:
        return bool(
            fx.get("_match_grade")
            or fx.get("_match_all_to")
            or fx.get("_match_similar")
        )

    effects = [
        fx for fx in (getattr(clip, "effects", None) or [])
        if fx.get("enabled", True) and (keep_match or not _is_match(fx))
    ]

    base = float(getattr(clip, "media_offset", 0.0) or 0.0)
    dur = float(getattr(clip, "duration", 0.0) or 0.0)
    n = max(int(n), 1)

    if dur <= 0.05:
        offsets = [base]
    else:
        # Evenly spaced sample centers, avoiding the exact clip edges.
        offsets = [base + dur * (i + 0.5) / n for i in range(n)]

    frames: list = []
    for off in offsets:
        frame = render_frame_with_effects(clip.path, effects, media_offset=max(off, 0.0))
        # Skip near-black frames (fades / hard cuts) — nudge inward once.
        if frame is not None and frame.mean() <= 5.0 and dur > 0.2:
            frame = render_frame_with_effects(
                clip.path, effects, media_offset=max(off + dur * 0.1, 0.0),
            )
        if frame is None:
            continue
        if crop is not None:
            frame = crop_region(frame, crop)
        if frame.size:
            frames.append(frame)

    if not frames:
        return None

    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    return np.concatenate(frames, axis=0)


# ---------------------------------------------------------------------------
# 3D LUT generation
# ---------------------------------------------------------------------------

def generate_lut3d(
    ref_frame: "np.ndarray",
    target_frame: "np.ndarray",
    output_path: str,
    size: int = 33,
) -> str:
    """Generate a 3D LUT .cube file via color-matcher MKL.

    Computes the MKL affine transform from the real target/ref pair,
    then applies that transform to an identity RGB lattice to build
    the LUT.  This preserves the full nonlinear colour mapping.

    *ref_frame* and *target_frame* are uint8 RGB (H, W, 3).
    Returns *output_path* on success.
    """
    import numpy as np
    from color_matcher import ColorMatcher

    # Step 1: compute MKL transform from the real images
    cm = ColorMatcher()
    cm._src = target_frame.astype(np.float64) / 255.0
    cm._ref = ref_frame.astype(np.float64) / 255.0
    cm.validate_color_chs()
    cm.init_vars()
    cm.mkl_solver()

    # cm.r = src pixels (target), cm.z = ref pixels
    # cm.mu_r = mean(target), cm.mu_z = mean(ref)
    # transform: result = T @ (pixel - mu_target) + mu_ref
    T = np.real(cm.transfer_mat)   # 3x3 (force real in degenerate cases)
    mu_src = np.real(cm.mu_r)      # 3x1 (target mean)
    mu_ref = np.real(cm.mu_z)      # 3x1 (ref mean)

    # Step 2: build identity lattice and apply the transform
    steps = np.linspace(0.0, 1.0, size, dtype=np.float64)
    # .cube ordering: R varies fastest (innermost loop), then G, then B
    bb, gg, rr = np.meshgrid(steps, steps, steps, indexing="ij")
    lattice = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)  # (size^3, 3)

    # Apply affine: T @ (pixel - mu_src) + mu_ref
    shifted = (lattice.T - mu_src)          # 3 x N
    transformed = T @ shifted + mu_ref      # 3 x N
    lut_float = np.clip(transformed.T, 0.0, 1.0)  # (N, 3)

    write_cube_file_float(output_path, lut_float, size)
    logger.info("Generated 3D LUT (%dx%dx%d) → %s", size, size, size, output_path)
    return output_path


def write_cube_file_float(
    path: str, lut_data: "np.ndarray", size: int,
) -> None:
    """Write a .cube 3D LUT file.  *lut_data* is (N, 3) float 0-1."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("TITLE \"SupremeDiffusion Match Grade\"\n")
        f.write(f"LUT_3D_SIZE {size}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
        for rgb in lut_data:
            f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")


def write_cube_file(
    path: str, lut_data: "np.ndarray", size: int,
) -> None:
    """Write a .cube 3D LUT file.  *lut_data* is (N, 3) uint8 0-255."""
    import numpy as np
    write_cube_file_float(path, lut_data.astype(np.float32) / 255.0, size)


def parse_cube_file(path: str) -> tuple["np.ndarray", int]:
    """Parse a .cube LUT file. Returns (float32 array of shape (size,size,size,3), size)."""
    import numpy as np

    size = 0
    data: list[list[float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if line.startswith("DOMAIN_MIN") or line.startswith("DOMAIN_MAX"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if size == 0 or len(data) != size ** 3:
        raise ValueError(
            f"Invalid .cube file: expected {size}^3={size**3} entries, got {len(data)}"
        )

    # .cube data: R varies fastest, then G, then B (outermost).
    # reshape gives (B, G, R, 3) in numpy C-order — which is exactly
    # what glTexImage3D(width=R, height=G, depth=B) expects since
    # OpenGL reads width (R) fastest from contiguous memory.
    arr = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
    return arr, size
