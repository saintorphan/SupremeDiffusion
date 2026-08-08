#!/usr/bin/env python3
"""Verify the Phase 1-5 color anchor fix by measuring drift in actual outputs.

Pure-analysis script — does NOT run generation. Run it on video files you've
already produced to confirm:

  (1) Pre-fix outputs (legacy `_seam_anchor_extension` path) had compounding
      drift across chained chunks.
  (2) Post-fix outputs (new `anchor_frames_for_chunk` with `anchor_image_mode`
      set) stay locked to the reference.

The script reports per-frame LAB stats and pairwise drift between clips so
the difference shows up numerically rather than just "looks better."

USAGE
=====

Intra-clip drift (single video — measures VAE warm bias accumulating across
frames of one generation):

    python3 scripts/verify_color_anchor.py --single path/to/clip.mp4

Inter-clip drift (compares the FIRST frame of each clip in order — useful when
you've chained N continuations):

    python3 scripts/verify_color_anchor.py --chain clip_0.mp4 clip_1.mp4 clip_2.mp4

Reference-vs-target drift (one reference image / frame against one video —
the most direct measurement of "did my fix work"):

    python3 scripts/verify_color_anchor.py --reference src.png --target generated.mp4

Output: per-frame and pairwise color stats in CIE L*a*b* — units are
JND-relevant (ΔE ~2.3 is a "just noticeable difference"). Anything under
ΔE 2-3 is rock-solid color stability; 3-6 is mild drift; >6 is visible drift.

EXAMPLES of what "good" looks like after the fix (anchor_image_mode='start'):

    Intra-clip max ΔE: 1.8 (was 5.2 before)
    Inter-clip drift between frame_0(clip_0) and frame_0(clip_3): 1.1 (was 6.4)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Logging — minimal so script output stays readable
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("verify_color_anchor")


# ---------------------------------------------------------------------------
# Frame extraction — uses ffmpeg with proper TV→PC range conversion
# ---------------------------------------------------------------------------


def iter_frames(video_path: str, *, stride: int = 1, max_frames: int = -1) -> Iterator:
    """Yield PIL.Image frames from a video. Lazy; uses ffmpeg + PNG pipe.

    Applies in_range=tv:out_range=pc so the analysis sees full-range RGB
    regardless of source encoding.
    """
    from PIL import Image
    import io

    p = Path(video_path)
    if not p.is_file():
        raise FileNotFoundError(video_path)

    vf = f"scale=in_range=tv:out_range=pc,select='not(mod(n\\,{stride}))'"
    if max_frames > 0:
        vf = f"{vf},select='lte(n\\,{max_frames * stride})'"

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(p),
        "-vf", vf,
        "-vsync", "0",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    # PNGs are concatenated in stdout — split by magic bytes
    buf = b""
    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
    PNG_END = b"IEND\xaeB`\x82"
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(PNG_MAGIC)
                if start < 0:
                    break
                end = buf.find(PNG_END, start)
                if end < 0:
                    break
                png_end = end + len(PNG_END)
                png_bytes = buf[start:png_end]
                buf = buf[png_end:]
                try:
                    img = Image.open(io.BytesIO(png_bytes))
                    img.load()
                    yield img.convert("RGB")
                except Exception:
                    continue
    finally:
        proc.wait()


def load_image(path: str):
    """Load any image / video first-frame as PIL.Image."""
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm", ".avi"):
        # Grab first frame via the same iterator
        for f in iter_frames(str(p), max_frames=1):
            return f
        raise RuntimeError(f"No frames in {path}")
    return Image.open(p).convert("RGB")


# ---------------------------------------------------------------------------
# Color stats — LAB mean + DeltaE pairwise
# ---------------------------------------------------------------------------


def lab_mean(img) -> tuple[float, float, float]:
    """Mean L*a*b* of a PIL.Image — via scikit-image (proper D65 conversion)."""
    import numpy as np
    from skimage.color import rgb2lab
    arr = np.asarray(img.convert("RGB")).astype("float32") / 255.0
    lab = rgb2lab(arr)
    return tuple(float(lab[..., i].mean()) for i in range(3))


def delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """CIE76 ΔE between two LAB-mean triples. Simple ΔE — fine for drift detection."""
    import math
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------


def mode_single(video_path: str, stride: int) -> None:
    """Per-frame drift report for one video."""
    print(f"\n=== Intra-clip drift: {Path(video_path).name} ===")
    first: tuple[float, float, float] | None = None
    last: tuple[float, float, float] | None = None
    max_de = 0.0
    n_frames = 0
    print(f"{'frame':>6} {'L*':>6} {'a*':>6} {'b*':>6} {'ΔE_from_frame0':>16}")
    for i, frame in enumerate(iter_frames(video_path, stride=stride)):
        m = lab_mean(frame)
        if first is None:
            first = m
        de = delta_e(first, m)
        max_de = max(max_de, de)
        last = m
        if i % 5 == 0 or i < 5:
            print(f"{i * stride:>6} {m[0]:>6.1f} {m[1]:>6.1f} {m[2]:>6.1f} {de:>16.2f}")
        n_frames += 1
    if first and last:
        print(f"\nFirst frame LAB: ({first[0]:.1f}, {first[1]:.1f}, {first[2]:.1f})")
        print(f"Last frame LAB:  ({last[0]:.1f}, {last[1]:.1f}, {last[2]:.1f})")
        print(f"ΔE first→last:   {delta_e(first, last):.2f}")
        print(f"Max ΔE over clip: {max_de:.2f}")
        print(f"Frames sampled:  {n_frames} (stride={stride})")
    print()
    print("INTERPRETATION:")
    print("  ΔE  < 2.3   → indistinguishable (color anchor working)")
    print("  ΔE  2.3-5   → mild drift (acceptable for short clips)")
    print("  ΔE  5-10    → noticeable drift (anchor likely off or persistence too low)")
    print("  ΔE  > 10    → severe drift (anchor not firing)")


def mode_chain(paths: list[str]) -> None:
    """Inter-clip drift — compare the first frame of each chained clip."""
    print(f"\n=== Inter-clip drift across {len(paths)} clips ===")
    means: list[tuple[float, float, float]] = []
    for i, p in enumerate(paths):
        m = lab_mean(load_image(p))
        means.append(m)
        print(f"clip {i} ({Path(p).name}): first-frame LAB = "
              f"({m[0]:.1f}, {m[1]:.1f}, {m[2]:.1f})")
    print()
    base = means[0]
    print(f"{'pair':>20} {'ΔE_from_clip0':>16}")
    for i in range(1, len(means)):
        de = delta_e(base, means[i])
        print(f"{'clip 0 → clip ' + str(i):>20} {de:>16.2f}")
    print()
    if len(means) >= 2:
        max_de = max(delta_e(base, m) for m in means[1:])
        print(f"Max inter-clip ΔE: {max_de:.2f}")
        print()
        print("INTERPRETATION:")
        print("  ΔE  < 2.3   → identical color across chain (anchor mode='start' working)")
        print("  ΔE  2.3-5   → mild drift (persistence may be too low)")
        print("  ΔE  > 5     → compound drift (anchor disabled or fire-and-forget)")


def mode_reference(ref_path: str, target_path: str, stride: int) -> None:
    """Reference-image vs every-frame-of-target drift."""
    print(f"\n=== Reference vs target drift ===")
    print(f"Reference: {ref_path}")
    print(f"Target:    {target_path}")
    ref = lab_mean(load_image(ref_path))
    print(f"\nReference LAB: ({ref[0]:.1f}, {ref[1]:.1f}, {ref[2]:.1f})")
    print(f"\n{'frame':>6} {'L*':>6} {'a*':>6} {'b*':>6} {'ΔE_from_ref':>14}")
    max_de = 0.0
    avg_de = 0.0
    n = 0
    for i, frame in enumerate(iter_frames(target_path, stride=stride)):
        m = lab_mean(frame)
        de = delta_e(ref, m)
        max_de = max(max_de, de)
        avg_de += de
        n += 1
        if i % 5 == 0 or i < 5:
            print(f"{i * stride:>6} {m[0]:>6.1f} {m[1]:>6.1f} {m[2]:>6.1f} {de:>14.2f}")
    if n:
        print(f"\nMax ΔE vs ref: {max_de:.2f}")
        print(f"Avg ΔE vs ref: {avg_de / n:.2f}")
        print(f"Frames sampled: {n} (stride={stride})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify color-anchor drift in SDQt outputs (post Phase 1-5 rebuild).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", metavar="VIDEO",
                   help="Single-video intra-clip drift report")
    g.add_argument("--chain", nargs="+", metavar="VIDEO",
                   help="Inter-clip drift across N chained clips (first-frame compare)")
    g.add_argument("--reference", metavar="IMG",
                   help="Reference image (use with --target)")
    parser.add_argument("--target", metavar="VIDEO",
                        help="Target video for --reference mode")
    parser.add_argument("--stride", type=int, default=4,
                        help="Sample every Nth frame for video modes (default 4, faster)")
    args = parser.parse_args()

    try:
        # Deps check
        try:
            from PIL import Image  # noqa
            from skimage.color import rgb2lab  # noqa
            import numpy  # noqa
        except ImportError as e:
            print(f"Missing dep: {e}. Run from SDQt venv: "
                  f"~/Projects/SupremeDiffusionQt/.venv/bin/python {sys.argv[0]}", file=sys.stderr)
            return 2

        if args.single:
            mode_single(args.single, stride=args.stride)
        elif args.chain:
            mode_chain(list(args.chain))
        elif args.reference:
            if not args.target:
                parser.error("--reference requires --target")
            mode_reference(args.reference, args.target, stride=args.stride)
        return 0
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
