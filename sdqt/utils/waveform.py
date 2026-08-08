"""Waveform extraction utility — extract amplitude peaks from audio via ffmpeg."""

from __future__ import annotations

import logging
import struct
import subprocess

logger = logging.getLogger(__name__)


def extract_waveform(clip_path: str, num_samples: int = 800) -> list[float]:
    """Extract normalized amplitude peaks from a media file.

    Uses ffmpeg to decode audio to raw float32 PCM, then buckets into
    *num_samples* peaks normalized to [0.0, 1.0].
    """
    try:
        cmd = [
            "ffmpeg", "-i", clip_path,
            "-ac", "1", "-ar", "8000",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "pipe:1",
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            return []

        raw = result.stdout
        if len(raw) < 4:
            return []

        # Parse raw float32 samples
        n_floats = len(raw) // 4
        samples = struct.unpack(f"<{n_floats}f", raw[:n_floats * 4])

        if not samples:
            return []

        # Bucket into num_samples peaks
        bucket_size = max(1, len(samples) // num_samples)
        peaks: list[float] = []
        for i in range(0, len(samples), bucket_size):
            chunk = samples[i:i + bucket_size]
            peaks.append(max(abs(s) for s in chunk))
            if len(peaks) >= num_samples:
                break

        # Normalize to [0.0, 1.0]
        max_peak = max(peaks) if peaks else 1.0
        if max_peak > 0:
            peaks = [p / max_peak for p in peaks]

        return peaks

    except Exception:
        logger.debug("Waveform extraction failed for %s", clip_path, exc_info=True)
        return []
