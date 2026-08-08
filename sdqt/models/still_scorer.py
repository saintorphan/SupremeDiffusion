"""Smart Still Scorer — quality scoring for extracted frames.

Scores frames by sharpness, face detection, and composition to help
users (especially beginners) identify the best frames automatically.

Usage::

    from sdqt.models.still_scorer import score_frames, pick_best
    scores = score_frames(["frame_001.png", "frame_002.png", ...])
    best = pick_best(scores, top_n=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    """Quality scores for a single frame."""
    path: str
    sharpness: float = 0.0       # Laplacian variance (higher = sharper)
    face_count: int = 0          # Number of detected faces
    composition: float = 0.0     # Rule-of-thirds score (0-1, higher = better)
    combined: float = 0.0        # Weighted overall score


def _score_sharpness(gray) -> float:
    """Laplacian variance — higher means sharper / more in focus."""
    import cv2
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _score_faces(gray) -> int:
    """Detect faces using OpenCV's Haar cascade."""
    import cv2
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def _score_composition(gray) -> float:
    """Score how well subject matter aligns with rule-of-thirds grid.

    Looks at edge density in the thirds intersections vs center.
    Frames with interesting content at thirds points score higher.
    """
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 50, 150)

    # Rule-of-thirds intersection regions (4 points)
    margin_x = w // 12  # region radius
    margin_y = h // 12
    thirds_x = [w // 3, 2 * w // 3]
    thirds_y = [h // 3, 2 * h // 3]

    thirds_density = 0.0
    for tx in thirds_x:
        for ty in thirds_y:
            x1 = max(0, tx - margin_x)
            x2 = min(w, tx + margin_x)
            y1 = max(0, ty - margin_y)
            y2 = min(h, ty + margin_y)
            region = edges[y1:y2, x1:x2]
            if region.size > 0:
                thirds_density += float(np.mean(region))

    # Center region (penalize too-centered compositions less,
    # but reward thirds placement more)
    cx, cy = w // 2, h // 2
    center_region = edges[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x]
    center_density = float(np.mean(center_region)) if center_region.size > 0 else 0.0

    # Normalize: thirds_density is sum of 4 regions
    # Score = thirds interest relative to total
    total = thirds_density + center_density + 1e-6
    score = thirds_density / total

    return min(1.0, score)


def score_frame(path: str) -> FrameScore:
    """Score a single frame image."""
    import cv2

    img = cv2.imread(path)
    if img is None:
        logger.warning("Could not read image: %s", path)
        return FrameScore(path=path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = _score_sharpness(gray)
    face_count = _score_faces(gray)
    composition = _score_composition(gray)

    # Weighted combined score:
    # - Sharpness is most important (blurry = unusable)
    # - Faces boost score (people are usually the subject)
    # - Composition is a tiebreaker
    # Normalize sharpness to roughly 0-100 range (typical values 10-500+)
    sharp_norm = min(100.0, sharpness / 5.0)
    face_bonus = face_count * 15.0  # each face adds 15 points
    comp_bonus = composition * 20.0  # up to 20 points

    combined = sharp_norm + face_bonus + comp_bonus

    return FrameScore(
        path=path,
        sharpness=sharpness,
        face_count=face_count,
        composition=composition,
        combined=combined,
    )


def score_frames(paths: list[str], progress_cb=None) -> list[FrameScore]:
    """Score multiple frames. Returns list sorted by combined score (best first).

    Args:
        paths: List of image file paths.
        progress_cb: Optional callback(fraction, description).
    """
    scores: list[FrameScore] = []
    total = len(paths)

    for i, path in enumerate(paths):
        if progress_cb:
            progress_cb((i + 1) / total, f"Scoring frame {i + 1}/{total}...")
        try:
            scores.append(score_frame(path))
        except Exception as e:
            logger.warning("Failed to score %s: %s", path, e)
            scores.append(FrameScore(path=path))

    scores.sort(key=lambda s: s.combined, reverse=True)
    return scores


def pick_best(scores: list[FrameScore], top_n: int = 3) -> list[FrameScore]:
    """Return the top N frames by combined score."""
    return scores[:top_n]
