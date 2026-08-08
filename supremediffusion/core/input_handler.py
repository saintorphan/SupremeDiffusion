"""Video decode, frame extraction, and image preparation utilities."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from supremediffusion.utils.video import extract_frames, extract_single_frame, probe_video

logger = logging.getLogger(__name__)


class InputHandler:
    """Static helpers for loading, resizing, and extracting frames from images and videos."""

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(path: str | Path) -> Image.Image:
        """Load an image from *path* and convert to RGB.

        Raises ``FileNotFoundError`` if the path does not exist,
        ``ValueError`` if the file cannot be decoded as an image.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            img = Image.open(path).convert("RGB")
            logger.debug("Loaded image %s  size=%s", path, img.size)
            return img
        except Exception as exc:
            raise ValueError(f"Failed to open image {path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Resize / crop
    # ------------------------------------------------------------------

    @staticmethod
    def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """Resize *image* maintaining aspect ratio, then center-crop to exact *width* x *height*.

        The shorter dimension is scaled to match the target; the longer
        dimension is then centre-cropped so the final size is exactly
        ``(width, height)``.
        """
        src_w, src_h = image.size
        scale = max(width / src_w, height / src_h)
        new_w = int(src_w * scale + 0.5)
        new_h = int(src_h * scale + 0.5)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Centre crop
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        image = image.crop((left, top, left + width, top + height))
        logger.debug("Resized to %dx%d (scale %.3f, crop from %dx%d)", width, height, scale, new_w, new_h)
        return image

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_frame_range(
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        fps: Optional[float] = None,
    ) -> list[Image.Image]:
        """Extract frames ``[start_frame, end_frame)`` from *video_path*.

        Returns a list of PIL Images.  Uses a temporary directory for the
        intermediate PNG files.
        """
        video_path = str(video_path)
        logger.info("Extracting frames %d–%d from %s", start_frame, end_frame, video_path)
        with tempfile.TemporaryDirectory(prefix="sd_frames_") as tmpdir:
            frame_paths = extract_frames(
                video_path,
                output_dir=tmpdir,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
            )
            frames = [Image.open(p).convert("RGB") for p in frame_paths]
        logger.info("Extracted %d frames", len(frames))
        return frames

    @staticmethod
    def extract_single_frame(video_path: str | Path, frame_idx: int) -> Image.Image:
        """Extract a single frame at *frame_idx* from *video_path*."""
        video_path = str(video_path)
        logger.debug("Extracting frame %d from %s", frame_idx, video_path)
        return extract_single_frame(video_path, frame_idx)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_video(video_path: str | Path) -> dict:
        """Probe *video_path* and return an info dict.

        Raises ``FileNotFoundError`` if the path does not exist and
        ``ValueError`` if ffprobe cannot decode the file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        try:
            info = probe_video(str(video_path))
        except Exception as exc:
            raise ValueError(f"Invalid video file {video_path}: {exc}") from exc
        logger.info(
            "Validated %s: %dx%d, %.2f fps, %d frames, %.2fs",
            video_path,
            info["width"],
            info["height"],
            info["fps"],
            info["num_frames"],
            info["duration"],
        )
        return info

    # ------------------------------------------------------------------
    # Mode-specific preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_input_for_mode(mode: int, **kwargs) -> dict:
        """Prepare pipeline inputs based on generation *mode*.

        Parameters vary by mode:

        * **Mode 1** (image-to-video): ``image_path``
        * **Mode 2** (first+last frame): ``first_frame_path``, ``last_frame_path``
        * **Mode 3** (video-to-video / re-encode): ``video_path``, ``start_frame``, ``end_frame``

        Returns a dict with the appropriate keys for downstream consumers.
        """
        handler = InputHandler

        if mode == 1:
            image_path = kwargs.get("image_path")
            if image_path is None:
                raise ValueError("Mode 1 requires 'image_path'")
            image = handler.load_image(image_path)
            logger.info("Mode 1: loaded image from %s", image_path)
            return {"image": image}

        if mode == 2:
            first_path = kwargs.get("first_frame_path")
            last_path = kwargs.get("last_frame_path")
            if first_path is None or last_path is None:
                raise ValueError("Mode 2 requires 'first_frame_path' and 'last_frame_path'")
            first_frame = handler.load_image(first_path)
            last_frame = handler.load_image(last_path)
            logger.info("Mode 2: loaded first=%s  last=%s", first_path, last_path)
            return {"image": first_frame, "last_frame": last_frame}

        if mode == 3:
            video_path = kwargs.get("video_path")
            start_frame = kwargs.get("start_frame")
            end_frame = kwargs.get("end_frame")
            if video_path is None or start_frame is None or end_frame is None:
                raise ValueError("Mode 3 requires 'video_path', 'start_frame', and 'end_frame'")
            frames = handler.extract_frame_range(video_path, start_frame, end_frame)
            logger.info("Mode 3: extracted %d frames from %s", len(frames), video_path)
            return {"frames": frames, "video_path": str(video_path)}

        raise ValueError(f"Unsupported mode: {mode}")
