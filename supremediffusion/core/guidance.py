"""Static guidance video generation for Supreme Diffusion."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from supremediffusion.utils.video import encode_video, probe_video

logger = logging.getLogger(__name__)


class GuidanceGenerator:
    """Create and load guidance videos used to steer generation."""

    @staticmethod
    def generate_static_guidance(
        frame: Image.Image,
        num_frames: int,
        fps: int,
        output_path: str | Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Create a guidance video where every frame is *frame*.

        If *width* and *height* are provided the frame is resized before
        encoding.  The video is written to *output_path* via
        :func:`supremediffusion.utils.video.encode_video`.

        Returns the string path of the saved video.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Optional resize
        if width is not None and height is not None:
            from supremediffusion.core.input_handler import InputHandler
            frame = InputHandler.resize_image(frame, width, height)

        arr = np.array(frame, dtype=np.uint8)  # (H, W, 3)
        # Stack into (N, H, W, 3)
        video_array = np.stack([arr] * num_frames, axis=0)

        logger.info(
            "Generating static guidance: %d frames @ %d fps, %dx%d -> %s",
            num_frames, fps, arr.shape[1], arr.shape[0], output_path,
        )
        encode_video(video_array, str(output_path), fps=fps)
        return str(output_path)

    @staticmethod
    def build_static_tensor(frame: Image.Image, num_frames: int,
                            width: Optional[int] = None,
                            height: Optional[int] = None) -> torch.Tensor:
        """Build a guidance tensor directly from a PIL image — no video encode.

        Every frame in the returned tensor is identical. This completely
        avoids the RGB→YUV→RGB colorspace roundtrip that video codecs
        introduce.
        """
        if width is not None and height is not None:
            from supremediffusion.core.input_handler import InputHandler
            frame = InputHandler.resize_image(frame, width, height)

        frame = frame.convert("RGB")
        arr = np.array(frame, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        result = tensor.unsqueeze(0).expand(num_frames, -1, -1, -1).contiguous()
        logger.info("Static guidance tensor: %d frames, shape %s", num_frames, result.shape)
        return result

    @staticmethod
    def load_static_guidance(frame_path: str | Path, num_frames: int,
                             width: int = 832, height: int = 480) -> torch.Tensor:
        """Build a guidance tensor directly from a single image — no video encode.

        Every frame in the returned tensor is identical. This avoids all
        colorspace conversion that video codecs introduce.
        """
        img = Image.open(str(frame_path)).convert("RGB")
        if img.width != width or img.height != height:
            img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        result = tensor.unsqueeze(0).expand(num_frames, -1, -1, -1).contiguous()
        logger.info("Static guidance tensor from %s: shape %s", frame_path, result.shape)
        return result

    @staticmethod
    def load_guidance_video(path: str | Path) -> torch.Tensor:
        """Load a guidance video and return a float tensor.

        The returned tensor has shape ``(num_frames, channels, height, width)``
        with values in ``[0, 1]``.

        Uses ffmpeg to decode frames into a temporary directory, then
        assembles them into a tensor.
        """
        import tempfile as _tmpmod

        path = str(path)
        info = probe_video(path)
        logger.info("Loading guidance video %s (%d frames)", path, info["num_frames"])

        from supremediffusion.utils.video import extract_frames

        with _tmpmod.TemporaryDirectory(prefix="sd_guidance_") as tmpdir:
            frame_paths = extract_frames(path, output_dir=tmpdir)
            frames = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
                frames.append(tensor)

        if not frames:
            raise ValueError(f"No frames extracted from guidance video: {path}")

        result = torch.stack(frames, dim=0)  # (N, 3, H, W)
        logger.debug("Guidance tensor shape: %s", result.shape)
        return result
