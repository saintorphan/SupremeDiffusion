"""Frame stitching and video encoding for Supreme Diffusion output."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from PIL import Image

from supremediffusion.utils.video import encode_video

logger = logging.getLogger(__name__)


class OutputHandler:
    """Utilities for converting generated frames into video files."""

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def frames_to_numpy(frames: Union[torch.Tensor, np.ndarray, list[Image.Image]]) -> np.ndarray:
        """Convert various frame representations to a ``(N, H, W, 3)`` uint8 numpy array.

        Accepted inputs:

        * ``torch.Tensor`` -- shapes ``(N, 3, H, W)`` or ``(N, H, W, 3)``
          with values in ``[0, 1]`` (float) or ``[0, 255]`` (uint8).
        * ``np.ndarray`` -- same shape conventions as above.
        * ``list[PIL.Image.Image]`` -- each image is converted to RGB.
        """
        if isinstance(frames, list):
            arrays = [np.array(f.convert("RGB"), dtype=np.uint8) for f in frames]
            result = np.stack(arrays, axis=0)
            logger.debug("Converted %d PIL images -> numpy %s", len(arrays), result.shape)
            return result

        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()

        # At this point frames is np.ndarray
        # Squeeze leading batch dimension if present: (1, N, H, W, C) -> (N, H, W, C)
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(f"Expected 4-D array, got shape {frames.shape}")

        # Handle (N, 3, H, W) -> (N, H, W, 3)
        if frames.shape[1] in (1, 3) and frames.shape[3] not in (1, 3):
            frames = np.transpose(frames, (0, 2, 3, 1))

        # Float [0,1] -> uint8
        if frames.dtype in (np.float32, np.float64):
            if frames.max() <= 1.0:
                frames = (frames * 255.0).clip(0, 255)
            frames = frames.astype(np.uint8)
        elif frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)

        logger.debug("frames_to_numpy result shape: %s dtype: %s", frames.shape, frames.dtype)
        return frames

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    @staticmethod
    def save_generation(
        frames: Union[torch.Tensor, np.ndarray, list[Image.Image]],
        output_path: str | Path,
        fps: int = 16,
        codec: Union[str, Sequence[str]] = "libx264",
        audio_path: str | None = None,
        color_profile=None,
    ) -> str:
        """Encode *frames* to a video file at *output_path*.

        Accepts any format understood by :meth:`frames_to_numpy`. The output
        is tagged with the active project's color profile (or the global
        default when *color_profile* is None) so every Wan/LTX inference
        result lands in the same color space the rest of the project uses.
        If *audio_path* is provided (WAV file), muxes audio into the video.
        Returns the string path of the saved file.

        *codec* may be a single encoder name (legacy, e.g. ``"libx264"``) or a
        full ffmpeg encoder-args list as returned by
        ``sdqt.utils.codec.configured_codec_args()`` (e.g.
        ``["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]``). Passing the
        list lets the primary generation save honor the user's
        ``video_output_codec`` (nvenc) setting instead of always using libx264.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np_frames = OutputHandler.frames_to_numpy(frames)
        logger.info(
            "Saving generation: %d frames @ %d fps, %dx%d -> %s",
            np_frames.shape[0], fps, np_frames.shape[2], np_frames.shape[1], output_path,
        )

        # A full ffmpeg arg list (e.g. configured_codec_args()) is encoded
        # directly so preset/cq/crf flags survive; a bare encoder name falls
        # back to the legacy encode_video() path (libx264-style crf/preset).
        codec_args = list(codec) if not isinstance(codec, str) else None

        if audio_path and Path(audio_path).is_file():
            import tempfile
            tmp_video = tempfile.mktemp(suffix=".mp4", prefix="sd_video_")
            if codec_args is not None:
                OutputHandler._encode_with_args(
                    np_frames, tmp_video, fps=fps, codec_args=codec_args,
                    color_profile=color_profile,
                )
            else:
                encode_video(
                    np_frames, tmp_video, fps=fps, codec=codec,
                    color_profile=color_profile,
                )
            OutputHandler._mux_audio(tmp_video, audio_path, str(output_path))
            try:
                Path(tmp_video).unlink()
            except OSError:
                pass
        else:
            if codec_args is not None:
                OutputHandler._encode_with_args(
                    np_frames, str(output_path), fps=fps, codec_args=codec_args,
                    color_profile=color_profile,
                )
            else:
                encode_video(
                    np_frames, str(output_path), fps=fps, codec=codec,
                    color_profile=color_profile,
                )
        return str(output_path)

    @staticmethod
    def _encode_with_args(
        np_frames: np.ndarray,
        output_path: str,
        *,
        fps: int,
        codec_args: list[str],
        color_profile=None,
    ) -> None:
        """Encode raw frames with an explicit ffmpeg encoder-args list.

        Mirrors :func:`encode_video` but substitutes the hardcoded
        ``-c:v <codec> -crf 16 -preset slow`` with the supplied *codec_args*
        (the encoder + its preset/cq flags). The encoder name is parsed out of
        *codec_args* (the token after ``-c:v``) so the color args still adapt
        to the selected encoder.
        """
        import subprocess

        from supremediffusion.utils.video import _color_args, strip_black_frames

        frames = strip_black_frames(np_frames)
        n, h, w, c = frames.shape

        encoder = "libx264"
        for i, a in enumerate(codec_args):
            if a == "-c:v" and i + 1 < len(codec_args):
                encoder = codec_args[i + 1]
                break

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            *codec_args,
            *_color_args(color_profile, encoder=encoder),
            str(output_path),
        ]
        result = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if result.returncode != 0:
            logger.error(
                "save_generation encode failed: %s",
                result.stderr.decode(errors="replace"),
            )
            result.check_returncode()

    @staticmethod
    def _mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
        """Mux a WAV audio file into a video using ffmpeg stream copy."""
        import subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning(
                "Audio mux failed, saving video without audio: %s",
                result.stderr.decode(errors="replace")[:200],
            )
            shutil.copy2(video_path, output_path)

    @staticmethod
    def save_clip_to_project(
        frames_or_video_path: Union[str, Path, torch.Tensor, np.ndarray, list[Image.Image]],
        project_path: Path,
        fps: int = 16,
    ) -> str:
        """Save a clip into the project's ``clips/`` directory.

        If *frames_or_video_path* is a string / Path pointing to an existing
        file it is copied directly; otherwise the frames are encoded to a new
        video.

        The clip filename is a UTC timestamp to avoid collisions.
        Returns the path of the saved clip.
        """
        clips_dir = Path(project_path) / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        clip_filename = f"clip_{timestamp}.mp4"
        dest = clips_dir / clip_filename

        # If given a path to an existing video, just copy it
        if isinstance(frames_or_video_path, (str, Path)):
            src = Path(frames_or_video_path)
            if src.is_file():
                shutil.copy2(src, dest)
                logger.info("Copied clip %s -> %s", src, dest)
                return str(dest)

        # Otherwise encode frames
        return OutputHandler.save_generation(frames_or_video_path, dest, fps=fps)
