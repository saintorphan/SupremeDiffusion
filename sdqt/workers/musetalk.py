"""Worker thread for MuseTalk lip sync inference."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import BaseWorker
from sdqt.utils.codec import configured_codec_args as _codec_args

logger = logging.getLogger(__name__)

_MUSETALK_DIR = Path("~/MuseTalk").expanduser()
_FACE_PAD_RATIO = 0.55  # Larger padding to include hair/neck during head tilts


def _match_color(src, ref):
    """Per-pixel brightness correction in BGR space.

    Computes a per-pixel ratio between reference and source, applies it
    to the MuseTalk output to exactly match the original brightness at
    every pixel. Works regardless of skin/background mix in the crop.
    """
    import numpy as np

    src_f = src.astype(np.float32)
    ref_f = ref.astype(np.float32)
    # Per-pixel luminance (simple average across channels)
    src_lum = src_f.mean(axis=2, keepdims=True).clip(1.0)
    ref_lum = ref_f.mean(axis=2, keepdims=True).clip(1.0)
    # Scale each pixel to match reference brightness
    ratio = ref_lum / src_lum
    result = (src_f * ratio).clip(0, 255).astype(np.uint8)
    return result


def _jaw_mask(h: int, w: int, mode: str = "jaw") -> "np.ndarray":
    """Create a feathered blend mask for face compositing.

    mode="jaw":  gradient from 0 at top to 255 at bottom ~40% line,
                 with feathered edges on all sides.
    mode="full": fully feathered ellipse covering the whole face.
    """
    import cv2
    import numpy as np

    mask = np.zeros((h, w), dtype=np.uint8)

    if mode == "full":
        # Feathered ellipse covering the whole crop
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 4, h // 2 - 4),
                     0, 0, 360, 255, -1)
        feather = max(h, w) // 6
        if feather > 1:
            mask = cv2.GaussianBlur(mask, (0, 0), feather)
        return mask

    # Jaw mode: bottom 60% is fully opaque, top fades out,
    # edges feathered so the seam is invisible
    cutoff = int(h * 0.40)  # top 40% fades in
    mask[cutoff:, :] = 255

    # Vertical gradient in the top portion
    if cutoff > 0:
        grad = np.linspace(0, 255, cutoff).astype(np.uint8)
        mask[:cutoff, :] = grad[:, np.newaxis]

    # Feather the left/right/bottom edges
    feather = max(w // 8, 8)
    if feather > 1:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)

    return mask


class MuseTalkWorker(BaseWorker):
    """Run MuseTalk lip sync natively.

    MuseTalk is a single-step UNet inpainting model in SD VAE latent space.
    Very fast (~30fps), low VRAM (~2-4 GB), best visual quality around mouth.

    Pipeline:
    1. Crop video to padded face region
    2. Run MuseTalk inference on cropped face + audio
    3. Composite synced face back onto original video
    4. Mux audio
    """

    def __init__(
        self,
        video_path: str,
        audio_path: str,
        face_bbox: list[int],
        output_path: str,
        *,
        version: str = "v1.5",
        blend_mode: str = "jaw",
        batch_size: int = 8,
        use_fp16: bool = True,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        face_pad_pct: int = 55,
        cheek_left: int = 90,
        cheek_right: int = 90,
        audio_offset: float = 0.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_path
        self._audio = audio_path
        self._bbox = face_bbox
        self._output = output_path
        self._version = version
        self._blend_mode = blend_mode
        self._batch_size = batch_size
        self._use_fp16 = use_fp16
        self._bbox_shift = bbox_shift
        self._extra_margin = extra_margin
        self._face_pad_pct = face_pad_pct
        self._cheek_left = cheek_left
        self._cheek_right = cheek_right
        self._audio_offset = audio_offset

    def do_work(self) -> str:
        if not _MUSETALK_DIR.is_dir():
            raise RuntimeError(
                "MuseTalk not found. Clone to ~/MuseTalk:\n"
                "git clone https://github.com/TMElyralab/MuseTalk ~/MuseTalk"
            )

        import torch

        # Fix torch.load weights_only for older MMLab/MuseTalk checkpoints
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

        tmp_dir = Path(tempfile.mkdtemp(prefix="musetalk_"))

        # Ensure MuseTalk is importable
        mt_str = str(_MUSETALK_DIR)
        if mt_str not in sys.path:
            sys.path.insert(0, mt_str)

        orig_cwd = os.getcwd()
        os.chdir(mt_str)

        try:
            logger.info("MuseTalk worker started, cwd=%s", os.getcwd())

            # 1. Crop video to face region
            self.progress.emit(0.0, "Cropping to face region...")
            crop_rect, video_w, video_h = self._compute_crop(self._video, self._bbox)
            cx, cy, cw, ch = crop_rect

            cropped_video = str(tmp_dir / "cropped.mp4")
            self._run_ffmpeg([
                "ffmpeg", "-y", "-i", self._video,
                "-vf", f"crop={cw}:{ch}:{cx}:{cy}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", cropped_video,
            ])
            if self.is_aborted:
                raise InterruptedError("Aborted")

            # 2. Run MuseTalk inference
            self.progress.emit(0.10, "Running MuseTalk inference...")
            synced_crop = str(tmp_dir / "synced_crop.mp4")
            self._run_musetalk(cropped_video, self._audio, synced_crop)

            if self.is_aborted:
                raise InterruptedError("Aborted")

            if not Path(synced_crop).is_file():
                raise RuntimeError("MuseTalk produced no output")

            # 3. Composite back
            self.progress.emit(0.85, "Compositing result...")
            composited = str(tmp_dir / "composited.mp4")
            self._run_ffmpeg([
                "ffmpeg", "-y",
                "-i", self._video,
                "-i", synced_crop,
                "-filter_complex",
                f"[1:v]scale={cw}:{ch}:flags=lanczos[crop];[0:v][crop]overlay={cx}:{cy}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", composited,
            ])
            if self.is_aborted:
                raise InterruptedError("Aborted")

            # 4. Mux audio (placed at audio_offset within the video)
            self.progress.emit(0.92, "Adding audio...")
            dur_result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", composited],
                capture_output=True, text=True,
            )
            vid_dur = dur_result.stdout.strip()

            if self._audio_offset > 0.01:
                # Delay audio by offset so it lines up with the synced frames
                mux_cmd = [
                    "ffmpeg", "-y",
                    "-i", composited,
                    "-i", self._audio,
                    "-filter_complex",
                    f"[1:a]adelay={int(self._audio_offset * 1000)}|{int(self._audio_offset * 1000)}[delayed]",
                    "-map", "0:v", "-map", "[delayed]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                ]
            else:
                mux_cmd = [
                    "ffmpeg", "-y",
                    "-i", composited,
                    "-i", self._audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                ]
            if vid_dur:
                mux_cmd += ["-t", vid_dur]
            else:
                mux_cmd += ["-shortest"]
            mux_cmd.append(self._output)
            self._run_ffmpeg(mux_cmd)

            self.progress.emit(1.0, "Done")
            return self._output
        finally:
            os.chdir(orig_cwd)
            torch.load = _orig_load
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _run_musetalk(self, video_path: str, audio_path: str, output_path: str) -> None:
        """Run MuseTalk inference natively using the correct API."""
        import numpy as np
        import torch
        import cv2

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (self._use_fp16 and torch.cuda.is_available()) else torch.float32

        # Determine model paths
        models_dir = _MUSETALK_DIR / "models"
        if self._version == "v1.5":
            unet_path = str(models_dir / "musetalkV15" / "unet.pth")
            unet_config = str(models_dir / "musetalkV15" / "musetalk.json")
        else:
            unet_path = str(models_dir / "musetalk" / "pytorch_model.bin")
            unet_config = str(models_dir / "musetalk" / "musetalk.json")

        self.progress.emit(0.15, "Loading MuseTalk models...")

        # Load VAE, UNet, PositionalEncoding
        from musetalk.utils.utils import load_all_model
        vae, unet, pe = load_all_model(
            unet_model_path=unet_path,
            vae_type="sd-vae",
            unet_config=unet_config,
            device=device,
        )

        self.progress.emit(0.25, "Processing audio...")

        # Probe video FPS early so whisper chunks are aligned
        import cv2
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS) or 25
        _cap.release()

        # Audio processing
        from musetalk.utils.audio_processor import AudioProcessor
        from transformers import WhisperModel

        audio_proc = AudioProcessor(
            feature_extractor_path=str(models_dir / "whisper"),
        )
        whisper_input_features, librosa_length = audio_proc.get_audio_feature(audio_path)

        whisper = WhisperModel.from_pretrained(
            str(models_dir / "whisper"),
        ).to(device).to(dtype)

        whisper_chunks = audio_proc.get_whisper_chunk(
            whisper_input_features, device, dtype, whisper,
            librosa_length, fps=fps,
        )

        # Free whisper after encoding
        del whisper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.progress.emit(0.35, "Processing video frames...")

        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("Could not read frames from video")

        # Save frames to temp dir (MuseTalk expects file paths)
        import tempfile as _tf
        frames_dir = str(Path(_tf.mkdtemp(prefix="mt_frames_")))
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = []
        for idx, frame in enumerate(frames):
            fp = os.path.join(frames_dir, f"{idx:08d}.png")
            cv2.imwrite(fp, frame)
            frame_paths.append(fp)

        # Get landmarks and face coords
        from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
        coord_list, frame_list = get_landmark_and_bbox(frame_paths, upperbondrange=self._bbox_shift)

        # Prepare input latents
        self.progress.emit(0.45, "Encoding face latents...")
        input_latent_list = []
        valid_coords = []
        for i, (frame, coord) in enumerate(zip(frame_list, coord_list)):
            if coord is None or coord is coord_placeholder:
                continue
            try:
                x1, y1, x2, y2 = coord  # MuseTalk uses x1,y1,x2,y2
                if self._version == "v1.5":
                    y2 = min(y2 + self._extra_margin, frame.shape[0])
                if y2 <= y1 or x2 <= x1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                latents = vae.get_latents_for_unet(crop)
                input_latent_list.append(latents)
                valid_coords.append(coord)
            except Exception:
                continue
        coord_list = valid_coords

        if not input_latent_list:
            raise RuntimeError("No valid face crops found")

        self.progress.emit(0.50, "Generating lip sync...")

        # Cycle the latent list (smooth looping like MuseTalk does)
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        coord_list_cycle = valid_coords + valid_coords[::-1]

        # Single-step UNet inference
        timesteps = torch.tensor([0], device=device, dtype=torch.long)
        output_frames = []
        batch_size = self._batch_size
        video_num = len(whisper_chunks)

        # Ensure UNet is in the right dtype (VAE stays fp32 for decode quality)
        unet.model = unet.model.to(dtype)

        from musetalk.utils.utils import datagen

        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device,
        )

        total_batches = int(np.ceil(float(video_num) / batch_size))

        for i, (whisper_batch, latent_batch) in enumerate(gen):
            if self.is_aborted:
                raise InterruptedError("Aborted")

            with torch.no_grad():
                audio_feature_batch = pe(whisper_batch).to(dtype=unet.model.dtype)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                pred_latents = unet.model(
                    latent_batch, timesteps,
                    encoder_hidden_states=audio_feature_batch,
                ).sample

                recon = vae.decode_latents(pred_latents)
                for frame in recon:
                    output_frames.append(frame)

            # Free intermediate tensors
            del pred_latents, latent_batch, audio_feature_batch
            torch.cuda.empty_cache()

            if i % 3 == 0:
                frac = 0.50 + 0.30 * ((i + 1) / max(total_batches, 1))
                self.progress.emit(min(frac, 0.80), f"Batch {i+1}/{total_batches}")

        self.progress.emit(0.82, "Blending and writing...")

        # Blend results back into original frames and write video.
        # With audio_offset > 0, preserve original frames before/after
        # the audio window so the clip length stays unchanged.
        start_frame = int(self._audio_offset * fps)
        sync_count = min(len(output_frames), len(coord_list_cycle))

        h_out, w_out = frames[0].shape[:2]

        # Pipe raw BGR frames straight to ffmpeg — no cv2.VideoWriter
        # (cv2 mp4v encodes limited-range YUV which washes out colors)
        from sdqt.utils.codec import pix_fmt_args
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w_out}x{h_out}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an", output_path,
        ]
        ffproc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        pipe_error = None
        total_frames = len(frames)
        try:
            for frame_idx in range(total_frames):
                if self.is_aborted:
                    break

                sync_idx = frame_idx - start_frame

                # Before audio window or after synced frames → original
                if sync_idx < 0 or sync_idx >= sync_count:
                    ffproc.stdin.write(frames[frame_idx].tobytes())
                    continue

                # Within audio window → blend synced face
                coord = coord_list_cycle[sync_idx % len(coord_list_cycle)]
                if coord is None or coord is coord_placeholder:
                    ffproc.stdin.write(frames[frame_idx].tobytes())
                    continue

                ori_frame = frames[frame_idx].copy()
                res_frame = output_frames[sync_idx]
                x1, y1, x2, y2 = coord
                if self._version == "v1.5":
                    y2 = min(y2 + self._extra_margin, ori_frame.shape[0])
                target_h, target_w = y2 - y1, x2 - x1
                if target_h > 0 and target_w > 0:
                    res_resized = cv2.resize(res_frame, (target_w, target_h),
                                             interpolation=cv2.INTER_LANCZOS4)
                    ey, ex = min(y2, ori_frame.shape[0]), min(x2, ori_frame.shape[1])
                    rh, rw = ey - y1, ex - x1
                    ori_frame[y1:ey, x1:ex] = res_resized[:rh, :rw]
                ffproc.stdin.write(ori_frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            pipe_error = exc

        try:
            ffproc.stdin.close()
        except OSError:
            pass
        ffproc.stdin = None  # prevent communicate() from flushing closed file
        _, stderr = ffproc.communicate(timeout=120)

        if pipe_error or ffproc.returncode != 0:
            err_msg = stderr.decode(errors="replace")[-500:] if stderr else str(pipe_error)
            raise RuntimeError(f"FFmpeg frame pipe failed: {err_msg}")

        # Cleanup
        del vae, unet, pe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_crop(
        self, video_path: str, bbox: list[int],
    ) -> tuple[tuple[int, int, int, int], int, int]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", video_path],
            capture_output=True, text=True, check=True,
        )
        dims = result.stdout.strip().split("x")
        video_w, video_h = int(dims[0]), int(dims[1])

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_ratio = self._face_pad_pct / 100.0
        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        cx = max(0, x1 - pad_x)
        cy = max(0, y1 - pad_y)
        cx2 = min(video_w, x2 + pad_x)
        cy2 = min(video_h, y2 + pad_y)

        cw = (cx2 - cx) & ~1
        ch = (cy2 - cy) & ~1

        return (cx, cy, cw, ch), video_w, video_h

    def _run_ffmpeg(self, cmd: list[str]) -> None:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        if self.is_aborted:
            return
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='replace')[-500:]}")


# coord_placeholder is imported lazily inside _run_musetalk to avoid
# triggering DWPose model loading at module import time.
