"""RIFE temporal interpolation for Supreme Diffusion.

Adapted from wan2gp's postprocessing/rife/inference.py (MIT License, Copyright 2024 Hzwer).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .ssim import ssim_matlab

logger = logging.getLogger(__name__)


def _np_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert (N, H, W, 3) uint8 numpy to (1, N, 3, H, W) float [0,1] tensor."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    return t.unsqueeze(0)  # add batch dim


def _tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, N, 3, H, W) float [0,1] tensor to (N, H, W, 3) uint8 numpy."""
    t = tensor.squeeze(0).clamp_(0, 1).mul_(255).to(torch.uint8)
    return t.permute(0, 2, 3, 1).cpu().numpy()


def temporal_interpolation(
    frames: np.ndarray,
    exp: int,
    model_path: str,
    device: str = "cuda",
    progress_cb: Optional[callable] = None,
) -> np.ndarray:
    """Interpolate frames using RIFE v4.

    Parameters
    ----------
    frames:
        (N, H, W, 3) uint8 numpy array.
    exp:
        Exponent: 1 = 2x frames, 2 = 4x frames.
    model_path:
        Path to rife4.26.pkl weights.
    device:
        Torch device.
    progress_cb:
        Optional callback(fraction, description) for progress.

    Returns
    -------
    Interpolated frames as (M, H, W, 3) uint8 numpy array.
    """
    from .RIFE_V4 import Model

    if progress_cb:
        progress_cb(0, "Loading RIFE model...")

    model = Model()
    model.load_model(model_path, -1, device=device)
    model.eval()
    model.to(device=device)

    # Convert to tensor: (1, N, 3, H, W)
    sample = _np_to_tensor(frames)
    n_frames = sample.shape[1]
    _, _, _, h, w = sample.shape

    pad_mod = model.pad_mod
    tmp = max(pad_mod, int(pad_mod / 1))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)

    def pad_image(img):
        return F.pad(img, padding)

    def get_frame(pos):
        if pos >= n_frames:
            return None
        return sample[:, pos]  # (1, 3, H, W)

    def make_inference(I0, I1, n):
        if n <= 0:
            return []
        return [model.inference(I0, I1, (i + 1) / (n + 1), 1.0) for i in range(n)]

    output_frames = []

    def add_frame(frame):
        f = frame.squeeze(0)[:, :h, :w]  # remove padding, (3, H, W)
        output_frames.append(f.cpu())

    lastframe_raw = get_frame(0)
    I1 = pad_image(lastframe_raw.to(device, non_blocking=True))
    temp = None

    with torch.no_grad():
        pos = 0
        total_pairs = n_frames - 1
        while True:
            if temp is not None:
                frame = temp
                temp = None
            else:
                pos += 1
                frame = get_frame(pos)
            if frame is None:
                break

            I0 = I1
            I1 = pad_image(frame.to(device, non_blocking=True))

            # SSIM check for static/very different frames
            I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            if ssim > 0.996 or pos > 100:
                pos += 1
                frame = get_frame(pos)
                if frame is None:
                    break_flag = True
                    frame = lastframe_raw
                else:
                    temp = frame
                I1 = pad_image(frame.to(device, non_blocking=True))
                I1 = model.inference(I0, I1, 0.5, 1.0)
                I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                frame = I1[:, :, :h, :w]

            if ssim < 0.2:
                output = [I0 for _ in range((2 ** exp) - 1)]
            else:
                output = make_inference(I0, I1, 2**exp - 1) if exp else []

            add_frame(lastframe_raw)
            for mid in output:
                add_frame(mid)
            lastframe_raw = frame

            if progress_cb:
                progress_cb(min(pos, total_pairs) / max(total_pairs, 1), f"Interpolating frame {pos}/{total_pairs}")

            if break_flag:
                break

        add_frame(lastframe_raw)

    # Stack and convert back to numpy
    result = torch.stack(output_frames, dim=0)  # (M, 3, H, W)
    result = result.clamp_(0, 1).mul_(255).to(torch.uint8)
    result = result.permute(0, 2, 3, 1).numpy()  # (M, H, W, 3)

    # Clean up RIFE model from GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("RIFE interpolation: %d -> %d frames (exp=%d)", n_frames, result.shape[0], exp)
    return result
