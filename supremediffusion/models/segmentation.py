"""BiRefNet auto-segmentation utility for foreground extraction.

Extracted from triposr_pipeline.py to be reusable across features
(3D modeling, Magic Mask, etc.).
"""

from __future__ import annotations

import gc as gc_module
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# Cache the loaded BiRefNet model + transform keyed on the source dir so repeated
# segment calls don't reload the ~1GB model every time. Call
# ``release_segmentation_model()`` to free VRAM when done.
_MODEL_CACHE: dict[str, Any] = {}


def _load_birefnet(birefnet_dir: str):
    """Load (or return cached) BiRefNet model + preprocessing transform."""
    cached = _MODEL_CACHE.get(birefnet_dir)
    if cached is not None:
        return cached["model"], cached["transform"]

    logger.info("Loading BiRefNet for segmentation from: %s", birefnet_dir)

    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms

    model = AutoModelForImageSegmentation.from_pretrained(
        birefnet_dir, trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _MODEL_CACHE[birefnet_dir] = {"model": model, "transform": transform}
    return model, transform


def release_segmentation_model() -> None:
    """Free any cached BiRefNet model and release its VRAM."""
    if not _MODEL_CACHE:
        return
    _MODEL_CACHE.clear()
    gc_module.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def segment_foreground(image_path: str, models_root: str) -> str:
    """Segment the foreground of an image using BiRefNet.

    Returns path to a binary mask PNG (white = foreground).
    The caller is responsible for deleting the temp file when done.

    The BiRefNet model is cached across calls; use
    ``release_segmentation_model()`` to free VRAM.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for segmentation.")

    birefnet_dir = str(Path(models_root) / "birefnet")
    if not Path(birefnet_dir).is_dir():
        # Try HuggingFace model ID as fallback
        birefnet_dir = "ZhengPeng7/BiRefNet"

    model, transform = _load_birefnet(birefnet_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():
        preds = model(input_tensor)[-1].sigmoid()

    # Convert to binary mask
    import numpy as np
    mask = preds[0].squeeze().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask, mode="L").resize((w, h), Image.LANCZOS)

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="magic_mask_", delete=False)
    mask_pil.save(tmp.name)
    tmp.close()

    # Release per-call tensors (model stays cached)
    del input_tensor, preds
    gc_module.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Segmentation mask saved: %s", tmp.name)
    return tmp.name
