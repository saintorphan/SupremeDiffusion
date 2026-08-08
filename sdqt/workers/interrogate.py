"""Worker threads for image interrogation (BLIP captioning, WD14 tagging, Qwen captioning)."""

from __future__ import annotations

import logging

from .base import BaseWorker

logger = logging.getLogger(__name__)


class QwenCaptionWorker(BaseWorker):
    """Image captioning using Qwen 2.5 VL 7B vision model.

    Loads the VL model, generates a detailed caption directly from the
    image, then unloads. Produces rich natural-language captions suitable
    for LoRA training datasets.
    """

    def __init__(self, image_path: str, state, *, parent=None) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._state = state

    def do_work(self) -> str:
        if self.is_aborted:
            raise InterruptedError

        self.status.emit("Loading Qwen VL...")
        self.progress.emit(0.1, "Loading vision model...")
        self._state.load_qwen_vl()

        if self.is_aborted:
            raise InterruptedError

        self.status.emit("Captioning image...")
        self.progress.emit(0.5, "Generating caption...")

        caption = self._state.caption_image(
            self._image_path,
            prompt=(
                "Describe this image in one detailed sentence for an AI training dataset. "
                "Cover the subject's appearance, pose, expression, clothing, the setting, "
                "lighting, camera angle, and artistic style. Be specific and factual. "
                "Output only the caption."
            ),
            max_new_tokens=200,
        )

        self._state.unload_qwen_vl()
        self.progress.emit(1.0, "Done")
        return caption.strip()


class BLIPInterrogateWorker(BaseWorker):
    """Run BLIP image captioning on a background thread.

    Uses Salesforce/blip-image-captioning-large via HuggingFace transformers.
    """

    def __init__(self, image_path: str, *, parent=None) -> None:
        super().__init__(parent)
        self._image_path = image_path

    def do_work(self) -> str:
        self.status.emit("Loading BLIP model...")
        self.progress.emit(0.1, "Loading BLIP...")

        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        if self.is_aborted:
            raise InterruptedError

        self.status.emit("Generating caption...")
        self.progress.emit(0.5, "Captioning...")

        img = Image.open(self._image_path).convert("RGB")

        # Unconditional caption
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=75)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Cleanup
        del model, processor, inputs, out
        self.progress.emit(1.0, "Done")
        return caption


class WD14TaggerWorker(BaseWorker):
    """Run WD14 tagger for danbooru-style tags on a background thread.

    Uses SmilingWolf/wd-swinv2-tagger-v3 via HuggingFace transformers + timm.
    Falls back to BLIP if dependencies are missing.
    """

    def __init__(self, image_path: str, threshold: float = 0.35, *, parent=None) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._threshold = threshold

    def do_work(self) -> str:
        self.status.emit("Loading WD14 tagger...")
        self.progress.emit(0.1, "Loading tagger model...")

        try:
            return self._run_wd14()
        except Exception as exc:
            logger.warning("WD14 tagger unavailable (%s), falling back to BLIP", exc)
            self.status.emit("WD14 unavailable, using BLIP instead...")
            return self._run_blip_fallback()

    def _run_wd14(self) -> str:
        import csv
        import os

        import numpy as np
        import torch
        from huggingface_hub import hf_hub_download
        from PIL import Image

        MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

        # Download model files
        self.progress.emit(0.2, "Downloading tagger...")
        model_path = hf_hub_download(MODEL_REPO, "model.onnx")
        tags_path = hf_hub_download(MODEL_REPO, "selected_tags.csv")

        # Load tags
        with open(tags_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            tags = [row[1] for row in reader]

        if self.is_aborted:
            raise InterruptedError

        self.progress.emit(0.4, "Running tagger...")

        # Try ONNX runtime first
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Preprocess image
        img = Image.open(self._image_path).convert("RGB")
        img = img.resize((448, 448), Image.LANCZOS)
        img_array = np.array(img).astype(np.float32)
        # BGR conversion and normalization
        img_array = img_array[:, :, ::-1].copy()
        img_array = np.expand_dims(img_array, 0)

        if self.is_aborted:
            raise InterruptedError

        self.progress.emit(0.7, "Generating tags...")

        input_name = session.get_inputs()[0].name
        probs = session.run(None, {input_name: img_array})[0][0]

        # Filter by threshold and format
        tag_probs = list(zip(tags, probs))
        # Skip first 4 tags (rating tags)
        tag_probs = tag_probs[4:]
        selected = [
            (tag, prob) for tag, prob in tag_probs if prob >= self._threshold
        ]
        selected.sort(key=lambda x: x[1], reverse=True)

        result = ", ".join(tag for tag, _ in selected)
        self.progress.emit(1.0, "Done")
        return result

    def _run_blip_fallback(self) -> str:
        """Fallback: use BLIP captioning if WD14 is unavailable."""
        import torch
        from PIL import Image
        from transformers import BlipProcessor, BlipForConditionalGeneration

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        img = Image.open(self._image_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=75)
        caption = processor.decode(out[0], skip_special_tokens=True)

        del model, processor, inputs, out
        self.progress.emit(1.0, "Done (BLIP fallback)")
        return caption
