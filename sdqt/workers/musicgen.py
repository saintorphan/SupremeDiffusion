"""Worker thread for MusicGen music generation.

Uses Meta's AudioCraft MusicGen to generate music from text prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .base import BaseWorker

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 32000  # MusicGen native sample rate

_MODEL_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
}


class MusicGenWorker(BaseWorker):
    """Generate music audio from a text prompt using MusicGen.

    Returns the output WAV file path.
    """

    def __init__(
        self,
        prompt: str,
        output_path: str,
        *,
        model_size: str = "medium",
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._prompt = prompt
        self._output_path = output_path
        self._model_size = model_size
        self._duration = duration
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._cfg_coef = cfg_coef

    def do_work(self) -> str:
        from audiocraft.models import MusicGen

        self.progress.emit(0.0, "Loading MusicGen model...")

        model_id = _MODEL_MAP.get(self._model_size, _MODEL_MAP["medium"])
        model = MusicGen.get_pretrained(model_id)

        if self.is_aborted:
            del model
            self._cleanup_gpu()
            raise InterruptedError("Aborted by user")

        self.progress.emit(0.3, "Generating music...")

        model.set_generation_params(
            duration=self._duration,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            cfg_coef=self._cfg_coef,
        )

        with torch.no_grad():
            wav = model.generate([self._prompt])  # shape: (1, 1, samples)

        if self.is_aborted:
            del model, wav
            self._cleanup_gpu()
            raise InterruptedError("Aborted by user")

        self.progress.emit(0.8, "Saving audio...")

        # Extract audio tensor and save
        audio = wav[0].cpu()  # (1, samples)

        # Clean up model before saving
        del model, wav
        self._cleanup_gpu()

        import torchaudio
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(self._output_path, audio, _SAMPLE_RATE)

        self.progress.emit(1.0, "Done.")
        return self._output_path
