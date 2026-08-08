"""Worker thread for AudioGen sound effect generation.

Uses Meta's AudioCraft AudioGen to generate sound effects from text prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .base import BaseWorker

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000  # AudioGen native sample rate
_MODEL_ID = "facebook/audiogen-medium"


class AudioGenWorker(BaseWorker):
    """Generate sound effects from a text prompt using AudioGen.

    Returns the output WAV file path.
    """

    def __init__(
        self,
        prompt: str,
        output_path: str,
        *,
        duration: float = 10.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._prompt = prompt
        self._output_path = output_path
        self._duration = duration
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._cfg_coef = cfg_coef

    def do_work(self) -> str:
        from audiocraft.models import AudioGen

        self.progress.emit(0.0, "Loading AudioGen model...")

        model = AudioGen.get_pretrained(_MODEL_ID)

        if self.is_aborted:
            del model
            self._cleanup_gpu()
            raise InterruptedError("Aborted by user")

        self.progress.emit(0.3, "Generating sound effect...")

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

        audio = wav[0].cpu()  # (1, samples)

        del model, wav
        self._cleanup_gpu()

        import torchaudio
        Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(self._output_path, audio, _SAMPLE_RATE)

        self.progress.emit(1.0, "Done.")
        return self._output_path
