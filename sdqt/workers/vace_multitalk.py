"""Worker thread for VACE MultiTalk lip sync generation."""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseWorker

logger = logging.getLogger(__name__)


class VaceMultitalkWorker(BaseWorker):
    """Run VACE MultiTalk lip sync generation natively.

    Takes an existing video + audio and regenerates it with lip-synced
    characters using Wan 2.2 diffusion with VACE control + MultiTalk audio.
    """

    def __init__(
        self,
        video_path: str,
        audio_left: str,
        audio_right: str,
        output_path: str,
        state,
        *,
        prompt: str = "",
        resolution: str = "832x480",
        num_frames: int = 81,
        steps: int = 20,
        guidance_scale: float = 1.0,
        flow_shift: float = 5.0,
        seed: int = -1,
        speaker_bboxes: list[list[int]] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video = video_path
        self._audio_left = audio_left
        self._audio_right = audio_right
        self._output = output_path
        self._state = state
        self._prompt = prompt
        self._resolution = resolution
        self._num_frames = num_frames
        self._steps = steps
        self._guidance = guidance_scale
        self._flow_shift = flow_shift
        self._seed = seed
        self._speaker_bboxes = speaker_bboxes

    def do_work(self) -> str:
        self.progress.emit(0.0, "Loading VACE MultiTalk pipeline...")

        pipeline = self._state.load_vace_multitalk_pipeline()

        if self.is_aborted:
            raise InterruptedError("Aborted")

        def _progress(frac: float, desc: str) -> None:
            self.progress.emit(frac, desc)

        result = pipeline.run(
            source_video=self._video,
            audio_left=self._audio_left,
            audio_right=self._audio_right or None,
            output_path=self._output,
            prompt=self._prompt,
            resolution=self._resolution,
            num_frames=self._num_frames,
            steps=self._steps,
            guidance_scale=self._guidance,
            flow_shift=self._flow_shift,
            seed=self._seed,
            speaker_bboxes=self._speaker_bboxes,
            progress_callback=_progress,
        )

        self.progress.emit(1.0, "Done")
        return result
