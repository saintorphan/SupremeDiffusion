"""Workers for the Txt2Prompt tab — LLM prompt writing + per-segment refine.

Both load the prompt-writer LLM via AppState (mutually exclusive with the
generation pipelines) and run the prompt_writer orchestrator off the UI thread.
"""
from __future__ import annotations

from sdqt.workers.base import BaseWorker
from supremediffusion.core import prompt_writer


class PromptWriteWorker(BaseWorker):
    """Generate a model-formatted prompt from an idea."""

    def __init__(self, state, req: dict, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._req = req

    def do_work(self):
        if self.is_aborted:
            raise InterruptedError("Aborted")
        self.status.emit("Loading prompt LLM (abliterated Qwen)...")
        self._state.load_prompt_llm()
        if self.is_aborted:
            raise InterruptedError("Aborted")
        self.status.emit("Writing prompt...")
        return prompt_writer.generate(self._state.prompt_llm_chat, **self._req)


class PromptRefineWorker(BaseWorker):
    """Refine a single prompt segment (the magic wand)."""

    def __init__(self, state, req: dict, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._req = req

    def do_work(self):
        if self.is_aborted:
            raise InterruptedError("Aborted")
        self.status.emit("Loading prompt LLM...")
        self._state.load_prompt_llm()
        if self.is_aborted:
            raise InterruptedError("Aborted")
        self.status.emit("Refining...")
        return prompt_writer.refine(self._state.prompt_llm_chat, **self._req)
