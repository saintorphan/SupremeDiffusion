"""Worker thread for multi-turn chat generation using Qwen 3.5 4B."""

from __future__ import annotations

import logging

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class ChatWorker(BaseWorker):
    """Generate a chat response on a background thread."""

    def __init__(self, state, messages: list[dict], max_new_tokens: int = 2048, *, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._messages = messages
        self._max_new_tokens = max_new_tokens

    def do_work(self) -> str:
        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.status.emit("Loading Qwen 3.5 4B...")
        self._state.load_qwen()

        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.status.emit("Generating response...")
        return self._state.generate_chat_response(
            self._messages, max_new_tokens=self._max_new_tokens,
        )
