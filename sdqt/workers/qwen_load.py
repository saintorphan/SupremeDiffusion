"""Worker thread for loading Qwen model without blocking the UI."""

from __future__ import annotations

import logging

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class QwenLoadWorker(BaseWorker):
    """Load Qwen 3.5 4B Instruct on a background thread."""

    def __init__(self, state, *, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    def do_work(self) -> bool:
        self.status.emit("Loading Qwen 3.5 4B...")
        self._state.load_qwen()
        return True
