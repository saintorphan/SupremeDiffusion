"""Shared pipeline-load worker — loads heavy models on a background thread."""

from __future__ import annotations

from sdqt.workers.base import BaseWorker


class PipelineLoadWorker(BaseWorker):
    """Call an AppState load method on a background thread."""

    def __init__(self, state, method_name: str, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._method = method_name

    def do_work(self):
        getattr(self._state, self._method)()
        return True
