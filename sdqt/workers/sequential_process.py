"""Sequential Processing controller — Mode 2 sliding-pair img2vid batch.

Thin subclass of BatchGenerateController that overrides three hooks:
per-item kwargs (Mode 2 instead of Mode 1), output filename (pair
numbers instead of sequential index), and human label. Loop step is
force-disabled. Color correct + post-process steps reuse the exact
same workers Batch uses via the inherited state machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .batch_generate import BatchGenerateController

logger = logging.getLogger(__name__)


class SequentialProcessingController(BatchGenerateController):
    """Iterate sliding (first, last) pairs of integer-named images."""

    def __init__(self, tab, settings: dict, parent=None) -> None:
        # Map pair dicts → opaque items for the parent class
        pairs = list(settings.get("pairs", []))
        # Parent expects settings["images"] as its items list; translate
        # pair dicts into that slot so the base class can iterate them.
        translated = dict(settings)
        translated["images"] = pairs
        super().__init__(tab, translated, parent=parent)

        self._guidance_frame = str(settings.get("guidance_frame", "first")).lower()
        if self._guidance_frame not in ("first", "last"):
            self._guidance_frame = "first"

        # Loop builder is never run for Sequential Processing regardless
        # of what the settings dict contained.
        self._do_loop = False

        # Recompute zero-pad width from the highest integer in the batch
        # rather than the default num_start-based width.
        max_num = int(settings.get("max_num", 0) or 0)
        if max_num:
            self._pad = max(3, len(str(max_num)))

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _build_worker_kwargs(self, item) -> dict:
        return self._tab._build_mode2_kwargs(
            item["first_path"],
            item["last_path"],
            self._use_guidance,
            self._guidance_frame,
        )

    def _describe_item(self, item) -> str:
        return f"{item['first_num']}→{item['last_num']}"

    def _final_filename(self, item, idx: int) -> str:
        first = int(item["first_num"])
        last = int(item["last_num"])
        pad = self._pad
        return f"{self._prefix}{first:0{pad}d}_{last:0{pad}d}.mp4"
