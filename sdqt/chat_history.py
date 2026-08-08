"""Per-project chat history persistence for Qwen assistant."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class ChatHistory:
    """Persists chat messages to a per-project JSON file."""

    def __init__(self, project_dir: str | Path) -> None:
        self._path = Path(project_dir) / "qwen_chat.json"
        self._messages: list[dict] = []

    def load(self) -> None:
        if self._path.is_file():
            try:
                self._messages = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Failed to load chat history from %s", self._path)
                self._messages = []
        else:
            self._messages = []

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._messages, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to save chat history to %s", self._path)

    def append(self, role: str, content: str) -> None:
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def clear(self) -> None:
        self._messages.clear()

    def get_messages_for_context(
        self,
        tokenizer,
        system_prompt: str,
        max_tokens: int = 12000,
    ) -> list[dict]:
        """Return system prompt + most recent messages that fit within max_tokens.

        Uses the tokenizer for actual token counts.
        """
        system_msg = {"role": "system", "content": system_prompt}
        system_tokens = len(tokenizer.encode(system_prompt))

        budget = max_tokens - system_tokens
        if budget <= 0:
            return [system_msg]

        # Walk backwards, accumulating messages that fit
        selected: list[dict] = []
        used = 0
        for msg in reversed(self._messages):
            msg_tokens = len(tokenizer.encode(msg["content"]))
            if used + msg_tokens > budget:
                break
            selected.append({"role": msg["role"], "content": msg["content"]})
            used += msg_tokens

        selected.reverse()
        return [system_msg] + selected

    def to_display_messages(self) -> list[dict]:
        """All messages (for UI display). Excludes system messages."""
        return [m for m in self._messages if m["role"] != "system"]

    @property
    def messages(self) -> list[dict]:
        return self._messages
