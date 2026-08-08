"""Floating Qwen chat assistant dialog."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sdqt.chat_history import ChatHistory
from sdqt.workers.chat import ChatWorker
from sdqt.workers.qwen_load import QwenLoadWorker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "resources" / "chat_system_prompt.txt"

_USER_BUBBLE = (
    "background: #1a3a5c; color: #ddd; border-radius: 8px;"
    " padding: 8px 12px; margin: 2px 4px 2px 40px;"
)
_ASSISTANT_BUBBLE = (
    "background: #2a2a2a; color: #ddd; border-radius: 8px;"
    " padding: 8px 12px; margin: 2px 40px 2px 4px;"
)
_STATUS_STYLE = "color: #888; font-size: 11px; padding: 0 8px;"


class ChatDialog(QDialog):
    """Non-modal floating chat window for Qwen 3.5 4B assistant."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._history: ChatHistory | None = None
        self._project_name: str = ""
        self._chat_worker: ChatWorker | None = None
        self._load_worker: QwenLoadWorker | None = None
        self._system_prompt = self._load_system_prompt()

        self.setWindowTitle("Qwen Assistant")
        self.setMinimumSize(500, 600)
        self.resize(600, 750)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._build_ui()

        # React to external Qwen unloads (e.g. pipeline load started)
        self._state.qwen_unloaded_signal.connect(self._on_qwen_unloaded)

    def _load_system_prompt(self) -> str:
        try:
            return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        except OSError:
            return "You are a helpful assistant for the Supreme Diffusion application."

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Status label
        self._status_label = QLabel("Initializing...")
        self._status_label.setStyleSheet(_STATUS_STYLE)
        layout.addWidget(self._status_label)

        # Message scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #333; background: #1e1e1e; }"
        )
        self._scroll = scroll

        self._messages_widget = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_widget)
        self._messages_layout.setContentsMargins(4, 4, 4, 4)
        self._messages_layout.setSpacing(4)
        self._messages_layout.addStretch()
        scroll.setWidget(self._messages_widget)
        layout.addWidget(scroll, 1)

        # Input area
        self._input = QTextEdit()
        self._input.setPlaceholderText("Ask about Supreme Diffusion...")
        self._input.setFixedHeight(70)
        self._input.setStyleSheet(
            "QTextEdit { background: #2a2a2a; color: #ddd; border: 1px solid #444;"
            " border-radius: 4px; padding: 4px; }"
        )
        self._input.setEnabled(False)
        layout.addWidget(self._input)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedHeight(30)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)

        btn_row.addStretch()

        clear_btn = QPushButton("Clear History")
        clear_btn.setFixedHeight(30)
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(clear_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(30)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    # -- Event overrides ---------------------------------------------------

    def keyPressEvent(self, event) -> None:
        # Enter sends, Shift+Enter for newline
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and self._send_btn.isEnabled()
        ):
            self._on_send()
            return
        super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load_history()
        if not self._state.qwen_loaded:
            self._start_loading()
        else:
            self._set_ready()

    def closeEvent(self, event) -> None:
        self._save_history()
        if self._chat_worker and self._chat_worker.isRunning():
            self._chat_worker.abort()
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.abort()
        self._state.unload_qwen()
        super().closeEvent(event)

    # -- Project management ------------------------------------------------

    def set_project(self, project_name: str) -> None:
        if project_name == self._project_name:
            return
        self._save_history()
        self._project_name = project_name
        self._load_history()

    # -- Model loading -----------------------------------------------------

    def _start_loading(self) -> None:
        self._status_label.setText("Loading Qwen 3.5 4B...")
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)

        from sdqt.deps import check_and_install
        if not check_and_install("transformers", self):
            self._status_label.setText("Transformers not installed.")
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self._state.model_registry, self,
            on_progress=lambda f, d: self._status_label.setText(d),
        ):
            self._status_label.setText("Model download cancelled.")
            return

        worker = QwenLoadWorker(self._state, parent=self)
        worker.finished_ok.connect(lambda _: self._set_ready())
        worker.error.connect(lambda msg: self._status_label.setText(f"Load error: {msg}"))
        worker.status.connect(lambda s: self._status_label.setText(s))
        worker.finished.connect(worker.deleteLater)
        self._load_worker = worker
        worker.start()

    def _set_ready(self) -> None:
        self._status_label.setText("Ready")
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input.setFocus()

    @Slot()
    def _on_qwen_unloaded(self) -> None:
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)
        self._status_label.setText("Model unloaded (generation in progress)")
        self._add_system_bubble("Model was unloaded for generation. Reopen chat to reload.")

    # -- Chat history ------------------------------------------------------

    def _load_history(self) -> None:
        # Clear UI
        while self._messages_layout.count() > 1:  # keep the stretch
            item = self._messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._project_name:
            return

        project_dir = self._state.project_manager.get_project_dir(self._project_name)
        self._history = ChatHistory(project_dir)
        self._history.load()

        # Populate UI from history
        for msg in self._history.to_display_messages():
            self._add_bubble(msg["role"], msg["content"])

        if not self._history.to_display_messages():
            self._add_bubble(
                "assistant",
                "Hello! I'm your Supreme Diffusion assistant. "
                "Ask me anything about the app — features, workflows, "
                "settings, or troubleshooting.",
            )

    def _save_history(self) -> None:
        if self._history:
            self._history.save()

    # -- Sending messages --------------------------------------------------

    @Slot()
    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return

        self._input.clear()
        self._add_bubble("user", text)
        if self._history:
            self._history.append("user", text)

        self._send_btn.setEnabled(False)
        self._input.setEnabled(False)
        self._status_label.setText("Thinking...")

        # Inject relevant upstream docs into system prompt based on user query
        from sdqt.chat_docs import find_relevant_docs
        doc_context = find_relevant_docs(text)
        system_prompt = self._system_prompt + doc_context

        # Build context messages
        if self._history and self._state._qwen_tokenizer:
            messages = self._history.get_messages_for_context(
                self._state._qwen_tokenizer,
                system_prompt,
            )
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]

        worker = ChatWorker(self._state, messages, parent=self)
        worker.finished_ok.connect(self._on_response)
        worker.error.connect(self._on_chat_error)
        worker.status.connect(lambda s: self._status_label.setText(s))
        worker.finished.connect(worker.deleteLater)
        self._chat_worker = worker
        worker.start()

    @Slot(object)
    def _on_response(self, reply: str) -> None:
        self._add_bubble("assistant", reply)
        if self._history:
            self._history.append("assistant", reply)
            self._history.save()
        self._status_label.setText("Ready")
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input.setFocus()

    @Slot(str)
    def _on_chat_error(self, msg: str) -> None:
        self._status_label.setText(f"Error: {msg}")
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)

    @Slot()
    def _on_clear(self) -> None:
        if self._history:
            self._history.clear()
            self._history.save()
        self._load_history()

    # -- Bubble rendering --------------------------------------------------

    def _add_bubble(self, role: str, content: str) -> None:
        label = QLabel(content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        if role == "user":
            label.setStyleSheet(_USER_BUBBLE)
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            label.setStyleSheet(_ASSISTANT_BUBBLE)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Insert before the stretch
        idx = self._messages_layout.count() - 1
        self._messages_layout.insertWidget(idx, label)

        # Scroll to bottom
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    def _add_system_bubble(self, content: str) -> None:
        label = QLabel(content)
        label.setWordWrap(True)
        label.setStyleSheet(
            "color: #888; font-style: italic; padding: 4px 12px; margin: 2px 20px;"
        )
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        idx = self._messages_layout.count() - 1
        self._messages_layout.insertWidget(idx, label)
