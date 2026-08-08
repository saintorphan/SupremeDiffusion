"""Console tab — live inference logs with copy/clear.

Features:
  - Color-coded by level: DEBUG=grey, INFO=white, WARNING=yellow, ERROR=red
  - Keyword highlighting: SUCCESS/COMPLETE/GENERATED/DONE = green,
    FAILED/CRASH/EXCEPTION = bright red
  - Thread-safe via Qt signal bridge
"""

from __future__ import annotations

import logging
import re

from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Qt-safe log handler: emits a signal so we can append from any thread
# ---------------------------------------------------------------------------

class _LogSignalBridge(QObject):
    """Bridge: logging handler emits this signal, GUI slot receives it."""
    log_record = Signal(str, int)  # (formatted_message, level)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal."""

    def __init__(self, bridge: _LogSignalBridge) -> None:
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._bridge.log_record.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Console tab widget
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    logging.DEBUG:    QColor(120, 120, 120),   # dim grey
    logging.INFO:     QColor(210, 210, 210),   # light grey / white
    logging.WARNING:  QColor(255, 213, 79),    # bright yellow
    logging.ERROR:    QColor(255, 82, 82),     # bright red
    logging.CRITICAL: QColor(255, 23, 68),     # vivid red
}

# Keywords that override the level color → green (success indicators)
_SUCCESS_RE = re.compile(
    r"\b(SUCCESS|COMPLETE|COMPLETED|GENERATED|DONE|FINISHED|SAVED|LOADED|READY)\b",
    re.IGNORECASE,
)
_SUCCESS_COLOR = QColor(105, 240, 105)  # bright green

_LEVEL_CHOICES = [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
]


class ConsoleTab(QWidget):
    """Live log console with level filter, copy, and clear."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._max_lines = 5000
        self._build_ui()
        self._install_handler()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Log Level:"))
        self._level_combo = QComboBox()
        for label, _ in _LEVEL_CHOICES:
            self._level_combo.addItem(label)
        self._level_combo.setCurrentIndex(0)  # DEBUG — show everything
        self._level_combo.currentIndexChanged.connect(self._on_level_changed)
        toolbar.addWidget(self._level_combo)
        toolbar.addStretch()

        self._line_count = QLabel("0 lines")
        toolbar.addWidget(self._line_count)

        self._copy_btn = QPushButton("Copy All")
        self._copy_btn.clicked.connect(self._on_copy)
        toolbar.addWidget(self._copy_btn)

        self._copy_sel_btn = QPushButton("Copy Selected")
        self._copy_sel_btn.clicked.connect(self._on_copy_selected)
        toolbar.addWidget(self._copy_sel_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        # Log view
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(self._max_lines)
        self._log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._log_view.setStyleSheet(
            "QPlainTextEdit { background: #111; font-family: monospace; font-size: 14px; }"
        )
        layout.addWidget(self._log_view, 1)

    def _install_handler(self) -> None:
        """Install a logging handler on the root logger."""
        self._bridge = _LogSignalBridge(self)
        self._bridge.log_record.connect(self._append_log)

        self._handler = QtLogHandler(self._bridge)
        self._handler.setLevel(logging.DEBUG)
        self._handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        ))

        logging.getLogger().addHandler(self._handler)

    # -- Slots ----------------------------------------------------------------

    @Slot(str, int)
    def _append_log(self, message: str, level: int) -> None:
        # Filter by selected level
        _, min_level = _LEVEL_CHOICES[self._level_combo.currentIndex()]
        if level < min_level:
            return

        # Pick base color from level
        color = _LEVEL_COLORS.get(level, QColor(210, 210, 210))

        # Override: success keywords get green regardless of level
        if _SUCCESS_RE.search(message):
            color = _SUCCESS_COLOR

        fmt = QTextCharFormat()
        fmt.setForeground(color)

        # Bold for warnings and above
        if level >= logging.WARNING:
            fmt.setFontWeight(700)  # bold

        cursor = self._log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + "\n", fmt)
        self._log_view.setTextCursor(cursor)

        # Auto-scroll to bottom
        scrollbar = self._log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        # Update line count
        count = self._log_view.document().blockCount()
        self._line_count.setText(f"{count} lines")

    @Slot()
    def _on_level_changed(self) -> None:
        _, min_level = _LEVEL_CHOICES[self._level_combo.currentIndex()]
        self._handler.setLevel(min_level)

    @Slot()
    def _on_copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        text = self._log_view.toPlainText()
        if text:
            QApplication.clipboard().setText(text)

    @Slot()
    def _on_copy_selected(self) -> None:
        from PySide6.QtWidgets import QApplication
        cursor = self._log_view.textCursor()
        text = cursor.selectedText()
        if text:
            # QPlainTextEdit uses paragraph separators — normalize to newlines
            text = text.replace("\u2029", "\n")
            QApplication.clipboard().setText(text)

    @Slot()
    def _on_clear(self) -> None:
        self._log_view.clear()
        self._line_count.setText("0 lines")

    def cleanup(self) -> None:
        """Remove the handler (call on app shutdown)."""
        logging.getLogger().removeHandler(self._handler)
