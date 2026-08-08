"""Collapsible section widget — card-style container with toggle header."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ── Shared card styles ──────────────────────────────────────────────────────

_HEADER_STYLE = (
    "QPushButton { text-align: left; font-weight: bold; font-size: 13px;"
    " padding: 8px 12px; color: #ccc;"
    " background: #252525; border: none;"
    " border-top-left-radius: 6px; border-top-right-radius: 6px; }"
    "QPushButton:hover { background: #303030; color: #fff; }"
)

_HEADER_COLLAPSED_STYLE = (
    "QPushButton { text-align: left; font-weight: bold; font-size: 13px;"
    " padding: 8px 12px; color: #888;"
    " background: #222; border: none; border-radius: 6px; }"
    "QPushButton:hover { background: #2a2a2a; color: #bbb; }"
)

_CARD_STYLE = (
    "QFrame#section_card {"
    " background: #1e1e1e;"
    " border: 1px solid #333;"
    " border-top: none;"
    " border-bottom-left-radius: 6px;"
    " border-bottom-right-radius: 6px;"
    " }"
)


class CollapsibleSection(QWidget):
    """A card-style container with a clickable header that toggles content.

    Looks like a modern settings card: rounded corners, subtle background,
    visible boundary between sections.

    Usage::

        section = CollapsibleSection("Advanced Settings", collapsed=True)
        section.add_widget(some_widget)
        section.add_layout(some_layout)
    """

    def __init__(self, title: str, collapsed: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._collapsed = collapsed

        # Header button
        self._toggle_btn = QPushButton()
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle)

        # Content card
        self._card = QFrame()
        self._card.setObjectName("section_card")
        self._card.setStyleSheet(_CARD_STYLE)

        self._content_layout = QVBoxLayout(self._card)
        self._content_layout.setContentsMargins(12, 10, 12, 10)
        self._content_layout.setSpacing(6)

        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(0)
        layout.addWidget(self._toggle_btn)
        layout.addWidget(self._card)

        self._title = title
        self._update_header()
        self._card.setVisible(not collapsed)

    # -- Public API --------------------------------------------------------

    def add_widget(self, widget: QWidget) -> None:
        self._content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._card.setVisible(not collapsed)
        self._update_header()

    @property
    def content_layout(self) -> QVBoxLayout:
        return self._content_layout

    # -- Private -----------------------------------------------------------

    @Slot()
    def _toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._card.setVisible(not self._collapsed)
        self._update_header()

    def set_title(self, title: str) -> None:
        """Update the header label text in place."""
        self._title = title
        self._update_header()

    def _update_header(self) -> None:
        arrow = "\u25b6" if self._collapsed else "\u25bc"
        self._toggle_btn.setText(f"{arrow}  {self._title}")
        if self._collapsed:
            self._toggle_btn.setStyleSheet(_HEADER_COLLAPSED_STYLE)
        else:
            self._toggle_btn.setStyleSheet(_HEADER_STYLE)
