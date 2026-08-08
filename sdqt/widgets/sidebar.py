"""Collapsible sidebar navigation for the modern UI layout."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.sidebar_button import SidebarButton, SIDEBAR_ICONS


_SIDEBAR_BG = "#141414"
_GROUP_HEADER_STYLE = (
    "font-size: 13px; font-weight: 600; color: #888;"
    " letter-spacing: 0.5px; padding: 12px 0 4px 16px;"
)
_SEPARATOR_STYLE = "background: #2a2a2a; max-height: 1px; margin: 4px 12px;"
_EXPANDED_WIDTH = 220
_COLLAPSED_WIDTH = 48


class _SidebarGroup(QWidget):
    """A labelled group of sidebar buttons."""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._header = QLabel(label.upper())
        self._header.setStyleSheet(_GROUP_HEADER_STYLE)
        self._layout.addWidget(self._header)

        self._buttons: list[SidebarButton] = []

    def add_button(self, button: SidebarButton) -> None:
        self._buttons.append(button)
        self._layout.addWidget(button)

    def set_collapsed(self, collapsed: bool) -> None:
        self._header.setVisible(not collapsed)
        for btn in self._buttons:
            btn.set_collapsed(collapsed)


class SidebarWidget(QFrame):
    """Vertical sidebar with grouped navigation buttons."""

    item_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setStyleSheet(
            f"QFrame#sidebar {{ background: {_SIDEBAR_BG};"
            f" border-right: 1px solid #2a2a2a; }}"
        )
        self.setFixedWidth(_EXPANDED_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self._collapsed = False
        self._buttons: dict[str, SidebarButton] = {}
        self._groups: list[_SidebarGroup] = []
        self._current_key: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header (logo + collapse toggle) ──
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet("background: #191919; border-bottom: 1px solid #2a2a2a;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 8, 0)

        self._title = QLabel("\u25c6 SD")  # ◆ SD
        self._title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #0078d4;"
        )
        h_layout.addWidget(self._title)
        h_layout.addStretch()

        self._collapse_btn = QPushButton()
        self._collapse_btn.setFixedSize(28, 28)
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.setToolTip("Collapse sidebar")
        self._collapse_btn.setStyleSheet(
            "QPushButton { border: none; background: transparent; }"
            "QPushButton:hover { background: #2a2a2a; border-radius: 4px; }"
        )
        self._collapse_btn.setIcon(self._make_collapse_icon())
        self._collapse_btn.clicked.connect(self.toggle_collapsed)
        h_layout.addWidget(self._collapse_btn)

        root.addWidget(header)

        # ── Scrollable button area ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { width: 6px; background: transparent; }"
            "QScrollBar::handle:vertical { background: #333; border-radius: 3px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        )

        self._scroll_content = QWidget()
        self._content_layout = QVBoxLayout(self._scroll_content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)

        scroll.setWidget(self._scroll_content)
        root.addWidget(scroll, 1)

        # ── System group (pinned at bottom) ──
        self._system_group = _SidebarGroup("System")
        root.addWidget(self._system_group)

    def add_group(self, label: str, items: list[tuple[str, str]]) -> None:
        """Add a navigation group with (key, label) items."""
        group = _SidebarGroup(label)
        for key, lbl in items:
            icon_fn = SIDEBAR_ICONS.get(key)
            icon = icon_fn() if icon_fn else None
            btn = SidebarButton(key, lbl, icon=icon)
            btn.clicked_key.connect(self._on_button_clicked)
            group.add_button(btn)
            self._buttons[key] = btn
        self._groups.append(group)
        self._content_layout.addWidget(group)

    def add_system_item(self, key: str, label: str) -> None:
        """Add an item to the bottom-pinned System group."""
        icon_fn = SIDEBAR_ICONS.get(key, SIDEBAR_ICONS.get("plugin"))
        icon = icon_fn() if icon_fn else None
        btn = SidebarButton(key, label, icon=icon)
        btn.clicked_key.connect(self._on_button_clicked)
        self._system_group.add_button(btn)
        self._buttons[key] = btn

    def add_plugin_item(self, key: str, label: str) -> None:
        """Add a dynamic plugin to the System group."""
        self.add_system_item(key, label)

    def finalize(self) -> None:
        """Call after all groups are added to insert bottom spacer."""
        self._content_layout.addStretch(1)

    def select_item(self, key: str, emit: bool = True) -> None:
        """Select a sidebar item by key, updating visual state."""
        if key == self._current_key:
            return
        # Deselect previous
        prev = self._buttons.get(self._current_key)
        if prev:
            prev.active = False
        # Select new
        btn = self._buttons.get(key)
        if btn:
            btn.active = True
            self._current_key = key
            if emit:
                self.item_selected.emit(key)

    @property
    def current_key(self) -> str:
        return self._current_key

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        target = _COLLAPSED_WIDTH if self._collapsed else _EXPANDED_WIDTH

        anim = QPropertyAnimation(self, b"maximumWidth")
        anim.setDuration(150)
        anim.setStartValue(self.width())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        # Keep reference so it doesn't get GC'd
        self._anim = anim
        anim.start()

        # Also animate minimumWidth
        anim2 = QPropertyAnimation(self, b"minimumWidth")
        anim2.setDuration(150)
        anim2.setStartValue(self.width())
        anim2.setEndValue(target)
        anim2.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim2 = anim2
        anim2.start()

        self._title.setVisible(not self._collapsed)
        self._collapse_btn.setToolTip(
            "Expand sidebar" if self._collapsed else "Collapse sidebar"
        )
        for group in self._groups:
            group.set_collapsed(self._collapsed)
        self._system_group.set_collapsed(self._collapsed)

    def _on_button_clicked(self, key: str) -> None:
        self.select_item(key)

    @staticmethod
    def _make_collapse_icon(size: int = 18) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(150, 150, 150), 1.5))
        # Hamburger menu
        for y in (4, 9, 14):
            p.drawLine(3, y, 15, y)
        p.end()
        return QIcon(pm)
