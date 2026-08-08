"""Sidebar navigation button — icon + label with active/hover states."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QPushButton


class SidebarButton(QPushButton):
    """A sidebar nav item with icon, label, and visual states."""

    clicked_key = Signal(str)

    _STYLE = """
        SidebarButton {{
            background: transparent;
            color: {text};
            border: none;
            border-left: 3px solid {accent};
            text-align: left;
            padding: 8px 12px 8px 10px;
            font-size: 13px;
            font-weight: normal;
        }}
        SidebarButton:hover {{
            background: #2a2a2a;
            color: #ccc;
        }}
    """

    def __init__(
        self,
        key: str,
        label: str,
        icon: QIcon | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.key = key
        self._label = label
        self._active = False

        self.setText(f"  {label}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(36)
        self.setIconSize(QSize(18, 18))
        if icon:
            self.setIcon(icon)

        self._apply_style()
        self.clicked.connect(lambda: self.clicked_key.emit(self.key))

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self._active = value
        self._apply_style()

    def set_collapsed(self, collapsed: bool) -> None:
        """Switch between icon-only and icon+label mode."""
        if collapsed:
            self.setText("")
            self.setToolTip(self._label)
        else:
            self.setText(f"  {self._label}")
            self.setToolTip("")

    def _apply_style(self) -> None:
        if self._active:
            self.setStyleSheet(self._STYLE.format(
                text="#ffffff", accent="#0078d4",
            ))
            self.setFont(self.font())
            f = self.font()
            f.setBold(True)
            self.setFont(f)
        else:
            self.setStyleSheet(self._STYLE.format(
                text="#888888", accent="transparent",
            ))
            f = self.font()
            f.setBold(False)
            self.setFont(f)


# ── Programmatic icon builders ──────────────────────────────────────────────
# Same QPainter approach as MainWindow._make_help_icon / _make_ai_icon.

def _px(size: int = 18) -> tuple[QPixmap, QPainter]:
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    return pm, p


def make_icon_sparkle(color: QColor = QColor(120, 180, 255)) -> QIcon:
    """Quick Generate — sparkle / star."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    # 4-point star
    p.drawLine(9, 1, 9, 17)
    p.drawLine(1, 9, 17, 9)
    p.drawLine(4, 4, 14, 14)
    p.drawLine(14, 4, 4, 14)
    p.end()
    return QIcon(pm)


def make_icon_wand(color: QColor = QColor(180, 130, 255)) -> QIcon:
    """Pipeline Wizard — magic wand."""
    pm, p = _px()
    pen = QPen(color, 1.8)
    p.setPen(pen)
    p.drawLine(3, 15, 12, 6)
    p.drawLine(12, 6, 15, 3)
    # Star tip
    p.drawLine(14, 1, 16, 3)
    p.drawLine(12, 2, 16, 2)
    p.drawLine(14, 0, 14, 4)
    p.end()
    return QIcon(pm)


def make_icon_film(color: QColor = QColor(100, 200, 150)) -> QIcon:
    """Video — film strip."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(2, 3, 14, 12)
    # Sprocket holes
    for y in (5, 9, 13):
        p.drawRect(3, y, 2, 1)
        p.drawRect(13, y, 2, 1)
    # Play triangle
    p.setBrush(color)
    from PySide6.QtGui import QPolygon
    from PySide6.QtCore import QPoint
    p.drawPolygon(QPolygon([QPoint(8, 6), QPoint(8, 12), QPoint(12, 9)]))
    p.end()
    return QIcon(pm)


def make_icon_image(color: QColor = QColor(100, 180, 230)) -> QIcon:
    """Image — picture frame with mountain."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(2, 3, 14, 12)
    # Mountain
    p.drawLine(4, 13, 8, 7)
    p.drawLine(8, 7, 10, 10)
    p.drawLine(10, 10, 12, 8)
    p.drawLine(12, 8, 14, 13)
    # Sun
    p.setBrush(color)
    p.drawEllipse(12, 5, 3, 3)
    p.end()
    return QIcon(pm)


def make_icon_speaker(color: QColor = QColor(255, 170, 80)) -> QIcon:
    """Audio — speaker."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(3, 6, 3, 6)
    p.drawLine(6, 6, 10, 3)
    p.drawLine(6, 12, 10, 15)
    p.drawLine(10, 3, 10, 15)
    # Sound waves
    from PySide6.QtCore import QRectF
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(12, 5, 4, 8), 45 * 16, 270 * 16)
    p.end()
    return QIcon(pm)


def make_icon_timeline(color: QColor = QColor(130, 200, 130)) -> QIcon:
    """Timeline — horizontal bars."""
    pm, p = _px()
    pen = QPen(color, 2.0)
    p.setPen(pen)
    p.drawLine(2, 5, 14, 5)
    p.drawLine(4, 9, 16, 9)
    p.drawLine(2, 13, 12, 13)
    p.end()
    return QIcon(pm)


def make_icon_sequence(color: QColor = QColor(200, 180, 100)) -> QIcon:
    """Sequences — stacked cards."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(4, 2, 12, 8)
    p.drawRect(2, 5, 12, 8)
    p.drawRect(0, 8, 12, 8)
    p.end()
    return QIcon(pm)


def make_icon_palette(color: QColor = QColor(230, 130, 130)) -> QIcon:
    """Color Correct — palette / sliders."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    # Three vertical sliders
    for x, y in ((5, 6), (9, 10), (13, 4)):
        p.drawLine(x, 2, x, 16)
        p.setBrush(color)
        p.drawEllipse(x - 2, y, 4, 4)
        p.setBrush(Qt.BrushStyle.NoBrush)
    p.end()
    return QIcon(pm)


def make_icon_folder(color: QColor = QColor(200, 180, 100)) -> QIcon:
    """Outputs — folder."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(2, 6, 14, 10)
    p.drawLine(2, 6, 2, 4)
    p.drawLine(2, 4, 8, 4)
    p.drawLine(8, 4, 9, 6)
    p.end()
    return QIcon(pm)


def make_icon_library(color: QColor = QColor(150, 150, 220)) -> QIcon:
    """Library — books / grid."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    # Book spines
    p.drawRect(2, 3, 3, 12)
    p.drawRect(6, 3, 3, 12)
    p.drawRect(10, 3, 3, 12)
    p.drawLine(14, 5, 14, 15)
    p.drawLine(14, 5, 16, 5)
    p.end()
    return QIcon(pm)


def make_icon_person(color: QColor = QColor(180, 140, 200)) -> QIcon:
    """Daz2Supreme — person / figure."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    # Head
    p.drawEllipse(6, 1, 6, 6)
    # Body
    p.drawLine(9, 7, 9, 12)
    p.drawLine(9, 12, 5, 17)
    p.drawLine(9, 12, 13, 17)
    p.drawLine(9, 9, 4, 12)
    p.drawLine(9, 9, 14, 12)
    p.end()
    return QIcon(pm)


def make_icon_gear(color: QColor = QColor(160, 160, 160)) -> QIcon:
    """Settings — gear."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    # Outer circle with notches
    p.drawEllipse(4, 4, 10, 10)
    p.drawEllipse(6, 6, 6, 6)
    # Gear teeth (lines radiating out)
    import math
    cx, cy = 9, 9
    for i in range(8):
        angle = i * math.pi / 4
        x1 = cx + 5 * math.cos(angle)
        y1 = cy + 5 * math.sin(angle)
        x2 = cx + 7 * math.cos(angle)
        y2 = cy + 7 * math.sin(angle)
        p.drawLine(int(x1), int(y1), int(x2), int(y2))
    p.end()
    return QIcon(pm)


def make_icon_terminal(color: QColor = QColor(100, 200, 100)) -> QIcon:
    """Console — terminal prompt."""
    pm, p = _px()
    pen = QPen(color, 1.8)
    p.setPen(pen)
    p.drawRect(1, 2, 16, 14)
    # Prompt: >_
    p.drawLine(4, 9, 7, 7)
    p.drawLine(7, 7, 4, 11)
    p.drawLine(9, 11, 13, 11)
    p.end()
    return QIcon(pm)


def make_icon_puzzle(color: QColor = QColor(200, 160, 100)) -> QIcon:
    """Plugin — puzzle piece."""
    pm, p = _px()
    pen = QPen(color, 1.5)
    p.setPen(pen)
    p.drawRect(2, 6, 6, 6)
    p.drawRect(10, 6, 6, 6)
    p.drawRect(6, 2, 6, 6)
    p.drawRect(6, 10, 6, 6)
    p.end()
    return QIcon(pm)


# Icon registry
SIDEBAR_ICONS = {
    "quick": make_icon_sparkle,
    "pipeline": make_icon_wand,
    "video": make_icon_film,
    "image": make_icon_image,
    "audio": make_icon_speaker,
    "timeline": make_icon_timeline,
    "sequences": make_icon_sequence,
    "color": make_icon_palette,
    "outputs": make_icon_folder,
    "library": make_icon_library,
    "daz": make_icon_person,
    "settings": make_icon_gear,
    "console": make_icon_terminal,
    "plugin": make_icon_puzzle,
}
