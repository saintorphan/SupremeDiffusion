"""Zoomable horizontal scrollbar — drag edges to zoom, drag middle to scroll.

Behaves like the timeline scrollbar in DAWs and video editors: the "thumb"
represents the visible portion of the timeline, and dragging its left or
right edge zooms in/out while keeping that edge anchored.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget


_BAR_HEIGHT = 14
_EDGE_ZONE = 7  # pixels from thumb edge that trigger resize cursor
_MIN_THUMB_PX = 20


class ZoomScrollBar(QWidget):
    """Custom horizontal scrollbar with edge-drag zoom.

    Signals:
        scroll_changed(float): normalised scroll position 0..1
        zoom_changed(float, float): (visible_start, visible_end) as fractions 0..1
    """

    scroll_changed = Signal(float)
    zoom_changed = Signal(float, float)  # (start_frac, end_frac)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_BAR_HEIGHT)
        self.setMouseTracking(True)

        # Visible range as fractions of total content [0..1]
        self._start: float = 0.0
        self._end: float = 1.0

        self._dragging: str = ""  # "", "left", "right", "middle"
        self._drag_offset: float = 0.0

    # -- Public API --------------------------------------------------------

    def set_range(self, start: float, end: float) -> None:
        """Set the visible range (0..1)."""
        self._start = max(0.0, min(start, 1.0))
        self._end = max(self._start + 0.001, min(end, 1.0))
        self.update()

    def visible_range(self) -> tuple[float, float]:
        return (self._start, self._end)

    # -- Painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Track background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(30, 30, 30)))
        p.drawRoundedRect(0, 0, w, h, 3, 3)

        # Thumb
        tx = int(self._start * w)
        tw = max(_MIN_THUMB_PX, int((self._end - self._start) * w))
        tw = min(tw, w - tx)

        thumb_color = QColor(80, 80, 90) if not self._dragging else QColor(100, 110, 130)
        p.setBrush(QBrush(thumb_color))
        p.setPen(QPen(QColor(60, 60, 70), 1))
        p.drawRoundedRect(tx, 1, tw, h - 2, 3, 3)

        # Edge grip lines
        grip_color = QColor(140, 140, 150) if not self._dragging else QColor(180, 180, 200)
        p.setPen(QPen(grip_color, 1))
        if tw > 30:
            # Left grip
            for dx in (3, 5):
                p.drawLine(tx + dx, 4, tx + dx, h - 4)
            # Right grip
            for dx in (3, 5):
                p.drawLine(tx + tw - dx, 4, tx + tw - dx, h - 4)

        p.end()

    # -- Mouse interaction -------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        x = event.position().x()
        w = self.width()
        tx = self._start * w
        tw = max(_MIN_THUMB_PX, (self._end - self._start) * w)

        if abs(x - tx) <= _EDGE_ZONE:
            self._dragging = "left"
        elif abs(x - (tx + tw)) <= _EDGE_ZONE:
            self._dragging = "right"
        elif tx <= x <= tx + tw:
            self._dragging = "middle"
            self._drag_offset = x - tx
        else:
            # Click outside thumb — jump scroll
            frac = x / w
            span = self._end - self._start
            self._start = max(0.0, min(frac - span / 2, 1.0 - span))
            self._end = self._start + span
            self._emit_zoom()
            self._dragging = "middle"
            self._drag_offset = (self._end - self._start) * w / 2

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        w = self.width()

        if not self._dragging:
            # Update cursor based on hover position
            tx = self._start * w
            tw = max(_MIN_THUMB_PX, (self._end - self._start) * w)
            if abs(x - tx) <= _EDGE_ZONE or abs(x - (tx + tw)) <= _EDGE_ZONE:
                self.setCursor(Qt.SizeHorCursor)
            elif tx <= x <= tx + tw:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        frac = max(0.0, min(x / w, 1.0))
        min_span = _MIN_THUMB_PX / max(w, 1)

        if self._dragging == "left":
            new_start = min(frac, self._end - min_span)
            self._start = max(0.0, new_start)
            self._emit_zoom()
        elif self._dragging == "right":
            new_end = max(frac, self._start + min_span)
            self._end = min(1.0, new_end)
            self._emit_zoom()
        elif self._dragging == "middle":
            span = self._end - self._start
            new_start = (x - self._drag_offset) / w
            new_start = max(0.0, min(new_start, 1.0 - span))
            self._start = new_start
            self._end = new_start + span
            self._emit_zoom()

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = ""
            self.update()
            self.setCursor(Qt.ArrowCursor)

    # -- Internals ---------------------------------------------------------

    def _emit_zoom(self) -> None:
        self.zoom_changed.emit(self._start, self._end)
