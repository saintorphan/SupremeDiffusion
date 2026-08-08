"""Range slider widget with draggable begin/end handles."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QBrush, QPen
from PySide6.QtWidgets import QWidget, QSizePolicy


class RangeSliderWidget(QWidget):
    """A dual-handle range slider for selecting begin/end on a timeline.

    Signals:
        range_changed(float, float): emitted when either handle moves (begin, end in 0..1)
        begin_dragging(float): emitted while the begin handle is being dragged
        end_dragging(float): emitted while the end handle is being dragged
        begin_released(float): emitted when begin handle drag finishes
        end_released(float): emitted when end handle drag finishes
    """

    range_changed = Signal(float, float)
    begin_dragging = Signal(float)
    end_dragging = Signal(float)
    begin_released = Signal(float)
    end_released = Signal(float)

    _HANDLE_W = 10
    _TRACK_H = 24
    _MARGIN = 12

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._begin = 0.0  # 0..1
        self._end = 1.0    # 0..1
        self._dragging: str | None = None  # "begin", "end", or None

        self.setMinimumHeight(self._TRACK_H + 4)
        self.setMaximumHeight(self._TRACK_H + 4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

    # -- Public API --------------------------------------------------------

    def set_range(self, begin: float, end: float) -> None:
        """Set the range programmatically (values in 0..1)."""
        self._begin = max(0.0, min(1.0, begin))
        self._end = max(self._begin, min(1.0, end))
        self.update()

    def begin(self) -> float:
        return self._begin

    def end(self) -> float:
        return self._end

    # -- Coordinate helpers ------------------------------------------------

    def _track_rect(self) -> QRect:
        m = self._MARGIN
        return QRect(m, 0, self.width() - 2 * m, self._TRACK_H)

    def _val_to_x(self, val: float) -> int:
        tr = self._track_rect()
        return int(tr.x() + val * tr.width())

    def _x_to_val(self, x: int) -> float:
        tr = self._track_rect()
        if tr.width() == 0:
            return 0.0
        return max(0.0, min(1.0, (x - tr.x()) / tr.width()))

    def _handle_rect(self, which: str) -> QRect:
        val = self._begin if which == "begin" else self._end
        x = self._val_to_x(val)
        hw = self._HANDLE_W
        return QRect(x - hw // 2, 0, hw, self._TRACK_H)

    def _hit_test(self, x: int) -> str | None:
        """Return which handle is under x, or None."""
        br = self._handle_rect("begin")
        er = self._handle_rect("end")
        # Give priority to closer handle
        d_begin = abs(x - br.center().x())
        d_end = abs(x - er.center().x())
        thresh = self._HANDLE_W + 4
        if d_begin <= thresh and d_end <= thresh:
            return "begin" if d_begin <= d_end else "end"
        if d_begin <= thresh:
            return "begin"
        if d_end <= thresh:
            return "end"
        return None

    # -- Paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        tr = self._track_rect()

        # Background track
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(60, 60, 60)))
        p.drawRoundedRect(tr, 4, 4)

        # Selected range highlight
        bx = self._val_to_x(self._begin)
        ex = self._val_to_x(self._end)
        sel = QRect(bx, tr.y() + 2, ex - bx, tr.height() - 4)
        p.setBrush(QBrush(QColor(70, 130, 200, 120)))
        p.drawRect(sel)

        # Begin handle (bracket shape)
        self._draw_handle(p, "begin", QColor(100, 180, 255))
        # End handle
        self._draw_handle(p, "end", QColor(100, 180, 255))

        p.end()

    def _draw_handle(self, p: QPainter, which: str, color: QColor) -> None:
        r = self._handle_rect(which)
        is_dragging = self._dragging == which

        fill = color if is_dragging else QColor(color.red(), color.green(), color.blue(), 180)
        p.setPen(QPen(QColor(255, 255, 255, 200), 1))
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(r, 3, 3)

        # Draw bracket lines
        p.setPen(QPen(QColor(255, 255, 255), 2))
        cx = r.center().x()
        p.drawLine(cx, r.y() + 4, cx, r.y() + r.height() - 4)

    # -- Mouse events ------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = self._hit_test(event.position().x())
            if self._dragging:
                self._update_from_mouse(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self._update_from_mouse(event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._dragging:
            which = self._dragging
            self._dragging = None
            self.update()
            val = self._begin if which == "begin" else self._end
            if which == "begin":
                self.begin_released.emit(val)
            else:
                self.end_released.emit(val)

    def _update_from_mouse(self, x: float) -> None:
        val = self._x_to_val(int(x))
        if self._dragging == "begin":
            self._begin = min(val, self._end - 0.001)
            self.begin_dragging.emit(self._begin)
        elif self._dragging == "end":
            self._end = max(val, self._begin + 0.001)
            self.end_dragging.emit(self._end)
        self.range_changed.emit(self._begin, self._end)
        self.update()
