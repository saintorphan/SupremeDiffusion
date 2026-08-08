"""Selection model -- QPainterPath-based selection with marching ants overlay."""

from __future__ import annotations

from PySide6.QtCore import QObject, QRectF, QTimer, Signal, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene


class Selection(QObject):
    """Selection state: a QPainterPath mask, or None (= everything selected)."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.path: QPainterPath | None = None

    def set_path(self, path: QPainterPath) -> None:
        self.path = path
        self.changed.emit()

    def clear(self) -> None:
        self.path = None
        self.changed.emit()

    def invert(self, canvas_rect: QRectF) -> None:
        if self.path is None:
            # Everything selected -> nothing selected (empty path)
            self.path = QPainterPath()
        else:
            full = QPainterPath()
            full.addRect(canvas_rect)
            self.path = full.subtracted(self.path)
        self.changed.emit()

    def select_all(self, canvas_rect: QRectF) -> None:
        self.path = None  # None = everything
        self.changed.emit()

    def to_mask(self, width: int, height: int) -> QImage:
        """Return a binary mask (white = selected, black = not)."""
        mask = QImage(width, height, QImage.Format.Format_Grayscale8)
        if self.path is None:
            mask.fill(QColor(255, 255, 255))
            return mask
        mask.fill(QColor(0, 0, 0))
        painter = QPainter(mask)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPath(self.path)
        painter.end()
        return mask

    def is_empty(self) -> bool:
        """True if a selection path exists and is empty (no area)."""
        if self.path is None:
            return False  # None means everything is selected
        return self.path.isEmpty()

    def combine(self, new_path: QPainterPath, mode: str) -> None:
        """Combine new_path with existing selection using mode."""
        if mode == "replace" or self.path is None:
            self.set_path(new_path)
        elif mode == "add":
            self.set_path(self.path.united(new_path))
        elif mode == "subtract":
            self.set_path(self.path.subtracted(new_path))
        elif mode == "intersect":
            self.set_path(self.path.intersected(new_path))
        else:
            self.set_path(new_path)

    def select_opaque(self, layer) -> None:
        """Select all non-transparent pixels in a layer."""
        img = layer.image
        w, h = img.width(), img.height()
        # Create alpha mask via compositing
        white = QImage(w, h, QImage.Format.Format_ARGB32)
        white.fill(QColor(255, 255, 255, 255))
        p = QPainter(white)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        p.drawImage(0, 0, img)
        p.end()
        alpha = white.convertToFormat(QImage.Format.Format_Grayscale8)

        # Build path from alpha using run-length spans
        ptr = alpha.constBits()
        if ptr is None:
            return
        data = bytes(ptr)
        bpl = alpha.bytesPerLine()
        path = QPainterPath()
        ox, oy = layer.offset_x, layer.offset_y
        for y in range(h):
            x = 0
            while x < w:
                if data[y * bpl + x] > 0:
                    x_start = x
                    while x < w and data[y * bpl + x] > 0:
                        x += 1
                    path.addRect(QRectF(x_start + ox, y + oy, x - x_start, 1))
                else:
                    x += 1
        if not path.isEmpty():
            path = path.simplified()
            self.set_path(path)

    @property
    def has_selection(self) -> bool:
        """True if there is an active selection (not 'select all')."""
        return self.path is not None


class MarchingAnts:
    """Animated dual-stroke dashed-border overlay for the current selection.

    A black solid background stroke (z=9999) sits behind a white dashed
    foreground stroke (z=10000), ensuring visibility on any background.
    """

    def __init__(self, scene: QGraphicsScene) -> None:
        self._scene = scene
        self._bg_item: QGraphicsPathItem | None = None
        self._fg_item: QGraphicsPathItem | None = None
        self._dash_offset = 0.0
        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._advance)

    def update_path(self, path: QPainterPath | None) -> None:
        self._remove_items()
        if path is None or path.isEmpty():
            self._timer.stop()
            return

        # Background stroke: solid black
        bg_pen = QPen(QColor(0, 0, 0), 1.5, Qt.PenStyle.SolidLine)
        self._bg_item = self._scene.addPath(path, bg_pen)
        self._bg_item.setZValue(9999)

        # Foreground stroke: white dashed (marching)
        fg_pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
        fg_pen.setDashOffset(self._dash_offset)
        self._fg_item = self._scene.addPath(path, fg_pen)
        self._fg_item.setZValue(10000)

        self._timer.start()

    def _advance(self) -> None:
        self._dash_offset += 1.0
        if self._dash_offset > 100:
            self._dash_offset = 0.0
        if self._fg_item is not None:
            pen = self._fg_item.pen()
            pen.setDashOffset(self._dash_offset)
            self._fg_item.setPen(pen)

    def _remove_items(self) -> None:
        if self._bg_item is not None:
            self._scene.removeItem(self._bg_item)
            self._bg_item = None
        if self._fg_item is not None:
            self._scene.removeItem(self._fg_item)
            self._fg_item = None

    def remove(self) -> None:
        self._timer.stop()
        self._remove_items()
