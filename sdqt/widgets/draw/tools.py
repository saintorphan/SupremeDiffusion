"""Tool classes for the Draw editor canvas."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem

if TYPE_CHECKING:
    from sdqt.widgets.draw.canvas import DrawCanvas


# ── Blend modes ─────────────────────────────────────────────────────────────

BLEND_MODES = {
    "normal": QPainter.CompositionMode.CompositionMode_SourceOver,
    "multiply": QPainter.CompositionMode.CompositionMode_Multiply,
    "screen": QPainter.CompositionMode.CompositionMode_Screen,
    "overlay": QPainter.CompositionMode.CompositionMode_Overlay,
    "darken": QPainter.CompositionMode.CompositionMode_Darken,
    "lighten": QPainter.CompositionMode.CompositionMode_Lighten,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _scene_to_layer(pos: QPointF, layer) -> QPointF:
    """Convert scene coordinates to layer-local pixel coordinates."""
    t = QTransform()
    t.translate(layer.offset_x, layer.offset_y)
    cx = layer.image.width() / 2
    cy = layer.image.height() / 2
    t.translate(cx, cy)
    if layer.flip_h:
        t.scale(-1, 1)
    if layer.flip_v:
        t.scale(1, -1)
    t.rotate(layer.rotation)
    t.scale(layer.scale_x, layer.scale_y)
    t.translate(-cx, -cy)
    inv, ok = t.inverted()
    if ok:
        return inv.map(pos)
    return QPointF(pos.x() - layer.offset_x, pos.y() - layer.offset_y)


def _color_distance(c1: QColor, c2: QColor) -> float:
    """Euclidean distance in RGBA space between two colors."""
    dr = c1.red() - c2.red()
    dg = c1.green() - c2.green()
    db = c1.blue() - c2.blue()
    da = c1.alpha() - c2.alpha()
    return math.sqrt(dr * dr + dg * dg + db * db + da * da)


def _get_select_mode(event) -> str:
    """Determine selection combine mode from keyboard modifiers."""
    mods = event.modifiers() if hasattr(event, 'modifiers') else Qt.KeyboardModifier.NoModifier
    shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
    alt = bool(mods & Qt.KeyboardModifier.AltModifier)
    if shift and alt:
        return "intersect"
    elif shift:
        return "add"
    elif alt:
        return "subtract"
    return "replace"


class Tool:
    """Base class for all drawing tools."""

    def __init__(self, canvas: DrawCanvas) -> None:
        self.canvas = canvas

    def on_press(self, pos: QPointF, event) -> None:
        pass

    def on_move(self, pos: QPointF, event) -> None:
        pass

    def on_release(self, pos: QPointF, event) -> None:
        pass

    def on_key(self, event) -> None:
        pass

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)


# ── Drawing Tools ────────────────────────────────────────────────────────────


class BrushTool(Tool):
    """Paint onto the active layer's QImage."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.color = QColor(255, 0, 0)
        self.secondary_color = QColor(255, 255, 255)
        self.size = 8
        self.opacity = 1.0
        self.hardness = 100
        self.blend_mode = "normal"
        self._painting = False
        self._last_pos: QPointF | None = None
        self._active_color: QColor = self.color
        self._pressure: float = 1.0

    def set_pressure(self, pressure: float) -> None:
        self._pressure = pressure

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        layer = self.canvas.active_layer
        if layer is None or layer.locked:
            return
        # Pick color based on button
        if event.button() == Qt.MouseButton.RightButton:
            self._active_color = self.secondary_color
        else:
            self._active_color = self.color
        self._painting = True
        self._last_pos = None
        self._paint_at(pos)

    def on_move(self, pos: QPointF, event) -> None:
        if self._painting:
            self._paint_at(pos)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._painting = False
            self._last_pos = None

    def _make_tip(self) -> QImage:
        """Generate a soft brush tip image using a radial gradient."""
        d = max(1, self.size)
        tip = QImage(d, d, QImage.Format.Format_ARGB32)
        tip.fill(QColor(0, 0, 0, 0))
        center = d / 2.0
        grad = QRadialGradient(center, center, center)
        inner = self.hardness / 100.0
        color = self._active_color
        grad.setColorAt(0, color)
        grad.setColorAt(max(0.01, inner), color)
        grad.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        p = QPainter(tip)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(0, 0, d, d)
        p.end()
        return tip

    def _paint_at(self, pos: QPointF) -> None:
        layer = self.canvas.active_layer
        if layer is None:
            return
        local = _scene_to_layer(pos, layer)

        # Apply pressure scaling
        effective_size = max(1, int(self.size * self._pressure))
        effective_opacity = self.opacity * self._pressure

        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(effective_opacity)
        painter.setCompositionMode(
            BLEND_MODES.get(self.blend_mode, QPainter.CompositionMode.CompositionMode_SourceOver)
        )

        if self.hardness < 100:
            # Soft brush: stamp tip at spacing intervals
            tip = self._make_tip()
            d = max(1, effective_size)
            spacing = max(1, int(d * 0.25))
            if self._last_pos is not None:
                dx = local.x() - self._last_pos.x()
                dy = local.y() - self._last_pos.y()
                dist = math.sqrt(dx * dx + dy * dy)
                steps = max(1, int(dist / spacing))
                for i in range(steps):
                    t = (i + 1) / steps
                    sx = self._last_pos.x() + dx * t
                    sy = self._last_pos.y() + dy * t
                    painter.drawImage(int(sx - d / 2), int(sy - d / 2), tip)
            else:
                painter.drawImage(int(local.x() - d / 2), int(local.y() - d / 2), tip)
        else:
            # Hard brush: fast path with drawLine / drawEllipse
            if self._last_pos is not None:
                pen = QPen(self._active_color, effective_size, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self._last_pos, local)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(self._active_color))
                r = effective_size / 2
                painter.drawEllipse(local, r, r)

        painter.end()
        self._last_pos = local
        self.canvas.layer_stack.layer_updated.emit(layer.uid)


class EraserTool(Tool):
    """Erase (clear to transparent) on the active layer."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.size = 16
        self.opacity = 1.0
        self.hardness = 100
        self._erasing = False
        self._last_pos: QPointF | None = None
        self._pressure: float = 1.0

    def set_pressure(self, pressure: float) -> None:
        self._pressure = pressure

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = self.canvas.active_layer
        if layer is None or layer.locked:
            return
        self._erasing = True
        self._last_pos = None
        self._erase_at(pos)

    def on_move(self, pos: QPointF, event) -> None:
        if self._erasing:
            self._erase_at(pos)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._erasing = False
            self._last_pos = None

    def _make_tip(self) -> QImage:
        """Generate a soft eraser tip using a radial gradient (white = erase)."""
        d = max(1, self.size)
        tip = QImage(d, d, QImage.Format.Format_ARGB32)
        tip.fill(QColor(0, 0, 0, 0))
        center = d / 2.0
        grad = QRadialGradient(center, center, center)
        inner = self.hardness / 100.0
        # Use white with full alpha for the eraser mask
        grad.setColorAt(0, QColor(255, 255, 255, 255))
        grad.setColorAt(max(0.01, inner), QColor(255, 255, 255, 255))
        grad.setColorAt(1.0, QColor(255, 255, 255, 0))
        p = QPainter(tip)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(0, 0, d, d)
        p.end()
        return tip

    def _erase_at(self, pos: QPointF) -> None:
        layer = self.canvas.active_layer
        if layer is None:
            return
        local = _scene_to_layer(pos, layer)

        # Apply pressure scaling
        effective_size = max(1, int(self.size * self._pressure))
        effective_opacity = self.opacity * self._pressure

        if self.hardness < 100:
            # Soft eraser: stamp a soft tip using DestinationOut compositing
            tip = self._make_tip()
            d = max(1, effective_size)
            spacing = max(1, int(d * 0.25))
            painter = QPainter(layer.image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setOpacity(effective_opacity)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            if self._last_pos is not None:
                dx = local.x() - self._last_pos.x()
                dy = local.y() - self._last_pos.y()
                dist = math.sqrt(dx * dx + dy * dy)
                steps = max(1, int(dist / spacing))
                for i in range(steps):
                    t = (i + 1) / steps
                    sx = self._last_pos.x() + dx * t
                    sy = self._last_pos.y() + dy * t
                    painter.drawImage(int(sx - d / 2), int(sy - d / 2), tip)
            else:
                painter.drawImage(int(local.x() - d / 2), int(local.y() - d / 2), tip)
            painter.end()
        else:
            # Hard eraser: fast path
            painter = QPainter(layer.image)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setOpacity(effective_opacity)
            if self._last_pos is not None:
                pen = QPen(QColor(0, 0, 0, 0), effective_size, Qt.PenStyle.SolidLine,
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self._last_pos, local)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
                r = effective_size / 2
                painter.drawEllipse(local, r, r)
            painter.end()

        self._last_pos = local
        self.canvas.layer_stack.layer_updated.emit(layer.uid)


class GrabTool(Tool):
    """Drag to move the active layer's offset position."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self._dragging = False
        self._drag_start: QPointF | None = None
        self._orig_offset: tuple[float, float] = (0, 0)

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = self.canvas.active_layer
        if layer is None or layer.locked:
            return
        self._dragging = True
        self._drag_start = pos
        self._orig_offset = (layer.offset_x, layer.offset_y)

    def on_move(self, pos: QPointF, event) -> None:
        if not self._dragging or self._drag_start is None:
            return
        layer = self.canvas.active_layer
        if layer is None:
            return
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        layer.offset_x = self._orig_offset[0] + dx
        layer.offset_y = self._orig_offset[1] + dy
        self.canvas.layer_stack.layer_updated.emit(layer.uid)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_start = None

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.OpenHandCursor)


# ── Shape Tools ──────────────────────────────────────────────────────────────
class _ShapeToolBase(Tool):
    """Base for rubber-band shape tools that rasterize onto the active layer."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.line_color = QColor(255, 255, 255)
        self.fill_color = QColor(100, 100, 255, 128)
        self.filled = True
        self.line_width = 2
        self._start: QPointF | None = None
        self._preview_item = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.canvas.snapshot_active()
        self._start = pos
        self._create_preview(pos)

    def on_move(self, pos: QPointF, event) -> None:
        if self._start is not None:
            self._update_preview(pos, event)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        self._remove_preview()
        self._rasterize(self._start, pos, event)
        self._start = None

    def _create_preview(self, pos: QPointF) -> None:
        pass

    def _update_preview(self, pos: QPointF, event=None) -> None:
        pass

    def _remove_preview(self) -> None:
        if self._preview_item is not None:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None

    def _rasterize(self, p1: QPointF, p2: QPointF, event=None) -> None:
        pass

    def _make_pen(self) -> QPen:
        return QPen(self.line_color, self.line_width)

    def _make_brush(self) -> QBrush:
        if self.filled:
            return QBrush(self.fill_color)
        return QBrush(Qt.BrushStyle.NoBrush)

    @staticmethod
    def _constrain(start: QPointF, end: QPointF, event) -> QPointF:
        """If Shift is held, constrain the shape (square for rect/ellipse, 45deg for line)."""
        if event is None:
            return end
        mods = event.modifiers() if hasattr(event, 'modifiers') else Qt.KeyboardModifier.NoModifier
        if not (mods & Qt.KeyboardModifier.ShiftModifier):
            return end
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        side = max(abs(dx), abs(dy))
        sx = side if dx >= 0 else -side
        sy = side if dy >= 0 else -side
        return QPointF(start.x() + sx, start.y() + sy)


class LineTool(_ShapeToolBase):
    """Draw a straight line onto the active layer."""

    def _create_preview(self, pos: QPointF) -> None:
        self._preview_item = self.canvas.scene().addLine(
            pos.x(), pos.y(), pos.x(), pos.y(), self._make_pen(),
        )
        self._preview_item.setZValue(9999)

    def _update_preview(self, pos: QPointF, event=None) -> None:
        if self._preview_item and self._start:
            end = self._constrain_line(self._start, pos, event)
            self._preview_item.setLine(
                self._start.x(), self._start.y(), end.x(), end.y(),
            )

    def _rasterize(self, p1: QPointF, p2: QPointF, event=None) -> None:
        layer = self.canvas.active_layer
        if layer is None:
            return
        p2 = self._constrain_line(p1, p2, event)
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._make_pen())
        painter.drawLine(_scene_to_layer(p1, layer), _scene_to_layer(p2, layer))
        painter.end()
        self.canvas.layer_stack.layer_updated.emit(layer.uid)

    @staticmethod
    def _constrain_line(start: QPointF, end: QPointF, event) -> QPointF:
        """Shift-constrain: snap line to nearest 45-degree angle."""
        if event is None:
            return end
        mods = event.modifiers() if hasattr(event, 'modifiers') else Qt.KeyboardModifier.NoModifier
        if not (mods & Qt.KeyboardModifier.ShiftModifier):
            return end
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return end
        angle = math.atan2(dy, dx)
        # Snap to nearest 45deg increment
        snapped = round(angle / (math.pi / 4)) * (math.pi / 4)
        return QPointF(start.x() + length * math.cos(snapped),
                       start.y() + length * math.sin(snapped))


class EllipseTool(_ShapeToolBase):
    """Draw an ellipse onto the active layer."""

    def _create_preview(self, pos: QPointF) -> None:
        self._preview_item = self.canvas.scene().addEllipse(
            QRectF(pos, pos), self._make_pen(), self._make_brush(),
        )
        self._preview_item.setZValue(9999)

    def _update_preview(self, pos: QPointF, event=None) -> None:
        if self._preview_item and self._start:
            end = self._constrain(self._start, pos, event)
            self._preview_item.setRect(QRectF(self._start, end).normalized())

    def _rasterize(self, p1: QPointF, p2: QPointF, event=None) -> None:
        layer = self.canvas.active_layer
        if layer is None:
            return
        p2 = self._constrain(p1, p2, event)
        rect = QRectF(
            _scene_to_layer(p1, layer),
            _scene_to_layer(p2, layer),
        ).normalized()
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._make_pen())
        painter.setBrush(self._make_brush())
        painter.drawEllipse(rect)
        painter.end()
        self.canvas.layer_stack.layer_updated.emit(layer.uid)


class RectangleTool(_ShapeToolBase):
    """Draw a rectangle onto the active layer."""

    def _create_preview(self, pos: QPointF) -> None:
        self._preview_item = self.canvas.scene().addRect(
            QRectF(pos, pos), self._make_pen(), self._make_brush(),
        )
        self._preview_item.setZValue(9999)

    def _update_preview(self, pos: QPointF, event=None) -> None:
        if self._preview_item and self._start:
            end = self._constrain(self._start, pos, event)
            self._preview_item.setRect(QRectF(self._start, end).normalized())

    def _rasterize(self, p1: QPointF, p2: QPointF, event=None) -> None:
        layer = self.canvas.active_layer
        if layer is None:
            return
        p2 = self._constrain(p1, p2, event)
        rect = QRectF(
            _scene_to_layer(p1, layer),
            _scene_to_layer(p2, layer),
        ).normalized()
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._make_pen())
        painter.setBrush(self._make_brush())
        painter.drawRect(rect)
        painter.end()
        self.canvas.layer_stack.layer_updated.emit(layer.uid)


# ── Eyedropper Tool ─────────────────────────────────────────────────────────

class EyedropperTool(Tool):
    """Sample color from the composited canvas at click position."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.on_color_sampled = None  # callback: fn(QColor)

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        # Sample from the flattened composite (what the user sees)
        composite = self.canvas.flatten()
        if composite.isNull():
            return
        x, y = int(pos.x()), int(pos.y())
        if 0 <= x < composite.width() and 0 <= y < composite.height():
            color = composite.pixelColor(x, y)
            if color.alpha() > 0 and self.on_color_sampled:
                self.on_color_sampled(color)

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)


# ── Flood Fill Tool ─────────────────────────────────────────────────────────

class FloodFillTool(Tool):
    """Flood-fill a region with the brush color using tolerance matching."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.color = QColor(255, 0, 0)
        self.tolerance = 32

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = self.canvas.active_layer
        if layer is None or layer.locked:
            return
        img = layer.image
        local = _scene_to_layer(pos, layer)
        x0, y0 = int(local.x()), int(local.y())
        w, h = img.width(), img.height()
        if x0 < 0 or y0 < 0 or x0 >= w or y0 >= h:
            return

        self.canvas.snapshot_active()

        # Get raw pixels for fast access (Format_ARGB32: 4 bytes per pixel)
        ptr = img.constBits()
        if ptr is None:
            return
        data = bytes(ptr)
        bpl = img.bytesPerLine()

        def px(x, y):
            off = y * bpl + x * 4
            return data[off:off + 4]

        target = px(x0, y0)
        tol_sq = self.tolerance * self.tolerance

        # BFS with flat boolean array
        visited = bytearray(w * h)
        mask = QImage(w, h, QImage.Format.Format_Grayscale8)
        mask.fill(QColor(0, 0, 0))
        mask_ptr = mask.bits()
        mask_bpl = mask.bytesPerLine()
        has_fill = False

        stack = [(x0, y0)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            idx = y * w + x
            if visited[idx]:
                continue
            visited[idx] = 1
            p = px(x, y)
            dist_sq = sum((a - b) ** 2 for a, b in zip(target, p))
            if dist_sq <= tol_sq:
                mask_ptr[y * mask_bpl + x] = 255
                has_fill = True
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

        if has_fill:
            # Build a colored image masked by the flood region
            clip_img = QImage(w, h, QImage.Format.Format_ARGB32)
            clip_img.fill(QColor(0, 0, 0, 0))
            cp = QPainter(clip_img)
            cp.setPen(Qt.PenStyle.NoPen)
            cp.setBrush(QBrush(self.color))
            cp.drawRect(0, 0, w, h)
            cp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            cp.drawImage(0, 0, mask.convertToFormat(QImage.Format.Format_ARGB32))
            cp.end()

            painter = QPainter(layer.image)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self.color))
            painter.drawImage(0, 0, clip_img)
            painter.end()
            self.canvas.layer_stack.layer_updated.emit(layer.uid)

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)


# ── Selection Tools ──────────────────────────────────────────────────────────

class FreehandSelectTool(Tool):
    """Freehand lasso -- draw a closed selection path."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self._path: QPainterPath | None = None
        self._preview_item = None
        self._drawing = False
        self._press_event = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press_event = event
        self._path = QPainterPath()
        self._path.moveTo(pos)
        self._drawing = True
        self._preview_item = self.canvas.scene().addPath(
            self._path, QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine),
        )
        self._preview_item.setZValue(9999)

    def on_move(self, pos: QPointF, event) -> None:
        if self._drawing and self._path:
            self._path.lineTo(pos)
            if self._preview_item:
                self._preview_item.setPath(self._path)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing = False
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None
        if self._path:
            self._path.closeSubpath()
            mode = _get_select_mode(self._press_event) if self._press_event else "replace"
            if mode == "replace":
                self.canvas.selection.set_path(self._path)
            else:
                self.canvas.selection.combine(self._path, mode)
        self._path = None
        self._press_event = None

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)


class PolygonLassoTool(Tool):
    """Click to add vertices, double-click to close the polygon selection."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self._points: list[QPointF] = []
        self._path = QPainterPath()
        self._preview_item = None
        self._press_event = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._press_event is None:
            self._press_event = event
        self._points.append(pos)
        self._update_path(pos)

    def on_move(self, pos: QPointF, event) -> None:
        if self._points:
            self._update_path(pos)

    def on_release(self, pos: QPointF, event) -> None:
        pass  # vertices added on press

    def on_key(self, event) -> None:
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self._close()
        elif event.key() == Qt.Key.Key_Escape:
            self._cancel()

    def _update_path(self, cursor_pos: QPointF) -> None:
        path = QPainterPath()
        if self._points:
            path.moveTo(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
            path.lineTo(cursor_pos)
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
        self._preview_item = self.canvas.scene().addPath(
            path, QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine),
        )
        self._preview_item.setZValue(9999)

    def _close(self) -> None:
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None
        if len(self._points) >= 3:
            path = QPainterPath()
            path.moveTo(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
            path.closeSubpath()
            mode = _get_select_mode(self._press_event) if self._press_event else "replace"
            if mode == "replace":
                self.canvas.selection.set_path(path)
            else:
                self.canvas.selection.combine(path, mode)
        self._points.clear()
        self._press_event = None

    def _cancel(self) -> None:
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None
        self._points.clear()
        self._press_event = None


class MagneticLassoTool(Tool):
    """Edge-snapping lasso using simple gradient-based edge detection."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self._points: list[QPointF] = []
        self._preview_item = None
        self._drawing = False
        self.snap_radius = 10
        self._press_event = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._press_event = event
        self._drawing = True
        snapped = self._snap_to_edge(pos)
        self._points = [snapped]
        self._update_preview()

    def on_move(self, pos: QPointF, event) -> None:
        if not self._drawing:
            return
        snapped = self._snap_to_edge(pos)
        # Only add point if moved enough distance
        if self._points:
            last = self._points[-1]
            dx = snapped.x() - last.x()
            dy = snapped.y() - last.y()
            if math.sqrt(dx * dx + dy * dy) > 4:
                self._points.append(snapped)
                self._update_preview()

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._drawing:
            return
        self._drawing = False
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None
        if len(self._points) >= 3:
            path = QPainterPath()
            path.moveTo(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
            path.closeSubpath()
            mode = _get_select_mode(self._press_event) if self._press_event else "replace"
            if mode == "replace":
                self.canvas.selection.set_path(path)
            else:
                self.canvas.selection.combine(path, mode)
        self._points.clear()
        self._press_event = None

    def _snap_to_edge(self, pos: QPointF) -> QPointF:
        """Snap pos to the nearest strong edge within snap_radius."""
        layer = self.canvas.active_layer
        if layer is None:
            return pos
        img = layer.image
        # Convert scene pos to layer-local for pixel sampling
        local = _scene_to_layer(pos, layer)
        x0, y0 = int(local.x()), int(local.y())
        r = self.snap_radius
        best_strength = 0.0
        best_lx, best_ly = x0, y0

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = x0 + dx, y0 + dy
                if 0 < px < img.width() - 1 and 0 < py < img.height() - 1:
                    c_l = QColor(img.pixel(px - 1, py)).lightnessF()
                    c_r = QColor(img.pixel(px + 1, py)).lightnessF()
                    c_t = QColor(img.pixel(px, py - 1)).lightnessF()
                    c_b = QColor(img.pixel(px, py + 1)).lightnessF()
                    gx = abs(c_r - c_l)
                    gy = abs(c_b - c_t)
                    strength = math.sqrt(gx * gx + gy * gy)
                    if strength > best_strength:
                        best_strength = strength
                        best_lx, best_ly = px, py

        # Convert best local coords back to scene coords
        return QPointF(best_lx + layer.offset_x, best_ly + layer.offset_y)

    def _update_preview(self) -> None:
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
        path = QPainterPath()
        if self._points:
            path.moveTo(self._points[0])
            for p in self._points[1:]:
                path.lineTo(p)
        self._preview_item = self.canvas.scene().addPath(
            path, QPen(QColor(0, 255, 255), 1, Qt.PenStyle.DashLine),
        )
        self._preview_item.setZValue(9999)


class MagicWandTool(Tool):
    """Flood-fill selection by color similarity."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.tolerance = 32

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = self.canvas.active_layer
        if layer is None:
            return
        img = layer.image
        # Convert scene pos to layer-local for pixel sampling
        local = _scene_to_layer(pos, layer)
        x0, y0 = int(local.x()), int(local.y())
        w, h = img.width(), img.height()
        if x0 < 0 or y0 < 0 or x0 >= w or y0 >= h:
            return

        # Get raw pixels for fast access
        ptr = img.constBits()
        if ptr is None:
            return
        data = bytes(ptr)
        bpl = img.bytesPerLine()

        def px(x, y):
            off = y * bpl + x * 4
            return data[off:off + 4]

        target = px(x0, y0)
        tol_sq = self.tolerance * self.tolerance

        # BFS with flat boolean array and grayscale mask
        visited = bytearray(w * h)
        mask = QImage(w, h, QImage.Format.Format_Grayscale8)
        mask.fill(QColor(0, 0, 0))
        mask_ptr = mask.bits()
        mask_bpl = mask.bytesPerLine()
        has_selection = False

        stack = [(x0, y0)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            idx = y * w + x
            if visited[idx]:
                continue
            visited[idx] = 1
            p = px(x, y)
            dist_sq = sum((a - b) ** 2 for a, b in zip(target, p))
            if dist_sq <= tol_sq:
                mask_ptr[y * mask_bpl + x] = 255
                has_selection = True
                stack.append((x + 1, y))
                stack.append((x - 1, y))
                stack.append((x, y + 1))
                stack.append((x, y - 1))

        if has_selection:
            # Read mask data and build path using run-length spans
            mask_data = bytes(mask.constBits())
            ox, oy = layer.offset_x, layer.offset_y
            path = QPainterPath()
            for y in range(h):
                x = 0
                while x < w:
                    if mask_data[y * mask_bpl + x]:
                        x_start = x
                        while x < w and mask_data[y * mask_bpl + x]:
                            x += 1
                        path.addRect(QRectF(x_start + ox, y + oy, x - x_start, 1))
                    else:
                        x += 1
            path = path.simplified()
            mode = _get_select_mode(event)
            if mode == "replace":
                self.canvas.selection.set_path(path)
            else:
                self.canvas.selection.combine(path, mode)

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.PointingHandCursor)


# ── Smudge Tool ─────────────────────────────────────────────────────────────

class SmudgeTool(Tool):
    """Smudge/blend tool -- samples color and pushes it along the stroke."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.size = 20
        self.strength = 0.5
        self._smudging = False
        self._last_pos: QPointF | None = None
        self._sample: QImage | None = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        layer = self.canvas.active_layer
        if layer is None or layer.locked:
            return
        self.canvas.snapshot_active()
        self._smudging = True
        local = _scene_to_layer(pos, layer)
        self._sample = self._grab_sample(layer.image, local)
        self._last_pos = local

    def on_move(self, pos: QPointF, event) -> None:
        if not self._smudging or self._sample is None:
            return
        layer = self.canvas.active_layer
        if layer is None:
            return
        local = _scene_to_layer(pos, layer)
        # Blend current sample with area under cursor
        new_sample = self._grab_sample(layer.image, local)
        if new_sample is None:
            return
        # Mix: result = old_sample * strength + new_sample * (1 - strength)
        blended = QImage(self._sample.size(), QImage.Format.Format_ARGB32)
        blended.fill(QColor(0, 0, 0, 0))
        p = QPainter(blended)
        p.drawImage(0, 0, new_sample)
        p.setOpacity(self.strength)
        p.drawImage(0, 0, self._sample)
        p.end()
        # Stamp blended onto layer
        d = self.size
        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawImage(int(local.x() - d / 2), int(local.y() - d / 2), blended)
        painter.end()
        self._sample = blended
        self._last_pos = local
        self.canvas.layer_stack.layer_updated.emit(layer.uid)

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._smudging = False
            self._last_pos = None
            self._sample = None

    def _grab_sample(self, image: QImage, center: QPointF) -> QImage | None:
        d = self.size
        x = int(center.x() - d / 2)
        y = int(center.y() - d / 2)
        return image.copy(x, y, d, d)

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)


# ── Text Tool ───────────────────────────────────────────────────────────────

class TextTool(Tool):
    """Click to place text on a new layer."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.font_family = "Sans Serif"
        self.font_size = 32
        self.color = QColor(255, 255, 255)
        self.bold = False
        self.italic = False

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        from PySide6.QtWidgets import (
            QCheckBox,
            QDialog,
            QFontComboBox,
            QHBoxLayout,
            QLabel,
            QPlainTextEdit,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
        )
        from PySide6.QtGui import QFont, QFontMetrics

        dlg = QDialog(self.canvas)
        dlg.setWindowTitle("Add Text")
        dlg.setMinimumWidth(350)
        layout = QVBoxLayout(dlg)

        text_edit = QPlainTextEdit()
        text_edit.setPlaceholderText("Enter text...")
        text_edit.setMaximumHeight(80)
        layout.addWidget(text_edit)

        r1 = QHBoxLayout()
        font_combo = QFontComboBox()
        r1.addWidget(font_combo)
        size_spin = QSpinBox()
        size_spin.setRange(6, 200)
        size_spin.setValue(self.font_size)
        r1.addWidget(size_spin)
        bold_check = QCheckBox("Bold")
        bold_check.setChecked(self.bold)
        r1.addWidget(bold_check)
        italic_check = QCheckBox("Italic")
        italic_check.setChecked(self.italic)
        r1.addWidget(italic_check)
        layout.addLayout(r1)

        r2 = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dlg.accept)
        r2.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        r2.addWidget(cancel_btn)
        layout.addLayout(r2)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        text = text_edit.toPlainText()
        if not text.strip():
            return

        font = QFont(font_combo.currentFont().family(), size_spin.value())
        font.setBold(bold_check.isChecked())
        font.setItalic(italic_check.isChecked())

        self.font_family = font_combo.currentFont().family()
        self.font_size = size_spin.value()
        self.bold = bold_check.isChecked()
        self.italic = italic_check.isChecked()

        # Measure text
        fm = QFontMetrics(font)
        lines = text.split('\n')
        text_w = max(fm.horizontalAdvance(line) for line in lines) + 10
        text_h = fm.height() * len(lines) + 10

        # Render text onto a new layer image
        w, h = self.canvas.canvas_size
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        p.setFont(font)
        p.setPen(QPen(self.color))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scene_pos = pos
        local_x = int(scene_pos.x())
        local_y = int(scene_pos.y())
        for i, line in enumerate(lines):
            p.drawText(local_x, local_y + fm.ascent() + i * fm.height(), line)
        p.end()

        self.canvas.add_image_layer(img, f"Text: {text[:20]}")

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.IBeamCursor)


# ── Gradient Tool ───────────────────────────────────────────────────────────

class GradientTool(Tool):
    """Draw a linear or radial gradient onto the active layer."""

    def __init__(self, canvas: DrawCanvas) -> None:
        super().__init__(canvas)
        self.color_start = QColor(255, 255, 255)
        self.color_end = QColor(0, 0, 0, 0)
        self.mode = "linear"  # or "radial"
        self._start: QPointF | None = None
        self._preview_item = None

    def on_press(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.canvas.snapshot_active()
        self._start = pos
        self._preview_item = self.canvas.scene().addLine(
            pos.x(), pos.y(), pos.x(), pos.y(),
            QPen(QColor(255, 255, 0, 128), 1, Qt.PenStyle.DashLine),
        )
        self._preview_item.setZValue(9999)

    def on_move(self, pos: QPointF, event) -> None:
        if self._start and self._preview_item:
            self._preview_item.setLine(
                self._start.x(), self._start.y(), pos.x(), pos.y()
            )

    def on_release(self, pos: QPointF, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._start is None:
            return
        if self._preview_item:
            self.canvas.scene().removeItem(self._preview_item)
            self._preview_item = None

        layer = self.canvas.active_layer
        if layer is None:
            self._start = None
            return

        p1 = _scene_to_layer(self._start, layer)
        p2 = _scene_to_layer(pos, layer)

        if self.mode == "radial":
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            radius = math.sqrt(dx * dx + dy * dy)
            if radius < 1:
                self._start = None
                return
            grad = QRadialGradient(p1, radius)
            grad.setColorAt(0, self.color_start)
            grad.setColorAt(1, self.color_end)
        else:
            grad = QLinearGradient(p1, p2)
            grad.setColorAt(0, self.color_start)
            grad.setColorAt(1, self.color_end)

        painter = QPainter(layer.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        # Clip to selection if active
        if self.canvas.selection.path is not None:
            # Convert scene selection to layer-local... for now just fill the whole image
            pass
        painter.drawRect(layer.image.rect())
        painter.end()
        self.canvas.layer_stack.layer_updated.emit(layer.uid)
        self._start = None

    def cursor(self) -> QCursor:
        return QCursor(Qt.CursorShape.CrossCursor)
