"""Interactive resize/rotate handles overlay for layer transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QBrush, QColor, QCursor, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
)

if TYPE_CHECKING:
    from sdqt.widgets.draw.canvas import DrawCanvas
    from sdqt.widgets.draw.layer_model import Layer

_HANDLE_SIZE = 10
_HANDLE_HALF = _HANDLE_SIZE / 2
_ROTATE_SIZE = 14
_ROTATE_HALF = _ROTATE_SIZE / 2
_ROTATE_OFFSET = 24  # distance outside the corner

# Handle indices for the 8 resize handles:
#   0=TL  1=T  2=TR  3=L  4=R  5=BL  6=B  7=BR
_CORNERS = {0, 2, 5, 7}   # proportional resize
_EDGES = {1, 3, 4, 6}     # single-axis resize

# Rotate pin indices: 8=TL_rot  9=TR_rot  10=BL_rot  11=BR_rot
_ROTATE_PINS = {8, 9, 10, 11}


class TransformHandles:
    """Overlay handles around a layer's bounding box.

    8 resize handles (white squares):
        Corners (TL/TR/BL/BR): proportional resize.
        Edges (T/B/L/R): single-axis resize.
    4 rotate pins (green circles) floating outside each corner.
    Interior drag: move.
    Enter: confirm.  Escape: cancel.
    """

    def __init__(self, canvas: DrawCanvas) -> None:
        self.canvas = canvas
        self._active = False
        self._layer_uid: str | None = None
        self._original_state: dict | None = None

        # Graphics items
        self._border: QGraphicsRectItem | None = None
        self._handles: list[QGraphicsItem] = []  # 12 items: 8 resize + 4 rotate

        # Interaction
        self._dragging_handle: int | None = None  # handle index, -1=move
        self._drag_start: QPointF | None = None
        self._orig_offset: tuple[float, float] = (0, 0)
        # Per-drag baseline (captured on press so consecutive drags don't jump)
        self._base_offset: tuple[float, float] = (0, 0)
        self._base_scale: tuple[float, float] = (1.0, 1.0)
        self._base_rotation: float = 0.0
        self._base_center: QPointF = QPointF(0, 0)

    @property
    def active(self) -> bool:
        return self._active

    def activate(self, layer_uid: str) -> None:
        layer = self.canvas.layer_stack.get(layer_uid)
        if layer is None:
            return
        if self._active:
            self._remove_handles()
        self._layer_uid = layer_uid
        self._active = True
        self._save_state(layer)
        self._create_handles()
        self._position_handles()

    def deactivate(self, confirm: bool = True) -> None:
        if not self._active:
            return
        if not confirm and self._original_state and self._layer_uid:
            self._restore_state()
        self._remove_handles()
        self._active = False
        self._layer_uid = None

    # -- Hit testing & interaction --------------------------------------------

    def handle_press(self, pos: QPointF) -> bool:
        """Returns True if press hit a handle or the interior."""
        if not self._active or not self._border:
            return False
        layer = self._get_layer()
        if layer is None:
            return False
        self._drag_start = pos

        # Capture a fresh baseline for THIS drag so successive resizes/rotations
        # compose correctly (otherwise a 2nd drag jumps back to activate-state).
        self._orig_offset = (layer.offset_x, layer.offset_y)
        self._base_offset = (layer.offset_x, layer.offset_y)
        self._base_scale = (layer.scale_x, layer.scale_y)
        self._base_rotation = layer.rotation
        # The layer is scaled/rotated about its centre (see DrawCanvas
        # ._apply_transform); that centre is scale-independent.
        cx = layer.image.width() / 2.0
        cy = layer.image.height() / 2.0
        self._base_center = QPointF(layer.offset_x + cx, layer.offset_y + cy)

        # Check individual handles (resize + rotate) — topmost (last drawn) first
        for i in reversed(range(len(self._handles))):
            h = self._handles[i]
            if h.contains(h.mapFromScene(pos)):
                self._dragging_handle = i
                self.canvas.snapshot_active()
                return True

        # Interior → move (border carries the layer transform, so this hit-test
        # respects rotation)
        if self._border.contains(self._border.mapFromScene(pos)):
            self._dragging_handle = -1
            self.canvas.snapshot_active()
            return True

        return False

    def handle_move(self, pos: QPointF) -> None:
        if not self._active or self._drag_start is None:
            return
        layer = self._get_layer()
        if layer is None:
            return

        orig_w = layer.image.width()
        orig_h = layer.image.height()
        if orig_w == 0 or orig_h == 0:
            return

        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        handle = self._dragging_handle
        bsx, bsy = self._base_scale
        center = self._base_center

        if handle == -1:
            # Move
            layer.offset_x = self._base_offset[0] + dx
            layer.offset_y = self._base_offset[1] + dy

        elif handle in _CORNERS:
            # Proportional resize: uniform scale by how far the cursor moved
            # from the layer centre relative to where the drag began. Works for
            # any corner and at any rotation, and the layer scales about its
            # centre exactly like DrawCanvas renders it.
            d0 = math.hypot(self._drag_start.x() - center.x(),
                            self._drag_start.y() - center.y())
            d1 = math.hypot(pos.x() - center.x(), pos.y() - center.y())
            if d0 < 1.0:
                return
            factor = max(0.05, d1 / d0)
            layer.scale_x = bsx * factor
            layer.scale_y = bsy * factor

        elif handle in _EDGES:
            # Single-axis resize. Project the drag onto the layer's local axes
            # so it stays correct when the layer is rotated.
            r = math.radians(self._base_rotation)
            cos_r, sin_r = math.cos(r), math.sin(r)
            along_x = dx * cos_r + dy * sin_r        # along local +x
            along_y = -dx * sin_r + dy * cos_r       # along local +y
            if handle == 1:    # Top edge
                layer.scale_y = bsy * max(0.05, 1.0 - along_y / (orig_h * bsy))
            elif handle == 6:  # Bottom edge
                layer.scale_y = bsy * max(0.05, 1.0 + along_y / (orig_h * bsy))
            elif handle == 3:  # Left edge
                layer.scale_x = bsx * max(0.05, 1.0 - along_x / (orig_w * bsx))
            elif handle == 4:  # Right edge
                layer.scale_x = bsx * max(0.05, 1.0 + along_x / (orig_w * bsx))

        elif handle in _ROTATE_PINS:
            # Rotate about the true (scale-independent) layer centre.
            a0 = math.atan2(self._drag_start.y() - center.y(),
                            self._drag_start.x() - center.x())
            a1 = math.atan2(pos.y() - center.y(), pos.x() - center.x())
            layer.rotation = self._base_rotation + math.degrees(a1 - a0)
        else:
            return

        self.canvas.layer_stack.layer_updated.emit(layer.uid)
        self._position_handles()

    def handle_release(self) -> None:
        self._dragging_handle = None
        self._drag_start = None

    # -- Private ---------------------------------------------------------------

    def _get_layer(self):
        if self._layer_uid:
            return self.canvas.layer_stack.get(self._layer_uid)
        return None

    def _save_state(self, layer: Layer) -> None:
        self._original_state = {
            "offset_x": layer.offset_x,
            "offset_y": layer.offset_y,
            "rotation": layer.rotation,
            "scale_x": layer.scale_x,
            "scale_y": layer.scale_y,
            "flip_h": layer.flip_h,
            "flip_v": layer.flip_v,
        }

    def _restore_state(self) -> None:
        if not self._original_state or not self._layer_uid:
            return
        layer = self.canvas.layer_stack.get(self._layer_uid)
        if layer is None:
            return
        for k, v in self._original_state.items():
            setattr(layer, k, v)
        self.canvas.layer_stack.layer_updated.emit(layer.uid)

    def _create_handles(self) -> None:
        scene = self.canvas.scene()
        border_pen = QPen(QColor(0, 120, 255), 1, Qt.PenStyle.DashLine)
        border_pen.setCosmetic(True)  # constant 1px even when the border is scaled
        resize_pen = QPen(QColor(0, 120, 255), 1)
        resize_brush = QBrush(QColor(255, 255, 255))
        rotate_pen = QPen(QColor(0, 200, 100), 1.5)
        rotate_brush = QBrush(QColor(0, 200, 100, 180))

        self._border = scene.addRect(QRectF(), border_pen)
        self._border.setZValue(9998)

        self._handles = []

        # 8 resize handles (all white squares): TL(0) T(1) TR(2) L(3) R(4) BL(5) B(6) BR(7)
        for i in range(8):
            h = scene.addRect(
                QRectF(-_HANDLE_HALF, -_HANDLE_HALF, _HANDLE_SIZE, _HANDLE_SIZE),
                resize_pen, resize_brush,
            )
            h.setZValue(9999)
            self._handles.append(h)

        # 4 rotate pins (green circles) outside each corner: TL(8) TR(9) BL(10) BR(11)
        for i in range(4):
            h = scene.addEllipse(
                QRectF(-_ROTATE_HALF, -_ROTATE_HALF, _ROTATE_SIZE, _ROTATE_SIZE),
                rotate_pen, rotate_brush,
            )
            h.setZValue(9999)
            self._handles.append(h)

    def _layer_transform(self, layer) -> QTransform:
        """Scene transform DrawCanvas applies to this layer's item (incl. pos).

        Mirrors DrawCanvas._apply_transform + item.setPos: scale/rotate/flip
        happen about the image centre, then the whole thing is offset. Mapping
        the image-local corner points through this gives the exact on-screen
        quad of the layer, so the handles wrap the PNG under scale AND rotation.
        """
        cx = layer.image.width() / 2.0
        cy = layer.image.height() / 2.0
        t = QTransform()
        t.translate(layer.offset_x, layer.offset_y)   # outermost (item.setPos)
        t.translate(cx, cy)
        if layer.flip_h:
            t.scale(-1, 1)
        if layer.flip_v:
            t.scale(1, -1)
        t.rotate(layer.rotation)
        t.scale(layer.scale_x, layer.scale_y)
        t.translate(-cx, -cy)
        return t

    def _position_handles(self) -> None:
        layer = self._get_layer()
        if layer is None or self._border is None:
            return
        w = layer.image.width()
        h = layer.image.height()
        t = self._layer_transform(layer)

        # Border overlays the layer exactly (carries the same transform, so it
        # scales + rotates with the PNG). Cosmetic pen keeps it 1px wide.
        self._border.setPos(0, 0)
        self._border.setTransform(t)
        self._border.setRect(0, 0, w, h)

        # 8 resize handle anchor points (image-local), mapped to scene:
        #   TL(0) T(1) TR(2) L(3) R(4) BL(5) B(6) BR(7)
        local = [
            QPointF(0, 0), QPointF(w / 2, 0), QPointF(w, 0),
            QPointF(0, h / 2), QPointF(w, h / 2),
            QPointF(0, h), QPointF(w / 2, h), QPointF(w, h),
        ]
        scene_pts = [t.map(p) for p in local]
        for i, p in enumerate(scene_pts):
            self._handles[i].setPos(p)

        # 4 rotate pins float just outside each corner, along the OUTWARD
        # diagonal in the (possibly rotated) frame: TL(8) TR(9) BL(10) BR(11)
        center = t.map(QPointF(w / 2.0, h / 2.0))
        for j, ci in enumerate((0, 2, 5, 7)):  # TL, TR, BL, BR corners
            c = scene_pts[ci]
            vx, vy = c.x() - center.x(), c.y() - center.y()
            mag = math.hypot(vx, vy) or 1.0
            self._handles[8 + j].setPos(
                QPointF(c.x() + vx / mag * _ROTATE_OFFSET,
                        c.y() + vy / mag * _ROTATE_OFFSET)
            )

    def _remove_handles(self) -> None:
        scene = self.canvas.scene()
        if self._border:
            scene.removeItem(self._border)
            self._border = None
        for h in self._handles:
            scene.removeItem(h)
        self._handles.clear()


class TransformTool:
    """Tool that activates/manages TransformHandles on the active layer."""

    def __init__(self, canvas: DrawCanvas) -> None:
        self.canvas = canvas
        self.handles = TransformHandles(canvas)

    def activate(self) -> None:
        layer = self.canvas.active_layer
        if layer:
            self.handles.activate(layer.uid)

    def deactivate(self, confirm: bool = True) -> None:
        self.handles.deactivate(confirm)

    def on_press(self, pos: QPointF, event) -> None:
        if not self.handles.active:
            self.activate()
        self.handles.handle_press(pos)

    def on_move(self, pos: QPointF, event) -> None:
        self.handles.handle_move(pos)

    def on_release(self, pos: QPointF, event) -> None:
        self.handles.handle_release()

    def on_key(self, event) -> None:
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            self.deactivate(confirm=True)
        elif event.key() == Qt.Key.Key_Escape:
            self.deactivate(confirm=False)

    def cursor(self):
        return QCursor(Qt.CursorShape.SizeAllCursor)
