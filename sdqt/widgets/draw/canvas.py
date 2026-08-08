"""DrawCanvas -- QGraphicsView managing layer items and tool delegation."""

from __future__ import annotations

import copy
import logging
import zlib
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from PySide6.QtGui import QTabletEvent

from sdqt.widgets.draw.layer_model import Layer, LayerStack
from sdqt.widgets.draw.selection import MarchingAnts, Selection

BLEND_MODES = {
    "normal": QPainter.CompositionMode.CompositionMode_SourceOver,
    "multiply": QPainter.CompositionMode.CompositionMode_Multiply,
    "screen": QPainter.CompositionMode.CompositionMode_Screen,
    "overlay": QPainter.CompositionMode.CompositionMode_Overlay,
    "darken": QPainter.CompositionMode.CompositionMode_Darken,
    "lighten": QPainter.CompositionMode.CompositionMode_Lighten,
    "soft_light": QPainter.CompositionMode.CompositionMode_SoftLight,
    "hard_light": QPainter.CompositionMode.CompositionMode_HardLight,
    "difference": QPainter.CompositionMode.CompositionMode_Difference,
}

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
_CHECK_SIZE = 16


def _make_checkerboard(tile: int = _CHECK_SIZE) -> QPixmap:
    size = tile * 2
    pm = QPixmap(size, size)
    pm.fill(QColor(204, 204, 204))
    painter = QPainter(pm)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRect(0, 0, tile, tile)
    painter.drawRect(tile, tile, tile, tile)
    painter.end()
    return pm


# ── Undo stack ───────────────────────────────────────────────────────────────

class _UndoSnapshot:
    """Stores a deep copy of one layer's QImage + transform state for undo/redo."""
    __slots__ = ("uid", "image_data", "image_w", "image_h", "image_fmt",
                 "offset_x", "offset_y", "rotation",
                 "scale_x", "scale_y", "flip_h", "flip_v")

    def __init__(self, uid: str, image: QImage, layer: Layer | None = None) -> None:
        self.uid = uid
        # Compress image data
        ptr = image.constBits()
        raw = bytes(ptr) if ptr else b''
        self.image_data = zlib.compress(raw, 1)
        self.image_w = image.width()
        self.image_h = image.height()
        self.image_fmt = image.format()
        if layer is not None:
            self.offset_x = layer.offset_x
            self.offset_y = layer.offset_y
            self.rotation = layer.rotation
            self.scale_x = layer.scale_x
            self.scale_y = layer.scale_y
            self.flip_h = layer.flip_h
            self.flip_v = layer.flip_v
        else:
            self.offset_x = 0.0
            self.offset_y = 0.0
            self.rotation = 0.0
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.flip_h = False
            self.flip_v = False

    def restore_image(self) -> QImage:
        raw = zlib.decompress(self.image_data)
        img = QImage(raw, self.image_w, self.image_h, self.image_fmt).copy()
        return img

    def store_image(self, image: QImage) -> None:
        ptr = image.constBits()
        raw = bytes(ptr) if ptr else b''
        self.image_data = zlib.compress(raw, 1)
        self.image_w = image.width()
        self.image_h = image.height()
        self.image_fmt = image.format()


class UndoStack:
    """Simple linear undo/redo stack for layer image edits."""

    def __init__(self, max_depth: int = 40) -> None:
        self._stack: list[_UndoSnapshot] = []
        self._index: int = -1
        self._max = max_depth

    def push(self, uid: str, image: QImage, layer: Layer | None = None) -> None:
        """Snapshot a layer BEFORE a modification."""
        # Trim any redo history beyond current index
        self._stack = self._stack[: self._index + 1]
        self._stack.append(_UndoSnapshot(uid, image, layer))
        # Adaptive max depth based on image size
        pixels = image.width() * image.height()
        if pixels > 4_000_000:
            effective_max = 20
        elif pixels > 1_000_000:
            effective_max = 30
        else:
            effective_max = self._max
        if len(self._stack) > effective_max:
            self._stack.pop(0)
        self._index = len(self._stack) - 1

    def undo(self, layer_stack: LayerStack) -> bool:
        if self._index < 0:
            return False
        snap = self._stack[self._index]
        layer = layer_stack.get(snap.uid)
        if layer is None:
            self._index -= 1
            return False
        # Swap: save current state for redo, restore snapshot
        current_image = layer.image.copy()
        cur_ox, cur_oy = layer.offset_x, layer.offset_y
        cur_rot = layer.rotation
        cur_sx, cur_sy = layer.scale_x, layer.scale_y
        cur_fh, cur_fv = layer.flip_h, layer.flip_v

        layer.image = snap.restore_image()
        layer.offset_x = snap.offset_x
        layer.offset_y = snap.offset_y
        layer.rotation = snap.rotation
        layer.scale_x = snap.scale_x
        layer.scale_y = snap.scale_y
        layer.flip_h = snap.flip_h
        layer.flip_v = snap.flip_v

        snap.store_image(current_image)
        snap.offset_x = cur_ox
        snap.offset_y = cur_oy
        snap.rotation = cur_rot
        snap.scale_x = cur_sx
        snap.scale_y = cur_sy
        snap.flip_h = cur_fh
        snap.flip_v = cur_fv

        layer_stack.layer_updated.emit(layer.uid)
        self._index -= 1
        return True

    def redo(self, layer_stack: LayerStack) -> bool:
        if self._index + 1 >= len(self._stack):
            return False
        self._index += 1
        snap = self._stack[self._index]
        layer = layer_stack.get(snap.uid)
        if layer is None:
            return False

        current_image = layer.image.copy()
        cur_ox, cur_oy = layer.offset_x, layer.offset_y
        cur_rot = layer.rotation
        cur_sx, cur_sy = layer.scale_x, layer.scale_y
        cur_fh, cur_fv = layer.flip_h, layer.flip_v

        layer.image = snap.restore_image()
        layer.offset_x = snap.offset_x
        layer.offset_y = snap.offset_y
        layer.rotation = snap.rotation
        layer.scale_x = snap.scale_x
        layer.scale_y = snap.scale_y
        layer.flip_h = snap.flip_h
        layer.flip_v = snap.flip_v

        snap.store_image(current_image)
        snap.offset_x = cur_ox
        snap.offset_y = cur_oy
        snap.rotation = cur_rot
        snap.scale_x = cur_sx
        snap.scale_y = cur_sy
        snap.flip_h = cur_fh
        snap.flip_v = cur_fv

        layer_stack.layer_updated.emit(layer.uid)
        return True

    def clear(self) -> None:
        self._stack.clear()
        self._index = -1

    @property
    def can_undo(self) -> bool:
        return self._index >= 0

    @property
    def can_redo(self) -> bool:
        return self._index + 1 < len(self._stack)


# ── Canvas ───────────────────────────────────────────────────────────────────

class DrawCanvas(QGraphicsView):
    """Layer-based drawing canvas."""

    flatten_requested = Signal(QImage)
    zoom_changed = Signal()
    canvas_size: tuple[int, int] = (1024, 1024)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet("QGraphicsView { background: #1a1a1a; border: 1px solid #555; }")

        # Solid dark background for scene (outside canvas area)
        self._scene.setBackgroundBrush(QBrush(QColor(0x1a, 0x1a, 0x1a)))

        # Checkerboard rect item bounded to canvas area
        self._checker_item = QGraphicsRectItem()
        self._checker_item.setBrush(QBrush(_make_checkerboard()))
        self._checker_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._checker_item.setZValue(-1)
        self._checker_item.setRect(QRectF(0, 0, *self.canvas_size))
        self._scene.addItem(self._checker_item)

        self.layer_stack = LayerStack(self)
        self.selection = Selection(self)
        self.undo_stack = UndoStack()
        self._ants = MarchingAnts(self._scene)

        self._items: dict[str, QGraphicsPixmapItem] = {}
        self._tool = None
        self._clipboard: QImage | None = None
        self._panning = False
        self._pan_start = None
        self._fit_on_resize = True
        self._view_flipped = False

        # Brush cursor overlay (circle showing brush size)
        self._cursor_circle: QGraphicsEllipseItem | None = None
        self._cursor_size: float = 8.0
        self._cursor_visible = False
        self._cursor_color = QColor(255, 255, 255, 220)

        self.layer_stack.layers_changed.connect(self._rebuild_items)
        self.layer_stack.layer_updated.connect(self.refresh_layer)
        self.selection.changed.connect(self._update_ants)

    # -- Coordinate helper: widget pos → scene pos ----------------------------

    def _to_scene(self, widget_pos) -> 'QPointF':
        """Convert a widget-relative position to scene coordinates.

        mapToScene expects *viewport* coordinates.  The viewport is inset
        from the widget by the frame, so we must translate first.
        """
        vp_pos = self.viewport().mapFromParent(widget_pos)
        return self.mapToScene(vp_pos)

    # -- Tool delegation ------------------------------------------------------

    def set_tool(self, tool) -> None:
        self._tool = tool
        has_size = tool is not None and hasattr(tool, "size")
        self._cursor_visible = has_size
        if has_size:
            self._cursor_size = tool.size
            self.viewport().setCursor(Qt.CursorShape.BlankCursor)
            self._ensure_cursor_circle()
        else:
            self._remove_cursor_circle()
            if tool is not None:
                self.viewport().setCursor(tool.cursor())
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def update_cursor_size(self, size: float) -> None:
        """Called when the brush/eraser size slider changes."""
        self._cursor_size = size
        rect = QRectF(-size / 2, -size / 2, size, size)
        if self._cursor_circle is not None:
            self._cursor_circle.setRect(rect)
        if hasattr(self, "_cursor_outline") and self._cursor_outline is not None:
            self._cursor_outline.setRect(rect)

    @property
    def active_layer(self) -> Layer | None:
        return self.layer_stack.active_layer

    # -- Brush cursor circle --------------------------------------------------

    def set_cursor_color(self, color: QColor) -> None:
        """Set the brush-size cursor circle color (default translucent white)."""
        self._cursor_color = color
        if self._cursor_circle is not None:
            self._cursor_circle.setPen(QPen(color, 1.0))

    def _ensure_cursor_circle(self) -> None:
        if self._cursor_circle is None:
            r = self._cursor_size / 2
            rect = QRectF(-r, -r, self._cursor_size, self._cursor_size)
            # Dark outline behind for contrast on light backgrounds
            outer_pen = QPen(QColor(0, 0, 0, 200), 1.5)
            self._cursor_outline = self._scene.addEllipse(rect, outer_pen)
            self._cursor_outline.setZValue(10000)
            # Bright inner circle
            inner_pen = QPen(self._cursor_color, 1.0)
            self._cursor_circle = self._scene.addEllipse(rect, inner_pen)
            self._cursor_circle.setZValue(10001)

    def _remove_cursor_circle(self) -> None:
        if self._cursor_circle is not None:
            self._scene.removeItem(self._cursor_circle)
            self._cursor_circle = None
        if hasattr(self, "_cursor_outline") and self._cursor_outline is not None:
            self._scene.removeItem(self._cursor_outline)
            self._cursor_outline = None

    def _move_cursor_circle(self, scene_pos) -> None:
        if self._cursor_circle is not None:
            self._cursor_circle.setPos(scene_pos)
        if hasattr(self, "_cursor_outline") and self._cursor_outline is not None:
            self._cursor_outline.setPos(scene_pos)

    # -- Image loading --------------------------------------------------------

    def load_background(self, path: str) -> None:
        img = QImage(path)
        if img.isNull():
            return
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        self.canvas_size = (img.width(), img.height())
        self._scene.setSceneRect(QRectF(0, 0, img.width(), img.height()))
        self._update_checker_rect()

        if self.layer_stack and len(self.layer_stack) > 0 and self.layer_stack[0].name == "Background":
            self.layer_stack.remove_layer(self.layer_stack[0].uid)

        bg = Layer(name="Background", image=img, locked=True)
        self.layer_stack._layers.insert(0, bg)
        self.layer_stack._active_uid = bg.uid
        self.layer_stack.layers_changed.emit()
        self.layer_stack.active_changed.emit(bg.uid)
        self.undo_stack.clear()
        self._fit_on_resize = True
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def add_image_layer(self, image: QImage, name: str = "Layer") -> Layer:
        if image.format() != QImage.Format.Format_ARGB32:
            image = image.convertToFormat(QImage.Format.Format_ARGB32)
        if not self.layer_stack:
            self.canvas_size = (image.width(), image.height())
            self._scene.setSceneRect(QRectF(0, 0, image.width(), image.height()))
            self._update_checker_rect()
        return self.layer_stack.add_layer(name, image, above=self.layer_stack.active_uid)

    def new_empty_layer(self, name: str = "Layer") -> Layer:
        w, h = self.canvas_size
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        return self.add_image_layer(img, name)

    # -- Checkerboard bounds --------------------------------------------------

    def _update_checker_rect(self) -> None:
        w, h = self.canvas_size
        self._checker_item.setRect(QRectF(0, 0, w, h))

    # -- Undo-aware snapshot helper -------------------------------------------

    def snapshot_active(self) -> None:
        """Push current active layer image + transforms onto undo stack (call BEFORE edit)."""
        layer = self.active_layer
        if layer is not None:
            self.undo_stack.push(layer.uid, layer.image, layer)

    def undo(self) -> None:
        self.undo_stack.undo(self.layer_stack)

    def redo(self) -> None:
        self.undo_stack.redo(self.layer_stack)

    # -- Flatten / render -----------------------------------------------------

    def flatten(self) -> QImage:
        """Composite all visible layers with transforms into canvas-sized image."""
        if len(self.layer_stack) == 0:
            return QImage()
        w, h = self.canvas_size
        result = QImage(w, h, QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for layer in self.layer_stack:
            if not layer.visible:
                continue
            painter.save()
            mode = BLEND_MODES.get(layer.blend_mode, QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setCompositionMode(mode)
            painter.setOpacity(layer.opacity)
            # Apply the same transform the scene items use
            painter.translate(layer.offset_x, layer.offset_y)
            cx = layer.image.width() / 2
            cy = layer.image.height() / 2
            painter.translate(cx, cy)
            if layer.flip_h:
                painter.scale(-1, 1)
            if layer.flip_v:
                painter.scale(1, -1)
            painter.rotate(layer.rotation)
            painter.scale(layer.scale_x, layer.scale_y)
            painter.translate(-cx, -cy)
            painter.drawImage(0, 0, layer.image)
            painter.restore()
        painter.end()
        return result

    def flatten_excluding(self, exclude_uids: set[str]) -> QImage:
        """Composite all visible layers except those whose uid is in exclude_uids."""
        if len(self.layer_stack) == 0:
            return QImage()
        w, h = self.canvas_size
        result = QImage(w, h, QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        for layer in self.layer_stack:
            if not layer.visible or layer.uid in exclude_uids:
                continue
            painter.save()
            mode = BLEND_MODES.get(layer.blend_mode, QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setCompositionMode(mode)
            painter.setOpacity(layer.opacity)
            painter.translate(layer.offset_x, layer.offset_y)
            cx = layer.image.width() / 2
            cy = layer.image.height() / 2
            painter.translate(cx, cy)
            if layer.flip_h:
                painter.scale(-1, 1)
            if layer.flip_v:
                painter.scale(1, -1)
            painter.rotate(layer.rotation)
            painter.scale(layer.scale_x, layer.scale_y)
            painter.translate(-cx, -cy)
            painter.drawImage(0, 0, layer.image)
            painter.restore()
        painter.end()
        return result

    def render_layer(self, uid: str) -> QImage:
        """Render a specific layer by uid with transforms baked into canvas-sized image."""
        layer = self.layer_stack.get(uid)
        if layer is None:
            return QImage()
        w, h = self.canvas_size
        result = QImage(w, h, QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        mode = BLEND_MODES.get(layer.blend_mode, QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setCompositionMode(mode)
        painter.setOpacity(layer.opacity)
        painter.translate(layer.offset_x, layer.offset_y)
        cx = layer.image.width() / 2
        cy = layer.image.height() / 2
        painter.translate(cx, cy)
        if layer.flip_h:
            painter.scale(-1, 1)
        if layer.flip_v:
            painter.scale(1, -1)
        painter.rotate(layer.rotation)
        painter.scale(layer.scale_x, layer.scale_y)
        painter.translate(-cx, -cy)
        painter.drawImage(0, 0, layer.image)
        painter.end()
        return result

    def render_active_layer(self) -> QImage:
        """Render the active layer with all transforms baked into canvas-sized image."""
        layer = self.active_layer
        if layer is None:
            return QImage()
        w, h = self.canvas_size
        result = QImage(w, h, QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        mode = BLEND_MODES.get(layer.blend_mode, QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setCompositionMode(mode)
        painter.setOpacity(layer.opacity)
        painter.translate(layer.offset_x, layer.offset_y)
        cx = layer.image.width() / 2
        cy = layer.image.height() / 2
        painter.translate(cx, cy)
        if layer.flip_h:
            painter.scale(-1, 1)
        if layer.flip_v:
            painter.scale(1, -1)
        painter.rotate(layer.rotation)
        painter.scale(layer.scale_x, layer.scale_y)
        painter.translate(-cx, -cy)
        painter.drawImage(0, 0, layer.image)
        painter.end()
        return result

    # -- Layer <-> Item sync --------------------------------------------------

    def _has_blend_modes(self) -> bool:
        """Check if any visible layer uses a non-normal blend mode."""
        return any(l.blend_mode != "normal" for l in self.layer_stack if l.visible)

    def _rebuild_items(self) -> None:
        # NOTE: Blend modes are applied correctly in flatten/export output.
        # The QGraphicsView preview uses per-item opacity but cannot composite
        # blend modes between items. Use flatten() for accurate blended output.
        for item in self._items.values():
            self._scene.removeItem(item)
        self._items.clear()
        for i, layer in enumerate(self.layer_stack):
            item = QGraphicsPixmapItem(QPixmap.fromImage(layer.image))
            item.setZValue(i)
            item.setVisible(layer.visible)
            item.setOpacity(layer.opacity)
            item.setPos(layer.offset_x, layer.offset_y)
            self._apply_transform(item, layer)
            self._scene.addItem(item)
            self._items[layer.uid] = item

    def refresh_layer(self, uid: str) -> None:
        layer = self.layer_stack.get(uid)
        item = self._items.get(uid)
        if layer is None or item is None:
            return
        item.setPixmap(QPixmap.fromImage(layer.image))
        item.setVisible(layer.visible)
        item.setOpacity(layer.opacity)
        item.setPos(layer.offset_x, layer.offset_y)
        self._apply_transform(item, layer)

    @staticmethod
    def _apply_transform(item: QGraphicsPixmapItem, layer: Layer) -> None:
        t = QTransform()
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
        item.setTransform(t)

    def _update_ants(self) -> None:
        self._ants.update_path(self.selection.path)

    # -- Clipboard operations -------------------------------------------------

    def copy_selection(self) -> None:
        layer = self.active_layer
        if layer is None:
            return
        result = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        if self.selection.path is not None:
            painter.setClipPath(self.selection.path)
        painter.drawImage(0, 0, layer.image)
        painter.end()
        self._clipboard = result

    def cut_selection(self) -> None:
        self.copy_selection()
        self.delete_selection()

    def delete_selection(self) -> None:
        layer = self.active_layer
        if layer is None or layer.locked:
            return
        self.snapshot_active()
        painter = QPainter(layer.image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        if self.selection.path is not None:
            painter.setClipPath(self.selection.path)
        painter.fillRect(layer.image.rect(), QColor(0, 0, 0, 0))
        painter.end()
        self.layer_stack.layer_updated.emit(layer.uid)

    def paste_clipboard(self) -> Layer | None:
        if self._clipboard is None:
            return None
        return self.add_image_layer(self._clipboard.copy(), "Pasted")

    # -- Mouse events → tool delegation (FIXED: widget→viewport→scene) --------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if self._tool is not None:
            scene_pos = self._to_scene(event.position().toPoint())
            # Snapshot for undo before brush/eraser/grab strokes begin
            if hasattr(self._tool, '_painting') or hasattr(self._tool, '_erasing') or hasattr(self._tool, '_dragging'):
                self.snapshot_active()
            self._tool.on_press(scene_pos, event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self._to_scene(event.position().toPoint())
        # Move brush cursor
        if self._cursor_visible:
            self._move_cursor_circle(scene_pos)
        if self._panning and self._pan_start is not None:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._tool is not None:
            self._tool.on_move(scene_pos, event)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            if self._cursor_visible:
                self.viewport().setCursor(Qt.CursorShape.BlankCursor)
            elif self._tool:
                self.viewport().setCursor(self._tool.cursor())
            else:
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._tool is not None:
            scene_pos = self._to_scene(event.position().toPoint())
            self._tool.on_release(scene_pos, event)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self._fit_on_resize = False
        self.zoom_changed.emit()
        event.accept()

    # -- Zoom API -------------------------------------------------------------

    @property
    def zoom_level(self) -> float:
        """Current zoom as a percentage (100.0 = 1:1)."""
        return self.transform().m11() * 100

    def set_zoom(self, percent: float) -> None:
        """Set zoom to exact percentage (e.g. 100.0 = 1:1)."""
        current = self.transform().m11()
        if current <= 0:
            return
        factor = (percent / 100.0) / current
        self.scale(factor, factor)
        self._fit_on_resize = False
        self.zoom_changed.emit()

    def zoom_to_100(self) -> None:
        """Reset zoom to 1:1 pixel mapping."""
        self.resetTransform()
        self._fit_on_resize = False
        self.zoom_changed.emit()

    def keyPressEvent(self, event) -> None:
        # Handle undo/redo directly so they work even when the canvas has focus
        mods = event.modifiers()
        key = event.key()
        if mods == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_Z:
                self.undo()
                event.accept()
                return
            elif key == Qt.Key.Key_Y:
                self.redo()
                event.accept()
                return
        elif mods == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            if key == Qt.Key.Key_Z:
                self.redo()
                event.accept()
                return
        if self._tool is not None:
            self._tool.on_key(event)
        super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        if self._cursor_circle:
            self._cursor_circle.setVisible(False)
        if hasattr(self, "_cursor_outline") and self._cursor_outline:
            self._cursor_outline.setVisible(False)
        super().leaveEvent(event)

    def enterEvent(self, event) -> None:
        if self._cursor_visible:
            if self._cursor_circle:
                self._cursor_circle.setVisible(True)
            if hasattr(self, "_cursor_outline") and self._cursor_outline:
                self._cursor_outline.setVisible(True)
        super().enterEvent(event)

    # -- Drag & drop ----------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _IMAGE_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _IMAGE_EXTS:
                img = QImage(path)
                if not img.isNull():
                    self.add_image_layer(img, Path(path).stem)
                event.acceptProposedAction()
                return

    # -- Fit on resize --------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_on_resize and self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # -- Pixel grid at high zoom -----------------------------------------------

    def drawForeground(self, painter, rect):
        zoom = self.transform().m11()
        if zoom < 4.0:
            return
        # Draw 1px grid
        painter.setPen(QPen(QColor(255, 255, 255, 30), 0))
        left = int(rect.left())
        right = int(rect.right()) + 1
        top = int(rect.top())
        bottom = int(rect.bottom()) + 1
        for x in range(left, right):
            painter.drawLine(x, top, x, bottom)
        for y in range(top, bottom):
            painter.drawLine(left, y, right, y)

    # -- Flip view (non-destructive) -------------------------------------------

    def toggle_flip_view(self) -> None:
        """Mirror the canvas view horizontally (non-destructive)."""
        self._view_flipped = not self._view_flipped
        if self._view_flipped:
            self.scale(-1, 1)
        else:
            # Undo the flip
            self.scale(-1, 1)
        self.zoom_changed.emit()

    # -- Tablet pressure support -----------------------------------------------

    def tabletEvent(self, event: QTabletEvent) -> None:
        pressure = event.pressure()
        if self._tool and hasattr(self._tool, 'set_pressure'):
            self._tool.set_pressure(pressure)
        # Let the regular mouse events handle the rest
        event.ignore()

    def fit_view(self) -> None:
        self._fit_on_resize = True
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit()
