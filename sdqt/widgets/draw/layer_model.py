"""Layer dataclass and LayerStack -- ordered layer model for the Draw editor."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QImage, QPainter


@dataclass
class Layer:
    """Single compositing layer."""

    uid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "Layer"
    image: QImage = field(default_factory=lambda: QImage())
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    flip_h: bool = False
    flip_v: bool = False
    blend_mode: str = "normal"


class LayerStack(QObject):
    """Ordered list of layers (index 0 = bottom, last = top).

    Emits signals on structural changes so canvas + panel stay in sync.
    """

    layers_changed = Signal()       # reorder, add, remove
    layer_updated = Signal(str)     # uid -- content or property changed
    active_changed = Signal(str)    # uid of newly active layer

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layers: list[Layer] = []
        self._uid_map: dict[str, Layer] = {}  # O(1) lookup by uid
        self._active_uid: str | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def active_uid(self) -> str | None:
        return self._active_uid

    @active_uid.setter
    def active_uid(self, uid: str | None) -> None:
        if uid != self._active_uid:
            self._active_uid = uid
            if uid is not None:
                self.active_changed.emit(uid)

    @property
    def active_layer(self) -> Layer | None:
        if self._active_uid is None:
            return None
        return self.get(self._active_uid)

    # -- CRUD -----------------------------------------------------------------

    def add_layer(
        self,
        name: str,
        image: QImage,
        above: str | None = None,
    ) -> Layer:
        """Add a new layer. If *above* is given, insert above that layer."""
        layer = Layer(name=name, image=image)
        if above is not None:
            idx = self._index_of(above)
            if idx is not None:
                self._layers.insert(idx + 1, layer)
            else:
                self._layers.append(layer)
        else:
            self._layers.append(layer)
        self._uid_map[layer.uid] = layer
        self._active_uid = layer.uid
        self.layers_changed.emit()
        self.active_changed.emit(layer.uid)
        return layer

    def remove_layer(self, uid: str) -> None:
        idx = self._index_of(uid)
        if idx is None:
            return
        self._uid_map.pop(uid, None)
        self._layers.pop(idx)
        if self._active_uid == uid:
            if self._layers:
                self._active_uid = self._layers[min(idx, len(self._layers) - 1)].uid
                self.active_changed.emit(self._active_uid)
            else:
                self._active_uid = None
        self.layers_changed.emit()

    def move_layer(self, uid: str, new_index: int) -> None:
        idx = self._index_of(uid)
        if idx is None:
            return
        layer = self._layers.pop(idx)
        new_index = max(0, min(new_index, len(self._layers)))
        self._layers.insert(new_index, layer)
        self.layers_changed.emit()

    def duplicate_layer(self, uid: str) -> Layer | None:
        src = self.get(uid)
        if src is None:
            return None
        dup = Layer(
            name=f"{src.name} copy",
            image=src.image.copy(),
            visible=src.visible,
            opacity=src.opacity,
            offset_x=src.offset_x,
            offset_y=src.offset_y,
            rotation=src.rotation,
            scale_x=src.scale_x,
            scale_y=src.scale_y,
            flip_h=src.flip_h,
            flip_v=src.flip_v,
            blend_mode=src.blend_mode,
        )
        idx = self._index_of(uid)
        self._uid_map[dup.uid] = dup
        self._layers.insert(idx + 1, dup)
        self._active_uid = dup.uid
        self.layers_changed.emit()
        self.active_changed.emit(dup.uid)
        return dup

    def merge_down(self, uid: str) -> None:
        """Merge layer *uid* into the layer below it."""
        idx = self._index_of(uid)
        if idx is None or idx == 0:
            return
        upper = self._layers[idx]
        lower = self._layers[idx - 1]
        if lower.locked:
            return
        # Paint upper onto lower
        painter = QPainter(lower.image)
        painter.setOpacity(upper.opacity)
        painter.drawImage(
            int(upper.offset_x - lower.offset_x),
            int(upper.offset_y - lower.offset_y),
            upper.image,
        )
        painter.end()
        self._uid_map.pop(upper.uid, None)
        self._layers.pop(idx)
        self._active_uid = lower.uid
        self.layers_changed.emit()
        self.active_changed.emit(lower.uid)

    def flatten(self) -> QImage:
        """Composite all visible layers into a single QImage."""
        if not self._layers:
            return QImage()
        # Find bounding box in a single pass
        w = h = 1
        for l in self._layers:
            w = max(w, l.image.width())
            h = max(h, l.image.height())
        result = QImage(w, h, QImage.Format.Format_ARGB32)
        result.fill(QColor(0, 0, 0, 0))
        painter = QPainter(result)
        for layer in self._layers:
            if not layer.visible:
                continue
            painter.setOpacity(layer.opacity)
            painter.drawImage(int(layer.offset_x), int(layer.offset_y), layer.image)
        painter.end()
        return result

    def clear(self) -> None:
        self._layers.clear()
        self._uid_map.clear()
        self._active_uid = None
        self.layers_changed.emit()

    # -- Access ---------------------------------------------------------------

    def get(self, uid: str) -> Layer | None:
        return self._uid_map.get(uid)

    def index_of(self, uid: str) -> int | None:
        return self._index_of(uid)

    def __iter__(self):
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, idx: int) -> Layer:
        return self._layers[idx]

    # -- Internal -------------------------------------------------------------

    def _index_of(self, uid: str) -> int | None:
        for i, l in enumerate(self._layers):
            if l.uid == uid:
                return i
        return None
