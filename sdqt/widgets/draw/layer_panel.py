"""LayerPanel -- tree widget + transform controls for the Draw editor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.draw.layer_model import LayerStack

if TYPE_CHECKING:
    from sdqt.widgets.draw.layer_model import Layer

logger = logging.getLogger(__name__)

_ICON_BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 2px 6px; font-size: 12px;"
    " color: #ccc; background: #333; border-radius: 3px; min-width: 24px; }"
    "QPushButton:hover { color: #fff; background: #555; }"
)

_THUMB_SIZE = 32


class LayerPanel(QWidget):
    """Right-side panel showing the layer stack + transform controls."""

    layer_selected = Signal(str)   # uid
    transform_changed = Signal()   # any transform value changed
    save_to_library_requested = Signal(str)  # uid -- right-click "Save to PNG Library"
    save_to_face_library_requested = Signal(str)  # uid -- right-click "Save to Face Library"

    def __init__(self, layer_stack: LayerStack, parent=None) -> None:
        super().__init__(parent)
        self._stack = layer_stack
        self._updating = False  # guard against feedback loops
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # -- Header row -------------------------------------------------------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addWidget(QLabel("<b>Layers</b>"))
        header.addStretch()

        self._btn_new = QPushButton("+")
        self._btn_new.setToolTip("New layer")
        self._btn_new.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_new.setFixedSize(28, 24)
        header.addWidget(self._btn_new)

        self._btn_dup = QPushButton("\u2398")
        self._btn_dup.setToolTip("Duplicate layer")
        self._btn_dup.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_dup.setFixedSize(28, 24)
        header.addWidget(self._btn_dup)

        self._btn_merge = QPushButton("\u2193")
        self._btn_merge.setToolTip("Merge down")
        self._btn_merge.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_merge.setFixedSize(28, 24)
        header.addWidget(self._btn_merge)

        self._btn_del = QPushButton("\u2715")
        self._btn_del.setToolTip("Delete layer")
        self._btn_del.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_del.setFixedSize(28, 24)
        header.addWidget(self._btn_del)

        layout.addLayout(header)
        layout.addSpacing(6)

        # -- Opacity row -------------------------------------------------------
        opacity_row = QHBoxLayout()
        opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.setSpacing(4)
        opacity_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        opacity_row.addWidget(self._opacity_slider, 1)
        self._opacity_label = QLabel("100%")
        self._opacity_label.setFixedWidth(36)
        opacity_row.addWidget(self._opacity_label)
        layout.addLayout(opacity_row)

        layout.addSpacing(4)

        # -- Blend mode row ----------------------------------------------------
        blend_row = QHBoxLayout()
        blend_row.setContentsMargins(0, 0, 0, 0)
        blend_row.setSpacing(4)
        blend_row.addWidget(QLabel("Blend:"))
        self._blend_combo = QComboBox()
        self._blend_combo.addItems([
            "normal", "multiply", "screen", "overlay",
            "darken", "lighten", "soft_light", "hard_light", "difference",
        ])
        self._blend_combo.setFixedWidth(100)
        blend_row.addWidget(self._blend_combo)
        blend_row.addStretch()
        layout.addLayout(blend_row)

        layout.addSpacing(4)

        # -- Layer tree --------------------------------------------------------
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setColumnCount(1)
        self._tree.setRootIsDecorated(False)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        layout.addWidget(self._tree, 1)

    def _connect_signals(self) -> None:
        self._stack.layers_changed.connect(self._rebuild_tree)
        self._stack.layer_updated.connect(self._on_layer_updated)
        self._stack.active_changed.connect(self._on_active_changed)
        self._tree.currentItemChanged.connect(self._on_tree_selection)
        self._tree.model().rowsMoved.connect(self._on_rows_moved)

        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        self._btn_new.clicked.connect(self._on_new)
        self._btn_dup.clicked.connect(self._on_dup)
        self._btn_merge.clicked.connect(self._on_merge)
        self._btn_del.clicked.connect(self._on_del)

        self._opacity_slider.valueChanged.connect(self._on_opacity)
        self._blend_combo.currentTextChanged.connect(self._on_blend_mode)

    # -- Context menu ----------------------------------------------------------

    def _on_tree_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        uid = item.data(0, Qt.ItemDataRole.UserRole)
        if not uid:
            return

        menu = QMenu(self)
        act_save = menu.addAction("Save to PNG Library...")
        act_save.triggered.connect(lambda: self.save_to_library_requested.emit(uid))
        act_face = menu.addAction("Save to Face Library...")
        act_face.triggered.connect(lambda: self.save_to_face_library_requested.emit(uid))
        menu.addSeparator()
        act_vis = menu.addAction("Toggle Visibility")
        act_vis.triggered.connect(lambda: self.toggle_visibility(uid))
        act_dup = menu.addAction("Duplicate")
        act_dup.triggered.connect(lambda: self._stack.duplicate_layer(uid))
        act_del = menu.addAction("Delete")
        act_del.triggered.connect(lambda: self._stack.remove_layer(uid))
        menu.exec(self._tree.viewport().mapToGlobal(pos))

    # -- Tree rebuild ----------------------------------------------------------

    def _rebuild_tree(self) -> None:
        self._updating = True
        self._tree.clear()
        # Show top layer first (reversed order)
        for layer in reversed(list(self._stack)):
            item = QTreeWidgetItem()
            item.setData(0, Qt.ItemDataRole.UserRole, layer.uid)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled
            )
            self._set_item_display(item, layer)
            self._tree.addTopLevelItem(item)
            if layer.uid == self._stack.active_uid:
                self._tree.setCurrentItem(item)
        self._updating = False
        self._sync_transform_controls()

    def _set_item_display(self, item: QTreeWidgetItem, layer: Layer) -> None:
        vis = "\U0001f441" if layer.visible else "\u25cb"
        lock = " \U0001f512" if layer.locked else ""
        item.setText(0, f"{vis} {layer.name}{lock}")
        # Thumbnail
        thumb = QPixmap.fromImage(layer.image).scaled(
            _THUMB_SIZE, _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        item.setIcon(0, QIcon(thumb))

    def _on_layer_updated(self, uid: str) -> None:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == uid:
                layer = self._stack.get(uid)
                if layer:
                    self._set_item_display(item, layer)
                break
        if uid == self._stack.active_uid:
            self._sync_transform_controls()

    def _on_active_changed(self, uid: str) -> None:
        self._updating = True
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item and item.data(0, Qt.ItemDataRole.UserRole) == uid:
                self._tree.setCurrentItem(item)
                break
        self._updating = False
        self._sync_transform_controls()

    # -- Tree interaction ------------------------------------------------------

    @Slot()
    def _on_tree_selection(self, current, previous) -> None:
        if self._updating or current is None:
            return
        uid = current.data(0, Qt.ItemDataRole.UserRole)
        if uid:
            self._stack.active_uid = uid
            self.layer_selected.emit(uid)

        # Toggle visibility on double-click of eye area
        # (handled via itemDoubleClicked would be better, but simple click-toggle here)

    def _on_rows_moved(self) -> None:
        """Sync layer stack order after drag-reorder in tree."""
        if self._updating:
            return
        new_order = []
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item:
                uid = item.data(0, Qt.ItemDataRole.UserRole)
                if uid:
                    new_order.append(uid)
        # Tree shows top-first, stack is bottom-first
        new_order.reverse()
        # Reorder stack to match
        self._updating = True
        for new_idx, uid in enumerate(new_order):
            self._stack.move_layer(uid, new_idx)
        self._updating = False
        self._stack.layers_changed.emit()

    # -- Button handlers -------------------------------------------------------

    @Slot()
    def _on_new(self) -> None:
        # Signal parent to add a new empty layer
        self.new_layer_requested()

    def new_layer_requested(self) -> None:
        """External hook: the Draw tab connects this to canvas.new_empty_layer."""
        pass  # overridden by connection in img_edit.py

    @Slot()
    def _on_dup(self) -> None:
        uid = self._stack.active_uid
        if uid:
            self._stack.duplicate_layer(uid)

    @Slot()
    def _on_merge(self) -> None:
        uid = self._stack.active_uid
        if uid:
            self._stack.merge_down(uid)

    @Slot()
    def _on_del(self) -> None:
        uid = self._stack.active_uid
        if uid:
            self._stack.remove_layer(uid)

    # -- Opacity ---------------------------------------------------------------

    @Slot(int)
    def _on_opacity(self, val: int) -> None:
        self._opacity_label.setText(f"{val}%")
        layer = self._stack.active_layer
        if layer and not self._updating:
            layer.opacity = val / 100.0
            self._stack.layer_updated.emit(layer.uid)

    # -- Blend mode ------------------------------------------------------------

    @Slot(str)
    def _on_blend_mode(self, text: str) -> None:
        layer = self._stack.active_layer
        if layer and not self._updating:
            layer.blend_mode = text
            self._stack.layer_updated.emit(layer.uid)

    # -- Sync controls ---------------------------------------------------------

    def _sync_transform_controls(self) -> None:
        """Sync opacity slider and blend mode combo to active layer."""
        layer = self._stack.active_layer
        self._updating = True
        if layer:
            self._opacity_slider.setValue(int(layer.opacity * 100))
            idx = self._blend_combo.findText(layer.blend_mode)
            if idx >= 0:
                self._blend_combo.setCurrentIndex(idx)
        else:
            self._opacity_slider.setValue(100)
            self._blend_combo.setCurrentIndex(0)
        self._updating = False

    # -- Visibility toggle (double-click tree item) ----------------------------

    def toggle_visibility(self, uid: str) -> None:
        layer = self._stack.get(uid)
        if layer:
            layer.visible = not layer.visible
            self._stack.layer_updated.emit(uid)
            self._rebuild_tree()
