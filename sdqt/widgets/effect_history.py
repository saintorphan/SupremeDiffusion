"""Effect history dialog — view, toggle, edit, and remove applied video effects."""

from __future__ import annotations

import copy
import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.effects import EFFECT_REGISTRY, effect_label, make_effect

logger = logging.getLogger(__name__)


class EffectHistoryDialog(QDialog):
    """Shows applied effects with toggle, edit, and remove controls.

    Signals:
        effects_changed: emitted when the stack changes (caller should re-apply).
    """

    effects_changed = Signal(list)  # emits the updated effect stack

    def __init__(self, effect_stack: list[dict], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Effect History")
        self.setMinimumWidth(420)
        self.setMinimumHeight(250)

        self._stack = copy.deepcopy(effect_stack)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list, 1)

        # Clear All button
        clear_btn = QPushButton("Clear All Effects")
        clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(clear_btn)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        self._list.clear()
        for i, fx in enumerate(self._stack):
            self._add_row(fx, i)

    def _add_row(self, fx: dict, index: int) -> None:
        etype = fx.get("type", "")
        enabled = fx.get("enabled", True)

        item = QListWidgetItem()
        item.setData(Qt.UserRole, index)
        self._list.addItem(item)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)

        # Enable checkbox
        cb = QCheckBox()
        cb.setChecked(enabled)
        cb.stateChanged.connect(lambda state, idx=index: self._toggle(idx, state))
        row_layout.addWidget(cb)

        # Label
        name_lbl = QLabel(effect_label(etype))
        row_layout.addWidget(name_lbl, 1)

        # Edit button
        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(40, 24)
        edit_btn.clicked.connect(lambda checked=False, idx=index: self._edit(idx))
        row_layout.addWidget(edit_btn)

        # Remove button
        del_btn = QPushButton("\u2717")
        del_btn.setFixedSize(24, 24)
        del_btn.setFlat(True)
        del_btn.clicked.connect(lambda checked=False, idx=index: self._remove(idx))
        row_layout.addWidget(del_btn)

        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _toggle(self, index: int, state: int) -> None:
        if 0 <= index < len(self._stack):
            self._stack[index]["enabled"] = bool(state)
            self.effects_changed.emit(self._stack)

    def _remove(self, index: int) -> None:
        if 0 <= index < len(self._stack):
            self._stack.pop(index)
            self._populate()
            self.effects_changed.emit(self._stack)

    def _edit(self, index: int) -> None:
        if index < 0 or index >= len(self._stack):
            return
        fx = self._stack[index]
        etype = fx.get("type", "")

        if etype == "crop_zoom":
            return  # Crop/zoom needs a visual editor — skip for now

        if etype not in EFFECT_REGISTRY:
            return

        from sdqt.widgets.effects_manager import EffectEditorDialog

        original_params = copy.deepcopy(fx.get("params", {}))
        dlg = EffectEditorDialog(fx, parent=self)
        if dlg.exec() == QDialog.Accepted:
            fx["params"] = dlg.values()
            self._populate()
            self.effects_changed.emit(self._stack)
        else:
            fx["params"] = original_params

    def _on_rows_moved(self, *args) -> None:
        new_order: list[dict] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            old_idx = item.data(Qt.UserRole)
            if old_idx is not None and 0 <= old_idx < len(self._stack):
                new_order.append(self._stack[old_idx])
        if len(new_order) == len(self._stack):
            self._stack[:] = new_order
        self._populate()
        self.effects_changed.emit(self._stack)

    def _clear_all(self) -> None:
        self._stack.clear()
        self._populate()
        self.effects_changed.emit(self._stack)

    def effect_stack(self) -> list[dict]:
        """Return the current (possibly modified) effect stack."""
        return copy.deepcopy(self._stack)
