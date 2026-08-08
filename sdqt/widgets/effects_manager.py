"""Floating effects manager panel and per-effect editor dialog."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.effects import (
    EFFECT_REGISTRY,
    EFFECT_TYPES,
    effect_label,
    make_effect,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EffectEditorDialog — modal, per-effect parameter editor
# ---------------------------------------------------------------------------

class EffectEditorDialog(QDialog):
    """Edit parameters for a single effect instance."""

    preview_changed = Signal()  # emitted on slider moves for live GL preview

    # Color Balance groups its 9 params into 3 sections
    _CB_GROUPS = {
        "Shadows": ["shadow_r", "shadow_g", "shadow_b"],
        "Midtones": ["midtone_r", "midtone_g", "midtone_b"],
        "Highlights": ["highlight_r", "highlight_g", "highlight_b"],
    }

    def __init__(self, fx: dict, parent=None) -> None:
        super().__init__(parent)
        self._fx = fx
        self._etype = fx["type"]
        self._widgets: dict[str, tuple[QSlider, QDoubleSpinBox]] = {}

        entry = EFFECT_REGISTRY.get(self._etype)
        if not entry:
            return
        label, param_defs = entry

        self.setWindowTitle(f"Edit: {label}")
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)

        # Add description for AI Enhance
        if self._etype == "ai_enhance":
            hint = QLabel(
                "Upscale video 4× using RealESRGAN.\n"
                "Lower tile size uses less VRAM. Set 0 for no tiling (fastest, most VRAM)."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
            layout.addWidget(hint)
        params = fx.get("params", {})

        if self._etype == "color_balance":
            self._build_color_balance(layout, param_defs, params)
        else:
            for pname, meta in param_defs.items():
                self._add_param_row(layout, pname, meta, params.get(pname, meta[2]))

        # Buttons
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)
        layout.addLayout(btn_layout)

    def _build_color_balance(self, layout, param_defs, params):
        for group_name, pnames in self._CB_GROUPS.items():
            group = QGroupBox(group_name)
            glayout = QVBoxLayout(group)
            for pname in pnames:
                meta = param_defs[pname]
                # Friendly label: shadow_r -> R
                short = pname.rsplit("_", 1)[-1].upper()
                self._add_param_row(glayout, pname, meta,
                                    params.get(pname, meta[2]), short)
            layout.addWidget(group)

    def _add_param_row(self, layout, pname: str, meta: tuple, value: float,
                       display_name: str = ""):
        pmin, pmax, default, decimals, step, suffix = meta
        if not display_name:
            display_name = pname.replace("_", " ").title()

        row = QHBoxLayout()
        lbl = QLabel(f"{display_name}:")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        # Slider (integer range mapped from float)
        slider_steps = max(1, int((pmax - pmin) / step))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, slider_steps)
        slider.setValue(int((value - pmin) / (pmax - pmin) * slider_steps))
        row.addWidget(slider, 1)

        # SpinBox
        spin = QDoubleSpinBox()
        spin.setRange(pmin, pmax)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(value)
        spin.setFixedWidth(85)
        row.addWidget(spin)

        # Link slider <-> spinbox
        def _slider_to_spin(pos, s=spin, sl=slider, mn=pmin, mx=pmax, st=slider_steps):
            s.blockSignals(True)
            s.setValue(mn + (pos / st) * (mx - mn))
            s.blockSignals(False)

        def _spin_to_slider(val, sl=slider, mn=pmin, mx=pmax, st=slider_steps):
            sl.blockSignals(True)
            sl.setValue(int((val - mn) / (mx - mn) * st))
            sl.blockSignals(False)

        slider.valueChanged.connect(_slider_to_spin)
        spin.valueChanged.connect(_spin_to_slider)

        # Live preview: update fx params on every value change
        def _on_value_changed(val, name=pname, s=spin):
            params = self._fx.get("params", {})
            params[name] = round(s.value(), s.decimals())
            self._fx["params"] = params
            self.preview_changed.emit()
        spin.valueChanged.connect(_on_value_changed)

        self._widgets[pname] = (slider, spin)
        layout.addLayout(row)

    def _reset(self):
        entry = EFFECT_REGISTRY.get(self._etype)
        if not entry:
            return
        _, param_defs = entry
        for pname, meta in param_defs.items():
            pmin, pmax, default, decimals, step, suffix = meta
            if pname in self._widgets:
                _, spin = self._widgets[pname]
                spin.setValue(default)

    def values(self) -> dict:
        """Return updated params dict."""
        result = {}
        for pname, (slider, spin) in self._widgets.items():
            result[pname] = round(spin.value(), spin.decimals())
        return result


# ---------------------------------------------------------------------------
# EffectsManagerPanel — floating, non-blocking tool window
# ---------------------------------------------------------------------------

class EffectsManagerPanel(QWidget):
    """Floating panel to manage the effect stack for a single clip."""

    effects_changed = Signal()
    preview_changed = Signal()  # emitted during live slider drags for GL preview

    def __init__(self, clip, parent=None) -> None:
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self._clip = clip
        self.setWindowTitle(f"Effects: {clip.name}")
        self.setMinimumWidth(360)
        self.setMinimumHeight(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Effect list with drag reorder
        self._list = QListWidget()
        self._list.setDragDropMode(QListWidget.InternalMove)
        self._list.setDefaultDropAction(Qt.MoveAction)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._list, 1)

        # Add Effect button
        add_btn = QPushButton("+ Add Effect")
        add_btn.clicked.connect(self._show_add_menu)
        layout.addWidget(add_btn)

        self._populate()

    def _populate(self):
        """Rebuild list from clip.effects."""
        self._list.clear()
        for i, fx in enumerate(self._clip.effects):
            self._add_row(fx, i)

    def _add_row(self, fx: dict, index: int):
        etype = fx.get("type", "")
        enabled = fx.get("enabled", True)
        label = effect_label(etype)
        if not enabled:
            label += "  (disabled)"

        item = QListWidgetItem()
        item.setData(Qt.UserRole, index)
        self._list.addItem(item)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(4, 2, 4, 2)

        # Drag grip label
        grip = QLabel("\u2261")  # ≡
        grip.setFixedWidth(16)
        row_layout.addWidget(grip)

        # Enable checkbox
        cb = QCheckBox()
        cb.setChecked(enabled)
        cb.stateChanged.connect(lambda state, idx=index: self._toggle_enabled(idx, state))
        row_layout.addWidget(cb)

        # Label
        name_lbl = QLabel(effect_label(etype))
        row_layout.addWidget(name_lbl, 1)

        # Edit button
        edit_btn = QPushButton("Edit")
        edit_btn.setFixedSize(40, 24)
        edit_btn.clicked.connect(lambda checked=False, idx=index: self._edit_effect(idx))
        row_layout.addWidget(edit_btn)

        # Delete button
        del_btn = QPushButton("\u2717")  # ✗
        del_btn.setFixedSize(24, 24)
        del_btn.setFlat(True)
        del_btn.clicked.connect(lambda checked=False, idx=index: self._delete_effect(idx))
        row_layout.addWidget(del_btn)

        item.setSizeHint(row_widget.sizeHint())
        self._list.setItemWidget(item, row_widget)

    def _toggle_enabled(self, index: int, state: int):
        if 0 <= index < len(self._clip.effects):
            self._clip.effects[index]["enabled"] = bool(state)
            self.effects_changed.emit()

    def _delete_effect(self, index: int):
        if 0 <= index < len(self._clip.effects):
            fx = self._clip.effects[index]
            # Clean up LUT file on disk if this is a lut3d effect
            if fx.get("type") == "lut3d":
                lut_file = fx.get("params", {}).get("lut_file", "")
                if lut_file:
                    from pathlib import Path
                    Path(lut_file).unlink(missing_ok=True)
            self._clip.effects.pop(index)
            self._populate()
            self.effects_changed.emit()

    def _edit_effect(self, index: int):
        """Open the editor for effect at index (called from Edit button)."""
        logger.info("_edit_effect called: index=%d, effects=%d", index, len(self._clip.effects))
        if index < 0 or index >= len(self._clip.effects):
            logger.warning("_edit_effect: index %d out of range", index)
            return
        item = self._list.item(index)
        if item:
            self._on_double_click(item)
        else:
            logger.warning("_edit_effect: no list item at index %d", index)

    def _on_double_click(self, item: QListWidgetItem):
        # Map visual row back to effect index
        row = self._list.row(item)
        if row < 0 or row >= len(self._clip.effects):
            logger.warning("Effects: click row %d out of range (effects=%d)", row, len(self._clip.effects))
            return
        fx = self._clip.effects[row]
        logger.info("Effects: editing %s (row=%d)", fx.get("type"), row)

        # Crop/Zoom gets its own visual editor
        if fx.get("type") == "crop_zoom":
            from sdqt.widgets.crop_editor import CropZoomDialog
            try:
                dlg = CropZoomDialog(self._clip, parent=self.window())
            except Exception:
                logger.error("CropZoomDialog creation failed", exc_info=True)
                return
            if dlg.exec() == QDialog.Accepted:
                fx["params"] = dlg.crop_values()
                self._populate()
                self.effects_changed.emit()
            return

        # Save original params for cancel-restore (live preview modifies fx in-place)
        import copy
        original_params = copy.deepcopy(fx.get("params", {}))
        dlg = EffectEditorDialog(fx, parent=self)
        dlg.preview_changed.connect(self.preview_changed.emit)
        if dlg.exec() == QDialog.Accepted:
            fx["params"] = dlg.values()
            self._populate()
            self.effects_changed.emit()
        else:
            # Restore original params on cancel
            fx["params"] = original_params
            self.preview_changed.emit()

    def _on_rows_moved(self, *args):
        """Sync clip.effects order after drag-reorder."""
        new_order = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            old_idx = item.data(Qt.UserRole)
            if old_idx is not None and 0 <= old_idx < len(self._clip.effects):
                new_order.append(self._clip.effects[old_idx])
        if len(new_order) == len(self._clip.effects):
            self._clip.effects[:] = new_order
        self._populate()
        self.effects_changed.emit()

    # Effect types that are auto-generated and not manually addable
    _HIDDEN_TYPES = {"lut3d"}

    def _show_add_menu(self):
        menu = QMenu(self)
        for etype in EFFECT_TYPES:
            if etype in self._HIDDEN_TYPES:
                continue
            label = effect_label(etype)
            act = menu.addAction(label)
            act.setData(etype)
        chosen = menu.exec(self.mapToGlobal(self.sender().pos()))
        if chosen:
            etype = chosen.data()
            fx = make_effect(etype)
            self._clip.effects.append(fx)
            self._populate()
            self.effects_changed.emit()

            # Open visual editor immediately for crop/zoom
            if etype == "crop_zoom":
                from sdqt.widgets.crop_editor import CropZoomDialog
                dlg = CropZoomDialog(self._clip, parent=self.window())
                if dlg.exec() == QDialog.Accepted:
                    fx["params"] = dlg.crop_values()
                    self._populate()
                    self.effects_changed.emit()
