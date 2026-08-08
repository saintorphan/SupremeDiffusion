"""DrawToolSettingsPanel -- shared context-sensitive tool controls for Draw/Inpaint."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from sdqt.widgets.draw.styles import MAX_RECENT_COLORS, PRESET_COLORS

# Category -> frozenset of widget keys to show
_VISIBILITY: dict[str, frozenset[str]] = {
    "brush": frozenset({
        "size", "size_lbl", "size_val",
        "opacity_lbl", "opacity", "opacity_val",
        "hardness_lbl", "hardness", "hardness_val",
        "color", "secondary_color", "swatches",
        "blend_lbl", "blend_mode",
    }),
    "eraser": frozenset({
        "size", "size_lbl", "size_val",
        "opacity_lbl", "opacity", "opacity_val",
        "hardness_lbl", "hardness", "hardness_val",
    }),
    "shape": frozenset({
        "color", "fill_color", "filled",
        "width_lbl", "width", "width_val",
    }),
    "wand": frozenset({"tol_lbl", "tol", "tol_val"}),
    "fill": frozenset({"color", "tol_lbl", "tol", "tol_val"}),
    "eyedropper": frozenset({"color"}),
    "smudge": frozenset({
        "size", "size_lbl", "size_val",
        "strength_lbl", "strength", "strength_val",
    }),
    "gradient": frozenset({
        "grad_lbl", "grad_mode", "grad_start", "grad_end",
    }),
    "text": frozenset({"color"}),
    "none": frozenset(),
}


class DrawToolSettingsPanel(QWidget):
    """Fixed-height panel that shows/hides controls based on tool category."""

    brush_size_changed = Signal(int)
    opacity_changed = Signal(int)
    hardness_changed = Signal(int)
    strength_changed = Signal(int)
    primary_color_changed = Signal(QColor)
    secondary_color_changed = Signal(QColor)
    fill_color_changed = Signal(QColor)
    tolerance_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)

        self._primary_color = QColor(255, 0, 0)
        self._secondary_color = QColor(255, 255, 255)
        self._fill_color = QColor(100, 100, 255, 128)
        self._grad_start_color = QColor(255, 255, 255)
        self._grad_end_color = QColor(0, 0, 0, 0)
        self._recent_colors: list[QColor] = []

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        # -- Brush size --------------------------------------------------------
        self._w: dict[str, QWidget] = {}

        lbl = QLabel("Size:")
        self._w["size_lbl"] = lbl
        row.addWidget(lbl)

        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(1, 100)
        sl.setValue(8)
        sl.setFixedWidth(100)
        sl.valueChanged.connect(self._on_size)
        self._size_slider = sl
        self._w["size"] = sl
        row.addWidget(sl)

        val = QLabel("8")
        val.setFixedWidth(24)
        self._size_val = val
        self._w["size_val"] = val
        row.addWidget(val)

        # -- Opacity -----------------------------------------------------------
        lbl2 = QLabel("Op:")
        self._w["opacity_lbl"] = lbl2
        row.addWidget(lbl2)

        osl = QSlider(Qt.Orientation.Horizontal)
        osl.setRange(0, 100)
        osl.setValue(100)
        osl.setFixedWidth(80)
        osl.valueChanged.connect(self._on_opacity)
        self._opacity_slider = osl
        self._w["opacity"] = osl
        row.addWidget(osl)

        oval = QLabel("100%")
        oval.setFixedWidth(30)
        self._opacity_val = oval
        self._w["opacity_val"] = oval
        row.addWidget(oval)

        # -- Hardness ----------------------------------------------------------
        hlbl = QLabel("Hard:")
        self._w["hardness_lbl"] = hlbl
        row.addWidget(hlbl)

        hsl = QSlider(Qt.Orientation.Horizontal)
        hsl.setRange(0, 100)
        hsl.setValue(100)
        hsl.setFixedWidth(70)
        hsl.valueChanged.connect(self._on_hardness)
        self._hardness_slider = hsl
        self._w["hardness"] = hsl
        row.addWidget(hsl)

        hval = QLabel("100")
        hval.setFixedWidth(24)
        self._hardness_val = hval
        self._w["hardness_val"] = hval
        row.addWidget(hval)

        # -- Strength (smudge) -------------------------------------------------
        stlbl = QLabel("Str:")
        self._w["strength_lbl"] = stlbl
        row.addWidget(stlbl)

        stsl = QSlider(Qt.Orientation.Horizontal)
        stsl.setRange(1, 100)
        stsl.setValue(50)
        stsl.setFixedWidth(70)
        stsl.valueChanged.connect(self._on_strength)
        self._strength_slider = stsl
        self._w["strength"] = stsl
        row.addWidget(stsl)

        stval = QLabel("50")
        stval.setFixedWidth(24)
        self._strength_val = stval
        self._w["strength_val"] = stval
        row.addWidget(stval)

        # -- Gradient mode -----------------------------------------------------
        gmlbl = QLabel("Grad:")
        self._w["grad_lbl"] = gmlbl
        row.addWidget(gmlbl)

        gm = QComboBox()
        gm.addItems(["linear", "radial"])
        gm.setFixedWidth(80)
        self._grad_combo = gm
        self._w["grad_mode"] = gm
        row.addWidget(gm)

        # -- Brush blend mode --------------------------------------------------
        bmlbl = QLabel("Mode:")
        self._w["blend_lbl"] = bmlbl
        row.addWidget(bmlbl)

        bm = QComboBox()
        bm.addItems(["normal", "multiply", "screen", "overlay", "darken", "lighten"])
        bm.setFixedWidth(90)
        self._blend_combo = bm
        self._w["blend_mode"] = bm
        row.addWidget(bm)

        # -- Color buttons -----------------------------------------------------
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(28, 28)
        self._color_btn.setStyleSheet("background: #ff0000;")
        self._color_btn.setToolTip("Primary color")
        self._color_btn.clicked.connect(self._pick_primary)
        self._w["color"] = self._color_btn
        row.addWidget(self._color_btn)

        self._secondary_btn = QPushButton()
        self._secondary_btn.setFixedSize(28, 28)
        self._secondary_btn.setStyleSheet("background: #ffffff;")
        self._secondary_btn.setToolTip("Secondary color (right click)")
        self._secondary_btn.clicked.connect(self._pick_secondary)
        self._w["secondary_color"] = self._secondary_btn
        row.addWidget(self._secondary_btn)

        self._fill_btn = QPushButton()
        self._fill_btn.setFixedSize(28, 28)
        self._fill_btn.setStyleSheet("background: rgba(100,100,255,128);")
        self._fill_btn.setToolTip("Fill color")
        self._fill_btn.clicked.connect(self._pick_fill)
        self._w["fill_color"] = self._fill_btn
        row.addWidget(self._fill_btn)

        # -- Gradient start/end colors -----------------------------------------
        self._grad_start_btn = QPushButton()
        self._grad_start_btn.setFixedSize(28, 28)
        self._grad_start_btn.setStyleSheet("background: #ffffff;")
        self._grad_start_btn.setToolTip("Gradient start color")
        self._grad_start_btn.clicked.connect(self._pick_grad_start)
        self._w["grad_start"] = self._grad_start_btn
        row.addWidget(self._grad_start_btn)

        self._grad_end_btn = QPushButton()
        self._grad_end_btn.setFixedSize(28, 28)
        self._grad_end_btn.setStyleSheet("background: rgba(0,0,0,0);")
        self._grad_end_btn.setToolTip("Gradient end color")
        self._grad_end_btn.clicked.connect(self._pick_grad_end)
        self._w["grad_end"] = self._grad_end_btn
        row.addWidget(self._grad_end_btn)

        # -- Filled checkbox + line width --------------------------------------
        self._filled_check = QCheckBox("Filled")
        self._filled_check.setChecked(True)
        self._filled_check.stateChanged.connect(lambda _: self._emit_sync())
        self._w["filled"] = self._filled_check
        row.addWidget(self._filled_check)

        wlbl = QLabel("Width:")
        self._w["width_lbl"] = wlbl
        row.addWidget(wlbl)

        wsl = QSlider(Qt.Orientation.Horizontal)
        wsl.setRange(1, 20)
        wsl.setValue(2)
        wsl.setFixedWidth(80)
        wsl.valueChanged.connect(self._on_width)
        self._width_slider = wsl
        self._w["width"] = wsl
        row.addWidget(wsl)

        wval = QLabel("2")
        wval.setFixedWidth(16)
        self._width_val = wval
        self._w["width_val"] = wval
        row.addWidget(wval)

        # -- Tolerance ---------------------------------------------------------
        tlbl = QLabel("Tolerance:")
        self._w["tol_lbl"] = tlbl
        row.addWidget(tlbl)

        tsl = QSlider(Qt.Orientation.Horizontal)
        tsl.setRange(1, 255)
        tsl.setValue(32)
        tsl.setFixedWidth(100)
        tsl.valueChanged.connect(self._on_tolerance)
        self._tol_slider = tsl
        self._w["tol"] = tsl
        row.addWidget(tsl)

        tval = QLabel("32")
        tval.setFixedWidth(24)
        self._tol_val = tval
        self._w["tol_val"] = tval
        row.addWidget(tval)

        # -- Swatches (preset + recent) ----------------------------------------
        swatch_container = QWidget()
        swatch_row = QHBoxLayout(swatch_container)
        swatch_row.setContentsMargins(0, 0, 0, 0)
        swatch_row.setSpacing(2)

        for hc in PRESET_COLORS:
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                f"QPushButton {{ border: 1px solid #555; background: {hc};"
                f" min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;"
                f" border-radius: 2px; }} "
                "QPushButton:hover { border-color: #fff; }"
            )
            btn.setToolTip(hc)
            btn.clicked.connect(lambda checked, c=hc: self._on_swatch(c))
            swatch_row.addWidget(btn)

        sep = QLabel("|")
        sep.setFixedWidth(8)
        sep.setStyleSheet("color: #555;")
        swatch_row.addWidget(sep)

        self._recent_btns: list[QPushButton] = []
        for _ in range(MAX_RECENT_COLORS):
            btn = QPushButton()
            btn.setFixedSize(20, 20)
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #444; background: #333;"
                " min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;"
                " border-radius: 2px; } "
                "QPushButton:hover { border-color: #fff; }"
            )
            btn.clicked.connect(lambda checked, b=btn: self._on_recent_swatch(b))
            swatch_row.addWidget(btn)
            self._recent_btns.append(btn)

        self._w["swatches"] = swatch_container
        row.addWidget(swatch_container)

        row.addStretch()

        # Start hidden
        for w in self._w.values():
            w.setVisible(False)

    # -- Public API -----------------------------------------------------------

    def show_for_tool(self, category: str) -> None:
        """Show controls appropriate for 'brush'|'eraser'|'shape'|'wand'|'fill'|'eyedropper'|'none'."""
        visible = _VISIBILITY.get(category, frozenset())
        for key, widget in self._w.items():
            widget.setVisible(key in visible)

    def sync_to_tool(self, tool) -> None:
        """Push current panel values onto a tool via hasattr."""
        if tool is None:
            return
        if hasattr(tool, "color"):
            tool.color = self._primary_color
        if hasattr(tool, "secondary_color"):
            tool.secondary_color = self._secondary_color
        if hasattr(tool, "size"):
            tool.size = self._size_slider.value()
        if hasattr(tool, "opacity"):
            tool.opacity = self._opacity_slider.value() / 100.0
        if hasattr(tool, "tolerance"):
            tool.tolerance = self._tol_slider.value()
        if hasattr(tool, "line_color"):
            tool.line_color = self._primary_color
        if hasattr(tool, "fill_color"):
            tool.fill_color = self._fill_color
        if hasattr(tool, "filled"):
            tool.filled = self._filled_check.isChecked()
        if hasattr(tool, "line_width"):
            tool.line_width = self._width_slider.value()
        if hasattr(tool, "hardness"):
            tool.hardness = self._hardness_slider.value()
        if hasattr(tool, "strength"):
            tool.strength = self._strength_slider.value() / 100.0
        if hasattr(tool, "blend_mode"):
            tool.blend_mode = self._blend_combo.currentText()
        if hasattr(tool, "mode"):  # gradient mode
            tool.mode = self._grad_combo.currentText()
        if hasattr(tool, "color_start"):
            tool.color_start = QColor(self._grad_start_color)
        if hasattr(tool, "color_end"):
            tool.color_end = QColor(self._grad_end_color)

    def set_primary_color(self, color: QColor) -> None:
        self._primary_color = QColor(color)
        self._color_btn.setStyleSheet(f"background: {color.name()};")
        self._add_recent(color)
        self.primary_color_changed.emit(color)

    @property
    def brush_size(self) -> int:
        return self._size_slider.value()

    @property
    def opacity(self) -> int:
        return self._opacity_slider.value()

    @property
    def primary_color(self) -> QColor:
        return QColor(self._primary_color)

    @property
    def secondary_color(self) -> QColor:
        return QColor(self._secondary_color)

    @property
    def fill_color(self) -> QColor:
        return QColor(self._fill_color)

    @property
    def is_filled(self) -> bool:
        return self._filled_check.isChecked()

    @property
    def line_width(self) -> int:
        return self._width_slider.value()

    @property
    def tolerance(self) -> int:
        return self._tol_slider.value()

    @property
    def hardness(self) -> int:
        return self._hardness_slider.value()

    @property
    def strength(self) -> int:
        return self._strength_slider.value()

    # -- Internal slots -------------------------------------------------------

    def _on_size(self, v: int) -> None:
        self._size_val.setText(str(v))
        self.brush_size_changed.emit(v)

    def _on_opacity(self, v: int) -> None:
        self._opacity_val.setText(f"{v}%")
        self.opacity_changed.emit(v)

    def _on_width(self, v: int) -> None:
        self._width_val.setText(str(v))
        self._emit_sync()

    def _on_tolerance(self, v: int) -> None:
        self._tol_val.setText(str(v))
        self.tolerance_changed.emit(v)

    def _on_hardness(self, v: int) -> None:
        self._hardness_val.setText(str(v))
        self.hardness_changed.emit(v)

    def _on_strength(self, v: int) -> None:
        self._strength_val.setText(str(v))
        self.strength_changed.emit(v)

    def _emit_sync(self) -> None:
        """Generic re-sync trigger for shape settings changes."""
        # Parents connect to brush_size_changed etc.; for filled/width we just
        # emit tolerance_changed(current) as a generic "settings changed" nudge.
        # This is a no-op signal that parents can ignore; the real sync is
        # done by the parent calling sync_to_tool() in response.
        self.tolerance_changed.emit(self._tol_slider.value())

    def _pick_primary(self) -> None:
        color = QColorDialog.getColor(
            self._primary_color, self, "Primary Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self.set_primary_color(color)

    def _pick_secondary(self) -> None:
        color = QColorDialog.getColor(self._secondary_color, self, "Secondary Color")
        if color.isValid():
            self._secondary_color = color
            self._secondary_btn.setStyleSheet(f"background: {color.name()};")
            self.secondary_color_changed.emit(color)

    def _pick_fill(self) -> None:
        color = QColorDialog.getColor(
            self._fill_color, self, "Fill Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._fill_color = color
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            self._fill_btn.setStyleSheet(f"background: rgba({r},{g},{b},{a});")
            self.fill_color_changed.emit(color)

    def _pick_grad_start(self) -> None:
        color = QColorDialog.getColor(
            self._grad_start_color, self, "Gradient Start",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._grad_start_color = color
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            self._grad_start_btn.setStyleSheet(f"background: rgba({r},{g},{b},{a});")

    def _pick_grad_end(self) -> None:
        color = QColorDialog.getColor(
            self._grad_end_color, self, "Gradient End",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._grad_end_color = color
            r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
            self._grad_end_btn.setStyleSheet(f"background: rgba({r},{g},{b},{a});")

    def _on_swatch(self, hex_color: str) -> None:
        color = QColor(hex_color)
        if color.isValid():
            self.set_primary_color(color)

    def _on_recent_swatch(self, btn: QPushButton) -> None:
        idx = self._recent_btns.index(btn) if btn in self._recent_btns else -1
        if 0 <= idx < len(self._recent_colors):
            self.set_primary_color(self._recent_colors[idx])

    def _add_recent(self, color: QColor) -> None:
        name = color.name()
        if name in PRESET_COLORS:
            return
        self._recent_colors = [c for c in self._recent_colors if c.name() != name]
        self._recent_colors.insert(0, QColor(color))
        self._recent_colors = self._recent_colors[:MAX_RECENT_COLORS]
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        for i, btn in enumerate(self._recent_btns):
            if i < len(self._recent_colors):
                c = self._recent_colors[i]
                btn.setStyleSheet(
                    f"QPushButton {{ border: 1px solid #555; background: {c.name()};"
                    f" min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;"
                    f" border-radius: 2px; }} "
                    "QPushButton:hover { border-color: #fff; }"
                )
                btn.setToolTip(c.name())
            else:
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid #444; background: #333;"
                    " min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;"
                    " border-radius: 2px; } "
                    "QPushButton:hover { border-color: #fff; }"
                )
                btn.setToolTip("")
