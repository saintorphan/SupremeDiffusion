"""Visual crop/zoom editor — preview frame with draggable rectangle."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

_ASPECT_PRESETS = [
    ("Free", None),
    ("16:9", 16 / 9),
    ("4:3", 4 / 3),
    ("1:1", 1.0),
    ("9:16", 9 / 16),
    ("3:4", 3 / 4),
    ("2.35:1", 2.35),
    ("21:9", 21 / 9),
]


class _CropCanvas(QWidget):
    """Preview image with a draggable/resizable crop rectangle."""

    rect_changed = Signal()  # emitted when user drags the rectangle

    _HANDLE_SIZE = 8
    _EDGE_ZONE = 6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._crop = QRectF(0.0, 0.0, 1.0, 1.0)  # normalised 0–1
        self._aspect: float | None = None
        self._drag_mode: str = ""  # "move", "tl", "tr", "bl", "br", "t", "b", "l", "r"
        self._drag_start = None
        self._drag_crop_start = None
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)

    def set_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def set_aspect(self, aspect: float | None) -> None:
        self._aspect = aspect
        if aspect is not None:
            self._constrain_aspect()
            self.rect_changed.emit()
        self.update()

    def crop_rect(self) -> tuple[float, float, float, float]:
        """Return (x, y, w, h) in 0–1 normalised coords."""
        return (self._crop.x(), self._crop.y(),
                self._crop.width(), self._crop.height())

    def set_crop_rect(self, x: float, y: float, w: float, h: float) -> None:
        self._crop = QRectF(x, y, w, h)
        self.update()

    # -- Coordinate mapping ------------------------------------------------

    def _image_rect(self) -> QRectF:
        """The rectangle where the preview image is drawn (letterboxed)."""
        if not self._pixmap:
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        dw, dh = pw * scale, ph * scale
        ox = (ww - dw) / 2
        oy = (wh - dh) / 2
        return QRectF(ox, oy, dw, dh)

    def _norm_to_widget(self, nx: float, ny: float) -> tuple[float, float]:
        r = self._image_rect()
        return r.x() + nx * r.width(), r.y() + ny * r.height()

    def _widget_to_norm(self, wx: float, wy: float) -> tuple[float, float]:
        r = self._image_rect()
        return ((wx - r.x()) / r.width(),
                (wy - r.y()) / r.height())

    def _crop_widget_rect(self) -> QRectF:
        r = self._image_rect()
        return QRectF(
            r.x() + self._crop.x() * r.width(),
            r.y() + self._crop.y() * r.height(),
            self._crop.width() * r.width(),
            self._crop.height() * r.height(),
        )

    # -- Painting ----------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(30, 30, 30))

        # Image
        if self._pixmap:
            ir = self._image_rect()
            p.drawPixmap(ir.toRect(), self._pixmap)

        # Dim outside crop
        cr = self._crop_widget_rect()
        dim = QColor(0, 0, 0, 150)
        ir = self._image_rect()
        # Top
        p.fillRect(QRectF(ir.x(), ir.y(), ir.width(), cr.y() - ir.y()), dim)
        # Bottom
        p.fillRect(QRectF(ir.x(), cr.bottom(), ir.width(), ir.bottom() - cr.bottom()), dim)
        # Left
        p.fillRect(QRectF(ir.x(), cr.y(), cr.x() - ir.x(), cr.height()), dim)
        # Right
        p.fillRect(QRectF(cr.right(), cr.y(), ir.right() - cr.right(), cr.height()), dim)

        # Crop border
        p.setPen(QPen(QColor(0, 120, 212), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(cr)

        # Rule-of-thirds lines
        p.setPen(QPen(QColor(0, 120, 212, 80), 1, Qt.PenStyle.DashLine))
        for i in (1, 2):
            x = cr.x() + cr.width() * i / 3
            p.drawLine(int(x), int(cr.y()), int(x), int(cr.bottom()))
            y = cr.y() + cr.height() * i / 3
            p.drawLine(int(cr.x()), int(y), int(cr.right()), int(y))

        # Corner handles
        hs = self._HANDLE_SIZE
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 120, 212))
        for cx, cy in [(cr.left(), cr.top()), (cr.right(), cr.top()),
                       (cr.left(), cr.bottom()), (cr.right(), cr.bottom())]:
            p.drawRect(QRectF(cx - hs / 2, cy - hs / 2, hs, hs))

        p.end()

    # -- Mouse interaction -------------------------------------------------

    def _hit_test(self, x: float, y: float) -> str:
        cr = self._crop_widget_rect()
        ez = self._EDGE_ZONE
        in_left = abs(x - cr.left()) < ez
        in_right = abs(x - cr.right()) < ez
        in_top = abs(y - cr.top()) < ez
        in_bottom = abs(y - cr.bottom()) < ez
        in_x = cr.left() - ez < x < cr.right() + ez
        in_y = cr.top() - ez < y < cr.bottom() + ez

        if in_left and in_top:
            return "tl"
        if in_right and in_top:
            return "tr"
        if in_left and in_bottom:
            return "bl"
        if in_right and in_bottom:
            return "br"
        if in_top and in_x:
            return "t"
        if in_bottom and in_x:
            return "b"
        if in_left and in_y:
            return "l"
        if in_right and in_y:
            return "r"
        if cr.contains(x, y):
            return "move"
        return ""

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = self._hit_test(event.position().x(), event.position().y())
            if self._drag_mode:
                self._drag_start = event.position()
                self._drag_crop_start = QRectF(self._crop)
            else:
                # Click outside crop rect — jump crop center to click position
                ir = self._image_rect()
                if ir.contains(event.position()):
                    nx, ny = self._widget_to_norm(
                        event.position().x(), event.position().y())
                    w, h = self._crop.width(), self._crop.height()
                    new_x = max(0.0, min(1.0 - w, nx - w / 2))
                    new_y = max(0.0, min(1.0 - h, ny - h / 2))
                    self._crop = QRectF(new_x, new_y, w, h)
                    self.update()
                    self.rect_changed.emit()
                    # Start drag from new position so user can keep adjusting
                    self._drag_mode = "move"
                    self._drag_start = event.position()
                    self._drag_crop_start = QRectF(self._crop)

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_mode or not self._drag_start:
            # Update cursor
            mode = self._hit_test(event.position().x(), event.position().y())
            cursors = {
                "tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
                "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                "move": Qt.CursorShape.SizeAllCursor,
            }
            self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
            return

        ir = self._image_rect()
        dx = (event.position().x() - self._drag_start.x()) / ir.width()
        dy = (event.position().y() - self._drag_start.y()) / ir.height()
        c = QRectF(self._drag_crop_start)
        mode = self._drag_mode
        MIN = 0.05

        if mode == "move":
            nx = max(0, min(1 - c.width(), c.x() + dx))
            ny = max(0, min(1 - c.height(), c.y() + dy))
            c.moveTopLeft(QRectF(nx, ny, 0, 0).topLeft())
        elif mode in ("tl", "t", "tr", "bl", "b", "br", "l", "r"):
            # Resize from edges/corners
            if "l" in mode:
                new_x = max(0, c.x() + dx)
                new_w = c.right() - new_x
                if new_w >= MIN:
                    c = QRectF(new_x, c.y(), new_w, c.height())
            if "r" in mode:
                new_w = max(MIN, min(1 - c.x(), c.width() + dx))
                c = QRectF(c.x(), c.y(), new_w, c.height())
            if "t" in mode:
                new_y = max(0, c.y() + dy)
                new_h = c.bottom() - new_y
                if new_h >= MIN:
                    c = QRectF(c.x(), new_y, c.width(), new_h)
            if "b" in mode:
                new_h = max(MIN, min(1 - c.y(), c.height() + dy))
                c = QRectF(c.x(), c.y(), c.width(), new_h)

        self._crop = c
        if self._aspect is not None:
            self._constrain_aspect()
        self.update()
        self.rect_changed.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = ""
        self._drag_start = None

    def _constrain_aspect(self) -> None:
        """Adjust height to match the locked aspect ratio."""
        if self._aspect is None:
            return
        w = self._crop.width()
        h = w / self._aspect
        if h > 1.0:
            h = 1.0
            w = h * self._aspect
        if self._crop.y() + h > 1.0:
            self._crop.moveBottom(1.0)
        if self._crop.x() + w > 1.0:
            self._crop.moveRight(1.0)
        self._crop.setWidth(w)
        self._crop.setHeight(h)


class CropZoomDialog(QDialog):
    """Modal dialog with visual crop rectangle editor."""

    def __init__(self, clip, parent=None) -> None:
        super().__init__(parent)
        self._clip = clip
        self._updating_zoom = False
        self._updating_fields = False
        self.setWindowTitle(f"Crop / Zoom — {clip.name}")
        self.setMinimumSize(640, 520)
        self.resize(720, 580)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Top row: Aspect + Fit presets + Reset
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Aspect:"))
        self._aspect_combo = QComboBox()
        for label, _ in _ASPECT_PRESETS:
            self._aspect_combo.addItem(label)
        self._aspect_combo.setFixedWidth(120)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        top_row.addWidget(self._aspect_combo)

        top_row.addSpacing(12)

        # Fit presets
        fit_fill_btn = QPushButton("Fill")
        fit_fill_btn.setToolTip("Zoom to fill — crop edges, no black bars")
        fit_fill_btn.setFixedWidth(50)
        fit_fill_btn.clicked.connect(self._fit_fill)
        top_row.addWidget(fit_fill_btn)

        fit_center_btn = QPushButton("Center")
        fit_center_btn.setToolTip("Center crop at current zoom level")
        fit_center_btn.setFixedWidth(55)
        fit_center_btn.clicked.connect(self._fit_center)
        top_row.addWidget(fit_center_btn)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #888;")
        top_row.addWidget(self._info_label, 1)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        top_row.addWidget(reset_btn)

        layout.addLayout(top_row)

        # Canvas
        self._canvas = _CropCanvas()
        self._canvas.rect_changed.connect(self._update_info)
        self._canvas.rect_changed.connect(self._sync_zoom_from_canvas)
        self._canvas.rect_changed.connect(self._refresh_fields)
        layout.addWidget(self._canvas, 1)

        # Zoom row
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(0, 95)  # 0% = no zoom (full frame), 95% = 20x zoom
        self._zoom_slider.setValue(0)
        self._zoom_slider.setTickInterval(10)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_row.addWidget(self._zoom_slider, 1)
        self._zoom_label = QLabel("1.0x")
        self._zoom_label.setFixedWidth(45)
        zoom_row.addWidget(self._zoom_label)
        zoom_row.addSpacing(8)
        zoom_row.addWidget(QLabel("Drag the crop rectangle to pan"))
        layout.addLayout(zoom_row)

        # Manual numeric entry (2 decimals) + anchor grid. Lets you type exact
        # crop geometry instead of dragging, and snap the region to a fixed
        # anchor (e.g. top-anchored to match an i2v conditioning crop).
        entry_row = QHBoxLayout()

        def _mk_field(suffix="  %"):
            sb = QDoubleSpinBox()
            sb.setRange(0.0, 100.0)
            sb.setDecimals(2)
            sb.setSingleStep(0.10)
            sb.setSuffix(suffix)
            sb.setFixedWidth(88)
            sb.valueChanged.connect(self._on_field_changed)
            return sb

        entry_row.addWidget(QLabel("Region X:"))
        self._field_x = _mk_field()
        entry_row.addWidget(self._field_x)
        entry_row.addWidget(QLabel("Y:"))
        self._field_y = _mk_field()
        entry_row.addWidget(self._field_y)
        entry_row.addSpacing(8)
        entry_row.addWidget(QLabel("Size W:"))
        self._field_w = _mk_field()
        self._field_w.setRange(0.01, 100.0)
        entry_row.addWidget(self._field_w)
        entry_row.addWidget(QLabel("H:"))
        self._field_h = _mk_field()
        self._field_h.setRange(0.01, 100.0)
        entry_row.addWidget(self._field_h)

        entry_row.addSpacing(14)
        entry_row.addWidget(QLabel("Anchor:"))
        anchor_grid = QGridLayout()
        anchor_grid.setSpacing(1)
        _anchors = [
            ("↖", "tl", 0, 0), ("↑", "t", 0, 1), ("↗", "tr", 0, 2),
            ("←", "l", 1, 0), ("•", "c", 1, 1), ("→", "r", 1, 2),
            ("↙", "bl", 2, 0), ("↓", "b", 2, 1), ("↘", "br", 2, 2),
        ]
        for sym, key, r, c in _anchors:
            b = QPushButton(sym)
            b.setFixedSize(24, 20)
            b.setToolTip(f"Anchor region: {key}")
            b.clicked.connect(lambda checked=False, k=key: self._apply_anchor(k))
            anchor_grid.addWidget(b, r, c)
        anchor_holder = QWidget()
        anchor_holder.setLayout(anchor_grid)
        entry_row.addWidget(anchor_holder)
        entry_row.addStretch()
        layout.addLayout(entry_row)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Load preview frame from clip
        self._load_preview()

        # Restore existing crop values if any
        fx_params = self._find_crop_params()
        if fx_params:
            self._canvas.set_crop_rect(
                fx_params.get("x", 0.0), fx_params.get("y", 0.0),
                fx_params.get("w", 1.0), fx_params.get("h", 1.0),
            )
        self._update_info()
        self._refresh_fields()

    def _find_crop_params(self) -> dict | None:
        for fx in self._clip.effects:
            if fx.get("type") == "crop_zoom":
                return fx.get("params", {})
        return None

    def _load_preview(self) -> None:
        """Extract first visible frame from the clip."""
        path = self._clip.path
        mo = getattr(self._clip, "media_offset", 0.0) or 0.0
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="crop_preview_")
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(mo), "-i", path,
                 "-vframes", "1", "-q:v", "2", tmp.name],
                capture_output=True, timeout=10,
            )
            pm = QPixmap(tmp.name)
            if not pm.isNull():
                self._canvas.set_image(pm)
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    def _on_aspect_changed(self, idx: int) -> None:
        _, aspect = _ASPECT_PRESETS[idx]
        self._canvas.set_aspect(aspect)
        self._update_info()

    def _on_reset(self) -> None:
        self._canvas.set_crop_rect(0.0, 0.0, 1.0, 1.0)
        self._aspect_combo.setCurrentIndex(0)
        self._canvas.set_aspect(None)
        self._zoom_slider.setValue(0)
        self._update_info()
        self._refresh_fields()

    def _on_zoom_changed(self, value: int) -> None:
        """Zoom slider changed — resize crop rect symmetrically around its center."""
        if self._updating_zoom:
            return
        # value 0 = full frame (w=h=1.0), value 95 = 5% of frame (20x zoom)
        frac = 1.0 - value / 100.0  # 1.0 → 0.05
        x, y, w, h = self._canvas.crop_rect()
        cx, cy = x + w / 2, y + h / 2

        aspect = self._canvas._aspect
        if aspect is not None:
            # Constrain to aspect ratio
            new_w = frac
            new_h = frac / aspect
            if new_h > 1.0:
                new_h = 1.0
                new_w = new_h * aspect
        else:
            new_w = frac
            new_h = frac

        # Center around current center, clamp to 0–1
        new_x = max(0.0, min(1.0 - new_w, cx - new_w / 2))
        new_y = max(0.0, min(1.0 - new_h, cy - new_h / 2))
        self._canvas.set_crop_rect(new_x, new_y, new_w, new_h)
        self._zoom_label.setText(f"{1.0 / max(frac, 0.05):.1f}x")
        self._update_info()

    def _sync_zoom_from_canvas(self) -> None:
        """Update zoom slider to match the canvas crop rect (after manual drag/resize)."""
        self._updating_zoom = True
        x, y, w, h = self._canvas.crop_rect()
        frac = max(w, h)  # use the larger dimension
        value = int((1.0 - frac) * 100)
        self._zoom_slider.setValue(max(0, min(95, value)))
        zoom = 1.0 / max(frac, 0.05)
        self._zoom_label.setText(f"{zoom:.1f}x")
        self._updating_zoom = False

    def _fit_fill(self) -> None:
        """Zoom to fill — use the largest centered crop that fills the frame."""
        self._canvas.set_crop_rect(0.0, 0.0, 1.0, 1.0)
        if self._canvas._aspect is not None:
            self._canvas._constrain_aspect()
            # Center the result
            x, y, w, h = self._canvas.crop_rect()
            self._canvas.set_crop_rect((1.0 - w) / 2, (1.0 - h) / 2, w, h)
        self._sync_zoom_from_canvas()
        self._canvas.rect_changed.emit()
        self._update_info()

    def _fit_center(self) -> None:
        """Center the crop rectangle at its current size."""
        x, y, w, h = self._canvas.crop_rect()
        new_x = max(0.0, (1.0 - w) / 2)
        new_y = max(0.0, (1.0 - h) / 2)
        self._canvas.set_crop_rect(new_x, new_y, w, h)
        self._canvas.rect_changed.emit()
        self._update_info()

    def _update_info(self) -> None:
        x, y, w, h = self._canvas.crop_rect()
        zoom = 1.0 / max(max(w, h), 0.05)
        self._info_label.setText(
            f"Region: {x:.1%}, {y:.1%}  Size: {w:.1%} x {h:.1%}  ({zoom:.1f}x)"
        )

    def _refresh_fields(self) -> None:
        """Sync the manual entry spinboxes to the current crop (from drag/zoom/etc.)."""
        if not hasattr(self, "_field_x"):
            return
        x, y, w, h = self._canvas.crop_rect()
        self._updating_fields = True
        self._field_x.setValue(x * 100.0)
        self._field_y.setValue(y * 100.0)
        self._field_w.setValue(w * 100.0)
        self._field_h.setValue(h * 100.0)
        self._updating_fields = False

    def _on_field_changed(self) -> None:
        """User typed a value — drive the crop rect from the four fields."""
        if self._updating_fields:
            return
        w = min(max(self._field_w.value() / 100.0, 0.0001), 1.0)
        h = min(max(self._field_h.value() / 100.0, 0.0001), 1.0)
        x = min(max(self._field_x.value() / 100.0, 0.0), 1.0 - w)
        y = min(max(self._field_y.value() / 100.0, 0.0), 1.0 - h)
        self._canvas.set_crop_rect(x, y, w, h)
        self._sync_zoom_from_canvas()
        self._update_info()

    def _apply_anchor(self, anchor: str) -> None:
        """Reposition the region (keeping its size) to a fixed 3x3 anchor point."""
        x, y, w, h = self._canvas.crop_rect()
        if anchor in ("tl", "l", "bl"):
            x = 0.0
        elif anchor in ("tr", "r", "br"):
            x = max(0.0, 1.0 - w)
        else:
            x = max(0.0, (1.0 - w) / 2.0)
        if anchor in ("tl", "t", "tr"):
            y = 0.0
        elif anchor in ("bl", "b", "br"):
            y = max(0.0, 1.0 - h)
        else:
            y = max(0.0, (1.0 - h) / 2.0)
        self._canvas.set_crop_rect(x, y, w, h)
        self._canvas.rect_changed.emit()
        self._update_info()

    def crop_values(self) -> dict:
        """Return the crop params dict."""
        x, y, w, h = self._canvas.crop_rect()
        return {"x": round(x, 4), "y": round(y, 4),
                "w": round(w, 4), "h": round(h, 4)}
