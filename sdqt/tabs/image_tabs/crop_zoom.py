"""Crop / Zoom sub-tab for Image Suite -- interactive image cropping."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, Slot, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.tabs.base import BaseTab
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget

logger = logging.getLogger(__name__)

_W_SPIN = 75

# ── Aspect ratio presets ────────────────────────────────────────────────────

_ASPECT_PRESETS: list[tuple[str, float | None]] = [
    ("Free", None),
    ("1:1", 1.0),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("3:2", 3 / 2),
    ("2:3", 2 / 3),
    ("21:9", 21 / 9),
    ("9:21", 9 / 21),
]

# ── Size presets ────────────────────────────────────────────────────────────

_SIZE_PRESETS: list[tuple[str, int, int]] = [
    ("512x512", 512, 512),
    ("768x768", 768, 768),
    ("832x480", 832, 480),
    ("480x832", 480, 832),
    ("1024x1024", 1024, 1024),
    ("1280x720", 1280, 720),
    ("720x1280", 720, 1280),
    ("1920x1080", 1920, 1080),
]

# ── Resize presets ─────────────────────────────────────────────────────

_RESIZE_PRESETS: list[tuple[str, int, int]] = [
    ("832x480", 832, 480),
    ("1024x576", 1024, 576),
    ("1280x720", 1280, 720),
    ("1920x1080", 1920, 1080),
    ("1024x1024", 1024, 1024),
    ("768x1344", 768, 1344),
    ("1344x768", 1344, 768),
    ("512x512", 512, 512),
]

_RESAMPLE_METHODS: list[tuple[str, Qt.TransformationMode]] = [
    ("Smooth (Bilinear)", Qt.TransformationMode.SmoothTransformation),
    ("Fast (Nearest)", Qt.TransformationMode.FastTransformation),
]

# ═══════════════════════════════════════════════════════════════════════════
#  _CropView — full-featured crop canvas with resize handles
# ═══════════════════════════════════════════════════════════════════════════


class _CropView(QWidget):
    """Image preview with draggable/resizable crop rectangle, dim overlay,
    rule-of-thirds guides, and aspect ratio locking."""

    crop_changed = Signal(int, int, int, int)  # x, y, w, h in pixels

    _HANDLE_SIZE = 8
    _EDGE_ZONE = 8

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._img_w = 0
        self._img_h = 0

        # Crop in pixel coords
        self._cx = 0
        self._cy = 0
        self._cw = 512
        self._ch = 512

        # Aspect lock (w/h ratio, or None = free)
        self._aspect: float | None = None

        # Drag state
        self._drag_mode = ""  # "move", "tl", "tr", "bl", "br", "t", "b", "l", "r"
        self._drag_start = None
        self._drag_crop_start: tuple[int, int, int, int] | None = None

        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #1a1a1a;")

    def load_image(self, path: str) -> tuple[int, int]:
        img = QPixmap(path)
        if img.isNull():
            return (0, 0)
        self._pixmap = img
        self._img_w = img.width()
        self._img_h = img.height()
        # Init crop to center, clamped to image
        self._cw = min(self._cw, self._img_w)
        self._ch = min(self._ch, self._img_h)
        self._cx = (self._img_w - self._cw) // 2
        self._cy = (self._img_h - self._ch) // 2
        self.update()
        self._emit()
        return (self._img_w, self._img_h)

    def set_crop(self, x: int, y: int, w: int, h: int) -> None:
        self._cx = max(0, min(x, self._img_w - 1))
        self._cy = max(0, min(y, self._img_h - 1))
        self._cw = max(1, min(w, self._img_w - self._cx))
        self._ch = max(1, min(h, self._img_h - self._cy))
        self.update()

    def get_crop(self) -> tuple[int, int, int, int]:
        return (self._cx, self._cy, self._cw, self._ch)

    def set_aspect(self, aspect: float | None) -> None:
        self._aspect = aspect
        if aspect is not None:
            self._constrain_aspect()
            self._emit()
        self.update()

    def crop_image(self, path: str) -> QImage | None:
        img = QImage(path)
        if img.isNull():
            return None
        return img.copy(self._cx, self._cy, self._cw, self._ch)

    def _emit(self) -> None:
        self.crop_changed.emit(self._cx, self._cy, self._cw, self._ch)

    # ── Coordinate mapping ───────────────────────────────────────────

    def _image_rect(self) -> QRectF:
        """The rectangle where the preview image is drawn (letterboxed)."""
        if not self._pixmap:
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._img_w, self._img_h
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        dw, dh = pw * scale, ph * scale
        return QRectF((ww - dw) / 2, (wh - dh) / 2, dw, dh)

    def _px_to_widget(self, px: int, py: int) -> tuple[float, float]:
        r = self._image_rect()
        if self._img_w == 0 or self._img_h == 0:
            return (0.0, 0.0)
        return (r.x() + px * r.width() / self._img_w,
                r.y() + py * r.height() / self._img_h)

    def _widget_to_px(self, wx: float, wy: float) -> tuple[int, int]:
        r = self._image_rect()
        if r.width() == 0 or r.height() == 0:
            return (0, 0)
        return (int((wx - r.x()) * self._img_w / r.width()),
                int((wy - r.y()) * self._img_h / r.height()))

    def _crop_widget_rect(self) -> QRectF:
        x1, y1 = self._px_to_widget(self._cx, self._cy)
        x2, y2 = self._px_to_widget(self._cx + self._cw, self._cy + self._ch)
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    # ── Painting ─────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(26, 26, 26))

        if self._pixmap:
            ir = self._image_rect()
            p.drawPixmap(ir.toRect(), self._pixmap)

            # Dim outside crop
            cr = self._crop_widget_rect()
            dim = QColor(0, 0, 0, 150)
            p.fillRect(QRectF(ir.x(), ir.y(), ir.width(), cr.y() - ir.y()), dim)
            p.fillRect(QRectF(ir.x(), cr.bottom(), ir.width(), ir.bottom() - cr.bottom()), dim)
            p.fillRect(QRectF(ir.x(), cr.y(), cr.x() - ir.x(), cr.height()), dim)
            p.fillRect(QRectF(cr.right(), cr.y(), ir.right() - cr.right(), cr.height()), dim)

            # Crop border
            p.setPen(QPen(QColor(0, 120, 212), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(cr)

            # Rule-of-thirds
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
            for hx, hy in [(cr.left(), cr.top()), (cr.right(), cr.top()),
                           (cr.left(), cr.bottom()), (cr.right(), cr.bottom())]:
                p.drawRect(QRectF(hx - hs / 2, hy - hs / 2, hs, hs))

            # Edge midpoint handles
            p.setBrush(QColor(0, 120, 212, 180))
            hs2 = hs * 0.7
            for hx, hy in [
                (cr.center().x(), cr.top()),
                (cr.center().x(), cr.bottom()),
                (cr.left(), cr.center().y()),
                (cr.right(), cr.center().y()),
            ]:
                p.drawRect(QRectF(hx - hs2 / 2, hy - hs2 / 2, hs2, hs2))

        p.end()

    # ── Hit testing ──────────────────────────────────────────────────

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

    # ── Mouse interaction ────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap:
            mode = self._hit_test(event.position().x(), event.position().y())
            if mode:
                self._drag_mode = mode
                self._drag_start = event.position()
                self._drag_crop_start = (self._cx, self._cy, self._cw, self._ch)
            else:
                # Click outside: jump crop center to click position
                ir = self._image_rect()
                if ir.contains(event.position()):
                    px, py = self._widget_to_px(event.position().x(), event.position().y())
                    nx = max(0, min(self._img_w - self._cw, px - self._cw // 2))
                    ny = max(0, min(self._img_h - self._ch, py - self._ch // 2))
                    self._cx, self._cy = nx, ny
                    self.update()
                    self._emit()
                    self._drag_mode = "move"
                    self._drag_start = event.position()
                    self._drag_crop_start = (self._cx, self._cy, self._cw, self._ch)

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_mode or not self._drag_start or not self._drag_crop_start:
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
        if ir.width() == 0 or ir.height() == 0:
            return
        scale_x = self._img_w / ir.width()
        scale_y = self._img_h / ir.height()
        dx = int((event.position().x() - self._drag_start.x()) * scale_x)
        dy = int((event.position().y() - self._drag_start.y()) * scale_y)
        ox, oy, ow, oh = self._drag_crop_start
        mode = self._drag_mode
        MIN_PX = 16

        if mode == "move":
            self._cx = max(0, min(self._img_w - ow, ox + dx))
            self._cy = max(0, min(self._img_h - oh, oy + dy))
        else:
            cx, cy, cw, ch = ox, oy, ow, oh
            if "l" in mode:
                new_x = max(0, ox + dx)
                new_w = (ox + ow) - new_x
                if new_w >= MIN_PX:
                    cx, cw = new_x, new_w
            if "r" in mode:
                cw = max(MIN_PX, min(self._img_w - cx, ow + dx))
            if "t" in mode:
                new_y = max(0, oy + dy)
                new_h = (oy + oh) - new_y
                if new_h >= MIN_PX:
                    cy, ch = new_y, new_h
            if "b" in mode:
                ch = max(MIN_PX, min(self._img_h - cy, oh + dy))
            self._cx, self._cy, self._cw, self._ch = cx, cy, cw, ch

        if self._aspect is not None:
            self._constrain_aspect()
        self.update()
        self._emit()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = ""
            self._drag_start = None

    # ── Aspect constraint ────────────────────────────────────────────

    def _constrain_aspect(self) -> None:
        if self._aspect is None or self._img_w == 0 or self._img_h == 0:
            return
        # Adjust height to match aspect, keep width
        new_h = int(self._cw / self._aspect)
        if new_h > self._img_h:
            new_h = self._img_h
            self._cw = int(new_h * self._aspect)
        if self._cy + new_h > self._img_h:
            self._cy = self._img_h - new_h
        if self._cx + self._cw > self._img_w:
            self._cx = self._img_w - self._cw
        self._cx = max(0, self._cx)
        self._cy = max(0, self._cy)
        self._ch = new_h


# ═══════════════════════════════════════════════════════════════════════════
#  CropZoomTab
# ═══════════════════════════════════════════════════════════════════════════


class CropZoomTab(BaseTab):
    """Interactive crop/zoom tool for images."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._source_path: str | None = None
        self._img_w = 0
        self._img_h = 0
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 0, 2, 2)
        outer.setSpacing(4)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setColumnMinimumWidth(1, 300)

        # ── Left: crop view ─────────────────────────────────────────
        self._crop_view = _CropView()
        self._crop_view.crop_changed.connect(self._on_crop_changed)
        grid.addWidget(self._crop_view, 0, 0, 8, 1)

        row = 0

        # ── Right: source ───────────────────────────────────────────
        self._source = ImageDropWidget("Source Image", thumb_height=120)
        self._source.image_loaded.connect(self._on_source_loaded)
        self._source.image_cleared.connect(self._on_source_cleared)
        grid.addWidget(self._source, row, 1)
        row += 1

        # ── Crop controls ───────────────────────────────────────────
        crop_group = QGroupBox("Crop Region")
        cg_layout = QVBoxLayout(crop_group)
        cg_layout.setSpacing(4)

        # Aspect ratio
        aspect_row = QHBoxLayout()
        aspect_row.setSpacing(6)
        aspect_row.addWidget(QLabel("Aspect:"))
        self._aspect_combo = QComboBox()
        self._aspect_combo.setFixedWidth(80)
        for label, _ in _ASPECT_PRESETS:
            self._aspect_combo.addItem(label)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        aspect_row.addWidget(self._aspect_combo)
        aspect_row.addStretch()
        cg_layout.addLayout(aspect_row)

        # Position
        pos_row = QHBoxLayout()
        pos_row.setSpacing(6)
        pos_row.addWidget(QLabel("X:"))
        self._spin_x = QSpinBox()
        self._spin_x.setRange(0, 9999)
        self._spin_x.setFixedWidth(_W_SPIN)
        pos_row.addWidget(self._spin_x)
        pos_row.addWidget(QLabel("Y:"))
        self._spin_y = QSpinBox()
        self._spin_y.setRange(0, 9999)
        self._spin_y.setFixedWidth(_W_SPIN)
        pos_row.addWidget(self._spin_y)
        pos_row.addStretch()
        cg_layout.addLayout(pos_row)

        # Dimensions
        dim_row = QHBoxLayout()
        dim_row.setSpacing(6)
        dim_row.addWidget(QLabel("W:"))
        self._spin_w = QSpinBox()
        self._spin_w.setRange(1, 9999)
        self._spin_w.setValue(512)
        self._spin_w.setFixedWidth(_W_SPIN)
        dim_row.addWidget(self._spin_w)
        dim_row.addWidget(QLabel("H:"))
        self._spin_h = QSpinBox()
        self._spin_h.setRange(1, 9999)
        self._spin_h.setValue(512)
        self._spin_h.setFixedWidth(_W_SPIN)
        dim_row.addWidget(self._spin_h)
        dim_row.addStretch()
        cg_layout.addLayout(dim_row)

        # Size presets (row 1)
        preset_row1 = QHBoxLayout()
        preset_row1.setSpacing(3)
        for label, w, h in _SIZE_PRESETS[:4]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, pw=w, ph=h: self._apply_size_preset(pw, ph))
            preset_row1.addWidget(btn)
        preset_row1.addStretch()
        cg_layout.addLayout(preset_row1)

        # Size presets (row 2)
        preset_row2 = QHBoxLayout()
        preset_row2.setSpacing(3)
        for label, w, h in _SIZE_PRESETS[4:]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, pw=w, ph=h: self._apply_size_preset(pw, ph))
            preset_row2.addWidget(btn)
        full_btn = QPushButton("Full")
        full_btn.setFixedHeight(22)
        full_btn.clicked.connect(self._apply_full)
        preset_row2.addWidget(full_btn)
        preset_row2.addStretch()
        cg_layout.addLayout(preset_row2)

        grid.addWidget(crop_group, row, 1)
        row += 1

        # ── Resize controls ─────────────────────────────────────
        resize_group = QGroupBox("Resize Output")
        rg_layout = QVBoxLayout(resize_group)
        rg_layout.setSpacing(4)

        # Output dimensions
        rdim_row = QHBoxLayout()
        rdim_row.setSpacing(6)
        rdim_row.addWidget(QLabel("W:"))
        self._resize_w = QSpinBox()
        self._resize_w.setRange(1, 9999)
        self._resize_w.setValue(1344)
        self._resize_w.setFixedWidth(_W_SPIN)
        rdim_row.addWidget(self._resize_w)
        rdim_row.addWidget(QLabel("H:"))
        self._resize_h = QSpinBox()
        self._resize_h.setRange(1, 9999)
        self._resize_h.setValue(768)
        self._resize_h.setFixedWidth(_W_SPIN)
        rdim_row.addWidget(self._resize_h)
        self._lock_aspect = QCheckBox("Lock")
        self._lock_aspect.setChecked(True)
        rdim_row.addWidget(self._lock_aspect)
        rdim_row.addStretch()
        rg_layout.addLayout(rdim_row)

        # Resample method
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(QLabel("Method:"))
        self._resample_combo = QComboBox()
        self._resample_combo.setFixedWidth(140)
        for label, _ in _RESAMPLE_METHODS:
            self._resample_combo.addItem(label)
        method_row.addWidget(self._resample_combo)
        method_row.addStretch()
        rg_layout.addLayout(method_row)

        # Resize presets (row 1)
        rp_row1 = QHBoxLayout()
        rp_row1.setSpacing(3)
        for label, w, h in _RESIZE_PRESETS[:4]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, rw=w, rh=h: self._apply_resize_preset(rw, rh))
            rp_row1.addWidget(btn)
        rp_row1.addStretch()
        rg_layout.addLayout(rp_row1)

        # Resize presets (row 2)
        rp_row2 = QHBoxLayout()
        rp_row2.setSpacing(3)
        for label, w, h in _RESIZE_PRESETS[4:]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda _, rw=w, rh=h: self._apply_resize_preset(rw, rh))
            rp_row2.addWidget(btn)
        rp_row2.addStretch()
        rg_layout.addLayout(rp_row2)

        grid.addWidget(resize_group, row, 1)
        row += 1

        # Wire lock-aspect ratio for resize spinners
        self._resize_aspect: float | None = 1344 / 768
        self._resize_updating = False
        self._resize_w.valueChanged.connect(self._on_resize_w_changed)
        self._resize_h.valueChanged.connect(self._on_resize_h_changed)

        # ── Info label ──────────────────────────────────────────────
        self._info = QLabel("Load an image to crop or resize.")
        self._info.setStyleSheet("color: #aaa;")
        grid.addWidget(self._info, row, 1)
        row += 1

        # ── Action buttons ──────────────────────────────────────────
        btn_widget = QWidget()
        btn_row = QHBoxLayout(btn_widget)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self._crop_btn = QPushButton("Crop")
        self._crop_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._crop_btn.clicked.connect(self._on_crop)
        btn_row.addWidget(self._crop_btn)

        self._resize_btn = QPushButton("Resize")
        self._resize_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._resize_btn.clicked.connect(self._on_resize)
        btn_row.addWidget(self._resize_btn)

        self._crop_resize_btn = QPushButton("Crop + Resize")
        self._crop_resize_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._crop_resize_btn.clicked.connect(self._on_crop_and_resize)
        btn_row.addWidget(self._crop_resize_btn)

        self._save_btn = QPushButton("Save to Project")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self._save_btn)

        self._clear_btn = QPushButton("Clear Results")
        self._clear_btn.clicked.connect(self._on_clear_results)
        btn_row.addWidget(self._clear_btn)

        grid.addWidget(btn_widget, row, 1)
        row += 1

        # ── Gallery ─────────────────────────────────────────────────
        self._gallery = ImageGalleryWidget("Cropped Results")
        grid.addWidget(self._gallery, row, 1)
        row += 1

        # Status
        self._status = QLabel("")
        grid.addWidget(self._status, row, 1)

        outer.addLayout(grid, 1)

        # Connect spinners — auto-apply on change
        self._spin_x.valueChanged.connect(self._on_spinner_changed)
        self._spin_y.valueChanged.connect(self._on_spinner_changed)
        self._spin_w.valueChanged.connect(self._on_spinner_changed)
        self._spin_h.valueChanged.connect(self._on_spinner_changed)
        self._updating = False

    # ── Source loading ───────────────────────────────────────────────

    def load_source(self, path: str) -> None:
        self._source.load_image(path)

    def load_image(self, path: str) -> None:
        """Alias for load_source (used by send-to wiring)."""
        self._source.load_image(path)

    @Slot(str)
    def _on_source_loaded(self, path: str) -> None:
        self._source_path = path
        self._img_w, self._img_h = self._crop_view.load_image(path)
        self._spin_x.setMaximum(max(0, self._img_w - 1))
        self._spin_y.setMaximum(max(0, self._img_h - 1))
        self._spin_w.setMaximum(self._img_w)
        self._spin_h.setMaximum(self._img_h)
        self._info.setText(f"Source: {self._img_w} x {self._img_h}")
        self._persist("cropzoom_source_path", path)

    @Slot()
    def _on_source_cleared(self) -> None:
        self._source_path = None
        self._persist("cropzoom_source_path", "")

    def _persist(self, key: str, value) -> None:
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            setattr(cfg, key, value)
            cfg.save(self.project_path)
        except Exception:
            pass

    # ── Aspect ratio ────────────────────────────────────────────────

    @Slot(int)
    def _on_aspect_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(_ASPECT_PRESETS):
            return
        _, aspect = _ASPECT_PRESETS[idx]
        self._crop_view.set_aspect(aspect)

    # ── Crop region sync ────────────────────────────────────────────

    @Slot(int, int, int, int)
    def _on_crop_changed(self, x: int, y: int, w: int, h: int) -> None:
        """Called when the view's crop rect changes (drag/resize)."""
        self._updating = True
        self._spin_x.setValue(x)
        self._spin_y.setValue(y)
        self._spin_w.setValue(w)
        self._spin_h.setValue(h)
        self._updating = False
        self._update_info()

    def _on_spinner_changed(self) -> None:
        """User edited spinners — auto-apply to view."""
        if self._updating:
            return
        self._crop_view.set_crop(
            self._spin_x.value(), self._spin_y.value(),
            self._spin_w.value(), self._spin_h.value(),
        )
        self._update_info()

    def _update_info(self) -> None:
        if self._img_w == 0:
            return
        x, y, w, h = self._crop_view.get_crop()
        pct_w = w / self._img_w * 100 if self._img_w else 0
        pct_h = h / self._img_h * 100 if self._img_h else 0
        self._info.setText(
            f"Source: {self._img_w}x{self._img_h}  |  "
            f"Crop: {w}x{h} ({pct_w:.0f}% x {pct_h:.0f}%)  at ({x}, {y})"
        )

    # ── Size presets ────────────────────────────────────────────────

    def _apply_size_preset(self, pw: int, ph: int) -> None:
        if self._img_w == 0 or self._img_h == 0:
            return
        w = min(pw, self._img_w)
        h = min(ph, self._img_h)
        x = (self._img_w - w) // 2
        y = (self._img_h - h) // 2
        self._crop_view.set_crop(x, y, w, h)
        self._crop_view._emit()

    def _apply_full(self) -> None:
        if self._img_w == 0:
            return
        self._crop_view.set_crop(0, 0, self._img_w, self._img_h)
        self._crop_view._emit()

    # ── Resize presets / lock-aspect ──────────────────────────────────

    def _apply_resize_preset(self, rw: int, rh: int) -> None:
        self._resize_updating = True
        self._resize_w.setValue(rw)
        self._resize_h.setValue(rh)
        self._resize_aspect = rw / rh if rh else None
        self._resize_updating = False

    def _on_resize_w_changed(self, val: int) -> None:
        if self._resize_updating:
            return
        if self._lock_aspect.isChecked() and self._resize_aspect:
            self._resize_updating = True
            self._resize_h.setValue(max(1, int(val / self._resize_aspect)))
            self._resize_updating = False
        else:
            h = self._resize_h.value()
            self._resize_aspect = val / h if h else None

    def _on_resize_h_changed(self, val: int) -> None:
        if self._resize_updating:
            return
        if self._lock_aspect.isChecked() and self._resize_aspect:
            self._resize_updating = True
            self._resize_w.setValue(max(1, int(val * self._resize_aspect)))
            self._resize_updating = False
        else:
            w = self._resize_w.value()
            self._resize_aspect = w / val if val else None

    # ── Resize helpers ──────────────────────────────────────────────

    def _get_resample_mode(self) -> Qt.TransformationMode:
        idx = self._resample_combo.currentIndex()
        if 0 <= idx < len(_RESAMPLE_METHODS):
            return _RESAMPLE_METHODS[idx][1]
        return Qt.TransformationMode.SmoothTransformation

    def _resize_image(self, img: QImage) -> QImage:
        tw, th = self._resize_w.value(), self._resize_h.value()
        return img.scaled(tw, th, Qt.AspectRatioMode.IgnoreAspectRatio, self._get_resample_mode())

    def _save_result(self, img: QImage, prefix: str) -> None:
        out_dir = self.project_path / "images" / "crops"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w, h = img.width(), img.height()
        out_path = out_dir / f"{prefix}_{w}x{h}_{ts}.png"
        img.save(str(out_path))
        self._gallery.load_images(self._list_results())
        self._save_btn.setEnabled(True)
        self._show_status(f"{prefix.title()}: {w}x{h} saved to {out_path.name}")

    # ── Crop action ─────────────────────────────────────────────────

    @Slot()
    def _on_crop(self) -> None:
        if not self._source_path:
            self._show_status("No source image.")
            return
        cropped = self._crop_view.crop_image(self._source_path)
        if cropped is None or cropped.isNull():
            self._show_status("Crop failed.")
            return
        self._save_result(cropped, "crop")

    # ── Resize action ───────────────────────────────────────────────

    @Slot()
    def _on_resize(self) -> None:
        if not self._source_path:
            self._show_status("No source image.")
            return
        img = QImage(self._source_path)
        if img.isNull():
            self._show_status("Failed to load source.")
            return
        resized = self._resize_image(img)
        if resized.isNull():
            self._show_status("Resize failed.")
            return
        self._save_result(resized, "resize")

    # ── Crop + Resize action ────────────────────────────────────────

    @Slot()
    def _on_crop_and_resize(self) -> None:
        if not self._source_path:
            self._show_status("No source image.")
            return
        cropped = self._crop_view.crop_image(self._source_path)
        if cropped is None or cropped.isNull():
            self._show_status("Crop failed.")
            return
        resized = self._resize_image(cropped)
        if resized.isNull():
            self._show_status("Resize failed.")
            return
        self._save_result(resized, "crop_resize")

    @Slot()
    def _on_save(self) -> None:
        path = self._gallery.selected_path
        if path:
            saved = self.save_frame_to_project(path)
            if saved:
                self._show_status("Saved to project.")

    @Slot()
    def _on_clear_results(self) -> None:
        import shutil
        d = self.project_path / "images" / "crops"
        if d.is_dir():
            shutil.rmtree(d)
        self._gallery.clear()
        self._save_btn.setEnabled(False)
        self._show_status("Results cleared.")

    def _list_results(self) -> list[str]:
        d = self.project_path / "images" / "crops"
        if not d.is_dir():
            return []
        return sorted(str(p) for p in d.glob("*.png"))

    # ── Project change ──────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._gallery.clear()
        self._save_btn.setEnabled(False)
        self._restoring = True
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        src = getattr(cfg, "cropzoom_source_path", "") or ""
        if src and Path(src).is_file():
            self._source.load_image(src)
        else:
            self._source.clear_image()
        results = self._list_results()
        if results:
            self._gallery.load_images(results)
            self._save_btn.setEnabled(True)
        self._restoring = False
