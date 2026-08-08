"""Mask editor widget -- QGraphicsView-based mask painting for inpainter."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from sdqt.widgets.send_targets import DISABLED_TARGETS

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

_ICON_BTN_STYLE = (
    "QPushButton { border: none; padding: 2px 4px; font-size: 13px; color: #aaa; }"
    "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
)

_ZOOM_BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 1px 6px; font-size: 13px;"
    " color: #ccc; background: #333; border-radius: 3px; min-width: 24px; }"
    "QPushButton:hover { color: #fff; background: #555; }"
)


class MaskEditorWidget(QWidget):
    """Mask painting editor for inpainting.

    Shows a source image and lets the user paint a mask over it.
    Supports drag/drop and browse to load the source image directly.
    Zoom: Ctrl+scroll or +/- buttons. Pan: middle-click drag.

    Signals:
        mask_updated(str): path to the mask PNG
        image_loaded(str): emitted when an image is loaded via drop/browse
    """

    mask_updated = Signal(str)
    image_loaded = Signal(str)
    magic_mask_requested = Signal()
    send_requested = Signal(str)
    final_frame_requested = Signal(str)
    guide_video_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._source_path: str | None = None
        self._brush_size = 30
        self._brush_hardness = 100  # 100 = hard edge, lower = soft radial falloff
        self._send_targets: list[tuple[str, str]] = []
        self._final_frame_targets: list[tuple[str, str]] = []
        self._guide_video_targets: list[tuple[str, str]] = []
        self._painting = False
        self._erasing = False
        self._erase_mode = False  # False=paint, True=erase (for toggle)
        self._panning = False
        self._pan_start = QPointF()
        self._last_point = QPointF()
        self._zoom_level = 1.0

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Title bar with browse/clear buttons
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(2)
        self._title_label = QLabel("<b>Mask Editor</b>")
        title_row.addWidget(self._title_label)
        title_row.addStretch()

        self._browse_btn = QPushButton("\u2191")  # ↑
        self._browse_btn.setFixedSize(28, 28)
        self._browse_btn.setToolTip("Browse for image")
        self._browse_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._browse_btn.clicked.connect(self._browse_image)
        title_row.addWidget(self._browse_btn)

        self._menu_btn = QPushButton("\u22EE")  # ⋮
        self._menu_btn.setFixedSize(28, 28)
        self._menu_btn.setToolTip("Send image to...")
        self._menu_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._menu_btn.clicked.connect(self._on_menu_click)
        self._menu_btn.setVisible(False)
        title_row.addWidget(self._menu_btn)

        self._clear_img_btn = QPushButton("\u2715")  # ✕
        self._clear_img_btn.setFixedSize(28, 28)
        self._clear_img_btn.setToolTip("Clear image")
        self._clear_img_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._clear_img_btn.clicked.connect(self._clear_all)
        self._clear_img_btn.setVisible(False)
        title_row.addWidget(self._clear_img_btn)

        layout.addLayout(title_row)

        # Graphics view
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.NoDrag)
        self._view.setAcceptDrops(True)
        self._view.setStyleSheet("QGraphicsView { background: #1a1a1a; border: 2px dashed #444; }")
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setMouseTracking(True)
        layout.addWidget(self._view, 1)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        # Zoom controls
        zoom_out = QPushButton("\u2212")  # −
        zoom_out.setFixedSize(28, 28)
        zoom_out.setStyleSheet(_ZOOM_BTN_STYLE)
        zoom_out.setToolTip("Zoom out")
        zoom_out.clicked.connect(lambda: self._zoom(0.8))
        ctrl.addWidget(zoom_out)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(42)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setStyleSheet("font-size: 13px; color: #aaa;")
        ctrl.addWidget(self._zoom_label)

        zoom_in = QPushButton("+")
        zoom_in.setFixedSize(28, 28)
        zoom_in.setStyleSheet(_ZOOM_BTN_STYLE)
        zoom_in.setToolTip("Zoom in")
        zoom_in.clicked.connect(lambda: self._zoom(1.25))
        ctrl.addWidget(zoom_in)

        fit_btn = QPushButton("Fit")
        fit_btn.setMinimumSize(48, 28)
        fit_btn.setStyleSheet(_ZOOM_BTN_STYLE)
        fit_btn.setToolTip("Fit to view")
        fit_btn.clicked.connect(self._fit_view)
        ctrl.addWidget(fit_btn)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #555;")
        ctrl.addWidget(sep)

        # Brush controls
        ctrl.addWidget(QLabel("Brush:"))
        self._brush_slider = QSlider(Qt.Horizontal)
        self._brush_slider.setRange(2, 200)
        self._brush_slider.setValue(30)
        self._brush_slider.valueChanged.connect(self._on_brush_change)
        ctrl.addWidget(self._brush_slider, 1)
        self._brush_label = QLabel("30")
        self._brush_label.setFixedWidth(32)
        ctrl.addWidget(self._brush_label)

        # Brush hardness (edge feathering). 100% = hard edge (legacy solid
        # brush); lower values paint with a radial alpha falloff for soft edges.
        ctrl.addWidget(QLabel("Hardness:"))
        self._hardness_spin = QSpinBox()
        self._hardness_spin.setRange(0, 100)
        self._hardness_spin.setValue(self._brush_hardness)
        self._hardness_spin.setSuffix("%")
        self._hardness_spin.setMinimumWidth(90)
        self._hardness_spin.setToolTip(
            "Brush edge hardness: 100% = hard edge, lower = soft feathered edge",
        )
        self._hardness_spin.valueChanged.connect(self._on_hardness_change)
        ctrl.addWidget(self._hardness_spin)

        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #555;")
        ctrl.addWidget(sep2)

        self._mode_btn = QPushButton("Paint")
        self._mode_btn.setMinimumSize(60, 28)
        self._mode_btn.setCheckable(True)
        self._mode_btn.setToolTip("Toggle between paint and erase (E)")
        self._mode_btn.setStyleSheet(
            "QPushButton { border: 1px solid #555; padding: 1px 8px; font-size: 13px;"
            " color: #ccc; background: #333; border-radius: 3px; }"
            "QPushButton:hover { color: #fff; background: #555; }"
            "QPushButton:checked { color: #fff; background: #744; border-color: #966; }"
        )
        self._mode_btn.toggled.connect(self._on_mode_toggled)
        ctrl.addWidget(self._mode_btn)

        self._erase_hint = QLabel("LMB=paint/erase  RMB=opposite  E=toggle  Scroll=zoom")
        self._erase_hint.setStyleSheet("color: #999; font-size: 13px;")
        ctrl.addWidget(self._erase_hint)

        clear_btn = QPushButton("Clear Mask")
        clear_btn.clicked.connect(self.clear_mask)
        ctrl.addWidget(clear_btn)

        invert_btn = QPushButton("Invert")
        invert_btn.clicked.connect(self._invert_mask)
        ctrl.addWidget(invert_btn)

        self._magic_mask_btn = QPushButton("Magic Mask")
        self._magic_mask_btn.setToolTip("Auto-segment foreground with AI (BiRefNet)")
        self._magic_mask_btn.clicked.connect(lambda: self.magic_mask_requested.emit())
        ctrl.addWidget(self._magic_mask_btn)

        layout.addLayout(ctrl)

        self._source_item: QGraphicsPixmapItem | None = None
        self._mask_image: QImage | None = None
        self._mask_item: QGraphicsPixmapItem | None = None

        # Brush cursor — high-contrast two-ring + crosshair so it stands out
        # on any background. _cursor_outer / _cursor_inner / _cursor_cross_*
        # are kept for sizing in _update_cursor_size().
        self._build_brush_cursor()

        # Install event filter on the viewport for painting + drag/drop + zoom
        self._view.viewport().installEventFilter(self)
        # Accept drops on the widget itself too
        self.setAcceptDrops(True)

    def _build_brush_cursor(self) -> None:
        """Construct a high-visibility brush cursor and add it to the scene.

        Built as a group of:
          - outer black ring (2px solid, full opacity)
          - inner white ring (2px solid, full opacity, slightly inside)
          - 4 crosshair tick lines through the centre (so the user can
            still see the click point even on a busy background)

        The two-ring + crosshair pattern is the standard for paint apps
        (Photoshop, Krita, GIMP) — guaranteed contrast on any background
        because at least one of the two rings always has high contrast
        against any pixel underneath.
        """
        cursor = QGraphicsItemGroup()
        cursor.setVisible(False)
        cursor.setZValue(100)  # always on top

        # Outer ring — black, thicker. Bumped to 2.5px so it reads at any zoom.
        self._cursor_outer = QGraphicsEllipseItem(parent=cursor)
        self._cursor_outer.setPen(QPen(QColor(0, 0, 0, 230), 2.5))
        self._cursor_outer.setBrush(QBrush(Qt.NoBrush))

        # Inner ring — white, slightly thinner, sitting 1px inside the outer
        # so they don't blend into one fat line at low zoom.
        self._cursor_inner = QGraphicsEllipseItem(parent=cursor)
        self._cursor_inner.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
        self._cursor_inner.setBrush(QBrush(Qt.NoBrush))

        # Crosshair — 4 short ticks meeting at the centre. Outer black halo
        # would compose into a "+", but two thin lines are enough at this
        # cursor size and keep the inside of the brush footprint visible.
        cross_pen = QPen(QColor(255, 255, 255, 230), 1.0)
        cross_pen_outer = QPen(QColor(0, 0, 0, 200), 2.0)
        self._cursor_cross_outer = []
        self._cursor_cross_inner = []
        for _ in range(4):
            o = QGraphicsLineItem(parent=cursor)
            o.setPen(cross_pen_outer)
            self._cursor_cross_outer.append(o)
            i = QGraphicsLineItem(parent=cursor)
            i.setPen(cross_pen)
            self._cursor_cross_inner.append(i)

        self._cursor_item = cursor
        self._scene.addItem(cursor)
        self._update_cursor_size()

    def _update_cursor_size(self) -> None:
        """Resize the brush cursor's rings and crosshair to match brush_size.

        Crosshair length scales with the brush radius so it stays visible
        for both tiny and huge brushes — but capped at 8px so it doesn't
        dominate the inside of large brushes.
        """
        r = self._brush_size / 2
        # Outer ring
        self._cursor_outer.setRect(QRectF(-r, -r, self._brush_size, self._brush_size))
        # Inner ring sits 1.5px inside outer so the two rings read as
        # distinct stripes rather than one fat line.
        inner_r = max(r - 1.5, 0.5)
        self._cursor_inner.setRect(
            QRectF(-inner_r, -inner_r, inner_r * 2, inner_r * 2)
        )
        # Crosshair — 4 ticks pointing inward from a small gap. Always 4–8px.
        tick = max(3.0, min(r * 0.25, 8.0))
        gap = max(1.5, tick * 0.4)
        # up, down, left, right
        coords = [
            (0, -tick - gap, 0, -gap),
            (0, gap, 0, tick + gap),
            (-tick - gap, 0, -gap, 0),
            (gap, 0, tick + gap, 0),
        ]
        for line_outer, line_inner, (x1, y1, x2, y2) in zip(
            self._cursor_cross_outer, self._cursor_cross_inner, coords,
        ):
            line_outer.setLine(x1, y1, x2, y2)
            line_inner.setLine(x1, y1, x2, y2)

    def _zoom(self, factor: float) -> None:
        """Apply zoom factor to the view."""
        new_zoom = self._zoom_level * factor
        new_zoom = max(0.1, min(new_zoom, 20.0))
        actual_factor = new_zoom / self._zoom_level
        self._zoom_level = new_zoom
        self._view.scale(actual_factor, actual_factor)
        self._zoom_label.setText(f"{int(self._zoom_level * 100)}%")

    def _fit_view(self) -> None:
        """Reset zoom to fit source image in view."""
        if self._source_item:
            self._view.resetTransform()
            self._zoom_level = 1.0
            self._view.fitInView(self._source_item, Qt.KeepAspectRatio)
            # Calculate actual zoom level from the transform
            t = self._view.transform()
            self._zoom_level = t.m11()
            self._zoom_label.setText(f"{int(self._zoom_level * 100)}%")

    def load_image(self, path: str) -> None:
        """Load source image for mask painting."""
        self._source_path = path
        self._scene.clear()
        self._source_item = None
        self._mask_item = None

        pixmap = QPixmap(path)
        if pixmap.isNull():
            return

        self._source_item = self._scene.addPixmap(pixmap)
        self._view.resetTransform()
        self._zoom_level = 1.0
        self._view.fitInView(self._source_item, Qt.KeepAspectRatio)
        t = self._view.transform()
        self._zoom_level = t.m11()
        self._zoom_label.setText(f"{int(self._zoom_level * 100)}%")
        self._view.setStyleSheet("QGraphicsView { background: #1a1a1a; border: 1px solid #555; }")
        self._clear_img_btn.setVisible(True)
        if self._send_targets or self._final_frame_targets or self._guide_video_targets:
            self._menu_btn.setVisible(True)

        # Preserve existing mask (scaled to new size) or create fresh one
        old_mask = self._mask_image
        self._mask_image = QImage(pixmap.size(), QImage.Format_ARGB32)
        if old_mask is not None and not old_mask.isNull():
            scaled = old_mask.scaled(
                pixmap.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
            self._mask_image = scaled.convertToFormat(QImage.Format_ARGB32)
        else:
            self._mask_image.fill(QColor(0, 0, 0, 0))
        self._mask_item = self._scene.addPixmap(QPixmap.fromImage(self._mask_image))
        self._mask_item.setOpacity(0.5)

        # Re-add brush cursor on top (load_image clears the scene)
        self._build_brush_cursor()

    def clear_mask(self) -> None:
        if self._mask_image:
            self._mask_image.fill(QColor(0, 0, 0, 0))
            self._update_mask_display()

    def save_mask_to_file(self, path: str) -> bool:
        """Save the raw mask overlay (ARGB32) to a PNG file for persistence."""
        if self._mask_image is None:
            return False
        return self._mask_image.save(path, "PNG")

    def load_mask_from_file(self, path: str) -> bool:
        """Restore a previously saved mask overlay from a PNG file."""
        if self._mask_image is None:
            return False
        if not Path(path).is_file():
            return False
        saved = QImage(path)
        if saved.isNull():
            return False
        # Resize to match current mask dimensions if needed
        if saved.size() != self._mask_image.size():
            saved = saved.scaled(
                self._mask_image.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
        saved = saved.convertToFormat(QImage.Format_ARGB32)
        self._mask_image = saved
        self._update_mask_display()
        return True

    def apply_mask_from_image(self, mask_path: str) -> None:
        """Import a binary mask from an external source (white=masked).

        Converts the grayscale mask to the ARGB32 overlay format used by
        the mask editor (semi-transparent red for painted areas).
        """
        if self._mask_image is None:
            return
        mask_img = QImage(mask_path)
        if mask_img.isNull():
            return
        # Resize to match current mask dimensions
        if mask_img.size() != self._mask_image.size():
            mask_img = mask_img.scaled(
                self._mask_image.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation,
            )
        # Convert to grayscale for consistent handling
        mask_img = mask_img.convertToFormat(QImage.Format_Grayscale8)
        w, h = mask_img.width(), mask_img.height()

        # Convert grayscale mask to ARGB32 red overlay. The mask brightness
        # lives in the RGB channels, so a plain convertToFormat(ARGB32) leaves
        # alpha=255 everywhere and DestinationIn would keep red at every pixel
        # (whole image masked). Instead drive the overlay's *alpha* directly
        # from the grayscale mask so red only appears where the mask is bright.
        red_overlay = QImage(w, h, QImage.Format_ARGB32)
        red_overlay.fill(QColor(255, 0, 0, 255))
        # setAlphaChannel reads the grayscale image's intensity as the alpha,
        # giving partial transparency for soft mask edges.
        red_overlay.setAlphaChannel(mask_img)
        self._mask_image = red_overlay

        self._update_mask_display()

    def _clear_all(self) -> None:
        """Clear image and mask."""
        self._source_path = None
        self._scene.clear()
        self._source_item = None
        self._mask_item = None
        self._mask_image = None
        self._clear_img_btn.setVisible(False)
        self._menu_btn.setVisible(False)
        self._view.resetTransform()
        self._zoom_level = 1.0
        self._zoom_label.setText("100%")
        self._view.setStyleSheet("QGraphicsView { background: #1a1a1a; border: 2px dashed #444; }")
        # Re-add cursor on top after scene clear
        self._build_brush_cursor()

    @property
    def image_path(self) -> str | None:
        return self._source_path

    def get_mask_path(self) -> str | None:
        """Save current mask to temp file and return path.

        Uses QPainter compositing instead of per-pixel loop for performance.
        """
        if self._mask_image is None:
            return None

        w, h = self._mask_image.width(), self._mask_image.height()
        # Create white image, use DestinationIn to keep only where mask has alpha
        white = QImage(w, h, QImage.Format_ARGB32)
        white.fill(QColor(255, 255, 255, 255))
        p = QPainter(white)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        p.drawImage(0, 0, self._mask_image)
        p.end()
        mask = white.convertToFormat(QImage.Format_Grayscale8)

        # Fast empty check
        bits = mask.constBits()
        if bits is None or not any(bytes(bits)):
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        mask.save(tmp.name)
        tmp.close()
        self.mask_updated.emit(tmp.name)
        return tmp.name

    def _on_mode_toggled(self, checked: bool) -> None:
        self._erase_mode = checked
        self._mode_btn.setText("Erase" if checked else "Paint")
        # Recolor the inner (light) ring to signal mode: white for paint,
        # bright red for erase. The outer black ring stays put — it provides
        # contrast against the canvas regardless of mode.
        color = QColor(255, 60, 60, 230) if checked else QColor(255, 255, 255, 230)
        self._cursor_inner.setPen(QPen(color, 1.5))
        # Also recolor the inner crosshair lines to match.
        cross_pen = QPen(color, 1.0)
        for line in self._cursor_cross_inner:
            line.setPen(cross_pen)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_E and not event.modifiers():
            self._mode_btn.toggle()
            return
        super().keyPressEvent(event)

    def _on_brush_change(self, val: int) -> None:
        self._brush_size = val
        self._brush_label.setText(str(val))
        self._update_cursor_size()

    def _on_hardness_change(self, val: int) -> None:
        self._brush_hardness = val

    @property
    def brush_hardness(self) -> int:
        """Current brush hardness percentage (0-100, 100 = hard edge)."""
        return self._brush_hardness

    def set_brush_hardness(self, val: int) -> None:
        """Set the brush hardness percentage (0-100, 100 = hard edge)."""
        val = max(0, min(int(val), 100))
        self._brush_hardness = val
        if self._hardness_spin.value() != val:
            self._hardness_spin.setValue(val)

    def _invert_mask(self) -> None:
        if self._mask_image is None:
            return
        w, h = self._mask_image.width(), self._mask_image.height()

        # Vectorised invert: painted (alpha>0) becomes clear, clear becomes
        # painted at alpha 180 (the same colour/strength _paint_at uses). The
        # old per-pixel Python double loop froze the UI for seconds on large
        # masks. Build an inverted binary alpha channel via numpy on the raw
        # alpha bytes so the binary edges stay crisp.
        import numpy as np

        cur = self._mask_image.convertToFormat(QImage.Format_ARGB32)
        alpha = cur.convertToFormat(QImage.Format_Alpha8)
        ptr = alpha.constBits()
        stride = alpha.bytesPerLine()
        buf = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(h, stride)[:, :w]
        # alpha>0 -> 0 (clear); alpha==0 -> 180 (painted).
        out = np.where(buf > 0, np.uint8(0), np.uint8(180))
        out = np.ascontiguousarray(out)

        new_alpha = QImage(out.data, w, h, w, QImage.Format_Grayscale8).copy()
        inverted = QImage(w, h, QImage.Format_ARGB32)
        inverted.fill(QColor(255, 255, 255, 180))
        inverted.setAlphaChannel(new_alpha)
        self._mask_image = inverted
        self._update_mask_display()

    def _paint_at(self, pos: QPointF, erase: bool = False) -> None:
        if self._mask_image is None:
            return
        radius = self._brush_size / 2
        painter = QPainter(self._mask_image)
        painter.setPen(Qt.NoPen)
        # The painted overlay colour: semi-transparent red so painted areas read
        # as masked. get_mask_path() reads the *alpha* channel, so the alpha
        # ramp below is what feathers the exported mask edge.
        full_alpha = 180

        if self._brush_hardness >= 100:
            # Hard edge — original solid brush behaviour.
            if erase:
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.setBrush(QBrush(QColor(0, 0, 0, 0)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255, full_alpha)))
            painter.drawEllipse(pos, radius, radius)
        else:
            # Soft edge — radial alpha falloff. The inner `hardness` fraction of
            # the radius stays at full strength, then ramps to zero alpha at the
            # rim. Erase uses Clear composition so the gradient alpha controls
            # how much is cleared (soft erase mirrors soft paint).
            grad = QRadialGradient(pos, radius)
            inner = max(0.0, min(self._brush_hardness / 100.0, 1.0))
            if erase:
                # DestinationOut removes destination alpha in proportion to the
                # source alpha, so the gradient gives a soft (feathered) erase.
                # Clear (used for the hard brush) ignores source alpha and would
                # wipe the whole footprint, defeating the falloff.
                painter.setCompositionMode(
                    QPainter.CompositionMode_DestinationOut
                )
                # Full alpha at centre so the core is fully erased; ramps to 0
                # at the rim for the feathered edge.
                base = QColor(0, 0, 0, 255)
                edge = QColor(0, 0, 0, 0)
            else:
                base = QColor(255, 255, 255, full_alpha)
                edge = QColor(255, 255, 255, 0)
            grad.setColorAt(0.0, base)
            if inner > 0.0:
                grad.setColorAt(inner, base)
            grad.setColorAt(1.0, edge)
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(pos, radius, radius)
        painter.end()
        self._update_mask_display()

    def _update_mask_display(self) -> None:
        if self._mask_item and self._mask_image:
            self._mask_item.setPixmap(QPixmap.fromImage(self._mask_image))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._source_item:
            self._fit_view()

    # -- Context menu targets -----------------------------------------------

    def set_send_targets(self, targets: list[tuple[str, str]]) -> None:
        self._send_targets = targets

    def set_final_frame_targets(self, targets: list[tuple[str, str]]) -> None:
        self._final_frame_targets = targets

    def set_guide_video_targets(self, targets: list[tuple[str, str]]) -> None:
        self._guide_video_targets = targets

    def _on_menu_click(self) -> None:
        pos = self._menu_btn.mapToGlobal(self._menu_btn.rect().bottomLeft())
        self._show_context_menu_at(pos)

    def _show_context_menu_at(self, global_pos) -> None:
        if not self._source_path:
            return

        from sdqt.widgets.send_targets import build_target_menu

        menu = QMenu(self)

        if self._send_targets:
            build_target_menu(menu, "Send to", self._send_targets,
                              self.send_requested.emit)

        if self._final_frame_targets:
            build_target_menu(menu, "Final Frame", self._final_frame_targets,
                              self.final_frame_requested.emit)

        if self._guide_video_targets:
            build_target_menu(menu, "Guide Video", self._guide_video_targets,
                              self.guide_video_requested.emit)

        if not menu.isEmpty():
            menu.exec(global_pos)

    # -- Drag/drop ---------------------------------------------------------

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
                self.load_image(path)
                self.image_loaded.emit(path)
                event.acceptProposedAction()
                return

    # -- Browse ------------------------------------------------------------

    def _browse_image(self) -> None:
        path, _ = get_open_filename(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
        )
        if path:
            self.load_image(path)
            self.image_loaded.emit(path)

    # -- Event filter for painting + zoom + pan ----------------------------

    def eventFilter(self, obj, event) -> bool:
        if obj is not self._view.viewport():
            return super().eventFilter(obj, event)

        from PySide6.QtCore import QEvent

        # Forward drag/drop from viewport to our handlers
        if event.type() == QEvent.DragEnter:
            self.dragEnterEvent(event)
            return event.isAccepted()
        elif event.type() == QEvent.DragMove:
            self.dragMoveEvent(event)
            return event.isAccepted()
        elif event.type() == QEvent.Drop:
            self.dropEvent(event)
            return event.isAccepted()

        # Scroll wheel zoom
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta > 0:
                self._zoom(1.15)
            elif delta < 0:
                self._zoom(1.0 / 1.15)
            return True

        # Mouse tracking for brush cursor
        if event.type() == QEvent.MouseMove and self._mask_image is not None:
            scene_pos = self._view.mapToScene(event.pos())
            self._cursor_item.setPos(scene_pos)
            if not self._cursor_item.isVisible():
                self._cursor_item.setVisible(True)

        # Middle-click pan
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self._view.setCursor(QCursor(Qt.ClosedHandCursor))
            return True
        if event.type() == QEvent.MouseMove and self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            hs = self._view.horizontalScrollBar()
            vs = self._view.verticalScrollBar()
            hs.setValue(hs.value() - int(delta.x()))
            vs.setValue(vs.value() - int(delta.y()))
            return True
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.MiddleButton:
            self._panning = False
            self._view.setCursor(QCursor(Qt.ArrowCursor))
            return True

        # Hide cursor when leaving
        if event.type() == QEvent.Leave:
            self._cursor_item.setVisible(False)

        # Painting / erasing
        # LMB uses current mode (paint/erase toggle), RMB uses opposite
        if self._mask_image is not None:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    erase = self._erase_mode
                    self._painting = not erase
                    self._erasing = erase
                    scene_pos = self._view.mapToScene(event.pos())
                    self._paint_at(scene_pos, erase=erase)
                    self._last_point = scene_pos
                    return True
                elif event.button() == Qt.RightButton:
                    erase = not self._erase_mode
                    self._painting = not erase
                    self._erasing = erase
                    scene_pos = self._view.mapToScene(event.pos())
                    self._paint_at(scene_pos, erase=erase)
                    self._last_point = scene_pos
                    return True
            elif event.type() == QEvent.MouseMove:
                if self._painting or self._erasing:
                    scene_pos = self._view.mapToScene(event.pos())
                    self._paint_at(scene_pos, erase=self._erasing)
                    self._last_point = scene_pos
                    return True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton and (self._painting or self._erasing):
                    self._painting = False
                    self._erasing = False
                    return True
                elif event.button() == Qt.RightButton and (self._painting or self._erasing):
                    self._painting = False
                    self._erasing = False
                    return True
        return super().eventFilter(obj, event)
