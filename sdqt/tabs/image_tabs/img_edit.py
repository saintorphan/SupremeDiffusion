"""Draw sub-tab for Image Suite -- layer-based image editor."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QSize, Slot, Signal
from PySide6.QtGui import QColor, QIcon, QImage
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QFileDialog,
    QSlider,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sdqt.tabs.base import BaseTab
from sdqt.widgets.draw.canvas import DrawCanvas
from sdqt.widgets.draw.file_io import load_sdraw, save_sdraw
from sdqt.widgets.draw.icons import make_tool_icon
from sdqt.widgets.draw.layer_panel import LayerPanel
from sdqt.widgets.draw.png_library import PNGLibrary
from sdqt.widgets.draw.styles import BTN_STYLE, SEND_BTN_STYLE, TOGGLE_STYLE, TOOL_BTN_STYLE, TOOL_SZ
from sdqt.widgets.draw.tool_settings import DrawToolSettingsPanel
from sdqt.widgets.draw.tools import (
    BrushTool,
    EllipseTool,
    EraserTool,
    EyedropperTool,
    FloodFillTool,
    FreehandSelectTool,
    GrabTool,
    GradientTool,
    LineTool,
    MagicWandTool,
    MagneticLassoTool,
    PolygonLassoTool,
    RectangleTool,
    SmudgeTool,
    TextTool,
)
from sdqt.widgets.draw.transform_handles import TransformTool
from sdqt.widgets.image_gallery import ImageGalleryWidget

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# Tool key -> settings panel category
_CATEGORY = {
    "brush": "brush", "eraser": "eraser", "grab": "none",
    "line": "shape", "ellipse": "shape", "rect": "shape",
    "eyedropper": "eyedropper", "fill": "fill",
    "smudge": "smudge", "text": "text", "gradient": "gradient",
    "freehand": "none", "lasso": "none", "magnetic": "none",
    "wand": "wand", "transform": "none",
}


class ImgEditTab(BaseTab):
    """Layer-based Draw editor -- canvas, toolbar, layer panel, PNG library."""

    send_flattened_requested = Signal(str)  # emits temp file path of flattened image

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._source_path: str | None = None
        self._png_lib_visible = False
        self._previous_tool_key: str = "brush"
        self._build_ui()
        self._create_tools()
        self._connect_signals()
        self._set_tool("brush")

    # ── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # ── Row 1: Tool buttons | Undo/Redo | Zoom | File/Send ───────────────
        row1 = QHBoxLayout()
        row1.setContentsMargins(2, 2, 2, 0)
        row1.setSpacing(2)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QToolButton] = {}

        tool_defs = [
            ("brush", "Brush"),
            ("eraser", "Eraser"),
            ("grab", "Grab"),
            ("|", None),
            ("line", "Line"),
            ("ellipse", "Ellipse"),
            ("rect", "Rectangle"),
            ("|", None),
            ("eyedropper", "Eyedropper"),
            ("fill", "Flood Fill"),
            ("smudge", "Smudge"),
            ("|", None),
            ("text", "Text"),
            ("gradient", "Gradient"),
            ("|", None),
            ("freehand", "Freehand Select"),
            ("lasso", "Polygon Lasso"),
            ("magnetic", "Magnetic Lasso"),
            ("wand", "Magic Wand"),
            ("|", None),
            ("transform", "Transform"),
        ]

        for key, label in tool_defs:
            if key == "|":
                sep = QLabel(" ")
                sep.setFixedWidth(4)
                row1.addWidget(sep)
                continue
            btn = QToolButton()
            btn.setIcon(QIcon(make_tool_icon(key)))
            btn.setIconSize(QSize(24, 24))
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setFixedSize(TOOL_SZ, TOOL_SZ)
            btn.setStyleSheet(TOOL_BTN_STYLE)
            self._tool_group.addButton(btn)
            self._tool_buttons[key] = btn
            btn.clicked.connect(lambda checked, k=key: self._set_tool(k))
            row1.addWidget(btn)

        row1.addWidget(QLabel("  "))

        # Undo / Redo
        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setStyleSheet(BTN_STYLE)
        self._btn_undo.setToolTip("Undo (Ctrl+Z)")
        self._btn_undo.clicked.connect(lambda: self._canvas.undo())
        row1.addWidget(self._btn_undo)

        self._btn_redo = QPushButton("Redo")
        self._btn_redo.setStyleSheet(BTN_STYLE)
        self._btn_redo.setToolTip("Redo (Ctrl+Y)")
        self._btn_redo.clicked.connect(lambda: self._canvas.redo())
        row1.addWidget(self._btn_redo)

        row1.addWidget(QLabel("  "))

        # Zoom controls
        btn_fit = QPushButton("Fit")
        btn_fit.setStyleSheet(BTN_STYLE)
        btn_fit.setToolTip("Fit canvas in view")
        btn_fit.clicked.connect(lambda: self._canvas.fit_view())
        row1.addWidget(btn_fit)

        btn_100 = QPushButton("100%")
        btn_100.setStyleSheet(BTN_STYLE)
        btn_100.setToolTip("Zoom to 1:1")
        btn_100.clicked.connect(lambda: self._canvas.zoom_to_100())
        row1.addWidget(btn_100)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 800)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(120)
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        row1.addWidget(self._zoom_slider)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(40)
        row1.addWidget(self._zoom_label)

        btn_flip = QPushButton("Flip")
        btn_flip.setStyleSheet(BTN_STYLE)
        btn_flip.setToolTip("Mirror canvas view (M)")
        btn_flip.clicked.connect(lambda: self._canvas.toggle_flip_view())
        row1.addWidget(btn_flip)

        row1.addStretch()

        # File / send buttons
        self._btn_save_sdraw = QPushButton("Save .sdraw")
        self._btn_save_sdraw.setStyleSheet(BTN_STYLE)
        self._btn_save_sdraw.clicked.connect(self._on_save_sdraw)
        row1.addWidget(self._btn_save_sdraw)

        self._btn_load_sdraw = QPushButton("Open")
        self._btn_load_sdraw.setStyleSheet(BTN_STYLE)
        self._btn_load_sdraw.clicked.connect(self._on_load_sdraw)
        row1.addWidget(self._btn_load_sdraw)

        self._btn_save_flat = QPushButton("Save Flat")
        self._btn_save_flat.setStyleSheet(BTN_STYLE)
        self._btn_save_flat.clicked.connect(self._on_save_flat)
        row1.addWidget(self._btn_save_flat)

        self._btn_send = QPushButton("Send Flat \u25bc")
        self._btn_send.setStyleSheet(SEND_BTN_STYLE)
        self._btn_send.setToolTip("Send all layers merged into one image")
        self._btn_send.clicked.connect(self._show_send_menu)
        row1.addWidget(self._btn_send)

        self._btn_send_layer = QPushButton("Send Layer \u25bc")
        self._btn_send_layer.setStyleSheet(SEND_BTN_STYLE)
        self._btn_send_layer.setToolTip("Send the active layer (with position baked in)")
        self._btn_send_layer.clicked.connect(self._show_send_layer_menu)
        row1.addWidget(self._btn_send_layer)

        self._btn_png_lib = QToolButton()
        self._btn_png_lib.setText("PNG Lib")
        self._btn_png_lib.setCheckable(True)
        self._btn_png_lib.setStyleSheet(TOGGLE_STYLE)
        self._btn_png_lib.clicked.connect(self._toggle_png_library)
        row1.addWidget(self._btn_png_lib)

        outer.addLayout(row1)

        # ── Row 2: DrawToolSettingsPanel ──────────────────────────────────────
        self._tool_settings = DrawToolSettingsPanel()
        outer.addWidget(self._tool_settings)

        # ── Main content area (splitter) ─────────────────────────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._canvas = DrawCanvas()
        self._splitter.addWidget(self._canvas)

        self._layer_panel = LayerPanel(self._canvas.layer_stack)
        self._splitter.addWidget(self._layer_panel)

        self._png_library = PNGLibrary()
        self._png_library.setVisible(False)
        self._splitter.addWidget(self._png_library)

        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setStretchFactor(2, 0)

        outer.addWidget(self._splitter, 1)

        # Status bar
        status_row = QHBoxLayout()
        status_row.setContentsMargins(4, 0, 4, 2)
        self._status = QLabel("")
        status_row.addWidget(self._status)
        status_row.addStretch()
        outer.addLayout(status_row)

    # ── Tools ────────────────────────────────────────────────────────────────

    def _create_tools(self) -> None:
        c = self._canvas
        self._tools: dict[str, object] = {
            "brush": BrushTool(c),
            "eraser": EraserTool(c),
            "grab": GrabTool(c),
            "line": LineTool(c),
            "ellipse": EllipseTool(c),
            "rect": RectangleTool(c),
            "eyedropper": EyedropperTool(c),
            "fill": FloodFillTool(c),
            "smudge": SmudgeTool(c),
            "text": TextTool(c),
            "gradient": GradientTool(c),
            "freehand": FreehandSelectTool(c),
            "lasso": PolygonLassoTool(c),
            "magnetic": MagneticLassoTool(c),
            "wand": MagicWandTool(c),
            "transform": TransformTool(c),
        }
        self._tools["eyedropper"].on_color_sampled = self._on_eyedropper_sample
        self._current_tool_key = "brush"

    def _set_tool(self, key: str) -> None:
        # Deactivate transform if switching away
        if self._current_tool_key == "transform" and key != "transform":
            tool = self._tools.get("transform")
            if tool and hasattr(tool, "deactivate"):
                tool.deactivate(confirm=True)

        if key != "eyedropper":
            self._previous_tool_key = self._current_tool_key

        self._current_tool_key = key
        tool = self._tools.get(key)
        if tool is None:
            return

        if key == "transform":
            self._canvas.set_tool(tool)
            tool.activate()
        else:
            self._canvas.set_tool(tool)

        btn = self._tool_buttons.get(key)
        if btn:
            btn.setChecked(True)

        self._tool_settings.show_for_tool(_CATEGORY[key])
        self._tool_settings.sync_to_tool(tool)

        # Update cursor size from panel
        if hasattr(tool, "size"):
            self._canvas.update_cursor_size(self._tool_settings.brush_size)

        self._show_status(f"Tool: {key}")

    # ── Signal connections ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Tool settings panel -> tool sync
        self._tool_settings.brush_size_changed.connect(self._on_brush_size)
        self._tool_settings.opacity_changed.connect(self._on_opacity_changed)
        self._tool_settings.primary_color_changed.connect(self._on_color_changed)
        self._tool_settings.secondary_color_changed.connect(self._on_color_changed)
        self._tool_settings.fill_color_changed.connect(self._on_color_changed)
        self._tool_settings.tolerance_changed.connect(self._on_settings_changed)

        # Zoom sync
        self._canvas.zoom_changed.connect(self._on_zoom_changed)

        # Layer panel
        self._layer_panel._btn_new.clicked.disconnect()
        self._layer_panel._btn_new.clicked.connect(lambda: self._canvas.new_empty_layer("Layer"))
        self._layer_panel._tree.itemDoubleClicked.connect(self._on_layer_double_click)
        self._layer_panel.save_to_library_requested.connect(self._on_save_layer_to_library)
        self._layer_panel.save_to_face_library_requested.connect(self._on_save_layer_to_face_library)

        # PNG library
        self._png_library.add_to_canvas_requested = self._add_png_to_canvas

        # Keyboard shortcuts
        from sdqt.widgets.draw.shortcuts import setup_canvas_shortcuts
        setup_canvas_shortcuts(
            self, self._canvas,
            paste_callback=self._paste_and_transform,
        )

    # ── Tool callbacks ───────────────────────────────────────────────────────

    @Slot(int)
    def _on_brush_size(self, val: int) -> None:
        tool = self._tools.get(self._current_tool_key)
        if tool and hasattr(tool, "size"):
            tool.size = val
        self._canvas.update_cursor_size(val)

    @Slot(int)
    def _on_opacity_changed(self, val: int) -> None:
        tool = self._tools.get(self._current_tool_key)
        if tool and hasattr(tool, "opacity"):
            tool.opacity = val / 100.0

    @Slot()
    def _on_color_changed(self, _color=None) -> None:
        self._tool_settings.sync_to_tool(self._tools.get(self._current_tool_key))

    @Slot()
    def _on_settings_changed(self, _val=None) -> None:
        self._tool_settings.sync_to_tool(self._tools.get(self._current_tool_key))

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def _on_zoom_slider(self, val: int) -> None:
        self._canvas.set_zoom(float(val))

    def _on_zoom_changed(self) -> None:
        z = self._canvas.zoom_level
        self._zoom_label.setText(f"{z:.0f}%")
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(max(10, min(800, int(z))))
        self._zoom_slider.blockSignals(False)

    # ── Eyedropper callback ──────────────────────────────────────────────────

    def _on_eyedropper_sample(self, color: QColor) -> None:
        self._tool_settings.set_primary_color(color)
        self._set_tool(self._previous_tool_key)

    # ── Image loading ────────────────────────────────────────────────────────

    def load_source(self, path: str) -> None:
        self._source_path = path
        self._canvas.load_background(path)
        self._canvas.new_empty_layer("Layer 1")
        self._show_status(f"Loaded: {Path(path).name}")

    # ── Paste + auto-transform ───────────────────────────────────────────────

    def _paste_and_transform(self) -> None:
        layer = self._canvas.paste_clipboard()
        if layer is not None:
            self._set_tool("transform")

    # ── Selection operations ─────────────────────────────────────────────────

    def _select_all(self) -> None:
        rect = self._canvas.scene().sceneRect()
        self._canvas.selection.select_all(rect)

    def _deselect(self) -> None:
        self._canvas.selection.clear()

    def _invert_selection(self) -> None:
        rect = self._canvas.scene().sceneRect()
        self._canvas.selection.invert(rect)

    # ── Layer visibility toggle ──────────────────────────────────────────────

    def _on_layer_double_click(self, item, column) -> None:
        uid = item.data(0, Qt.ItemDataRole.UserRole)
        if uid:
            self._layer_panel.toggle_visibility(uid)

    # ── PNG Library ──────────────────────────────────────────────────────────

    @Slot()
    def _toggle_png_library(self) -> None:
        self._png_lib_visible = not self._png_lib_visible
        self._png_library.setVisible(self._png_lib_visible)
        if self._png_lib_visible:
            try:
                lib_dir = self.state.global_config.png_library_dir
                if lib_dir:
                    self._png_library.set_root_dir(lib_dir)
            except AttributeError:
                pass

    @Slot(str)
    def _on_save_layer_to_library(self, uid: str) -> None:
        layer = self._canvas.layer_stack.get(uid)
        if layer is None:
            return
        try:
            lib_dir = self.state.global_config.png_library_dir
        except AttributeError:
            lib_dir = ""
        if not lib_dir:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "PNG Library", "No library directory configured.\nSet path in Settings.")
            return
        Path(lib_dir).mkdir(parents=True, exist_ok=True)
        self._png_library.set_root_dir(lib_dir)
        self._png_library.save_layer_image(layer.image, layer.name)
        self._show_status(f"Saved layer '{layer.name}' to PNG Library")

    @Slot(str)
    def _on_save_layer_to_face_library(self, uid: str) -> None:
        from PySide6.QtWidgets import QInputDialog, QMessageBox

        layer = self._canvas.layer_stack.get(uid)
        if layer is None:
            return
        try:
            face_dir = self.state.global_config.face_library_dir
        except AttributeError:
            face_dir = ""
        if not face_dir:
            QMessageBox.warning(self, "Face Library", "No face library directory configured.\nSet path in Settings.")
            return
        face_path = Path(face_dir)
        face_path.mkdir(parents=True, exist_ok=True)

        name, ok = QInputDialog.getText(
            self, "Save to Face Library", "Face name (without .png):",
            text=layer.name,
        )
        if not ok or not name.strip():
            return
        out = face_path / f"{name.strip()}.png"
        if out.exists():
            ret = QMessageBox.question(
                self, "Overwrite?", f"{out.name} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
        layer.image.save(str(out), "PNG")
        self._show_status(f"Saved layer '{layer.name}' to Face Library")

    def _add_png_to_canvas(self, path: str) -> None:
        img = QImage(path)
        if not img.isNull():
            self._canvas.add_image_layer(img, Path(path).stem)

    # ── File I/O ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_save_sdraw(self) -> None:
        draw_dir = self.project_path / "draw"
        draw_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(draw_dir / f"drawing_{ts}.sdraw")

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Drawing", default_path, "Sdraw Files (*.sdraw)",
        )
        if path:
            save_sdraw(path, self._canvas.layer_stack, canvas_size=self._canvas.canvas_size)
            self._show_status(f"Saved: {Path(path).name}")

    @Slot()
    def _on_load_sdraw(self) -> None:
        draw_dir = self.project_path / "draw"
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Drawing", str(draw_dir), "Sdraw Files (*.sdraw)",
        )
        if path:
            w, h = load_sdraw(path, self._canvas.layer_stack)
            self._canvas.canvas_size = (w, h)
            self._canvas.scene().setSceneRect(QRectF(0, 0, w, h))
            self._canvas.fit_view()
            self._show_status(f"Opened: {Path(path).name}")

    @Slot()
    def _on_save_flat(self) -> None:
        flat = self._canvas.flatten()
        if flat.isNull():
            self._show_status("Nothing to save.")
            return
        images_dir = self.project_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = images_dir / f"draw_{ts}.png"
        flat.save(str(out_path))
        self._show_status(f"Saved: {out_path.name}")

    # ── Send Flattened ───────────────────────────────────────────────────────

    @Slot()
    def _show_send_menu(self) -> None:
        from sdqt.widgets.send_targets import IMAGE_TARGETS_SIMPLE, _populate_menu
        menu = QMenu(self)
        _populate_menu(menu, IMAGE_TARGETS_SIMPLE, self._do_send_flattened)
        menu.exec(self._btn_send.mapToGlobal(self._btn_send.rect().bottomLeft()))

    def _do_send_flattened(self, target: str) -> None:
        flat = self._canvas.flatten()
        if flat.isNull():
            self._show_status("Nothing to send.")
            return
        self._send_image(flat, target, "flattened")

    # ── Send Active Layer ────────────────────────────────────────────────────

    @Slot()
    def _show_send_layer_menu(self) -> None:
        from sdqt.widgets.send_targets import IMAGE_TARGETS_SIMPLE, _populate_menu
        menu = QMenu(self)
        _populate_menu(menu, IMAGE_TARGETS_SIMPLE, self._do_send_layer)
        menu.exec(self._btn_send_layer.mapToGlobal(self._btn_send_layer.rect().bottomLeft()))

    def _do_send_layer(self, target: str) -> None:
        img = self._canvas.render_active_layer()
        if img.isNull():
            self._show_status("No active layer to send.")
            return
        self._send_image(img, target, "layer")

    # ── Shared send helper ───────────────────────────────────────────────────

    def _send_image(self, image, target: str, label: str) -> None:
        import tempfile as _tf
        tmp = _tf.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp.name)
        tmp.close()
        self.send_flattened_requested.emit(target + "|" + tmp.name)
        self._show_status(f"Sent {label} to {target}")

    # ── Drag & drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                suffix = Path(url.toLocalFile()).suffix.lower()
                if suffix in _IMAGE_EXTS or suffix == ".sdraw":
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            suffix = Path(path).suffix.lower()
            if suffix == ".sdraw":
                w, h = load_sdraw(path, self._canvas.layer_stack)
                self._canvas.canvas_size = (w, h)
                self._canvas.scene().setSceneRect(QRectF(0, 0, w, h))
                self._canvas.fit_view()
                event.acceptProposedAction()
                return
            elif suffix in _IMAGE_EXTS:
                if len(self._canvas.layer_stack) == 0:
                    self._canvas.load_background(path)
                else:
                    img = QImage(path)
                    if not img.isNull():
                        self._canvas.add_image_layer(img, Path(path).stem)
                event.acceptProposedAction()
                return

    # ── Workspace persistence ────────────────────────────────────────────────

    @property
    def _workspace_path(self) -> Path:
        return self.project_path / "draw" / "workspace.sdraw"

    def save_workspace(self) -> None:
        if len(self._canvas.layer_stack) == 0:
            ws = self._workspace_path
            if ws.is_file():
                ws.unlink(missing_ok=True)
            return
        ws = self._workspace_path
        ws.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_sdraw(
                str(ws),
                self._canvas.layer_stack,
                canvas_size=self._canvas.canvas_size,
            )
            logger.debug("Draw workspace saved: %s", ws)
        except Exception as exc:
            logger.warning("Failed to save draw workspace: %s", exc)

    def load_workspace(self) -> None:
        ws = self._workspace_path
        if not ws.is_file():
            return
        try:
            w, h = load_sdraw(str(ws), self._canvas.layer_stack)
            self._canvas.canvas_size = (w, h)
            self._canvas.scene().setSceneRect(QRectF(0, 0, w, h))
            self._canvas.fit_view()
            self._show_status("Workspace restored")
            logger.debug("Draw workspace loaded: %s", ws)
        except Exception as exc:
            logger.warning("Failed to load draw workspace: %s", exc)

    # ── Project change ───────────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        self.save_workspace()
        super().on_project_changed(project_name)
        self._canvas.layer_stack.clear()
        self._canvas.canvas_size = (1024, 1024)
        self._canvas.scene().setSceneRect(QRectF(0, 0, 1024, 1024))
        self._show_status("")
        self.load_workspace()
        self._set_tool(self._current_tool_key)
