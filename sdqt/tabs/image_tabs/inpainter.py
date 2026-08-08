"""Inpainter sub-tab — dual-toolbar Draw editor with mask + paint tools."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QRectF, QSize, QTimer, Slot, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.tabs.base import BaseTab
from sdqt.tabs.image_tabs.model_family import SDXL_STRATEGY
from sdqt.widgets.draw.canvas import DrawCanvas
from sdqt.widgets.draw.icons import make_tool_icon
from sdqt.widgets.draw.layer_panel import LayerPanel
from sdqt.widgets.draw.png_library import PNGLibrary
from sdqt.widgets.draw.styles import (
    BTN_STYLE,
    MASK_BTN_STYLE,
    MASK_COLOR,
    ROW_LABEL_STYLE,
    TOGGLE_STYLE,
    TOOL_BTN_STYLE,
    TOOL_SZ,
)
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
)
from sdqt.widgets.draw.transform_handles import TransformTool
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.image_params import ImageParamsWidget
from sdqt.widgets.prompt_enhance import PromptEnhanceWidget
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

logger = logging.getLogger(__name__)

# Paint tool key -> settings panel category
_CATEGORY = {
    "brush": "brush", "eraser": "eraser", "grab": "none",
    "line": "shape", "ellipse": "shape", "rect": "shape",
    "eyedropper": "eyedropper", "fill": "fill",
    "smudge": "smudge", "gradient": "gradient",
    "freehand": "none", "lasso": "none", "magnetic": "none",
    "wand": "wand", "transform": "none",
}


class InpainterTab(BaseTab):
    """Inpainting tab with dual-toolbar: Mask Tools (row 1) + Paint Tools (row 2).

    Mask tools auto-target the Mask layer with semi-transparent red.
    Paint tools work on the active (non-mask) layer for compositing edits.
    On Generate, non-mask layers are flattened and the mask is extracted.
    """

    send_flattened_requested = Signal(str)

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._source_path: str | None = None
        self._mask_layer_uid: str | None = None
        self._worker = None
        self._gen_temp_files: list[str] = []
        self._strategy = SDXL_STRATEGY
        self._png_lib_visible = False
        self._previous_tool_key: str = "m_brush"
        self._enhance_worker = None
        self._is_mask_tool = True
        self._build_ui()
        self._create_tools()
        self._connect_signals()
        self._set_tool("m_brush")

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(1)

        # Shared button group across both rows (exclusive)
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QToolButton] = {}

        # ── Mask Tools (two wrap-aware rows) ──────────────────────────────────
        # Row 1a: the mask tool buttons. Row 1b: mask actions + brush context.
        mask_row = QHBoxLayout()
        mask_row.setContentsMargins(2, 2, 2, 0)
        mask_row.setSpacing(4)

        lbl_mask = QLabel("Mask")
        lbl_mask.setStyleSheet(ROW_LABEL_STYLE)
        mask_row.addWidget(lbl_mask)

        mask_tool_defs = [
            ("m_brush", "Mask Brush"), ("m_eraser", "Mask Eraser"),
            ("|", None),
            ("m_line", "Mask Line"), ("m_ellipse", "Mask Ellipse"), ("m_rect", "Mask Rectangle"),
            ("m_fill", "Mask Flood Fill"),
            ("|", None),
            ("m_freehand", "Mask Freehand"), ("m_lasso", "Mask Polygon Lasso"),
            ("m_magnetic", "Mask Magnetic Lasso"), ("m_wand", "Mask Magic Wand"),
        ]
        self._add_tool_buttons(mask_row, mask_tool_defs)
        mask_row.addStretch()
        outer.addLayout(mask_row)

        # Row 1b: mask action buttons + size/tolerance context
        mask_actions_row = QHBoxLayout()
        mask_actions_row.setContentsMargins(2, 0, 2, 0)
        mask_actions_row.setSpacing(8)

        # Mask action buttons
        btn_clear_mask = QPushButton("Clear Mask")
        btn_clear_mask.setStyleSheet(MASK_BTN_STYLE)
        btn_clear_mask.setToolTip("Clear the entire mask layer")
        btn_clear_mask.clicked.connect(self._clear_mask)
        mask_actions_row.addWidget(btn_clear_mask)

        btn_invert = QPushButton("Invert Mask")
        btn_invert.setStyleSheet(MASK_BTN_STYLE)
        btn_invert.setToolTip("Swap masked/unmasked regions")
        btn_invert.clicked.connect(self._invert_mask)
        mask_actions_row.addWidget(btn_invert)

        mask_actions_row.addSpacing(12)

        # Mask context: size + tolerance (inline, only 2 controls)
        self._mask_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_size_slider.setRange(1, 100)
        self._mask_size_slider.setValue(16)
        self._mask_size_slider.setFixedWidth(120)
        self._mask_size_label = QLabel("16")
        self._mask_size_label.setFixedWidth(32)
        self._mask_size_slider.valueChanged.connect(
            lambda v: self._mask_size_label.setText(str(v)))
        self._mask_size_slider.valueChanged.connect(self._on_mask_size)

        self._mask_tol_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_tol_slider.setRange(1, 255)
        self._mask_tol_slider.setValue(32)
        self._mask_tol_slider.setFixedWidth(120)
        self._mask_tol_label = QLabel("32")
        self._mask_tol_label.setFixedWidth(32)
        self._mask_tol_slider.valueChanged.connect(
            lambda v: self._mask_tol_label.setText(str(v)))
        self._mask_tol_slider.valueChanged.connect(self._on_mask_tolerance)

        self._mask_ctx = [
            QLabel("Size:"), self._mask_size_slider, self._mask_size_label,
            QLabel("Tol:"), self._mask_tol_slider, self._mask_tol_label,
        ]
        for w in self._mask_ctx:
            mask_actions_row.addWidget(w)

        mask_actions_row.addStretch()
        outer.addLayout(mask_actions_row)

        # ── Paint Tools (two wrap-aware rows) ─────────────────────────────────
        # Row 2a: the paint tool buttons. Row 2b: undo/redo + tool settings +
        # zoom controls + panel toggles. Splitting avoids cramming 20+ widgets
        # onto a single line.
        paint_row = QHBoxLayout()
        paint_row.setContentsMargins(2, 0, 2, 0)
        paint_row.setSpacing(4)

        lbl_paint = QLabel("Paint")
        lbl_paint.setStyleSheet(ROW_LABEL_STYLE)
        paint_row.addWidget(lbl_paint)

        paint_tool_defs = [
            ("p_brush", "Paint Brush"), ("p_eraser", "Paint Eraser"), ("p_grab", "Grab Layer"),
            ("|", None),
            ("p_line", "Line"), ("p_ellipse", "Ellipse"), ("p_rect", "Rectangle"),
            ("|", None),
            ("p_eyedropper", "Eyedropper"), ("p_fill", "Flood Fill"),
            ("p_smudge", "Smudge"), ("p_gradient", "Gradient"),
            ("|", None),
            ("p_freehand", "Freehand Select"), ("p_lasso", "Polygon Lasso"),
            ("p_magnetic", "Magnetic Lasso"), ("p_wand", "Magic Wand"),
            ("|", None),
            ("p_transform", "Transform"),
        ]
        self._add_tool_buttons(paint_row, paint_tool_defs)
        paint_row.addStretch()
        outer.addLayout(paint_row)

        # Row 2b: undo/redo, tool settings, zoom, panel toggles
        paint_opts_row = QHBoxLayout()
        paint_opts_row.setContentsMargins(2, 0, 2, 0)
        paint_opts_row.setSpacing(8)

        # Undo / Redo
        for text, slot in [("Undo", lambda: self._canvas.undo()),
                           ("Redo", lambda: self._canvas.redo())]:
            b = QPushButton(text)
            b.setStyleSheet(BTN_STYLE)
            b.clicked.connect(slot)
            paint_opts_row.addWidget(b)

        paint_opts_row.addSpacing(12)

        # DrawToolSettingsPanel (replaces all inline context widgets)
        self._tool_settings = DrawToolSettingsPanel()
        paint_opts_row.addWidget(self._tool_settings)

        paint_opts_row.addSpacing(12)

        # Zoom controls
        btn_fit = QPushButton("Fit")
        btn_fit.setStyleSheet(BTN_STYLE)
        btn_fit.setToolTip("Fit canvas in view")
        btn_fit.clicked.connect(lambda: self._canvas.fit_view())
        paint_opts_row.addWidget(btn_fit)

        btn_100 = QPushButton("100%")
        btn_100.setStyleSheet(BTN_STYLE)
        btn_100.setToolTip("Zoom to 1:1")
        btn_100.clicked.connect(lambda: self._canvas.zoom_to_100())
        paint_opts_row.addWidget(btn_100)

        self._zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self._zoom_slider.setRange(10, 800)
        self._zoom_slider.setValue(100)
        self._zoom_slider.setFixedWidth(120)
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        paint_opts_row.addWidget(self._zoom_slider)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(40)
        paint_opts_row.addWidget(self._zoom_label)

        paint_opts_row.addStretch()

        # Panel toggles (far right of paint options row)
        self._btn_layers = QToolButton()
        self._btn_layers.setText("Layers")
        self._btn_layers.setCheckable(True)
        self._btn_layers.setStyleSheet(TOGGLE_STYLE)
        self._btn_layers.clicked.connect(self._toggle_layer_panel)
        paint_opts_row.addWidget(self._btn_layers)

        self._btn_png_lib = QToolButton()
        self._btn_png_lib.setText("PNG Lib")
        self._btn_png_lib.setCheckable(True)
        self._btn_png_lib.setStyleSheet(TOGGLE_STYLE)
        self._btn_png_lib.clicked.connect(self._toggle_png_library)
        paint_opts_row.addWidget(self._btn_png_lib)

        outer.addLayout(paint_opts_row)

        # ── Splitter (canvas | controls | layers | png lib) ──────────────────
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._canvas = DrawCanvas()
        self._splitter.addWidget(self._canvas)

        # Right panel
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setMinimumWidth(400)
        right_panel = QWidget()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(8, 6, 8, 6)
        rp.setSpacing(8)

        # Vertical splitter for right panel sections — drag to resize
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter = right_splitter
        right_splitter.setChildrenCollapsible(True)
        right_splitter.setHandleWidth(5)
        # A real minimum so the splitter cannot collapse to ~0 — that let the
        # whole right column shrink to the viewport instead of scrolling, which
        # crushed the params rows into each other and the PNG library to a sliver.
        right_splitter.setMinimumHeight(300)
        right_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )

        self._layer_panel = LayerPanel(self._canvas.layer_stack)
        self._layer_panel.setVisible(False)
        self._layer_panel.setMinimumHeight(260)
        right_splitter.addWidget(self._layer_panel)

        self._png_library = PNGLibrary()
        self._png_library.setVisible(False)
        self._png_library.setMinimumHeight(320)
        right_splitter.addWidget(self._png_library)

        self._gallery = ImageGalleryWidget("Results")
        self._gallery.setMinimumHeight(220)
        right_splitter.addWidget(self._gallery)
        right_splitter.setStretchFactor(2, 1)

        rp.addWidget(right_splitter, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._gen_btn = QPushButton("Inpaint")
        self._gen_btn.setObjectName("primary")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)
        self._unload_btn = QPushButton("Unload SD")
        self._unload_btn.setToolTip("Free SD image model from VRAM")
        self._unload_btn.clicked.connect(self._on_unload)
        btn_row.addWidget(self._unload_btn)
        rp.addLayout(btn_row)

        self._status = QLabel("")
        rp.addWidget(self._status)

        prompt_group = QGroupBox("Prompt")
        pg = QVBoxLayout(prompt_group)
        pg.setContentsMargins(8, 6, 8, 6)
        pg.setSpacing(8)
        self._enhance = PromptEnhanceWidget(mode="image")
        self._enhance.enhance_requested.connect(self._on_enhance)
        pg.addWidget(self._enhance)
        self._prompt = QPlainTextEdit()
        self._prompt.setMaximumHeight(85)
        self._prompt.setPlaceholderText("Describe what should fill the masked area...")
        pg.addWidget(self._prompt)
        self._neg_prompt = QPlainTextEdit()
        self._neg_prompt.setMaximumHeight(55)
        self._neg_prompt.setPlaceholderText("Negative prompt...")
        pg.addWidget(self._neg_prompt)
        rp.addWidget(prompt_group)

        self._inp_group = QGroupBox("Inpaint Settings")
        inp_group = self._inp_group
        inp = QVBoxLayout(inp_group)
        inp.setContentsMargins(8, 6, 8, 6)
        inp.setSpacing(8)
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        r1.addWidget(QLabel("Denoise:"))
        self._denoising = QDoubleSpinBox()
        self._denoising.setRange(0, 1)
        self._denoising.setDecimals(2)
        self._denoising.setSingleStep(0.05)
        self._denoising.setValue(0.75)
        self._denoising.setMinimumWidth(95)
        r1.addWidget(self._denoising)
        r1.addWidget(QLabel("Blur:"))
        self._mask_blur = QSpinBox()
        self._mask_blur.setRange(0, 64)
        self._mask_blur.setValue(4)
        self._mask_blur.setMinimumWidth(90)
        r1.addWidget(self._mask_blur)
        self._invert_mask_gen = QCheckBox("Invert")
        r1.addWidget(self._invert_mask_gen)
        r1.addStretch()
        inp.addLayout(r1)

        self._adv_inpaint_widget = QWidget()
        adv_col = QVBoxLayout(self._adv_inpaint_widget)
        adv_col.setContentsMargins(0, 0, 0, 0)
        adv_col.setSpacing(8)
        r2 = QHBoxLayout()
        r2.setSpacing(8)
        r2.addWidget(QLabel("Fill:"))
        self._inpaint_fill = QComboBox()
        self._inpaint_fill.addItems(["fill", "original", "latent noise", "latent nothing"])
        self._inpaint_fill.setMinimumWidth(160)
        r2.addWidget(self._inpaint_fill)
        self._full_res = QCheckBox("Only Masked")
        r2.addWidget(self._full_res)
        r2.addWidget(QLabel("Pad:"))
        self._full_res_padding = QSpinBox()
        self._full_res_padding.setRange(0, 256)
        self._full_res_padding.setValue(32)
        self._full_res_padding.setMinimumWidth(90)
        r2.addWidget(self._full_res_padding)
        r2.addStretch()
        adv_col.addLayout(r2)

        # Phase 4b: Laplacian pyramid paste-back (seam-free composite).
        # Its own row so the long label doesn't crowd the fill/padding controls.
        r3 = QHBoxLayout()
        r3.setSpacing(8)
        self._laplacian_blend = QCheckBox("Laplacian Blend")
        self._laplacian_blend.setToolTip(
            "Multi-level frequency-domain blend on paste-back. Hides the mask "
            "boundary at every frequency band (Photoshop healing-brush style). "
            "Slightly slower than the default alpha blend. Recommended for "
            "tight inpaint masks where the seam would otherwise be visible."
        )
        r3.addWidget(self._laplacian_blend)
        r3.addStretch()
        adv_col.addLayout(r3)
        inp.addWidget(self._adv_inpaint_widget)
        rp.addWidget(inp_group)

        self._params = ImageParamsWidget()
        self._params.lora_toggled.connect(self._on_lora_toggled)
        self._params.params_changed.connect(self._persist_params)
        self._params.set_dim_source_callback(self._source_image_size)
        rp.addWidget(self._params)

        # ADetailer post-step panel (face refinement after generation)
        from sdqt.widgets.adetailer_panel import ADetailerPanel
        self._adetailer = ADetailerPanel()
        self._adetailer.changed.connect(self._persist_params)
        rp.addWidget(self._adetailer)

        # Tab-level fields (prompt, denoise, mask/inpaint options) were only
        # saved on Generate — persist on change too, debounced for text fields.
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_params)
        self._prompt.textChanged.connect(self._schedule_persist)
        self._neg_prompt.textChanged.connect(self._schedule_persist)
        self._denoising.valueChanged.connect(self._schedule_persist)
        self._mask_blur.valueChanged.connect(self._schedule_persist)
        self._invert_mask_gen.toggled.connect(self._schedule_persist)
        self._inpaint_fill.currentIndexChanged.connect(self._schedule_persist)
        self._full_res.toggled.connect(self._schedule_persist)
        self._full_res_padding.valueChanged.connect(self._schedule_persist)
        self._laplacian_blend.toggled.connect(self._schedule_persist)

        # Batch Post-Processing — RealESRGAN sharpen/upscale, etc.
        from sdqt.widgets.batch_postprocess import BatchPostProcessWidget
        self._postproc = BatchPostProcessWidget(self.state)
        self._postproc.progress_changed.connect(self._show_status)
        self._postproc.finished.connect(self._on_postproc_done)
        rp.addWidget(self._postproc)
        rp.addStretch()

        right_scroll.setWidget(right_panel)
        self._splitter.addWidget(right_scroll)

        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setHandleWidth(2)
        outer.addWidget(self._splitter, 1)

        srow = QHBoxLayout()
        srow.setContentsMargins(4, 0, 4, 2)
        self._canvas_status = QLabel("")
        srow.addWidget(self._canvas_status)
        srow.addStretch()
        outer.addLayout(srow)

    def _add_tool_buttons(self, row: QHBoxLayout, defs: list) -> None:
        for key, label in defs:
            if key == "|":
                sep = QLabel(" ")
                sep.setFixedWidth(4)
                row.addWidget(sep)
                continue
            icon_key = key.split("_", 1)[1] if "_" in key else key
            btn = QToolButton()
            btn.setIcon(QIcon(make_tool_icon(icon_key)))
            btn.setIconSize(QSize(24, 24))
            btn.setToolTip(label)
            btn.setCheckable(True)
            btn.setFixedSize(TOOL_SZ, TOOL_SZ)
            btn.setStyleSheet(TOOL_BTN_STYLE)
            self._tool_group.addButton(btn)
            self._tool_buttons[key] = btn
            btn.clicked.connect(lambda checked, k=key: self._set_tool(k))
            row.addWidget(btn)

    # ── Tools ────────────────────────────────────────────────────────────────

    def _create_tools(self) -> None:
        c = self._canvas
        self._tools: dict[str, object] = {
            # Mask tools
            "m_brush": BrushTool(c), "m_eraser": EraserTool(c),
            "m_line": LineTool(c), "m_ellipse": EllipseTool(c), "m_rect": RectangleTool(c),
            "m_fill": FloodFillTool(c),
            "m_freehand": FreehandSelectTool(c), "m_lasso": PolygonLassoTool(c),
            "m_magnetic": MagneticLassoTool(c), "m_wand": MagicWandTool(c),
            # Paint tools
            "p_brush": BrushTool(c), "p_eraser": EraserTool(c), "p_grab": GrabTool(c),
            "p_line": LineTool(c), "p_ellipse": EllipseTool(c), "p_rect": RectangleTool(c),
            "p_eyedropper": EyedropperTool(c), "p_fill": FloodFillTool(c),
            "p_smudge": SmudgeTool(c), "p_gradient": GradientTool(c),
            "p_freehand": FreehandSelectTool(c), "p_lasso": PolygonLassoTool(c),
            "p_magnetic": MagneticLassoTool(c), "p_wand": MagicWandTool(c),
            "p_transform": TransformTool(c),
        }
        self._tools["p_eyedropper"].on_color_sampled = self._on_eyedropper_sample
        self._current_tool_key = "m_brush"

    def _set_tool(self, key: str) -> None:
        # Deactivate transform if switching away
        if self._current_tool_key == "p_transform" and key != "p_transform":
            t = self._tools.get("p_transform")
            if t and hasattr(t, "deactivate"):
                t.deactivate(confirm=True)
        if key != "p_eyedropper":
            self._previous_tool_key = self._current_tool_key

        self._current_tool_key = key
        self._is_mask_tool = key.startswith("m_")
        tool = self._tools.get(key)
        if tool is None:
            return

        # Mask tools auto-switch to the Mask layer
        if self._is_mask_tool and len(self._canvas.layer_stack) > 0:
            self._ensure_mask_layer()
            if self._mask_layer_uid:
                self._canvas.layer_stack.active_uid = self._mask_layer_uid

        # Red brush cursor for mask tools so it stands out over the mask
        # overlay; default white for paint tools.
        self._canvas.set_cursor_color(
            QColor(255, 60, 60, 235) if self._is_mask_tool
            else QColor(255, 255, 255, 220)
        )

        if key == "p_transform":
            self._canvas.set_tool(tool)
            tool.activate()
        else:
            self._canvas.set_tool(tool)

        btn = self._tool_buttons.get(key)
        if btn:
            btn.setChecked(True)

        self._show_context(key)
        self._sync_tool_settings()

        display = key.split("_", 1)[1] if "_" in key else key
        prefix = "Mask" if self._is_mask_tool else "Paint"
        self._canvas_status.setText(f"{prefix}: {display}")

    def _show_context(self, key: str) -> None:
        base = key.split("_", 1)[1] if "_" in key else key
        is_mask = key.startswith("m_")

        # Mask row context
        show_mask_size = is_mask and base in ("brush", "eraser", "line", "ellipse", "rect", "fill")
        show_mask_tol = is_mask and base in ("wand", "fill")
        for w in self._mask_ctx:
            w.setVisible(False)
        if show_mask_size:
            for w in self._mask_ctx[:3]:
                w.setVisible(True)
        if show_mask_tol:
            for w in self._mask_ctx[3:]:
                w.setVisible(True)

        # Paint tool settings panel
        if is_mask:
            self._tool_settings.show_for_tool("none")
        else:
            category = _CATEGORY.get(base, "none")
            self._tool_settings.show_for_tool(category)

    def _sync_tool_settings(self) -> None:
        tool = self._tools.get(self._current_tool_key)
        if tool is None:
            return
        if self._is_mask_tool:
            # Mask tools always use mask color and mask size
            if hasattr(tool, "color"):
                tool.color = MASK_COLOR
            if hasattr(tool, "secondary_color"):
                tool.secondary_color = MASK_COLOR
            mask_size = self._mask_size_slider.value()
            if hasattr(tool, "size"):
                tool.size = mask_size
            if hasattr(tool, "tolerance"):
                tool.tolerance = self._mask_tol_slider.value()
            if hasattr(tool, "line_color"):
                tool.line_color = MASK_COLOR
            if hasattr(tool, "fill_color"):
                tool.fill_color = MASK_COLOR
            if hasattr(tool, "filled"):
                tool.filled = True
            if hasattr(tool, "line_width"):
                tool.line_width = mask_size
            if hasattr(tool, "opacity"):
                tool.opacity = 1.0
            self._canvas.update_cursor_size(mask_size)
        else:
            # Paint tools: delegate to shared panel
            self._tool_settings.sync_to_tool(tool)
            if hasattr(tool, "size"):
                self._canvas.update_cursor_size(self._tool_settings.brush_size)

    # ── Signals ──────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Tool settings panel
        self._tool_settings.brush_size_changed.connect(self._on_brush_size)
        self._tool_settings.opacity_changed.connect(self._on_opacity_changed)
        self._tool_settings.primary_color_changed.connect(self._on_paint_settings_changed)
        self._tool_settings.secondary_color_changed.connect(self._on_paint_settings_changed)
        self._tool_settings.fill_color_changed.connect(self._on_paint_settings_changed)
        self._tool_settings.tolerance_changed.connect(self._on_paint_settings_changed)

        # Zoom sync
        self._canvas.zoom_changed.connect(self._on_zoom_changed)

        # Layer panel
        self._layer_panel._btn_new.clicked.disconnect()
        self._layer_panel._btn_new.clicked.connect(lambda: self._canvas.new_empty_layer("Layer"))
        self._layer_panel._tree.itemDoubleClicked.connect(
            lambda item, col: self._layer_panel.toggle_visibility(
                item.data(0, Qt.ItemDataRole.UserRole) or ""))
        self._layer_panel.save_to_library_requested.connect(self._on_save_layer_to_library)
        self._png_library.add_to_canvas_requested = self._add_png_to_canvas

        from sdqt.widgets.draw.shortcuts import setup_canvas_shortcuts
        setup_canvas_shortcuts(
            self, self._canvas,
            paste_callback=lambda: (
                self._canvas.paste_clipboard() and self._set_tool("p_transform")
            ),
            extra=[("Shift+F", self._fill_selection_to_mask)],
        )

    # ── Family switching ─────────────────────────────────────────────────────

    def set_family(self, strategy) -> None:
        self._strategy = strategy
        self._params.set_family(strategy)
        self._enhance.set_family(strategy)
        self._neg_prompt.setVisible(strategy.has_negative_prompt)
        self._adv_inpaint_widget.setVisible(strategy.has_inpaint_advanced)
        self._mask_blur.setVisible(strategy.has_inpaint_advanced)
        self._invert_mask_gen.setVisible(strategy.has_inpaint_advanced)

    # ── Image loading ────────────────────────────────────────────────────────

    def load_source(self, path: str) -> None:
        self._source_path = path
        # Clear all layers so stale masks/layers from a previous image
        # don't persist at the wrong size
        self._canvas.layer_stack.clear()
        self._canvas.load_background(path)
        self._ensure_mask_layer()
        self._set_tool(self._current_tool_key)
        self._canvas_status.setText(f"Loaded: {Path(path).name}")
        # Persist so the canvas restores on restart (was only saved on
        # Generate before).
        if not getattr(self, "_restoring", False):
            try:
                cfg = ProjectConfig.load(self.project_path)
                cfg.inpaint_source_path = path
                cfg.save(self.project_path)
            except Exception as exc:
                logger.debug("inpainter source persist failed: %s", exc)

    def _ensure_mask_layer(self) -> None:
        w, h = self._canvas.canvas_size
        for layer in self._canvas.layer_stack:
            if layer.name == "Mask":
                if layer.image.width() == w and layer.image.height() == h:
                    self._mask_layer_uid = layer.uid
                    return
                self._canvas.layer_stack.remove_layer(layer.uid)
                break
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        mask = self._canvas.add_image_layer(img, "Mask")
        self._mask_layer_uid = mask.uid

    # ── Mask operations ──────────────────────────────────────────────────────

    def _fill_selection_to_mask(self) -> None:
        self._ensure_mask_layer()
        mask = self._canvas.layer_stack.get(self._mask_layer_uid)
        if mask is None:
            return
        self._canvas.snapshot_active()
        prev_uid = self._canvas.layer_stack.active_uid
        self._canvas.layer_stack.active_uid = self._mask_layer_uid
        painter = QPainter(mask.image)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(MASK_COLOR))
        if self._canvas.selection.path is not None:
            painter.setClipPath(self._canvas.selection.path)
        painter.fillRect(mask.image.rect(), MASK_COLOR)
        painter.end()
        self._canvas.layer_stack.layer_updated.emit(mask.uid)
        if prev_uid and prev_uid != self._mask_layer_uid:
            self._canvas.layer_stack.active_uid = prev_uid
        self._canvas_status.setText("Selection filled to mask.")

    def _clear_mask(self) -> None:
        if not self._mask_layer_uid:
            return
        mask = self._canvas.layer_stack.get(self._mask_layer_uid)
        if mask is None:
            return
        self._canvas.snapshot_active()
        mask.image.fill(QColor(0, 0, 0, 0))
        self._canvas.layer_stack.layer_updated.emit(mask.uid)
        self._canvas_status.setText("Mask cleared.")

    def _invert_mask(self) -> None:
        if not self._mask_layer_uid:
            return
        mask = self._canvas.layer_stack.get(self._mask_layer_uid)
        if mask is None:
            return
        self._canvas.snapshot_active()
        w, h = mask.image.width(), mask.image.height()
        inv = QImage(w, h, QImage.Format.Format_ARGB32)
        inv.fill(MASK_COLOR)
        p = QPainter(inv)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        p.drawImage(0, 0, mask.image)
        p.end()
        mask.image = inv
        self._canvas.layer_stack.layer_updated.emit(mask.uid)
        self._canvas_status.setText("Mask inverted.")

    def _get_mask_path(self) -> str | None:
        """Extract mask as greyscale PNG. Uses QPainter compositing (no pixel loop)."""
        if not self._mask_layer_uid:
            return None
        mask_img = self._canvas.render_layer(self._mask_layer_uid)
        if mask_img.isNull():
            return None
        w, h = mask_img.width(), mask_img.height()

        # Compositing approach: create white image, use DestinationIn to keep
        # only pixels where mask has alpha > 0, then convert to greyscale.
        white = QImage(w, h, QImage.Format.Format_ARGB32)
        white.fill(QColor(255, 255, 255, 255))
        p = QPainter(white)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
        p.drawImage(0, 0, mask_img)
        p.end()

        grey = white.convertToFormat(QImage.Format.Format_Grayscale8)

        # Fast empty check via constBits
        bits = grey.constBits()
        if bits is None:
            return None
        data = bytes(bits)
        if not any(data):
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        grey.save(tmp.name)
        tmp.close()
        return tmp.name

    def _get_source_path(self) -> str | None:
        extra = sum(1 for l in self._canvas.layer_stack if l.name not in ("Background", "Mask"))
        if extra == 0 and self._source_path:
            return self._source_path
        exclude = {self._mask_layer_uid} if self._mask_layer_uid else set()
        flat = self._canvas.flatten_excluding(exclude)
        if flat.isNull():
            return self._source_path
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        flat.save(tmp.name)
        tmp.close()
        return tmp.name

    # ── Generation ───────────────────────────────────────────────────────────

    @Slot()
    def _on_generate(self) -> None:
        s = self._strategy
        pipeline = s.get_pipeline(self.state)
        if pipeline is None:
            self._show_status(f"{s.display_name} pipeline not loaded. Use Load button.")
            return
        src = self._get_source_path()
        if not src:
            self._show_status("No source image. Drop an image onto the canvas.")
            return
        mask_path = self._get_mask_path()
        if not mask_path:
            self._show_status("No mask. Use Mask tools to paint the inpaint region.")
            return

        # All inputs validated — claim the GPU lock (released in done/error).
        if not self.acquire_gpu("Image"):
            return

        # Track temp PNGs so they can be pruned once the worker consumes them.
        # The mask is always a fresh temp; src is only a temp when the canvas
        # was flattened (i.e. it differs from the persistent source path).
        self._gen_temp_files = [mask_path]
        if src and src != self._source_path:
            self._gen_temp_files.append(src)

        cfg = ProjectConfig.load(self.project_path)
        s.set_config(cfg, "prompt", self._prompt.toPlainText())
        s.set_config(cfg, "negative_prompt", self._neg_prompt.toPlainText())
        s.set_config(cfg, "denoising_strength", self._denoising.value())
        if s.has_inpaint_advanced:
            cfg.img_mask_blur = self._mask_blur.value()
            cfg.img_mask_invert = self._invert_mask_gen.isChecked()
            cfg.img_inpainting_fill = self._inpaint_fill.currentIndex()
            cfg.img_inpaint_full_res = self._full_res.isChecked()
            cfg.img_inpaint_full_res_padding = self._full_res_padding.value()
            if hasattr(self, "_laplacian_blend"):
                cfg.img_inpaint_laplacian_blend = self._laplacian_blend.isChecked()
        cfg.inpaint_source_path = self._source_path or src
        self._params.collect_to_config(cfg, strategy=s)
        cfg.save(self.project_path)

        self._gen_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Inpainting...")

        WorkerCls = s.resolve_worker("inpaint")
        worker = WorkerCls(pipeline, self.project_name, src, mask_path, cfg, parent=self)
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _cleanup_gen_temp_files(self) -> None:
        """Delete the mask/source temp PNGs created for the last generation."""
        for path in self._gen_temp_files:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Failed to remove temp file %s: %s", path, exc)
        self._gen_temp_files = []

    def _on_done(self, paths: list[str]) -> None:
        self.release_gpu("Image")
        self._cleanup_gen_temp_files()
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        if paths and self._postproc.is_enabled():
            self._show_status(f"Inpainted {len(paths)} image(s); post-processing...")
            self._postproc.process(paths)
        else:
            self._gallery.load_images(paths)
            self._show_status(f"Inpainted {len(paths)} image(s).")

    def _on_postproc_done(self, paths: list[str]) -> None:
        self._gallery.load_images(paths)
        self._show_status(f"Inpainted + post-processed {len(paths)} image(s).")

    def _on_error(self, msg: str) -> None:
        self.release_gpu("Image")
        self._cleanup_gen_temp_files()
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    @Slot()
    def _on_unload(self) -> None:
        method = getattr(self.state, self._strategy.unload_method, None)
        if method:
            method()
        self._show_status(f"{self._strategy.display_name} pipelines unloaded.")

    # ── Tool callbacks ───────────────────────────────────────────────────────

    def _on_mask_size(self, val: int) -> None:
        if self._is_mask_tool:
            tool = self._tools.get(self._current_tool_key)
            if tool and hasattr(tool, "size"):
                tool.size = val
            if tool and hasattr(tool, "line_width"):
                tool.line_width = val
            self._canvas.update_cursor_size(val)

    def _on_mask_tolerance(self, val: int) -> None:
        if self._is_mask_tool:
            tool = self._tools.get(self._current_tool_key)
            if tool and hasattr(tool, "tolerance"):
                tool.tolerance = val

    def _on_brush_size(self, val: int) -> None:
        if not self._is_mask_tool:
            tool = self._tools.get(self._current_tool_key)
            if tool and hasattr(tool, "size"):
                tool.size = val
            self._canvas.update_cursor_size(val)

    def _on_opacity_changed(self, val: int) -> None:
        if not self._is_mask_tool:
            tool = self._tools.get(self._current_tool_key)
            if tool and hasattr(tool, "opacity"):
                tool.opacity = val / 100.0

    @Slot()
    def _on_paint_settings_changed(self, _val=None) -> None:
        if not self._is_mask_tool:
            self._tool_settings.sync_to_tool(self._tools.get(self._current_tool_key))

    def _on_eyedropper_sample(self, color: QColor) -> None:
        self._tool_settings.set_primary_color(color)
        self._set_tool(self._previous_tool_key)

    def _on_lora_toggled(self, tag: str, checked: bool) -> None:
        current = self._prompt.toPlainText()
        if checked:
            if tag not in current:
                sep = " " if current and not current.endswith(" ") else ""
                self._prompt.setPlainText(current + sep + tag)
        else:
            self._prompt.setPlainText(current.replace(tag, "").strip())

    @Slot()
    def _schedule_persist(self) -> None:
        """Debounced persist for high-frequency fields (prompt keystrokes)."""
        if getattr(self, "_restoring", False):
            return
        self._persist_timer.start()

    @Slot()
    def _persist_params(self) -> None:
        """Auto-save model/sampler/scheduler/etc. whenever the user changes them."""
        if getattr(self, "_restoring", False):
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            s = self._strategy
            self._params.collect_to_config(cfg, strategy=s)
            self._adetailer.save_to_cfg(cfg)
            s.set_config(cfg, "prompt", self._prompt.toPlainText())
            s.set_config(cfg, "negative_prompt", self._neg_prompt.toPlainText())
            s.set_config(cfg, "denoising_strength", self._denoising.value())
            if s.has_inpaint_advanced:
                cfg.img_mask_blur = self._mask_blur.value()
                cfg.img_mask_invert = self._invert_mask_gen.isChecked()
                cfg.img_inpainting_fill = self._inpaint_fill.currentIndex()
                cfg.img_inpaint_full_res = self._full_res.isChecked()
                cfg.img_inpaint_full_res_padding = self._full_res_padding.value()
                cfg.img_inpaint_laplacian_blend = self._laplacian_blend.isChecked()
            cfg.save(self.project_path)
        except Exception as exc:
            logger.debug("inpainter _persist_params failed: %s", exc)

    def _source_image_size(self) -> tuple[int, int] | None:
        """Return the source image's (width, height), or None if no source."""
        if self._source_path:
            from PySide6.QtGui import QImageReader
            size = QImageReader(self._source_path).size()
            if size.isValid() and size.width() > 0:
                return (size.width(), size.height())
        w, h = self._canvas.canvas_size
        if w > 0 and h > 0:
            return (w, h)
        return None

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def _on_zoom_slider(self, val: int) -> None:
        self._canvas.set_zoom(float(val))

    def _on_zoom_changed(self) -> None:
        z = self._canvas.zoom_level
        self._zoom_label.setText(f"{z:.0f}%")
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(max(10, min(800, int(z))))
        self._zoom_slider.blockSignals(False)

    # ── Panels ───────────────────────────────────────────────────────────────

    def _toggle_layer_panel(self) -> None:
        self._layer_panel.setVisible(not self._layer_panel.isVisible())

    def _toggle_png_library(self) -> None:
        self._png_lib_visible = not self._png_lib_visible
        self._png_library.setVisible(self._png_lib_visible)
        if self._png_lib_visible:
            # Give the library a real slice of the media area (it shares the
            # vertical splitter with the results gallery) so it is actually
            # visible rather than collapsed to a sliver.
            total = sum(self._right_splitter.sizes()) or 640
            lib_h = max(340, int(total * 0.55))
            gal_h = max(200, total - lib_h)
            self._right_splitter.setSizes([0, lib_h, gal_h])
            try:
                lib_dir = self.state.global_config.png_library_dir
                if lib_dir:
                    self._png_library.set_root_dir(lib_dir)
            except AttributeError:
                pass

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
            QMessageBox.warning(self, "PNG Library", "No library directory configured.")
            return
        Path(lib_dir).mkdir(parents=True, exist_ok=True)
        self._png_library.set_root_dir(lib_dir)
        self._png_library.save_layer_image(layer.image, layer.name)
        self._canvas_status.setText(f"Saved '{layer.name}' to PNG Library")

    def _add_png_to_canvas(self, path: str) -> None:
        img = QImage(path)
        if not img.isNull():
            self._canvas.add_image_layer(img, Path(path).stem)

    # ── Prompt enhancement ───────────────────────────────────────────────────

    def _on_enhance(self, which: str, _text: str, style: str) -> None:
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        current = self._prompt.toPlainText() if which == "pos" else self._neg_prompt.toPlainText()
        self._enhance.set_enabled_buttons(False)
        worker = PromptEnhanceWorker(current, style, which, self.state, parent=self)
        worker.finished_ok.connect(lambda r: self._on_enhance_done(which, r))
        worker.error.connect(self._on_enhance_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    def _on_enhance_done(self, which: str, result: str) -> None:
        self._enhance.set_enabled_buttons(True)
        (self._prompt if which == "pos" else self._neg_prompt).setPlainText(result)
        self._show_status("Prompt enhanced.")

    def _on_enhance_error(self, msg: str) -> None:
        self._enhance.set_enabled_buttons(True)
        self._show_status(f"Enhance error: {msg}")

    # ── Drag & drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
                if len(self._canvas.layer_stack) == 0:
                    self.load_source(path)
                else:
                    img = QImage(path)
                    if not img.isNull():
                        self._canvas.add_image_layer(img, Path(path).stem)
                event.acceptProposedAction()
                return

    # ── Project persistence ──────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._gallery.clear()
        self._canvas.layer_stack.clear()
        self._canvas.canvas_size = (1024, 1024)
        self._canvas.scene().setSceneRect(QRectF(0, 0, 1024, 1024))
        self._mask_layer_uid = None
        self._source_path = None
        self._restoring = True
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        s = self._strategy
        src = cfg.inpaint_source_path or ""
        if src and Path(src).is_file():
            self.load_source(src)
        self._prompt.setPlainText(s.get_config(cfg, "prompt") or "")
        self._neg_prompt.setPlainText(s.get_config(cfg, "negative_prompt") or "")
        self._denoising.setValue(s.get_config(cfg, "denoising_strength") or 0.75)
        if s.has_inpaint_advanced:
            self._mask_blur.setValue(cfg.img_mask_blur)
            self._invert_mask_gen.setChecked(cfg.img_mask_invert)
            self._inpaint_fill.setCurrentIndex(cfg.img_inpainting_fill)
            self._full_res.setChecked(cfg.img_inpaint_full_res)
            self._full_res_padding.setValue(cfg.img_inpaint_full_res_padding)
            if hasattr(self, "_laplacian_blend"):
                self._laplacian_blend.setChecked(
                    bool(getattr(cfg, "img_inpaint_laplacian_blend", False))
                )
        self._params.restore_from_config(cfg, strategy=s)
        self._adetailer.load_from_cfg(cfg)
        self._restoring = False

    def populate_options(self, **kwargs) -> None:
        self._params.populate_options(**kwargs)
