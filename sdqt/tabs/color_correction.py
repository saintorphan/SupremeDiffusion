"""Color Correction tab — BATCH retroactive correction for existing files.

Scope after the Phase 5 rebuild: fresh generations now apply color anchoring
**in-pipeline** automatically (see :mod:`supremediffusion.core.post_color_anchor`
and the Generate tab's "Auto Color Match" section). This tab is no longer the
main color-correction surface — it exists for two specific use cases:

1. **Retroactive correction** of files that were generated BEFORE the new
   pipeline was in place, or imported from external sources.
2. **Per-category** correction (separately adjusting skin / sky / etc.) which
   the in-pipeline color anchor doesn't expose.

For fresh in-pipeline generation, prefer the Generate tab's color match
controls + anchor mode. Use this tab when you have a file already.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMenu, QProgressBar, QPushButton,
    QScrollArea, QSlider, QSplitter, QVBoxLayout, QWidget, QAbstractItemView,
)

from sdqt.tabs.base import BaseTab
from sdqt.widgets.video_player import VideoPlayerWidget

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".ts", ".m4v"}
_ALL_MEDIA = _IMAGE_EXTS | _VIDEO_EXTS
_THUMB_H = 160


def _make_thumbnail(frame, max_w: int = 280, max_h: int = _THUMB_H) -> QPixmap | None:
    """Convert numpy RGB array to a scaled QPixmap."""
    if frame is None:
        return None
    try:
        h, w = frame.shape[:2]
        qimg = QImage(frame.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    except Exception:
        return None


# ── Parameter definitions per correction key ─────────────────────────────
# (min, max, default/identity, decimals, step, label)

_PARAM_DEFS: dict[str, tuple[float, float, float, int, float, str]] = {
    "brightness":   (-0.50, 0.50, 0.0, 2, 0.01, "Brightness"),
    "contrast":     (0.50, 2.00, 1.0, 2, 0.01, "Contrast"),
    "rgb_gain_r":   (0.50, 2.00, 1.0, 3, 0.005, "R Gain"),
    "rgb_gain_g":   (0.50, 2.00, 1.0, 3, 0.005, "G Gain"),
    "rgb_gain_b":   (0.50, 2.00, 1.0, 3, 0.005, "B Gain"),
    "temperature":  (-50.0, 50.0, 0.0, 1, 0.5, "Temperature"),
    "tint":         (-50.0, 50.0, 0.0, 1, 0.5, "Tint"),
    "saturation":   (0.00, 3.00, 1.0, 2, 0.01, "Saturation"),
    "black_point":  (0.0, 80.0, 0.0, 0, 1.0, "Black Point"),
    "white_point":  (180.0, 255.0, 255.0, 0, 1.0, "White Point"),
    "gamma":        (0.50, 2.00, 1.0, 2, 0.01, "Gamma"),
    "shadows":      (-0.50, 0.50, 0.0, 2, 0.01, "Shadows"),
    "highlights":   (-0.50, 0.50, 0.0, 2, 0.01, "Highlights"),
}

# Which param keys belong to each category
_CAT_PARAMS: dict[str, list[str]] = {
    "brightness_contrast": ["brightness", "contrast"],
    "color_balance":       ["rgb_gain_r", "rgb_gain_g", "rgb_gain_b"],
    "color_temperature":   ["temperature", "tint"],
    "hue_sat":             ["saturation"],
    "levels":              ["black_point", "white_point", "gamma"],
    "shadows_highlights":  ["shadows", "highlights"],
}


# ── Category row widget ───────────────────────────────────────────────────

class _CategoryRow(QWidget):
    """Correction category: checkbox + strength slider + per-parameter spinboxes."""

    def __init__(self, key: str, label: str, parent=None) -> None:
        super().__init__(parent)
        self.key = key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Header row: checkbox + strength slider
        header = QHBoxLayout()
        header.setSpacing(6)

        self._checkbox = QCheckBox(label)
        self._checkbox.setChecked(True)
        self._checkbox.setStyleSheet("font-weight: bold;")
        header.addWidget(self._checkbox)

        header.addStretch()

        str_label = QLabel("Str:")
        str_label.setStyleSheet("color: #aaa; font-size: 13px;")
        header.addWidget(str_label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(100)
        self._slider.setFixedWidth(120)
        header.addWidget(self._slider)

        self._pct_label = QLabel("100%")
        self._pct_label.setFixedWidth(40)
        self._pct_label.setStyleSheet("color: #aaa;")
        header.addWidget(self._pct_label)

        layout.addLayout(header)

        self._slider.valueChanged.connect(
            lambda v: self._pct_label.setText(f"{v}%")
        )

        # Parameter spinboxes
        self._spins: dict[str, QDoubleSpinBox] = {}
        param_keys = _CAT_PARAMS.get(key, [])
        params_row = QHBoxLayout()
        params_row.setSpacing(12)
        params_row.setContentsMargins(24, 0, 0, 0)

        for pk in param_keys:
            pmin, pmax, pdef, decimals, step, plabel = _PARAM_DEFS[pk]

            lbl = QLabel(f"{plabel}:")
            lbl.setStyleSheet("color: #aaa;")
            params_row.addWidget(lbl)

            spin = QDoubleSpinBox()
            spin.setRange(pmin, pmax)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setValue(pdef)
            spin.setMinimumWidth(95)
            params_row.addWidget(spin)
            self._spins[pk] = spin

        # Reset button
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setFixedWidth(50)
        self._reset_btn.clicked.connect(self._reset_params)
        params_row.addStretch()
        params_row.addWidget(self._reset_btn)

        layout.addLayout(params_row)

    @property
    def enabled(self) -> bool:
        return self._checkbox.isChecked()

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self._checkbox.setChecked(val)

    @property
    def strength(self) -> float:
        return self._slider.value() / 100.0

    @strength.setter
    def strength(self, val: float) -> None:
        self._slider.setValue(int(val * 100))

    def get_param(self, key: str) -> float:
        spin = self._spins.get(key)
        return spin.value() if spin else _PARAM_DEFS.get(key, (0, 0, 0))[2]

    def set_param(self, key: str, value: float) -> None:
        spin = self._spins.get(key)
        if spin:
            spin.setValue(value)

    def _reset_params(self) -> None:
        for pk, spin in self._spins.items():
            _, _, pdef, *_ = _PARAM_DEFS[pk]
            spin.setValue(pdef)

    def set_values_text(self, text: str) -> None:
        """Compat — no longer used but kept for safety."""
        pass

    def clear_values(self) -> None:
        self._reset_params()


# ── Drop-enabled file list ────────────────────────────────────────────────

class _DropThumbnailList(QListWidget):
    """Thumbnail grid that accepts drag-and-drop of media files."""

    files_dropped = Signal(list)  # list of file paths
    send_requested = Signal(str, str)  # (target_key, file_path)

    _THUMB_SIZE = 96

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))
        self.setGridSize(QSize(self._THUMB_SIZE + 8, self._THUMB_SIZE + 28))
        self.setResizeMode(QListWidget.Adjust)
        self.setSpacing(4)
        self.setWrapping(True)
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def add_file(self, path: str) -> None:
        """Add a file with its thumbnail (async for videos)."""
        from PySide6.QtGui import QIcon
        name = Path(path).stem  # stem only, no extension clutter
        item = QListWidgetItem(name)
        item.setToolTip(path)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setSizeHint(QSize(self._THUMB_SIZE + 8, self._THUMB_SIZE + 28))
        self.addItem(item)

        # Generate thumbnail without blocking — images are fast, videos deferred
        try:
            suffix = Path(path).suffix.lower()
            if suffix in _IMAGE_EXTS:
                pix = QPixmap(path)
                if not pix.isNull():
                    pix = pix.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                    item.setIcon(QIcon(pix))
            elif suffix in _VIDEO_EXTS:
                # Defer video thumbnail to avoid blocking the UI
                from PySide6.QtCore import QTimer
                QTimer.singleShot(50, lambda p=path, it=item: self._load_video_thumb(p, it))
        except Exception:
            pass

    def _load_video_thumb(self, path: str, item: QListWidgetItem) -> None:
        """Load video thumbnail in a deferred callback."""
        try:
            from PySide6.QtGui import QIcon
            from sdqt.utils.grade_compute import extract_frame_from_path
            frame = extract_frame_from_path(path, 0)
            if frame is not None:
                pix = _make_thumbnail(frame, max_w=self._THUMB_SIZE, max_h=self._THUMB_SIZE)
                if pix:
                    item.setIcon(QIcon(pix))
        except Exception:
            pass

    def _show_context_menu(self, pos) -> None:
        """Right-click menu on batch file thumbnails."""
        item = self.itemAt(pos)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path:
            return

        menu = QMenu(self)

        send_menu = menu.addMenu("Send to")
        for key, label in [
            ("timeline", "Timeline"),
            ("ve", "Video Extender"),
            ("longshot", "Longshot"),
            ("stillgrabber", "Still Grabber"),
            ("cc_ref", "Color Reference"),
        ]:
            act = send_menu.addAction(label)
            act.triggered.connect(lambda checked, k=key, p=path: self.send_requested.emit(k, p))

        menu.addSeparator()
        remove_act = menu.addAction("Remove")
        remove_act.triggered.connect(lambda: self._remove_item(item))

        menu.exec(self.mapToGlobal(pos))

    def _remove_item(self, item: QListWidgetItem) -> None:
        """Remove a single item by reference."""
        row = self.row(item)
        if row >= 0:
            self.takeItem(row)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and Path(p).suffix.lower() in _ALL_MEDIA:
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


# ── Main tab ──────────────────────────────────────────────────────────────

class ColorCorrectionTab(BaseTab):
    """Batch retroactive color correction for existing video / image files.

    Post Phase-5 rebuild: this tab is NOT the place for new-generation color
    matching — that's automatic in the gen pipeline now. This tab is for:
      - Correcting files generated before the new pipeline existed
      - Per-category masking (skin / sky / etc.) which the auto-anchor lacks
      - Matching imported external footage to a reference
    """

    color_ref_received = Signal(str)  # frame path from external sources
    send_requested = Signal(str, str)  # (target_key, result_path)
    replace_on_timeline = Signal(str, str)  # (original_source_path, result_path)

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)

        self._ref_path: str | None = None
        self._ref_frame = None  # numpy array
        self._source_paths: list[str] = []
        self._source_frame = None  # numpy array (first source file)
        self._corrections: dict | None = None
        self._worker = None
        self._last_result_path: str | None = None
        self._source_from_timeline: bool = False  # tracks if source came from timeline

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 0, 2, 2)
        root.setSpacing(4)

        # Scope banner — fresh generations are auto-corrected in-pipeline now.
        from PySide6.QtWidgets import QLabel as _QL
        banner = _QL(
            "Batch tool for existing files. Fresh generations are auto-color-anchored "
            "in the Generate tab — use this only for files already on disk or imported."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background:#2a2f3a; color:#ccc; padding:6px 10px; "
            "border-left:3px solid #4a90d9; font-size:13px;"
        )
        root.addWidget(banner)

        # ── Top: vertical splitter (video players on top, controls below) ──
        main_splitter = QSplitter(Qt.Vertical)

        # ── Video players row ──
        players_widget = QWidget()
        players_layout = QHBoxLayout(players_widget)
        players_layout.setContentsMargins(0, 0, 0, 0)
        players_layout.setSpacing(4)

        self._source_player = VideoPlayerWidget("Source")
        players_layout.addWidget(self._source_player, 1)

        self._result_player = VideoPlayerWidget("Result")
        self._result_player.set_send_targets(
            clip_targets=[
                ("timeline", "Timeline"),
                ("ve", "Video Extender"),
                ("longshot", "Longshot"),
                ("stillgrabber", "Still Grabber"),
                ("cc_source", "Color Correct"),
            ],
            frame_targets=[
                ("img2img", "Img2Img"),
                ("inpaint", "Inpaint"),
                ("face_swap", "Face Swap"),
                ("cc_ref", "Color Reference"),
            ],
        )
        self._result_player.enable_color_ref_action(True)
        self._result_player.send_clip_requested.connect(
            lambda key: self.send_requested.emit(key, self._last_result_path or "")
        )
        self._result_player.send_frame_requested.connect(
            lambda key: self._emit_frame_send(key)
        )
        players_layout.addWidget(self._result_player, 1)

        main_splitter.addWidget(players_widget)

        # ── Controls row (reference + corrections + batch) ──
        controls_widget = QWidget()
        controls_splitter = QSplitter(Qt.Horizontal)

        # ── Left: reference frame ──
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        ref_header = QHBoxLayout()
        ref_header.addWidget(QLabel("<b>Reference</b>"))
        ref_header.addStretch()
        self._load_ref_btn = QPushButton("Load...")
        self._load_ref_btn.setFixedWidth(60)
        self._load_ref_btn.clicked.connect(self._load_reference_dialog)
        ref_header.addWidget(self._load_ref_btn)
        left_lay.addLayout(ref_header)

        self._ref_label = QLabel("Right-click any media\n→ Set as Color Reference")
        self._ref_label.setAlignment(Qt.AlignCenter)
        self._ref_label.setMinimumHeight(_THUMB_H)
        self._ref_label.setStyleSheet(
            "background: #1a1a1a; border: 1px dashed #555; color: #666;"
        )
        left_lay.addWidget(self._ref_label)

        # Source load button
        src_header = QHBoxLayout()
        src_header.addWidget(QLabel("<b>Source</b>"))
        src_header.addStretch()
        self._load_src_btn = QPushButton("Load...")
        self._load_src_btn.setFixedWidth(60)
        self._load_src_btn.clicked.connect(self._load_source_dialog)
        src_header.addWidget(self._load_src_btn)
        left_lay.addLayout(src_header)

        left_lay.addStretch()
        controls_splitter.addWidget(left)

        # ── Center: corrections ──
        center = QWidget()
        center.setMinimumWidth(500)
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(4)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(6)
        toggle_row.addWidget(QLabel("<b>Corrections</b>"))
        toggle_row.addStretch()
        self._select_all_btn = QPushButton("All")
        self._select_all_btn.setFixedWidth(40)
        self._select_all_btn.clicked.connect(lambda: self._toggle_all(True))
        self._select_none_btn = QPushButton("None")
        self._select_none_btn.setFixedWidth(50)
        self._select_none_btn.clicked.connect(lambda: self._toggle_all(False))
        toggle_row.addWidget(self._select_all_btn)
        toggle_row.addWidget(self._select_none_btn)
        center_lay.addLayout(toggle_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget()
        self._cat_layout = QVBoxLayout(scroll_content)
        self._cat_layout.setContentsMargins(0, 0, 0, 0)
        self._cat_layout.setSpacing(4)

        from sdqt.utils.grade_compute import CORRECTION_CATEGORIES, CATEGORY_LABELS

        self._cat_rows: dict[str, _CategoryRow] = {}
        for cat_key in CORRECTION_CATEGORIES:
            label = CATEGORY_LABELS.get(cat_key, cat_key)
            row = _CategoryRow(cat_key, label)
            self._cat_rows[cat_key] = row
            self._cat_layout.addWidget(row)

        self._cat_layout.addStretch()
        scroll.setWidget(scroll_content)
        center_lay.addWidget(scroll, 1)

        controls_splitter.addWidget(center)

        # ── Right: batch file list ──
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        right_lay.addWidget(QLabel("<b>Batch Files</b>"))

        self._file_list = _DropThumbnailList()
        self._file_list.files_dropped.connect(self._add_files)
        self._file_list.send_requested.connect(
            lambda key, path: self.send_requested.emit(key, path)
        )
        right_lay.addWidget(self._file_list, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._add_files_dialog)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_files)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._remove_btn)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch()
        right_lay.addLayout(btn_row)

        controls_splitter.addWidget(right)

        controls_splitter.setStretchFactor(0, 1)
        controls_splitter.setStretchFactor(1, 3)
        controls_splitter.setStretchFactor(2, 1)

        controls_lay = QVBoxLayout(controls_widget)
        controls_lay.setContentsMargins(0, 0, 0, 0)
        controls_lay.addWidget(controls_splitter)

        main_splitter.addWidget(controls_widget)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)

        root.addWidget(main_splitter, 1)

        # ── Bottom bar ──
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        bottom.addWidget(QLabel("Overall:"))
        self._overall_slider = QSlider(Qt.Horizontal)
        self._overall_slider.setRange(0, 100)
        self._overall_slider.setValue(100)
        self._overall_slider.setFixedWidth(160)
        bottom.addWidget(self._overall_slider)
        self._overall_pct = QLabel("100%")
        self._overall_pct.setFixedWidth(40)
        bottom.addWidget(self._overall_pct)
        self._overall_slider.valueChanged.connect(
            lambda v: self._overall_pct.setText(f"{v}%")
        )

        bottom.addSpacing(12)
        bottom.addWidget(QLabel("Match:"))
        self._method_combo = QComboBox()
        self._method_combo.setFixedWidth(200)
        # Friendly labels → color-matcher method keys. "Match colors" is the
        # same MKL transfer as the Mode-2 "match frame" LF/RF buttons.
        self._method_combo.addItem("Match colors (recommended)", "mkl")
        self._method_combo.addItem("Strong match", "hm-mvgd-hm")
        self._method_combo.addItem("Subtle (color cast only)", "mean-only-lab")
        self._method_combo.setToolTip(
            "How the source clip is matched to the reference image, frame by "
            "frame.\n• Match colors — same as the Mode-2 'match frame' "
            "buttons (recommended)\n• Strong match — force the full color "
            "distribution to the reference\n• Subtle — shift color cast "
            "only, leave the look mostly alone"
        )
        bottom.addWidget(self._method_combo)

        bottom.addStretch()

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setMinimumWidth(140)
        self._analyze_btn.clicked.connect(self._analyze)
        bottom.addWidget(self._analyze_btn)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("primary")
        self._apply_btn.setMinimumWidth(140)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        bottom.addWidget(self._apply_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setMinimumWidth(90)
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._abort)
        bottom.addWidget(self._abort_btn)

        bottom.addSpacing(12)

        self._progress = QProgressBar()
        self._progress.setMinimumWidth(200)
        self._progress.setVisible(False)
        bottom.addWidget(self._progress)

        root.addLayout(bottom)

        # ── Send-to row (visible after results) ──
        send_row = QHBoxLayout()
        send_row.setSpacing(6)

        from sdqt.widgets.send_to import SendToWidget
        self._send_to = SendToWidget(
            label="Send result to:",
            targets=[
                ("timeline", "Timeline"),
                ("ve", "Video Extender"),
                ("longshot", "Longshot"),
                ("trim", "Trim & Crop"),
                ("stillgrabber", "Still Grabber"),
            ],
        )
        self._send_to.send_requested.connect(self._on_send_to)
        self._send_to.set_enabled(False)
        send_row.addWidget(self._send_to)

        self._replace_btn = QPushButton("Replace on Timeline")
        self._replace_btn.setToolTip(
            "Replace the original timeline clip with the corrected version"
        )
        self._replace_btn.setFixedWidth(160)
        self._replace_btn.setVisible(False)
        self._replace_btn.clicked.connect(self._on_replace_timeline)
        send_row.addWidget(self._replace_btn)

        send_row.addStretch()
        root.addLayout(send_row)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa; padding: 2px; font-size: 13px;")
        root.addWidget(self._status)

    # ── Public API (called by MainWindow) ─────────────────────────────────

    def set_reference(self, path: str) -> None:
        """Set reference image/video frame. Called from context menu routing."""
        from sdqt.utils.grade_compute import extract_frame_from_path

        frame = extract_frame_from_path(path)
        if frame is None:
            self._show_status(f"Failed to load reference: {Path(path).name}")
            return

        self._ref_path = path
        self._ref_frame = frame

        pix = _make_thumbnail(frame)
        if pix:
            self._ref_label.setPixmap(pix)
            self._ref_label.setStyleSheet("background: #1a1a1a; border: 1px solid #555;")

        self._show_status(
            f"Reference set: {Path(path).name} — load a clip and hit Apply"
        )
        self._corrections = None
        self._clear_correction_values()
        self._maybe_enable_apply()

    def _maybe_enable_apply(self) -> None:
        """Enable Apply whenever there's at least one source to act on.

        The per-frame reference match needs no Analyze pass: load a reference,
        load a clip, press Apply. (With no reference loaded, Apply falls back
        to the manual-grade sliders.)
        """
        self._apply_btn.setEnabled(bool(self._source_paths))

    def add_source(self, path: str, *, from_timeline: bool = False) -> None:
        """Set the primary source file and show its preview.

        When called via send-to (single clip), replaces the current source.
        Use _add_files for batch drag-and-drop.

        Args:
            path: Path to the source media file.
            from_timeline: If True, enables "Replace on Timeline" after correction.
        """
        # Replace current source — clear previous state
        self._file_list.clear()
        self._source_paths.clear()
        self._corrections = None
        self._clear_correction_values()
        self._source_from_timeline = from_timeline
        self._result_player.load_video(None)
        self._send_to.set_enabled(False)
        self._replace_btn.setVisible(False)

        self._source_paths.append(path)
        self._file_list.add_file(path)
        self._load_source_into_player(path)
        self._maybe_enable_apply()

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_source_into_player(self, path: str) -> None:
        """Load a source file into the source video player and extract frame."""
        suffix = Path(path).suffix.lower()
        if suffix in _VIDEO_EXTS:
            self._source_player.load_video(path)
        elif suffix in _IMAGE_EXTS:
            # Show image as a static frame in the player
            self._source_player.load_video(path)

        from sdqt.utils.grade_compute import extract_frame_from_path
        self._source_frame = extract_frame_from_path(path)

    def _load_reference_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image/Video", "",
            "Media Files (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv);;All (*)",
        )
        if path:
            self.set_reference(path)

    def _load_source_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Source Files", "",
            "Media Files (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv);;All (*)",
        )
        if paths:
            self._add_files(paths)

    def _add_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Files to Batch", "",
            "Media Files (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv);;All (*)",
        )
        if paths:
            self._add_files(paths)

    def _add_files(self, paths: list[str]) -> None:
        for p in paths:
            if p not in self._source_paths:
                self._source_paths.append(p)
                self._file_list.add_file(p)
        if self._source_paths and self._source_frame is None:
            self._load_source_into_player(self._source_paths[0])
        self._maybe_enable_apply()
        self._show_status(f"{len(self._source_paths)} file(s) in batch")

    def _remove_selected(self) -> None:
        for item in reversed(self._file_list.selectedItems()):
            path = item.data(Qt.ItemDataRole.UserRole)
            self._file_list.takeItem(self._file_list.row(item))
            if path and path in self._source_paths:
                self._source_paths.remove(path)
        if self._source_paths:
            # Reload the first remaining file into the player
            self._load_source_into_player(self._source_paths[0])
        else:
            self._source_frame = None
            self._source_player.load_video(None)
        self._maybe_enable_apply()

    def _clear_files(self) -> None:
        self._file_list.clear()
        self._source_paths.clear()
        self._source_frame = None
        self._source_player.load_video(None)
        self._result_player.load_video(None)
        self._corrections = None
        self._clear_correction_values()
        self._apply_btn.setEnabled(False)

    def _toggle_all(self, state: bool) -> None:
        for row in self._cat_rows.values():
            row.enabled = state

    def _clear_correction_values(self) -> None:
        for row in self._cat_rows.values():
            row.clear_values()

    # ── Analyze ───────────────────────────────────────────────────────────

    @Slot()
    def _analyze(self) -> None:
        if self._ref_frame is None:
            self._show_status("No reference frame set")
            return
        if not self._source_paths:
            self._show_status("No source files loaded")
            return

        # Extract source frame if needed
        if self._source_frame is None:
            from sdqt.utils.grade_compute import extract_frame_from_path
            self._source_frame = extract_frame_from_path(self._source_paths[0])
            if self._source_frame is None:
                self._show_status("Failed to extract frame from source")
                return

        # Generate 3D LUT via color-matcher MKL
        from sdqt.utils.grade_compute import (
            generate_lut3d, derive_gains_from_transfer,
            color_match_frame, CORRECTION_CATEGORIES,
        )

        self._show_status("Generating 3D LUT (MKL)...")

        lut_dir = str(self.project_path / "luts")
        import time
        lut_path = str(Path(lut_dir) / f"cc_{int(time.time())}.cube")

        try:
            generate_lut3d(
                self._ref_frame, self._source_frame, lut_path,
            )
        except Exception as exc:
            logger.error("LUT generation failed: %s", exc)
            self._show_status(f"Analysis failed: {exc}")
            return

        self._lut_path = lut_path

        # Compute full corrections for UI spinboxes (informational —
        # the 3D LUT is what actually gets applied)
        from sdqt.utils.grade_compute import compute_grade
        self._corrections = compute_grade(
            self._ref_frame, self._source_frame,
        ) or {}

        # Populate spinboxes with derived values (for display/tweaking)
        active_cats = 0
        for cat_key, keys in CORRECTION_CATEGORIES.items():
            row = self._cat_rows.get(cat_key)
            if not row:
                continue
            has_correction = any(k in self._corrections for k in keys)
            row.enabled = has_correction
            if has_correction:
                active_cats += 1
                row.strength = 1.0
            else:
                row._reset_params()
                continue

            for k in keys:
                if k not in self._corrections:
                    continue
                val = self._corrections[k]
                if k == "rgb_gain" and isinstance(val, list):
                    row.set_param("rgb_gain_r", val[0])
                    row.set_param("rgb_gain_g", val[1])
                    row.set_param("rgb_gain_b", val[2])
                else:
                    row.set_param(k, val)

        self._apply_btn.setEnabled(True)
        self._show_status(
            f"Analysis complete — 3D LUT generated ({active_cats} "
            f"category(ies) shown for reference)"
        )

    # ── Apply ─────────────────────────────────────────────────────────────

    def _read_corrections_from_ui(self) -> dict:
        """Read current correction values from the spinboxes."""
        corrections: dict = {}
        from sdqt.utils.grade_compute import CORRECTION_CATEGORIES, _IDENTITY

        for cat_key, keys in CORRECTION_CATEGORIES.items():
            row = self._cat_rows.get(cat_key)
            if not row or not row.enabled:
                continue
            strength = row.strength * (self._overall_slider.value() / 100.0)
            if strength < 0.001:
                continue
            for k in keys:
                if k == "rgb_gain":
                    r = row.get_param("rgb_gain_r")
                    g = row.get_param("rgb_gain_g")
                    b = row.get_param("rgb_gain_b")
                    identity = _IDENTITY["rgb_gain"]
                    scaled = [
                        id_v + (v - id_v) * strength
                        for v, id_v in zip([r, g, b], identity)
                    ]
                    if any(abs(s - 1.0) > 0.001 for s in scaled):
                        corrections["rgb_gain"] = scaled
                else:
                    val = row.get_param(k)
                    identity = _IDENTITY.get(k, 0.0)
                    scaled_val = identity + (val - identity) * strength
                    if abs(scaled_val - identity) > 0.001:
                        corrections[k] = scaled_val
        return corrections

    @Slot()
    def _apply(self) -> None:
        if not self._source_paths:
            self._show_status("No source files loaded")
            return

        # Primary path: a reference is loaded → run the REAL per-frame color
        # match (same MKL transfer as the Mode-2 "match frame" buttons, applied
        # to every frame). This is the "match a clip to a source image" tool.
        if self._ref_frame is not None:
            self._apply_reference_match()
            return

        # No reference → fall back to the manual-grade effects path (the
        # per-category correction sliders). Only reachable when the user has
        # tweaked corrections without loading a reference image.
        from sdqt.utils.grade_compute import corrections_to_effects
        corrections = self._read_corrections_from_ui()
        if not corrections:
            self._show_status("Load a reference image, or set correction values.")
            return
        effects = corrections_to_effects(corrections)
        if not effects:
            self._show_status("No applicable effects generated")
            return

        out_dir = str(self.project_path / "outputs")

        from sdqt.workers.color_correction import ColorCorrectionBatchWorker

        self._begin_worker_ui("Applying corrections...")

        worker = ColorCorrectionBatchWorker(
            effects=effects,
            files=list(self._source_paths),
            output_dir=out_dir,
            parent=self,
        )

        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)

        self._worker = worker
        worker.start()

    def _apply_reference_match(self) -> None:
        """Run true per-frame reference color matching on all source files.

        Writes the reference frame to a stable temp PNG so the reference is a
        plain full-range RGB still (identical convention to the source frames
        the matcher decodes) — this is why it matches cleanly, the same reason
        the Mode-2 LF/RF buttons work: both sides are plain RGB, no ffmpeg
        colorspace/range guessing in between.
        """
        import numpy as np
        from PIL import Image

        try:
            ref_tmp = tempfile.NamedTemporaryFile(
                suffix=".png", prefix="ccref_", delete=False,
            )
            Image.fromarray(np.asarray(self._ref_frame)).save(ref_tmp.name)
            ref_tmp.close()
        except Exception as exc:
            self._show_status(f"Could not prepare reference: {exc}")
            return

        method = self._method_combo.currentData() or "mkl"
        strength = self._overall_slider.value() / 100.0
        out_dir = str(self.project_path / "outputs")

        from sdqt.workers.color_correction import ClipColorMatchBatchWorker

        self._begin_worker_ui(
            f"Matching {len(self._source_paths)} file(s) to reference…"
        )

        worker = ClipColorMatchBatchWorker(
            reference=ref_tmp.name,
            files=list(self._source_paths),
            output_dir=out_dir,
            method=method,
            strength=strength,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)

        self._worker = worker
        worker.start()

    def _begin_worker_ui(self, status: str) -> None:
        """Shared setup for a batch run — progress bar + button states."""
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._abort_btn.setVisible(True)
        self._apply_btn.setEnabled(False)
        self._analyze_btn.setEnabled(False)
        self._show_status(status)

    @Slot()
    def _abort(self) -> None:
        if self._worker:
            self._worker.abort()
            self._show_status("Aborting...")

    def _on_progress(self, frac: float, msg: str) -> None:
        self._progress.setValue(int(frac * 100))
        self._show_status(msg)

    def _on_done(self, results: object) -> None:
        self._progress.setVisible(False)
        self._abort_btn.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._analyze_btn.setEnabled(True)
        self._worker = None

        if isinstance(results, list) and results:
            count = len(results)
            first_result = results[0]
            self._last_result_path = first_result
            suffix = Path(first_result).suffix.lower()
            if suffix in _VIDEO_EXTS:
                self._result_player.load_video(first_result)
            elif suffix in _IMAGE_EXTS:
                self._result_player.load_video(first_result)

            self._send_to.set_enabled(True)
            self._replace_btn.setVisible(self._source_from_timeline)
            self._show_status(f"Done — {count} file(s) saved to project outputs")
        else:
            self._send_to.set_enabled(False)
            self._replace_btn.setVisible(False)
            self._show_status("No results produced")

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._abort_btn.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._analyze_btn.setEnabled(True)
        self._worker = None
        self._show_status(f"Error: {msg}")

    # ── Send-to handlers ─────────────────────────────────────────────────

    def _emit_frame_send(self, key: str) -> None:
        """Capture a frame from the result player and emit send_requested."""
        frame_path = self._result_player.capture_frame()
        if frame_path:
            self.send_requested.emit(key, frame_path)

    @Slot(str)
    def _on_send_to(self, target: str) -> None:
        """Emit send_requested with the last result path."""
        if self._last_result_path and Path(self._last_result_path).is_file():
            self.send_requested.emit(target, self._last_result_path)

    @Slot()
    def _on_replace_timeline(self) -> None:
        """Replace the original timeline clip with the corrected version."""
        if not self._last_result_path or not Path(self._last_result_path).is_file():
            self._show_status("No result to replace with")
            return
        if not self._source_paths:
            self._show_status("No source clip to replace")
            return
        original_path = self._source_paths[0]
        self.replace_on_timeline.emit(original_path, self._last_result_path)
        self._show_status(f"Replacing timeline clip: {Path(original_path).name}")
