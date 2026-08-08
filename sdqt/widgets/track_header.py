"""Track header widget — name, volume, mute/visibility controls, drag handle."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QCursor, QMouseEvent, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.timeline_track import TrackType

logger = logging.getLogger(__name__)

_HEADER_WIDTH = 150


class TrackHeaderWidget(QWidget):
    """Header for a single timeline track — name, volume, mute, visibility."""

    volume_changed = Signal(str, float)   # (track_id, volume)
    mute_toggled = Signal(str, bool)      # (track_id, muted)
    visible_toggled = Signal(str, bool)   # (track_id, visible)
    lock_toggled = Signal(str, bool)      # (track_id, locked)
    context_menu_requested = Signal(str, QPoint)  # (track_id, global_pos)
    clicked = Signal(str)  # track_id — emitted on left-click outside drag handle

    def __init__(
        self,
        track_id: str,
        name: str,
        track_type: TrackType,
        height: int = 70,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._track_id = track_id
        self._track_type = track_type
        self._selected = False
        self._drag_start_y: int | None = None
        self._dragging = False

        self.setFixedWidth(_HEADER_WIDTH)
        self.setFixedHeight(height)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

        self._build_ui(name, track_type)

    @property
    def track_id(self) -> str:
        return self._track_id

    def _build_ui(self, name: str, track_type: TrackType) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 4, 2)
        layout.setSpacing(2)

        # Drag handle
        self._handle = QWidget()
        self._handle.setFixedWidth(14)
        self._handle.setCursor(QCursor(Qt.SizeAllCursor))
        self._handle.setStyleSheet("background: #333; border-right: 1px solid #555;")
        layout.addWidget(self._handle)

        # Right column: name + controls
        right = QVBoxLayout()
        right.setContentsMargins(2, 0, 0, 0)
        right.setSpacing(1)

        # Track name
        self._name_label = QLabel(name)
        self._name_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #ddd;")
        self._name_label.setToolTip(name)
        right.addWidget(self._name_label)

        # Volume slider
        vol_row = QHBoxLayout()
        vol_row.setSpacing(2)
        vol_row.setContentsMargins(0, 0, 0, 0)

        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(70)
        self._volume_slider.setFixedHeight(14)
        self._volume_slider.setToolTip("Track volume")
        self._volume_slider.valueChanged.connect(self._on_volume)
        vol_row.addWidget(self._volume_slider)
        right.addLayout(vol_row)

        # Mute + Visibility buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        btn_row.setContentsMargins(0, 0, 0, 0)

        btn_style = (
            "QPushButton { min-width: 28px; max-width: 28px; min-height: 24px;"
            "  max-height: 24px; font-size: 13px; border: 1px solid #555;"
            "  border-radius: 2px; padding: 0; }"
            "QPushButton:checked { background: #4a90d9; border-color: #6ab0ff; }"
            "QPushButton:hover { background: #3a3a3a; }"
        )

        self._mute_btn = QPushButton("M")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setToolTip("Mute track")
        self._mute_btn.setStyleSheet(btn_style)
        self._mute_btn.toggled.connect(self._on_mute)
        btn_row.addWidget(self._mute_btn)

        if track_type == TrackType.VIDEO:
            self._vis_btn = QPushButton("\U0001F441")  # 👁
            self._vis_btn.setCheckable(True)
            self._vis_btn.setChecked(True)
            self._vis_btn.setToolTip("Toggle visibility")
            self._vis_btn.setStyleSheet(btn_style)
            self._vis_btn.toggled.connect(self._on_visible)
            btn_row.addWidget(self._vis_btn)
        else:
            self._vis_btn = None

        self._lock_btn = QPushButton("\U0001F513")  # 🔓
        self._lock_btn.setCheckable(True)
        self._lock_btn.setToolTip("Lock / Unlock track")
        self._lock_btn.setStyleSheet(btn_style)
        self._lock_btn.toggled.connect(self._on_lock)
        btn_row.addWidget(self._lock_btn)

        btn_row.addStretch()
        right.addLayout(btn_row)

        layout.addLayout(right, 1)

    def set_name(self, name: str) -> None:
        self._name_label.setText(name)

    def set_volume(self, volume: float) -> None:
        self._volume_slider.blockSignals(True)
        self._volume_slider.setValue(int(volume * 100))
        self._volume_slider.blockSignals(False)

    def set_muted(self, muted: bool) -> None:
        self._mute_btn.blockSignals(True)
        self._mute_btn.setChecked(muted)
        self._mute_btn.blockSignals(False)

    def set_visible(self, visible: bool) -> None:
        if self._vis_btn:
            self._vis_btn.blockSignals(True)
            self._vis_btn.setChecked(visible)
            self._vis_btn.blockSignals(False)

    def _on_volume(self, value: int) -> None:
        self.volume_changed.emit(self._track_id, value / 100.0)

    def _on_mute(self, checked: bool) -> None:
        self.mute_toggled.emit(self._track_id, checked)

    def _on_visible(self, checked: bool) -> None:
        self.visible_toggled.emit(self._track_id, checked)

    def _on_lock(self, checked: bool) -> None:
        self._lock_btn.setText("\U0001F512" if checked else "\U0001F513")  # 🔒 / 🔓
        self._lock_btn.setToolTip("Unlock track" if checked else "Lock track")
        self.lock_toggled.emit(self._track_id, checked)

    def set_locked(self, locked: bool) -> None:
        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(locked)
        self._lock_btn.setText("\U0001F512" if locked else "\U0001F513")
        self._lock_btn.blockSignals(False)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def _on_context_menu(self, pos) -> None:
        self.context_menu_requested.emit(self._track_id, self.mapToGlobal(pos))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            x = int(event.position().x())
            # Always select on click — drag handle also selects
            self.clicked.emit(self._track_id)
            if x < 16:
                # Drag handle area — also start tracking for reorder drag
                self._drag_start_y = int(event.position().y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_y is not None
            and abs(int(event.position().y()) - self._drag_start_y) > 8
        ):
            from PySide6.QtCore import QMimeData
            from PySide6.QtGui import QDrag
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData("application/x-track-header", self._track_id.encode("utf-8"))
            drag.setMimeData(mime)
            drag.exec(Qt.MoveAction)
            self._drag_start_y = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start_y = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)

        # Selected highlight background + accent bar
        if self._selected:
            p.fillRect(self.rect(), QColor("#2a3a4a"))
            p.fillRect(0, 0, 3, self.height(), QColor("#4a90d9"))

        p.end()

        # Let Qt paint child widgets on top
        super().paintEvent(event)

        # Grip lines on the drag handle (painted last, on top)
        p = QPainter(self)
        p.setPen(QPen(QColor("#666"), 1))
        x = 5
        for y in range(12, self.height() - 8, 6):
            p.drawLine(x, y, x + 4, y)
        p.end()
