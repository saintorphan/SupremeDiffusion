"""Full-screen image lightbox dialog -- double-click to view large."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.send_targets import DISABLED_TARGETS


_NAV_BTN_STYLE = """
QPushButton {
    background: rgba(0, 0, 0, 160);
    color: #fff;
    border: none;
    font-size: 28px;
    font-weight: bold;
    padding: 12px 8px;
    min-width: 44px;
    min-height: 80px;
}
QPushButton:hover {
    background: rgba(60, 60, 60, 200);
}
QPushButton:pressed {
    background: rgba(100, 100, 100, 220);
}
"""


class ImageLightbox(QDialog):
    """Modal dialog that shows an image at full/near-full size.

    - Left/Right arrows or overlay buttons to cycle images.
    - Right-click for send-to menu.
    - Click the image or press Escape to close.
    - Scrollable if the image is larger than the screen.

    Parameters
    ----------
    image_path:
        Path of the image to display initially.
    image_paths:
        Optional list of all image paths for navigation.
    send_targets:
        Optional list of (key, display_name) for right-click send-to menu.
    parent:
        Parent widget.
    """

    send_requested = Signal(str)  # key from send-to menu
    action_requested = Signal(str, str)  # (action_key, image_path)

    def __init__(
        self,
        image_path: str,
        *,
        image_paths: list[str] | None = None,
        send_targets: list[tuple[str, str]] | None = None,
        extra_menu_builder=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(Path(image_path).name)
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setStyleSheet("background: #111;")

        self._send_targets = send_targets or []
        self._extra_menu_builder = extra_menu_builder

        # -- Image list bookkeeping --
        if image_paths and len(image_paths) > 1:
            self._paths = list(image_paths)
            try:
                self._index = self._paths.index(image_path)
            except ValueError:
                self._index = 0
        else:
            self._paths = [image_path]
            self._index = 0

        # -- Layout: scroll area with nav overlays --
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Prev button
        self._prev_btn = QPushButton("\u276e")  # ❮
        self._prev_btn.setStyleSheet(_NAV_BTN_STYLE)
        self._prev_btn.setCursor(Qt.PointingHandCursor)
        self._prev_btn.setFocusPolicy(Qt.NoFocus)
        self._prev_btn.clicked.connect(self._show_prev)
        container_layout.addWidget(self._prev_btn)

        # Scroll area for image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setCursor(Qt.PointingHandCursor)
        self._label.setContextMenuPolicy(Qt.CustomContextMenu)
        self._label.customContextMenuRequested.connect(self._show_send_menu)
        scroll.setWidget(self._label)
        container_layout.addWidget(scroll, 1)

        # Next button
        self._next_btn = QPushButton("\u276f")  # ❯
        self._next_btn.setStyleSheet(_NAV_BTN_STYLE)
        self._next_btn.setCursor(Qt.PointingHandCursor)
        self._next_btn.setFocusPolicy(Qt.NoFocus)
        self._next_btn.clicked.connect(self._show_next)
        container_layout.addWidget(self._next_btn)

        outer.addWidget(container, 1)

        # Info bar
        self._info = QLabel()
        self._info.setStyleSheet(
            "QLabel { color: #aaa; padding: 4px 8px; font-size: 13px; }"
        )
        self._info.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._info)

        # Cache screen metrics
        screen = self.screen().availableGeometry()
        self._max_w = int(screen.width() * 0.9)
        self._max_h = int(screen.height() * 0.9)

        self._load_current()

        has_nav = len(self._paths) > 1
        self._prev_btn.setVisible(has_nav)
        self._next_btn.setVisible(has_nav)

        QShortcut(QKeySequence("Escape"), self, self.close)
        QShortcut(QKeySequence("Left"), self, self._show_prev)
        QShortcut(QKeySequence("Right"), self, self._show_next)

    @property
    def current_path(self) -> str:
        return self._paths[self._index] if self._paths else ""

    # -- Navigation --

    def _show_prev(self) -> None:
        if len(self._paths) <= 1:
            return
        self._index = (self._index - 1) % len(self._paths)
        self._load_current()

    def _show_next(self) -> None:
        if len(self._paths) <= 1:
            return
        self._index = (self._index + 1) % len(self._paths)
        self._load_current()

    def _load_current(self) -> None:
        path = self._paths[self._index]
        self.setWindowTitle(Path(path).name)

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            count_text = f"  |  {self._index + 1} / {len(self._paths)}" if len(self._paths) > 1 else ""
            self._info.setText(
                f"{Path(path).name}  |  {pixmap.width()} x {pixmap.height()}{count_text}"
                "  |  Right-click for send-to"
            )

            if not self.isVisible():
                disp_w = min(pixmap.width() + 110, self._max_w)
                disp_h = min(pixmap.height() + 50, self._max_h)
                self.resize(disp_w, disp_h)

            if pixmap.width() > self._max_w - 110 or pixmap.height() > self._max_h - 50:
                pixmap = pixmap.scaled(
                    self._max_w - 110, self._max_h - 50,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            self._label.setPixmap(pixmap)
        else:
            self._label.setText("Failed to load image.")
            self._info.setText(Path(path).name)

        self._prev_btn.setEnabled(len(self._paths) > 1)
        self._next_btn.setEnabled(len(self._paths) > 1)

    # -- Send-to context menu --

    def _show_send_menu(self, pos) -> None:
        menu = QMenu(self)
        has_items = False

        if self._send_targets:
            for key, name in self._send_targets:
                action = menu.addAction(name)
                if key in DISABLED_TARGETS:
                    action.setEnabled(False)
                else:
                    action.triggered.connect(lambda checked, k=key: self.send_requested.emit(k))
            has_items = True

        if self._extra_menu_builder:
            if has_items:
                menu.addSeparator()
            self._extra_menu_builder(menu, self.current_path, self)
            has_items = True

        if not has_items:
            return
        menu.exec(self._label.mapToGlobal(pos))

    # -- Mouse --

    def mousePressEvent(self, event) -> None:
        """Left-click to close (but not right-click or nav buttons)."""
        if event.button() == Qt.RightButton:
            return  # let context menu handle it
        child = self.childAt(event.pos())
        if child in (self._prev_btn, self._next_btn):
            return
        self.close()
