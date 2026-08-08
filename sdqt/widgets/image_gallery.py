"""Image gallery widget -- thumbnail grid with selection."""

from __future__ import annotations

import logging
from pathlib import Path

from sdqt.widgets.send_targets import DISABLED_TARGETS

from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_save_filename
from sdqt.widgets.image_lightbox import ImageLightbox

logger = logging.getLogger(__name__)


class ImageGalleryWidget(QWidget):
    """Thumbnail grid for browsing generated images.

    Signals:
        image_selected(str): path to the selected image
    """

    image_selected = Signal(str)
    send_requested = Signal(str)  # key e.g. "img2img", "inpaint"
    final_frame_requested = Signal(str)  # key e.g. "img2vid", "ve"
    guide_video_requested = Signal(str)  # key e.g. "img2vid_m1", "ve"
    postprocess_requested = Signal(str)  # image path to post-process
    color_ref_requested = Signal(str)  # image path for Color Correct tab

    def __init__(self, label: str = "Gallery", parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel(f"<b>{label}</b>")
        layout.addWidget(title)

        self._list = QListWidget()
        self._list.setViewMode(QListWidget.IconMode)
        self._list.setIconSize(QSize(128, 128))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setSpacing(4)
        self._list.setWrapping(True)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._list, 1)

        self._empty_label = QLabel("No images yet. Generate or drag-and-drop to get started.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #aaa; font-style: italic; font-size: 13px; padding: 8px;")
        layout.addWidget(self._empty_label)

        self._paths: list[str] = []
        self._send_targets: list[tuple[str, str]] = []  # (key, display_name)
        self._final_frame_targets: list[tuple[str, str]] = []  # (key, display_name)
        self._guide_video_targets: list[tuple[str, str]] = []  # (key, display_name)

    def load_images(self, paths: list[str]) -> None:
        """Populate the gallery with image paths."""
        self._list.clear()
        self._paths = list(paths)
        self._empty_label.setVisible(not paths)
        self._list.setVisible(bool(paths))

        for p in paths:
            pp = Path(p)
            if not pp.is_file():
                continue
            pixmap = QPixmap(str(pp))
            if pixmap.isNull():
                continue
            item = QListWidgetItem()
            thumb = pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(thumb))
            item.setText(pp.name)
            item.setData(Qt.UserRole, str(pp))
            item.setToolTip(str(pp))
            self._list.addItem(item)

    def clear(self) -> None:
        self._list.clear()
        self._paths = []
        self._empty_label.setVisible(True)
        self._list.setVisible(False)

    @property
    def selected_path(self) -> str | None:
        item = self._list.currentItem()
        if item:
            return item.data(Qt.UserRole)
        return None

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_item_changed(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if current:
            path = current.data(Qt.UserRole)
            if path:
                self.image_selected.emit(path)

    @Slot(QListWidgetItem)
    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            lightbox = ImageLightbox(
                path,
                image_paths=self._paths,
                send_targets=self._send_targets,
                parent=self.window(),
            )
            lightbox.send_requested.connect(
                lambda key: self._on_lightbox_send(key, lightbox)
            )
            lightbox.exec()

    def _on_lightbox_send(self, key: str, lightbox) -> None:
        """Route send from lightbox — sync gallery selection to lightbox's current image first."""
        current = lightbox.current_path
        if current:
            # Select the matching item in the gallery
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item and item.data(Qt.UserRole) == current:
                    self._list.setCurrentRow(i)
                    break
        self.send_requested.emit(key)

    def set_send_targets(self, targets: list[tuple[str, str]]) -> None:
        """Configure right-click send-to targets: list of (key, display_name)."""
        self._send_targets = targets

    def set_final_frame_targets(self, targets: list[tuple[str, str]]) -> None:
        """Configure right-click Final Frame targets: list of (key, display_name)."""
        self._final_frame_targets = targets

    def set_guide_video_targets(self, targets: list[tuple[str, str]]) -> None:
        """Configure right-click Guide Video targets: list of (key, display_name)."""
        self._guide_video_targets = targets

    @Slot()
    def _show_context_menu(self, pos) -> None:
        path = self.selected_path
        if not path:
            return

        from sdqt.widgets.send_targets import build_target_menu

        menu = QMenu(self)
        has_items = False

        if self._send_targets:
            build_target_menu(menu, "Send to", self._send_targets,
                              self.send_requested.emit)
            has_items = True

        if self._final_frame_targets:
            build_target_menu(menu, "Final Frame", self._final_frame_targets,
                              self.final_frame_requested.emit)
            has_items = True

        if self._guide_video_targets:
            build_target_menu(menu, "Guide Video", self._guide_video_targets,
                              self.guide_video_requested.emit)
            has_items = True

        if has_items:
            menu.addSeparator()

        cc_ref_action = menu.addAction("Set as Color Reference")
        cc_ref_action.triggered.connect(lambda _checked, p=path: self.color_ref_requested.emit(p))

        menu.addSeparator()

        pp_action = menu.addAction("Post-Process")
        pp_action.triggered.connect(lambda _checked, p=path: self._run_postprocess(p))

        save_as = menu.addAction("Save Image As...")
        save_as.triggered.connect(self._save_as)
        menu.exec(self._list.mapToGlobal(pos))

    def _run_postprocess(self, path: str) -> None:
        """Run image post-processing: dialog → worker → save prompt."""
        from PySide6.QtWidgets import QDialog, QMessageBox
        from sdqt.tabs.timeline_dialogs import ImagePostProcessDialog

        if not path or not Path(path).is_file():
            QMessageBox.warning(self, "Post-Process", f"Image not found:\n{path}")
            return

        dlg = ImagePostProcessDialog(self.window())
        if dlg.exec() != QDialog.Accepted:
            return
        if not dlg.has_any_enabled():
            QMessageBox.information(self, "Post-Process", "No post-processing options enabled.")
            return

        # Resolve model dirs from AppState via the window hierarchy
        upscaler_dir = ""
        face_models_dir = ""
        main_win = self.window()
        if hasattr(main_win, "state"):
            upscaler_dir = main_win.state.global_config.model_paths.get("upscaler_dir", "")
            face_models_dir = main_win.state.global_config.model_paths.get("face_models_dir", "")

        from sdqt.workers.image_postprocess import ImagePostProcessWorker
        worker = ImagePostProcessWorker(
            source_path=path,
            ai_denoise=dlg.ai_denoise(),
            ai_sharpen=dlg.ai_sharpen(),
            ai_enhance=dlg.ai_enhance(),
            ai_face_restore=dlg.ai_face_restore(),
            ai_tile=dlg.ai_tile_size(),
            spatial=dlg.spatial(),
            upscaler_dir=upscaler_dir,
            face_models_dir=face_models_dir,
            parent=self,
        )

        # Progress → status bar if available
        status_bar = getattr(main_win, "_status_bar", None)
        if status_bar:
            worker.progress.connect(lambda f, d: status_bar.showMessage(d))

        worker.error.connect(
            lambda msg: QMessageBox.critical(self.window(), "Post-Process Error", msg)
        )

        def _on_done(result_path):
            box = QMessageBox(self.window())
            box.setWindowTitle("Save Post-Processed Image")
            box.setText(f"Post-processing complete.\n\n{Path(result_path).name}")
            box.setInformativeText("Save as a copy or overwrite the original?")
            copy_btn = box.addButton("Save as Copy", QMessageBox.AcceptRole)
            overwrite_btn = box.addButton("Overwrite Original", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()

            import shutil
            clicked = box.clickedButton()
            if clicked == overwrite_btn:
                shutil.move(result_path, path)
                final_path = path
            elif clicked == copy_btn:
                final_path = result_path
            else:
                return

            # Reload gallery with result
            paths = list(self._paths)
            if final_path not in paths:
                paths.append(final_path)
            self.load_images(paths)
            if status_bar:
                status_bar.showMessage(f"Saved: {Path(final_path).name}", 5000)

        worker.finished_ok.connect(_on_done)
        worker.finished.connect(worker.deleteLater)
        self._pp_worker = worker
        worker.start()
        if status_bar:
            status_bar.showMessage("Post-processing image...")

    @Slot()
    def _save_as(self) -> None:
        path = self.selected_path
        if not path or not Path(path).is_file():
            return
        default_name = Path(path).name
        dest, _ = get_save_filename(
            self, "Save Image As", default_name, "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if dest:
            import shutil
            shutil.copy2(path, dest)
