"""Library Suite — top-level tab combining all global asset libraries.

Sub-tabs:
  - PNG Library (images — moved from Outputs browser)
  - Voice Library (audio references — moved from Audio suite)
  - Sound Library (SFX and ambient audio)
  - Character Library (body/pose references for Body Double / RePose)
  - Face Library (face reference images for Face Swap)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab
from sdqt.utils.file_dialog import get_open_filenames
from sdqt.widgets.draggable_list import DraggableFileList

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Base: reusable image library tab (file list + image preview + CRUD)
# ═══════════════════════════════════════════════════════════════════════════════

class _ImageLibraryTab(BaseTab):
    """Base class for image-based libraries (PNG, Character, Face)."""

    send_image_requested = Signal(str, str)  # (key, path)

    # Subclasses override:
    _title = "Library"
    _description = "Global library"
    _subdir = "library"          # under ~/.supremediffusion/library/<subdir>
    _extensions = _IMAGE_EXTS
    _file_filter = "Images (*.png *.jpg *.jpeg *.webp)"
    _import_label = "Import"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._files: list[str] = []
        self._send_targets: list = []
        self._build_ui()

    @property
    def _lib_dir(self) -> Path:
        return Path(self.state.global_config.projects_root).parent / "library" / self._subdir

    _THUMB_SIZE = 120

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(4)

        # Header
        layout.addWidget(QLabel(f"<b>{self._title}</b> — {self._description}"))

        # Thumbnail grid
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.IconMode)
        self._list.setIconSize(QSize(self._THUMB_SIZE, self._THUMB_SIZE))
        self._list.setResizeMode(QListWidget.Adjust)
        self._list.setSpacing(8)
        self._list.setWrapping(True)
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setMinimumHeight(300)
        layout.addWidget(self._list, 1)

        # Footer: count + buttons
        footer = QHBoxLayout()
        footer.setSpacing(6)

        self._count_label = QLabel("0 files")
        footer.addWidget(self._count_label)
        footer.addStretch()

        self._upload_btn = QPushButton(self._import_label)
        self._upload_btn.setToolTip(f"Import file(s) to the {self._title.lower()}")
        self._upload_btn.clicked.connect(self._on_upload)
        footer.addWidget(self._upload_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        footer.addWidget(self._refresh_btn)

        layout.addLayout(footer)

        # Hidden preview label kept for backward compat (context menu uses _status)
        self._preview = QLabel()
        self._preview.hide()
        self._status = QLabel("")
        self._status.hide()

    def set_send_targets(self, targets: list) -> None:
        self._send_targets = targets

    # -- Public API -----------------------------------------------------------

    def save_to_library(self, path: str) -> bool:
        """Copy a file into this library. Returns True on success."""
        src = Path(path)
        if not src.is_file():
            return False
        lib = self._lib_dir
        lib.mkdir(parents=True, exist_ok=True)
        dest = lib / src.name
        if dest.exists():
            stem, suffix, idx = src.stem, src.suffix, 1
            while dest.exists():
                dest = lib / f"{stem}_{idx}{suffix}"
                idx += 1
        shutil.copy2(str(src), str(dest))
        self._refresh()
        self._show_status(f"Saved {dest.name}")
        return True

    # -- Scanning -------------------------------------------------------------

    def _refresh(self) -> None:
        self._list.clear()
        self._files = []
        lib = self._lib_dir
        if not lib.is_dir():
            self._count_label.setText("0 files")
            return
        ts = self._THUMB_SIZE
        for f in sorted(lib.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if f.suffix.lower() in self._extensions and f.is_file():
                self._files.append(str(f))
                pixmap = QPixmap(str(f))
                item = QListWidgetItem()
                item.setText(f.stem)
                item.setToolTip(str(f))
                item.setData(Qt.UserRole, str(f))
                if not pixmap.isNull():
                    item.setIcon(QIcon(
                        pixmap.scaled(ts, ts, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    ))
                self._list.addItem(item)
        n = len(self._files)
        self._count_label.setText(f"{n} file{'s' if n != 1 else ''}")

    @Slot(QListWidgetItem)
    def _on_double_click(self, item: QListWidgetItem) -> None:
        """Open the image in a lightbox on double-click."""
        path = item.data(Qt.UserRole)
        if not path or not Path(path).is_file():
            return
        from sdqt.widgets.image_lightbox import ImageLightbox
        lightbox = ImageLightbox(
            path,
            image_paths=self._files,
            send_targets=self._send_targets,
            parent=self.window(),
        )
        lightbox.send_requested.connect(
            lambda key: self.send_image_requested.emit(key, path)
        )
        lightbox.exec()

    # -- Context menu ---------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if not item:
            return
        self._list.setCurrentItem(item)
        path = item.data(Qt.UserRole)
        if not path or not Path(path).is_file():
            return

        menu = QMenu(self)

        if self._send_targets:
            from sdqt.widgets.send_targets import build_target_menu
            build_target_menu(
                menu, "Send to", self._send_targets,
                lambda k, p=path: self.send_image_requested.emit(k, p),
            )
            menu.addSeparator()

        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self._on_rename(path))

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._on_delete(path))

        menu.exec(self._list.mapToGlobal(pos))

    # -- CRUD -----------------------------------------------------------------

    @Slot()
    def _on_upload(self) -> None:
        paths, _ = get_open_filenames(
            self, f"Import to {self._title}", "", self._file_filter)
        if not paths:
            return
        lib = self._lib_dir
        lib.mkdir(parents=True, exist_ok=True)
        count = 0
        for p in paths:
            src = Path(p)
            if not src.is_file() or src.suffix.lower() not in self._extensions:
                continue
            dest = lib / src.name
            if dest.exists():
                stem, suffix, idx = src.stem, src.suffix, 1
                while dest.exists():
                    dest = lib / f"{stem}_{idx}{suffix}"
                    idx += 1
            shutil.copy2(str(src), str(dest))
            count += 1
        self._refresh()
        self._show_status(f"Imported {count} file(s).")

    def _on_rename(self, path: str) -> None:
        old = Path(path)
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=old.stem)
        if not ok or not new_name.strip():
            return
        new_path = old.parent / f"{new_name.strip()}{old.suffix}"
        if new_path.exists() and new_path != old:
            QMessageBox.warning(self, "Error", f"'{new_path.name}' already exists.")
            return
        old.rename(new_path)
        self._refresh()
        self._show_status(f"Renamed to {new_path.name}")

    def _on_delete(self, path: str) -> None:
        p = Path(path)
        reply = QMessageBox.question(
            self, "Delete", f"Delete '{p.name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                p.unlink()
                self._refresh()
                self._show_status(f"Deleted {p.name}")
            except Exception as exc:
                self._show_status(f"Error: {exc}")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._refresh()


# ═══════════════════════════════════════════════════════════════════════════════
#  Base: reusable audio library tab (file list + audio preview + CRUD)
# ═══════════════════════════════════════════════════════════════════════════════

class _AudioLibraryTab(BaseTab):
    """Base class for audio-based libraries (Voice, Sound)."""

    send_audio_requested = Signal(str, str)  # (key, path)

    _title = "Audio Library"
    _description = "Global audio library"
    _subdir = "audio"
    _extensions = _AUDIO_EXTS
    _file_filter = "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)"
    _import_label = "Import"

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._files: list[str] = []
        self._build_ui()

    @property
    def _lib_dir(self) -> Path:
        return Path(self.state.global_config.projects_root).parent / "library" / self._subdir

    def _build_ui(self) -> None:
        from sdqt.widgets.audio_player import AudioPlayerWidget

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left: file list --------------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        left_layout.addWidget(QLabel(f"<b>{self._title}</b> — {self._description}"))

        self._list = DraggableFileList()
        self._list.setSelectionMode(DraggableFileList.SingleSelection)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self._list, 1)

        self._count_label = QLabel("0 files")
        left_layout.addWidget(self._count_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._upload_btn = QPushButton(self._import_label)
        self._upload_btn.setToolTip(f"Import file(s) to the {self._title.lower()}")
        self._upload_btn.clicked.connect(self._on_upload)
        btn_row.addWidget(self._upload_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # -- Right: audio preview ---------------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._audio_preview = AudioPlayerWidget(f"{self._title} Preview")
        self._audio_preview.send_audio_requested.connect(self.send_audio_requested)
        right_layout.addWidget(self._audio_preview)

        self._status = QLabel("")
        right_layout.addWidget(self._status)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

    def set_audio_send_targets(self, targets: list) -> None:
        self._audio_preview.set_send_targets(targets)

    # -- Public API -----------------------------------------------------------

    def save_to_library(self, path: str) -> bool:
        """Copy an audio file into this library. Returns True on success."""
        src = Path(path)
        if not src.is_file():
            return False
        lib = self._lib_dir
        lib.mkdir(parents=True, exist_ok=True)
        dest = lib / src.name
        if dest.exists():
            stem, suffix, idx = src.stem, src.suffix, 1
            while dest.exists():
                dest = lib / f"{stem}_{idx}{suffix}"
                idx += 1
        shutil.copy2(str(src), str(dest))
        self._refresh()
        self._show_status(f"Saved {dest.name}")
        return True

    def load_audio(self, path: str) -> None:
        if path and Path(path).is_file():
            self._audio_preview.load_audio(path)

    # -- Scanning -------------------------------------------------------------

    def _refresh(self) -> None:
        self._list.clear()
        self._files = []
        lib = self._lib_dir
        if not lib.is_dir():
            self._count_label.setText("0 files")
            return
        for f in sorted(lib.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if f.suffix.lower() in self._extensions and f.is_file():
                self._files.append(str(f))
                self._list.addItem(f.name)
        n = len(self._files)
        self._count_label.setText(f"{n} file{'s' if n != 1 else ''}")

    @Slot(int)
    def _on_selection_changed(self, row: int) -> None:
        if 0 <= row < len(self._files):
            self._audio_preview.load_audio(self._files[row])

    # -- Context menu ---------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if not item:
            return
        self._list.setCurrentItem(item)
        row = self._list.row(item)
        if row < 0 or row >= len(self._files):
            return
        path = self._files[row]

        menu = QMenu(self)

        from sdqt.widgets.send_targets import AUDIO_TARGETS, build_target_menu
        build_target_menu(
            menu, "Send to", AUDIO_TARGETS,
            lambda k, p=path: self.send_audio_requested.emit(k, p),
        )
        menu.addSeparator()

        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self._on_rename(path))

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._on_delete(path))

        menu.exec(self._list.mapToGlobal(pos))

    # -- CRUD -----------------------------------------------------------------

    @Slot()
    def _on_upload(self) -> None:
        paths, _ = get_open_filenames(
            self, f"Import to {self._title}", "", self._file_filter)
        if not paths:
            return
        lib = self._lib_dir
        lib.mkdir(parents=True, exist_ok=True)
        count = 0
        for p in paths:
            src = Path(p)
            if not src.is_file() or src.suffix.lower() not in self._extensions:
                continue
            dest = lib / src.name
            if dest.exists():
                stem, suffix, idx = src.stem, src.suffix, 1
                while dest.exists():
                    dest = lib / f"{stem}_{idx}{suffix}"
                    idx += 1
            shutil.copy2(str(src), str(dest))
            count += 1
        self._refresh()
        self._show_status(f"Imported {count} file(s).")

    def _on_rename(self, path: str) -> None:
        old = Path(path)
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=old.stem)
        if not ok or not new_name.strip():
            return
        new_path = old.parent / f"{new_name.strip()}{old.suffix}"
        if new_path.exists() and new_path != old:
            QMessageBox.warning(self, "Error", f"'{new_path.name}' already exists.")
            return
        # Release file and block signals to prevent re-entrancy during rename
        self._audio_preview.clear_audio()
        self._list.blockSignals(True)
        old.rename(new_path)
        self._refresh()
        self._list.blockSignals(False)
        # Re-select the renamed file
        for i, f in enumerate(self._files):
            if f == str(new_path):
                self._list.setCurrentRow(i)
                break
        self._show_status(f"Renamed to {new_path.name}")

    def _on_delete(self, path: str) -> None:
        p = Path(path)
        # Release file before deleting — QMediaPlayer holds a lock
        self._audio_preview.clear_audio()
        reply = QMessageBox.question(
            self, "Delete", f"Delete '{p.name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self._list.blockSignals(True)
                p.unlink()
                self._refresh()
                self._show_status(f"Deleted {p.name}")
            except Exception as exc:
                self._show_status(f"Error: {exc}")
            finally:
                self._list.blockSignals(False)

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._refresh()


# ═══════════════════════════════════════════════════════════════════════════════
#  Concrete library tabs
# ═══════════════════════════════════════════════════════════════════════════════

class PNGLibraryTab(_ImageLibraryTab):
    _title = "PNG Library"
    _description = "image assets for compositing and generation"
    _subdir = "images"
    _extensions = {".png"}
    _file_filter = "PNG Images (*.png)"
    _import_label = "Import PNG"

    @property
    def _lib_dir(self) -> Path:
        d = getattr(self.state.global_config, "png_library_dir", "")
        if d:
            return Path(d)
        return super()._lib_dir


class VoiceLibraryTab(_AudioLibraryTab):
    _title = "Voice Library"
    _description = "saved voices available across all projects"
    _subdir = "voices"
    _import_label = "Import Voice"


class SoundLibraryTab(_AudioLibraryTab):
    _title = "Sound Library"
    _description = "SFX and ambient audio clips"
    _subdir = "sounds"
    _import_label = "Import Sound"


class CharacterLibraryTab(_ImageLibraryTab):
    _title = "Character Library"
    _description = "body and pose references for Body Double / RePose"
    _subdir = "characters"
    _import_label = "Import Character"

    @property
    def _lib_dir(self) -> Path:
        d = getattr(self.state.global_config, "characters_dir", "")
        if d:
            return Path(d)
        return super()._lib_dir

    def _refresh(self) -> None:
        """Override to scan character subdirectories (each has base.png + poses/)."""
        self._list.clear()
        self._files = []
        lib = self._lib_dir
        if not lib.is_dir():
            self._count_label.setText("0 characters")
            return
        ts = self._THUMB_SIZE
        for d in sorted(lib.iterdir()):
            if not d.is_dir():
                continue
            # Look for base.png or character.json to confirm it's a character
            base = d / "base.png"
            meta = d / "character.json"
            if not base.is_file() and not meta.is_file():
                continue
            # Load name from metadata if available
            name = d.name
            if meta.is_file():
                try:
                    import json as _json
                    m = _json.loads(meta.read_text())
                    name = m.get("name", d.name)
                except Exception:
                    pass
            # Use base.png as thumbnail
            item = QListWidgetItem()
            item.setText(name)
            item.setToolTip(str(d))
            item.setData(Qt.UserRole, str(base) if base.is_file() else str(d))
            if base.is_file():
                pm = QPixmap(str(base))
                if not pm.isNull():
                    item.setIcon(QIcon(
                        pm.scaled(ts, ts, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    ))
            self._list.addItem(item)
            self._files.append(str(base) if base.is_file() else "")
        n = self._list.count()
        self._count_label.setText(f"{n} character{'s' if n != 1 else ''}")

    @Slot(QListWidgetItem)
    def _on_double_click(self, item: QListWidgetItem) -> None:
        """Open character folder — show all images (base + poses) in lightbox."""
        tooltip = item.toolTip()  # character directory path
        char_dir = Path(tooltip)
        if not char_dir.is_dir():
            path = item.data(Qt.UserRole)
            if path and Path(path).is_file():
                from sdqt.widgets.image_lightbox import ImageLightbox
                ImageLightbox(path, parent=self.window()).exec()
            return
        # Collect all images from the character directory
        all_images = []
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self._extensions:
                all_images.append(str(f))
        poses_dir = char_dir / "poses"
        if poses_dir.is_dir():
            for f in sorted(poses_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in self._extensions:
                    all_images.append(str(f))
        if all_images:
            from sdqt.widgets.image_lightbox import ImageLightbox
            lightbox = ImageLightbox(
                all_images[0], image_paths=all_images,
                send_targets=self._send_targets,
                extra_menu_builder=self._build_postprocess_menu,
                parent=self.window(),
            )
            lightbox.send_requested.connect(
                lambda key: self.send_image_requested.emit(key, lightbox.current_path)
            )
            lightbox.exec()

    def _build_postprocess_menu(self, menu, image_path: str, lightbox) -> None:
        """Add post-processing actions to the lightbox right-click menu."""
        from PySide6.QtWidgets import QMenu as _QMenu

        # Send-to (categorised)
        if self._send_targets:
            from sdqt.widgets.send_targets import build_target_menu
            build_target_menu(
                menu, "Send to", self._send_targets,
                lambda k: self.send_image_requested.emit(k, lightbox.current_path),
            )
            menu.addSeparator()

        # Post-processing submenu
        pp_menu = menu.addMenu("Post Processing")

        # Face Restore (uses .pth models via spandrel)
        restore_menu = pp_menu.addMenu("Face Restore")
        _RESTORE_MODELS = {
            "GFPGAN v1.4": "GFPGANv1.4.pth",
            "CodeFormer v0.1": "codeformer-v0.1.0.pth",
        }
        face_dir = Path(self.state.global_config.model_paths.get("face_models_dir", ""))
        for label, filename in _RESTORE_MODELS.items():
            act = restore_menu.addAction(label)
            act.triggered.connect(
                lambda checked, fn=filename: self._run_face_restore(
                    lightbox.current_path, fn, lightbox)
            )

        # AI Upscale
        upscale_act = pp_menu.addAction("AI Upscale (2x)")
        upscale_act.triggered.connect(
            lambda: self._run_ai_upscale(lightbox.current_path, lightbox)
        )

        # Save As
        menu.addSeparator()
        save_act = menu.addAction("Save Image As...")
        save_act.triggered.connect(
            lambda: self._save_lightbox_image(lightbox.current_path)
        )

    def _run_face_restore(self, image_path: str, model_filename: str, lightbox) -> None:
        """Run face restore on the current image using spandrel (.pth models)."""
        try:
            import numpy as np
            from PIL import Image as PILImage

            # Find model — check face dir first, then upscaler dir
            face_dir = self.state.global_config.model_paths.get("face_models_dir", "")
            upscaler_dir = self.state.global_config.model_paths.get("upscaler_dir", "")
            model_path = ""
            for d in (face_dir, upscaler_dir):
                if d and (Path(d) / model_filename).is_file():
                    model_path = str(Path(d) / model_filename)
                    break

            if not model_path:
                # Try auto-download
                reply = QMessageBox.question(
                    lightbox, "Download Model",
                    f"{model_filename} not found.\n\nDownload it now?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                target_dir = face_dir or upscaler_dir
                if not target_dir:
                    self._show_status("No model directory configured.")
                    return
                self._show_status(f"Downloading {model_filename}...")
                try:
                    from supremediffusion.postprocessing.ai_upscale import download_model
                    model_path = download_model(
                        target_dir, model_filename,
                        progress_cb=lambda f, d: self._show_status(d),
                    )
                except FileNotFoundError:
                    # No URL in the upscaler registry — try known URLs
                    _FACE_RESTORE_URLS = {
                        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                        "codeformer-v0.1.0.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
                    }
                    url = _FACE_RESTORE_URLS.get(model_filename)
                    if not url:
                        self._show_status(f"No download URL for {model_filename}")
                        return
                    import urllib.request
                    dest = Path(target_dir) / model_filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    self._show_status(f"Downloading {model_filename}...")
                    urllib.request.urlretrieve(url, str(dest))
                    model_path = str(dest)
                    self._show_status(f"Downloaded {model_filename}")

            # Run face restore
            self._show_status("Running face restore...")
            from supremediffusion.postprocessing.ai_upscale import face_restore_frames

            img = np.array(PILImage.open(image_path).convert("RGB"))
            frames = np.expand_dims(img, 0)  # [1, H, W, 3]
            result = face_restore_frames(frames, model_path)

            src = Path(image_path)
            out_path = src.parent / f"{src.stem}_restored{src.suffix}"
            PILImage.fromarray(result[0]).save(str(out_path))
            self._show_status(f"Face restored: {out_path.name}")

            # Show in lightbox
            lightbox._paths.append(str(out_path))
            lightbox._index = len(lightbox._paths) - 1
            lightbox._load_current()
        except Exception as exc:
            self._show_status(f"Face restore failed: {exc}")
            logger.warning("Face restore failed: %s", exc, exc_info=True)

    def _run_ai_upscale(self, image_path: str, lightbox) -> None:
        """Run AI upscale on the current image. Auto-downloads model if missing."""
        try:
            from supremediffusion.postprocessing.ai_upscale import upscale_frames, _resolve_model
            import numpy as np
            from PIL import Image as PILImage

            models_dir = self.state.global_config.model_paths.get("upscaler_dir", "")
            if not models_dir:
                models_dir = str(Path(self.state.global_config.models_root) / "upscalers")

            self._show_status("Preparing upscale model (downloading if needed)...")
            model_path = _resolve_model(
                models_dir, "RealESRGAN_x2plus.pth",
                progress_cb=lambda f, d: self._show_status(d),
            )

            self._show_status("Upscaling...")
            img = np.array(PILImage.open(image_path).convert("RGB"))
            frames = np.expand_dims(img, 0)
            result = upscale_frames(frames, model_path, tile_size=512)

            src = Path(image_path)
            out_path = src.parent / f"{src.stem}_upscaled{src.suffix}"
            PILImage.fromarray(result[0]).save(str(out_path))
            self._show_status(f"Upscaled: {out_path.name}")

            lightbox._paths.append(str(out_path))
            lightbox._index = len(lightbox._paths) - 1
            lightbox._load_current()
        except Exception as exc:
            self._show_status(f"AI upscale failed: {exc}")
            logger.warning("AI upscale failed: %s", exc, exc_info=True)

    def _save_lightbox_image(self, image_path: str) -> None:
        """Save the current lightbox image to a user-chosen location."""
        from sdqt.utils.file_dialog import get_save_filename
        dest, _ = get_save_filename(
            self, "Save Image As", Path(image_path).name,
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if dest:
            shutil.copy2(image_path, dest)
            self._show_status(f"Saved: {Path(dest).name}")


class FaceLibraryTab(_ImageLibraryTab):
    _title = "Face Library"
    _description = "face references for Face Swap"
    _subdir = "faces"
    _import_label = "Import Face"

    @property
    def _lib_dir(self) -> Path:
        d = getattr(self.state.global_config, "face_library_dir", "")
        if d:
            return Path(d)
        return super()._lib_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  LibrarySuiteTab — top-level container
# ═══════════════════════════════════════════════════════════════════════════════

class LibrarySuiteTab(BaseTab):
    """Top-level Library tab combining all global asset libraries."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._sub_tabs: list[BaseTab] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()

        self._png_library = PNGLibraryTab(self.state)
        self._tab_widget.addTab(self._png_library, "PNG Library")

        self._voice_library = VoiceLibraryTab(self.state)
        self._tab_widget.addTab(self._voice_library, "Voice Library")

        self._sound_library = SoundLibraryTab(self.state)
        self._tab_widget.addTab(self._sound_library, "Sound Library")

        self._character_library = CharacterLibraryTab(self.state)
        self._tab_widget.addTab(self._character_library, "Character Library")

        self._face_library = FaceLibraryTab(self.state)
        self._tab_widget.addTab(self._face_library, "Face Library")

        self._sub_tabs = [
            self._png_library, self._voice_library, self._sound_library,
            self._character_library, self._face_library,
        ]

        layout.addWidget(self._tab_widget)

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        for tab in self._sub_tabs:
            tab.on_project_changed(project_name)

    # -- Public API for cross-tab wiring --------------------------------------

    def save_to_voice_library(self, path: str) -> bool:
        return self._voice_library.save_to_library(path)

    def save_to_sound_library(self, path: str) -> bool:
        return self._sound_library.save_to_library(path)

    def save_to_png_library(self, path: str) -> bool:
        return self._png_library.save_to_library(path)

    def save_to_face_library(self, path: str) -> bool:
        return self._face_library.save_to_library(path)

    def save_to_character_library(self, path: str) -> bool:
        return self._character_library.save_to_library(path)
