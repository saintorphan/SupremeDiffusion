"""Output Browser tab -- browse and manage generated files."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from sdqt.widgets.send_targets import DISABLED_TARGETS

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filenames

from sdqt.state import AppState
from sdqt.widgets.audio_player import AudioPlayerWidget
from sdqt.widgets.video_player import VideoPlayerWidget

from .base import BaseTab

logger = logging.getLogger(__name__)


def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    if nbytes < 1024:
        return f"{nbytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        nbytes /= 1024
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
    return f"{nbytes:.1f} PB"

_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
_AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

# (tab_label, subdir_relative_to_project, file_extensions, category)
# category: "video", "image", "face"
_BROWSER_GROUPS = [
    ("Video Gen", [
        ("Img2Vid",     "clips/generate",      _VIDEO_EXTS, "video"),
        ("Extend",      "clips/extend",        _VIDEO_EXTS, "video"),
        ("Longshot",    "clips/longshot",       _VIDEO_EXTS, "video"),
        ("Trims",       "clips/trims",          _VIDEO_EXTS, "video"),
        ("Lip Sync",    "clips/lipsync",        _VIDEO_EXTS, "video"),
        ("Timeline",    "clips/timeline",       _VIDEO_EXTS, "video"),
        ("Combined",    "clips/combined",       _VIDEO_EXTS, "video"),
        ("3D Renders",  "clips/3d_renders",     _VIDEO_EXTS, "video"),
        ("Outputs",     "outputs",              _VIDEO_EXTS, "video"),
    ]),
    ("Image Gen", [
        ("Txt2Img",     "images/txt2img",       _IMAGE_EXTS, "image"),
        ("Img2Img",     "images/img2img",       _IMAGE_EXTS, "image"),
        ("Inpaint",     "images/inpaint",       _IMAGE_EXTS, "image"),
        ("ControlNet",  "images/controlnet",    _IMAGE_EXTS, "image"),
        ("Face Swap",   "images/faceswap",      _IMAGE_EXTS, "image"),
        ("Body Double", "images/bodydouble",    _IMAGE_EXTS, "image"),
        ("RePose",      "images/repose",        _IMAGE_EXTS, "image"),
        ("Crop/Zoom",   "images/crops",         _IMAGE_EXTS, "image"),
        ("Draw",        "images",               _IMAGE_EXTS, "image"),
    ]),
    ("Other", [
        ("Frames",      "frames",               _IMAGE_EXTS, "image"),
        ("Faces",       "faces",                _IMAGE_EXTS, "face"),
        ("Orpheus TTS", "audio/orpheus",        _AUDIO_EXTS, "audio"),
        ("OpenVoice",   "audio/openvoice",      _AUDIO_EXTS, "audio"),
        ("RVC",         "audio/rvc",            _AUDIO_EXTS, "audio"),
    ]),
]

# Flat list for backward compat (used by _refresh, _on_selection_changed)
_BROWSER_SECTIONS = []
for _group_name, _sections in _BROWSER_GROUPS:
    _BROWSER_SECTIONS.extend(_sections)

# -- Shared context-menu target definitions --------------------------------

from sdqt.widgets.send_targets import (
    CLIP_TARGETS as _CLIP_TARGETS,
    IMAGE_TARGETS as _FRAME_TARGETS,
    IMAGE_TARGETS as _IMAGE_SEND_TARGETS,
    IMAGE_TARGETS_SIMPLE as _PNG_LIB_SEND_TARGETS,
    FINAL_FRAME_TARGETS as _FINAL_FRAME_TARGETS,
    GUIDE_VIDEO_TARGETS as _GUIDE_VIDEO_TARGETS,
    AUDIO_TARGETS as _AUDIO_SEND_TARGETS,
)

_FACE_SEND_TARGETS = [
    ("Face / Body", [
        ("faceswap", "Face Swap"),
    ]),
    ("Generate", [
        ("inpaint", "Inpaint"),
    ]),
    ("Edit", [
        ("imgedit", "Draw"),
    ]),
]

_PREVIEW_HEIGHT = 320


# =========================================================================
#  FileBrowserWidget — reusable list for a single project subdirectory
# =========================================================================

class FileBrowserWidget(QWidget):
    """Generic file browser for a project subdirectory."""

    send_clip_requested = Signal(str, str)   # (key, file_path)
    send_image_requested = Signal(str, str)  # (key, file_path)
    send_audio_requested = Signal(str, str)  # (key, file_path)
    rename_requested = Signal(str)           # file_path
    delete_requested = Signal(str)           # file_path
    color_ref_requested = Signal(str)        # file path for Color Correct tab

    def __init__(self, label: str, extensions: list[str],
                 category: str, parent=None) -> None:
        super().__init__(parent)
        self._extensions = extensions
        self._category = category
        self._files: list[str] = []
        self._dir_size: int = 0
        self._scan_dir: Path | None = None
        self._state = None  # AppState, injected by BrowserTab via set_state()

    def set_state(self, state) -> None:
        """Inject AppState so right-click menus can run Export / Quick Export."""
        self._state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_list_context_menu)
        layout.addWidget(self._list, 1)

        self._count_label = QLabel("0 files")
        layout.addWidget(self._count_label)

    @property
    def dir_size(self) -> int:
        """Total size in bytes of matched files in the last scan."""
        return self._dir_size

    @property
    def scan_dir(self) -> Path | None:
        return self._scan_dir

    def scan(self, directory: Path) -> None:
        self._list.clear()
        self._files = []
        self._dir_size = 0
        self._scan_dir = directory
        if not directory.is_dir():
            self._count_label.setText("0 files")
            return
        # Use os.scandir() — caches stat info, avoids extra syscalls
        import os
        entries: list[tuple[float, str, str, int]] = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        suffix = Path(entry.name).suffix.lower()
                        if suffix in self._extensions:
                            st = entry.stat()
                            entries.append((st.st_mtime, str(entry.path), entry.name, st.st_size))
        except OSError:
            pass
        entries.sort(key=lambda e: e[0], reverse=True)
        total = 0
        for _mtime, fpath, fname, fsize in entries:
            self._files.append(fpath)
            self._list.addItem(fname)
            total += fsize
        self._dir_size = total
        self._count_label.setText(
            f"{len(self._files)} files — {_format_size(total)}"
        )

    @property
    def selected_path(self) -> str | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    def path_at(self, pos) -> str | None:
        item = self._list.itemAt(pos)
        if item is None:
            return None
        row = self._list.row(item)
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    @property
    def file_list(self) -> QListWidget:
        return self._list

    # -- Context menu on list items ----------------------------------------

    def _show_list_context_menu(self, pos) -> None:
        path = self.path_at(pos)
        if not path:
            return
        item = self._list.itemAt(pos)
        if item:
            self._list.setCurrentItem(item)

        menu = QMenu(self)

        if self._category == "video":
            self._add_clip_send_menu(menu, path)
            self._add_frame_send_menu(menu, path)
            menu.addSeparator()
        elif self._category == "image":
            self._add_image_send_menu(menu, path, _IMAGE_SEND_TARGETS)
            menu.addSeparator()
        elif self._category == "face":
            self._add_image_send_menu(menu, path, _FACE_SEND_TARGETS)
            menu.addSeparator()
        elif self._category == "audio":
            self._add_audio_send_menu(menu, path)
            menu.addSeparator()

        if self._category in ("video", "image", "face"):
            cc_ref_act = menu.addAction("Set as Color Reference")
            cc_ref_act.triggered.connect(
                lambda _checked, p=path: self._emit_color_ref(p))
            menu.addSeparator()

        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self.rename_requested.emit(path))

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self.delete_requested.emit(path))

        # Export / Quick Export / Create Loop for video files
        if self._category == "video" and self._state is not None:
            from sdqt.tabs.clip_menu_actions import add_clip_export_actions
            add_clip_export_actions(
                menu, self, self._state, path,
                media_offset=0.0, duration=0.0,
                clip_name=Path(path).stem,
            )

        menu.exec(self._list.mapToGlobal(pos))

    def _add_clip_send_menu(self, menu: QMenu, path: str) -> None:
        from sdqt.widgets.send_targets import build_target_menu
        build_target_menu(menu, "Send clip to", _CLIP_TARGETS,
                          lambda k, p=path: self.send_clip_requested.emit(k, p))

    def _add_frame_send_menu(self, menu: QMenu, path: str) -> None:
        from sdqt.widgets.send_targets import build_target_menu
        build_target_menu(menu, "Send frame to", _FRAME_TARGETS,
                          lambda k, p=path: self.send_image_requested.emit(k, p))

    def _add_image_send_menu(self, menu: QMenu, path: str, targets: list) -> None:
        from sdqt.widgets.send_targets import build_target_menu
        build_target_menu(menu, "Send to", targets,
                          lambda k, p=path: self.send_image_requested.emit(k, p))

    def _add_audio_send_menu(self, menu: QMenu, path: str) -> None:
        from sdqt.widgets.send_targets import build_target_menu
        build_target_menu(menu, "Send to", _AUDIO_SEND_TARGETS,
                          lambda k, p=path: self.send_audio_requested.emit(k, p))
        voice_lib_action = menu.addAction("Save to Voice Library")
        voice_lib_action.triggered.connect(
            lambda checked, p=path: self.send_audio_requested.emit("voice_library", p))

    def _emit_color_ref(self, path: str) -> None:
        """Emit color reference — extract first frame for videos, pass images directly."""
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
            self.color_ref_requested.emit(path)
        else:
            import tempfile
            from sdqt.utils.grade_compute import extract_frame_from_path
            frame = extract_frame_from_path(path, 0)
            if frame is not None:
                from PIL import Image
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                Image.fromarray(frame).save(tmp.name)
                self.color_ref_requested.emit(tmp.name)


# =========================================================================
#  _ImagePreviewLabel — image preview with right-click send-to
# =========================================================================

class _ImagePreviewLabel(QLabel):
    """Image preview label with right-click context menu for send-to targets."""

    send_requested = Signal(str)  # key

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_path: str | None = None
        self._send_targets: list[tuple[str, str]] = []
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_send_targets(self, targets: list[tuple[str, str]]) -> None:
        self._send_targets = targets

    def set_current_path(self, path: str | None) -> None:
        self._current_path = path

    def _show_context_menu(self, pos) -> None:
        if not self._current_path or not self._send_targets:
            return
        menu = QMenu(self)
        send_menu = menu.addMenu("Send to")
        for key, name in self._send_targets:
            action = send_menu.addAction(name)
            if key in DISABLED_TARGETS:
                action.setEnabled(False)
            else:
                action.triggered.connect(
                    lambda checked, k=key: self.send_requested.emit(k))
        menu.exec(self.mapToGlobal(pos))


# =========================================================================
#  _DropListWidget — QListWidget that accepts PNG file drops
# =========================================================================

class _DropListWidget(QListWidget):
    """QListWidget that accepts drag-drop of PNG files."""

    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() == ".png":
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() == ".png" and Path(p).is_file():
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()


# =========================================================================
#  _GenerationsSubTab — the module-based file browser (formerly the whole tab)
# =========================================================================

class _GenerationsSubTab(QWidget):
    """Generations browser: per-module tabs with video/image preview."""

    send_clip_requested = Signal(str, str)
    send_frame_requested = Signal(str)
    send_image_requested = Signal(str, str)
    send_audio_requested = Signal(str, str)

    def __init__(self, state: AppState, get_project_path, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._get_project_path = get_project_path
        self._browsers: list[FileBrowserWidget] = []
        self._section_dirs: list[str] = []
        self._section_cats: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # Left: file lists
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._section_tabs = QTabWidget()

        # Build grouped tabs: parent tab → nested QTabWidget with sub-sections
        self._group_tabs: list[QTabWidget] = []
        for group_name, sections in _BROWSER_GROUPS:
            group_tw = QTabWidget()
            for label, subdir, exts, cat in sections:
                browser = FileBrowserWidget(label, exts, cat)
                browser.set_state(self._state)
                browser.file_list.currentRowChanged.connect(self._on_selection_changed)
                browser.send_clip_requested.connect(self.send_clip_requested)
                browser.send_image_requested.connect(self.send_image_requested)
                browser.send_audio_requested.connect(self.send_audio_requested)
                browser.rename_requested.connect(self._on_rename_file)
                browser.delete_requested.connect(self._on_delete_file)
                group_tw.addTab(browser, label)
                self._browsers.append(browser)
                self._section_dirs.append(subdir)
                self._section_cats.append(cat)
            group_tw.currentChanged.connect(self._on_section_changed)
            self._section_tabs.addTab(group_tw, group_name)
            self._group_tabs.append(group_tw)

        self._section_tabs.currentChanged.connect(self._on_group_changed)
        left_layout.addWidget(self._section_tabs, 1)
        splitter.addWidget(left)

        # Right: preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._video_preview = VideoPlayerWidget("Video Preview")
        self._video_preview.setFixedHeight(_PREVIEW_HEIGHT)
        self._video_preview.set_send_targets(
            clip_targets=_CLIP_TARGETS,
            frame_targets=_FRAME_TARGETS,
            final_frame_targets=_FINAL_FRAME_TARGETS,
        )
        self._video_preview.set_guide_video_targets(_GUIDE_VIDEO_TARGETS)
        self._video_preview.enable_color_ref_action(True)
        right_layout.addWidget(self._video_preview)

        self._image_preview = _ImagePreviewLabel()
        self._image_preview.setAlignment(Qt.AlignCenter)
        self._image_preview.setFixedHeight(_PREVIEW_HEIGHT)
        self._image_preview.setStyleSheet("background: #1a1a1a;")
        self._image_preview.set_send_targets(_IMAGE_SEND_TARGETS)
        self._image_preview.send_requested.connect(self._on_image_preview_send)
        right_layout.addWidget(self._image_preview)

        self._audio_preview = AudioPlayerWidget("Audio Preview")
        self._audio_preview.set_send_targets(_AUDIO_SEND_TARGETS)
        self._audio_preview.send_audio_requested.connect(self.send_audio_requested)
        self._audio_preview.setVisible(False)
        right_layout.addWidget(self._audio_preview)

        # Generation metadata panel (shows prompt, settings for selected image)
        self._meta_panel = QGroupBox("Generation Info")
        meta_layout = QVBoxLayout(self._meta_panel)
        meta_layout.setContentsMargins(4, 4, 4, 4)
        self._meta_text = QTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setMaximumHeight(180)
        self._meta_text.setStyleSheet("background: #1a1a1a; font-size: 13px;")
        self._meta_text.setPlaceholderText("Select an image to see generation parameters...")
        meta_layout.addWidget(self._meta_text)
        self._meta_panel.setVisible(False)
        right_layout.addWidget(self._meta_panel)

        btn_row = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self._refresh_btn)

        self._cleanup_btn = QPushButton("Clean Up Directory")
        self._cleanup_btn.setToolTip("Delete all files in the active output directory")
        self._cleanup_btn.clicked.connect(self._on_cleanup)
        btn_row.addWidget(self._cleanup_btn)

        btn_row.addStretch()

        self._total_size_label = QLabel("")
        self._total_size_label.setStyleSheet("color: #999; font-size: 13px; padding: 0 6px;")
        btn_row.addWidget(self._total_size_label)
        right_layout.addLayout(btn_row)

        # Face management (only visible on Faces tab)
        self._face_mgmt_group = QGroupBox("Face management")
        fm_layout = QHBoxLayout(self._face_mgmt_group)
        self._upload_face_btn = QPushButton("Upload Face")
        self._upload_face_btn.clicked.connect(self._on_upload_face)
        fm_layout.addWidget(self._upload_face_btn)
        self._face_mgmt_group.setVisible(False)
        right_layout.addWidget(self._face_mgmt_group)

        self._status = QLabel("")
        right_layout.addWidget(self._status)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    @property
    def _project_path(self) -> Path:
        return self._get_project_path()

    def _refresh(self) -> None:
        pp = self._project_path
        for browser, subdir in zip(self._browsers, self._section_dirs):
            browser.scan(pp / subdir)
        self._update_total_size()

    def _update_total_size(self) -> None:
        total = sum(b.dir_size for b in self._browsers)
        self._total_size_label.setText(f"Project outputs: {_format_size(total)}")

    def _active_browser_index(self) -> int:
        """Get the flat index of the currently active browser across all groups."""
        group_idx = self._section_tabs.currentIndex()
        if group_idx < 0 or group_idx >= len(self._group_tabs):
            return -1
        group_tw = self._group_tabs[group_idx]
        sub_idx = group_tw.currentIndex()
        if sub_idx < 0:
            return -1
        # Count browsers in preceding groups
        offset = 0
        for gi, (_, sections) in enumerate(_BROWSER_GROUPS):
            if gi == group_idx:
                return offset + sub_idx
            offset += len(sections)
        return -1

    @Slot(int)
    def _on_group_changed(self, idx: int) -> None:
        """Parent group tab changed — update preview visibility."""
        self._on_section_changed(0)

    @Slot(int)
    def _on_section_changed(self, idx: int) -> None:
        flat_idx = self._active_browser_index()
        cat = self._section_cats[flat_idx] if 0 <= flat_idx < len(self._section_cats) else ""
        self._face_mgmt_group.setVisible(cat == "face")
        is_audio = cat == "audio"
        self._video_preview.setVisible(not is_audio)
        self._image_preview.setVisible(not is_audio)
        self._audio_preview.setVisible(is_audio)
        if is_audio:
            self._audio_preview.clear_audio()
        if cat == "face":
            self._image_preview.set_send_targets(_FACE_SEND_TARGETS)
        else:
            self._image_preview.set_send_targets(_IMAGE_SEND_TARGETS)

    @Slot(int)
    def _on_selection_changed(self, row: int) -> None:
        flat_idx = self._active_browser_index()
        if flat_idx < 0 or flat_idx >= len(self._browsers):
            return
        browser = self._browsers[flat_idx]
        path = browser.selected_path
        if not path:
            return

        ext = Path(path).suffix.lower()
        is_video = ext in _VIDEO_EXTS
        is_image = ext in _IMAGE_EXTS
        is_audio = ext in _AUDIO_EXTS

        if is_video:
            self._video_preview.load_video(path)
            self._video_preview.setVisible(True)
            self._image_preview.clear()
            self._image_preview.set_current_path(None)
            self._image_preview.setVisible(True)
            self._audio_preview.setVisible(False)
        elif is_image:
            self._video_preview.load_video(None)
            self._video_preview.setVisible(True)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                self._image_preview.setPixmap(
                    pixmap.scaled(self._image_preview.size(),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self._image_preview.set_current_path(path)
            self._image_preview.setVisible(True)
            self._audio_preview.setVisible(False)
        elif is_audio:
            self._video_preview.setVisible(False)
            self._image_preview.setVisible(False)
            self._audio_preview.setVisible(True)
            self._audio_preview.load_audio(path)

        # Show generation metadata for PNG images
        self._load_metadata(path if is_image else None)

    def _load_metadata(self, path: str | None) -> None:
        """Read and display PNG generation metadata for the selected image."""
        if not path or not path.lower().endswith(".png"):
            self._meta_panel.setVisible(False)
            self._meta_text.clear()
            return
        try:
            from PIL import Image
            img = Image.open(path)
            params = img.info.get("parameters", "")
            if params:
                # Format nicely with HTML
                lines = params.replace("\n", "<br>")
                html = f"<pre style='color: #ccc; font-size: 13px;'>{lines}</pre>"
                self._meta_text.setHtml(html)
                self._meta_panel.setVisible(True)
            else:
                self._meta_panel.setVisible(False)
                self._meta_text.clear()
        except Exception:
            self._meta_panel.setVisible(False)
            self._meta_text.clear()

    @Slot(str)
    def _on_delete_file(self, path: str) -> None:
        if not path:
            return
        reply = QMessageBox.question(
            self, "Delete File", f"Delete '{Path(path).name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                os.remove(path)
                self._refresh()
                self._status.setText(f"Deleted {Path(path).name}")
            except Exception as exc:
                self._status.setText(f"Error: {exc}")

    @Slot(str)
    def _on_rename_file(self, path: str) -> None:
        if not path:
            return
        old_path = Path(path)
        new_name, ok = QInputDialog.getText(
            self, "Rename File", "New name:", text=old_path.stem)
        if ok and new_name.strip():
            new_path = old_path.parent / f"{new_name.strip()}{old_path.suffix}"
            if new_path.exists():
                QMessageBox.warning(
                    self, "Error", f"'{new_path.name}' already exists.")
                return
            old_path.rename(new_path)
            self._refresh()
            self._status.setText(f"Renamed to {new_path.name}")

    @Slot()
    def _on_upload_face(self) -> None:
        paths, _ = get_open_filenames(
            self, "Upload Face Image(s)", "",
            "Images (*.png *.jpg *.jpeg *.webp)")
        if not paths:
            return
        faces_dir = self._project_path / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            dest = faces_dir / Path(p).name
            if dest.exists():
                stem, suffix, idx = Path(p).stem, Path(p).suffix, 1
                while dest.exists():
                    dest = faces_dir / f"{stem}_{idx}{suffix}"
                    idx += 1
            shutil.copy2(p, dest)
        self._refresh()
        self._status.setText(f"Uploaded {len(paths)} face(s).")

    @Slot()
    def _on_cleanup(self) -> None:
        """Delete all files in the currently active output directory."""
        flat_idx = self._active_browser_index()
        if flat_idx < 0 or flat_idx >= len(self._browsers):
            return
        browser = self._browsers[flat_idx]
        scan_dir = browser.scan_dir
        if not scan_dir or not scan_dir.is_dir():
            self._status.setText("No directory to clean.")
            return
        n = len(browser._files)
        if n == 0:
            self._status.setText("Directory is already empty.")
            return
        size_str = _format_size(browser.dir_size)
        section_label = _BROWSER_SECTIONS[flat_idx][0] if flat_idx < len(_BROWSER_SECTIONS) else "this"
        reply = QMessageBox.warning(
            self, "Clean Up Directory",
            f"Delete all {n} file(s) ({size_str}) from {section_label}?\n\n"
            f"Directory: {scan_dir}\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        deleted = 0
        for fpath in list(browser._files):
            try:
                os.remove(fpath)
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", fpath, exc)
        self._refresh()
        self._status.setText(f"Cleaned up {deleted} file(s) from {section_label}.")

    @Slot(str)
    def _on_image_preview_send(self, key: str) -> None:
        path = self._image_preview._current_path
        if path:
            self.send_image_requested.emit(key, path)

    def on_project_changed(self) -> None:
        self._refresh()


# =========================================================================
#  _PNGLibrarySubTab — project PNG library browser
# =========================================================================

class _PNGLibrarySubTab(QWidget):
    """PNG Library browser: list with image preview, CRUD, upload, drag-drop."""

    send_image_requested = Signal(str, str)  # (key, image_path)

    def __init__(self, state: AppState, get_project_path, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._get_project_path = get_project_path
        self._files: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)

        # Left: file list with drag-drop
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._list = _DropListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._list.files_dropped.connect(self._on_files_dropped)
        left_layout.addWidget(self._list, 1)

        self._count_label = QLabel("0 files")
        left_layout.addWidget(self._count_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._upload_btn = QPushButton("Upload")
        self._upload_btn.setToolTip("Upload PNG(s) to the project PNG library")
        self._upload_btn.clicked.connect(self._on_upload)
        btn_row.addWidget(self._upload_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        btn_row.addWidget(self._refresh_btn)

        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: image preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._image_preview = _ImagePreviewLabel()
        self._image_preview.setAlignment(Qt.AlignCenter)
        self._image_preview.setFixedHeight(_PREVIEW_HEIGHT)
        self._image_preview.setStyleSheet("background: #1a1a1a;")
        self._image_preview.set_send_targets(_PNG_LIB_SEND_TARGETS)
        self._image_preview.send_requested.connect(self._on_preview_send)
        right_layout.addWidget(self._image_preview)

        self._status = QLabel("")
        right_layout.addWidget(self._status)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    @property
    def _project_path(self) -> Path:
        return self._get_project_path()

    @property
    def _lib_dir(self) -> Path:
        return self._project_path / "png_lib"

    def _refresh(self) -> None:
        self._list.clear()
        self._files = []
        lib = self._lib_dir
        if not lib.is_dir():
            self._count_label.setText("0 files")
            return
        for f in sorted(lib.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if f.suffix.lower() == ".png" and f.is_file():
                self._files.append(str(f))
                self._list.addItem(f.name)
        self._count_label.setText(f"{len(self._files)} files")

    @property
    def selected_path(self) -> str | None:
        row = self._list.currentRow()
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    @Slot(int)
    def _on_selection_changed(self, row: int) -> None:
        path = self.selected_path
        if not path:
            self._image_preview.clear()
            self._image_preview.set_current_path(None)
            return
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._image_preview.setPixmap(
                pixmap.scaled(self._image_preview.size(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._image_preview.set_current_path(path)

    # -- Context menu ------------------------------------------------------

    def _show_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        self._list.setCurrentItem(item)
        path = self.selected_path
        if not path:
            return

        menu = QMenu(self)

        from sdqt.widgets.send_targets import build_target_menu
        build_target_menu(menu, "Send to", _PNG_LIB_SEND_TARGETS,
                          lambda k, p=path: self.send_image_requested.emit(k, p))
        menu.addSeparator()

        rename_action = menu.addAction("Rename")
        rename_action.triggered.connect(lambda: self._on_rename(path))

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._on_delete(path))

        menu.exec(self._list.mapToGlobal(pos))

    # -- CRUD --------------------------------------------------------------

    @Slot()
    def _on_upload(self) -> None:
        paths, _ = get_open_filenames(
            self, "Upload PNG(s)", "", "PNG Images (*.png)")
        if paths:
            self._import_files(paths)

    @Slot(list)
    def _on_files_dropped(self, paths: list[str]) -> None:
        self._import_files(paths)

    def _import_files(self, paths: list[str]) -> None:
        lib = self._lib_dir
        lib.mkdir(parents=True, exist_ok=True)
        count = 0
        for p in paths:
            src = Path(p)
            if not src.is_file() or src.suffix.lower() != ".png":
                continue
            dest = lib / src.name
            if dest.exists():
                stem, idx = src.stem, 1
                while dest.exists():
                    dest = lib / f"{stem}_{idx}.png"
                    idx += 1
            shutil.copy2(str(src), str(dest))
            count += 1
        self._refresh()
        self._status.setText(f"Uploaded {count} PNG(s).")

    def _on_rename(self, path: str) -> None:
        old = Path(path)
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name (without .png):", text=old.stem)
        if not ok or not new_name.strip():
            return
        new_path = old.parent / f"{new_name.strip()}.png"
        if new_path.exists() and new_path != old:
            QMessageBox.warning(self, "Error", f"'{new_path.name}' already exists.")
            return
        old.rename(new_path)
        self._refresh()
        self._status.setText(f"Renamed to {new_path.name}")

    def _on_delete(self, path: str) -> None:
        p = Path(path)
        reply = QMessageBox.question(
            self, "Delete", f"Delete '{p.name}'?",
            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                p.unlink()
                self._refresh()
                self._status.setText(f"Deleted {p.name}")
            except Exception as exc:
                self._status.setText(f"Error: {exc}")

    @Slot(str)
    def _on_preview_send(self, key: str) -> None:
        path = self._image_preview._current_path
        if path:
            self.send_image_requested.emit(key, path)

    def on_project_changed(self) -> None:
        self._refresh()


# =========================================================================
#  BrowserTab — top-level tab for browsing generated outputs
# =========================================================================

class BrowserTab(BaseTab):
    """Browse and manage generated outputs."""

    # Signals for MainWindow to wire up
    send_clip_requested = Signal(str, str)
    send_frame_requested = Signal(str)
    send_image_requested = Signal(str, str)
    send_audio_requested = Signal(str, str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._top_tabs = QTabWidget()

        # -- Generations subtab ------------------------------------------------
        self._generations = _GenerationsSubTab(
            self.state, lambda: self.project_path)
        self._generations.send_clip_requested.connect(self.send_clip_requested)
        self._generations.send_frame_requested.connect(self.send_frame_requested)
        self._generations.send_image_requested.connect(self.send_image_requested)
        self._generations.send_audio_requested.connect(self.send_audio_requested)
        self._top_tabs.addTab(self._generations, "Generations")

        layout.addWidget(self._top_tabs)

    # -- Delegate attributes that MainWindow accesses directly -----------------

    @property
    def _video_preview(self) -> VideoPlayerWidget:
        return self._generations._video_preview

    @property
    def _section_tabs(self) -> QTabWidget:
        return self._generations._section_tabs

    @property
    def _browsers(self) -> list[FileBrowserWidget]:
        return self._generations._browsers

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._generations.on_project_changed()
