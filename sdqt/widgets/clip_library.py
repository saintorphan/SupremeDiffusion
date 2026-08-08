"""Clip library widget — scrollable clip list with thumbnails and drag support."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QMimeData, QPoint, QSize
from PySide6.QtGui import QAction, QDrag, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.timeline_track import MIME_TIMELINE_CLIP

logger = logging.getLogger(__name__)

_THUMB_W, _THUMB_H = 80, 45
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac"}
_MEDIA_EXTS = _VIDEO_EXTS | _AUDIO_EXTS


class _ClipCard(QWidget):
    """Individual clip card with thumbnail, name, duration, and remove button."""

    remove_requested = Signal(str)  # path
    add_to_timeline = Signal(str)   # path
    context_menu_requested = Signal(str, QPoint)  # (path, global_pos)

    def __init__(self, path: str, duration: float, thumbnail: QPixmap | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.path = path
        self.duration = duration
        self._drag_start: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(_THUMB_W, _THUMB_H)
        self._thumb.setStyleSheet("background: #222; border: 1px solid #444;")
        self._thumb.setAlignment(Qt.AlignCenter)
        if thumbnail and not thumbnail.isNull():
            self._thumb.setPixmap(
                thumbnail.scaled(_THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        layout.addWidget(self._thumb)

        # Info column
        info = QVBoxLayout()
        info.setSpacing(1)
        info.setContentsMargins(0, 0, 0, 0)

        name = Path(path).stem
        if len(name) > 28:
            name = name[:26] + "..."
        self._name_label = QLabel(f"<b>{name}</b>")
        info.addWidget(self._name_label)

        self._dur_label = QLabel(f"{duration:.1f}s")
        self._dur_label.setStyleSheet("font-size: 13px; color: #ccc;")
        info.addWidget(self._dur_label)

        # Audio indicator for audio-only files
        if Path(path).suffix.lower() in _AUDIO_EXTS:
            audio_lbl = QLabel("\U0001F3B5")  # 🎵
            audio_lbl.setStyleSheet("font-size: 13px; color: #1abc9c;")
            audio_lbl.setToolTip("Audio file")
            info.addWidget(audio_lbl)

        info.addStretch()
        layout.addLayout(info, 1)

        # Remove button
        self._remove_btn = QPushButton("\u2715")
        self._remove_btn.setFixedSize(20, 20)
        self._remove_btn.setToolTip("Remove from library")
        self._remove_btn.setStyleSheet(
            "QPushButton { border: none; color: #888; font-size: 14px; }"
            "QPushButton:hover { color: #f44; }"
        )
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.path))
        layout.addWidget(self._remove_btn, 0, Qt.AlignTop)

        self.setFixedHeight(65)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self._set_tech_tooltip()

    def _set_tech_tooltip(self) -> None:
        """Set a tooltip with technical video info."""
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_range",
                 "-show_entries", "format=duration",
                 "-of", "csv=p=0", self.path],
                capture_output=True, text=True, timeout=5,
            )
            parts = r.stdout.strip().split("\n")
            if len(parts) >= 2:
                vp = parts[0].split(",")
                dur = parts[1].strip()
                if len(vp) >= 6:
                    codec, w, h, pix, cr, rfr = vp[0], vp[1], vp[2], vp[3], vp[4], vp[5]
                    if "/" in rfr:
                        n, d = rfr.split("/")
                        fps = f"{float(n)/float(d):.0f}"
                    else:
                        fps = rfr
                    dur_s = f"{float(dur):.1f}s" if dur else "?"
                    self.setToolTip(
                        f"{Path(self.path).name}\n"
                        f"{w}x{h} | {codec} {pix} {cr}\n"
                        f"{fps} fps | {dur_s}"
                    )
                    return
        except Exception:
            pass
        self.setToolTip(self.path)

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._thumb.setPixmap(
            pixmap.scaled(_THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_path(self, new_path: str) -> None:
        """Update the card after the underlying file was renamed on disk."""
        self.path = new_path
        name = Path(new_path).stem
        if len(name) > 28:
            name = name[:26] + "..."
        self._name_label.setText(f"<b>{name}</b>")
        self._set_tech_tooltip()

    def _show_menu(self, pos) -> None:
        self.context_menu_requested.emit(self.path, self.mapToGlobal(pos))

    # -- Drag out ----------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start is not None
            and (event.position().toPoint() - self._drag_start).manhattanLength() > 10
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(MIME_TIMELINE_CLIP, self.path.encode("utf-8"))
            drag.setMimeData(mime)
            if self._thumb.pixmap() and not self._thumb.pixmap().isNull():
                drag.setPixmap(self._thumb.pixmap().scaled(60, 34, Qt.KeepAspectRatio))
            drag.exec(Qt.CopyAction)
            self._drag_start = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start = None
        super().mouseReleaseEvent(event)


class ClipLibraryWidget(QWidget):
    """Scrollable vertical list of clip cards with drag support.

    Signals:
        clip_added(str, float): path and duration when a clip is added
        clip_removed(str): path when a clip is removed
        add_to_timeline_requested(str): path when user wants to add clip to timeline
    """

    clip_added = Signal(str, float)
    clip_removed = Signal(str)
    add_to_timeline_requested = Signal(str)
    clip_context_menu_requested = Signal(str, QPoint)  # (path, global_pos)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._clips: dict[str, tuple[float, QPixmap | None]] = {}  # path -> (duration, thumb)
        self._cards: dict[str, _ClipCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        lbl = QLabel("<b>Clip Library</b>")
        header.addWidget(lbl)
        header.addStretch()

        import_btn = QPushButton("Import")
        import_btn.setObjectName("primary")
        import_btn.setToolTip("Import media files into the clip library")
        import_btn.clicked.connect(self._import_clips)
        header.addWidget(import_btn)

        layout.addLayout(header)

        # Clip list area
        self._list_widget = QListWidget()
        self._list_widget.setSpacing(1)
        self._list_widget.setStyleSheet(
            "QListWidget { background: #1a1a1a; border: none; }"
            "QListWidget::item { background: #252525; border-radius: 3px; }"
            "QListWidget::item:hover { background: #2a2a2a; }"
        )
        layout.addWidget(self._list_widget, 1)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

    # -- Public API --------------------------------------------------------

    def add_clip(self, path: str, duration: float, thumbnail: QPixmap | None = None) -> None:
        if path in self._clips:
            return
        self._clips[path] = (duration, thumbnail)

        card = _ClipCard(path, duration, thumbnail)
        card.remove_requested.connect(self.remove_clip)
        card.add_to_timeline.connect(self.add_to_timeline_requested.emit)
        card.context_menu_requested.connect(self.clip_context_menu_requested.emit)
        self._cards[path] = card

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, card.height()))
        item.setData(Qt.UserRole, path)
        self._list_widget.addItem(item)
        self._list_widget.setItemWidget(item, card)

        self.clip_added.emit(path, duration)

        # Generate thumbnail async if not provided
        if thumbnail is None:
            self._generate_thumbnail(path)

    def remove_clip(self, path: str) -> None:
        if path not in self._clips:
            return
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item and item.data(Qt.UserRole) == path:
                self._list_widget.takeItem(i)
                break
        del self._clips[path]
        self._cards.pop(path, None)
        self.clip_removed.emit(path)

    def rename_clip(self, old_path: str, new_path: str) -> None:
        """Point an existing entry at a renamed file — no remove/add signals."""
        if old_path not in self._clips or new_path in self._clips:
            return
        self._clips[new_path] = self._clips.pop(old_path)
        card = self._cards.pop(old_path, None)
        if card is not None:
            card.set_path(new_path)
            self._cards[new_path] = card
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item and item.data(Qt.UserRole) == old_path:
                item.setData(Qt.UserRole, new_path)
                break

    def clear(self) -> None:
        paths = list(self._clips.keys())
        for p in paths:
            self.remove_clip(p)

    def clip_paths(self) -> list[str]:
        return list(self._clips.keys())

    def clip_data(self) -> dict[str, tuple[float, QPixmap | None]]:
        return dict(self._clips)

    def get_duration(self, path: str) -> float:
        return self._clips.get(path, (0.0, None))[0]

    def get_thumbnail(self, path: str) -> QPixmap | None:
        return self._clips.get(path, (0.0, None))[1]

    def set_thumbnail(self, path: str, pixmap: QPixmap) -> None:
        if path in self._clips:
            dur, _ = self._clips[path]
            self._clips[path] = (dur, pixmap)
            if path in self._cards:
                self._cards[path].set_thumbnail(pixmap)

    # -- Import ------------------------------------------------------------

    def _import_clips(self) -> None:
        """Open a file dialog to import media files into the library."""
        exts = " ".join(f"*{e}" for e in sorted(_MEDIA_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Clips", "",
            f"Media Files ({exts});;All Files (*)",
        )
        if not paths:
            return
        for p in paths:
            if p in self._clips:
                continue
            dur = 0.0
            ext = Path(p).suffix.lower()
            try:
                if ext in _VIDEO_EXTS:
                    from supremediffusion.utils.video import probe_video
                    info = probe_video(p)
                    dur = info.get("duration", 0.0)
                elif ext in _AUDIO_EXTS:
                    import subprocess, json
                    result = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_format", str(p)],
                        capture_output=True, text=True, timeout=10,
                    )
                    data = json.loads(result.stdout)
                    dur = float(data.get("format", {}).get("duration", 0))
            except Exception:
                logger.debug("Could not probe duration for %s", p, exc_info=True)
            self.add_clip(p, dur)
            logger.info("Imported clip: %s (%.1fs)", p, dur)

    # -- Thumbnail generation ----------------------------------------------

    def _generate_thumbnail(self, path: str) -> None:
        """Generate a thumbnail from the first frame of the video."""
        # Skip for audio-only files
        if Path(path).suffix.lower() in _AUDIO_EXTS:
            return
        try:
            from supremediffusion.utils.video import extract_single_frame
            img = extract_single_frame(path, 0)
            if img is not None:
                from PySide6.QtGui import QImage
                if img.mode != "RGB":
                    img = img.convert("RGB")
                data = img.tobytes("raw", "RGB")
                qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.set_thumbnail(path, pixmap)
        except Exception:
            logger.debug("Thumbnail generation failed for %s", path, exc_info=True)
