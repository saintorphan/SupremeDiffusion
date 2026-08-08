"""Video player widget with native scrubbing, frame capture, drag/drop, and toolbar."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from sdqt.widgets.send_targets import DISABLED_TARGETS

from PySide6.QtCore import Qt, Signal, Slot, QUrl, QMimeData, QTimer, QBuffer, QIODevice
from PySide6.QtGui import QAction, QDrag, QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QAudio, QAudioSink, QAudioFormat, QMediaDevices
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename, get_save_filename

logger = logging.getLogger(__name__)

# Module-level audio device ID — set by SettingsTab, read by VideoPlayerWidget
_audio_device_id: str = ""


def set_audio_device_id(device_id: str) -> None:
    """Set the preferred audio output device ID (called from Settings)."""
    global _audio_device_id
    _audio_device_id = device_id

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class _DropVideoWidget(QVideoWidget):
    """QVideoWidget subclass that accepts file drops and supports drag-out."""

    file_dropped = Signal(str)
    browse_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drag_start_pos = None
        self._file_path: str | None = None

        # Browse overlay button (visible when empty)
        self._browse_overlay = QPushButton("\u2191", self)  # ↑
        self._browse_overlay.setToolTip("Browse for video file")
        self._browse_overlay.setFixedSize(28, 28)
        self._browse_overlay.setStyleSheet(
            "QPushButton { background: rgba(80,80,80,180); color: #ddd; border: 1px solid #666; "
            "border-radius: 6px; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(100,140,200,220); color: #fff; }"
        )
        self._browse_overlay.clicked.connect(self.browse_requested.emit)
        self._browse_overlay.setCursor(Qt.PointingHandCursor)

    def set_file_path(self, path: str | None) -> None:
        self._file_path = path
        self._browse_overlay.setVisible(path is None)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Position overlay in upper-right corner
        self._browse_overlay.move(self.width() - self._browse_overlay.width() - 6, 6)

    # -- Drop in -------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _VIDEO_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _VIDEO_EXTS:
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return

    # -- Drag out ------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_pos is not None
            and self._file_path
            and (event.position().toPoint() - self._drag_start_pos).manhattanLength() > 20
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self._file_path)])
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction)
            self._drag_start_pos = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)


class _ClickSlider(QSlider):
    """QSlider that jumps to the clicked position instead of page-stepping."""

    click_position = Signal(int)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), self.width(),
            )
            self.setValue(val)
            self.click_position.emit(val)
            # Still let the base class start its drag tracking
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)


class VideoPlayerWidget(QWidget):
    """Embeddable video player with playback controls, frame capture, and drag/drop.

    Signals:
        position_changed(float): current playback position in seconds
        frame_captured(str): path to captured frame PNG
        video_loaded(str): emitted when a video is loaded (path)
        video_cleared(): emitted when the video is cleared
    """

    position_changed = Signal(float)
    frame_captured = Signal(str)
    video_loaded = Signal(str)
    video_cleared = Signal()
    send_clip_requested = Signal(str)   # key e.g. "ve", "trim"
    send_frame_requested = Signal(str)  # key e.g. "img2img", "inpaint"
    final_frame_requested = Signal(str) # key e.g. "img2vid", "ve"
    guide_video_requested = Signal(str) # key e.g. "img2vid_m1", "ve"
    color_ref_requested = Signal(str)   # emits captured frame path

    def __init__(self, label: str = "Preview", parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._current_path: str | None = None
        self._duration_ms: int = 0
        self._label_text = label

        # -- Effects system ------------------------------------------------
        self._effects_enabled = False
        self._effect_stack: list[dict] = []
        self._original_path: str | None = None
        self._modified_path: str | None = None
        self._effect_worker = None

        # -- Media stack ---------------------------------------------------
        self._player = QMediaPlayer(self)
        # QMediaPlayer FFmpeg backend audio is broken on some systems;
        # use QAudioSink as a fallback for videos with audio tracks.
        self._audio = QAudioOutput(self)
        self._audio.setVolume(0.0)  # mute QMediaPlayer's own audio
        self._player.setAudioOutput(self._audio)
        self._audio_sink: QAudioSink | None = None
        self._audio_buffer: QBuffer | None = None
        self._audio_pcm: bytes = b""
        self._has_audio_track = False
        self._volume: float = 1.0  # 0.0 to 1.0

        self._video_widget = _DropVideoWidget()
        self._video_widget.setMinimumHeight(120)
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
        self._video_widget.setStyleSheet("background: black;")
        self._video_widget.file_dropped.connect(self.load_video)
        self._video_widget.browse_requested.connect(self._browse_video)
        self._player.setVideoOutput(self._video_widget)

        # -- Controls (large by default — easy to hit on every tab) --------
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(64, 40)
        self._play_btn.setStyleSheet(
            "QPushButton { font-size: 20px; font-weight: bold; }"
        )
        self._play_btn.clicked.connect(self._toggle_play)

        self._slider = _ClickSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setFixedHeight(28)
        self._slider.sliderMoved.connect(self._seek)
        self._slider.click_position.connect(self._seek)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setMinimumWidth(120)
        self._time_label.setFixedHeight(28)
        self._time_label.setStyleSheet("font-size: 14px;")

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(self._play_btn)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._time_label)

        # -- Title toolbar: [label] [spacer] [folder] [clear] -------------
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(4)

        self._title_label = QLabel(f"<b>{label}</b>")
        self._title_label.setMinimumHeight(24)
        self._title_label.setWordWrap(True)
        title_row.addWidget(self._title_label)
        title_row.addStretch()

        btn_style = (
            "QPushButton { border: none; padding: 2px 4px; font-size: 14px; color: #aaa; }"
            "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
        )

        self._folder_btn = QPushButton("\U0001F4C2")  # 📂
        self._folder_btn.setFixedSize(32, 26)
        self._folder_btn.setToolTip("Open containing folder")
        self._folder_btn.setStyleSheet(btn_style)
        self._folder_btn.clicked.connect(self._open_folder)
        self._folder_btn.setVisible(False)
        title_row.addWidget(self._folder_btn)

        self._clear_btn = QPushButton("\u2715")  # ✕
        self._clear_btn.setFixedSize(32, 26)
        self._clear_btn.setToolTip("Clear video")
        self._clear_btn.setStyleSheet(btn_style)
        self._clear_btn.clicked.connect(self.clear_video)
        self._clear_btn.setVisible(False)
        title_row.addWidget(self._clear_btn)

        # -- Original / Modified tab bar -----------------------------------
        self._mode_bar = QTabBar()
        self._mode_bar.addTab("Original")
        self._mode_bar.addTab("Modified")
        self._mode_bar.setFixedHeight(28)
        self._mode_bar.setStyleSheet(
            "QTabBar::tab { padding: 5px 12px; font-size: 13px; }"
        )
        self._mode_bar.currentChanged.connect(self._on_mode_tab_changed)
        self._mode_bar.setVisible(False)

        # -- Layout --------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(title_row)
        layout.addWidget(self._mode_bar)
        layout.addWidget(self._video_widget, 1)
        layout.addLayout(controls)

        # -- Signals -------------------------------------------------------
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_media_status)

        # Flag: when True, auto-pause after the first frame renders
        self._pending_show_frame = False
        self._pending_auto_play = False

        # Context menu targets (configurable per instance)
        self._clip_targets: list[tuple[str, str]] = []   # (key, display_name)
        self._frame_targets: list[tuple[str, str]] = []  # (key, display_name)
        self._final_frame_targets: list[tuple[str, str]] = []  # (key, display_name)
        self._guide_video_targets: list[tuple[str, str]] = []  # (key, display_name)
        self._show_color_ref: bool = False

        # Enable context menu on the video widget
        self._video_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._video_widget.customContextMenuRequested.connect(self._show_context_menu)

    # -- Fallback drop handling on outer widget ----------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _VIDEO_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _VIDEO_EXTS:
                self.load_video(path)
                event.acceptProposedAction()
                return

    # -- Visibility handling -----------------------------------------------

    def showEvent(self, event) -> None:
        """Re-display the current frame when the widget becomes visible.

        QMediaPlayer cannot render to a hidden QVideoWidget, so videos
        loaded while the tab is not visible end up black.  When we become
        visible, force a frame render by briefly playing then pausing.
        """
        super().showEvent(event)
        if not self._current_path:
            return
        if self._player.mediaStatus() in (
            QMediaPlayer.NoMedia,
            QMediaPlayer.InvalidMedia,
        ):
            return
        # Clear deferred flag now that we're visible
        self._pending_show_frame = False
        # Force a frame render if not already playing
        if self._player.playbackState() != QMediaPlayer.PlayingState:
            pos = self._player.position()
            self._player.play()
            self._player.setPosition(pos)
            QTimer.singleShot(50, self._player.pause)

    # -- Public API --------------------------------------------------------

    def set_send_targets(
        self,
        clip_targets: list[tuple[str, str]] | None = None,
        frame_targets: list[tuple[str, str]] | None = None,
        final_frame_targets: list[tuple[str, str]] | None = None,
    ) -> None:
        """Configure right-click send-to targets.

        Args:
            clip_targets: List of (key, display_name) for "Send clip to" submenu
            frame_targets: List of (key, display_name) for "Send frame to" submenu
            final_frame_targets: List of (key, display_name) for "Final Frame" submenu
        """
        if clip_targets is not None:
            self._clip_targets = clip_targets
        if frame_targets is not None:
            self._frame_targets = frame_targets
        if final_frame_targets is not None:
            self._final_frame_targets = final_frame_targets

    def set_guide_video_targets(self, targets: list[tuple[str, str]]) -> None:
        """Configure right-click Guide Video targets."""
        self._guide_video_targets = targets

    def enable_color_ref_action(self, enabled: bool = True) -> None:
        """Show a 'Color Reference Frame' action in the right-click menu."""
        self._show_color_ref = enabled

    def _stop_audio_sink(self, clear_data: bool = False) -> None:
        """Stop the QAudioSink fallback if active."""
        if self._audio_sink is not None:
            self._audio_sink.stop()
            self._audio_sink = None
        if self._audio_buffer is not None:
            self._audio_buffer.close()
            self._audio_buffer = None
        if clear_data:
            self._audio_pcm = b""
            self._has_audio_track = False

    def _extract_audio(self, path: str) -> None:
        """Extract PCM audio from a video file for QAudioSink playback."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a",
                 "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=5,
            )
            if not result.stdout.strip():
                return  # no audio track
            result = subprocess.run(
                ["ffmpeg", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
                 "-ar", "44100", "-ac", "2", "-"],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout:
                self._audio_pcm = result.stdout
                self._has_audio_track = True
        except Exception:
            logger.debug("Audio extraction failed for %s", path, exc_info=True)

    @staticmethod
    def _find_audio_device() -> "QAudioDevice":
        """Find the configured audio output device, or system default."""
        if _audio_device_id:
            for dev in QMediaDevices.audioOutputs():
                if dev.id().data().decode() == _audio_device_id:
                    return dev
        return QMediaDevices.defaultAudioOutput()

    def _start_audio_sink(self, start_pos_ms: int = 0) -> None:
        """Prepare and start QAudioSink playback of extracted PCM data.

        Args:
            start_pos_ms: Video position in ms to sync audio start to.
        """
        if not self._has_audio_track or not self._audio_pcm:
            return
        self._stop_audio_sink()
        try:
            audio_dev = self._find_audio_device()
            if audio_dev.isNull():
                logger.warning("No audio output device available, skipping audio")
                return
            fmt = QAudioFormat()
            fmt.setSampleRate(44100)
            fmt.setChannelCount(2)
            fmt.setSampleFormat(QAudioFormat.Int16)
            if not audio_dev.isFormatSupported(fmt):
                logger.warning("Audio format not supported by device %s", audio_dev.description())
                return
            self._audio_sink = QAudioSink(audio_dev, fmt, self)
            from PySide6.QtCore import QByteArray
            self._audio_buffer = QBuffer(self)
            self._audio_buffer.setData(QByteArray(self._audio_pcm))
            self._audio_buffer.open(QIODevice.ReadOnly)
            # Seek to correct position BEFORE starting playback
            byte_pos = int(start_pos_ms / 1000.0 * 44100 * 4)
            byte_pos = max(0, min(byte_pos, len(self._audio_pcm)))
            self._audio_buffer.seek(byte_pos)
            self._audio_sink.setVolume(self._volume)
            self._audio_sink.start(self._audio_buffer)
        except Exception:
            logger.warning("Failed to start audio sink, playing without audio", exc_info=True)
            self._stop_audio_sink()

    def load_video(self, path: str | None, auto_play: bool = False) -> None:
        """Load a video file (or clear if None).

        Args:
            path: Video file path, or None to clear.
            auto_play: If True, start playback after loading (for fresh results).
                       If False (default), show first frame then pause.
        """
        self._stop_audio_sink(clear_data=True)
        self._pending_show_frame = False
        self._pending_auto_play = False
        self._current_path = path
        # Reset effects state on new video load
        self._original_path = path
        self._modified_path = None
        self._effect_stack = []
        self._mode_bar.setVisible(False)
        if self._mode_bar.currentIndex() != 0:
            self._mode_bar.blockSignals(True)
            self._mode_bar.setCurrentIndex(0)
            self._mode_bar.blockSignals(False)
        if path and Path(path).is_file():
            self._extract_audio(path)
            self._player.setSource(QUrl.fromLocalFile(path))
            self._video_widget.set_file_path(path)
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            if auto_play:
                # Let it play continuously after buffer
                self._pending_auto_play = True
            # Show first frame: play briefly then pause (or continue if auto_play)
            self._pending_show_frame = True
            self._player.play()
            self.video_loaded.emit(path)
        else:
            self._player.setSource(QUrl())
            self._video_widget.set_file_path(None)
            self._slider.setRange(0, 0)
            self._time_label.setText("0:00 / 0:00")
            self._folder_btn.setVisible(False)
            self._clear_btn.setVisible(False)

    def clear_video(self) -> None:
        """Stop and clear the loaded video."""
        self._pending_show_frame = False
        self._stop_audio_sink(clear_data=True)
        # setSource(QUrl()) implicitly stops playback — skip explicit stop()
        # as it can block for 30+ seconds on the FFmpeg backend.
        if self._current_path is not None:
            self._player.setSource(QUrl())
        self._current_path = None
        self._original_path = None
        self._modified_path = None
        self._effect_stack = []
        self._mode_bar.setVisible(False)
        self._video_widget.set_file_path(None)
        self._slider.setRange(0, 0)
        self._time_label.setText("0:00 / 0:00")
        self._folder_btn.setVisible(False)
        self._clear_btn.setVisible(False)
        self.video_cleared.emit()

    def get_position_sec(self) -> float:
        """Current playback position in seconds."""
        return self._player.position() / 1000.0

    def capture_frame(self) -> str | None:
        """Extract current frame as PNG and return path.

        Uses ffmpeg via the core video utils for frame-accurate extraction.
        Captures from whichever version (Original/Modified) is currently shown.
        """
        capture_path = self._current_path
        if not capture_path or not Path(capture_path).is_file():
            return None
        try:
            from supremediffusion.utils.video import extract_single_frame, probe_video

            timestamp = self.get_position_sec()
            info = probe_video(capture_path)
            fps = info.get("fps", 16)
            frame_num = max(0, min(int(timestamp * fps), info["num_frames"] - 1))
            # Send the frame exactly as previewed — do NOT neutralize here.
            # Color-neutralization is reserved for the automated chained-
            # generation feed-forward (to break warm drift); applying it to a
            # manual "send frame to" pushed warm frames cool/blue so the sent
            # image no longer matched the preview.
            img = extract_single_frame(capture_path, frame_num)

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()
            self.frame_captured.emit(tmp.name)
            return tmp.name
        except Exception:
            logger.warning("Frame capture failed", exc_info=True)
            return None

    @Slot()
    def _set_current_frame_as_neutralizer_ref(self) -> None:
        """Extract current frame raw and persist as neutralizer reference."""
        capture_path = self._current_path
        if not capture_path or not Path(capture_path).is_file():
            return
        try:
            from supremediffusion.utils.video import extract_single_frame, probe_video
            from sdqt.utils.color_neutralize import save_as_neutralizer_reference
            from supremediffusion.config.global_config import GlobalConfig

            timestamp = self.get_position_sec()
            info = probe_video(capture_path)
            fps = info.get("fps", 16) or 16
            frame_num = max(0, min(int(timestamp * fps), info["num_frames"] - 1))
            img = extract_single_frame(capture_path, frame_num)
            cfg = GlobalConfig.load()
            ref_path = save_as_neutralizer_reference(img, cfg)
            logger.info("Neutralizer reference set → %s", ref_path)
        except Exception:
            logger.warning("Set neutralizer reference failed", exc_info=True)

    @property
    def video_path(self) -> str | None:
        """Return the modified path when effects are applied, else the current path."""
        if self._modified_path and self._effect_stack:
            return self._modified_path
        return self._current_path

    @property
    def original_path(self) -> str | None:
        """Return the original (un-effected) video path."""
        return self._original_path or self._current_path

    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        if self._audio_sink is not None:
            self._audio_sink.setVolume(self._volume)

    # -- Slots -------------------------------------------------------------

    @Slot()
    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
            self._stop_audio_sink()
        else:
            # Restart from beginning if at end or stopped
            if (
                self._player.playbackState() == QMediaPlayer.StoppedState
                or self._player.position() >= self._duration_ms - 100
            ):
                self._player.setPosition(0)
            self._player.play()
            # Start audio after video so they begin together
            self._ensure_audio_playing()

    def _ensure_audio_playing(self) -> None:
        """Start or restart the QAudioSink synced to current video position."""
        if not self._has_audio_track:
            return
        try:
            self._start_audio_sink(start_pos_ms=self._player.position())
        except Exception:
            logger.warning("Audio start failed, playing without audio", exc_info=True)

    @Slot(int)
    def _seek(self, position_ms: int) -> None:
        # If stopped, switch to paused so the frame is visible
        if self._player.playbackState() == QMediaPlayer.StoppedState:
            self._player.play()
            self._player.pause()
        self._player.setPosition(position_ms)
        # Resync audio buffer position
        if self._audio_buffer is not None and self._has_audio_track:
            byte_pos = int(position_ms / 1000.0 * 44100 * 4)  # 2ch * 16bit = 4 bytes/sample
            byte_pos = max(0, min(byte_pos, len(self._audio_pcm)))
            self._audio_buffer.seek(byte_pos)

    @Slot(int)
    def _on_position(self, pos_ms: int) -> None:
        if not self._slider.isSliderDown():
            self._slider.setValue(pos_ms)
        self._time_label.setText(
            f"{self._fmt(pos_ms)} / {self._fmt(self._duration_ms)}"
        )
        self.position_changed.emit(pos_ms / 1000.0)

    @Slot(int)
    def _on_duration(self, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        self._slider.setRange(0, dur_ms)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_state(self, state: QMediaPlayer.PlaybackState) -> None:
        self._play_btn.setText(
            "⏸" if state == QMediaPlayer.PlayingState else "▶"
        )

    @Slot(QMediaPlayer.MediaStatus)
    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        # After loading, either pause on first frame or continue playing.
        if self._pending_show_frame and status in (
            QMediaPlayer.BufferedMedia,
            QMediaPlayer.LoadedMedia,
        ):
            if getattr(self, "_pending_auto_play", False):
                # Result auto-play: let it keep playing
                self._pending_auto_play = False
                self._pending_show_frame = False
            else:
                # Normal load: pause on first frame
                self._player.pause()
                self._stop_audio_sink()
                if self.isVisible():
                    self._pending_show_frame = False
        # When playback reaches the end, just stop and seek back.
        if status == QMediaPlayer.EndOfMedia:
            self._stop_audio_sink()
            self._player.setPosition(max(0, self._duration_ms - 50))

    @Slot()
    def _browse_video(self) -> None:
        path, _ = get_open_filename(
            self, "Select Video", "", "Videos (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if path:
            self.load_video(path)

    @Slot()
    def _save_as(self) -> None:
        if not self._current_path or not Path(self._current_path).is_file():
            return
        default_name = Path(self._current_path).name
        dest, _ = get_save_filename(
            self, "Save Clip As", default_name, "Videos (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if dest:
            import shutil
            shutil.copy2(self._current_path, dest)

    @Slot()
    def _open_folder(self) -> None:
        if self._current_path and Path(self._current_path).is_file():
            folder = str(Path(self._current_path).parent)
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", self._current_path])
                elif os.name == "posix":
                    subprocess.Popen(["xdg-open", folder])
                else:
                    subprocess.Popen(["open", folder])
            except Exception:
                logger.warning("Failed to open folder", exc_info=True)

    @Slot()
    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu with send-to options."""
        if not self._current_path:
            return

        from sdqt.widgets.send_targets import build_target_menu

        menu = QMenu(self)

        if self._clip_targets:
            build_target_menu(menu, "Send clip to", self._clip_targets,
                              self.send_clip_requested.emit)

        if self._frame_targets:
            build_target_menu(menu, "Send frame to", self._frame_targets,
                              self.send_frame_requested.emit)

        if self._final_frame_targets:
            build_target_menu(menu, "Final Frame", self._final_frame_targets,
                              self.final_frame_requested.emit)

        if self._guide_video_targets:
            build_target_menu(menu, "Guide Video", self._guide_video_targets,
                              self.guide_video_requested.emit)

        if self._show_color_ref:
            cc_action = menu.addAction("Color Reference Frame")
            cc_action.triggered.connect(self._emit_color_ref)

        # "Set current frame as Neutralizer Reference" — pulls the current
        # playhead frame and saves it as the global neutralize_reference_path
        # used by the frame neutralizer.
        neut_action = menu.addAction("Set current frame as Neutralizer Reference")
        neut_action.setToolTip(
            "Save the current frame as the color reference for the frame "
            "neutralizer (Settings → Color Drift Neutralization)."
        )
        neut_action.triggered.connect(self._set_current_frame_as_neutralizer_ref)

        if self._clip_targets or self._frame_targets or self._final_frame_targets or self._guide_video_targets or self._show_color_ref:
            menu.addSeparator()

        # -- Effects menu (only on result players) --------------------------
        if self._effects_enabled:
            from sdqt.models.effects import EFFECT_CATEGORIES, EFFECT_REGISTRY, effect_label as _fx_label

            fx_menu = menu.addMenu("Effects")
            for cat_name, type_keys in EFFECT_CATEGORIES.items():
                cat_sub = fx_menu.addMenu(cat_name)
                for etype in type_keys:
                    if etype not in EFFECT_REGISTRY:
                        continue
                    label = _fx_label(etype)
                    act = cat_sub.addAction(label)
                    act.triggered.connect(
                        lambda checked, et=etype: self._add_effect(et)
                    )

            if self._effect_stack:
                fx_menu.addSeparator()
                hist_action = fx_menu.addAction("Effect History...")
                hist_action.triggered.connect(self._show_effect_history)

        # Trim / Crop / Lip Sync — available on any player with a loaded video
        menu.addSeparator()
        trim_action = menu.addAction("Trim...")
        trim_action.triggered.connect(self._open_trim_dialog)

        crop_action = menu.addAction("Crop...")
        crop_action.triggered.connect(self._open_crop_dialog)

        ls_menu = menu.addMenu("Lip Sync")
        for key, label in [
            ("ltx_a2vid", "LTX Audio-to-Video"),
            ("musetalk", "MuseTalk"),
            ("vace_multitalk", "VACE MultiTalk"),
            ("latentsync", "LatentSync"),
            ("video_retalking", "Video Retalking"),
        ]:
            act = ls_menu.addAction(label)
            act.triggered.connect(
                lambda checked, k=key: self._open_lipsync_dialog(k)
            )
        menu.addSeparator()

        # Always available actions
        save_as = menu.addAction("Save Clip As...")
        save_as.triggered.connect(self._save_as)

        open_folder = menu.addAction("Open containing folder")
        open_folder.triggered.connect(self._open_folder)

        # Export / Quick Export / Create Loop — shared clip actions.
        # Requires set_state_ref() to have been called on this player.
        if getattr(self, "_app_state", None) is not None and self._current_path:
            from sdqt.tabs.clip_menu_actions import add_clip_export_actions
            add_clip_export_actions(
                menu, self, self._app_state, self._current_path,
                media_offset=0.0, duration=0.0,
                clip_name=Path(self._current_path).stem,
            )

        menu.exec(self._video_widget.mapToGlobal(pos))

    @Slot()
    def _emit_color_ref(self) -> None:
        """Capture current frame and emit as color reference."""
        path = self.capture_frame()
        if path:
            self.color_ref_requested.emit(path)

    # -- Effects system public API -----------------------------------------

    def enable_effects_menu(self, enabled: bool = True) -> None:
        """Enable or disable the Effects submenu on this player's context menu."""
        self._effects_enabled = enabled

    def set_state_ref(self, state) -> None:
        """Store a reference to AppState for dialog use (lip sync, etc.)."""
        self._app_state = state

    def set_controls_size(self, large: bool = True) -> None:
        """Scale up the playback controls for high-attention tabs.

        Makes the play button, slider, and time label visibly larger so
        users on the Generate/Img2Vid tab can hit them without aiming.
        """
        if large:
            self._play_btn.setFixedSize(64, 40)
            self._play_btn.setStyleSheet(
                "QPushButton { font-size: 20px; font-weight: bold; }"
            )
            self._slider.setFixedHeight(28)
            self._time_label.setFixedHeight(28)
            self._time_label.setFixedWidth(120)
            self._time_label.setStyleSheet("font-size: 14px;")
        else:
            self._play_btn.setFixedSize(32, 20)
            self._play_btn.setStyleSheet("")
            self._slider.setFixedHeight(16)
            self._time_label.setFixedHeight(16)
            self._time_label.setFixedWidth(90)
            self._time_label.setStyleSheet("")

    # -- Effects application -----------------------------------------------

    def _add_effect(self, effect_type: str) -> None:
        """Open an editor for a new effect and append to the stack on OK."""
        from sdqt.models.effects import make_effect, EFFECT_REGISTRY

        if effect_type == "crop_zoom":
            return

        fx = make_effect(effect_type)
        logger.info("_add_effect: type=%s, original_path=%s, current_path=%s",
                     effect_type, self._original_path, self._current_path)

        from sdqt.widgets.effects_manager import EffectEditorDialog
        dlg = EffectEditorDialog(fx, parent=self)
        result = dlg.exec()
        logger.info("_add_effect: dialog result=%s (Accepted=%s)", result, QDialog.Accepted)
        if result == QDialog.Accepted:
            fx["params"] = dlg.values()
            self._effect_stack.append(fx)
            logger.info("_add_effect: stack now has %d effects, applying...", len(self._effect_stack))
            self._apply_effect_stack()

    def _apply_effect_stack(self) -> None:
        """Re-apply the full effect stack from the original source."""
        src = self._original_path
        logger.info("_apply_effect_stack: original_path=%s, exists=%s",
                     src, Path(src).is_file() if src else "N/A")
        if not src or not Path(src).is_file():
            logger.warning("_apply_effect_stack: no valid source, aborting")
            self._title_label.setText(
                f"<b>{self._label_text}</b> <i style='color:#e55;'>No source video</i>")
            return

        # Filter out identity/disabled effects
        active = [fx for fx in self._effect_stack if fx.get("enabled", True)]
        if not active:
            # No active effects — switch back to original
            self._modified_path = None
            self._mode_bar.setVisible(False)
            self._switch_to_path(src)
            return

        from sdqt.workers.video_effects import VideoEffectWorker

        # Stop any running worker
        if self._effect_worker is not None and self._effect_worker.isRunning():
            self._effect_worker.abort()
            self._effect_worker.wait(2000)

        # Pass upscaler model dir + the active project's color profile if we
        # have an AppState reference. The worker uses the profile to tag its
        # output so it matches the project's other encodes.
        upscaler_dir = ""
        global_config = None
        project_config = None
        state = getattr(self, "_app_state", None)
        if state:
            upscaler_dir = state.global_config.model_paths.get("upscaler_dir", "")
            global_config = state.global_config
            try:
                from supremediffusion.config.project_config import ProjectConfig
                pm = getattr(state, "project_manager", None)
                cur = getattr(state, "current_project", "") or ""
                if pm and cur:
                    project_config = ProjectConfig.load(pm.get_project_path(cur))
            except Exception:
                project_config = None

        logger.info("_apply_effect_stack: creating worker src=%s effects=%d upscaler=%s",
                     src, len(active), upscaler_dir)
        worker = VideoEffectWorker(
            src, active, upscaler_dir=upscaler_dir,
            project_config=project_config, global_config=global_config,
            parent=self,
        )
        worker.finished_ok.connect(self._on_effects_applied)
        worker.error.connect(self._on_effects_error)
        worker.status.connect(self._on_effects_status)
        worker.finished.connect(worker.deleteLater)
        self._effect_worker = worker
        self._title_label.setText(f"<b>{self._label_text}</b> <i style='color:#888;'>Processing...</i>")
        logger.info("_apply_effect_stack: starting worker thread")
        worker.start()

    def _on_effects_status(self, msg: str) -> None:
        """Show worker status in the title label."""
        self._title_label.setText(
            f"<b>{self._label_text}</b> <i style='color:#888;'>{msg}</i>"
        )

    def _on_effects_error(self, msg: str) -> None:
        """Show error and restore title."""
        logger.warning("Effect apply failed: %s", msg)
        self._title_label.setText(f"<b>{self._label_text}</b> <i style='color:#e55;'>Error: {msg[:60]}</i>")
        QTimer.singleShot(5000, lambda: self._title_label.setText(f"<b>{self._label_text}</b>"))

    def _on_effects_applied(self, output_path: str) -> None:
        """Load the effects-baked result as the Modified version."""
        self._title_label.setText(f"<b>{self._label_text}</b>")
        self._modified_path = output_path
        self._mode_bar.setVisible(True)
        self._mode_bar.blockSignals(True)
        self._mode_bar.setCurrentIndex(1)  # Switch to Modified
        self._mode_bar.blockSignals(False)
        self._switch_to_path(output_path)

    def _switch_to_path(self, path: str) -> None:
        """Switch the media player to a different file without resetting effects state."""
        self._stop_audio_sink(clear_data=True)
        self._current_path = path
        if path and Path(path).is_file():
            self._extract_audio(path)
            self._player.setSource(QUrl.fromLocalFile(path))
            self._video_widget.set_file_path(path)
            self._pending_show_frame = True
            self._player.play()
        else:
            self._player.setSource(QUrl())

    @Slot(int)
    def _on_mode_tab_changed(self, index: int) -> None:
        """Switch between Original (0) and Modified (1)."""
        if index == 0 and self._original_path:
            self._switch_to_path(self._original_path)
        elif index == 1 and self._modified_path:
            self._switch_to_path(self._modified_path)

    def _show_effect_history(self) -> None:
        """Open the Effect History dialog."""
        from sdqt.widgets.effect_history import EffectHistoryDialog

        dlg = EffectHistoryDialog(self._effect_stack, parent=self)
        dlg.effects_changed.connect(self._on_history_changed)
        dlg.exec()

    def _on_history_changed(self, new_stack: list[dict]) -> None:
        """Update the effect stack from EffectHistoryDialog changes."""
        self._effect_stack = new_stack
        self._apply_effect_stack()

    # -- Trim / Crop / Lip Sync popup dialogs ------------------------------

    def _open_trim_dialog(self) -> None:
        src = self._original_path or self._current_path
        if not src:
            return
        from sdqt.widgets.trim_dialog import TrimDialog
        dlg = TrimDialog(src, parent=self.window())
        if dlg.exec() == QDialog.Accepted and dlg.result_path:
            self._load_dialog_result(dlg.result_path)

    def _open_crop_dialog(self) -> None:
        src = self._original_path or self._current_path
        if not src:
            return
        from sdqt.widgets.crop_dialog import CropDialog
        dlg = CropDialog(src, parent=self.window())
        if dlg.exec() == QDialog.Accepted and dlg.result_path:
            self._load_dialog_result(dlg.result_path)

    def _open_lipsync_dialog(self, engine: str) -> None:
        src = self._original_path or self._current_path
        if not src:
            return
        state = getattr(self, "_app_state", None)
        if engine == "ltx_a2vid":
            from sdqt.widgets.lipsync_ltx_dialog import LipSyncLTXDialog
            dlg = LipSyncLTXDialog(src, state, parent=self.window())
        elif engine == "musetalk":
            from sdqt.widgets.lipsync_musetalk_dialog import LipSyncMuseTalkDialog
            dlg = LipSyncMuseTalkDialog(src, state, parent=self.window())
        elif engine == "vace_multitalk":
            from sdqt.widgets.lipsync_vace_multitalk_dialog import LipSyncVaceMultitalkDialog
            dlg = LipSyncVaceMultitalkDialog(src, state, parent=self.window())
        elif engine == "latentsync":
            from sdqt.widgets.lipsync_latentsync_dialog import LipSyncLatentSyncDialog
            dlg = LipSyncLatentSyncDialog(src, state, parent=self.window())
        elif engine == "video_retalking":
            from sdqt.widgets.lipsync_video_retalking_dialog import LipSyncVideoRetalkingDialog
            dlg = LipSyncVideoRetalkingDialog(src, state, parent=self.window())
        else:
            return
        if dlg.exec() == QDialog.Accepted and getattr(dlg, "result_path", None):
            self._load_dialog_result(dlg.result_path)

    def _load_dialog_result(self, result_path: str) -> None:
        """Load a dialog result as the new Modified video (replacing the original)."""
        # The dialog result becomes the new base — reset effects
        self._original_path = result_path
        self._modified_path = None
        self._effect_stack = []
        self._mode_bar.setVisible(False)
        self._mode_bar.blockSignals(True)
        self._mode_bar.setCurrentIndex(0)
        self._mode_bar.blockSignals(False)
        # Load via normal path (keeps buttons/signals intact)
        self._stop_audio_sink(clear_data=True)
        self._current_path = result_path
        if result_path and Path(result_path).is_file():
            self._extract_audio(result_path)
            self._player.setSource(QUrl.fromLocalFile(result_path))
            self._video_widget.set_file_path(result_path)
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            self._pending_show_frame = True
            self._pending_auto_play = True
            self._player.play()
            self.video_loaded.emit(result_path)

    @staticmethod
    def _fmt(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"
