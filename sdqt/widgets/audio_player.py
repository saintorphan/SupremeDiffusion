"""Reusable audio-only player widget with playback controls and send-to."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

_ICON_BTN_STYLE = (
    "QPushButton { border: none; padding: 2px 4px; font-size: 13px; color: #aaa; }"
    "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
)


class _ClickSlider(QSlider):
    """QSlider that jumps to the clicked position."""

    click_position = Signal(int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = QStyle.sliderValueFromPosition(
                self.minimum(), self.maximum(),
                int(event.position().x()), self.width(),
            )
            self.setValue(val)
            self.click_position.emit(val)
        super().mousePressEvent(event)


class AudioPlayerWidget(QWidget):
    """Compact audio player with play/pause, scrubber, time label, and send-to.

    Signals:
        audio_loaded(str): emitted when an audio file is loaded (path)
        audio_cleared(): emitted when audio is cleared
        send_audio_requested(str, str): (key, path) for send-to routing
    """

    audio_loaded = Signal(str)
    audio_cleared = Signal()
    send_audio_requested = Signal(str, str)

    def __init__(self, label: str = "Audio Preview", parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFixedHeight(80)
        self._current_path: str | None = None
        self._duration_ms: int = 0
        self._send_targets: list = []

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(1.0)
        self._player.setAudioOutput(self._audio_output)

        # -- Layout --------------------------------------------------------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        # Title row
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(2)

        self._title_label = QLabel(f"<b>{label}</b>")
        self._title_label.setFixedHeight(18)
        title_row.addWidget(self._title_label)

        self._file_label = QLabel("")
        self._file_label.setStyleSheet("color: #888; font-size: 11px;")
        title_row.addWidget(self._file_label, 1)

        self._browse_btn = QPushButton("\u2191")  # ↑
        self._browse_btn.setFixedSize(24, 20)
        self._browse_btn.setToolTip("Browse for audio file")
        self._browse_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._browse_btn.clicked.connect(self._browse_audio)
        title_row.addWidget(self._browse_btn)

        self._folder_btn = QPushButton("\U0001F4C2")  # 📂
        self._folder_btn.setFixedSize(24, 20)
        self._folder_btn.setToolTip("Open containing folder")
        self._folder_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._folder_btn.clicked.connect(self._open_folder)
        self._folder_btn.setVisible(False)
        title_row.addWidget(self._folder_btn)

        self._clear_btn = QPushButton("\u2715")  # ✕
        self._clear_btn.setFixedSize(24, 20)
        self._clear_btn.setToolTip("Clear audio")
        self._clear_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._clear_btn.clicked.connect(self.clear_audio)
        self._clear_btn.setVisible(False)
        title_row.addWidget(self._clear_btn)

        layout.addLayout(title_row)

        # Controls row
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(32, 24)
        self._play_btn.clicked.connect(self._toggle_play)
        controls.addWidget(self._play_btn)

        self._slider = _ClickSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setFixedHeight(16)
        self._slider.sliderMoved.connect(self._seek)
        self._slider.click_position.connect(self._seek)
        controls.addWidget(self._slider, 1)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setFixedWidth(90)
        self._time_label.setFixedHeight(16)
        controls.addWidget(self._time_label)

        layout.addLayout(controls)

        # -- Signals -------------------------------------------------------
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)

        # Context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    # -- Public API --------------------------------------------------------

    def load_audio(self, path: str | None) -> None:
        """Load an audio file for playback."""
        # Block signals during source swap to prevent re-entrant state changes
        self._player.blockSignals(True)
        try:
            self._player.stop()
            if path and Path(path).is_file():
                self._player.setSource(QUrl.fromLocalFile(path))
            else:
                self._player.setSource(QUrl())
        finally:
            self._player.blockSignals(False)
        self._current_path = path
        if path and Path(path).is_file():
            self._file_label.setText(Path(path).name)
            self._file_label.setToolTip(path)
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            self._play_btn.setText("\u25b6")
            self.audio_loaded.emit(path)
        else:
            self._duration_ms = 0
            self._file_label.setText("")
            self._slider.setRange(0, 0)
            self._time_label.setText("0:00 / 0:00")
            self._play_btn.setText("\u25b6")
            self._folder_btn.setVisible(False)
            self._clear_btn.setVisible(False)

    def clear_audio(self) -> None:
        """Stop and clear the loaded audio."""
        # Block signals to prevent re-entrant state change handlers
        self._player.blockSignals(True)
        try:
            self._player.stop()
            self._player.setSource(QUrl())
        finally:
            self._player.blockSignals(False)
        self._current_path = None
        self._duration_ms = 0
        self._file_label.setText("")
        self._file_label.setToolTip("")
        self._slider.setRange(0, 0)
        self._time_label.setText("0:00 / 0:00")
        self._play_btn.setText("\u25b6")
        self._folder_btn.setVisible(False)
        self._clear_btn.setVisible(False)
        self.audio_cleared.emit()

    @property
    def audio_path(self) -> str | None:
        return self._current_path

    def set_send_targets(self, targets: list) -> None:
        """Set send-to targets for context menu. Same nested tuple format as _CLIP_TARGETS."""
        self._send_targets = targets

    # -- Drag/drop ---------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _AUDIO_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _AUDIO_EXTS:
                self.load_audio(path)
                event.acceptProposedAction()
                return

    # -- Playback controls -------------------------------------------------

    @Slot()
    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            if (
                self._player.playbackState() == QMediaPlayer.StoppedState
                or self._player.position() >= self._duration_ms - 100
            ):
                self._player.setPosition(0)
            self._player.play()

    @Slot(int)
    def _seek(self, position_ms: int) -> None:
        self._player.setPosition(position_ms)

    @Slot(int)
    def _on_position(self, pos_ms: int) -> None:
        if not self._slider.isSliderDown():
            self._slider.setValue(pos_ms)
        self._time_label.setText(
            f"{self._fmt(pos_ms)} / {self._fmt(self._duration_ms)}"
        )

    @Slot(int)
    def _on_duration(self, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        self._slider.setRange(0, dur_ms)

    @Slot(QMediaPlayer.PlaybackState)
    def _on_state(self, state: QMediaPlayer.PlaybackState) -> None:
        self._play_btn.setText(
            "\u23f8" if state == QMediaPlayer.PlayingState else "\u25b6"
        )

    # -- Context menu ------------------------------------------------------

    @Slot()
    def _show_context_menu(self, pos) -> None:
        if not self._current_path or not self._send_targets:
            return
        menu = QMenu(self)
        send_menu = menu.addMenu("Send to")
        for entry in self._send_targets:
            if isinstance(entry[1], list):
                sub = send_menu.addMenu(entry[0])
                for key, name in entry[1]:
                    action = sub.addAction(name)
                    action.triggered.connect(
                        lambda checked, k=key: self.send_audio_requested.emit(
                            k, self._current_path))
            else:
                key, name = entry
                action = send_menu.addAction(name)
                action.triggered.connect(
                    lambda checked, k=key: self.send_audio_requested.emit(
                        k, self._current_path))
        menu.exec(self.mapToGlobal(pos))

    # -- Browse / folder ---------------------------------------------------

    @Slot()
    def _browse_audio(self) -> None:
        path, _ = get_open_filename(
            self, "Select Audio", "",
            "Audio (*.wav *.mp3 *.flac *.m4a *.ogg)",
        )
        if path:
            self.load_audio(path)

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

    @staticmethod
    def _fmt(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"
