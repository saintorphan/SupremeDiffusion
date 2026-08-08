"""Audio suite — container with Chatterbox TTS, Dia TTS, MusicGen, and AudioGen sub-tabs."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QTabWidget, QVBoxLayout

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab
from sdqt.tabs.chatterbox_tts import ChatterboxTTSTab
from sdqt.tabs.dia_tts import DiaTTSTab
from sdqt.tabs.musicgen_tab import MusicGenTab
from sdqt.tabs.audiogen_tab import AudioGenTab
from sdqt.tabs.sadtalker_tab import SadTalkerTab

logger = logging.getLogger(__name__)


class AudioSuiteTab(BaseTab):
    """Container tab with Chatterbox TTS, Dia TTS, MusicGen, and AudioGen sub-tabs."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._sub_tabs: list[BaseTab] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()

        self._chatterbox = ChatterboxTTSTab(self.state)
        self._tab_widget.addTab(self._chatterbox, "Chatterbox TTS")

        self._dia = DiaTTSTab(self.state)
        self._tab_widget.addTab(self._dia, "Dia TTS")

        self._musicgen = MusicGenTab(self.state)
        self._tab_widget.addTab(self._musicgen, "MusicGen")

        self._audiogen = AudioGenTab(self.state)
        self._tab_widget.addTab(self._audiogen, "AudioGen")

        self._sadtalker = SadTalkerTab(self.state)
        self._tab_widget.addTab(self._sadtalker, "SadTalker")

        self._sub_tabs = [
            self._chatterbox, self._dia, self._musicgen, self._audiogen,
            self._sadtalker,
        ]

        layout.addWidget(self._tab_widget)

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        for tab in self._sub_tabs:
            tab.on_project_changed(project_name)

    def load_audio(self, path: str) -> None:
        """Load audio into the Chatterbox tab preview."""
        self._chatterbox.load_audio(path)

    def load_ref_audio(self, path: str) -> None:
        """Load a reference voice for Chatterbox voice cloning."""
        self._chatterbox.load_ref_audio(path)

    def load_chatterbox_audio(self, path: str) -> None:
        """Load audio into the Chatterbox tab preview."""
        self._chatterbox.load_audio(path)

    def load_chatterbox_ref_audio(self, path: str) -> None:
        """Load a reference voice for Chatterbox voice cloning."""
        self._chatterbox.load_ref_audio(path)

    def load_dia_audio(self, path: str) -> None:
        """Load audio into the Dia tab preview."""
        self._dia.load_audio(path)

    def load_dia_ref_audio(self, path: str) -> None:
        """Load a reference voice for Dia voice cloning."""
        self._dia.load_ref_audio(path)

    def load_sadtalker_source(self, path: str) -> None:
        """Route an image into the SadTalker source slot."""
        self._sadtalker.load_source(path)
        self._tab_widget.setCurrentWidget(self._sadtalker)

    def load_sadtalker_audio(self, path: str) -> None:
        """Route audio into the SadTalker driving-audio slot."""
        self._sadtalker.load_audio(path)
        self._tab_widget.setCurrentWidget(self._sadtalker)

    def load_sadtalker_ref_video(self, path: str) -> None:
        """Route a video into the SadTalker reference-video slot."""
        self._sadtalker.load_ref_video(path)
        self._tab_widget.setCurrentWidget(self._sadtalker)
