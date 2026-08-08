"""Character Tools tab — container for Face Swap, Body Double, RePose, ControlNet."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QTabWidget, QVBoxLayout

from sdqt.tabs.base import BaseTab
from sdqt.tabs.image_tabs.face_swap import FaceSwapTab
from sdqt.tabs.image_tabs.body_double import BodyDoubleTab
from sdqt.tabs.image_tabs.repose import RePoseTab
from sdqt.tabs.image_tabs.controlnet_tab import ControlNetTab

logger = logging.getLogger(__name__)


class CharacterToolsTab(BaseTab):
    """Container for character-oriented image tools."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()

        self._face_swap = FaceSwapTab(self.state)
        self._tab_widget.addTab(self._face_swap, "Face Swap")

        self._body_double = BodyDoubleTab(self.state)
        self._tab_widget.addTab(self._body_double, "Body Double")

        self._repose = RePoseTab(self.state)
        self._tab_widget.addTab(self._repose, "RePose")

        self._controlnet = ControlNetTab(self.state)
        self._tab_widget.addTab(self._controlnet, "ControlNet")

        self._sub_tabs = [self._face_swap, self._body_double,
                          self._repose, self._controlnet]
        layout.addWidget(self._tab_widget)

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        for tab in self._sub_tabs:
            tab.on_project_changed(project_name)

    def populate_options(self, **kwargs) -> None:
        """Forward SD options to sub-tabs that use ImageParamsWidget."""
        for tab in self._sub_tabs:
            if hasattr(tab, "populate_options"):
                tab.populate_options(**kwargs)
