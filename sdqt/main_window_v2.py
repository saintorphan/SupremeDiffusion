"""Modern sidebar-based main window layout.

Inherits all cross-tab wiring, shortcuts, status bar, and plugin logic
from MainWindow.  Only _build_ui is replaced — the flat QTabWidget tab
strip becomes a collapsible sidebar + QStackedWidget.

The trick:  self._tab_widget is set to the QStackedWidget.  Since
QStackedWidget has the same setCurrentWidget / currentWidget /
currentIndex / setCurrentIndex API as QTabWidget, all 800+ lines
of cross-tab dispatch code in MainWindow work unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.widgets.sidebar import SidebarWidget
from sdqt.widgets.project_bar import ProjectBar

# Import all tab classes (same as main_window.py)
from sdqt.tabs.generate import GenerateTab
from sdqt.tabs.video_extender import VideoExtenderTab
from sdqt.tabs.longshot import LongshotTab
from sdqt.tabs.still_grabber import StillGrabberTab
from sdqt.tabs.video_lora import VideoLoRATab
from sdqt.tabs.image_suite import ImageSuiteTab
from sdqt.tabs.browser import BrowserTab
from sdqt.tabs.audio_suite import AudioSuiteTab
from sdqt.tabs.timeline import TimelineTab
from sdqt.tabs.library_suite import LibrarySuiteTab
from sdqt.tabs.settings import SettingsTab
from sdqt.tabs.console import ConsoleTab
from sdqt.tabs.sequences_tab import SequencesTab
from sdqt.tabs.daz2supreme import Daz2SupremeTab
from sdqt.tabs.color_correction import ColorCorrectionTab
from sdqt.tabs.quick_gen import QuickGenTab
from sdqt.tabs.pipeline_wizard import PipelineWizardTab

# Import MainWindow for inheritance (cross-tab wiring, etc.)
from sdqt.main_window import MainWindow

logger = logging.getLogger(__name__)

# ── Styles ──────────────────────────────────────────────────────────────────

_BANNER_STYLE = (
    "QFrame#banner_v2 { background: #1e1e1e; border-bottom: 1px solid #333;"
    " padding: 0px; }"
)

_TITLE_STYLE = (
    "font-size: 17px; font-weight: bold; color: #0078d4;"
    " padding: 0 10px 0 4px;"
)

_HELP_BTN_STYLE = (
    "QPushButton { border: 1px solid #444; padding: 0px;"
    " background: #2a2a2a; border-radius: 4px;"
    " min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px; } "
    "QPushButton:hover { background: #505050; border-color: #888; } "
    "QPushButton:pressed { background: #606060; }"
)


class MainWindowV2(MainWindow):
    """Modern sidebar layout — inherits all logic from MainWindow."""

    # Map: sidebar key -> QStackedWidget widget
    # Built during _build_ui, used to sync sidebar <-> stack.
    _key_to_widget: dict[str, QWidget]
    _widget_to_key: dict[int, str]  # id(widget) -> key

    def _build_ui(self) -> None:
        """Override: sidebar + QStackedWidget instead of QTabWidget."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Sidebar (left) ──────────────────────────────────────────────
        self._sidebar = SidebarWidget()
        self._sidebar.item_selected.connect(self._on_sidebar_selected)

        self._sidebar.add_group("Create", [
            ("quick", "Quick Generate"),
            ("pipeline", "Pipeline Wizard"),
            ("video", "Video"),
            ("image", "Image"),
            ("audio", "Audio"),
        ])
        self._sidebar.add_group("Refine", [
            ("timeline", "Timeline"),
            ("sequences", "Sequences"),
            ("color", "Color Correct"),
        ])
        self._sidebar.add_group("Export", [
            ("outputs", "Outputs"),
        ])
        self._sidebar.add_group("Manage", [
            ("library", "Library"),
            ("daz", "Daz2Supreme"),
        ])

        # System items (pinned at bottom)
        self._sidebar.add_system_item("settings", "Settings")
        self._sidebar.add_system_item("console", "Console")

        self._sidebar.finalize()
        root.addWidget(self._sidebar)

        # ── Right panel (banner + content + status) ─────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Banner
        banner = self._build_banner()
        right_layout.addWidget(banner)

        # Content stack (replaces QTabWidget)
        self._content_stack = QStackedWidget()
        right_layout.addWidget(self._content_stack, 1)

        root.addWidget(right, 1)

        # ── Create all tab instances ────────────────────────────────────

        # Quick Generate
        self._quick_tab = QuickGenTab(self._state)

        # Pipeline Wizard
        self._pipeline_tab = PipelineWizardTab(self._state)

        # Timeline
        self._timeline_tab = TimelineTab(self._state)

        # Video suite (inner QTabWidget keeps its tab bar)
        self._generate_suite = QWidget()
        gs_layout = QVBoxLayout(self._generate_suite)
        gs_layout.setContentsMargins(0, 0, 0, 0)
        self._generate_tabs = QTabWidget()
        gs_layout.addWidget(self._generate_tabs)

        self._gen_tab = GenerateTab(self._state)
        self._generate_tabs.addTab(self._gen_tab, "Img2Vid")
        self._ve_tab = VideoExtenderTab(self._state)
        self._generate_tabs.addTab(self._ve_tab, "Video Extender")
        self._ls_tab = LongshotTab(self._state)
        self._generate_tabs.addTab(self._ls_tab, "Longshot")
        self._still_tab = StillGrabberTab(self._state)
        self._generate_tabs.addTab(self._still_tab, "StillGrabber")
        self._video_lora_tab = VideoLoRATab(self._state)
        self._generate_tabs.addTab(self._video_lora_tab, "Video LoRA")

        # Image suite
        self._img_tab = ImageSuiteTab(self._state)

        # Audio suite
        self._audio_tab = AudioSuiteTab(self._state)

        # Sequences
        self._sequences_tab = SequencesTab(self._state)

        # Color correction
        self._cc_tab = ColorCorrectionTab(self._state)

        # Outputs
        self._browser_tab = BrowserTab(self._state)

        # Library
        self._library_tab = LibrarySuiteTab(self._state)

        # Daz2Supreme
        self._daz_tab = Daz2SupremeTab(self._state)

        # Settings
        self._settings_tab = SettingsTab(self._state)

        # Console
        self._console_tab = ConsoleTab()

        # ── Populate content stack + key mapping ────────────────────────
        self._key_to_widget = {}
        self._widget_to_key = {}

        _items = [
            ("quick", self._quick_tab),
            ("pipeline", self._pipeline_tab),
            ("video", self._generate_suite),
            ("image", self._img_tab),
            ("audio", self._audio_tab),
            ("timeline", self._timeline_tab),
            ("sequences", self._sequences_tab),
            ("color", self._cc_tab),
            ("outputs", self._browser_tab),
            ("library", self._library_tab),
            ("daz", self._daz_tab),
            ("settings", self._settings_tab),
            ("console", self._console_tab),
        ]
        for key, widget in _items:
            self._content_stack.addWidget(widget)
            self._key_to_widget[key] = widget
            self._widget_to_key[id(widget)] = key

        # ── Plugins ─────────────────────────────────────────────────────
        self._plugin_tabs: list = []
        try:
            from plugins import discover_plugins
            app_root = Path(__file__).resolve().parent.parent
            for manifest, plugin_cls in discover_plugins(app_root / "plugins"):
                try:
                    tab = plugin_cls(self._state)
                    label = manifest.tab_label or manifest.name
                    self._content_stack.addWidget(tab)
                    key = f"plugin_{id(tab)}"
                    self._key_to_widget[key] = tab
                    self._widget_to_key[id(tab)] = key
                    self._sidebar.add_plugin_item(key, label)
                    self._plugin_tabs.append(tab)
                    logger.info("Plugin tab added: %s v%s", manifest.name, manifest.version)
                except Exception:
                    logger.exception("Failed to init plugin '%s'", manifest.name)
        except Exception:
            logger.debug("No plugins loaded (plugins/ not found or import error)")

        # ── Track all tabs for project change broadcast ─────────────────
        self._tabs = [
            self._quick_tab, self._pipeline_tab, self._timeline_tab, self._gen_tab,
            self._ve_tab, self._ls_tab, self._still_tab, self._video_lora_tab,
            self._img_tab, self._browser_tab, self._audio_tab,
            self._sequences_tab, self._cc_tab,
            self._library_tab, self._daz_tab, self._settings_tab,
        ] + self._plugin_tabs

        # ── KEY TRICK: alias _tab_widget to _content_stack ──────────────
        # QStackedWidget has the same navigation API as QTabWidget:
        #   setCurrentWidget, currentWidget, currentIndex, setCurrentIndex,
        #   count, indexOf
        # So all 800+ lines of cross-tab dispatch in MainWindow._connect_cross_tab
        # work unchanged.
        self._tab_widget = self._content_stack

        # Sync sidebar visual state when the stack changes (from cross-tab sends)
        self._content_stack.currentChanged.connect(self._sync_sidebar_from_stack)

        # Select initial item
        self._sidebar.select_item("quick", emit=False)

    def _build_banner(self) -> QFrame:
        """Build the top banner (project bar + help/chat buttons)."""
        banner = QFrame()
        banner.setObjectName("banner_v2")
        banner.setStyleSheet(_BANNER_STYLE)
        banner.setFixedHeight(50)
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # App title (shorter — sidebar has full branding)
        title = QLabel("\u25c6 Supreme Diffusion")
        title.setStyleSheet(_TITLE_STYLE)
        layout.addWidget(title)

        # Project bar
        self._project_bar = ProjectBar(self._state)
        self._project_bar.project_changed.connect(self._on_project_changed)
        layout.addWidget(self._project_bar)

        layout.addStretch()

        # Help button
        help_btn = QPushButton()
        help_btn.setToolTip("Help")
        help_btn.setIcon(self._make_help_icon())
        help_btn.setIconSize(QPixmap(24, 24).size())
        help_btn.setStyleSheet(_HELP_BTN_STYLE)
        help_btn.clicked.connect(self._show_help)
        layout.addWidget(help_btn)

        # AI chat button
        self._chat_btn = QPushButton()
        self._chat_btn.setToolTip("Qwen Assistant")
        self._chat_btn.setIcon(self._make_ai_icon())
        self._chat_btn.setIconSize(QPixmap(24, 24).size())
        self._chat_btn.setStyleSheet(_HELP_BTN_STYLE)
        self._chat_btn.clicked.connect(self._on_chat_clicked)
        layout.addWidget(self._chat_btn)

        return banner

    # ── Sidebar <-> Stack sync ──────────────────────────────────────────────

    @Slot(str)
    def _on_sidebar_selected(self, key: str) -> None:
        """User clicked a sidebar item — switch the content stack."""
        widget = self._key_to_widget.get(key)
        if widget is not None:
            self._content_stack.setCurrentWidget(widget)

    @Slot(int)
    def _sync_sidebar_from_stack(self, index: int) -> None:
        """Stack changed (via cross-tab send) — update sidebar highlight."""
        widget = self._content_stack.widget(index)
        if widget is not None:
            key = self._widget_to_key.get(id(widget), "")
            if key:
                self._sidebar.select_item(key, emit=False)

    # ── Overrides for sidebar-aware navigation ──────────────────────────────

    def _toggle_console(self) -> None:
        self._content_stack.setCurrentWidget(self._console_tab)

    def _on_generate_shortcut(self) -> None:
        """Ctrl+G — generate in active video tab."""
        if self._content_stack.currentWidget() is not self._generate_suite:
            return
        current = self._generate_tabs.currentWidget()
        if current is self._gen_tab:
            self._gen_tab._on_generate()
        elif current is self._ve_tab:
            self._ve_tab._on_generate()

    def _on_abort_shortcut(self) -> None:
        """Escape — abort in active video tab."""
        if self._content_stack.currentWidget() is not self._generate_suite:
            return
        current = self._generate_tabs.currentWidget()
        if current is self._gen_tab:
            self._gen_tab._on_abort()
        elif current is self._ve_tab:
            self._ve_tab._on_abort()

    # ── Session restore/save with sidebar key ───────────────────────────────

    def _restore_last_session(self) -> None:
        try:
            cfg = self._state.global_config
            if cfg.last_project:
                projects = self._state.project_manager.list_projects()
                if cfg.last_project in projects:
                    self._project_bar._combo.setCurrentText(cfg.last_project)
            # Restore sidebar selection
            key = cfg.last_sidebar_key
            if key and key in self._key_to_widget:
                self._sidebar.select_item(key)
            # Restore video sub-tab
            if 0 <= cfg.last_video_tab_idx < self._generate_tabs.count():
                self._generate_tabs.setCurrentIndex(cfg.last_video_tab_idx)
        except Exception:
            pass

    def _save_session_state(self) -> None:
        try:
            cfg = self._state.global_config
            cfg.last_project = self._project_bar.current_project or ""
            cfg.last_sidebar_key = self._sidebar.current_key
            # Also save tab index for classic mode compatibility
            cfg.last_parent_tab_idx = self._content_stack.currentIndex()
            cfg.last_video_tab_idx = self._generate_tabs.currentIndex()
            cfg.save()
        except Exception:
            pass
