"""Main application window."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.utils.codec import (
    configured_codec_args as _codec_args,
    configured_encoder_name as _enc_name,
    pix_fmt_args as _pix_fmt,
)
from sdqt.widgets.project_bar import ProjectBar
from sdqt.tabs.generate import GenerateTab
from sdqt.tabs.video_extender import VideoExtenderTab
from sdqt.tabs.txt2prompt import Txt2PromptTab
from sdqt.tabs.longshot import LongshotTab
from sdqt.tabs.pose_animate import PoseAnimateTab
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

logger = logging.getLogger(__name__)

# ── Banner styles ────────────────────────────────────────────────────────────

_BANNER_STYLE = (
    "QFrame#banner { background: #1e1e1e; border-bottom: 1px solid #333;"
    " padding: 0px; }"
)

_TITLE_STYLE = (
    "font-size: 17px; font-weight: bold; color: #0078d4;"
    " padding: 0 10px 0 4px;"
)

_HELP_BTN_STYLE = (
    "QPushButton { border: 1px solid #444; padding: 0px;"
    " background: #2a2a2a; border-radius: 4px;"
    " min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; } "
    "QPushButton:hover { background: #505050; border-color: #888; } "
    "QPushButton:pressed { background: #606060; }"
)

_TAB_STYLE = (
    "QTabWidget::pane { border: none; }"
    "QTabBar { qproperty-drawBase: 0; }"
    "QTabBar::tab { padding: 8px 18px; font-size: 14px; font-weight: bold;"
    " color: #999; background: #1e1e1e; border: none; min-width: 100px;"
    " border-bottom: 2px solid transparent; margin-right: 2px; }"
    "QTabBar::tab:selected { color: #fff; border-bottom: 2px solid #0078d4;"
    " background: #252525; }"
    "QTabBar::tab:hover { color: #ddd; background: #2a2a2a; }"
    "QTabBar::scroller { width: 20px; }"
)


class MainWindow(QMainWindow):
    """Supreme Diffusion Qt -- main window with header banner and tab container."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Supreme Diffusion")
        self.setMinimumSize(1500, 950)

        self._state = AppState()
        self._tabs: list = []
        self._chat_dialog = None

        self._build_ui()
        self._connect_cross_tab()
        self._setup_shortcuts()
        self._setup_status_bar()

        # Populate Image Suite dropdowns eagerly (before project restore so
        # checkpoint/sampler/VAE selections can be restored from config)
        try:
            self._img_tab.populate_sd_options()
        except Exception:
            pass

        # Restore last project + tab from global config
        self._restore_last_session()

        # Fire initial on_project_changed for all tabs (profiles, config restore)
        self._project_bar.emit_initial_project()

        # Auto-load configured pipelines after the window shows
        QTimer.singleShot(100, self._auto_load_pipelines)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header Banner ────────────────────────────────────────────────────
        banner = QFrame()
        banner.setObjectName("banner")
        banner.setStyleSheet(_BANNER_STYLE)
        banner.setFixedHeight(58)
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        banner_layout.setSpacing(8)

        # App title
        title = QLabel("\u25c6 Supreme Diffusion")  # ◆
        title.setStyleSheet(_TITLE_STYLE)
        banner_layout.addWidget(title)

        # Project bar (combo + CRUD buttons)
        self._project_bar = ProjectBar(self._state)
        self._project_bar.project_changed.connect(self._on_project_changed)
        banner_layout.addWidget(self._project_bar)

        banner_layout.addStretch()

        # Help button
        help_btn = QPushButton()
        help_btn.setToolTip("Help")
        help_btn.setIcon(self._make_help_icon())
        help_btn.setIconSize(QPixmap(28, 28).size())
        help_btn.setStyleSheet(_HELP_BTN_STYLE)
        help_btn.clicked.connect(self._show_help)
        banner_layout.addWidget(help_btn)

        # AI chat button
        self._chat_btn = QPushButton()
        self._chat_btn.setToolTip("Qwen Assistant")
        self._chat_btn.setIcon(self._make_ai_icon())
        self._chat_btn.setIconSize(QPixmap(28, 28).size())
        self._chat_btn.setStyleSheet(_HELP_BTN_STYLE)
        self._chat_btn.clicked.connect(self._on_chat_clicked)
        banner_layout.addWidget(self._chat_btn)

        root.addWidget(banner)

        # ── Tab container ────────────────────────────────────────────────────
        tab_container = QWidget()
        tab_layout = QVBoxLayout(tab_container)
        tab_layout.setContentsMargins(4, 2, 4, 0)
        tab_layout.setSpacing(0)

        self._tab_widget = QTabWidget()
        self._tab_widget.setStyleSheet(_TAB_STYLE)
        # 15+ tabs: enable horizontal scroll buttons + elide long labels so
        # nothing truncates / falls off on narrower (laptop) windows.
        self._tab_widget.setUsesScrollButtons(True)
        self._tab_widget.tabBar().setUsesScrollButtons(True)
        self._tab_widget.setElideMode(Qt.TextElideMode.ElideRight)
        tab_layout.addWidget(self._tab_widget, 1)

        # ── 1. Timeline (direct top-level tab) ──
        self._timeline_tab = TimelineTab(self._state)
        self._tab_widget.addTab(self._timeline_tab, "Timeline")

        # ── 2. Video (Img2Vid, Video Extender, Longshot, Process) ──
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

        self._mimicmotion_tab = PoseAnimateTab(self._state)
        self._generate_tabs.addTab(self._mimicmotion_tab, "Pose Animate")

        self._still_tab = StillGrabberTab(self._state)
        self._generate_tabs.addTab(self._still_tab, "StillGrabber")

        self._video_lora_tab = VideoLoRATab(self._state)
        self._generate_tabs.addTab(self._video_lora_tab, "Video LoRA")

        self._tab_widget.addTab(self._generate_suite, "Video")

        # ── Txt2Prompt (LLM-written, model-formatted prompts) ──
        self._txt2prompt_tab = Txt2PromptTab(self._state)
        self._tab_widget.addTab(self._txt2prompt_tab, "Txt2Prompt")
        self._txt2prompt_tab.send_to_requested.connect(self._dispatch_prompt_send)

        # ── 3. Image Suite ──
        self._img_tab = ImageSuiteTab(self._state)
        self._tab_widget.addTab(self._img_tab, "Image")

        # ── 4. Audio ──
        self._audio_tab = AudioSuiteTab(self._state)
        self._tab_widget.addTab(self._audio_tab, "Audio")

        # ── 5. Quick Generate (easy mode) ──
        self._quick_tab = QuickGenTab(self._state)
        self._tab_widget.addTab(self._quick_tab, "Quick")

        # ── 5b. Pipeline Wizard (guided workflow) ──
        self._pipeline_tab = PipelineWizardTab(self._state)
        self._tab_widget.addTab(self._pipeline_tab, "Pipeline")

        # ── 6. Sequences ──
        self._sequences_tab = SequencesTab(self._state)
        self._tab_widget.addTab(self._sequences_tab, "Sequences")

        # ── 6. Color Correct ──
        self._cc_tab = ColorCorrectionTab(self._state)
        self._tab_widget.addTab(self._cc_tab, "Color Correct")

        # ── 7. Outputs ──
        self._browser_tab = BrowserTab(self._state)
        self._tab_widget.addTab(self._browser_tab, "Outputs")

        # ── 7. Library ──
        self._library_tab = LibrarySuiteTab(self._state)
        self._tab_widget.addTab(self._library_tab, "Library")

        # ── 8. Daz2Supreme ──
        self._daz_tab = Daz2SupremeTab(self._state)
        self._tab_widget.addTab(self._daz_tab, "Daz2Supreme")

        # ── 9. Settings ──
        self._settings_tab = SettingsTab(self._state)
        self._tab_widget.addTab(self._settings_tab, "Settings")

        # ── 10. Console (far right) ──
        self._console_tab = ConsoleTab()
        self._tab_widget.addTab(self._console_tab, "Console")

        # ── 11. Dynamic plugins (manifest-validated) ──
        self._plugin_tabs: list = []
        try:
            from plugins import discover_plugins
            app_root = Path(__file__).resolve().parent.parent
            for manifest, plugin_cls in discover_plugins(app_root / "plugins"):
                try:
                    tab = plugin_cls(self._state)
                    label = manifest.tab_label or manifest.name
                    self._tab_widget.insertTab(
                        self._tab_widget.indexOf(self._console_tab), tab, label,
                    )
                    self._plugin_tabs.append(tab)
                    logger.info("Plugin tab added: %s v%s", manifest.name, manifest.version)
                except Exception:
                    logger.exception("Failed to init plugin '%s'", manifest.name)
        except Exception:
            logger.debug("No plugins loaded (plugins/ not found or import error)")

        root.addWidget(tab_container, 1)

        # Track all tabs for project change broadcast
        self._tabs = [
            self._quick_tab, self._pipeline_tab, self._timeline_tab, self._gen_tab,
            self._ve_tab, self._ls_tab, self._mimicmotion_tab, self._still_tab,
            self._video_lora_tab, self._txt2prompt_tab,
            self._img_tab, self._browser_tab, self._audio_tab,
            self._sequences_tab, self._cc_tab,
            self._library_tab, self._daz_tab, self._settings_tab,
        ] + self._plugin_tabs

    def _dispatch_prompt_send(self, target: str, payload: dict) -> None:
        """Route a Txt2Prompt 'send-to' into Generate / Video Extender."""
        self._tab_widget.setCurrentWidget(self._generate_suite)
        if target == "video_extender":
            self._generate_tabs.setCurrentWidget(self._ve_tab)
            self._ve_tab.apply_prompt_payload(payload)
        else:
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab.apply_prompt_payload(payload)

    def _connect_cross_tab(self) -> None:
        """Wire cross-tab send-to via right-click context menus on video players."""

        # Route all tab status messages to the main status bar
        for tab in self._tabs:
            if hasattr(tab, "status_message"):
                tab.status_message.connect(self._on_tab_status)
        # Also connect Image Suite sub-tabs (nested inside the container)
        for sub in getattr(self._img_tab, "_sub_tabs", []):
            if hasattr(sub, "status_message"):
                sub.status_message.connect(self._on_tab_status)
            # And sub-sub-tabs (SDXL container has txt2img/img2img/inpaint etc.)
            for nested in getattr(sub, "_sub_tabs", []):
                if hasattr(nested, "status_message"):
                    nested.status_message.connect(self._on_tab_status)
        # Audio suite sub-tabs
        for sub in getattr(self._audio_tab, "_sub_tabs", []):
            if hasattr(sub, "status_message"):
                sub.status_message.connect(self._on_tab_status)
            # Wire send_audio_requested from each audio sub-tab's preview
            if hasattr(sub, "_audio_preview"):
                sub._audio_preview.send_audio_requested.connect(
                    self._dispatch_audio_send
                )
            # Wire Voice Library's own send signal
            if hasattr(sub, "send_audio_requested"):
                sub.send_audio_requested.connect(self._dispatch_audio_send)

        # Library suite sub-tabs
        for sub in getattr(self._library_tab, "_sub_tabs", []):
            if hasattr(sub, "status_message"):
                sub.status_message.connect(self._on_tab_status)
            if hasattr(sub, "_audio_preview"):
                sub._audio_preview.send_audio_requested.connect(
                    self._dispatch_audio_send)
            if hasattr(sub, "send_audio_requested"):
                sub.send_audio_requested.connect(self._dispatch_audio_send)
            if hasattr(sub, "send_image_requested"):
                sub.send_image_requested.connect(
                    lambda key, path: self._send_to_image_tab(key, path))

        # SD pipeline loaded -> populate Image Suite dropdowns
        self._settings_tab.sd_pipelines_loaded.connect(self._img_tab.populate_sd_options)

        # Denoising loop -> gate wan2gp-only controls in video gen tabs
        self._settings_tab.denoising_loop_changed.connect(self._gen_tab.set_denoising_loop)
        self._settings_tab.denoising_loop_changed.connect(self._ve_tab._params.set_denoising_loop)
        # Apply initial value
        _initial_loop = self._state.global_config.denoising_loop or "standard"
        self._gen_tab.set_denoising_loop(_initial_loop)
        self._ve_tab._params.set_denoising_loop(_initial_loop)

        # -- Shared target lists (centralized in send_targets.py) ----------------
        from sdqt.widgets.send_targets import (
            CLIP_TARGETS, IMAGE_TARGETS, IMAGE_TARGETS_SIMPLE,
            FINAL_FRAME_TARGETS, GUIDE_VIDEO_TARGETS,
        )
        _clip_targets = CLIP_TARGETS
        _frame_targets = IMAGE_TARGETS
        _final_frame_targets = FINAL_FRAME_TARGETS
        _guide_video_targets = GUIDE_VIDEO_TARGETS

        # -- Generate tab: preview player --------------------------------------
        self._gen_tab._preview.set_send_targets(
            clip_targets=_clip_targets, frame_targets=_frame_targets,
            final_frame_targets=_final_frame_targets,
        )
        self._gen_tab._preview.set_guide_video_targets(_guide_video_targets)
        self._gen_tab._preview.send_clip_requested.connect(
            lambda key: self._dispatch_video_send(key, self._gen_tab._preview.video_path)
        )
        self._gen_tab._preview.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(self._gen_tab._preview, key)
        )
        self._gen_tab._preview.final_frame_requested.connect(
            lambda key: self._capture_and_send_final_frame(self._gen_tab._preview, key)
        )
        self._gen_tab._preview.guide_video_requested.connect(
            lambda key: self._capture_and_send_guide_video(self._gen_tab._preview, key)
        )

        # -- Generate tab: source image widgets (Mode 1 image, Mode 2 frames) ---
        for _img_widget in (
            self._gen_tab._image_source,
            self._gen_tab._first_frame,
            self._gen_tab._last_frame,
        ):
            _img_widget.set_send_targets(_frame_targets)
            _img_widget.set_final_frame_targets(_final_frame_targets)
            _img_widget.send_requested.connect(
                lambda key, w=_img_widget: self._send_to_image_tab(key, w.image_path)
            )
            _img_widget.final_frame_requested.connect(
                lambda key, w=_img_widget: self._send_final_frame(w.image_path, key)
            )

        # -- Video Extender: source + preview players --------------------------
        self._ve_tab._source_player.set_send_targets(
            clip_targets=_clip_targets, frame_targets=_frame_targets,
            final_frame_targets=_final_frame_targets,
        )
        self._ve_tab._source_player.send_clip_requested.connect(
            lambda key: self._dispatch_video_send(key, self._ve_tab._source_player.video_path)
        )
        self._ve_tab._source_player.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(self._ve_tab._source_player, key)
        )
        self._ve_tab._source_player.final_frame_requested.connect(
            lambda key: self._capture_and_send_final_frame(self._ve_tab._source_player, key)
        )
        self._ve_tab._source_player.set_guide_video_targets(_guide_video_targets)
        self._ve_tab._source_player.guide_video_requested.connect(
            lambda key: self._capture_and_send_guide_video(self._ve_tab._source_player, key)
        )

        self._ve_tab._preview.set_send_targets(
            clip_targets=_clip_targets, frame_targets=_frame_targets,
            final_frame_targets=_final_frame_targets,
        )
        self._ve_tab._preview.send_clip_requested.connect(
            lambda key: self._dispatch_video_send(key, self._ve_tab._preview.video_path)
        )
        self._ve_tab._preview.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(self._ve_tab._preview, key)
        )
        self._ve_tab._preview.final_frame_requested.connect(
            lambda key: self._capture_and_send_final_frame(self._ve_tab._preview, key)
        )
        self._ve_tab._preview.set_guide_video_targets(_guide_video_targets)
        self._ve_tab._preview.guide_video_requested.connect(
            lambda key: self._capture_and_send_guide_video(self._ve_tab._preview, key)
        )

        # -- Longshot: source, gapfill, output players -------------------------
        for player in (self._ls_tab._source_player, self._ls_tab._gapfill_player, self._ls_tab._output_player):
            player.set_send_targets(
                clip_targets=_clip_targets, frame_targets=_frame_targets,
                final_frame_targets=_final_frame_targets,
            )
            player.send_clip_requested.connect(
                lambda key, p=player: self._dispatch_video_send(key, p.video_path)
            )
            player.send_frame_requested.connect(
                lambda key, p=player: self._capture_and_send_frame(p, key)
            )
            player.final_frame_requested.connect(
                lambda key, p=player: self._capture_and_send_final_frame(p, key)
            )
            player.set_guide_video_targets(_guide_video_targets)
            player.guide_video_requested.connect(
                lambda key, p=player: self._capture_and_send_guide_video(p, key)
            )

        # -- Pose Animate: driving + preview players ---------------------------
        for player in (self._mimicmotion_tab._driver_player, self._mimicmotion_tab._preview):
            player.set_send_targets(
                clip_targets=_clip_targets, frame_targets=_frame_targets,
                final_frame_targets=_final_frame_targets,
            )
            player.send_clip_requested.connect(
                lambda key, p=player: self._dispatch_video_send(key, p.video_path)
            )
            player.send_frame_requested.connect(
                lambda key, p=player: self._capture_and_send_frame(p, key)
            )
            player.final_frame_requested.connect(
                lambda key, p=player: self._capture_and_send_final_frame(p, key)
            )
        # Reference image drop: route "Send to" actions back through image dispatch
        self._mimicmotion_tab._ref_drop.set_send_targets(_frame_targets)
        self._mimicmotion_tab._ref_drop.send_requested.connect(
            lambda key: self._send_to_image_tab(key, self._mimicmotion_tab._ref_drop.image_path())
        )

        # -- StillGrabber player ---------------------------------------------------
        self._still_tab._player.set_send_targets(
            clip_targets=_clip_targets, frame_targets=_frame_targets,
            final_frame_targets=_final_frame_targets,
        )
        self._still_tab._player.send_clip_requested.connect(
            lambda key: self._dispatch_video_send(key, self._still_tab._player.video_path)
        )
        self._still_tab._player.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(self._still_tab._player, key)
        )
        self._still_tab._player.final_frame_requested.connect(
            lambda key: self._capture_and_send_final_frame(self._still_tab._player, key)
        )
        self._still_tab.send_image_requested.connect(
            lambda key, path: self._send_to_image_tab(
                key, path, settings=self._get_active_image_settings())
        )

        # -- Enable Effects menu + popup dialogs on result players ---------------
        _result_players = [
            self._gen_tab._preview,
            self._ve_tab._preview,
            self._ls_tab._output_player,
        ]
        for player in _result_players:
            player.enable_effects_menu(True)
            player.set_state_ref(self._state)

        # -- Browser: send-to signals (context menus + video player right-click) -
        self._browser_tab.send_clip_requested.connect(
            lambda key, path: self._dispatch_video_send(key, path)
        )
        self._browser_tab.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(
                self._browser_tab._video_preview, key)
        )
        self._browser_tab.send_image_requested.connect(
            lambda key, path: self._send_to_image_tab(
                key, path, settings=self._get_active_image_settings())
        )
        self._browser_tab.send_audio_requested.connect(self._dispatch_audio_send)
        self._browser_tab._video_preview.send_clip_requested.connect(
            lambda key: self._dispatch_video_send(
                key, self._browser_tab._video_preview.video_path)
        )
        self._browser_tab._video_preview.send_frame_requested.connect(
            lambda key: self._capture_and_send_frame(
                self._browser_tab._video_preview, key)
        )
        self._browser_tab._video_preview.final_frame_requested.connect(
            lambda key: self._capture_and_send_final_frame(
                self._browser_tab._video_preview, key)
        )
        self._browser_tab._video_preview.guide_video_requested.connect(
            lambda key: self._capture_and_send_guide_video(
                self._browser_tab._video_preview, key)
        )
        self._browser_tab._video_preview.color_ref_requested.connect(
            self._send_color_ref
        )

        # -- Daz2Supreme send-to ---------------------------------------------------
        self._daz_tab.send_image_requested.connect(
            lambda key, path: self._send_to_image_tab(
                key, path, settings=self._get_active_image_settings())
        )

        # -- Image Suite sub-tab send-to (right-click on gallery) ---------------
        _image_send_targets = IMAGE_TARGETS_SIMPLE
        _gallery_tabs = []
        _gallery_tabs.extend(self._img_tab._gen.get_sub_tabs())
        # Add top-level image suite tabs that aren't in containers
        for t in [self._img_tab._face_swap, self._img_tab._body_double, self._img_tab._repose, self._img_tab._crop_zoom, self._img_tab._img_edit, self._img_tab._model3d]:
            if hasattr(t, "_gallery"):
                _gallery_tabs.append(t)
        for sub_tab in _gallery_tabs:
            if not hasattr(sub_tab, "_gallery"):
                continue
            sub_tab._gallery.set_send_targets(_image_send_targets)
            sub_tab._gallery.set_final_frame_targets(_final_frame_targets)
            sub_tab._gallery.set_guide_video_targets(_guide_video_targets)
            sub_tab._gallery.send_requested.connect(
                lambda key, t=sub_tab: self._dispatch_image_subtab_send(t, key)
            )
            sub_tab._gallery.final_frame_requested.connect(
                lambda key, t=sub_tab: self._dispatch_gallery_final_frame(t, key)
            )
            sub_tab._gallery.guide_video_requested.connect(
                lambda key, t=sub_tab: self._dispatch_gallery_guide_video(t, key)
            )
            if hasattr(sub_tab._gallery, "color_ref_requested"):
                sub_tab._gallery.color_ref_requested.connect(self._send_color_ref)

        # -- Browser file list: wire color_ref_requested if present ------------
        for browser in self._browser_tab._browsers:
            if hasattr(browser, "color_ref_requested"):
                browser.color_ref_requested.connect(self._send_color_ref)

        # -- Source image widgets (ImageDropWidget / MaskEditorWidget) ----------
        _source_widgets = []
        # Unified gen tab sub-tabs
        for sub_tab in self._img_tab._gen.get_sub_tabs():
            if hasattr(sub_tab, "_source"):
                _source_widgets.append(sub_tab._source)
        # face_swap source
        if hasattr(self._img_tab._face_swap, "_source"):
            _source_widgets.append(self._img_tab._face_swap._source)
        # body_double source + target
        if hasattr(self._img_tab._body_double, "_source"):
            _source_widgets.append(self._img_tab._body_double._source)
        if hasattr(self._img_tab._body_double, "_target"):
            _source_widgets.append(self._img_tab._body_double._target)
        # repose source + pose_ref
        if hasattr(self._img_tab._repose, "_source"):
            _source_widgets.append(self._img_tab._repose._source)
        if hasattr(self._img_tab._repose, "_pose_ref"):
            _source_widgets.append(self._img_tab._repose._pose_ref)
        # crop_zoom source
        if hasattr(self._img_tab._crop_zoom, "_source"):
            _source_widgets.append(self._img_tab._crop_zoom._source)

        for widget in _source_widgets:
            widget.set_send_targets(IMAGE_TARGETS)
            widget.set_final_frame_targets(_final_frame_targets)
            widget.set_guide_video_targets(_guide_video_targets)
            widget.send_requested.connect(
                lambda key, w=widget: self._send_to_image_tab(
                    key, w.image_path, settings=self._get_active_image_settings(),
                )
            )
            widget.final_frame_requested.connect(
                lambda key, w=widget: self._send_final_frame(w.image_path, key)
            )
            widget.guide_video_requested.connect(
                lambda key, w=widget: self._send_guide_video(w.image_path, key)
            )

        # QwenAlyzer send-with-prompt
        self._img_tab._png_info.send_with_prompt.connect(self._dispatch_qwenalyzer_send)

        # QwenAlyzer gallery (legacy wiring, if gallery exists)
        if hasattr(self._img_tab._png_info, "_gallery"):
            self._img_tab._png_info._gallery.set_send_targets(_image_send_targets)
            self._img_tab._png_info._gallery.set_final_frame_targets(_final_frame_targets)
            self._img_tab._png_info._gallery.set_guide_video_targets(_guide_video_targets)
            self._img_tab._png_info._gallery.send_requested.connect(
                lambda key: self._dispatch_pnginfo_send(key)
            )
            self._img_tab._png_info._gallery.final_frame_requested.connect(
                lambda key: self._send_final_frame(self._img_tab._png_info._path.text(), key)
            )
            self._img_tab._png_info._gallery.guide_video_requested.connect(
                lambda key: self._send_guide_video(self._img_tab._png_info._path.text(), key)
            )

        # 3D Modeling -> send composite to image/video tabs
        self._img_tab._model3d.send_composite.connect(self._on_3d_send_composite)
        self._img_tab._model3d.lora_dataset_ready.connect(self._on_3d_lora_dataset)

        # Draw tab -> send flattened composite to other image tabs
        self._img_tab._img_edit.send_flattened_requested.connect(
            self._on_draw_send_flattened
        )

        # -- Timeline: preview player (removed — GL preview replaced VideoPlayerWidget)

        # -- Timeline: clip context menus (library + track) ------------------------
        self._timeline_tab.set_send_targets(
            clip_targets=_clip_targets, frame_targets=_frame_targets,
            final_frame_targets=_final_frame_targets,
            guide_video_targets=_guide_video_targets,
        )
        self._timeline_tab.send_clip_to.connect(
            lambda key, path: self._dispatch_video_send(key, path)
        )
        self._timeline_tab.send_frame_to.connect(
            lambda key, path: self._dispatch_timeline_frame(key, path)
        )
        self._timeline_tab.send_final_frame_to.connect(
            lambda key, path: self._send_final_frame(path, key)
        )
        self._timeline_tab.send_guide_video_to.connect(
            lambda key, path: self._send_guide_video(path, key)
        )
        self._timeline_tab.send_cc_ref_to.connect(
            lambda path: self._send_color_ref(path)
        )
        self._timeline_tab.send_cc_source_from_timeline.connect(
            self._send_to_cc_source_from_timeline
        )

        # -- Enable "Color Reference Frame" on ALL video players ---------------
        _all_video_players = [
            self._gen_tab._preview,
            self._ve_tab._source_player, self._ve_tab._preview,
            self._ls_tab._source_player, self._ls_tab._gapfill_player, self._ls_tab._output_player,
            self._cc_tab._result_player,
        ]
        for player in _all_video_players:
            player.enable_color_ref_action(True)
            player.color_ref_requested.connect(
                lambda path, p=player: self._send_color_ref(path)
            )

        # Color Correct tab → send-to / replace on timeline
        self._cc_tab.send_requested.connect(self._dispatch_cc_send)
        self._cc_tab.replace_on_timeline.connect(self._replace_clip_on_timeline)

    # ── Console toggle ────────────────────────────────────────────────────────

    def _toggle_console(self) -> None:
        """Switch to the Console tab (Ctrl+`)."""
        self._tab_widget.setCurrentWidget(self._console_tab)

    # ── Help dialog ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_help_icon(size: int = 24) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(100, 180, 255), 2.5))
        p.drawEllipse(3, 3, 18, 18)
        p.setPen(QPen(QColor(210, 210, 210), 2.5))
        # Question mark
        p.drawArc(8, 5, 8, 8, 30 * 16, 180 * 16)
        p.drawLine(12, 13, 12, 15)
        p.drawPoint(12, 18)
        p.end()
        return QIcon(pm)

    @staticmethod
    def _make_ai_icon(size: int = 24) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        from PySide6.QtGui import QPainterPath
        # Chat bubble
        path = QPainterPath()
        path.moveTo(4, 4)
        path.lineTo(20, 4)
        path.quadTo(22, 4, 22, 6)
        path.lineTo(22, 15)
        path.quadTo(22, 17, 20, 17)
        path.lineTo(10, 17)
        path.lineTo(6, 21)
        path.lineTo(6, 17)
        path.lineTo(4, 17)
        path.quadTo(2, 17, 2, 15)
        path.lineTo(2, 6)
        path.quadTo(2, 4, 4, 4)
        p.setPen(QPen(QColor(180, 100, 255), 1.5))
        p.setBrush(QColor(50, 30, 70))
        p.drawPath(path)
        # Three dots inside
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(200, 160, 255))
        for x in (8, 12, 16):
            p.drawEllipse(x - 1, 9, 3, 3)
        p.end()
        return QIcon(pm)

    def _show_help(self) -> None:
        from PySide6.QtWidgets import QDialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Supreme Diffusion — Help")
        dlg.setMinimumSize(750, 600)
        dlg.resize(900, 700)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        # Load help from HTML file
        help_path = Path(__file__).parent / "resources" / "help.html"
        try:
            browser.setHtml(help_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            browser.setHtml("<h2>Help file not found</h2><p>Expected at: "
                            f"<code>{help_path}</code></p>")
        layout.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    # ── Chat assistant ───────────────────────────────────────────────────────

    def _on_chat_clicked(self) -> None:
        from sdqt.widgets.chat_dialog import ChatDialog
        if self._chat_dialog is None:
            self._chat_dialog = ChatDialog(self._state, parent=self)
        project = self._project_bar.current_project or ""
        self._chat_dialog.set_project(project)
        self._chat_dialog.show()
        self._chat_dialog.raise_()
        self._chat_dialog.activateWindow()

    # ── Send-to color reference ──────────────────────────────────────────────

    def _send_color_ref(self, frame_path: str) -> None:
        """Route a captured color reference frame to the Color Correct tab."""
        self._cc_tab.set_reference(frame_path)
        self._tab_widget.setCurrentWidget(self._cc_tab)
        self._status_bar.showMessage("Color reference set in Color Correct tab.", 3000)

    def _send_to_cc_source(self, path: str) -> None:
        """Send a file to the Color Correct tab as a source to correct."""
        self._cc_tab.add_source(path)
        self._tab_widget.setCurrentWidget(self._cc_tab)
        self._status_bar.showMessage("Source added to Color Correct tab.", 3000)

    def _send_to_cc_ref(self, path: str) -> None:
        """Send a file to the Color Correct tab as the reference."""
        self._send_color_ref(path)

    def _send_to_cc_source_from_timeline(self, path: str) -> None:
        """Send a timeline clip to Color Correct as source (enables Replace)."""
        self._cc_tab.add_source(path, from_timeline=True)
        self._tab_widget.setCurrentWidget(self._cc_tab)
        self._status_bar.showMessage("Timeline clip sent to Color Correct.", 3000)

    def _dispatch_cc_send(self, target: str, path: str) -> None:
        """Route Color Correct send-to requests (clips and frames)."""
        if not path:
            return
        # Try clip handlers first
        clip_handlers = {
            "timeline": self._send_to_timeline,
            "ve": self._send_to_ve,
            "longshot": self._send_to_longshot,
            "stillgrabber": self._send_to_stillgrabber,
            "cc_source": self._send_to_cc_source,
            "cc_ref": self._send_to_cc_ref,
        }
        handler = clip_handlers.get(target)
        if handler:
            handler(path)
            return
        # Frame-based sends (img2img, inpaint, face_swap, etc.)
        if target in ("img2img", "inpaint", "face_swap", "img2vid"):
            self._send_to_image_tab(target, path)
            return

    def _replace_clip_on_timeline(self, original_path: str, result_path: str) -> None:
        """Replace a timeline clip's media path with the corrected version."""
        from pathlib import Path as _Path
        replaced = False
        for track in self._timeline_tab._multitrack.tracks():
            for clip in track.clips:
                if clip.path == original_path or _Path(clip.path).name == _Path(original_path).name:
                    persisted = self._timeline_tab._persist_clip(result_path)
                    clip.path = persisted
                    clip.name = _Path(persisted).stem
                    replaced = True
                    break
            if replaced:
                break
        if replaced:
            self._timeline_tab._multitrack.update()
            self._timeline_tab._on_tracks_changed()
            self._tab_widget.setCurrentWidget(self._timeline_tab)
            self._status_bar.showMessage("Timeline clip replaced with corrected version.", 3000)
        else:
            self._status_bar.showMessage("Could not find matching clip on timeline.", 3000)

    def _get_browser_selected(self) -> str | None:
        tab_idx = self._browser_tab._active_browser_index()
        if 0 <= tab_idx < len(self._browser_tab._browsers):
            return self._browser_tab._browsers[tab_idx].selected_path
        return None

    def _get_browser_video(self) -> str | None:
        return self._get_browser_selected()

    # ── Send-to video tab navigation ─────────────────────────────────────────

    def _send_to_longshot(self, path: str | None) -> None:
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._ls_tab)
            self._ls_tab.load_source(path)

    def _send_to_stillgrabber(self, path: str | None) -> None:
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._still_tab)
            self._still_tab.load_source(path)

    def _send_to_ve(self, path: str | None) -> None:
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._ve_tab)
            self._ve_tab.load_source(path)

    def _send_to_mimicmotion(self, path: str | None) -> None:
        """Route a video clip to Pose Animate as the driving (pose source) video."""
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._mimicmotion_tab)
            self._mimicmotion_tab.load_source(path)

    def _send_to_mimicmotion_ref(self, path: str | None) -> None:
        """Route an image to Pose Animate as the reference (appearance) image."""
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._mimicmotion_tab)
            self._mimicmotion_tab.load_reference_image(path)

    def _send_image_to_gen(self, path: str | None) -> None:
        if path:
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._image_source.load_image(path)
            self._gen_tab._mode1_rb.setChecked(True)


    # ── Audio send-to navigation ─────────────────────────────────────────────

    def _dispatch_audio_send(self, key: str, path: str) -> None:
        """Route audio send-to by key string."""
        if not path:
            return
        handlers = {
            "timeline": self._send_to_timeline,
            "chatterbox": self._send_to_chatterbox,
            "chatterbox_ref": self._send_to_chatterbox_ref,
            "dia_tts": self._send_to_tts,
            "dia_ref": self._send_to_dia_ref,
            "voice_library": self._save_to_voice_library,
            "sound_library": self._save_to_sound_library,
            "musicgen": self._send_to_musicgen,
            "audiogen": self._send_to_audiogen,
        }
        handler = handlers.get(key)
        if handler:
            handler(path)
            self._status_bar.showMessage(f"Audio sent to {key}.", 3000)

    def _save_to_voice_library(self, path: str) -> None:
        self._library_tab.save_to_voice_library(path)
        self._tab_widget.setCurrentWidget(self._library_tab)
        self._library_tab._tab_widget.setCurrentWidget(self._library_tab._voice_library)

    def _save_to_sound_library(self, path: str) -> None:
        self._library_tab.save_to_sound_library(path)
        self._tab_widget.setCurrentWidget(self._library_tab)
        self._library_tab._tab_widget.setCurrentWidget(self._library_tab._sound_library)

    def _send_to_chatterbox(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._chatterbox)
        self._audio_tab._chatterbox.load_audio(path)

    def _send_to_chatterbox_ref(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._chatterbox)
        self._audio_tab._chatterbox.load_ref_audio(path)

    def _send_to_tts(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._dia)
        self._audio_tab._dia.load_audio(path)

    def _send_to_dia_ref(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._dia)
        self._audio_tab._dia.load_ref_audio(path)

    def _send_to_musicgen(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._musicgen)
        self._audio_tab._musicgen.load_audio(path)

    def _send_to_sadtalker_src(self, path: str | None) -> None:
        """Route an image to SadTalker as the source image."""
        if not path:
            return
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab.load_sadtalker_source(path)

    def _send_to_sadtalker_audio(self, path: str | None) -> None:
        """Route an audio clip to SadTalker as the driving audio."""
        if not path:
            return
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab.load_sadtalker_audio(path)

    def _send_to_audiogen(self, path: str) -> None:
        self._tab_widget.setCurrentWidget(self._audio_tab)
        self._audio_tab._tab_widget.setCurrentWidget(self._audio_tab._audiogen)
        self._audio_tab._audiogen.load_audio(path)

    def _send_to_img2vid_clip(self, path: str | None) -> None:
        """Send a video clip to Img2Vid -- extract last frame as source image."""
        if path:
            frame = self._capture_last_frame_path(path)
            if frame:
                self._send_image_to_gen(frame)

    def _capture_last_frame_path(self, video_path: str) -> str | None:
        """Extract the last frame of a video as a temp PNG."""
        try:
            from supremediffusion.utils.video import extract_single_frame, probe_video
            from sdqt.utils.color_neutralize import neutralize_if_enabled
            info = probe_video(video_path)
            last = max(0, info.get("num_frames", 1) - 1)
            img = extract_single_frame(video_path, last)
            if img:
                img = neutralize_if_enabled(img, self.state.global_config)
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                tmp.close()
                return tmp.name
        except Exception:
            pass
        return None

    def _send_to_timeline(self, path: str | None) -> None:
        if path:
            self._tab_widget.setCurrentWidget(self._timeline_tab)
            self._timeline_tab.add_clip(path)

    # ── Gen settings extract/apply ───────────────────────────────────────────

    @staticmethod
    def _extract_gen_settings(tab) -> dict:
        """Extract prompt/sampler/scheduler settings from an image sub-tab."""
        settings: dict = {}
        if hasattr(tab, "_prompt") and hasattr(tab._prompt, "toPlainText"):
            settings["prompt"] = tab._prompt.toPlainText()
        if hasattr(tab, "_neg_prompt"):
            w = tab._neg_prompt
            if hasattr(w, "toPlainText"):
                settings["negative_prompt"] = w.toPlainText()
            elif hasattr(w, "text"):
                settings["negative_prompt"] = w.text()
        if hasattr(tab, "_params"):
            p = tab._params
            if hasattr(p, "sampler"):
                settings["sampler"] = p.sampler.currentText()
            if hasattr(p, "scheduler"):
                settings["scheduler"] = p.scheduler.currentText()
        if hasattr(tab, "_sampler") and hasattr(tab._sampler, "currentText"):
            settings["sampler"] = tab._sampler.currentText()
        if hasattr(tab, "_scheduler") and hasattr(tab._scheduler, "currentText"):
            settings["scheduler"] = tab._scheduler.currentText()
        return settings

    @staticmethod
    def _apply_gen_settings(tab, settings: dict) -> None:
        """Apply prompt/sampler/scheduler settings to an image sub-tab."""
        if not settings:
            return
        prompt = settings.get("prompt", "")
        neg = settings.get("negative_prompt", "")
        sampler = settings.get("sampler", "")
        scheduler = settings.get("scheduler", "")
        if prompt and hasattr(tab, "_prompt") and hasattr(tab._prompt, "setPlainText"):
            tab._prompt.setPlainText(prompt)
        if neg and hasattr(tab, "_neg_prompt"):
            w = tab._neg_prompt
            if hasattr(w, "setPlainText"):
                w.setPlainText(neg)
            elif hasattr(w, "setText"):
                w.setText(neg)
        if hasattr(tab, "_params"):
            p = tab._params
            if sampler and hasattr(p, "sampler"):
                idx = p.sampler.findText(sampler)
                if idx >= 0:
                    p.sampler.setCurrentIndex(idx)
            if scheduler and hasattr(p, "scheduler"):
                idx = p.scheduler.findText(scheduler)
                if idx >= 0:
                    p.scheduler.setCurrentIndex(idx)
        if sampler and hasattr(tab, "_sampler") and hasattr(tab._sampler, "findText"):
            idx = tab._sampler.findText(sampler)
            if idx >= 0:
                tab._sampler.setCurrentIndex(idx)
        if scheduler and hasattr(tab, "_scheduler") and hasattr(tab._scheduler, "findText"):
            idx = tab._scheduler.findText(scheduler)
            if idx >= 0:
                tab._scheduler.setCurrentIndex(idx)

    def _get_active_image_settings(self) -> dict:
        """Extract generation settings from the currently active image sub-tab."""
        current = self._img_tab._tab_widget.currentWidget()
        if current is None:
            return {}
        if hasattr(current, "_tab_widget"):
            inner = current._tab_widget.currentWidget()
            if inner is not None:
                return self._extract_gen_settings(inner)
        return self._extract_gen_settings(current)

    # ── Send-to image tab navigation ─────────────────────────────────────────

    def _send_to_image_tab(self, target: str, path: str | None, settings: dict | None = None) -> None:
        if not path:
            return
        if target == "img2vid":
            self._send_image_to_gen(path)
            return
        if target == "timeline":
            self._send_to_timeline(path)
            return
        if target == "cc_source":
            self._send_to_cc_source(path)
            return
        if target == "cc_ref":
            self._send_to_cc_ref(path)
            return
        if target == "mimicmotion_ref":
            self._send_to_mimicmotion_ref(path)
            return
        if target == "sadtalker_src":
            self._send_to_sadtalker_src(path)
            return

        gen = self._img_tab._gen
        tab_map = {
            "txt2img": gen,
            "img2img": gen,
            "inpaint": gen,
            "img2flux": gen,
            "flux_fill": gen,
            "img2zimg": gen,
            "zimg_fill": gen,
            "faceswap": self._img_tab._face_swap,
            "faceswap_src": self._img_tab._face_swap,
            "faceswap_tgt": self._img_tab._face_swap,
            "bodydouble": self._img_tab._body_double,
            "bodydouble_src": self._img_tab._body_double,
            "bodydouble_tgt": self._img_tab._body_double,
            "repose": self._img_tab._repose,
            "repose_src": self._img_tab._repose,
            "repose_pose": self._img_tab._repose,
            "imgedit": self._img_tab._img_edit,
            "model3d": self._img_tab._model3d,
            "qwenalyzer": self._img_tab._png_info,
            "cropzoom": self._img_tab._crop_zoom,
            "controlnet_cond": self._img_tab._controlnet,
            "controlnet_src": self._img_tab._controlnet,
        }
        tab = tab_map.get(target)
        if tab:
            # Unified gen tab routing
            if target == "txt2img":
                gen.switch_to("txt2img")
            elif target in ("img2img", "img2flux", "img2zimg"):
                gen.load_source(path)
            elif target in ("inpaint", "flux_fill", "zimg_fill"):
                gen.load_fill_source(path)
            elif target == "imgedit":
                tab.load_source(path)
            elif target == "faceswap":
                tab.load_target(path)
            elif target == "faceswap_src":
                tab.load_source(path)
            elif target == "faceswap_tgt":
                tab.load_target(path)
            elif target == "bodydouble_src":
                tab.load_source(path)
            elif target == "bodydouble_tgt":
                tab.load_target(path)
            elif target == "bodydouble":
                tab.load_target(path)
            elif target == "repose_src":
                tab.load_source(path)
            elif target == "repose_pose":
                tab.load_pose_ref(path)
            elif target == "repose":
                tab.load_source(path)
            elif target == "controlnet_cond":
                tab.load_condition(path)
            elif target == "controlnet_src":
                tab.load_source(path)
            elif target == "qwenalyzer":
                tab.load_image(path)
            elif hasattr(tab, "load_source"):
                tab.load_source(path)

            # Apply generation settings
            if settings:
                dest = tab
                if tab is gen:
                    # Route to the actual sub-tab
                    if target in ("img2img", "img2flux", "img2zimg"):
                        dest = gen._img2img
                    elif target in ("inpaint", "flux_fill", "zimg_fill"):
                        dest = gen._inpainter
                    elif target == "txt2img":
                        dest = gen._txt2img
                self._apply_gen_settings(dest, settings)

            self._tab_widget.setCurrentWidget(self._img_tab)
            # Navigate to the correct parent tab within Image Suite
            char_tools = self._img_tab._character_tools
            if tab in (char_tools._face_swap, char_tools._body_double,
                       char_tools._repose, char_tools._controlnet):
                self._img_tab._tab_widget.setCurrentWidget(char_tools)
                char_tools._tab_widget.setCurrentWidget(tab)
            else:
                self._img_tab._tab_widget.setCurrentWidget(tab)

    def _on_draw_send_flattened(self, payload: str) -> None:
        """Handle 'Send Flattened To' from the Draw tab. Payload: 'target|path'."""
        if "|" not in payload:
            return
        target, path = payload.split("|", 1)
        if target == "img2vid":
            self._send_image_to_gen(path)
        else:
            self._send_to_image_tab(target, path, settings=self._get_active_image_settings())

    def _on_3d_send_composite(self, payload: str) -> None:
        """Handle 'Send Composite' from the 3D Modeling tab. Payload: 'target|path'."""
        if "|" not in payload:
            return
        target, path = payload.split("|", 1)
        if target == "img2vid":
            self._send_image_to_gen(path)
        else:
            self._send_to_image_tab(target, path, settings=self._get_active_image_settings())
        self._status_bar.showMessage(f"3D composite sent to {target}.", 3000)

    def _on_3d_lora_dataset(self, dataset_path: str) -> None:
        """3D tab rendered multi-angle views → switch to LoRA tab and auto-caption."""
        lora = self._img_tab._lora
        self._tab_widget.setCurrentWidget(self._img_tab)
        self._img_tab._tab_widget.setCurrentWidget(lora)
        lora._dataset.set_dataset_path(dataset_path)
        self._status_bar.showMessage("3D dataset loaded in LoRA training. Starting Qwen VL captioning...", 5000)
        lora._on_batch_caption("qwen")

    def _on_3d_lora_dataset_with_model(self, dataset_path: str, model_type: str, checkpoint: str) -> None:
        """Wizard rendered multi-angle views → switch to LoRA tab with model preset."""
        lora = self._img_tab._lora
        self._tab_widget.setCurrentWidget(self._img_tab)
        self._img_tab._tab_widget.setCurrentWidget(lora)
        lora._dataset.set_dataset_path(dataset_path)
        if model_type:
            idx = lora._model_type.findText(model_type)
            if idx >= 0:
                lora._model_type.setCurrentIndex(idx)
        if checkpoint:
            idx = lora._checkpoint_combo.findText(checkpoint)
            if idx >= 0:
                lora._checkpoint_combo.setCurrentIndex(idx)
        self._status_bar.showMessage("3D dataset loaded in LoRA training. Starting Qwen VL captioning...", 5000)
        lora._on_batch_caption("qwen")

    def _on_video_lora_dataset(self, dataset_path: str, trigger_token: str, config: dict) -> None:
        """Character Dataset Builder wizard → switch to Video LoRA tab."""
        self._tab_widget.setCurrentWidget(self._generate_suite)
        self._generate_tabs.setCurrentWidget(self._video_lora_tab)
        self._video_lora_tab.set_video_dataset(dataset_path, trigger_token, config)

    def _capture_and_send_frame(self, player, target: str) -> None:
        """Capture a frame from a video player and send to an image tab."""
        frame_path = player.capture_frame()
        if frame_path:
            self._send_to_image_tab(target, frame_path, settings=self._get_active_image_settings())
            self._status_bar.showMessage(f"Frame sent to {target}.", 3000)
        else:
            self._status_bar.showMessage("Frame capture failed — no video loaded or ffmpeg error.", 5000)
            logger.warning("Frame capture failed — no video loaded or ffmpeg error.")

    def _get_image_subtab_path(self, source_tab) -> str | None:
        if hasattr(source_tab, "_gallery"):
            return source_tab._gallery.selected_path
        if hasattr(source_tab, "_result_path"):
            return source_tab._result_path
        return None

    def _dispatch_video_send(self, key: str, path: str | None) -> None:
        """Route a send-to-video dropdown selection to the correct handler."""
        if not path:
            return
        handlers = {
            "timeline": self._send_to_timeline,
            "img2vid_clip": self._send_to_img2vid_clip,
            "ve": self._send_to_ve,
            "longshot": self._send_to_longshot,
            "stillgrabber": self._send_to_stillgrabber,
            "mimicmotion": self._send_to_mimicmotion,
            "cc_source": self._send_to_cc_source,
            "cc_ref": self._send_to_cc_ref,
            "chatterbox": self._send_to_chatterbox,
            "chatterbox_ref": self._send_to_chatterbox_ref,
            "dia_tts": self._send_to_tts,
            "dia_ref": self._send_to_dia_ref,
            "sadtalker_audio": self._send_to_sadtalker_audio,
        }
        handler = handlers.get(key)
        if handler:
            handler(path)

    def _dispatch_image_subtab_send(self, source_tab, key: str) -> None:
        path = self._get_image_subtab_path(source_tab)
        if not path:
            return
        settings = self._extract_gen_settings(source_tab)
        if key == "img2vid":
            self._send_image_to_gen(path)
        elif key == "timeline":
            self._send_to_timeline(path)
        elif key == "cc_source":
            self._send_to_cc_source(path)
        elif key == "cc_ref":
            self._send_to_cc_ref(path)
        else:
            self._send_to_image_tab(key, path, settings=settings)

    def _dispatch_gallery_final_frame(self, source_tab, key: str) -> None:
        path = self._get_image_subtab_path(source_tab)
        if path:
            self._send_final_frame(path, key)

    def _capture_and_send_final_frame(self, player, key: str) -> None:
        frame_path = player.capture_frame()
        if frame_path:
            self._send_final_frame(frame_path, key)
        else:
            self._status_bar.showMessage("Frame capture failed.", 5000)

    def _dispatch_timeline_frame(self, key: str, path: str) -> None:
        if not path:
            return
        if key == "img2vid_first":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._first_frame.load_image(path)
            self._status_bar.showMessage("First frame loaded into Img2Vid Mode 2.", 3000)
        elif key == "img2vid_last":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._last_frame.load_image(path)
            self._status_bar.showMessage("Last frame loaded into Img2Vid Mode 2.", 3000)
        else:
            self._send_to_image_tab(key, path, settings=self._get_active_image_settings())

    def _send_final_frame(self, path: str, key: str) -> None:
        if not path:
            return
        if key == "img2vid":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._use_final_frame_m2.setChecked(True)
            self._gen_tab._final_frame_m2.load_image(path)
            self._status_bar.showMessage("Final frame loaded into Img2Vid Mode 2.", 3000)
        elif key == "img2vid_ff":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._first_frame.load_image(path)
            self._status_bar.showMessage("First frame loaded into Img2Vid Mode 2.", 3000)
        elif key == "img2vid_lf":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._last_frame.load_image(path)
            self._status_bar.showMessage("Last frame loaded into Img2Vid Mode 2.", 3000)
        elif key == "ve":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._ve_tab)
            self._ve_tab._use_final_frame.setChecked(True)
            self._ve_tab._final_frame.load_image(path)
            self._status_bar.showMessage("Final frame loaded into Video Extender.", 3000)

    def _dispatch_gallery_guide_video(self, source_tab, key: str) -> None:
        path = self._get_image_subtab_path(source_tab)
        if path:
            self._send_guide_video(path, key)

    def _capture_and_send_guide_video(self, player, key: str) -> None:
        frame_path = player.capture_frame()
        if frame_path:
            self._send_guide_video(frame_path, key)
        else:
            self._status_bar.showMessage("Frame capture failed.", 5000)

    def _image_to_still_video(self, image_path: str, duration: float = 5.0) -> str | None:
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out.close()
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    *_codec_args(),
                    "-t", str(duration),
                    *_pix_fmt(_enc_name()),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=full",
                    "-r", "24",
                    out.name,
                ],
                capture_output=True,
                timeout=30,
            )
            if Path(out.name).stat().st_size > 0:
                return out.name
        except Exception:
            logger.warning("Failed to generate still video", exc_info=True)
        return None

    def _send_guide_video(self, image_path: str, key: str) -> None:
        if not image_path:
            return
        self._status_bar.showMessage("Generating still guidance clip...")
        video_path = self._image_to_still_video(image_path)
        if not video_path:
            self._status_bar.showMessage("Failed to generate guidance clip.", 5000)
            return

        if key == "img2vid_m1":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode1_rb.setChecked(True)
            self._gen_tab._use_guidance_m1.setChecked(True)
            self._gen_tab._guidance_video_m1.load_file(video_path)
            self._status_bar.showMessage("Guide video loaded into Img2Vid (single).", 3000)
        elif key == "img2vid_m2":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._gen_tab)
            self._gen_tab._mode2_rb.setChecked(True)
            self._gen_tab._use_guidance_m2.setChecked(True)
            self._gen_tab._guidance_video_m2.load_file(video_path)
            self._status_bar.showMessage("Guide video loaded into Img2Vid F/L.", 3000)
        elif key == "ve":
            self._tab_widget.setCurrentWidget(self._generate_suite)
            self._generate_tabs.setCurrentWidget(self._ve_tab)
            self._ve_tab._use_guidance.setChecked(True)
            self._ve_tab._guidance_video.load_file(video_path)
            self._status_bar.showMessage("Guide video loaded into Video Extender.", 3000)

    def _dispatch_pnginfo_send(self, key: str) -> None:
        path = self._img_tab._png_info._path.text()
        if not path:
            return
        if key == "img2vid":
            self._send_image_to_gen(path)
        else:
            self._send_to_image_tab(key, path, settings=self._get_active_image_settings())

    def _dispatch_qwenalyzer_send(self, target: str, path: str, prompt: str, negative: str) -> None:
        """Handle QwenAlyzer send — routes image + prompt to the target tab."""
        if not path:
            return
        if target == "img2vid":
            self._send_image_to_gen(path)
            return

        # First send the image
        self._send_to_image_tab(target, path, settings=self._get_active_image_settings())

        # Then inject the prompts into the actual destination sub-tab
        dest = None
        gen = self._img_tab._gen
        if target == "txt2img":
            dest = gen._txt2img
        elif target in ("img2img", "img2flux", "img2zimg"):
            dest = gen._img2img
        elif target in ("inpaint", "flux_fill", "zimg_fill"):
            dest = gen._inpainter
        elif target == "imgedit":
            dest = self._img_tab._img_edit
        elif target in ("bodydouble_src", "bodydouble_tgt"):
            dest = self._img_tab._body_double
        elif target in ("repose_src", "repose_pose"):
            dest = self._img_tab._repose
        elif target in ("controlnet_cond", "controlnet_src"):
            dest = self._img_tab._controlnet

        if dest and prompt:
            if hasattr(dest, "_prompt"):
                if hasattr(dest._prompt, "setPlainText"):
                    dest._prompt.setPlainText(prompt)
                elif hasattr(dest._prompt, "setText"):
                    dest._prompt.setText(prompt)
            if negative and hasattr(dest, "_neg_prompt"):
                if hasattr(dest._neg_prompt, "setPlainText"):
                    dest._neg_prompt.setPlainText(negative)
                elif hasattr(dest._neg_prompt, "setText"):
                    dest._neg_prompt.setText(negative)

        self._status_bar.showMessage(f"QwenAlyzer sent to {target}.", 3000)

    # ── Keyboard shortcuts ───────────────────────────────────────────────────

    def _setup_shortcuts(self) -> None:
        gen_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        gen_shortcut.activated.connect(self._on_generate_shortcut)

        abort_shortcut = QShortcut(QKeySequence("Escape"), self)
        abort_shortcut.activated.connect(self._on_abort_shortcut)

        console_shortcut = QShortcut(QKeySequence("Ctrl+`"), self)
        console_shortcut.activated.connect(self._toggle_console)

    @Slot()
    def _on_generate_shortcut(self) -> None:
        if self._tab_widget.currentWidget() is not self._generate_suite:
            return
        current = self._generate_tabs.currentWidget()
        if current is self._gen_tab:
            self._gen_tab._on_generate()
        elif current is self._ve_tab:
            self._ve_tab._on_generate()

    @Slot()
    def _on_abort_shortcut(self) -> None:
        if self._tab_widget.currentWidget() is not self._generate_suite:
            return
        current = self._generate_tabs.currentWidget()
        if current is self._gen_tab:
            self._gen_tab._on_abort()
        elif current is self._ve_tab:
            self._ve_tab._on_abort()

    # ── Status bar ───────────────────────────────────────────────────────────

    def _setup_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("QStatusBar { min-height: 26px; }")
        self._status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self._status_bar)

        self._pipeline_label = QLabel("")
        self._pipeline_label.setStyleSheet("font-size: 13px; color: #bbb;")
        self._status_bar.addPermanentWidget(self._pipeline_label)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status_bar)
        self._status_timer.start(5000)

    def _on_tab_status(self, msg: str) -> None:
        """Receive status messages from any tab and show in the main status bar."""
        self._status_bar.showMessage(msg, 10000)

    def _update_status_bar(self) -> None:
        parts = []
        if self._state.pipelines_loaded:
            parts.append("Video: Ready")

        if self._state.img_pipeline is not None:
            parts.append("Image: Ready")

        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                parts.append(f"GPU: {mem:.1f}/{total:.1f} GB")
        except Exception:
            pass

        self._pipeline_label.setText("  |  ".join(parts) if parts else "")

    # ── Auto-load pipelines ──────────────────────────────────────────────────

    def _auto_load_pipelines(self) -> None:
        self._status_bar.showMessage("Ready — pipelines load on first generation.")

    def _run_next_auto_load(self) -> None:
        if not self._auto_load_queue:
            return
        name, load_fn, success_msg, after = self._auto_load_queue.pop(0)

        from sdqt.workers.base import BaseWorker

        class _Loader(BaseWorker):
            def __init__(self, fn, msg, parent=None):
                super().__init__(parent)
                self._fn = fn
                self._msg = msg
            def do_work(self):
                self._fn()
                return self._msg

        self._status_bar.showMessage(f"Auto-loading {name} pipelines...")
        worker = _Loader(load_fn, success_msg, parent=self)
        worker.finished_ok.connect(lambda msg: self._on_auto_loaded(msg, after))
        worker.error.connect(lambda msg: self._on_auto_load_error(name, msg))
        worker.finished.connect(worker.deleteLater)
        self._auto_load_worker = worker
        worker.start()

    def _on_auto_loaded(self, msg: str, after=None) -> None:
        self._status_bar.showMessage(msg)
        if after:
            try:
                after()
            except Exception:
                pass
        self._run_next_auto_load()

    def _on_auto_load_error(self, name: str, msg: str) -> None:
        logger.warning("Auto-load %s failed: %s", name, msg)
        self._status_bar.showMessage(f"Auto-load {name} failed: {msg}")
        self._run_next_auto_load()

    # ── Project change broadcast ─────────────────────────────────────────────

    @Slot(str)
    def _on_project_changed(self, project_name: str) -> None:
        for tab in self._tabs:
            tab.on_project_changed(project_name)
        if self._chat_dialog and self._chat_dialog.isVisible():
            self._chat_dialog.set_project(project_name)

    def _restore_last_session(self) -> None:
        try:
            cfg = self._state.global_config
            if cfg.last_project:
                projects = self._state.project_manager.list_projects()
                if cfg.last_project in projects:
                    self._project_bar._combo.setCurrentText(cfg.last_project)
            if 0 <= cfg.last_parent_tab_idx < self._tab_widget.count():
                self._tab_widget.setCurrentIndex(cfg.last_parent_tab_idx)
            if 0 <= cfg.last_video_tab_idx < self._generate_tabs.count():
                self._generate_tabs.setCurrentIndex(cfg.last_video_tab_idx)
        except Exception:
            pass

    def _save_session_state(self) -> None:
        try:
            cfg = self._state.global_config
            cfg.last_project = self._project_bar.current_project or ""
            cfg.last_parent_tab_idx = self._tab_widget.currentIndex()
            cfg.last_video_tab_idx = self._generate_tabs.currentIndex()
            cfg.save()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self._status_timer.stop()
        self._save_session_state()
        if self._chat_dialog:
            self._chat_dialog.close()
        # Flush pending debounced field-persists (prompt edits made in the
        # last ~0.6s would otherwise die with their timers)
        for sub in (getattr(self._img_tab._gen, "_img2img", None),
                    getattr(self._img_tab._gen, "_inpainter", None)):
            try:
                timer = getattr(sub, "_persist_timer", None)
                if timer is not None and timer.isActive():
                    timer.stop()
                    sub._persist_params()
            except Exception:
                pass
        # Persist Draw workspace before exit
        try:
            self._img_tab._img_edit.save_workspace()
        except Exception:
            pass
        try:
            self._img_tab._model3d._save_scene()
        except Exception:
            pass
        try:
            self._state.unload_pipelines()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        super().closeEvent(event)
