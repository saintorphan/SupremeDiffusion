"""Chatterbox TTS tab — voice cloning with emotion control."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.widgets.audio_player import AudioPlayerWidget
from sdqt.widgets.draggable_list import DraggableFileList
from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.send_targets import AUDIO_TARGETS as _AUDIO_SEND_TARGETS
from sdqt.widgets.send_targets import build_target_menu
from sdqt.workers.chatterbox_tts import ChatterboxTTSWorker
from .base import BaseTab

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

_SECTION_STYLE = (
    "font-weight: bold; color: #4ca6ff;"
    " padding: 2px 0;"
)

_TAG_BTN_STYLE = (
    "QPushButton { padding: 2px 8px; font-size: 13px;"
    " background: #2a2a2a; border: 1px solid #444; border-radius: 3px; }"
    "QPushButton:hover { background: #3a3a3a; border-color: #666; }"
)


class ChatterboxTTSTab(BaseTab):
    """Chatterbox TTS: voice cloning with emotion/exaggeration control."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: ChatterboxTTSWorker | None = None
        self._output_files: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left column: text (top) + settings (bottom) ──────────────────────
        left_splitter = QSplitter(Qt.Vertical)

        # -- Top: text input ------------------------------------------------
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(4, 4, 4, 2)
        top_layout.setSpacing(4)

        header = QLabel("Chatterbox TTS")
        header.setStyleSheet(_SECTION_STYLE)
        top_layout.addWidget(header)

        self._text = QPlainTextEdit()
        self._text.setPlaceholderText(
            "Enter text to synthesize. Use a reference audio clip "
            "for voice cloning.\n\n"
            "Adjust Emotion (exaggeration) and Speaker Adherence "
            "(cfg weight) to control output style."
        )
        top_layout.addWidget(self._text, 1)

        left_splitter.addWidget(top_widget)

        # -- Bottom: reference, params, status ------------------------------
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(4, 2, 4, 4)
        bottom_layout.setSpacing(4)

        # Quick reference
        ref_label = QLabel(
            "<b>Voice clone:</b> Drop a ~5s reference audio clip below. "
            "<b>Emotion:</b> 0.0 = neutral, 1.0 = highly expressive. "
            "<b>Speaker Adherence:</b> Higher = closer to reference voice."
        )
        ref_label.setWordWrap(True)
        ref_label.setStyleSheet("color: #aaa; font-size: 13px; padding: 2px 0;")
        bottom_layout.addWidget(ref_label)

        # Reference audio
        self._ref_drop = FileDropWidget(
            "Reference Voice (optional — ~5s clip)", extensions=_AUDIO_EXTS
        )
        self._ref_drop.setMaximumHeight(100)
        self._ref_drop.setMinimumHeight(90)
        bottom_layout.addWidget(self._ref_drop)

        # Params row — the two controls share the first line, the action
        # buttons drop to their own row so neither group gets crammed.
        params_row = QHBoxLayout()
        params_row.setSpacing(8)

        params_row.addWidget(QLabel("Emotion:"))
        self._exaggeration = QDoubleSpinBox()
        self._exaggeration.setRange(0.0, 1.0)
        self._exaggeration.setDecimals(1)
        self._exaggeration.setSingleStep(0.1)
        self._exaggeration.setValue(0.5)
        self._exaggeration.setMinimumWidth(95)
        params_row.addWidget(self._exaggeration)

        params_row.addWidget(QLabel("Speaker Adherence:"))
        self._cfg_weight = QDoubleSpinBox()
        self._cfg_weight.setRange(0.0, 1.0)
        self._cfg_weight.setDecimals(1)
        self._cfg_weight.setSingleStep(0.1)
        self._cfg_weight.setValue(0.5)
        self._cfg_weight.setMinimumWidth(95)
        params_row.addWidget(self._cfg_weight)
        params_row.addStretch()

        bottom_layout.addLayout(params_row)

        # Action row (second line)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addStretch()

        self._go_btn = QPushButton("Generate")
        self._go_btn.setObjectName("primary")
        self._go_btn.clicked.connect(self._on_generate)
        action_row.addWidget(self._go_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet("font-weight: bold; color: #e55;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        action_row.addWidget(self._abort_btn)

        bottom_layout.addLayout(action_row)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa;")
        bottom_layout.addWidget(self._status)
        bottom_layout.addStretch()

        left_splitter.addWidget(bottom_widget)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)

        # Wrap the left splitter in a scroll area so the params + action rows
        # stay reachable when the window is short.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(left_splitter)
        splitter.addWidget(left_scroll)

        # ── Right column: output list + preview ──────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        out_header = QHBoxLayout()
        out_header.addWidget(QLabel("<b>Chatterbox Outputs:</b>"))
        out_header.addStretch()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_outputs)
        out_header.addWidget(self._clear_btn)
        right_layout.addLayout(out_header)

        self._output_list = DraggableFileList()
        self._output_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._output_list.customContextMenuRequested.connect(self._show_output_menu)
        self._output_list.currentRowChanged.connect(self._on_output_selected)
        right_layout.addWidget(self._output_list, 1)

        self._audio_preview = AudioPlayerWidget("Preview")
        self._audio_preview.set_send_targets(_AUDIO_SEND_TARGETS)
        right_layout.addWidget(self._audio_preview)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, 1)

    # -- Public API --------------------------------------------------------

    def load_audio(self, path: str) -> None:
        """Load an audio file into the preview player."""
        if path and Path(path).is_file():
            self._audio_preview.load_audio(path)

    def load_ref_audio(self, path: str) -> None:
        """Load a reference voice audio for voice cloning."""
        if path and Path(path).is_file():
            self._ref_drop.load_file(path)

    # -- Generation --------------------------------------------------------

    @Slot()
    def _on_generate(self) -> None:
        text = self._text.toPlainText().strip()
        if not text:
            self._show_status("Enter text first.")
            return

        self._save_to_config()

        out_dir = self.project_path / "audio" / "chatterbox"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output = str(out_dir / f"chatterbox_{ts}.wav")

        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Starting Chatterbox TTS...")

        worker = ChatterboxTTSWorker(
            text=text,
            output_path=output,
            ref_audio=self._ref_drop.file_path,
            exaggeration=self._exaggeration.value(),
            cfg_weight=self._cfg_weight.value(),
            app_state=self.state,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_gen_done)
        worker.error.connect(self._on_gen_error)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: self._restore_buttons())
        self._worker = worker
        worker.start()

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    def _restore_buttons(self) -> None:
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)

    # -- Output handling ---------------------------------------------------

    def _on_gen_done(self, result: str) -> None:
        self.state._force_gc()
        logger.info("Chatterbox TTS complete: %s", result)
        self._show_status(f"Done: {Path(result).name}")
        self._rescan_outputs()
        for i in range(self._output_list.count()):
            if self._output_list.item(i).toolTip() == result:
                self._output_list.setCurrentRow(i)
                break

    def _on_gen_error(self, msg: str) -> None:
        self.state._force_gc()
        logger.error("Chatterbox TTS error: %s", msg)
        self._show_status(f"Error: {msg}")

    # -- Output list -------------------------------------------------------

    def _rescan_outputs(self) -> None:
        self._output_list.clear()
        self._output_files = []
        out_dir = self.project_path / "audio" / "chatterbox"
        if not out_dir.is_dir():
            return
        for f in sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime,
                        reverse=True):
            if f.suffix.lower() in (".wav", ".mp3", ".flac"):
                self._output_files.append(str(f))
                self._output_list.addItem(f.name)
                self._output_list.item(
                    self._output_list.count() - 1
                ).setToolTip(str(f))

    @Slot(int)
    def _on_output_selected(self, row: int) -> None:
        if 0 <= row < len(self._output_files):
            self._audio_preview.load_audio(self._output_files[row])

    def _show_output_menu(self, pos) -> None:
        item = self._output_list.itemAt(pos)
        if not item:
            return
        self._output_list.setCurrentItem(item)
        row = self._output_list.row(item)
        if row < 0 or row >= len(self._output_files):
            return
        path = self._output_files[row]

        menu = QMenu(self)
        build_target_menu(
            menu, "Send to", _AUDIO_SEND_TARGETS,
            lambda k, p=path: self._audio_preview.send_audio_requested.emit(k, p),
        )

        menu.addSeparator()
        voice_lib_action = menu.addAction("Save to Voice Library")
        voice_lib_action.triggered.connect(
            lambda: self._save_to_voice_library(path)
        )

        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_output(path))

        menu.exec(self._output_list.mapToGlobal(pos))

    def _save_to_voice_library(self, path: str) -> None:
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "save_to_voice_library"):
                if parent.save_to_voice_library(path):
                    self._show_status(f"Saved {Path(path).name} to Voice Library")
                return
            parent = parent.parent()
        self._show_status("Voice Library not available")

    def _delete_output(self, path: str) -> None:
        try:
            Path(path).unlink()
            self._rescan_outputs()
            self._show_status(f"Deleted {Path(path).name}")
        except Exception as exc:
            logger.error("Failed to delete %s: %s", path, exc)
            self._show_status(f"Error: {exc}")

    @Slot()
    def _on_clear_outputs(self) -> None:
        import shutil
        out_dir = self.project_path / "audio" / "chatterbox"
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        self._output_list.clear()
        self._output_files = []
        self._audio_preview.clear_audio()
        self._show_status("Outputs cleared.")

    # -- Persistence -------------------------------------------------------

    def _save_to_config(self) -> None:
        from supremediffusion.config.project_config import ProjectConfig
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        cfg.chatterbox_text = self._text.toPlainText()
        cfg.chatterbox_ref_audio = self._ref_drop.file_path or ""
        cfg.chatterbox_exaggeration = self._exaggeration.value()
        cfg.chatterbox_cfg_weight = self._cfg_weight.value()
        cfg.save(self.project_path)

    def _restore_from_config(self) -> None:
        from supremediffusion.config.project_config import ProjectConfig
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        if cfg.chatterbox_text:
            self._text.setPlainText(cfg.chatterbox_text)
        if cfg.chatterbox_ref_audio and Path(cfg.chatterbox_ref_audio).is_file():
            self._ref_drop.load_file(cfg.chatterbox_ref_audio)
        self._exaggeration.setValue(cfg.chatterbox_exaggeration)
        self._cfg_weight.setValue(cfg.chatterbox_cfg_weight)

    # -- Project change ----------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        self._save_to_config()
        super().on_project_changed(project_name)
        self._audio_preview.clear_audio()
        self._show_status("")
        self._rescan_outputs()
        self._restore_from_config()
