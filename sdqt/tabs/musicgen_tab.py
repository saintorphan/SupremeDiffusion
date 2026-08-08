"""MusicGen tab — text-to-music generation using Meta AudioCraft."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.widgets.audio_player import AudioPlayerWidget
from sdqt.widgets.draggable_list import DraggableFileList
from sdqt.widgets.send_targets import AUDIO_TARGETS as _AUDIO_SEND_TARGETS
from sdqt.widgets.send_targets import build_target_menu
from sdqt.workers.musicgen import MusicGenWorker

from .base import BaseTab

logger = logging.getLogger(__name__)


class MusicGenTab(BaseTab):
    """Text-to-music generation using MusicGen."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: MusicGenWorker | None = None
        self._output_files: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        splitter = QSplitter(Qt.Horizontal)

        # -- Left: controls --------------------------------------------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 6, 8, 6)
        left_layout.setSpacing(8)

        # Prompt
        left_layout.addWidget(QLabel("<b>Prompt:</b>"))
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText(
            "Describe the music you want to generate..."
        )
        self._prompt.setMinimumHeight(80)
        self._prompt.setMaximumHeight(200)
        left_layout.addWidget(self._prompt, 1)

        # Params row 1: Duration, Model
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        row1.addWidget(QLabel("Duration:"))
        self._duration = QDoubleSpinBox()
        self._duration.setRange(1.0, 47.0)
        self._duration.setDecimals(1)
        self._duration.setSingleStep(1.0)
        self._duration.setValue(10.0)
        self._duration.setSuffix("s")
        self._duration.setMinimumWidth(95)
        row1.addWidget(self._duration)

        row1.addWidget(QLabel("Model:"))
        self._model_size = QComboBox()
        self._model_size.addItems(["small", "medium", "large"])
        self._model_size.setCurrentText("medium")
        self._model_size.setMinimumWidth(120)
        row1.addWidget(self._model_size)

        row1.addStretch()
        left_layout.addLayout(row1)

        # Params row 2: Temperature, Top-K, CFG
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        row2.addWidget(QLabel("Temp:"))
        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.1, 2.0)
        self._temperature.setDecimals(2)
        self._temperature.setSingleStep(0.05)
        self._temperature.setValue(1.0)
        self._temperature.setMinimumWidth(95)
        row2.addWidget(self._temperature)

        row2.addWidget(QLabel("Top-K:"))
        self._top_k = QSpinBox()
        self._top_k.setRange(0, 500)
        self._top_k.setValue(250)
        self._top_k.setMinimumWidth(90)
        row2.addWidget(self._top_k)

        row2.addWidget(QLabel("CFG:"))
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 10.0)
        self._cfg.setDecimals(1)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(3.0)
        self._cfg.setMinimumWidth(95)
        row2.addWidget(self._cfg)

        row2.addStretch()
        left_layout.addLayout(row2)

        # Action row: Generate / Abort on their own line
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addStretch()

        self._go_btn = QPushButton("Generate")
        self._go_btn.setObjectName("primary")
        self._go_btn.clicked.connect(self._on_generate)
        row3.addWidget(self._go_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet("font-weight: bold; color: #e55;")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        row3.addWidget(self._abort_btn)

        left_layout.addLayout(row3)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa;")
        left_layout.addWidget(self._status)

        # Scroll-wrap the left controls so they stay reachable on short windows.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # -- Right: output list + preview ------------------------------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        out_header = QHBoxLayout()
        out_header.addWidget(QLabel("<b>Outputs:</b>"))
        out_header.addStretch()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_outputs)
        out_header.addWidget(self._clear_btn)
        right_layout.addLayout(out_header)

        self._output_list = DraggableFileList()
        self._output_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._output_list.customContextMenuRequested.connect(
            self._show_output_menu
        )
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

    # -- Generation --------------------------------------------------------

    @Slot()
    def _on_generate(self) -> None:
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._show_status("Enter a prompt first.")
            return

        from sdqt.deps import check_and_install
        if not check_and_install("audiocraft", self):
            return

        self._save_to_config()

        out_dir = self.project_path / "audio" / "musicgen"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output = str(out_dir / f"musicgen_{ts}.wav")

        self._go_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._show_status("Freeing VRAM...")
        self.state.unload_pipelines()
        self._show_status("Loading MusicGen...")

        worker = MusicGenWorker(
            prompt=prompt,
            output_path=output,
            model_size=self._model_size.currentText(),
            duration=self._duration.value(),
            temperature=self._temperature.value(),
            top_k=self._top_k.value(),
            cfg_coef=self._cfg.value(),
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, result: str) -> None:
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self.state._force_gc()
        logger.info("MusicGen complete: %s", result)
        self._show_status(f"Done: {Path(result).name}")
        self._rescan_outputs()
        for i in range(self._output_list.count()):
            if self._output_list.item(i).toolTip() == result:
                self._output_list.setCurrentRow(i)
                break

    def _on_error(self, msg: str) -> None:
        self._go_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self.state._force_gc()
        logger.error("MusicGen error: %s", msg)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    # -- Output list -------------------------------------------------------

    def _rescan_outputs(self) -> None:
        self._output_list.clear()
        self._output_files = []
        out_dir = self.project_path / "audio" / "musicgen"
        if not out_dir.is_dir():
            return
        for f in sorted(
            out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
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
        snd_lib = menu.addAction("Save to Sound Library")
        snd_lib.triggered.connect(lambda: self._save_to_sound_library(path))
        menu.addSeparator()
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_output(path))
        menu.exec(self._output_list.mapToGlobal(pos))

    def _save_to_sound_library(self, path: str) -> None:
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "save_to_sound_library"):
                if parent.save_to_sound_library(path):
                    self._show_status(f"Saved {Path(path).name} to Sound Library")
                return
            parent = parent.parent()
        # Fallback: emit via audio_preview send signal
        self._audio_preview.send_audio_requested.emit("sound_library", path)

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
        out_dir = self.project_path / "audio" / "musicgen"
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
        cfg.musicgen_prompt = self._prompt.toPlainText()
        cfg.musicgen_duration = self._duration.value()
        cfg.musicgen_model_size = self._model_size.currentText()
        cfg.musicgen_temperature = self._temperature.value()
        cfg.musicgen_top_k = self._top_k.value()
        cfg.musicgen_cfg = self._cfg.value()
        cfg.save(self.project_path)

    def _restore_from_config(self) -> None:
        from supremediffusion.config.project_config import ProjectConfig
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        prompt = getattr(cfg, "musicgen_prompt", "") or ""
        if prompt:
            self._prompt.setPlainText(prompt)
        else:
            self._prompt.clear()
        self._duration.setValue(getattr(cfg, "musicgen_duration", 10.0))
        model = getattr(cfg, "musicgen_model_size", "medium") or "medium"
        idx = self._model_size.findText(model)
        if idx >= 0:
            self._model_size.setCurrentIndex(idx)
        self._temperature.setValue(getattr(cfg, "musicgen_temperature", 1.0))
        self._top_k.setValue(getattr(cfg, "musicgen_top_k", 250))
        self._cfg.setValue(getattr(cfg, "musicgen_cfg", 3.0))

    # -- Project change ----------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        self._save_to_config()
        super().on_project_changed(project_name)
        self._audio_preview.clear_audio()
        self._show_status("")
        self._rescan_outputs()
        self._restore_from_config()
