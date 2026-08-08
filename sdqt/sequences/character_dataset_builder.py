"""Character Dataset Builder — wizard to create a video LoRA training dataset.

Flow:
  1. Pick reference still(s) of the actor
  2. Set character name, trigger token, batch generation config
  3. Batch-generate short I2V clips from the stills
  4. Review grid — keep/reject each clip
  5. Auto-caption, configure dual noise training, send to Video LoRA tab
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "person talking naturally",
    "person turning head to the left",
    "person turning head to the right",
    "person smiling",
    "person looking down then up",
    "person gesturing with hands",
    "person nodding",
    "person tilting head",
]


@dataclass
class _State:
    """Shared state across wizard pages."""
    reference_stills: list[str] = field(default_factory=list)
    character_name: str = ""
    trigger_token: str = ""
    num_clips: int = 8
    clip_frames: int = 49
    fps: int = 16
    action_prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    generated_clips: list[str] = field(default_factory=list)
    kept: list[bool] = field(default_factory=list)
    dataset_dir: str = ""


# ── Page 1: Reference Stills ─────────────────────────────────────────────────

class _ReferenceStillsPage(WizardPage):
    """Select one or more reference images of the actor."""

    def __init__(self, ws: _State, state, parent=None):
        super().__init__(parent)
        self._ws = ws
        self._state = state

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Step 1: Select Reference Still(s)</b><br>"
            "Pick the best frame(s) of your actor — clear face, good lighting, "
            "varied angles if possible."
        ))

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self._list, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Images...")
        add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _on_add(self) -> None:
        from sdqt.utils.file_dialog import get_open_filenames
        paths, _ = get_open_filenames(
            self, "Select Reference Images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        for p in (paths or []):
            if p not in self._ws.reference_stills:
                self._ws.reference_stills.append(p)
                self._list.addItem(Path(p).name)

    def _on_remove(self) -> None:
        for item in reversed(self._list.selectedItems()):
            idx = self._list.row(item)
            self._list.takeItem(idx)
            if 0 <= idx < len(self._ws.reference_stills):
                self._ws.reference_stills.pop(idx)

    def validate(self) -> bool:
        return len(self._ws.reference_stills) > 0


# ── Page 2: Character Config ─────────────────────────────────────────────────

class _CharacterConfigPage(WizardPage):
    """Set character name, prompts, and generation params."""

    def __init__(self, ws: _State, parent=None):
        super().__init__(parent)
        self._ws = ws

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Step 2: Character & Generation Settings</b>"
        ))

        # Name / token
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Character Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. megan")
        self._name.setFixedWidth(200)
        name_row.addWidget(self._name)
        name_row.addWidget(QLabel("Trigger Token:"))
        self._token = QLineEdit()
        self._token.setPlaceholderText("auto")
        self._token.setFixedWidth(200)
        name_row.addWidget(self._token)
        name_row.addStretch()
        layout.addLayout(name_row)

        self._name.textChanged.connect(
            lambda t: self._token.setText(f"sks_{t.strip().lower().replace(' ', '_')}")
            if t and not self._token.isModified() else None
        )

        # Generation params
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("Clips per still:"))
        self._num_clips = QSpinBox()
        self._num_clips.setRange(1, 50)
        self._num_clips.setValue(8)
        self._num_clips.setFixedWidth(60)
        gen_row.addWidget(self._num_clips)
        gen_row.addWidget(QLabel("Frames:"))
        self._frames = QSpinBox()
        self._frames.setRange(17, 121)
        self._frames.setValue(49)
        self._frames.setSingleStep(8)
        self._frames.setFixedWidth(60)
        gen_row.addWidget(self._frames)
        gen_row.addWidget(QLabel("FPS:"))
        self._fps = QSpinBox()
        self._fps.setRange(8, 30)
        self._fps.setValue(16)
        self._fps.setFixedWidth(60)
        gen_row.addWidget(self._fps)
        gen_row.addStretch()
        layout.addLayout(gen_row)

        # Action prompts
        layout.addWidget(QLabel("Action Prompts (one per line):"))
        self._prompts = QPlainTextEdit()
        self._prompts.setPlainText("\n".join(DEFAULT_PROMPTS))
        self._prompts.setMaximumHeight(200)
        layout.addWidget(self._prompts)

    def on_leave(self) -> None:
        self._ws.character_name = self._name.text().strip()
        self._ws.trigger_token = self._token.text().strip() or f"sks_{self._ws.character_name.lower().replace(' ', '_')}"
        self._ws.num_clips = self._num_clips.value()
        self._ws.clip_frames = self._frames.value()
        self._ws.fps = self._fps.value()
        self._ws.action_prompts = [
            l.strip() for l in self._prompts.toPlainText().split("\n") if l.strip()
        ]

    def validate(self) -> bool:
        return bool(self._name.text().strip())


# ── Page 3: Batch Generate ───────────────────────────────────────────────────

class _BatchGeneratePage(WizardPage):
    """Generate short I2V clips from reference stills."""

    def __init__(self, ws: _State, state, parent=None):
        super().__init__(parent)
        self._ws = ws
        self._app_state = state
        self._worker = None
        self._generating = False
        self._queue: list[tuple[str, str]] = []  # (image_path, prompt) pairs

        layout = QVBoxLayout(self)
        self._desc = QLabel("")
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        self._gen_btn = QPushButton("Generate Clips")
        self._gen_btn.setStyleSheet("font-weight: bold;")
        self._gen_btn.clicked.connect(self._start_generation)
        layout.addWidget(self._gen_btn)

        self._skip_btn = QPushButton("Skip (use external clips)")
        self._skip_btn.setStyleSheet("color: #888;")
        self._skip_btn.clicked.connect(self._skip)
        layout.addWidget(self._skip_btn)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status)

        from PySide6.QtWidgets import QProgressBar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        layout.addStretch()

    def on_enter(self) -> None:
        n_stills = len(self._ws.reference_stills)
        n_prompts = len(self._ws.action_prompts)
        total = n_stills * n_prompts
        self._desc.setText(
            f"<b>Step 3: Generate Training Clips</b><br>"
            f"{n_stills} reference still(s) × {n_prompts} action prompts = "
            f"<b>{total} clips</b> to generate.<br>"
            f"Character: {self._ws.trigger_token}"
        )
        # Create dataset dir
        proj = self._app_state.project_manager.get_project_path(
            self._app_state.current_project or "_default"
        )
        ds_dir = proj / "datasets" / "video_lora" / (
            self._ws.character_name.lower().replace(" ", "_") or "character"
        )
        ds_dir.mkdir(parents=True, exist_ok=True)
        self._ws.dataset_dir = str(ds_dir)
        # Reset state
        self._ws.generated_clips.clear()
        self._gen_btn.setVisible(True)
        self._skip_btn.setVisible(True)

    def _skip(self) -> None:
        """Allow user to place clips externally."""
        if not self._ws.dataset_dir:
            ds_dir = Path(tempfile.mkdtemp(prefix="char_dataset_"))
            self._ws.dataset_dir = str(ds_dir)
        self._status.setText(
            f"Skipped generation. Place clips in:\n{self._ws.dataset_dir}"
        )
        self._generating = False

    def _start_generation(self) -> None:
        """Build the generation queue and start sequential I2V generation."""
        # Ensure pipeline is loaded
        pipeline = self._app_state.pipeline
        if pipeline is None:
            self._status.setText("Loading video pipeline...")
            try:
                self._app_state.load_pipelines()
                pipeline = self._app_state.pipeline
            except Exception as exc:
                self._status.setText(f"Pipeline load failed: {exc}")
                return
        if pipeline is None:
            self._status.setText("Pipeline not available. Check Settings.")
            return

        # Build queue: (image_path, prompt) for each still × prompt combo
        self._queue.clear()
        for still_path in self._ws.reference_stills:
            for prompt in self._ws.action_prompts:
                self._queue.append((still_path, prompt))

        self._total = len(self._queue)
        self._completed = 0
        self._generating = True
        self._gen_btn.setVisible(False)
        self._skip_btn.setVisible(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, self._total)
        self._progress.setValue(0)
        self._run_next()

    def _run_next(self) -> None:
        """Generate the next clip in the queue."""
        if not self._queue:
            self._on_all_done()
            return

        image_path, prompt = self._queue.pop(0)
        self._completed += 1
        clip_num = self._completed
        self._status.setText(
            f"Generating clip {clip_num}/{self._total}: {prompt[:50]}"
        )
        self._progress.setValue(clip_num - 1)

        # Build config for this clip
        from supremediffusion.config.project_config import ProjectConfig
        cfg = ProjectConfig()
        cfg.prompt = f"{self._ws.trigger_token} {prompt}"
        cfg.negative_prompt = ""
        cfg.video_length = self._ws.clip_frames
        cfg.fps = self._ws.fps
        cfg.num_inference_steps = 4  # Lightning fast
        cfg.guidance_scale = 1.0
        cfg.seed = -1  # Random each time
        cfg.mode = 1

        from sdqt.workers.inference import InferenceWorker
        worker = InferenceWorker(
            pipeline=self._app_state.pipeline,
            project_name=self._app_state.current_project or "_default",
            project_config=cfg,
            mode=1,
            image_path=image_path,
            global_config=self._app_state.global_config,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._status.setText(d))
        worker.finished_ok.connect(self._on_clip_done)
        worker.error.connect(self._on_clip_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_clip_done(self, result_path: str) -> None:
        """Move generated clip to dataset dir and continue."""
        if result_path and Path(result_path).is_file():
            ds = Path(self._ws.dataset_dir)
            dest = ds / f"clip_{self._completed:03d}.mp4"
            shutil.copy2(result_path, str(dest))
            self._ws.generated_clips.append(str(dest))
            logger.info("Generated training clip: %s", dest.name)
        self._run_next()

    def _on_clip_error(self, msg: str) -> None:
        """Log error and continue to next clip."""
        logger.warning("Clip generation failed: %s", msg)
        self._status.setText(f"Clip {self._completed} failed: {msg[:60]}. Continuing...")
        self._run_next()

    def _on_all_done(self) -> None:
        """All clips generated."""
        self._generating = False
        self._progress.setValue(self._total)
        n = len(self._ws.generated_clips)
        self._status.setText(
            f"Done — {n}/{self._total} clips generated in {self._ws.dataset_dir}"
        )
        self._gen_btn.setVisible(False)
        self._skip_btn.setVisible(False)

    def validate(self) -> bool:
        if self._generating:
            return False  # Don't advance while generating
        if not self._ws.generated_clips and not self._ws.dataset_dir:
            self._status.setText("Generate clips or skip to use external clips.")
            return False
        return True


# ── Page 4: Review Grid ─────────────────────────────────────────────────────

class _ReviewGridPage(WizardPage):
    """Review generated clips — keep or reject each one."""

    def __init__(self, ws: _State, parent=None):
        super().__init__(parent)
        self._ws = ws

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Step 4: Review & Curate</b><br>"
            "Uncheck any clips where the face/body drifted. Keep only consistent ones."
        ))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(8)
        scroll.setWidget(self._grid_widget)
        layout.addWidget(scroll, 1)

        self._checks: list[QCheckBox] = []

    def on_enter(self) -> None:
        # Clear old grid
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._checks.clear()

        # Populate from dataset dir
        ds = Path(self._ws.dataset_dir)
        clips = []
        if ds.is_dir():
            exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            clips = sorted(p for p in ds.iterdir() if p.suffix.lower() in exts)

        if not clips and self._ws.generated_clips:
            clips = [Path(p) for p in self._ws.generated_clips if Path(p).is_file()]

        self._ws.kept = [True] * len(clips)
        cols = 4
        for i, clip_path in enumerate(clips):
            card = QVBoxLayout()
            # Thumbnail
            thumb_label = QLabel()
            thumb_label.setFixedSize(160, 90)
            thumb_label.setStyleSheet("background: #333; border: 1px solid #555;")
            thumb_label.setAlignment(Qt.AlignCenter)
            try:
                from supremediffusion.utils.video import extract_single_frame
                img = extract_single_frame(str(clip_path), 0)
                from PySide6.QtGui import QImage
                qimg = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                thumb_label.setPixmap(pix)
            except Exception:
                thumb_label.setText(clip_path.stem[:20])

            cb = QCheckBox(clip_path.stem[:25])
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, idx=i: self._on_toggle(idx, checked))
            self._checks.append(cb)

            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(2)
            cl.addWidget(thumb_label)
            cl.addWidget(cb)
            self._grid.addWidget(container, i // cols, i % cols)

    def _on_toggle(self, idx: int, checked: bool) -> None:
        if 0 <= idx < len(self._ws.kept):
            self._ws.kept[idx] = checked

    def validate(self) -> bool:
        return any(self._ws.kept)


# ── Page 5: Caption & Launch ─────────────────────────────────────────────────

class _CaptionLaunchPage(WizardPage):
    """Auto-caption clips and send to Video LoRA tab."""

    def __init__(self, ws: _State, wizard: CharacterDatasetWizard, parent=None):
        super().__init__(parent)
        self._ws = ws
        self._wizard = wizard

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Step 5: Caption & Launch Training</b><br>"
            "Captions will use your trigger token. Adjust training config below."
        ))

        # Caption preview
        self._caption_preview = QLabel("")
        self._caption_preview.setStyleSheet("color: #aaa; padding: 4px; background: #2a2a2a; border-radius: 3px;")
        self._caption_preview.setWordWrap(True)
        layout.addWidget(self._caption_preview)

        # Summary
        self._summary = QLabel("")
        self._summary.setStyleSheet("color: #888; padding-top: 6px;")
        layout.addWidget(self._summary)

        layout.addStretch()

    def on_enter(self) -> None:
        kept = sum(1 for k in self._ws.kept if k)
        total = len(self._ws.kept)
        token = self._ws.trigger_token or "sks_character"
        self._caption_preview.setText(
            f"Example caption:\n\"{token} person, [action description]\""
        )
        self._summary.setText(
            f"Keeping {kept}/{total} clips. Trigger token: {token}\n"
            f"Click 'Finish' to prepare the dataset and open Video LoRA training."
        )

    def validate(self) -> bool:
        # Prepare final dataset
        self._prepare_dataset()
        # Emit signal to main window
        self._wizard.send_to_video_lora.emit(
            self._ws.dataset_dir,
            self._ws.trigger_token,
            {},
        )
        return True

    def _prepare_dataset(self) -> None:
        """Copy kept clips to dataset dir and write caption files."""
        ds = Path(self._ws.dataset_dir)
        ds.mkdir(parents=True, exist_ok=True)

        # Collect source clips
        source_clips = []
        if ds.is_dir():
            exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            source_clips = sorted(p for p in ds.iterdir() if p.suffix.lower() in exts)

        if not source_clips and self._ws.generated_clips:
            source_clips = [Path(p) for p in self._ws.generated_clips if Path(p).is_file()]

        token = self._ws.trigger_token or "sks_character"
        kept_idx = 0
        for i, clip_path in enumerate(source_clips):
            if i < len(self._ws.kept) and not self._ws.kept[i]:
                # Remove rejected clips from dataset dir
                if clip_path.parent == ds:
                    clip_path.unlink(missing_ok=True)
                continue
            # Write caption file
            caption = f"{token} person"
            if kept_idx < len(self._ws.action_prompts):
                caption = f"{token} {self._ws.action_prompts[kept_idx]}"
            caption_path = clip_path.with_suffix(".txt")
            caption_path.write_text(caption, encoding="utf-8")
            kept_idx += 1


# ── Main Wizard ──────────────────────────────────────────────────────────────

class CharacterDatasetWizard(SequenceWizard):
    """Wizard to build a video LoRA training dataset for character consistency."""

    send_to_video_lora = Signal(str, str, dict)  # dataset_dir, trigger_token, config

    def __init__(self, state, *, parent=None):
        super().__init__("Character Dataset Builder", state, parent=parent)
        self._ws = _State()

        self._pages = [
            _ReferenceStillsPage(self._ws, state),
            _CharacterConfigPage(self._ws),
            _BatchGeneratePage(self._ws, state),
            _ReviewGridPage(self._ws),
            _CaptionLaunchPage(self._ws, self),
        ]
        self._finish_setup()
