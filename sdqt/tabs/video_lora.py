"""Video LoRA Training tab — Wan 2.2 dual high/low noise character LoRA training."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab

logger = logging.getLogger(__name__)


class VideoLoRATab(BaseTab):
    """Video LoRA Training for Wan 2.2 — dual high/low noise character LoRA."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Header
        header = QLabel("<b>Video LoRA Training (Wan 2.2)</b>")
        header.setStyleSheet("font-size: 14px; padding-bottom: 4px;")
        layout.addWidget(header)

        # Character / Trigger Token
        char_row = QHBoxLayout()
        char_row.setSpacing(6)
        char_row.addWidget(QLabel("Character Name:"))
        self._char_name = QLineEdit()
        self._char_name.setPlaceholderText("e.g. megan")
        self._char_name.setFixedWidth(200)
        self._char_name.textChanged.connect(self._update_trigger)
        char_row.addWidget(self._char_name)
        char_row.addWidget(QLabel("Trigger Token:"))
        self._trigger = QLineEdit()
        self._trigger.setPlaceholderText("auto-generated")
        self._trigger.setFixedWidth(200)
        char_row.addWidget(self._trigger)
        char_row.addStretch()
        layout.addLayout(char_row)

        # Dataset panel
        ds_group = QGroupBox("Training Clips")
        ds_layout = QVBoxLayout(ds_group)

        ds_row = QHBoxLayout()
        ds_row.setSpacing(6)
        self._ds_path = QLineEdit()
        self._ds_path.setReadOnly(True)
        self._ds_path.setPlaceholderText("No dataset — use Character Dataset Builder wizard")
        ds_row.addWidget(self._ds_path, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse)
        ds_row.addWidget(self._browse_btn)
        ds_layout.addLayout(ds_row)

        self._clip_list = QListWidget()
        self._clip_list.setMaximumHeight(150)
        ds_layout.addWidget(self._clip_list)

        self._clip_count = QLabel("0 clips")
        self._clip_count.setStyleSheet("color: #888;")
        ds_layout.addWidget(self._clip_count)

        layout.addWidget(ds_group)

        # Dual noise config
        config_row = QHBoxLayout()
        config_row.setSpacing(12)

        for label, prefix in [("High-Noise LoRA", "hn"), ("Low-Noise LoRA", "ln")]:
            group = QGroupBox(label)
            g_layout = QVBoxLayout(group)
            g_layout.setSpacing(4)

            def _param_row(parent_layout, name, default, min_v, max_v, step, decimals=0, prefix=prefix):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{name}:"))
                if decimals > 0:
                    spin = QDoubleSpinBox()
                    spin.setDecimals(decimals)
                else:
                    spin = QSpinBox()
                spin.setRange(min_v, max_v)
                spin.setValue(default)
                spin.setSingleStep(step)
                spin.setFixedWidth(80)
                row.addWidget(spin)
                row.addStretch()
                parent_layout.addLayout(row)
                setattr(self, f"_{prefix}_{name.lower().replace(' ', '_')}", spin)

            _param_row(g_layout, "Rank", 64, 1, 512, 16)
            _param_row(g_layout, "Alpha", 32, 1, 512, 16)
            _param_row(g_layout, "LR", 1e-4, 1e-6, 1e-2, 1e-5, decimals=6)
            _param_row(g_layout, "Steps", 1000, 100, 50000, 100)

            config_row.addWidget(group)

        layout.addLayout(config_row)

        # General settings
        gen_row = QHBoxLayout()
        gen_row.setSpacing(6)
        gen_row.addWidget(QLabel("Batch Size:"))
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 8)
        self._batch_size.setValue(1)
        self._batch_size.setFixedWidth(60)
        gen_row.addWidget(self._batch_size)
        gen_row.addWidget(QLabel("Gradient Accum:"))
        self._grad_accum = QSpinBox()
        self._grad_accum.setRange(1, 64)
        self._grad_accum.setValue(4)
        self._grad_accum.setFixedWidth(60)
        gen_row.addWidget(self._grad_accum)
        gen_row.addStretch()
        layout.addLayout(gen_row)

        # Training controls
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)
        self._train_btn = QPushButton("Start Training")
        self._train_btn.setStyleSheet("font-weight: bold;")
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train)
        ctrl_row.addWidget(self._train_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        ctrl_row.addWidget(self._abort_btn)
        ctrl_row.addStretch()
        help_btn = QPushButton("Workflow Guide")
        help_btn.clicked.connect(self._show_help)
        ctrl_row.addWidget(help_btn)
        layout.addLayout(ctrl_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._status)

        layout.addStretch()

    # ── Public API (for wizard) ──────────────────────────────────────────────

    def set_video_dataset(self, path: str, trigger_token: str = "",
                          config: dict | None = None) -> None:
        """Populate the tab from the Character Dataset Builder wizard."""
        self._ds_path.setText(path)
        if trigger_token:
            self._trigger.setText(trigger_token)
            # Extract char name from trigger
            name = trigger_token.replace("sks_", "").replace("_", " ")
            self._char_name.setText(name)
        self._refresh_clip_list(path)
        if config:
            for key, val in config.items():
                widget = getattr(self, f"_{key}", None)
                if widget and hasattr(widget, "setValue"):
                    widget.setValue(val)
        self._train_btn.setEnabled(True)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_trigger(self, text: str) -> None:
        if text and not self._trigger.isModified():
            token = "sks_" + text.strip().lower().replace(" ", "_")
            self._trigger.setText(token)

    @Slot()
    def _on_browse(self) -> None:
        from sdqt.utils.file_dialog import get_existing_directory
        path = get_existing_directory(self, "Select Video Dataset Folder")
        if path:
            self._ds_path.setText(path)
            self._refresh_clip_list(path)
            self._train_btn.setEnabled(True)

    def _refresh_clip_list(self, path: str) -> None:
        self._clip_list.clear()
        d = Path(path)
        if not d.is_dir():
            self._clip_count.setText("0 clips")
            return
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        clips = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        for c in clips:
            self._clip_list.addItem(c.name)
        self._clip_count.setText(f"{len(clips)} clips")

    @Slot()
    def _show_help(self) -> None:
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle("Video LoRA Workflow Guide")
        dlg.setMinimumSize(620, 500)
        layout = QVBoxLayout(dlg)
        text = QTextBrowser()
        text.setOpenExternalLinks(False)
        text.setHtml("""
<h2>Character Consistency with Video LoRA</h2>

<h3>Single Character in One Scene</h3>
<ol>
  <li><b>Daz2Supreme</b> — Render your character in the scene (final composition)</li>
  <li><b>Image Suite → Img2Img</b> — Run a realism pass on the full render (character + scene together)</li>
  <li><b>Video → Img2Vid</b> — Generate a short clip (2-3 sec) from the realistic image</li>
  <li><b>Video → StillGrabber</b> — Extract 10+ frames with varied poses/expressions</li>
  <li><b>Sequences → Character Dataset Builder</b> — Use those stills to batch-generate training clips with different actions</li>
  <li><b>Curate</b> — Keep clips where face/body stayed consistent, reject drifted ones</li>
  <li><b>Train</b> — Train a scene+character LoRA here (covers both identity and environment)</li>
  <li><b>Generate</b> — New poses/angles in the same scene with LoRA at 0.5 strength</li>
</ol>

<h3>Single Character in Multiple Scenes</h3>
<ol>
  <li><b>Daz2Supreme</b> — Render character on a <b>neutral/solid background</b></li>
  <li><b>Img2Img realism pass</b> — Photorealistic character, still neutral bg</li>
  <li><b>Img2Vid → StillGrabber → Dataset Builder</b> — Same as above</li>
  <li><b>Train character-only LoRA</b> (neutral bg means no scene contamination)</li>
  <li>For each scene: Daz render scene → realism pass → composite character in → generate with character LoRA</li>
</ol>

<h3>Two Characters Together</h3>
<ol>
  <li>Train <b>scene+charA LoRA</b> (character A composited into the scene, full pipeline)</li>
  <li>Train <b>charB LoRA on neutral bg</b> (character B alone, no scene)</li>
  <li>Stack at generation time: scene+charA at 0.5 + charB at 0.4-0.5</li>
  <li>Source image: Daz render of both characters in the scene → realism pass</li>
  <li>The scene comes from the first LoRA, charB identity from the second</li>
</ol>

<h3>Realism Pass Rules</h3>
<ul>
  <li><b>Always</b> run the realism pass before generating I2V clips — both for training data and production</li>
  <li><b>Never</b> do separate realism passes on character and scene — do them together in one pass to keep lighting/color consistent</li>
  <li>Exception: neutral-bg character training (character cutout only, no scene to conflict with)</li>
</ul>

<h3>LoRA Strength Guide</h3>
<ul>
  <li><b>0.3-0.4</b> — Subtle influence, good for scene LoRAs</li>
  <li><b>0.5-0.6</b> — Standard for character identity</li>
  <li><b>0.7+</b> — Strong lock, may reduce prompt responsiveness</li>
  <li>Total stacked weight should stay under ~1.5 to avoid artifacts</li>
</ul>

<h3>Training Tips (Wan 2.2 Lightning)</h3>
<ul>
  <li>Train <b>dual LoRAs</b> — one for high-noise transformer, one for low-noise</li>
  <li>Use the same dataset for both</li>
  <li>8-15 curated training clips is enough</li>
  <li>Clips should be 2-3 seconds, varied actions</li>
  <li>Use distinct trigger tokens per character (e.g. sks_megan, sks_john)</li>
  <li>Put trigger token at the start of each caption</li>
</ul>
""")
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.exec()

    @Slot()
    def _on_train(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        from sdqt.workers.video_lora_train import find_ai_toolkit, VideoLoRATrainWorker

        # Validate dataset
        ds = self._ds_path.text()
        if not ds or not Path(ds).is_dir():
            self._show_status("No dataset folder selected.")
            return
        exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        clips = [p for p in Path(ds).iterdir() if p.suffix.lower() in exts]
        if not clips:
            self._show_status("No video clips found in dataset folder.")
            return

        # Find ai-toolkit
        toolkit_dir = find_ai_toolkit()
        if not toolkit_dir:
            QMessageBox.warning(
                self, "ai-toolkit Not Found",
                "ai-toolkit is required for video LoRA training.\n\n"
                "Install it to one of these locations:\n"
                "  - third_party/ai-toolkit/\n"
                "  - ~/ai-toolkit/\n"
                "  - ~/Projects/ai-toolkit/\n\n"
                "Clone from: https://github.com/ostris/ai-toolkit",
            )
            return

        # Get model path
        model_path = self.state.global_config.model_paths.get("wan_model_dir", "")
        if not model_path:
            model_path = self.state.global_config.model_paths.get("transformer", "")
        if not model_path:
            QMessageBox.warning(self, "Model Path", "No Wan 2.2 model path configured in Settings.")
            return

        # Output dir
        output_dir = str(self.project_path / "lora_output" / (
            self._char_name.text().strip().lower().replace(" ", "_") or "video_lora"
        ))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        trigger = self._trigger.text().strip() or "sks_character"
        name = self._char_name.text().strip().lower().replace(" ", "_") or "character"

        worker = VideoLoRATrainWorker(
            ai_toolkit_dir=toolkit_dir,
            name=name,
            model_path=model_path,
            dataset_dir=ds,
            output_dir=output_dir,
            hn_rank=self._hn_rank.value(),
            hn_alpha=self._hn_alpha.value(),
            hn_lr=self._hn_lr.value(),
            hn_steps=self._hn_steps.value(),
            ln_rank=self._ln_rank.value(),
            ln_alpha=self._ln_alpha.value(),
            ln_lr=self._ln_lr.value(),
            ln_steps=self._ln_steps.value(),
            batch_size=self._batch_size.value(),
            grad_accum=self._grad_accum.value(),
            num_frames=49,
            fps=16,
            trigger_token=trigger,
            parent=self,
        )

        self._train_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._abort_btn.clicked.connect(lambda: worker.abort())
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)

        worker.progress.connect(lambda f, d: (
            self._progress.setValue(int(f * 100)),
            self._show_status(d),
        ))
        worker.finished_ok.connect(self._on_train_done)
        worker.error.connect(self._on_train_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_train_done(self, result: dict) -> None:
        self._train_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._progress.setVisible(False)

        parts = []
        if result.get("high_noise"):
            parts.append(f"High-noise: {result['high_noise']}")
        if result.get("low_noise"):
            parts.append(f"Low-noise: {result['low_noise']}")

        if parts:
            self._show_status("Training complete! " + " | ".join(parts))
            # Offer to copy to LoRA directory
            self._offer_copy_lora(result)
        else:
            self._show_status("Training finished but no LoRA files found.")

    def _on_train_error(self, msg: str) -> None:
        self._train_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._progress.setVisible(False)
        self._show_status(f"Training error: {msg}")

    def _offer_copy_lora(self, result: dict) -> None:
        from PySide6.QtWidgets import QMessageBox
        import shutil

        lora_dir = self.state.global_config.model_paths.get("lora_dir", "")
        if not lora_dir:
            return

        reply = QMessageBox.question(
            self, "Copy LoRA",
            f"Copy trained LoRA files to the LoRA directory?\n\n{lora_dir}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        lora_path = Path(lora_dir)
        lora_path.mkdir(parents=True, exist_ok=True)
        copied = []
        for key in ("high_noise", "low_noise"):
            src = result.get(key)
            if src and Path(src).is_file():
                name = self._char_name.text().strip().lower().replace(" ", "_") or "character"
                dest = lora_path / f"{name}_{key}.safetensors"
                shutil.copy2(src, str(dest))
                copied.append(dest.name)

        if copied:
            self._show_status(f"Copied to LoRA dir: {', '.join(copied)}")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._clip_list.clear()
        self._clip_count.setText("0 clips")
