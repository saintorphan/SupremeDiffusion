"""LoRA Training tab — dataset prep, config, training execution, progress monitoring."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Slot, QTimer, QFileSystemWatcher
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig
from sdqt.tabs.base import BaseTab
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.dataset_panel import DatasetPanel
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.loss_graph import LossGraphWidget
from sdqt.workers.lora_train import LoRATrainWorker, find_kohya_script

logger = logging.getLogger(__name__)

# Step/loss regex for parsing worker progress descriptions
_STEP_LOSS_RE = re.compile(r"Step (\d+)/\d+\s+Loss:\s*([\d.]+)")

# ── Presets ──────────────────────────────────────────────────────────────────

_PRESETS = {
    "SD 1.5 Quick": {
        "model_type": "SD 1.5", "rank": 32, "alpha": 32, "lr": "1e-4",
        "optimizer": "AdamW8bit", "steps": 1500, "scheduler": "cosine",
        "grad_ckpt": False, "precision": "fp16", "resolution": "512x512",
    },
    "SD 1.5 Quality": {
        "model_type": "SD 1.5", "rank": 64, "alpha": 64, "lr": "5e-5",
        "optimizer": "Prodigy", "steps": 3000, "scheduler": "cosine",
        "grad_ckpt": False, "precision": "bf16", "resolution": "512x512",
    },
    "SDXL Efficient": {
        "model_type": "SDXL", "rank": 16, "alpha": 16, "lr": "5e-5",
        "optimizer": "Prodigy", "steps": 2000, "scheduler": "cosine",
        "grad_ckpt": True, "precision": "bf16", "resolution": "1024x1024",
    },
    "SDXL Quality": {
        "model_type": "SDXL", "rank": 32, "alpha": 32, "lr": "3e-5",
        "optimizer": "Prodigy", "steps": 4000, "scheduler": "cosine",
        "grad_ckpt": True, "precision": "bf16", "resolution": "1024x1024",
    },
    "Pony Character": {
        "model_type": "SDXL", "rank": 16, "alpha": 16, "lr": "5e-5",
        "optimizer": "Prodigy", "steps": 3000, "scheduler": "cosine",
        "grad_ckpt": True, "precision": "bf16", "resolution": "1024x1024",
    },
    "Illustrious Style": {
        "model_type": "SDXL", "rank": 16, "alpha": 16, "lr": "5e-5",
        "optimizer": "Prodigy", "steps": 2500, "scheduler": "cosine",
        "grad_ckpt": True, "precision": "bf16", "resolution": "1024x1024",
    },
}


class LoRATrainingTab(BaseTab):
    """LoRA Training — dataset prep, training config, execution, and monitoring."""

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: LoRATrainWorker | None = None
        self._kohya_script = find_kohya_script()
        self._sample_watcher: QFileSystemWatcher | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        if not self._kohya_script:
            # Show install instructions
            msg_layout = QVBoxLayout()
            msg = QLabel(
                "LoRA Training requires kohya sd-scripts.\n\n"
                "Install with:\n"
                "  git clone https://github.com/kohya-ss/sd-scripts.git third_party/sd-scripts\n"
                "  pip install -r third_party/sd-scripts/requirements.txt\n\n"
                "Then restart the application.\n\n"
                "The dataset preparation tools below still work without kohya."
            )
            msg.setStyleSheet("font-size: 14px; color: #aaa; padding: 20px;")
            msg.setWordWrap(True)
            msg_layout.addWidget(msg)
            msg_layout.addStretch()
            # Still build the full UI — just disable the train button later
            # Fall through to build_ui

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Dataset Panel ──────────────────────────────────────────
        self._dataset = DatasetPanel()
        self._dataset.caption_batch_requested.connect(self._on_batch_caption)
        self._dataset.prepare_requested.connect(self._on_prepare_dataset)
        splitter.addWidget(self._dataset)

        # ── Right: Config + Progress ─────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(4, 4, 4, 4)
        rv.setSpacing(4)

        # Presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("(custom)")
        self._preset_combo.addItems(list(_PRESETS.keys()))
        self._preset_combo.setFixedWidth(160)
        self._preset_combo.currentTextChanged.connect(self._on_preset)
        preset_row.addWidget(self._preset_combo)
        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet("color: #888; font-size: 12px;")
        preset_row.addWidget(self._vram_label)
        preset_row.addStretch()
        rv.addLayout(preset_row)

        # Detect VRAM
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
                self._vram_label.setText(f"GPU: {vram:.0f} GB VRAM")
        except Exception:
            pass

        # ── Model & Architecture ─────────────────────────────────────────
        model_section = CollapsibleSection("Model & Architecture")

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Checkpoint:"))
        self._checkpoint_combo = QComboBox()
        self._checkpoint_combo.setMinimumWidth(200)
        model_row.addWidget(self._checkpoint_combo, 1)
        model_row.addWidget(QLabel("Type:"))
        self._model_type = QComboBox()
        self._model_type.addItems(["SD 1.5", "SDXL"])
        self._model_type.setFixedWidth(80)
        self._model_type.currentTextChanged.connect(self._on_model_type_changed)
        model_row.addWidget(self._model_type)
        mw = QWidget()
        mw.setLayout(model_row)
        model_section.add_widget(mw)

        arch_row = QHBoxLayout()
        arch_row.addWidget(QLabel("Network:"))
        self._network_type = QComboBox()
        self._network_type.addItems(["LoRA", "LoCon", "LoHa"])
        self._network_type.setFixedWidth(80)
        arch_row.addWidget(self._network_type)
        arch_row.addWidget(QLabel("Rank:"))
        self._rank = QComboBox()
        self._rank.addItems(["4", "8", "16", "32", "64", "128"])
        self._rank.setCurrentText("16")
        self._rank.setFixedWidth(60)
        arch_row.addWidget(self._rank)
        arch_row.addWidget(QLabel("Alpha:"))
        self._alpha = QSpinBox()
        self._alpha.setRange(1, 128)
        self._alpha.setValue(16)
        self._alpha.setFixedWidth(60)
        arch_row.addWidget(self._alpha)
        arch_row.addStretch()
        aw = QWidget()
        aw.setLayout(arch_row)
        model_section.add_widget(aw)

        rv.addWidget(model_section)

        # ── Training Parameters ──────────────────────────────────────────
        train_section = CollapsibleSection("Training Parameters")

        t_row1 = QHBoxLayout()
        t_row1.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(100, 50000)
        self._steps.setValue(2000)
        self._steps.setFixedWidth(80)
        t_row1.addWidget(self._steps)
        t_row1.addWidget(QLabel("LR:"))
        self._lr = QLineEdit("5e-5")
        self._lr.setFixedWidth(80)
        t_row1.addWidget(self._lr)
        t_row1.addWidget(QLabel("Scheduler:"))
        self._scheduler = QComboBox()
        self._scheduler.addItems(["cosine", "cosine_with_restarts", "constant",
                                   "constant_with_warmup", "polynomial"])
        self._scheduler.setFixedWidth(140)
        t_row1.addWidget(self._scheduler)
        t_row1.addStretch()
        tw1 = QWidget()
        tw1.setLayout(t_row1)
        train_section.add_widget(tw1)

        t_row2 = QHBoxLayout()
        t_row2.addWidget(QLabel("Warmup:"))
        self._warmup = QSpinBox()
        self._warmup.setRange(0, 1000)
        self._warmup.setValue(0)
        self._warmup.setFixedWidth(70)
        t_row2.addWidget(self._warmup)
        t_row2.addWidget(QLabel("TE LR:"))
        self._te_lr = QLineEdit("0")
        self._te_lr.setFixedWidth(70)
        self._te_lr.setToolTip("Text encoder learning rate (0 = freeze)")
        t_row2.addWidget(self._te_lr)
        self._train_te = QCheckBox("Train Text Encoder")
        t_row2.addWidget(self._train_te)
        t_row2.addStretch()
        tw2 = QWidget()
        tw2.setLayout(t_row2)
        train_section.add_widget(tw2)

        rv.addWidget(train_section)

        # ── Optimization ─────────────────────────────────────────────────
        opt_section = CollapsibleSection("Optimization")

        o_row1 = QHBoxLayout()
        o_row1.addWidget(QLabel("Optimizer:"))
        self._optimizer = QComboBox()
        self._optimizer.addItems(["Prodigy", "AdamW", "AdamW8bit", "DAdaptAdam", "Lion"])
        self._optimizer.setFixedWidth(110)
        o_row1.addWidget(self._optimizer)
        o_row1.addWidget(QLabel("Precision:"))
        self._precision = QComboBox()
        self._precision.addItems(["bf16", "fp16", "no"])
        self._precision.setFixedWidth(70)
        o_row1.addWidget(self._precision)
        self._grad_ckpt = QCheckBox("Grad Checkpoint")
        self._grad_ckpt.setToolTip("Trade compute for VRAM — required for SDXL on 12 GB")
        o_row1.addWidget(self._grad_ckpt)
        o_row1.addStretch()
        ow1 = QWidget()
        ow1.setLayout(o_row1)
        opt_section.add_widget(ow1)

        o_row2 = QHBoxLayout()
        self._cache_latents = QCheckBox("Cache Latents")
        self._cache_latents.setChecked(True)
        o_row2.addWidget(self._cache_latents)
        self._cache_te = QCheckBox("Cache TE Outputs")
        self._cache_te.setChecked(True)
        o_row2.addWidget(self._cache_te)
        o_row2.addWidget(QLabel("Noise Offset:"))
        self._noise_offset = QDoubleSpinBox()
        self._noise_offset.setRange(0, 0.1)
        self._noise_offset.setDecimals(3)
        self._noise_offset.setSingleStep(0.005)
        self._noise_offset.setValue(0.0)
        self._noise_offset.setFixedWidth(70)
        o_row2.addWidget(self._noise_offset)
        o_row2.addWidget(QLabel("Min SNR:"))
        self._min_snr = QDoubleSpinBox()
        self._min_snr.setRange(0, 20)
        self._min_snr.setValue(5.0)
        self._min_snr.setFixedWidth(60)
        self._min_snr.setToolTip("Min-SNR gamma loss weighting (0 = off)")
        o_row2.addWidget(self._min_snr)
        o_row2.addStretch()
        ow2 = QWidget()
        ow2.setLayout(o_row2)
        opt_section.add_widget(ow2)

        rv.addWidget(opt_section)

        # ── Output ───────────────────────────────────────────────────────
        out_section = CollapsibleSection("Output")

        out_row1 = QHBoxLayout()
        out_row1.addWidget(QLabel("Name:"))
        self._output_name = QLineEdit("my_lora")
        self._output_name.setFixedWidth(200)
        out_row1.addWidget(self._output_name)
        out_row1.addWidget(QLabel("Save every:"))
        self._save_every = QSpinBox()
        self._save_every.setRange(0, 10000)
        self._save_every.setValue(500)
        self._save_every.setFixedWidth(70)
        self._save_every.setToolTip("Save checkpoint every N steps (0 = disabled)")
        out_row1.addWidget(self._save_every)
        out_row1.addStretch()
        outw1 = QWidget()
        outw1.setLayout(out_row1)
        out_section.add_widget(outw1)

        out_row2 = QHBoxLayout()
        out_row2.addWidget(QLabel("Sample every:"))
        self._sample_every = QSpinBox()
        self._sample_every.setRange(0, 10000)
        self._sample_every.setValue(200)
        self._sample_every.setFixedWidth(70)
        out_row2.addWidget(self._sample_every)
        out_row2.addWidget(QLabel("Prompt:"))
        self._sample_prompt = QLineEdit()
        self._sample_prompt.setPlaceholderText("(uses first caption if empty)")
        out_row2.addWidget(self._sample_prompt, 1)
        outw2 = QWidget()
        outw2.setLayout(out_row2)
        out_section.add_widget(outw2)

        rv.addWidget(out_section)

        # ── Action Buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train LoRA")
        self._train_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; color: #fff;"
            " background: #0078d4; border-radius: 4px; padding: 8px 20px; } "
            "QPushButton:hover { background: #005a9e; } "
            "QPushButton:disabled { background: #444; color: #888; }")
        self._train_btn.clicked.connect(self._on_train)
        if not self._kohya_script:
            self._train_btn.setEnabled(False)
            self._train_btn.setToolTip("kohya sd-scripts not installed")
        btn_row.addWidget(self._train_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; color: #fff;"
            " background: #cc3333; border-radius: 4px; padding: 8px 16px; } "
            "QPushButton:hover { background: #aa2222; }")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)

        self._open_output_btn = QPushButton("Open Output")
        self._open_output_btn.setVisible(False)
        self._open_output_btn.clicked.connect(self._on_open_output)
        btn_row.addWidget(self._open_output_btn)

        btn_row.addStretch()
        rv.addLayout(btn_row)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setFixedHeight(20)
        rv.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("font-size: 12px; color: #ccc;")
        rv.addWidget(self._progress_label)

        # ── Loss Graph ───────────────────────────────────────────────────
        self._loss_graph = LossGraphWidget()
        self._loss_graph.setMinimumHeight(150)
        rv.addWidget(self._loss_graph)

        # ── Sample Previews ──────────────────────────────────────────────
        self._samples = ImageGalleryWidget("Training Samples")
        self._samples.setMaximumHeight(160)
        rv.addWidget(self._samples)

        rv.addStretch()
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        # Populate checkpoints
        self._populate_checkpoints()

    # ── Presets ───────────────────────────────────────────────────────────

    @Slot(str)
    def _on_preset(self, name: str) -> None:
        preset = _PRESETS.get(name)
        if not preset:
            return
        self._model_type.setCurrentText(preset.get("model_type", "SDXL"))
        self._rank.setCurrentText(str(preset.get("rank", 16)))
        self._alpha.setValue(preset.get("alpha", 16))
        self._lr.setText(preset.get("lr", "5e-5"))
        self._optimizer.setCurrentText(preset.get("optimizer", "Prodigy"))
        self._steps.setValue(preset.get("steps", 2000))
        self._scheduler.setCurrentText(preset.get("scheduler", "cosine"))
        self._grad_ckpt.setChecked(preset.get("grad_ckpt", True))
        self._precision.setCurrentText(preset.get("precision", "bf16"))
        # Update dataset resolution
        res = preset.get("resolution", "1024x1024")
        idx = self._dataset.resolution.findText(res)
        if idx >= 0:
            self._dataset.resolution.setCurrentIndex(idx)

    @Slot(str)
    def _on_model_type_changed(self, model_type: str) -> None:
        """Gate settings based on model type + VRAM."""
        is_sdxl = model_type == "SDXL"
        if is_sdxl:
            self._grad_ckpt.setChecked(True)
            self._train_te.setChecked(False)
            # Cap rank for low VRAM
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
                    if vram < 16:
                        current = int(self._rank.currentText())
                        if current > 32:
                            self._rank.setCurrentText("32")
            except Exception:
                pass

    # ── Populate checkpoints ─────────────────────────────────────────────

    def _populate_checkpoints(self) -> None:
        self._checkpoint_combo.clear()
        try:
            ckpt_dir = self.state.global_config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir and Path(ckpt_dir).is_dir():
                for f in sorted(Path(ckpt_dir).iterdir()):
                    if f.suffix.lower() in (".safetensors", ".ckpt"):
                        self._checkpoint_combo.addItem(f.name, str(f))
        except Exception:
            pass
        if self._checkpoint_combo.count() == 0:
            self._checkpoint_combo.addItem("(no checkpoints found)")

    # ── Config generation ────────────────────────────────────────────────

    def _generate_config(self) -> Path:
        """Build kohya TOML config from UI values."""
        dataset_path = self._dataset.dataset_path
        if not dataset_path:
            raise ValueError("No dataset folder selected")

        ckpt_path = self._checkpoint_combo.currentData()
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise ValueError("No valid checkpoint selected")

        output_dir = self.project_path / "loras"
        output_dir.mkdir(parents=True, exist_ok=True)

        res_text = self._dataset.resolution.currentText()
        res = int(res_text.split("x")[0])

        is_sdxl = self._model_type.currentText() == "SDXL"
        network_module = {
            "LoRA": "networks.lora",
            "LoCon": "lycoris.kohya",
            "LoHa": "lycoris.kohya",
        }.get(self._network_type.currentText(), "networks.lora")

        rank = int(self._rank.currentText())

        # Build dataset TOML structure
        # kohya expects a specific directory layout: num_repeats_conceptname/
        # We'll create a symlink structure or just reference directly
        repeats = self._dataset.repeats.value()

        config = {
            "pretrained_model_name_or_path": ckpt_path,
            "train_data_dir": dataset_path,
            "output_dir": str(output_dir),
            "output_name": self._output_name.text() or "lora",
            "network_module": network_module,
            "network_dim": rank,
            "network_alpha": self._alpha.value(),
            "resolution": f"{res},{res}",
            "max_train_steps": self._steps.value(),
            "learning_rate": float(self._lr.text()),
            "lr_scheduler": self._scheduler.currentText(),
            "lr_warmup_steps": self._warmup.value(),
            "optimizer_type": self._optimizer.currentText(),
            "mixed_precision": self._precision.currentText(),
            "gradient_checkpointing": self._grad_ckpt.isChecked(),
            "cache_latents": self._cache_latents.isChecked(),
            "cache_latents_to_disk": self._cache_latents.isChecked(),
            "cache_text_encoder_outputs": self._cache_te.isChecked(),
            "enable_bucket": self._dataset.enable_buckets.isChecked(),
            "flip_aug": self._dataset.flip_aug.isChecked(),
            "keep_tokens": self._dataset.keep_tokens.value(),
            "seed": 42,
            "max_token_length": 225,
            "save_every_n_steps": self._save_every.value() if self._save_every.value() > 0 else None,
            "train_batch_size": 1,
            "dataset_repeats": repeats,
        }

        if self._noise_offset.value() > 0:
            config["noise_offset"] = self._noise_offset.value()
        if self._min_snr.value() > 0:
            config["min_snr_gamma"] = self._min_snr.value()
        if self._train_te.isChecked():
            config["train_text_encoder"] = True
            if self._te_lr.text() and self._te_lr.text() != "0":
                config["text_encoder_lr"] = float(self._te_lr.text())

        # Sample generation
        if self._sample_every.value() > 0:
            config["sample_every_n_steps"] = self._sample_every.value()
            prompt = self._sample_prompt.text()
            if not prompt:
                # Use first caption from dataset
                captions = list(Path(dataset_path).glob("*.txt"))
                if captions:
                    prompt = captions[0].read_text(encoding="utf-8").strip()[:200]
            if prompt:
                config["sample_prompts"] = prompt
                config["sample_sampler"] = "euler_a"

        # SDXL-specific
        if is_sdxl:
            config["sdxl"] = True

        # LyCORIS-specific args
        if self._network_type.currentText() in ("LoCon", "LoHa"):
            algo = "locon" if self._network_type.currentText() == "LoCon" else "loha"
            config["network_args"] = [f"algo={algo}"]

        # Write TOML
        try:
            import toml
            config_path = Path(tempfile.mktemp(suffix=".toml", prefix="lora_config_"))
            # Filter out None values
            config = {k: v for k, v in config.items() if v is not None}
            with open(config_path, "w") as f:
                toml.dump(config, f)
        except ImportError:
            # Fallback: write as JSON (kohya also supports --config_file with json)
            config_path = Path(tempfile.mktemp(suffix=".json", prefix="lora_config_"))
            config = {k: v for k, v in config.items() if v is not None}
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        logger.info("Training config written: %s", config_path)
        return config_path

    # ── Training execution ───────────────────────────────────────────────

    @Slot()
    def _on_train(self) -> None:
        if not self._kohya_script:
            self._show_status("kohya sd-scripts not installed.")
            return
        if not self._dataset.dataset_path:
            self._show_status("No dataset folder selected.")
            return
        if self._dataset.image_count == 0:
            self._show_status("Dataset folder is empty.")
            return

        try:
            config_path = self._generate_config()
        except Exception as exc:
            self._show_status(f"Config error: {exc}")
            return

        output_name = self._output_name.text() or "lora"
        output_dir = self.project_path / "loras"
        output_path = output_dir / f"{output_name}.safetensors"

        is_sdxl = self._model_type.currentText() == "SDXL"

        self._train_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._loss_graph.clear()

        # Watch for sample images
        sample_dir = output_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        self._sample_watcher = QFileSystemWatcher([str(sample_dir)], self)
        self._sample_watcher.directoryChanged.connect(self._on_samples_changed)

        worker = LoRATrainWorker(
            config_path=str(config_path),
            kohya_script=self._kohya_script,
            output_path=str(output_path),
            is_sdxl=is_sdxl,
            parent=self,
        )
        worker.progress.connect(self._on_progress)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()
        self._show_status("Training started...")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()
            self._show_status("Aborting...")

    def _on_progress(self, frac: float, desc: str) -> None:
        self._progress_bar.setValue(int(frac * 1000))
        self._progress_label.setText(desc)
        self._show_status(desc)

        # Parse step/loss for graph
        m = _STEP_LOSS_RE.search(desc)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            self._loss_graph.add_point(step, loss)

    def _on_done(self, path: str) -> None:
        self._train_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        self._open_output_btn.setVisible(True)
        self._show_status(f"Training complete: {Path(path).name}")

        # Auto-copy to LoRA directory
        self._auto_copy_lora(path)

    def _on_error(self, msg: str) -> None:
        self._train_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        self._show_status(f"Training error: {msg}")

    def _auto_copy_lora(self, path: str) -> None:
        """Copy trained LoRA to the SD LoRA directory so it appears in pickers."""
        try:
            lora_dir = self.state.global_config.model_paths.get("sd_lora_dir", "")
            if lora_dir and Path(lora_dir).is_dir():
                dest = Path(lora_dir) / Path(path).name
                if not dest.exists():
                    shutil.copy2(path, dest)
                    self._show_status(f"LoRA copied to {dest.name} — refresh LoRA lists to use it.")
                    logger.info("Auto-copied LoRA to %s", dest)
        except Exception as exc:
            logger.warning("Auto-copy LoRA failed: %s", exc)

    @Slot(str)
    def _on_samples_changed(self, path: str) -> None:
        """Reload sample gallery when new sample images appear."""
        sample_dir = Path(path)
        if sample_dir.is_dir():
            pngs = sorted(sample_dir.glob("*.png"), key=lambda f: f.stat().st_mtime)
            if pngs:
                self._samples.load_images([str(p) for p in pngs[-8:]])

    @Slot()
    def _on_open_output(self) -> None:
        """Open the output LoRA directory."""
        import subprocess, sys
        output_dir = self.project_path / "loras"
        if output_dir.is_dir():
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(output_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(output_dir)])

    # ── Batch captioning ─────────────────────────────────────────────────

    @Slot(str, str)
    def _on_prepare_dataset(self, source: str, dest: str) -> None:
        """After Prepare Dataset copies and renames, auto-caption with Qwen VL."""
        self._show_status(f"Dataset prepared at {Path(dest).name}. Starting Qwen VL captioning...")
        self._on_batch_caption("qwen")

    @Slot(str)
    def _on_batch_caption(self, method: str) -> None:
        """Run BLIP, WD14, or Qwen VL on all uncaptioned images."""
        if method == "qwen":
            from sdqt.deps import check_and_install
            if not check_and_install("qwen_vl", self):
                return
            from sdqt.models.manager import check_and_prompt_download
            if not check_and_prompt_download(
                "qwen_vl", self.state.model_registry, self,
                on_progress=lambda f, d: self._show_status(d),
            ):
                return

        uncaptioned = self._dataset.get_uncaptioned_paths()
        if not uncaptioned:
            self._show_status("All images already captioned.")
            return
        self._show_status(f"Captioning {len(uncaptioned)} images with {method.upper()}...")
        # Run captioning in sequence using a timer to keep UI responsive
        self._caption_queue = list(uncaptioned)
        self._caption_method = method
        self._caption_total = len(uncaptioned)
        self._caption_idx = 0
        self._caption_next()

    def _caption_next(self) -> None:
        if self._caption_idx >= len(self._caption_queue):
            self._dataset.set_batch_progress(0, 0)
            self._show_status(f"Captioning complete ({self._caption_total} images).")
            return

        path = self._caption_queue[self._caption_idx]
        self._dataset.set_batch_progress(self._caption_idx, self._caption_total)

        if self._caption_method == "blip":
            from sdqt.workers.interrogate import BLIPInterrogateWorker
            worker = BLIPInterrogateWorker(path, self.state, parent=self)
        elif self._caption_method == "qwen":
            from sdqt.workers.interrogate import QwenCaptionWorker
            worker = QwenCaptionWorker(path, self.state, parent=self)
        else:
            from sdqt.workers.interrogate import WD14TaggerWorker
            worker = WD14TaggerWorker(path, self.state, parent=self)

        worker.finished_ok.connect(self._on_caption_result)
        worker.error.connect(lambda msg: self._on_caption_result(""))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_caption_result(self, caption: str) -> None:
        if caption and self._caption_idx < len(self._caption_queue):
            path = self._caption_queue[self._caption_idx]
            txt_path = Path(path).with_suffix(".txt")
            txt_path.write_text(caption, encoding="utf-8")
        self._caption_idx += 1
        # Use a timer to process next to keep UI responsive
        QTimer.singleShot(10, self._caption_next)

    # ── Project change ───────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        # Save current state before switching
        self._save_lora_state()
        super().on_project_changed(project_name)
        self._populate_checkpoints()
        self._loss_graph.clear()
        self._samples.clear()
        self._restore_lora_state()

    def _save_lora_state(self) -> None:
        """Persist LoRA training config to project."""
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.lora_dataset_path = self._dataset.dataset_path or ""
            cfg.lora_output_name = self._output_name.text()
            cfg.lora_model_type = self._model_type.currentText()
            cfg.lora_rank = self._rank.currentText()
            cfg.lora_alpha = self._alpha.value()
            cfg.lora_steps = self._steps.value()
            cfg.lora_lr = self._lr.text()
            cfg.lora_scheduler = self._scheduler.currentText()
            cfg.lora_optimizer = self._optimizer.currentText()
            cfg.lora_precision = self._precision.currentText()
            cfg.lora_grad_ckpt = self._grad_ckpt.isChecked()
            cfg.lora_cache_latents = self._cache_latents.isChecked()
            cfg.lora_cache_te = self._cache_te.isChecked()
            cfg.lora_noise_offset = self._noise_offset.value()
            cfg.lora_min_snr = self._min_snr.value()
            cfg.lora_save_every = self._save_every.value()
            cfg.lora_sample_every = self._sample_every.value()
            cfg.lora_sample_prompt = self._sample_prompt.text()
            ckpt_idx = self._checkpoint_combo.currentIndex()
            cfg.lora_checkpoint_idx = ckpt_idx
            cfg.save(self.project_path)
        except Exception:
            pass

    def _restore_lora_state(self) -> None:
        """Restore LoRA training config from project."""
        self._restoring = True
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        ds = getattr(cfg, "lora_dataset_path", "") or ""
        if ds and Path(ds).is_dir():
            self._dataset.set_dataset_path(ds)
        self._output_name.setText(getattr(cfg, "lora_output_name", "my_lora") or "my_lora")
        mt = getattr(cfg, "lora_model_type", "")
        if mt:
            idx = self._model_type.findText(mt)
            if idx >= 0:
                self._model_type.setCurrentIndex(idx)
        rk = getattr(cfg, "lora_rank", "")
        if rk:
            idx = self._rank.findText(str(rk))
            if idx >= 0:
                self._rank.setCurrentIndex(idx)
        self._alpha.setValue(getattr(cfg, "lora_alpha", 16) or 16)
        self._steps.setValue(getattr(cfg, "lora_steps", 2000) or 2000)
        self._lr.setText(getattr(cfg, "lora_lr", "5e-5") or "5e-5")
        sched = getattr(cfg, "lora_scheduler", "")
        if sched:
            idx = self._scheduler.findText(sched)
            if idx >= 0:
                self._scheduler.setCurrentIndex(idx)
        opt = getattr(cfg, "lora_optimizer", "")
        if opt:
            idx = self._optimizer.findText(opt)
            if idx >= 0:
                self._optimizer.setCurrentIndex(idx)
        prec = getattr(cfg, "lora_precision", "")
        if prec:
            idx = self._precision.findText(prec)
            if idx >= 0:
                self._precision.setCurrentIndex(idx)
        self._grad_ckpt.setChecked(getattr(cfg, "lora_grad_ckpt", True))
        self._cache_latents.setChecked(getattr(cfg, "lora_cache_latents", True))
        self._cache_te.setChecked(getattr(cfg, "lora_cache_te", True))
        self._noise_offset.setValue(getattr(cfg, "lora_noise_offset", 0.0) or 0.0)
        self._min_snr.setValue(getattr(cfg, "lora_min_snr", 5.0) or 5.0)
        self._save_every.setValue(getattr(cfg, "lora_save_every", 500) or 500)
        self._sample_every.setValue(getattr(cfg, "lora_sample_every", 200) or 200)
        self._sample_prompt.setText(getattr(cfg, "lora_sample_prompt", "") or "")
        ckpt_idx = getattr(cfg, "lora_checkpoint_idx", 0) or 0
        if 0 <= ckpt_idx < self._checkpoint_combo.count():
            self._checkpoint_combo.setCurrentIndex(ckpt_idx)
        self._restoring = False
