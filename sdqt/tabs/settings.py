"""Settings tab -- global config and pipeline management."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.state import AppState
from sdqt.utils.file_dialog import get_existing_directory, get_open_filename
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.workers.base import BaseWorker

from .base import BaseTab

logger = logging.getLogger(__name__)


class PipelineLoadWorker(BaseWorker):
    """Load video pipelines on a background thread."""

    def __init__(self, state: AppState, method: str = "load_pipelines", parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._method = method

    def do_work(self) -> str:
        self.status.emit("Loading pipelines...")
        getattr(self._state, self._method)()
        return "Pipelines loaded."


class SDPipelineLoadWorker(BaseWorker):
    """Load SD image pipelines on a background thread."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    def do_work(self) -> str:
        self.status.emit("Loading image pipelines...")
        self._state.load_sd_pipelines()
        return "Image pipelines loaded."


class FluxPipelineLoadWorker(BaseWorker):
    """Load FLUX pipelines on a background thread."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    def do_work(self) -> str:
        self.status.emit("Loading FLUX pipelines...")
        self._state.load_flux_pipelines()
        return "FLUX pipelines loaded."


class ZImagePipelineLoadWorker(BaseWorker):
    """Load Z-Image Turbo pipelines on a background thread."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

    def do_work(self) -> str:
        self.status.emit("Loading Z-Image pipelines...")
        self._state.load_zimage_pipelines()
        return "Z-Image pipelines loaded."


class QwenDownloadWorker(BaseWorker):
    """Download Qwen model to a local directory on a background thread."""

    def __init__(self, local_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self._local_dir = local_dir

    def do_work(self) -> str:
        from huggingface_hub import snapshot_download

        model_id = "huihui-ai/Qwen2.5-7B-Instruct-abliterated"
        self.status.emit(f"Downloading {model_id}...")

        self._local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(self._local_dir),
            local_dir_use_symlinks=False,
        )
        self.status.emit("Download complete.")
        return "Qwen downloaded."


class SettingsTab(BaseTab):
    """Global settings and pipeline management."""

    sd_pipelines_loaded = Signal()
    flux_pipelines_loaded = Signal()
    zimage_pipelines_loaded = Signal()
    denoising_loop_changed = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: BaseWorker | None = None
        self._build_ui()
        # Apply saved audio device at startup
        saved_audio = getattr(state.global_config, "audio_device", "") or ""
        if saved_audio:
            from sdqt.widgets.video_player import set_audio_device_id
            set_audio_device_id(saved_audio)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)

        cfg = self.state.global_config

        # ── Appearance ──────────────────────────────────────────────
        appear_section = CollapsibleSection("Appearance")
        ui_row = QHBoxLayout()
        ui_row.addWidget(QLabel("UI Layout:"))
        self._ui_layout = QComboBox()
        self._ui_layout.addItems(["classic", "modern"])
        self._ui_layout.setCurrentText(cfg.ui_layout or "classic")
        self._ui_layout.currentTextChanged.connect(self._on_ui_layout_changed)
        ui_row.addWidget(self._ui_layout)
        ui_row.addStretch()
        appear_section.add_layout(ui_row)

        self._ui_layout_note = QLabel("")
        self._ui_layout_note.setWordWrap(True)
        self._ui_layout_note.setStyleSheet("padding: 4px; color: #e8a33d;")
        self._ui_layout_note.setVisible(False)
        appear_section.add_widget(self._ui_layout_note)

        layout.addWidget(appear_section)

        # ── Wan Video Model Paths ────────────────────────────────────
        self._path_fields: dict[str, QLineEdit] = {}
        wan_paths_section = CollapsibleSection("Wan Video Model Paths")
        for key in ("transformer", "transformer_2", "vae", "text_encoder", "image_encoder", "lora_dir"):
            # Label above the input so long paths get the full row width.
            wan_paths_section.add_widget(QLabel(f"{key}:"))
            row = QHBoxLayout()
            row.setSpacing(8)
            field = QLineEdit()
            field.setText(cfg.model_paths.get(key, ""))
            row.addWidget(field, 1)
            browse = QPushButton("Browse...")
            browse.setMinimumWidth(80)
            browse.clicked.connect(lambda checked, k=key, f=field: self._browse_path(k, f))
            row.addWidget(browse)
            wan_paths_section.add_layout(row)
            self._path_fields[key] = field
        layout.addWidget(wan_paths_section)

        # ── LTX Video Model Paths ───────────────────────────────────
        ltx_paths_section = CollapsibleSection("LTX Video Model Paths", collapsed=True)
        _ltx_keys = {
            "ltx_transformer": "Transformer",
            "ltx_vae": "VAE",
            "ltx_text_encoder": "Text Encoder (Gemma 3)",
            "ltx_embeddings_connector": "Embeddings Connector",
            "ltx_text_projection": "Text Projection",
            "ltx_lora_dir": "LoRA Dir",
            "ltx_spatial_upsampler": "Spatial Upscaler (2x)",
            "ltx_temporal_upsampler": "Temporal Upscaler (2x)",
        }
        for key, label in _ltx_keys.items():
            ltx_paths_section.add_widget(QLabel(f"{label}:"))
            row = QHBoxLayout()
            row.setSpacing(8)
            field = QLineEdit()
            field.setText(cfg.model_paths.get(key, ""))
            row.addWidget(field, 1)
            browse = QPushButton("Browse...")
            browse.setMinimumWidth(80)
            browse.clicked.connect(lambda checked, k=key, f=field: self._browse_path(k, f))
            row.addWidget(browse)
            ltx_paths_section.add_layout(row)
            self._path_fields[key] = field
        layout.addWidget(ltx_paths_section)

        # ── Image Model Paths ────────────────────────────────────────
        img_paths_section = CollapsibleSection("Image Model Paths", collapsed=True)
        self._img_path_fields: dict[str, QLineEdit] = {}
        for key in ("sd_checkpoint_dir", "sd_vae_dir", "sd_lora_dir", "face_models_dir"):
            img_paths_section.add_widget(QLabel(f"{key}:"))
            row = QHBoxLayout()
            row.setSpacing(8)
            field = QLineEdit()
            field.setText(cfg.model_paths.get(key, ""))
            row.addWidget(field, 1)
            browse = QPushButton("Browse...")
            browse.setMinimumWidth(80)
            browse.clicked.connect(lambda checked, k=key, f=field: self._browse_path(k, f))
            row.addWidget(browse)
            img_paths_section.add_layout(row)
            self._img_path_fields[key] = field
        layout.addWidget(img_paths_section)

        # ── FLUX Model Paths ───────────────────────────────────────
        flux_paths_section = CollapsibleSection("FLUX Model Paths", collapsed=True)
        self._flux_path_fields: dict[str, QLineEdit] = {}
        for key in ("flux_chroma_dir", "flux_fill_dir", "flux_lora_dir", "birefnet_dir", "triposr_dir"):
            flux_paths_section.add_widget(QLabel(f"{key}:"))
            row = QHBoxLayout()
            row.setSpacing(8)
            field = QLineEdit()
            field.setText(cfg.model_paths.get(key, ""))
            row.addWidget(field, 1)
            browse = QPushButton("Browse...")
            browse.setMinimumWidth(80)
            browse.clicked.connect(lambda checked, k=key, f=field: self._browse_path(k, f))
            row.addWidget(browse)
            flux_paths_section.add_layout(row)
            self._flux_path_fields[key] = field
        layout.addWidget(flux_paths_section)

        # ── Z-Image Model Paths ────────────────────────────────────
        zimage_paths_section = CollapsibleSection("Z-Image Model Paths", collapsed=True)
        self._zimage_path_fields: dict[str, QLineEdit] = {}
        zimage_paths_section.add_widget(QLabel("zimage_dir (file or dir):"))
        row = QHBoxLayout()
        row.setSpacing(8)
        field = QLineEdit()
        field.setPlaceholderText("Path to .safetensors file or diffusers directory")
        field.setText(cfg.model_paths.get("zimage_dir", ""))
        row.addWidget(field, 1)
        browse_file = QPushButton("File...")
        browse_file.setMinimumWidth(70)
        browse_file.clicked.connect(lambda: self._browse_zimage_file(field))
        row.addWidget(browse_file)
        browse_dir = QPushButton("Dir...")
        browse_dir.setMinimumWidth(70)
        browse_dir.clicked.connect(lambda: self._browse_zimage_dir(field))
        row.addWidget(browse_dir)
        zimage_paths_section.add_layout(row)
        self._zimage_path_fields["zimage_dir"] = field

        zimage_paths_section.add_widget(QLabel("zimage_lora_dir:"))
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        lora_field = QLineEdit()
        lora_field.setText(cfg.model_paths.get("zimage_lora_dir", ""))
        row2.addWidget(lora_field, 1)
        lora_browse = QPushButton("Browse...")
        lora_browse.setMinimumWidth(80)
        lora_browse.clicked.connect(lambda: self._browse_zimage_dir(lora_field))
        row2.addWidget(lora_browse)
        zimage_paths_section.add_layout(row2)
        self._zimage_path_fields["zimage_lora_dir"] = lora_field

        layout.addWidget(zimage_paths_section)

        # ── External UI Auto-Import ──────────────────────────────────
        ext_section = CollapsibleSection("Import Paths from Other UIs", collapsed=True)
        ext_desc = QLabel(
            "Point to your existing ComfyUI, Forge, or A1111 install to auto-detect "
            "checkpoint and LoRA directories. This sets the path fields above."
        )
        ext_desc.setWordWrap(True)
        ext_desc.setStyleSheet("color: #aaa;")
        ext_section.add_widget(ext_desc)

        ext_row = QHBoxLayout()
        ext_row.setSpacing(8)
        self._ext_ui_path = QLineEdit()
        self._ext_ui_path.setPlaceholderText("Path to ComfyUI / Forge / A1111 root folder...")
        ext_row.addWidget(self._ext_ui_path, 1)
        ext_browse = QPushButton("Browse...")
        ext_browse.setMinimumWidth(80)
        ext_browse.clicked.connect(lambda: self._browse_ext_ui())
        ext_row.addWidget(ext_browse)
        ext_apply = QPushButton("Auto-Detect")
        ext_apply.setMinimumWidth(100)
        ext_apply.setMinimumHeight(40)
        ext_apply.setToolTip(
            "Scan the selected folder for known model directory layouts\n"
            "(ComfyUI, Forge, A1111) and fill in the path fields."
        )
        ext_apply.clicked.connect(self._on_auto_detect_ext)
        ext_row.addWidget(ext_apply)
        ext_section.add_layout(ext_row)

        self._ext_status = QLabel("")
        self._ext_status.setWordWrap(True)
        ext_section.add_widget(self._ext_status)

        layout.addWidget(ext_section)

        # ── Hardware & Quality ───────────────────────────────────────
        hw_section = CollapsibleSection("Hardware & Quality Tier")

        # Hardware info banner
        hw = self.state.hardware_profile
        hw_info = QLabel(
            f"<b>{hw.gpu_name}</b> ({hw.gpu_vram_gb}GB VRAM) &nbsp;|&nbsp; "
            f"{hw.system_ram_gb}GB RAM &nbsp;|&nbsp; {hw.cpu_name}"
        )
        hw_info.setWordWrap(True)
        hw_info.setStyleSheet("padding: 6px; background: rgba(255,255,255,0.05); border-radius: 4px;")
        hw_section.add_widget(hw_info)

        # Quality tier selector
        tier_row = QHBoxLayout()
        tier_row.addWidget(QLabel("Quality Tier:"))
        self._quality_tier = QComboBox()
        self._quality_tier.addItems(["auto", "draft", "standard", "production", "ultra"])
        self._quality_tier.setCurrentText(cfg.quality_tier or "auto")
        self._quality_tier.currentTextChanged.connect(self._on_quality_tier_changed)
        tier_row.addWidget(self._quality_tier)
        tier_row.addStretch()
        hw_section.add_layout(tier_row)

        # Tier description + advisory
        active = self.state.quality_tier
        self._tier_info = QLabel(
            f"<b>{active.label}</b> -- {active.description}<br>"
            f"<i>{active.speed_hint} &nbsp;|&nbsp; {active.quality_hint}</i>"
        )
        self._tier_info.setWordWrap(True)
        self._tier_info.setStyleSheet("padding: 4px; color: #aaa;")
        hw_section.add_widget(self._tier_info)

        self._tier_advisory = QLabel("")
        self._tier_advisory.setWordWrap(True)
        self._tier_advisory.setStyleSheet("padding: 4px; color: #e8a33d;")
        self._tier_advisory.setVisible(False)
        hw_section.add_widget(self._tier_advisory)

        layout.addWidget(hw_section)

        # ── Performance ──────────────────────────────────────────────
        perf_section = CollapsibleSection("Performance")

        loop_row = QHBoxLayout()
        loop_row.addWidget(QLabel("Denoising Loop:"))
        self._denoising_loop = QComboBox()
        self._denoising_loop.addItems(["standard", "wan2gp"])
        self._denoising_loop.setCurrentText(cfg.denoising_loop or "standard")
        self._denoising_loop.currentTextChanged.connect(self.denoising_loop_changed.emit)
        loop_row.addWidget(self._denoising_loop)
        loop_row.addStretch()
        perf_section.add_layout(loop_row)

        # GPU Optimisation Profile (named wrapper around mmgp memory profiles)
        from supremediffusion.config.hardware_detect import (
            GPU_OPT_PROFILES, GPU_OPT_BY_KEY, gpu_opt_advisory,
        )
        gpu_opt_row = QHBoxLayout()
        gpu_opt_row.addWidget(QLabel("GPU Profile:"))
        self._gpu_opt_profile = QComboBox()
        # Populate with available profiles, marking recommended
        hw = self.state.hardware_profile
        current_mmgp = cfg.memory_profile
        current_opt_key = ""
        for opt in GPU_OPT_PROFILES:
            suffix = ""
            if opt.key == hw.recommended_gpu_opt:
                suffix = " (recommended)"
            self._gpu_opt_profile.addItem(f"{opt.label}{suffix}", opt.key)
            # Match current config to a profile by mmgp number + quantization
            if opt.mmgp_profile == current_mmgp and not current_opt_key:
                current_opt_key = opt.key
        # If no match found, default to recommended
        if not current_opt_key:
            current_opt_key = hw.recommended_gpu_opt
        # Set current selection
        for i in range(self._gpu_opt_profile.count()):
            if self._gpu_opt_profile.itemData(i) == current_opt_key:
                self._gpu_opt_profile.setCurrentIndex(i)
                break
        gpu_opt_row.addWidget(self._gpu_opt_profile)
        gpu_opt_row.addStretch()
        perf_section.add_layout(gpu_opt_row)

        # GPU profile description (updates on selection)
        opt_obj = GPU_OPT_BY_KEY.get(current_opt_key, GPU_OPT_PROFILES[1])
        self._gpu_opt_info = QLabel(
            f"<b>{opt_obj.label}</b> -- {opt_obj.description}<br>"
            f"<span style='color:#999;'>{opt_obj.details.replace(chr(10), '<br>')}</span>"
        )
        self._gpu_opt_info.setWordWrap(True)
        self._gpu_opt_info.setStyleSheet("padding: 4px; color: #aaa;")
        perf_section.add_widget(self._gpu_opt_info)

        # GPU profile advisory
        self._gpu_opt_advisory = QLabel("")
        self._gpu_opt_advisory.setWordWrap(True)
        self._gpu_opt_advisory.setStyleSheet("padding: 4px; color: #e8a33d;")
        self._gpu_opt_advisory.setVisible(False)
        perf_section.add_widget(self._gpu_opt_advisory)

        self._gpu_opt_profile.currentIndexChanged.connect(self._on_gpu_opt_changed)

        # ── Auto-tune ────────────────────────────────────────────────
        # Detect the local GPU/RAM and fill the performance widgets with
        # recommended values. Does NOT auto-save — the user reviews then
        # clicks Save Settings.
        # Note appears above the button so the result of the last run is
        # visible before re-running, not hidden below the action.
        self._detect_apply_note = QLabel("")
        self._detect_apply_note.setWordWrap(True)
        self._detect_apply_note.setStyleSheet("padding: 4px; color: #4fc3f7;")
        self._detect_apply_note.setVisible(False)
        perf_section.add_widget(self._detect_apply_note)

        autotune_row = QHBoxLayout()
        autotune_row.setSpacing(8)
        self._detect_apply_btn = QPushButton("Detect & Apply Recommended")
        self._detect_apply_btn.setMinimumHeight(44)
        self._detect_apply_btn.setToolTip(
            "Probe this machine's GPU VRAM / system RAM and set the\n"
            "performance options to a tuned recommendation. Review,\n"
            "then click Save Settings to persist."
        )
        self._detect_apply_btn.clicked.connect(self._on_detect_apply)
        autotune_row.addWidget(self._detect_apply_btn)
        autotune_row.addStretch()
        perf_section.add_layout(autotune_row)

        # Hidden spinbox to hold the actual mmgp value for save
        self._memory_profile = QSpinBox()
        self._memory_profile.setRange(1, 5)
        self._memory_profile.setValue(cfg.memory_profile)
        self._memory_profile.setVisible(False)
        perf_section.add_widget(self._memory_profile)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Attention:"))
        self._attention = QComboBox()
        # Pull modes from the upstream attention module so the dropdown only
        # shows backends that are actually installed and supported on this GPU
        # (sage/sage2/sage3 require triton + the right capability; flash needs
        # flash_attn). Names must match the strings the elif chain in
        # ``pay_attention()`` switches on — "sage_attn"/"flash_attn" do NOT
        # work and would crash with UnboundLocalError.
        try:
            from supremediffusion.models.ltx2.shared.attention import (
                get_supported_attention_modes,
            )
            modes = get_supported_attention_modes() or ["sdpa"]
        except Exception:
            modes = ["sdpa", "xformers"]
        self._attention.addItems(modes)
        # Migrate legacy values written by the old hardcoded dropdown.
        legacy_map = {"sage_attn": "sage", "flash_attn": "flash"}
        current = legacy_map.get(cfg.attention_mode, cfg.attention_mode)
        if current not in modes:
            current = "sdpa" if "sdpa" in modes else modes[0]
        self._attention.setCurrentText(current)
        row1.addWidget(self._attention)
        row1.addStretch()
        perf_section.add_layout(row1)

        # The quantization / precision / compile options are laid out as a
        # two-column grid (label, combo per cell) so each control gets its own
        # cell with breathing room instead of cramming two pairs per HBox.
        perf_grid = QGridLayout()
        perf_grid.setHorizontalSpacing(10)
        perf_grid.setVerticalSpacing(10)

        self._transformer_quant = QComboBox()
        self._transformer_quant.addItems(["int8", "fp8", "bf16", "none"])
        self._transformer_quant.setCurrentText(cfg.transformer_quantization or "int8")
        self._transformer_quant.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Transformer Quantization:"), 0, 0)
        perf_grid.addWidget(self._transformer_quant, 0, 1)

        self._transformer_dtype = QComboBox()
        self._transformer_dtype.addItems(["auto", "fp16", "bf16"])
        self._transformer_dtype.setCurrentText(cfg.transformer_dtype_policy or "auto")
        self._transformer_dtype.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Transformer Dtype:"), 0, 2)
        perf_grid.addWidget(self._transformer_dtype, 0, 3)

        self._mixed_precision = QComboBox()
        self._mixed_precision.addItems(["0", "1"])
        self._mixed_precision.setCurrentText(str(cfg.mixed_precision) if cfg.mixed_precision else "0")
        self._mixed_precision.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Mixed Precision:"), 1, 0)
        perf_grid.addWidget(self._mixed_precision, 1, 1)

        self._text_enc_quant = QComboBox()
        self._text_enc_quant.addItems(["int8", "fp8", "bf16", "none"])
        self._text_enc_quant.setCurrentText(cfg.text_encoder_quantization or "int8")
        self._text_enc_quant.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Text Encoder Quant:"), 1, 2)
        perf_grid.addWidget(self._text_enc_quant, 1, 3)

        self._lm_decoder = QComboBox()
        self._lm_decoder.addItems(["", "legacy", "cg", "vllm"])
        self._lm_decoder.setCurrentText(cfg.lm_decoder_engine or "")
        self._lm_decoder.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("LM Decoder Engine:"), 2, 0)
        perf_grid.addWidget(self._lm_decoder, 2, 1)

        self._vae_precision = QComboBox()
        self._vae_precision.addItems(["16", "32"])
        self._vae_precision.setCurrentText(str(cfg.vae_precision) if cfg.vae_precision else "16")
        self._vae_precision.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("VAE Precision:"), 2, 2)
        perf_grid.addWidget(self._vae_precision, 2, 3)

        self._compile_transformer = QComboBox()
        self._compile_transformer.addItems(["", "transformer"])
        self._compile_transformer.setCurrentText(cfg.compile_transformer or "")
        self._compile_transformer.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Compile Transformer:"), 3, 0)
        perf_grid.addWidget(self._compile_transformer, 3, 1)

        self._vae_tiling = QComboBox()
        self._vae_tiling.addItems(["Auto", "Disabled", "256", "128"])
        _vt_val = cfg.vae_tiling if hasattr(cfg, 'vae_tiling') else 0
        if isinstance(_vt_val, int) and 0 <= _vt_val <= 3:
            self._vae_tiling.setCurrentIndex(_vt_val)
        self._vae_tiling.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("VAE Tiling:"), 3, 2)
        perf_grid.addWidget(self._vae_tiling, 3, 3)

        self._boost = QComboBox()
        self._boost.addItems(["1", "2"])
        _boost_val = cfg.boost if hasattr(cfg, 'boost') else 1
        self._boost.setCurrentText(str(_boost_val) if _boost_val else "1")
        self._boost.setMinimumWidth(120)
        perf_grid.addWidget(QLabel("Boost:"), 4, 0)
        perf_grid.addWidget(self._boost, 4, 1)

        self._int8_kernels = QCheckBox("INT8 Kernels")
        self._int8_kernels.setChecked(cfg.enable_int8_kernels if hasattr(cfg, 'enable_int8_kernels') else False)
        perf_grid.addWidget(self._int8_kernels, 4, 2, 1, 2)

        perf_grid.setColumnStretch(4, 1)
        perf_section.add_layout(perf_grid)

        layout.addWidget(perf_section)

        # ── Memory Management ────────────────────────────────────────
        mem_section = CollapsibleSection("Memory Management", collapsed=True)

        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Preload in VRAM (MB):"))
        self._preload_vram = QSpinBox()
        self._preload_vram.setRange(0, 40000)
        self._preload_vram.setValue(cfg.preload_in_vram if hasattr(cfg, 'preload_in_vram') else 0)
        mem_row.addWidget(self._preload_vram)
        mem_row.addWidget(QLabel("Max Reserved LoRAs (MB):"))
        self._max_loras = QSpinBox()
        self._max_loras.setRange(-1, 10000)
        self._max_loras.setValue(cfg.max_reserved_loras if hasattr(cfg, 'max_reserved_loras') else 0)
        mem_row.addWidget(self._max_loras)
        mem_section.add_layout(mem_row)

        layout.addWidget(mem_section)

        # ── Color Drift Neutralization ───────────────────────────────
        # LTX/Wan output drifts warm; that drift becomes conditioning for
        # the next clip and compounds. This neutralizes extracted frames
        # before they re-enter the pipeline (send-to-img2vid, last-frame
        # capture, timeline extracts).
        neutral_section = CollapsibleSection(
            "Color Drift Neutralization", collapsed=True,
        )

        neut_enable_row = QHBoxLayout()
        self._neutralize_enabled = QCheckBox(
            "Neutralize extracted frames before send-to / next-clip conditioning"
        )
        self._neutralize_enabled.setChecked(
            bool(getattr(cfg, "neutralize_frames_enabled", False))
        )
        self._neutralize_enabled.setToolTip(
            "When enabled: every frame extracted from a video (send-to image "
            "tabs, capture last frame for extender, timeline frame export) is "
            "color-balanced before being saved as a temp PNG. Breaks the "
            "compound warm-bias chain in LTX/Wan i2v generations."
        )
        neut_enable_row.addWidget(self._neutralize_enabled)
        neut_enable_row.addStretch()
        neutral_section.add_layout(neut_enable_row)

        neut_ref_row = QHBoxLayout()
        neut_ref_row.setSpacing(8)
        neut_ref_row.addWidget(QLabel("Reference image:"))
        self._neutralize_ref = QLineEdit()
        self._neutralize_ref.setPlaceholderText(
            "Empty = gray-world (auto). Pick an anchor frame to preserve "
            "intentional warmth (sunset, candlelight)."
        )
        self._neutralize_ref.setText(
            getattr(cfg, "neutralize_reference_path", "") or ""
        )
        neut_ref_row.addWidget(self._neutralize_ref, 1)
        neut_ref_browse = QPushButton("Browse...")
        neut_ref_browse.setMinimumWidth(80)
        def _pick_neut_ref():
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self, "Select reference frame", "",
                "Images (*.png *.jpg *.jpeg *.webp)",
            )
            if path:
                self._neutralize_ref.setText(path)
        neut_ref_browse.clicked.connect(_pick_neut_ref)
        neut_ref_row.addWidget(neut_ref_browse)
        neutral_section.add_layout(neut_ref_row)

        neut_strength_row = QHBoxLayout()
        neut_strength_row.addWidget(QLabel("Strength:"))
        self._neutralize_strength = QDoubleSpinBox()
        self._neutralize_strength.setRange(0.0, 1.0)
        self._neutralize_strength.setSingleStep(0.05)
        self._neutralize_strength.setDecimals(2)
        self._neutralize_strength.setValue(
            float(getattr(cfg, "neutralize_strength", 1.0) or 1.0)
        )
        self._neutralize_strength.setToolTip(
            "0.00 = original frame. 1.00 = fully neutralized. "
            "Blend in between if 1.0 is too aggressive."
        )
        neut_strength_row.addWidget(self._neutralize_strength)
        neut_strength_row.addStretch()
        neutral_section.add_layout(neut_strength_row)

        layout.addWidget(neutral_section)

        # ── Output ───────────────────────────────────────────────────
        output_section = CollapsibleSection("Output", collapsed=True)

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        out_row.addWidget(QLabel("Video Codec:"))
        from sdqt.utils.codec import codec_labels, default_codec_label
        self._video_codec = QComboBox()
        self._video_codec.addItems(codec_labels())
        saved = cfg.video_output_codec or default_codec_label()
        idx = self._video_codec.findText(saved)
        if idx >= 0:
            self._video_codec.setCurrentIndex(idx)
        self._video_codec.setMinimumWidth(200)
        out_row.addWidget(self._video_codec)
        out_row.addWidget(QLabel("Container:"))
        self._video_container = QLineEdit()
        self._video_container.setText(cfg.video_container or "mp4")
        # Fixed-content field (mp4/mkv/mov) — kept compact on purpose.
        self._video_container.setMaximumWidth(100)
        out_row.addWidget(self._video_container)
        out_row.addStretch()
        output_section.add_layout(out_row)

        # Color Profile — global default for new projects. Existing projects
        # store their own per-project override on ProjectConfig.color_profile.
        cp_row = QHBoxLayout()
        cp_row.addWidget(QLabel("Color Profile:"))
        from supremediffusion.config.color_profile import (
            list_profiles as _list_profiles,
            DEFAULT_PROFILE_KEY as _DEFAULT_KEY,
        )
        self._color_profile = QComboBox()
        self._color_profile.setMinimumWidth(280)
        self._color_profile.setToolTip(
            "Default color profile for new projects. Existing projects keep "
            "their own per-project override."
        )
        for p in _list_profiles():
            self._color_profile.addItem(p.label, p.key)
        saved_key = cfg.default_color_profile or _DEFAULT_KEY
        cp_idx = self._color_profile.findData(saved_key)
        if cp_idx < 0:
            cp_idx = self._color_profile.findData(_DEFAULT_KEY)
        if cp_idx >= 0:
            self._color_profile.setCurrentIndex(cp_idx)
        cp_row.addWidget(self._color_profile)
        cp_row.addStretch()
        output_section.add_layout(cp_row)

        layout.addWidget(output_section)

        # ── Audio ────────────────────────────────────────────────────
        audio_section = CollapsibleSection("Audio Output")

        audio_row = QHBoxLayout()
        audio_row.setSpacing(6)
        audio_row.addWidget(QLabel("Device:"))
        self._audio_device = QComboBox()
        self._audio_device.setFixedWidth(300)
        self._populate_audio_devices()
        audio_row.addWidget(self._audio_device)
        refresh_audio = QPushButton("Refresh")
        refresh_audio.setFixedWidth(60)
        refresh_audio.clicked.connect(self._populate_audio_devices)
        audio_row.addWidget(refresh_audio)
        audio_row.addStretch()
        audio_section.add_layout(audio_row)

        layout.addWidget(audio_section)

        # ── Storage ──────────────────────────────────────────────────
        storage_section = CollapsibleSection("Storage", collapsed=True)

        stor_row = QHBoxLayout()
        stor_row.setSpacing(8)
        stor_row.addWidget(QLabel("Projects Root:"))
        self._projects_root = QLineEdit()
        self._projects_root.setText(cfg.projects_root or "")
        stor_row.addWidget(self._projects_root, 1)
        root_browse = QPushButton("Browse...")
        root_browse.setMinimumWidth(80)
        root_browse.clicked.connect(self._browse_projects_root)
        stor_row.addWidget(root_browse)
        storage_section.add_layout(stor_row)

        png_row = QHBoxLayout()
        png_row.setSpacing(8)
        png_row.addWidget(QLabel("PNG Library Dir:"))
        self._png_library_dir = QLineEdit()
        self._png_library_dir.setText(getattr(cfg, "png_library_dir", "") or "")
        png_row.addWidget(self._png_library_dir, 1)
        png_browse = QPushButton("Browse...")
        png_browse.setMinimumWidth(80)
        png_browse.clicked.connect(self._browse_png_library_dir)
        png_row.addWidget(png_browse)
        storage_section.add_layout(png_row)

        face_row = QHBoxLayout()
        face_row.setSpacing(8)
        face_row.addWidget(QLabel("Face Library Dir:"))
        self._face_library_dir = QLineEdit()
        self._face_library_dir.setText(getattr(cfg, "face_library_dir", "") or "")
        face_row.addWidget(self._face_library_dir, 1)
        face_browse = QPushButton("Browse...")
        face_browse.setMinimumWidth(80)
        face_browse.clicked.connect(self._browse_face_library_dir)
        face_row.addWidget(face_browse)
        storage_section.add_layout(face_row)

        layout.addWidget(storage_section)

        # ── Pipeline Controls ────────────────────────────────────────
        pipeline_group = QGroupBox("Pipeline")
        pl_layout = QVBoxLayout(pipeline_group)

        # Video model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Video Model:"))
        self._video_model_combo = QComboBox()
        self._video_model_combo.setMinimumWidth(220)
        self._video_model_combo.addItem("— Select a model —", userData="")
        self._video_model_combo.addItem("Wan 2.1 I2V 14B (480P)", userData="video_gen_21_480p")
        self._video_model_combo.addItem("Wan 2.1 I2V 14B (720P)", userData="video_gen_21_720p")
        self._video_model_combo.addItem("Wan 2.2 I2V 14B", userData="video_gen")
        self._video_model_combo.addItem("LTX 2.3 Dev (22B)", userData="ltx_video_dev")
        self._video_model_combo.addItem("LTX 2.3 Distilled (22B)", userData="ltx_video_distilled")
        self._video_model_combo.currentIndexChanged.connect(self._on_video_model_selected)
        model_row.addWidget(self._video_model_combo, 1)
        self._video_model_status = QLabel("")
        self._video_model_status.setStyleSheet("font-size: 13px; color: #bbb;")
        model_row.addWidget(self._video_model_status)
        pl_layout.addLayout(model_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        # The Video pipeline button is the primary load action on this page.
        self._load_btn = QPushButton()
        self._load_btn.setObjectName("primary")
        self._load_btn.clicked.connect(self._toggle_video_pipelines)
        btn_row.addWidget(self._load_btn)

        self._load_img_btn = QPushButton()
        self._load_img_btn.setMinimumHeight(44)
        self._load_img_btn.clicked.connect(self._toggle_sd_pipelines)
        btn_row.addWidget(self._load_img_btn)

        self._load_flux_btn = QPushButton()
        self._load_flux_btn.setMinimumHeight(44)
        self._load_flux_btn.clicked.connect(self._toggle_flux_pipelines)
        btn_row.addWidget(self._load_flux_btn)

        self._load_zimage_btn = QPushButton()
        self._load_zimage_btn.setMinimumHeight(44)
        self._load_zimage_btn.clicked.connect(self._toggle_zimage_pipelines)
        btn_row.addWidget(self._load_zimage_btn)

        self._unload_btn = QPushButton("Unload All")
        self._unload_btn.setMinimumHeight(44)
        self._unload_btn.clicked.connect(self._unload_pipelines)
        btn_row.addWidget(self._unload_btn)
        pl_layout.addLayout(btn_row)

        # Pipeline state surfaced in a colored banner so users notice it.
        self._pipeline_status = QLabel("")
        self._pipeline_status.setWordWrap(True)
        self._pipeline_status.setStyleSheet(
            "padding: 6px 8px; background: rgba(255,255,255,0.05); "
            "border-radius: 4px; color: #ccc;"
        )
        pl_layout.addWidget(self._pipeline_status)
        self._sync_pipeline_buttons()

        layout.addWidget(pipeline_group)

        # ── Plugin Manager ──────────────────────────────────────────
        self._build_plugin_manager(layout)

        # ── Installed Models ─────────────────────────────────────────
        models_section = CollapsibleSection("Installed Models", collapsed=True)

        self._models_table = QLabel("Click Refresh to scan models.")
        self._models_table.setWordWrap(True)
        models_section.add_widget(self._models_table)

        # Row 1: scan / download actions.
        models_scan_row = QHBoxLayout()
        models_scan_row.setSpacing(8)
        refresh_models = QPushButton("Refresh")
        refresh_models.setMinimumHeight(40)
        refresh_models.setMinimumWidth(120)
        refresh_models.clicked.connect(self._refresh_models_status)
        models_scan_row.addWidget(refresh_models)

        download_missing = QPushButton("Download Missing")
        download_missing.setMinimumHeight(40)
        download_missing.setToolTip(
            "Download all missing models for your VRAM tier.\n"
            "Models also auto-download when you first generate."
        )
        download_missing.clicked.connect(self._download_missing_models)
        models_scan_row.addWidget(download_missing)
        models_scan_row.addStretch()
        models_section.add_layout(models_scan_row)

        # Row 2: package installers.
        models_install_row = QHBoxLayout()
        models_install_row.setSpacing(8)
        install_wan = QPushButton("Install Wan Package")
        install_wan.setMinimumHeight(40)
        install_wan.setToolTip(
            "Download all required Wan models for the selected variant.\n"
            "Auto-selects the best quantization for your GPU."
        )
        install_wan.clicked.connect(self._install_wan_package)
        models_install_row.addWidget(install_wan)

        install_ltx = QPushButton("Install LTX Package")
        install_ltx.setMinimumHeight(40)
        install_ltx.setToolTip(
            "Download all required LTX 2.3 models (transformer, VAE, Gemma 3, connectors).\n"
            "Auto-selects the best quantization for your GPU."
        )
        install_ltx.clicked.connect(self._install_ltx_package)
        models_install_row.addWidget(install_ltx)
        models_install_row.addStretch()
        models_section.add_layout(models_install_row)

        # Row 3: Models Root path (label above input for long paths).
        models_section.add_widget(QLabel("Models Root:"))
        models_root_row = QHBoxLayout()
        models_root_row.setSpacing(8)
        self._models_root = QLineEdit()
        self._models_root.setText(getattr(cfg, "models_root", "") or "")
        models_root_row.addWidget(self._models_root, 1)
        models_root_browse = QPushButton("Browse...")
        models_root_browse.setMinimumWidth(80)
        models_root_browse.clicked.connect(self._browse_models_root)
        models_root_row.addWidget(models_root_browse)
        models_section.add_layout(models_root_row)

        layout.addWidget(models_section)

        # ── Qwen Local Download ─────────────────────────────────────
        qwen_section = CollapsibleSection("Qwen (Prompt Enhancer)", collapsed=True)

        qwen_info = QLabel(
            "Qwen is the built-in AI that enhances your prompts. "
            "Download it locally so it never contacts HuggingFace again."
        )
        qwen_info.setWordWrap(True)
        qwen_info.setStyleSheet("color: #aaa; font-size: 13px;")
        qwen_section.add_widget(qwen_info)

        qwen_row = QHBoxLayout()
        self._qwen_download_btn = QPushButton("Download Qwen Locally")
        self._qwen_download_btn.setToolTip(
            "Download Qwen 2.5 7B to your models directory.\n"
            "After this, prompt enhancement works fully offline."
        )
        self._qwen_download_btn.clicked.connect(self._download_qwen_locally)
        qwen_row.addWidget(self._qwen_download_btn)

        self._qwen_status = QLabel("")
        self._qwen_status.setStyleSheet("color: #aaa; font-size: 13px;")
        qwen_row.addWidget(self._qwen_status, 1)
        qwen_section.add_layout(qwen_row)

        # Check if already downloaded
        self._check_qwen_local_status()

        layout.addWidget(qwen_section)

        # ── Save Button ──────────────────────────────────────────────
        # Primary action for the page — styled by the global #primary rule.
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("primary")
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ── External UI auto-detect ─────────────────────────────────────

    def _browse_ext_ui(self) -> None:
        path = get_existing_directory(self, "Select ComfyUI / Forge / A1111 root")
        if path:
            self._ext_ui_path.setText(path)

    def _on_auto_detect_ext(self) -> None:
        """Auto-detect model paths from a ComfyUI, Forge, or A1111 install."""
        from pathlib import Path
        root = Path(self._ext_ui_path.text().strip())
        if not root.is_dir():
            self._ext_status.setText("Invalid directory.")
            return

        found: dict[str, str] = {}
        ui_name = "Unknown"

        # ComfyUI layout
        comfy_models = root / "models"
        if comfy_models.is_dir():
            ui_name = "ComfyUI"
            for sub, key in [
                ("checkpoints", "sd_checkpoint_dir"),
                ("loras", "sd_lora_dir"),
                ("vae", "sd_vae_dir"),
            ]:
                d = comfy_models / sub
                if d.is_dir():
                    found[key] = str(d)

        # A1111 / Forge layout
        a1111_models = root / "models"
        if (a1111_models / "Stable-diffusion").is_dir():
            ui_name = "A1111/Forge"
            mapping = {
                "Stable-diffusion": "sd_checkpoint_dir",
                "Lora": "sd_lora_dir",
                "VAE": "sd_vae_dir",
            }
            for sub, key in mapping.items():
                d = a1111_models / sub
                if d.is_dir():
                    found[key] = str(d)

        if not found:
            self._ext_status.setText(
                f"No known model layout detected in:\n{root}\n"
                "Expected ComfyUI (models/checkpoints) or A1111/Forge (models/Stable-diffusion)."
            )
            return

        # Apply to path fields
        applied = []
        for key, path_str in found.items():
            if key in self._img_path_fields:
                self._img_path_fields[key].setText(path_str)
                applied.append(f"{key} = {path_str}")
            elif key in self._path_fields:
                self._path_fields[key].setText(path_str)
                applied.append(f"{key} = {path_str}")

        self._ext_status.setText(
            f"Detected: {ui_name}\n" + "\n".join(applied)
        )
        self._ext_status.setStyleSheet("color: #4fc3f7;")
        logger.info("Auto-detected %s paths from %s: %s", ui_name, root, found)

    def _browse_path(self, key: str, field: QLineEdit) -> None:
        if key.endswith("_dir"):
            path = get_existing_directory(self, f"Select {key}")
        else:
            path, _ = get_open_filename(self, f"Select {key}")
        if path:
            field.setText(path)

    def _browse_zimage_file(self, field: QLineEdit) -> None:
        path, _ = get_open_filename(self, "Select Z-Image checkpoint (.safetensors)", filter="SafeTensors (*.safetensors)")
        if path:
            field.setText(path)

    def _browse_zimage_dir(self, field: QLineEdit) -> None:
        path = get_existing_directory(self, "Select Z-Image model directory")
        if path:
            field.setText(path)

    def _browse_projects_root(self) -> None:
        path = get_existing_directory(self, "Select Projects Root")
        if path:
            self._projects_root.setText(path)

    def _browse_png_library_dir(self) -> None:
        path = get_existing_directory(self, "Select PNG Library Directory")
        if path:
            self._png_library_dir.setText(path)

    def _browse_face_library_dir(self) -> None:
        path = get_existing_directory(self, "Select Face Library Directory")
        if path:
            self._face_library_dir.setText(path)

    def _populate_audio_devices(self) -> None:
        from PySide6.QtMultimedia import QMediaDevices
        saved = getattr(self.state.global_config, "audio_device", "") or ""
        self._audio_device.clear()
        self._audio_device.addItem("(System Default)", "")
        selected_idx = 0
        for dev in QMediaDevices.audioOutputs():
            dev_id = dev.id().data().decode()
            self._audio_device.addItem(dev.description(), dev_id)
            if saved and dev_id == saved:
                selected_idx = self._audio_device.count() - 1
        self._audio_device.setCurrentIndex(selected_idx)

    @Slot()
    def _on_quality_tier_changed(self, key: str) -> None:
        """Update UI when the quality tier dropdown changes."""
        advisory = self.state.set_quality_tier(key)
        active = self.state.quality_tier
        self._tier_info.setText(
            f"<b>{active.label}</b> -- {active.description}<br>"
            f"<i>{active.speed_hint} &nbsp;|&nbsp; {active.quality_hint}</i>"
        )
        if advisory:
            self._tier_advisory.setText(advisory)
            self._tier_advisory.setVisible(True)
        else:
            self._tier_advisory.setVisible(False)

    @Slot(str)
    def _on_ui_layout_changed(self, value: str) -> None:
        """Show restart notice when UI layout changes."""
        current = self.state.global_config.ui_layout or "classic"
        if value != current:
            self._ui_layout_note.setText(
                f"Switching to '{value}' layout requires a restart. "
                "Save settings, then close and reopen the app."
            )
            self._ui_layout_note.setVisible(True)
        else:
            self._ui_layout_note.setVisible(False)

    @Slot()
    def _on_gpu_opt_changed(self, index: int) -> None:
        """Update UI when the GPU optimisation profile dropdown changes."""
        from supremediffusion.config.hardware_detect import (
            GPU_OPT_BY_KEY, gpu_opt_advisory,
        )
        key = self._gpu_opt_profile.itemData(index)
        if not key:
            return
        opt = GPU_OPT_BY_KEY.get(key)
        if opt is None:
            return

        # Update description panel
        self._gpu_opt_info.setText(
            f"<b>{opt.label}</b> -- {opt.description}<br>"
            f"<span style='color:#999;'>{opt.details.replace(chr(10), '<br>')}</span>"
        )

        # Update the hidden mmgp spinbox so save picks up the right value
        self._memory_profile.setValue(opt.mmgp_profile)

        # Show advisory if applicable
        hw = self.state.hardware_profile
        advisory = gpu_opt_advisory(key, hw)
        if advisory:
            self._gpu_opt_advisory.setText(advisory)
            self._gpu_opt_advisory.setVisible(True)
        else:
            self._gpu_opt_advisory.setVisible(False)

    @Slot()
    def _on_detect_apply(self) -> None:
        """Probe hardware and fill the performance widgets with recommendations.

        Does NOT save — the user reviews the populated values and clicks
        "Save Settings" to persist them.
        """
        from supremediffusion.utils.hardware import (
            detect_hardware, recommend_settings,
        )

        hw = detect_hardware()
        rec = recommend_settings(hw)

        # Memory profile — set the hidden spinbox and sync the GPU Profile
        # dropdown to the first named profile that maps to this mmgp number.
        mmgp = int(rec["memory_profile"])
        self._memory_profile.setValue(mmgp)
        for i in range(self._gpu_opt_profile.count()):
            key = self._gpu_opt_profile.itemData(i)
            try:
                from supremediffusion.config.hardware_detect import GPU_OPT_BY_KEY
                opt = GPU_OPT_BY_KEY.get(key)
            except Exception:
                opt = None
            if opt is not None and opt.mmgp_profile == mmgp:
                # blockSignals so _on_gpu_opt_changed doesn't clobber the
                # mmgp value we just set (matching profiles share it anyway).
                self._gpu_opt_profile.blockSignals(True)
                self._gpu_opt_profile.setCurrentIndex(i)
                self._gpu_opt_profile.blockSignals(False)
                # Refresh the description panel manually since signals were off.
                self._on_gpu_opt_changed(i)
                break

        # Attention — map the recommended backend name onto whatever the
        # combo actually contains (mirrors the legacy_map + availability
        # fallback used when the combo was first populated).
        rec_attn = rec["attention_mode"]
        legacy_map = {"sage_attn": "sage", "flash_attn": "flash"}
        rec_attn = legacy_map.get(rec_attn, rec_attn)
        modes = [self._attention.itemText(i) for i in range(self._attention.count())]
        if rec_attn not in modes:
            # Walk the preference ladder down to whatever is installed.
            for fallback in ("sage", "xformers", "sdpa"):
                if fallback in modes:
                    rec_attn = fallback
                    break
            else:
                rec_attn = modes[0] if modes else "sdpa"
        self._attention.setCurrentText(rec_attn)

        # Quantization
        self._transformer_quant.setCurrentText(rec["transformer_quantization"])
        self._text_enc_quant.setCurrentText(rec["text_encoder_quantization"])
        # Transformer dtype: bf16 quant pairs with a bf16 dtype policy, else auto.
        self._transformer_dtype.setCurrentText(
            "bf16" if rec["transformer_quantization"] == "bf16" else "auto"
        )

        # Precision / mixed precision
        self._vae_precision.setCurrentText(rec["vae_precision"])
        self._mixed_precision.setCurrentText(rec["mixed_precision"])

        # VAE tiling — combo items are "Auto"/"Disabled"/"256"/"128".
        vt_idx = self._vae_tiling.findText(rec["vae_tiling"])
        if vt_idx >= 0:
            self._vae_tiling.setCurrentIndex(vt_idx)

        # Boost
        self._boost.setCurrentText(str(rec["boost"]))

        # Preload VRAM
        self._preload_vram.setValue(int(rec["preload_in_vram"]))

        vram = hw.get("vram_gb") or 0.0
        gpu = hw.get("gpu_name") or "GPU"
        self._detect_apply_note.setText(
            f"Detected {gpu} ({vram} GB VRAM, {hw.get('ram_gb', 0.0)} GB RAM) — "
            f"applied recommended performance settings. "
            f"Review above, then click Save Settings to persist."
        )
        self._detect_apply_note.setVisible(True)

    def _save_settings(self) -> None:
        cfg = self.state.global_config

        # Video model paths
        for key, field in self._path_fields.items():
            cfg.model_paths[key] = field.text()

        # Image model paths
        for key, field in self._img_path_fields.items():
            cfg.model_paths[key] = field.text()

        # FLUX model paths
        for key, field in self._flux_path_fields.items():
            cfg.model_paths[key] = field.text()

        # Z-Image model paths
        for key, field in self._zimage_path_fields.items():
            cfg.model_paths[key] = field.text()

        # Quality tier
        cfg.quality_tier = self._quality_tier.currentText()

        # UI Layout
        cfg.ui_layout = self._ui_layout.currentText()

        # Performance
        cfg.denoising_loop = self._denoising_loop.currentText()
        cfg.memory_profile = self._memory_profile.value()
        cfg.attention_mode = self._attention.currentText()
        cfg.transformer_quantization = self._transformer_quant.currentText()
        cfg.transformer_dtype_policy = self._transformer_dtype.currentText()
        cfg.mixed_precision = self._mixed_precision.currentText()
        cfg.text_encoder_quantization = self._text_enc_quant.currentText()
        cfg.lm_decoder_engine = self._lm_decoder.currentText()
        cfg.vae_precision = self._vae_precision.currentText()
        cfg.compile_transformer = self._compile_transformer.currentText()
        cfg.vae_tiling = self._vae_tiling.currentIndex()
        cfg.boost = int(self._boost.currentText()) if self._boost.currentText() else 1
        cfg.enable_int8_kernels = self._int8_kernels.isChecked()

        # Memory
        cfg.preload_in_vram = self._preload_vram.value()
        cfg.max_reserved_loras = self._max_loras.value()

        # Color Drift Neutralization
        cfg.neutralize_frames_enabled = self._neutralize_enabled.isChecked()
        cfg.neutralize_reference_path = self._neutralize_ref.text().strip()
        cfg.neutralize_strength = float(self._neutralize_strength.value())

        # Output
        cfg.video_output_codec = self._video_codec.currentText()
        cfg.video_container = self._video_container.text()
        # Color profile — saved as the slug (currentData), not the label.
        cp_data = self._color_profile.currentData()
        if cp_data:
            cfg.default_color_profile = cp_data
        # Drop the codec args cache so a new lookup picks up the new default.
        try:
            from sdqt.utils import codec as _codec_mod
            _codec_mod._configured_cache = None  # type: ignore[attr-defined]
        except Exception:
            pass

        # Storage
        cfg.projects_root = self._projects_root.text()
        cfg.png_library_dir = self._png_library_dir.text()
        cfg.face_library_dir = self._face_library_dir.text()
        cfg.models_root = self._models_root.text()

        # Audio
        cfg.audio_device = self._audio_device.currentData() or ""
        from sdqt.widgets.video_player import set_audio_device_id
        set_audio_device_id(cfg.audio_device)

        cfg.save()
        self._pipeline_status.setText("Settings saved.")

    # -- Plugin Manager -------------------------------------------------------

    def _build_plugin_manager(self, parent_layout) -> None:
        """Build the plugin manager UI section."""
        from plugins.manifest import load_manifest, ManifestError, set_app_version

        # Ensure app version is set so min_app_version checks pass
        try:
            toml_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
            if toml_path.is_file():
                for line in toml_path.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("version"):
                        ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                        set_app_version(ver)
                        break
        except Exception:
            pass

        section = CollapsibleSection("Plugin Manager", collapsed=False)

        # Scan all plugin directories
        app_root = Path(__file__).resolve().parent.parent.parent
        plugins_dir = app_root / "plugins"
        self._plugin_dir = plugins_dir
        self._plugin_rows: dict[str, dict] = {}  # key -> {manifest, checkbox, labels}

        disabled = self._load_disabled_plugins()

        if not plugins_dir.is_dir():
            section.add_widget(QLabel("No plugins directory found."))
            parent_layout.addWidget(section)
            return

        found_any = False
        for candidate in sorted(plugins_dir.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name.startswith(("_", ".")):
                continue

            manifest_path = candidate / "plugin.json"
            if not manifest_path.is_file():
                continue

            try:
                manifest = load_manifest(candidate)
            except ManifestError:
                continue

            found_any = True
            key = candidate.name  # directory name is the plugin key

            row = QHBoxLayout()
            row.setSpacing(8)

            # Enable/disable checkbox
            cb = QCheckBox()
            cb.setChecked(key not in disabled)
            cb.setToolTip("Enable or disable this plugin (requires restart)")
            row.addWidget(cb)

            # Plugin info
            icon = manifest.icon or ""
            name_label = QLabel(
                f"<b>{icon} {manifest.name}</b> "
                f"<span style='color:#888;'>v{manifest.version}</span>"
            )
            name_label.setMinimumWidth(180)
            row.addWidget(name_label)

            author_label = QLabel(manifest.author or "")
            author_label.setStyleSheet("color: #999; font-size: 13px;")
            author_label.setMinimumWidth(90)
            row.addWidget(author_label)

            desc_label = QLabel(manifest.description or "")
            desc_label.setStyleSheet("color: #aaa; font-size: 13px;")
            desc_label.setWordWrap(True)
            row.addWidget(desc_label, 1)

            # Status indicator
            status = QLabel("")
            status.setMinimumWidth(70)
            status.setStyleSheet("font-size: 13px;")
            if key in disabled:
                status.setText("Disabled")
                status.setStyleSheet("color: #e85050; font-size: 13px;")
                name_label.setStyleSheet("color: #666;")
            else:
                status.setText("Active")
                status.setStyleSheet("color: #50e850; font-size: 13px;")
            row.addWidget(status)

            # Wire the toggle
            cb.toggled.connect(
                lambda checked, k=key, s=status, nl=name_label: self._on_plugin_toggled(k, checked, s, nl)
            )

            self._plugin_rows[key] = {
                "manifest": manifest,
                "checkbox": cb,
                "status": status,
                "name_label": name_label,
            }

            section.add_layout(row)

        if not found_any:
            section.add_widget(QLabel("No plugins found in plugins/ directory."))

        # Restart hint
        hint = QLabel(
            "<i>Changes take effect after restarting the app.</i>"
        )
        hint.setStyleSheet("color: #999; font-size: 13px; padding-top: 4px;")
        section.add_widget(hint)

        parent_layout.addWidget(section)

    def _on_plugin_toggled(self, key: str, enabled: bool, status_label, name_label) -> None:
        """Handle a plugin enable/disable toggle."""
        disabled = self._load_disabled_plugins()
        if enabled:
            disabled.discard(key)
            status_label.setText("Active")
            status_label.setStyleSheet("color: #50e850; font-size: 13px;")
            name_label.setStyleSheet("")
        else:
            disabled.add(key)
            status_label.setText("Disabled")
            status_label.setStyleSheet("color: #e85050; font-size: 13px;")
            name_label.setStyleSheet("color: #666;")
        self._save_disabled_plugins(disabled)

    def _load_disabled_plugins(self) -> set[str]:
        """Load the set of disabled plugin directory names."""
        import json
        disabled_path = self._get_disabled_path()
        if disabled_path.is_file():
            try:
                with open(disabled_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return set(data)
            except Exception:
                pass
        return set()

    def _save_disabled_plugins(self, disabled: set[str]) -> None:
        """Persist the set of disabled plugin directory names."""
        import json
        disabled_path = self._get_disabled_path()
        disabled_path.parent.mkdir(parents=True, exist_ok=True)
        with open(disabled_path, "w", encoding="utf-8") as f:
            json.dump(sorted(disabled), f, indent=2)

    def _get_disabled_path(self) -> Path:
        app_root = Path(__file__).resolve().parent.parent.parent
        return app_root / "plugins" / "disabled.json"

    # -- Qwen local download --------------------------------------------------

    def _get_qwen_local_dir(self) -> Path:
        return Path(self.state.global_config.models_root) / "qwen2.5_7b_abliterated"

    def _check_qwen_local_status(self) -> None:
        """Update the status label based on whether Qwen is downloaded locally."""
        local_dir = self._get_qwen_local_dir()
        if local_dir.is_dir() and any(local_dir.iterdir()):
            self._qwen_status.setText(f"Downloaded locally: {local_dir}")
            self._qwen_status.setStyleSheet("color: #50e850; font-size: 13px;")
            self._qwen_download_btn.setText("Re-download Qwen")
        else:
            self._qwen_status.setText("Not downloaded — using HuggingFace cache")
            self._qwen_status.setStyleSheet("color: #e8a33d; font-size: 13px;")

    @Slot()
    def _download_qwen_locally(self) -> None:
        """Download Qwen model to the local models directory."""
        self._qwen_download_btn.setEnabled(False)
        self._qwen_status.setText("Downloading... this may take a few minutes.")
        self._qwen_status.setStyleSheet("color: #e8a33d; font-size: 13px;")

        worker = QwenDownloadWorker(self._get_qwen_local_dir(), parent=self)
        worker.finished_ok.connect(self._on_qwen_downloaded)
        worker.error.connect(self._on_qwen_download_error)
        worker.status.connect(lambda s: self._qwen_status.setText(s))
        worker.finished.connect(worker.deleteLater)
        self._qwen_worker = worker
        worker.start()

    def _on_qwen_downloaded(self, msg) -> None:
        self._qwen_download_btn.setEnabled(True)
        self._check_qwen_local_status()

    def _on_qwen_download_error(self, msg: str) -> None:
        self._qwen_download_btn.setEnabled(True)
        self._qwen_status.setText(f"Download failed: {msg}")
        self._qwen_status.setStyleSheet("color: #e85050; font-size: 13px;")

    @Slot()
    # -- Pipeline button sync ------------------------------------------------

    def _sync_pipeline_buttons(self) -> None:
        """Update button labels to reflect current pipeline state."""
        s = self.state
        self._load_btn.setText(
            "Unload Video" if s.pipelines_loaded else "Load Video Pipelines")
        self._load_img_btn.setText(
            "Unload Image" if s.sd_pipelines_loaded else "Load Image Pipelines")
        self._load_flux_btn.setText(
            "Unload FLUX" if s.flux_pipelines_loaded else "Load FLUX Pipelines")
        self._load_zimage_btn.setText(
            "Unload Z-Image" if s.zimage_pipelines_loaded else "Load Z-Image Pipelines")

        parts = []
        if s.pipelines_loaded:
            parts.append("Video")
        if s.sd_pipelines_loaded:
            parts.append("Image (SD)")
        if s.flux_pipelines_loaded:
            parts.append("FLUX")
        if s.zimage_pipelines_loaded:
            parts.append("Z-Image")
        self._pipeline_status.setText(
            f"Loaded: {', '.join(parts)}" if parts else "No pipelines loaded")

    # -- Video model selector -----------------------------------------------

    def _on_video_model_selected(self, index: int) -> None:
        """Auto-fill model path fields when a video model is selected."""
        feature_key = self._video_model_combo.currentData()
        if not feature_key:
            self._video_model_status.setText("")
            return

        registry = self.state.model_registry
        fm = registry.get_feature_models(feature_key)
        if fm is None:
            self._video_model_status.setText("Unknown model set")
            self._video_model_status.setStyleSheet("color: #e85050; font-size: 13px;")
            return

        filled = []
        missing = []
        for model in fm.models:
            if not model.config_key:
                continue
            local_path = registry.get_model_local_path(model)
            if registry._is_model_present(model):
                # Auto-fill the path field
                path_str = str(local_path)
                if model.config_key in self._path_fields:
                    self._path_fields[model.config_key].setText(path_str)
                filled.append(model.config_key)
            else:
                missing.append(model.name)

        # Clear transformer_2 if this model set doesn't use it
        has_t2 = any(m.config_key == "transformer_2" for m in fm.models)
        if not has_t2 and "transformer_2" in self._path_fields:
            self._path_fields["transformer_2"].setText("")

        if missing:
            self._video_model_status.setText(
                f"Paths set ({len(filled)}). Missing: {', '.join(missing)}")
            self._video_model_status.setStyleSheet("color: #e8a33d; font-size: 13px;")
        else:
            self._video_model_status.setText(
                f"All paths set ({len(filled)} components found)")
            self._video_model_status.setStyleSheet("color: #50e850; font-size: 13px;")

    # -- Toggle handlers ----------------------------------------------------

    def _toggle_video_pipelines(self) -> None:
        if self.state.pipelines_loaded:
            self.state.unload_video_pipelines()
            self.state._force_gc()
            self._sync_pipeline_buttons()
        else:
            self._save_settings()
            self._load_btn.setEnabled(False)
            # Pick load method based on selected video model
            feature = self._video_model_combo.currentData() or ""
            if feature.startswith("ltx_"):
                method = "load_ltx_pipelines"
                label = "LTX"
            else:
                method = "load_pipelines"
                label = "Wan"
            self._pipeline_status.setText(f"Loading {label} video pipelines...")
            worker = PipelineLoadWorker(self.state, method, parent=self)
            worker.finished_ok.connect(self._on_loaded)
            worker.error.connect(self._on_load_error)
            worker.status.connect(lambda s: self._pipeline_status.setText(s))
            worker.finished.connect(worker.deleteLater)
            self._worker = worker
            worker.start()

    def _toggle_sd_pipelines(self) -> None:
        if self.state.sd_pipelines_loaded:
            self.state.unload_sd_pipelines()
            self.state._force_gc()
            self._sync_pipeline_buttons()
        else:
            self._save_settings()
            self._load_img_btn.setEnabled(False)
            self._pipeline_status.setText("Loading image pipelines...")
            worker = SDPipelineLoadWorker(self.state, parent=self)
            worker.finished_ok.connect(self._on_sd_loaded)
            worker.error.connect(self._on_sd_load_error)
            worker.status.connect(lambda s: self._pipeline_status.setText(s))
            worker.finished.connect(worker.deleteLater)
            self._worker = worker
            worker.start()

    def _toggle_flux_pipelines(self) -> None:
        if self.state.flux_pipelines_loaded:
            self.state.unload_flux_pipelines()
            self.state._force_gc()
            self._sync_pipeline_buttons()
        else:
            self._save_settings()
            self._load_flux_btn.setEnabled(False)
            self._pipeline_status.setText("Loading FLUX pipelines...")
            worker = FluxPipelineLoadWorker(self.state, parent=self)
            worker.finished_ok.connect(self._on_flux_loaded)
            worker.error.connect(self._on_flux_load_error)
            worker.status.connect(lambda s: self._pipeline_status.setText(s))
            worker.finished.connect(worker.deleteLater)
            self._worker = worker
            worker.start()

    def _toggle_zimage_pipelines(self) -> None:
        if self.state.zimage_pipelines_loaded:
            self.state.unload_zimage_pipelines()
            self.state._force_gc()
            self._sync_pipeline_buttons()
        else:
            self._save_settings()
            self._load_zimage_btn.setEnabled(False)
            self._pipeline_status.setText("Loading Z-Image pipelines...")
            worker = ZImagePipelineLoadWorker(self.state, parent=self)
            worker.finished_ok.connect(self._on_zimage_loaded)
            worker.error.connect(self._on_zimage_load_error)
            worker.status.connect(lambda s: self._pipeline_status.setText(s))
            worker.finished.connect(worker.deleteLater)
            self._worker = worker
            worker.start()

    # -- Load callbacks -----------------------------------------------------

    def _on_loaded(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._sync_pipeline_buttons()

    def _on_load_error(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._pipeline_status.setText(f"Error: {msg}")
        self._sync_pipeline_buttons()

    def _on_sd_loaded(self, msg: str) -> None:
        self._load_img_btn.setEnabled(True)
        self._sync_pipeline_buttons()
        self.sd_pipelines_loaded.emit()

    def _on_sd_load_error(self, msg: str) -> None:
        self._load_img_btn.setEnabled(True)
        self._pipeline_status.setText(f"Error: {msg}")
        self._sync_pipeline_buttons()

    def _on_flux_loaded(self, msg: str) -> None:
        self._load_flux_btn.setEnabled(True)
        self._sync_pipeline_buttons()
        self.flux_pipelines_loaded.emit()

    def _on_flux_load_error(self, msg: str) -> None:
        self._load_flux_btn.setEnabled(True)
        self._pipeline_status.setText(f"Error: {msg}")
        self._sync_pipeline_buttons()

    def _on_zimage_loaded(self, msg: str) -> None:
        self._load_zimage_btn.setEnabled(True)
        self._sync_pipeline_buttons()
        self.zimage_pipelines_loaded.emit()

    def _on_zimage_load_error(self, msg: str) -> None:
        self._load_zimage_btn.setEnabled(True)
        self._pipeline_status.setText(f"Error: {msg}")
        self._sync_pipeline_buttons()

    @Slot()
    def _unload_pipelines(self) -> None:
        self.state.unload_pipelines()
        self._sync_pipeline_buttons()

    def _browse_models_root(self) -> None:
        path = get_existing_directory(self, "Select Models Root Directory")
        if path:
            self._models_root.setText(path)

    @Slot()
    def _install_wan_package(self) -> None:
        """Download all models for the currently selected Wan variant."""
        from sdqt.models.manager import check_and_prompt_download
        feature = self._video_model_combo.currentData() or ""
        if not feature or feature.startswith("ltx_"):
            feature = "video_gen"
        registry = self.state.model_registry
        ok = check_and_prompt_download(
            feature, registry, self,
            on_progress=lambda f, d: self._pipeline_status.setText(d),
        )
        self._refresh_models_status()
        if ok:
            self._pipeline_status.setText("Wan package installed.")

    def _install_ltx_package(self) -> None:
        """Download all models for the selected LTX variant."""
        from sdqt.models.manager import check_and_prompt_download
        feature = self._video_model_combo.currentData() or ""
        if not feature.startswith("ltx_"):
            feature = "ltx_video_distilled"
        registry = self.state.model_registry
        ok = check_and_prompt_download(
            feature, registry, self,
            on_progress=lambda f, d: self._pipeline_status.setText(d),
        )
        self._refresh_models_status()
        if ok:
            self._pipeline_status.setText("LTX package installed.")

    def _download_missing_models(self) -> None:
        """Prompt to download all missing models across all features."""
        from sdqt.models.manager import check_and_prompt_download
        registry = self.state.model_registry
        for feature_key in registry.all_features():
            missing = registry.get_missing_models(feature_key)
            if missing:
                ok = check_and_prompt_download(
                    feature_key, registry, self,
                    on_progress=lambda f, d: self._pipeline_status.setText(d),
                )
                if not ok:
                    break  # User cancelled
        self._refresh_models_status()
        self._pipeline_status.setText("Model check complete.")

    def _refresh_models_status(self) -> None:
        """Scan model registry and update the installed models display."""
        try:
            registry = self.state.model_registry
            statuses = registry.get_installed_status()
            if not statuses:
                self._models_table.setText("No model features configured.")
                return
            lines = []
            for s in statuses:
                icon = "\u2705" if s["missing"] == 0 else "\u274c"
                lines.append(
                    f"{icon}  <b>{s['label']}</b>: "
                    f"{s['installed']}/{s['total']} installed "
                    f"({s['total_size']}) — {s['status']}"
                )
            self._models_table.setText("<br>".join(lines))
        except Exception as exc:
            self._models_table.setText(f"Error scanning models: {exc}")
