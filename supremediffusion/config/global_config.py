"""App-wide configuration for Supreme Diffusion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import ClassVar

from .defaults import APP_ROOT, GLOBAL_DEFAULTS


def _default_model_paths() -> dict[str, str]:
    return dict(GLOBAL_DEFAULTS["model_paths"])


@dataclass
class GlobalConfig:
    """Application-wide settings persisted as JSON."""

    model_paths: dict[str, str] = field(default_factory=_default_model_paths)
    attention_mode: str = GLOBAL_DEFAULTS["attention_mode"]
    memory_profile: int = GLOBAL_DEFAULTS["memory_profile"]
    text_encoder_quantization: str = GLOBAL_DEFAULTS["text_encoder_quantization"]
    vae_precision: str = GLOBAL_DEFAULTS["vae_precision"]
    enable_int8_kernels: bool = GLOBAL_DEFAULTS["enable_int8_kernels"]
    # Performance settings (matching wan2gp Performance tab)
    transformer_quantization: str = GLOBAL_DEFAULTS["transformer_quantization"]
    transformer_dtype_policy: str = GLOBAL_DEFAULTS["transformer_dtype_policy"]
    mixed_precision: str = GLOBAL_DEFAULTS["mixed_precision"]
    lm_decoder_engine: str = GLOBAL_DEFAULTS["lm_decoder_engine"]
    compile_transformer: str = GLOBAL_DEFAULTS["compile_transformer"]
    vae_tiling: int = GLOBAL_DEFAULTS["vae_tiling"]
    boost: int = GLOBAL_DEFAULTS["boost"]
    preload_in_vram: int = GLOBAL_DEFAULTS["preload_in_vram"]
    max_reserved_loras: int = GLOBAL_DEFAULTS["max_reserved_loras"]
    denoising_loop: str = GLOBAL_DEFAULTS["denoising_loop"]
    video_output_codec: str = GLOBAL_DEFAULTS["video_output_codec"]
    video_container: str = GLOBAL_DEFAULTS["video_container"]
    default_color_profile: str = GLOBAL_DEFAULTS["default_color_profile"]
    projects_root: str = GLOBAL_DEFAULTS["projects_root"]
    png_library_dir: str = GLOBAL_DEFAULTS["png_library_dir"]
    face_library_dir: str = GLOBAL_DEFAULTS["face_library_dir"]
    characters_dir: str = GLOBAL_DEFAULTS["characters_dir"]
    meshes_dir: str = GLOBAL_DEFAULTS["meshes_dir"]
    models_root: str = GLOBAL_DEFAULTS["models_root"]
    audio_device: str = GLOBAL_DEFAULTS["audio_device"]
    last_project: str = GLOBAL_DEFAULTS["last_project"]
    last_parent_tab_idx: int = GLOBAL_DEFAULTS["last_parent_tab_idx"]
    last_video_tab_idx: int = GLOBAL_DEFAULTS["last_video_tab_idx"]
    quality_tier: str = GLOBAL_DEFAULTS["quality_tier"]
    ui_layout: str = GLOBAL_DEFAULTS["ui_layout"]
    last_sidebar_key: str = GLOBAL_DEFAULTS["last_sidebar_key"]
    tl_clip_zoom: int = GLOBAL_DEFAULTS["tl_clip_zoom"]
    last_export_dir: str = ""
    batch_generate_settings: dict = field(default_factory=dict)
    sequential_process_settings: dict = field(default_factory=dict)
    neutralize_frames_enabled: bool = GLOBAL_DEFAULTS["neutralize_frames_enabled"]
    neutralize_reference_path: str = GLOBAL_DEFAULTS["neutralize_reference_path"]
    neutralize_strength: float = GLOBAL_DEFAULTS["neutralize_strength"]

    DEFAULT_PATH: ClassVar[Path] = APP_ROOT / "config.json"

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise to JSON. Returns the path written."""
        target = Path(path) if path else self.DEFAULT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> GlobalConfig:
        """Load from JSON. Returns defaults when the file is missing.

        Merges saved model_paths with defaults so newly added keys
        (e.g. sd_checkpoint_dir) appear even in older config files.
        """
        target = Path(path) if path else cls.DEFAULT_PATH
        if not target.exists():
            return cls()
        data = json.loads(target.read_text(encoding="utf-8"))
        # Merge model_paths: defaults first, then saved values on top
        if "model_paths" in data:
            merged = dict(GLOBAL_DEFAULTS["model_paths"])
            merged.update(data["model_paths"])
            data["model_paths"] = merged
        # Backfill empty model_paths from ~/.supremediffusion/config.json
        # (shared config from Gradio app or manual setup)
        shared_cfg = Path.home() / ".supremediffusion" / "config.json"
        if shared_cfg.exists() and shared_cfg != target:
            try:
                shared = json.loads(shared_cfg.read_text(encoding="utf-8"))
                shared_paths = shared.get("model_paths", {})
                for k, v in shared_paths.items():
                    if v and not data.get("model_paths", {}).get(k):
                        data.setdefault("model_paths", {})[k] = v
            except Exception:
                pass
        # Backfill empty strings with defaults for fields that have non-empty defaults
        # (handles pre-existing configs saved before defaults were set)
        for key in ("png_library_dir", "face_library_dir", "characters_dir", "meshes_dir"):
            if key not in data or not data[key]:
                data[key] = GLOBAL_DEFAULTS.get(key, "")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
