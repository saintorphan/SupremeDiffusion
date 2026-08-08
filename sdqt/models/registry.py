"""Model registry — inventory of all models, presence checks, tier selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """A single downloadable model component."""

    name: str                          # Human-readable name
    repo_id: str                       # HuggingFace repo (e.g. "Wan-AI/Wan2.1-I2V-14B-480P")
    subfolder: str = ""                # Subfolder within repo (e.g. "transformer")
    filename: str = ""                 # Single file to download (for direct files)
    url: str = ""                      # Direct download URL (non-HF)
    size_bytes: int = 0                # Approximate download size
    config_key: str = ""               # GlobalConfig.model_paths key to update after download
    local_subdir: str = ""             # Subdirectory under models_root to store in
    min_tier: int = 0                  # Minimum VRAM tier required
    max_tier: int = 4                  # Maximum VRAM tier (above this, a better variant exists)
    variant: str = ""                  # Variant label (e.g. "int8", "bf16")
    min_arch: str = ""                 # Minimum GPU arch (e.g. "sm_89" for Ada/nvfp4)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def size_display(self) -> str:
        gb = self.size_gb
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        mb = self.size_bytes / (1024 ** 2)
        return f"{mb:.0f} MB"


@dataclass
class FeatureModels:
    """Models required for a specific feature."""

    feature: str                       # Feature key (e.g. "video_gen")
    label: str                         # Human-readable label
    models: list[ModelEntry] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(m.size_bytes for m in self.models)

    @property
    def total_size_display(self) -> str:
        gb = self.total_size_bytes / (1024 ** 3)
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        mb = self.total_size_bytes / (1024 ** 2)
        return f"{mb:.0f} MB"


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _wan21_480p_models() -> list[ModelEntry]:
    """Wan 2.1 I2V 14B 480P models."""
    return [
        ModelEntry(
            name="Wan 2.1 Transformer (int8)",
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="transformer",
            size_bytes=int(14.5 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.1-480p/transformer",
            min_tier=0, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="Wan 2.1 Transformer (bf16)",
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="transformer",
            size_bytes=int(28.0 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.1-480p/transformer_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
        ModelEntry(
            name="Wan 2.1 Text Encoder",
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="text_encoder",
            size_bytes=int(2.5 * 1024**3),
            config_key="text_encoder",
            local_subdir="wan/2.1-480p/text_encoder",
        ),
        ModelEntry(
            name="Wan 2.1 VAE",
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="vae",
            size_bytes=int(335 * 1024**2),
            config_key="vae",
            local_subdir="wan/2.1-480p/vae",
        ),
        ModelEntry(
            name="Wan CLIP Image Encoder",
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="image_encoder",
            size_bytes=int(600 * 1024**2),
            config_key="image_encoder",
            local_subdir="wan/image_encoder",
        ),
    ]


def _wan21_720p_models() -> list[ModelEntry]:
    """Wan 2.1 I2V 14B 720P models."""
    return [
        ModelEntry(
            name="Wan 2.1 720P Transformer (int8)",
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="transformer",
            size_bytes=int(14.5 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.1-720p/transformer",
            min_tier=0, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="Wan 2.1 720P Transformer (bf16)",
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="transformer",
            size_bytes=int(28.0 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.1-720p/transformer_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
        ModelEntry(
            name="Wan 2.1 720P Text Encoder",
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="text_encoder",
            size_bytes=int(2.5 * 1024**3),
            config_key="text_encoder",
            local_subdir="wan/2.1-720p/text_encoder",
        ),
        ModelEntry(
            name="Wan 2.1 720P VAE",
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="vae",
            size_bytes=int(335 * 1024**2),
            config_key="vae",
            local_subdir="wan/2.1-720p/vae",
        ),
        ModelEntry(
            name="Wan CLIP Image Encoder",
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            subfolder="image_encoder",
            size_bytes=int(600 * 1024**2),
            config_key="image_encoder",
            local_subdir="wan/image_encoder",
        ),
    ]


def _wan22_models() -> list[ModelEntry]:
    """Wan 2.2 I2V A14B models (MoE dual-expert)."""
    return [
        ModelEntry(
            name="Wan 2.2 Transformer (int8)",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="transformer",
            size_bytes=int(14.5 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.2/transformer",
            min_tier=0, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="Wan 2.2 Transformer 2 (int8)",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="transformer_2",
            size_bytes=int(14.5 * 1024**3),
            config_key="transformer_2",
            local_subdir="wan/2.2/transformer_2",
            min_tier=0, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="Wan 2.2 Transformer (bf16)",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="transformer",
            size_bytes=int(28.0 * 1024**3),
            config_key="transformer",
            local_subdir="wan/2.2/transformer_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
        ModelEntry(
            name="Wan 2.2 Transformer 2 (bf16)",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="transformer_2",
            size_bytes=int(28.0 * 1024**3),
            config_key="transformer_2",
            local_subdir="wan/2.2/transformer_2_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
        ModelEntry(
            name="Wan 2.2 Text Encoder",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="text_encoder",
            size_bytes=int(2.5 * 1024**3),
            config_key="text_encoder",
            local_subdir="wan/2.2/text_encoder",
        ),
        ModelEntry(
            name="Wan 2.2 VAE",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="vae",
            size_bytes=int(335 * 1024**2),
            config_key="vae",
            local_subdir="wan/2.2/vae",
        ),
        ModelEntry(
            name="Wan CLIP Image Encoder",
            repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            subfolder="image_encoder",
            size_bytes=int(600 * 1024**2),
            config_key="image_encoder",
            local_subdir="wan/image_encoder",
        ),
    ]


def _lipsync_models() -> list[ModelEntry]:
    """LatentSync lip sync models."""
    return [
        ModelEntry(
            name="LatentSync UNet",
            repo_id="chunyu-li/LatentSync",
            filename="latentsync_unet.pt",
            size_bytes=int(2.0 * 1024**3),
            local_subdir="lipsync",
        ),
        ModelEntry(
            name="Whisper Tiny",
            repo_id="openai/whisper-tiny",
            size_bytes=int(151 * 1024**2),
            local_subdir="lipsync/whisper_tiny",
        ),
        ModelEntry(
            name="SD VAE ft-MSE",
            repo_id="stabilityai/sd-vae-ft-mse",
            size_bytes=int(335 * 1024**2),
            local_subdir="lipsync/sd_vae",
        ),
    ]


def _musetalk_models() -> list[ModelEntry]:
    """MuseTalk lip sync models."""
    return [
        ModelEntry(
            name="MuseTalk UNet",
            repo_id="TMElyralab/MuseTalk",
            filename="pytorch_model.bin",
            subfolder="musetalk",
            size_bytes=int(800 * 1024**2),
            local_subdir="musetalk",
        ),
        ModelEntry(
            name="MuseTalk VAE",
            repo_id="TMElyralab/MuseTalk",
            filename="musetalk.json",
            subfolder="musetalk",
            size_bytes=int(400 * 1024**2),
            local_subdir="musetalk",
        ),
        ModelEntry(
            name="MuseTalk Whisper",
            repo_id="openai/whisper-tiny",
            size_bytes=int(600 * 1024**2),
            local_subdir="musetalk/whisper",
        ),
    ]


def _face_swap_models() -> list[ModelEntry]:
    """Face swap models (InsightFace + inswapper)."""
    return [
        ModelEntry(
            name="InsightFace buffalo_l",
            repo_id="public-data/insightface",
            subfolder="models/buffalo_l",
            size_bytes=int(350 * 1024**2),
            local_subdir="face/buffalo_l",
            config_key="face_models_dir",
        ),
        ModelEntry(
            name="inswapper_128",
            repo_id="ashleykleynhans/inswapper",
            filename="inswapper_128.onnx",
            size_bytes=int(400 * 1024**2),
            local_subdir="face",
            config_key="face_models_dir",
        ),
    ]


def _adetailer_models() -> list[ModelEntry]:
    """ADetailer YOLOv8 face detector (optional — falls back to InsightFace).

    Downloads to ``models_root/face/face_yolov8s.pt`` which is the same dir
    ADetailer resolves via ``face_models_dir`` (== models_root/face). No
    config_key so presence is checked directly against the file, not the
    (already-populated) face dir.
    """
    return [
        ModelEntry(
            name="ADetailer YOLOv8s face",
            repo_id="Bingsu/adetailer",
            filename="face_yolov8s.pt",
            size_bytes=int(22 * 1024**2),
            local_subdir="face",
        ),
    ]


def _vace_multitalk_models() -> list[ModelEntry]:
    """VACE MultiTalk models.

    NOTE: The VACE MultiTalk pipeline uses Wan2GP-format int8 quantized
    checkpoints (not HF diffusers repos).  These must be downloaded
    manually from MeiGen-AI/MeiGen-MultiTalk or quantized locally.
    The registry entries below are placeholders so the Settings page
    can show their status; auto-download is not yet supported for these.
    """
    return [
        # Placeholder — these require Wan2GP-format quantized checkpoints
        # that are not available via standard HF diffusers repos.
        # Users must place .safetensors files manually in the ckpts dir.
        ModelEntry(
            name="Wav2Vec2",
            repo_id="facebook/wav2vec2-base-960h",
            size_bytes=int(380 * 1024**2),
            local_subdir="vace_multitalk/wav2vec2",
        ),
    ]


def _flux_models() -> list[ModelEntry]:
    """FLUX GGUF + supporting models — tier-aware quantization."""
    return [
        # Q4_K_S for low VRAM (8-16 GB)
        ModelEntry(
            name="FLUX Transformer (Q4_K_S GGUF)",
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q4_K_S.gguf",
            size_bytes=int(6.8 * 1024**3),
            config_key="flux_gguf_dir",
            local_subdir="flux",
            min_tier=0, max_tier=2,
            variant="Q4_K_S",
        ),
        # Q8_0 for mid VRAM (16-24 GB)
        ModelEntry(
            name="FLUX Transformer (Q8_0 GGUF)",
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q8_0.gguf",
            size_bytes=int(12.3 * 1024**3),
            config_key="flux_gguf_dir",
            local_subdir="flux",
            min_tier=3, max_tier=3,
            variant="Q8_0",
        ),
        # Full precision for high VRAM (24+ GB)
        ModelEntry(
            name="FLUX Transformer (F16 GGUF)",
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-F16.gguf",
            size_bytes=int(23.8 * 1024**3),
            config_key="flux_gguf_dir",
            local_subdir="flux",
            min_tier=4, max_tier=4,
            variant="F16",
        ),
        ModelEntry(
            name="FLUX VAE (ae)",
            repo_id="camenduru/FLUX.1-dev-ungated",
            subfolder="vae",
            size_bytes=int(335 * 1024**2),
            local_subdir="flux/vae",
        ),
        ModelEntry(
            name="T5 XXL (fp8)",
            repo_id="comfyanonymous/flux_text_encoders",
            filename="t5xxl_fp8_e4m3fn.safetensors",
            size_bytes=int(4.9 * 1024**3),
            local_subdir="flux/text_encoder",
        ),
        ModelEntry(
            name="CLIP-L",
            repo_id="openai/clip-vit-large-patch14",
            size_bytes=int(890 * 1024**2),
            local_subdir="flux/clip",
        ),
    ]


def _zimage_models() -> list[ModelEntry]:
    """Z-Image Turbo models."""
    return [
        ModelEntry(
            name="Z-Image Turbo",
            repo_id="",  # User-provided path typically
            url="",
            size_bytes=int(12.0 * 1024**3),
            config_key="zimage_dir",
            local_subdir="zimage",
        ),
    ]


def _prompt_enhance_models() -> list[ModelEntry]:
    """Qwen 3.5 4B for prompt enhancement and chat assistant."""
    return [
        ModelEntry(
            name="Qwen 3.5 4B (BF16)",
            repo_id="Qwen/Qwen3.5-4B",
            size_bytes=int(5.0 * 1024**3),
            local_subdir="qwen3.5_4b",
        ),
    ]


def _qwen_vl_models() -> list[ModelEntry]:
    """Qwen 2.5 VL 3B Instruct for image captioning (QwenAlyzer, LoRA captions)."""
    return [
        ModelEntry(
            name="Qwen 2.5 VL 3B Instruct",
            repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
            size_bytes=int(6.0 * 1024**3),
            local_subdir="qwen_vl_3b",
        ),
    ]


def _orpheus_tts_models() -> list[ModelEntry]:
    """Orpheus TTS 3B BF16 model (unsloth ungated mirror)."""
    return [
        ModelEntry(
            name="Orpheus 3B (BF16)",
            repo_id="unsloth/orpheus-3b-0.1-ft",
            size_bytes=int(6.0 * 1024**3),
            local_subdir="orpheus_tts",
        ),
    ]


def _openvoice_models() -> list[ModelEntry]:
    """OpenVoice V2 models."""
    return [
        ModelEntry(
            name="OpenVoice V2",
            repo_id="myshell-ai/OpenVoiceV2",
            size_bytes=int(500 * 1024**2),
            local_subdir="openvoice",
        ),
    ]


def _rvc_models() -> list[ModelEntry]:
    """RVC pretrained models (rvc + fairseq-fixed auto-downloads its own base models)."""
    return []


def _3d_modeling_models() -> list[ModelEntry]:
    """BiRefNet background removal + TripoSR 3D mesh generation."""
    return [
        ModelEntry(
            name="BiRefNet (Background Removal)",
            repo_id="ZhengPeng7/BiRefNet",
            size_bytes=int(350 * 1024**2),
            config_key="birefnet_dir",
            local_subdir="birefnet",
        ),
        ModelEntry(
            name="TripoSR (3D Mesh Generation)",
            repo_id="stabilityai/TripoSR",
            size_bytes=int(3.0 * 1024**3),
            config_key="triposr_dir",
            local_subdir="triposr",
        ),
    ]


def _body_double_models() -> list[ModelEntry]:
    """ControlNet OpenPose + IP-Adapter for Body Double."""
    return [
        ModelEntry(
            name="ControlNet OpenPose",
            repo_id="lllyasviel/control_v11p_sd15_openpose",
            size_bytes=int(1.4 * 1024**3),
            local_subdir="body_double/controlnet_openpose",
        ),
        ModelEntry(
            name="IP-Adapter Plus",
            repo_id="h94/IP-Adapter",
            filename="ip-adapter-plus_sd15.safetensors",
            subfolder="models",
            size_bytes=int(100 * 1024**2),
            local_subdir="body_double/ip_adapter",
        ),
        ModelEntry(
            name="CLIP ViT-H (IP-Adapter)",
            repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            size_bytes=int(2.0 * 1024**3),
            local_subdir="body_double/clip_vit_h",
        ),
    ]


def _ltx23_control_lora_models() -> list[ModelEntry]:
    """LTX 2.3 IC-LoRA for control video conditioning."""
    return [
        ModelEntry(
            name="LTX 2.3 IC-LoRA Union Control",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
            size_bytes=int(654 * 1024**2),
            local_subdir="ltx/2.3/loras",
            config_key="ltx_lora_dir",
        ),
    ]


def _ltx23_upsampler_models() -> list[ModelEntry]:
    """LTX 2.3 spatial and temporal upsampler models."""
    return [
        ModelEntry(
            name="LTX 2.3 Spatial Upscaler (2x)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            size_bytes=int(996 * 1024**2),
            config_key="ltx_spatial_upsampler",
            local_subdir="ltx/2.3/spatial_upscaler",
        ),
        ModelEntry(
            name="LTX 2.3 Temporal Upscaler (2x)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
            size_bytes=int(262 * 1024**2),
            config_key="ltx_temporal_upsampler",
            local_subdir="ltx/2.3/temporal_upscaler",
        ),
    ]


def _ltx23_shared_models() -> list[ModelEntry]:
    """LTX 2.3 shared components (VAE, connectors, text encoder)."""
    return [
        ModelEntry(
            name="LTX 2.3 VAE",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b_vae.safetensors",
            size_bytes=int(1.45 * 1024**3),
            config_key="ltx_vae",
            local_subdir="ltx/2.3/vae",
        ),
        ModelEntry(
            name="LTX 2.3 Embeddings Connector",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b_embeddings_connector.safetensors",
            size_bytes=int(4.03 * 1024**3),
            config_key="ltx_embeddings_connector",
            local_subdir="ltx/2.3/embeddings_connector",
        ),
        ModelEntry(
            name="LTX 2.3 Text Embedding Projection",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b_text_embedding_projection.safetensors",
            size_bytes=int(2.31 * 1024**3),
            config_key="ltx_text_projection",
            local_subdir="ltx/2.3/text_projection",
        ),
        ModelEntry(
            name="Gemma 3 12B Text Encoder",
            repo_id="DeepBeepMeep/LTX-2",
            subfolder="gemma-3-12b-it-qat-q4_0-unquantized",
            size_bytes=int(24.0 * 1024**3),
            config_key="ltx_text_encoder",
            local_subdir="ltx/gemma3_12b",
            min_tier=3, max_tier=4,
            variant="bf16",
        ),
        ModelEntry(
            name="Gemma 3 12B Text Encoder (int8)",
            repo_id="DeepBeepMeep/LTX-2",
            subfolder="gemma-3-12b-it-qat-q4_0-unquantized",
            size_bytes=int(12.0 * 1024**3),
            config_key="ltx_text_encoder",
            local_subdir="ltx/gemma3_12b_int8",
            min_tier=0, max_tier=2,
            variant="int8",
        ),
    ]


def _ltx23_dev_models() -> list[ModelEntry]:
    """LTX 2.3 Dev 22B models (full quality, 20-26 steps)."""
    return [
        ModelEntry(
            name="LTX 2.3 Dev Transformer (int8)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-dev_diffusion_model_quanto_int8.safetensors",
            size_bytes=int(19.4 * 1024**3),
            config_key="ltx_transformer",
            local_subdir="ltx/2.3/dev_transformer_int8",
            min_tier=1, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="LTX 2.3 Dev Transformer (nvfp4)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-dev-nvfp4_diffusion_model.safetensors",
            size_bytes=int(13.5 * 1024**3),
            config_key="ltx_transformer",
            local_subdir="ltx/2.3/dev_transformer_nvfp4",
            min_tier=2, max_tier=3,
            variant="nvfp4",
            min_arch="sm_89",
        ),
        ModelEntry(
            name="LTX 2.3 Dev Transformer (bf16)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-dev_diffusion_model.safetensors",
            size_bytes=int(38.0 * 1024**3),
            config_key="ltx_transformer",
            local_subdir="ltx/2.3/dev_transformer_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
    ] + _ltx23_shared_models() + _ltx23_upsampler_models() + _ltx23_control_lora_models()


def _ltx23_distilled_models() -> list[ModelEntry]:
    """LTX 2.3 Distilled 22B models (fast, 4-8 steps, no CFG)."""
    return [
        ModelEntry(
            name="LTX 2.3 Distilled Transformer (int8)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-distilled-1.1_diffusion_model_quanto_bf16_int8.safetensors",
            size_bytes=int(19.4 * 1024**3),
            config_key="ltx_transformer",
            local_subdir="ltx/2.3/distilled_transformer_int8",
            min_tier=1, max_tier=3,
            variant="int8",
        ),
        ModelEntry(
            name="LTX 2.3 Distilled Transformer (bf16)",
            repo_id="DeepBeepMeep/LTX-2",
            filename="ltx-2.3-22b-distilled-1.1_diffusion_model_bf16.safetensors",
            size_bytes=int(38.0 * 1024**3),
            config_key="ltx_transformer",
            local_subdir="ltx/2.3/distilled_transformer_bf16",
            min_tier=4, max_tier=4,
            variant="bf16",
        ),
    ] + _ltx23_shared_models() + _ltx23_upsampler_models() + _ltx23_control_lora_models()


# ---------------------------------------------------------------------------
# Feature → model mapping
# ---------------------------------------------------------------------------

FEATURE_MODELS: dict[str, FeatureModels] = {
    "video_gen_21_480p": FeatureModels(
        feature="video_gen_21_480p",
        label="Wan 2.1 I2V 14B (480P)",
        models=_wan21_480p_models(),
    ),
    "video_gen_21_720p": FeatureModels(
        feature="video_gen_21_720p",
        label="Wan 2.1 I2V 14B (720P)",
        models=_wan21_720p_models(),
    ),
    "video_gen": FeatureModels(
        feature="video_gen",
        label="Wan 2.2 I2V 14B",
        models=_wan22_models(),
    ),
    "ltx_video_dev": FeatureModels(
        feature="ltx_video_dev",
        label="LTX 2.3 Dev (22B)",
        models=_ltx23_dev_models(),
    ),
    "ltx_video_distilled": FeatureModels(
        feature="ltx_video_distilled",
        label="LTX 2.3 Distilled (22B)",
        models=_ltx23_distilled_models(),
    ),
    "lipsync": FeatureModels(
        feature="lipsync",
        label="Lip Sync (LatentSync)",
        models=_lipsync_models(),
    ),
    "musetalk": FeatureModels(
        feature="musetalk",
        label="Lip Sync (MuseTalk)",
        models=_musetalk_models(),
    ),
    "face_swap": FeatureModels(
        feature="face_swap",
        label="Face Swap",
        models=_face_swap_models(),
    ),
    "adetailer": FeatureModels(
        feature="adetailer",
        label="ADetailer Face Model (YOLOv8s)",
        models=_adetailer_models(),
    ),
    "vace_multitalk": FeatureModels(
        feature="vace_multitalk",
        label="VACE MultiTalk",
        models=_vace_multitalk_models(),
    ),
    "flux": FeatureModels(
        feature="flux",
        label="FLUX Image Generation",
        models=_flux_models(),
    ),
    "zimage": FeatureModels(
        feature="zimage",
        label="Z-Image Turbo",
        models=_zimage_models(),
    ),
    "prompt_enhance": FeatureModels(
        feature="prompt_enhance",
        label="Qwen 3.5 4B Assistant",
        models=_prompt_enhance_models(),
    ),
    "qwen_vl": FeatureModels(
        feature="qwen_vl",
        label="Qwen 2.5 VL 3B (Image Captioning)",
        models=_qwen_vl_models(),
    ),
    "body_double": FeatureModels(
        feature="body_double",
        label="Body Double (ControlNet + IP-Adapter)",
        models=_body_double_models(),
    ),
    "orpheus_tts": FeatureModels(
        feature="orpheus_tts",
        label="Orpheus TTS (3B FP16)",
        models=_orpheus_tts_models(),
    ),
    "openvoice": FeatureModels(
        feature="openvoice",
        label="OpenVoice V2",
        models=_openvoice_models(),
    ),
    "rvc": FeatureModels(
        feature="rvc",
        label="RVC Voice Conversion",
        models=_rvc_models(),
    ),
    "3d_modeling": FeatureModels(
        feature="3d_modeling",
        label="3D Modeling (BiRefNet + TripoSR)",
        models=_3d_modeling_models(),
    ),
    "auto_segment": FeatureModels(
        feature="auto_segment",
        label="Magic Mask (BiRefNet)",
        models=[
            ModelEntry(
                name="BiRefNet",
                repo_id="ZhengPeng7/BiRefNet",
                size_bytes=int(350 * 1024**2),
                local_subdir="birefnet",
            ),
        ],
    ),
    # image_gen (SD/SDXL) is user-provided — no auto-download
}


def _detect_vram_tier() -> int:
    """Detect VRAM tier from available GPU memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        if vram_gb < 8:
            return 0
        elif vram_gb < 12:
            return 1
        elif vram_gb < 16:
            return 2
        elif vram_gb < 24:
            return 3
        else:
            return 4
    except Exception:
        return 1  # Safe default


def _detect_gpu_arch() -> str:
    """Detect GPU compute capability as 'sm_XY' string (e.g. 'sm_89' for Ada)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        props = torch.cuda.get_device_properties(0)
        return f"sm_{props.major}{props.minor}"
    except Exception:
        return ""


class ModelRegistry:
    """Central model inventory with presence checks and tier-aware selection."""

    def __init__(self, models_root: str | Path, config_model_paths: dict[str, str]) -> None:
        self._models_root = Path(models_root)
        self._config_paths = config_model_paths
        self._vram_tier = _detect_vram_tier()
        self._gpu_arch = _detect_gpu_arch()
        logger.info(
            "ModelRegistry: models_root=%s  vram_tier=%d  gpu_arch=%s",
            self._models_root, self._vram_tier, self._gpu_arch or "(none)",
        )

    @property
    def models_root(self) -> Path:
        return self._models_root

    @property
    def vram_tier(self) -> int:
        return self._vram_tier

    def get_feature_models(self, feature: str) -> FeatureModels | None:
        """Get the model set for a feature, filtered to the user's VRAM tier."""
        fm = FEATURE_MODELS.get(feature)
        if fm is None:
            return None
        # Filter models to those appropriate for this VRAM tier and GPU arch
        filtered = [
            m for m in fm.models
            if m.min_tier <= self._vram_tier <= m.max_tier
            and (not m.min_arch or (self._gpu_arch and self._gpu_arch >= m.min_arch))
        ]
        return FeatureModels(
            feature=fm.feature,
            label=fm.label,
            models=filtered,
        )

    def get_missing_models(self, feature: str) -> list[ModelEntry]:
        """Return models needed for a feature that aren't present locally."""
        fm = self.get_feature_models(feature)
        if fm is None:
            return []
        missing = []
        for model in fm.models:
            if not self._is_model_present(model):
                missing.append(model)
        return missing

    def _is_model_present(self, model: ModelEntry) -> bool:
        """Check if a model is already downloaded or configured."""
        # First check if GlobalConfig already has a valid path for this model
        if model.config_key:
            configured = self._config_paths.get(model.config_key, "")
            if configured:
                p = Path(configured)
                if p.is_file():
                    logger.debug("Model '%s' found via config_key '%s' → %s", model.name, model.config_key, configured)
                    return True
                elif p.is_dir() and any(p.iterdir()):
                    logger.debug("Model '%s' found via config_key '%s' → %s", model.name, model.config_key, configured)
                    return True
            elif configured:
                logger.warning("Model '%s' config_key '%s' path does not exist: %s", model.name, model.config_key, configured)

        # Check models_root
        if model.local_subdir:
            local_dir = self._models_root / model.local_subdir
            if model.filename:
                found = (local_dir / model.filename).is_file()
            else:
                # Directory-based model — check if dir has actual model files
                # (not just .cache dirs or READMEs from failed downloads)
                _MODEL_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf"}
                found = local_dir.is_dir() and any(
                    f for f in local_dir.rglob("*")
                    if f.is_file() and f.suffix.lower() in _MODEL_EXTS
                )
            if not found:
                logger.debug("Model '%s' not found at %s", model.name, local_dir)
            return found
        return False

    def get_model_local_path(self, model: ModelEntry) -> Path:
        """Return the expected local path for a model."""
        base = self._models_root / model.local_subdir
        if model.filename:
            return base / model.filename
        return base

    def all_features(self) -> list[str]:
        """Return all known feature keys."""
        return list(FEATURE_MODELS.keys())

    def get_installed_status(self) -> list[dict]:
        """Return installation status for all features (for Settings UI)."""
        result = []
        for key, fm in FEATURE_MODELS.items():
            tier_models = self.get_feature_models(key)
            if not tier_models or not tier_models.models:
                continue
            missing = self.get_missing_models(key)
            result.append({
                "feature": key,
                "label": fm.label,
                "total": len(tier_models.models),
                "installed": len(tier_models.models) - len(missing),
                "missing": len(missing),
                "total_size": tier_models.total_size_display,
                "status": "Installed" if not missing else f"{len(missing)} missing",
            })
        return result
