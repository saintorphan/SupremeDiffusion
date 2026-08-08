"""Unified model loader for Supreme Diffusion.

Detects model format from filename and loads transformer, text encoder,
and VAE components accordingly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy / guarded imports -- let the app start even when deps are absent
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch is not installed. Model loading will be unavailable.")

try:
    import safetensors.torch as safetensors_torch
except ImportError:
    safetensors_torch = None  # type: ignore[assignment]
    logger.warning("safetensors is not installed. Safetensors loading will be unavailable.")

try:
    from diffusers import AutoencoderKLWan
except ImportError:
    AutoencoderKLWan = None  # type: ignore[assignment]
    logger.warning("diffusers is not installed or missing AutoencoderKLWan.")

try:
    from diffusers.models import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None  # type: ignore[assignment]
    logger.warning("diffusers is not installed or missing WanTransformer3DModel.")

try:
    from transformers import UMT5EncoderModel, T5EncoderModel, AutoTokenizer
except ImportError:
    UMT5EncoderModel = None  # type: ignore[assignment]
    T5EncoderModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    logger.warning("transformers is not installed. Text encoder loading will be unavailable.")

try:
    from optimum.quanto import quantize, freeze, qint8
except ImportError:
    quantize = None  # type: ignore[assignment]
    freeze = None  # type: ignore[assignment]
    qint8 = None  # type: ignore[assignment]
    logger.warning("optimum-quanto is not installed. Int8 quantization will be unavailable.")

try:
    import gguf  # type: ignore[import-untyped]
except ImportError:
    gguf = None  # type: ignore[assignment]
    logger.warning("gguf package is not installed. GGUF model loading will be unavailable.")

try:
    from mmgp.offload import fast_load_transformers_model as mmgp_load  # type: ignore[import-untyped]
except ImportError:
    mmgp_load = None  # type: ignore[assignment]


# Default config for Wan 2.2 I2V 14B (used when model file lacks embedded config)
_DEFAULT_TRANSFORMER_CONFIG = Path(__file__).parent / "configs" / "wan_i2v_2_2_14B.json"


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Determine the quantization / format of a model checkpoint from its path.

    Returns one of ``"quanto_int8"``, ``"gguf"``, ``"fp8"``, or ``"diffusers"``.

    When *path* is a directory (e.g. a diffusers-format transformer folder),
    the files inside are inspected.  A directory containing standard
    safetensors with no int8/fp8/gguf markers is reported as ``"diffusers"``.

    Raises ``ValueError`` when the format cannot be determined.
    """
    p = Path(path)

    # --- Directory path (diffusers-format download) ---
    if p.is_dir():
        # Collect model files inside the directory
        names = [f.name.lower() for f in p.iterdir() if f.is_file()]

        if any(n.endswith(".gguf") for n in names):
            return "gguf"
        if any(re.search(r"quanto.*int8.*\.safetensors$", n) or re.search(r"int8.*\.safetensors$", n) for n in names):
            return "quanto_int8"
        if any(re.search(r"lightning.*fp8.*\.safetensors$", n) or re.search(r"fp8.*\.safetensors$", n) for n in names):
            return "fp8"
        if any(n.endswith(".safetensors") for n in names):
            return "diffusers"

        raise ValueError(
            f"Cannot detect model format from directory '{p.name}': "
            "no recognised model files found inside."
        )

    # --- Single file path ---
    name = p.name.lower()

    if name.endswith(".gguf"):
        return "gguf"

    # quanto / int8 patterns  (e.g.  "…quanto_int8.safetensors", "…int8.safetensors")
    # Plain substring match: real quantized filenames embed the token without a
    # separator (e.g. "...I2VFP8HIGH.safetensors", "model_int8.safetensors").
    # A word-boundary anchor broke those, sending them down the diffusers path.
    if re.search(r"quanto.*int8.*\.safetensors$", name) or re.search(r"int8.*\.safetensors$", name):
        return "quanto_int8"

    # FP8 patterns  (e.g.  "…Lightning_FP8.safetensors", "…fp8.safetensors")
    if re.search(r"lightning.*fp8.*\.safetensors$", name) or re.search(r"fp8.*\.safetensors$", name):
        return "fp8"

    raise ValueError(
        f"Cannot detect model format from filename '{p.name}'. "
        "Provide the format explicitly (quanto_int8, gguf, fp8, or diffusers)."
    )


# ---------------------------------------------------------------------------
# Transformer loader
# ---------------------------------------------------------------------------

def _read_safetensors_config_kwargs(path: str) -> dict[str, Any]:
    """Read config from safetensors metadata and map Wan field names to diffusers params.

    Returns a dict suitable for ``configKwargs`` in mmgp's ``fast_load_transformers_model``.

    Note: We read the raw file header instead of using ``safetensors.safe_open``
    because mmgp monkey-patches that function with a version that lacks ``.metadata()``.
    """
    try:
        import base64
        import json as _json
        import struct

        with open(path, "rb") as fp:
            header_size = struct.unpack("<Q", fp.read(8))[0]
            header_json = fp.read(header_size)

        header = _json.loads(header_json)
        meta = header.get("__metadata__", {})
        config_b64 = meta.get("config_base64") or meta.get("config", None)
        if not config_b64:
            return {}
        raw = _json.loads(base64.b64decode(config_b64))
    except Exception:
        return {}

    # Map Wan-native field names → diffusers WanTransformer3DModel params
    kwargs: dict[str, Any] = {}
    if "in_dim" in raw:
        kwargs["in_channels"] = raw["in_dim"]
    if "out_dim" in raw:
        kwargs["out_channels"] = raw["out_dim"]
    if "num_heads" in raw:
        kwargs["num_attention_heads"] = raw["num_heads"]
    if "dim" in raw and "num_heads" in raw:
        kwargs["attention_head_dim"] = raw["dim"] // raw["num_heads"]
    if "ffn_dim" in raw:
        kwargs["ffn_dim"] = raw["ffn_dim"]
    if "freq_dim" in raw:
        kwargs["freq_dim"] = raw["freq_dim"]
    if "num_layers" in raw:
        kwargs["num_layers"] = raw["num_layers"]
    if "eps" in raw:
        kwargs["eps"] = raw["eps"]
    # I2V models embed CLIP image features via image_embedder
    if "image_dim" in raw:
        kwargs["image_dim"] = raw["image_dim"]
    if "pos_embed_seq_len" in raw:
        kwargs["pos_embed_seq_len"] = raw["pos_embed_seq_len"]

    if kwargs:
        logger.info("Mapped safetensors config to diffusers kwargs: %s", kwargs)
    return kwargs


# Wan-native → diffusers key mapping tables
_WAN_PREFIX_MAP = [
    ("time_projection.1.", "condition_embedder.time_proj."),
    ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
    ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
    ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
    ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
    ("head.head.", "proj_out."),
    ("head.modulation", "scale_shift_table"),
]

_WAN_BLOCK_MAP = [
    (".self_attn.q.", ".attn1.to_q."),
    (".self_attn.q", ".attn1.to_q"),  # for quantization map keys (no trailing dot)
    (".self_attn.k.", ".attn1.to_k."),
    (".self_attn.k", ".attn1.to_k"),
    (".self_attn.v.", ".attn1.to_v."),
    (".self_attn.v", ".attn1.to_v"),
    (".self_attn.o.", ".attn1.to_out.0."),
    (".self_attn.o", ".attn1.to_out.0"),
    (".self_attn.norm_q.", ".attn1.norm_q."),
    (".self_attn.norm_k.", ".attn1.norm_k."),
    (".cross_attn.q.", ".attn2.to_q."),
    (".cross_attn.q", ".attn2.to_q"),
    (".cross_attn.k.", ".attn2.to_k."),
    (".cross_attn.k", ".attn2.to_k"),
    (".cross_attn.v.", ".attn2.to_v."),
    (".cross_attn.v", ".attn2.to_v"),
    (".cross_attn.o.", ".attn2.to_out.0."),
    (".cross_attn.o", ".attn2.to_out.0"),
    (".cross_attn.norm_q.", ".attn2.norm_q."),
    (".cross_attn.norm_k.", ".attn2.norm_k."),
    (".ffn.0.", ".ffn.net.0.proj."),
    (".ffn.0", ".ffn.net.0.proj"),
    (".ffn.2.", ".ffn.net.2."),
    (".ffn.2", ".ffn.net.2"),
    (".modulation", ".scale_shift_table"),
    (".norm3.", ".norm2."),
]


def _remap_key(key: str) -> str:
    """Remap a single Wan-native key to a diffusers key."""
    for old, new in _WAN_PREFIX_MAP:
        if key.startswith(old):
            return new + key[len(old):]

    for old, new in _WAN_BLOCK_MAP:
        if old in key:
            return key.replace(old, new, 1)

    return key


def _strip_prefix(key: str) -> str:
    """Strip common checkpoint prefixes (ComfyUI/CivitAI format)."""
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _remap_wan_state_dict(sd: dict, quantization_map: Optional[dict] = None, *_args) -> Any:
    """Remap Wan-native checkpoint keys to diffusers WanTransformer3DModel keys.

    Handles both the state dict keys and the quantization map keys.
    mmgp calls this with ``(state_dict, quantization_map, tied_weights_map)``.
    If quantization_map is provided, returns ``(new_sd, new_qmap)`` tuple.
    """
    new_sd = {_remap_key(_strip_prefix(k)): v for k, v in sd.items()}

    remapped = sum(1 for a, b in zip(sd.keys(), new_sd.keys()) if a != b)
    if remapped:
        logger.info("Remapped %d/%d state dict keys from Wan-native to diffusers format.", remapped, len(sd))

    if quantization_map is not None:
        new_qmap = {_remap_key(_strip_prefix(k)): v for k, v in quantization_map.items()}
        logger.info("Remapped %d quantization map keys.", len(new_qmap))
        return new_sd, new_qmap

    return new_sd


def _remap_t5_state_dict(sd: dict, quantization_map: Optional[dict] = None, *_args) -> Any:
    """Remap Wan-native T5 encoder keys to transformers UMT5EncoderModel keys.

    mmgp calls this with ``(state_dict, quantization_map, tied_weights_map)``.
    """
    import re as _re

    def _remap_t5_key(key: str) -> str:
        # blocks.N.X → encoder.block.N.layer.L.X
        m = _re.match(r"blocks\.(\d+)\.(.*)", key)
        if m:
            block_idx = m.group(1)
            rest = m.group(2)

            # Attention: layer.0
            for attn_key in ("attn.q.", "attn.k.", "attn.v.", "attn.o.",
                             "attn.q", "attn.k", "attn.v", "attn.o"):
                if rest.startswith(attn_key):
                    sa_key = attn_key.replace("attn.", "")
                    suffix = rest[len(attn_key):]
                    return f"encoder.block.{block_idx}.layer.0.SelfAttention.{sa_key}{suffix}"

            # Norms
            if rest.startswith("norm1"):
                suffix = rest[len("norm1"):]
                return f"encoder.block.{block_idx}.layer.0.layer_norm{suffix}"
            if rest.startswith("norm2"):
                suffix = rest[len("norm2"):]
                return f"encoder.block.{block_idx}.layer.1.layer_norm{suffix}"

            # FFN: layer.1.DenseReluDense
            if rest.startswith("ffn.gate.0"):
                suffix = rest[len("ffn.gate.0"):]
                return f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_0{suffix}"
            if rest.startswith("ffn.fc1"):
                suffix = rest[len("ffn.fc1"):]
                return f"encoder.block.{block_idx}.layer.1.DenseReluDense.wi_1{suffix}"
            if rest.startswith("ffn.fc2"):
                suffix = rest[len("ffn.fc2"):]
                return f"encoder.block.{block_idx}.layer.1.DenseReluDense.wo{suffix}"

            # Positional embedding
            if rest.startswith("pos_embedding.embedding"):
                suffix = rest[len("pos_embedding.embedding"):]
                return f"encoder.block.{block_idx}.layer.0.SelfAttention.relative_attention_bias{suffix}"

        # Global keys
        if key.startswith("norm.") or key == "norm":
            return key.replace("norm", "encoder.final_layer_norm", 1)
        if key.startswith("token_embedding."):
            return key.replace("token_embedding.", "shared.", 1)

        return key

    new_sd = {_remap_t5_key(k): v for k, v in sd.items()}

    # Duplicate shared.weight → encoder.embed_tokens.weight (tied in UMT5)
    for k in list(new_sd.keys()):
        if k.startswith("shared."):
            tied_key = k.replace("shared.", "encoder.embed_tokens.", 1)
            if tied_key not in new_sd:
                new_sd[tied_key] = new_sd[k]

    remapped = sum(1 for a, b in zip(sd.keys(), new_sd.keys()) if a != b)
    if remapped:
        logger.info("Remapped %d/%d T5 encoder keys from Wan-native to transformers format.", remapped, len(sd))

    if quantization_map is not None:
        new_qmap = {_remap_t5_key(k): v for k, v in quantization_map.items()}
        logger.info("Remapped %d T5 quantization map keys.", len(new_qmap))
        return new_sd, new_qmap

    return new_sd


def _ensure_torch(label: str) -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required to {label} but is not installed.")


def load_transformer(
    path: str,
    format: Optional[str] = None,
    device: str = "cpu",
) -> Any:
    """Load a Wan transformer checkpoint.

    Parameters
    ----------
    path:
        Filesystem path to the model file.
    format:
        One of ``"quanto_int8"``, ``"gguf"``, ``"fp8"``.
        When *None* the format is auto-detected from the filename.
    device:
        Target device (e.g. ``"cpu"``, ``"cuda"``).

    Returns
    -------
    A ``WanTransformer3DModel`` instance with weights loaded.
    """
    _ensure_torch("load a transformer")

    if WanTransformer3DModel is None:
        raise RuntimeError(
            "diffusers.models.WanTransformer3DModel is not available. "
            "Install a compatible version of diffusers."
        )

    if format is None:
        format = detect_format(path)

    logger.info("Loading transformer from %s (format=%s, device=%s)", path, format, device)

    if format == "quanto_int8":
        return _load_transformer_quanto_int8(path, device)
    elif format == "gguf":
        return _load_transformer_gguf(path, device)
    elif format == "fp8":
        return _load_transformer_fp8(path, device)
    elif format == "diffusers":
        return _load_transformer_diffusers(path, device)
    else:
        raise ValueError(f"Unsupported transformer format: {format!r}")


def _load_transformer_quanto_int8(path: str, device: str) -> Any:
    """Load quanto int8 safetensors using mmgp's fast loader."""
    if mmgp_load is None:
        raise RuntimeError(
            "mmgp is required for loading quanto int8 models but is not installed."
        )

    try:
        config_kwargs = _read_safetensors_config_kwargs(path)
        transformer = mmgp_load(
            path,
            modelClass=WanTransformer3DModel,
            do_quantize=False,
            verboseLevel=1,
            configKwargs=config_kwargs,
            preprocess_sd=_remap_wan_state_dict,
        )
        logger.info("Transformer loaded with quanto int8 via mmgp from %s", path)
        return transformer

    except Exception as exc:
        raise RuntimeError(f"Failed to load quanto_int8 transformer from '{path}': {exc}") from exc


def _load_transformer_gguf(path: str, device: str) -> Any:
    """Load a GGUF transformer using mmgp's fast loader."""
    if mmgp_load is None:
        raise RuntimeError("mmgp is required for loading GGUF models but is not installed.")

    try:
        transformer = mmgp_load(
            path,
            modelClass=WanTransformer3DModel,
            do_quantize=False,
            verboseLevel=1,
            defaultConfigPath=str(_DEFAULT_TRANSFORMER_CONFIG),
            preprocess_sd=_remap_wan_state_dict,
        )
        logger.info("Transformer loaded from GGUF via mmgp from %s", path)
        return transformer

    except Exception as exc:
        raise RuntimeError(f"Failed to load GGUF transformer from '{path}': {exc}") from exc


def _load_transformer_fp8(path: str, device: str) -> Any:
    """Load an FP8 safetensors transformer using mmgp's fast loader."""
    if mmgp_load is None:
        raise RuntimeError("mmgp is required for loading FP8 models but is not installed.")

    try:
        config_kwargs = _read_safetensors_config_kwargs(path)
        load_kwargs: dict[str, Any] = {
            "modelClass": WanTransformer3DModel,
            "do_quantize": False,
            "verboseLevel": 1,
            "preprocess_sd": _remap_wan_state_dict,
        }
        if config_kwargs:
            load_kwargs["configKwargs"] = config_kwargs
        else:
            load_kwargs["defaultConfigPath"] = str(_DEFAULT_TRANSFORMER_CONFIG)
            logger.info("No config in safetensors metadata; using default config file.")
        transformer = mmgp_load(path, **load_kwargs)
        logger.info("Transformer loaded with FP8 via mmgp from %s", path)
        return transformer

    except Exception as exc:
        raise RuntimeError(f"Failed to load FP8 transformer from '{path}': {exc}") from exc


def _load_transformer_diffusers(path: str, device: str) -> Any:
    """Load a transformer from a diffusers-format directory.

    The directory should contain ``config.json`` and one or more
    ``.safetensors`` weight files (the standard diffusers layout).
    """
    p = Path(path)
    config_path = p / "config.json"

    # Try mmgp first — it handles diffusers directories with quantised weights
    if mmgp_load is not None:
        try:
            load_kwargs: dict[str, Any] = {
                "modelClass": WanTransformer3DModel,
                "do_quantize": False,
                "verboseLevel": 1,
                "preprocess_sd": _remap_wan_state_dict,
            }
            if config_path.is_file():
                load_kwargs["defaultConfigPath"] = str(config_path)
            transformer = mmgp_load(path, **load_kwargs)
            logger.info("Transformer loaded (diffusers dir) via mmgp from %s", path)
            return transformer
        except Exception as exc:
            logger.warning("mmgp failed on diffusers dir, falling back to from_pretrained: %s", exc)

    # Fallback: use diffusers' own from_pretrained
    try:
        transformer = WanTransformer3DModel.from_pretrained(path)
        logger.info("Transformer loaded via from_pretrained from %s", path)
        return transformer
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load diffusers transformer from '{path}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Text encoder loader
# ---------------------------------------------------------------------------

def load_text_encoder(
    path: str,
    quantization: str = "int8",
) -> Any:
    """Load a T5-family text encoder, optionally quantized.

    Parameters
    ----------
    path:
        Path to the pretrained text encoder directory or checkpoint.
    quantization:
        ``"int8"`` to apply optimum-quanto int8 quantization, or ``"none"``
        to load at full precision.

    Returns
    -------
    A text encoder model (``UMT5EncoderModel`` or ``T5EncoderModel``).
    """
    _ensure_torch("load a text encoder")

    if UMT5EncoderModel is None and T5EncoderModel is None:
        raise RuntimeError(
            "transformers is required for text encoder loading but is not installed."
        )

    logger.info("Loading text encoder from %s (quantization=%s)", path, quantization)

    encoder_path = Path(path)

    try:
        # If path is a directory, look for a quanto int8 safetensors file inside
        quanto_file = None
        if encoder_path.is_dir():
            for f in encoder_path.iterdir():
                if f.suffix == ".safetensors" and ("quanto" in f.name.lower() or "int8" in f.name.lower()):
                    quanto_file = f
                    break

        # Use mmgp fast loader for quanto int8 files
        if quanto_file is not None and mmgp_load is not None:
            encoder_class = UMT5EncoderModel if UMT5EncoderModel is not None else T5EncoderModel
            # Point mmgp to the config.json in the same directory
            config_path = quanto_file.parent / "config.json"
            kwargs: dict[str, Any] = {
                "do_quantize": False,
                "verboseLevel": 1,
                "modelClass": encoder_class,
                "preprocess_sd": _remap_t5_state_dict,
            }
            if config_path.is_file():
                kwargs["defaultConfigPath"] = str(config_path)
            encoder = mmgp_load(str(quanto_file), **kwargs)
            logger.info("Loaded text encoder via mmgp from %s", quanto_file)
            return encoder

        # Fallback: standard from_pretrained (for non-quantized models)
        try:
            encoder = UMT5EncoderModel.from_pretrained(
                str(encoder_path),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Loaded UMT5EncoderModel from %s", path)
        except Exception:
            encoder = T5EncoderModel.from_pretrained(
                str(encoder_path),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            logger.info("Loaded T5EncoderModel from %s", path)

        # Apply quantization if requested
        if quantization == "int8":
            if quantize is None or freeze is None or qint8 is None:
                logger.warning(
                    "optimum-quanto is not installed; skipping int8 quantization "
                    "for text encoder."
                )
            else:
                quantize(encoder, weights=qint8)
                freeze(encoder)
                logger.info("Applied int8 quantization to text encoder.")

        return encoder

    except Exception as exc:
        raise RuntimeError(f"Failed to load text encoder from '{path}': {exc}") from exc


# ---------------------------------------------------------------------------
# VAE loader
# ---------------------------------------------------------------------------

def load_vae(
    path: str,
    precision: str = "16",
) -> Any:
    """Load an ``AutoencoderKLWan`` VAE.

    Parameters
    ----------
    path:
        Path to the pretrained VAE directory or checkpoint.
    precision:
        ``"16"`` for float16, ``"32"`` for float32, ``"bf16"`` for bfloat16.
    """
    _ensure_torch("load a VAE")

    if AutoencoderKLWan is None:
        raise RuntimeError(
            "diffusers.AutoencoderKLWan is not available. "
            "Install a compatible version of diffusers."
        )

    dtype_map = {
        "16": torch.float16,
        "32": torch.float32,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(precision)
    if dtype is None:
        raise ValueError(
            f"Unsupported VAE precision '{precision}'. Choose from: {list(dtype_map.keys())}"
        )

    logger.info("Loading VAE from %s (precision=%s)", path, precision)

    vae_path = Path(path)

    try:
        # Single safetensors file — use from_single_file (has built-in key conversion)
        if vae_path.is_file() and vae_path.suffix == ".safetensors":
            vae = AutoencoderKLWan.from_single_file(
                str(vae_path),
                torch_dtype=dtype,
            )
            logger.info("VAE loaded via from_single_file at %s precision.", precision)
            return vae

        # Directory — standard from_pretrained
        vae = AutoencoderKLWan.from_pretrained(
            str(path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        logger.info("VAE loaded at %s precision.", precision)
        return vae

    except Exception as exc:
        raise RuntimeError(f"Failed to load VAE from '{path}': {exc}") from exc
