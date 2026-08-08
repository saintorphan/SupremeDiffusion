"""Checkpoint/VAE/LoRA discovery for Stable Diffusion models."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CHECKPOINT_EXTENSIONS = {".safetensors", ".ckpt"}
_VAE_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}
_LORA_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}


@dataclass
class CheckpointInfo:
    filename: str       # full path
    name: str           # display name (stem)
    model_type: str     # "sd15", "sdxl", "flux", or "zimage"
    subtype: str = ""   # e.g. "chroma", "fill" for FLUX
    group: str = ""     # display group (subfolder name or model_type label)


_ZIMAGE_KEYWORDS = {"zimage", "z-image", "z_image"}


def _is_zimage_name(name: str) -> bool:
    low = name.lower()
    return any(kw in low for kw in _ZIMAGE_KEYWORDS)


def _read_safetensors_header_text(path: Path) -> str | None:
    """Read a safetensors file's JSON header as text (capped at 10MB).

    Returns the decoded header text, or ``None`` if the file can't be read.
    Shared by the family/type detectors so the header is parsed in one place.
    """
    try:
        with open(path, "rb") as f:
            # safetensors format: 8-byte LE header length, then JSON header
            header_size = struct.unpack("<Q", f.read(8))[0]
            # Cap read to 10MB to avoid reading huge headers
            header_bytes = f.read(min(header_size, 10 * 1024 * 1024))
            return header_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.debug("Could not read safetensors header for %s: %s", path, exc)
        return None


def _sd_type_from_header(header_text: str) -> str:
    """Classify SD15 vs SDXL from a safetensors header text blob."""
    # SDXL-specific keys (dual text encoder)
    if "conditioner.embedders.1." in header_text:
        return "sdxl"
    # Also check for text_model_2 pattern (some SDXL checkpoints)
    if "text_model_2" in header_text:
        return "sdxl"
    return "sd15"


def detect_model_family(path: str | Path) -> str:
    """Detect the model family of a checkpoint file or directory.

    Returns one of: ``"sd15"``, ``"sdxl"``, ``"flux"``, ``"zimage"``.
    """
    path = Path(path)

    # GGUF → FLUX
    if path.suffix.lower() == ".gguf":
        return "flux"

    # Filename heuristic for Z-Image
    if _is_zimage_name(path.stem if path.is_file() else path.name):
        return "zimage"

    # Diffusers directory: check model_index.json
    if path.is_dir():
        index_file = path / "model_index.json"
        if index_file.exists():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                class_name = data.get("_class_name", "")
                if "Flux" in class_name:
                    return "flux"
                if _is_zimage_name(class_name) or _is_zimage_name(path.name):
                    return "zimage"
            except Exception:
                pass
        # Check for FLUX-specific subdirs
        if (path / "transformer").is_dir():
            # Could be FLUX diffusers format
            for sub in ("double_blocks", "single_blocks"):
                if any((path / "transformer").glob(f"*{sub}*")):
                    return "flux"
        return "sdxl"  # default for diffusers dirs

    # Safetensors header inspection
    if path.suffix.lower() == ".safetensors" and path.is_file():
        header_text = _read_safetensors_header_text(path)
        if header_text is not None:
            # FLUX indicators
            if "double_blocks" in header_text or "single_blocks" in header_text:
                return "flux"
            if "img_in" in header_text and "txt_in" in header_text:
                return "flux"

            # Fall through to SD15/SDXL detection
            return _sd_type_from_header(header_text)

    # Fallback: file size heuristic
    try:
        size_gb = path.stat().st_size / (1024 ** 3)
        if size_gb > 5.0:
            return "sdxl"
    except OSError:
        pass

    return "sd15"


def scan_all_image_models(global_config) -> list[CheckpointInfo]:
    """Scan all configured model directories and return a unified list.

    *global_config* must expose ``model_paths`` (dict-like).
    """
    results: list[CheckpointInfo] = []
    mp = global_config.model_paths

    # SD checkpoints (sd15/sdxl)
    sd_dir = mp.get("sd_checkpoint_dir", "")
    if sd_dir:
        for ci in scan_checkpoints(sd_dir):
            results.append(ci)

    # FLUX GGUF models
    flux_gguf_dir = mp.get("flux_gguf_dir", "")
    if flux_gguf_dir:
        base = Path(flux_gguf_dir)
        for subdir, subtype in [("chroma", "chroma"), ("fill", "fill")]:
            d = base / subdir
            if d.is_dir():
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() == ".gguf" and f.is_file():
                        results.append(CheckpointInfo(
                            filename=str(f),
                            name=f.stem,
                            model_type="flux",
                            subtype=subtype,
                            group="Chroma" if subtype == "chroma" else "Flux",
                        ))

    # Z-Image models
    zimage_dir = mp.get("zimage_dir", "")
    if zimage_dir:
        zp = Path(zimage_dir)
        # Determine scan directory
        if zp.is_dir() and (zp / "model_index.json").exists():
            scan_dir = zp.parent
        elif zp.is_file():
            scan_dir = zp.parent
        elif zp.is_dir():
            scan_dir = zp
        else:
            scan_dir = None

        if scan_dir and scan_dir.is_dir():
            for entry in sorted(scan_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() == ".safetensors" and _is_zimage_name(entry.stem):
                    results.append(CheckpointInfo(
                        filename=str(entry),
                        name=entry.stem,
                        model_type="zimage",
                    group="Z-Image",
                    ))
                elif entry.is_dir() and _is_zimage_name(entry.name):
                    if (entry / "model_index.json").exists():
                        results.append(CheckpointInfo(
                            filename=str(entry),
                            name=entry.name,
                            model_type="zimage",
                        ))
                    else:
                        for sub in sorted(entry.iterdir()):
                            if sub.is_file() and sub.suffix.lower() == ".safetensors":
                                results.append(CheckpointInfo(
                                    filename=str(sub),
                                    name=sub.stem,
                                    model_type="zimage",
                                ))
                            elif sub.is_dir() and (sub / "model_index.json").exists():
                                results.append(CheckpointInfo(
                                    filename=str(sub),
                                    name=sub.name,
                                    model_type="zimage",
                                ))

    return results


def detect_model_type(path: str | Path) -> str:
    """Detect whether a safetensors checkpoint is SD 1.5 or SDXL.

    SDXL models have dual text encoder keys like
    ``conditioner.embedders.1.*`` in their metadata/tensor names.
    Fallback: file size heuristic (>5GB = SDXL).
    """
    path = Path(path)

    if path.suffix == ".safetensors":
        header_text = _read_safetensors_header_text(path)
        if header_text is not None:
            return _sd_type_from_header(header_text)

    # Fallback: file size heuristic
    try:
        size_gb = path.stat().st_size / (1024 ** 3)
        if size_gb > 5.0:
            return "sdxl"
    except OSError:
        pass

    return "sd15"


def scan_checkpoints(directory: str | Path, recursive: bool = True) -> list[CheckpointInfo]:
    """Scan a directory for checkpoint files and return info about each.

    When *recursive* is True, scans subdirectories and prefixes the
    checkpoint name with the subfolder (e.g. ``PonyXL/catpony_realV31``).
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Checkpoint directory does not exist: %s", directory)
        return []

    results = []

    def _scan_dir(d: Path, prefix: str = "", group: str = "") -> None:
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in _CHECKPOINT_EXTENSIONS:
                model_type = detect_model_type(f)
                display_name = f"{prefix}{f.stem}" if prefix else f.stem
                results.append(CheckpointInfo(
                    filename=str(f),
                    name=display_name,
                    model_type=model_type,
                    group=group,
                ))
            elif recursive and f.is_dir():
                # Top-level subfolder becomes the group name
                sub_group = group or f.name
                _scan_dir(f, prefix=f"{prefix}{f.name}/", group=sub_group)

    _scan_dir(directory)
    logger.info("Found %d checkpoints in %s", len(results), directory)
    return results


def scan_vaes(directory: str | Path) -> list[str]:
    """Scan a directory (recursively) for VAE files. Returns display names (stems)."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("VAE directory does not exist: %s", directory)
        return []

    results = []
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in _VAE_EXTENSIONS and f.is_file():
            results.append(f.stem)

    logger.info("Found %d VAEs in %s", len(results), directory)
    return results


def scan_loras(directory: str | Path) -> list[str]:
    """Scan a directory (recursively) for LoRA files. Returns filenames."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("LoRA directory does not exist: %s", directory)
        return []

    results = []
    for f in sorted(directory.rglob("*")):
        if f.suffix.lower() in _LORA_EXTENSIONS and f.is_file():
            results.append(f.name)

    logger.info("Found %d LoRAs in %s", len(results), directory)
    return results
