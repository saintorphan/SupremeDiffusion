"""PNG metadata reading and A1111-style parameter parsing."""

from __future__ import annotations

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Regex for "Key: value" pairs on the parameters line (A1111 format)
_RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def read_png_info(image_path: str) -> dict[str, str]:
    """Read all text chunks from a PNG file.

    Returns a dict of chunk keys → values. Common keys:
    - "parameters": A1111-style generation info
    - "Comment": NovelAI / ComfyUI metadata
    - "prompt", "workflow": ComfyUI metadata
    """
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
    except ImportError:
        return {}

    try:
        img = Image.open(image_path)
    except Exception as exc:
        logger.warning("Could not open image %s: %s", image_path, exc)
        return {}

    info = {}

    # PNG text chunks are in img.info (dict) or img.text (dict)
    raw = getattr(img, "info", {}) or {}
    text = getattr(img, "text", {}) or {}

    # Merge both sources
    for d in (raw, text):
        for k, v in d.items():
            if isinstance(v, str):
                info[k] = v
            elif isinstance(v, bytes):
                try:
                    info[k] = v.decode("utf-8", errors="replace")
                except Exception:
                    pass

    # Also check EXIF for tools that embed there (e.g. JPEG from A1111)
    if not info.get("parameters"):
        try:
            exif = img.getexif()
            if exif:
                # UserComment tag (0x9286) is used by some tools
                user_comment = exif.get(0x9286, "")
                if isinstance(user_comment, bytes):
                    user_comment = user_comment.decode("utf-8", errors="replace")
                if user_comment and "Steps:" in user_comment:
                    info["parameters"] = user_comment
        except Exception:
            pass

    return info


def parse_a1111_parameters(params_text: str) -> dict[str, Any]:
    """Parse an A1111-format parameters string into a structured dict.

    Input format:
        prompt text (possibly multiline)
        Negative prompt: negative text
        Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 12345, Size: 512x768, Model: modelname, ...

    Returns dict with keys like:
        "prompt", "negative_prompt", "Steps", "Sampler", "Schedule type",
        "CFG scale", "Seed", "Size", "Width", "Height", "Model", "Clip skip", etc.
    """
    if not params_text or not params_text.strip():
        return {}

    lines = params_text.strip().split("\n")
    result: dict[str, Any] = {}

    # Find where the last "parameters line" starts (contains "Steps:")
    params_line_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if "Steps:" in lines[i]:
            params_line_idx = i
            break

    if params_line_idx == -1:
        # No structured parameters found — treat entire text as prompt
        result["prompt"] = params_text.strip()
        return result

    # Everything before the params line = prompt + negative prompt
    prompt_lines = []
    negative_lines = []
    in_negative = False

    for line in lines[:params_line_idx]:
        if line.startswith("Negative prompt:"):
            in_negative = True
            neg_text = line[len("Negative prompt:"):].strip()
            if neg_text:
                negative_lines.append(neg_text)
        elif in_negative:
            negative_lines.append(line)
        else:
            prompt_lines.append(line)

    result["prompt"] = "\n".join(prompt_lines).strip()
    result["negative_prompt"] = "\n".join(negative_lines).strip()

    # Parse the key-value parameters line
    params_str = "\n".join(lines[params_line_idx:])
    for match in _RE_PARAM.finditer(params_str):
        key = match.group(1).strip()
        value = match.group(2).strip()
        # Unquote if quoted
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('\\"', '"')
        result[key] = value

    # Split "Size: WxH" into Width/Height
    if "Size" in result:
        size = result["Size"]
        if "x" in size:
            parts = size.split("x")
            try:
                result["Width"] = int(parts[0])
                result["Height"] = int(parts[1])
            except ValueError:
                pass

    # Normalize common key variations
    _aliases = {
        "Sampling method": "Sampler",
        "Sampling steps": "Steps",
        "Schedule type": "Schedule type",
    }
    for alias, canonical in _aliases.items():
        if alias in result and canonical not in result:
            result[canonical] = result[alias]

    return result


def format_info_html(info: dict[str, str]) -> str:
    """Format raw PNG info dict as readable HTML for display."""
    if not info:
        return "<p>No metadata found.</p>"

    parts = []
    for key, value in info.items():
        # Escape HTML
        safe_key = key.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_val = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Truncate extremely long values (ComfyUI workflows can be huge)
        if len(safe_val) > 5000:
            safe_val = safe_val[:5000] + "... (truncated)"
        parts.append(f"**{safe_key}:**\n```\n{safe_val}\n```")

    return "\n\n".join(parts)


def format_params_display(params: dict[str, Any]) -> str:
    """Format parsed parameters as readable markdown."""
    if not params:
        return ""

    lines = []

    prompt = params.get("prompt", "")
    if prompt:
        lines.append(f"**Prompt:** {prompt}")

    neg = params.get("negative_prompt", "")
    if neg:
        lines.append(f"**Negative prompt:** {neg}")

    # Generation settings
    settings = []
    for key in ["Steps", "Sampler", "Schedule type", "CFG scale", "Seed",
                 "Size", "Model", "Model hash", "Clip skip", "Denoising strength",
                 "Hires upscale", "Hires steps", "Hires upscaler",
                 "Lora hashes", "Version"]:
        if key in params:
            settings.append(f"**{key}:** {params[key]}")

    # Any remaining keys not already shown
    shown = {"prompt", "negative_prompt", "Steps", "Sampler", "Schedule type",
             "CFG scale", "Seed", "Size", "Model", "Model hash", "Clip skip",
             "Denoising strength", "Hires upscale", "Hires steps", "Hires upscaler",
             "Lora hashes", "Version", "Width", "Height"}
    for key, val in params.items():
        if key not in shown:
            settings.append(f"**{key}:** {val}")

    if settings:
        lines.append("\n".join(settings))

    return "\n\n".join(lines)
