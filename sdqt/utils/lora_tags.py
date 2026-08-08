"""Shared LoRA tag utilities — regex, extraction, and reinsertion.

Used by quick_gen, pipeline_wizard, create_character, and character_video
to strip/preserve ``<lora:name:weight>`` tags around Qwen prompt enhancement.
"""

from __future__ import annotations

import re

# Matches <lora:name:weight> optionally followed by trigger text
LORA_TAG_RE = re.compile(r"<lora:[^>]+>(?:\s+[^<,\n]+)?")

# Simpler variant that captures (name, weight) groups
LORA_TAG_CAPTURE_RE = re.compile(r"<lora:([^:>]+):([0-9]*\.?[0-9]+)>")


def extract_lora_tags(text: str) -> tuple[str, list[str]]:
    """Split prompt into (clean_text, [lora_tags]).

    Removes all ``<lora:...>`` tags (plus trailing trigger text) from *text*
    and returns the cleaned text alongside the extracted tag strings.
    """
    tags = LORA_TAG_RE.findall(text)
    clean = LORA_TAG_RE.sub("", text).strip()
    clean = re.sub(r"[, ]{2,}", ", ", clean).strip(", ")
    return clean, tags


def reinsert_lora_tags(enhanced: str, tags: list[str]) -> str:
    """Prepend LoRA tags back onto a Qwen-enhanced prompt."""
    if not tags:
        return enhanced
    return f"{' '.join(tags)} {enhanced}".strip()


def strip_lora_tags(text: str) -> str:
    """Remove ``<lora:...>`` tags from prompt text.

    These are A1111 syntax that diffusers doesn't parse — they waste
    CLIP's 77-token budget if left in.  LoRA files are applied separately.
    """
    cleaned = LORA_TAG_CAPTURE_RE.sub("", text)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    return cleaned.strip(", ")
