"""Keyword-based upstream doc search for Qwen chat context injection."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DOCS_DIR = Path(__file__).parent.parent / "docs" / "upstream"

# Mapping: filename -> list of keyword triggers (lowercase)
# Mirrors the UPSTREAM_DOCS list in scripts/download_upstream_docs.py
_DOC_KEYWORDS: dict[str, list[str]] = {
    "wan_video.md": ["wan", "wan2.1", "wan2.2", "i2v", "video generation", "14b"],
    "diffusers.md": ["diffusers", "huggingface", "pipeline", "scheduler", "sampler"],
    "stable_diffusion.md": ["stable diffusion", "sd", "sd15", "sdxl", "txt2img", "img2img"],
    "flux.md": ["flux", "chroma", "flux fill", "black forest"],
    "controlnet.md": ["controlnet", "control net", "openpose", "canny", "depth", "condition"],
    "ip_adapter.md": ["ip-adapter", "ip adapter", "image prompt", "style transfer"],
    "latentsync.md": ["latentsync", "latent sync", "lip sync", "lipsync", "audio driven"],
    "musetalk.md": ["musetalk", "muse talk", "lip sync", "talking head"],
    "video_retalking.md": ["video retalking", "retalking", "lip sync", "face reenactment"],
    "insightface.md": ["insightface", "face detection", "face recognition", "buffalo", "face analysis"],
    "roop.md": ["roop", "face swap", "faceswap", "inswapper"],
    "gfpgan.md": ["gfpgan", "face restoration", "face enhance", "face upscale"],
    "codeformer.md": ["codeformer", "code former", "face restoration"],
    "triposr.md": ["triposr", "tripo", "3d", "mesh", "3d modeling", "3d generation"],
    "real_esrgan.md": ["real-esrgan", "esrgan", "upscale", "super resolution", "upscaling"],
    "rife.md": ["rife", "frame interpolation", "interpolation", "slow motion"],
    "orpheus_tts.md": ["orpheus", "tts", "text to speech", "speech synthesis"],
    "openvoice.md": ["openvoice", "open voice", "voice cloning", "voice clone"],
    "rvc.md": ["rvc", "voice conversion", "retrieval", "voice change"],
    "lora.md": ["lora", "low-rank", "low rank", "adaptation", "fine-tune", "finetune"],
    "xformers.md": ["xformers", "memory efficient", "attention"],
    "birefnet.md": ["birefnet", "background removal", "segmentation", "magic mask", "matting"],
    "sd3.md": ["sd3", "sd3.5", "stable diffusion 3", "mmdit"],
    "whisper.md": ["whisper", "speech recognition", "transcription"],
}

# Max chars to inject per doc (keeps context budget reasonable)
_MAX_DOC_CHARS = 6000


def find_relevant_docs(user_message: str, max_docs: int = 2) -> str:
    """Search upstream docs for content relevant to the user's message.

    Returns a formatted string to inject into the chat context, or empty
    string if nothing matches.
    """
    if not _DOCS_DIR.is_dir():
        return ""

    msg_lower = user_message.lower()

    # Score each doc by keyword match count
    scores: list[tuple[str, int]] = []
    for filename, keywords in _DOC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # Use word boundary matching for short keywords to avoid false positives
            if len(kw) <= 3:
                if re.search(rf'\b{re.escape(kw)}\b', msg_lower):
                    score += 2
            elif kw in msg_lower:
                score += 2 if len(kw) > 5 else 1
        if score > 0:
            scores.append((filename, score))

    if not scores:
        return ""

    # Take top N docs by score
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:max_docs]

    parts: list[str] = []
    for filename, _score in top:
        doc_path = _DOCS_DIR / filename
        if not doc_path.is_file():
            continue
        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError:
            continue

        # Truncate to budget
        if len(content) > _MAX_DOC_CHARS:
            content = content[:_MAX_DOC_CHARS] + "\n\n[... truncated ...]"

        doc_name = filename.replace(".md", "").replace("_", " ").title()
        parts.append(f"--- Reference: {doc_name} ---\n{content}")

    if not parts:
        return ""

    header = (
        "\n\nThe following upstream documentation may help answer the user's question. "
        "Use it as reference material — cite specific details when relevant:\n\n"
    )
    return header + "\n\n".join(parts)
