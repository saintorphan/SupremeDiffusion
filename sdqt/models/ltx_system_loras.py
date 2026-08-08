"""Registry of official LTX 2.3 22B "system" IC-LoRAs distributed by
Lightricks / DeepBeepMeep — exposed in the UI as labeled checkbox cards
grouped by category, separate from the user's own LoRA collection.

A LoRA is considered "system" if its filename starts with any of the
prefixes below. ``activated_loras`` in the project config stores the FULL
filename (not the prefix) so the existing pipeline LoRA loader works
unchanged.
"""

from __future__ import annotations

# (category, filename_prefix, display_name, description)
LTX_SYSTEM_LORAS: list[tuple[str, str, str, str]] = [
    # ── Structural / motion control ────────────────────────────────
    (
        "Control",
        "ltx-2.3-22b-ic-lora-union-control",
        "Union Control",
        "Pose / depth / canny / edge conditioning from a reference video. "
        "Set video_prompt_type letters: P=pose, D=depth, E=canny, V=raw, G=guide.",
    ),
    (
        "Control",
        "ltx-2.3-22b-ic-lora-outpaint",
        "Outpaint",
        "Extend frame edges. Pair with an outpainting source image / mask.",
    ),

    # ── Identity ───────────────────────────────────────────────────
    (
        "Identity",
        "id-lora-celebvhq-ltx2.3",
        "ID LoRA (CelebV-HQ)",
        "Locks character identity from a short REFERENCE VOICE clip (audio). "
        "Supply a reference audio file — identity is keyed off the voice, not "
        "the start image. Reference is truncated to ~4.8s.",
    ),

    # ── Visual quality / look ──────────────────────────────────────
    (
        "Visual",
        "ltx-2.3-22b-ic-lora-hdr-0.9",
        "HDR Cinema Look",
        "Cinematic grading + extended dynamic range. Pair with HDR Scene "
        "Embeddings for best results.",
    ),
    (
        "Visual",
        "ltx-2.3-22b-ic-lora-hdr-scene-emb",
        "HDR Scene Embeddings",
        "Scene-aware HDR conditioning. Used together with HDR Cinema Look.",
    ),
    (
        "Visual",
        "ltx-2.3-22b-ic-lora-refocus",
        "Refocus",
        "Detail recovery / focus restoration on soft inputs.",
    ),
    (
        "Visual",
        "ltx-2.3-22b-ic-lora-uncompress",
        "Uncompress",
        "Removes compression artifacts from low-bitrate source frames.",
    ),

    # ── Distillation helper (LoRA form, used with Dev model) ───────
    (
        "Pipeline",
        "ltx-2.3-22b-distilled-lora-384-1.1",
        "Distilled LoRA v1.1",
        "Apply distillation as a LoRA on top of the Dev transformer "
        "(skip if you're already using the distilled checkpoint).",
    ),

    # ── Community LoRAs (community-trained, not from Lightricks) ───
    (
        "Community",
        "LTX-2.3-22b-AV-LoRA-talking-head",
        "AV Talking Head (elix3r)",
        "First community audio-visual LoRA — trained on joint audio-video "
        "cross-attention for synchronized lip-sync from a soundtrack. "
        "Trigger: 'oh wx person'. LoRA strength 1.0.",
    ),
    (
        "Community",
        "ltx23_inpaint_masked_t2v_rank128_v1_10000steps",
        "Inpaint v1 (Alissonerdx, 10k steps)",
        "Mask-based inpainting LoRA (rank-128, 10k-step variant — best "
        "prompt adherence). Use with mask + IC-LoRA workflow.",
    ),
    (
        "Community",
        "ltx23_inpaint_masked_t2v_rank128_v1_02500steps",
        "Inpaint v1 (Alissonerdx, 2.5k steps)",
        "Earlier checkpoint of the rank-128 inpaint LoRA. "
        "Use only if 10k version is unavailable.",
    ),
    (
        "Community",
        "ltx23_inpaint_masked_r2v_rank32_v1_3000steps",
        "Inpaint Lite (Alissonerdx, rank-32)",
        "Smaller rank-32 inpaint LoRA for faster loading / less VRAM.",
    ),
    (
        "Community",
        "ltx2.3-transition",
        "Transition (joyfox / ValiantCat)",
        "Smooth first-to-last-frame guided transitions and morphs. "
        "Trigger word: 'zhuanchang'. LoRA strength 1.0.",
    ),
    (
        "Community",
        "galaxy-ace-realism",
        "Galaxy Ace Realism (placeholder)",
        "Early 2010s low-end Android phone footage aesthetic — sensor noise, "
        "compression artifacts, lens softness, handheld jitter. "
        "Trigger: 'early 2010s low-end android phone footage'. "
        "FILE NOT INSTALLED: drop the .safetensors into ltx_lora_dir.",
    ),
]


def is_system_lora(filename: str) -> bool:
    """True if the given filename matches any LTX system LoRA prefix."""
    stem = filename.rsplit(".", 1)[0]
    return any(stem.startswith(p) for _, p, _, _ in LTX_SYSTEM_LORAS)


def match_system_lora(filename: str) -> tuple[str, str, str, str] | None:
    """Return the registry entry for *filename*, or None."""
    stem = filename.rsplit(".", 1)[0]
    for entry in LTX_SYSTEM_LORAS:
        if stem.startswith(entry[1]):
            return entry
    return None
