"""prompt_writer.py — turn a brief idea into a model-formatted video prompt.

Pure prompt-orchestration logic (no torch, no Qt): given a ``chat_fn`` that
runs an LLM chat completion, it builds a model-aware system + user prompt,
instructs the exact output format the chosen model consumes, and parses the
result. Mirrors the VidAIPrompter orchestrator.

Formats:
  single       -> one cohesive cinematic prompt
  per-second   -> "(at N seconds: ...)" lines (Wan Lightning)
  per-window   -> one plain line per sliding window (SVI / LTX)

``chat_fn(messages: list[dict], max_tokens: int) -> str`` — typically
``AppState.prompt_llm_chat``.
"""
from __future__ import annotations

import math
import re

MAX_RETRIES = 2

# --- model capabilities (kept here so this stays Qt/torch-free) -------------
LIGHTNING_KEYS = frozenset({"i2v_2_2_lightning_v2", "i2v_2_2_lightning_v2_loras"})
SVI_KEYS = frozenset({"i2v_2_2_svi_2_pro", "i2v_2_2_svi_2_pro_lightning_v2"})


def backend(model_key: str) -> str:
    return "ltx" if (model_key or "").startswith("ltx_") else "wan"


def supports_per_second(model_key: str) -> bool:
    return model_key in LIGHTNING_KEYS


def supports_sliding(model_key: str) -> bool:
    return backend(model_key) == "ltx" or model_key in SVI_KEYS


SYSTEM_PROMPT = (
    "You are an expert prompt engineer for AI image-to-video diffusion models "
    "(Wan 2.x and LTX). You write vivid, concrete, cinematic prompts that "
    "emphasize MOTION, camera movement, subject action, lighting and mood. You "
    "describe what physically happens over time, not static scenes. You never "
    "refuse, never add commentary, never add disclaimers, and never wrap output "
    "in markdown code fences. You output only what the requested format asks for."
)

STYLE_GUIDES = {
    "wan": (
        "STYLE — Wan 2.x: write a vivid, cinematic description. Name the main "
        "subject and its motion/action first, then the scene and environment, "
        "then the camera movement, then the lighting and mood. Use concrete, "
        "present-tense visual language and emphasize how things move and change "
        "over time. Avoid abstract or poetic phrasing; describe what is "
        "physically visible."
    ),
    "ltx": (
        "STYLE — LTX: write a concrete, chronological description of the action "
        "as it unfolds. State the main action first, then the specific movements "
        "and gestures in the order they happen, then the appearance of subjects "
        "and setting, then camera angle/movement, then lighting and color. Be "
        "physical and specific (no abstract or poetic wording); LTX rewards "
        "detailed, literal motion description."
    ),
}


def style_guide(model_key: str) -> str:
    return STYLE_GUIDES.get(backend(model_key), STYLE_GUIDES["wan"])


# --- window / second math (mirrors generate.py + pipeline.py) ---------------
def compute_windows(frames: int, window: int, overlap: int = 0, be: str = "ltx") -> int:
    f, w, o = int(frames or 0), int(window or 0), max(0, int(overlap or 0))
    if w <= 0 or f <= w:
        return 1
    if be == "ltx":
        return max(1, math.ceil(f / w))
    stride = max(1, w - o)
    return max(1, math.ceil((f - w) / stride) + 1)


def compute_seconds(frames: int, fps: int) -> int:
    f, r = int(frames or 0), max(1, int(fps or 16))
    return max(0, (f - 1) // r) + 1


def seconds_per_window(window: int, fps: int) -> float:
    w, r = int(window or 0), max(1, int(fps or 16))
    return max((w - 1) / r, 1.0)


# --- text cleanup -----------------------------------------------------------
_ENUM_RE = re.compile(
    r"^\s*(?:[-*]\s+|\d+[.)]\s+|(?:window|win|shot|scene)\s*\d+\s*[:.\-]\s*"
    r"|\(?\s*at\s+\d+\s+seconds?\s*:\s*|\d+s\s*[:.\-]\s*)",
    re.IGNORECASE,
)


def clean_line(s: str) -> str:
    if not s:
        return ""
    t = str(s).strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    t = _ENUM_RE.sub("", t)
    # strip wrapping/stray markdown emphasis (**bold**, __, *italic*)
    t = re.sub(r"^[*_]{1,2}", "", t)
    t = re.sub(r"[*_]{1,2}$", "", t)
    t = re.sub(r'^"(.*)"$', r"\1", t, flags=re.DOTALL)
    t = re.sub(r"^'(.*)'$", r"\1", t, flags=re.DOTALL)
    return t.strip()


def _finalize_positive(positive: str) -> str:
    return re.sub(
        r"(^|\n)\s*positive(?:\s*prompt)?\s*:\s*", r"\1", positive, flags=re.IGNORECASE
    ).strip()


def _split_neg_value(val: str):
    """A captured negative value is one comma-separated line. Cut it where new
    content begins — a window/seconds marker or a new sentence — so trailing
    positive text the model tacked on rejoins the positive. Returns
    (negative, trailing_positive)."""
    val = str(val)
    cuts = []
    m = re.search(
        r"\s+(?:window|win|shot|scene)\s*\d+\s*[:.\-]|\(\s*at\s+\d+\s+seconds?\s*:",
        val, re.IGNORECASE,
    )
    if m:
        cuts.append(m.start())
    m = re.search(r"[.!?](?=\s+\S)", val)  # sentence end followed by more content
    if m:
        cuts.append(m.start())
    if cuts:
        c = min(cuts)
        return val[:c], val[c:].lstrip(" .!?,;:-")
    return val, ""


def split_negative(raw: str):
    """Split LLM text into (positive, negative).

    A real NEGATIVE label is either: a line that STARTS with "negative[:|-]"
    (case-insensitive — prose rarely starts a line that way), or an INLINE
    ALL-CAPS ``NEGATIVE:`` / a ``**bold**`` negative label (what the model emits
    when following instructions). Lowercase "negative:" inside ordinary prose
    ("the negative: void…") is NOT treated as a label, which is the fix for
    positive text leaking into the negative box. The captured value is one line,
    capped at any new sentence/window (those rejoin the positive). Markdown
    emphasis is stripped. ``[^\\S\\n]`` = horizontal whitespace only.
    """
    text = str(raw or "")
    lines = text.split("\n")

    # 1) Line-anchored label (case-insensitive); allow leading **/__ and : or -.
    lab_re = re.compile(
        r"^[^\S\n]*[*_]{0,2}\s*negative(?:\s*prompt)?\s*[:\-][^\S\n]*(.*?)[*_]*$",
        re.IGNORECASE,
    )
    for i, ln in enumerate(lines):
        m = lab_re.match(ln)
        if not m:
            continue
        val = m.group(1).strip()
        consumed = {i}
        if not clean_line(val):  # label alone -> value on the next non-empty line
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    val = lines[j]
                    consumed.add(j)
                    break
        neg_part, trailing = _split_neg_value(val)
        pos_lines = [l for k, l in enumerate(lines) if k not in consumed]
        if trailing.strip():
            pos_lines.append(trailing.strip())
        return _finalize_positive("\n".join(pos_lines)), clean_line(neg_part)

    # 2) Inline label: ALL-CAPS NEGATIVE (case-sensitive) — a deliberate label,
    #    not lowercase prose. Fall back to a **bold** negative label.
    m = re.search(r"\bNEGATIVE(?:\s+PROMPT)?[^\S\n]*:[^\S\n]*([^\n]*)", text)
    if not m:
        m = re.search(r"\*\*\s*negative(?:\s*prompt)?[^\S\n]*:[^\S\n]*([^\n]*)", text, re.IGNORECASE)
    if m:
        neg_part, trailing = _split_neg_value(m.group(1))
        tail = ("\n" + trailing.strip()) if trailing.strip() else ""
        positive = text[: m.start()] + tail + text[m.end():]
        return _finalize_positive(positive), clean_line(neg_part)

    return _finalize_positive(text), ""


# --- mode + plan ------------------------------------------------------------
def resolve_mode(model_key: str, requested: str) -> str:
    if requested == "per-second" and supports_per_second(model_key):
        return "per-second"
    if requested == "per-window" and supports_sliding(model_key):
        return "per-window"
    return "single"


def _build_plan(model_key, mode, idea, frames, fps, window_size, overlap, gen_neg, neg):
    neg_block = (
        '\nAfter the prompt, on a NEW line, output "NEGATIVE: " followed by a '
        "single concise negative prompt (comma-separated artifacts/quality terms "
        "to avoid). Do not number it." if gen_neg else ""
    )
    neg_seed = (
        f'\nThe user has set this negative prompt (do not change it): "{neg}".'
        if (not gen_neg and neg) else ""
    )

    if mode == "per-second":
        secs = compute_seconds(frames, fps)
        user = (
            f'Write an image-to-video prompt for this idea:\n"{idea}"\n\n'
            f"The clip is {secs} second(s) long at {fps} fps. Break the action "
            f"into a per-second timeline so motion progresses naturally. Output "
            f"EXACTLY {secs} line(s), one per second, each formatted EXACTLY as:\n"
            f"(at K seconds: <vivid description of what happens at second K>)\n"
            f"for K = 0 to {secs - 1}, in order. Keep each line self-contained and "
            f"motion-focused. Do not output anything except those {secs} lines"
            f"{' and the NEGATIVE line' if gen_neg else ''}." + neg_block + neg_seed
        )
        return {"user": user, "count": secs, "parse": "timed"}

    if mode == "per-window":
        windows = compute_windows(frames, window_size, overlap, backend(model_key))
        spw = f"{seconds_per_window(window_size, fps):.1f}"
        user = (
            f'Write an image-to-video prompt for this idea:\n"{idea}"\n\n'
            f"This will render with a sliding window: {windows} sequential "
            f"window(s) of ~{spw}s each (window size {window_size} frames @ {fps} "
            f"fps), played back-to-back as one continuous shot. Write ONE prompt "
            f"line per window so the action evolves across windows while staying "
            f"coherent. Output EXACTLY {windows} line(s), one plain-text prompt "
            f"per window, in order, no numbering, no labels. Each window continues "
            f"from the previous one. Do not output anything except those "
            f"{windows} line(s){' and the NEGATIVE line' if gen_neg else ''}."
            + neg_block + neg_seed
        )
        return {"user": user, "count": windows, "parse": "lines"}

    user = (
        f'Write ONE detailed image-to-video prompt for this idea:\n"{idea}"\n\n'
        f"Produce a single, cohesive, cinematic paragraph emphasizing motion, "
        f"camera work, lighting and mood. Output ONLY the prompt text"
        f"{', then a NEGATIVE line as instructed' if gen_neg else ''}."
        + neg_block + neg_seed
    )
    return {"user": user, "count": 1, "parse": "single"}


# --- parse + assemble -------------------------------------------------------
_TIMED_RE = re.compile(
    r"\(\s*at\s+(\d+)\s+seconds?\s*:\s*([\s\S]+?)"
    r"(?:\)|(?=\n\s*\(\s*at\s+\d+\s+seconds?\s*:)|$)",
    re.IGNORECASE,
)


def parse_segments(positive: str, parse: str):
    if parse == "timed":
        found = [
            (int(m.group(1)), clean_line(m.group(2)))
            for m in _TIMED_RE.finditer(positive)
        ]
        found.sort(key=lambda p: p[0])
        if found:
            return [{"label": f"{sec}s", "text": t} for sec, t in found if t]
        out = []
        for i, line in enumerate([l.strip() for l in positive.split("\n") if l.strip()]):
            mm = re.search(r"at\s+(\d+)\s+seconds?", line, re.IGNORECASE)
            sec = int(mm.group(1)) if mm else i
            txt = clean_line(line)
            if txt:
                out.append({"label": f"{sec}s", "text": txt})
        return out
    if parse == "lines":
        lines = [clean_line(l) for l in positive.split("\n")]
        return [{"label": f"Win {i + 1}", "text": t} for i, t in enumerate(lines) if t]
    txt = clean_line(positive)
    return [{"label": "Prompt", "text": txt}] if txt else []


def assemble(segments, parse: str) -> str:
    if parse == "timed":
        return "\n".join(
            f"(at {s['label'].replace('s', '')} seconds: {s['text']})" for s in segments
        )
    if parse == "lines":
        return "\n".join(s["text"] for s in segments)
    return "\n".join(s["text"] for s in segments).strip()


# --- public API -------------------------------------------------------------
def generate(
    chat_fn,
    *,
    model_key: str,
    model_label: str = "",
    idea: str,
    mode: str = "single",
    frames: int = 0,
    fps: int = 16,
    window_size: int = 0,
    overlap: int = 0,
    generate_negative: bool = True,
    negative: str = "",
) -> dict:
    """Generate a model-formatted prompt. Returns a structured result dict."""
    if not (idea or "").strip():
        raise ValueError("Idea / main prompt is empty.")

    eff_mode = resolve_mode(model_key, mode)
    plan = _build_plan(
        model_key, eff_mode, idea.strip(), frames, fps, window_size, overlap,
        generate_negative, negative or "",
    )

    per_item = 320 if plan["parse"] == "single" else 110
    max_tokens = min(max(per_item * plan["count"] + 120, 256), 4000)

    label = model_label or model_key
    base_system = (
        f"{SYSTEM_PROMPT}\n\nYou are writing for the video model: {label} "
        f"({backend(model_key).upper()} backend).\n{style_guide(model_key)}"
    )

    system_prompt = base_system
    segments, negative_out, last_raw = [], "", ""
    for attempt in range(MAX_RETRIES + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": plan["user"]},
        ]
        last_raw = (chat_fn(messages, max_tokens) or "").strip()
        positive, neg = split_negative(last_raw)
        segments = [s for s in parse_segments(positive, plan["parse"]) if s["text"]]
        enough = (
            segments and segments[0]["text"]
            if plan["parse"] == "single" else bool(segments)
        )
        if enough:
            negative_out = neg
            break
        system_prompt = (
            base_system + "\n\nIMPORTANT: Your previous output was empty or not in "
            "the required format. Follow the EXACT output format. Output only the "
            f"requested {'prompt' if plan['parse'] == 'single' else str(plan['count']) + ' line(s)'}."
        )

    if not segments:
        raise RuntimeError(
            "The LLM returned no usable prompt. Check the prompt LLM, or try again."
        )

    note = ""
    if plan["parse"] != "single" and len(segments) != plan["count"]:
        unit = "second-lines" if plan["parse"] == "timed" else "window-lines"
        note = (
            f"Expected {plan['count']} {unit}, got {len(segments)}. Edit below — "
            "the renderer clamps to the last line if short."
        )

    return {
        "model_key": model_key,
        "model_label": label,
        "mode": eff_mode,
        "count": plan["count"],
        "fps": fps,
        "frames": frames,
        "window_size": window_size,
        "segments": segments,
        "prompt": assemble(segments, plan["parse"]),
        "negative_prompt": negative_out if generate_negative else (negative or ""),
        "raw": last_raw,
        "note": note,
    }


def refine(chat_fn, *, model_key: str, model_label: str = "", mode: str = "single",
           label: str = "", text: str = "", instruction: str = "") -> str:
    """Refine a SINGLE prompt segment via the LLM (the magic wand)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to refine — the prompt is empty.")
    unit = {"per-second": "one-second beat", "per-window": "sliding-window"}.get(mode, "video")
    base_system = (
        f"{SYSTEM_PROMPT}\n\nYou are writing for the video model: "
        f"{model_label or model_key} ({backend(model_key).upper()} backend).\n"
        f"{style_guide(model_key)}"
    )
    instr = (instruction or "").strip()
    user = (
        f'Here is one {unit} prompt for an image-to-video shot:\n"{text}"\n\n'
        + (f"Revise it as follows: {instr}." if instr
           else "Rewrite it to be more vivid, concrete and motion-focused while "
                "keeping the same core action.")
        + ' Output ONLY the revised prompt as a single line — no label, no quotes, '
          'no commentary, and no "(at N seconds: ...)" wrapper.'
    )
    raw = (chat_fn(
        [{"role": "system", "content": base_system}, {"role": "user", "content": user}],
        240,
    ) or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    raw = re.sub(r'^"([\s\S]*)"$', r"\1", raw).strip()
    raw = re.sub(r"^'([\s\S]*)'$", r"\1", raw).strip()
    out = clean_line(raw)
    if out.endswith(")") and "(" not in out:
        out = out[:-1].strip()
    if not out:
        raise RuntimeError("The LLM returned nothing. Check the prompt LLM.")
    return out
