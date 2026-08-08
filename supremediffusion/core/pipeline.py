"""Orchestrates mode -> clip -> extend generation flow for Supreme Diffusion."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image

from supremediffusion.config.project_config import ProjectConfig
from supremediffusion.core.guidance import GuidanceGenerator
from supremediffusion.core.input_handler import InputHandler
from supremediffusion.core.output_handler import OutputHandler
from supremediffusion.models.lora import LoRAManager, parse_phase_multipliers, compute_switch_step

logger = logging.getLogger(__name__)


# _color_correct_frames was removed in Phase 5. All call sites migrated to
# supremediffusion.core.post_color_anchor.anchor_frames_for_chunk which adds
# per-frame fade + persistence + multi-mode reference resolution. To match
# the legacy flat-correction behavior, set anchor_image_mode='start' (or
# 'single' / 'previous_last') with color_anchor_persistence=1.0 (no fade).


class GenerationPipeline:
    """High-level pipeline that ties input handling, guidance, generation, and output together.

    Parameters
    ----------
    wan_pipeline:
        A ``WanI2VPipeline`` or ``LTXVideoPipeline`` instance (or any
        compatible object) that exposes ``.generate()`` and ``.is_loaded``.
    project_manager:
        A ``ProjectManager`` instance for resolving project paths.
    """

    # SVI 2 Pro model_type keys — these route through the sliding-window
    # orchestrator when video_length exceeds sliding_window_size.
    SVI_MODEL_KEYS = frozenset({
        "i2v_2_2_svi_2_pro",
        "i2v_2_2_svi_2_pro_lightning_v2",
    })

    def __init__(self, wan_pipeline: Any, project_manager: Any) -> None:
        self.wan = wan_pipeline
        self.project_manager = project_manager
        self.input_handler = InputHandler()
        self.guidance_gen = GuidanceGenerator()
        self.output_handler = OutputHandler()

    # ------------------------------------------------------------------
    # SVI 2 Pro — sliding-window orchestration
    # ------------------------------------------------------------------

    def _should_use_sliding_window(self, project_config: ProjectConfig) -> bool:
        """Whether to route this generation through the sliding-window loop.

        Triggered for SVI 2 Pro models when the requested ``video_length``
        exceeds ``sliding_window_size`` (after latent-alignment), and for LTX
        (which upstream wan2gp also drives through an external sliding-window
        loop with one prompt per window).
        """
        model_type = getattr(project_config, "model_type", "") or ""
        if model_type not in self.SVI_MODEL_KEYS and not model_type.startswith("ltx_"):
            return False
        window_size = getattr(project_config, "sliding_window_size", 0) or 0
        video_length = getattr(project_config, "video_length", 0) or 0
        return window_size > 0 and video_length > window_size

    def _resolve_anchor_images(
        self,
        project_config: ProjectConfig,
        num_windows: int,
        fallback: Image.Image,
    ) -> List[Image.Image]:
        """Return the anchor (first-frame) image to use for each SVI window.

        - ``start`` mode: every window starts from the prior window's last
          surviving frame, so we return ``[None] * num_windows`` to signal
          "use chained continuation". Window 0 still uses *fallback*.
        - ``single`` mode: every window's start image is the configured
          anchor. Identity is locked across the entire shot.
        - ``per_window`` mode: ";"-separated paths used as the *first frame*
          of each window. Hard cuts between windows.
        - ``per_window_keyframes`` mode: anchors are *end* targets, not start
          frames — see :meth:`_resolve_keyframes`. We return chained-mode
          starts here so each window's first frame comes from the prior
          tail, producing smooth motion that converges to keyframe[N].
        """
        mode = getattr(project_config, "anchor_image_mode", "start") or "start"
        path_str = getattr(project_config, "anchor_image_path", "") or ""

        if mode == "single" and path_str:
            anchor = self.input_handler.load_image(path_str)
            return [anchor] * num_windows

        if mode == "per_window" and path_str:
            # Position-preserving: paths[i] is window i's anchor. An empty entry
            # (e.g. "p1;;p3") means "no anchor for this window" -> chain from the
            # prior tail, so the user can anchor only some windows. The last
            # non-empty entry clamps for windows past the end of the list.
            raw = [p.strip() for p in path_str.split(";")]
            if not any(raw):
                return [None] * num_windows
            last_idx = max(i for i, p in enumerate(raw) if p)
            images: List[Optional[Image.Image]] = []
            for i in range(num_windows):
                if i < len(raw):
                    p = raw[i]
                elif i >= last_idx:
                    p = raw[last_idx]  # clamp to last provided anchor
                else:
                    p = ""
                images.append(self.input_handler.load_image(p) if p else None)
            return images  # type: ignore[return-value]

        # "start" or "per_window_keyframes" — chained continuation. Window 0
        # uses fallback; subsequent windows use prior tail (handled in
        # _run_sliding_window).
        result: List[Optional[Image.Image]] = [None] * num_windows
        result[0] = fallback
        return result  # type: ignore[return-value]

    def _resolve_keyframes(
        self,
        project_config: ProjectConfig,
        num_windows: int,
    ) -> List[Optional[Image.Image]]:
        """Return per-window end-keyframe targets (or [None]*N if not active).

        Only meaningful in ``per_window_keyframes`` mode: each window must
        converge to ``keyframes[N]`` at its final frame, with smooth motion
        chained from the prior window's tail. Use case: stretching a fast
        action (balloon inflation, slow drink, transformation) across many
        windows by handing the model progressive stage images.
        """
        mode = getattr(project_config, "anchor_image_mode", "start") or "start"
        if mode != "per_window_keyframes":
            return [None] * num_windows

        path_str = getattr(project_config, "anchor_image_path", "") or ""
        paths = [p.strip() for p in path_str.split(";") if p.strip()]
        if not paths:
            return [None] * num_windows

        keyframes: List[Optional[Image.Image]] = []
        for i in range(num_windows):
            p = paths[i] if i < len(paths) else paths[-1]
            keyframes.append(self.input_handler.load_image(p))
        return keyframes

    @staticmethod
    def _compute_num_windows(
        total_frames: int, window_size: int, overlap: int, discard: int,
    ) -> int:
        """Mirror wan2gp's compute_sliding_window_no.

        Each window after the first contributes ``window_size - overlap -
        discard`` net new frames (the overlap blends with the previous window
        and the discarded tail is dropped). Window 0 contributes
        ``window_size - discard``.
        """
        if total_frames <= window_size:
            return 1
        net_per_window = max(window_size - overlap - discard, 1)
        remaining = total_frames - (window_size - discard)
        import math
        return 1 + math.ceil(remaining / net_per_window)

    # _seam_anchor_extension was removed in Phase 5. All call sites now use
    # supremediffusion.core.post_color_anchor.anchor_frames_for_chunk which
    # adds multi-mode reference resolution + configurable persistence. The
    # old fade-to-zero behavior is reproducible via persistence=0.0.

    @staticmethod
    def _match_image_to_source(
        image: Image.Image, source: Image.Image, method: str, strength: float = 0.8,
    ) -> Image.Image:
        """Pre-match *image*'s color profile to *source* before it's fed
        into the generation pipeline.

        Use case: in Video Extender / Mode 2, the dest frame and guidance
        video can come from a different camera/encoder than the source clip
        — different gamma, different range, different white balance. Without
        this, the model sees "go from color profile A (source last frame)
        toward color profile B (dest)" and the chain inherits B's profile.
        Pre-matching keeps everything in source's profile so the model only
        worries about *content* changes.
        """
        if image is None or source is None or strength <= 0:
            return image
        from supremediffusion.utils.color_correct import apply_color_match

        src_np = np.asarray(source.convert("RGB")).astype(np.float32) / 255.0
        tgt_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        try:
            matched = apply_color_match(tgt_np, src_np, method=method, strength=strength)
        except Exception as exc:
            logger.warning("Pre-match image_to_source failed: %s — using raw", exc)
            return image
        out = np.clip(matched * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    @staticmethod
    def _match_tensor_to_source(
        tensor: Any, source: Image.Image, method: str, strength: float = 0.8,
    ) -> Any:
        """Pre-match every frame of a guidance video tensor to *source*.

        *tensor* is shape ``(N, 3, H, W)`` in ``[0, 1]`` (the format
        :func:`GuidanceGenerator.load_guidance_video` returns). Returns a
        same-shape tensor with each frame statistically matched to the
        source frame's Lab color profile.
        """
        if tensor is None or source is None or strength <= 0:
            return tensor
        try:
            import torch as _torch
        except ImportError:
            return tensor
        from supremediffusion.utils.color_correct import apply_color_match

        src_np = np.asarray(source.convert("RGB")).astype(np.float32) / 255.0
        # tensor (N, 3, H, W) in [0,1] → (N, H, W, 3) numpy
        if hasattr(tensor, "cpu"):
            np_frames = tensor.detach().cpu().numpy()
        else:
            np_frames = np.asarray(tensor)
        np_frames = np.transpose(np_frames, (0, 2, 3, 1)).astype(np.float32)
        try:
            matched = apply_color_match(np_frames, src_np, method=method, strength=strength)
        except Exception as exc:
            logger.warning("Pre-match tensor_to_source failed: %s — using raw", exc)
            return tensor
        # Back to (N, 3, H, W) tensor of original dtype/device
        matched = np.clip(matched, 0.0, 1.0).astype(np.float32)
        out = np.transpose(matched, (0, 3, 1, 2))
        if hasattr(tensor, "cpu"):
            return _torch.from_numpy(out).to(dtype=tensor.dtype, device=tensor.device)
        return out

    @staticmethod
    def _warn_guidance_length(guidance_tensor: Any, expected_length: int, where: str) -> None:
        """Log a WARN when a guidance tensor's frame count materially differs
        from the requested ``video_length``.

        A guidance video that is shorter than the generation horizon leaves the
        tail windows un-guided; one that is much longer is silently truncated by
        the pipeline. Either case usually means the user fed a clip whose length
        doesn't match the run — surfacing it next to the ``wan.generate()``
        handoff makes the mismatch debuggable instead of producing quietly
        degraded motion. "Materially" = more than one frame off (small ±1
        rounding from VAE temporal stride is expected and not worth a warning).
        """
        if guidance_tensor is None:
            return
        try:
            actual = int(guidance_tensor.shape[0])
        except (AttributeError, IndexError, TypeError):
            return
        expected = int(expected_length or 0)
        if expected > 0 and abs(actual - expected) > 1:
            logger.warning(
                "%s: guidance tensor has %d frames but video_length=%d "
                "(%+d) — guidance and output horizon are misaligned",
                where, actual, expected, actual - expected,
            )

    @staticmethod
    def _slice_prompt_for_window(
        prompt: str,
        window_idx: int,
        window_size: int,
        fps: int,
    ) -> str:
        """Extract time-anchored prompt segments belonging to *window_idx*.

        Wan2GP's prompt format: ``(at N seconds: description)`` per line.
        For multi-window SVI runs we want each window to receive only the
        segments describing its own time slice — otherwise every window sees
        the entire shot's plan and the model tries to compress 15 seconds of
        action into its 5-second horizon (the user's "rushed action" issue).

        Segments inside the window's time range are kept and re-mapped so
        their seconds are *relative to the window start* (the model treats
        each window as a fresh generation). Segments outside the window are
        dropped.

        If the prompt contains no time anchors, returns it unchanged — so
        plain prompts still work; everyone in every window sees the same
        description.

        If the window has no segments (gap in the user's plan), falls back
        to the segment from the most-recent earlier second so the model has
        *some* guidance instead of silence.
        """
        import re

        if "(at " not in prompt or " seconds:" not in prompt:
            # No timed anchors. Upstream wan2gp's per-window model: a multi-line
            # prompt = one line per window, indexed by window with last-line
            # clamp (mirrors prompt_parser.split_prompt_units "W" mode). A
            # single-line prompt passes through unchanged — every window shares it.
            lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
            if len(lines) > 1:
                return lines[window_idx] if window_idx < len(lines) else lines[-1]
            return prompt

        # Match `(at N seconds: ...)` — the inner text may contain commas,
        # parens, anything except an unbalanced `)`. Use a lazy/non-greedy
        # match and require the closing paren.
        pattern = re.compile(
            r"\(\s*at\s+(\d+)\s+seconds?\s*:\s*(.+?)\s*\)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = [(int(m.group(1)), m.group(2).strip()) for m in pattern.finditer(prompt)]
        if not matches:
            return prompt

        seconds_per_window = max((window_size - 1) / max(fps, 1), 1.0)
        win_start = window_idx * seconds_per_window
        win_end = win_start + seconds_per_window

        in_window = [(sec, text) for sec, text in matches if win_start <= sec < win_end]

        if not in_window:
            # No segments in this window — find the most recent earlier
            # segment to give the model something to anchor on.
            earlier = [(sec, text) for sec, text in matches if sec < win_start]
            if earlier:
                last_sec, last_text = max(earlier, key=lambda p: p[0])
                return f"(at 0 seconds: {last_text})"
            # Or if no earlier, take the first segment overall.
            first_sec, first_text = matches[0]
            return f"(at 0 seconds: {first_text})"

        # Re-map seconds to be relative to window start
        out_lines = []
        offset = int(win_start)
        for sec, text in in_window:
            local = max(sec - offset, 0)
            out_lines.append(f"(at {local} seconds: {text})")
        return "\n".join(out_lines)

    def _run_sliding_window(
        self,
        project_path: Path,
        start_image: Image.Image,
        project_config: ProjectConfig,
        callback: Optional[Callable] = None,
        guidance_tensor: Optional[Any] = None,
        last_image: Optional[Image.Image] = None,
        return_frames: bool = False,
    ) -> Any:
        """Run multi-window SVI 2 Pro generation.

        Loops ``wan.generate()`` once per window, chains the start image
        between windows according to ``anchor_image_mode``, and stitches
        the per-window outputs together with overlap + discard trimming.

        Parameters
        ----------
        last_image:
            Optional terminal frame for first/last-frame (Mode 2) SVI runs.
            Passed to ``wan.generate()`` as ``last_frame`` only on the
            *final* window — earlier windows are unconstrained at their
            tail so the orchestrator can chain them freely.
        """
        from copy import copy as _shallow_copy

        window_size = int(getattr(project_config, "sliding_window_size", 81) or 81)
        overlap = int(getattr(project_config, "sliding_window_overlap", 0) or 0)
        discard = int(getattr(project_config, "sliding_window_discard_last_frames", 0) or 0)
        cc_strength = float(
            getattr(project_config, "sliding_window_color_correction_strength", 0.0) or 0.0
        )
        total_frames = int(getattr(project_config, "video_length", window_size) or window_size)
        # LTX has no latent-overlap prefill (its generate() lacks
        # overlap_frames/overlap_noise); it chains windows via the start image
        # only. Detect it once so the loop can skip those Wan-only mechanics.
        is_ltx = (getattr(project_config, "model_type", "") or "").startswith("ltx_")
        # Log label — this orchestrator is shared by SVI 2 Pro and LTX.
        win_tag = "LTX" if is_ltx else "SVI"

        if overlap >= window_size:
            overlap = max(window_size - 1, 0)
        if discard >= window_size - overlap:
            discard = max(window_size - overlap - 1, 0)

        # Net new frames each window contributes after overlap + discard trim.
        # The clamps above keep this ≥ 1, but guard explicitly so a degenerate
        # config can never make the window loop spin without advancing.
        net_new = window_size - overlap - discard
        if net_new <= 0:
            raise ValueError(
                f"SVI window config yields no net-new frames per window "
                f"(window_size={window_size}, overlap={overlap}, discard={discard})"
            )

        num_windows = self._compute_num_windows(
            total_frames, window_size, overlap, discard,
        )
        anchors = self._resolve_anchor_images(project_config, num_windows, start_image)
        keyframes = self._resolve_keyframes(project_config, num_windows)

        # Pre-match keyframes + per-window anchors to the start image's color
        # profile. Keyframes are end targets the model must converge to; if
        # they come from a different camera, the end-of-window content
        # inherits the keyframe's profile and the chain drifts. Same logic
        # applies to per-window start anchors. Strength is the configurable
        # pre-match strength (independent of the post-decode correction slider).
        cc_method = getattr(project_config, "color_correction_method", "mean-only-lab")
        prematch_strength = float(getattr(project_config, "anchor_prematch_strength", 0.8))
        if start_image is not None and prematch_strength > 0:
            keyframes = [
                self._match_image_to_source(k, start_image, cc_method, strength=prematch_strength)
                if k is not None else None
                for k in keyframes
            ]
            anchors = [
                self._match_image_to_source(a, start_image, cc_method, strength=prematch_strength)
                if a is not None and a is not start_image else a
                for a in anchors
            ]

        logger.info(
            "%s sliding window: %d windows of %d frames (overlap=%d, discard=%d, "
            "anchor_mode=%s) for total %d frames",
            win_tag, num_windows, window_size, overlap, discard,
            getattr(project_config, "anchor_image_mode", "start"), total_frames,
        )

        # Per-window config: each call generates exactly window_size frames.
        # We trim overlap + discard at the orchestrator level.
        win_cfg = _shallow_copy(project_config)
        win_cfg.video_length = window_size
        # Thread the sliding-window color-correction slider into the key the
        # post-anchor helper actually reads (color_correction_strength).
        # Without this anchor_frames_for_chunk would fall back to the default
        # 0.8 and the slider would only act as an on/off toggle.
        win_cfg.color_correction_strength = cc_strength

        overlap_noise = float(
            getattr(project_config, "sliding_window_overlap_noise", 0.0) or 0.0
        )

        # LTX self-manages LoRAs inside generate() (from ltx_lora_dir); the Wan
        # PEFT path here reads a different dir and would only leave a stale
        # manager that _remove_loras then runs against the LTX pipe. Skip it.
        if not is_ltx:
            self._apply_loras(project_config)
        all_frames: List[Image.Image] = []
        window_audio: List[str] = []  # LTX emits one audio track per window
        prev_window_tail: List[Image.Image] = []

        try:
            lora_cb = None if is_ltx else self._get_lora_step_callback()
            for window_idx in range(num_windows):
                if window_idx == 0:
                    win_start = anchors[0] if anchors[0] is not None else start_image
                else:
                    # If anchors[i] was supplied (single or per_window mode) use
                    # it; otherwise chain from the previous window's last frame.
                    win_start = (
                        anchors[window_idx]
                        if anchors[window_idx] is not None
                        else (prev_window_tail[-1] if prev_window_tail else start_image)
                    )

                # End-frame target. Per-window-keyframes mode applies a
                # keyframe to *every* window. Otherwise Mode 2's last_image is
                # the terminal target and applies to the final window only.
                final_window = window_idx == num_windows - 1
                if keyframes[window_idx] is not None:
                    win_last = keyframes[window_idx]
                elif final_window and last_image is not None:
                    win_last = last_image
                else:
                    win_last = None

                # Build overlap_frames for windows ≥ 1 — these get VAE-encoded
                # and injected into the start of the next window's noise tensor
                # so the model jointly denoises overlap+new frames.
                #
                # Skip overlap prefill in per_window mode: the user explicitly
                # asked for a different scene at this window, so injecting
                # prior frames as latent context would make the model try to
                # bridge unrelated content (worse than a clean cut).
                anchor_mode = getattr(project_config, "anchor_image_mode", "start") or "start"
                use_overlap = (
                    window_idx > 0
                    and overlap > 0
                    and prev_window_tail
                    and anchor_mode != "per_window"
                    and not is_ltx  # LTX chains via start image, not latent overlap
                )
                win_overlap = prev_window_tail[-overlap:] if use_overlap else None

                logger.info(
                    "%s window %d/%d: generating %d frames from %s%s%s",
                    win_tag, window_idx + 1, num_windows, window_size,
                    "anchor" if anchors[window_idx] is not None else "chained last frame",
                    f" + {len(win_overlap)} overlap frames" if win_overlap else "",
                    " → last_frame target" if win_last is not None else "",
                )

                # Slice the prompt for this window's time range. For plain
                # prompts (no `(at N seconds:)` anchors) this is a no-op.
                fps = getattr(project_config, "fps", 16) or 16
                win_prompt = self._slice_prompt_for_window(
                    project_config.prompt or "", window_idx, window_size, fps,
                )
                if win_prompt != project_config.prompt:
                    logger.info(
                        "%s window %d: time-sliced prompt (%d chars → %d chars)",
                        win_tag, window_idx + 1,
                        len(project_config.prompt or ""),
                        len(win_prompt),
                    )

                # Slice the guidance video per window. The full tensor maps
                # 1:1 to output frames, so window N starts at the cumulative
                # count of net-new frames emitted by prior windows. Net new
                # frames per window is (window_size - overlap - discard), so the
                # stride must include discard — otherwise the guidance slides
                # ahead of the output and the two desync when discard > 0.
                # Falls back to None for any window past the end of the guidance.
                win_guidance = None
                if guidance_tensor is not None:
                    g_start = window_idx * net_new
                    g_end = g_start + window_size
                    if g_start < guidance_tensor.shape[0]:
                        slice_ = guidance_tensor[g_start:g_end]
                        # Only pass if we got at least a few frames; tiny
                        # tail slices at the end of the guidance produce
                        # noisy results.
                        if slice_.shape[0] >= 4:
                            win_guidance = slice_

                self._warn_guidance_length(
                    win_guidance, window_size, f"{win_tag} window {window_idx}",
                )
                gen_kwargs = dict(
                    image=win_start,
                    prompt=win_prompt,
                    negative_prompt=project_config.negative_prompt,
                    project_config=win_cfg,
                    last_frame=win_last,
                    guidance_video=win_guidance,
                    callback=callback,
                    lora_step_callback=lora_cb,
                )
                if not is_ltx:
                    # LTX.generate() has no overlap_frames/overlap_noise params;
                    # passing them would raise TypeError. Wan accepts both.
                    gen_kwargs["overlap_frames"] = win_overlap
                    gen_kwargs["overlap_noise"] = overlap_noise
                window_frames = self.wan.generate(**gen_kwargs)

                if not isinstance(window_frames, list):
                    # Some pipeline paths return a tensor — normalise to list
                    # of PIL images via OutputHandler.frames_to_numpy → PIL.
                    np_frames = OutputHandler.frames_to_numpy(window_frames)
                    window_frames = [Image.fromarray(f) for f in np_frames]

                # Capture this window's audio track (LTX emits one ~window-long
                # WAV per generate()). Collected so the final mux spans the whole
                # video — otherwise only the last window's audio would be muxed
                # and ffmpeg -shortest would truncate the video to it.
                _wa = getattr(self.wan, "last_audio_path", None)
                if _wa and Path(_wa).is_file():
                    window_audio.append(_wa)

                # Drop the tail discard before color correcting / overlap-trim
                if discard > 0 and len(window_frames) > discard:
                    window_frames = window_frames[:-discard]

                # Inter-window color correction: routed through the unified
                # post_color_anchor helper which respects anchor_image_mode.
                # - "start"       → all_frames[0] (first generated frame)
                # - "single"      → cfg.anchor_image_path
                # - "per_window"  → rotating ';'-separated paths in
                #                   cfg.anchor_image_path (chunk_index=window_idx)
                # - "off"         → skipped
                # Default is "start" which matches the legacy behavior the
                # old code had hardcoded.
                if cc_strength > 0 and all_frames:
                    from supremediffusion.core.post_color_anchor import (
                        anchor_frames_for_chunk,
                    )
                    window_frames = anchor_frames_for_chunk(
                        window_frames,
                        config=win_cfg,
                        source_image=all_frames[0],
                        chunk_index=window_idx,
                        previous_last_frame=all_frames[-1] if all_frames else None,
                    )

                if window_idx == 0:
                    all_frames.extend(window_frames)
                elif use_overlap:
                    # The first `overlap` frames of this window are the
                    # latent-prefilled re-denoising of the previous window's
                    # tail. The model used them as temporal context, so the
                    # transition is already baked in. Drop them from output —
                    # we keep only the genuinely new content.
                    all_frames.extend(window_frames[overlap:])
                else:
                    # No prefill (per_window anchor mode): the window is
                    # independent, keep every frame. The cut between windows
                    # is intentional.
                    all_frames.extend(window_frames)

                # Snapshot this window's full output (post-discard) so the
                # next window can pull its overlap tail. Use the full window
                # rather than the trimmed `keep` so prev_window_tail[-overlap:]
                # always picks up the actual generated content.
                prev_window_tail = list(window_frames)

                if len(all_frames) >= total_frames:
                    break
        finally:
            if not is_ltx:
                self._remove_loras()

        # Final trim to exact requested length
        if len(all_frames) > total_frames:
            all_frames = all_frames[:total_frames]

        logger.info("%s sliding window complete: %d frames", win_tag, len(all_frames))

        # Stitch the per-window audio so the muxed track spans the full video.
        # Using only one window's track would let ffmpeg -shortest cut the whole
        # video down to that single window's duration (the "12s -> 4s" bug).
        clips = [a for a in window_audio if a and Path(a).is_file()]
        if len(clips) > 1:
            audio_path = self._concat_audio(clips)
            logger.info("%s: concatenated %d window audio tracks", win_tag, len(clips))
        elif clips:
            audio_path = clips[0]
        else:
            audio_path = getattr(self.wan, "last_audio_path", None)

        preview_path = self._make_preview_path(project_path)
        if return_frames:
            return all_frames, str(preview_path)
        self.output_handler.save_generation(
            all_frames, preview_path, fps=project_config.fps,
            audio_path=audio_path,
        )
        logger.info("%s preview saved: %s", win_tag, preview_path)
        return str(preview_path)

    @staticmethod
    def _concat_audio(paths: List[str]) -> Optional[str]:
        """Concatenate WAV clips (the per-window LTX audio) into one track via
        the ffmpeg concat filter. Returns the new file, or the last clip on
        failure (never raises — audio is best-effort)."""
        import subprocess
        try:
            out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            for p in paths:
                cmd += ["-i", p]
            n = len(paths)
            filt = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[a]"
            cmd += ["-filter_complex", filt, "-map", "[a]", out]
            subprocess.run(cmd, check=True, capture_output=True)
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio concat failed (%s); using last window's audio.", exc)
            return paths[-1] if paths else None

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def _apply_loras(self, project_config: ProjectConfig) -> None:
        """Apply LoRAs from project config to the Wan pipeline.

        Supports wan2gp-style phase multipliers (e.g. ``"1;0 0;1"``).
        When phase syntax is detected, LoRAs are loaded as unfused PEFT
        adapters with per-step weight scheduling.
        """
        lora_names = getattr(project_config, "activated_loras", [])
        if not lora_names:
            return

        # LTX handles its own LoRAs inside LTXVideoPipeline._setup_loras (resolved
        # against ltx_lora_dir and fused/threaded through LTX2.generate). This
        # Wan-path loader resolves against the Wan lora_dir, so for an LTX run it
        # would look in the wrong folder and spuriously warn "not found ...
        # skipping" for every LTX LoRA (e.g. union-control) — making it look
        # unloaded when it isn't. Bail out for the LTX backend.
        if hasattr(self.wan, "ltx2"):
            return

        # Ensure the pipeline is loaded before applying LoRAs
        if not self.wan.is_loaded:
            logger.info("Auto-loading pipeline before LoRA application.")
            self.wan.load()

        lora_dir = ""
        if hasattr(self.wan, "config"):
            lora_dir = getattr(self.wan.config, "model_paths", {}).get("lora_dir", "")
        if not lora_dir:
            logger.warning("LoRA(s) selected but no lora_dir configured; skipping.")
            return

        # Resolve filenames — the dropdown stores stems, we need full filenames
        lora_mgr = LoRAManager(lora_dir)
        available = lora_mgr.list_available()
        resolved = []
        for name in lora_names:
            match = None
            for avail in available:
                if avail == name or avail.rsplit(".", 1)[0] == name:
                    match = avail
                    break
            if match:
                resolved.append(match)
            else:
                logger.warning("LoRA '%s' not found in %s; skipping.", name, lora_dir)

        if not resolved:
            self._active_lora_mgr = lora_mgr
            return

        mult_str = getattr(project_config, "loras_multipliers", "") or ""
        num_phases = getattr(project_config, "guidance_phases", 1) or 1
        has_phase_syntax = ";" in mult_str

        if has_phase_syntax and num_phases >= 2:
            # --- Phase-aware LoRA loading ---
            num_steps = getattr(project_config, "num_inference_steps", 4)
            switch_threshold = getattr(project_config, "switch_threshold", 900) or 900

            # Compute timesteps to find the switch step index
            from supremediffusion.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
            import torch
            flow_shift = getattr(project_config, "flow_shift", 5.0) or 5.0
            sched = FlowUniPCMultistepScheduler(shift=float(flow_shift))
            sched.set_timesteps(num_steps, device="cpu")
            timesteps = sched.timesteps.tolist()

            switch_step = compute_switch_step(timesteps, switch_threshold)
            logger.info(
                "Phase LoRA: %d phases, switch_threshold=%d, switch_step=%d/%d, timesteps=%s",
                num_phases, switch_threshold, switch_step, num_steps, timesteps,
            )

            step_scales = parse_phase_multipliers(
                mult_str, len(resolved), num_steps,
                num_phases=num_phases, switch_step=switch_step,
            )

            lora_mgr.apply_loras_phased(self.wan.pipe, resolved, step_scales)
        else:
            # --- Simple constant multipliers ---
            mult_parts = [s.strip() for s in mult_str.replace(";", ",").split(",") if s.strip()]
            multipliers = []
            for p in mult_parts:
                try:
                    multipliers.append(float(p))
                except ValueError:
                    multipliers.append(1.0)
            while len(multipliers) < len(resolved):
                multipliers.append(1.0)
            multipliers = multipliers[:len(resolved)]
            lora_mgr.apply_loras(self.wan.pipe, resolved, multipliers)

        self._active_lora_mgr = lora_mgr

    def _get_lora_step_callback(self) -> Optional[Callable]:
        """Return a callback that updates LoRA adapter weights each step, or None."""
        mgr = getattr(self, "_active_lora_mgr", None)
        if mgr is None or not mgr.is_phased:
            return None

        def _lora_step_cb(pipe, step_index, timestep, cb_kwargs):
            # Update adapter weights for the NEXT step (callback runs after current step)
            mgr.set_step_scales(pipe, step_index + 1)
            return cb_kwargs

        return _lora_step_cb

    def _remove_loras(self) -> None:
        """Remove any LoRAs applied during generation."""
        mgr = getattr(self, "_active_lora_mgr", None)
        if mgr is not None and self.wan.pipe is not None:
            mgr.remove_loras(self.wan.pipe)
            self._active_lora_mgr = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_project_path(self, project_name: str) -> Path:
        """Return the project root directory."""
        return self.project_manager.get_project_path(project_name)

    def _make_preview_path(self, project_path: Path) -> Path:
        """Return a unique path inside the project outputs/ for a preview video."""
        outputs_dir = project_path / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return outputs_dir / f"preview_{ts}.mp4"

    def _make_guidance_path(self, project_path: Path, tag: str = "static") -> Path:
        """Return a path inside the project guidance/ directory."""
        guidance_dir = project_path / "guidance"
        guidance_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return guidance_dir / f"guidance_{tag}_{ts}.mp4"

    # ------------------------------------------------------------------
    # Mode 1: Image to video
    # ------------------------------------------------------------------

    def run_mode1(
        self,
        project_name: str,
        image_path: str,
        project_config: ProjectConfig,
        generate_guidance: bool = False,
        callback: Optional[Callable] = None,
        return_frames: bool = False,
        guidance_tensor: Optional[Any] = None,
        guidance_from_source: bool = False,
    ) -> str:
        """Run mode 1 (image-to-video) generation.

        Steps:
            1. Load input image.
            2. Optionally create a static guidance video from the image.
            3. Call the Wan pipeline to generate video frames.
            4. Save the result as a preview video.

        Returns the path to the preview video.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info("Mode 1: project=%s image=%s guidance=%s", project_name, image_path, generate_guidance)

        try:
            # 1. Load image
            inputs = self.input_handler.prepare_input_for_mode(1, image_path=image_path)
            image = inputs["image"]

            # 2. Guidance — use pre-loaded tensor if provided, else generate.
            # guidance_from_source arrives as a parameter: callers that built
            # the tensor from the source image (Generate tab static guidance)
            # declare it so the pre-match below is skipped.
            if guidance_tensor is None and generate_guidance:
                guidance_tensor = self.guidance_gen.build_static_tensor(
                    frame=image,
                    num_frames=project_config.video_length,
                )
                guidance_from_source = True
                logger.info("Static guidance tensor built from source frame")

            # 2a. Pre-match guidance video to the source image's color profile.
            # Static guidance built from the source image is already matched —
            # running rgb2lab/lab2rgb on 81 source-broadcast frames is wasted
            # work AND a ~10 GB CPU allocation that systemd-oomd kills on the
            # second generation once the 22B model is resident.
            cc_method = getattr(project_config, "color_correction_method", "mean-only-lab")
            prematch_strength = float(getattr(project_config, "anchor_prematch_strength", 0.8))
            if (
                guidance_tensor is not None
                and not guidance_from_source
                and prematch_strength > 0
            ):
                guidance_tensor = self._match_tensor_to_source(
                    guidance_tensor, image, cc_method, strength=prematch_strength,
                )
                logger.info("Mode 1: pre-matched guidance video to source color profile (method=%s)", cc_method)

            # 2b. SVI 2 Pro — multi-window orchestration
            if self._should_use_sliding_window(project_config):
                logger.info("Mode 1: routing through sliding-window orchestrator")
                return self._run_sliding_window(
                    project_path=project_path,
                    start_image=image,
                    project_config=project_config,
                    callback=callback,
                    guidance_tensor=guidance_tensor,
                    return_frames=return_frames,
                )

            # 3. Generate (with LoRA support)
            logger.info("Starting generation (mode 1)…")
            self._warn_guidance_length(
                guidance_tensor, getattr(project_config, "video_length", 0), "Mode 1",
            )
            self._apply_loras(project_config)
            try:
                lora_cb = self._get_lora_step_callback()
                result_frames = self.wan.generate(
                    image=image,
                    prompt=project_config.prompt,
                    negative_prompt=project_config.negative_prompt,
                    project_config=project_config,
                    guidance_video=guidance_tensor,
                    callback=callback,
                    lora_step_callback=lora_cb,
                )
            finally:
                self._remove_loras()

            # 4. Discard trailing frames if requested
            discard = getattr(project_config, "discard_last_frames", 0)
            if discard > 0 and isinstance(result_frames, list) and len(result_frames) > discard:
                logger.info("Discarding last %d frames (had %d)", discard, len(result_frames))
                result_frames = result_frames[:-discard]

            # 4b. Color anchor — unified post-step. Replaces the old flat
            # _color_correct_frames call. Picks reference per anchor_image_mode:
            #   'start' (default)  → the input source image (`image`)
            #   'single'           → anchor_image_path
            #   'per_window'       → ignored on single-chunk (chunk_index=0)
            #   'previous_last'    → also `image` (no prior chunk in Mode 1)
            #   'off'              → no correction
            # Persistence keeps correction strong across the whole clip
            # instead of fading to zero by frame 16 (VAE warm bias re-accumulating).
            cc_strength = getattr(project_config, "color_correction_strength", 0.0) or 0.0
            if cc_strength > 0 and isinstance(result_frames, list) and result_frames:
                from supremediffusion.core.post_color_anchor import (
                    anchor_frames_for_chunk,
                )
                logger.info(
                    "Mode 1: color anchor mode=%s strength=%.2f persistence=%.2f method=%s",
                    getattr(project_config, "anchor_image_mode", "start"),
                    cc_strength,
                    float(getattr(project_config, "color_anchor_persistence", 0.6) or 0.6),
                    getattr(project_config, "color_correction_method", "mean-only-lab"),
                )
                result_frames = anchor_frames_for_chunk(
                    result_frames,
                    config=project_config,
                    source_image=image,
                    chunk_index=0,
                    previous_last_frame=image,
                )

            # 5. Save preview (or return raw frames for post-processing)
            audio_path = getattr(self.wan, "last_audio_path", None)
            preview_path = self._make_preview_path(project_path)
            if return_frames:
                return result_frames, str(preview_path)
            self.output_handler.save_generation(
                result_frames, preview_path, fps=project_config.fps,
                audio_path=audio_path,
            )
            logger.info("Mode 1 preview saved: %s", preview_path)
            return str(preview_path)

        except Exception:
            logger.exception("Mode 1 generation failed")
            raise

    # ------------------------------------------------------------------
    # Mode 2: First + last frame
    # ------------------------------------------------------------------

    def run_mode2(
        self,
        project_name: str,
        first_frame_path: str,
        last_frame_path: str,
        project_config: ProjectConfig,
        generate_guidance: bool = False,
        guidance_frame: str = "first",
        callback: Optional[Callable] = None,
        return_frames: bool = False,
        guidance_tensor: Optional[Any] = None,
        guidance_from_source: bool = False,
    ) -> str:
        """Run mode 2 (first + last frame) generation.

        Steps:
            1. Load both frames.
            2. Optionally create a static guidance video from the chosen frame.
            3. Call the Wan pipeline with first frame, last frame, and optional guidance.
            4. Save the result as a preview video.

        Returns the path to the preview video (or (frames, path) if return_frames=True).
        """
        project_path = self._resolve_project_path(project_name)
        logger.info(
            "Mode 2: project=%s first=%s last=%s guidance=%s (frame=%s)",
            project_name, first_frame_path, last_frame_path, generate_guidance, guidance_frame,
        )

        try:
            # 1. Load frames
            inputs = self.input_handler.prepare_input_for_mode(
                2, first_frame_path=first_frame_path, last_frame_path=last_frame_path,
            )
            first_image = inputs["image"]
            last_image = inputs["last_frame"]

            # 2. Guidance — use pre-loaded tensor if provided, else generate.
            # guidance_from_source arrives as a parameter: callers that built
            # the tensor from a source frame declare it so the pre-match is
            # skipped (it's already in the source's color profile).
            guidance_path = None
            if guidance_tensor is None and generate_guidance:
                guide_frame = first_image if guidance_frame == "first" else last_image
                guidance_tensor = self.guidance_gen.build_static_tensor(
                    frame=guide_frame,
                    num_frames=project_config.video_length,
                )
                guidance_from_source = True
                logger.info("Static guidance tensor built from %s frame", guidance_frame)

            # 2a. Pre-match the LAST frame to the FIRST frame's color profile.
            # If the user picked first/last from different cameras / different
            # encoders, the model would be asked to bridge a color gap on top
            # of the content gap. Pre-matching keeps the chain in one profile.
            cc_method = getattr(project_config, "color_correction_method", "mean-only-lab")
            prematch_strength = float(getattr(project_config, "anchor_prematch_strength", 0.8))
            if last_image is not None and first_image is not None and prematch_strength > 0:
                last_image = self._match_image_to_source(
                    last_image, first_image, cc_method, strength=prematch_strength,
                )
                logger.info("Mode 2: pre-matched last frame to first frame's color profile (method=%s)", cc_method)
            # Skip when auto-built from the source — already matched, and the
            # rgb2lab/lab2rgb allocation gets the process OOM-killed on run 2
            # once the 22B model is resident in CPU RAM.
            if (
                guidance_tensor is not None
                and first_image is not None
                and not guidance_from_source
                and prematch_strength > 0
            ):
                guidance_tensor = self._match_tensor_to_source(
                    guidance_tensor, first_image, cc_method, strength=prematch_strength,
                )
                logger.info("Mode 2: pre-matched guidance video to first frame's color profile (method=%s)", cc_method)

            # 2b. SVI 2 Pro — multi-window orchestration. The terminal frame
            # is constrained on the final window only; earlier windows chain
            # freely so they can converge naturally toward last_image.
            if self._should_use_sliding_window(project_config):
                logger.info("Mode 2: routing through sliding-window orchestrator")
                return self._run_sliding_window(
                    project_path=project_path,
                    start_image=first_image,
                    project_config=project_config,
                    callback=callback,
                    guidance_tensor=guidance_tensor,
                    last_image=last_image,
                    return_frames=return_frames,
                )

            # 3. Generate (with LoRA support)
            logger.info("Starting generation (mode 2)…")
            self._warn_guidance_length(
                guidance_tensor, getattr(project_config, "video_length", 0), "Mode 2",
            )
            self._apply_loras(project_config)
            try:
                lora_cb = self._get_lora_step_callback()
                result_frames = self.wan.generate(
                    image=first_image,
                    prompt=project_config.prompt,
                    negative_prompt=project_config.negative_prompt,
                    project_config=project_config,
                    last_frame=last_image,
                    guidance_video=guidance_tensor,
                    callback=callback,
                    lora_step_callback=lora_cb,
                )
            finally:
                self._remove_loras()

            # 4. Discard trailing frames if requested
            discard = getattr(project_config, "discard_last_frames", 0)
            if discard > 0 and isinstance(result_frames, list) and len(result_frames) > discard:
                logger.info("Discarding last %d frames (had %d)", discard, len(result_frames))
                result_frames = result_frames[:-discard]

            # 4b. Color anchor — unified post-step (see Mode 1 comments). For
            # Mode 2 the reference defaults to ``first_image`` (the user's
            # supplied first frame). When anchor_image_mode='single' or
            # 'per_window' the user-configured path overrides.
            cc_strength = getattr(project_config, "color_correction_strength", 0.0) or 0.0
            if cc_strength > 0 and isinstance(result_frames, list) and result_frames:
                from supremediffusion.core.post_color_anchor import (
                    anchor_frames_for_chunk,
                )
                logger.info(
                    "Mode 2: color anchor mode=%s strength=%.2f persistence=%.2f method=%s",
                    getattr(project_config, "anchor_image_mode", "start"),
                    cc_strength,
                    float(getattr(project_config, "color_anchor_persistence", 0.6) or 0.6),
                    getattr(project_config, "color_correction_method", "mean-only-lab"),
                )
                result_frames = anchor_frames_for_chunk(
                    result_frames,
                    config=project_config,
                    source_image=first_image,
                    chunk_index=0,
                    previous_last_frame=first_image,
                )

            # 5. Save preview (or return raw frames for post-processing)
            audio_path = getattr(self.wan, "last_audio_path", None)
            preview_path = self._make_preview_path(project_path)
            if return_frames:
                return result_frames, str(preview_path)
            self.output_handler.save_generation(
                result_frames, preview_path, fps=project_config.fps,
                audio_path=audio_path,
            )
            logger.info("Mode 2 preview saved: %s", preview_path)
            return str(preview_path)

        except Exception:
            logger.exception("Mode 2 generation failed")
            raise

    # ------------------------------------------------------------------
    # Mode 3: Video to video (extract + re-encode)
    # ------------------------------------------------------------------

    def run_mode3(
        self,
        project_name: str,
        video_path: str,
        start_frame: int,
        end_frame: int,
        project_config: ProjectConfig,
        generate_guidance: bool = False,
        guidance_frame_idx: int = 0,
        callback: Optional[Callable] = None,
    ) -> str:
        """Run mode 3 (video-to-video / clip extraction).

        Steps:
            1. Extract the frame range from the source video.
            2. Optionally create a static guidance video from a specific frame.
            3. Encode the extracted frames as a new clip.

        Returns the path to the encoded clip.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info(
            "Mode 3: project=%s video=%s frames=%d–%d guidance=%s (idx=%d)",
            project_name, video_path, start_frame, end_frame, generate_guidance, guidance_frame_idx,
        )

        try:
            # 1. Extract frames
            inputs = self.input_handler.prepare_input_for_mode(
                3, video_path=video_path, start_frame=start_frame, end_frame=end_frame,
            )
            frames = inputs["frames"]

            if not frames:
                raise ValueError(f"No frames extracted from {video_path} [{start_frame}:{end_frame}]")

            # Mode 3 is pure extraction + re-encode — no generation happens, so
            # no guidance tensor is built (the previous build was dead: nothing
            # downstream consumed it). generate_guidance/guidance_frame_idx are
            # accepted for signature symmetry with modes 1/2 but unused here.

            # 2. Encode extracted frames as clip
            clip_path = self.output_handler.save_clip_to_project(
                frames, project_path, fps=project_config.fps,
            )
            logger.info("Mode 3 clip saved: %s", clip_path)

            if callback is not None:
                callback({"status": "complete", "clip_path": clip_path})

            return clip_path

        except Exception:
            logger.exception("Mode 3 extraction failed")
            raise

    # ------------------------------------------------------------------
    # Video Extension (Continue Video — wan2gp-style)
    # ------------------------------------------------------------------

    def run_extend(
        self,
        project_name: str,
        video_path: str,
        project_config: ProjectConfig,
        final_frame_path: Optional[str] = None,
        guidance_video_path: Optional[str] = None,
        guidance_tensor: Optional[Any] = None,
        start_image_path: Optional[str] = None,
        callback: Optional[Callable] = None,
        return_frames: bool = False,
        guidance_from_source: bool = False,
    ) -> str:
        """Continue a video by extracting the last frame as image_start and
        using the source video as guidance for smooth continuation.

        This follows the wan2gp approach:
        1. Extract the last frame from the source video as ``image_start``
        2. Use the source video as a guidance video for continuity blending
        3. Call wan.generate() — essentially Mode 1 (or Mode 2 if final_frame given)

        Parameters
        ----------
        video_path:
            Path to the source video to continue from.
        final_frame_path:
            Optional destination frame — if provided, uses first/last frame mode.
        guidance_video_path:
            Optional separate guidance video. If not provided, the source video
            itself is used as guidance.
        start_image_path:
            Optional lossless start frame image. When provided, used instead of
            extracting the last frame from *video_path*, avoiding a lossy
            decode round-trip that degrades detail over chained continuations.

        Returns the path to the generated preview video.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info(
            "Extend: project=%s video=%s final_frame=%s start_image=%s",
            project_name, video_path, final_frame_path, start_image_path,
        )

        try:
            # LTX-native extension via RetakePipeline (when available)
            if hasattr(self.wan, "extend_video"):
                logger.info("Using LTX-native video extension")
                # extend_video doesn't forward these inputs — warn so they
                # don't silently do nothing on an LTX extend.
                if final_frame_path:
                    logger.warning(
                        "LTX extend ignores final_frame_path=%s (not supported by "
                        "extend_video)", final_frame_path,
                    )
                if guidance_video_path:
                    logger.warning(
                        "LTX extend ignores guidance_video_path=%s (not supported by "
                        "extend_video)", guidance_video_path,
                    )
                if start_image_path:
                    logger.warning(
                        "LTX extend ignores start_image_path=%s (not supported by "
                        "extend_video)", start_image_path,
                    )
                result_frames = self.wan.extend_video(
                    video_path=video_path,
                    prompt=project_config.prompt,
                    negative_prompt=project_config.negative_prompt,
                    project_config=project_config,
                    callback=callback,
                )

                # Color anchor — the LTX short-circuit returned before the
                # shared post-anchor block below, so apply it here too. Resolve
                # the reference from the project's persistent source image.
                cc_strength = getattr(project_config, "color_correction_strength", 0.0) or 0.0
                if cc_strength > 0 and isinstance(result_frames, list) and result_frames:
                    from supremediffusion.core.post_color_anchor import (
                        anchor_frames_for_chunk,
                    )
                    source_image_obj = None
                    src_path = getattr(project_config, "image_path", "") or ""
                    if src_path:
                        try:
                            from pathlib import Path as _P
                            if _P(src_path).is_file():
                                source_image_obj = Image.open(src_path).convert("RGB")
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("LTX extend anchor source load failed: %s", exc)
                    logger.info(
                        "LTX extend: color anchor mode=%s strength=%.2f frames=%d",
                        getattr(project_config, "anchor_image_mode", "start"),
                        cc_strength, len(result_frames),
                    )
                    result_frames = anchor_frames_for_chunk(
                        result_frames,
                        config=project_config,
                        source_image=source_image_obj,
                        chunk_index=0,
                        previous_last_frame=None,
                    )

                audio_path = getattr(self.wan, "last_audio_path", None)
                preview_path = self._make_preview_path(project_path)
                if return_frames:
                    return result_frames, str(preview_path)
                self.output_handler.save_generation(
                    result_frames, preview_path, fps=project_config.fps,
                    audio_path=audio_path,
                )
                logger.info("LTX extension preview saved: %s", preview_path)
                return str(preview_path)

            # 1. Use provided lossless start image, or extract last frame
            #    from source video as image_start.
            #
            # Both paths SHOULD produce equivalent color:
            #  - start_image_path: PNG saved by VideoExtendWorker from raw
            #    post-CC frames (already full-range)
            #  - extract_single_frame: applies _png_export_filter → tv-to-pc
            #    range conversion driven by ffprobe (probe_color_range)
            #
            # If they DON'T match (e.g. saved PNG drifted from VAE imperfection
            # while the source video is the original clean reference), the
            # color stats logged below let you diagnose chained-extension drift.
            if start_image_path:
                image_start = self.input_handler.load_image(start_image_path)
                logger.info("Using provided lossless start image: %s", start_image_path)
            else:
                info = self.input_handler.validate_video(video_path)
                last_idx = max(info["num_frames"] - 1, 0)
                image_start = self.input_handler.extract_single_frame(video_path, last_idx)
                logger.info("Extracted last frame (idx=%d) from source video", last_idx)

            # Log color stats so chained-extension drift is visible in the
            # logs. If the mean shifts noticeably between runs, the saved PNG
            # (start_image_path) is drifting away from the source clip's
            # actual last frame, and you'd want to switch to re-extracting
            # from the encoded video instead.
            try:
                _src_arr = np.asarray(image_start.convert("RGB"))
                _r, _g, _b = _src_arr.mean(axis=(0, 1))
                logger.info(
                    "Extend: image_start color stats — mean RGB (%.1f, %.1f, %.1f)  range [%d..%d]",
                    _r, _g, _b, int(_src_arr.min()), int(_src_arr.max()),
                )
            except Exception:
                pass

            # 2. Load guidance video.
            #    - Caller-supplied tensor wins (skips encode/decode roundtrip).
            #    - Explicit guidance video: always use it.
            #    - No explicit guidance + denoise < 1.0: use source video as guidance.
            # guidance_from_source arrives as a parameter — callers that built
            # the tensor from the source declare it to skip the pre-match.
            if guidance_tensor is not None:
                logger.info(
                    "Using caller-supplied guidance tensor (shape=%s) — skipping codec roundtrip",
                    tuple(guidance_tensor.shape),
                )
            elif guidance_video_path:
                try:
                    guidance_tensor = self.guidance_gen.load_guidance_video(guidance_video_path)
                    logger.info("Loaded explicit guidance video from %s", guidance_video_path)
                except Exception as exc:
                    logger.warning("Failed to load guidance video %s: %s", guidance_video_path, exc)
            elif project_config.denoising_strength < 1.0:
                try:
                    guidance_tensor = self.guidance_gen.load_guidance_video(video_path)
                    guidance_from_source = True
                    logger.info("Loaded source video as guidance from %s (denoise < 1.0)", video_path)
                except Exception as exc:
                    logger.warning("Failed to load source as guidance %s: %s", video_path, exc)

            # 3. Load optional final frame for first/last frame mode
            last_image = None
            if final_frame_path:
                last_image = self.input_handler.load_image(final_frame_path)
                logger.info("Using final frame: %s", final_frame_path)

            # 3a. Pre-match dest frame + guidance video to the source video's
            # color profile. Without this the model is told to interpolate
            # between two color spaces and the extension inherits the dest's
            # profile instead of the source's. Strength is the configurable
            # pre-match strength because this is technical normalization, not
            # creative shaping.
            cc_method = getattr(project_config, "color_correction_method", "mean-only-lab")
            prematch_strength = float(getattr(project_config, "anchor_prematch_strength", 0.8))
            if last_image is not None and prematch_strength > 0:
                last_image = self._match_image_to_source(
                    last_image, image_start, cc_method, strength=prematch_strength,
                )
                logger.info("Extend: pre-matched final frame to source color profile (method=%s)", cc_method)
            # Skip when guidance was loaded from the source video itself — the
            # color profile is already the source's, and the rgb2lab/lab2rgb
            # allocation gets the process OOM-killed on run 2 once the 22B
            # model is resident in CPU RAM.
            if (
                guidance_tensor is not None
                and not guidance_from_source
                and prematch_strength > 0
            ):
                guidance_tensor = self._match_tensor_to_source(
                    guidance_tensor, image_start, cc_method, strength=prematch_strength,
                )
                logger.info("Extend: pre-matched guidance video to source color profile (method=%s)", cc_method)

            # 3b. SVI 2 Pro continuation — when the user wants to chain many
            # SVI windows off an existing video, use sliding-window from the
            # extracted last frame. Single-window extends fall through to the
            # original code path so they don't pay the orchestrator overhead.
            if self._should_use_sliding_window(project_config):
                logger.info("Extend: routing through sliding-window orchestrator")
                return self._run_sliding_window(
                    project_path=project_path,
                    start_image=image_start,
                    project_config=project_config,
                    callback=callback,
                    guidance_tensor=guidance_tensor,
                    last_image=last_image,
                    return_frames=return_frames,
                )

            # 4. Generate with LoRA support
            logger.info("Starting generation (extend)…")
            self._warn_guidance_length(
                guidance_tensor, getattr(project_config, "video_length", 0), "Extend",
            )
            self._apply_loras(project_config)
            try:
                lora_cb = self._get_lora_step_callback()
                result_frames = self.wan.generate(
                    image=image_start,
                    prompt=project_config.prompt,
                    negative_prompt=project_config.negative_prompt,
                    project_config=project_config,
                    last_frame=last_image,
                    guidance_video=guidance_tensor,
                    callback=callback,
                    lora_step_callback=lora_cb,
                )
            finally:
                self._remove_loras()

            # 5. Color anchor (unified post-step) — picks the reference based
            #    on project_config.anchor_image_mode:
            #      - "start"          → original source image (project.image_path)
            #      - "single"         → anchor_image_path file
            #      - "per_window"     → rotating list of ';'-separated paths
            #      - "previous_last"  → image_start (legacy behavior)
            #      - "off"            → skip anchoring
            #    Default 'start' fights compound drift across long chains by
            #    keeping every chunk locked to the original palette rather than
            #    the (already-drifted) previous chunk's last frame.
            cc_strength = getattr(project_config, "color_correction_strength", 0.0)
            if cc_strength > 0 and isinstance(result_frames, list) and result_frames:
                from supremediffusion.core.post_color_anchor import (
                    anchor_frames_for_chunk,
                )
                # Try to locate the project's persistent original source.
                source_image_obj = None
                src_path = getattr(project_config, "image_path", "") or ""
                if src_path:
                    try:
                        from pathlib import Path as _P
                        if _P(src_path).is_file():
                            source_image_obj = Image.open(src_path).convert("RGB")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("anchor source load failed: %s", exc)
                # image_start IS the previous chunk's last frame on VE calls,
                # so the legacy 'previous_last' mode keeps the same behavior.
                logger.info(
                    "Extend: anchor mode=%s strength=%.2f persistence=%.2f method=%s frames=%d",
                    getattr(project_config, "anchor_image_mode", "start"),
                    cc_strength,
                    float(getattr(project_config, "color_anchor_persistence", 0.6) or 0.6),
                    getattr(project_config, "color_correction_method", "mean-only-lab"),
                    len(result_frames),
                )
                result_frames = anchor_frames_for_chunk(
                    result_frames,
                    config=project_config,
                    source_image=source_image_obj,
                    chunk_index=0,  # single-chunk extend; multi-chunk path uses sliding-window code
                    previous_last_frame=image_start,
                )

            # 5b. Discard trailing frames if requested
            discard = getattr(project_config, "discard_last_frames", 0)
            if discard > 0 and isinstance(result_frames, list) and len(result_frames) > discard:
                logger.info("Discarding last %d frames (had %d)", discard, len(result_frames))
                result_frames = result_frames[:-discard]

            # 6. Save preview
            audio_path = getattr(self.wan, "last_audio_path", None)
            preview_path = self._make_preview_path(project_path)
            if return_frames:
                return result_frames, str(preview_path)
            self.output_handler.save_generation(
                result_frames, preview_path, fps=project_config.fps,
                audio_path=audio_path,
            )
            logger.info("Extension preview saved: %s", preview_path)
            return str(preview_path)

        except Exception:
            logger.exception("Video extension failed")
            raise

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    def run(self, project_name: str, **kwargs):
        """Dispatch to the appropriate mode handler based on kwargs.

        Expects ``mode`` (int 1/2/3) in kwargs or infers from input keys.
        All project config fields can be passed as kwargs and will be used
        to construct a ProjectConfig.

        If ``return_frames=True`` is passed, returns ``(frames, preview_path)``
        instead of saving and returning just the path.
        """
        mode = kwargs.pop("mode", None)
        return_frames = kwargs.pop("return_frames", False)

        # Infer mode from input keys if not explicitly provided
        if mode is None:
            if "video_path" in kwargs and "start_frame" in kwargs:
                mode = 3
            elif "first_frame_path" in kwargs and "last_frame_path" in kwargs:
                mode = 2
            elif "image_path" in kwargs:
                mode = 1
            else:
                raise ValueError("Cannot determine mode from provided inputs.")

        # Extract input-specific args
        image_path = kwargs.pop("image_path", None)
        first_frame_path = kwargs.pop("first_frame_path", None)
        last_frame_path = kwargs.pop("last_frame_path", None)
        video_path = kwargs.pop("video_path", None)
        start_frame = kwargs.pop("start_frame", 0)
        end_frame = kwargs.pop("end_frame", 48)
        generate_guidance = kwargs.pop("guidance_video_source", "none") != "none"
        guidance_frame = kwargs.pop("guidance_video_frame", "first")
        callback = kwargs.pop("callback", None)

        # Pop extension kwargs (used by Video Extender tab / mode 3)
        extend = kwargs.pop("extend", False)
        extend_frames = kwargs.pop("extend_frames", 81)
        extend_final_frame = kwargs.pop("extend_final_frame", None)
        # Lossless start image for chained extension (not a config field).
        start_image_path = kwargs.pop("start_image_path", None)
        # Pop guidance path (passed separately, not a config field)
        guidance_video_path = kwargs.pop("guidance_video_path", None)

        # Pop guidance checkbox fields (saved in config but not pipeline params)
        kwargs.pop("use_guidance_m1", None)
        kwargs.pop("generate_static_m1", None)
        kwargs.pop("use_guidance_m2", None)
        kwargs.pop("generate_static_m2", None)

        # If a guidance video path was provided, load it as a tensor
        guidance_tensor = None
        if guidance_video_path:
            try:
                guidance_tensor = self.guidance_gen.load_guidance_video(guidance_video_path)
                logger.info("Loaded guidance video from %s", guidance_video_path)
            except Exception as exc:
                logger.warning("Failed to load guidance video %s: %s", guidance_video_path, exc)

        # Build ProjectConfig from remaining kwargs
        config_fields = {k: v for k, v in kwargs.items() if k in ProjectConfig.__dataclass_fields__}
        project_config = ProjectConfig(**config_fields)

        # Video extension — route to run_extend() instead of mode 3
        if extend and video_path:
            return self.run_extend(
                project_name, video_path, project_config,
                final_frame_path=extend_final_frame,
                guidance_video_path=guidance_video_path,
                guidance_tensor=guidance_tensor,
                start_image_path=start_image_path,
                callback=callback, return_frames=return_frames,
            )

        if mode == 1:
            return self.run_mode1(
                project_name, image_path, project_config,
                generate_guidance=generate_guidance, callback=callback,
                return_frames=return_frames,
                guidance_tensor=guidance_tensor,
            )
        elif mode == 2:
            return self.run_mode2(
                project_name, first_frame_path, last_frame_path, project_config,
                generate_guidance=generate_guidance, guidance_frame=guidance_frame,
                callback=callback, return_frames=return_frames,
                guidance_tensor=guidance_tensor,
            )
        elif mode == 3:
            return self.run_mode3(
                project_name, video_path, int(start_frame), int(end_frame),
                project_config, generate_guidance=generate_guidance,
                guidance_frame_idx=0, callback=callback,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    # ------------------------------------------------------------------
    # Accept / promote a preview to a project clip
    # ------------------------------------------------------------------

    def accept_clip(self, preview_path: str, project_name: str) -> str:
        """Copy a preview video into the project's ``clips/`` directory.

        Returns the final clip path.
        """
        project_path = self._resolve_project_path(project_name)
        logger.info("Accepting clip: %s -> project '%s'", preview_path, project_name)

        try:
            clip_path = self.output_handler.save_clip_to_project(
                preview_path, project_path,
            )
            logger.info("Clip accepted: %s", clip_path)
            return clip_path
        except Exception:
            logger.exception("Failed to accept clip %s", preview_path)
            raise
