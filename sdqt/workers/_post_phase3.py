"""Shared helper for invoking the unified PostProcessPipeline from any worker.

Used by GenerationWorker (Mode 1/2/3), VideoExtendWorker, LongshotChunkWorker,
LongshotRenderWorker, PoseAnimateWorker, and any future generation worker.
Each worker calls :func:`apply_phase3_postprocessing` at the end of its
``do_work()`` to apply the user-selected Quality preset's post-processing
chain (color anchor / upscale / RIFE) on the output video file.

This module exists to DRY the Phase 3 integration — without it, the same
30-line block would be copy-pasted into every worker.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def apply_phase3_postprocessing(
    *,
    result: Any,
    config: Any,
    global_config: Any,
    project_name: str,
    state: Any,
    progress_emit: Callable[[float, str], None],
    status_emit: Callable[[str], None],
    is_aborted: Callable[[], bool],
    fps_override: Optional[int] = None,
) -> Any:
    """Apply the configured Phase 3 post-processing pipeline to ``result``.

    Args:
        result: path-or-not — only acts on mp4/mkv strings. Anything else is
            returned unchanged.
        config: ProjectConfig (reads ``output_quality``, ``upscale_model``,
            ``rife_multiplier``, ``fps``).
        global_config: GlobalConfig (reads ``model_paths.upscaler_dir``).
        project_name: passed to the pipeline.
        state: AppState reference for the pipeline's state attribute.
        progress_emit / status_emit / is_aborted: forwarded to the pipeline
            so it can stream progress back to the worker's signals.
        fps_override: explicit fps. Falls back to ``config.fps`` then 16.

    Returns the (possibly new) path. On any error, returns ``result`` unchanged
    with a warning logged.
    """
    try:
        return _apply(
            result=result, config=config, global_config=global_config,
            project_name=project_name, state=state,
            progress_emit=progress_emit, status_emit=status_emit,
            is_aborted=is_aborted, fps_override=fps_override,
        )
    except Exception as exc:  # noqa: BLE001
        # Surface the skip to the user instead of only logging. Without this
        # a dead upscale/RIFE step (e.g. missing model, bad config) looked
        # like a working "Quality" preset that silently produced raw output.
        logger.warning("Phase 3 post-processing failed: %s — keeping raw result", exc)
        try:
            status_emit(f"Quality post-processing skipped: {exc}")
        except Exception:  # noqa: BLE001
            pass
        return result


def _apply(
    *,
    result: Any,
    config: Any,
    global_config: Any,
    project_name: str,
    state: Any,
    progress_emit: Callable[[float, str], None],
    status_emit: Callable[[str], None],
    is_aborted: Callable[[], bool],
    fps_override: Optional[int],
) -> Any:
    if not (isinstance(result, str) and result.endswith((".mp4", ".mkv"))):
        return result

    quality = (getattr(config, "output_quality", "standard") or "standard").lower()
    if quality not in ("quality", "master"):
        return result  # draft / standard handled by earlier pipeline stages

    from sdqt.workers.post_pipeline import build_from_quality_preset

    # Resolve upscale model path from chkpts/upscalers/
    upscale_model_name = (
        getattr(config, "upscale_model", "4x-UltraSharp.pth") or "4x-UltraSharp.pth"
    )
    upscale_dir = ""
    if global_config is not None:
        mp = getattr(global_config, "model_paths", {}) or {}
        upscale_dir = mp.get("upscaler_dir", "") or ""
    candidates = [
        Path(upscale_dir) / upscale_model_name if upscale_dir else None,
        Path("/data/shared/ai/chkpts/upscalers") / upscale_model_name,
        Path.home() / "Projects" / "SupremeDiffusionQt" / "models" / "upscalers" / upscale_model_name,
    ]
    upscale_path = next((str(c) for c in candidates if c and c.is_file()), None)
    if upscale_path is None:
        logger.warning(
            "Phase 3: upscale model %s not found — skipping upscale", upscale_model_name,
        )
        if quality == "quality":
            # Quality preset is upscale-only; nothing else to do. Tell the user
            # why instead of silently returning the un-upscaled video.
            try:
                status_emit(
                    f"Quality post-processing skipped: upscale model "
                    f"'{upscale_model_name}' not found"
                )
            except Exception:  # noqa: BLE001
                pass
            return result

    rife_mult = float(getattr(config, "rife_multiplier", 2.0) or 2.0)

    pipeline = build_from_quality_preset(
        quality,
        state=state,
        project_name=project_name,
        global_config=global_config,
        source_image=None,  # color anchor already applied earlier in core pipeline
        upscale_model_path=upscale_path,
        rife_multiplier=rife_mult,
        progress_cb=lambda f, d: progress_emit(0.95 + 0.05 * f, d),
        status_cb=status_emit,
        abort_cb=is_aborted,
    )

    fps = int(fps_override) if fps_override else int(getattr(config, "fps", 16) or 16)
    return pipeline.run(result, fps=fps)
