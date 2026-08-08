"""SadTalker worker — audio-driven talking head via subprocess bridge."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from sdqt.utils.codec import configured_codec_args

from .base import BaseWorker

logger = logging.getLogger(__name__)


class SadTalkerWorker(BaseWorker):
    """Runs the SadTalker subprocess pipeline. Re-encodes output through the
    project's configured codec so it integrates with the timeline."""

    def __init__(
        self,
        state: Any,
        project_name: str,
        *,
        source_image: str,
        driven_audio: str | None = None,
        preprocess: str = "extfull",
        still_mode: bool = False,
        enhancer_method: str | None = None,
        background_enhancer: bool = False,
        batch_size: int = 2,
        size: int = 256,
        pose_style: int = 0,
        exp_scale: float = 1.0,
        use_ref_video: bool = False,
        ref_video: str | None = None,
        ref_info: str | None = None,
        use_idle_mode: bool = False,
        length_of_audio: int = 0,
        use_blink: bool = True,
        input_yaw_str: str = "",
        input_pitch_str: str = "",
        input_roll_str: str = "",
        face3dvis: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._project_name = project_name
        self._kwargs = dict(
            source_image=source_image,
            driven_audio=driven_audio,
            preprocess=preprocess,
            still_mode=still_mode,
            enhancer_method=enhancer_method,
            background_enhancer=background_enhancer,
            batch_size=batch_size,
            size=size,
            pose_style=pose_style,
            exp_scale=exp_scale,
            use_ref_video=use_ref_video,
            ref_video=ref_video,
            ref_info=ref_info,
            use_idle_mode=use_idle_mode,
            length_of_audio=length_of_audio,
            use_blink=use_blink,
            input_yaw_str=input_yaw_str,
            input_pitch_str=input_pitch_str,
            input_roll_str=input_roll_str,
            face3dvis=face3dvis,
        )

    def do_work(self) -> str:
        self.status.emit("Preparing SadTalker pipeline...")
        self.progress.emit(0.0, "Loading")
        if self._state.sadtalker_pipeline is None:
            self._state.load_sadtalker_pipelines()
        pipeline = self._state.sadtalker_pipeline
        if pipeline is None:
            raise RuntimeError("SadTalker pipeline failed to initialise.")

        project_root = self._state.project_manager.get_project_path(self._project_name)
        outputs = Path(project_root) / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        intermediate_dir = outputs / f".sadtalker_tmp_{ts}"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        def _progress(frac: float, desc: str) -> None:
            self.progress.emit(frac, desc)

        def _aborted() -> bool:
            return self.is_aborted

        try:
            raw_output = pipeline.generate(
                result_dir=str(intermediate_dir),
                progress_cb=_progress,
                abort_cb=_aborted,
                **self._kwargs,
            )
        except InterruptedError:
            shutil.rmtree(intermediate_dir, ignore_errors=True)
            raise

        # Re-encode through the project's configured codec so the result
        # blends with the rest of the timeline (LTX / Wan / MimicMotion all
        # write through configured_codec_args()).
        final_path = str(outputs / f"sadtalker_{ts}.mp4")
        self.status.emit("Re-encoding to project codec...")
        self.progress.emit(0.95, "Encoding mp4")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", raw_output,
            *configured_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            final_path,
        ]
        try:
            self._run_ffmpeg(cmd)
        except Exception as exc:  # noqa: BLE001
            # If re-encode fails, just copy the original. Don't lose the work.
            logger.warning("SadTalker re-encode failed (%s) — saving raw output.", exc)
            shutil.copy(raw_output, final_path)
        finally:
            shutil.rmtree(intermediate_dir, ignore_errors=True)

        self.progress.emit(1.0, "Complete")
        return final_path
