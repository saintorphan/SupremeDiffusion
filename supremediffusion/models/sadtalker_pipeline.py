"""SadTalker (audio-driven talking head) pipeline wrapper.

SadTalker pins numpy 1.21 / torch 1.12 / basicsr 1.4 / kornia 0.6 — versions
incompatible with the modern stack the rest of SupremeDiffusion runs on. Rather
than try to share a venv (which would break LTX / Wan / MimicMotion / ADetailer),
this wrapper invokes SadTalker through its own venv as a subprocess.

The bridge call is JSON-args in, single-output-mp4-path out. All controls
exposed by SadTalker's ``SadTalker.test()`` (the gradio_demo entry we modified
to surface every knob) are passed through.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Upstream paths — the clone we set up earlier in this project.
_UPSTREAM_REPO = Path.home() / "Projects" / "SadTalker"
_UPSTREAM_VENV_PY = _UPSTREAM_REPO / "venv" / "bin" / "python"


# The bridge script — written into a temp file at runtime. Kept here as a
# string so the wrapper is self-contained (no extra files to deploy).
_BRIDGE_SCRIPT = r'''
"""Bridge: read JSON args on stdin, run SadTalker, print output mp4 path."""
import json, sys, os, traceback

# numpy 1.21 + sadtalker chain expects this attribute that pre-existed in old
# numpy. Some sub-deps reference it; harmless to alias.
import numpy as _np
if not hasattr(_np, "VisibleDeprecationWarning"):
    _np.VisibleDeprecationWarning = DeprecationWarning  # type: ignore[attr-defined]

REPO = os.environ.get("SADTALKER_REPO")
sys.path.insert(0, REPO)
os.chdir(REPO)  # SadTalker uses cwd-relative checkpoint paths

from src.gradio_demo import SadTalker  # noqa: E402

def main() -> int:
    args = json.loads(sys.stdin.read())
    talker = SadTalker(
        checkpoint_path=args.get("checkpoint_path", "checkpoints"),
        config_path=args.get("config_path", "src/config"),
        lazy_load=True,
    )
    output = talker.test(
        source_image=args["source_image"],
        driven_audio=args.get("driven_audio") or None,
        preprocess=args.get("preprocess", "extfull"),
        still_mode=args.get("still_mode", False),
        enhancer_method=args.get("enhancer_method") or None,
        background_enhancer=args.get("background_enhancer") or None,
        batch_size=int(args.get("batch_size", 2)),
        size=int(args.get("size", 256)),
        pose_style=int(args.get("pose_style", 0)),
        exp_scale=float(args.get("exp_scale", 1.0)),
        use_ref_video=args.get("use_ref_video", False),
        ref_video=args.get("ref_video") or None,
        ref_info=args.get("ref_info") or None,
        use_idle_mode=args.get("use_idle_mode", False),
        length_of_audio=int(args.get("length_of_audio", 0)),
        use_blink=args.get("use_blink", True),
        input_yaw_str=args.get("input_yaw_str", "") or "",
        input_pitch_str=args.get("input_pitch_str", "") or "",
        input_roll_str=args.get("input_roll_str", "") or "",
        face3dvis=args.get("face3dvis", False),
        result_dir=args.get("result_dir", "./results/"),
    )
    # Final line is the mp4 path — caller parses last line.
    print("\nSADTALKER_OUTPUT::" + str(output))
    return 0

try:
    sys.exit(main())
except Exception as e:
    print("SADTALKER_ERROR::" + str(e), file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
'''


class SadTalkerPipeline:
    """Subprocess wrapper around SadTalker's gradio_demo.SadTalker.test().

    Lifecycle is shaped like the other pipeline wrappers (load / unload /
    is_loaded) for AppState consistency, but `load()` does nothing heavy —
    every ``generate()`` spawns a fresh subprocess. SadTalker uses ``lazy_load``
    so model weights load on first inference inside the subprocess.
    """

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self._loaded: bool = False
        self._bridge_path: Optional[str] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Verify upstream + venv are present. No model load happens here."""
        if not _UPSTREAM_REPO.is_dir():
            raise RuntimeError(
                f"SadTalker upstream repo not found at {_UPSTREAM_REPO}. "
                "Clone with: git clone https://github.com/OpenTalker/SadTalker "
                f"{_UPSTREAM_REPO}"
            )
        if not _UPSTREAM_VENV_PY.is_file():
            raise RuntimeError(
                f"SadTalker venv python not found at {_UPSTREAM_VENV_PY}. "
                "Set up SadTalker's venv per its README first."
            )
        if not (_UPSTREAM_REPO / "checkpoints").is_dir():
            logger.warning(
                "SadTalker checkpoints dir missing at %s/checkpoints. "
                "Weights will fail to load on first generate.",
                _UPSTREAM_REPO,
            )
        # Write the bridge script to a stable temp location once.
        if self._bridge_path is None or not Path(self._bridge_path).exists():
            tf = tempfile.NamedTemporaryFile(
                mode="w", suffix="_sadtalker_bridge.py", delete=False
            )
            tf.write(_BRIDGE_SCRIPT)
            tf.flush()
            tf.close()
            self._bridge_path = tf.name
        self._loaded = True

    def unload(self) -> None:
        """No persistent state to release — subprocess teardown is per-call."""
        if self._bridge_path and Path(self._bridge_path).exists():
            try:
                Path(self._bridge_path).unlink()
            except OSError:
                pass
        self._bridge_path = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    def generate(
        self,
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
        result_dir: str | None = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        abort_cb: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Run a single SadTalker inference, return path to the output mp4."""
        if not self._loaded:
            self.load()
        if not source_image or not os.path.isfile(source_image):
            raise RuntimeError(f"Source image not found: {source_image}")

        if result_dir is None:
            result_dir = str(Path.cwd() / "results")
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        args = {
            "source_image": str(source_image),
            "driven_audio": str(driven_audio) if driven_audio else None,
            "preprocess": preprocess,
            "still_mode": bool(still_mode),
            "enhancer_method": enhancer_method or None,
            "background_enhancer": "realesrgan" if background_enhancer else None,
            "batch_size": int(batch_size),
            "size": int(size),
            "pose_style": int(pose_style),
            "exp_scale": float(exp_scale),
            "use_ref_video": bool(use_ref_video),
            "ref_video": str(ref_video) if ref_video else None,
            "ref_info": ref_info or None,
            "use_idle_mode": bool(use_idle_mode),
            "length_of_audio": int(length_of_audio),
            "use_blink": bool(use_blink),
            "input_yaw_str": input_yaw_str or "",
            "input_pitch_str": input_pitch_str or "",
            "input_roll_str": input_roll_str or "",
            "face3dvis": bool(face3dvis),
            "result_dir": str(result_dir),
            "checkpoint_path": "checkpoints",
            "config_path": "src/config",
        }

        env = os.environ.copy()
        env["SADTALKER_REPO"] = str(_UPSTREAM_REPO)
        # Avoid the subprocess inheriting our venv's PYTHONPATH.
        env.pop("PYTHONPATH", None)

        if progress_cb:
            progress_cb(0.05, "Spawning SadTalker subprocess...")

        proc = subprocess.Popen(
            [str(_UPSTREAM_VENV_PY), self._bridge_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        try:
            proc.stdin.write(json.dumps(args))
            proc.stdin.close()
        except BrokenPipeError as exc:
            proc.kill()
            raise RuntimeError(f"SadTalker bridge died on launch: {exc}") from exc

        # Stream stdout for the user. SadTalker prints progress lines like
        # 'audio2exp::  10%|...' which is enough to surface.
        output_path: str | None = None
        last_emit = time.time()
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("SADTALKER_OUTPUT::"):
                output_path = line.removeprefix("SADTALKER_OUTPUT::").strip()
                continue
            if abort_cb and abort_cb():
                proc.terminate()
                raise InterruptedError("Aborted by user")
            now = time.time()
            if progress_cb and now - last_emit > 0.5:
                progress_cb(0.5, line[:120])
                last_emit = now
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            raise RuntimeError(f"SadTalker failed (rc={proc.returncode}):\n{stderr[:2000]}")

        if not output_path or not os.path.isfile(output_path):
            raise RuntimeError("SadTalker finished but no output path was returned.")

        if progress_cb:
            progress_cb(1.0, "Complete")
        return output_path
