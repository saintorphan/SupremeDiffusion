"""Dia TTS worker — generates speech using Nari Labs Dia 1.6B.

Runs Dia in its own venv (~/Projects/dia/.venv) via subprocess to avoid
torch version conflicts. The app's torch 2.10 produces degraded audio
from Dia which was built for torch 2.6.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_DIA_DIR = Path.home() / "Projects" / "dia"
_DIA_PYTHON = _DIA_DIR / ".venv" / "bin" / "python3"
_MODEL_ID = "nari-labs/Dia-1.6B-0626"
_SAMPLE_RATE = 44100

# Dia prints a progress line every 86 decode steps.
# With max_tokens=3072, that's ~35 lines — enough for responsive abort.
_STEP_RE = re.compile(r"generate step (\d+):")


class DiaTTSWorker(BaseWorker):
    """Generate speech audio from text/dialogue using Dia 1.6B.

    Runs in Dia's own venv via subprocess to ensure correct torch version.
    """

    def __init__(
        self,
        text: str,
        output_path: str,
        *,
        ref_audio: str | None = None,
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.2,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 45,
        speed_factor: float = 0.94,
        app_state: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._output_path = output_path
        self._ref_audio = ref_audio
        self._max_tokens = max_tokens
        self._cfg_scale = cfg_scale
        self._temperature = temperature
        self._top_p = top_p
        self._cfg_filter_top_k = cfg_filter_top_k
        self._speed_factor = speed_factor
        self._app_state = app_state

    def do_work(self) -> str:
        if not _DIA_PYTHON.is_file():
            raise RuntimeError(
                f"Dia venv not found at {_DIA_PYTHON}. "
                "Clone https://github.com/nari-labs/dia into ~/Projects/dia "
                "and create a venv with its requirements."
            )

        # Free VRAM on the worker thread (not UI thread) so the subprocess
        # has GPU memory available
        if self._app_state is not None:
            self.progress.emit(0.0, "Freeing VRAM for Dia...")
            try:
                self._app_state.unload_pipelines()
            except Exception:
                pass

        # Ensure text starts with [S1]
        text = self._text.strip()
        if not text.startswith("[S1]") and not text.startswith("[S2]"):
            text = "[S1] " + text

        # Auto-scale max_tokens to text length to prevent runaway generation.
        # Dia uses ~100 codec tokens per second of speech, and English speech
        # averages ~15 characters/second.  We estimate duration from the text
        # (stripping speaker tags and nonverbal markers), add a generous 2x
        # headroom, and clamp to [768, user_max].
        plain = re.sub(r"\[S[12]\]|\([^)]*\)", "", text)
        est_chars = len(plain.strip())
        est_seconds = max(est_chars / 15.0, 2.0)  # at least 2s
        est_tokens = int(est_seconds * 100 * 2.0)  # 2x headroom
        max_tokens = max(768, min(est_tokens, self._max_tokens))
        logger.info("Dia: %d chars → est %.1fs → %d max_tokens (user cap %d)",
                     est_chars, est_seconds, max_tokens, self._max_tokens)

        self.progress.emit(0.1, "Starting Dia TTS (subprocess)...")

        # Build the generation script to run in Dia's venv
        params = json.dumps({
            "text": text,
            "output_path": self._output_path,
            "ref_audio": self._ref_audio or "",
            "max_tokens": max_tokens,
            "cfg_scale": self._cfg_scale,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "cfg_filter_top_k": self._cfg_filter_top_k,
            "speed_factor": self._speed_factor,
            "model_id": _MODEL_ID,
        })

        # Dia verbose=True prints step progress every 86 tokens to stdout.
        # We parse these for progress updates and abort checking.
        script = textwrap.dedent("""\
            import json, sys, os
            import numpy as np
            import soundfile as sf
            from pathlib import Path

            params = json.loads(sys.argv[1])

            # Add dia repo to path
            dia_dir = os.path.expanduser("~/Projects/dia")
            if dia_dir not in sys.path:
                sys.path.insert(0, dia_dir)

            from dia.model import Dia

            print("PROGRESS:0.15:Loading Dia model...", flush=True)
            model = Dia.from_pretrained(params["model_id"], compute_dtype="float16")

            print("PROGRESS:0.25:Generating speech...", flush=True)

            gen_kwargs = dict(
                max_tokens=params["max_tokens"],
                cfg_scale=params["cfg_scale"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                cfg_filter_top_k=params["cfg_filter_top_k"],
                use_torch_compile=False,
                verbose=True,
            )
            ref = params["ref_audio"]
            if ref and Path(ref).is_file():
                gen_kwargs["audio_prompt"] = ref

            output = model.generate(params["text"], **gen_kwargs)

            print("PROGRESS:0.85:Processing audio...", flush=True)

            import torch
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if output is None:
                print("ERROR:Dia generated no audio output.", flush=True)
                sys.exit(1)

            audio = np.asarray(output, dtype=np.float32)
            if audio.size == 0:
                print("ERROR:Dia generated empty audio.", flush=True)
                sys.exit(1)

            # Normalize to -0.95..+0.95
            peak = np.max(np.abs(audio))
            if peak < 1e-6:
                print("ERROR:Dia generated only silence.", flush=True)
                sys.exit(1)
            audio = audio * (0.95 / peak)

            # Trim trailing silence — use a low absolute threshold on the
            # normalized audio so quiet tails are trimmed but speech is kept
            abs_audio = np.abs(audio)
            silence_thresh = 0.005
            above = np.where(abs_audio > silence_thresh)[0]
            if len(above) > 0:
                # Keep 300ms of padding after last audible sample
                pad_samples = int(0.3 * 44100)
                end = min(len(audio), above[-1] + pad_samples)
                audio = audio[:end]

            # Speed adjustment
            spd = params["speed_factor"]
            if abs(spd - 1.0) > 0.02:
                orig_len = len(audio)
                new_len = int(orig_len / spd)
                if new_len > 0 and new_len != orig_len:
                    indices = np.linspace(0, orig_len - 1, new_len)
                    audio = np.interp(indices, np.arange(orig_len), audio)

            # Save as int16 wav
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            Path(params["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            sf.write(params["output_path"], audio_int16, 44100, subtype="PCM_16")

            dur = len(audio_int16) / 44100
            print(f"PROGRESS:1.0:Done \\u2014 {dur:.1f}s of audio.", flush=True)
            print(f"RESULT:{params['output_path']}", flush=True)
        """)

        proc = subprocess.Popen(
            [str(_DIA_PYTHON), "-c", script, params],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(_DIA_DIR),
        )

        # Stream stdout for progress updates.
        # Dia verbose mode prints "generate step N: ..." every 86 tokens,
        # giving us a heartbeat to check abort and report progress.
        result_path = None
        error_msg = None
        max_tokens = self._max_tokens

        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            if line.startswith("PROGRESS:"):
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    try:
                        self.progress.emit(float(parts[1]), parts[2])
                    except ValueError:
                        pass
            elif line.startswith("RESULT:"):
                result_path = line[7:]
            elif line.startswith("ERROR:"):
                error_msg = line[6:]
            else:
                # Parse Dia verbose step lines for progress
                m = _STEP_RE.match(line)
                if m:
                    step = int(m.group(1))
                    # Map steps to 0.25–0.85 range (generation phase)
                    frac = 0.25 + 0.6 * min(step / max_tokens, 1.0)
                    self.progress.emit(frac, f"Generating... step {step}")

            if self.is_aborted:
                proc.kill()
                proc.wait()
                raise InterruptedError("Aborted by user")

        proc.wait()

        if error_msg:
            raise RuntimeError(error_msg)

        if proc.returncode != 0:
            # Capture stderr for diagnostics
            stderr_text = ""
            try:
                stderr_text = proc.stderr.read().strip()
            except Exception:
                pass
            if stderr_text:
                lines = [l for l in stderr_text.split("\n") if l.strip()]
                tail = lines[-1] if lines else stderr_text[:200]
                raise RuntimeError(f"Dia subprocess failed: {tail}")
            raise RuntimeError(f"Dia subprocess failed (exit code {proc.returncode})")

        if result_path and Path(result_path).is_file():
            return result_path

        raise RuntimeError("Dia subprocess completed but no output file produced.")
