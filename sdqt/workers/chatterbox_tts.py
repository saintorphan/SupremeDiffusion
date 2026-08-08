"""Chatterbox TTS worker — generates speech using Resemble AI Chatterbox.

Runs Chatterbox in its own venv (~/Projects/chatterbox/.venv) via subprocess
to avoid pkg_resources / torch version conflicts with the app's Python 3.12.
"""

from __future__ import annotations

import json
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

_CB_DIR = Path.home() / "Projects" / "chatterbox"
_CB_PYTHON = _CB_DIR / ".venv" / "bin" / "python3"


class ChatterboxTTSWorker(BaseWorker):
    """Generate speech audio from text using Chatterbox TTS.

    Runs in Chatterbox's own venv via subprocess.
    """

    def __init__(
        self,
        text: str,
        output_path: str,
        *,
        ref_audio: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        app_state: Any = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._text = text
        self._output_path = output_path
        self._ref_audio = ref_audio
        self._exaggeration = exaggeration
        self._cfg_weight = cfg_weight
        self._app_state = app_state

    def do_work(self) -> str:
        if not _CB_PYTHON.is_file():
            raise RuntimeError(
                f"Chatterbox venv not found at {_CB_PYTHON}. "
                "Clone https://github.com/resemble-ai/chatterbox into "
                "~/Projects/chatterbox and create a venv with: "
                "pip install chatterbox-tts"
            )

        # Free VRAM so the subprocess has GPU memory available
        if self._app_state is not None:
            self.progress.emit(0.0, "Freeing VRAM for Chatterbox...")
            try:
                self._app_state.unload_pipelines()
            except Exception:
                pass

        self.progress.emit(0.1, "Starting Chatterbox TTS (subprocess)...")

        params = json.dumps({
            "text": self._text.strip(),
            "output_path": self._output_path,
            "ref_audio": self._ref_audio or "",
            "exaggeration": self._exaggeration,
            "cfg_weight": self._cfg_weight,
        })

        script = textwrap.dedent("""\
            import json, sys
            from pathlib import Path

            params = json.loads(sys.argv[1])

            print("PROGRESS:0.15:Loading Chatterbox model...", flush=True)
            from chatterbox.tts import ChatterboxTTS
            import torchaudio
            import torch

            model = ChatterboxTTS.from_pretrained(device="cuda")

            print("PROGRESS:0.30:Generating speech...", flush=True)

            gen_kwargs = dict(
                exaggeration=params["exaggeration"],
                cfg_weight=params["cfg_weight"],
            )
            ref = params["ref_audio"]
            if ref and Path(ref).is_file():
                gen_kwargs["audio_prompt_path"] = ref

            wav = model.generate(params["text"], **gen_kwargs)

            print("PROGRESS:0.85:Saving audio...", flush=True)

            if wav is None or wav.numel() == 0:
                print("ERROR:Chatterbox generated no audio output.", flush=True)
                sys.exit(1)

            # Ensure 2D shape [channels, samples]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            Path(params["output_path"]).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(params["output_path"], wav.cpu(), model.sr)

            dur = wav.shape[-1] / model.sr
            print(f"PROGRESS:1.0:Done \\u2014 {dur:.1f}s of audio.", flush=True)
            print(f"RESULT:{params['output_path']}", flush=True)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        """)

        proc = subprocess.Popen(
            [str(_CB_PYTHON), "-c", script, params],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(_CB_DIR),
        )

        result_path = None
        error_msg = None

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

            if self.is_aborted:
                proc.kill()
                proc.wait()
                raise InterruptedError("Aborted by user")

        proc.wait()

        if error_msg:
            raise RuntimeError(error_msg)

        if proc.returncode != 0:
            stderr_text = ""
            try:
                stderr_text = proc.stderr.read().strip()
            except Exception:
                pass
            if stderr_text:
                lines = [l for l in stderr_text.split("\n") if l.strip()]
                tail = lines[-1] if lines else stderr_text[:200]
                raise RuntimeError(f"Chatterbox subprocess failed: {tail}")
            raise RuntimeError(
                f"Chatterbox subprocess failed (exit code {proc.returncode})"
            )

        if result_path and Path(result_path).is_file():
            return result_path

        raise RuntimeError(
            "Chatterbox subprocess completed but no output file produced."
        )
