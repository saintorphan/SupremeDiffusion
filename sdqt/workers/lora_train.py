"""Worker for running kohya LoRA training as a managed subprocess."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

# Parse kohya training output for progress
_STEP_RE = re.compile(
    r"steps?:\s*(\d+)/(\d+).*?loss[=:\s]*([\d.]+)",
    re.IGNORECASE,
)
_LR_RE = re.compile(r"lr[=:\s]*([\d.eE+-]+)")
_EPOCH_RE = re.compile(r"epoch\s*(\d+)")
_SAVING_RE = re.compile(r"saving.*?(\S+\.safetensors)", re.IGNORECASE)


def find_kohya_script() -> str | None:
    """Find the kohya sd-scripts training script.

    Searches:
    1. third_party/sd-scripts/ (bundled)
    2. ~/sd-scripts/ (user-installed)
    3. Common install locations
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "third_party" / "sd-scripts",
        Path.home() / "sd-scripts",
        Path.home() / "kohya_ss" / "sd-scripts",
    ]
    for base in candidates:
        script = base / "sdxl_train_network.py"
        if script.is_file():
            return str(script)
        # Also check for the older train_network.py (SD 1.5)
        script = base / "train_network.py"
        if script.is_file():
            return str(script)
    return None


class LoRATrainWorker(BaseWorker):
    """Run kohya training as a subprocess, parse output for progress.

    Emits:
        progress(float, str): fraction 0-1, description text
        status(str): status messages
        loss_update(int, float): (step, loss) for the loss graph
        finished_ok(str): path to the trained .safetensors file
        error(str): error message
    """

    # Custom signal for loss graph — we'll emit via progress with a parseable format
    # The tab will parse step/loss from the description string

    def __init__(
        self,
        config_path: str,
        kohya_script: str,
        output_path: str,
        is_sdxl: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._kohya_script = kohya_script
        self._output_path = output_path
        self._is_sdxl = is_sdxl
        self._loss_history: list[tuple[int, float]] = []

    @property
    def loss_history(self) -> list[tuple[int, float]]:
        return self._loss_history

    def do_work(self) -> str:
        # Determine which script to use
        script = self._kohya_script
        if self._is_sdxl and "sdxl_train_network" not in script:
            # Try to find the SDXL variant
            base = Path(script).parent
            sdxl_script = base / "sdxl_train_network.py"
            if sdxl_script.is_file():
                script = str(sdxl_script)

        cmd = [
            sys.executable, script,
            "--config_file", self._config_path,
        ]

        self.status.emit(f"Starting training: {Path(script).name}")
        logger.info("Training command: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(script).parent),
        )

        last_step = 0
        total_steps = 0

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            # Log everything to the console
            logger.info("[kohya] %s", line)

            # Parse step/loss progress
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                total = int(m.group(2))
                loss = float(m.group(3))
                total_steps = total
                last_step = step
                self._loss_history.append((step, loss))

                # Parse LR if present
                lr_str = ""
                lr_m = _LR_RE.search(line)
                if lr_m:
                    lr_str = f"  LR: {lr_m.group(1)}"

                frac = step / total if total > 0 else 0
                desc = f"Step {step}/{total}  Loss: {loss:.5f}{lr_str}"
                self.progress.emit(frac, desc)

            # Check for saving
            save_m = _SAVING_RE.search(line)
            if save_m:
                self.status.emit(f"Saved checkpoint: {save_m.group(1)}")

            # Check for epoch
            epoch_m = _EPOCH_RE.search(line)
            if epoch_m and not m:
                self.status.emit(f"Epoch {epoch_m.group(1)}")

            # Abort check
            if self.is_aborted:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("Training aborted by user")

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Training failed (exit code {process.returncode})")

        self.progress.emit(1.0, "Training complete.")

        # Find the output file
        output = Path(self._output_path)
        if output.is_file():
            return str(output)

        # Search for it in the output directory
        out_dir = output.parent
        safetensors = sorted(out_dir.glob("*.safetensors"), key=lambda f: f.stat().st_mtime)
        if safetensors:
            return str(safetensors[-1])

        return str(output)
