"""Background worker for downloading models from HuggingFace Hub or direct URLs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from sdqt.workers.base import BaseWorker
from .registry import ModelEntry

logger = logging.getLogger(__name__)


class ModelDownloadWorker(BaseWorker):
    """Download a list of models with progress reporting.

    Emits ``progress(fraction, description)`` as each file completes.
    On success, ``finished_ok`` emits a list of (ModelEntry, local_path) tuples.
    """

    def __init__(
        self,
        models: list[ModelEntry],
        models_root: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._models = models
        self._models_root = models_root

    def do_work(self) -> list[tuple[ModelEntry, str]]:
        results: list[tuple[ModelEntry, str]] = []
        total = len(self._models)

        for i, model in enumerate(self._models):
            if self.is_aborted:
                raise InterruptedError("Download cancelled by user")

            size_mb = f" ({model.size_bytes / 1024**2:.0f} MB)" if model.size_bytes else ""
            label = f"Downloading {model.name}{size_mb} ({i + 1}/{total})"
            self.progress.emit(i / total, label)
            self.status.emit(label)
            logger.info("%s", label)

            try:
                local_path = self._download_one(model)
                results.append((model, str(local_path)))
            except InterruptedError:
                raise
            except Exception as exc:
                logger.error("Failed to download %s: %s", model.name, exc)
                raise RuntimeError(f"Failed to download {model.name}: {exc}") from exc

        self.progress.emit(1.0, "All downloads complete.")
        return results

    def _download_one(self, model: ModelEntry) -> Path:
        """Download a single model entry. Returns the local path."""
        dest = self._models_root / model.local_subdir
        dest.mkdir(parents=True, exist_ok=True)

        if model.url:
            return self._download_url(model, dest)
        elif model.repo_id:
            return self._download_hf(model, dest)
        else:
            raise ValueError(f"No repo_id or url for model {model.name}")

    def _download_hf(self, model: ModelEntry, dest: Path) -> Path:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for model downloads. "
                "Install with: pip install huggingface-hub"
            )

        if self.is_aborted:
            raise InterruptedError()

        if model.filename:
            # Single file download — download to HF cache first, then copy
            # to dest.  Passing subfolder + local_dir to hf_hub_download
            # nests the subfolder inside local_dir, so we avoid that.
            repo_path = (
                f"{model.subfolder}/{model.filename}"
                if model.subfolder
                else model.filename
            )
            cached = hf_hub_download(
                repo_id=model.repo_id,
                filename=repo_path,
            )
            target = dest / model.filename
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, target)
            return target
        else:
            # Full subfolder / repo download.
            # snapshot_download places repo files at local_dir/<subfolder>/,
            # but we need them at dest (= models_root / local_subdir).
            # Strategy: download into a temp staging dir, then move the
            # subfolder contents into dest.
            if model.subfolder:
                # Download each file in the subfolder individually to
                # dest.  snapshot_download + allow_patterns is unreliable
                # across huggingface_hub versions.
                from huggingface_hub import list_repo_files
                repo_files = list_repo_files(model.repo_id)
                prefix = model.subfolder.rstrip("/") + "/"
                sub_files = [f for f in repo_files if f.startswith(prefix)]

                # Also grab related files from sibling folders that are
                # needed at load time (e.g. image_processor/preprocessor_config.json
                # is required by CLIPImageProcessor but lives outside image_encoder/)
                _SIBLING_FILES = {
                    "image_encoder": ["image_processor/preprocessor_config.json"],
                }
                for extra in _SIBLING_FILES.get(model.subfolder, []):
                    if extra in repo_files:
                        sub_files.append(extra)

                if sub_files:
                    dest.mkdir(parents=True, exist_ok=True)
                    for repo_path in sub_files:
                        if repo_path.startswith(prefix):
                            rel = repo_path[len(prefix):]
                        else:
                            # Sibling file — use just the filename
                            rel = Path(repo_path).name
                        cached = hf_hub_download(
                            repo_id=model.repo_id,
                            filename=repo_path,
                        )
                        target = dest / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cached, target)
                    logger.info(
                        "Downloaded %d files for %s into %s",
                        len(sub_files), model.name, dest,
                    )
                else:
                    logger.warning(
                        "No files found under %s in repo %s for %s",
                        model.subfolder, model.repo_id, model.name,
                    )
            else:
                snapshot_download(
                    repo_id=model.repo_id,
                    local_dir=str(dest),
                    local_dir_use_symlinks=False,
                )
            return dest

    def _download_url(self, model: ModelEntry, dest: Path) -> Path:
        """Download from a direct URL."""
        import urllib.request

        filename = model.filename or model.url.split("/")[-1]
        target = dest / filename

        if target.exists():
            return target

        self.status.emit(f"Downloading {model.name} from URL...")

        tmp = target.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(model.url, str(tmp))
            shutil.move(str(tmp), str(target))
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

        return target
