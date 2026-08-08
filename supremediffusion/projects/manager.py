"""Project CRUD and path management for Supreme Diffusion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from supremediffusion.config.project_config import ProjectConfig

_SUBDIRS = ("inputs", "guidance", "clips", "outputs", "trims", "images", "scenes")
_DEFAULT_PROJECT = "_default"


class ProjectManager:
    """Create, list, rename and delete projects under a shared root."""

    def __init__(self, projects_root: Path | str | None = None) -> None:
        if projects_root is None:
            projects_root = Path("~/.supremediffusion/projects").expanduser()
        self.root = Path(projects_root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Guarantee the _default project always exists.
        if not (self.root / _DEFAULT_PROJECT).exists():
            self.create_project(_DEFAULT_PROJECT)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_projects(self) -> List[str]:
        """Return sorted names of every project directory."""
        return sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )

    def get_project_path(self, name: str) -> Path:
        """Return the root directory for the given project."""
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"Project '{name}' does not exist.")
        return path

    def get_project_config_path(self, name: str) -> Path:
        """Return the path to *project.json* for the given project."""
        return self.get_project_path(name) / "project.json"

    def get_subdir(self, name: str, subdir: str) -> Path:
        """Return the path to a standard subdirectory inside a project.

        *subdir* must be one of: inputs, guidance, clips, outputs.
        """
        if subdir not in _SUBDIRS:
            raise ValueError(
                f"Invalid subdir '{subdir}'. Must be one of {_SUBDIRS}."
            )
        return self.get_project_path(name) / subdir

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def create_project(self, name: str) -> Path:
        """Create a new project with default subdirectories and config.

        Returns the project directory path.
        """
        project_dir = self.root / name
        if project_dir.exists():
            raise FileExistsError(f"Project '{name}' already exists.")

        project_dir.mkdir(parents=True)
        for sub in _SUBDIRS:
            (project_dir / sub).mkdir()

        config = ProjectConfig()
        config.save(project_dir / "project.json")

        return project_dir

    def rename_project(self, old_name: str, new_name: str) -> Path:
        """Rename *old_name* to *new_name*. Returns the new path."""
        old_path = self.get_project_path(old_name)
        new_path = self.root / new_name

        if old_name == _DEFAULT_PROJECT:
            raise ValueError("Cannot rename the _default project.")
        if new_path.exists():
            raise FileExistsError(f"Project '{new_name}' already exists.")

        old_path.rename(new_path)
        return new_path

    def copy_project(self, source_name: str, new_name: str) -> Path:
        """Copy *source_name* verbatim to *new_name*. Returns the new path."""
        source_path = self.get_project_path(source_name)
        new_path = self.root / new_name
        if new_path.exists():
            raise FileExistsError(f"Project '{new_name}' already exists.")
        shutil.copytree(source_path, new_path)
        return new_path

    def delete_project(self, name: str) -> None:
        """Permanently delete a project and all its contents."""
        if name == _DEFAULT_PROJECT:
            raise ValueError("Cannot delete the _default project.")

        path = self.get_project_path(name)
        shutil.rmtree(path)
