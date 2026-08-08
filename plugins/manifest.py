"""Plugin manifest loader and validator.

Every plugin directory must contain a ``plugin.json`` manifest.  The manifest
is validated BEFORE any Python is imported — a bad manifest means the plugin
is skipped entirely, protecting the app from broken code.

Manifest fields
---------------
Required:
  name          str   Display name
  version       str   Semver-ish plugin version
  entry_point   str   Python file containing the plugin class (e.g. "plugin.py")

Optional:
  author          str           Author name/handle
  description     str           One-line description
  min_app_version str           Minimum SupremeDiffusion version (e.g. "2.2.0")
  tab_label       str           Tab label in the UI (defaults to *name*)
  requires        list[str]     Pipeline keys needed (e.g. ["sd_pipeline", "wan_pipeline"])
  content_schema  str           Content schema version this plugin's JSONs follow
  icon            str           Emoji or relative path to icon image
  tags            list[str]     Searchable tags
  homepage        str           URL for docs / source
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Current app version — used for min_app_version checks
_APP_VERSION: str = "0.0.0"

def set_app_version(version: str) -> None:
    """Call once at startup to set the running app version."""
    global _APP_VERSION
    _APP_VERSION = version


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse '2.2.0' → (2, 2, 0). Ignores non-numeric suffixes."""
    parts: list[int] = []
    for segment in v.split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    """Parsed and validated plugin manifest."""

    # Required
    name: str
    version: str
    entry_point: str

    # Optional
    author: str = ""
    description: str = ""
    min_app_version: str = ""
    tab_label: str = ""
    requires: list[str] = field(default_factory=list)
    content_schema: str = ""
    icon: str = ""
    tags: list[str] = field(default_factory=list)
    homepage: str = ""

    # Internal — set by the loader, not from JSON
    plugin_dir: Path = field(default_factory=lambda: Path("."))


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ManifestError(Exception):
    """Raised when a plugin.json is invalid."""


# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {"name", "version", "entry_point"}

def load_manifest(plugin_dir: Path) -> PluginManifest:
    """Load and validate ``plugin.json`` from *plugin_dir*.

    Raises :class:`ManifestError` with a human-readable message on failure.
    """
    manifest_path = plugin_dir / "plugin.json"
    if not manifest_path.is_file():
        raise ManifestError(
            f"Missing plugin.json in {plugin_dir.name}/\n"
            f"Every plugin must have a plugin.json manifest."
        )

    # ── Parse JSON ──
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"Invalid JSON in {plugin_dir.name}/plugin.json: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json must be a JSON object, "
            f"got {type(data).__name__}"
        )

    # ── Check required fields ──
    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json missing required fields: "
            f"{', '.join(sorted(missing))}\n"
            f"Required: {', '.join(sorted(_REQUIRED_FIELDS))}"
        )

    # ── Type checks ──
    for fld in ("name", "version", "entry_point"):
        if not isinstance(data[fld], str) or not data[fld].strip():
            raise ManifestError(
                f"{plugin_dir.name}/plugin.json: '{fld}' must be a non-empty string"
            )

    # ── entry_point must point to an existing .py file ──
    entry = data["entry_point"]
    if not entry.endswith(".py"):
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json: entry_point must end with .py, "
            f"got '{entry}'"
        )
    entry_path = plugin_dir / entry
    if not entry_path.is_file():
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json: entry_point '{entry}' not found.\n"
            f"Expected at: {entry_path}"
        )

    # ── min_app_version check ──
    min_ver = data.get("min_app_version", "")
    if min_ver:
        if not isinstance(min_ver, str):
            raise ManifestError(
                f"{plugin_dir.name}/plugin.json: min_app_version must be a string"
            )
        if _parse_version(min_ver) > _parse_version(_APP_VERSION):
            raise ManifestError(
                f"Plugin '{data['name']}' requires app version {min_ver}, "
                f"but running {_APP_VERSION}. Please update SupremeDiffusion."
            )

    # ── requires check (type only — actual pipeline check is at runtime) ──
    requires = data.get("requires", [])
    if not isinstance(requires, list):
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json: 'requires' must be a list"
        )

    # ── tags check ──
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        raise ManifestError(
            f"{plugin_dir.name}/plugin.json: 'tags' must be a list"
        )

    # ── Build manifest ──
    manifest = PluginManifest(
        name=data["name"].strip(),
        version=data["version"].strip(),
        entry_point=data["entry_point"].strip(),
        author=str(data.get("author", "")).strip(),
        description=str(data.get("description", "")).strip(),
        min_app_version=min_ver,
        tab_label=str(data.get("tab_label", "")).strip(),
        requires=[str(r) for r in requires],
        content_schema=str(data.get("content_schema", "")).strip(),
        icon=str(data.get("icon", "")).strip(),
        tags=[str(t) for t in tags],
        homepage=str(data.get("homepage", "")).strip(),
        plugin_dir=plugin_dir,
    )

    return manifest


def generate_manifest_template() -> dict:
    """Return a template plugin.json dict with all fields and example values."""
    return {
        "name": "My Plugin",
        "version": "1.0.0",
        "author": "Your Name",
        "description": "A short description of what this plugin does.",
        "min_app_version": "2.2.0",
        "entry_point": "plugin.py",
        "tab_label": "My Plugin",
        "requires": ["sd_pipeline"],
        "content_schema": "v1",
        "icon": "",
        "tags": ["image", "video"],
        "homepage": ""
    }
