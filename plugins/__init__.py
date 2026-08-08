"""Plugin discovery — manifest-first loading with validation before Python import."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_plugins(plugins_dir: Path | str) -> list[tuple]:
    """Scan *plugins_dir* for plugin packages and return (manifest, class) pairs.

    Loading order per plugin:
      1. Check for ``plugin.json`` — REQUIRED. No manifest = skip.
      2. Validate manifest fields, version compatibility, entry_point exists.
      3. Only then import the Python module and extract PLUGIN_CLASS.

    Directories whose name starts with ``_`` or ``.`` are skipped.

    Returns a list of (PluginManifest, plugin_class) tuples.
    """
    plugins_dir = Path(plugins_dir)
    if not plugins_dir.is_dir():
        return []

    # Set app version for manifest compatibility checks
    try:
        from plugins.manifest import set_app_version
        _set_version(set_app_version)
    except Exception:
        pass

    from plugins.manifest import load_manifest, ManifestError

    # Load disabled plugins list
    disabled: set[str] = set()
    disabled_path = plugins_dir / "disabled.json"
    if disabled_path.is_file():
        try:
            import json as _json
            with open(disabled_path, "r", encoding="utf-8") as _f:
                _data = _json.load(_f)
            if isinstance(_data, list):
                disabled = set(_data)
        except Exception:
            pass

    found: list[tuple] = []
    for candidate in sorted(plugins_dir.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name.startswith(("_", ".")):
            continue
        if candidate.name in disabled:
            logger.info("Plugin skipped (disabled): %s", candidate.name)
            continue

        # ── Step 1: Load and validate manifest BEFORE any Python import ──
        try:
            manifest = load_manifest(candidate)
        except ManifestError as exc:
            logger.warning("Plugin skipped (%s): %s", candidate.name, exc)
            continue
        except Exception:
            logger.exception("Unexpected error reading manifest for %s", candidate.name)
            continue

        # ── Step 2: Import the Python module ──
        init_file = candidate / "__init__.py"
        if not init_file.is_file():
            logger.warning(
                "Plugin %s has plugin.json but no __init__.py — skipped",
                candidate.name,
            )
            continue

        module_name = f"plugins.{candidate.name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(init_file))
            if spec is None or spec.loader is None:
                logger.warning("Plugin %s: cannot create module spec", candidate.name)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            logger.exception(
                "Plugin '%s' failed to import — skipped", manifest.name,
            )
            # Clean up partial import
            sys.modules.pop(module_name, None)
            continue

        # ── Step 3: Extract PLUGIN_CLASS ──
        plugin_cls = getattr(module, "PLUGIN_CLASS", None)
        if plugin_cls is None:
            logger.warning(
                "Plugin '%s' (%s) has no PLUGIN_CLASS export — skipped",
                manifest.name, candidate.name,
            )
            continue

        # ── Step 4: Verify it's a proper subclass ──
        try:
            from plugins.base_plugin import PluginBase
            if not issubclass(plugin_cls, PluginBase):
                logger.warning(
                    "Plugin '%s': PLUGIN_CLASS is not a PluginBase subclass — skipped",
                    manifest.name,
                )
                continue
        except Exception:
            pass  # If we can't check, allow it through

        # Patch manifest metadata onto the class (so MainWindow can read it)
        plugin_cls._manifest = manifest

        found.append((manifest, plugin_cls))
        logger.info(
            "Discovered plugin: %s v%s by %s (%s)",
            manifest.name, manifest.version,
            manifest.author or "unknown", candidate.name,
        )

    return found


def _set_version(setter_fn) -> None:
    """Read app version from pyproject.toml and set it for manifest checks."""
    try:
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if toml_path.is_file():
            for line in toml_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version"):
                    ver = line.split("=", 1)[1].strip().strip('"').strip("'")
                    setter_fn(ver)
                    return
    except Exception:
        pass
