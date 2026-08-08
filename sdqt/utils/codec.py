"""Video codec utilities — central source of truth for ffmpeg encoder args."""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading

logger = logging.getLogger(__name__)

# Thread-local active color profile slot. Workers set this in their ``run()``
# (via :func:`active_profile_context`) so every encoder call below resolves
# to the project that owns the operation, not the global default. Without it
# we'd have to thread project_cfg through ~25 worker files explicitly.
_ACTIVE = threading.local()

# (display_label, encoder_name, extra_args)
_CODEC_OPTIONS = [
    ("H.264 (NVENC, GPU)", "h264_nvenc", ["-preset", "p4", "-cq", "18"]),
    ("H.264 (CPU)", "libx264", ["-preset", "fast", "-crf", "18"]),
    ("H.265 (NVENC, GPU)", "hevc_nvenc", ["-preset", "p4", "-cq", "20"]),
    ("H.265 (CPU)", "libx265", ["-preset", "fast", "-crf", "20"]),
]

_available_cache: list[tuple[str, str, list[str]]] | None = None


def available_codecs() -> list[tuple[str, str, list[str]]]:
    """Return list of (label, encoder, extra_args) for encoders present on this system."""
    global _available_cache
    if _available_cache is not None:
        return _available_cache

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5,
        )
        output = result.stdout
    except Exception:
        output = ""

    avail = []
    for label, enc, args in _CODEC_OPTIONS:
        if enc in output:
            avail.append((label, enc, args))

    if not avail:
        # Fallback — libx264 should always exist
        avail = [("H.264 (CPU)", "libx264", ["-preset", "fast", "-crf", "18"])]

    _available_cache = avail
    return avail


def codec_labels() -> list[str]:
    """Return display labels for the settings dropdown."""
    return [label for label, _, _ in available_codecs()]


def get_codec_args(setting: str = "") -> list[str]:
    """Return ffmpeg encoder args for the configured codec.

    Args:
        setting: The display label or encoder name from config.
                 If empty/invalid, returns the first available codec.

    Returns:
        List like ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"]
    """
    codecs = available_codecs()

    # Match by label or encoder name
    for label, enc, args in codecs:
        if setting in (label, enc):
            return ["-c:v", enc] + args

    # Default to first available
    _, enc, args = codecs[0]
    return ["-c:v", enc] + args


def default_codec_label() -> str:
    """Return the label of the best default codec (GPU if available)."""
    codecs = available_codecs()
    return codecs[0][0] if codecs else "H.264 (CPU)"


def configured_encoder_name() -> str:
    """Return the encoder name from the user's configured codec."""
    args = configured_codec_args()
    for i, a in enumerate(args):
        if a == "-c:v" and i + 1 < len(args):
            return args[i + 1]
    return "libx264"


# ── Color profile API ──────────────────────────────────────────────────────
#
# Every encode in the app routes its color/pix_fmt/range/colorspace flags
# through here. The active profile is project-level (ProjectConfig
# .color_profile), with a global default (GlobalConfig.default_color_profile)
# and a hard fallback of ``bt709_limited`` (matches the previous app-wide
# GODOT_COLOR_ARGS behavior so legacy projects don't shift on first run).
#
# Three public entry points:
#   * project_color_args(project_cfg, global_cfg, encoder=None) — encoder flags
#   * project_input_filter(project_cfg, global_cfg, source_path=…) — scale + range filter
#   * probe_color_range(path) — read a file's actual range for filter chains
#
# The legacy ``FULL_RANGE_FILTER`` / ``GODOT_COLOR_ARGS`` / ``pix_fmt_args``
# names are kept as thin wrappers so callers that haven't been swept yet
# continue to work, but new code should use the project_*_args helpers.


def _resolve_profile(project_cfg=None, global_cfg=None):
    """Resolve the active ColorProfile from the supplied configs.

    Resolution order:
      1. Explicit *project_cfg* if it carries a resolved profile.
      2. Thread-local active profile (set via :func:`active_profile_context`).
      3. Explicit *global_cfg.default_color_profile*.
      4. Hard fallback ``bt709_limited``.

    Lazy-imports the profile module to avoid a config→codec import cycle.
    """
    from supremediffusion.config.color_profile import get_profile
    if project_cfg is not None and hasattr(project_cfg, "resolved_color_profile"):
        return project_cfg.resolved_color_profile(global_cfg)
    active = getattr(_ACTIVE, "profile", None)
    if active is not None:
        return active
    key = ""
    if global_cfg is not None:
        key = getattr(global_cfg, "default_color_profile", "") or ""
    return get_profile(key)


def _global_default_profile():
    """Resolve the active profile, falling back to GlobalConfig's default.

    Checks the thread-local first so worker-level encodes pick up the
    project they were started for, then falls back to GlobalConfig.
    """
    active = getattr(_ACTIVE, "profile", None)
    if active is not None:
        return active
    try:
        from supremediffusion.config.global_config import GlobalConfig
        cfg = GlobalConfig.load()
        return _resolve_profile(global_cfg=cfg)
    except Exception:
        from supremediffusion.config.color_profile import get_profile
        return get_profile(None)


def set_active_color_profile(profile) -> None:
    """Set the thread-local active ColorProfile.

    Called by worker entry points (BaseWorker subclasses) so every ffmpeg
    invocation inside ``do_work()`` reads from the right project, even
    deep inside library helpers that didn't get plumbed.
    """
    _ACTIVE.profile = profile


def clear_active_color_profile() -> None:
    """Clear the thread-local active profile."""
    if hasattr(_ACTIVE, "profile"):
        del _ACTIVE.profile


@contextlib.contextmanager
def active_profile_context(project_cfg=None, global_cfg=None, profile=None):
    """Context manager that pushes a profile onto the thread-local slot.

    Pass either an explicit *profile* or *project_cfg*+*global_cfg* (the
    profile is resolved from them). On exit the previous value is restored.
    """
    if profile is None:
        profile = _resolve_profile(project_cfg, global_cfg)
    previous = getattr(_ACTIVE, "profile", None)
    _ACTIVE.profile = profile
    try:
        yield profile
    finally:
        if previous is None:
            clear_active_color_profile()
        else:
            _ACTIVE.profile = previous


def project_color_args(
    project_cfg=None, global_cfg=None, encoder: str = "",
) -> list[str]:
    """Return ``-pix_fmt … -color_range … -colorspace … -color_primaries … -color_trc …``.

    *project_cfg* / *global_cfg* are optional — when omitted the app's
    GlobalConfig default is used. *encoder* defaults to the user's
    configured encoder name.
    """
    profile = _resolve_profile(project_cfg, global_cfg)
    enc = encoder or configured_encoder_name()
    return profile.encoder_args(enc)


def probe_color_range(path: str) -> str:
    """Return ``"full"`` or ``"tv"`` for the source file at *path*.

    Reads ``color_range`` and ``pix_fmt`` from ffprobe and combines them
    with a heuristic: explicit ``pc``/``full`` → full, explicit
    ``tv``/``limited`` → tv, otherwise ``yuvj*`` / ``rgb*`` are assumed
    full and everything else (plain ``yuv420p`` etc.) is assumed limited
    — which is what this app actually writes.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_range,pix_fmt",
             "-of", "default=nw=1", path],
            capture_output=True, text=True, timeout=15,
        )
        # Parse by key. ffprobe emits fields in stream order (pix_fmt before
        # color_range), NOT the order requested, so positional parsing would
        # swap them — silently misdetecting every full-range (pc) clip as tv,
        # which darkens/crushes it when the combine path retags it as tv.
        fields: dict[str, str] = {}
        for ln in r.stdout.splitlines():
            k, sep, v = ln.strip().lower().partition("=")
            if sep:
                fields[k] = v
        color_range = fields.get("color_range", "")
        pix_fmt = fields.get("pix_fmt", "")
        if color_range in ("pc", "full"):
            return "full"
        if color_range in ("tv", "limited"):
            return "tv"
        if pix_fmt.startswith("yuvj") or pix_fmt.startswith("rgb"):
            return "full"
        return "tv"
    except Exception:
        return "tv"


def probe_color_space(path: str) -> str:
    """Return the source's YUV matrix as an ffmpeg matrix name, or ``""``.

    Reads the ``color_space`` tag and normalizes it to a name accepted by
    the scale filter's ``in_color_matrix``/``out_color_matrix`` options.
    BT.601-family tags (``smpte170m``/``bt470bg``/``bt601``) share the same
    luma matrix and normalize to ``smpte170m``. Unknown/untagged → ``""``
    (caller should not force a conversion it can't verify).
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_space",
             "-of", "default=nw=1", path],
            capture_output=True, text=True, timeout=15,
        )
        cs = ""
        for ln in r.stdout.splitlines():
            k, sep, v = ln.strip().lower().partition("=")
            if sep and k == "color_space":
                cs = v
        if cs in ("bt709",):
            return "bt709"
        if cs in ("smpte170m", "bt470bg", "bt601"):
            return "smpte170m"
        if cs in ("bt2020nc", "bt2020c", "bt2020"):
            return "bt2020"
        return ""
    except Exception:
        return ""


def matrix_convert_filter(source_path: str, profile=None) -> str:
    """Return a ``colorspace=iall=…:all=…`` fragment converting *source_path*'s
    YUV matrix to *profile*'s, or ``""`` when no conversion is needed.

    The scale filter's ``in_color_matrix``/``out_color_matrix`` options only
    apply to YUV↔RGB conversions — YUV→YUV rescales silently ignore them —
    so real matrix conversion (e.g. BT.601 imports → BT.709 project) must go
    through the dedicated ``colorspace`` filter. Untagged sources return
    ``""`` (no guessed conversions).
    """
    if profile is None:
        profile = _resolve_profile()
    src_mat = probe_color_space(source_path) if source_path else ""
    if src_mat and src_mat != profile.colorspace:
        return f"colorspace=iall={src_mat}:all={profile.colorspace}"
    return ""


def project_input_filter(
    project_cfg=None,
    global_cfg=None,
    source_path: str = "",
    source_range: str = "",
    *,
    add_setsar: bool = True,
    add_setparams: bool = False,
    add_matrix: bool = False,
) -> str:
    """Build a ``scale=…:in_range=…:out_range=…`` filter for an input.

    *source_range* (``"full"`` or ``"tv"``) overrides the probe; otherwise
    *source_path* is probed via :func:`probe_color_range`. The output range
    matches the project's color profile, so a limited→full project (or
    vice versa) gets a real conversion instead of silent retagging.

    ``add_matrix`` probes the source's YUV matrix and, when it is known and
    differs from the profile's colorspace, prepends a ``colorspace`` filter
    so BT.601 imports are truly converted to the profile matrix instead of
    silently retagged (which shifts hue). Untagged sources are left alone.

    ``add_setparams`` stamps the profile's range/colorspace/primaries/trc
    onto the stream — required for concat where every input must declare
    matching tags.
    """
    profile = _resolve_profile(project_cfg, global_cfg)
    in_r = source_range or (probe_color_range(source_path) if source_path else "tv")
    out_r = profile.output_range()
    parts = []
    if add_matrix and source_path:
        conv = matrix_convert_filter(source_path, profile)
        if conv:
            parts.append(conv)
    parts.append(f"scale=in_range={in_r}:out_range={out_r}:flags=lanczos")
    if add_setsar:
        parts.append("setsar=1")
    if add_setparams:
        parts.append(profile.setparams_filter())
    return ",".join(parts)


# ── Legacy aliases (kept for callers not yet swept to project_color_args) ──
#
# These delegate to the active profile via ``_global_default_profile()`` —
# i.e. they used to be hardcoded "always full-range" / "always Godot tv"
# fragments and now reflect whatever the user's global default is. Once the
# sweep completes there should be no consumers outside this file.

def _legacy_full_range_filter() -> str:
    profile = _global_default_profile()
    out_r = profile.output_range()
    # out_color_matrix pins the RGB→YUV conversion to the profile's matrix.
    # Without it, swscale converts with its own default (BT.601 — even for HD)
    # while the encoder stamps the profile's colorspace tag, so every raw-RGB
    # encode picked up a real hue shift on playback (measured |Δ|≈8/255 on
    # saturated colors). Harmless for YUV inputs already in the profile matrix.
    return (
        f"scale=in_range=full:out_range={out_r}"
        f":out_color_matrix={profile.colorspace}"
    )


def full_range_filter() -> str:
    """Return a ``scale=in_range=full:out_range=<active-profile-out>`` filter.

    Resolves the active profile via the thread-local first, then global
    default. For full-range **RGB** inputs (rawvideo/PNG pipes) only — for
    video-file inputs use :func:`project_input_filter`, which probes the
    source's actual range instead of assuming full.
    """
    return _legacy_full_range_filter()


# DEPRECATED — use project_input_filter(...) for input-aware range conversion
# or full_range_filter() for the dynamic-profile equivalent of this constant.
# Kept as a module-level string for callers that interpolate it into f-strings.
FULL_RANGE_FILTER: str = _legacy_full_range_filter()


def full_range_vf(vf: str = "") -> str:
    """DEPRECATED: prepend the legacy full-range scale filter to *vf*."""
    base = full_range_filter()
    if not vf:
        return base
    return f"{base},{vf}"


# DEPRECATED — use project_color_args(...) instead.
GODOT_COLOR_ARGS: list[str] = _global_default_profile().encoder_args("libx264")


def pix_fmt_args(encoder: str = "") -> list[str]:
    """DEPRECATED: returns project-default pix_fmt/range flags.

    Old behavior: hardcoded ``yuvj420p`` (libx264) or ``yuv420p -color_range pc``
    (NVENC). New behavior: returns the active profile's flags. Callers
    should migrate to :func:`project_color_args`.
    """
    return project_color_args(encoder=encoder)


_configured_cache: list[str] | None = None


def configured_codec_args() -> list[str]:
    """Return codec args from the user's GlobalConfig setting (cached).

    Reads ``video_output_codec`` from GlobalConfig on first call,
    then caches the result for the lifetime of the process.
    Falls back to the best available codec if the setting is empty
    or GlobalConfig is unavailable.
    """
    global _configured_cache
    if _configured_cache is not None:
        return list(_configured_cache)

    setting = ""
    try:
        from supremediffusion.config.global_config import GlobalConfig
        cfg = GlobalConfig.load()
        setting = getattr(cfg, "video_output_codec", "") or ""
    except Exception:
        pass

    _configured_cache = get_codec_args(setting)
    return list(_configured_cache)
