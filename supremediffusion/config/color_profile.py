"""Project-level color profile.

A ``ColorProfile`` describes the full set of color/range/colorspace knobs an
ffmpeg encode needs so that every output in a project is consistent. One
profile is selected per project (default ``bt709_full``), and every encoder
callsite in the app threads it through instead of picking its own
``pix_fmt`` / range / primaries combo.

The four built-in presets cover the realistic shipping cases:

* ``bt709_full``     — Godot/HD full-range, the default.
* ``bt709_limited``  — Broadcast / streaming standard.
* ``bt601_limited``  — Legacy SD / DV imports.
* ``srgb_full``      — Photo-style outputs (sRGB transfer).

Anything more exotic (HDR, Rec.2020) intentionally isn't here yet — adding
a new preset is just appending to ``_PRESETS``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorProfile:
    """A complete color/range/colorspace setting for ffmpeg encodes."""

    key: str
    label: str
    pix_fmt_libx264: str    # libx264-friendly pix_fmt (yuvj420p signals full range)
    pix_fmt_nvenc: str      # NVENC pix_fmt (always yuv420p; range stamped via -color_range)
    color_range: str        # "pc" (full) or "tv" (limited)
    colorspace: str         # "bt709" / "bt601"
    color_primaries: str    # "bt709" / "smpte170m" / "iec61966-2-1"
    color_trc: str          # "bt709" / "smpte170m" / "iec61966-2-1"

    def encoder_args(self, encoder_name: str = "") -> list[str]:
        """Return the full ``-pix_fmt ... -color_range ...`` block for an encoder.

        NVENC silently ignores ``yuvj420p`` and falls back to limited range, so
        we use ``yuv420p`` + an explicit ``-color_range pc/tv`` for it. libx264
        honors ``yuvj420p`` properly so the dedicated full-range pix_fmt is
        used when the profile is full-range.
        """
        if encoder_name and "nvenc" in encoder_name:
            pix_fmt = self.pix_fmt_nvenc
        else:
            pix_fmt = self.pix_fmt_libx264
        return [
            "-pix_fmt", pix_fmt,
            "-color_range", self.color_range,
            "-colorspace", self.colorspace,
            "-color_primaries", self.color_primaries,
            "-color_trc", self.color_trc,
        ]

    def output_range(self) -> str:
        """Return ``"full"`` or ``"tv"`` for use in scale ``out_range=`` tags."""
        return "full" if self.color_range == "pc" else "tv"

    def setparams_filter(self) -> str:
        """Return a ``setparams=…`` filter that stamps range/colorspace on a stream.

        Useful right before a concat where every input must declare matching
        tags or ffmpeg refuses to splice them.
        """
        return (
            f"setparams=range={self.output_range()}"
            f":colorspace={self.colorspace}"
            f":color_primaries={self.color_primaries}"
            f":color_trc={self.color_trc}"
        )


_PRESETS: dict[str, ColorProfile] = {
    # Default. Matches the previous app-wide GODOT_COLOR_ARGS: yuv420p, tv
    # (limited) range, bt709 primaries/colorspace/trc. This is what Godot
    # decodes correctly out of the box and what every existing project has
    # been silently using.
    "bt709_limited": ColorProfile(
        key="bt709_limited",
        label="BT.709 Limited (Godot / broadcast — default)",
        pix_fmt_libx264="yuv420p",
        pix_fmt_nvenc="yuv420p",
        color_range="tv",
        colorspace="bt709",
        color_primaries="bt709",
        color_trc="bt709",
    ),
    "bt709_full": ColorProfile(
        key="bt709_full",
        label="BT.709 Full-Range",
        pix_fmt_libx264="yuvj420p",
        pix_fmt_nvenc="yuv420p",
        color_range="pc",
        colorspace="bt709",
        color_primaries="bt709",
        color_trc="bt709",
    ),
    "bt601_limited": ColorProfile(
        key="bt601_limited",
        label="BT.601 Limited (legacy SD)",
        pix_fmt_libx264="yuv420p",
        pix_fmt_nvenc="yuv420p",
        color_range="tv",
        colorspace="smpte170m",
        color_primaries="smpte170m",
        color_trc="smpte170m",
    ),
    "srgb_full": ColorProfile(
        key="srgb_full",
        label="sRGB Full-Range (photo-style)",
        pix_fmt_libx264="yuvj420p",
        pix_fmt_nvenc="yuv420p",
        color_range="pc",
        colorspace="bt709",
        color_primaries="bt709",
        color_trc="iec61966-2-1",
    ),
}

DEFAULT_PROFILE_KEY: str = "bt709_limited"


def get_profile(key: str | None) -> ColorProfile:
    """Resolve a profile by slug. Falls back to the default on empty/unknown."""
    if key and key in _PRESETS:
        return _PRESETS[key]
    return _PRESETS[DEFAULT_PROFILE_KEY]


def list_profiles() -> list[ColorProfile]:
    """Return every preset in display order."""
    return list(_PRESETS.values())
