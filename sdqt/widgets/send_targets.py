"""Shared send-to target definitions and menu builder.

All context-menu target lists are defined here so every image, video, and clip
location throughout the app shows consistent, categorised options.

Target entries are either:
  ("key", "Label")                          — flat action
  ("Subcategory Name", [("key", "Label")…]) — nested submenu
"""

from __future__ import annotations

from PySide6.QtWidgets import QMenu

# ── Disabled (greyed-out "Coming Soon") targets ─────────────────────────────

DISABLED_TARGETS: set[str] = set()  # All targets now routed through unified gen tab

# ── Clip send-to (video → video tab) ────────────────────────────────────────

CLIP_TARGETS: list = [
    ("timeline", "Timeline"),
    ("Video", [
        ("img2vid_clip", "Img2Vid"),
        ("ve", "Video Extender"),
        ("longshot", "Longshot"),
        ("stillgrabber", "StillGrabber"),
        ("mimicmotion", "Pose Animate (driver)"),
    ]),
    ("Color", [
        ("cc_source", "Color Correct (source)"),
        ("cc_ref", "Color Correct (reference)"),
    ]),
    ("Audio", [
        ("chatterbox", "Chatterbox TTS (preview)"),
        ("chatterbox_ref", "Chatterbox TTS (reference voice)"),
        ("dia_tts", "Dia TTS (preview)"),
        ("dia_ref", "Dia TTS (reference voice)"),
    ]),
]

# ── Frame / Image send-to (image → image tab) ──────────────────────────────
# Used by both "Send frame to" on video players and "Send to" on image widgets.

IMAGE_TARGETS: list = [
    ("Video", [
        ("img2vid", "Img2Vid"),
        ("timeline", "Timeline Library"),
        ("mimicmotion_ref", "Pose Animate (reference)"),
        ("sadtalker_src", "SadTalker (source)"),
    ]),
    ("Generate", [
        ("img2img", "Img2Img"),
        ("inpaint", "Inpaint"),
        ("controlnet_cond", "ControlNet (condition)"),
        ("controlnet_src", "ControlNet (source)"),
    ]),
    ("Edit", [
        ("imgedit", "Draw"),
        ("cropzoom", "Crop/Zoom"),
    ]),
    ("Face / Body", [
        ("faceswap_src", "FaceSwap (source)"),
        ("faceswap_tgt", "FaceSwap (target)"),
        ("bodydouble_src", "Body Double (source)"),
        ("bodydouble_tgt", "Body Double (target)"),
        ("repose_src", "RePose (source)"),
        ("repose_pose", "RePose (pose ref)"),
    ]),
    ("Tools", [
        ("model3d", "3D Modeling"),
        ("daz_scene", "Daz2Supreme"),
        ("qwenalyzer", "QwenAlyzer"),
    ]),
    ("Color", [
        ("cc_source", "Color Correct (source)"),
        ("cc_ref", "Color Correct (reference)"),
    ]),
]

# Simplified image targets for gallery right-click (no source/target split)
IMAGE_TARGETS_SIMPLE: list = [
    ("Video", [
        ("img2vid", "Img2Vid"),
        ("timeline", "Timeline Library"),
        ("mimicmotion_ref", "Pose Animate (reference)"),
        ("sadtalker_src", "SadTalker (source)"),
    ]),
    ("Generate", [
        ("img2img", "Img2Img"),
        ("inpaint", "Inpaint"),
    ]),
    ("Edit", [
        ("imgedit", "Draw"),
        ("cropzoom", "Crop/Zoom"),
    ]),
    ("Face / Body", [
        ("faceswap", "Face Swap"),
        ("bodydouble", "Body Double"),
        ("repose", "RePose"),
    ]),
    ("Tools", [
        ("model3d", "3D Modeling"),
        ("daz_scene", "Daz2Supreme"),
        ("qwenalyzer", "QwenAlyzer"),
    ]),
    ("Color", [
        ("cc_source", "Color Correct (source)"),
        ("cc_ref", "Color Correct (reference)"),
    ]),
]

# ── Final Frame targets ─────────────────────────────────────────────────────

FINAL_FRAME_TARGETS: list = [
    ("img2vid", "Img2Vid"),
    ("img2vid_ff", "Img2Vid FF (first frame)"),
    ("img2vid_lf", "Img2Vid LF (last frame)"),
    ("ve", "Video Extender"),
]

# ── Guide Video targets ─────────────────────────────────────────────────────

GUIDE_VIDEO_TARGETS: list = [
    ("img2vid_m1", "Img2Vid (single)"),
    ("img2vid_m2", "Img2Vid F/L"),
    ("ve", "Video Extender"),
]

# ── Audio send-to ───────────────────────────────────────────────────────────

AUDIO_TARGETS: list = [
    ("timeline", "Timeline"),
    ("Audio", [
        ("chatterbox", "Chatterbox TTS"),
        ("chatterbox_ref", "Chatterbox Ref Voice"),
        ("dia_tts", "Dia TTS"),
        ("dia_ref", "Dia Ref Voice"),
        ("musicgen", "MusicGen"),
        ("audiogen", "AudioGen"),
        ("sadtalker_audio", "SadTalker (driving audio)"),
    ]),
    ("Library", [
        ("voice_library", "Voice Library"),
        ("sound_library", "Sound Library"),
    ]),
]


# ── Menu builder helper ─────────────────────────────────────────────────────

def build_target_menu(
    parent_menu: QMenu,
    title: str,
    targets: list,
    callback,
) -> QMenu:
    """Add a submenu with categorised send-to actions.

    Args:
        parent_menu: The QMenu to add the submenu to.
        title: Submenu title (e.g. "Send clip to").
        targets: Target list — flat or nested subcategory format.
        callback: Called with (key: str) when an action is triggered.

    Returns:
        The created submenu.
    """
    menu = parent_menu.addMenu(title)
    _populate_menu(menu, targets, callback)
    return menu


def _populate_menu(menu: QMenu, targets: list, callback) -> None:
    """Recursively populate a menu from a target list."""
    for entry in targets:
        if isinstance(entry[1], list):
            # Subcategory
            sub = menu.addMenu(entry[0])
            for key, name in entry[1]:
                action = sub.addAction(name)
                if key in DISABLED_TARGETS:
                    action.setEnabled(False)
                else:
                    action.triggered.connect(
                        lambda checked, k=key: callback(k)
                    )
        else:
            key, name = entry
            action = menu.addAction(name)
            if key in DISABLED_TARGETS:
                action.setEnabled(False)
            else:
                action.triggered.connect(
                    lambda checked, k=key: callback(k)
                )
