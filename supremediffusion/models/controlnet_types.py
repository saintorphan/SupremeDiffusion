"""ControlNet type registry and IP-Adapter variant registry.

Central mapping of user-facing names to HuggingFace repos, preprocessor
classes, and default parameters for ControlNet conditioning types and
IP-Adapter weight variants.
"""

from __future__ import annotations

import gc as gc_module
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cache of instantiated preprocessor detectors, keyed by (type, repo/class,
# init kwargs). Reused across calls so a multi-frame extraction loads each
# model ONCE instead of rebuilding + tearing it down per frame (that reload
# was ~3s/frame of pure overhead vs ~0.3s of actual inference). Freed via
# clear_preprocessor_cache() when an extraction finishes.
_DETECTOR_CACHE: dict = {}


def clear_preprocessor_cache() -> None:
    """Release any cached preprocessor detectors to free VRAM/RAM."""
    if _DETECTOR_CACHE:
        _DETECTOR_CACHE.clear()
        gc_module.collect()


try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ControlNet type definitions
# ---------------------------------------------------------------------------

@dataclass
class ControlNetType:
    """Registry entry for a ControlNet conditioning type."""

    key: str                        # "openpose", "canny", "depth", etc.
    name: str                       # Display name
    sd15_repo: str                  # HuggingFace repo for SD 1.5
    sdxl_repo: str                  # HuggingFace repo for SDXL ("" = not available)
    preprocessor_class: str         # Class name in controlnet_aux
    preprocessor_repo: str          # Pretrained model path for preprocessor
    default_strength: float         # Default conditioning scale
    preprocessor_kwargs: dict = field(default_factory=dict)        # passed to detector.__call__
    preprocessor_init_kwargs: dict = field(default_factory=dict)   # passed to from_pretrained


CONTROLNET_TYPES: dict[str, ControlNetType] = {
    "openpose": ControlNetType(
        key="openpose",
        name="OpenPose",
        sd15_repo="lllyasviel/control_v11p_sd15_openpose",
        sdxl_repo="thibaud/controlnet-openpose-sdxl-1.0",
        preprocessor_class="OpenposeDetector",
        preprocessor_repo="lllyasviel/ControlNet",
        default_strength=0.7,
    ),
    "canny": ControlNetType(
        key="canny",
        name="Canny",
        sd15_repo="lllyasviel/control_v11p_sd15_canny",
        sdxl_repo="diffusers/controlnet-canny-sdxl-1.0",
        preprocessor_class="CannyDetector",
        preprocessor_repo="",
        default_strength=0.8,
    ),
    "depth": ControlNetType(
        key="depth",
        name="Depth (Midas)",
        sd15_repo="lllyasviel/control_v11f1p_sd15_depth",
        sdxl_repo="diffusers/controlnet-depth-sdxl-1.0",
        preprocessor_class="MidasDetector",
        preprocessor_repo="lllyasviel/ControlNet",
        default_strength=0.7,
    ),
    "lineart": ControlNetType(
        key="lineart",
        name="Lineart",
        sd15_repo="lllyasviel/control_v11p_sd15_lineart",
        sdxl_repo="",
        preprocessor_class="LineartDetector",
        preprocessor_repo="lllyasviel/Annotators",
        default_strength=0.75,
    ),
    "normal": ControlNetType(
        key="normal",
        name="Normal (BAE)",
        sd15_repo="lllyasviel/control_v11p_sd15_normalbae",
        sdxl_repo="",  # SD 1.5 only
        preprocessor_class="NormalBaeDetector",
        preprocessor_repo="lllyasviel/Annotators",
        default_strength=0.7,
    ),
    "mlsd": ControlNetType(
        key="mlsd",
        name="M-LSD Lines",
        sd15_repo="lllyasviel/control_v11p_sd15_mlsd",
        sdxl_repo="",
        preprocessor_class="MLSDdetector",
        preprocessor_repo="lllyasviel/ControlNet",
        default_strength=0.7,
    ),
    "softedge": ControlNetType(
        key="softedge",
        name="Soft Edge (HED)",
        sd15_repo="lllyasviel/control_v11p_sd15_softedge",
        sdxl_repo="SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
        preprocessor_class="HEDdetector",
        preprocessor_repo="lllyasviel/ControlNet",
        default_strength=0.7,
    ),
    "scribble": ControlNetType(
        key="scribble",
        name="Scribble",
        sd15_repo="lllyasviel/control_v11p_sd15_scribble",
        sdxl_repo="",
        preprocessor_class="PidiNetDetector",
        preprocessor_repo="lllyasviel/Annotators",
        default_strength=0.7,
        preprocessor_kwargs={"safe": True},
    ),
    "seg": ControlNetType(
        key="seg",
        name="Segmentation",
        sd15_repo="lllyasviel/control_v11p_sd15_seg",
        sdxl_repo="",
        preprocessor_class="SamDetector",
        preprocessor_repo="ybelkada/segment-anything",
        default_strength=0.7,
        # model_type/filename are SamDetector.from_pretrained args, not __call__
        # args — keep them paired so the chosen checkpoint actually loads.
        preprocessor_init_kwargs={
            "model_type": "vit_b",
            "filename": "sam_vit_b_01ec64.pth",
        },
    ),
    "shuffle": ControlNetType(
        key="shuffle",
        name="Content Shuffle",
        sd15_repo="lllyasviel/control_v11e_sd15_shuffle",
        sdxl_repo="",
        preprocessor_class="ContentShuffleDetector",
        preprocessor_repo="",
        default_strength=0.7,
    ),
    "tile": ControlNetType(
        key="tile",
        name="Tile / Upscale",
        sd15_repo="lllyasviel/control_v11f1e_sd15_tile",
        sdxl_repo="",
        preprocessor_class="",
        preprocessor_repo="",
        default_strength=0.8,
    ),
    "ip2p": ControlNetType(
        key="ip2p",
        name="Instruct Pix2Pix",
        sd15_repo="lllyasviel/control_v11e_sd15_ip2p",
        sdxl_repo="",
        preprocessor_class="",
        preprocessor_repo="",
        default_strength=0.7,
    ),
    "lineart_anime": ControlNetType(
        key="lineart_anime",
        name="Lineart Anime",
        sd15_repo="lllyasviel/control_v11p_sd15s2_lineart_anime",
        sdxl_repo="",
        preprocessor_class="LineartAnimeDetector",
        preprocessor_repo="lllyasviel/Annotators",
        default_strength=0.75,
    ),
    "dwpose": ControlNetType(
        key="dwpose",
        name="DWPose (body+hands+face)",
        sd15_repo="lllyasviel/control_v11p_sd15_openpose",
        sdxl_repo="thibaud/controlnet-openpose-sdxl-1.0",
        preprocessor_class="DWposeDetector",
        preprocessor_repo="",
        default_strength=0.7,
    ),
}


def list_controlnet_types() -> list[str]:
    """Return available ControlNet type keys."""
    return list(CONTROLNET_TYPES.keys())


def list_controlnet_names() -> list[tuple[str, str]]:
    """Return (key, display_name) pairs for all ControlNet types."""
    return [(ct.key, ct.name) for ct in CONTROLNET_TYPES.values()]


# Call-time preprocessor kwargs (route to ``detector.__call__``). Anything
# else in ``overrides`` that names a known init-time arg is split out and
# routed to ``from_pretrained`` instead, so callers can override either side.
_CALL_TIME_KWARGS = frozenset({
    "low_threshold",
    "high_threshold",
    "detect_resolution",
    "image_resolution",
    "include_hand",
    "include_face",
    "include_body",
    "hand_and_face",
    "safe",
    "scribble",
    "output_type",
})
_INIT_TIME_KWARGS = frozenset({
    "model_type",
    "filename",
    "subfolder",
    "cache_dir",
})


def run_preprocessor(
    cn_type_key: str,
    image: Any,
    init_overrides: Optional[dict] = None,
    **overrides,
) -> Any:
    """Run a ControlNet preprocessor on an image.

    Dynamically imports the detector class, runs it, and cleans up.
    Returns a PIL Image of the condition map.

    ``overrides`` are call-time kwargs forwarded to ``detector.__call__``
    (e.g. ``low_threshold``/``high_threshold`` for Canny,
    ``detect_resolution``/``image_resolution`` for resolution control,
    ``include_hand``/``include_face`` for OpenPose). Any override that names
    a known init-time arg (``model_type``/``filename``/...) is split out and
    routed to ``from_pretrained`` so it actually reaches model construction
    rather than being silently swallowed by the detector call. ``init_overrides``
    may be passed explicitly for the same purpose.
    """
    ct = CONTROLNET_TYPES.get(cn_type_key)
    if ct is None:
        raise ValueError(f"Unknown ControlNet type: {cn_type_key}")

    # Ensure PIL image early (needed for passthrough and preprocessing)
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Types with no preprocessor (tile, ip2p) — return image as-is
    if not ct.preprocessor_class:
        return image

    try:
        import controlnet_aux
    except ImportError:
        raise RuntimeError(
            "controlnet_aux is required for ControlNet preprocessing.\n"
            "Install it with: pip install controlnet_aux"
        )

    # Get the detector class
    detector_cls = getattr(controlnet_aux, ct.preprocessor_class, None)
    if detector_cls is None:
        raise RuntimeError(f"Preprocessor {ct.preprocessor_class} not found in controlnet_aux")

    # Split overrides into init-time (from_pretrained) vs call-time (__call__).
    # An override naming a known init arg is rerouted so it actually lands on
    # from_pretrained; everything else is treated as a call-time kwarg.
    init_kwargs = dict(ct.preprocessor_init_kwargs)
    if init_overrides:
        init_kwargs.update(init_overrides)
    call_overrides = dict(overrides)
    for key in list(call_overrides):
        if key in _INIT_TIME_KWARGS:
            init_kwargs[key] = call_overrides.pop(key)

    # Instantiate (cached) — reuse a loaded detector across calls instead of
    # rebuilding + deleting it every frame. init kwargs (e.g. SAM
    # model_type/filename) route to from_pretrained. Call
    # clear_preprocessor_cache() once the extraction loop is done to free it.
    ckey = (cn_type_key, ct.preprocessor_repo or ct.preprocessor_class,
            tuple(sorted((k, repr(v)) for k, v in init_kwargs.items())))
    detector = _DETECTOR_CACHE.get(ckey)
    if detector is None:
        if ct.preprocessor_repo:
            detector = detector_cls.from_pretrained(ct.preprocessor_repo, **init_kwargs)
        else:
            detector = detector_cls()
        _DETECTOR_CACHE[ckey] = detector

    # Merge default call kwargs with call-time overrides
    kwargs = dict(ct.preprocessor_kwargs)
    kwargs.update(call_overrides)

    # Run the preprocessor (detector stays cached for the next frame)
    result = detector(image, **kwargs)

    return result


def preprocessor_overrides_for(cn_type_key: str, config: Any) -> dict:
    """Resolve call-time preprocessor overrides for ``cn_type_key`` from config.

    Reads the ``cn_*`` ProjectConfig fields and returns only the kwargs that
    are meaningful for the chosen preprocessor type. Safe to call with any
    config object — missing fields fall back to controlnet_aux defaults.
    """
    overrides: dict = {}
    # Resolution applies to every controlnet_aux detector.
    overrides["detect_resolution"] = int(getattr(config, "cn_detect_resolution", 512))
    overrides["image_resolution"] = int(getattr(config, "cn_image_resolution", 512))

    if cn_type_key == "canny":
        overrides["low_threshold"] = int(getattr(config, "cn_canny_low", 100))
        overrides["high_threshold"] = int(getattr(config, "cn_canny_high", 200))
    elif cn_type_key in ("openpose", "dwpose"):
        overrides["include_hand"] = bool(getattr(config, "cn_openpose_include_hand", True))
        overrides["include_face"] = bool(getattr(config, "cn_openpose_include_face", True))

    return overrides


def get_controlnet_model(cn_type_key: str, is_sdxl: bool) -> Any:
    """Load a ControlNetModel from HuggingFace for the given type.

    Returns a diffusers ControlNetModel instance.
    """
    ct = CONTROLNET_TYPES.get(cn_type_key)
    if ct is None:
        raise ValueError(f"Unknown ControlNet type: {cn_type_key}")

    repo = ct.sdxl_repo if is_sdxl else ct.sd15_repo
    if not repo:
        raise ValueError(
            f"ControlNet type '{ct.name}' is not available for "
            f"{'SDXL' if is_sdxl else 'SD 1.5'}"
        )

    from diffusers import ControlNetModel

    # fp16 is invalid on CPU (many ops unimplemented) — pick dtype per device.
    dtype = torch.float16 if (torch is not None and torch.cuda.is_available()) else torch.float32
    logger.info("Loading ControlNet %s from %s (%s)...", ct.name, repo, dtype)
    model = ControlNetModel.from_pretrained(repo, torch_dtype=dtype)
    return model


# ---------------------------------------------------------------------------
# IP-Adapter variant definitions
# ---------------------------------------------------------------------------

@dataclass
class IPAdapterVariant:
    """Registry entry for an IP-Adapter weight variant."""

    key: str                # "base", "plus", "plus_face", "full_face"
    name: str               # Display name
    sd15_weight: str        # Weight filename for SD 1.5
    sdxl_weight: str        # Weight filename for SDXL ("" = not available)
    sd15_subfolder: str     # Subfolder in h94/IP-Adapter for SD 1.5
    sdxl_subfolder: str     # Subfolder in h94/IP-Adapter for SDXL


IP_ADAPTER_VARIANTS: dict[str, IPAdapterVariant] = {
    "base": IPAdapterVariant(
        key="base",
        name="IP-Adapter",
        sd15_weight="ip-adapter_sd15.safetensors",
        sdxl_weight="ip-adapter_sdxl_vit-h.safetensors",
        sd15_subfolder="models",
        sdxl_subfolder="sdxl_models",
    ),
    "plus": IPAdapterVariant(
        key="plus",
        name="IP-Adapter Plus",
        sd15_weight="ip-adapter-plus_sd15.safetensors",
        sdxl_weight="ip-adapter-plus_sdxl_vit-h.safetensors",
        sd15_subfolder="models",
        sdxl_subfolder="sdxl_models",
    ),
    "plus_face": IPAdapterVariant(
        key="plus_face",
        name="IP-Adapter Plus Face",
        sd15_weight="ip-adapter-plus-face_sd15.safetensors",
        sdxl_weight="ip-adapter-plus-face_sdxl_vit-h.safetensors",
        sd15_subfolder="models",
        sdxl_subfolder="sdxl_models",
    ),
    "full_face": IPAdapterVariant(
        key="full_face",
        name="IP-Adapter Full Face",
        sd15_weight="ip-adapter-full-face_sd15.safetensors",
        sdxl_weight="",  # SD 1.5 only
        sd15_subfolder="models",
        sdxl_subfolder="",
    ),
    "faceid": IPAdapterVariant(
        key="faceid",
        name="IP-Adapter FaceID",
        sd15_weight="ip-adapter-faceid_sd15.bin",
        sdxl_weight="ip-adapter-faceid_sdxl.bin",
        sd15_subfolder="",
        sdxl_subfolder="",
    ),
    "faceid_plus": IPAdapterVariant(
        key="faceid_plus",
        name="IP-Adapter FaceID Plus",
        sd15_weight="ip-adapter-faceid-plus_sd15.bin",
        sdxl_weight="ip-adapter-faceid-plusv2_sdxl.bin",
        sd15_subfolder="",
        sdxl_subfolder="",
    ),
}


def get_ip_adapter_config(variant_key: str, is_sdxl: bool) -> tuple[str, str]:
    """Return (subfolder, weight_name) for loading an IP-Adapter variant.

    Raises ValueError if the variant isn't available for the model type.
    """
    variant = IP_ADAPTER_VARIANTS.get(variant_key)
    if variant is None:
        raise ValueError(f"Unknown IP-Adapter variant: {variant_key}")

    # FaceID variants are handled directly by the pipeline loader
    if variant_key.startswith("faceid"):
        weight = variant.sdxl_weight if is_sdxl else variant.sd15_weight
        if not weight:
            raise ValueError(f"IP-Adapter variant '{variant.name}' not available for {'SDXL' if is_sdxl else 'SD 1.5'}")
        return None, weight  # subfolder=None for FaceID

    if is_sdxl:
        if not variant.sdxl_weight:
            raise ValueError(
                f"IP-Adapter variant '{variant.name}' is not available for SDXL"
            )
        return variant.sdxl_subfolder, variant.sdxl_weight
    else:
        return variant.sd15_subfolder, variant.sd15_weight


def list_ip_adapter_variants() -> list[tuple[str, str]]:
    """Return (key, display_name) pairs for all IP-Adapter variants."""
    return [(v.key, v.name) for v in IP_ADAPTER_VARIANTS.values()]
