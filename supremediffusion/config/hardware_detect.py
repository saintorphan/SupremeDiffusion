"""Auto-detect GPU/system hardware and recommend quality settings.

Probes CUDA devices and system RAM on import, then exposes a
`detect()` function that returns a `HardwareProfile` with recommended
defaults for memory profile, resolution, quantization, etc.

The recommendations are conservative — they target *reliable* generation
without OOM, not maximum possible quality.  Users can always override
upward (with warnings) or downward (silently).
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Quality tier definitions ────────────────────────────────────────────────
#
# Each tier bundles every setting that affects speed vs quality into a single
# named preset.  Content packs can reference tiers by name, and the UI shows
# human-readable labels + time/quality trade-off hints.

@dataclass(frozen=True)
class QualityTier:
    """A named quality/speed preset."""

    key: str                    # machine key: "draft", "standard", "production", "ultra"
    label: str                  # UI label
    description: str            # one-liner shown under the dropdown
    speed_hint: str             # e.g. "~2 min per clip"
    quality_hint: str           # e.g. "Preview only — not for final output"

    # Video generation
    video_resolution: str       # e.g. "832x480"
    video_steps: int
    video_guidance: float
    video_frames: int           # 33=2s, 49=3s, 81=5s

    # Post-processing
    upscale: str                # "none", "2x", "4x"
    upscale_tile: int
    frame_interpolation: bool   # RIFE between clips
    face_restore: bool

    # Speed optimisations (all zero-quality-loss)
    vae_tiling: bool
    compile_transformer: bool

    # Minimum VRAM (GB) to run this tier without OOM
    min_vram_gb: float


# ── VRAM reality check (mid-2026 consumer GPU landscape) ────────────────
#
#  8 GB  (RTX 4060, 5060)         → SD/SDXL fine, video gen at 480p with
#                                    aggressive offloading + int8 + VAE tiling
# 12 GB  (RTX 3060, 4070)         → WAN 480p comfortable, community workhorse
# 16 GB  (RTX 4070 Ti S, 5060 Ti) → WAN 480p fast, longer clips, compile helps
# 24 GB  (RTX 3090, 4090)         → WAN native 720p viable, full pipeline
# 32 GB+ (RTX 5090, A6000)        → native 720p+, no compromises, long clips
#
# IMPORTANT: Post-processing (upscale, face restore, RIFE) runs AFTER the
# generation model is fully unloaded.  ESRGAN needs ~2-3GB VRAM regardless
# of input resolution (tiled processing).  So upscale tiers do NOT add to
# the generation VRAM requirement.  This means even 8GB cards can do 4x
# upscale — they just need to generate at lower resolution first.
#
# Users can always select one tier higher — advisory warns, doesn't block.

QUALITY_TIERS: tuple[QualityTier, ...] = (
    QualityTier(
        key="draft",
        label="Draft",
        description="Fast preview — check composition and motion before committing",
        speed_hint="~2 min per clip",
        quality_hint="Quick iteration at 512x512. Upscale later if it looks good.",
        video_resolution="512x512",
        video_steps=12,
        video_guidance=5.0,
        video_frames=33,
        upscale="none",
        upscale_tile=512,
        frame_interpolation=False,
        face_restore=False,
        vae_tiling=True,
        compile_transformer=False,
        min_vram_gb=6.0,
    ),
    QualityTier(
        key="standard",
        label="Standard",
        description="Solid everyday quality — native 480p with 2x AI upscale",
        speed_hint="~8 min per clip",
        quality_hint="832x480 generation + 2x ESRGAN (1664x960). Good for most work.",
        video_resolution="832x480",
        video_steps=20,
        video_guidance=5.0,
        video_frames=49,
        upscale="2x",
        upscale_tile=512,
        frame_interpolation=False,
        face_restore=False,
        vae_tiling=True,
        compile_transformer=False,
        min_vram_gb=10.0,
    ),
    QualityTier(
        key="production",
        label="Production",
        description="Publish-ready — 4x upscale to near-4K + RIFE + face restore",
        speed_hint="~15 min per clip",
        quality_hint="832x480 + 4x ESRGAN (3328x1920) + frame interpolation. Final render quality.",
        video_resolution="832x480",
        video_steps=20,
        video_guidance=5.0,
        video_frames=49,
        upscale="4x",
        upscale_tile=512,
        frame_interpolation=True,
        face_restore=True,
        vae_tiling=True,
        compile_transformer=True,
        min_vram_gb=14.0,
    ),
    QualityTier(
        key="ultra",
        label="Ultra",
        description="Maximum quality — native 720p + full post-processing pipeline",
        speed_hint="~25 min per clip",
        quality_hint="1280x720 native + 4x ESRGAN (5120x2880) + RIFE + face restore.",
        video_resolution="1280x720",
        video_steps=20,
        video_guidance=5.0,
        video_frames=81,
        upscale="4x",
        upscale_tile=384,
        frame_interpolation=True,
        face_restore=True,
        vae_tiling=True,
        compile_transformer=True,
        min_vram_gb=20.0,
    ),
)

TIER_BY_KEY: dict[str, QualityTier] = {t.key: t for t in QUALITY_TIERS}


def get_tier(key: str) -> QualityTier:
    """Look up a quality tier by key.  Raises KeyError if not found."""
    return TIER_BY_KEY[key]


# ── GPU Optimization Profiles ─────────────────────────────────────────────
#
# These map VRAM classes to recommended global performance settings.
# Each profile controls how the model is loaded and processed — this is
# about GPU utilisation and speed, NOT output quality.
#
# IMPORTANT: These are SUGGESTIONS shown in the UI.  They never auto-apply
# over the user's saved config.  The creator's 12GB defaults (Conservative,
# int8, xformers) are the baseline — detect() must match them for 12GB cards.
#
# The old numeric mmgp profiles (1-5) still work under the hood.  These
# named profiles wrap them with human-readable labels and descriptions.

@dataclass(frozen=True)
class GpuOptProfile:
    """A named GPU optimization preset mapping to an mmgp memory profile."""

    key: str                    # machine key
    label: str                  # UI label
    mmgp_profile: int           # underlying mmgp number (1-5)
    min_vram_gb: float          # minimum VRAM to use this profile safely
    min_ram_gb: float           # minimum system RAM recommended
    quantization: str           # "int8", "fp8", or "none"
    vae_tiling: bool            # whether VAE tiling is recommended
    compile_transformer: bool   # whether torch.compile is recommended
    description: str            # one-liner for the UI dropdown
    details: str                # multi-line explanation shown on selection


GPU_OPT_PROFILES: tuple[GpuOptProfile, ...] = (
    GpuOptProfile(
        key="minimal",
        label="Minimal",
        mmgp_profile=5,
        min_vram_gb=0.0,
        min_ram_gb=8.0,
        quantization="int8",
        vae_tiling=True,
        compile_transformer=False,
        description="Extreme offload -- bare minimum VRAM, for 6-8GB GPUs",
        details=(
            "How it works: Only the active model layer stays on the GPU. "
            "Everything else lives in system RAM and gets swapped in as needed.\n"
            "What this means: Generation is slow due to constant RAM-to-GPU transfers, "
            "but it lets small GPUs (RTX 4060, 3060 8GB) run models that wouldn't otherwise fit.\n"
            "Settings: mmgp profile 5, int8 quantization, VAE tiling ON, torch.compile OFF.\n"
            "Best for: 6-8GB VRAM with 16GB+ system RAM."
        ),
    ),
    GpuOptProfile(
        key="conservative",
        label="Conservative",
        mmgp_profile=4,
        min_vram_gb=10.0,
        min_ram_gb=16.0,
        quantization="int8",
        vae_tiling=True,
        compile_transformer=False,
        description="Heavy offload -- reliable on tight VRAM, for 10-12GB GPUs",
        details=(
            "How it works: The model is split between GPU and system RAM. "
            "The GPU holds the active portion, RAM stores the rest. "
            "Int8 quantization shrinks the model from ~14GB to ~7GB.\n"
            "What this means: Reliable generation without out-of-memory crashes. "
            "Some speed is traded for stability -- the GPU waits during RAM transfers.\n"
            "Settings: mmgp profile 4, int8 quantization, VAE tiling ON, torch.compile OFF.\n"
            "Best for: 10-12GB VRAM (RTX 3060 12GB, RTX 4070). "
            "This is the original default, tuned for the creator's 12GB RTX 30-series."
        ),
    ),
    GpuOptProfile(
        key="balanced",
        label="Balanced",
        mmgp_profile=3,
        min_vram_gb=12.0,
        min_ram_gb=16.0,
        quantization="int8",
        vae_tiling=True,
        compile_transformer=False,
        description="Moderate offload -- good speed/safety tradeoff, for 12-16GB GPUs",
        details=(
            "How it works: More of the model stays on the GPU at once, "
            "with RAM as overflow. Less swapping means faster generation.\n"
            "What this means: Noticeably faster than Conservative, "
            "with the int8 model (~7GB) fitting more comfortably in VRAM.\n"
            "Settings: mmgp profile 3, int8 quantization, VAE tiling ON, torch.compile OFF.\n"
            "Best for: 12-16GB VRAM with 32GB+ system RAM. "
            "Good middle ground if Conservative feels too slow."
        ),
    ),
    GpuOptProfile(
        key="performance",
        label="Performance",
        mmgp_profile=2,
        min_vram_gb=14.0,
        min_ram_gb=16.0,
        quantization="int8",
        vae_tiling=True,
        compile_transformer=True,
        description="Light offload -- most of model stays on GPU, for 16-20GB GPUs",
        details=(
            "How it works: The int8 model (~7GB) mostly stays in VRAM "
            "with minimal RAM overflow. torch.compile pre-optimises the model "
            "for your specific GPU (adds ~30s startup, then every step is faster).\n"
            "What this means: Fast generation with very little RAM-GPU swapping. "
            "The compile step pays for itself after a few clips.\n"
            "Settings: mmgp profile 2, int8 quantization, VAE tiling ON, torch.compile ON.\n"
            "Best for: 16-20GB VRAM (RTX 5060 Ti, RTX 4070 Ti Super)."
        ),
    ),
    GpuOptProfile(
        key="maximum",
        label="Maximum",
        mmgp_profile=1,
        min_vram_gb=20.0,
        min_ram_gb=16.0,
        quantization="int8",
        vae_tiling=False,
        compile_transformer=True,
        description="No offload -- full VRAM, fastest possible, for 24GB+ GPUs",
        details=(
            "How it works: The entire model loads into VRAM. "
            "No RAM involvement, no swapping, no waiting. "
            "torch.compile further optimises every operation for your GPU.\n"
            "What this means: Maximum generation speed. "
            "The GPU runs at full utilisation with zero transfer overhead.\n"
            "Settings: mmgp profile 1, int8 quantization, VAE tiling OFF, torch.compile ON.\n"
            "Best for: 24GB+ VRAM (RTX 3090, RTX 4090, RTX 5090)."
        ),
    ),
    GpuOptProfile(
        key="uncompressed",
        label="Uncompressed",
        mmgp_profile=1,
        min_vram_gb=30.0,
        min_ram_gb=32.0,
        quantization="none",
        vae_tiling=False,
        compile_transformer=True,
        description="Full precision -- no quantization, no compromise, for 32GB+ GPUs",
        details=(
            "How it works: The model runs at full precision (bf16/fp16) "
            "without int8 quantization. The full ~14GB model loads entirely into VRAM "
            "with room to spare. torch.compile optimises everything.\n"
            "What this means: Highest possible fidelity -- no quantization artifacts "
            "(which are minimal with int8 anyway, but this eliminates them entirely). "
            "Fastest generation with no compromises anywhere.\n"
            "Settings: mmgp profile 1, no quantization, VAE tiling OFF, torch.compile ON.\n"
            "Best for: 32GB+ VRAM (RTX 5090, A6000, workstation GPUs)."
        ),
    ),
)

GPU_OPT_BY_KEY: dict[str, GpuOptProfile] = {p.key: p for p in GPU_OPT_PROFILES}


def get_gpu_opt_profile(key: str) -> GpuOptProfile:
    """Look up a GPU optimisation profile by key.  Raises KeyError if not found."""
    return GPU_OPT_BY_KEY[key]


# ── Hardware profile ────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    """Detected hardware capabilities + recommended settings."""

    # Raw hardware info
    gpu_name: str = "Unknown"
    gpu_vram_gb: float = 0.0
    gpu_arch: str = ""             # e.g. "sm_89" for Ada Lovelace
    system_ram_gb: float = 0.0
    cpu_name: str = ""
    os: str = ""
    cuda_version: str = ""

    # Recommended GPU optimisation profile (SUGGESTION ONLY — never auto-applied)
    recommended_gpu_opt: str = "conservative"

    # What GPU opt profiles are safe to run
    available_gpu_opts: list[str] = field(default_factory=list)

    # Recommended global config overrides (kept for backward compat)
    recommended_memory_profile: int = 4
    recommended_attention: str = "sdpa"
    recommended_quantization: str = "int8"
    recommended_vae_tiling: bool = True
    recommended_compile: bool = False

    # Recommended default quality tier
    recommended_tier: str = "draft"

    # What tiers are safe to run
    available_tiers: list[str] = field(default_factory=list)

    # Human-readable summary
    summary: str = ""


def _detect_gpu() -> tuple[str, float, str, str]:
    """Return (gpu_name, vram_gb, arch, cuda_version) or fallback values."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ("No CUDA GPU", 0.0, "", "")

        props = torch.cuda.get_device_properties(0)
        name = props.name
        # Attribute name varies across PyTorch versions
        total_bytes = (
            getattr(props, "total_memory", None)
            or getattr(props, "total_global_mem", None)
            or getattr(props, "total_mem", 0)
        )
        vram_gb = round(total_bytes / (1024 ** 3), 1)
        arch = f"sm_{props.major}{props.minor}"
        cuda_ver = torch.version.cuda or ""
        return (name, vram_gb, arch, cuda_ver)
    except Exception as exc:
        logger.warning("GPU detection failed: %s", exc)
        return ("Detection failed", 0.0, "", "")


def _detect_system_ram() -> float:
    """Return system RAM in GB.  Works on Windows, Linux, and macOS."""
    # Best: psutil (cross-platform, pip-installable)
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    except Exception:
        pass

    # Linux: read /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # e.g. "MemTotal:       65536000 kB"
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except (FileNotFoundError, OSError, ValueError):
        pass

    # Windows: ctypes kernel32
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mem = ctypes.c_ulonglong(0)
        kernel32.GetPhysicallyInstalledMemory(ctypes.byref(mem))
        return round(mem.value / (1024 * 1024), 1)
    except (AttributeError, OSError):
        pass

    # macOS: sysctl
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
    except (FileNotFoundError, OSError, ValueError):
        pass

    return 0.0


def _detect_cpu() -> str:
    """Return CPU model string.  Works on Windows, Linux, and macOS."""
    # platform.processor() is often empty or unhelpful on Linux
    cpu = platform.processor()
    if cpu and cpu != platform.machine():
        return cpu

    # Linux: parse /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, OSError):
        pass

    # macOS: sysctl
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, OSError):
        pass

    return platform.machine() or "Unknown"


def detect() -> HardwareProfile:
    """Probe hardware and return a HardwareProfile with recommendations.

    This is designed to be called once at startup.  It takes ~100ms on
    most systems (the CUDA property query is instant if the driver is loaded).
    """
    gpu_name, vram_gb, arch, cuda_ver = _detect_gpu()
    ram_gb = _detect_system_ram()
    cpu = _detect_cpu()
    os_name = f"{platform.system()} {platform.release()}"

    profile = HardwareProfile(
        gpu_name=gpu_name,
        gpu_vram_gb=vram_gb,
        gpu_arch=arch,
        system_ram_gb=ram_gb,
        cpu_name=cpu,
        os=os_name,
        cuda_version=cuda_ver,
    )

    # ── Determine available GPU optimisation profiles ──
    available_opts = []
    best_opt = "minimal"  # fallback
    for opt in GPU_OPT_PROFILES:
        if vram_gb >= opt.min_vram_gb and ram_gb >= opt.min_ram_gb:
            available_opts.append(opt.key)
            best_opt = opt.key  # last one that fits = highest
    if not available_opts:
        available_opts = ["minimal"]
    profile.available_gpu_opts = available_opts
    profile.recommended_gpu_opt = best_opt

    # Derive legacy recommended_* fields from the best GPU opt profile
    opt = GPU_OPT_BY_KEY[best_opt]
    profile.recommended_memory_profile = opt.mmgp_profile
    profile.recommended_quantization = opt.quantization
    profile.recommended_vae_tiling = opt.vae_tiling
    profile.recommended_compile = opt.compile_transformer

    # ── Determine which quality tiers are available ──
    available = []
    for tier in QUALITY_TIERS:
        if vram_gb >= tier.min_vram_gb:
            available.append(tier.key)
    if not available:
        available = ["draft"]  # always allow draft
    profile.available_tiers = available

    # ── Pick the best default quality tier for this GPU ──
    # Choose the highest tier that fits comfortably (with ~1GB headroom)
    best_tier = "draft"
    for tier in QUALITY_TIERS:
        if vram_gb >= tier.min_vram_gb + 1.0:
            best_tier = tier.key
    profile.recommended_tier = best_tier

    # ── Attention backend ──
    try:
        import xformers  # noqa: F401
        profile.recommended_attention = "xformers"
    except ImportError:
        profile.recommended_attention = "sdpa"

    # ── Build summary ──
    tier_obj = TIER_BY_KEY[best_tier]
    opt_obj = GPU_OPT_BY_KEY[best_opt]
    profile.summary = (
        f"GPU: {gpu_name} ({vram_gb}GB VRAM) | RAM: {ram_gb}GB | "
        f"Recommended: {tier_obj.label} quality, {opt_obj.label} GPU profile"
    )

    logger.info("Hardware detected: %s", profile.summary)
    logger.info(
        "Available tiers: %s | Default: %s",
        ", ".join(available), best_tier,
    )
    logger.info(
        "Available GPU profiles: %s | Recommended: %s (mmgp %d)",
        ", ".join(available_opts), best_opt, opt.mmgp_profile,
    )

    return profile


# ── Advisory messages ───────────────────────────────────────────────────────
#
# These return user-facing strings for the UI when someone changes their
# quality tier above or below the recommended level.

def tier_advisory(selected_key: str, profile: HardwareProfile) -> Optional[str]:
    """Return an advisory message when the user picks a tier, or None if fine.

    Messages are informational, never blocking — the user can always proceed.
    """
    selected = TIER_BY_KEY.get(selected_key)
    recommended = TIER_BY_KEY.get(profile.recommended_tier)
    if selected is None or recommended is None:
        return None

    tier_order = [t.key for t in QUALITY_TIERS]
    sel_idx = tier_order.index(selected.key)
    rec_idx = tier_order.index(recommended.key)

    if sel_idx > rec_idx + 1:
        # User selected well above recommended
        if profile.gpu_vram_gb < selected.min_vram_gb:
            return (
                f"Your GPU has {profile.gpu_vram_gb}GB VRAM — "
                f"{selected.label} needs at least {selected.min_vram_gb}GB. "
                f"This will likely fail with an out-of-memory error. "
                f"Recommended: {recommended.label}."
            )
        return (
            f"{selected.label} will produce the best results but generation "
            f"will take significantly longer ({selected.speed_hint}). "
            f"For faster iteration, try {recommended.label} first, then "
            f"re-render at {selected.label} once you're happy with the result."
        )
    elif sel_idx > rec_idx:
        # One step above — gentle note
        return (
            f"{selected.label} is a step above the recommended tier for your hardware. "
            f"Generation will be slower ({selected.speed_hint}) but should work. "
            f"If you run into issues, drop back to {recommended.label}."
        )
    elif sel_idx < rec_idx:
        # Below recommended — note about quality
        return (
            f"{selected.label} will be fast ({selected.speed_hint}) but "
            f"quality will be lower. Your hardware can handle {recommended.label} "
            f"-- consider upgrading the tier for better results."
        )

    return None  # At recommended level — no advisory needed


def gpu_opt_advisory(
    selected_key: str,
    profile: HardwareProfile,
) -> Optional[str]:
    """Return an advisory when the user picks a GPU opt profile, or None if fine.

    Compares the selected profile against what's safe for the detected hardware.
    Messages are informational, never blocking.
    """
    selected = GPU_OPT_BY_KEY.get(selected_key)
    recommended = GPU_OPT_BY_KEY.get(profile.recommended_gpu_opt)
    if selected is None or recommended is None:
        return None

    if selected_key == profile.recommended_gpu_opt:
        return None  # Perfect match — no advisory

    opt_order = [p.key for p in GPU_OPT_PROFILES]
    sel_idx = opt_order.index(selected.key)
    rec_idx = opt_order.index(recommended.key)

    if sel_idx > rec_idx:
        # User picked a more aggressive profile than recommended
        if profile.gpu_vram_gb < selected.min_vram_gb:
            return (
                f"Your GPU has {profile.gpu_vram_gb}GB VRAM -- "
                f"{selected.label} needs at least {selected.min_vram_gb}GB. "
                f"This will likely cause out-of-memory crashes. "
                f"Recommended for your hardware: {recommended.label}."
            )
        if profile.system_ram_gb < selected.min_ram_gb:
            return (
                f"Your system has {profile.system_ram_gb}GB RAM -- "
                f"{selected.label} works best with {selected.min_ram_gb}GB+. "
                f"You may experience slowdowns or crashes."
            )
        return (
            f"{selected.label} is above the recommended profile for your hardware. "
            f"Generation may be unstable or crash with out-of-memory errors. "
            f"Recommended: {recommended.label}."
        )
    elif sel_idx < rec_idx:
        # User picked a more conservative profile — totally safe, just slower
        return (
            f"{selected.label} will work fine but is more conservative than needed. "
            f"Your hardware can handle {recommended.label} for faster generation "
            f"with no quality difference."
        )

    return None


def tier_for_content_pack(
    pack_tier_key: Optional[str],
    profile: HardwareProfile,
) -> QualityTier:
    """Resolve the quality tier for a content pack run.

    If the content pack specifies a preferred tier, use it — but clamp
    to the hardware's available tiers.  If None, use the recommended tier.
    """
    if pack_tier_key and pack_tier_key in TIER_BY_KEY:
        tier = TIER_BY_KEY[pack_tier_key]
        if pack_tier_key in profile.available_tiers:
            return tier
        # Pack wants a tier we can't run — fall back to best available
        logger.warning(
            "Content pack requests '%s' tier but hardware only supports %s. "
            "Falling back to '%s'.",
            pack_tier_key, profile.available_tiers, profile.recommended_tier,
        )
    return TIER_BY_KEY[profile.recommended_tier]
