"""Application state -- single shared object passed to all components."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class _QwenSignals(QObject):
    """Qt signals for Qwen lifecycle events (must live on a QObject)."""
    qwen_unloaded = Signal()  # emitted when Qwen is evicted (e.g. by pipeline load)


class AppState:
    """Holds all shared runtime objects.

    Mirrors the ``app_state`` dict from the Gradio version but as a proper
    object with typed accessors.  The heavy model objects (wan_pipeline,
    sd_pipeline) are loaded lazily via the Settings tab.
    """

    def __init__(self) -> None:
        from supremediffusion.config.global_config import GlobalConfig
        from supremediffusion.projects.manager import ProjectManager

        # Config
        self.global_config: GlobalConfig = GlobalConfig.load()
        self.global_config.save()  # ensure file exists on disk

        # Projects
        self.project_manager: ProjectManager = ProjectManager(
            self.global_config.projects_root,
        )

        # Lock to prevent concurrent pipeline loads (they share GPU memory)
        self._load_lock = threading.Lock()

        # Global generation lock — prevents multiple tabs from generating at once
        self._generation_lock = threading.Lock()
        self._generation_owner: str | None = None  # tab name that owns the GPU

        # Model registry (lazy, lightweight)
        self._model_registry = None

        # Qwen 3.5 4B (shared BF16 model for prompt enhance + chat)
        self._qwen_model: Any = None
        self._qwen_tokenizer: Any = None
        self._qwen_signals = _QwenSignals()

        # Qwen 2.5 VL 7B (vision-language model for image captioning)
        self._qwen_vl_model: Any = None
        self._qwen_vl_processor: Any = None

        # Prompt-writer LLM — abliterated Qwen GGUF via llama-cpp-python
        # (Txt2Prompt tab + the shared "Enhance prompt" path). GPU, mutually
        # exclusive with all generation pipelines.
        self._prompt_llm: Any = None

        # Hardware detection (runs once at startup, ~100ms)
        from supremediffusion.config.hardware_detect import detect as _detect_hw
        self.hardware_profile = _detect_hw()
        logger.info("Hardware: %s", self.hardware_profile.summary)

        # Resolve the active quality tier
        self._resolve_quality_tier()

        # Pipelines -- None until explicitly loaded (heavy GPU objects)
        self._wan_pipeline: Any = None
        self._lora_manager: Any = None
        self._gen_pipeline: Any = None
        self._sd_pipeline: Any = None
        self._img_pipeline: Any = None
        self._flux_model_pipeline: Any = None
        self._flux_pipeline: Any = None
        self._zimage_model_pipeline: Any = None
        self._zimage_pipeline: Any = None
        self._lipsync_pipeline: Any = None
        self._vace_multitalk_pipeline: Any = None
        self._triposr_pipeline: Any = None
        self._ltx_pipeline: Any = None
        self._ltx_gen_pipeline: Any = None
        self._mimicmotion_pipeline: Any = None
        self._sadtalker_pipeline: Any = None

        logger.info(
            "AppState initialised.  Projects root: %s  (%d projects)",
            self.global_config.projects_root,
            len(self.project_manager.list_projects()),
        )

    # -- Quality tier --------------------------------------------------------

    def _resolve_quality_tier(self) -> None:
        """Resolve the active quality tier from config + hardware."""
        from supremediffusion.config.hardware_detect import (
            get_tier, tier_for_content_pack, TIER_BY_KEY,
        )
        tier_key = self.global_config.quality_tier
        if tier_key == "auto" or tier_key not in TIER_BY_KEY:
            self._active_tier = get_tier(self.hardware_profile.recommended_tier)
            logger.info(
                "Quality tier: auto -> %s (based on %s)",
                self._active_tier.label, self.hardware_profile.gpu_name,
            )
        else:
            self._active_tier = get_tier(tier_key)
            logger.info("Quality tier: %s (user override)", self._active_tier.label)

    @property
    def quality_tier(self):
        """The active QualityTier object."""
        return self._active_tier

    def set_quality_tier(self, key: str) -> str | None:
        """Change quality tier.  Returns advisory message or None."""
        from supremediffusion.config.hardware_detect import (
            get_tier, tier_advisory, TIER_BY_KEY,
        )
        if key == "auto":
            self.global_config.quality_tier = "auto"
        elif key in TIER_BY_KEY:
            self.global_config.quality_tier = key
        self._resolve_quality_tier()
        self.global_config.save()
        return tier_advisory(self._active_tier.key, self.hardware_profile)

    # -- Model registry ----------------------------------------------------

    @property
    def model_registry(self):
        """Lazy-loaded ModelRegistry for download management."""
        if self._model_registry is None:
            from sdqt.models.registry import ModelRegistry
            self._model_registry = ModelRegistry(
                self.global_config.models_root,
                self.global_config.model_paths,
            )
        return self._model_registry

    def sync_config_after_download(self) -> None:
        """Save GlobalConfig after model downloads update model_paths."""
        self.global_config.save()

    # -- Lazy pipeline accessors ------------------------------------------

    @property
    def wan_pipeline(self) -> Any:
        return self._wan_pipeline

    @property
    def lora_manager(self) -> Any:
        return self._lora_manager

    @property
    def pipeline(self) -> Any:
        """The active GenerationPipeline (Wan or LTX, whichever is loaded)."""
        if self._ltx_gen_pipeline is not None:
            return self._ltx_gen_pipeline
        return self._gen_pipeline

    @property
    def active_video_backend(self) -> str:
        """Return 'ltx' if LTX is loaded, 'wan' if Wan is loaded, else ''."""
        if self._ltx_pipeline is not None:
            return "ltx"
        if self._wan_pipeline is not None:
            return "wan"
        return ""

    @property
    def sd_pipeline(self) -> Any:
        return self._sd_pipeline

    @property
    def img_pipeline(self) -> Any:
        return self._img_pipeline

    @property
    def flux_pipeline(self) -> Any:
        """The FluxGenerationPipeline orchestrator."""
        return self._flux_pipeline

    @property
    def zimage_pipeline(self) -> Any:
        """The ZImageGenerationPipeline orchestrator."""
        return self._zimage_pipeline

    @property
    def pipelines_loaded(self) -> bool:
        return self._wan_pipeline is not None or self._ltx_pipeline is not None

    @property
    def sd_pipelines_loaded(self) -> bool:
        return self._sd_pipeline is not None

    @property
    def flux_pipelines_loaded(self) -> bool:
        return self._flux_pipeline is not None

    @property
    def zimage_pipelines_loaded(self) -> bool:
        return self._zimage_pipeline is not None

    # ── Generation lock ────────────────────────────────────────────────

    def acquire_generation(self, owner: str) -> bool:
        """Try to claim the GPU for generation. Returns True if acquired."""
        with self._generation_lock:
            if self._generation_owner is None:
                self._generation_owner = owner
                return True
            return False

    def release_generation(self, owner: str) -> None:
        """Release the GPU generation lock."""
        with self._generation_lock:
            if self._generation_owner == owner:
                self._generation_owner = None

    @property
    def generation_owner(self) -> str | None:
        """Who currently owns the GPU, or None if free."""
        with self._generation_lock:
            return self._generation_owner

    def load_pipelines(self) -> None:
        """Instantiate video pipeline objects.  Call from a worker thread.

        Auto-unloads all image pipelines first to free VRAM.
        """
        from supremediffusion.models.wan_i2v import WanI2VPipeline
        from supremediffusion.models.lora import LoRAManager
        from supremediffusion.core.pipeline import GenerationPipeline

        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            # LTX is the sibling video backend and is ~22B — it can't co-reside
            # with Wan on the GPU. Symmetric with load_ltx_pipelines(), which
            # unloads Wan. Without this, loading Wan left LTX resident and the
            # `pipeline` property (which prefers LTX) kept routing Wan projects
            # to the LTX distilled pipeline — wrong model, 8-step lock, etc.
            self.unload_ltx_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self._force_gc()

        logger.info("Loading video pipelines...")

        self._wan_pipeline = WanI2VPipeline(self.global_config)
        self._lora_manager = LoRAManager(
            self.global_config.model_paths.get("lora_dir", ""),
        )
        self._gen_pipeline = GenerationPipeline(
            self._wan_pipeline, self.project_manager,
        )

        logger.info("Core pipelines ready.")

    def load_ltx_pipelines(self) -> None:
        """Instantiate LTX Video 2.3 pipeline objects.  Call from a worker thread.

        Auto-unloads all other pipelines first to free VRAM.
        """
        from supremediffusion.models.ltx_pipeline import LTXVideoPipeline
        from supremediffusion.core.pipeline import GenerationPipeline

        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self._force_gc()

        logger.info("Loading LTX video pipelines...")

        self._ltx_pipeline = LTXVideoPipeline(self.global_config)
        self._ltx_gen_pipeline = GenerationPipeline(
            self._ltx_pipeline, self.project_manager,
        )

        logger.info("LTX pipelines ready.")

    def load_video_pipeline_for_backend(self, backend: str) -> None:
        """Load the requested video backend ('wan' or 'ltx') and ensure the
        other one is unloaded, so exactly one video backend is ever resident.

        Wan and LTX are each ~22B and can't share the GPU. Beyond VRAM, a stale
        backend made `pipeline` / `active_video_backend` resolve to it (the
        `pipeline` property prefers LTX whenever it's loaded), so a Wan project
        silently rendered on the previously-loaded LTX distilled pipeline —
        wrong model, 8-step lock, flicker, etc. The old early-return left the
        other backend loaded; this does not.
        """
        if backend == "ltx":
            if self._wan_pipeline is not None:
                self.unload_video_pipelines()
            if self._ltx_pipeline is None:
                self.load_ltx_pipelines()
        else:
            if self._ltx_pipeline is not None:
                self.unload_ltx_pipelines()
            if self._wan_pipeline is None:
                self.load_pipelines()

    @property
    def ltx_pipeline(self) -> Any:
        return self._ltx_pipeline

    @property
    def ltx_gen_pipeline(self) -> Any:
        return self._ltx_gen_pipeline

    def unload_ltx_pipelines(self) -> None:
        """Release LTX video pipeline GPU resources."""
        if self._ltx_pipeline is not None and hasattr(self._ltx_pipeline, "unload"):
            self._ltx_pipeline.unload()
        self._ltx_pipeline = None
        self._ltx_gen_pipeline = None
        logger.info("LTX pipelines unloaded.")

    def load_sd_pipelines(self) -> None:
        """Instantiate SD image pipelines (optional, for Image Suite).

        Auto-unloads video, FLUX and Z-Image first to free VRAM.
        """
        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self._force_gc()

            from supremediffusion.models.sd_pipeline import SDImagePipeline
            from supremediffusion.core.image_pipeline import ImageGenerationPipeline

            self._sd_pipeline = SDImagePipeline(self.global_config)
            self._img_pipeline = ImageGenerationPipeline(
                self._sd_pipeline, self.project_manager,
            )
            logger.info("SD image pipelines ready.")

    def load_flux_pipelines(self, *, load_chroma: bool = True, load_fill: bool = False) -> None:
        """Instantiate FLUX pipelines (optional, for FLUX tabs).

        Args:
            load_chroma: Load the Chroma txt2img/img2img pipeline.
            load_fill: Load the FluxFill inpainting pipeline.

        Auto-unloads SD and Z-Image first to free VRAM.
        Acquires _load_lock to wait for any in-progress Z-Image load.
        """
        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_sd_pipelines()
            self.unload_zimage_pipelines()
            self._force_gc()

            from supremediffusion.models.flux_pipeline import FluxImagePipeline
            from supremediffusion.core.flux_pipeline import FluxGenerationPipeline

            self._flux_model_pipeline = FluxImagePipeline(self.global_config)

            if load_chroma:
                chroma_dir = self.global_config.model_paths.get("flux_chroma_dir", "")
                if chroma_dir:
                    self._flux_model_pipeline.load_chroma(chroma_dir)

            if load_fill:
                fill_dir = self.global_config.model_paths.get("flux_fill_dir", "")
                if fill_dir:
                    self._flux_model_pipeline.load_fill(fill_dir)

            self._flux_pipeline = FluxGenerationPipeline(
                self._flux_model_pipeline, self.project_manager,
            )
            logger.info("FLUX pipelines ready.")

    def load_zimage_pipelines(self, model_path: str | None = None) -> None:
        """Instantiate Z-Image Turbo pipelines (optional, for Z-Image tabs).

        Args:
            model_path: Explicit path to a .safetensors file or diffusers dir.
                        Falls back to the last-used path stored in
                        ``zimage_active_model``, then to ``zimage_dir``.

        Auto-unloads SD and FLUX first to free VRAM.
        Thread-safe: only one pipeline load can run at a time.
        """
        if not self._load_lock.acquire(blocking=False):
            # Another load is already in progress — wait for it
            logger.info("Pipeline load already in progress, waiting...")
            with self._load_lock:
                if self._zimage_pipeline is not None:
                    return  # loaded by the other thread
                # Other thread failed; fall through to retry

        try:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()

            from supremediffusion.models.zimage_pipeline import ZImageModelPipeline
            from supremediffusion.core.zimage_pipeline import ZImageGenerationPipeline

            if not model_path:
                model_path = self.global_config.model_paths.get("zimage_active_model", "")
            if not model_path:
                model_path = self.global_config.model_paths.get("zimage_dir", "")
            if not model_path:
                raise RuntimeError("No Z-Image model path configured.")

            # Remember which model we loaded
            self.global_config.model_paths["zimage_active_model"] = model_path
            self.global_config.save()

            self._zimage_model_pipeline = ZImageModelPipeline(self.global_config, model_dir=model_path)
            self._zimage_pipeline = ZImageGenerationPipeline(
                self._zimage_model_pipeline, self.project_manager,
            )
            logger.info("Z-Image pipelines ready (model: %s).", model_path)
        finally:
            self._load_lock.release()

    def unload_sd_pipelines(self) -> None:
        """Release SD image pipeline GPU resources."""
        if self._sd_pipeline is not None and hasattr(self._sd_pipeline, "unload"):
            self._sd_pipeline.unload()
        self._sd_pipeline = None
        self._img_pipeline = None
        logger.info("SD image pipelines unloaded.")

    def unload_flux_pipelines(self) -> None:
        """Release FLUX GPU resources."""
        if self._flux_model_pipeline is not None and hasattr(self._flux_model_pipeline, "unload"):
            self._flux_model_pipeline.unload()
        self._flux_model_pipeline = None
        self._flux_pipeline = None
        logger.info("FLUX pipelines unloaded.")

    def unload_zimage_pipelines(self) -> None:
        """Release Z-Image GPU resources."""
        if self._zimage_model_pipeline is not None and hasattr(self._zimage_model_pipeline, "unload"):
            self._zimage_model_pipeline.unload()
        self._zimage_model_pipeline = None
        self._zimage_pipeline = None
        logger.info("Z-Image pipelines unloaded.")

    def unload_video_pipelines(self) -> None:
        """Release all video pipeline GPU resources (Wan + LTX + MimicMotion)."""
        if self._wan_pipeline is not None and hasattr(self._wan_pipeline, "unload"):
            self._wan_pipeline.unload()
        self._wan_pipeline = None
        self._lora_manager = None
        self._gen_pipeline = None
        self.unload_ltx_pipelines()
        self.unload_mimicmotion_pipelines()
        logger.info("Video pipelines unloaded.")

    def load_mimicmotion_pipelines(self) -> None:
        """Instantiate the MimicMotion pipeline. Mutually exclusive with all other GPU pipelines."""
        from supremediffusion.models.mimicmotion_pipeline import MimicMotionPipeline

        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            # Avoid recursion: unload_video_pipelines() calls back here, so
            # tear down siblings inline rather than via the umbrella.
            if self._wan_pipeline is not None and hasattr(self._wan_pipeline, "unload"):
                self._wan_pipeline.unload()
            self._wan_pipeline = None
            self._lora_manager = None
            self._gen_pipeline = None
            self.unload_ltx_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self._force_gc()

        logger.info("Loading MimicMotion pipeline...")
        self._mimicmotion_pipeline = MimicMotionPipeline(self.global_config)
        logger.info("MimicMotion pipeline ready (will load weights on first generate).")

    def unload_mimicmotion_pipelines(self) -> None:
        """Release MimicMotion GPU resources."""
        if self._mimicmotion_pipeline is not None and hasattr(self._mimicmotion_pipeline, "unload"):
            self._mimicmotion_pipeline.unload()
        self._mimicmotion_pipeline = None
        logger.info("MimicMotion pipeline unloaded.")

    @property
    def mimicmotion_pipeline(self) -> Any:
        return self._mimicmotion_pipeline

    def load_sadtalker_pipelines(self) -> None:
        """Prepare SadTalker pipeline. It runs in its own subprocess so no
        GPU pre-allocation happens here, but we still acquire the load lock
        and unload other in-process pipelines first as a courtesy.

        NOTE: because SadTalker is subprocess-isolated, you do NOT have to
        unload before using it — VRAM is shared at runtime. This method
        exists for API symmetry and for the optional pre-warm of the bridge.
        """
        from supremediffusion.models.sadtalker_pipeline import SadTalkerPipeline

        with self._load_lock:
            self._force_gc()

        logger.info("Initialising SadTalker subprocess pipeline...")
        self._sadtalker_pipeline = SadTalkerPipeline(self.global_config)
        self._sadtalker_pipeline.load()
        logger.info("SadTalker pipeline ready (subprocess will spawn on first generate).")

    def unload_sadtalker_pipelines(self) -> None:
        """Release the SadTalker bridge."""
        if self._sadtalker_pipeline is not None and hasattr(self._sadtalker_pipeline, "unload"):
            self._sadtalker_pipeline.unload()
        self._sadtalker_pipeline = None
        logger.info("SadTalker pipeline unloaded.")

    @property
    def sadtalker_pipeline(self) -> Any:
        return self._sadtalker_pipeline

    def load_lipsync_pipeline(self) -> Any:
        """Load LatentSync pipeline, unloading everything else first."""
        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self._force_gc()

            from supremediffusion.models.lipsync_pipeline import NativeLipSyncPipeline
            if self._lipsync_pipeline is None:
                self._lipsync_pipeline = NativeLipSyncPipeline()
            self._lipsync_pipeline.load()
            logger.info("LipSync pipeline ready.")
            return self._lipsync_pipeline

    @property
    def lipsync_pipeline(self) -> Any:
        return self._lipsync_pipeline

    def unload_lipsync_pipeline(self) -> None:
        """Release LatentSync GPU resources."""
        if self._lipsync_pipeline is not None and hasattr(self._lipsync_pipeline, "unload"):
            self._lipsync_pipeline.unload()
        self._lipsync_pipeline = None
        logger.info("LipSync pipeline unloaded.")

    def load_vace_multitalk_pipeline(self) -> Any:
        """Load VACE MultiTalk pipeline, unloading everything else first."""
        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self._force_gc()

            from supremediffusion.models.vace_multitalk_pipeline import NativeVaceMultitalkPipeline
            if self._vace_multitalk_pipeline is None:
                self._vace_multitalk_pipeline = NativeVaceMultitalkPipeline()
            self._vace_multitalk_pipeline.load()
            logger.info("VACE MultiTalk pipeline ready.")
            return self._vace_multitalk_pipeline

    @property
    def vace_multitalk_pipeline(self) -> Any:
        return self._vace_multitalk_pipeline

    def unload_vace_multitalk_pipeline(self) -> None:
        """Release VACE MultiTalk GPU resources."""
        if self._vace_multitalk_pipeline is not None and hasattr(self._vace_multitalk_pipeline, "unload"):
            self._vace_multitalk_pipeline.unload()
        self._vace_multitalk_pipeline = None
        logger.info("VACE MultiTalk pipeline unloaded.")

    # -- TripoSR (3D Modeling) ------------------------------------------------

    def load_triposr_pipeline(self) -> Any:
        """Load TripoSR + BiRefNet pipeline, unloading heavy pipelines first."""
        with self._load_lock:
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_lipsync_pipeline()
            self.unload_vace_multitalk_pipeline()
            self._force_gc()

            from supremediffusion.models.triposr_pipeline import TripoSRPipeline
            if self._triposr_pipeline is None:
                self._triposr_pipeline = TripoSRPipeline(self.global_config.model_paths)
            self._triposr_pipeline.load_tsr()
            logger.info("TripoSR pipeline ready.")
            return self._triposr_pipeline

    @property
    def triposr_pipeline(self) -> Any:
        return self._triposr_pipeline

    def unload_triposr_pipeline(self) -> None:
        """Release TripoSR + BiRefNet GPU resources."""
        if self._triposr_pipeline is not None and hasattr(self._triposr_pipeline, "unload"):
            self._triposr_pipeline.unload()
        self._triposr_pipeline = None
        logger.info("TripoSR pipeline unloaded.")

    # -- Qwen 3.5 4B (shared BF16 model) ------------------------------------

    @property
    def qwen_loaded(self) -> bool:
        return self._qwen_model is not None

    @property
    def qwen_unloaded_signal(self):
        """Signal emitted when Qwen is unloaded externally."""
        return self._qwen_signals.qwen_unloaded

    def load_qwen(self) -> None:
        """Load Qwen 3.5 4B at bfloat16 via transformers.

        Unloads all generation pipelines first (but not Qwen itself).
        Call from a worker thread.
        """
        with self._load_lock:
            if self._qwen_model is not None:
                return  # already loaded

            # Free VRAM from generation pipelines
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self.unload_vace_multitalk_pipeline()
            self.unload_triposr_pipeline()
            self._force_gc()

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Prefer Qwen 2.5 7B abliterated (local), fall back to 3.5 4B (HF)
            from pathlib import Path
            local_7b = Path(self.global_config.models_root) / "qwen2.5_7b_abliterated"
            local_4b = Path(self.global_config.models_root) / "qwen3.5_4b"

            if local_7b.is_dir() and any(local_7b.iterdir()):
                model_id = str(local_7b)
                logger.info("Loading Qwen 2.5 7B abliterated from %s ...", model_id)
            elif local_4b.is_dir() and any(local_4b.iterdir()):
                model_id = str(local_4b)
                logger.info("Loading Qwen 3.5 4B (fallback) from %s ...", model_id)
            else:
                model_id = "Qwen/Qwen3.5-4B"
                logger.info(
                    "No local Qwen model found. Downloading Qwen 3.5 4B from HF. "
                    "For better results, go to Settings → Qwen → Download Qwen Locally."
                )

            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
            )

            # Ensure VRAM is fully reclaimed before loading
            self._force_gc()

            # Load directly to GPU (no device_map="auto" which can leave
            # layers on meta device).  Fall back to CPU if VRAM is tight.
            free_vram = 0
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0] // (1024 ** 3)
            logger.info("Qwen load: free VRAM ~%dGiB", free_vram)

            if free_vram >= 15:
                logger.info("Loading Qwen directly to cuda:0")
                self._qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to("cuda:0")
            else:
                logger.info("Loading Qwen to CPU (VRAM too low for bf16 on GPU)")
                self._qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                ).to("cpu")
            logger.info("Qwen loaded from %s", model_id)

    def unload_qwen(self) -> None:
        """Release Qwen model/tokenizer and free VRAM."""
        if self._qwen_model is None:
            return
        del self._qwen_model
        del self._qwen_tokenizer
        self._qwen_model = None
        self._qwen_tokenizer = None
        self._force_gc()
        self._qwen_signals.qwen_unloaded.emit()
        logger.info("Qwen model unloaded.")

    # -- Prompt-writer LLM (abliterated Qwen GGUF, llama-cpp-python) ---------

    @property
    def prompt_llm_loaded(self) -> bool:
        return self._prompt_llm is not None

    def _resolve_prompt_llm_path(self) -> str:
        """Locate the prompt-writer GGUF: explicit config first, then scan the
        models dirs for an abliterated Qwen GGUF, then any .gguf."""
        from pathlib import Path
        explicit = (self.global_config.model_paths or {}).get("prompt_llm_gguf", "")
        if explicit and Path(explicit).is_file():
            return explicit
        candidates: list[Path] = []
        seen: set[str] = set()
        for d in (self.global_config.models_root, "/data/models"):
            try:
                dd = Path(d)
                if dd.is_dir() and str(dd.resolve()) not in seen:
                    seen.add(str(dd.resolve()))
                    candidates += sorted(dd.glob("*.gguf"))
            except Exception:
                pass
        def _is_ablit_qwen(name: str) -> bool:
            return "qwen" in name and "ablit" in name
        # Prefer a llama.cpp-compatible arch: Qwen2.5 ('qwen2') loads on older
        # llama-cpp-python; newer 'qwen3'/'qwen35' may not. So pick 2.5 first.
        for c in candidates:
            n = c.name.lower()
            if _is_ablit_qwen(n) and ("qwen2" in n or "2.5" in n or "2_5" in n):
                return str(c)
        for c in candidates:  # then any abliterated Qwen
            if _is_ablit_qwen(c.name.lower()):
                return str(c)
        return str(candidates[0]) if candidates else ""

    def load_prompt_llm(self) -> None:
        """Load the prompt-writer LLM (abliterated Qwen GGUF) on GPU.

        Mutually exclusive with all generation pipelines + the chat Qwen.
        Call from a worker thread.
        """
        with self._load_lock:
            if self._prompt_llm is not None:
                return
            # Free VRAM from every competing pipeline.
            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_qwen_vl()
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self.unload_vace_multitalk_pipeline()
            self.unload_triposr_pipeline()
            self._force_gc()

            gguf = self._resolve_prompt_llm_path()
            if not gguf:
                raise RuntimeError(
                    "No prompt-LLM GGUF found. Place an abliterated Qwen .gguf in "
                    "your models dir, or set model_paths['prompt_llm_gguf']."
                )
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is required for the prompt LLM "
                    "(pip install llama-cpp-python)."
                ) from exc
            n_ctx = int(getattr(self.global_config, "prompt_llm_ctx", 8192) or 8192)
            logger.info("Loading prompt LLM (GGUF) from %s (n_ctx=%d) ...", gguf, n_ctx)
            self._prompt_llm = Llama(
                model_path=str(gguf),
                n_gpu_layers=-1,   # offload all layers to GPU
                n_ctx=n_ctx,
                verbose=False,
            )
            logger.info("Prompt LLM loaded.")

    def unload_prompt_llm(self) -> None:
        """Release the prompt-writer LLM and free VRAM."""
        if self._prompt_llm is None:
            return
        try:
            del self._prompt_llm
        except Exception:
            pass
        self._prompt_llm = None
        self._force_gc()
        logger.info("Prompt LLM unloaded.")

    def prompt_llm_chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.92,
    ) -> str:
        """Chat-completion with the loaded prompt LLM. Returns the reply text."""
        llm = self._prompt_llm
        if llm is None:
            raise RuntimeError("Prompt LLM not loaded. Call load_prompt_llm() first.")
        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    def generate_chat_response(
        self,
        messages: list[dict],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat response using the loaded Qwen model.

        Args:
            messages: List of {role, content} dicts (system/user/assistant).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The assistant's reply text.
        """
        # Grab local references so a concurrent unload_qwen() can't
        # null them out while we're generating.
        model = self._qwen_model
        tokenizer = self._qwen_tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError("Qwen model not loaded. Call load_qwen() first.")

        import torch

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
            )

        new_tokens = outputs[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -- Qwen 2.5 VL 7B (vision-language for image captioning) ---------------

    @property
    def qwen_vl_loaded(self) -> bool:
        return self._qwen_vl_model is not None

    def load_qwen_vl(self) -> None:
        """Load Qwen 2.5 VL 7B Instruct for image captioning.

        Unloads all other pipelines (including text Qwen) first.
        Call from a worker thread.
        """
        with self._load_lock:
            if self._qwen_vl_model is not None:
                return

            self.unload_qwen()
            self.unload_prompt_llm()
            self.unload_video_pipelines()
            self.unload_sd_pipelines()
            self.unload_flux_pipelines()
            self.unload_zimage_pipelines()
            self.unload_lipsync_pipeline()
            self.unload_vace_multitalk_pipeline()
            self.unload_triposr_pipeline()
            self._force_gc()

            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

            logger.info("Loading Qwen 2.5 VL 3B Instruct (bf16)...")
            self._qwen_vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._qwen_vl_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            logger.info("Qwen 2.5 VL 3B loaded.")

    def unload_qwen_vl(self) -> None:
        """Release Qwen VL model and free VRAM."""
        if self._qwen_vl_model is None:
            return
        del self._qwen_vl_model
        del self._qwen_vl_processor
        self._qwen_vl_model = None
        self._qwen_vl_processor = None
        self._force_gc()
        logger.info("Qwen VL model unloaded.")

    def caption_image(self, image_path: str, prompt: str = "", max_new_tokens: int = 300) -> str:
        """Generate a caption for an image using Qwen 2.5 VL.

        Args:
            image_path: Path to the image file.
            prompt: Optional instruction (default: general captioning).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            The generated caption text.
        """
        if self._qwen_vl_model is None or self._qwen_vl_processor is None:
            raise RuntimeError("Qwen VL not loaded. Call load_qwen_vl() first.")

        from PIL import Image as PILImage

        if not prompt:
            prompt = (
                "Describe this image in one detailed sentence covering the subject, "
                "appearance, pose, expression, clothing, setting, lighting, and style. "
                "Be specific and factual. Output only the description."
            )

        image = PILImage.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]

        text = self._qwen_vl_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._qwen_vl_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._qwen_vl_model.device)

        import torch
        with torch.no_grad():
            output_ids = self._qwen_vl_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Decode only new tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._qwen_vl_processor.decode(generated, skip_special_tokens=True).strip()

    def unload_pipelines(self) -> None:
        """Release GPU resources."""
        self.unload_qwen()
        self.unload_prompt_llm()
        self.unload_qwen_vl()
        self.unload_video_pipelines()
        self.unload_sd_pipelines()
        self.unload_flux_pipelines()
        self.unload_zimage_pipelines()
        self.unload_lipsync_pipeline()
        self.unload_vace_multitalk_pipeline()
        self.unload_triposr_pipeline()
        self._force_gc()
        logger.info("All pipelines unloaded.")

    @staticmethod
    def _force_gc() -> None:
        """Aggressively reclaim memory after unloading pipelines."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        gc.collect()

    # -- Project helpers --------------------------------------------------

    @property
    def current_project(self) -> Optional[str]:
        """Convenience -- actual selection lives in the UI."""
        projects = self.project_manager.list_projects()
        return projects[0] if projects else None
