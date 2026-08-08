"""Model family strategy — adapts UI and pipeline dispatch per model type."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelFamilyStrategy:
    """Describes how to configure UI and dispatch generation for a model family."""

    family: str              # "sd15", "sdxl", "flux", "zimage"
    display_name: str
    config_prefix: str       # "img", "flux", "zimage"
    pipeline_attr: str       # "img_pipeline", "flux_pipeline", "zimage_pipeline"
    load_method: str         # "load_sd_pipelines", "load_flux_pipelines", etc.
    unload_method: str       # "unload_sd_pipelines", etc.
    lora_dir_key: str        # key in global_config.model_paths

    # Worker dotted import paths (module:class)
    txt2img_worker: str
    img2img_worker: str
    inpaint_worker: str

    # UI visibility flags
    has_checkpoint_selector: bool = True
    has_vae_selector: bool = True
    has_sampler: bool = True
    has_clip_skip: bool = True
    has_refiner: bool = False
    has_max_seq_len: bool = False
    has_remove_bg: bool = False
    has_inpaint_advanced: bool = True  # blur, fill mode, padding
    has_negative_prompt: bool = True
    has_scheduler: bool = True

    # Defaults
    default_steps: int = 20
    default_cfg: float = 7.0
    default_width: int = 512
    default_height: int = 512
    cfg_range: tuple[float, float] = (0.0, 30.0)
    scheduler_choices: list[str] = field(default_factory=list)

    # --- helpers ---

    def get_config(self, cfg, field_name: str):
        """Read ``{prefix}_{field_name}`` from a ProjectConfig."""
        return getattr(cfg, f"{self.config_prefix}_{field_name}", None)

    def set_config(self, cfg, field_name: str, value):
        """Write ``{prefix}_{field_name}`` on a ProjectConfig."""
        setattr(cfg, f"{self.config_prefix}_{field_name}", value)

    def get_pipeline(self, state):
        """Return the loaded pipeline from AppState, or None."""
        return getattr(state, self.pipeline_attr, None)

    def resolve_worker(self, mode: str):
        """Import and return the worker class for *mode* (txt2img/img2img/inpaint)."""
        spec = {"txt2img": self.txt2img_worker,
                "img2img": self.img2img_worker,
                "inpaint": self.inpaint_worker}[mode]
        module_path, cls_name = spec.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)


# ── Strategy instances ────────────────────────────────────────────────────────

SD15_STRATEGY = ModelFamilyStrategy(
    family="sd15",
    display_name="SD 1.5",
    config_prefix="img",
    pipeline_attr="img_pipeline",
    load_method="load_sd_pipelines",
    unload_method="unload_sd_pipelines",
    lora_dir_key="sd_lora_dir",
    txt2img_worker="sdqt.workers.image.Txt2ImgWorker",
    img2img_worker="sdqt.workers.image.Img2ImgWorker",
    inpaint_worker="sdqt.workers.image.InpaintWorker",
    has_checkpoint_selector=True,
    has_vae_selector=True,
    has_sampler=True,
    has_clip_skip=True,
    has_refiner=False,
    has_max_seq_len=False,
    has_remove_bg=False,
    has_inpaint_advanced=True,
    has_negative_prompt=True,
    has_scheduler=True,
    default_steps=20,
    default_cfg=7.0,
    default_width=512,
    default_height=512,
    cfg_range=(1.0, 30.0),
)

SDXL_STRATEGY = ModelFamilyStrategy(
    family="sdxl",
    display_name="SDXL",
    config_prefix="img",
    pipeline_attr="img_pipeline",
    load_method="load_sd_pipelines",
    unload_method="unload_sd_pipelines",
    lora_dir_key="sd_lora_dir",
    txt2img_worker="sdqt.workers.image.Txt2ImgWorker",
    img2img_worker="sdqt.workers.image.Img2ImgWorker",
    inpaint_worker="sdqt.workers.image.InpaintWorker",
    has_checkpoint_selector=True,
    has_vae_selector=True,
    has_sampler=True,
    has_clip_skip=True,
    has_refiner=True,
    has_max_seq_len=False,
    has_remove_bg=False,
    has_inpaint_advanced=True,
    has_negative_prompt=True,
    has_scheduler=True,
    default_steps=20,
    default_cfg=7.0,
    default_width=1024,
    default_height=1024,
    cfg_range=(1.0, 30.0),
)

FLUX_STRATEGY = ModelFamilyStrategy(
    family="flux",
    display_name="FLUX",
    config_prefix="flux",
    pipeline_attr="flux_pipeline",
    load_method="load_flux_pipelines",
    unload_method="unload_flux_pipelines",
    lora_dir_key="flux_lora_dir",
    txt2img_worker="sdqt.workers.flux.FluxTxt2ImgWorker",
    img2img_worker="sdqt.workers.flux.FluxImg2ImgWorker",
    inpaint_worker="sdqt.workers.flux.FluxFillWorker",
    has_checkpoint_selector=False,
    has_vae_selector=False,
    has_sampler=False,
    has_clip_skip=False,
    has_refiner=False,
    has_max_seq_len=True,
    has_remove_bg=True,
    has_inpaint_advanced=False,
    has_negative_prompt=True,
    has_scheduler=True,
    default_steps=35,
    default_cfg=3.5,
    default_width=1024,
    default_height=1024,
    cfg_range=(0.0, 30.0),
    scheduler_choices=["Euler", "Heun", "DPM++ 2M", "DEIS 2M", "UniPC"],
)

ZIMAGE_STRATEGY = ModelFamilyStrategy(
    family="zimage",
    display_name="Z-Image",
    config_prefix="zimage",
    pipeline_attr="zimage_pipeline",
    load_method="load_zimage_pipelines",
    unload_method="unload_zimage_pipelines",
    lora_dir_key="zimage_lora_dir",
    txt2img_worker="sdqt.workers.zimage.ZImageTxt2ImgWorker",
    img2img_worker="sdqt.workers.zimage.ZImageImg2ImgWorker",
    inpaint_worker="sdqt.workers.zimage.ZImageInpaintWorker",
    has_checkpoint_selector=False,
    has_vae_selector=False,
    has_sampler=False,
    has_clip_skip=False,
    has_refiner=False,
    has_max_seq_len=False,
    has_remove_bg=True,
    has_inpaint_advanced=False,
    has_negative_prompt=True,
    has_scheduler=False,
    default_steps=9,
    default_cfg=0.0,
    default_width=1024,
    default_height=1024,
    cfg_range=(0.0, 10.0),
)

STRATEGY_MAP: dict[str, ModelFamilyStrategy] = {
    "sd15": SD15_STRATEGY,
    "sdxl": SDXL_STRATEGY,
    "flux": FLUX_STRATEGY,
    "zimage": ZIMAGE_STRATEGY,
}
