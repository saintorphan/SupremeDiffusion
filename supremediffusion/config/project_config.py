"""Per-project configuration for Supreme Diffusion."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .defaults import PROJECT_DEFAULTS

logger = logging.getLogger(__name__)

# Sentinel so cfg_get can tell "attribute genuinely absent" apart from a real
# stored value of None.
_MISSING = object()


def cfg_get(config: object, name: str, default: Any = None) -> Any:
    """Read ``config.<name>`` with dev-mode validation of the attribute.

    Behaves like ``getattr(config, name, default)`` but, when the attribute is
    missing on ``config``, it flags the access instead of silently masking what
    is almost always a mistyped config-key name. If the ``SDQT_DEBUG_CONFIG``
    environment flag is set (any truthy value) the missing access raises an
    ``AttributeError``; otherwise it logs a warning and returns ``default``.

    Catches the ``getattr-default-masks-typo`` class of bug where a typo'd field
    name silently falls back to a default forever.
    """
    value = getattr(config, name, _MISSING)
    if value is _MISSING:
        msg = (
            f"cfg_get: '{name}' is not an attribute of "
            f"{type(config).__name__}; returning default ({default!r})"
        )
        if os.environ.get("SDQT_DEBUG_CONFIG"):
            raise AttributeError(msg)
        logger.warning(msg)
        return default
    return value


def _default_loras() -> list[str]:
    return list(PROJECT_DEFAULTS["activated_loras"])


def _default_slg_layers() -> list[int]:
    return list(PROJECT_DEFAULTS["slg_layers"])


def _default_img_loras() -> list:
    return list(PROJECT_DEFAULTS["img_loras"])


@dataclass
class ProjectConfig:
    """Settings that live inside each project directory."""

    last_parent_tab: str = PROJECT_DEFAULTS["last_parent_tab"]
    last_child_tab: str = PROJECT_DEFAULTS["last_child_tab"]
    mode: int = PROJECT_DEFAULTS["mode"]
    # Color profile slug (see supremediffusion.config.color_profile._PRESETS).
    # Empty string means "inherit GlobalConfig.default_color_profile". Existing
    # projects without this field set fall back to bt709_full.
    color_profile: str = ""
    # Input file paths (remembered per project)
    image_path: str = PROJECT_DEFAULTS["image_path"]
    first_frame_path: str = PROJECT_DEFAULTS["first_frame_path"]
    last_frame_path: str = PROJECT_DEFAULTS["last_frame_path"]
    guidance_video_path_m1: str = PROJECT_DEFAULTS["guidance_video_path_m1"]
    guidance_video_path_m2: str = PROJECT_DEFAULTS["guidance_video_path_m2"]
    m3_video_path: str = PROJECT_DEFAULTS["m3_video_path"]
    preview_path_m1: str = PROJECT_DEFAULTS["preview_path_m1"]
    preview_path_m2: str = PROJECT_DEFAULTS["preview_path_m2"]
    ve_preview_path: str = PROJECT_DEFAULTS["ve_preview_path"]
    ve_video_path: str = PROJECT_DEFAULTS["ve_video_path"]
    ve_last_frame_path: str = PROJECT_DEFAULTS["ve_last_frame_path"]
    ve_guidance_video_path: str = PROJECT_DEFAULTS["ve_guidance_video_path"]
    ve_guidance_video_frame: str = PROJECT_DEFAULTS["ve_guidance_video_frame"]
    ltx_inpaint_mask_path: str = PROJECT_DEFAULTS["ltx_inpaint_mask_path"]
    # LTX Union-Control (IC-LoRA) inputs — the Guidance Video is the control source.
    ltx_control_type: str = PROJECT_DEFAULTS["ltx_control_type"]
    ltx_control_strength: float = PROJECT_DEFAULTS["ltx_control_strength"]
    ltx_control_start_seconds: float = PROJECT_DEFAULTS["ltx_control_start_seconds"]
    ltx_control_ramp_frames: int = PROJECT_DEFAULTS["ltx_control_ramp_frames"]
    # LTX AV talking-head (community LoRA) driving-audio source.
    ltx_audio_path: str = PROJECT_DEFAULTS["ltx_audio_path"]
    video_backend: str = PROJECT_DEFAULTS["video_backend"]
    model_type: str = PROJECT_DEFAULTS["model_type"]
    resolution: str = PROJECT_DEFAULTS["resolution"]
    fps: int = PROJECT_DEFAULTS["fps"]
    video_length: int = PROJECT_DEFAULTS["video_length"]
    prompt: str = PROJECT_DEFAULTS["prompt"]
    negative_prompt: str = PROJECT_DEFAULTS["negative_prompt"]
    num_inference_steps: int = PROJECT_DEFAULTS["num_inference_steps"]
    guidance_scale: float = PROJECT_DEFAULTS["guidance_scale"]
    guidance2_scale: float = PROJECT_DEFAULTS["guidance2_scale"]
    alt_guidance_scale: float = PROJECT_DEFAULTS["alt_guidance_scale"]
    guidance_phases: int = PROJECT_DEFAULTS["guidance_phases"]
    switch_threshold: int = PROJECT_DEFAULTS["switch_threshold"]
    flow_shift: float = PROJECT_DEFAULTS["flow_shift"]
    sample_solver: str = PROJECT_DEFAULTS["sample_solver"]
    seed: int = PROJECT_DEFAULTS["seed"]
    denoising_strength: float = PROJECT_DEFAULTS["denoising_strength"]
    masking_strength: float = PROJECT_DEFAULTS["masking_strength"]
    activated_loras: list[str] = field(default_factory=_default_loras)
    loras_multipliers: str = PROJECT_DEFAULTS["loras_multipliers"]
    NAG_scale: float = PROJECT_DEFAULTS["NAG_scale"]
    NAG_tau: float = PROJECT_DEFAULTS["NAG_tau"]
    NAG_alpha: float = PROJECT_DEFAULTS["NAG_alpha"]
    cfg_star_switch: int = PROJECT_DEFAULTS["cfg_star_switch"]
    cfg_zero_step: int = PROJECT_DEFAULTS["cfg_zero_step"]
    tea_cache_setting: str = PROJECT_DEFAULTS["tea_cache_setting"]
    tea_cache_start_step_perc: float = PROJECT_DEFAULTS["tea_cache_start_step_perc"]
    use_guidance_m1: bool = PROJECT_DEFAULTS["use_guidance_m1"]
    generate_static_m1: bool = PROJECT_DEFAULTS["generate_static_m1"]
    use_guidance_m2: bool = PROJECT_DEFAULTS["use_guidance_m2"]
    generate_static_m2: bool = PROJECT_DEFAULTS["generate_static_m2"]
    guidance_video_source: str = PROJECT_DEFAULTS["guidance_video_source"]
    guidance_video_frame: str = PROJECT_DEFAULTS["guidance_video_frame"]
    # Quality / Advanced enable gates
    quality_overrides_enabled: bool = PROJECT_DEFAULTS["quality_overrides_enabled"]
    nag_tea_enabled: bool = PROJECT_DEFAULTS["nag_tea_enabled"]
    # Quality
    slg_switch: int = PROJECT_DEFAULTS["slg_switch"]
    slg_layers: list[int] = field(default_factory=_default_slg_layers)
    slg_start_perc: int = PROJECT_DEFAULTS["slg_start_perc"]
    slg_end_perc: int = PROJECT_DEFAULTS["slg_end_perc"]
    apg_switch: int = PROJECT_DEFAULTS["apg_switch"]
    motion_amplitude: float = PROJECT_DEFAULTS["motion_amplitude"]
    self_refiner_setting: int = PROJECT_DEFAULTS["self_refiner_setting"]
    self_refiner_uncertainty: float = PROJECT_DEFAULTS["self_refiner_uncertainty"]
    self_refiner_certainty_skip: float = PROJECT_DEFAULTS["self_refiner_certainty_skip"]
    # Discard last frames
    discard_last_frames: int = PROJECT_DEFAULTS["discard_last_frames"]
    # Color correction
    color_correction_strength: float = PROJECT_DEFAULTS["color_correction_strength"]
    color_correction_method: str = PROJECT_DEFAULTS["color_correction_method"]
    color_anchor_persistence: float = PROJECT_DEFAULTS["color_anchor_persistence"]
    anchor_prematch_strength: float = PROJECT_DEFAULTS["anchor_prematch_strength"]
    # Wan 2.2 LoRA presets (Lightning / SVI Pro)
    acceleration_preset: str = PROJECT_DEFAULTS["acceleration_preset"]
    antidrift_preset: str = PROJECT_DEFAULTS["antidrift_preset"]
    # Post-processing pipeline (Phase 3 — unified Quality preset)
    output_quality: str = PROJECT_DEFAULTS["output_quality"]
    upscale_model: str = PROJECT_DEFAULTS["upscale_model"]
    rife_multiplier: float = PROJECT_DEFAULTS["rife_multiplier"]
    # SVI 2 Pro — Sliding Window + Anchor Image
    sliding_window_size: int = PROJECT_DEFAULTS["sliding_window_size"]
    sliding_window_overlap: int = PROJECT_DEFAULTS["sliding_window_overlap"]
    sliding_window_color_correction_strength: float = PROJECT_DEFAULTS["sliding_window_color_correction_strength"]
    sliding_window_overlap_noise: float = PROJECT_DEFAULTS["sliding_window_overlap_noise"]
    sliding_window_discard_last_frames: int = PROJECT_DEFAULTS["sliding_window_discard_last_frames"]
    anchor_image_mode: str = PROJECT_DEFAULTS["anchor_image_mode"]
    anchor_image_path: str = PROJECT_DEFAULTS["anchor_image_path"]
    # Post-processing
    temporal_upsampling: str = PROJECT_DEFAULTS["temporal_upsampling"]
    spatial_upsampling: str = PROJECT_DEFAULTS["spatial_upsampling"]
    film_grain_intensity: float = PROJECT_DEFAULTS["film_grain_intensity"]
    film_grain_saturation: float = PROJECT_DEFAULTS["film_grain_saturation"]
    # Trim / Crop persisted paths
    tc_source_path: str = PROJECT_DEFAULTS["tc_source_path"]
    trim_video_path: str = PROJECT_DEFAULTS["trim_video_path"]
    crop_video_path: str = PROJECT_DEFAULTS["crop_video_path"]
    # Image suite source image paths
    img2img_source_path: str = PROJECT_DEFAULTS["img2img_source_path"]
    inpaint_source_path: str = PROJECT_DEFAULTS["inpaint_source_path"]
    imgedit_source_path: str = PROJECT_DEFAULTS["imgedit_source_path"]
    cropzoom_source_path: str = PROJECT_DEFAULTS["cropzoom_source_path"]
    # Image generation settings
    selected_image_model_filename: str = PROJECT_DEFAULTS["selected_image_model_filename"]
    selected_image_model_type: str = PROJECT_DEFAULTS["selected_image_model_type"]
    selected_image_model_subtype: str = PROJECT_DEFAULTS["selected_image_model_subtype"]
    img_checkpoint: str = PROJECT_DEFAULTS["img_checkpoint"]
    img_vae: str = PROJECT_DEFAULTS["img_vae"]
    sd_vae_precision: str = PROJECT_DEFAULTS["sd_vae_precision"]
    img_sampler: str = PROJECT_DEFAULTS["img_sampler"]
    img_scheduler: str = PROJECT_DEFAULTS["img_scheduler"]
    img_steps: int = PROJECT_DEFAULTS["img_steps"]
    img_cfg_scale: float = PROJECT_DEFAULTS["img_cfg_scale"]
    img_width: int = PROJECT_DEFAULTS["img_width"]
    img_height: int = PROJECT_DEFAULTS["img_height"]
    img_seed: int = PROJECT_DEFAULTS["img_seed"]
    img_batch_size: int = PROJECT_DEFAULTS["img_batch_size"]
    img_batch_count: int = PROJECT_DEFAULTS["img_batch_count"]
    img_prompt: str = PROJECT_DEFAULTS["img_prompt"]
    img_negative_prompt: str = PROJECT_DEFAULTS["img_negative_prompt"]
    img_denoising_strength: float = PROJECT_DEFAULTS["img_denoising_strength"]
    # Optional IP-Adapter reference-look for txt2img (e.g. character creator
    # applying a canonical look to generated poses). scale 0 = off.
    img_ipadapter_path: str = PROJECT_DEFAULTS["img_ipadapter_path"]
    img_ipadapter_scale: float = PROJECT_DEFAULTS["img_ipadapter_scale"]
    img_ipadapter_variant: str = PROJECT_DEFAULTS["img_ipadapter_variant"]
    img_resize_mode: int = PROJECT_DEFAULTS["img_resize_mode"]
    img_mask_blur: int = PROJECT_DEFAULTS["img_mask_blur"]
    img_inpainting_fill: int = PROJECT_DEFAULTS["img_inpainting_fill"]
    img_mask_invert: bool = PROJECT_DEFAULTS["img_mask_invert"]
    img_inpaint_laplacian_blend: bool = PROJECT_DEFAULTS["img_inpaint_laplacian_blend"]
    img_inpaint_full_res: bool = PROJECT_DEFAULTS["img_inpaint_full_res"]
    img_inpaint_full_res_padding: int = PROJECT_DEFAULTS["img_inpaint_full_res_padding"]
    img_clip_skip: int = PROJECT_DEFAULTS["img_clip_skip"]
    img_loras: list = field(default_factory=_default_img_loras)
    img_lora_multipliers: str = PROJECT_DEFAULTS["img_lora_multipliers"]
    # SDXL Refiner settings
    img_refiner_enabled: bool = PROJECT_DEFAULTS["img_refiner_enabled"]
    img_refiner_checkpoint: str = PROJECT_DEFAULTS["img_refiner_checkpoint"]
    img_refiner_switch_at: float = PROJECT_DEFAULTS["img_refiner_switch_at"]
    img_refiner_steps: int = PROJECT_DEFAULTS["img_refiner_steps"]
    img_refiner_cfg_scale: float = PROJECT_DEFAULTS["img_refiner_cfg_scale"]
    # Face swap settings
    faceswap_source_path: str = PROJECT_DEFAULTS["faceswap_source_path"]
    faceswap_target_path: str = PROJECT_DEFAULTS["faceswap_target_path"]
    faceswap_model: str = PROJECT_DEFAULTS["faceswap_model"]
    faceswap_enhancer: str = PROJECT_DEFAULTS["faceswap_enhancer"]
    faceswap_blend_ratio: float = PROJECT_DEFAULTS["faceswap_blend_ratio"]
    faceswap_source_face_idx: int = PROJECT_DEFAULTS["faceswap_source_face_idx"]
    faceswap_target_face_idx: int = PROJECT_DEFAULTS["faceswap_target_face_idx"]
    faceswap_swap_all: bool = PROJECT_DEFAULTS["faceswap_swap_all"]
    faceswap_enhancer_strength: float = PROJECT_DEFAULTS["faceswap_enhancer_strength"]
    faceswap_enhance_source: bool = PROJECT_DEFAULTS["faceswap_enhance_source"]
    # ADetailer (post-gen face refinement)
    adetailer_enabled: bool = PROJECT_DEFAULTS["adetailer_enabled"]
    adetailer_detector: str = PROJECT_DEFAULTS["adetailer_detector"]
    adetailer_threshold: float = PROJECT_DEFAULTS["adetailer_threshold"]
    adetailer_dilation_pct: float = PROJECT_DEFAULTS["adetailer_dilation_pct"]
    adetailer_feather_px: int = PROJECT_DEFAULTS["adetailer_feather_px"]
    adetailer_denoise: float = PROJECT_DEFAULTS["adetailer_denoise"]
    adetailer_steps_add: int = PROJECT_DEFAULTS["adetailer_steps_add"]
    adetailer_inpaint_size: int = PROJECT_DEFAULTS["adetailer_inpaint_size"]
    adetailer_prompt_override: str = PROJECT_DEFAULTS["adetailer_prompt_override"]
    adetailer_neg_prompt_override: str = PROJECT_DEFAULTS["adetailer_neg_prompt_override"]
    # Body Double settings
    bodydouble_checkpoint: str = PROJECT_DEFAULTS["bodydouble_checkpoint"]
    bodydouble_sampler: str = PROJECT_DEFAULTS["bodydouble_sampler"]
    bodydouble_scheduler: str = PROJECT_DEFAULTS["bodydouble_scheduler"]
    bodydouble_source_path: str = PROJECT_DEFAULTS["bodydouble_source_path"]
    bodydouble_target_path: str = PROJECT_DEFAULTS["bodydouble_target_path"]
    bodydouble_prompt: str = PROJECT_DEFAULTS["bodydouble_prompt"]
    bodydouble_negative_prompt: str = PROJECT_DEFAULTS["bodydouble_negative_prompt"]
    bodydouble_steps: int = PROJECT_DEFAULTS["bodydouble_steps"]
    bodydouble_cfg_scale: float = PROJECT_DEFAULTS["bodydouble_cfg_scale"]
    bodydouble_seed: int = PROJECT_DEFAULTS["bodydouble_seed"]
    bodydouble_denoising_strength: float = PROJECT_DEFAULTS["bodydouble_denoising_strength"]
    bodydouble_controlnet_strength: float = PROJECT_DEFAULTS["bodydouble_controlnet_strength"]
    bodydouble_ip_adapter_scale: float = PROJECT_DEFAULTS["bodydouble_ip_adapter_scale"]
    bodydouble_mask_blur: int = PROJECT_DEFAULTS["bodydouble_mask_blur"]
    bodydouble_mask_path: str = PROJECT_DEFAULTS["bodydouble_mask_path"]
    bodydouble_controlnet_type: str = PROJECT_DEFAULTS["bodydouble_controlnet_type"]
    bodydouble_controlnet2_type: str = PROJECT_DEFAULTS["bodydouble_controlnet2_type"]
    bodydouble_controlnet2_strength: float = PROJECT_DEFAULTS["bodydouble_controlnet2_strength"]
    bodydouble_ip_adapter_variant: str = PROJECT_DEFAULTS["bodydouble_ip_adapter_variant"]
    bodydouble_width: int = PROJECT_DEFAULTS["bodydouble_width"]
    bodydouble_height: int = PROJECT_DEFAULTS["bodydouble_height"]
    bodydouble_guidance_start: float = PROJECT_DEFAULTS["bodydouble_guidance_start"]
    bodydouble_guidance_end: float = PROJECT_DEFAULTS["bodydouble_guidance_end"]
    bodydouble_clip_skip: int = PROJECT_DEFAULTS["bodydouble_clip_skip"]
    bodydouble_style_preset: str = PROJECT_DEFAULTS["bodydouble_style_preset"]
    bodydouble_ip_adapter2_variant: str = PROJECT_DEFAULTS["bodydouble_ip_adapter2_variant"]
    bodydouble_ip_adapter2_scale: float = PROJECT_DEFAULTS["bodydouble_ip_adapter2_scale"]
    bodydouble_style_ref_path: str = PROJECT_DEFAULTS["bodydouble_style_ref_path"]
    # RePose settings
    repose_checkpoint: str = PROJECT_DEFAULTS["repose_checkpoint"]
    repose_sampler: str = PROJECT_DEFAULTS["repose_sampler"]
    repose_scheduler: str = PROJECT_DEFAULTS["repose_scheduler"]
    repose_source_path: str = PROJECT_DEFAULTS["repose_source_path"]
    repose_pose_path: str = PROJECT_DEFAULTS["repose_pose_path"]
    repose_prompt: str = PROJECT_DEFAULTS["repose_prompt"]
    repose_negative_prompt: str = PROJECT_DEFAULTS["repose_negative_prompt"]
    repose_steps: int = PROJECT_DEFAULTS["repose_steps"]
    repose_cfg_scale: float = PROJECT_DEFAULTS["repose_cfg_scale"]
    repose_seed: int = PROJECT_DEFAULTS["repose_seed"]
    repose_denoising_strength: float = PROJECT_DEFAULTS["repose_denoising_strength"]
    repose_controlnet_strength: float = PROJECT_DEFAULTS["repose_controlnet_strength"]
    repose_ip_adapter_scale: float = PROJECT_DEFAULTS["repose_ip_adapter_scale"]
    repose_mask_blur: int = PROJECT_DEFAULTS["repose_mask_blur"]
    repose_mask_path: str = PROJECT_DEFAULTS["repose_mask_path"]
    repose_controlnet_type: str = PROJECT_DEFAULTS["repose_controlnet_type"]
    repose_controlnet2_type: str = PROJECT_DEFAULTS["repose_controlnet2_type"]
    repose_controlnet2_strength: float = PROJECT_DEFAULTS["repose_controlnet2_strength"]
    repose_ip_adapter_variant: str = PROJECT_DEFAULTS["repose_ip_adapter_variant"]
    repose_width: int = PROJECT_DEFAULTS["repose_width"]
    repose_height: int = PROJECT_DEFAULTS["repose_height"]
    repose_guidance_start: float = PROJECT_DEFAULTS["repose_guidance_start"]
    repose_guidance_end: float = PROJECT_DEFAULTS["repose_guidance_end"]
    repose_clip_skip: int = PROJECT_DEFAULTS["repose_clip_skip"]
    repose_preprocessed: bool = PROJECT_DEFAULTS["repose_preprocessed"]
    repose_appearance_path: str = PROJECT_DEFAULTS["repose_appearance_path"]
    repose_style_preset: str = PROJECT_DEFAULTS["repose_style_preset"]
    repose_ip_adapter2_variant: str = PROJECT_DEFAULTS["repose_ip_adapter2_variant"]
    repose_ip_adapter2_scale: float = PROJECT_DEFAULTS["repose_ip_adapter2_scale"]
    repose_style_ref_path: str = PROJECT_DEFAULTS["repose_style_ref_path"]
    # ControlNet tab settings
    controlnet_mode: str = PROJECT_DEFAULTS["controlnet_mode"]
    controlnet_checkpoint: str = PROJECT_DEFAULTS["controlnet_checkpoint"]
    controlnet_sampler: str = PROJECT_DEFAULTS["controlnet_sampler"]
    controlnet_scheduler: str = PROJECT_DEFAULTS["controlnet_scheduler"]
    controlnet_prompt: str = PROJECT_DEFAULTS["controlnet_prompt"]
    controlnet_negative_prompt: str = PROJECT_DEFAULTS["controlnet_negative_prompt"]
    controlnet_steps: int = PROJECT_DEFAULTS["controlnet_steps"]
    controlnet_cfg_scale: float = PROJECT_DEFAULTS["controlnet_cfg_scale"]
    controlnet_seed: int = PROJECT_DEFAULTS["controlnet_seed"]
    controlnet_width: int = PROJECT_DEFAULTS["controlnet_width"]
    controlnet_height: int = PROJECT_DEFAULTS["controlnet_height"]
    controlnet_denoising_strength: float = PROJECT_DEFAULTS["controlnet_denoising_strength"]
    controlnet_type: str = PROJECT_DEFAULTS["controlnet_type"]
    controlnet_type2: str = PROJECT_DEFAULTS["controlnet_type2"]
    controlnet_strength: float = PROJECT_DEFAULTS["controlnet_strength"]
    controlnet_strength2: float = PROJECT_DEFAULTS["controlnet_strength2"]
    controlnet_source_path: str = PROJECT_DEFAULTS["controlnet_source_path"]
    controlnet_condition_path: str = PROJECT_DEFAULTS["controlnet_condition_path"]
    controlnet_condition2_path: str = PROJECT_DEFAULTS["controlnet_condition2_path"]
    controlnet_mask_path: str = PROJECT_DEFAULTS["controlnet_mask_path"]
    controlnet_batch_count: int = PROJECT_DEFAULTS["controlnet_batch_count"]
    controlnet_guidance_start: float = PROJECT_DEFAULTS["controlnet_guidance_start"]
    controlnet_guidance_end: float = PROJECT_DEFAULTS["controlnet_guidance_end"]
    controlnet_guess_mode: bool = PROJECT_DEFAULTS["controlnet_guess_mode"]
    controlnet_clip_skip: int = PROJECT_DEFAULTS["controlnet_clip_skip"]
    controlnet_preprocessed: bool = PROJECT_DEFAULTS["controlnet_preprocessed"]
    # ControlNet preprocessor params
    cn_canny_low: int = PROJECT_DEFAULTS["cn_canny_low"]
    cn_canny_high: int = PROJECT_DEFAULTS["cn_canny_high"]
    cn_detect_resolution: int = PROJECT_DEFAULTS["cn_detect_resolution"]
    cn_image_resolution: int = PROJECT_DEFAULTS["cn_image_resolution"]
    cn_openpose_include_hand: bool = PROJECT_DEFAULTS["cn_openpose_include_hand"]
    cn_openpose_include_face: bool = PROJECT_DEFAULTS["cn_openpose_include_face"]
    # Longshot settings
    ls_source_path: str = PROJECT_DEFAULTS["ls_source_path"]
    ls_range_start: float = PROJECT_DEFAULTS["ls_range_start"]
    ls_range_end: float = PROJECT_DEFAULTS["ls_range_end"]
    ls_subdivisions: int = PROJECT_DEFAULTS["ls_subdivisions"]
    ls_gapfill_duration: float = PROJECT_DEFAULTS["ls_gapfill_duration"]
    ls_final_video_path: str = PROJECT_DEFAULTS["ls_final_video_path"]
    # FLUX generation settings
    flux_prompt: str = PROJECT_DEFAULTS["flux_prompt"]
    flux_negative_prompt: str = PROJECT_DEFAULTS["flux_negative_prompt"]
    flux_steps: int = PROJECT_DEFAULTS["flux_steps"]
    flux_cfg_scale: float = PROJECT_DEFAULTS["flux_cfg_scale"]
    flux_width: int = PROJECT_DEFAULTS["flux_width"]
    flux_height: int = PROJECT_DEFAULTS["flux_height"]
    flux_seed: int = PROJECT_DEFAULTS["flux_seed"]
    flux_batch_count: int = PROJECT_DEFAULTS["flux_batch_count"]
    flux_denoising_strength: float = PROJECT_DEFAULTS["flux_denoising_strength"]
    flux_max_seq_len: int = PROJECT_DEFAULTS["flux_max_seq_len"]
    flux_scheduler: str = PROJECT_DEFAULTS["flux_scheduler"]
    flux_activated_loras: list = field(default_factory=lambda: list(PROJECT_DEFAULTS["flux_activated_loras"]))
    flux_lora_multipliers: str = PROJECT_DEFAULTS["flux_lora_multipliers"]
    flux_source_path: str = PROJECT_DEFAULTS["flux_source_path"]
    flux_remove_bg: bool = PROJECT_DEFAULTS["flux_remove_bg"]
    flux_img2img_source_path: str = PROJECT_DEFAULTS["flux_img2img_source_path"]
    flux_fill_source_path: str = PROJECT_DEFAULTS["flux_fill_source_path"]
    flux_fill_prompt: str = PROJECT_DEFAULTS["flux_fill_prompt"]
    flux_fill_negative_prompt: str = PROJECT_DEFAULTS["flux_fill_negative_prompt"]
    flux_fill_steps: int = PROJECT_DEFAULTS["flux_fill_steps"]
    flux_fill_cfg_scale: float = PROJECT_DEFAULTS["flux_fill_cfg_scale"]
    flux_fill_strength: float = PROJECT_DEFAULTS["flux_fill_strength"]
    flux_fill_width: int = PROJECT_DEFAULTS["flux_fill_width"]
    flux_fill_height: int = PROJECT_DEFAULTS["flux_fill_height"]
    flux_fill_seed: int = PROJECT_DEFAULTS["flux_fill_seed"]
    flux_fill_remove_bg: bool = PROJECT_DEFAULTS["flux_fill_remove_bg"]
    # Z-Image generation settings
    zimage_prompt: str = PROJECT_DEFAULTS["zimage_prompt"]
    zimage_negative_prompt: str = PROJECT_DEFAULTS["zimage_negative_prompt"]
    zimage_cfg_normalization: bool = PROJECT_DEFAULTS["zimage_cfg_normalization"]
    zimage_cfg_truncation: float = PROJECT_DEFAULTS["zimage_cfg_truncation"]
    zimage_steps: int = PROJECT_DEFAULTS["zimage_steps"]
    zimage_cfg_scale: float = PROJECT_DEFAULTS["zimage_cfg_scale"]
    zimage_width: int = PROJECT_DEFAULTS["zimage_width"]
    zimage_height: int = PROJECT_DEFAULTS["zimage_height"]
    zimage_seed: int = PROJECT_DEFAULTS["zimage_seed"]
    zimage_batch_count: int = PROJECT_DEFAULTS["zimage_batch_count"]
    zimage_denoising_strength: float = PROJECT_DEFAULTS["zimage_denoising_strength"]
    zimage_remove_bg: bool = PROJECT_DEFAULTS["zimage_remove_bg"]
    zimage_activated_loras: list[str] = field(default_factory=lambda: list(PROJECT_DEFAULTS["zimage_activated_loras"]))
    zimage_lora_multipliers: str = PROJECT_DEFAULTS["zimage_lora_multipliers"]
    zimage_source_path: str = PROJECT_DEFAULTS["zimage_source_path"]
    zimage_inpaint_source_path: str = PROJECT_DEFAULTS["zimage_inpaint_source_path"]
    zimage_inpaint_prompt: str = PROJECT_DEFAULTS["zimage_inpaint_prompt"]
    zimage_inpaint_steps: int = PROJECT_DEFAULTS["zimage_inpaint_steps"]
    zimage_inpaint_cfg_scale: float = PROJECT_DEFAULTS["zimage_inpaint_cfg_scale"]
    zimage_inpaint_strength: float = PROJECT_DEFAULTS["zimage_inpaint_strength"]
    zimage_inpaint_width: int = PROJECT_DEFAULTS["zimage_inpaint_width"]
    zimage_inpaint_height: int = PROJECT_DEFAULTS["zimage_inpaint_height"]
    zimage_inpaint_seed: int = PROJECT_DEFAULTS["zimage_inpaint_seed"]
    zimage_inpaint_remove_bg: bool = PROJECT_DEFAULTS["zimage_inpaint_remove_bg"]
    # Lip Sync settings
    lipsync_source_path: str = PROJECT_DEFAULTS["lipsync_source_path"]
    lipsync_audio_path: str = PROJECT_DEFAULTS["lipsync_audio_path"]
    lipsync_result_path: str = PROJECT_DEFAULTS["lipsync_result_path"]
    lipsync_face_bbox: list[int] = field(default_factory=lambda: list(PROJECT_DEFAULTS["lipsync_face_bbox"]))
    lipsync_steps: int = PROJECT_DEFAULTS["lipsync_steps"]
    lipsync_guidance: float = PROJECT_DEFAULTS["lipsync_guidance"]
    # VACE MultiTalk settings
    vmt_source_path: str = PROJECT_DEFAULTS["vmt_source_path"]
    vmt_audio_left: str = PROJECT_DEFAULTS["vmt_audio_left"]
    vmt_audio_right: str = PROJECT_DEFAULTS["vmt_audio_right"]
    vmt_result_path: str = PROJECT_DEFAULTS["vmt_result_path"]
    vmt_resolution: str = PROJECT_DEFAULTS["vmt_resolution"]
    vmt_steps: int = PROJECT_DEFAULTS["vmt_steps"]
    vmt_shift: float = PROJECT_DEFAULTS["vmt_shift"]
    vmt_seed: int = PROJECT_DEFAULTS["vmt_seed"]
    vmt_speaker_bboxes: list = field(default_factory=lambda: list(PROJECT_DEFAULTS["vmt_speaker_bboxes"]))
    vmt_speaker_assignments: dict = field(default_factory=lambda: dict(PROJECT_DEFAULTS["vmt_speaker_assignments"]))
    # Timeline settings
    tl_library_clips: list[str] = field(default_factory=lambda: list(PROJECT_DEFAULTS["tl_library_clips"]))
    tl_timeline_clips: list[str] = field(default_factory=lambda: list(PROJECT_DEFAULTS["tl_timeline_clips"]))
    tl_tracks: list[dict] = field(default_factory=lambda: list(PROJECT_DEFAULTS["tl_tracks"]))
    tl_export_settings: dict = field(default_factory=lambda: dict(PROJECT_DEFAULTS["tl_export_settings"]))
    tl_versions: dict = field(default_factory=lambda: dict(PROJECT_DEFAULTS["tl_versions"]))
    tl_active_version: str = PROJECT_DEFAULTS["tl_active_version"]
    tl_preview_path: str = PROJECT_DEFAULTS["tl_preview_path"]
    tl_fps: float = PROJECT_DEFAULTS["tl_fps"]
    tl_zoom_start: float = PROJECT_DEFAULTS["tl_zoom_start"]
    tl_zoom_end: float = PROJECT_DEFAULTS["tl_zoom_end"]
    # Color Correct tab settings
    cc_source_path: str = PROJECT_DEFAULTS["cc_source_path"]
    cc_ref_path: str = PROJECT_DEFAULTS["cc_ref_path"]
    cc_strength: float = PROJECT_DEFAULTS["cc_strength"]
    cc_histogram_match: bool = PROJECT_DEFAULTS["cc_histogram_match"]
    cc_result_path: str = PROJECT_DEFAULTS["cc_result_path"]
    # Brightness/Contrast tab settings
    bc_source_path: str = PROJECT_DEFAULTS["bc_source_path"]
    bc_ref_path: str = PROJECT_DEFAULTS["bc_ref_path"]
    bc_brightness: float = PROJECT_DEFAULTS["bc_brightness"]
    bc_contrast: float = PROJECT_DEFAULTS["bc_contrast"]
    bc_strength: float = PROJECT_DEFAULTS["bc_strength"]
    bc_mode: int = PROJECT_DEFAULTS["bc_mode"]
    bc_result_path: str = PROJECT_DEFAULTS["bc_result_path"]
    # Hue/Saturation tab settings
    hs_source_path: str = PROJECT_DEFAULTS["hs_source_path"]
    hs_ref_path: str = PROJECT_DEFAULTS["hs_ref_path"]
    hs_hue_shift: float = PROJECT_DEFAULTS["hs_hue_shift"]
    hs_saturation: float = PROJECT_DEFAULTS["hs_saturation"]
    hs_strength: float = PROJECT_DEFAULTS["hs_strength"]
    hs_mode: int = PROJECT_DEFAULTS["hs_mode"]
    hs_result_path: str = PROJECT_DEFAULTS["hs_result_path"]
    # Speed Control tab settings
    sc_source_path: str = PROJECT_DEFAULTS["sc_source_path"]
    sc_preset_idx: int = PROJECT_DEFAULTS["sc_preset_idx"]
    sc_result_path: str = PROJECT_DEFAULTS["sc_result_path"]

    # Project defaults (set in Project Settings dialog)
    image_resolution: str = ""
    default_video_model: str = ""
    default_image_model: str = ""
    default_image_steps: int = 0
    default_image_cfg: float = 0.0
    default_image_sampler: str = ""

    # QwenAlyzer
    qwenalyzer_image_path: str = ""
    qwenalyzer_style: str = "sdxl"
    qwenalyzer_prompt: str = ""
    qwenalyzer_negative: str = ""

    # LoRA Training
    lora_dataset_path: str = ""
    lora_output_name: str = "my_lora"
    lora_model_type: str = ""
    lora_rank: str = "16"
    lora_alpha: int = 16
    lora_steps: int = 2000
    lora_lr: str = "5e-5"
    lora_scheduler: str = "cosine"
    lora_optimizer: str = "Prodigy"
    lora_precision: str = "bf16"
    lora_grad_ckpt: bool = True
    lora_cache_latents: bool = True
    lora_cache_te: bool = True
    lora_noise_offset: float = 0.0
    lora_min_snr: float = 5.0
    lora_save_every: int = 500
    lora_sample_every: int = 200
    lora_sample_prompt: str = ""
    lora_checkpoint_idx: int = 0

    # Chatterbox TTS
    chatterbox_text: str = PROJECT_DEFAULTS["chatterbox_text"]
    chatterbox_ref_audio: str = PROJECT_DEFAULTS["chatterbox_ref_audio"]
    chatterbox_exaggeration: float = PROJECT_DEFAULTS["chatterbox_exaggeration"]
    chatterbox_cfg_weight: float = PROJECT_DEFAULTS["chatterbox_cfg_weight"]

    # SadTalker (audio-driven talking head)
    sadtalker_source_path: str = PROJECT_DEFAULTS["sadtalker_source_path"]
    sadtalker_audio_path: str = PROJECT_DEFAULTS["sadtalker_audio_path"]
    sadtalker_preprocess: str = PROJECT_DEFAULTS["sadtalker_preprocess"]
    sadtalker_size: int = PROJECT_DEFAULTS["sadtalker_size"]
    sadtalker_pose_style: int = PROJECT_DEFAULTS["sadtalker_pose_style"]
    sadtalker_exp_scale: float = PROJECT_DEFAULTS["sadtalker_exp_scale"]
    sadtalker_still_mode: bool = PROJECT_DEFAULTS["sadtalker_still_mode"]
    sadtalker_use_blink: bool = PROJECT_DEFAULTS["sadtalker_use_blink"]
    sadtalker_batch_size: int = PROJECT_DEFAULTS["sadtalker_batch_size"]
    sadtalker_enhancer_method: str = PROJECT_DEFAULTS["sadtalker_enhancer_method"]
    sadtalker_background_enhancer: bool = PROJECT_DEFAULTS["sadtalker_background_enhancer"]
    sadtalker_ref_video: str = PROJECT_DEFAULTS["sadtalker_ref_video"]
    sadtalker_ref_info: str = PROJECT_DEFAULTS["sadtalker_ref_info"]
    sadtalker_use_ref_video: bool = PROJECT_DEFAULTS["sadtalker_use_ref_video"]
    sadtalker_use_idle_mode: bool = PROJECT_DEFAULTS["sadtalker_use_idle_mode"]
    sadtalker_length_of_audio: int = PROJECT_DEFAULTS["sadtalker_length_of_audio"]
    sadtalker_input_yaw_str: str = PROJECT_DEFAULTS["sadtalker_input_yaw_str"]
    sadtalker_input_pitch_str: str = PROJECT_DEFAULTS["sadtalker_input_pitch_str"]
    sadtalker_input_roll_str: str = PROJECT_DEFAULTS["sadtalker_input_roll_str"]
    sadtalker_face3dvis: bool = PROJECT_DEFAULTS["sadtalker_face3dvis"]
    sadtalker_result_path: str = PROJECT_DEFAULTS["sadtalker_result_path"]

    # MimicMotion (pose-driven body animation)
    mimicmotion_reference_path: str = PROJECT_DEFAULTS["mimicmotion_reference_path"]
    mimicmotion_driving_path: str = PROJECT_DEFAULTS["mimicmotion_driving_path"]
    mimicmotion_audio_path: str = PROJECT_DEFAULTS["mimicmotion_audio_path"]
    mimicmotion_result_path: str = PROJECT_DEFAULTS["mimicmotion_result_path"]
    mimicmotion_resolution: str = PROJECT_DEFAULTS["mimicmotion_resolution"]
    mimicmotion_num_frames: int = PROJECT_DEFAULTS["mimicmotion_num_frames"]
    mimicmotion_steps: int = PROJECT_DEFAULTS["mimicmotion_steps"]
    mimicmotion_guidance: float = PROJECT_DEFAULTS["mimicmotion_guidance"]
    mimicmotion_seed: int = PROJECT_DEFAULTS["mimicmotion_seed"]
    mimicmotion_fps: int = PROJECT_DEFAULTS["mimicmotion_fps"]
    mimicmotion_min_guidance: float = PROJECT_DEFAULTS["mimicmotion_min_guidance"]
    mimicmotion_max_guidance: float = PROJECT_DEFAULTS["mimicmotion_max_guidance"]
    mimicmotion_noise_aug: float = PROJECT_DEFAULTS["mimicmotion_noise_aug"]
    mimicmotion_pose_align: bool = PROJECT_DEFAULTS["mimicmotion_pose_align"]

    # Dia TTS
    dia_text: str = ""
    dia_ref_audio: str = ""
    dia_cfg_scale: float = 3.0
    dia_temperature: float = 1.2
    dia_top_p: float = 0.95
    dia_speed: float = 0.94
    dia_max_tokens: int = 3072

    CONFIG_FILENAME: ClassVar[str] = "project.json"

    # -- color profile resolution --------------------------------------------

    def resolved_color_profile(self, global_cfg=None):
        """Return the ColorProfile that should drive every encode in this project.

        Resolution order:
          1. ``self.color_profile`` if set to a known slug.
          2. ``global_cfg.default_color_profile`` if a GlobalConfig is supplied.
          3. The hardcoded default (``bt709_full``).
        """
        from .color_profile import get_profile
        key = self.color_profile or ""
        if not key and global_cfg is not None:
            key = getattr(global_cfg, "default_color_profile", "") or ""
        return get_profile(key)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise to JSON inside the given directory (or exact file path).

        Uses atomic write (write-to-temp-then-rename) to avoid race conditions
        where concurrent readers see a truncated file.
        """
        import tempfile
        target = Path(path)
        if target.is_dir():
            target = target / self.CONFIG_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(asdict(self), indent=2) + "\n"
        # Atomic write: write to temp file in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(target)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        """Load from a project directory or exact JSON file. Returns defaults on missing/corrupt file."""
        target = Path(path)
        if target.is_dir():
            target = target / cls.CONFIG_FILENAME
        if not target.exists():
            return cls()
        try:
            text = target.read_text(encoding="utf-8")
            if not text.strip():
                return cls()
            data = json.loads(text)
        except (json.JSONDecodeError, OSError):
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Keys that intentionally live in PROJECT_DEFAULTS but are NOT persisted
# ProjectConfig fields (transient / dialog-only state). Empty today; listing a
# key here is the deliberate opt-out from the parity guard below.
DIALOG_ONLY_EXTRAS: frozenset[str] = frozenset()

# Import-time parity guard. save() serialises declared dataclass fields and
# load() filters on __dataclass_fields__, so any PROJECT_DEFAULTS key that is
# not a declared field silently evaporates on save/reload. Fail loudly at import
# if a UI-written key was added to PROJECT_DEFAULTS without a matching field.
_missing_fields = (
    set(PROJECT_DEFAULTS) - DIALOG_ONLY_EXTRAS - set(ProjectConfig.__dataclass_fields__)
)
assert not _missing_fields, (
    "PROJECT_DEFAULTS keys missing from ProjectConfig (won't persist): "
    f"{sorted(_missing_fields)}"
)
del _missing_fields
