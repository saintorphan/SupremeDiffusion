# Supreme Diffusion — 3.1.0

*A thousand tons of AI in a five-pound bucket.*

![Version](https://img.shields.io/badge/version-3.1.1-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PySide6](https://img.shields.io/badge/frontend-PySide6-green)
![PyTorch](https://img.shields.io/badge/backend-PyTorch%202.1%2B-red)
![License](https://img.shields.io/badge/license-see%20below-lightgrey)

> **v3.1.0** — GL-rendered Quick Export, multi-clip batch export, combine
> preserves clip resolution + effects, Post-Process prompt on export.
> Bug reports welcome via
> [GitHub Issues](https://github.com/saintorphan/SupremeDiffusion/issues).

---

## What's New in 3.0

- **Batch Generate (Mode 1, img2vid)** — One-click batch over a folder or multi-file
  selection of source images. Optional per-item Color Correct (with reference image/video),
  Create Loops, and full Post-Process preset. Clip numbering continues from the highest
  existing index in the output folder when enabled. All dialog fields persist across
  launches.
- **Sequential Processing (Mode 2, first/last frame)** — Sliding-window pair iteration
  over a numerically-named image sequence. Validates contiguity before starting. Same
  Color Correct + Post-Process pipeline as Batch. "Guidance Frame" dropdown picks First
  or Last as the static guidance source per pair.
- **Quick Export — timeline WYSIWYG** — Right-click any clip on the timeline, Multitrack
  canvas, browser, or clip library → Quick Export bakes the clip's effect stack (LUTs,
  colour correction, rotation, crop, etc.) exactly as shown in the timeline preview, then
  optionally runs denoise + sharpen + face restore + lanczos 1.5× + RIFE 2×. A Yes/No
  prompt asks whether to run post-processing so already-processed clips aren't
  double-processed.
- **Multi-clip Quick Export** — Highlight multiple clips on the timeline, right-click →
  "Quick Export N clips…" runs the same pipeline sequentially over every clip, with a
  unified progress dialog and "Open Folder" result summary.
- **Create Loop** — Right-click a clip → Create Loop opens a dedicated dialog that
  builds a seamless forward + reverse loop via a single-pass ffmpeg filter_complex
  (no codec drift between halves), trims seam-black frames automatically, and previews
  the result inline. Save to the project clips folder, or send the loop straight through
  Quick Export / Export.
- **Match Grade → / ←** — Timeline clip context menu now supports single-direction and
  chain-mode match grading against any reference clip, saving generated LUTs per-clip
  for repeatable reruns.
- **Colour pipeline rewrite (the "wash-out" fix)** — Every ffmpeg encode in the app now
  outputs via `yuvj420p` instead of the broken `yuv420p + color_range pc` combo. libx264
  silently squeezes full-range RGB into TV-range [16,235] unless the output pix_fmt is
  explicitly `yuvj420p`; the old convention meant every exported file had blacks at
  RGB 16 (grey) instead of 0. 17 call sites were swept — Quick Export, Batch, Sequential,
  Timeline Export, Bake Effects, Color Correct, Loop Build, Trim, Crop, Speed, MuseTalk,
  LatentSync, Video Retalking, render3d, and more. If you've ever wondered why your
  exports looked washed out compared to the preview, this is why.
- **Multi-drop = sequential placement** — Dragging multiple clips from a file manager
  onto the Multitrack canvas now lays them down end-to-end in drop order instead of
  stacking them on top of each other at the drop point.
- **Drag-and-drop reordering + numeric-stem validation** in Sequential Processing's
  source picker.
- **Batch Generate/Sequential field persistence** via `GlobalConfig.batch_generate_settings`
  and `sequential_process_settings`.
- **Image Suite auto-save** — Model, sampler, scheduler, steps, CFG, LoRA selections,
  and all other img2img / inpaint params now persist on every change (not just at
  generate time) so your SDXL + DPM++ Karras choice sticks even if you walk away without
  hitting Generate.
- **GPU stability** — Systemd-managed `nvidia-smi -pl` power cap helper to mitigate
  RTX 3080 Ti / 3090 transient-spike Xid 79 "GPU fell off the bus" crashes on heavy
  inference loops.

## What's New in 3.1

- **GL-Rendered Quick Export** — When Quick Export is invoked from a timeline clip
  that has a GL preview available, frames are now captured directly from the OpenGL
  shader pipeline (the same path the timeline preview uses) instead of going through
  ffmpeg's `lut3d` filter. This guarantees the exported file is pixel-identical to
  what's shown in the preview — LUTs, colour correction, effects, and all. The AI
  post-process preset (denoise/sharpen/face/lanczos/RIFE) is then optionally chained
  on top of the GL-rendered result.
- **"Use Post Processing?" prompt** — Quick Export (single and batch) now asks
  Yes/No before running the AI preset. Selecting **No** gives a pure WYSIWYG bake
  so already-processed clips don't get double-processed (which compounds upscaling,
  frame-rate doubling, and sharpening).
- **Multi-Clip Batch Quick Export** — Highlight multiple clips on the timeline canvas,
  right-click → "Quick Export N clips…". Picks an output folder, asks the Post-Process
  prompt once, then exports every selected clip sequentially with a unified progress
  dialog. Cancel stops after the current clip. Each output is named after the clip and
  auto-suffixed on collision.
- **Combine preserves source resolution** — The "Combine N clips" action previously
  hardcoded all output to 1280×720. It now probes the first clip's native dimensions
  and uses those as the target for all scale + pad operations so your 720×1248
  portrait clips don't get letter-boxed into a landscape frame.
- **Combine preserves per-clip effects** — Each clip's effect stack (LUTs from Match
  Grade, colour correction, hue/sat, etc.) is now baked into its segment of the
  combined output via `_build_video_filters()`, matching the timeline preview.
  Previously all effects were silently dropped during combine.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Tab Reference](#tab-reference)
- [Architecture](#architecture)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Configuration](#configuration)
- [Acknowledgments & Credits](#acknowledgments--credits)

---

## Features

- **Video Generation** — Wan 2.1/2.2 Image-to-Video with three modes: single image, first/last frame interpolation, and video continuation
- **Video Extender** — Extend existing clips with AI-generated continuations, optional guidance video and final frame control
- **Longshot** — Multi-chunk long-form video generation with automatic gap-filling and assembly
- **StillGrabber** — Extract evenly-spaced still frames from video clips with configurable output resolution and right-click send-to for all image tabs
- **Video LoRA** — Train custom video LoRAs using ai-toolkit backend with batch generation and full parameter control
- **Image Suite** — Unified Generate tab (All2Img, Inpaint, Draw, Crop/Zoom), Character Tools (Face Swap, Body Double, RePose, ControlNet), 3D Modeling, QwenAlyzer (LLM image analysis + PNG metadata). Supports SD 1.5/SDXL, FLUX, Z-Image Turbo, Magic Mask (AI auto-segmentation)
- **Audio Suite** — Chatterbox TTS (voice cloning with emotion control), Dia TTS (multi-speaker dialogue synthesis), MusicGen (text-to-music via Meta AudioCraft), AudioGen (text-to-sound-effects via Meta AudioCraft)
- **Lip Sync** — MuseTalk (fast/real-time) accessible via timeline clip right-click context menu. VACE MultiTalk, LatentSync, and Video Retalking integrations available
- **Timeline NLE** — Multi-track non-linear editor with zoom (DAW-style scrollbar), razor/slip/slide tools, markers, ripple editing, fade handles, crossfade transitions, clip thumbnails, GL Render export (pixel-identical to preview) with RIFE temporal upsampling, and a comprehensive effects system (15 effects across Color, Filter, Transform, Enhance, and Encode categories including AI Enhance, Crop/Zoom/Fit, and speed)
- **Timeline Effects** — Hue/Saturation, Brightness/Contrast, Levels, Shadows/Highlights, Color Balance, Color Temperature, Vibrance, Sharpen, Blur, Denoise, Vignette, Film Grain, Crop/Zoom/Fit, Speed, AI Enhance, Compress — all applied per-clip via OpenGL preview or rendered export
- **Color Correct** — Dedicated tab for reference-based color matching with per-category controls (brightness, contrast, color balance, temperature, saturation, levels, shadows/highlights), individual strength sliders, manual parameter editing, batch file processing, and side-by-side source/result video players. Right-click any image or video anywhere in the app to set it as the color reference.
- **GL Render Export** — Renders each frame through the OpenGL shader pipeline (pixel-identical to what you see in the preview) and pipes raw RGB to ffmpeg for encoding only. Eliminates color mismatch between preview and export. Enabled by default in the export dialog.
- **Full-Range Color Preservation** — All ffmpeg encode paths use `scale=in_range=full:out_range=full` and `-color_range pc` to prevent the RGB-to-YUV limited-range clamp that causes darkening and yellowing on every encode
- **Processing Tools** — Trim and crop (standalone tabs); color correction and speed control are now timeline effects
- **3D Modeling** — TripoSR single-image mesh generation, OpenGL viewport with gizmo-based translate/rotate/scale, background compositing, configurable lighting, camera presets, keyframe animation timeline with 5 presets (turntable, dolly, spin, reveal, orbit+rise), frame sequence rendering to video, send to Img2Vid for AI refinement. Import meshes from Daz3D/Blender (OBJ, GLB, FBX via pyassimp). UV unwrapping (xatlas) + texture baking. Scene auto-save/restore.
- **Daz2Supreme** — Full Daz3D scene import with PBR textures, shadows, FABRIK IK solver with viewport drag posing, pose library, clothing conform, batch render, normal maps, HDRI skybox, object list, bone picking, and light editor
- **Sequence Wizards** — 7 guided multi-step workflows for common video production tasks
- **LoRA Training** — Native LoRA training for SD 1.5, SDXL, Pony, and Illustrious via kohya sd-scripts. Dataset browser with image thumbnails and caption editor. Batch auto-captioning (BLIP/WD14). Caption tools (prepend trigger word, append, find/replace). 6 presets optimized for 12 GB VRAM. Real-time loss curve graph and sample image previews during training. Auto-copies trained LoRA to scan directory.
- **Library** — Top-level tab with PNG Library, Voice Library, Sound Library, Character Library, and Face Library for cross-project asset reuse
- **Project System** — Named projects with automatic state persistence, file organization, and cross-tab send-to routing
- **AI Assistant** — Built-in Qwen 3.5 4B chat assistant with per-project history, context-aware upstream documentation lookup, app knowledge base, and VRAM-aware lifecycle
- **Model Manager** — Automatic VRAM tier detection, one-click downloads from Hugging Face, feature-based model grouping
- **Draw/Paint** — Layer-based canvas with brush, eraser, shapes, lasso/magic wand selection, flood fill, eyedropper, transform handles, and PNG library
- **Configurable Video Codec** — GPU-first encoding (nvenc → CPU fallback), selectable from Settings

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA with 8 GB VRAM (CUDA) | NVIDIA with 12 GB+ VRAM |
| RAM | 16 GB | 32 GB+ |
| Storage | 30 GB (models) | 100 GB+ (multiple model variants) |

> An NVIDIA GPU with CUDA support is **required** for inference. The app auto-detects your VRAM tier (0–4) and selects quantized model variants accordingly.

**This application's inference orchestrator was developed, tested, and optimized on a Linux PC with an NVIDIA RTX 3080 Ti (12 GB VRAM) and 32 GB DDR4 RAM.** With the following settings, it successfully runs both Wan 2.2 I2V 14B and Wan 2.2 I2V 14B Lightning v2 at 480P (512x512, 832x480, 480x832, and equivalent aspect ratios):

| Setting | Value | Notes |
|---------|-------|-------|
| Transformer quantization | **int8** | ~14.5 GB model compressed to fit in 12 GB VRAM |
| Text encoder quantization | **int8** | Reduces text encoder memory footprint |
| Attention mode | **xformers** | Memory-efficient attention — critical for 12 GB cards |
| Memory profile | **4** (aggressive) | Maximum offloading between pipeline stages |
| Boost | **2** | Enables MMGP memory management optimizations |
| VAE tiling | **Auto** | Tiles VAE decode to avoid VRAM spikes on larger resolutions |
| Mixed precision | **Enabled** | fp16 compute where possible |

With these settings, Wan 2.2 I2V 14B Lightning v2 generates a 5-second 480P video in ~4 steps. The standard Wan 2.2 I2V 14B model uses ~20–30 steps for higher quality output. Both use the same ~14.5 GB int8-quantized transformer weights.

Generated 480P videos can then be post-processed in a single pass: **RIFE x2** doubles the frame rate (16 fps → 32 fps) for smoother motion, and **Lanczos spatial upscale** doubles the resolution to 1024x1024 (or 1664x960 / 960x1664 for landscape/portrait equivalents). Both options are available as checkboxes in the generation parameters — no external tools needed.

### Software

- Python 3.10 – 3.12
- CUDA toolkit (matching your PyTorch build)
- FFmpeg (on `PATH` — used for video encode/decode, trimming, compositing)
- Git (for cloning)

---

## Installation & Launch

### Prerequisites

- **Python 3.10–3.12** installed and on your PATH
- **NVIDIA GPU** with CUDA drivers installed
- **Git** for cloning

> FFmpeg is also required but the start script will detect if it's missing and offer to install it for you.

### Quick Start (recommended)

The start scripts handle everything automatically — venv creation, PyTorch, dependencies, factory directories, update checks, and launch:

```bash
git clone https://github.com/saintorphan/SupremeDiffusion.git
cd SupremeDiffusion
./start.sh          # Linux / macOS
```

```bat
start.bat            # Windows
```

**On first run, the start script will:**

1. Check for **FFmpeg** and offer to install it via your package manager if missing
2. Create a `.venv/` Python virtual environment if one doesn't exist
3. Install **PyTorch** with the CUDA build matching your NVIDIA driver (cu118/cu124/cu126/cu128, auto-detected — if PyTorch isn't already installed)
4. Install all Python dependencies via `pip install -e .` (PySide6, diffusers, transformers, accelerate, peft, safetensors, optimum-quanto, gguf, mmgp, xformers, Pillow, numpy, opencv-python, sentencepiece, protobuf, ftfy)
5. Ask **"Install optional components?"** — if Y, prompt per-feature: face swap, 3D modeling (TripoSR), LoRA training (kohya sd-scripts), Chatterbox TTS, Dia TTS, AudioCraft (MusicGen/AudioGen) — answer Y or N for each. If N, skips all optional installs.
6. Create the factory model and project directories
7. Launch the application

**On subsequent runs**, it skips steps 1–5, checks GitHub for updates and pulls them automatically, then launches.

> If you need a different CUDA version (e.g. 11.8 or 12.1), install PyTorch manually before running the start script — it will detect the existing installation and skip the PyTorch step.

### Manual Installation

If you prefer to set things up yourself:

```bash
git clone https://github.com/saintorphan/SupremeDiffusion.git
cd SupremeDiffusion
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .
python run.py
```

---

## Getting Started

### First-time setup

1. **Settings tab** — Model paths default to factory directories inside the app folder (`models/checkpoints/`, `models/vae/`, etc.). Drop your model files into these folders, or point the paths to existing directories on your system.

2. **Models download automatically** — When you click Generate for the first time, the app checks if the required models are present. If they're missing, it prompts you to download them from Hugging Face. The correct quantization variant (int8, Q4, Q8, F16, bf16) is selected automatically based on your detected VRAM tier. You don't need to manually find or download models.

3. **Or download in advance** — Open **Settings → Installed Models**, click **Refresh** to see what's installed, then click **Download Missing** to grab everything at once.

4. **Create a project** — Use the project bar at the top to create a named project. All outputs, clips, frames, and state are saved per-project.

5. **Generate** — Go to the **Video → Img2Vid** tab, drop in a source image, write a prompt, and click **Generate** (or press `Ctrl+G`).

### Tips

- **Don't upscale until you're done editing.** RIFE frame interpolation and Lanczos spatial upscale should only be applied to your final, finished clip — not to intermediate clips you plan to extend, trim, or color correct further. Every re-encode introduces quality loss, and stacking upscale → re-encode → upscale compounds into visible degradation. Keep your working clips at native 480P resolution and only enable RIFE/upscale on your last export.

- **Color correct every continuation clip immediately.** Wan's VAE has a baked-in warm color shift that accumulates across generations. When extending a video (Video Extender or Longshot), each new clip will be slightly warmer than the last. To combat this: after generating each continuation clip, send it to **Color Correct** and use the **first frame of your very first clip** as the reference image. Apply color correction before stitching, extending further, or doing any other processing. This keeps the entire sequence color-consistent.

- **Right-click everything.** Video players and image galleries have right-click context menus with send-to options — send clips to other video tabs, send frames to image tabs, extract color correction references, and more. This is the fastest way to move content between modules.

- **Use projects to organize.** Each project keeps its own generated clips, frames, settings, and timeline state. Create separate projects for separate scenes or experiments — switching is instant and preserves your full workspace.

### Model downloads

The app auto-detects your VRAM and downloads the best model variant for your GPU. You can also pre-download from **Settings → Installed Models → Download Missing**. Models are grouped by feature:

| Feature | Models | Approx. Size |
|---------|--------|-------------|
| Video Generation | Wan 2.1/2.2 Transformer + Text Encoder + VAE | 15–30 GB (int8–bf16) |
| Lip Sync (MuseTalk) | UNet + VAE + Whisper | ~2 GB |
| Lip Sync (LatentSync) | UNet + Whisper Tiny + SD VAE | ~3 GB |
| VACE MultiTalk | Processor + Transformer + Audio Encoder + Wav2Vec2 | ~30 GB |
| Face Swap | InsightFace buffalo_l + inswapper_128 | ~350 MB |
| Body Double | ControlNet (OpenPose/Canny/Depth/Lineart/Normal) + IP-Adapter + CLIP ViT-H | ~8 GB |
| Magic Mask | BiRefNet foreground segmentation | ~350 MB |
| FLUX | FLUX GGUF + VAE + T5-XXL fp8 + CLIP-L | ~10–30 GB (Q4→F16 by VRAM) |
| 3D Modeling | BiRefNet + TripoSR | ~3.5 GB |
| AI Assistant / Prompt Enhance | Qwen 3.5 4B (BF16) | ~5 GB |
| Image Captioning (QwenAlyzer / LoRA) | Qwen 2.5 VL 3B Instruct (bf16) | ~6 GB |
| AI Enhance | Real-ESRGAN 4x | ~65 MB |
| LoRA Training | kohya sd-scripts (cloned, not downloaded) | ~50 MB |
| Chatterbox TTS | Chatterbox (voice cloning + emotion) | ~1 GB |
| Dia TTS | Dia (multi-speaker dialogue) | ~2 GB |
| MusicGen | Meta AudioCraft MusicGen | ~3.3 GB |
| AudioGen | Meta AudioCraft AudioGen | ~3.3 GB |

---

## Tab Reference

### Timeline

Multi-track non-linear video editor with:

- Drag-and-drop clip library
- Multiple video and audio tracks with volume, mute, solo, lock
- DAW-style zoom scrollbar, Ctrl+Scroll zoom at cursor, zoom-to-fit
- Tools: Select (ghost drag preview), Scrub, Razor (click-to-split), Slip, Slide
- Ripple edit mode (delete/trim shifts subsequent clips)
- Roll edit (Ctrl+trim adjusts adjacent clip)
- Edge snapping with priority-aware thresholds
- Timeline markers (M key or double-click ruler)
- Clip thumbnails and waveform visualization
- Fade in/out handles with visual triangles
- Crossfade transition indicators
- Clip grouping, rotate 90 CW/CCW
- Undo/redo with command-based stack
- **15 clip effects** via right-click: Hue/Saturation, Brightness/Contrast, Levels, Shadows/Highlights, Color Balance, Color Temperature, Vibrance, Sharpen, Blur, Denoise, Vignette, Film Grain, Crop/Zoom/Fit, Speed, AI Enhance, Compress
- OpenGL real-time preview with effects applied
- **GL Render export** — renders each frame through the GL shader pipeline (pixel-identical to preview), pipes raw RGB to ffmpeg for encode-only. Eliminates color mismatch between preview and rendered output. Enabled by default.
- FFmpeg filter_complex export (alternative path) with parameterized resolution/fps
- Full-range color preservation on all encodes (`in_range=full:out_range=full` + `-color_range pc`)
- **Export post-processing** — Denoise (SCUNet), Sharpen (RealESRGAN 2x), Upscale (RealESRGAN 4x), Face Restore (GFPGAN), Lanczos spatial upscale (1.5x/2.0x), Film Grain — all configurable in the export dialog
- RIFE temporal upsampling on export (x2 or x4)
- Configurable video codec (GPU-first: nvenc → CPU fallback)
- Preview caching (skips rebuild when timeline unchanged)
- Playhead auto-scroll during playback
- **Lip Sync** — Right-click any clip → "Lip Sync (MuseTalk)..." to open the MuseTalk dialog with audio browser, face detection, and blend controls

### Video

- **Img2Vid** — Wan 2.1/2.2 Image-to-Video. Mode 1: single image → video. Mode 2: first + last frame interpolation. Mode 3: video continuation. Supports guidance videos for motion/composition control. Full parameter control: steps, guidance scales, LoRA, post-processing (RIFE temporal upscale, spatial upscale, film grain), prompt profiles.
- **Video Extender** — Load a video and extend it with AI-generated continuation. Supports guidance video overlay, final frame control, and all generation parameters.
- **Longshot** — Multi-chunk generation for long-form video. Three-stage pipeline: chunk generation → gap-fill rendering → final assembly.
- **StillGrabber** — Extract evenly-spaced still frames from video clips. Configurable output resolution, frame count, and right-click send-to for all image tabs.
- **Video LoRA** — Train custom video LoRAs using ai-toolkit backend. Batch video generation with trained LoRAs. Full parameter control for training configuration.

### Image

- **Generate** — Unified image generation tab with sub-tabs:
  - **All2Img** — Combined txt2img and img2img with SD 1.5/SDXL checkpoints, VAE, sampler/scheduler selection, LoRA support, batch generation, optional SDXL refiner, and configurable denoising strength.
  - **Inpaint** — Full layer-based inpainting editor with dual-toolbar (Mask + Paint), selection tools (lasso, magic wand, magnetic lasso), Sel→Mask workflow, Magic Mask AI auto-segmentation, and layer panel.
  - **Draw** — Layer-based painting with brushes, shapes, lasso tools, magic wand, flood fill, transform handles. Includes a **PNG Library** for overlaid compositing. Send flattened output to any tab.
  - **Crop/Zoom** — Crop and zoom into image regions with aspect presets.
- **Character Tools** — Sub-tabs for person/face work:
  - **Face Swap** — InsightFace detection + inswapper with face enhancement (GFPGAN, CodeFormer).
  - **Body Double** — Configurable ControlNet (OpenPose, Canny, Depth, Lineart, Normal) + IP-Adapter (Base, Plus, Plus Face, Full Face) for identity-preserving person replacement. Dual ControlNet, Magic Mask.
  - **RePose** — Same technology as Body Double with separate source and pose reference inputs.
  - **ControlNet** — General-purpose single or dual ControlNet-conditioned txt2img and img2img with live preprocessing preview.
- **3D Modeling** — TripoSR single-image mesh generation, OpenGL viewport with translate/rotate/scale gizmos, background plate compositing, adjustable lighting, camera presets, keyframe animation timeline (turntable, dolly, spin, reveal, orbit+rise), frame sequence rendering to MP4, send composite to any image/video tab. Import from Daz3D/Blender (OBJ, GLB, FBX). UV unwrapping + texture baking. Scene persistence.
- **QwenAlyzer** — Vision-powered image analysis: Qwen 2.5 VL 3B sees the image, then Qwen 3.5 4B formats the description into style-specific prompts for any model family (SD 1.5/SDXL/SD3/FLUX/Pony/Illustrious/NoobAI). Send image + prompts to generation tabs. Also reads/edits PNG metadata.

### Audio

- **Chatterbox TTS** — Voice cloning with emotion control. Clone any voice from a short reference recording and generate expressive speech.
- **Dia TTS** — Multi-speaker dialogue synthesis with voice cloning. Generate natural conversations between multiple speakers.
- **MusicGen** — Text-to-music generation using Meta AudioCraft. Describe the music you want and generate instrumental tracks.
- **AudioGen** — Text-to-sound-effect generation using Meta AudioCraft. Generate ambient sounds, foley, and effects from text descriptions.

### Color Correct

Dedicated color correction tab with:

- **Reference-based matching** — Right-click any image or video anywhere in the app → "Set as Color Reference" to define the target look
- **Per-category controls** — Brightness/Contrast, Color Balance (RGB gains), Color Temperature/Tint, Saturation, Levels (black point, white point, gamma), Shadows/Highlights — each with enable checkbox and strength slider
- **Manual parameter editing** — All values exposed as editable spinboxes; use Analyze to auto-fill from reference, or dial in values manually
- **Batch processing** — Add multiple source files via drag-drop or file picker, correct all against the same reference
- **Side-by-side preview** — Source and Result video players for immediate comparison after Apply
- **Send-to integration** — "Send to Color Correct (source)" and "Send to Color Correct (reference)" in all send-to menus throughout the app

### Sequences

7 guided multi-step Sequence Wizards for common video production workflows. Each wizard walks you through a structured pipeline with send-to routing between steps.

### Daz2Supreme

Full Daz3D scene import and rendering:

- Import Daz3D scenes with PBR textures, normal maps, and shadow maps
- FABRIK IK solver with viewport drag posing
- Pose library with save/load
- Clothing conform and object hierarchy
- HDRI skybox backgrounds
- Bone picking and light editor
- Batch render to image sequence or video
- Send rendered output to any image/video tab

### Outputs

Project file explorer with thumbnail previews, send-to routing, and context menus for all media types.

### Library

Top-level asset library with five sub-tabs for cross-project reuse:

- **PNG Library** — Store and organize reference images, overlays, and compositing assets.
- **Voice Library** — Save favourite generated voices for reuse across projects.
- **Sound Library** — Store music and sound effects for easy access.
- **Character Library** — Save character references (face + body) for consistent identity across projects.
- **Face Library** — Manage face images for face swap operations.

### LoRA Training

- **Dataset Prep** — Image folder browser with thumbnails, inline caption editor (auto-save), batch auto-captioning (BLIP natural language, WD14 danbooru tags), caption tools (prepend trigger word, append text, find/replace across all captions).
- **Training Config** — 6 presets (SD 1.5 Quick/Quality, SDXL Efficient/Quality, Pony Character, Illustrious Style). Configurable: base checkpoint (auto-detect SD 1.5 vs SDXL), network type (LoRA/LoCon/LoHa), rank, alpha, learning rate, scheduler, optimizer (Prodigy, AdamW, AdamW8bit, DAdaptAdam, Lion), gradient checkpointing, latent/TE caching, noise offset, min-SNR gamma.
- **Training Execution** — Runs kohya sd-scripts as managed subprocess. Real-time loss curve graph, progress bar, sample image gallery (auto-refreshes). Supports abort. Auto-copies trained .safetensors to LoRA scan directory on completion.
- **12 GB VRAM Optimized** — Presets and auto-detection ensure SDXL training fits in 12 GB: gradient checkpointing ON, rank capped at 32, all caching enabled, Prodigy optimizer.

### Other

- **AI Assistant** — Click the **AI** button in the header to open a floating chat window powered by Qwen 3.5 4B (BF16). Ask about features, workflows, settings, or troubleshooting. Automatically injects relevant upstream documentation (ControlNet, LoRA, FLUX, etc.) into the context based on your question. Per-project chat history, multi-turn conversation with ~12K token context, VRAM-aware model lifecycle (auto-unloads when generation starts, auto-loads when chat opens).
- **Settings** — Model paths, VRAM profile, attention mode, quantization, video codec (GPU-first), pipeline load/unload, model downloads.
- **Console** — Real-time debug log output (toggle with `` Ctrl+` ``).

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+G` | Start generation (Video tab — Img2Vid / Video Extender) |
| `Escape` | Abort current generation |
| `Ctrl+Z` | Undo (Timeline) |
| `Ctrl+Y` | Redo (Timeline) |
| `Ctrl+=` / `Ctrl+-` | Zoom in / out (Timeline) |
| `Ctrl+0` | Zoom to fit (Timeline) |
| `Ctrl+Scroll` | Zoom at cursor (Timeline) |
| `M` | Add marker at playhead (Timeline) |
| `Y` | Slip tool (Timeline) |
| `U` | Slide tool (Timeline) |
| `Space` | Play / Pause (Timeline) |
| `Delete` | Delete selected clip (Timeline) |
| `Left` / `Right` | Nudge playhead (Timeline) |
| `` Ctrl+` `` | Toggle console log |

---

## Configuration

### Global config

Stored at `~/.supremediffusion/config.json`. Key settings:

| Setting | Values | Description |
|---------|--------|-------------|
| `memory_profile` | 1–5 | VRAM management aggressiveness |
| `attention_mode` | `sdpa`, `flash_attn`, `sage_attn`, `xformers` | Attention implementation |
| `transformer_quantization` | `int8`, `bf16` | Transformer weight precision |
| `vae_precision` | `16`, `32` | VAE float precision |
| `mixed_precision` | `0`, `1` | Enable/disable mixed precision |
| `vae_tiling` | `Auto`, `Disabled`, `256`, `128` | VAE tiling for large images |
| `preload_in_vram` | 0–40000 (MB) | Pre-load VRAM budget |

### Project config

Each project stores its state in `<projects_root>/<project_name>/config.json`, including all UI settings, file paths, timeline tracks, markers, and library contents. Switching projects fully restores the previous session.

---

## Acknowledgments & Credits

Portions of the following inference pipelines and libraries were referenced when building Supreme Diffusion. I gratefully acknowledge these projects and their authors:

### Video Generation

- [**Wan 2.1 / 2.2**](https://github.com/Wan-Video/Wan2.1) — Alibaba / Wan-Video — Open large-scale video generative models, core I2V backbone
- [**Wan2GP**](https://github.com/deepbeepmeep/Wan2GP) — deepbeepmeep — GPU-poor optimizations, VRAM reduction, RIFLEx, quantization
- [**VACE**](https://github.com/ali-vilab/VACE) — Alibaba / ali-vilab — All-in-one video creation and editing framework
- [**MultiTalk**](https://github.com/MeiGen-AI/MultiTalk) — MeiGen-AI — Audio-driven multi-person conversational video generation

### Image Generation

- [**Stable Diffusion / SDXL**](https://github.com/Stability-AI/stablediffusion) — Stability AI / CompVis — Foundation text-to-image diffusion models
- [**Hugging Face Diffusers**](https://github.com/huggingface/diffusers) — Hugging Face — Diffusion model inference library
- [**FLUX**](https://huggingface.co/black-forest-labs) — Black Forest Labs — Advanced image generation models
- [**reForge**](https://github.com/Panchovix/stable-diffusion-webui-reForge) — Panchovix (fork of lllyasviel's Forge) — SD pipeline loading patterns reference
- [**TripoSR**](https://github.com/VAST-AI-Research/TripoSR) — VAST-AI / Stability AI / Tripo — Single-image 3D mesh reconstruction

### Lip Sync

- [**MuseTalk**](https://github.com/TMElyralab/MuseTalk) — TMElyralab (Tencent) — Real-time latent-space lip synchronization
- [**LatentSync**](https://github.com/bytedance/LatentSync) — ByteDance — SD-based lip sync with SyncNet supervision
- [**Video Retalking**](https://github.com/OpenTalker/video-retalking) — OpenTalker — Audio-based lip sync with expression editing and face enhancement

### Face Swap & Analysis

- [**InsightFace**](https://github.com/deepinsight/insightface) — DeepInsight — 2D/3D face detection, recognition, and alignment
- [**Roop**](https://github.com/s0md3v/roop) — s0md3v — One-click face swap, inswapper architecture
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN) — TencentARC — GAN-based blind face restoration
- [**CodeFormer**](https://github.com/sczhou/CodeFormer) — sczhou (S-Lab, NTU) — Codebook lookup transformer for face restoration

### Pose, Identity & Segmentation

- [**ControlNet**](https://github.com/lllyasviel/ControlNet) — lllyasviel (Lvmin Zhang) — Conditional control for diffusion models (OpenPose, Canny, Depth, Lineart, Normal)
- [**IP-Adapter**](https://github.com/tencent-ailab/IP-Adapter) — Tencent AI Lab — Image prompt adapter for identity-preserving generation
- [**BiRefNet**](https://github.com/ZhengPeng7/BiRefNet) — ZhengPeng7 — Bilateral reference network for high-resolution dichotomous image segmentation (Magic Mask)

### Training

- [**kohya sd-scripts**](https://github.com/kohya-ss/sd-scripts) — kohya-ss — LoRA/LoCon/LoHa training scripts for SD 1.5, SDXL, and derivatives

### Post-Processing

- [**RIFE**](https://github.com/hzwer/ECCV2022-RIFE) — Megvii Research — Real-time frame interpolation for temporal upsampling
- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN) — Xinntao — Practical image/video restoration for spatial upscaling

### Audio & Speech

- [**Chatterbox TTS**](https://github.com/resemble-ai/chatterbox) — Resemble AI — Voice cloning with emotion control
- [**Dia TTS**](https://github.com/nari-labs/dia) — Nari Labs — Multi-speaker dialogue synthesis
- [**MusicGen**](https://github.com/facebookresearch/audiocraft) — Meta / AudioCraft — Text-to-music generation
- [**AudioGen**](https://github.com/facebookresearch/audiocraft) — Meta / AudioCraft — Text-to-sound-effect generation
- [**Whisper**](https://github.com/openai/whisper) — OpenAI — Speech recognition for audio feature extraction in lip sync
- [**Wav2Vec 2.0**](https://huggingface.co/facebook/wav2vec2-base-960h) — Meta — Self-supervised audio representation for VACE MultiTalk

### Infrastructure

- [**PyTorch**](https://pytorch.org) — Meta — Deep learning framework
- [**PySide6 (Qt)**](https://www.qt.io/qt-for-python) — The Qt Company — Desktop UI framework
- [**FFmpeg**](https://ffmpeg.org) — FFmpeg team — Video/audio encoding, decoding, and compositing
- [**xformers**](https://github.com/facebookresearch/xformers) — Meta — Memory-efficient attention
- [**MMGP**](https://github.com/deepbeepmeep/mmgp) — deepbeepmeep — Memory management for GPU-poor setups

---

## License

Supreme Diffusion is provided as-is for personal and research use. Individual upstream components carry their own licenses — please refer to each project's repository for terms. Notable restrictions:

- **InsightFace / inswapper** models are for non-commercial research use only
- **CodeFormer** is released under the S-Lab License 1.0
- **MuseTalk**, **LatentSync**, and **Video Retalking** have their own research-use terms
- **Wan 2.1** models are released under the Apache 2.0 License

Users are responsible for ensuring they comply with all applicable licenses and local regulations when using face swap, lip sync, and deepfake-adjacent features. Always obtain consent when using real people's likenesses and clearly label generated content.

---
