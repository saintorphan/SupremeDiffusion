#!/usr/bin/env python3
"""Download README files from upstream projects for Qwen chat reference.

Run once (or periodically to refresh):
    python scripts/download_upstream_docs.py

Downloads to docs/upstream/{name}.md
"""

import urllib.request
import urllib.error
import ssl
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs" / "upstream"

# (local_filename, raw_readme_url, keywords for search matching)
UPSTREAM_DOCS = [
    (
        "wan_video.md",
        "https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/README.md",
        ["wan", "wan2.1", "wan2.2", "i2v", "video generation", "14b"],
    ),
    (
        "diffusers.md",
        "https://raw.githubusercontent.com/huggingface/diffusers/main/README.md",
        ["diffusers", "huggingface", "pipeline", "scheduler", "sampler"],
    ),
    (
        "stable_diffusion.md",
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/README.md",
        ["stable diffusion", "sd", "sd15", "sdxl", "txt2img", "img2img"],
    ),
    (
        "flux.md",
        "https://raw.githubusercontent.com/black-forest-labs/flux/main/README.md",
        ["flux", "chroma", "flux fill", "black forest"],
    ),
    (
        "controlnet.md",
        "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/README.md",
        ["controlnet", "control net", "openpose", "canny", "depth", "condition"],
    ),
    (
        "ip_adapter.md",
        "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/README.md",
        ["ip-adapter", "ip adapter", "image prompt", "style transfer"],
    ),
    (
        "latentsync.md",
        "https://raw.githubusercontent.com/bytedance/LatentSync/main/README.md",
        ["latentsync", "latent sync", "lip sync", "lipsync", "audio driven"],
    ),
    (
        "musetalk.md",
        "https://raw.githubusercontent.com/TMElyralab/MuseTalk/main/README.md",
        ["musetalk", "muse talk", "lip sync", "talking head"],
    ),
    (
        "video_retalking.md",
        "https://raw.githubusercontent.com/OpenTalker/video-retalking/main/README.md",
        ["video retalking", "retalking", "lip sync", "face reenactment"],
    ),
    (
        "insightface.md",
        "https://raw.githubusercontent.com/deepinsight/insightface/master/README.md",
        ["insightface", "face detection", "face recognition", "buffalo", "face analysis"],
    ),
    (
        "roop.md",
        "https://raw.githubusercontent.com/s0md3v/roop/main/README.md",
        ["roop", "face swap", "faceswap", "inswapper"],
    ),
    (
        "gfpgan.md",
        "https://raw.githubusercontent.com/TencentARC/GFPGAN/master/README.md",
        ["gfpgan", "face restoration", "face enhance", "face upscale"],
    ),
    (
        "codeformer.md",
        "https://raw.githubusercontent.com/sczhou/CodeFormer/master/README.md",
        ["codeformer", "code former", "face restoration"],
    ),
    (
        "triposr.md",
        "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/README.md",
        ["triposr", "tripo", "3d", "mesh", "3d modeling", "3d generation"],
    ),
    (
        "real_esrgan.md",
        "https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/README.md",
        ["real-esrgan", "esrgan", "upscale", "super resolution", "upscaling"],
    ),
    (
        "rife.md",
        "https://raw.githubusercontent.com/hzwer/ECCV2022-RIFE/main/README.md",
        ["rife", "frame interpolation", "interpolation", "slow motion"],
    ),
    (
        "orpheus_tts.md",
        "https://raw.githubusercontent.com/canopyai/Orpheus-TTS/main/README.md",
        ["orpheus", "tts", "text to speech", "speech synthesis"],
    ),
    (
        "openvoice.md",
        "https://raw.githubusercontent.com/myshell-ai/OpenVoice/main/README.md",
        ["openvoice", "open voice", "voice cloning", "voice clone", "tts"],
    ),
    (
        "rvc.md",
        "https://raw.githubusercontent.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/main/README.md",
        ["rvc", "voice conversion", "retrieval", "voice change"],
    ),
    (
        "lora.md",
        "https://raw.githubusercontent.com/cloneofsimo/lora/master/README.md",
        ["lora", "low-rank", "low rank", "adaptation", "fine-tune", "finetune"],
    ),
    (
        "xformers.md",
        "https://raw.githubusercontent.com/facebookresearch/xformers/main/README.md",
        ["xformers", "memory efficient", "attention"],
    ),
    (
        "birefnet.md",
        "https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/main/README.md",
        ["birefnet", "background removal", "segmentation", "magic mask", "matting"],
    ),
    (
        "sd3.md",
        "https://raw.githubusercontent.com/Stability-AI/sd3.5/main/README.md",
        ["sd3", "sd3.5", "stable diffusion 3", "mmdit"],
    ),
    (
        "whisper.md",
        "https://raw.githubusercontent.com/openai/whisper/main/README.md",
        ["whisper", "speech recognition", "transcription", "audio"],
    ),
]


def download_all() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Allow unverified SSL for corporate/proxy environments
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    success = 0
    failed = 0

    for filename, url, _keywords in UPSTREAM_DOCS:
        dest = DOCS_DIR / filename
        print(f"  {filename} ... ", end="", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SupremeDiffusion/1.0"})
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            dest.write_text(content, encoding="utf-8")
            print(f"OK ({len(content)} bytes)")
            success += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

    print(f"\nDone: {success} downloaded, {failed} failed")


if __name__ == "__main__":
    print("Downloading upstream docs to docs/upstream/\n")
    download_all()
