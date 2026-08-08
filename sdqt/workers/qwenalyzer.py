"""Worker thread for QwenAlyzer — image analysis and prompt generation via Qwen 3.5 4B."""

from __future__ import annotations

import logging

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

# System prompts for image-to-prompt analysis
_ANALYZE_PROMPTS = {
    "sd15": (
        "You are an expert at analyzing images and generating Stable Diffusion 1.5 prompts. "
        "Given a description of an image, generate a detailed comma-separated tag prompt that would "
        "reproduce this image. Start with quality tags like 'masterpiece, best quality, highly detailed'. "
        "Include subject, pose, clothing, expression, setting, lighting, camera angle, and style tags. "
        "Use danbooru/booru-style tags. Output only the prompt, nothing else."
    ),
    "sdxl": (
        "You are an expert at analyzing images and generating SDXL prompts. "
        "Given a description of an image, generate a detailed natural language prompt that would "
        "reproduce this image. Describe the subject, composition, lighting, atmosphere, colors, "
        "and style in flowing prose (2-4 sentences). Include quality modifiers. "
        "Output only the prompt, nothing else."
    ),
    "sd3": (
        "You are an expert at analyzing images and generating Stable Diffusion 3 prompts. "
        "SD3 uses T5 and understands full natural language. Given a description of an image, "
        "generate a detailed flowing prompt (3-5 sentences). Describe subject, scene, composition, "
        "lighting, color palette, and artistic style in plain English. No booru tags. "
        "Output only the prompt, nothing else."
    ),
    "flux": (
        "You are an expert at analyzing images and generating FLUX prompts. "
        "FLUX excels at long, descriptive natural language. Given a description of an image, "
        "generate a richly detailed paragraph (4-6 sentences). Describe the subject in detail, "
        "then environment, lighting, camera angle, mood, color grading, and artistic style. "
        "Be specific and vivid. No booru tags — write flowing prose. "
        "Output only the prompt, nothing else."
    ),
    "pony_real": (
        "You are an expert at analyzing images and generating Pony Diffusion (realistic) prompts. "
        "Given a description of an image, generate a prompt starting with 'score_9, score_8_up, score_7_up, ' "
        "followed by realistic/photographic tags: detailed skin, natural lighting, photorealism. "
        "Use booru-style comma-separated tags but lean toward real-world descriptors. "
        "Output only the prompt, nothing else."
    ),
    "pony_anime": (
        "You are an expert at analyzing images and generating Pony Diffusion (anime) prompts. "
        "Given a description of an image, generate a prompt starting with 'score_9, score_8_up, score_7_up, ' "
        "followed by anime-style booru tags. Include style tags like 'anime coloring, detailed eyes'. "
        "Output only the prompt, nothing else."
    ),
    "illustrious": (
        "You are an expert at analyzing images and generating Illustrious Diffusion prompts. "
        "Given a description of an image, generate a prompt with quality tags (masterpiece, best quality, "
        "absurdres) followed by detailed booru-style descriptors of the subject, scene, and style. "
        "Output only the prompt, nothing else."
    ),
    "noobai": (
        "You are an expert at analyzing images and generating NoobAI prompts. "
        "Given a description of an image, generate a prompt starting with 'masterpiece, best quality, "
        "amazing quality, very aesthetic, absurdres, ' followed by detailed booru-style tags. "
        "Include character descriptors, scene details, and style tags in danbooru format. "
        "Output only the prompt, nothing else."
    ),
}

_ANALYZE_NEG_PROMPTS = {
    "sd15": (
        "You are a negative prompt generator for Stable Diffusion 1.5. "
        "Given a positive prompt describing an image, generate appropriate negative tags "
        "to avoid artifacts and low quality. Include 'worst quality, low quality, normal quality, "
        "lowres, bad anatomy, bad hands, deformed, blurry, watermark, text, extra fingers'. "
        "Output only the negative prompt, nothing else."
    ),
    "sdxl": (
        "You are a negative prompt generator for SDXL. "
        "Given a positive prompt, generate appropriate negative terms. "
        "Include 'low quality, blurry, distorted, deformed, disfigured, bad anatomy, "
        "watermark, text, oversaturated, ugly'. Output only the negative prompt, nothing else."
    ),
    "sd3": (
        "You are a negative prompt generator for Stable Diffusion 3. "
        "SD3 uses natural language for negatives. Generate a short sentence or two describing "
        "what to avoid: low quality, blurriness, distortion, bad anatomy, artifacts. "
        "Keep it in plain English. Output only the negative prompt, nothing else."
    ),
    "flux": (
        "FLUX does not use negative prompts. Output only: "
        "'(FLUX does not support negative prompts — describe what you want in the positive prompt instead.)'"
    ),
    "pony_real": (
        "You are a negative prompt generator for Pony Diffusion (realistic). "
        "Generate negative tags starting with 'score_4, score_3, score_2, score_1, ' "
        "followed by 'worst quality, low quality, bad anatomy, watermark, text, "
        "anime, cartoon, illustration, painting, drawing, 3d render'. "
        "Output only the negative prompt, nothing else."
    ),
    "pony_anime": (
        "You are a negative prompt generator for Pony Diffusion (anime). "
        "Generate negative tags starting with 'score_4, score_3, score_2, score_1, ' "
        "followed by 'worst quality, low quality, bad anatomy, watermark, text, "
        "realistic, photo, 3d render, bad hands, extra fingers'. "
        "Output only the negative prompt, nothing else."
    ),
    "illustrious": (
        "You are a negative prompt generator for Illustrious Diffusion. "
        "Generate negative tags including 'worst quality, low quality, normal quality, "
        "lowres, bad anatomy, bad hands, extra digits, fewer digits, text, watermark, "
        "blurry, jpeg artifacts'. Output only the negative prompt, nothing else."
    ),
    "noobai": (
        "You are a negative prompt generator for NoobAI. "
        "Generate negative tags including 'worst quality, bad quality, low quality, "
        "normal quality, lowres, bad anatomy, bad hands, extra digits, fewer digits, "
        "text, watermark, signature, blurry, jpeg artifacts, ugly'. "
        "Output only the negative prompt, nothing else."
    ),
}


class QwenAlyzerWorker(BaseWorker):
    """Analyze an image using Qwen 2.5 VL and generate pos/neg prompts.

    Stage 1: Qwen VL sees the image and generates a detailed description.
    Stage 2: Qwen 3.5 4B text model rewrites the description into a
    style-specific prompt (SD 1.5, SDXL, FLUX, etc.).

    Returns a dict with "pos" and/or "neg" keys depending on `which`.
    """

    def __init__(
        self,
        image_path: str,
        style_key: str,
        style_name: str,
        which: str,
        state,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._style_key = style_key
        self._style_name = style_name
        self._which = which  # "pos", "neg", or "both"
        self._state = state

    def do_work(self) -> dict:
        if self.is_aborted:
            raise InterruptedError("Aborted")

        # Stage 1: VL model describes the image
        self.status.emit("Loading Qwen VL...")
        self._state.load_qwen_vl()

        if self.is_aborted:
            raise InterruptedError("Aborted")

        self.status.emit("Analyzing image...")
        description = self._state.caption_image(
            self._image_path,
            prompt=(
                "Describe this image in thorough detail. Cover the subject (appearance, "
                "pose, expression, clothing, hair), the setting/background, lighting, "
                "camera angle, color palette, and artistic style. Be specific and factual."
            ),
            max_new_tokens=400,
        )
        logger.info("VL description: %s", description[:200])

        # Unload VL, load text Qwen for prompt formatting
        self._state.unload_qwen_vl()

        if self.is_aborted:
            raise InterruptedError("Aborted")

        # Stage 2: Text Qwen formats into style-specific prompt
        self.status.emit("Loading Qwen 3.5 4B...")
        self._state.load_qwen()

        result = {}

        if self._which in ("pos", "both"):
            if self.is_aborted:
                raise InterruptedError("Aborted")

            system_prompt = _ANALYZE_PROMPTS.get(
                self._style_key, _ANALYZE_PROMPTS["sdxl"]
            )
            user_text = (
                f"Based on this detailed image description, generate a {self._style_name} "
                f"prompt to reproduce the image.\n\n{description}"
            )

            self.status.emit("Generating prompt...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]
            result["pos"] = self._state.generate_chat_response(messages, max_new_tokens=512)

        if self._which in ("neg", "both"):
            if self.is_aborted:
                raise InterruptedError("Aborted")

            neg_system = _ANALYZE_NEG_PROMPTS.get(
                self._style_key, _ANALYZE_NEG_PROMPTS["sdxl"]
            )
            pos_context = result.get("pos", description)
            user_text = f"Generate a negative prompt for this image:\n\n{pos_context}"

            self.status.emit("Generating negative prompt...")
            messages = [
                {"role": "system", "content": neg_system},
                {"role": "user", "content": user_text},
            ]
            result["neg"] = self._state.generate_chat_response(messages, max_new_tokens=256)

        return result
