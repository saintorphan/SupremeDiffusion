"""Face swap pipeline wrapping InsightFace + ONNX swap models.

Self-contained module — no dependency on roop-unleashed source code.
Algorithms faithfully adapted from roop-unleashed's face_util,
FaceSwapInsightFace, ProcessMgr, and Enhance_* modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ArcFace 5-point alignment template (112x112 normalised)
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Swap model metadata
SWAP_MODELS = {
    "inswapper_128": {"file": "inswapper_128.onnx", "size": 128},
    "reswapper_128": {"file": "reswapper_128.onnx", "size": 128},
    "reswapper_256": {"file": "reswapper_256.onnx", "size": 256},
}

ENHANCER_MODELS = {
    "gfpgan": {"file": "GFPGANv1.4.onnx", "size": 512},
    "codeformer": {"file": "CodeFormer/CodeFormerv0.1.onnx", "size": 512},
    "gpen": {"file": "GPEN-BFR-512.onnx", "size": 512},
    "restoreformer": {"file": "restoreformer_plus_plus.onnx", "size": 512},
}


def _estimate_affine(src_pts: np.ndarray, dst_pts: np.ndarray):
    """Estimate similarity transform (rotation + scale + translation)."""
    from skimage.transform import SimilarityTransform

    if hasattr(SimilarityTransform, "from_estimate"):
        tform = SimilarityTransform.from_estimate(src_pts, dst_pts)
    else:
        tform = SimilarityTransform()
        tform.estimate(src_pts, dst_pts)
    return tform.params[:2]  # 2x3 affine matrix


def _create_onnx_session(model_path: str):
    """Create an ONNX Runtime inference session with GPU, falling back to CPU."""
    import onnxruntime as ort

    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            return ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            logger.warning("CUDA failed for %s, falling back to CPU", Path(model_path).name)
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _extract_emap(model_path: str) -> np.ndarray:
    """Extract the embedding projection matrix from the ONNX model graph.

    roop-unleashed reads the last initializer from the ONNX graph — this is
    a learned projection matrix that transforms the source face embedding
    into the swap model's latent space.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    graph = model.graph
    # The emap is the last initializer in the graph
    emap = numpy_helper.to_array(graph.initializer[-1])
    return emap


def _color_transfer_lab(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Transfer colour statistics from target to source in LAB space.

    Matches mean and std of each LAB channel so the swapped face
    takes on the target's skin tone / lighting.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean, src_std = cv2.meanStdDev(src_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)

    src_mean = src_mean.reshape(1, 1, 3)
    src_std = src_std.reshape(1, 1, 3)
    tgt_mean = tgt_mean.reshape(1, 1, 3)
    tgt_std = tgt_std.reshape(1, 1, 3)

    # Prevent division by zero
    src_std = np.maximum(src_std, 1e-6)

    result = (src_lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


class FaceSwapPipeline:
    """Face swap pipeline using InsightFace analysis + ONNX swap models."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self._analyser = None
        self._swapper_cache: dict[str, Any] = {}
        self._enhancer_cache: dict[str, Any] = {}

    def _load_analyser(self):
        """Load InsightFace buffalo_l face analyser."""
        if self._analyser is not None:
            return self._analyser

        from insightface.app import FaceAnalysis

        # InsightFace resolves the model dir as ``<root>/models/<name>`` (it
        # always inserts a literal ``models`` segment). Our models live at
        # ``<face_models_dir>/buffalo_l`` (registry local_subdir="face/buffalo_l"),
        # so we must choose ``root`` such that ``root/models/buffalo_l`` lands on
        # the actual files — otherwise InsightFace looks at the wrong path and
        # silently re-downloads (or fails offline).
        bare_dir = self.models_dir / "buffalo_l"            # face/buffalo_l
        native_dir = self.models_dir / "models" / "buffalo_l"  # face/models/buffalo_l
        if native_dir.is_dir():
            # Already in InsightFace's native layout.
            root = self.models_dir
        elif bare_dir.is_dir():
            # Bare layout: point root at a staging dir whose ``models`` subdir
            # resolves to our model dir, so check and loader agree.
            root = self.models_dir
            staged = self.models_dir / "models" / "buffalo_l"
            if not staged.exists():
                staged.parent.mkdir(parents=True, exist_ok=True)
                try:
                    staged.symlink_to(bare_dir, target_is_directory=True)
                except (OSError, NotImplementedError):
                    # Windows / no-symlink filesystems: copy instead.
                    import shutil
                    shutil.copytree(bare_dir, staged)
        else:
            raise FileNotFoundError(
                f"InsightFace buffalo_l model not found at {bare_dir}"
            )

        self._analyser = FaceAnalysis(
            name="buffalo_l",
            root=str(root),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._analyser.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Loaded InsightFace buffalo_l analyser")
        return self._analyser

    def _load_swapper(self, model_key: str):
        """Load an ONNX swap model session + its embedding projection matrix."""
        if model_key in self._swapper_cache:
            return self._swapper_cache[model_key]

        info = SWAP_MODELS.get(model_key)
        if info is None:
            raise ValueError(f"Unknown swap model: {model_key}")

        path = self.models_dir / info["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Swap model not found: {path}")

        session = _create_onnx_session(path)
        emap = _extract_emap(str(path))

        self._swapper_cache[model_key] = {
            "session": session,
            "size": info["size"],
            "inputs": [inp.name for inp in session.get_inputs()],
            "output": session.get_outputs()[0].name,
            "emap": emap,
        }
        logger.info("Loaded swap model: %s (emap shape %s)", model_key, emap.shape)
        return self._swapper_cache[model_key]

    def _load_enhancer(self, name: str):
        """Load a face enhancement ONNX model.

        Only one enhancer is kept loaded at a time to conserve VRAM.
        """
        key = name.lower().replace("++", "").replace(" ", "")
        if key in self._enhancer_cache:
            return self._enhancer_cache[key]

        info = ENHANCER_MODELS.get(key)
        if info is None:
            raise ValueError(f"Unknown enhancer: {name}")

        # Release any previously loaded enhancer to free VRAM
        if self._enhancer_cache:
            logger.info("Releasing previous enhancer to free VRAM")
            self._enhancer_cache.clear()

        path = self.models_dir / info["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Enhancer model not found: {path}")

        # Enhancers run on CPU to avoid VRAM contention with swap models
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._enhancer_cache[key] = {
            "session": session,
            "size": info["size"],
            "key": key,
            "inputs": [inp.name for inp in session.get_inputs()],
            "outputs": [out.name for out in session.get_outputs()],
        }
        logger.info("Loaded enhancer: %s", name)
        return self._enhancer_cache[key]

    def detect_faces(self, image_path: str) -> list[dict]:
        """Detect faces in an image, return list of face info dicts."""
        analyser = self._load_analyser()
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = analyser.get(img)
        results = []
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = bbox
            pad = int((x2 - x1) * 0.2)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            results.append(
                {
                    "bbox": bbox.tolist(),
                    "kps": face.kps.tolist() if face.kps is not None else None,
                    "embedding": face.normed_embedding,
                    "det_score": float(getattr(face, "det_score", 1.0)),
                    "index": i,
                    "crop_image": Image.fromarray(crop_rgb),
                    "_face_obj": face,
                }
            )
        return results

    def _align_face(self, img: np.ndarray, kps: np.ndarray, size: int) -> tuple:
        """Align and crop face using 5-point landmarks → arcface template.

        Matches roop-unleashed's face_util.align_crop / estimate_norm.
        """
        dst = ARCFACE_DST.copy()
        # Scale template to target size and apply size-dependent offsets
        ratio = size / 112.0
        dst = dst * ratio
        if size >= 512:
            dst += np.array([1.5, 1.5], dtype=np.float32)
        elif size >= 320:
            dst += np.array([0.75, 0.75], dtype=np.float32)
        elif size >= 256:
            dst += np.array([0.5, 0.5], dtype=np.float32)
        elif size >= 160:
            dst += np.array([0.1, 0.1], dtype=np.float32)

        M = _estimate_affine(kps, dst)
        aligned = cv2.warpAffine(
            img, M, (size, size), borderValue=0.0,
        )
        return aligned, M

    def _run_swap(
        self, swapper: dict, aligned: np.ndarray, source_embedding: np.ndarray,
    ) -> np.ndarray:
        """Run the ONNX swap model on an aligned face.

        Matches roop-unleashed's ProcessMgr.prepare_crop_frame →
        FaceSwapInsightFace.process → ProcessMgr.normalize_swap_frame.

        InSwapper normalisation: BGR→RGB, /255, mean=0 std=1 → [0,1] range.
        Embedding is projected through the emap matrix then L2-normalised.
        """
        session = swapper["session"]
        emap = swapper["emap"]

        # -- Prepare face tensor (roop: prepare_crop_frame) --
        # BGR → RGB, then [0, 1] float, NCHW
        blob = aligned[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # 1×C×H×W

        # -- Project embedding through emap (roop: FaceSwapInsightFace) --
        latent = source_embedding.reshape(1, -1)
        latent = np.dot(latent, emap).astype(np.float32)
        latent /= np.linalg.norm(latent)

        # -- Run inference --
        inputs = swapper["inputs"]
        feed = {inputs[0]: blob, inputs[1]: latent}
        output = session.run([swapper["output"]], feed)[0]

        # -- Denormalise (roop: normalize_swap_frame) --
        # NCHW → HWC, × 255, RGB → BGR
        result = output[0].transpose(1, 2, 0)
        result = (result * 255.0).round()
        result = result[:, :, ::-1]  # RGB → BGR
        return np.clip(result, 0, 255).astype(np.uint8)

    def _run_enhancer(
        self, enhancer: dict, face_img: np.ndarray, strength: float = 0.5,
    ) -> tuple[np.ndarray, int]:
        """Run face enhancement model on a face crop.

        Matches roop-unleashed's Enhance_GFPGAN / Enhance_CodeFormer:
        - Input: resize to 512, BGR→RGB, /255, (x−0.5)/0.5 → [-1,1], NCHW
        - Output: clip [-1,1], (x+1)/2 → [0,1], × 255, RGB→BGR

        Args:
            strength: CodeFormer fidelity weight (0=quality, 1=fidelity).
                      Ignored by non-CodeFormer enhancers.

        Returns (enhanced_bgr, scale_factor).
        """
        session = enhancer["session"]
        key = enhancer["key"]
        size = enhancer["size"]
        input_size = face_img.shape[1]

        # Resize to model input
        resized = cv2.resize(face_img, (size, size), interpolation=cv2.INTER_CUBIC)

        # BGR → RGB, [0,1], then (x−0.5)/0.5 → [-1,1]
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = (blob - 0.5) / 0.5
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)  # 1×C×H×W

        # Build feed dict
        inputs = enhancer["inputs"]
        feed = {inputs[0]: blob}
        if key == "codeformer" and len(inputs) > 1:
            # CodeFormer quality weight (double precision as roop does)
            feed[inputs[1]] = np.array([strength], dtype=np.float64)

        outputs = enhancer["outputs"]
        result = session.run([outputs[0]], feed)[0]

        # Post-process: NCHW → HWC, clip [-1,1], → [0,1], × 255, RGB→BGR
        result = result[0].transpose(1, 2, 0)
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2.0
        result = cv2.cvtColor((result * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        scale_factor = max(1, result.shape[1] // input_size)
        return result, scale_factor

    def _paste_back(
        self,
        target_img: np.ndarray,
        swapped_face: np.ndarray,
        M: np.ndarray,
        face_size: int,
        num_erosion_iterations: int = 1,
        blur_amount: int = 20,
    ) -> np.ndarray:
        """Paste swapped face back onto target with mask, erosion, and blur.

        Matches roop-unleashed's ProcessMgr.paste_upscale:
        1. Create a white rectangle mask in face space
        2. Warp both face and mask back via inverse affine
        3. Erode mask edges, then Gaussian blur for feathering
        4. Normalised float blend: result = mask * face + (1−mask) * target
        """
        result = target_img.copy()
        h, w = result.shape[:2]

        # Scale affine matrix if swapped face was upscaled (e.g. by enhancer)
        scale = swapped_face.shape[0] / face_size
        M_scaled = M * scale

        IM = cv2.invertAffineTransform(M_scaled)

        # Create mask in face space — shrink inward slightly to avoid
        # warping artifacts at the very edge of the aligned crop
        face_h, face_w = swapped_face.shape[:2]
        img_matte = np.zeros((face_h, face_w), dtype=np.uint8)
        border = max(1, face_h // 16)  # ~6% inset
        img_matte[border:face_h - border, border:face_w - border] = 255

        # Warp face and mask back to target image space
        face_warped = cv2.warpAffine(
            swapped_face, IM, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        img_matte = cv2.warpAffine(
            img_matte, IM, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        # Clamp border pixels to prevent edge artifacts
        img_matte[:1, :] = 0
        img_matte[-1:, :] = 0
        img_matte[:, :1] = 0
        img_matte[:, -1:] = 0

        # Feather the mask edges with Gaussian blur for smooth blending
        mask_indices = np.where(img_matte > 0)
        if len(mask_indices[0]) > 0:
            mask_h = np.max(mask_indices[0]) - np.min(mask_indices[0])
            mask_w = np.max(mask_indices[1]) - np.min(mask_indices[1])
            mask_size = max(1, int(np.sqrt(mask_h * mask_w)))

            # Light erosion — just clean up interpolation fringes
            ek = max(2, mask_size // 40)
            erosion_kernel = np.ones((ek, ek), np.uint8)
            img_matte = cv2.erode(
                img_matte, erosion_kernel, iterations=num_erosion_iterations,
            )

            # Feather blur — proportional to face size for smooth edges
            bk = max(3, mask_size // 10)
            bk = bk if bk % 2 == 1 else bk + 1  # must be odd
            img_matte = cv2.GaussianBlur(img_matte, (bk, bk), 0)

        # Normalise mask to [0, 1] float for blending
        matte_float = img_matte.astype(np.float32) / 255.0
        matte_3ch = matte_float[:, :, np.newaxis]

        # Blend
        blended = matte_3ch * face_warped.astype(np.float32) + \
                  (1.0 - matte_3ch) * result.astype(np.float32)
        return blended.astype(np.uint8)

    def _enhance_source_face(self, source_path: str) -> str:
        """Pre-enhance the source image's face with GFPGAN.

        Inswapper's embedding extractor runs on an aligned 112×112 face crop;
        a sharper, less-noisy source face yields a stronger identity embedding
        and stronger downstream transfer. Returns a path to a temp PNG with the
        enhanced face composited back into the original source image. Falls
        back to ``source_path`` unchanged if no face is detected.
        """
        import tempfile

        faces = self.detect_faces(source_path)
        if not faces:
            return source_path
        img = cv2.imread(str(source_path))
        if img is None:
            return source_path
        kps = np.array(faces[0]["kps"])
        aligned, M = self._align_face(img, kps, 512)
        enhancer = self._load_enhancer("gfpgan")
        enhanced, _scale = self._run_enhancer(enhancer, aligned, strength=1.0)
        result = self._paste_back(img, enhanced, M, 512)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, result)
        tmp.close()
        return tmp.name

    def swap(
        self,
        source_path: str,
        target_path: str,
        source_face_idx: int = 0,
        target_face_idx: int = 0,
        swap_model: str = "inswapper_128",
        enhancer: str | None = None,
        blend_ratio: float = 0.5,
        swap_all: bool = False,
        enhancer_strength: float = 0.5,
        source_face_map: dict[int, int] | None = None,
        enhance_source: bool = False,
    ) -> Image.Image:
        """Run the full face swap pipeline.

        Pipeline (matching roop-unleashed):
        1. Detect faces in source & target
        2. Align target face via 5-point arcface template
        3. Run ONNX swap model (embedding projected through emap)
        4. Optional: run enhancer (GFPGAN/CodeFormer/GPEN/RestoreFormer++)
        5. Colour-correct swapped face to match target skin tone (LAB)
        6. Paste back with eroded + blurred mask

        Args:
            enhancer_strength: CodeFormer fidelity weight (0=quality, 1=fidelity).
            source_face_map: Optional mapping {target_face_idx: source_face_idx}
                for multi-source swap. Overrides source_face_idx when provided.
        """
        # Optional source pre-enhance — sharpens the source face before
        # InsightFace extracts the identity embedding.
        enhanced_tmp: str | None = None
        if enhance_source:
            new_path = self._enhance_source_face(source_path)
            if new_path != source_path:
                enhanced_tmp = new_path
            source_path = new_path

        try:
            # Detect faces
            source_faces = self.detect_faces(source_path)
        finally:
            # The enhanced source is only needed for embedding extraction;
            # prune the temp PNG now that detection has read it.
            if enhanced_tmp is not None:
                try:
                    Path(enhanced_tmp).unlink()
                except OSError:
                    pass
        if not source_faces:
            raise ValueError("No face detected in source image")
        if source_face_idx >= len(source_faces):
            raise ValueError(
                f"Source face index {source_face_idx} out of range "
                f"(found {len(source_faces)} faces)"
            )

        target_faces = self.detect_faces(target_path)
        if not target_faces:
            raise ValueError("No face detected in target image")

        default_embedding = source_faces[source_face_idx]["embedding"]

        # Load swap model
        swapper = self._load_swapper(swap_model)
        swap_size = swapper["size"]

        # Load enhancer if requested
        enhancer_model = None
        if enhancer and enhancer.lower() != "none":
            enhancer_model = self._load_enhancer(enhancer)

        # Read target image
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            raise ValueError(f"Could not read target image: {target_path}")

        result = target_img.copy()

        # Determine which faces to swap
        if swap_all:
            faces_to_swap = target_faces
        else:
            if target_face_idx >= len(target_faces):
                raise ValueError(
                    f"Target face index {target_face_idx} out of range "
                    f"(found {len(target_faces)} faces)"
                )
            faces_to_swap = [target_faces[target_face_idx]]

        for face_info in faces_to_swap:
            face_obj = face_info["_face_obj"]
            face_idx = face_info["index"]
            kps = (
                np.array(face_info["kps"], dtype=np.float32)
                if isinstance(face_info["kps"], list)
                else face_obj.kps
            )

            # Resolve source embedding for this target face
            embedding = default_embedding
            if source_face_map and face_idx in source_face_map:
                src_idx = source_face_map[face_idx]
                if 0 <= src_idx < len(source_faces):
                    embedding = source_faces[src_idx]["embedding"]

            # Align target face at swap model resolution
            aligned, M = self._align_face(result, kps, swap_size)

            # Run swap
            swapped = self._run_swap(swapper, aligned, embedding)

            # Colour-correct swapped face to match the aligned target region
            swapped = _color_transfer_lab(swapped, aligned)

            # Optional enhancement
            paste_size = swap_size
            if enhancer_model is not None:
                enhanced, scale_factor = self._run_enhancer(
                    enhancer_model, swapped, strength=enhancer_strength,
                )
                # Upscale raw swap to match enhanced resolution for blending
                if enhanced.shape[:2] != swapped.shape[:2]:
                    swapped = cv2.resize(
                        swapped, (enhanced.shape[1], enhanced.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                # blend_ratio controls how much enhancement to apply
                swapped = cv2.addWeighted(
                    swapped, 1.0 - blend_ratio, enhanced, blend_ratio, 0,
                )
                paste_size = swap_size  # M was computed at swap_size

            # Paste back with proper mask erosion + blur
            result = self._paste_back(result, swapped, M, paste_size)

        # Convert BGR → RGB → PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def swap_frame(
        self,
        target_img: np.ndarray,
        source_embedding: np.ndarray,
        target_faces: list[dict],
        swap_model: str = "inswapper_128",
        enhancer: str | None = None,
        blend_ratio: float = 0.5,
        enhancer_strength: float = 0.5,
        swap_all: bool = False,
        target_face_idx: int = 0,
        source_embeddings: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Swap face on a pre-loaded BGR frame. Used for video processing.

        Unlike swap(), this takes a raw numpy frame and pre-computed faces
        to avoid repeated file I/O and face detection per frame.

        Args:
            source_embedding: Default identity embedding applied to every
                swapped target face.
            source_embeddings: Optional mapping {target_face_idx: embedding}
                for multi-source video/batch swap. When a target face's
                ``index`` is present, its embedding overrides
                ``source_embedding``; otherwise the default is used. Keeps
                backward compatibility with single-source callers.

        Returns the result as a BGR numpy array.
        """
        swapper = self._load_swapper(swap_model)
        swap_size = swapper["size"]

        enhancer_model = None
        if enhancer and enhancer.lower() != "none":
            enhancer_model = self._load_enhancer(enhancer)

        result = target_img.copy()

        if swap_all:
            faces = target_faces
        else:
            if target_face_idx < len(target_faces):
                faces = [target_faces[target_face_idx]]
            else:
                faces = target_faces[:1] if target_faces else []

        for face_info in faces:
            face_obj = face_info["_face_obj"]
            kps = (
                np.array(face_info["kps"], dtype=np.float32)
                if isinstance(face_info["kps"], list)
                else face_obj.kps
            )

            # Resolve source embedding for this target face (multi-source map
            # overrides the default when the target face index is present).
            embedding = source_embedding
            if source_embeddings:
                mapped = source_embeddings.get(face_info.get("index"))
                if mapped is not None:
                    embedding = mapped

            aligned, M = self._align_face(result, kps, swap_size)
            swapped = self._run_swap(swapper, aligned, embedding)
            swapped = _color_transfer_lab(swapped, aligned)

            paste_size = swap_size
            if enhancer_model is not None:
                enhanced, _ = self._run_enhancer(
                    enhancer_model, swapped, strength=enhancer_strength,
                )
                if enhanced.shape[:2] != swapped.shape[:2]:
                    swapped = cv2.resize(
                        swapped, (enhanced.shape[1], enhanced.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )
                swapped = cv2.addWeighted(
                    swapped, 1.0 - blend_ratio, enhanced, blend_ratio, 0,
                )
                paste_size = swap_size

            result = self._paste_back(result, swapped, M, paste_size)

        return result

    def release(self):
        """Free ONNX sessions and analyser."""
        self._swapper_cache.clear()
        self._enhancer_cache.clear()
        self._analyser = None
        logger.info("Released FaceSwapPipeline resources")
