"""TripoSR pipeline -- image to 3D mesh conversion with BiRefNet BG removal."""

from __future__ import annotations

import gc as gc_module
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


class TripoSRPipeline:
    """Wraps BiRefNet background removal and TripoSR 3D mesh generation.

    Pipeline:
        1. Remove background from input image (BiRefNet)
        2. Generate 3D mesh from the BG-removed image (TripoSR)
        3. Export mesh as OBJ/GLB

    Both models are loaded lazily and cached.
    """

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self._birefnet_model: Any = None
        self._birefnet_transform: Any = None
        self._tsr_model: Any = None
        self._loaded_birefnet: bool = False
        self._loaded_tsr: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_birefnet(self, model_path: str | None = None) -> None:
        """Load BiRefNet background removal model."""
        if self._loaded_birefnet:
            return

        if torch is None:
            raise RuntimeError("PyTorch required.")

        path = model_path or self.config.model_paths.get("birefnet_dir", "")
        if not path:
            raise ValueError("BiRefNet model path not configured.")

        logger.info("Loading BiRefNet from: %s", path)

        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        self._birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            path, trust_remote_code=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._birefnet_model = self._birefnet_model.to(device)
        self._birefnet_model.eval()

        self._birefnet_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self._loaded_birefnet = True
        logger.info("BiRefNet loaded.")

    def load_tsr(self, model_path: str | None = None) -> None:
        """Load TripoSR 3D reconstruction model."""
        if self._loaded_tsr:
            return

        if torch is None:
            raise RuntimeError("PyTorch required.")

        path = model_path or self.config.model_paths.get("triposr_dir", "")
        if not path:
            # Try default HuggingFace model ID
            path = "stabilityai/TripoSR"

        logger.info("Loading TripoSR from: %s", path)

        # TripoSR is not on PyPI — add the cloned repo to sys.path
        import sys
        _tsr_repo = str(Path.home() / "Projects" / "TripoSR")
        if _tsr_repo not in sys.path and Path(_tsr_repo).is_dir():
            sys.path.insert(0, _tsr_repo)

        from tsr.system import TSR

        self._tsr_model = TSR.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tsr_model.to(device)

        # Set reasonable chunk size for VRAM management
        self._tsr_model.renderer.set_chunk_size(8192)

        self._loaded_tsr = True
        logger.info("TripoSR loaded.")

    # ------------------------------------------------------------------
    # Background Removal
    # ------------------------------------------------------------------

    def remove_background(self, image_path: str) -> Image.Image:
        """Remove background from an image using BiRefNet.

        Returns RGBA PIL Image with transparent background.
        """
        if not self._loaded_birefnet:
            self.load_birefnet()

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        device = next(self._birefnet_model.parameters()).device
        input_tensor = self._birefnet_transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            preds = self._birefnet_model(input_tensor)[-1].sigmoid()

        # Resize mask back to original dimensions
        mask = preds[0].squeeze()
        mask_pil = transforms_to_pil(mask, (w, h))

        # Apply mask as alpha channel
        result = img.copy()
        result.putalpha(mask_pil)

        return result

    # ------------------------------------------------------------------
    # 3D Mesh Generation
    # ------------------------------------------------------------------

    def generate_mesh(
        self,
        image: str | Image.Image,
        output_path: str | None = None,
        resolution: int = 256,
        output_format: str = "obj",
        remove_bg: bool = True,
        callback: Optional[Callable] = None,
    ) -> str:
        """Convert an image to a 3D mesh.

        Args:
            image: Path to image or PIL Image
            output_path: Where to save the mesh (auto-generated if None)
            resolution: Mesh resolution (higher = more detail, more VRAM)
            output_format: "obj", "glb", "ply", or "stl"
            remove_bg: Whether to remove background first
            callback: Optional progress callback

        Returns:
            Path to the saved mesh file.
        """
        if not self._loaded_tsr:
            self.load_tsr()

        # Load image
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image

        # Remove background if requested
        if remove_bg:
            if callback:
                callback("Removing background...")
            if isinstance(image, str):
                img = self.remove_background(image)
            else:
                # Save to temp, remove BG, reload
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(tmp.name)
                tmp.close()
                img = self.remove_background(tmp.name)

            # Convert RGBA → RGB on white background for TripoSR
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg

        if callback:
            callback("Generating 3D mesh...")

        # Generate scene codes
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            scene_codes = self._tsr_model([img], device=device)

        if callback:
            callback("Extracting mesh...")

        # Extract mesh
        meshes = self._tsr_model.extract_mesh(scene_codes, resolution=resolution)
        mesh = meshes[0]

        # Determine output path
        if output_path is None:
            output_path = tempfile.mktemp(suffix=f".{output_format}", prefix="mesh_")

        # Ensure correct extension
        out_p = Path(output_path)
        if out_p.suffix.lower() != f".{output_format}":
            output_path = str(out_p.with_suffix(f".{output_format}"))

        # Try UV unwrapping + texture baking if xatlas is available
        self._try_uv_and_texture(mesh, img, output_path)

        mesh.export(output_path)
        logger.info("Mesh exported: %s", output_path)

        self._cleanup()
        return output_path

    def _try_uv_and_texture(self, mesh, source_image, output_path: str) -> None:
        """Attempt UV unwrapping (xatlas) and texture baking from vertex colors.

        This is a best-effort step — if xatlas is not installed, the mesh
        exports with vertex colors only (which still works fine in the viewport).
        """
        try:
            import xatlas
            import numpy as np
        except ImportError:
            logger.debug("xatlas not installed — skipping UV unwrapping")
            return

        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            return

        try:
            if callback := getattr(self, '_current_callback', None):
                callback("UV unwrapping...")

            verts = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.uint32)

            # Run xatlas parametrization
            vmapping, indices, uvs = xatlas.parametrize(verts, faces)

            # Remap mesh to new topology
            mesh.vertices = verts[vmapping]
            mesh.faces = indices
            if hasattr(mesh, 'vertex_normals'):
                norms = np.array(mesh.vertex_normals, dtype=np.float32)
                mesh.vertex_normals = norms[vmapping]

            # Store UVs (trimesh uses visual.uv)
            try:
                from trimesh.visual import TextureVisuals
                mesh.visual = TextureVisuals(uv=uvs)
            except Exception:
                pass

            # Bake vertex colors to texture if available
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
                self._bake_texture(mesh, vmapping, uvs, output_path)

            logger.info("UV unwrapping complete (%d vertices, %d UVs)", len(mesh.vertices), len(uvs))

        except Exception as exc:
            logger.warning("UV unwrapping failed (non-fatal): %s", exc)

    @staticmethod
    def _bake_texture(mesh, vmapping, uvs, output_path: str, tex_size: int = 1024) -> None:
        """Bake vertex colors into a UV-mapped texture image."""
        try:
            import numpy as np

            vc = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vc = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
            elif hasattr(mesh, '_vertex_colors'):
                vc = np.array(mesh._vertex_colors[:, :3], dtype=np.float32) / 255.0

            if vc is None or len(vc) == 0:
                return

            # Create texture image
            texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)

            # Simple nearest-neighbor bake: for each UV, write the vertex color
            for i, (u, v) in enumerate(uvs):
                px = int(u * (tex_size - 1))
                py = int((1.0 - v) * (tex_size - 1))
                px = max(0, min(tex_size - 1, px))
                py = max(0, min(tex_size - 1, py))
                vi = min(i, len(vc) - 1)
                texture[py, px] = (vc[vi] * 255).clip(0, 255).astype(np.uint8)

            # Save texture alongside the mesh
            tex_path = Path(output_path).with_name(Path(output_path).stem + "_albedo.png")
            Image.fromarray(texture).save(str(tex_path))
            logger.info("Texture baked: %s", tex_path)

        except Exception as exc:
            logger.warning("Texture baking failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release all models and GPU memory."""
        self._birefnet_model = None
        self._birefnet_transform = None
        self._tsr_model = None
        self._loaded_birefnet = False
        self._loaded_tsr = False
        self._cleanup()
        logger.info("TripoSR pipeline unloaded.")

    @staticmethod
    def _cleanup() -> None:
        gc_module.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def transforms_to_pil(tensor, size: tuple[int, int]):
    """Convert a single-channel torch tensor to a PIL Image (L mode)."""
    import numpy as np
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask = Image.fromarray(arr, mode="L")
    return mask.resize(size, Image.LANCZOS)
