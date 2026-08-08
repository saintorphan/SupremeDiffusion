"""OpenGL-based timeline preview widget with GLSL effect compositing.

Renders decoded video frames through a chain of GLSL shaders matching
the timeline effect stack. Uses ping-pong FBOs for multi-pass rendering.

Three textures:
  - _input_tex: uploaded with the decoded frame (video resolution)
  - _fbo_tex_a / _fbo_tex_b: ping-pong targets (widget resolution)

GL resources are fully destroyed when tabbing away (release_gl_resources)
and lazily re-created on the next paintGL call.
"""

from __future__ import annotations

import ctypes
import logging
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from sdqt.preview import HAS_OPENGL, HAS_PYAV
from sdqt.preview.frame_cache import FrameCache
from sdqt.widgets.timeline_track import TimelineTrack, TrackType

if HAS_OPENGL:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL import GL
    from sdqt.preview.shaders import (
        VERTEX_SHADER, PASSTHROUGH_FRAG, EFFECT_SHADERS,
        COMPOSITE_FRAG,
    )

logger = logging.getLogger(__name__)

# Fullscreen quad: 2 triangles covering NDC [-1,1]
_QUAD_VERTICES = np.array([
    # x,    y,   u,   v
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0, -1.0,  0.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 1.0,
], dtype=np.float32)


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a GLSL shader, raise on error."""
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile error: {log}")
    return shader


def _link_program(vert_src: str, frag_src: str) -> int:
    """Compile and link a shader program, binding attribute locations."""
    vs = _compile_shader(vert_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(frag_src, GL.GL_FRAGMENT_SHADER)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    # Bind attribute locations explicitly (works on compat + core profiles)
    GL.glBindAttribLocation(program, 0, "a_position")
    GL.glBindAttribLocation(program, 1, "a_texcoord")
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program)
        if isinstance(log, bytes):
            log = log.decode(errors="replace")
        GL.glDeleteProgram(program)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        raise RuntimeError(f"Program link error: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return program


def _create_texture(filter_mode=None) -> int:
    """Create a GL texture with standard params. Returns texture ID."""
    if filter_mode is None:
        filter_mode = GL.GL_LINEAR
    tex = int(GL.glGenTextures(1))
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filter_mode)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filter_mode)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    return tex


class _GLPreviewCanvas(QOpenGLWidget if HAS_OPENGL else QWidget):
    """Internal OpenGL canvas for timeline frame rendering with effects.

    Lazily creates GL resources on first paint and after release.
    """

    frame_rendered = Signal()

    def __init__(self, frame_cache: FrameCache, parent=None) -> None:
        super().__init__(parent)
        self._cache = frame_cache
        self._tracks: list[TimelineTrack] = []
        self._position: float = 0.0
        self._frame_num: int = 0

        # GL resources (created lazily, destroyed on release)
        self._gl_ready = False
        self._vao = 0
        self._vbo = 0
        self._input_tex = 0       # decoded frame upload target
        self._fbo_tex_a = 0       # ping-pong texture A
        self._fbo_tex_b = 0       # ping-pong texture B
        self._fbo_a = 0           # FBO for texture A
        self._fbo_b = 0           # FBO for texture B
        self._fbo_width = 0
        self._fbo_height = 0
        self._input_tex_w = 0     # current input texture dimensions
        self._input_tex_h = 0

        # Shader programs: effect_type -> program ID
        self._programs: dict[str, int] = {}
        self._passthrough_prog = 0
        self._composite_prog = 0

        # 3D LUT texture cache: cube_path -> (file mtime_ns, GL texture ID).
        # The mtime is checked on every lookup — Match Grade rewrites .cube
        # files at the same path, so a path-only cache would keep showing the
        # previous run's LUT.
        self._lut3d_textures: dict[str, tuple[int, int]] = {}

        # Flag: set True when hideEvent should destroy resources
        self._released = False

        # Cached last-rendered frame for capture after tab switch
        self._last_frame_image = None  # QImage
        self._last_frame_src_w = 0    # source video width at time of cache
        self._last_frame_src_h = 0    # source video height at time of cache
        self._last_frame_is_tv = True  # active clip color range for TV→PC fix

        self.setMinimumHeight(120)

    @property
    def _default_fbo(self) -> int:
        """Qt's internal FBO — must be queried dynamically, can change between paints."""
        return self.defaultFramebufferObject()

    # -- GL Lifecycle --

    def initializeGL(self) -> None:
        """Called once by Qt on first show. We do the real init in _ensure_gl."""
        self._ensure_gl()

    def _ensure_gl(self) -> None:
        """Create all GL resources if not already created.

        Safe to call multiple times — no-ops if already ready.
        All GL errors are caught to prevent app crashes.
        """
        if self._gl_ready or not HAS_OPENGL:
            return
        try:
            self._do_init_gl()
        except Exception:
            logger.error("GL preview init failed", exc_info=True)
            self._gl_ready = False

    def _do_init_gl(self) -> None:
        """Actual GL initialization. Separated for error handling."""

        version = GL.glGetString(GL.GL_VERSION)
        logger.debug("GL preview init — GL version: %s", version)

        # Clear any stale GL errors from context creation
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        # -- Compile shaders --
        try:
            self._passthrough_prog = _link_program(VERTEX_SHADER, PASSTHROUGH_FRAG)
        except RuntimeError as e:
            logger.error("GL preview: passthrough shader failed: %s", e)
            return

        try:
            self._composite_prog = _link_program(VERTEX_SHADER, COMPOSITE_FRAG)
        except RuntimeError as e:
            logger.warning("GL preview: composite shader failed: %s", e)
            self._composite_prog = self._passthrough_prog

        for effect_type, frag_src in EFFECT_SHADERS.items():
            try:
                self._programs[effect_type] = _link_program(VERTEX_SHADER, frag_src)
            except RuntimeError as e:
                logger.warning("Shader compile failed for %s: %s", effect_type, e)

        # -- Fullscreen quad VAO/VBO --
        self._vao = int(GL.glGenVertexArrays(1))
        GL.glBindVertexArray(self._vao)

        self._vbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, _QUAD_VERTICES.nbytes,
                        _QUAD_VERTICES, GL.GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats * 4 bytes
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE,
                                 stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE,
                                 stride, ctypes.c_void_p(2 * 4))
        GL.glBindVertexArray(0)

        # -- Textures --
        self._input_tex = _create_texture()
        self._fbo_tex_a = _create_texture()
        self._fbo_tex_b = _create_texture()

        # -- FBOs --
        self._fbo_a = int(GL.glGenFramebuffers(1))
        self._fbo_b = int(GL.glGenFramebuffers(1))

        # Allocate FBO textures at widget size
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        self._resize_fbos(w, h)

        self._gl_ready = True
        self._released = False
        logger.debug("GL preview: resources created (%dx%d)", w, h)

    def resizeGL(self, w: int, h: int) -> None:
        if not self._gl_ready or w <= 0 or h <= 0:
            return
        try:
            # Clear any stale GL errors before proceeding
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass
            GL.glViewport(0, 0, w, h)
            self._resize_fbos(w, h)
        except Exception:
            logger.debug("resizeGL error", exc_info=True)

    def _resize_fbos(self, w: int, h: int) -> None:
        """Resize the ping-pong FBO textures to w x h."""
        if w == self._fbo_width and h == self._fbo_height:
            return
        self._fbo_width = w
        self._fbo_height = h

        for tex, fbo in [(self._fbo_tex_a, self._fbo_a),
                         (self._fbo_tex_b, self._fbo_b)]:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                            w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                      GL.GL_TEXTURE_2D, tex, 0)
            status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
            if status != GL.GL_FRAMEBUFFER_COMPLETE:
                logger.warning("FBO incomplete: status=0x%x", status)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)

    # -- 3D LUT texture management --

    def _load_lut3d_texture(self, cube_path: str) -> int:
        """Load or return cached GL_TEXTURE_3D for a .cube LUT file.

        Cached per (path, mtime): if the file was rewritten since the texture
        was uploaded, the stale texture is dropped and the LUT reloads.
        """
        import os
        try:
            mtime = os.stat(cube_path).st_mtime_ns
        except OSError:
            mtime = -1

        cached = self._lut3d_textures.get(cube_path)
        if cached is not None:
            cached_mtime, cached_tex = cached
            if mtime < 0 or mtime == cached_mtime:
                return cached_tex
            # File changed on disk — delete the stale texture and reload.
            # Called from the render pass, so the GL context is current.
            del self._lut3d_textures[cube_path]
            try:
                GL.glDeleteTextures(1, [cached_tex])
            except Exception:
                pass

        from sdqt.utils.grade_compute import parse_cube_file
        try:
            lut_data, size = parse_cube_file(cube_path)
        except Exception as exc:
            logger.warning("Failed to parse LUT %s: %s", cube_path, exc)
            return 0

        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE)
        import numpy as np
        lut_contiguous = np.ascontiguousarray(lut_data)
        GL.glTexImage3D(
            GL.GL_TEXTURE_3D, 0, GL.GL_RGB32F,
            size, size, size, 0,
            GL.GL_RGB, GL.GL_FLOAT, lut_contiguous,
        )
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

        self._lut3d_textures[cube_path] = (mtime, tex)
        logger.info("Loaded 3D LUT texture %s (%dx%dx%d) → GL tex %d",
                     cube_path, size, size, size, tex)
        return tex

    def invalidate_lut3d(self, cube_path: str) -> None:
        """Delete a cached 3D LUT texture so it gets reloaded next frame."""
        _, tex = self._lut3d_textures.pop(cube_path, (0, 0))
        if tex:
            try:
                self.makeCurrent()
                GL.glDeleteTextures(1, [tex])
                self.doneCurrent()
            except Exception:
                pass

    # -- Rendering --

    def paintGL(self) -> None:
        if not HAS_OPENGL:
            return

        # Lazy init / re-init after release
        if not self._gl_ready:
            self._ensure_gl()
        if not self._gl_ready:
            return

        try:
            # Clear stale errors
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass

            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
            GL.glViewport(0, 0, max(self.width(), 1), max(self.height(), 1))
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # Find active clips at current position (bottom-to-top)
            active_clips = []
            for track in reversed(self._tracks):
                if track.track_type != TrackType.VIDEO or not track.visible:
                    continue
                for clip in track.clips:
                    clip_end = clip.start_time + clip.duration
                    if clip.start_time <= self._position < clip_end:
                        active_clips.append((track, clip))
                        break

            if not active_clips:
                if self._tracks:
                    logger.debug("GL paintGL: no active clips at pos=%.3f, "
                                 "tracks=%d, total_clips=%d",
                                 self._position, len(self._tracks),
                                 sum(len(t.clips) for t in self._tracks))
                return

            if len(active_clips) == 1:
                _, clip = active_clips[0]
                self._render_clip_to_screen(clip)
            else:
                self._render_composited(active_clips)

            # Cache rendered frame + source dimensions for capture after tab switch.
            # Also cache whether the active clip is TV-range so the capture path
            # can apply the correct range expansion (fixes the blue-wash bug
            # on Send-frame from the GL preview).
            try:
                self._last_frame_image = self.grabFramebuffer()
                self._last_frame_src_w = self._input_tex_w
                self._last_frame_src_h = self._input_tex_h
                # If multiple clips composited, use the first as the range hint.
                # Mixed-range timelines are rare; the heuristic is safe.
                if active_clips:
                    _, _first_clip = active_clips[0]
                    self._last_frame_is_tv = bool(getattr(_first_clip, "is_tv_range", True))
                else:
                    self._last_frame_is_tv = True
            except Exception:
                self._last_frame_image = None

            self.frame_rendered.emit()
        except Exception:
            logger.debug("paintGL error", exc_info=True)

    def _upload_frame(self, clip) -> bool:
        """Decode and upload clip's current frame to _input_tex.

        Applies rotation (metadata flag) via numpy before upload.
        Returns True on success. Sets _input_tex_w/h.
        """
        clip_time = self._position - clip.start_time + clip.media_offset
        decoder = self._cache.get_decoder(clip.path)
        if decoder is None:
            return False
        frame_num = max(0, int(clip_time * decoder.fps))
        frame_data = self._cache.get_or_decode(clip.path, frame_num)
        if frame_data is None:
            return False

        # Apply rotation metadata (numpy rot90 is cheap on CPU)
        rot = getattr(clip, "rotation", 0) % 360
        if rot == 90:
            frame_data = np.rot90(frame_data, k=3)  # CW 90 = 3x CCW 90
        elif rot == 180:
            frame_data = np.rot90(frame_data, k=2)
        elif rot == 270:
            frame_data = np.rot90(frame_data, k=1)

        # NOTE: no manual range expansion here — PyAV's ``to_ndarray("rgb24")``
        # (in frame_cache) already outputs full-range RGB, correctly honoring
        # each clip's color_range tag (tv → expanded, full → left as-is, and
        # even untagged → swscale's limited-assumed expansion). This was
        # verified to match the ffmpeg frame-extraction path ("send frame" /
        # captures) to the bit across tv, full, and untagged clips. The old
        # ``(x-16)*255/219`` stretch double-expanded already-full-range data,
        # making the GL timeline darker/higher-contrast than the frames it
        # exported — the "timeline color ≠ extracted frame" bug. Don't add it
        # back: expand at decode time (frame_cache), never here.
        h, w = frame_data.shape[:2]
        upload_data = np.ascontiguousarray(frame_data[::-1])  # flip Y for GL

        GL.glBindTexture(GL.GL_TEXTURE_2D, self._input_tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)  # RGB rows aren't 4-byte aligned
        # Re-allocate only if dimensions changed
        if w != self._input_tex_w or h != self._input_tex_h:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8,
                            w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                            upload_data)
            self._input_tex_w = w
            self._input_tex_h = h
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0,
                               w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                               upload_data)
        return True

    def _letterbox_viewport(self) -> tuple[int, int, int, int]:
        """Compute a viewport that preserves the input frame's aspect ratio.

        Returns (x, y, w, h) for glViewport — centered with black bars.
        """
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        fw, fh = self._input_tex_w, self._input_tex_h
        if fw <= 0 or fh <= 0:
            return 0, 0, ww, wh
        frame_ar = fw / fh
        widget_ar = ww / wh
        if frame_ar > widget_ar:
            # Pillarbox (frame wider than widget) — fit to width
            vw = ww
            vh = int(ww / frame_ar)
            vx = 0
            vy = (wh - vh) // 2
        else:
            # Letterbox (frame taller than widget) — fit to height
            vh = wh
            vw = int(wh * frame_ar)
            vx = (ww - vw) // 2
            vy = 0
        return vx, vy, max(vw, 1), max(vh, 1)

    def _render_clip_to_screen(self, clip) -> None:
        """Render a clip through its effects to the default framebuffer (screen)."""
        if not self._upload_frame(clip):
            return

        effects = [fx for fx in clip.effects
                   if fx.get("enabled", True) and fx.get("type") in self._programs]

        vx, vy, vw, vh = self._letterbox_viewport()

        if not effects:
            # No effects: passthrough input directly to screen
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
            GL.glViewport(vx, vy, vw, vh)
            self._draw_quad(self._passthrough_prog, self._input_tex)
            return

        # Ping-pong: input_tex → FBO_A → FBO_B → ... → screen
        read_tex = self._input_tex
        ping_fbo, pong_fbo = self._fbo_a, self._fbo_b
        ping_tex, pong_tex = self._fbo_tex_a, self._fbo_tex_b

        for i, fx in enumerate(effects):
            is_last = (i == len(effects) - 1)
            program = self._programs.get(fx["type"])
            if not program:
                continue

            if is_last:
                # Render to screen with letterbox
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
                GL.glViewport(vx, vy, vw, vh)
            else:
                # Render to ping FBO
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ping_fbo)
                GL.glViewport(0, 0, self._fbo_width, self._fbo_height)

            GL.glUseProgram(program)
            self._set_effect_uniforms(program, fx)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, read_tex)
            loc = GL.glGetUniformLocation(program, "u_texture")
            if loc >= 0:
                GL.glUniform1i(loc, 0)

            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

            if not is_last:
                # Next pass reads from what we just wrote
                read_tex = ping_tex
                # Swap ping/pong for next iteration
                ping_fbo, pong_fbo = pong_fbo, ping_fbo
                ping_tex, pong_tex = pong_tex, ping_tex

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)

    def _render_clip_to_fbo(self, clip, target_fbo: int, target_tex: int) -> bool:
        """Render a clip through its effects into target_fbo. Returns True on success."""
        if not self._upload_frame(clip):
            return False

        effects = [fx for fx in clip.effects
                   if fx.get("enabled", True) and fx.get("type") in self._programs]

        if not effects:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            GL.glViewport(0, 0, self._fbo_width, self._fbo_height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._draw_quad(self._passthrough_prog, self._input_tex)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
            return True

        # For rendering to FBO, we need to use the OTHER FBO as intermediate.
        # We'll use a simplified single-pass approach for FBO targets:
        # if 1 effect: input_tex → target_fbo
        # if 2+ effects: input_tex → temp_fbo → ... → target_fbo
        # Use the non-target FBO as the temp.
        if target_fbo == self._fbo_a:
            temp_fbo, temp_tex = self._fbo_b, self._fbo_tex_b
        else:
            temp_fbo, temp_tex = self._fbo_a, self._fbo_tex_a

        read_tex = self._input_tex
        for i, fx in enumerate(effects):
            is_last = (i == len(effects) - 1)
            program = self._programs.get(fx["type"])
            if not program:
                continue

            write_fbo = target_fbo if is_last else temp_fbo
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, write_fbo)
            GL.glViewport(0, 0, self._fbo_width, self._fbo_height)
            if i == 0 or is_last:
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            GL.glUseProgram(program)
            self._set_effect_uniforms(program, fx)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, read_tex)
            loc = GL.glGetUniformLocation(program, "u_texture")
            if loc >= 0:
                GL.glUniform1i(loc, 0)

            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glBindVertexArray(0)

            if not is_last:
                read_tex = temp_tex

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
        return True

    def _render_composited(self, active_clips: list) -> None:
        """Render multiple clips and composite bottom-to-top."""
        # Bottom clip renders directly to screen
        _, clip0 = active_clips[0]
        self._render_clip_to_screen(clip0)

        if len(active_clips) <= 1:
            return

        # Each additional clip: render to FBO A, then blend onto screen
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        for track, clip in active_clips[1:]:
            if not self._render_clip_to_fbo(clip, self._fbo_a, self._fbo_tex_a):
                continue

            # Compute fade alpha
            alpha = 1.0
            clip_time_rel = self._position - clip.start_time
            if clip.fade_in > 0 and clip_time_rel < clip.fade_in:
                alpha *= clip_time_rel / clip.fade_in
            clip_end_rel = clip.duration - clip_time_rel
            if clip.fade_out > 0 and clip_end_rel < clip.fade_out:
                alpha *= clip_end_rel / clip.fade_out

            # Draw overlay on screen with alpha blending (letterboxed)
            vx, vy, vw, vh = self._letterbox_viewport()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
            GL.glViewport(vx, vy, vw, vh)

            # Use passthrough, modulating via blend
            # TODO: use composite shader with u_alpha uniform for proper fade
            self._draw_quad(self._passthrough_prog, self._fbo_tex_a)

        GL.glDisable(GL.GL_BLEND)

    # -- Export rendering (offscreen at arbitrary resolution) --

    def render_frame_for_export(
        self, position: float, width: int, height: int,
    ) -> bytes | None:
        """Render a frame at *position* into an offscreen FBO at *width* x *height*.

        Returns raw RGB bytes (width * height * 3) or None on failure.
        Must be called from the main/GL thread.
        """
        if not self._gl_ready:
            self._ensure_gl()
        if not self._gl_ready:
            return None

        try:
            # Make GL context current for offscreen rendering
            self.makeCurrent()

            # Clear stale errors
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass

            # Seek to position
            self._position = position

            # Resize FBOs to export resolution (will be restored later)
            old_fbo_w, old_fbo_h = self._fbo_width, self._fbo_height
            self._resize_fbos(width, height)

            # Create a dedicated export FBO + texture
            export_tex = _create_texture()
            GL.glBindTexture(GL.GL_TEXTURE_2D, export_tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8,
                width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None,
            )
            export_fbo = int(GL.glGenFramebuffers(1))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, export_fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                GL.GL_TEXTURE_2D, export_tex, 0,
            )

            GL.glViewport(0, 0, width, height)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            # Find active clips at this position
            active_clips = []
            for track in reversed(self._tracks):
                if track.track_type != TrackType.VIDEO or not track.visible:
                    continue
                for clip in track.clips:
                    clip_end = clip.start_time + clip.duration
                    if clip.start_time <= position < clip_end:
                        active_clips.append((track, clip))
                        break

            if active_clips:
                if len(active_clips) == 1:
                    _, clip = active_clips[0]
                    # Render to export FBO instead of screen
                    if not self._upload_frame(clip):
                        self._cleanup_export_fbo(export_fbo, export_tex, old_fbo_w, old_fbo_h)
                        self.doneCurrent()
                        return None

                    effects = [fx for fx in clip.effects
                               if fx.get("enabled", True) and fx.get("type") in self._programs]

                    if not effects:
                        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, export_fbo)
                        GL.glViewport(0, 0, width, height)
                        self._draw_quad(self._passthrough_prog, self._input_tex)
                    else:
                        read_tex = self._input_tex
                        ping_fbo, pong_fbo = self._fbo_a, self._fbo_b
                        ping_tex, pong_tex = self._fbo_tex_a, self._fbo_tex_b

                        for i, fx in enumerate(effects):
                            is_last = (i == len(effects) - 1)
                            program = self._programs.get(fx["type"])
                            if not program:
                                continue

                            write_fbo = export_fbo if is_last else ping_fbo
                            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, write_fbo)
                            GL.glViewport(0, 0, width, height)

                            GL.glUseProgram(program)
                            self._set_effect_uniforms(program, fx)
                            GL.glActiveTexture(GL.GL_TEXTURE0)
                            GL.glBindTexture(GL.GL_TEXTURE_2D, read_tex)
                            loc = GL.glGetUniformLocation(program, "u_texture")
                            if loc >= 0:
                                GL.glUniform1i(loc, 0)

                            GL.glBindVertexArray(self._vao)
                            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
                            GL.glBindVertexArray(0)

                            if not is_last:
                                read_tex = ping_tex
                                ping_fbo, pong_fbo = pong_fbo, ping_fbo
                                ping_tex, pong_tex = pong_tex, ping_tex
                else:
                    # Multi-clip compositing to export FBO
                    _, clip0 = active_clips[0]
                    self._render_clip_to_fbo(clip0, export_fbo, export_tex)

                    if len(active_clips) > 1:
                        GL.glEnable(GL.GL_BLEND)
                        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

                        for track, clip in active_clips[1:]:
                            if not self._render_clip_to_fbo(clip, self._fbo_a, self._fbo_tex_a):
                                continue
                            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, export_fbo)
                            GL.glViewport(0, 0, width, height)
                            self._draw_quad(self._passthrough_prog, self._fbo_tex_a)

                        GL.glDisable(GL.GL_BLEND)

            # Read back pixels
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, export_fbo)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

            # OpenGL reads bottom-up, flip vertically
            row_bytes = width * 3
            frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            frame = np.flipud(frame).copy()

            self._cleanup_export_fbo(export_fbo, export_tex, old_fbo_w, old_fbo_h)
            self.doneCurrent()
            return frame.tobytes()

        except Exception:
            logger.error("GL export frame failed", exc_info=True)
            try:
                self.doneCurrent()
            except Exception:
                pass
            return None

    def _cleanup_export_fbo(
        self, fbo: int, tex: int, restore_w: int, restore_h: int,
    ) -> None:
        """Delete export FBO/texture and restore FBO sizes."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._default_fbo)
        GL.glDeleteFramebuffers(1, [fbo])
        GL.glDeleteTextures(1, [tex])
        if restore_w > 0 and restore_h > 0:
            self._resize_fbos(restore_w, restore_h)

    def _draw_quad(self, program: int, texture: int) -> None:
        """Draw a fullscreen quad with the given program and texture."""
        GL.glUseProgram(program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        loc = GL.glGetUniformLocation(program, "u_texture")
        if loc >= 0:
            GL.glUniform1i(loc, 0)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def _set_effect_uniforms(self, program: int, fx: dict) -> None:
        """Set shader uniforms from effect params."""
        params = fx.get("params", {})
        fx_type = fx["type"]

        def _set(name: str, value):
            loc = GL.glGetUniformLocation(program, name)
            if loc < 0:
                return
            if isinstance(value, (list, tuple)):
                if len(value) == 2:
                    GL.glUniform2f(loc, float(value[0]), float(value[1]))
                elif len(value) == 3:
                    GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))
                elif len(value) == 4:
                    GL.glUniform4f(loc, float(value[0]), float(value[1]),
                                   float(value[2]), float(value[3]))
            else:
                GL.glUniform1f(loc, float(value))

        # Texel size for convolution shaders
        if fx_type in ("sharpen", "blur", "denoise"):
            tw = max(self._input_tex_w, 1)
            th = max(self._input_tex_h, 1)
            _set("u_texel_size", (1.0 / tw, 1.0 / th))

        if fx_type == "hue_sat":
            _set("u_hue_shift", params.get("hue_shift", 0.0))
            _set("u_saturation", params.get("saturation", 1.0))
        elif fx_type == "brightness_contrast":
            _set("u_brightness", params.get("brightness", 0.0))
            _set("u_contrast", params.get("contrast", 1.0))
        elif fx_type == "levels":
            _set("u_black_point", params.get("black_point", 0.0))
            _set("u_white_point", params.get("white_point", 255.0))
            _set("u_gamma", params.get("gamma", 1.0))
        elif fx_type == "shadows_highlights":
            _set("u_shadows", params.get("shadows", 0.0))
            _set("u_highlights", params.get("highlights", 0.0))
        elif fx_type == "color_balance":
            _set("u_shadow_rgb", (
                params.get("shadow_r", 0.0),
                params.get("shadow_g", 0.0),
                params.get("shadow_b", 0.0),
            ))
            _set("u_midtone_rgb", (
                params.get("midtone_r", 0.0),
                params.get("midtone_g", 0.0),
                params.get("midtone_b", 0.0),
            ))
            _set("u_highlight_rgb", (
                params.get("highlight_r", 0.0),
                params.get("highlight_g", 0.0),
                params.get("highlight_b", 0.0),
            ))
        elif fx_type == "channel_mixer":
            _set("u_rr", params.get("rr", 1.0))
            _set("u_gg", params.get("gg", 1.0))
            _set("u_bb", params.get("bb", 1.0))
        elif fx_type == "color_temperature":
            _set("u_temperature", params.get("temperature", 0.0))
            _set("u_tint", params.get("tint", 0.0))
        elif fx_type == "vibrance":
            _set("u_vibrance", params.get("vibrance", 1.0))
        elif fx_type == "sharpen":
            _set("u_amount", params.get("amount", 0.0))
        elif fx_type == "blur":
            _set("u_amount", params.get("amount", 0.0))
        elif fx_type == "denoise":
            _set("u_strength", params.get("strength", 0.0))
        elif fx_type == "vignette":
            _set("u_intensity", params.get("intensity", 0.0))
        elif fx_type == "film_grain":
            _set("u_intensity", params.get("intensity", 0.0))
            _set("u_time", float(self._frame_num))
        elif fx_type == "crop_zoom":
            _set("u_crop_rect", (
                params.get("x", 0.0),
                params.get("y", 0.0),
                params.get("w", 1.0),
                params.get("h", 1.0),
            ))
        elif fx_type == "lut3d":
            lut_file = params.get("lut_file", "")
            if lut_file:
                lut_tex = self._load_lut3d_texture(lut_file)
                if lut_tex:
                    GL.glActiveTexture(GL.GL_TEXTURE1)
                    GL.glBindTexture(GL.GL_TEXTURE_3D, lut_tex)
                    loc = GL.glGetUniformLocation(program, "u_lut3d")
                    if loc >= 0:
                        GL.glUniform1i(loc, 1)
                    GL.glActiveTexture(GL.GL_TEXTURE0)  # restore
            _set("u_lut3d_strength", params.get("strength", 1.0))

    # -- Public API --

    def set_timeline_data(self, tracks: list[TimelineTrack]) -> None:
        self._tracks = tracks

    def seek(self, position_sec: float) -> None:
        """Seek to a position and schedule a repaint."""
        self._position = max(0.0, position_sec)
        self.update()  # triggers paintGL (which calls _ensure_gl if needed)

    def set_frame_num(self, n: int) -> None:
        self._frame_num = n

    def capture_frame(self) -> str | None:
        """Return the last rendered frame as a PNG at source video resolution.

        Crops letterbox bars from the cached QImage, then scales to the
        source video resolution. Works even after GL release.
        """
        if self._last_frame_image is None:
            return None
        try:
            from PySide6.QtCore import Qt as _Qt
            from PySide6.QtGui import QImage
            img = self._last_frame_image
            # Convert to RGB (strip alpha)
            if img.format() != QImage.Format_RGB888:
                img = img.convertToFormat(QImage.Format_RGB888)

            src_w = self._last_frame_src_w or self._input_tex_w
            src_h = self._last_frame_src_h or self._input_tex_h

            # Crop letterbox bars by computing the content viewport
            if src_w > 0 and src_h > 0:
                ww, wh = img.width(), img.height()
                frame_ar = src_w / src_h
                widget_ar = ww / wh
                if frame_ar > widget_ar:
                    vw = ww
                    vh = int(ww / frame_ar)
                    vx = 0
                    vy = (wh - vh) // 2
                else:
                    vh = wh
                    vw = int(wh * frame_ar)
                    vx = (ww - vw) // 2
                    vy = 0
                # Crop to content area (remove black bars)
                img = img.copy(vx, vy, vw, vh)
                # Scale to exact source resolution
                src_w -= src_w % 2
                src_h -= src_h % 2
                if src_w > 0 and src_h > 0:
                    img = img.scaled(src_w, src_h, _Qt.IgnoreAspectRatio, _Qt.SmoothTransformation)

            # NOTE: TV→PC range expansion is now applied at upload time in
            # _upload_frame(), so the framebuffer here is already full-range
            # RGB. No additional expansion needed at capture time. (Previous
            # version did capture-side expansion; removed to avoid double-
            # correcting now that the upload path handles it.)

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="gl_cap_")
            tmp.close()
            img.save(tmp.name, "PNG")
            if Path(tmp.name).stat().st_size > 100:
                return tmp.name
            return None
        except Exception:
            logger.warning("GL frame capture failed", exc_info=True)
            return None

    # -- VRAM Lifecycle --

    def release_gl_resources(self) -> None:
        """Destroy all GL resources to free VRAM."""
        if not self._gl_ready:
            return
        self._released = True
        try:
            self.makeCurrent()
            # Clear stale errors before cleanup
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass

            for prog in self._programs.values():
                if prog:
                    GL.glDeleteProgram(prog)
            self._programs.clear()

            for prog in (self._passthrough_prog, self._composite_prog):
                if prog:
                    GL.glDeleteProgram(prog)
            self._passthrough_prog = 0
            self._composite_prog = 0

            textures = [t for t in (self._input_tex, self._fbo_tex_a, self._fbo_tex_b) if t]
            if textures:
                GL.glDeleteTextures(textures)
            self._input_tex = 0
            self._fbo_tex_a = 0
            self._fbo_tex_b = 0
            self._input_tex_w = 0
            self._input_tex_h = 0

            # Clean up 3D LUT textures
            lut_textures = [tex for _, tex in self._lut3d_textures.values()]
            if lut_textures:
                GL.glDeleteTextures(lut_textures)
            self._lut3d_textures.clear()

            fbos = [f for f in (self._fbo_a, self._fbo_b) if f]
            if fbos:
                for fbo in fbos:
                    GL.glDeleteFramebuffers(1, [fbo])
            self._fbo_a = 0
            self._fbo_b = 0

            if self._vao:
                GL.glDeleteVertexArrays(1, [self._vao])
                self._vao = 0
            if self._vbo:
                GL.glDeleteBuffers(1, [self._vbo])
                self._vbo = 0

            self.doneCurrent()
        except Exception:
            logger.debug("GL resource release error", exc_info=True)

        self._gl_ready = False
        self._fbo_width = 0
        self._fbo_height = 0
        logger.debug("GL preview: resources released")

    def restore_gl_resources(self) -> None:
        """Schedule re-creation of GL resources on next paint."""
        self._released = False
        # _ensure_gl will be called in paintGL
        self.update()


class TimelineGLPreview(QWidget):
    """High-level GL preview widget for the timeline.

    Wraps _GLPreviewCanvas with fallback if GL/PyAV unavailable.
    Provides play/pause/seek API matching VideoPlayerWidget interface subset.
    """

    position_changed = Signal(float)
    frame_captured = Signal(str)

    def __init__(self, frame_cache: FrameCache, parent=None) -> None:
        super().__init__(parent)
        self._cache = frame_cache
        self._canvas: _GLPreviewCanvas | None = None
        self._playing = False
        self._position = 0.0
        self._duration = 0.0
        self._fps = 25.0
        self._frame_num = 0

        # Playback timer
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / self._fps))
        self._play_timer.timeout.connect(self._on_play_tick)

        # Audio mixer (lazy init)
        self._audio_mixer = None
        self._master_volume: float = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if HAS_OPENGL and HAS_PYAV:
            self._canvas = _GLPreviewCanvas(frame_cache)
            self._canvas.setStyleSheet("background: black;")
            layout.addWidget(self._canvas, 1)
        else:
            missing = []
            if not HAS_PYAV:
                missing.append("PyAV (pip install av)")
            if not HAS_OPENGL:
                missing.append("PyOpenGL (pip install PyOpenGL)")
            lbl = QLabel(f"GL Preview unavailable.\nInstall: {', '.join(missing)}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #999; font-size: 13px; background: black;")
            layout.addWidget(lbl, 1)

    @property
    def available(self) -> bool:
        return self._canvas is not None

    def set_timeline_data(self, tracks: list[TimelineTrack]) -> None:
        """Update the track data for rendering."""
        if self._canvas:
            self._canvas.set_timeline_data(tracks)
        # Calculate total duration
        dur = 0.0
        for t in tracks:
            for c in t.clips:
                dur = max(dur, c.start_time + c.duration)
        self._duration = dur
        # Update audio mixer (create early so PCM extraction starts now)
        self._ensure_audio_mixer()
        if self._audio_mixer:
            self._audio_mixer.set_timeline_data(tracks)

    def seek(self, position_sec: float) -> None:
        """Decode and render frame at position. Instant."""
        self._position = max(0.0, min(position_sec, self._duration)) if self._duration > 0 else 0.0
        if self._canvas:
            self._canvas.seek(self._position)
        self.position_changed.emit(self._position)

    def play(self) -> None:
        """Start playback from current position."""
        if self._playing or not self._canvas:
            return
        logger.info("GL preview: play at pos=%.3f fps=%.0f dur=%.1f",
                     self._position, self._fps, self._duration)
        self._playing = True
        self._ensure_audio_mixer()
        if self._audio_mixer:
            self._audio_mixer.seek(self._position)
            self._audio_mixer.start()
        if self._cache:
            self._cache.stop_prefetch()
        self._play_timer.start()

    def pause(self) -> None:
        """Pause playback — instant, no blocking."""
        logger.debug("GL preview: pause at pos=%.3f", self._position)
        self._playing = False
        self._play_timer.stop()
        if self._audio_mixer:
            self._audio_mixer.stop()

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self.pause()
        self._position = 0.0

    def is_playing(self) -> bool:
        return self._playing

    def capture_frame(self) -> str | None:
        """Capture current GL frame as PNG."""
        if self._canvas:
            path = self._canvas.capture_frame()
            if path:
                self.frame_captured.emit(path)
            return path
        return None

    def render_frame_for_export(
        self, position: float, width: int, height: int,
    ) -> bytes | None:
        """Render a frame at full resolution for export. Main-thread only."""
        if self._canvas:
            return self._canvas.render_frame_for_export(position, width, height)
        return None

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def fps(self) -> float:
        return self._fps

    def release_gl_resources(self) -> None:
        """Free all VRAM. Called when tabbing away."""
        self.pause()
        if self._canvas:
            self._canvas.release_gl_resources()

    def restore_gl_resources(self) -> None:
        """Restore GL resources. Called when tabbing back."""
        if self._canvas:
            self._canvas.restore_gl_resources()

    # -- Internal --

    def _on_play_tick(self) -> None:
        """Advance playhead by one frame interval."""
        dt = 1.0 / self._fps
        self._position += dt
        self._frame_num += 1

        if self._position >= self._duration:
            self.pause()
            self._position = max(0.0, self._duration)
            self.position_changed.emit(self._position)
            return

        if self._canvas:
            self._canvas.set_frame_num(self._frame_num)
            self._canvas.seek(self._position)
        # Keep audio mixer in sync with video position
        if self._audio_mixer and self._audio_mixer._playing:
            self._audio_mixer._position = self._position
        self.position_changed.emit(self._position)

    def _ensure_audio_mixer(self) -> None:
        """Lazy-init the audio mixer."""
        if self._audio_mixer is not None:
            return
        try:
            from sdqt.preview.audio_mixer import TimelineAudioMixer
            self._audio_mixer = TimelineAudioMixer(parent=self)
            self._audio_mixer._ext_sync = True  # position driven by _on_play_tick
            self._audio_mixer.set_volume(self._master_volume)
            if self._canvas:
                self._audio_mixer.set_timeline_data(self._canvas._tracks)
        except Exception:
            logger.debug("Audio mixer init failed", exc_info=True)

    def _start_prefetch(self) -> None:
        """Prefetch upcoming frames for smooth playback."""
        if not self._canvas:
            return
        for track in self._canvas._tracks:
            if track.track_type != TrackType.VIDEO:
                continue
            for clip in track.clips:
                clip_end = clip.start_time + clip.duration
                if clip.start_time <= self._position < clip_end:
                    decoder = self._cache.get_decoder(clip.path)
                    if decoder:
                        clip_time = self._position - clip.start_time + clip.media_offset
                        start_frame = int(clip_time * decoder.fps)
                        self._cache.prefetch_range(clip.path, start_frame, 50)

    def set_fps(self, fps: float) -> None:
        """Update playback frame rate."""
        self._fps = max(1.0, fps)
        self._play_timer.setInterval(int(1000 / self._fps))
        logger.debug("GL preview: FPS set to %.0f (interval=%dms)", self._fps, self._play_timer.interval())

    def set_volume(self, volume: float) -> None:
        """Set audio volume."""
        self._master_volume = volume
        if self._audio_mixer:
            self._audio_mixer.set_volume(volume)

    def hideEvent(self, event) -> None:
        """Auto-release GL resources and stop audio when widget is hidden."""
        self.release_gl_resources()
        if self._audio_mixer:
            self._audio_mixer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        """Restore GL resources when widget becomes visible again."""
        self.restore_gl_resources()
        super().showEvent(event)
