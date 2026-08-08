"""PyAV-based frame decoder and LRU frame cache for timeline GL preview."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from sdqt.preview import HAS_PYAV

if HAS_PYAV:
    import av

logger = logging.getLogger(__name__)


class ClipDecoder:
    """Wraps av.InputContainer for frame-level decode with seeking."""

    def __init__(self, clip_path: str) -> None:
        self._path = clip_path
        self._container = av.open(clip_path)
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.thread_type = "AUTO"
        self._stream.codec_context.thread_count = 0
        self._fps = float(self._stream.average_rate or self._stream.guessed_rate or 24)
        self._duration = float(self._stream.duration * self._stream.time_base) if self._stream.duration else 0.0
        self._frame_count = int(self._duration * self._fps) if self._duration > 0 else 0
        # For sequential decode tracking
        self._last_decoded_frame = -1
        self._decoder_iter = None
        self._decode_lock = threading.Lock()

    def decode_frame(self, frame_num: int) -> np.ndarray | None:
        """Decode a specific frame number. Returns RGB uint8 HWC array or None.

        Thread-safe: serialized via _decode_lock to prevent concurrent
        seek+decode on the same av.InputContainer (which would segfault).
        """
        if frame_num < 0:
            frame_num = 0
        with self._decode_lock:
            return self._decode_frame_unlocked(frame_num)

    def _decode_frame_unlocked(self, frame_num: int) -> np.ndarray | None:
        try:
            # If this is the next sequential frame, continue iterating
            if self._decoder_iter is not None and frame_num == self._last_decoded_frame + 1:
                return self._decode_next(frame_num)

            # Otherwise seek and decode
            target_pts = int(frame_num / self._fps / self._stream.time_base)
            self._container.seek(target_pts, stream=self._stream, backward=True)
            self._decoder_iter = self._container.decode(video=0)

            # Decode until we reach or pass our target frame
            for frame in self._decoder_iter:
                current_frame = int(float(frame.pts * self._stream.time_base) * self._fps)
                if current_frame >= frame_num:
                    self._last_decoded_frame = frame_num
                    return frame.to_ndarray(format="rgb24")
            return None
        except Exception:
            logger.debug("Failed to decode frame %d from %s", frame_num, self._path, exc_info=True)
            self._decoder_iter = None
            return None

    def _decode_next(self, frame_num: int) -> np.ndarray | None:
        """Decode the next frame from the current iterator position."""
        try:
            frame = next(self._decoder_iter)
            self._last_decoded_frame = frame_num
            return frame.to_ndarray(format="rgb24")
        except (StopIteration, Exception):
            self._decoder_iter = None
            return None

    def close(self) -> None:
        """Close the container."""
        try:
            self._decoder_iter = None
            self._container.close()
        except Exception:
            pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._stream.codec_context.width

    @property
    def height(self) -> int:
        return self._stream.codec_context.height


class FrameCache:
    """Thread-safe LRU cache: (clip_path, frame_num) -> numpy.ndarray.

    Manages ClipDecoder instances per clip path, evicts frames when
    the cache exceeds max_bytes.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._decoders: dict[str, ClipDecoder] = {}
        self._lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()

    def get_or_decode(self, clip_path: str, frame_num: int) -> np.ndarray | None:
        """Return cached frame or decode it. Thread-safe."""
        key = (clip_path, frame_num)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Decode outside the lock to avoid blocking
        decoder = self._get_decoder(clip_path)
        if decoder is None:
            return None
        frame = decoder.decode_frame(frame_num)
        if frame is None:
            return None

        with self._lock:
            # Check again in case another thread decoded it
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = frame
            self._current_bytes += frame.nbytes
            self._evict()
        return frame

    def get_decoder(self, clip_path: str) -> ClipDecoder | None:
        """Get or create a decoder for the given clip path."""
        return self._get_decoder(clip_path)

    def stop_prefetch(self) -> None:
        """Cancel any active prefetch thread."""
        self._prefetch_stop.set()
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0.5)
        self._prefetch_thread = None

    def prefetch_range(self, clip_path: str, start: int, count: int) -> None:
        """Pre-decode a range of frames on a background thread."""
        self.stop_prefetch()
        self._prefetch_stop.clear()

        def _worker():
            for i in range(start, start + count):
                if self._prefetch_stop.is_set():
                    break
                self.get_or_decode(clip_path, i)

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def clear(self) -> None:
        """Clear all cached frames and close all decoders."""
        self._prefetch_stop.set()
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
            for dec in self._decoders.values():
                dec.close()
            self._decoders.clear()

    def release_clip(self, clip_path: str) -> None:
        """Close decoder and evict all frames for a clip."""
        with self._lock:
            if clip_path in self._decoders:
                self._decoders[clip_path].close()
                del self._decoders[clip_path]
            keys_to_remove = [k for k in self._cache if k[0] == clip_path]
            for k in keys_to_remove:
                self._current_bytes -= self._cache[k].nbytes
                del self._cache[k]

    def cleanup_stale_decoders(self, active_paths: set[str]) -> None:
        """Release decoders for clips no longer on the timeline."""
        with self._lock:
            stale = [p for p in self._decoders if p not in active_paths]
        for p in stale:
            self.release_clip(p)

    def _get_decoder(self, clip_path: str) -> ClipDecoder | None:
        """Get or create a decoder, thread-safe."""
        with self._lock:
            if clip_path in self._decoders:
                return self._decoders[clip_path]
        try:
            decoder = ClipDecoder(clip_path)
            with self._lock:
                self._decoders[clip_path] = decoder
            return decoder
        except Exception:
            logger.debug("Failed to open decoder for %s", clip_path, exc_info=True)
            return None

    def _evict(self) -> None:
        """Evict oldest frames until under max_bytes. Must hold lock."""
        while self._current_bytes > self._max_bytes and self._cache:
            _, frame = self._cache.popitem(last=False)
            self._current_bytes -= frame.nbytes
