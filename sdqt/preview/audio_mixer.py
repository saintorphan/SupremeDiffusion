"""Timeline audio mixer — pre-extracts PCM via ffmpeg, mixes in memory.

Architecture: Audio is written AHEAD of video position to compensate for
the audio output pipeline latency (~500ms on PulseAudio/PipeWire via paplay).
Audio writes are non-blocking to prevent stalling the Qt event loop.

The strategy mirrors ffplay/MLT: write audio far enough ahead that by the
time it exits the audio pipeline buffer, the video has caught up to that
position.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSink

from sdqt.widgets.timeline_track import TimelineTrack

logger = logging.getLogger(__name__)

_HAS_PAPLAY = shutil.which("paplay") is not None

# 44100 Hz, 2 channels, int16 = 4 bytes per sample-pair
_RATE = 44100
_CHANNELS = 2
_SAMPLE_BYTES = _CHANNELS * 2  # 4 bytes per stereo sample
_BPS = _RATE * _SAMPLE_BYTES  # bytes per second of PCM

# How far ahead to write audio (seconds).  This must exceed the audio output
# pipeline latency so that written audio reaches the speakers at the same
# time as the corresponding video frame.  paplay/PipeWire typically adds
# ~500ms.  We pre-buffer 800ms to be safe.
_PREBUFFER_SEC = 0.8

# Feed timer interval — how often we check if more audio needs writing.
_FEED_INTERVAL_MS = 40


def _extract_pcm(path: str) -> bytes:
    """Extract full audio as raw s16le PCM via ffmpeg. Returns empty bytes on failure."""
    try:
        # Quick check: has audio?
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=5,
        )
        if not probe.stdout.strip():
            return b""
        result = subprocess.run(
            ["ffmpeg", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
             "-ar", str(_RATE), "-ac", str(_CHANNELS), "-"],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        logger.debug("PCM extraction failed for %s", path, exc_info=True)
    return b""


class TimelineAudioMixer(QObject):
    """Pre-extracts clip audio via ffmpeg, mixes at playback time.

    Key design: audio is written AHEAD of the video position by
    _PREBUFFER_SEC, so that by the time it passes through the audio
    pipeline buffer it arrives at the speakers in sync with video.
    Writes to the audio pipe are non-blocking to avoid stalling the
    Qt event loop.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tracks: list[TimelineTrack] = []
        self._position: float = 0.0  # current video position (set externally)
        self._playing = False
        self._volume: float = 1.0
        self._ext_sync: bool = False  # True when position is driven externally
        self._audio_write_pos: float = 0.0  # how far ahead we've written audio

        # Pre-extracted PCM data per clip path (bounded by _max_cache_bytes)
        self._pcm_cache: dict[str, bytes] = {}
        self._pcm_cache_bytes: int = 0
        self._max_cache_bytes: int = 512 * 1024 * 1024  # 512 MB
        self._pcm_lock = threading.Lock()  # guards _pcm_cache from bg threads
        self._pcm_extracting: set[str] = set()  # paths currently being extracted

        # Output: paplay subprocess (primary) or QAudioSink (fallback)
        self._paplay_proc: subprocess.Popen | None = None
        self._audio_sink: QAudioSink | None = None
        self._io_device = None

        # Feed timer
        self._chunk_timer = QTimer(self)
        self._chunk_timer.setInterval(_FEED_INTERVAL_MS)
        self._chunk_timer.timeout.connect(self._feed_chunk)

    def set_timeline_data(self, tracks: list[TimelineTrack]) -> None:
        clip_count = sum(len(t.clips) for t in tracks)
        logger.info("Audio mixer: set_timeline_data — %d tracks, %d clips", len(tracks), clip_count)
        self._tracks = tracks
        self._ensure_pcm_extracted()

    def _flush_and_resync(self) -> None:
        """Stop the audio sink to flush its buffer, then restart at current position."""
        logger.debug("Audio mixer: flush and resync at pos=%.3f", self._position)
        self._stop_sink()
        self._audio_write_pos = self._position
        self._setup_sink()

    def seek(self, position_sec: float) -> None:
        logger.debug("Audio mixer: seek to %.3fs", position_sec)
        if self._playing and abs(position_sec - self._position) > 0.2:
            self._flush_and_resync()
        self._position = position_sec
        self._audio_write_pos = position_sec

    def start(self) -> None:
        logger.info("Audio mixer: start at pos=%.3f, cached=%d, extracting=%d",
                     self._position, len(self._pcm_cache), len(self._pcm_extracting))
        self._playing = True
        self._audio_write_pos = self._position
        self._ensure_pcm_extracted()
        self._setup_sink()
        # Pre-fill the audio buffer immediately so audio starts on time
        self._prefill()
        self._chunk_timer.start()

    def stop(self) -> None:
        logger.info("Audio mixer: stop")
        self._playing = False
        self._chunk_timer.stop()
        self._stop_sink()

    # -- PCM extraction (background, one-time per clip) --------------------

    def _ensure_pcm_extracted(self) -> None:
        """Kick off background extraction for any clips not yet cached."""
        for track in self._tracks:
            for clip in track.clips:
                if not clip.has_audio:
                    continue
                path = clip.path
                if path in self._pcm_cache or path in self._pcm_extracting:
                    continue
                if not Path(path).is_file():
                    logger.warning("Audio mixer: file not found: %s", path)
                    continue
                logger.info("Audio mixer: starting PCM extraction for %s (track=%s)",
                            Path(path).name, track.name)
                self._pcm_extracting.add(path)
                threading.Thread(
                    target=self._extract_worker, args=(path,), daemon=True
                ).start()

    def _extract_worker(self, path: str) -> None:
        pcm = _extract_pcm(path)
        with self._pcm_lock:
            self._pcm_cache[path] = pcm
            self._pcm_cache_bytes += len(pcm)
            while self._pcm_cache_bytes > self._max_cache_bytes and self._pcm_cache:
                evict_key = next(iter(self._pcm_cache))
                evicted = self._pcm_cache.pop(evict_key)
                self._pcm_cache_bytes -= len(evicted)
        self._pcm_extracting.discard(path)
        if pcm:
            logger.info("Audio mixer: extracted %.1fs PCM from %s (%d bytes)",
                        len(pcm) / _BPS, Path(path).name, len(pcm))

    # -- Output setup (paplay primary, QAudioSink fallback) ----------------

    def _setup_sink(self) -> None:
        self._stop_sink()

        if _HAS_PAPLAY:
            try:
                if os.name == "nt":
                    raise OSError("paplay not available on Windows")
                self._paplay_proc = subprocess.Popen(
                    ["paplay", "--raw",
                     f"--rate={_RATE}", f"--channels={_CHANNELS}",
                     "--format=s16le",
                     "--latency-msec=50",
                     "--stream-name=SupremeDiffusion Timeline"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                # Make stdin non-blocking so writes never stall the event loop
                fd = self._paplay_proc.stdin.fileno()
                import fcntl
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                # Increase pipe buffer to hold more pre-buffered audio (1MB)
                try:
                    fcntl.fcntl(fd, 1031, 1048576)  # F_SETPIPE_SZ
                except OSError:
                    pass  # not fatal, default 64KB still works
                logger.debug("Audio mixer: using paplay (non-blocking)")
                return
            except Exception:
                logger.debug("paplay launch failed", exc_info=True)

        # Fallback: QAudioSink
        try:
            fmt = QAudioFormat()
            fmt.setSampleRate(_RATE)
            fmt.setChannelCount(_CHANNELS)
            fmt.setSampleFormat(QAudioFormat.Int16)
            self._audio_sink = QAudioSink(fmt, self)
            self._audio_sink.setVolume(self._volume)
            self._io_device = self._audio_sink.start()
            logger.debug("Audio mixer: using QAudioSink")
        except Exception:
            logger.warning("Audio mixer: sink setup failed", exc_info=True)

    def _stop_sink(self) -> None:
        if self._paplay_proc is not None:
            try:
                self._paplay_proc.stdin.close()
            except Exception:
                pass
            try:
                self._paplay_proc.kill()
            except Exception:
                pass
            self._paplay_proc = None
        if self._audio_sink is not None:
            try:
                self._audio_sink.stop()
            except Exception:
                pass
            self._audio_sink = None
        self._io_device = None

    # -- Chunk mixing & output ---------------------------------------------

    def _write_pcm(self, data: bytes) -> bool:
        """Write PCM to audio output. Returns True if all data was written."""
        if self._paplay_proc is not None:
            try:
                self._paplay_proc.stdin.write(data)
                self._paplay_proc.stdin.flush()
                return True
            except BlockingIOError:
                # Pipe buffer full — drop this chunk, we're ahead enough
                return False
            except (BrokenPipeError, OSError):
                self._paplay_proc = None
                return False
        elif self._io_device is not None:
            try:
                self._io_device.write(data)
                return True
            except Exception:
                return False
        return False

    def _mix_range(self, write_from: float, chunk_sec: float) -> bytes:
        """Mix all tracks for the given time range, return PCM bytes."""
        chunk_samples = max(1, int(chunk_sec * _RATE))
        mix = np.zeros((chunk_samples, _CHANNELS), dtype=np.float32)

        for track in self._tracks:
            if track.muted:
                continue
            for clip in track.clips:
                if not clip.has_audio or getattr(clip, "muted", False):
                    continue
                with self._pcm_lock:
                    pcm = self._pcm_cache.get(clip.path)
                if not pcm:
                    continue

                clip_start = clip.start_time
                clip_end = clip_start + clip.duration
                if write_from >= clip_end or write_from + chunk_sec <= clip_start:
                    continue

                media_time = write_from - clip_start + clip.media_offset
                byte_offset = int(media_time * _RATE) * _SAMPLE_BYTES
                byte_length = chunk_samples * _SAMPLE_BYTES
                byte_offset = max(0, min(byte_offset, len(pcm)))
                chunk_bytes = pcm[byte_offset:byte_offset + byte_length]

                if not chunk_bytes or len(chunk_bytes) < _SAMPLE_BYTES:
                    continue

                usable = len(chunk_bytes) - (len(chunk_bytes) % _SAMPLE_BYTES)
                samples = np.frombuffer(chunk_bytes[:usable], dtype=np.int16).reshape(-1, _CHANNELS)
                audio = samples.astype(np.float32) / 32768.0

                vol = clip.volume * track.volume
                if clip.fade_in > 0:
                    rel = write_from - clip_start
                    if rel < clip.fade_in:
                        vol *= rel / clip.fade_in
                if clip.fade_out > 0:
                    rel_end = clip_end - write_from
                    if rel_end < clip.fade_out:
                        vol *= rel_end / clip.fade_out

                length = min(len(audio), chunk_samples)
                mix[:length] += audio[:length] * vol

        mix *= self._volume
        np.clip(mix, -1.0, 1.0, out=mix)
        return (mix * 32767).astype(np.int16).tobytes()

    def _prefill(self) -> None:
        """Pre-fill the audio buffer at start so audio is ready when it needs to play."""
        if self._paplay_proc is None and self._io_device is None:
            return
        # Write _PREBUFFER_SEC of audio ahead of current position
        target = self._position + _PREBUFFER_SEC
        chunk_sec = target - self._audio_write_pos
        if chunk_sec <= 0:
            return
        # Write in small chunks to avoid huge allocations
        step = 0.1  # 100ms chunks
        while self._audio_write_pos < target:
            sec = min(step, target - self._audio_write_pos)
            pcm = self._mix_range(self._audio_write_pos, sec)
            if not self._write_pcm(pcm):
                break  # pipe full, stop pre-filling
            self._audio_write_pos += sec

    def _feed_chunk(self) -> None:
        """Keep the audio buffer filled ahead of the video position."""
        if not self._playing:
            return
        if self._paplay_proc is None and self._io_device is None:
            return
        try:
            # Target: always stay _PREBUFFER_SEC ahead of video position
            target = self._position + _PREBUFFER_SEC

            if target <= self._audio_write_pos:
                return  # already buffered far enough ahead

            # Don't write too much at once (cap at 200ms per tick)
            available = target - self._audio_write_pos
            chunk_sec = min(available, 0.2)

            if chunk_sec < 0.005:
                return  # too small, skip

            pcm = self._mix_range(self._audio_write_pos, chunk_sec)
            if self._write_pcm(pcm):
                self._audio_write_pos += chunk_sec
            # If write failed (pipe full), we'll try again next tick
        except Exception:
            logger.debug("Audio feed chunk failed", exc_info=True)

    # -- Volume ------------------------------------------------------------

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        if self._audio_sink is not None:
            self._audio_sink.setVolume(self._volume)
