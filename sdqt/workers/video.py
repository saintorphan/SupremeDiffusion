"""Worker threads for video processing (trim, concat, chunk, assemble)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from supremediffusion.utils.video import (
    concat_videos_multi,
    extract_single_frame,
    probe_video,
    trim_boundary_frames,
    trim_video_precise,
)

from .base import BaseWorker

logger = logging.getLogger(__name__)


class AcceptEditsWorker(BaseWorker):
    """Stitch longshot output back into the original source, replacing the selected range.

    Result: original[0:start] + longshot + original[end:duration]
    Emits ``finished_ok`` with the final video path string.
    """

    def __init__(
        self,
        source_path: str,
        longshot_path: str,
        range_start: float,
        range_end: float,
        source_duration: float,
        output_dir: Path,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._longshot = longshot_path
        self._start = range_start
        self._end = range_end
        self._duration = source_duration
        self._output_dir = output_dir

    def do_work(self) -> str:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        segments: list[str] = []

        # Head: original[0 : range_start] — frame-accurate trim
        if self._start > 0.1:
            self.progress.emit(0.1, "Trimming head segment...")
            head = str(self._output_dir / "accept_head.mp4")
            trim_video_precise(self._source, head, 0.0, self._start)
            segments.append(head)

        # Middle: longshot output
        segments.append(self._longshot)

        # Tail: original[range_end : duration] — frame-accurate trim
        if self._end < self._duration - 0.1:
            self.progress.emit(0.4, "Trimming tail segment...")
            tail = str(self._output_dir / "accept_tail.mp4")
            trim_video_precise(self._source, tail, self._end, self._duration)
            segments.append(tail)

        if len(segments) == 1:
            final = segments[0]
        else:
            self.progress.emit(0.6, "Concatenating segments...")
            final = str(self._output_dir / "accepted_final.mp4")
            concat_videos_multi(segments, final)

        result = str(self._output_dir / "accepted_final.mp4")
        if final != result:
            shutil.copy2(final, result)

        self.progress.emit(1.0, "Edits accepted")
        return result


class ChunkWorker(BaseWorker):
    """Subdivide a source clip into chunks and extract gapfill frame pairs.

    Emits ``finished_ok`` with a dict::

        {
            "chunks": [{"path": str, "start": float, "end": float}, ...],
            "gapfills": [{"index": int, "frame_a_path": str, "frame_b_path": str,
                          "render_path": "", "status": "pending"}, ...],
        }
    """

    def __init__(
        self,
        source_path: str,
        ls_dir: Path,
        range_start: float,
        range_end: float,
        subdivisions: int,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._ls_dir = ls_dir
        self._start = range_start
        self._end = range_end
        self._subdivs = subdivisions

    def do_work(self) -> dict:
        self._ls_dir.mkdir(parents=True, exist_ok=True)
        n = self._subdivs
        chunk_len = (self._end - self._start) / n

        # Snap boundaries to exact frame times to prevent duration drift
        fps = probe_video(self._source)["fps"]
        boundaries = [round((self._start + i * chunk_len) * fps) / fps for i in range(n + 1)]
        # Pin first/last to the snapped range edges
        boundaries[0] = round(self._start * fps) / fps
        boundaries[-1] = round(self._end * fps) / fps

        # Trim chunks
        chunks = []
        for i in range(n):
            if self.is_aborted:
                raise InterruptedError
            self.progress.emit(i / (2 * n), f"Trimming chunk {i + 1}/{n}")
            path = str(self._ls_dir / f"chunk_{i:03d}.mp4")
            trim_video_precise(self._source, path, boundaries[i], boundaries[i + 1])
            chunks.append({"path": path, "start": boundaries[i], "end": boundaries[i + 1]})

        # Extract gapfill frame pairs
        gapfills = []
        for i in range(n - 1):
            if self.is_aborted:
                raise InterruptedError
            self.progress.emit((n + i) / (2 * n), f"Extracting frames for gapfill {i + 1}/{n - 1}")

            info_i = probe_video(chunks[i]["path"])
            last_frame_num = max(0, info_i["num_frames"] - 1)
            frame_a = extract_single_frame(chunks[i]["path"], last_frame_num)
            frame_a_path = str(self._ls_dir / f"gapfill_{i:03d}_frame_a.png")
            frame_a.save(frame_a_path)

            frame_b = extract_single_frame(chunks[i + 1]["path"], 0)
            frame_b_path = str(self._ls_dir / f"gapfill_{i:03d}_frame_b.png")
            frame_b.save(frame_b_path)

            gapfills.append({
                "index": i,
                "frame_a_path": frame_a_path,
                "frame_b_path": frame_b_path,
                "render_path": "",
                "status": "pending",
            })

        self.progress.emit(1.0, "Chunking complete")
        return {"chunks": chunks, "gapfills": gapfills}


class AssembleWorker(BaseWorker):
    """Concatenate chunks and gapfill renders into a final video.

    Emits ``finished_ok`` with the final video path string.
    """

    def __init__(
        self,
        chunks: list[dict],
        gapfills: list[dict],
        ls_dir: Path,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._chunks = chunks
        self._gapfills = gapfills
        self._ls_dir = ls_dir

    def do_work(self) -> str:
        # Trim duplicate boundary frames from each gapfill render.
        # run_mode2() produces videos whose first frame IS frame_a and last
        # frame IS frame_b, so without trimming each boundary frame appears
        # twice (end of chunk + start of gapfill, and vice versa).
        trimmed_gapfills: dict[int, str] = {}
        for i, gf in enumerate(self._gapfills):
            gf_path = gf.get("render_path", "")
            if not gf_path:
                continue
            if self.is_aborted:
                raise InterruptedError
            self.progress.emit(i / max(len(self._gapfills), 1) * 0.3, f"Trimming gapfill {i + 1} boundaries...")
            trimmed = str(self._ls_dir / f"gapfill_{i:03d}_trimmed.mp4")
            trim_boundary_frames(gf_path, trimmed)
            trimmed_gapfills[i] = trimmed

        # Build segment list: chunk_0, gapfill_0, chunk_1, gapfill_1, ..., chunk_N
        segments = []
        for i, chunk in enumerate(self._chunks):
            segments.append(chunk["path"])
            if i in trimmed_gapfills:
                segments.append(trimmed_gapfills[i])

        if len(segments) < 2:
            raise ValueError("Not enough segments to assemble")

        # Single-pass concatenation — no iterative re-encoding
        self.progress.emit(0.4, "Concatenating all segments...")
        final_path = str(self._ls_dir / "final_longshot.mp4")
        concat_videos_multi(segments, final_path)

        self.progress.emit(1.0, "Assembly complete")
        return final_path
