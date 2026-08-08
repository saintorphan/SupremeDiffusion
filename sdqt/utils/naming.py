"""Derived-filename helpers — cap suffix-accumulated clip names.

Timeline operations derive output names from the source stem plus an
operation suffix (combine, bake, speed, reverse, quick export, ...).
Chained operations compound those suffixes until names hit the 255-byte
filename limit and further edits fail outright. cap_stem() bounds the
stem while keeping the head and tail readable and embedding a short hash
of the full name so capped names stay unique.
"""

from __future__ import annotations

import hashlib

# Leaves room for one more operation suffix, tempfile randomness, and the
# extension below the 255-byte filename limit.
MAX_STEM = 110


def cap_stem(stem: str, max_len: int = MAX_STEM) -> str:
    """Cap a filename stem (no extension) to *max_len* characters.

    Over-long stems become ``<head>~<hash8>~<tail>`` — the tail keeps the
    most recent operation suffixes visible, the hash keeps distinct long
    names from colliding after truncation.
    """
    if len(stem) <= max_len:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8", "replace")).hexdigest()[:8]
    keep = max_len - len(digest) - 2  # two "~" separators
    head = keep // 3
    tail = keep - head
    return f"{stem[:head]}~{digest}~{stem[-tail:]}"
