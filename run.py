#!/usr/bin/env python3
"""Launch Supreme Diffusion Qt."""
import sys
from pathlib import Path

# Optional external model repos (these are NOT bundled)
for _name in ("LatentSync", "OVI", "Wan2GP", "MuseTalk"):
    _p = Path(f"~/{_name}").expanduser()
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sdqt.app import main

if __name__ == "__main__":
    main()
