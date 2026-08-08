"""LTX-2 model integration package.

The vendored LTX2 code is copied from Lightricks-LTX-2 / DeepBeepMeep Wan2GP.
Many of its modules use absolute imports like ``from shared.utils import ...``
which assume ``shared/`` is on ``sys.path``. We bundle ``shared/`` next to
``ltx2/`` and add this directory to ``sys.path`` at import time so those
imports resolve.
"""
import os
import sys

_LTX2_DIR = os.path.dirname(os.path.abspath(__file__))
if _LTX2_DIR not in sys.path:
    sys.path.insert(0, _LTX2_DIR)
