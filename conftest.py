"""Repository-root path anchor for the test suite.

pytest inserts each test file's directory (``tests/``) into ``sys.path``, but a
few tests import modules that live at the repository root or under
``experiments/`` (e.g. ``experiments.openillumination_allocation``). Putting the
repo root on ``sys.path`` here makes those imports work identically on every
machine and in CI, without depending on the caller's working directory.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
