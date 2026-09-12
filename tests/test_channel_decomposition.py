"""P-CHANNEL-DECOMP artifact gates (H4).

CI-safe: reads only the committed
results/openillumination/channel_decomposition.json (no raw data). Pins:

1. Direction channel <= 2% at every level/object (the claim registered as
   C7); the max over all rows is exactly the recorded value.
2. Intensity-only reproduces the joint dynamic range: median |joint -
   intensity| across objects is small (<= 0.2 pp) at every level, and the
   level-0.5 medians agree to < 0.1 pp.
3. Grid completeness: 11 objects x 3 channels x 8 levels = 264 rows.
4. Manifest integrity (config sha256 matches committed config).
"""

import json
import math
from pathlib import Path

import numpy as np

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/channel_decomposition.json"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_direction_channel_below_2pct():
    """direction channel <= 2% at every level/object. """
    j = _load()
    dir_rows = [r for r in j["rows"] if r["channel"] == "direction"]
    assert all(r["D"] <= 0.02 + 1e-9 for r in dir_rows)
    assert j["display"]["direction_max_pct"] <= 2.0
    assert j["display"]["direction_max_pct"] == \
        round(max(r["D"] for r in dir_rows) * 100, 3)


def test_intensity_reproduces_joint():
    """intensity-only reproduces the joint dynamic range. """
    j = _load()
    rows = j["rows"]
    levels = sorted({r["level"] for r in rows})
    for lv in levels:
        joint = [r["D"] for r in rows if r["channel"] == "joint" and r["level"] == lv]
        intr = [r["D"] for r in rows if r["channel"] == "intensity" and r["level"] == lv]
        med_diff = abs(float(np.median(joint)) - float(np.median(intr)))
        assert med_diff <= 0.002, (lv, med_diff)   # <= 0.2 pp
    # level 0.5 特写
    j05 = [r["D"] for r in rows if r["channel"] == "joint" and r["level"] == 0.5]
    i05 = [r["D"] for r in rows if r["channel"] == "intensity" and r["level"] == 0.5]
    assert abs(float(np.median(j05)) - float(np.median(i05))) < 0.001


def test_grid_complete():
    """11 objects x 3 channels x 8 levels = 264 rows. """
    j = _load()
    assert len(j["rows"]) == 11 * 3 * 8
    assert j["n_objects"] == 11
    assert len(j["channels"]) == 3


def test_manifest_integrity():
    """config hash recorded matches the committed config. """
    import hashlib
    j = _load()
    cfg_hash = hashlib.sha256(
        (REPO / "configs/channel_decomposition.yaml").read_bytes()).hexdigest()
    assert j["manifest"]["config_sha256"] == cfg_hash
    assert j["manifest"]["git_sha"]


import numpy as np  # noqa: E402  (kept at the end to mirror test-file order)