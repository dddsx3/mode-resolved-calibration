"""P-CERT-LEVELS artifact gates (H1): dynamic range as a curve in level.

CI-safe: reads only the committed results/certification/certified_gaps_levels.json
and cross-checks the level-0.5 point against the frozen P-CERT artifact
(no raw data). Pins:

1. The level curve medians match the committed D0 probe exactly
   (0.7 -> 89.85 % across the 11-point grid; 62.87 % at level 0.5).
2. Every row satisfies the universal ceiling D <= 1 - 1/kappa
   (docs/methods.md section 7; a violation would be a theorem failure).
3. The level-0.5 median reproduces the P-CERT headline (62.87 %).
4. Manifest integrity (config sha256 matches the committed config).
"""

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/certification/certified_gaps_levels.json"
PCERT = REPO / "results/certification/certified_gaps.json"

# D0 probe table (verified independently; same functional, kappa=10)
EXPECTED_MEDIAN_PCT = {
    "0.025": 0.7, "0.05": 2.24, "0.1": 7.82, "0.2": 24.49, "0.35": 47.91,
    "0.5": 62.87, "0.75": 75.48, "1.0": 81.2, "2.0": 87.61, "4.0": 89.39,
    "8.0": 89.85,
}


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_level_curve_medians():
    """by_level median curve matches the committed D0 probe. """
    j = _load()
    assert j["analysis_status"] == "certified_gaps_levels_v1"
    bl = j["by_level"]
    for k, exp in EXPECTED_MEDIAN_PCT.items():
        assert abs(bl[k]["median_pct"] - exp) <= 1e-6, (k, bl[k]["median_pct"], exp)


def test_level_grid_complete():
    """All 11 levels x 11 objects present (121 rows)."""
    j = _load()
    levels = sorted({r["level"] for r in j["rows"]})
    assert len(levels) == 11
    assert len(j["rows"]) == 11 * 11
    assert j["n_objects"] == 11


def test_every_row_satisfies_ceiling():
    """D <= 1 - 1/kappa on every row (universal ceiling, methods.md §7)."""
    j = _load()
    kappa = j["kappa"]
    ceiling = 1.0 - 1.0 / kappa
    for r in j["rows"]:
        assert r["D"] <= ceiling + 1e-9, r
    assert j["all_ceiling_bounds_ok"] is True


def test_level_05_reproduces_pcert_headline():
    """Median D at level 0.5 = 62.87 %, matching the frozen P-CERT. """
    j = _load()
    assert abs(j["by_level"]["0.5"]["median_pct"] - 62.87) <= 1e-6
    pcert = json.loads(PCERT.read_text(encoding="utf-8"))
    assert pcert["display"]["dynamic_range_median_pct_across_objects"] == 62.87


def test_manifest_integrity():
    """config hash recorded matches the committed config. """
    import hashlib
    j = _load()
    cfg_hash = hashlib.sha256(
        (REPO / "configs/certified_gaps_levels.yaml").read_bytes()).hexdigest()
    assert j["manifest"]["config_sha256"] == cfg_hash
    assert j["manifest"]["git_sha"]