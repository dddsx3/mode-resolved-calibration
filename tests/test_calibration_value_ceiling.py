"""P-CEILING artifact gates (H2): committed level-64 saturation table.

CI-safe: reads only results/magnitude/calibration_value_ceiling.json (no
raw data). Pins the two empirical claims:

1. **Universal bound holds on the committed rows** — every D <= 1 - 1/kappa
   (the theorem inequality; a violation would be a theorem failure).
2. **The ceiling is met at level 64** — the audit table showed slack
   <= 1e-4 on 5 objects; this guards the "90.0% at kappa=10 is the ceiling
   attained, not empirical saturation" statement.

The theorem itself is bound on the synthetic family in
tests/test_math_foundations.py::test_calibration_value_ceiling.
"""

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/magnitude/calibration_value_ceiling.json"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_committed_rows_satisfy_ceiling():
    """D <= 1 - 1/kappa on every committed row. """
    j = _load()
    assert j["analysis_status"] == "calibration_value_ceiling_v1"
    for r in j["rows"]:
        assert r["D"] <= r["ceiling"] + 1e-9, r


def test_ceiling_met_at_level_64():
    """At level=64 the D(kappa) table saturates to 1-1/kappa (slack <= 1e-4)."""
    j = _load()
    assert j["level"] == 64.0
    assert [float(x) for x in j["kappa_grid"]] == [2, 3, 5, 10, 20, 50, 100]
    rows = j["rows"]
    assert len(rows) == 5 * 7                 # 5 objects x 7 kappa
    nonempty = [r["slack"] for r in rows]
    assert max(nonempty) <= 1e-4, max(nonempty)
    # the headline: kappa=10 row has D within 1e-4 of 0.9
    k10 = [r for r in rows if r["kappa"] == 10]
    assert all(abs(r["D"] - 0.9) <= 1e-4 for r in k10)


def test_manifest_integrity():
    """config hash recorded in the artifact matches the committed config."""
    import hashlib
    j = _load()
    cfg_hash = hashlib.sha256(
        (REPO / "configs/calibration_value_ceiling.yaml").read_bytes()).hexdigest()
    assert j["manifest"]["config_sha256"] == cfg_hash
    assert j["manifest"]["git_sha"]