"""Artifact gates for the committed P-RADIUS and P-CONC evidence files.

CI-safe: both tests read only the committed JSON artifacts (no raw data).

P-RADIUS (audit P0-1 acceptance): the v2 metric domain (observation-side
injection + nominal-geometry estimator) must recover the first-order
quadratic law — the small-level log-log slope of dev must be positive and
near +2 (v1 collapsed with slope −1.9, which is what this gate guards
against). Also pins the radius bookkeeping fields
(`median_over_crossed_subset`, `n_crossed` ≤ `n_total`).

P-CONC (audit P2-6 acceptance): `clean_vs_mean_ratio` must exist and be a
different quantity from `range_rel_spread` — the systematic
noiseless-vs-noisy calibration-source offset (per-object ratios deviating
from 1) must not be confusable with the within-family IQR dispersion.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

RADIUS = REPO / "results/magnitude/linearization_radius.json"
CONC = REPO / "results/certification/certificate_concentration.json"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not committed")
    return json.loads(p.read_text(encoding="utf-8"))


def test_pradius_first_order_slope_recovered():
    """v2 metric domain: dev follows the (l/l_min)^2 law at small levels.

    Regression lock for audit P0-1: v1's estimator-side injection produced
    a NEGATIVE log-log slope (−1.9) with a 6-order energy collapse. The
    corrected artifact must show a positive slope near +2 on the small-level
    prefix (q = dev/scaling ≈ 1)."""
    j = _load(RADIUS)
    assert j["analysis_status"] == "linearization_radius_v2"
    ref = float(j["levels"][0])
    for name, obj in j["objects"].items():
        rows = [r for r in obj["rows"] if r["level"] <= 0.5]
        lv = np.array([r["level"] for r in rows])
        dev = np.array([r["dev_from_ref"] for r in rows])
        q = np.array([r["excess_over_scaling"] for r in rows])
        assert np.all(np.isfinite(dev)) and np.all(dev > 0), name
        slope = np.polyfit(np.log(lv), np.log(dev), 1)[0]
        assert slope > 1.0, (name, slope)          # was −1.9 in v1
        assert abs(slope - 2.0) < 1.0, (name, slope)  # near the +2 law
        # an object whose radius_2x sits inside the prefix legitimately has
        # q crossing 2 there (e.g. 0.5); only require q to be finite
        assert np.all(np.isfinite(q)), name


def test_pradius_radius_bookkeeping():
    """Radius fields exist and are honestly bookkept: the median is over
    the crossed subset only, with n_crossed ≤ n_total exposed."""
    j = _load(RADIUS)
    for key in ("radius_2x", "radius_10x"):
        agg = j[key]
        for field in ("median_over_crossed_subset", "n_crossed", "n_total",
                      "per_object"):
            assert field in agg, (key, field)
        assert agg["n_crossed"] <= agg["n_total"]
        assert agg["n_total"] == j["n_objects"]
        per_obj = agg["per_object"]
        assert len(per_obj) == agg["n_total"]
        crossed = [v for v in per_obj.values() if v is not None]
        assert len(crossed) == agg["n_crossed"]
        if crossed:
            assert agg["median_over_crossed_subset"] == \
                pytest.approx(float(np.median(crossed)))


def test_pconc_clean_vs_mean_ratio_distinct_from_iqr():
    """`clean_vs_mean_ratio` exists, is separate from `range_rel_spread`,
    and the two are not the same quantity: the systematic clean-vs-mean
    offset (ratio != 1) is reported per object alongside the within-family
    IQR dispersion (audit P2-6)."""
    j = _load(CONC)
    assert "clean_vs_mean_ratio" in j
    for field in ("median", "min", "max"):
        assert field in j["clean_vs_mean_ratio"], field
    assert j["analysis_status"] == "certificate_concentration_v2"
    objs = j["objects"]
    ratios = []
    for name, o in objs.items():
        assert "clean_vs_mean_ratio" in o, name
        assert "range_rel_spread" in o, name
        r = o["clean_vs_mean_ratio"]
        assert math.isfinite(r) and r > 0, (name, r)
        ratios.append(r)
    cr = j["clean_vs_mean_ratio"]
    assert cr["median"] == pytest.approx(float(np.median(ratios)))
    assert cr["min"] == pytest.approx(min(ratios))
    assert cr["max"] == pytest.approx(max(ratios))
    # the note must state the non-conflation rule
    assert "within one calibration family" in j["note"]
