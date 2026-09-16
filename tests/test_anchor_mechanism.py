"""P-ANCHOR-MECHANISM artifact gates (acceptance 530f991 section 5).

CI-safe: reads only results/openillumination/anchor_mechanism.json.
Pins:

1. Preregistered rule self-consistency: mechanism-supported iff some
   statistic has |pooled Spearman| >= 0.7 AND a consistent per-cohort
   sign; the measured outcome is mechanism-supported.
2. The mechanism statistic: dir_int_weak_ratio pooled -0.899 with
   per-cohort -0.927 (OI) / -1.000 (DiLiGenT) -- the direction share is
   monotone-decreasing in the direction-nuisance / intensity-nuisance
   weak-subspace energy ratio on BOTH cohorts independently.
3. The secondary statistic (finf_spread, -0.868) and the non-supporting
   ones (light_spread_deg / light_isotropy / gauge_cos: inconsistent
   sign) recorded as such.
4. The 21 object rows carry both cohorts' shares read from the committed
   artifacts (source sha256 anchors).
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/anchor_mechanism.json"
CFG = REPO / "configs/anchor_mechanism.yaml"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_rule_self_consistency():
    art = _load()
    supported = sorted(sn for sn, c in art["correlations"].items()
                       if abs(c["pooled"]) >= 0.7 and c["consistent_sign"])
    assert art["mechanism_supported"] == supported
    expected = "mechanism-supported" if supported else "no-single-statistic"
    assert art["outcome"] == expected
    assert art["outcome"] == "mechanism-supported"
    assert len(art["rows"]) == 21


def test_mechanism_statistic_pinned():
    art = _load()
    c = art["correlations"]["dir_int_weak_ratio"]
    assert c["pooled"] == -0.898701
    assert c["per_cohort"]["oi"] == -0.927273
    assert c["per_cohort"]["dq"] == -1.0
    assert c["consistent_sign"] is True
    # DiLiGenT 内是完美秩相关(10 物体)
    assert abs(c["per_cohort"]["dq"]) == 1.0


def test_secondary_and_non_supporting():
    art = _load()
    assert art["correlations"]["finf_spread"]["pooled"] == -0.867532
    assert art["correlations"]["finf_spread"]["consistent_sign"] is True
    for sn in ("light_spread_deg", "light_isotropy", "gauge_cos"):
        assert art["correlations"][sn]["consistent_sign"] is False, sn
        assert sn not in art["mechanism_supported"], sn


def test_sources_anchored():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert art["manifest"]["oi_shares_sha256"] == hashlib.sha256(
        (REPO / "results/openillumination/ball_anchor.json")
        .read_bytes()).hexdigest()
    assert art["manifest"]["diligent_shares_sha256"] == hashlib.sha256(
        (REPO / "results/diligent/diligent_queue.json")
        .read_bytes()).hexdigest()
