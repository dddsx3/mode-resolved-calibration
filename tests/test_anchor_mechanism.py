"""P-ANCHOR-MECHANISM artifact gates (acceptance 530f991 section 5;
rule revised per acceptance 533a279 section 3.3).

CI-safe: reads only results/openillumination/anchor_mechanism.json. Pins:

1. Degeneracy marking: 7 of 21 objects carry `degenerate_share`
   (|share| < 0.01 -- the same 0/0 semantics as diligent_queue.json's
   `degenerate_objects`).
2. Both calibers are reported: pooled all-21 (-0.899) and pooled
   valid-sample (-0.688); the OI-internal valid correlation is -0.927
   (n=11) and the DiLiGenT valid subset is not computable (3 objects).
3. Outcome consistency: `oi-internal-only` -- the valid-sample pooled
   correlation is BELOW the 0.7 threshold while the OI-internal one is
   above it; the v1 rule (all-21) is retained as reference only.
4. The statistic's corrected description (weak-subspace loading, not a
   per-light nuisance-energy ratio) is recorded in the artifact note.
5. Source anchors.
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


def test_degeneracy_marking():
    art = _load()
    assert art["degenerate_share_threshold"] == 0.01
    assert art["n_degenerate"] == 7
    degen = sorted(r["object"] for r in art["rows"]
                   if r["degenerate_share"])
    assert degen == ["ballPNG", "bearPNG", "buddhaPNG", "gobletPNG",
                     "harvestPNG", "pot2PNG", "readingPNG"]
    # 非退化的 DiLiGenT 物体:cat(0.221)、pot1(0.295)、cow(-0.028)
    dq_valid = sorted(r["object"] for r in art["rows"]
                      if r["cohort"] == "dq" and not r["degenerate_share"])
    assert dq_valid == ["catPNG", "cowPNG", "pot1PNG"]


def test_both_calibers_reported():
    art = _load()
    c = art["correlations"]["dir_int_weak_ratio"]
    assert c["pooled"] == -0.898701
    assert c["pooled_excl_degenerate"] == -0.687912
    assert c["n_valid"] == 14
    assert c["per_cohort_excl_degenerate"]["oi"] == -0.927273
    assert c["per_cohort_excl_degenerate"]["dq"] is None


def test_outcome_is_oi_internal_only():
    art = _load()
    assert art["outcome"].startswith("oi-internal-only")
    # 有效样本 pooled 低于门限,OI 内部高于门限
    c = art["correlations"]["dir_int_weak_ratio"]
    assert abs(c["pooled_excl_degenerate"]) < 0.7
    assert abs(c["per_cohort_excl_degenerate"]["oi"]) >= 0.7
    # v1 判据(all-21)保留为参照:该口径下 dir_int_weak_ratio 过线
    assert "dir_int_weak_ratio" in         art["mechanism_supported_including_degenerate"]


def test_corrected_description_recorded():
    art = _load()
    note = art["note"]
    assert "weak-subspace" in note or "weak-subspace loading" in note or         "not a per-light" in note


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
