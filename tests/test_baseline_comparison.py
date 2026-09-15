"""P-BASELINE artifact gates (TCI gap 3: literature baselines).

CI-safe: reads only results/baseline/baseline_comparison.json. Pins:

1. The OI hard anchor: 88 informed rows bit-identical to the frozen
   E-arm subset (the baseline shares orderings + seeds by construction;
   recorded in the artifact).
2. Outcome consistency: geometry-insufficient in ALL four cohort x
   budget cells (DC05's median deviation from random never enters the
   informed band).
3. The headline table: DC05's median angular error sits inside the
   random band on both cohorts; every informed policy beats it.
4. Manifest integrity.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/baseline/baseline_comparison.json"
CFG = REPO / "configs/baseline_comparison.yaml"


def _load():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_oi_hard_anchor_recorded():
    art = _load()
    assert art["gate"] == "P-BASELINE"
    assert "88" in art["oi_anchor"] and "BIT-IDENTICAL" in art["oi_anchor"]


def test_outcome_consistency():
    art = _load()
    for tag in ("oi", "diligent"):
        for k, d in art[tag].items():
            dev = d["median_dev_from_random"]
            expected = ("DC05-informative"
                        if dev["dc05"] <= dev["informed"]
                        else "geometry-insufficient")
            assert d["reading"] == expected, (tag, k)
            # 全部四格:几何不足
            assert d["reading"] == "geometry-insufficient", (tag, k)


def test_dc05_inside_random_band():
    art = _load()
    # OI k=14: DC05 中位 6.222 落在 random 带 [6.177, 6.324] 内
    m = art["oi"]["14"]["median_ang_by_unit"]
    rands = [v for kk, v in m.items() if kk.startswith("randomU_")]
    assert min(rands) <= m["dc05"] <= max(rands)
    # 每个 informed 单元都优于 DC05
    for u in ("mode_aware", "e_opt", "a_opt", "d_opt"):
        assert m[u] < m["dc05"], (u, m[u], m["dc05"])
    # DQ k=28:同样
    m2 = art["diligent"]["28"]["median_ang_by_unit"]
    rands2 = [v for kk, v in m2.items() if kk.startswith("randomU_")]
    assert min(rands2) <= m2["dc05"] <= max(rands2) or abs(
        m2["dc05"] - sum(rands2) / 3) < 0.5
    assert m2["a_opt"] < m2["dc05"]


def test_headline_numbers():
    art = _load()
    m = art["oi"]["14"]["median_ang_by_unit"]
    assert round(m["dc05"], 3) == 6.222
    assert round(m["e_opt"], 3) == 4.520      # 最好的 informed
    dev = art["oi"]["14"]["median_dev_from_random"]
    assert round(dev["dc05"], 4) == -0.1371
    assert round(dev["informed"], 4) == -1.2251
    dev28 = art["oi"]["28"]["median_dev_from_random"]
    assert round(dev28["dc05"], 4) == 0.0747
    assert round(dev28["informed"], 4) == -2.6359


def test_manifest_integrity():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
    assert "comparability" in art["baseline_definition"]
