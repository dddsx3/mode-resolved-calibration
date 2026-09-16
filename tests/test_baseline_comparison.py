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
    """v1.1 判读(相对 randomA48,active 同预算口径):规则自洽 + 实测结局。

    OI:geometry-insufficient(dc05_active 有增益但 informed 显著更好);
    DiLiGenT:geometry-informative —— 因为 a_opt 在 DQ 上不再优于
    active-random(其 dev 为正),不是几何变强了。
    """
    art = _load()
    for tag in ("oi", "diligent"):
        for k, d in art[tag].items():
            dev = d["median_dev_from"]
            expected = ("geometry-insufficient"
                        if dev["dc05_active_vs_randomA48"]
                        >= dev["informed_vs_randomA48"]
                        else "geometry-informative")
            assert d["reading"] == expected, (tag, k)
    assert art["oi"]["14"]["reading"] == "geometry-insufficient"
    assert art["oi"]["28"]["reading"] == "geometry-insufficient"
    assert art["diligent"]["14"]["reading"] == "geometry-informative"
    assert art["diligent"]["28"]["reading"] == "geometry-informative"


def test_dq_informed_loses_to_active_random():
    """DQ 上的真发现:a_opt 相对 randomA48 的中位偏差为正(k14 +0.242,
    k28 +0.474)——预测模型的逐灯排序在 DiLiGenT 上不优于 active-random。
    与 E 臂 het 发现同族(排序有效性是条件性的)。"""
    art = _load()
    for k, want in (("14", 0.242046), ("28", 0.473932)):
        dev = art["diligent"][k]["median_dev_from"]
        assert round(dev["informed_vs_randomA48"], 6) == want, k
        assert dev["informed_vs_randomA48"] > 0, k
    # v1 混淆自文档化:all-pool dc05 的 active 重合度远低于预算
    ov14 = art["oi"]["14"]["dc05_active_overlap_at_k"]
    assert max(ov14.values()) <= 9 and min(ov14.values()) >= 2


def test_dc05_active_between_random_and_informed_oi():
    """OI v1.1 核心读法:dc05_active 优于 randomA48(几何有增益)但
   劣于每个 informed 策略(校准模型仍显著更好)。"""
    art = _load()
    for k in ("14", "28"):
        m = art["oi"][k]["median_ang_by_unit"]
        rA = [v for u, v in m.items() if u.startswith("randomA48_")]
        assert m["dc05_active"] < max(rA), k          # 优于最差 active-random
        for u in ("mode_aware", "e_opt", "a_opt", "d_opt"):
            assert m[u] < m["dc05_active"], (k, u)
    # v1(all-pool dc05)确实落在 universe-random 带内(v1 混淆的复现)
    m = art["oi"]["14"]["median_ang_by_unit"]
    rU = [v for u, v in m.items() if u.startswith("randomU_")]
    assert min(rU) <= m["dc05"] <= max(rU)


def test_headline_numbers():
    art = _load()
    m = art["oi"]["14"]["median_ang_by_unit"]
    assert round(m["dc05"], 3) == 6.222
    assert round(m["dc05_active"], 3) == 5.235
    assert round(m["e_opt"], 3) == 4.520      # 最好的 informed
    dev = art["oi"]["14"]["median_dev_from"]
    assert round(dev["dc05_active_vs_randomA48"], 4) == -0.4379
    assert round(dev["informed_vs_randomA48"], 4) == -0.7041
    assert round(dev["dc05_vs_randomU"], 4) == -0.1371
    dev28 = art["oi"]["28"]["median_dev_from"]
    assert round(dev28["dc05_active_vs_randomA48"], 4) == -0.615
    assert round(dev28["informed_vs_randomA48"], 4) == -1.2042
    # DQ:dc05_active 与 randomA48 打平(k28 中位 6.686 vs 带 [6.710, 6.934])
    m2 = art["diligent"]["28"]["median_ang_by_unit"]
    assert round(m2["dc05_active"], 3) == 6.686
    assert round(m2["a_opt"], 3) == 7.004


def test_manifest_integrity():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
    assert "comparability" in art["baseline_definition"]
