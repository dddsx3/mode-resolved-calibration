"""P-BASELINE artifact gates (TCI gap 3: literature baselines).

CI-safe: reads only results/baseline/baseline_comparison.json. Pins:

1. The OI hard anchor: 88 informed rows bit-identical to the frozen
   E-arm subset (the baseline shares orderings + seeds by construction;
   recorded in the artifact).
2. Outcome consistency (v1.1 reading, active-restricted, loader-corrected):
   the combined `reading` is recomputed from `median_dev_from` and must
   match; the measured outcome is OI geometry-insufficient (both
   budgets) and DiLiGenT geometry-informative-at-k14 (both beat random)
   / geometry-insufficient-at-k28. With the corrected DiLiGenT loader
   BOTH geometry-only and the informed family beat active-restricted
   random on DQ -- the pre-correction "informed loses to random" claim
   is withdrawn (P-DILIGENT-LOADER-FIX).
3. The headline table: on OI, dc05_active beats randomA48 but no
   informed policy loses to it; the v1 all-pool DC05's chance-level
   performance is reproduced and self-documented via the recorded
   active-set overlaps.
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
            assert d["reading"].startswith(expected), (tag, k)
    assert art["oi"]["14"]["reading"] == "geometry-insufficient"
    assert art["oi"]["28"]["reading"] == "geometry-insufficient"
    # loader-corrected: DQ 上是 "both beat random"(k=14,几何名义更好)与
    # geometry-insufficient(k=28, informed 更好)——"model loses" 已撤回
    assert art["diligent"]["14"]["reading"] == (
        "geometry-informative (both beat random)")
    assert art["diligent"]["28"]["reading"] == "geometry-insufficient"


def test_dq_both_beat_random_after_loader_fix():
    """loader 修正后(DQ 半):几何与 informed 在 DQ 两个预算上都优于
    active-random;"a_opt 输给随机"是丢光强归一化的伪影,已撤回
    (P-DILIGENT-LOADER-FIX)。"""
    art = _load()
    for k, dca, dinf in (("14", -0.336354, -0.273633),
                         ("28", -0.334681, -0.369331)):
        dev = art["diligent"][k]["median_dev_from"]
        assert round(dev["dc05_active_vs_randomA48"], 6) == dca, k
        assert round(dev["informed_vs_randomA48"], 6) == dinf, k
        assert dev["dc05_active_vs_randomA48"] < 0, k
        assert dev["informed_vs_randomA48"] < 0, k
    # v1 混淆自文档化:all-pool dc05 的 active 重合度远低于预算
    ov14 = art["oi"]["14"]["dc05_allpool_overlap_at_k"]
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
    # DQ(loader 修正后):k28 中位 dc05_active 6.881 / a_opt 7.000
    m2 = art["diligent"]["28"]["median_ang_by_unit"]
    assert round(m2["dc05_active"], 3) == 6.884
    assert round(m2["a_opt"], 3) == 6.682


def test_v1_1_fields_present_and_consistent():
    """P3/P5/P6/P7/P8 字段:浪费份额由产物算出、逐物体偏差数组可重建
    reading、正交判据、版本标记。"""
    import numpy as np
    art = _load()
    assert art["analysis_status"] == "baseline_comparison_v1_1"
    for tag in ("oi", "diligent"):
        for k, d in art[tag].items():
            assert "dc05_allpool_overlap_at_k" in d
            assert "wasted_budget_share" in d
            # wasted = 1 − overlap/k(逐物体算术自洽)
            for obj, ov in d["dc05_allpool_overlap_at_k"].items():
                assert abs(d["wasted_budget_share"][obj]
                           - (1.0 - ov / int(k))) < 1e-6
            assert d["wasted_share_range"] == [
                min(d["wasted_budget_share"].values()),
                max(d["wasted_budget_share"].values())]
            # 逐物体偏差数组可重建 reading(P7)
            devs = d["dev_per_object"]
            med_a = float(np.median([x for x in
                                     devs["dc05_active_vs_randomA48"]
                                     if x is not None]))
            med_i = float(np.median([x for x in
                                     devs["informed_vs_randomA48"]
                                     if x is not None]))
            assert abs(med_a - d["median_dev_from"]
                       ["dc05_active_vs_randomA48"]) < 1e-6
            assert abs(med_i - d["median_dev_from"]
                       ["informed_vs_randomA48"]) < 1e-6
            # 正交判据与中位数符号一致
            assert d["geometry_vs_randomA48"] == (
                "geometry-beats-randomA48" if med_a < 0
                else "geometry-at-randomA48")
            assert d["model_vs_randomA48"] == (
                "model-beats-randomA48" if med_i < 0
                else "model-loses-to-randomA48")
    # OI k=14 的浪费份额范围 = 产物给的 [0.357143, 0.857143]
    assert art["oi"]["14"]["wasted_share_range"] == [0.357143, 0.857143]


def test_manifest_integrity():
    art = _load()
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        CFG.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
    assert "comparability" in art["baseline_definition"]
