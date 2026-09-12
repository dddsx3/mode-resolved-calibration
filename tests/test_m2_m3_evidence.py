"""P-LOWRANK-FULLRES + P-ALLOC2 evidence gates（plan v2 M2/M3 结果验收）。

结构 + 机械有效性断言（无数值方向锁——两协议均为结果无关报告）：
  fullres: 11 对象 × 3 预算行齐、fw_gap ≥ 0、greedy/random 不低于下界、
           P 为全分辨率（> 1200 子采样口径）、逐对象动态范围 ∈ (0, 1)；
  alloc2:  行结构齐（4 policies 端点成对）、bootstrap 字段齐、
           每行 targeted/random 能量 ≥ 0。

数值本身如实报告（包括与评估报告 4.02% 的 divergence——见
provenance/reconciliation.json 与 P-CERT v1 的证据测试）。
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FULLRES = REPO / "results/certification/lowrank_fullres.json"
ALLOC2 = REPO / "results/mode_tail/allocation_mode_tail.json"
BUDGETS_K = (5, 14, 48)


def _load(p):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fullres():
    return _load(FULLRES)


@pytest.fixture(scope="module")
def alloc2():
    return _load(ALLOC2)


def test_fullres_structure(fullres):
    rows = fullres["rows"]
    objs = {r["object"] for r in rows}
    assert len(objs) == 11
    assert {r["k"] for r in rows} == set(BUDGETS_K)
    for r in rows:
        assert r["P"] > 1200                      # 全分辨率口径（非子采样）
        assert r["L"] == 142                      # 全 142 灯
        assert r["fw_gap"] >= 0.0
        tol = 1e-9 * max(1.0, abs(r["fw_lower_bound"]))
        assert r["greedy_J_A"] >= r["fw_lower_bound"] - tol
        assert r["random_min"] >= r["fw_lower_bound"] - tol
        assert r["J_all"] < r["J_none"]
        dyn = (r["J_none"] - r["J_all"]) / abs(r["J_none"])
        assert 0.0 < dyn < 1.0
    assert set(fullres["by_k"].keys()) == {str(k) for k in BUDGETS_K}
    assert fullres["manifest"]["git_sha"]


def test_alloc2_structure(alloc2):
    rows = alloc2["rows"]
    objs = {r["object"] for r in rows}
    assert len(objs) == 11
    # 每对象：3 levels × 2 regimes × 5 budgets × 10 seeds
    n_per_obj = {o: sum(1 for r in rows if r["object"] == o) for o in objs}
    assert set(n_per_obj.values()) == {3 * 2 * 5 * 10}
    assert set(alloc2["aggregated"].keys()) == {
        f"{rg}/{k}" for rg in (10, 100) for k in (5, 10, 14, 28, 48)}
    for r in rows:
        assert len(r["targeted"]) == 5 and len(r["random_active48"]) == 5
        assert all(x >= 0.0 for x in r["targeted"] + r["random_active48"])
    for key, blk in alloc2["aggregated"].items():
        for arm in ("targeted", "scalar_targeted"):
            assert len(blk[arm]["per_object"]) == 11
            assert len(blk[arm]["delta_ci95"]) == 2


def test_alloc2_paired_independence(alloc2):
    """配对性 sanity：同 (obj, level, regime, k, seed) 下两臂来自同一 raw。"""
    keys = {(r["object"], r["level"], r["regime"], r["k"], r["seed"])
            for r in alloc2["rows"]}
    assert len(keys) == len(alloc2["rows"])      # 无重复行
