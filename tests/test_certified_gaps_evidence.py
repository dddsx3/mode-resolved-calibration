"""P-CERT certified-gap table evidence gate（results/certification/）。

自包含：只读已提交的 JSON 证据。认证表是描述性/认证性的（预注册条款：
无符号判据），所以这里锁的是**结构有效性**而非具体数值方向：
  - 表结构齐全（11 对象 × 5 预算）；
  - 证书机械有效性：fw_gap ≥ 0、greedy/random 均不低于认证下界、
    J_all < J_none（分配方向 sanity）；
  - 动态范围在 (0, 1) 内（构造保证）。
数值本身如实报告——包括与潜力评估报告 4.02% 的 divergence
（那是合成场景实例数字；真实对象 36–89%，见 provenance/reconciliation.json）。
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GAPS = REPO / "results/certification/certified_gaps.json"
RECON = REPO / "results/certification/provenance/reconciliation.json"
BUDGETS_K = (5, 10, 14, 28, 48)


@pytest.fixture(scope="module")
def gaps():
    return json.loads(GAPS.read_text(encoding="utf-8"))


def test_table_structure(gaps):
    rows = gaps["rows"]
    assert len(rows) == 55
    objs = {r["object"] for r in rows}
    assert len(objs) == 11
    ks = {r["k"] for r in rows}
    assert ks == set(BUDGETS_K)
    for r in rows:
        for field in ("J_none", "J_all", "fw_J_A", "fw_gap", "fw_lower_bound",
                      "greedy_J_A", "random_mean"):
            assert field in r
    assert set(gaps["by_k"].keys()) == {str(k) for k in BUDGETS_K}
    assert gaps["manifest"]["git_sha"]


def test_certificate_mechanics_hold(gaps):
    """认证机械有效性：fw_gap ≥ 0；任何候选不低于下界；方向 sanity。"""
    for r in gaps["rows"]:
        tol = 1e-9 * max(1.0, abs(r["fw_lower_bound"]))
        assert r["fw_gap"] >= 0.0
        assert r["greedy_J_A"] >= r["fw_lower_bound"] - tol
        assert r["random_min"] >= r["fw_lower_bound"] - tol
        assert r["J_all"] < r["J_none"]            # 分配方向 sanity
        for x in r["random_J_A"]:
            assert x >= r["fw_lower_bound"] - tol


def test_dynamic_range_bounded_and_recorded(gaps):
    """动态范围按构造 ∈ (0, 1)，且逐对象值已被记录（55 行全部在档）。"""
    dyn = {(r["object"], r["k"]):
           (r["J_none"] - r["J_all"]) / abs(r["J_none"]) for r in gaps["rows"]}
    assert len(dyn) == 55
    for v in dyn.values():
        assert 0.0 < v < 1.0


def test_reconciliation_registered(gaps=None):
    """与潜力评估报告 4.02% 的 divergence 必须已注册（B4 门留痕）。"""
    rec = json.loads(RECON.read_text(encoding="utf-8"))
    assert rec["conclusion"]["b4_gate"] == "CLOSED with documented divergence"
    assert "synthetic" in rec["divergence_from_assessment_headline"]["clarification"].lower()
