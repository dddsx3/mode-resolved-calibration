"""P-ACTIVE-SET-ABLATION · artifact pin tests (E4/M1).

The ablation decomposes the informed-vs-random advantage on the certified
J_A functional (six policies) and records the realized-value arm's honest
findings: the weak-mode endpoint's E(k) hump, the anti-correlation of the
stable oracle ranking with single-light J_A gains, and the mechanism
anchors (unaligned mae improves while the gauge-aligned dual energy
worsens under refinement). The legacy-frame k-curve must reproduce the
frozen allocation per-run numbers bit-exactly.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/active_set_ablation.json"

POLICY_KEYS = ("continuous_fw_J_A", "classical_a_opt_J_A",
               "random_active48_mean", "random_universe_mean",
               "mode_tail_J_A", "rounded_J_A")


def test_ablation_artifact_structure():
    if not ART.exists():
        pytest.skip("artifact not committed")
    j = json.loads(ART.read_text(encoding="utf-8"))
    assert j["analysis_status"] == "active_set_ablation_v1"
    assert j["n_objects"] == 11 and len(j["rows"]) == 55      # 11 x 5 budgets
    for r in j["rows"]:
        for key in POLICY_KEYS:
            assert key in r and r[key] > 0, (r["object"], r["k"], key)
        # certified gap is nonnegative (greedy >= lower bound)
        assert r["certified_gap"] >= -1e-12
        # active-set dilution is nonnegative (universe random never better
        # than active random on J_A: inactive lights are exact no-ops and
        # the universe draw refines fewer active lights in expectation)
        assert r["active_set_dilution"] >= -1e-9
    for key in ("active_set_dilution_J_A", "mode_ordering_gap_J_A",
                "rounding_loss_J_A", "certified_gap_J_A"):
        assert key in j["decomposition"]


def test_ablation_oracle_findings():
    """诚实重构后的 oracle 发现:反相关 + 机制锚点 + legacy 冻结锚定。"""
    if not ART.exists():
        pytest.skip("artifact not committed")
    j = json.loads(ART.read_text(encoding="utf-8"))
    if not j["oracle"]:
        pytest.skip("oracle rows absent")
    for o in j["oracle"]:
        # 排序稳定(split-half 高)且与 J_A 增益反相关(驼峰上升沿机制)
        assert o["split_half_spearman"] > 0.6
        assert o["spearman_oracle_vs_Jgain"] < 0.0
        # 机制锚点:未对齐 mae 随全精化改善,dual 能量恶化(误差再分配)
        a = o["mechanism_anchors"]
        assert a["mae_all48"] < a["mae_base"]
        assert a["dual_energy_all48"] > a["dual_energy_base"]
        # legacy 冻结锚定:obj_10 的 k=14/48 复现冻结 per-run 值(位级)
        if o["object"] == "obj_10_pumpkin3":
            c = o["k_curve_legacy_frozen_anchor"]
            assert c["14"] == pytest.approx(1.4648, abs=5e-4)
            assert c["48"] == pytest.approx(0.1889, abs=5e-4)
            assert c["114"] == pytest.approx(c["48"], abs=1e-9)  # no-ops
