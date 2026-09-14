"""P-SIGMA-FAMILY artifact gates (S3): predictor side + C arm.

CI-safe: reads only committed artifacts (no raw data). Pins:

Predictor side (results/openillumination/corruption_family_sensitivity.json):
1. Grid completeness: 45 unique family points x 11 objects = 495 rows.
2. Stored bit-anchors: the anchor / intensity_only / dir_only@0.5-deg
   points reproduce the frozen channel_decomposition.json joint /
   intensity / direction @ level 0.5 rows EXACTLY (J_A_1, J_A_kappa, D
   float ==; the closed-channel convention is shared).
3. Gauge anchor: the anchor point's two-term-law quantities match the
   frozen goal_orientation.json gauge_mechanism_rho_mean at level 0.5
   (q, r, V_pred, dV at 6 decimals; V_meas == V_rho_mean).
4. S2-1 outcome consistency (robust; r* censored high 11/11; direction
   share at the operating point tiny; direction-only max <= 2% -- the
   C7 generalization).
5. S2-2 outcome consistency (parameterization-specific: median |dV|
   within 0.01 but max beyond 0.05).
6. S2-3 ceiling guard: max violation <= 1e-9 (theorem M10/M10').

C arm (results/magnitude/linearization_radius_family.json):
7. Control bit-anchor: joint-shape rows E_mean == frozen
   linearization_radius.json per (object, level).
8. Outcome consistency (channel-dependent): joint/intensity_only radii
   equal the frozen medians; direction_only never crosses (0/11);
   joint_het radius shrunk below the factor-4 band.
9. Manifest integrity (config sha256 matches committed config).
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/corruption_family_sensitivity.json"
ART_C = REPO / "results/magnitude/linearization_radius_family.json"
FROZEN_CHAN = REPO / "results/openillumination/channel_decomposition.json"
FROZEN_GOAL = REPO / "results/goal_oriented/goal_orientation.json"
FROZEN_RAD = REPO / "results/magnitude/linearization_radius.json"


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not committed")
    return json.loads(p.read_text(encoding="utf-8"))


def _anchor_row(art, obj, sig_logI, sig_dir_deg):
    """按族点 (sig_logI, sig_dir_deg, het=0, rho=0) 取行。"""
    grid = art["grid"]
    for r in art["rows"]:
        if r["object"] != obj:
            continue
        pt = grid[r["grid_idx"]]
        if (pt["sig_logI"] == sig_logI and pt["sig_dir_deg"] == sig_dir_deg
                and pt["het_sigma"] == 0.0 and pt["rho_c"] == 0.0):
            return r
    raise KeyError(f"family point ({sig_logI}, {sig_dir_deg}) not found "
                   f"for {obj}")


def test_grid_completeness():
    art = _load(ART)
    assert art["gate"] == "P-SIGMA-FAMILY"
    assert len(art["grid"]) == 45
    assert len(art["rows"]) == 45 * 11
    tags = {t for pt in art["grid"] for t in pt["tags"]}
    assert {"anchor", "dir_sweep", "logI_sweep", "reverse_control",
            "het_sweep", "rho_sweep", "tier2", "tier3", "dir_only",
            "intensity_only", "both_closed"} <= tags


def test_stored_bit_anchors_vs_frozen_channel_decomposition():
    """锚点/单通道点(存档值)== 冻结 channel_decomposition @ level 0.5。"""
    art, frz = _load(ART), _load(FROZEN_CHAN)
    frz_rows = {(r["object"], r["channel"]): r for r in frz["rows"]
                if r["level"] == 0.5}
    for obj in sorted({r["object"] for r in art["rows"]}):
        for chan, li, sd in (("joint", 0.5, 0.5),
                             ("intensity", 0.5, 0.001),
                             ("direction", 1e-06, 0.5)):
            mine = _anchor_row(art, obj, li, sd)
            f = frz_rows[(obj, chan)]
            assert mine["J_A_1"] == f["J_A_1"], (obj, chan, "J_A_1")
            assert mine["J_A_kappa"] == f["J_A_kappa"], (obj, chan, "J_A_kappa")
            assert mine["D"] == f["D"], (obj, chan, "D")


def test_gauge_anchor_matches_goal_orientation():
    """锚点两项定律量 == 冻结 goal_orientation level-0.5 gauge 机制。"""
    art, frz = _load(ART), _load(FROZEN_GOAL)
    gm = {r["object"]: r for r in frz["rows"] if r["level"] == 0.5}
    for obj in sorted({r["object"] for r in art["rows"]}):
        mine = _anchor_row(art, obj, 0.5, 0.5)
        g = gm[obj]["gauge_mechanism_rho_mean"]
        assert round(mine["q"], 6) == g["q"], (obj, "q")
        assert round(mine["r"], 6) == g["r"], (obj, "r")
        assert round(mine["V_pred"], 6) == g["V_pred"], (obj, "V_pred")
        # 冻结 dV 是"已舍入分量之差再舍入"(goal_orientation 的写法),
        # 家族侧存的是全精度 —— 镜像同一计算路径后比较
        assert round(round(mine["V_pred"], 6) - round(mine["V_meas"], 6),
                     6) == g["dV"], (obj, "dV")
        assert mine["V_meas"] == gm[obj]["V_rho_mean"], (obj, "V_meas")


def test_s21_outcome_consistency():
    art = _load(ART)
    s21 = art["s21_direction_share"]
    assert s21["outcome"] == "robust"
    assert s21["r_star_median"] == ">25"
    assert s21["n_censored_high"] == 11
    assert s21["n_censored_low"] == 0
    # 工作点(锚点)方向份额:中位 < 0.2%
    assert s21["share_at_anchor_median"] < 0.002
    # C7 直接推广:方向单通道 D 全网格 <= 2%
    assert s21["direction_only_max_pct"] <= 2.0
    assert s21["direction_only_max_pct"] == 1.68
    # 反向控制:固定 σ_dir=1°,强度份额处处 >= 0.85
    rc = art["s21_reverse_control"]["share_int_by_sigma_logI"]
    assert all(v["min"] >= 0.85 for v in rc.values())


def test_s22_outcome_consistency():
    art = _load(ART)
    s22 = art["s22_two_term_law"]
    assert s22["outcome"] == "parameterization-specific"
    ov = s22["overall"]
    # 预注册判据的自洽:median <= 0.01 且 max > 0.05 ⇒ 该结局
    assert ov["median_abs_dV"] <= 0.01
    assert ov["max_abs_dV"] > 0.05
    # 破坏点集中在通道隔离/极端组合
    assert s22["by_tag"]["dir_only"]["max_abs_dV"] == ov["max_abs_dV"]
    assert s22["by_tag"]["anchor"]["max_abs_dV"] <= 0.01
    # gauge 恒等式(Σ 无关)逐对象机器精度
    assert max(s22["align_residual_per_object"].values()) < 1e-9


def test_s23_ceiling_guard():
    art = _load(ART)
    s23 = art["s23_ceiling"]
    assert s23["max_violation"] <= 1e-9
    assert s23["n_rows"] == 495
    ceiling = 1.0 - 1.0 / art["kappa"]
    assert max(r["D"] for r in art["rows"]) <= ceiling + 1e-9


def test_manifest_integrity():
    import hashlib
    art = _load(ART)
    cfg = REPO / "configs/corruption_family_sensitivity.yaml"
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        cfg.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40


# ---------------- C 臂 ----------------
def test_c_control_bit_anchor():
    art, frz = _load(ART_C), _load(FROZEN_RAD)
    for obj, d in art["objects"]["joint"].items():
        for r in d["rows"]:
            f = next(x for x in frz["objects"][obj]["rows"]
                     if x["level"] == r["level"])
            assert r["E_mean"] == f["E_mean"], (obj, r["level"], "E_mean")


def test_c_outcome_consistency():
    art = _load(ART_C)
    assert art["outcome"] == "channel-dependent"
    ss = art["shapes_summary"]
    # 控制/强度单通道 == 冻结中位(1.0 / 1.5)
    assert ss["joint"]["radius_2x"]["median_over_crossed_subset"] == 1.0
    assert ss["joint"]["radius_10x"]["median_over_crossed_subset"] == 1.5
    assert ss["intensity_only"]["radius_2x"][
        "median_over_crossed_subset"] == 1.0
    assert ss["intensity_only"]["radius_10x"][
        "median_over_crossed_subset"] == 1.5
    # 方向单通道:整个网格 [0.05, 8] 内从未越限(0/11)
    assert ss["direction_only"]["radius_2x"]["n_crossed"] == 0
    assert ss["direction_only"]["radius_10x"]["n_crossed"] == 0
    # het=1.0:半径缩小到 0.2/0.35(带外,判 channel-dependent)
    assert ss["joint_het"]["radius_2x"]["median_over_crossed_subset"] == 0.2
    assert ss["joint_het"]["radius_10x"][
        "median_over_crossed_subset"] == 0.35


def test_c_manifest_integrity():
    import hashlib
    art = _load(ART_C)
    cfg = REPO / "configs/linearization_radius_family.yaml"
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        cfg.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40


# ---------------- E 臂(reduced decision quality) ----------------
ART_E = REPO / "results/openillumination/decision_quality_family.json"
FROZEN_DQ = REPO / "results/openillumination/decision_quality.json"


def test_e_control_rows_bit_identical_to_frozen_subset():
    """anchor 臂的行(存档值)== 冻结 decision_quality.json 的
    (level 0.5, regime 10, k∈{14,28}, 4 informed) 子集,逐位。"""
    art, frz = _load(ART_E), _load(FROZEN_DQ)
    frz_rows = {(r["object"], r["unit"], r["k"]): r for r in frz["rows"]
                if r["level"] == 0.5 and r["regime"] == 10
                and r["k"] in (14, 28)}
    n = 0
    for r in art["rows"]:
        if r["arm"] != "anchor":
            continue
        f = frz_rows[(r["object"], r["unit"], r["k"])]
        for field in ("pred_J_A", "ang_mean_deg", "mse_aligned",
                      "dual_mean"):
            assert r[field] == f[field], (r["object"], r["unit"], r["k"],
                                          field)
        n += 1
    assert n == 88                    # 11 obj x 4 units x 2 budgets


def test_e_outcome_consistency():
    """结局字段必须与预注册判据作用于存档值的结果一致(判据本身受测)。"""
    art = _load(ART_E)
    assert art["gate"] == "P-SIGMA-FAMILY"
    arms = art["arms"]
    assert set(arms) == {"anchor", "dir_heavy", "het"}
    rates = {a: d["informed_pairwise_sign"]["rate"] for a, d in arms.items()}
    assert all(r is not None for r in rates.values())
    expected = ("robust" if (art["control_reproduces_subset"]
                             and all(r >= 0.60 for r in rates.values()))
                else "parameterization-specific")
    assert art["outcome"] == expected
    # 控制臂必须精确复现冻结子集 88/125
    assert art["control_reproduces_subset"] is True
    comp = art["frozen_reference"]["subset_level05_regime10_k1428"]
    a = arms["anchor"]["informed_pairwise_sign"]
    assert (a["agree"], a["total"]) == (comp["agree"], comp["total"])
    assert a["total"] == 125
    # 每臂配对统计的算术自洽
    for d in arms.values():
        s = d["informed_pairwise_sign"]
        assert 0 <= s["agree"] <= s["total"]
        assert abs(s["rate"] - s["agree"] / s["total"]) < 5e-7
    # manifest
    import hashlib
    cfg = REPO / "configs/decision_quality_family.yaml"
    assert art["manifest"]["config_sha256"] == hashlib.sha256(
        cfg.read_bytes()).hexdigest()
    assert len(art["manifest"]["git_sha"]) == 40
