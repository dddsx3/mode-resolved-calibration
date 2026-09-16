"""P1-b · 头条数字字段级绑定(docs/methods.md + docs/claims.md)。

背景(验收 P1-b 的量化):docs 数字门禁(存在性绊线)对 236 个 token
只有 8 个(3.4%)唯一保护;整数/2 位小数随机错值通过率 ~99-100%,
4 位小数在密集值域 64.8%——**审稿人真正会核的头条数全在无保护区**。

本测试把头条数字改为 README traced-map 式的**字段级绑定**:每个条目
(token, 产物, 提取器, 精度)双向校验——
  (a) token 必须以词边界出现在指定 doc(改 doc 即失败);
  (b) token 的数值必须与指定产物字段的提取值在打印精度内一致
      (改产物/漂移即失败)。
变异方向因此被双向覆盖,与数字池密度无关。

少数派生/结构量(2200 = 5×88×5、1e-4 上界断言、1.0045 的散文锚)
在 SPECIALS 里单独声明并注明理由。
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = {"methods": REPO / "docs/methods.md", "claims": REPO / "docs/claims.md",
        "experiments": REPO / "docs/EXPERIMENTS.md"}
ARTS = {
    "amp": REPO / "results/magnitude/directional_amplitude_summary.json",
    "ab": REPO / "results/submodularity/alpha_bound.json",
    "go": REPO / "results/goal_oriented/goal_orientation.json",
    "vm": REPO / "results/magnitude/validity_map.json",
    "dq": REPO / "results/openillumination/decision_quality.json",
    "abl": REPO / "results/openillumination/active_set_ablation.json",
    "cv": REPO / "results/magnitude/calibration_value_ceiling.json",
    "fam": REPO / "results/openillumination/corruption_family_sensitivity.json",
    "famc": REPO / "results/magnitude/linearization_radius_family.json",
    "fame": REPO / "results/openillumination/decision_quality_family.json",
    "ediag": REPO / "results/openillumination/corruption_family_e_diag.json",
    "ba": REPO / "results/openillumination/ball_anchor.json",
    "dq2": REPO / "results/diligent/diligent_queue.json",
    "bcmp": REPO / "results/baseline/baseline_comparison.json",
    "dqf": REPO / "results/openillumination/decision_quality_feasible.json",
    "ablf": REPO / "results/openillumination/active_set_ablation_feasible.json",
}


def _num(tok):
    return float(tok.replace("\u2212", "-").replace(",", ""))


def _dec(tok):
    s = tok.replace("\u2212", "-").replace(",", "")
    if "e" in s.lower():
        return max(0, -_ord(s))          # 3.9e-2 -> 2 位有效
    return len(s.split(".")[1]) if "." in s else 0


def _ord(s):
    m = re.search(r"e-(\d+)", s.lower())
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- 提取器
def _ab_min(field):
    return lambda j: min(r[field] for r in j["toy_instances"])


def _go_v(level, key, scale=1.0):
    return lambda j: j["headline"]["value_curves_by_level"][level][key] * scale


def _dq_boot(key, idx=None):
    def ext(j):
        v = j["bootstrap_dAUC"][key]
        return v["median_dAUC"] if idx is None else v["ci95"][idx]
    return ext


def _dq_a48(ext):
    def f(j):
        vals = [v["median_dAUC"] for k, v in j["bootstrap_dAUC"].items()
                if "vs_randomA48" in k and "ang_mean_deg" in k]
        return ext(vals)
    return f


def _abl_oracle(obj, field):
    def ext(j):
        return next(o for o in j["oracle"] if o["object"] == obj)[field]
    return ext


def _abl_curve(obj, k):
    return lambda j: next(o for o in j["oracle"]
                          if o["object"] == obj)["k_curve_legacy_frozen_anchor"][k]


M, C, E = "methods", "claims", "experiments"
_CLOUD_INFO = (REPO / "results/openillumination/provenance/"
               "decision_quality_cloud42bc299.json")
U = "\u2212"                                     # unicode minus,与 docs 一致

BINDINGS = [
    # ---- 幅值有效域 / α 界(methods + claims)----
    (M, "15.98", "amp", lambda j: j["arm_D_corrected"]["ratio_stats"]["median"], 2),
    (M, "0.635", "ab", lambda j: j["gamma_overall"]["gamma_lower_bound_min"], 3),
    (M, "162,000", "ab", lambda j: j["proof_limits"]["adversarial_triples"], 0),
    (C, "162,000", "ab", lambda j: j["proof_limits"]["adversarial_triples"], 0),
    (M, "12,493", "ab", lambda j: j["proof_limits"]["m_inv_sq_reversal_samples"], 0),
    (C, "12,493", "ab", lambda j: j["proof_limits"]["m_inv_sq_reversal_samples"], 0),
    (M, "0.271", "ab", lambda j: j["proof_limits"]["min_ratio_observed"], 3),
    (M, "1.31", "ab", lambda j: j["proof_limits"]["min_tightness"], 2),
    (M, "1.001011", "ab", _ab_min("min_ub_over_val"), 6),
    (M, "1.000096", "ab", _ab_min("min_val_over_lb"), 6),
    # ---- goal-oriented / 两项定律 ----
    (M, "62.87", "go", _go_v("0.5", "all", 100.0), 2),
    (M, "89.76", "go", _go_v("0.5", "mean", 100.0), 2),
    (M, "87.93", "go", _go_v("0.1", "mean", 100.0), 2),
    (M, "7.82", "go", _go_v("0.1", "all", 100.0), 2),
    (M, "0.878", "go", _go_v("0.1", "rho_mean"), 3),
    (M, "0.9000", "go", _go_v("8.0", "rho_mean"), 4),
    (M, "0.936", "go",
     lambda j: j["headline"]["spearman_mean_vs_contrast"]["median"], 3),
    (C, "0.936", "go",
     lambda j: j["headline"]["spearman_mean_vs_contrast"]["median"], 3),
    (C, "0.0008", "go",
     lambda j: j["headline"]["gauge_mechanism_rho_mean"]["median_abs_dV"], 4),
    (M, "3.9e-2", "go",
     lambda j: j["headline"]["gauge_mechanism_rho_mean"]["by_level"]["0.1"]
     ["max_abs_dV"], 2),
    # ---- decision quality v1.1(双基线)----
    (C, f"{U}3.90", "dq", _dq_boot("a_opt|10|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}6.52", "dq", _dq_boot("a_opt|10|ang_mean_deg|vs_randomU", 0), 2),
    (C, f"{U}2.45", "dq", _dq_boot("a_opt|10|ang_mean_deg|vs_randomU", 1), 2),
    (C, f"{U}5.03", "dq", _dq_boot("a_opt|100|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}7.92", "dq", _dq_boot("a_opt|100|ang_mean_deg|vs_randomU", 0), 2),
    (C, f"{U}3.51", "dq", _dq_boot("a_opt|100|ang_mean_deg|vs_randomU", 1), 2),
    (C, f"{U}3.68", "dq", _dq_boot("mode_aware|10|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}4.84", "dq", _dq_boot("mode_aware|100|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}3.88", "dq", _dq_boot("e_opt|10|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}5.14", "dq", _dq_boot("e_opt|100|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}3.61", "dq", _dq_boot("d_opt|10|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}4.75", "dq", _dq_boot("d_opt|100|ang_mean_deg|vs_randomU"), 2),
    (C, f"{U}0.17", "dq", _dq_a48(max), 2),   # A48 中位全负:max=最接近零
    (C, f"{U}0.55", "dq", _dq_a48(min), 2),
    (M, "0.917", "dq",
     lambda j: j["spearman_pred_vs_realized"]["ang_mean_deg"]["median"], 3),
    (C, "0.917", "dq",
     lambda j: j["spearman_pred_vs_realized"]["ang_mean_deg"]["median"], 3),
    (M, "0.687", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["rate"], 3),
    (C, "0.687", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["rate"], 3),
    (M, "0.657", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][0], 3),
    (C, "0.657", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][0], 3),
    (M, "0.716", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][1], 3),
    (C, "0.716", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][1], 3),
    (M, "678", "dq", lambda j: j["informed_pairwise_sign_pooled"]["agree"], 0),
    (C, "678", "dq", lambda j: j["informed_pairwise_sign_pooled"]["agree"], 0),
    (M, "987", "dq", lambda j: j["informed_pairwise_sign_pooled"]["total"], 0),
    (C, "987", "dq", lambda j: j["informed_pairwise_sign_pooled"]["total"], 0),
    (C, f"{U}0.575", "dq",
     lambda j: j["spearman_pred_vs_realized"]["mse_aligned"]["median"], 3),
    (C, f"{U}0.525", "dq",
     lambda j: j["spearman_pred_vs_realized"]["dual_mean"]["median"], 3),
    # ---- validity map(B8)----
    (C, "7.7", "vm", lambda j: j["by_excess_bin"][0]["ratio_median_of_medians"], 1),
    (C, "10.4", "vm", lambda j: j["by_excess_bin"][1]["ratio_median_of_medians"], 1),
    (C, "72.0", "vm", lambda j: j["by_excess_bin"][2]["ratio_median_of_medians"], 1),
    (C, "106.5", "vm", lambda j: j["by_excess_bin"][3]["ratio_median_of_medians"], 1),
    (C, "0.8", "vm",
     lambda j: min(b["spearman_median"] for b in j["by_excess_bin"]), 1),
    (C, "0.7", "vm",
     lambda j: min(b["kendall_median"] for b in j["by_excess_bin"]), 1),
    (C, "0.6", "vm", lambda j: j["by_level"]["0.2"]["kendall_median"], 1),
    # ---- active-set ablation(C8)----
    (C, "0.063", "abl",
     lambda j: j["decomposition"]["active_set_dilution_J_A"]["median"], 3),
    (C, "0.004", "abl",
     lambda j: j["decomposition"]["mode_ordering_gap_J_A"]["max"], 3),
    (M, f"{U}0.36", "abl", _abl_oracle("obj_03_pumpkin",
                                       "spearman_oracle_vs_Jgain"), 2),
    (M, f"{U}0.82", "abl", _abl_oracle("obj_10_pumpkin3",
                                       "spearman_oracle_vs_Jgain"), 2),
    (C, "0.82", "abl", _abl_oracle("obj_03_pumpkin", "split_half_spearman"), 2),
    (C, "0.88", "abl", _abl_oracle("obj_10_pumpkin3", "split_half_spearman"), 2),
    (C, "1.07", "abl", _abl_curve("obj_10_pumpkin3", "0"), 2),
    (C, "1.46", "abl", _abl_curve("obj_10_pumpkin3", "14"), 2),
    (C, "0.19", "abl", _abl_curve("obj_10_pumpkin3", "48"), 2),
    (C, "0.59", "abl", _abl_curve("obj_03_pumpkin", "0"), 2),
    (C, "1.21", "abl", _abl_curve("obj_03_pumpkin", "14"), 2),
    (C, "0.20", "abl", _abl_curve("obj_03_pumpkin", "48"), 2),

    # ---- EXPERIMENTS.md 头条(P2-5:纳入字段级绑定)----
    (E, "3960", "dq", lambda j: float(len(j["rows"])), 0),
    (E, "12.4", "dq", lambda j: j["manifest"]["elapsed_s"] / 3600.0, 1),
    (E, "36.4", "go",
     lambda j: json.loads(_CLOUD_INFO.read_text(encoding="utf-8"))
     ["manifest"]["elapsed_s"] / 60.0, 1),
    (E, "0.687", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["rate"], 3),
    (E, "0.657", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][0], 3),
    (E, "0.716", "dq",
     lambda j: j["informed_pairwise_sign_pooled"]["ci95"][1], 3),
    (E, f"{U}1.000", "dq",
     lambda j: j["spearman_informed_only"]["ang_mean_deg"]["min"], 3),
    (E, "2200", "dq",
     lambda j: 5 * 88 * 5, 0),   # 未变单元行数(5×88×5,交叉验证语义)
    (E, "678", "dq", lambda j: j["informed_pairwise_sign_pooled"]["agree"], 0),
    (E, "987", "dq", lambda j: j["informed_pairwise_sign_pooled"]["total"], 0),

    # ---- Σ_φ parameter-family sensitivity (C9 / P-SIGMA-FAMILY) ----
    (C, "0.07", "fam",
     lambda j: j["s21_direction_share"]["share_at_anchor_median"] * 100.0, 2),
    (E, "0.07", "fam",
     lambda j: j["s21_direction_share"]["share_at_anchor_median"] * 100.0, 2),
    (C, "0.0076", "fam",
     lambda j: j["s22_two_term_law"]["overall"]["median_abs_dV"], 4),
    (E, "0.0076", "fam",
     lambda j: j["s22_two_term_law"]["overall"]["median_abs_dV"], 4),
    (C, "0.464", "fam",
     lambda j: j["s22_two_term_law"]["overall"]["max_abs_dV"], 3),
    (E, "0.464", "fam",
     lambda j: j["s22_two_term_law"]["overall"]["max_abs_dV"], 3),
    (C, f"{U}0.0028", "fam",
     lambda j: j["s23_ceiling"]["max_violation"], 4),
    (M, f"{U}0.0028", "fam",
     lambda j: j["s23_ceiling"]["max_violation"], 4),
    (C, "0.863", "fam",
     lambda j: min(v["min"] for v in
                   j["s21_reverse_control"]["share_int_by_sigma_logI"]
                   .values()), 3),
    (C, "495", "fam", lambda j: float(len(j["rows"])), 0),
    (M, "495", "fam", lambda j: float(len(j["rows"])), 0),
    (E, "495", "fam", lambda j: float(len(j["rows"])), 0),
    (M, "45", "fam", lambda j: float(len(j["grid"])), 0),
    (E, "45", "fam", lambda j: float(len(j["grid"])), 0),
    (C, "0.35", "famc",
     lambda j: j["shapes_summary"]["joint_het"]["radius_10x"]
     ["median_over_crossed_subset"], 2),
    (E, "0.35", "famc",
     lambda j: j["shapes_summary"]["joint_het"]["radius_10x"]
     ["median_over_crossed_subset"], 2),
    # ---- E arm (C9 / P-SIGMA-FAMILY 21b) ----
    (C, "0.704", "fame",
     lambda j: j["arms"]["anchor"]["informed_pairwise_sign"]["rate"], 3),
    (E, "0.704", "fame",
     lambda j: j["arms"]["anchor"]["informed_pairwise_sign"]["rate"], 3),
    (C, "0.889", "fame",
     lambda j: j["arms"]["dir_heavy"]["informed_pairwise_sign"]["rate"], 3),
    (E, "0.889", "fame",
     lambda j: j["arms"]["dir_heavy"]["informed_pairwise_sign"]["rate"], 3),
    (C, "0.353", "fame",
     lambda j: j["arms"]["het"]["informed_pairwise_sign"]["rate"], 3),
    (E, "0.353", "fame",
     lambda j: j["arms"]["het"]["informed_pairwise_sign"]["rate"], 3),
    (C, "88", "fame",
     lambda j: float(j["arms"]["anchor"]["informed_pairwise_sign"]["agree"]), 0),
    (E, "88", "fame",
     lambda j: float(j["arms"]["anchor"]["informed_pairwise_sign"]["agree"]), 0),
    (C, "125", "fame",
     lambda j: float(j["arms"]["anchor"]["informed_pairwise_sign"]["total"]), 0),
    (E, "125", "fame",
     lambda j: float(j["arms"]["anchor"]["informed_pairwise_sign"]["total"]), 0),

    # ---- E cross-diagnosis (P-SIGMA-FAMILY-E-DIAG) ----
    (C, "0.949", "ediag", lambda j: j["summary"]["s_pred"], 3),
    (E, "0.949", "ediag", lambda j: j["summary"]["s_pred"], 3),
    (C, f"{U}0.325", "ediag", lambda j: j["summary"]["s_real"], 3),
    (E, f"{U}0.325", "ediag", lambda j: j["summary"]["s_real"], 3),
    (E, "0.789", "ediag", lambda j: j["summary"]["fit_anc"], 3),
    (C, f"{U}0.778", "ediag", lambda j: j["summary"]["fit_het"], 3),
    (E, f"{U}0.778", "ediag", lambda j: j["summary"]["fit_het"], 3),
    (C, f"{U}0.738", "ediag", lambda j: j["summary"]["cross_ah"], 3),
    (E, f"{U}0.738", "ediag", lambda j: j["summary"]["cross_ah"], 3),
    (C, "0.738", "ediag", lambda j: j["summary"]["cross_ha"], 3),
    (E, "0.738", "ediag", lambda j: j["summary"]["cross_ha"], 3),

    # ---- ball anchor (C10 / P-BALL-ANCHOR) ----
    (C, "2.96", "ba",
     lambda j: j["measured_anchor"]["sig_dir_deg"], 2),
    (E, "2.96", "ba",
     lambda j: j["measured_anchor"]["sig_dir_deg"], 2),
    (C, "0.0159", "ba",
     lambda j: j["measured_anchor"]["sig_logI"], 4),
    (E, "0.0159", "ba",
     lambda j: j["measured_anchor"]["sig_logI"], 4),
    (C, "36.0", "ba",
     lambda j: j["direction_share_at_anchor"]["median"] * 100.0, 1),
    (E, "36.0", "ba",
     lambda j: j["direction_share_at_anchor"]["median"] * 100.0, 1),
    (C, "10.6", "ba",
     lambda j: (__import__("math").radians(
         j["measured_anchor"]["sig_dir_deg"])
         / j["measured_anchor"]["sig_logI"]) ** 2, 1),
    (E, "4.47", "ba",
     lambda j: j["gt_comparison"]["max_direction_error_deg"], 2),
    # ---- DiLiGenT queue (C11 / P-DILIGENT-QUEUE) ----
    (C, "0.20", "dq2",
     lambda j: j["channel_decomposition"]["direction_max_pct"], 2),
    (E, "0.75", "dq2",
     lambda j: min(j["linearization_radius"]["radius_2x"]
                   ["per_object"].values()), 2),
    (E, "1.5", "dq2",
     lambda j: max(j["linearization_radius"]["radius_2x"]
                   ["per_object"].values()), 1),
    # ---- baseline comparison (C12 / P-BASELINE) ----
    (E, "2.894", "bcmp",
     lambda j: j["oi"]["28"]["median_ang_by_unit"]["e_opt"], 3),
    (E, "4.005", "bcmp",
     lambda j: j["oi"]["28"]["median_ang_by_unit"]["dc05_active"], 3),
    (E, f"{U}0.438", "bcmp",
     lambda j: j["oi"]["14"]["median_dev_from"]
     ["dc05_active_vs_randomA48"], 3),
    (E, f"{U}0.615", "bcmp",
     lambda j: j["oi"]["28"]["median_dev_from"]
     ["dc05_active_vs_randomA48"], 3),
    (E, "6.88", "bcmp",
     lambda j: j["diligent"]["28"]["median_ang_by_unit"]["dc05_active"], 2),
    (E, "6.68", "bcmp",
     lambda j: j["diligent"]["28"]["median_ang_by_unit"]["a_opt"], 2),
    # ---- feasible-budget DQ diagnostic (P-DQ-FEASIBLE) ----
    (C, "−1.40", "dqf",
     lambda j: min(v["median_dAUC"] for v in
                   j["feasible_dAUC_vs_randomA48"].values()), 2),
    (E, "−1.40", "dqf",
     lambda j: min(v["median_dAUC"] for v in
                   j["feasible_dAUC_vs_randomA48"].values()), 2),
    (C, "−0.38", "dqf",
     lambda j: max(v["median_dAUC"] for v in
                   j["feasible_dAUC_vs_randomA48"].values()), 2),
    (E, "−0.38", "dqf",
     lambda j: max(v["median_dAUC"] for v in
                   j["feasible_dAUC_vs_randomA48"].values()), 2),
    # ---- ablation budget-axis caliber (P-ABLATION-FEASIBLE) ----
    (C, "1364", "ablf",
     lambda j: j["ratio_with_k48"]["as_multiple"], 0),
    (C, "1001", "ablf",
     lambda j: j["ratio_without_k48"]["as_multiple"], 0),
    (C, "71.4", "ba",
     lambda j: max(j["direction_share_at_anchor"]["per_object"].values())
     * 100.0, 1),
]

# SPECIALS 追加(数值侧由产物 pin 测试守):11/44 = goal_orientation 的
# top3 n_cells_disjoint / n_cells(复合字面量,只做存在性)

# 派生/结构量:只做存在性+理由登记(数值侧由对应产物的 pin 测试守)
SPECIALS = [
    # 2200 = 5 个未变单元 × 88 cells × 5 budgets(结构性计数,由
    # dq_cross_env_repro / reuse-equivalence 记录的语义守)
    (C, "2200", "structural count: 5 unchanged units x 88 cells x 5 budgets"),
    # matched-GLS 合成 MC 比(冻结基准记录;产物内为散文锚)
    (M, "1.0045", "frozen MC record; prose-anchored in "
                  "directional_amplitude_summary.amplitude_interpretation"),
    (C, "11/44", "goal_orientation headline: top3 n_cells_disjoint / n_cells "
                 "(compound literal; value pinned by test_goal_oriented)"),]


# Occurrence-count anchoring (P1-b mutation-tested hardening: 0.635
# appears twice in methods.md - mutating ONE copy passes a presence-only
# check; anchoring the count catches any single-copy mutation).
# Deliberate additions/removals must update this table (same
# discipline as the README traced map). Covers methods/claims/EXPERIMENTS.
OCCURRENCE_COUNTS = {
    # ---- Σ_φ family sensitivity (C9 / P-SIGMA-FAMILY) ----
    ("claims", "0.07"): 1,
    ("experiments", "0.07"): 1,
    ("claims", "0.0076"): 1,
    ("experiments", "0.0076"): 1,
    ("claims", "0.464"): 1,
    ("experiments", "0.464"): 2,
    ("claims", "−0.0028"): 1,
    ("methods", "−0.0028"): 1,
    ("claims", "0.863"): 1,
    ("claims", "495"): 1,
    ("methods", "495"): 1,
    ("experiments", "495"): 3,
    ("methods", "45"): 1,
    ("experiments", "45"): 3,
    ("claims", "0.35"): 1,
    ("experiments", "0.35"): 3,
    ("claims", "3283"): 2,
    ("methods", "3283"): 1,
    ("experiments", "3283"): 3,
    # ---- baseline comparison (24 / C12) ----
    ("claims", "88"): 9,
    ("experiments", "88"): 14,
    # ---- DiLiGenT queue (23 / C11) ----
    ("claims", "0.75"): 1,
    ("experiments", "0.75"): 5,
    ("experiments", "433"): 1,
    # ---- ball anchor (22 / C10) ----
    ("claims", "2.96"): 2,
    ("experiments", "2.96"): 5,
    ("claims", "0.0159"): 1,
    ("experiments", "0.0159"): 4,
    ("claims", "36.0"): 2,
    ("experiments", "36.0"): 3,
    ("claims", "10.6"): 2,
    ("experiments", "10.6"): 1,
    ("claims", "4.47"): 1,
    ("experiments", "4.47"): 1,
    ("claims", "71.4"): 2,
    ("experiments", "71.4"): 4,
    ("experiments", "192"): 1,
    # ---- E cross-diagnosis (21b / E-DIAG) ----
    ("claims", "0.949"): 1,
    ("experiments", "0.949"): 1,
    ("claims", "−0.325"): 1,
    ("experiments", "−0.325"): 1,
    ("claims", "−0.738"): 1,
    ("experiments", "−0.738"): 1,
    ("claims", "0.738"): 1,
    ("experiments", "0.738"): 2,
    ("experiments", "0.789"): 2,
    ("claims", "−0.778"): 1,
    ("experiments", "−0.778"): 3,
    # ---- E arm (21b) ----
    ("claims", "0.704"): 3,
    ("experiments", "0.704"): 3,
    ("claims", "0.889"): 1,
    ("experiments", "0.889"): 2,
    ("claims", "0.353"): 1,
    ("experiments", "0.353"): 3,
    ("claims", "88"): 9,
    ("experiments", "88"): 14,
    ("claims", "125"): 2,
    ("experiments", "125"): 2,
    ("claims", "0.817"): 1,
    ("experiments", "0.817"): 1,
    ("claims", "0.268"): 1,
    ("experiments", "0.268"): 1,
    ("experiments", "104"): 1,
    ("claims", "42"): 1,
    ("experiments", "42"): 2,
    ("claims", "0.0008"): 2,
    ("claims", "0.004"): 1,
    ("claims", "0.063"): 1,
    ("claims", "0.19"): 1,
    ("claims", "0.20"): 2,
    ("claims", "0.59"): 1,
    ("claims", "0.6"): 1,
    ("claims", "0.657"): 1,
    ("claims", "0.687"): 1,
    ("claims", "0.7"): 4,
    ("claims", "0.716"): 1,
    ("claims", "0.8"): 1,
    ("claims", "0.82"): 2,
    ("claims", "0.88"): 1,
    ("claims", "0.917"): 1,
    ("claims", "0.936"): 1,
    ("claims", "1.07"): 1,
    ("claims", "1.21"): 1,
    ("claims", "1.46"): 1,
    ("claims", "10.4"): 1,
    ("claims", "106.5"): 1,
    ("claims", "12,493"): 1,
    ("claims", "162,000"): 1,
    ("claims", "678"): 1,
    ("claims", "7.7"): 1,
    ("claims", "72.0"): 1,
    ("claims", "987"): 1,
    ("claims", "−0.17"): 1,
    ("claims", "−0.525"): 1,
    ("claims", "−0.55"): 1,
    ("claims", "−0.575"): 1,
    ("claims", "−2.45"): 1,
    ("claims", "−3.51"): 1,
    ("claims", "−3.61"): 1,
    ("claims", "−3.68"): 1,
    ("claims", "−3.88"): 1,
    ("claims", "−3.90"): 1,
    ("claims", "−4.75"): 1,
    ("claims", "−4.84"): 1,
    ("claims", "−5.03"): 1,
    ("claims", "−5.14"): 1,
    ("claims", "−6.52"): 1,
    ("claims", "−7.92"): 1,
    ("experiments", "0.657"): 3,
    ("experiments", "0.687"): 5,
    ("experiments", "0.716"): 3,
    ("experiments", "12.4"): 2,
    ("experiments", "2200"): 1,
    ("experiments", "36.4"): 1,
    ("experiments", "3960"): 1,
    ("experiments", "678"): 4,
    ("experiments", "987"): 4,
    ("experiments", "−1.000"): 1,
    ("methods", "0.271"): 2,
    ("methods", "0.635"): 2,
    ("methods", "0.657"): 1,
    ("methods", "0.687"): 1,
    ("methods", "0.716"): 1,
    ("methods", "0.878"): 1,
    ("methods", "0.9000"): 2,
    ("methods", "0.917"): 1,
    ("methods", "0.936"): 1,
    ("methods", "1.000096"): 1,
    ("methods", "1.001011"): 1,
    ("methods", "1.31"): 1,
    ("methods", "12,493"): 2,
    ("methods", "15.98"): 1,
    ("methods", "162,000"): 2,
    ("methods", "3.9e-2"): 1,
    ("methods", "62.87"): 1,
    ("methods", "678"): 1,
    ("methods", "7.82"): 1,
    ("methods", "87.93"): 1,
    ("methods", "89.76"): 1,
    ("methods", "987"): 1,
    ("methods", "−0.36"): 1,
    ("methods", "−0.82"): 1
}



def test_headline_numbers_field_bound():
    texts = {k: p.read_text(encoding="utf-8") for k, p in DOCS.items()}
    arts = {}
    for doc, tok, art, ext, dec in BINDINGS:
        pat = r"(?<![\w.])" + re.escape(tok) + r"(?![\w])"
        hits = re.findall(pat, texts[doc])
        assert hits, \
            f"[{doc}] headline token {tok!r} not found (mutated or removed?)"
        want = OCCURRENCE_COUNTS.get((doc, tok))
        if want is not None:
            assert len(hits) == want, \
                f"[{doc}] {tok!r}: {len(hits)} occurrences, expected {want}"
        if art not in arts:
            arts[art] = json.loads(ARTS[art].read_text(encoding="utf-8"))
        val = ext(arts[art])
        tv = _num(tok) if "/" not in tok else _num(tok.split("/")[0])
        tol = 0.5 * 10 ** -dec + 1e-12
        assert abs(tv - val) <= tol, \
            f"[{doc}] {tok!r} vs artifact {val!r} (dec={dec})"


def test_headline_specials_present():
    texts = {k: p.read_text(encoding="utf-8") for k, p in DOCS.items()}
    for doc, tok, why in SPECIALS:
        pat = r"(?<![\w.])" + re.escape(tok) + r"(?![\w])"
        assert re.search(pat, texts[doc]), f"[{doc}] {tok!r} missing ({why})"


def test_m10_ceiling_tightness_bound():
    """M10 'met to <= 1e-4 at level 64':kappa=10 全部 slack ≤ 1e-4。"""
    cv = json.loads(ARTS["cv"].read_text(encoding="utf-8"))
    sl = [r["slack"] for r in cv["rows"] if r["kappa"] == 10.0]
    assert sl and max(sl) <= 1.0e-4, max(sl)
