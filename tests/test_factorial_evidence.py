"""MF-0.3 factorial evidence gate（数学冻结 v1.0 §57/§58）。

自包含：只读已提交的 CSV/JSON 证据，不需要原始数据、不执行重算。
锁三件事：
  1. factorial 证据存在且结构齐全（四臂、protocol、display）；
  2. 机制校验通过且 A 臂锚点与冻结 benchmark 逐位一致（pred/emp 相对差 0
     容差带 + A 臂统计量与冻结 validation_summary.json 逐字段一致）；
  3. 论文口径 = D（arm D 为 paper-facing，固定先验），其主结果字段存在。
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FACT = REPO / "results/openillumination/correctness/mf0_factorial_summary.json"
FROZEN = REPO / "results/openillumination/validation_summary.json"


@pytest.fixture(scope="module")
def factorial():
    return json.loads(FACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frozen_summary():
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def test_factorial_structure(factorial):
    """四臂 + protocol + display 结构齐全，arm 语义正确。"""
    assert set(factorial["variants"]) == {"A", "B", "C", "D"}
    expect = {"A": ("legacy", "legacy"), "B": ("corrected", "legacy"),
              "C": ("legacy", "dual"), "D": ("corrected", "dual")}
    for arm, (conv, coord) in expect.items():
        v = factorial["variants"][arm]
        assert v["noise_fit_convention"] == conv
        assert v["mode_coordinate"] == coord
        for field in ("RA", "RA_ci95", "n_pos_cells", "n_pos_objects",
                      "stratified_median_mode", "pooled_spearman"):
            assert field in v
        # v1.1 字段语义修正：stratified_valid_levels 必须是**有限层数计数**
        # （v1 误存布尔值，曾使审计误读为"6 层只有 1 层有效"）
        assert v["stratified_valid_levels"] == 6
    assert factorial["paper_variant"] == "D"
    assert factorial["protocol"]["n_cells"] == 66
    assert factorial["protocol"]["n_modes"] == 5
    assert factorial["protocol"]["bootstrap"] == dict(reps=10000,
                                                      seed=20260908,
                                                      unit="object")


def test_machinery_check_pass(factorial):
    """机制校验：A 臂逐位复现冻结基准（容差带内；实测 0.0）。"""
    m = factorial["machinery_check"]
    assert m["pass_"] is True
    assert m["pred_deg_max_rel"] <= 1e-7
    assert m["emp_deg_max_rel"] <= 1e-7
    assert m["pooled_spearman_abs_diff"] <= 1e-7


def test_arm_a_matches_frozen_summary(factorial, frozen_summary):
    """A 臂统计量与冻结 validation_summary.json 逐字段一致（复刻忠实性）。"""
    a = factorial["variants"]["A"]
    assert a["RA"] == pytest.approx(frozen_summary["RA"], abs=1e-12)
    assert a["RA_ci95"][0] == pytest.approx(frozen_summary["RA_ci95"][0], abs=1e-9)
    assert a["RA_ci95"][1] == pytest.approx(frozen_summary["RA_ci95"][1], abs=1e-9)
    assert a["stratified_median_mode"] == pytest.approx(
        frozen_summary["stratified_median_mode"], abs=1e-9)
    assert a["best_scalar_name"] == frozen_summary["best_scalar_name"]
    assert a["delta_mode_vs_best_scalar"] == pytest.approx(
        frozen_summary["delta_mode_vs_best_scalar"], abs=1e-12)
    assert a["delta_mode_ci95"][0] == pytest.approx(
        frozen_summary["delta_mode_ci95"][0], abs=1e-9)
    assert factorial["display"]["A"]["stratified_median_mode"] == 0.536


def test_arm_d_primary_result_stable(factorial):
    """主结果（within-cell mode ranking）在 corrected 口径下保持：
    R_A = 0.90、≥65/66 正 cells、11/11 正 objects。"""
    d = factorial["variants"]["D"]
    assert d["RA"] == pytest.approx(0.90, abs=5e-3)
    assert d["n_pos_cells"] >= 65
    assert d["n_pos_objects"] == 11
    assert d["RA_ci95"][0] > 0.5          # 仍明显为正（D2 分支：CI 变宽）


def test_stratified_flip_recorded(factorial):
    """stratified 关联在 corrected 噪声拟合下翻转——诚实记录的字段级证据：
    A > 0（复现冻结 +0.536），D < 0，且翻转由噪声拟合驱动（B 翻、C 不翻）。"""
    v = factorial["variants"]
    assert v["A"]["stratified_median_mode"] > 0.5
    assert v["D"]["stratified_median_mode"] < 0
    assert v["B"]["stratified_median_mode"] < 0       # 噪声拟合单独已翻转
    assert v["C"]["stratified_median_mode"] > 0.3     # 投影修正单独不翻转


def test_display_matches_variants(factorial):
    """README 引用的 display 值必须由 variants 字段四舍五入而来（claim 链）。"""
    for arm in ("A", "B", "C", "D"):
        disp = factorial["display"][arm]
        var = factorial["variants"][arm]
        assert disp["RA"] == round(var["RA"], 2)
        assert disp["RA_ci95"] == [round(var["RA_ci95"][0], 2),
                                   round(var["RA_ci95"][1], 2)]
        assert disp["stratified_median_mode"] == round(
            var["stratified_median_mode"], 3)


# --------------------------------------------------------------- EOL-robust integrity
def test_frozen_artifact_sha_eol_robust():
    """configs/openillumination.yaml 记录的冻结产物哈希是 CRLF 工作树口径，
    而 .gitattributes 钉死 LF 检出——裸 sha256 在任何规范检出不匹配
    （severity 脚本在新鲜 clone 上曾必然失败）。修复后的 EOL-鲁棒校验
    （LF/CRLF 双候选、内容篡改仍零容忍）必须对本仓库的两个冻结产物通过。"""
    import sys

    sys.path.insert(0, str(REPO))
    import yaml

    from experiments.openillumination_severity import _eol_robust_sha_matches

    cfg = yaml.safe_load(
        (REPO / "configs/openillumination.yaml").read_text(encoding="utf-8"))
    checked = 0
    for f, h in cfg["frozen_artifacts_sha256"].items():
        path = REPO / f
        if not path.exists():
            continue
        assert _eol_robust_sha_matches(str(path), h), f
        checked += 1
    assert checked == 2          # ci04_formal_summary.json + ci04_formal_manifest.json
    # 篡改检测仍然有效：改一个字节 → 两个候选都不匹配
    import hashlib

    raw = (REPO / "results/openillumination/ci04_formal_summary.json").read_bytes()
    tampered = hashlib.sha256(raw + b"x").hexdigest()
    assert not _eol_robust_sha_matches(
        str(REPO / "results/openillumination/ci04_formal_summary.json"), tampered)
