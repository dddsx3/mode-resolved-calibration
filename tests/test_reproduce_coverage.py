"""P-REPRODUCE-COVERAGE · 复现入口的覆盖门禁(验收 665d01e §5)。

背景:旧的 `reproduce_paper.sh` 自称 "full reproduction chain",但实测
只覆盖早期合成实验——论文的 headline 产物(C8/C10/C11/C12/B9)一条都不在,
且没有任何文档引用它。""一个能跑通但不覆盖结论的入口,比一个明确标注
partial 的入口更糟""(artifact reviewer 的第一个动作就是跑它)。

本门禁把它变成机器断言(与 CONTRIBUTING 第 5 条同精神:规则要么全局
执行,要么不写):
  1. `docs/REPRODUCIBILITY.md` 复现矩阵里的每条
     `python experiments/X.py` 命令,其脚本必须在 reproduce_paper.sh 的
     runner 表中出现;
  2. runner 表里登记的所有产物路径必须实际存在于仓库;
  3. README(en)必须引用 reproduce_paper.sh(防止再次零引用)。
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "reproduce_paper.sh"
MATRIX = REPO / "docs/REPRODUCIBILITY.md"
README = REPO / "README.md"


def _script_text():
    if not SCRIPT.exists():
        import pytest
        pytest.skip("reproduce_paper.sh not present")
    return SCRIPT.read_text(encoding="utf-8")


def test_matrix_commands_covered():
    """复现矩阵中的每条实验命令都在 runner 表里(或显式豁免)。"""
    text = _script_text()
    matrix = MATRIX.read_text(encoding="utf-8")
    cmds = set(re.findall(r"python (experiments/[\w/]+\.py)", matrix))
    assert cmds, "no experiment commands found in the reproduction matrix"
    # 已登记的独立复现脚本(矩阵引用)必须被入口覆盖
    missing = sorted(c for c in cmds
                     if c not in text)
    assert not missing, (
        f"reproduce_paper.sh does not run these matrix experiments: "
        f"{missing}")


def test_runner_artifacts_exist():
    """runner 表登记的产物路径必须实际存在(打字错误即失败)。"""
    text = _script_text()
    tbl = re.search(r"RUNNERS_TABLE\(\) \{.*?cat <<'TABLE'\n(.*?)\nTABLE",
                    text, re.S)
    assert tbl, "RUNNERS_TABLE block not found"
    paths = []
    for line in tbl.group(1).splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            paths.append(parts[2])
    assert paths, "runner table empty"
    missing = [p for p in paths if not (REPO / p).exists()]
    assert not missing, f"runner artifacts missing from the repo: {missing}"


def test_readme_references_the_entry_point():
    """README(en)必须引用 reproduce_paper.sh(防零引用回归)。"""
    readme = README.read_text(encoding="utf-8")
    assert "reproduce_paper.sh" in readme, (
        "README does not reference reproduce_paper.sh -- the reproduction "
        "entry point must be discoverable (acceptance 665d01e section 5)")


def test_diag_stage_is_zero_cost():
    """diag 阶段只含零成本诊断(读已提交产物),不得含需要原始数据的步骤。"""
    text = _script_text()
    diag_block = text.split('_CURRENT_STAGE="diag"')[1].split(
        '_CURRENT_STAGE="oi"')[0]
    # diag 阶段的 requires 全部为 none
    reqs = re.findall(r"results/[\w/\.\-]+\.json (none|oi|dq|oidq) \d", diag_block)
    assert reqs, "diag block has no run_step entries"
    assert all(r == "none" for r in reqs), (
        f"diag stage contains data-dependent steps: {reqs}")
