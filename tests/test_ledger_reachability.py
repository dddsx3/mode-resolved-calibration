"""P0-e · 台账机器校验(REPRODUCIBILITY §1.1 的"账实相符"自动化)。

背景(验收 P0-e):两个 provenance 记录的 manifest.git_sha 指向一个
**不可达提交**(生成器跑完后 HEAD 被 amend,SHA 成孤儿),而台账的
"Reachable from HEAD"列(手工维护)仍声明 yes——这是 58366b8 之后的
第二次同类事故。根因:该列没有任何东西校验它。

本测试把台账从散文变成机器断言:
  1. 解析 REPRODUCIBILITY.md 的表格行(sha12, reachable_claimed);
  2. 每行 `git merge-base --is-ancestor <sha> HEAD` 实测可达性,
     断言与声明一致;
  3. results/** 中每个 12 位 git_sha 都必须在台账里(防漏登记);
  4. 头部 "N distinct" 与实际不同 SHA 数一致。

防 ghost 的机制纪律(已入 CONTRIBUTING):生成器跑完后不要再 amend/
rebase;若必须动,重跑生成器。本测试是事后审计的自动化。
"""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "docs/REPRODUCIBILITY.md"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=str(REPO))


def ledger_rows():
    """[(sha12, reachable_claimed)] —— 只取 §1.1 表格的数据行。"""
    rows = []
    in_table = False
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.startswith("| git_sha"):
            in_table = True
            continue
        if in_table:
            if line.startswith("|---"):
                continue                  # 表头分隔行
            if not line.startswith("| `"):
                break                     # 表格结束
            m = re.match(r"\| `([0-9a-f]{12})` \|[^|]+\| *([^|]+)\|", line)
            if m:
                rows.append((m.group(1), m.group(2).strip().lower()))
    assert rows, "ledger table not found"
    return rows


def test_ledger_reachability_matches_claims():
    """每行 'yes/no (ancestor...)' 声明与 git 实测一致。"""
    for sha, claimed in ledger_rows():
        rc = _git("merge-base", "--is-ancestor", sha, "HEAD").returncode
        actual = (rc == 0)
        claimed_yes = claimed.startswith("yes")
        assert actual == claimed_yes, (
            f"ledger row {sha}: claims '{claimed}' but "
            f"merge-base --is-ancestor -> {rc} "
            f"({'reachable' if actual else 'UNREACHABLE (ghost sha?)'}); "
            f"if the generating commit was amended, RERUN the generator "
            f"so the record pins the reachable commit")


def test_results_git_shas_all_in_ledger():
    """results/** 的每个 12 位 git_sha 必须在台账(防漏登记)。"""
    out = _git("grep", "-rho", r'"git_sha": *"[0-9a-f]\{12\}', "results")
    found = {m.group(1) for m in re.finditer(r'"([0-9a-f]{12})', out.stdout)}
    ledger = {sha for sha, _ in ledger_rows()}
    missing = found - ledger
    assert not missing, f"results git_shas missing from ledger: {missing}"


def test_ledger_distinct_count_consistent():
    """头部 'N distinct' = 台账行不同 SHA 数 = results 实际数。"""
    text = LEDGER.read_text(encoding="utf-8")
    m = re.search(r"([\w-]+) distinct `git_sha` values are recorded", text)
    assert m, "header count sentence not found"
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
             "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
    # 复合数词(twenty-one):逐段求和
    parts = re.split(r"[-\s]+", m.group(1).lower())
    claimed = 0
    for part in parts:
        v = words.get(part)
        assert v is not None, f"unparsed count word: {part}"
        claimed += v
    assert claimed is not None, f"unparsed count word: {m.group(1)}"
    ledger_distinct = len({sha for sha, _ in ledger_rows()})
    assert ledger_distinct == claimed, \
        f"header says {claimed} distinct, table has {ledger_distinct}"
    out = _git("grep", "-rho", r'"git_sha": *"[0-9a-f]\{12\}', "results")
    found = len({mm.group(1)
                 for mm in re.finditer(r'"([0-9a-f]{12})', out.stdout)})
    assert found == ledger_distinct, \
        f"results has {found} distinct git_shAs, ledger {ledger_distinct}"
