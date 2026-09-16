"""P-ABLATION-FEASIBLE · C8 源产物的预算轴退化标记(验收 665d01e §4/N-1)。

背景:""跨预算统计量必须在每个预算上检查双方是否退化重合""这条规则
(刚写入 CONTRIBUTING 第 5 条)没有应用到触发它的源产物本身。
`active_set_ablation.json`(C8 的证据)用预算 {5,10,14,28,48},**最大
那个恰好等于 |active| = 48**:在 k=48 处两侧的排序前缀都选满 48 盏灯,
`mode_ordering_gap` 恒为 0——**定义性零,不是测量结果**;而
`active_set_dilution` 在 k=48 仍非零(该预算对 dilution 项仍然有效)。
这是"部分退化":k>|active| 时整个比较死掉;k=|active| 时只有
**ordering 子项**死掉,需要各自的标记。

本诊断(零成本纯算术,数据源 = 已入库的 active_set_ablation.json 55 行):
  1. 逐预算退化标记(区分 ordering 子项退化 / dilution 项退化);
  2. 两种口径的 ordering/dilution 比:含 k=48(定义性零稀释)与剔除;
  3. 与 C8 引用的 1370× 口径对齐说明。

输出: results/openillumination/active_set_ablation_feasible.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

SRC = REPO / "results/openillumination/active_set_ablation.json"
N_ACTIVE = 48


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def run(src_path=SRC,
        out_path=REPO / "results/openillumination/"
                       "active_set_ablation_feasible.json"):
    art = json.loads(Path(src_path).read_text(encoding="utf-8"))
    rows = art["rows"]
    budgets = list(art["budgets_k"])

    # ---- 1) 逐预算退化标记 ----
    degenerate = {}
    for k in budgets:
        rs = [r for r in rows if r["k"] == k]
        gaps = [r["mode_ordering_gap"] for r in rs]
        dils = [r["active_set_dilution"] for r in rs]
        n_gap_zero = sum(1 for g in gaps if g == 0.0)
        # k = |active| → 排序前缀两侧选同一批灯,ordering 子项定义性零;
        # k > |active| 时会整项退化(本产物无此预算)
        ord_deg = (k >= N_ACTIVE) and (n_gap_zero == len(rs))
        degenerate[str(k)] = dict(
            n_rows=len(rs), n_ordering_gap_zero=n_gap_zero,
            ordering_subterm_degenerate=bool(ord_deg),
            dilution_term_degenerate=bool(k > N_ACTIVE),
            note=("k = |active| = 48: both ordering prefixes select all "
                  "48 lights, so mode_ordering_gap is definitionally 0 "
                  "(not a measurement); active_set_dilution remains "
                  "valid at this budget"
                  if k == N_ACTIVE else
                  "k < |active|: both terms measured normally"
                  if k < N_ACTIVE else
                  "k > |active|: the whole comparison is degenerate"))

    # ---- 2) 两种口径 ----
    def ratio(sel_k):
        gaps = [r["mode_ordering_gap"] for r in rows if r["k"] in sel_k]
        dils = [r["active_set_dilution"] for r in rows if r["k"] in sel_k]
        med_g = float(np.median(gaps))
        med_d = float(np.median(dils))
        return dict(median_ordering_gap=round(med_g, 10),
                    median_dilution=round(med_d, 6),
                    ratio=round(med_g / med_d, 10),
                    as_multiple=round(med_d / med_g, 1))

    with_48 = ratio(set(budgets))
    without_48 = ratio({k for k in budgets if k < N_ACTIVE})

    summary = dict(
        gate="P-ABLATION-FEASIBLE",
        analysis_status="ablation_feasible_v1",
        source_artifact=("results/openillumination/"
                         "active_set_ablation.json"),
        source_manifest_sha256=_sha(Path(src_path)),
        source_git_sha=art["manifest"]["git_sha"],
        n_active=N_ACTIVE,
        budgets_k=budgets,
        degenerate_budgets=degenerate,
        ratio_with_k48=with_48,
        ratio_without_k48=without_48,
        caliber_note=("C8's quoted ~1370x uses the full grid INCLUDING "
                      "k=48, where the ordering sub-term is "
                      "definitionally zero and therefore dilutes the "
                      "ratio; excluding k=48 the ratio is ~1001x. Both "
                      "are order-of-magnitude statements; the conclusion "
                      "is unchanged. The caliber must be stated because "
                      "the ordering sub-term at k=|active| carries no "
                      "information."),
        manifest=dict(git_sha=_git_sha()),
        note="Zero-cost arithmetic on the committed ablation rows: "
             "budget-axis degeneracy marking for the C8 decomposition "
             "(ordering sub-term dies at k=|active|=48; the dilution "
             "term stays valid there and dies only for k>48).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[abl-feas] ordering-gap zero rows at k=48: "
          f"{degenerate['48']['n_ordering_gap_zero']}/"
          f"{degenerate['48']['n_rows']}")
    print(f"[abl-feas] ratio with k=48: {with_48['ratio']:.2e} "
          f"({with_48['as_multiple']}x) | without: "
          f"{without_48['ratio']:.2e} ({without_48['as_multiple']}x)")
    print(f"[abl-feas] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/"
                                   "active_set_ablation_feasible.json"))
    args = ap.parse_args()
    run(args.src, args.out)
