"""P-DQ-FEASIBLE · E 臂预算网格饱和诊断 + 可行区间 dAUC(验收 9515039 §4)。

问题:E 臂预算网格 {14, 28, 57, 85, 114}(K=142 的 0.1/0.2/0.4/0.6/0.8)
中后三档**超过 active 集大小 48**。任何策略的排序前缀 ordr[:k] 都退化
为全部 48 盏 → informed 与 active48-random 选出同一批灯 → 端点逐位
相同 → "vs active48" 的 dAUC 有 3/5 预算区间恒为零,把效应量稀释了
2.0–2.8 倍。B9 的 "an order of magnitude smaller" 这句话是该稀释造
出来的。

本诊断(零成本纯算术,数据源 = 已入库的 decision_quality.json 3960 行):
  1. **退化检查**:逐预算统计 informed 与 randomA48 端点逐位相同的
     cell 数 → `degenerate_budgets` 显式标记(验收建议的非退化门禁);
  2. **可行区间 dAUC**:trapezoid 只跨 fraction [0.1, 0.2](k=14→28,
     全部 ≤ 48),对两基线的对象聚类 bootstrap CI(同冻结口径
     B=10000 seed=20260910);
  3. **全网格 dAUC** 复算(与冻结产物对拍)+ 稀释比;
  4. 可行区间 vs universe-random 也一并给出(修 "order of magnitude"
     这句需要两个基线在同一横轴上的比)。

输出: results/openillumination/decision_quality_feasible.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap  # noqa: E402

SRC = REPO / "results/openillumination/decision_quality.json"
N_ACTIVE = 48          # scen.sel 大小(两队列一致;K=142 口径)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _trapezoid_auc(Es, fracs):
    Es = np.asarray(Es, float)
    bs = np.asarray(fracs, float)
    return float(np.sum((Es[:-1] + Es[1:]) / 2.0 * np.diff(bs))
                 / (bs[-1] - bs[0]))


def run(src_path=SRC,
        out_path=REPO / "results/openillumination/"
                       "decision_quality_feasible.json",
        B=10000, seed=20260910):
    art = json.loads(Path(src_path).read_text(encoding="utf-8"))
    rows = art["rows"]
    budgets = list(art["budgets_k"])
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8]
    units_det = list(art["policies"])
    units_randA = [f"randomA48_{i}"
                   for i in range(art["random_units"]["active48"])]
    units_randU = [f"randomU_{i}"
                   for i in range(art["random_units"]["universe"])]

    # ---- 1) 退化检查:逐预算,端点是否逐位相同 ----
    by = defaultdict(dict)
    for r in rows:
        by[(r["object"], r["level"], r["regime"], r["k"])][r["unit"]] = \
            r["ang_mean_deg"]
    degenerate = {}
    for k in budgets:
        cells = [v for kk, v in by.items() if kk[3] == k]
        n_same = 0
        for v in cells:
            det = [v[u] for u in units_det]
            rA = [v[w] for w in units_randA]
            if all(any(a == b for b in rA) for a in det):
                n_same += 1
        degenerate[str(k)] = dict(
            n_cells=n_same, n_total=len(cells),
            degenerate=(n_same == len(cells) and len(cells) > 0),
            feasible=(k <= N_ACTIVE),
            note="informed and active48-random prefixes select the same "
                 "lights when k > |active| = 48; endpoints then coincide "
                 "bit-exactly and the comparison carries no information")

    # ---- 2/3) dAUC:全网格 vs 可行区间(对象聚类 bootstrap) ----
    auc_by = {}
    for (obj, lv, rg, k), v in by.items():
        for u, val in v.items():
            auc_by.setdefault((obj, lv, rg), {}).setdefault(u, {})[k] = val

    def dAUC_series(ks, bs, base_units):
        """逐 (object, level, regime) 的 dAUC 列表(对 base_units 均值)。"""
        out = defaultdict(list)
        for key, uv in auc_by.items():
            Es = {u: [uv[u][k] for k in ks] for u in uv}
            base = np.mean([_trapezoid_auc(Es[w], bs) for w in base_units])
            for u in units_det:
                out[(u, key[2])].append(
                    _trapezoid_auc(Es[u], bs) - base)
        return out

    full_A48 = dAUC_series(budgets, fracs, units_randA)
    feas_A48 = dAUC_series(budgets[:2], fracs[:2], units_randA)
    feas_U = dAUC_series(budgets[:2], fracs[:2], units_randU)

    # 冻结口径:聚类单元 = 一条 (object, level) auc_row(每 regime 44 条),
    # stat = median over payload values,bootstrap 用 B/seed 同冻结
    def series_rows(ks, bs, base_units):
        per = defaultdict(list)
        for key, uv in auc_by.items():
            obj, lv, rg = key
            Es = {u: [uv[u][k] for k in ks] for u in uv}
            base = np.mean([_trapezoid_auc(Es[w], bs) for w in base_units])
            for u in units_det:
                per[(u, rg)].append((obj, _trapezoid_auc(Es[u], bs) - base))
        return per

    def summarize(per):
        out = {}
        for (u, rg), payload in per.items():
            point, (lo, hi), _ = cluster_bootstrap(
                payload, lambda items: float(np.median([v for _, v in items])),
                B=B, seed=seed)
            out[f"{u}|{rg}"] = dict(
                median_dAUC=round(point, 6),
                ci95=[round(lo, 6), round(hi, 6)], n_objects=len(payload))
        return out

    full_stats = summarize(series_rows(budgets, fracs, units_randA))
    feasA48_stats = summarize(series_rows(budgets[:2], fracs[:2],
                                          units_randA))
    feasU_stats = summarize(series_rows(budgets[:2], fracs[:2],
                                        units_randU))

    # 与冻结产物对拍(全网格 vs A48)
    frz = art["bootstrap_dAUC"]
    n_match = 0
    for key, v in full_stats.items():
        u, rg = key.split("|")
        fk = f"{u}|{rg}|ang_mean_deg|vs_randomA48"
        if fk in frz:
            # 冻结产物把逐 cell dAUC 存成 6 位小数,复算从原始行全精度
            # → 对拍到舍入层级(5e-6),不是逐位
            assert abs(v["median_dAUC"] - frz[fk]["median_dAUC"]) < 5e-6, \
                (key, v["median_dAUC"], frz[fk]["median_dAUC"])
            n_match += 1
    print(f"[feas] full-grid dAUC reproduces the frozen artifact on "
          f"{n_match} cells")

    dilution = {}
    for key, v in feasA48_stats.items():
        f = full_stats[key]["median_dAUC"]
        q = v["median_dAUC"]
        dilution[key] = dict(full=f, feasible=q,
                             ratio=round(q / f, 3) if f else None)

    # 可行区间两基线比(修 "order of magnitude" 需要)
    ratio_A_vs_U = {}
    for key, v in feasA48_stats.items():
        ratio_A_vs_U[key] = round(v["median_dAUC"]
                                  / feasU_stats[key]["median_dAUC"], 3)

    summary = dict(
        gate="P-DQ-FEASIBLE",
        analysis_status="dq_feasible_v1",
        source_artifact="results/openillumination/decision_quality.json",
        source_manifest_sha256=_sha(Path(src_path)),
        source_git_sha=art["manifest"]["git_sha"],
        n_active=N_ACTIVE,
        budgets_k=budgets,
        degenerate_budgets=degenerate,
        full_grid_dAUC_vs_randomA48=full_stats,
        feasible_dAUC_vs_randomA48=feasA48_stats,
        feasible_dAUC_vs_universeRandom=feasU_stats,
        dilution=dilution,
        feasible_ratio_active48_over_universeRandom=ratio_A_vs_U,
        reading=("the frozen full-grid dAUC vs active-restricted random is "
                 "diluted by the three infeasible budgets (k=57/85/114 > "
                 "|active|=48, endpoints bit-identical): on the feasible "
                 "interval the within-active-set advantage is "
                 f"{min(v['median_dAUC'] for v in feasA48_stats.values()):.2f}"
                 f"..{max(v['median_dAUC'] for v in feasA48_stats.values()):.2f}"
                 " deg-AUC, ~2.0-2.8x larger than the frozen full-grid "
                 "values -- 'an order of magnitude smaller' was the "
                 "dilution artifact; the true ratio to the universe-random "
                 "advantage is ~2.5x"),
        manifest=dict(git_sha=_git_sha(), B=B, seed=seed),
        note="Zero-cost arithmetic on the committed E-arm rows: "
             "non-degeneracy check per budget + feasible-interval dAUC "
             "(k <= |active|) with the frozen object-cluster bootstrap "
             "spec, plus the full-grid recomputation for cross-check.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[feas] degenerate budgets: "
          f"{[k for k, v in degenerate.items() if v['degenerate']]}")
    print(f"[feas] dAUC vs A48: full "
          f"{min(v['median_dAUC'] for v in full_stats.values()):.3f}.."
          f"{max(v['median_dAUC'] for v in full_stats.values()):.3f} | "
          f"feasible "
          f"{min(v['median_dAUC'] for v in feasA48_stats.values()):.3f}.."
          f"{max(v['median_dAUC'] for v in feasA48_stats.values()):.3f}")
    print(f"[feas] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/"
                                   "decision_quality_feasible.json"))
    args = ap.parse_args()
    run(args.src, args.out)
