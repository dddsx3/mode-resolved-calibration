"""P-CROSS-ENV · decision-quality 跨环境复现对拍(P2-1 的诚实版)。

比较同一网格的两份产物(典型:本地 2-workers vs 云端 11-workers,不同
CPU/numpy/OpenBLAS),输出分层报告并落盘证据 JSON:

  1) 结构:行数/AUC cells/bootstrap 键数/随机拆分;
  2) 行级:四量(pred_J_A, ang, mse, dual)的逐位一致数、容差内数、
     最大绝对/相对差;按 (cell, unit) 归因 pred_J_A 发散(>1e-9 rel)
     的组合——贪心近平局翻转的定位;
  3) 结论稳健性:16 个 ang bootstrap CI 的符号与显著性是否两侧一致;
     informed 统计是否逐位一致。

判定语义(诚实):跨机对拍**不期望逐位一致**——eigh/inv/pinv 的末位
ULP 差异会在贪心近平局处翻转选择。verdict = 'statistically-identical'
当且仅当:结构一致、发散 cell-unit 占比 < 5%、全部 ang CI 同号且显著、
informed 统计逐位一致。

用法:
  python experiments/verify_dq_cross_env.py \
      --a results/openillumination/decision_quality.json \
      --b <cloud artifact path> \
      [--out results/openillumination/provenance/dq_cross_env_repro.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FIELDS = ("pred_J_A", "ang_mean_deg", "mse_aligned", "dual_mean")


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(path_a: Path, path_b: Path):
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))
    ra = {(x["object"], x["level"], x["regime"], x["unit"], x["k"]): x
          for x in a["rows"]}
    rb = {(x["object"], x["level"], x["regime"], x["unit"], x["k"]): x
          for x in b["rows"]}
    keys_match = set(ra) == set(rb)
    n_exact = n_tol = n_total = 0
    max_abs = {f: 0.0 for f in FIELDS}
    max_rel = {f: 0.0 for f in FIELDS}
    div = set()
    for key in (set(ra) & set(rb)):
        for f in FIELDS:
            va, vb = ra[key][f], rb[key][f]
            n_total += 1
            if va == vb:
                n_exact += 1
            else:
                d = abs(va - vb)
                s = max(abs(va), abs(vb), 1e-300)
                max_abs[f] = max(max_abs[f], d)
                max_rel[f] = max(max_rel[f], d / s)
                if d / s <= 1e-9:
                    n_tol += 1
        rel_p = abs(ra[key]["pred_J_A"] - rb[key]["pred_J_A"]) / max(
            abs(ra[key]["pred_J_A"]), 1e-300)
        if rel_p > 1e-9:
            div.add(key[:4])                      # (obj, lv, rg, unit)
    n_cell_unit = len({k[:4] for k in ra})
    # 结论稳健性
    ci_ok = True
    for k in a["bootstrap_dAUC"]:
        if "ang_mean_deg" not in k:
            continue                     # 只审 ang 键;mse/dual 键跳过
        if k not in b["bootstrap_dAUC"]:
            ci_ok = False
            break
        ca, cb = a["bootstrap_dAUC"][k], b["bootstrap_dAUC"][k]
        if not ((ca["median_dAUC"] < 0) == (cb["median_dAUC"] < 0)
                and ((ca["ci95"][1] < 0) or (ca["ci95"][0] > 0))
                and ((cb["ci95"][1] < 0) or (cb["ci95"][0] > 0))):
            ci_ok = False
            break
    informed_same = (a["informed_pairwise_sign_pooled"]
                     == b["informed_pairwise_sign_pooled"])
    spearman_same = (a["spearman_pred_vs_realized"]
                     == b["spearman_pred_vs_realized"])
    struct_ok = (a["analysis_status"] == b["analysis_status"]
                 and len(a["rows"]) == len(b["rows"])
                 and len(a["auc_rows"]) == len(b["auc_rows"])
                 and len(a["bootstrap_dAUC"]) == len(b["bootstrap_dAUC"])
                 and a["random_units"] == b["random_units"])
    verdict = ("statistically-identical"
               if (keys_match and struct_ok and ci_ok and informed_same
                   and spearman_same and len(div) / max(n_cell_unit, 1) < 0.05)
               else "DIVERGED")
    return dict(
        struct_ok=bool(struct_ok and keys_match),
        rows=dict(n_total=n_total, n_bit_exact=n_exact, n_within_1e9=n_tol,
                  max_abs={f: f"{v:.3e}" for f, v in max_abs.items()},
                  max_rel={f: f"{v:.3e}" for f, v in max_rel.items()}),
        cell_units=dict(n_total=n_cell_unit, n_diverged=len(div),
                        diverged=[list(d) for d in sorted(div)]),
        conclusions=dict(
            all_ang_ci_same_sign_and_significant=bool(ci_ok),
            informed_pooled_identical=bool(informed_same),
            spearman_summaries_identical=bool(spearman_same)),
        verdict=verdict)


def run(path_a, path_b, out_path):
    t0 = time.time()
    path_a, path_b = Path(path_a), Path(path_b)
    res = compare(path_a, path_b)
    rec = dict(
        gate="P-CROSS-ENV",
        analysis_status="dq_cross_env_repro_v1",
        question="does the decision-quality grid reproduce across "
                 "environments (local 2-workers AMD/numpy-2.4.1 vs cloud "
                 "11-workers x86/numpy-2.4.6)? Expectation: NOT bit-exact "
                 "(last-ULP BLAS differences flip greedy near-ties); the "
                 "registered claim is statistical identity",
        method="layered comparison: structure, row-level four-quantity "
               "tolerances, (cell,unit) divergence attribution via "
               "pred_J_A > 1e-9 rel, and conclusion robustness (16 ang "
               "CI signs+significance, informed pooled, spearman "
               "summaries)",
        inputs=dict(
            a=str(path_a), a_sha256=_sha(path_a),
            b=str(path_b), b_sha256=_sha(path_b)),
        results=res,
        manifest=dict(git_sha=_git_sha(),
                      elapsed_s=round(time.time() - t0, 1)),
        note="cloud package: decision_quality_42bc299f_20260914_120403.zip "
             "(RUN_INFO: git_sha=launch=42bc299..., config_sha256=b76eafa..., "
             "elapsed 2183.5s = 36.4 min on 32 cores / 11 workers, "
             "Linux-6.6.98 tl4, numpy 2.4.6). The cloud artifact is archived "
             "alongside this record for full auditability. Same-machine "
             "worker-invariance is proven separately (resume-vs-clean "
             "bit-exact test in the 42bc299 commit trail); this record "
             "covers the CROSS-environment property.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(rec, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(json.dumps(res, indent=1))
    print(f"verdict: {res['verdict']}; wrote {out_path}")
    if res["verdict"] == "DIVERGED":
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(
        REPO / "results/openillumination/decision_quality.json"))
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=str(
        REPO / "results/openillumination/provenance/dq_cross_env_repro.json"))
    x = ap.parse_args()
    run(x.a, x.b, x.out)
