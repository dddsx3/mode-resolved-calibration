"""MF-0.6 · frozen allocation states 的秩不变检查（数学冻结 v1.0 §57 MF-0.6）。

对**冻结 allocation 状态**（frozen selection_orders.json 的完整 142 灯排序 +
冻结 rng 规程重建的 SelectionState）检查：ΔF 的数值秩 / 正子空间维是否跨
budget 前缀保持。结论决定论文命名（v1.0 §36/§57）：
  - 秩保持 ⇒ 正文可称 E/A/D-optimal baselines；
  - 秩变化 ⇒ 必须改称 "implemented pseudo-A / positive-subspace D criteria"。

需要原始 11 对象数据（与冻结 allocation run 相同的 data_root/meta）；
无数据时退出并提示。输出（新目录，不触碰冻结产物）：
  results/openillumination/correctness/allocation_rank_check.json

用法：
  python experiments/allocation_rank_check.py \
      [--data D:/data/OpenIllumination] [--meta D:/data/OpenIllumination_meta]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.policies import SelectionState              # noqa: E402
from calibinfo.allocation.rank_invariance import (                    # noqa: E402
    rank_invariance_report)
from calibinfo.datasets.openillumination import load_object           # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator           # noqa: E402
from experiments.openillumination_allocation import (                 # noqa: E402
    BUDGET_COUNTS, LEVELS, REGIMES)

ORDERS = REPO / "results/openillumination/allocation/selection_orders.json"
OUT = REPO / "results/openillumination/correctness/allocation_rank_check.json"
POLICIES = ("mode_aware", "e_opt", "a_opt", "d_opt",
            "random_0", "random_1", "random_2", "random_3", "random_4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="D:/data/OpenIllumination")
    ap.add_argument("--meta", default="D:/data/OpenIllumination_meta")
    ap.add_argument("--regimes", default="10", help="comma list, e.g. 10,100")
    args = ap.parse_args()
    regimes = [int(r) for r in args.regimes.split(",")]

    cfg = yaml.safe_load(
        (REPO / "configs/openillumination_allocation.yaml").read_text(
            encoding="utf-8"))
    cohort = list(cfg["cohort"])
    orders = json.loads(ORDERS.read_text(encoding="utf-8"))

    results = {}
    all_invariant = True
    for obj_idx, obj_name in enumerate(cohort):
        obj = load_object(args.data, obj_name, data_meta=args.meta)
        # 冻结 allocation run 的逐位 rng 规程
        scen = NominalScene_for_state(obj, obj_idx)
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        finf = scen.Finf_diag
        K = cfg["K_lights"]
        u142 = np.zeros((K, finf.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        lam0142 = np.stack([np.eye(3)] * K)
        u142[scen.sel] = u_act
        M0142[scen.sel] = M0_act
        active142 = np.zeros(K, bool)
        active142[scen.sel] = True

        obj_res = {}
        for level in LEVELS:
            gen = CorruptionGenerator("joint", level)
            base_lam0 = np.linalg.inv(gen.sigma_phi_diag())
            lam0142[scen.sel] = base_lam0
            state = SelectionState(u=u142, M0=M0142, lam0=lam0142.copy(),
                                   finf=finf, active=active142)
            for regime in regimes:
                for pol in POLICIES:
                    key = f"{obj_name}|{level}|{regime}|{pol}"
                    ordr = orders.get(key)
                    if ordr is None:
                        continue
                    rep = rank_invariance_report(state, ordr, regime=float(regime),
                                                 budget_counts=BUDGET_COUNTS)
                    obj_res[key] = dict(
                        rank_invariant=rep["rank_invariant"],
                        ranks={str(k): v for k, v in rep["ranks"].items()},
                        pos_dims={str(k): v for k, v in rep["pos_dims"].items()},
                        finf_rank=rep["finf_rank"],
                        lam_min_min=float(min(rep["lam_min"].values())))
                    all_invariant &= rep["rank_invariant"]
        results[obj_name] = obj_res
        n_inv = sum(1 for v in obj_res.values() if v["rank_invariant"])
        print(f"[rank-check] {obj_name}: {n_inv}/{len(obj_res)} orderings "
              f"rank-invariant", flush=True)

    rank_values = sorted({str(v["ranks"]["0"]) for o in results.values()
                          for v in o.values()})
    pos_values = sorted({str(v["pos_dims"]["0"]) for o in results.values()
                         for v in o.values()})
    summary = dict(
        gate="MF-0.6 allocation rank invariance on frozen states",
        analysis_status="mf0_rank_invariance_check",
        all_rank_invariant=bool(all_invariant),
        distinct_base_ranks=rank_values,
        distinct_base_pos_dims=pos_values,
        finf_ranks=sorted({str(v["finf_rank"]) for o in results.values()
                           for v in o.values()}),
        budget_counts=BUDGET_COUNTS, levels=LEVELS, regimes=regimes,
        n_orderings=sum(len(o) for o in results.values()),
        per_object=results,
        conclusion=("rank preserved across all budget prefixes and orderings: the "
                    "working subspace is common and the E/A/D-optimal naming is "
                    "licensed (A/D implemented on the positive subspace, which "
                    "coincides with the full working subspace)"
                    if all_invariant else
                    "RANK CHANGED for some state: the classical baselines must be "
                    "named 'implemented pseudo-A / positive-subspace D criteria'"),
        note="frozen selection_orders.json orderings; SelectionState rebuilt with "
             "the frozen rng spec (default_rng([20260910, obj_idx])) and the "
             "per-level base precision (wiring gate F1)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"[rank-check] all_rank_invariant = {all_invariant} -> {OUT}")


def NominalScene_for_state(obj, obj_idx):
    """冻结 allocation run 的场景构建（rng 规程逐位一致）。"""
    from experiments.openillumination_validation import NominalScene
    return NominalScene(obj, np.random.default_rng([20260910, obj_idx]))


if __name__ == "__main__":
    main()
