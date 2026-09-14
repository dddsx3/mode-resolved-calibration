"""P-REUSE-EQUIV · v1→v1.1 未变单元的位级等价验证(重用决定的审计证据)。

背景:v1.1 网格(4 informed + 3U/2A48)相对 v1(2 informed + 5U)只新增了
e_opt/d_opt 与 active48 随机;mode_aware、a_opt、randomU_0..2(= v1 的
random_0..2,同一 rng 规范)的计算路径逐字节未动。本脚本把"可复用/全量
重跑等价"的论断变成机器可查的证据:从 **git 提交历史** 读 v1 产物(不读
工作树——它可能已被后续运行覆盖),在抽样点上完整重推导,断言**位级一致**。

抽样(obj_03_pumpkin, regime 10, level ∈ {0.2, 0.5}):
  - pred_J_A:mode_aware / a_opt / randomU_0 × k ∈ {14, 57} × 2 level = 12 检查
    (重算排序 + 同状态 J_A;确定性,无种子)
  - 实测端点:mode_aware × k ∈ {14, 57} × {ang_mean_deg, mse_aligned,
    dual_mean} @ level 0.2 = 6 检查(10 seeds 冻结种子规范;dual 按 v1 的
    聚合顺序:逐 arm mean-of-5,再对 seeds 求 mean——之前一次 ad-hoc
    "失配"正是验证方聚合顺序不同所致,本脚本全程用 v1 顺序)

任何一位不一致即非零退出。输出:
  results/openillumination/provenance/dq_v1_reuse_equivalence.json

用法:
  python experiments/verify_dq_v1_reuse_equivalence.py \
      [--v1-ref fc38e9d] \
      [--out results/openillumination/provenance/dq_v1_reuse_equivalence.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.corruption import raw_innovations           # noqa: E402
from calibinfo.allocation.policies import (                           # noqa: E402
    SelectionState, budget_scales, select_ordering,
    select_ordering_mode_aware)
from calibinfo.allocation.convex import CertificateProblem            # noqa: E402
from calibinfo.allocation.blocks import LightBlocks                   # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator           # noqa: E402
from experiments.allocation_mode_tail import arm_metrics              # noqa: E402
from experiments.active_set_ablation import _load_scene               # noqa: E402

V1_PATH = "results/openillumination/decision_quality.json"


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def run(v1_ref="fc38e9d",
        out_path=REPO / "results/openillumination/provenance/"
                        "dq_v1_reuse_equivalence.json"):
    t_start = time.time()
    # ---- v1 产物从 git 历史读(免疫工作树覆盖)----
    v1_bytes = subprocess.run(
        ["git", "show", f"{v1_ref}:{V1_PATH}"], capture_output=True,
        cwd=str(REPO), check=True).stdout
    v1 = json.loads(v1_bytes)
    assert v1["analysis_status"] == "decision_quality_v1"
    v1rows = {(r["object"], r["level"], r["regime"], r["unit"], r["k"]): r
              for r in v1["rows"]}

    cfg = dict(data_root="D:/data/OpenIllumination",
               data_meta="D:/data/OpenIllumination_meta",
               noise_fit_convention="corrected")
    obj_idx, obj_name, regime = 0, "obj_03_pumpkin", 10
    levels = [0.2, 0.5]                       # v1 levels 列表 = lv_i 0 / 1
    K = 142
    checks = []

    scen = _load_scene(cfg, obj_idx, obj_name)
    for level in levels:
        lv_i = levels.index(level)
        gen = CorruptionGenerator("joint", level)
        sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
        _deg, W_dual = scen.predicted_degradation(
            gen.sigma_phi_diag(), mode_coordinate="dual")
        u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        lam0142 = np.stack([np.eye(3)] * K)
        u142[scen.sel] = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0142[scen.sel] = np.einsum(
            "kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
        lam0142[scen.sel] = np.linalg.inv(gen.sigma_phi_diag())
        act142 = np.zeros(K, bool)
        act142[scen.sel] = True
        state = SelectionState(u=u142, M0=M0142, lam0=lam0142.copy(),
                               finf=scen.Finf_diag, active=act142)
        prob = CertificateProblem(
            blocks=LightBlocks(u142, M0142, lam0142.copy(),
                               scen.Finf_diag, act142),
            kappa=float(regime))

        ords = {"mode_aware": select_ordering_mode_aware(
                    state, float(regime))[0],
                "a_opt": select_ordering(state, "a_opt", float(regime))[0]}
        rperm = np.random.default_rng([20260911, obj_idx, 0, 0])
        ords["random_0"] = select_ordering(state, "random", rng=rperm)[0]

        # ---- pred_J_A 检查(12)----
        for unit in ("mode_aware", "a_opt", "random_0"):
            for k in (14, 57):
                t = np.ones(K)
                t[np.asarray(ords[unit][:k])] = float(regime)
                got = prob.J_A(t)
                ref = v1rows[(obj_name, level, regime, unit, k)]["pred_J_A"]
                checks.append(dict(
                    kind="pred_J_A", level=level, unit=unit, k=k,
                    recomputed=repr(got), v1=repr(ref),
                    bit_exact=bool(got == ref)))

        # ---- 实测端点检查(level 0.2,6)----
        if level == 0.2:
            for k in (14, 57):
                acc = dict(ang=[], mse=[], dual=[])
                for s_i in range(10):
                    z_rng = np.random.default_rng(
                        [20260910, 7777, obj_idx, lv_i, s_i])
                    raw = raw_innovations(z_rng, len(scen.sel))
                    scales = budget_scales(ords["mode_aware"], k,
                                           regime)[scen.sel]
                    m = arm_metrics(scales, raw, scen, sig_logI,
                                    sig_rad, W_dual)
                    acc["ang"].append(m["ang_mean_deg"])
                    acc["mse"].append(m["mse_aligned"])
                    acc["dual"].append(float(np.mean(m["dual"])))
                ref = v1rows[(obj_name, level, regime, "mode_aware", k)]
                for ep in ("ang_mean_deg", "mse_aligned", "dual_mean"):
                    got = float(np.mean(acc[ep.split("_")[0]]))
                    rv = ref[ep]
                    checks.append(dict(
                        kind="arm_mean", level=level, unit="mode_aware",
                        k=k, endpoint=ep, recomputed=repr(got), v1=repr(rv),
                        bit_exact=bool(got == rv)))
        print(f"[equiv] level {level}: "
              f"{sum(c['bit_exact'] for c in checks)}/{len(checks)} checks "
              f"bit-exact so far", flush=True)

    n_ok = sum(c["bit_exact"] for c in checks)
    verdict = "pass" if n_ok == len(checks) else "FAIL"
    out = dict(
        gate="P-REUSE-EQUIV",
        analysis_status="dq_v1_reuse_equivalence_v1",
        question="are the v1->v1.1 unchanged selection units "
                 "(mode_aware, a_opt, randomU_0..2 = v1 random_0..2, "
                 "same rng spec) bit-exact reproducible from the committed "
                 "v1 artifact, so their rows are reusable and a full v1.1 "
                 "rerun cross-validates instead of duplicating?",
        method="re-derive orderings + J_A + 10-seed arm means on the "
               "sample points from scratch and compare to the v1 rows "
               "read from git history (immune to working-tree overwrites); "
               "dual uses the v1 aggregation order (per-arm mean-of-5, "
               "then mean over seeds)",
        v1_ref=v1_ref,
        v1_artifact_sha256=hashlib.sha256(v1_bytes).hexdigest(),
        sample=dict(object=obj_name, regime=regime, levels=levels,
                    units=["mode_aware", "a_opt", "random_0"],
                    pred_ks=[14, 57], arm_level=0.2, seeds=10),
        n_checks=len(checks), n_bit_exact=n_ok, verdict=verdict,
        checks=checks,
        manifest=dict(git_sha=_git_sha(),
                      elapsed_s=round(time.time() - t_start, 1)),
        note="seed-determinism: these units' selection, J_A, and arm code "
             "paths are byte-identical between the v1 and v1.1 commits, "
             "and all per-unit seeds carry object+level+seed indices - "
             "the values are worker-count invariant. Evidence for the "
             "reuse decision recorded in d877355's commit trail.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(out, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[equiv] {n_ok}/{len(checks)} bit-exact -> {verdict}; "
          f"wrote {out_path}")
    if verdict != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-ref", default="fc38e9d")
    ap.add_argument("--out", default=str(
        REPO / "results/openillumination/provenance/"
               "dq_v1_reuse_equivalence.json"))
    a = ap.parse_args()
    run(a.v1_ref, a.out)
