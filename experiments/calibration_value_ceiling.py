"""P-CEILING · universal ceiling on the calibration-budget value (H2).

Theorem (proved, machine-checked in tests/test_math_foundations.py): with
per-light precision multipliers t, U_k(t) = u_k(M0_k + t Lambda0_k)^{-1} u_k^T
is Loewner-decreasing in t. For a UNIFORM multiplier t . 1,

    M0_k + t Lambda0_k >= (1/t)(M0_k + Lambda0_k)
        <=>  (t-1) M0_k + (t^2 - 1) Lambda0_k >= 0,   t >= 1,  M0_k, Lambda0_k >= 0,

so U_k(t) <= t U_k(1) (Loewner), hence DeltaF(t) <= t DeltaF(1) and
J_A(t) >= J_A(1)/t (tr X^{-1} Loewner-decreasing). Therefore

    D = 1 - J_A(kappa . 1)/J_A(1)  <=  1 - 1/kappa        [P-CEILING]

The celebrated '90.0% at kappa=10' is this ceiling being MET: with the
identity precision block M0 dominant over Lambda0 (large level = small
Lambda0), the bound is tight to <= 1e-4.

This script regenerates the audit's level=64 saturation table with the
exact P-CERT assembly (uniform multiplier, corrected convention, 5 objects
x the kappa grid): every D(kappa) and its slack to the ceiling are
reported, no sign-based gate.

Output: results/magnitude/calibration_value_ceiling.json

Usage:
  python experiments/calibration_value_ceiling.py [--config ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.blocks import LightBlocks  # noqa: E402
from calibinfo.datasets.openillumination import load_object              # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator              # noqa: E402
from experiments.openillumination_validation import NominalScene         # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def build_state(scen, level: float, K: int) -> LightBlocks:
    """P-CERT 同款装配:142 灯全域、48 个 Fisher-active 灯。"""
    gen = CorruptionGenerator("joint", level)
    base_lam0 = np.linalg.inv(gen.sigma_phi_diag())
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi, scen.w[:, :, None] * scen.B_phi)
    u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    lam0142 = np.stack([np.eye(3)] * K)
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    lam0142[scen.sel] = base_lam0
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True
    return LightBlocks(u142, M0142, lam0142, scen.Finf_diag, active142)


def J_A_uniform(blk: LightBlocks, t: float) -> float:
    """J_A(t . 1) = tr DeltaF(t . 1)^{-1}(dense;P~1200 一次 inverse 可承受)。"""
    lam = np.stack([t * blk.lam0[k] for k in range(blk.L)])
    DF = blk.assemble(lam)
    return float(np.trace(np.linalg.inv(DF)))


def run(config_path=REPO / "configs/calibration_value_ceiling.yaml",
        out_path=REPO / "results/magnitude/calibration_value_ceiling.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa_grid = [float(x) for x in cfg["kappa_grid"]]
    level = float(cfg["level"])
    K = int(cfg["K_lights"])
    t_start = time.time()

    rows = []
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        blk = build_state(scen, level, K)
        J1 = J_A_uniform(blk, 1.0)
        for kap in kappa_grid:
            Jk = J_A_uniform(blk, kap)
            D = 1.0 - Jk / J1
            bound = 1.0 - 1.0 / kap
            rows.append(dict(object=obj_name, kappa=kap,
                             J_A_1=float(J1), J_A_kappa=float(Jk),
                             D=float(D), ceiling=float(bound),
                             slack=float(bound - D),
                             bounds_ok=bool(D <= bound + 1e-9)))
        print(f"[ceiling] {obj_name}: D @ kappa=10 = "
              f"{[r['D'] for r in rows if r['object'] == obj_name and r['kappa'] == 10][0]:.6f} "
              f"(ceiling 0.9)", flush=True)

    all_ok = all(r["bounds_ok"] for r in rows)
    # 汇总:每 κ 的 slack 统计(所有物体取最大 slack = 最接近上界的)
    by_kappa = {}
    for kap in kappa_grid:
        rs = [r for r in rows if r["kappa"] == kap]
        by_kappa[str(kap)] = dict(
            D_median=round(float(np.median([r["D"] for r in rs])), 6),
            D_min=round(float(np.min([r["D"] for r in rs])), 6),
            D_max=round(float(np.max([r["D"] for r in rs])), 6),
            slack_max=round(float(np.max([r["slack"] for r in rs])), 6),
            ceiling=1.0 - 1.0 / kap)

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        theorem="D = 1 - J_A(kappa)/J_A(1) <= 1 - 1/kappa (Loewner inverse bound)",
        level=level, kappa_grid=kappa_grid,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        rows=rows, by_kappa=by_kappa,
        all_bounds_ok=all_ok,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="Universal ceiling on the calibration-budget value: the "
             "'90.0% at kappa=10' headine is the ceiling 1-1/kappa being "
             "met at large level (identity precision block dominant), not "
             "an empirical saturation artifact. Every row reported; the "
             "only hard gate is the theorem inequality D <= 1-1/kappa.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                               .encode("utf-8"))
    print(f"[ceiling] all bounds ok: {all_ok}; max slack over rows: "
          f"{max(r['slack'] for r in rows):.4g}")
    print(f"[ceiling] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/calibration_value_ceiling.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/magnitude/calibration_value_ceiling.json"))
    args = ap.parse_args()
    run(args.config, args.out)