"""P-CERT-LEVELS · certified dynamic range as a curve in `level` (H1).

The dynamic range of the calibration-budget value is a function of the
calibration-uncertainty operating point `level` — not a single number. This
script re-samples the same functional P-CERT reports (J_none vs J_all on
every active light, kappa=10) across an 11-point level grid, on the full
11-object cohort, with the exact P-CERT assembly (corrected convention,
142-light universe, 48 Fisher-active lights). The scene is assembled once
per object (`NominalScene` is level-independent; `level` only enters
Sigma_phi in `_build_state`), so the level sweep is cheap.

Relationship to P-CERT: level 0.5 is the P-CERT operating point; the median
D(0.5) here reproduces the certified 62.87% (uniform all-refined endpoint).
The curve saturates toward the universal ceiling 1-1/kappa = 0.9 at large
level (docs/methods.md section 7) — that ceiling is what bounds the
allocation value at strong calibration uncertainty.

Output: results/certification/certified_gaps_levels.json

Usage:
  python experiments/certified_gaps_levels.py [--config ...]
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

from calibinfo.allocation.blocks import LightBlocks               # noqa: E402
from calibinfo.datasets.openillumination import load_object       # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator       # noqa: E402
from experiments.openillumination_validation import NominalScene  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def build_state(scen, level: float, K: int) -> LightBlocks:
    """P-CERT 同款装配(`_build_state` 复制;level 只进 Σ_φ)。"""
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


def J_A(blk: LightBlocks, tvec: np.ndarray) -> float:
    lam = np.stack([tvec[k] * blk.lam0[k] for k in range(blk.L)])
    DF = blk.assemble(lam)
    return float(np.trace(np.linalg.inv(DF)))


def run(config_path=REPO / "configs/certified_gaps_levels.yaml",
        out_path=REPO / "results/certification/certified_gaps_levels.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    levels = [float(x) for x in cfg["levels"]]
    K = int(cfg["K_lights"])
    t_start = time.time()

    rows = []
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        t_one = np.ones(K)
        for lv in levels:
            blk = build_state(scen, lv, K)
            active = list(np.flatnonzero(blk.active))
            t_all = t_one.copy()
            t_all[active] = kappa
            J1 = J_A(blk, t_one)
            Jk = J_A(blk, t_all)
            D = (J1 - Jk) / J1
            rows.append(dict(object=obj_name, level=lv,
                             J_A_1=float(J1), J_A_kappa=float(Jk),
                             D=float(D),
                             ceiling=1.0 - 1.0 / kappa,
                             bounds_ok=bool(D <= 1.0 - 1.0 / kappa + 1e-9)))
        print(f"[levels] {obj_name}: D(0.1)={[r['D'] for r in rows if r['object'] == obj_name and r['level'] == 0.1][0]*100:.1f}%  "
              f"D(0.5)={[r['D'] for r in rows if r['object'] == obj_name and r['level'] == 0.5][0]*100:.1f}%  "
              f"D(8.0)={[r['D'] for r in rows if r['object'] == obj_name and r['level'] == 8.0][0]*100:.1f}%",
              flush=True)

    # 汇总:每 level 的中位/min/max
    by_level = {}
    for lv in levels:
        rs = [r for r in rows if r["level"] == lv]
        by_level[str(lv)] = dict(
            median_pct=round(float(np.median([r["D"] for r in rs])) * 100, 2),
            min_pct=round(float(np.min([r["D"] for r in rs])) * 100, 2),
            max_pct=round(float(np.max([r["D"] for r in rs])) * 100, 2))

    all_ok = all(r["bounds_ok"] for r in rows)
    display = dict(
        median_pct_by_level={str(lv): by_level[str(lv)]["median_pct"] for lv in levels},
        n_objects=len(cfg["cohort"]),
        spanning_driven_by="level (calibration-uncertainty operating point)",
        note="median 0.7% at level 0.025 -> 89.9% at level 8.0 (37x); "
             "level 0.5 reproduces the P-CERT 62.87%.")

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        kappa=kappa, levels=levels,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        by_level=by_level, display=display,
        all_ceiling_bounds_ok=all_ok,
        rows=rows,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=_git_sha(),
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="Dynamic range as a curve in level (uniform all-refined "
             "endpoint, kappa=10). Descriptive curve: no sign-based gate, "
             "every level/object reported. Saturates toward the universal "
             "ceiling 1-1/kappa at large level (methods.md section 7).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                               .encode("utf-8"))
    print(f"[levels] ceiling bounds all ok: {all_ok}")
    print(f"[levels] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/certified_gaps_levels.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/certification/certified_gaps_levels.json"))
    args = ap.parse_args()
    run(args.config, args.out)