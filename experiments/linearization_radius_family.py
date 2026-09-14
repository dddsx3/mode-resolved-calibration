"""P-SIGMA-FAMILY (C 臂) · 线性化有效半径在 Σ_φ 参数族上的稳定性(S2-4 C)。

冻结结论 C(claim B6 / linearization_radius v2):联合参数化下
q(level) = dev(level)/(level/level_ref)² 首过 2×/10× 的水平
(radius_2x 中位 1.0、radius_10x 中位 1.5)。本实验把腐蚀形状换成
三轴族的四个成员(joint 控制 / intensity_only / direction_only /
joint_het),同网格同种子重测半径。

控制臂锚点:joint 臂与冻结 linearization_radius.json **逐位一致**
(观测侧注入 + 名义几何估计 v2 度量域;raw_innovations_family /
apply_corruption_family 的标量退化路径逐位等于 v2 用的
raw_innovations / apply_scaled_corruption,S0-2 钉死)——运行时断言,
不一致即失败。

输出: results/magnitude/linearization_radius_family.json
(不覆盖冻结的 linearization_radius.json)
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

from calibinfo.allocation.corruption import (                        # noqa: E402
    raw_innovations_family, apply_corruption_family)
from calibinfo.datasets.openillumination import load_object          # noqa: E402
from calibinfo.models.corruption_family import CorruptionFamily      # noqa: E402
from experiments.openillumination_validation import NominalScene     # noqa: E402

FROZEN = REPO / "results/magnitude/linearization_radius.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def _shape_params(shape_cfg, level):
    """shape 的 (sig_logI, sig_dir_deg, het) —— 'level' 占位替换。"""
    def _v(x):
        return float(level) if x == "level" else float(x)
    return (_v(shape_cfg["sig_logI"]), _v(shape_cfg["sig_dir_deg"]),
            float(shape_cfg["het_sigma"]))


def run(config_path=REPO / "configs/linearization_radius_family.yaml",
        out_path=REPO / "results/magnitude/"
                       "linearization_radius_family.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    levels = [float(lv) for lv in cfg["levels"]]
    seeds = int(cfg["seeds"])
    ref_level = float(cfg.get("reference_level", levels[0]))
    crossings = [float(c) for c in cfg["crossings"]]
    shapes = cfg["shapes"]
    family_seed = int(cfg.get("family_seed", 20260915))
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    t_start = time.time()
    sha_at_launch = _git_sha()

    out = {}
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name,
                          data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        n_act = len(scen.sel)
        for shape_name, shape_cfg in shapes.items():
            rows_obj = []
            for level in levels:
                sI, sd, het = _shape_params(shape_cfg, level)
                fam = CorruptionFamily(sI, sd, het_sigma=het, rho_c=0.0,
                                       seed=family_seed,
                                       n_lights=int(cfg["K_lights"]))
                # 预测侧:bottom-5 dual 子空间(逐灯块须按场景灯序对齐)
                sig_blocks = fam.sigma_phi_block()[scen.sel]
                _deg, W_dual = scen.predicted_degradation(
                    sig_blocks, mode_coordinate="dual")
                rho_pred = np.sort(_deg)
                # 注入侧:het 用逐灯 σ 向量;het=0 退化为标量(控制臂
                # 与 v2 逐位一致)
                if het == 0.0:
                    sig_I_vec, sig_R_vec = sI, np.radians(sd)
                else:
                    mult = fam.sigma_logI_vec()[scen.sel] / sI
                    sig_I_vec = sI * mult
                    sig_R_vec = np.radians(sd) * mult
                energies = np.empty(seeds)
                for s_i in range(seeds):
                    rng = np.random.default_rng(
                        [20260916, 4242, obj_idx, int(level * 1000), s_i])
                    raw = raw_innovations_family(rng, n_act, rho_c=0.0)
                    scales = np.ones(n_act)
                    d2, g = apply_corruption_family(
                        scen.dirs, sig_I_vec, sig_R_vec, scales, raw)
                    # 观测侧注入(v2 方案 a):全非线性前向,含翻转
                    I_obs = scen.rho[None, :] \
                        * np.maximum(scen.n @ d2.T, 0.0).T * g[:, None]
                    scen_obs = NominalScene._from_arrays(scen, I_obs)
                    rho_t = scen_obs.estimate_albedo(
                        scen.dirs, np.ones(n_act))
                    e = rho_t - scen.rho
                    sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
                    e = e - sg * scen.rho
                    energies[s_i] = float(np.sum((e @ W_dual) ** 2))
                rows_obj.append(dict(level=level, E_mean=float(energies.mean()),
                                     pred_deg_bottom5=[float(x) for x in
                                                       1.0 / rho_pred[:5]],
                                     n_seeds=seeds))
            # 参考能量 + 一阶标度 + 半径(与 v2 相同语义)
            E_ref = rows_obj[0]["E_mean"]
            for r in rows_obj:
                scale2 = (r["level"] / ref_level) ** 2
                r["dev_from_ref"] = (r["E_mean"] / E_ref
                                     if E_ref > 0 else float("nan"))
                r["first_order_scaling"] = scale2
                r["excess_over_scaling"] = (r["dev_from_ref"] / scale2
                                            if E_ref > 0 else float("nan"))
            radius = {}
            for thr in crossings:
                hit = [r["level"] for r in rows_obj
                       if r["excess_over_scaling"] > thr]
                radius[f"cross_{thr:g}x"] = (min(hit) if hit else None)
            out.setdefault(shape_name, {})[obj_name] = dict(
                rows=rows_obj, radius=radius, E_ref=E_ref)
        # 控制臂锚点:joint 行与冻结 v2 逐位一致
        for level in levels:
            mine = out["joint"][obj_name]["rows"]
            frz = frozen["objects"][obj_name]["rows"]
            m = next(r for r in mine if r["level"] == level)
            f = next(r for r in frz if r["level"] == level)
            if m["E_mean"] != f["E_mean"]:
                raise RuntimeError(
                    f"control anchor broken: {obj_name} level {level} "
                    f"E_mean {m['E_mean']!r} != frozen {f['E_mean']!r}")
        print(f"[famC] {obj_name}: 4 shapes done "
              f"(joint radius2x={out['joint'][obj_name]['radius']['cross_2x']}"
              f", dir_only radius2x="
              f"{out['direction_only'][obj_name]['radius']['cross_2x']}; "
              f"{time.time() - t_start:.0f}s)", flush=True)

    # ---- 汇总(每 shape:与 v2 相同的聚合 + 族比较)----
    def _agg(shape, key):
        vals = [v["radius"][key] for v in out[shape].values()
                if v["radius"][key] is not None]
        return dict(
            median_over_crossed_subset=(float(np.median(vals))
                                        if vals else None),
            n_crossed=len(vals), n_total=len(out[shape]),
            per_object={k: v["radius"][key] for k, v in out[shape].items()})

    shapes_summary = {}
    for shape_name in shapes:
        shapes_summary[shape_name] = {
            f"radius_{int(t)}x": _agg(shape_name, f"cross_{t:g}x")
            for t in crossings}
    # 预注册判据
    frz2 = frozen["radius_2x"]["median_over_crossed_subset"]
    frz10 = frozen["radius_10x"]["median_over_crossed_subset"]
    robust = True
    for shape_name in shapes:
        for key, frz_med, lo, hi in (("radius_2x", frz2, frz2 / 4, frz2 * 4),
                                     ("radius_10x", frz10, frz10 / 4,
                                      frz10 * 4)):
            med = shapes_summary[shape_name][key][
                "median_over_crossed_subset"]
            if med is None or not (lo <= med <= hi):
                robust = False
    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        levels=levels, seeds=seeds, crossings=crossings,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]), shapes=list(shapes),
        family_seed=family_seed,
        frozen_reference=dict(
            artifact="results/magnitude/linearization_radius.json",
            radius_2x_median=frz2, radius_10x_median=frz10),
        shapes_summary=shapes_summary,
        outcome=("robust" if robust else "channel-dependent"),
        outcome_rule=cfg["outcome_rule"].strip(),
        control_anchor="joint arm E_mean BIT-IDENTICAL to the frozen v2 "
                       "artifact per (object, level); runtime-asserted",
        objects=out,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            scene_rng_spec=cfg["scene_rng_spec"],
            elapsed_s=round(time.time() - t_start, 1)),
        note="Linearization radius per family shape (S2-4 C): same v2 "
             "metric domain (observation-side injection, nominal-geometry "
             "estimator, q = dev / first-order scaling), same level grid "
             "and seed spec as the frozen run; the joint arm is a "
             "bit-exact control. Closed channels use the frozen "
             "closed-channel convention on both sides "
             "(1e-3 deg / 1e-6 logI).")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[famC] outcome: {summary['outcome']}")
    for sn in shapes:
        print(f"[famC] {sn}: radius2x median "
              f"{shapes_summary[sn]['radius_2x']['median_over_crossed_subset']}"
              f", radius10x median "
              f"{shapes_summary[sn]['radius_10x']['median_over_crossed_subset']}")
    print(f"[famC] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/"
                                "linearization_radius_family.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/magnitude/"
                                "linearization_radius_family.json"))
    args = ap.parse_args()
    run(args.config, args.out)
