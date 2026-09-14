"""P-SIGMA-FAMILY · Σ_φ 参数族敏感性(S1 三层网格 + S2-1/2/3,预测侧)。

把结论 A(两项定律)/B(上限)/D(强度通道主导,C7)从"取决于我们的参数化"
(joint level,logI/dir 方差比被钉死在 ~3283)升级为"在三轴参数族上成立
(或不成立)":通道比(sig_logI vs sig_dir_deg 独立)、逐灯异质性(het_sigma)、
通道耦合(rho_c)。

预测侧功能量与冻结实验**同一代码路径**:D 用 experiments/channel_
decomposition.J_A(装配镜像 build_state,唯一替换点 Σ_φ 来源换成
CorruptionFamily);两项定律用 goal_orientation 的 woodbury_quad_risk
(rho_mean 任务,精确 gauge 对齐)。退化锚点:S0-3 测试证明该路径对冻结
channel_decomposition.json 的 264 行逐位复现(commit ec2058f)。

预注册(configs/corruption_family_sensitivity.yaml,先于运行提交):
  - S2-1: share_dir 与 r*(log10 线性插值首过 0.02),robust/artifact/scoped
    三分支判据;反向控制(固定 σ_dir=1° 扫 σ_logI);
  - S2-2: 两项定律 |dV| 判据(median ≤ 0.01 且 max ≤ 0.05);
  - S2-3: 上限定理运行时守卫(max 违反 ≤ 1e-9;是守卫,不是证据)。

输出: results/openillumination/corruption_family_sensitivity.json
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
from calibinfo.information.lowrank import woodbury_quad_risk      # noqa: E402
from calibinfo.models.corruption_family import CorruptionFamily   # noqa: E402
from experiments.channel_decomposition import J_A                 # noqa: E402
from experiments.openillumination_validation import NominalScene  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def build_grid(cfg):
    """枚举族网格点(按精确四元组去重),每点带 tags(层级归属)。"""
    pts = {}

    def add(li, sd, het, rho, tag):
        key = (float(li), float(sd), float(het), float(rho))
        if key not in pts:
            pts[key] = dict(sig_logI=key[0], sig_dir_deg=key[1],
                            het_sigma=key[2], rho_c=key[3], tags=[])
        if tag not in pts[key]["tags"]:
            pts[key]["tags"].append(tag)

    a = cfg["anchor"]
    add(a["sig_logI"], a["sig_dir_deg"], a["het_sigma"], a["rho_c"], "anchor")
    for sd in cfg["dir_sweep_sigma_deg"]:
        add(a["sig_logI"], sd, a["het_sigma"], a["rho_c"], "dir_sweep")
    for li in cfg["logI_sweep_sigma"]:
        add(li, a["sig_dir_deg"], a["het_sigma"], a["rho_c"], "logI_sweep")
    for li in cfg["logI_sweep_sigma"]:
        add(li, cfg["reverse_control_sigma_dir_deg"], a["het_sigma"],
            a["rho_c"], "reverse_control")
    for het in cfg["het_sweep"]:
        add(a["sig_logI"], a["sig_dir_deg"], het, a["rho_c"], "het_sweep")
    for rho in cfg["rho_sweep"]:
        add(a["sig_logI"], a["sig_dir_deg"], a["het_sigma"], rho, "rho_sweep")
    for c in cfg["tier2_combos"]:
        add(c["sig_logI"], c["sig_dir_deg"], c["het_sigma"], c["rho_c"],
            "tier2")
    for sd in cfg["tier3_sigma_dir_deg"]:
        for het in cfg["tier3_het"]:
            add(a["sig_logI"], sd, het, a["rho_c"], "tier3")
    cli = float(cfg["closed_intensity_sigma_logI"])
    for sd in cfg["dir_only_sigma_dir_deg"]:
        add(cli, sd, 0.0, 0.0, "dir_only")
    io = cfg["intensity_only"]
    add(io["sig_logI"], io["sig_dir_deg"], 0.0, 0.0, "intensity_only")
    bc = cfg["both_closed"]
    add(bc["sig_logI"], bc["sig_dir_deg"], 0.0, 0.0, "both_closed")
    return list(pts.values())


def align_residual(scen, a_dir):
    """gauge 恒等式 A·a = B·c̄ 的对齐残差(Σ 无关;每对象一次)。"""
    nsel = scen.w.shape[0]
    num_l = den_l = 0.0
    for j in range(nsel):
        Aw_a = np.sqrt(scen.w[j]) * scen.s_hat[j] * a_dir
        Bk = np.sqrt(scen.w[j])[:, None] * scen.B_phi[j]
        c, res, *_ = np.linalg.lstsq(Bk, Aw_a, rcond=None)
        r2 = float(res[0] ** 2) if len(res) else float(
            np.sum((Bk @ c - Aw_a) ** 2))
        num_l += r2
        den_l += float(Aw_a @ Aw_a)
    return float(np.sqrt(num_l / den_l)) if den_l > 0 else float("inf")


def eval_point(grid_pt, st, kappa, K, family_seed):
    """单族点:J_A 端点 → D;woodbury J_a 端点 → V_meas;闭式 q,r → V_pred。"""
    fam = CorruptionFamily(grid_pt["sig_logI"], grid_pt["sig_dir_deg"],
                           het_sigma=grid_pt["het_sigma"],
                           rho_c=grid_pt["rho_c"],
                           seed=family_seed, n_lights=K)
    lam0142 = np.linalg.inv(fam.sigma_phi_block())       # (K,3,3) 批量
    blk = LightBlocks(st["u142"], st["M0142"], lam0142,
                      st["finf"], st["active142"])
    t_one = np.ones(K)
    t_all = t_one.copy()
    t_all[st["active_idx"]] = kappa
    J1 = J_A(blk, t_one)                                 # 冻结功能量(同 cd)
    Jk = J_A(blk, t_all)
    D = (J1 - Jk) / J1
    H = st["a"][None, :]
    j1 = woodbury_quad_risk(st["finf"], st["u142"], st["M0142"], lam0142,
                            st["active142"], t_one, H)
    jk = woodbury_quad_risk(st["finf"], st["u142"], st["M0142"], lam0142,
                            st["active142"], t_all, H)
    V_meas = 1.0 - jk / j1
    rho2 = float(st["rho"] @ st["rho"])
    q = float(sum(lam0142[k][0, 0] for k in st["active_idx"]) / rho2)
    r = rho2 / float(np.sum(st["rho"] ** 2 * st["finf"]))
    V_pred = (1.0 - 1.0 / kappa) / (1.0 + q * r)
    return dict(J_A_1=float(J1), J_A_kappa=float(Jk), D=float(D),
                V_meas=float(V_meas), q=q, r=float(r),
                V_pred=float(V_pred), dV=float(V_pred - V_meas))


def r_star_of(curve, thresh=0.02):
    """curve: [(sigma_dir_deg, share)] 升序。log10 线性插值首过 thresh。

    返回 (r_star 或 None, 状态字符串)。"""
    if curve[0][1] >= thresh:
        return None, f"<{curve[0][0]:g}"
    for i in range(1, len(curve)):
        sd0, s0 = curve[i - 1]
        sd1, s1 = curve[i]
        if s1 >= thresh:
            if s1 == s0:
                return sd1, "crossed"
            x0, x1 = np.log10(sd0), np.log10(sd1)
            x = x0 + (x1 - x0) * (thresh - s0) / (s1 - s0)
            return float(10.0 ** x), "crossed"
    return None, f">{curve[-1][0]:g}"


def run(config_path=REPO / "configs/corruption_family_sensitivity.yaml",
        out_path=REPO / "results/openillumination/"
                       "corruption_family_sensitivity.json"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    kappa = float(cfg["kappa"])
    K = int(cfg["K_lights"])
    family_seed = int(cfg.get("family_seed", 20260915))
    ceiling = 1.0 - 1.0 / kappa
    grid = build_grid(cfg)
    t_start = time.time()
    sha_at_launch = _git_sha()

    rows = []
    align_res = {}
    for obj_idx, obj_name in enumerate(cfg["cohort"]):
        obj = load_object(cfg["data_root"], obj_name,
                          data_meta=cfg.get("data_meta"))
        scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                            noise_fit_convention=cfg["noise_fit_convention"])
        # P-CERT 同款装配(镜像 channel_decomposition.build_state)
        u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
        M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                           scen.w[:, :, None] * scen.B_phi)
        u142 = np.zeros((K, scen.Finf_diag.shape[0], 3))
        M0142 = np.zeros((K, 3, 3))
        u142[scen.sel] = u_act
        M0142[scen.sel] = M0_act
        active142 = np.zeros(K, bool)
        active142[scen.sel] = True
        st = dict(u142=u142, M0142=M0142, finf=scen.Finf_diag,
                  active142=active142, active_idx=np.flatnonzero(active142),
                  rho=scen.rho, a=scen.rho / np.linalg.norm(scen.rho))
        # gauge 恒等式残差(Σ 无关,每对象一次;A 的恒等式部分零依赖)
        align_res[obj_name] = round(align_residual(scen, st["a"]), 12)
        for gi, pt in enumerate(grid):
            res = eval_point(pt, st, kappa, K, family_seed)
            viol = res["D"] - ceiling
            if viol > 1e-9:
                raise RuntimeError(
                    f"ceiling guard violated at {obj_name} {pt}: "
                    f"D={res['D']:.12f} > {ceiling:.12f}")
            rows.append(dict(object=obj_name, grid_idx=gi, **res,
                             ceiling_margin=float(ceiling - res["D"])))
        print(f"[fam] {obj_name}: {len(grid)} points done "
              f"({time.time() - t_start:.0f}s)", flush=True)

    # D 检索表:(obj, sig_logI, sig_dir_deg) → D(仅 het=0/rho=0 点用)
    Dmap = {}
    for row, pt in ((r, grid[r["grid_idx"]]) for r in rows):
        if pt["het_sigma"] == 0.0 and pt["rho_c"] == 0.0:
            Dmap[(row["object"], pt["sig_logI"], pt["sig_dir_deg"])] = row["D"]

    # ---------------- S2-1: 方向通道份额与 r* ----------------
    a = cfg["anchor"]
    closed_dir = float(cfg["closed_direction_sigma_deg"])
    closed_li = float(cfg["closed_intensity_sigma_logI"])
    dir_grid = [float(x) for x in cfg["dir_sweep_sigma_deg"]]
    per_obj_s21 = {}
    for obj_name in cfg["cohort"]:
        d_base = Dmap[(obj_name, a["sig_logI"], closed_dir)]
        curve = [(closed_dir, 0.0)]
        for sd in dir_grid:
            d = Dmap[(obj_name, a["sig_logI"], sd)]
            curve.append((sd, (d - d_base) / d))
        # r* 只在扫描点之间插值(预注册:censored '<0.1' 若首个扫描点已越限)
        rs, status = r_star_of(curve[1:])
        per_obj_s21[obj_name] = dict(
            share_curve=[dict(sigma_dir_deg=sd, share=round(s, 6))
                         for sd, s in curve],
            r_star=(round(rs, 4) if rs is not None else None),
            r_star_status=status,
            share_at_anchor=round(
                dict(curve)[a["sig_dir_deg"]], 6))

    def _s21_val(o):
        d = per_obj_s21[o]
        if d["r_star"] is not None:
            return d["r_star"]
        # censored-high → +inf;censored-low → 0(排序用的保守哨兵)
        return np.inf if d["r_star_status"].startswith(">") else 0.0
    vals = sorted(_s21_val(o) for o in cfg["cohort"])
    n_cens_hi = sum(1 for o in cfg["cohort"]
                    if per_obj_s21[o]["r_star"] is None
                    and per_obj_s21[o]["r_star_status"].startswith(">"))
    n_cens_lo = sum(1 for o in cfg["cohort"]
                    if per_obj_s21[o]["r_star"] is None
                    and per_obj_s21[o]["r_star_status"].startswith("<"))
    med = vals[len(vals) // 2]        # 11 个对象 → 第 6 顺序统计量
    if np.isinf(med):
        med_repr = ">25"
    elif med == 0.0:
        med_repr = "<0.1"
    else:
        med_repr = round(med, 4)
    if np.isinf(med) or med > 4.0:
        outcome = "robust"
    elif med <= 1.0:
        outcome = "artifact"
    else:
        outcome = "scoped"
    dir_only_max = max(r["D"] for r, pt in ((r, grid[r["grid_idx"]])
                                            for r in rows)
                       if "dir_only" in pt["tags"])
    s21 = dict(
        definition=cfg["s21_direction_share"].strip(),
        baseline_point=f"D({a['sig_logI']}, closed_dir={closed_dir}) "
                       "(== frozen intensity@0.5 row)",
        per_object=per_obj_s21,
        r_star_median=med_repr,
        r_star_min=(round(min(v for v in vals if np.isfinite(v)), 4)
                    if any(np.isfinite(v) for v in vals) else None),
        n_censored_high=n_cens_hi,
        n_censored_low=n_cens_lo,
        outcome=outcome,
        share_at_anchor_median=round(float(np.median(
            [per_obj_s21[o]["share_at_anchor"] for o in cfg["cohort"]])), 6),
        direction_only_max_pct=round(dir_only_max * 100, 3),
        direction_only_note="direct C7 generalization: D(closed_logI, "
                            "sigma_dir) over the sweep; the frozen "
                            "measurement's max was 1.68%")

    # ---------------- S2-1 反向控制(固定 σ_dir=1°,扫 σ_logI) ----------------
    rc_sd = float(cfg["reverse_control_sigma_dir_deg"])
    per_li = {}
    for li in cfg["logI_sweep_sigma"]:
        li = float(li)
        shares = []
        for obj_name in cfg["cohort"]:
            d = Dmap[(obj_name, li, rc_sd)]
            d0 = Dmap[(obj_name, closed_li, rc_sd)]
            shares.append((d - d0) / d)
        per_li[str(li)] = dict(
            median=round(float(np.median(shares)), 6),
            min=round(float(np.min(shares)), 6),
            max=round(float(np.max(shares)), 6))
    s21_rc = dict(
        definition=cfg["s21_reverse_control"].strip(),
        fixed_sigma_dir_deg=rc_sd,
        baseline_point=f"D(closed_logI={closed_li}, {rc_sd} deg)",
        share_int_by_sigma_logI=per_li)

    # ---------------- S2-2: 两项定律 ----------------
    def dv_stats(sel):
        dvs = [abs(r["dV"]) for r, pt in ((r, grid[r["grid_idx"]])
                                          for r in rows) if sel(pt)]
        if not dvs:
            return None
        return dict(n=len(dvs),
                    median_abs_dV=round(float(np.median(dvs)), 6),
                    max_abs_dV=round(float(np.max(dvs)), 6))
    tags = sorted({t for pt in grid for t in pt["tags"]})
    by_tag = {t: dv_stats(lambda pt, t=t: t in pt["tags"]) for t in tags}
    overall = dv_stats(lambda pt: True)
    s22_outcome = ("robust" if overall["median_abs_dV"] <= 0.01
                   and overall["max_abs_dV"] <= 0.05
                   else "parameterization-specific")
    s22 = dict(
        definition=cfg["s22_two_term_law"].strip(),
        overall=overall, by_tag=by_tag, outcome=s22_outcome,
        align_residual_per_object=align_res,
        align_note="gauge identity A·a = B·c̄ residual (Sigma-independent, "
                   "computed once per object; A's identity part has zero "
                   "family dependence by construction)")

    # ---------------- S2-3: 上限守卫 ----------------
    s23 = dict(
        definition=cfg["s23_ceiling"].strip(),
        max_violation=round(float(max(r["D"] - ceiling for r in rows)), 12),
        n_rows=len(rows),
        note="runtime guard on theorem M10/M10' (zero Sigma-dependence); "
             "NOT experimental evidence")

    summary = dict(
        gate=cfg["gate"], analysis_status=cfg["analysis_status"],
        kappa=kappa, K_lights=K,
        noise_fit_convention=cfg["noise_fit_convention"],
        n_objects=len(cfg["cohort"]),
        anchor=cfg["anchor"], family_seed=family_seed,
        grid=grid, rows=rows,
        s21_direction_share=s21, s21_reverse_control=s21_rc,
        s22_two_term_law=s22, s23_ceiling=s23,
        manifest=dict(
            config_sha256=_sha(Path(config_path)),
            git_sha=sha_at_launch,
            scene_rng_spec=cfg["scene_rng_spec"],
            s0_anchor="S0-3 end-to-end anchor: 264/264 channel_decomposition "
                      "rows bit-exact through the family path "
                      "(tests/test_corruption_family_degeneracy.py)",
            elapsed_s=round(time.time() - t_start, 1)),
        note="Predictor-side family sensitivity: D via the frozen "
             "channel-decomposition functional (J_A), two-term law via "
             "woodbury_quad_risk on the exactly gauge-aligned rho_mean task. "
             "Preregistered outcome rules in the config; every number is "
             "reported regardless of shape.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"))
    print(f"[fam] S2-1 outcome: {outcome} (r* median {med_repr})")
    print(f"[fam] S2-2 outcome: {s22_outcome} "
          f"(median |dV| {overall['median_abs_dV']}, "
          f"max {overall['max_abs_dV']})")
    print(f"[fam] S2-3 max ceiling violation: {s23['max_violation']}")
    print(f"[fam] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "configs/"
                                "corruption_family_sensitivity.yaml"))
    ap.add_argument("--out",
                    default=str(REPO / "results/openillumination/"
                                "corruption_family_sensitivity.json"))
    args = ap.parse_args()
    run(args.config, args.out)
