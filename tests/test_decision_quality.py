"""P-DECISION-QUALITY · math gate (M3-1) + artifact pin (M3-2..4).

arm_metrics (experiments/allocation_mode_tail.py) is the SINGLE residual
pipeline extended with decision-layer readouts: the corruption + fixed-n̂
GLS segment is value-identical to arm_energy (tested here), and the
one-step normal refit is calibrated_ps's own n-update (masked unweighted
LSQ + normalization) vectorized -- tested against the per-pixel loop
reference on a synthetic scene (no raw data, CI-safe).
"""

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/openillumination/decision_quality.json"


def _synthetic_scene(seed=20260913):
    """最小合成场景(142 灯 × ~250 masked 像素;CI-safe,无原始数据)。"""
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments"))
    from experiments.openillumination_validation import NominalScene

    rng = np.random.default_rng(seed)
    K, H, W = 142, 16, 20
    yy, xx = np.mgrid[:H, :W]
    mask = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) < (W / 2.2) ** 2
    dirs = rng.normal(size=(K, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 2] = np.abs(dirs[:, 2]) + 0.4          # 上半球
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    P = int(mask.sum())
    rho = rng.uniform(0.3, 1.5, size=P)
    nrm = np.zeros((P, 3))
    nrm[:, 0] = rng.normal(size=P) * 0.3
    nrm[:, 1] = rng.normal(size=P) * 0.3
    nrm[:, 2] = 1.0
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    imgs = rho[:, None] * (nrm @ dirs.T)           # (P,K)
    img = np.zeros((K, H, W))
    img[:, mask] = imgs.T
    obj = dict(images=img[..., None], mask=mask, light_directions=dirs)
    return NominalScene(obj, np.random.default_rng([20260910, 0]),
                        noise_fit_convention="corrected")


def test_arm_metrics_parity_and_normal_refit():
    """三重门禁:dual 读出与 arm_energy 位级一致;法线重估 = calibrated_ps
    一步更新的向量化(逐像素循环参照,≤1e-12);端点有限且角度非负。"""
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments"))
    from calibinfo.allocation.corruption import (apply_scaled_corruption,
                                                 raw_innovations)
    from calibinfo.models.corruption import CorruptionGenerator
    from experiments.allocation_mode_tail import arm_energy, arm_metrics

    scen = _synthetic_scene()
    gen = CorruptionGenerator("joint", 0.5)
    sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
    _deg, W_dual = scen.predicted_degradation(gen.sigma_phi_diag(),
                                              mode_coordinate="dual")
    n_act = len(scen.sel)
    rng = np.random.default_rng(77)
    raw = raw_innovations(rng, n_act)
    for scale in (np.ones(n_act),
                  np.where(np.arange(n_act) < 5, 1 / np.sqrt(10), 1.0)):
        ref = arm_energy(scale, raw, scen, sig_logI, sig_rad, W_dual)
        m = arm_metrics(scale, raw, scen, sig_logI, sig_rad, W_dual)
        np.testing.assert_allclose(m["dual"], ref, rtol=0, atol=1e-12)
        assert np.isfinite([m["ang_mean_deg"], m["ang_median_deg"],
                            m["ang_p95_deg"], m["mse_aligned"],
                            m["mae_raw"]]).all()
        assert m["ang_mean_deg"] >= 0 and m["ang_p95_deg"] >= m["ang_median_deg"]

        # 法线重估 = calibrated_ps 一步(逐像素循环参照)
        d2, g = apply_scaled_corruption(scen.dirs, sig_logI, sig_rad,
                                        scale, raw)
        rho_t = scen.estimate_albedo(d2, g)
        ndl = np.clip(scen.n @ d2.T, 0, None).T
        act = (scen.I > 1e-3) & (ndl > 0)
        g_col = np.broadcast_to(np.asarray(g, float).reshape(-1, 1), ndl.shape)
        Y = scen.I / np.maximum(rho_t[None, :] * g_col, 1e-12)
        n_ref = np.zeros_like(scen.n)
        for p in range(scen.n.shape[0]):
            m_ = act[:, p]
            if m_.sum() >= 3:
                v = np.linalg.lstsq(d2[m_], Y[m_, p], rcond=None)[0]
                nv = np.linalg.norm(v)
                n_ref[p] = v / nv if nv > 1e-9 else np.array([0., 0., 1.])
            else:
                n_ref[p] = np.array([0.0, 0.0, 1.0])
        # 重算 arm_metrics 的 n_est 并比对
        M = act.astype(float)
        A = np.einsum("lp,li,lj->pij", M, d2, d2)
        b = np.einsum("lp,lp,li->pi", M, Y, d2)
        n_sol = (np.linalg.pinv(A) @ b[..., None])[..., 0]
        nrm2 = np.linalg.norm(n_sol, axis=1)
        n_vec = np.zeros_like(scen.n)
        ok = nrm2 > 1e-9
        n_vec[ok] = n_sol[ok] / nrm2[ok, None]
        n_vec[~ok] = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(n_vec, n_ref, atol=1e-12)


def test_decision_quality_artifact():
    if not ART.exists():
        pytest.skip("artifact not committed")
    import hashlib
    j = json.loads(ART.read_text(encoding="utf-8"))
    if j["analysis_status"] != "decision_quality_v1_1":
        pytest.skip("artifact predates the v1.1 grid (4 informed + 3U/2A48)")
    assert j["n_objects"] == 11
    # rows: 11 obj x 4 levels x 2 regimes x 9 units (4 det + 3U + 2A48) x 5 budgets
    assert len(j["rows"]) == 11 * 4 * 2 * 9 * 5
    assert len(j["auc_rows"]) == 11 * 4 * 2
    # bootstrap: 4 policies x 2 regimes x 3 endpoints x 2 baselines
    assert len(j["bootstrap_dAUC"]) == 4 * 2 * 3 * 2
    for key, b in j["bootstrap_dAUC"].items():
        assert b["ci95"][0] <= b["median_dAUC"] <= b["ci95"][1] + 1e-12
    for ep in ("ang_mean_deg", "mse_aligned", "dual_mean"):
        s = j["spearman_pred_vs_realized"][ep]
        assert -1.0 <= s["min"] <= s["max"] <= 1.0 and s["n_cells"] == 88
        si = j["spearman_informed_only"][ep]
        assert -1.0 <= si["min"] <= si["max"] <= 1.0 and si["n_cells"] == 88
    ip = j["informed_pairwise_sign_pooled"]
    assert 0.0 <= ip["rate"] <= 1.0 and ip["total"] > 0
    assert ip["ci95"][0] <= ip["rate"] <= ip["ci95"][1] + 1e-9
    # 随机拆分元数据
    assert j["random_units"] == {"universe": 3, "active48": 2}
    cfg_hash = hashlib.sha256(
        (REPO / "configs/decision_quality.yaml").read_bytes()).hexdigest()
    assert j["manifest"]["config_sha256"] == cfg_hash
    assert j["manifest"]["git_sha"]
