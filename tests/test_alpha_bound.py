"""P-ALPHA-BOUND gates (C3): the α-approximate submodularity theorem.

Two layers:
1. **CI-safe theorem re-derivation** — on the P-SUBMOD toy family (pure
   synthetic, no raw data), re-verify γ_measured ≥ 1/(1+α) − 1e-9 on all
   seed/family instances and zero violations of the two-sided sandwich
   (1600 (S,x) pairs). This is the math gate; it runs on any machine.
2. **Committed-artifact gates** — the real-object α and the overall
   statement are read from results/submodularity/alpha_bound.json only (no
   raw data in CI), with range pins to catch regressions.

The theorem itself (methods.md §9, L9–L13):
    γ ≥ 1/(1+α),  α = max_x λmax( ΔF(1)^{-1} W_x ),
    W_x = u_x [K_x(1) − K_x(κ)] u_x^T ⪰ 0,
where α is nominal-design-only (no truth, no measurements).
"""

import json
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from calibinfo.allocation.blocks import LightBlocks, sym_inv
from calibinfo.allocation.alpha_bound import (
    decompose, alpha_of, gamma_lower_bound, delta_gain_interval)

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/submodularity/alpha_bound.json"

KAPPA = 10.0
SEEDS = [20260918 + i for i in range(10)]
FAMILIES = ["random", "adversarial"]
L, P = 5, 40


def _build_instance(seed, family):
    """P-SUBMOD 同款实例族(与 experiments/submodularity_search.py 一致)。"""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    lam_diag = np.array([30.0, 800.0, 800.0])
    if family == "adversarial":
        base_dir = rng.normal(size=(P, 3))
        for k in range(L):
            B[k] = base_dir[None, :] * rng.uniform(0.8, 1.2) \
                + 0.05 * rng.normal(size=(P, 3))
        s = s ** 3
        s = s * (1.0 + rng.uniform(0, 20, size=(L, 1)))
        lam_diag = np.array([30.0 * 1e-2, 800.0, 800.0 * 1e2])
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag(lam_diag)] * L)
    finf = (w * s ** 2).sum(0)
    return u, M0, lam0, finf


def _F_values(u, M0, lam0, finf):
    vals = {}
    for n in range(0, L + 1):
        for S_tuple in itertools.combinations(range(L), n):
            t = np.ones(L)
            for k in S_tuple:
                t[k] = KAPPA
            DF = np.diag(finf)
            for k in range(L):
                KK = sym_inv(M0[k] + t[k] * lam0[k])
                DF -= (u[k] @ KK) @ u[k].T
            vals[frozenset(S_tuple)] = float(np.trace(np.linalg.inv(DF)))
    return vals


def _measured_gamma(u, M0, lam0, finf):
    """穷举 (A ⊆ B, x ∉ B) 的实测 γ = min dA/dB(与 P-SUBMOD 同语义)。"""
    vals = _F_values(u, M0, lam0, finf)
    F0 = vals[frozenset()]
    G = {S: F0 - v for S, v in vals.items()}
    gamma = np.inf
    for n in range(1, L + 1):
        for Bt in itertools.combinations(range(L), n):
            Bset = frozenset(Bt)
            for x in range(L):
                if x in Bset:
                    continue
                dB = G[Bset | {x}] - G[Bset]
                if dB <= 1e-15:
                    continue
                for r in range(0, n + 1):
                    for At in itertools.combinations(sorted(Bset), r):
                        dA = G[frozenset(At) | {x}] - G[frozenset(At)]
                        if dA > 1e-15:
                            gamma = min(gamma, dA / dB)
    return float(gamma) if np.isfinite(gamma) else None


# ---------------------------------------------------------------- 1. 定理(CI-safe)
def test_gamma_bound_holds_on_toy_family():
    """γ_measured ≥ 1/(1+α) − 1e-9 对所有 20 个 toy 实例成立。"""
    for family in FAMILIES:
        for seed in SEEDS:
            u, M0, lam0, finf = _build_instance(seed, family)
            blk = LightBlocks(u, M0, lam0, finf, active=np.ones(L, bool))
            A, C = decompose(blk, KAPPA)
            a = alpha_of(A, blk, C)
            lb = gamma_lower_bound(a)
            g = _measured_gamma(u, M0, lam0, finf)
            assert g is not None and g >= lb - 1e-9, (family, seed, a, lb, g)


def test_refined_set_matches_direct_assembly():
    from calibinfo.allocation.alpha_bound import _m_s_inv

    u, M0, lam0, finf = _build_instance(SEEDS[0], "random")
    blocks = LightBlocks(u, M0, lam0, finf, np.ones(L, bool))
    A, C = decompose(blocks, KAPPA)
    for size in range(L):
        for selected in itertools.combinations(range(L), size):
            t = np.ones(L)
            t[list(selected)] = KAPPA
            direct = blocks.assemble(t[:, None, None] * lam0)
            inverse, matrix = _m_s_inv(A, blocks, C, selected, KAPPA)
            np.testing.assert_allclose(matrix, direct, rtol=1e-11, atol=1e-11)
            for x in set(range(L)) - set(selected):
                updated = t.copy()
                updated[x] = KAPPA
                refined = blocks.assemble(updated[:, None, None] * lam0)
                expected = np.trace(inverse) - np.trace(np.linalg.inv(refined))
                lower, upper, value = delta_gain_interval(
                    A, blocks, C, selected, x, KAPPA)
                assert value == pytest.approx(expected, abs=1e-11)
                assert lower - 1e-11 <= value <= upper + 1e-11


def test_two_sided_sandwich_zero_violations():
    """两端不等式 1600 个 (S,x) 对零违反。"""
    rng = np.random.default_rng(12345)
    n_viol = 0
    n_pairs = 0
    for seed in SEEDS[:4]:
        u, M0, lam0, finf = _build_instance(seed, "random")
        blk = LightBlocks(u, M0, lam0, finf, active=np.ones(L, bool))
        A, C = decompose(blk, KAPPA)
        for _ in range(400):
            S = frozenset(rng.choice(L, size=int(rng.integers(0, L + 1)),
                                     replace=False).tolist())
            x = int(rng.integers(0, L))
            if x in S:
                x = (x + 1) % L
            lb, ub, val = delta_gain_interval(A, blk, C, S, x, KAPPA)
            n_pairs += 1
            if val < lb - 1e-9 or val > ub + 1e-9:
                n_viol += 1
    assert n_pairs == 1600
    assert n_viol == 0, f"two-sided sandwich violated on {n_viol} pairs"


def test_alpha_peaks_in_interior():
    """极限行为:Λ0 极强/极弱都令 α→0;中间档取最大。"""
    u, M0, lam0, finf = _build_instance(20260918, "random")
    alphas = []
    for scale in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
        blk = LightBlocks(u, M0, lam0 * scale, finf, active=np.ones(L, bool))
        A, C = decompose(blk, KAPPA)
        alphas.append(alpha_of(A, blk, C))
    assert alphas[0] < 0.01 and alphas[-1] < 0.01      # 两端 α→0
    assert max(alphas) > min(alphas) * 5                # 中间显著更大


# ---------------------------------------------------------------- 2. 产物门禁(no raw data)
def _load_artifact():
    if not ART.exists():
        pytest.skip(f"{ART.name} not committed")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_real_alpha_in_range():
    """真实物体 α 落在 [0.15, 0.80](防回归)。"""
    j = _load_artifact()
    for row in j["real"]:
        assert 0.15 <= row["alpha"] <= 0.80, row


def test_overall_statement_pinned():
    """整体措辞:γ 下界 ≥ 0.635,worst 物体 = obj_10_pumpkin3。"""
    j = _load_artifact()
    g = j["gamma_overall"]
    assert g["gamma_lower_bound_min"] >= 0.6350
    assert g["at_object"] == "obj_10_pumpkin3"
    assert g["at_level"] == 0.1
    assert "0.635-supermodular" in g["phrasing"]


def test_alpha_bound_manifest_integrity():
    """manifest:config 当前 hash 匹配(configs/alpha_bound.yaml)。"""
    import hashlib
    j = _load_artifact()
    cfg_hash = hashlib.sha256((REPO / "configs/alpha_bound.yaml").read_bytes()
                              ).hexdigest()
    assert j["manifest"]["config_sha256"] == cfg_hash
    assert j["manifest"]["git_sha"]
    assert j["analysis_status"] == "alpha_bound_v1"


def test_toy_counts_in_artifact():
    """产物如实登记 20/20 定理成立、3200 对零违反、范围与实测一致。"""
    j = _load_artifact()
    toy = j["toy"]
    assert toy["n_instances"] == 20
    assert toy["n_theorem_holds"] == 20
    assert toy["inequality_violations_total"] == 0
    assert toy["inequality_pairs_total"] >= 1600
    ra = toy["alpha_range"]["random"]
    aa = toy["alpha_range"]["adversarial"]
    assert ra["min"] > 0.05 and ra["max"] < 0.13
    assert aa["min"] < 0.10 and aa["max"] < 0.18