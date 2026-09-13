"""N1 · gamma-bound proof-limits regression（把本轮的发现固化）。

Two locked facts (methods.md §9, results/submodularity/alpha_bound.json
`proof_limits`):

1. **The `M^{-2}` trace ordering genuinely reverses** — a pinned 2×2
   counterexample: with `M ⪯ N = M + H` (H ⪰ 0) and `W ⪰ 0`, the marginal
   gain still *increases* (`d(M,W) = 1/3 < d(N,W) = 1/2`) while
   `tr[W M^{-2}] = 1 < tr[W N^{-2}] = 1.25`. The sandwich upper endpoint
   `tr[W M(S)^{-2}]` is therefore NOT monotone along the refinement
   lattice, which is exactly why the collapse step of the γ ≥ 1/(1+α)
   proof cannot go through `M^{-2}` monotonicity (Löwner: `t ↦ t^p` is
   operator-monotone iff |p| ≤ 1). If someone "helpfully" reinstates that
   step, assertion 1 fails.

2. **The bound itself holds on adversarial exhaustive families** — random
   near-singular / counterexample-rotated `(A, {W_k ⪰ 0})` instances,
   exhaustive `(S ⊆ T ⊆ N\\{x}, x)` triples, zero violations of
   `Δ_xG(S)/Δ_xG(T) ≥ 1/(1+α) − 1e-9`, including triples where the
   `M^{-2}` ordering is reversed.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "results/submodularity/alpha_bound.json"

M2 = np.array([[1.0, -1.0], [-1.0, 2.0]])
H2 = np.array([[1.0, -1.0], [-1.0, 1.0]])
W2 = np.array([[1.0, -2.0], [-2.0, 4.0]])


def _gain(M, W):
    """d(M, W) = tr M^{-1} − tr (M+W)^{-1} (the A-opt marginal gain)."""
    return float(np.trace(np.linalg.inv(M))
                 - np.trace(np.linalg.inv(M + W)))


def _alpha(A, W):
    w, v = np.linalg.eigh(A)
    Ah = (v / np.sqrt(w)) @ v.T
    return float(np.linalg.eigvalsh(Ah @ W @ Ah).max())


# ------------------------------------------------ 1. 2×2 反例(固化反转)
def test_counterexample_2x2_pins_m_inv_sq_reversal():
    """d(M,W)=1/3 < d(N,W)=1/2 while tr[W M^-2] < tr[W N^-2] — the M^-2
    sandwich endpoint is NOT monotone; the collapse step may not use it."""
    N2 = M2 + H2
    Minv, Ninv = np.linalg.inv(M2), np.linalg.inv(N2)
    d_MW, d_NW = _gain(M2, W2), _gain(N2, W2)
    tr_M = float(np.trace(W2 @ Minv @ Minv))
    tr_N = float(np.trace(W2 @ Ninv @ Ninv))

    assert d_MW == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert d_NW == pytest.approx(0.5, abs=1e-12)
    assert d_MW < d_NW                                  # gain increases
    assert tr_M == pytest.approx(1.0, abs=1e-12)
    assert tr_N == pytest.approx(1.25, abs=1e-12)
    assert tr_M < tr_N                                  # M^-2 ordering reversed

    # the bound still holds on the counterexample itself (2/3 ≥ 1/3)
    a = _alpha(M2, W2)
    assert a == pytest.approx(2.0, abs=1e-12)
    assert d_MW / d_NW >= 1.0 / (1.0 + a) - 1e-9


# ------------------------------------- 2. 小规模穷举:界在对抗族上成立
def _exhaustive_check(A, Ws, tol=1e-15):
    """全 (S ⊆ T ⊆ N\\{x}, x) 三元组:返回 (violations, reversals, min_ratio)。"""
    L = len(Ws)
    alpha = max(_alpha(A, W) for W in Ws)
    Ms, Fs = {}, {}
    for n in range(L + 1):
        for S in itertools.combinations(range(L), n):
            M = A.copy()
            for k in S:
                M = M + Ws[k]
            Sf = frozenset(S)
            Ms[Sf] = M
            Fs[Sf] = float(np.trace(np.linalg.inv(M)))
    F0 = Fs[frozenset()]
    G = {S: F0 - v for S, v in Fs.items()}
    viol = rev = 0
    min_ratio = np.inf
    for x in range(L):
        Wx = Ws[x]
        rest = [k for k in range(L) if k != x]
        for m in range(len(rest) + 1):
            for Tt in itertools.combinations(rest, m):
                Tf = frozenset(Tt)
                dT = G[Tf | {x}] - G[Tf]
                if dT <= tol:
                    continue
                MTinv = np.linalg.inv(Ms[Tf])
                trT = float(np.trace(Wx @ MTinv @ MTinv))
                for j in range(m + 1):
                    for St in itertools.combinations(Tt, j):
                        Sf = frozenset(St)
                        dS = G[Sf | {x}] - G[Sf]
                        if dS <= tol:
                            continue
                        ratio = dS / dT
                        min_ratio = min(min_ratio, ratio)
                        if ratio < 1.0 / (1.0 + alpha) - 1e-9:
                            viol += 1
                        MSinv = np.linalg.inv(Ms[Sf])
                        if float(np.trace(Wx @ MSinv @ MSinv)) < trT - 1e-12:
                            rev += 1
    return viol, rev, float(min_ratio)


def test_gamma_bound_holds_on_adversarial_psd_family():
    """随机 (A, {W_k ⪰ 0}) 族 + 旋转反例嵌入族:穷举零违反,且反转确实出现。"""
    P, L = 6, 4
    total_viol = total_rev = 0
    for seed in range(50):
        r = np.random.default_rng(31000 + seed)
        Q = np.linalg.qr(r.normal(size=(P, P)))[0]
        if seed % 2 == 0:                       # near-singular family
            A = (Q * 10.0 ** r.uniform(-6, 2, size=P)) @ Q.T
            Ws = []
            for _ in range(L):
                G = r.normal(size=(P, 3))
                Ws.append((G @ G.T) * 10.0 ** r.uniform(-2, 2))
        else:                                   # counterexample-rotated family
            D = np.diag(10.0 ** r.uniform(-2, 2, size=P - 2))
            A = Q @ np.block([[M2, np.zeros((2, P - 2))],
                              [np.zeros((P - 2, 2)), D]]) @ Q.T

            def emb(X):
                return Q @ np.block([
                    [X, np.zeros((2, P - 2))],
                    [np.zeros((P - 2, 2)), np.zeros((P - 2, P - 2))]]) @ Q.T

            Ws = [emb(H2), emb(W2)]
            for _ in range(L - 2):
                v = r.normal(size=P)
                v /= np.linalg.norm(v)
                Ws.append(np.outer(v, v) * r.uniform(0.01, 0.1))
        viol, rev, _ = _exhaustive_check(A, Ws)
        total_viol += viol
        total_rev += rev
    assert total_viol == 0, "gamma bound violated on adversarial family"
    assert total_rev > 0, "expected at least one M^-2 reversal (rotated family)"


# ------------------------------------- 3. 产物 proof_limits 字段(可追溯)
def test_proof_limits_artifact_fields():
    """alpha_bound.json 的 proof_limits 与本测试的独立重算一致。"""
    if not ART.exists():
        pytest.skip("artifact not committed")
    j = json.loads(ART.read_text(encoding="utf-8"))
    if "proof_limits" not in j:
        pytest.skip("proof_limits field not yet in artifact")
    pl = j["proof_limits"]
    assert pl["gamma_bound_violations"] == 0
    assert pl["adversarial_triples"] >= 100_000
    assert pl["m_inv_sq_reversal_samples"] > 0
    assert pl["min_ratio_observed"] > 0
    # 2x2 反例数值与测试 1 的硬编码一致
    ce = pl["m_inv_sq_ordering_counterexample_2x2"]
    assert ce["d_MW"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert ce["d_NW"] == pytest.approx(0.5, abs=1e-5)
    assert ce["m_inv_sq_ordering_reversed"] is True
    assert ce["bound_holds_here"] is True
