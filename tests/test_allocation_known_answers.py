"""Allocation known-answer tests (V-B1..V-B6 + two bindings).

All tests run on small synthetic per-light systems in the CI04 whitened per-light
form; none require the raw datasets.
"""

import inspect
from dataclasses import fields

import numpy as np
import pytest

from calibinfo.allocation.blocks import (
    LightBlocks,
    mode_gains,
    objective_of,
    sym_inv,
)
from calibinfo.allocation.corruption import paired_corruption
from calibinfo.allocation.policies import (
    SelectionState,
    budget_scales,
    select_ordering,
    select_ordering_mode_aware,
)
from calibinfo.information.schur import delta_f
from calibinfo.models.corruption import CorruptionGenerator


# ---------------------------------------------------------------- fixtures
def _random_blocks(seed=0, L=5, P=40, n_inactive=0):
    """Random per-light system in the CI04 whitened form (A_k diagonal)."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.5, 2.0, size=(L, P))
    s = rng.uniform(0.2, 1.5, size=(L, P))
    B = rng.normal(0.0, 0.5, size=(L, P, 3))
    if n_inactive:                      # lights outside the analysis subset
        w[-n_inactive:] = 0.0
        s[-n_inactive:] = 0.0
        B[-n_inactive:] = 0.0
    u = (w * s)[:, :, None] * B
    M0 = np.einsum("kpi,kpj->kij", B, w[:, :, None] * B)
    lam0 = np.stack([np.diag([30.0, 800.0, 800.0])] * L)
    finf = (w * s ** 2).sum(0)
    active = np.ones(L, bool)
    if n_inactive:
        active[-n_inactive:] = False
    return u, M0, lam0, finf, active, w, s, B


def _state(seed=0, L=5, P=40, n_inactive=0):
    u, M0, lam0, finf, active = _random_blocks(seed, L, P, n_inactive)[:5]
    return SelectionState(u=u, M0=M0, lam0=lam0, finf=finf, active=active)


def _reference_greedy(state, policy, regime=10.0, rng=None):
    """Definition-level reference: rebuild DeltaF from scratch for every candidate
    (no incremental updates), independent objective/mode computation."""
    L = state.u.shape[0]
    if policy == "random":
        return [int(i) for i in rng.permutation(L)], []
    blocks = LightBlocks(state.u, state.M0, state.lam0, state.finf, state.active)
    lam = state.lam0.copy()
    kind = {"e_opt": "e", "a_opt": "a", "d_opt": "d"}[policy]
    DF = blocks.assemble(lam)
    cur = objective_of(DF, kind)
    ordering, steps = [], []
    unselected = set(range(L))
    for _t in range(L):
        best_idx, best_j = None, None
        for idx in sorted(unselected):
            lam_c = lam.copy()
            if state.active[idx]:
                lam_c[idx] = regime * lam_c[idx]
                cand = objective_of(blocks.assemble(lam_c), kind)
                gain = (cur - cand) if kind == "a" else (cand - cur)
            else:
                gain = 0.0
            if best_j is None or gain > best_j:
                best_idx, best_j = idx, gain
        ordering.append(best_idx)
        unselected.remove(best_idx)
        steps.append({"index": best_idx, "J": float(best_j)})
        lam[best_idx] = regime * lam[best_idx]
        DF = blocks.assemble(lam)
        cur = objective_of(DF, kind)
    return ordering, steps


def _reference_mode_aware(state, regime=10.0):
    L = state.u.shape[0]
    blocks = LightBlocks(state.u, state.M0, state.lam0, state.finf, state.active)
    lam = state.lam0.copy()
    DF = blocks.assemble(lam)
    ordering, steps = [], []
    unselected = set(range(L))
    for _t in range(L):
        _w, V = np.linalg.eigh(DF)
        modes = V[:, :5]
        g = mode_gains(state.u, state.M0, lam, state.lam0, state.active, modes)
        uns = sorted(unselected)
        denom = g[uns].sum(0) + 1e-12
        Gs = (g / denom).mean(1)
        best_idx, best_j = None, None
        for idx in uns:
            if best_j is None or Gs[idx] > best_j:
                best_idx, best_j = idx, float(Gs[idx])
        ordering.append(best_idx)
        unselected.remove(best_idx)
        steps.append({"index": best_idx, "J": best_j})
        lam[best_idx] = regime * lam[best_idx]
        DF = blocks.assemble(lam)
    return ordering, steps


# ---------------------------------------------------------------- V-B1
def test_VB1_lo_monotonicity():
    """Lambda_2 >= Lambda_1 (per-light block scaling) implies
    DeltaF(Lambda_2) >= DeltaF(Lambda_1), numerically."""
    u, M0, lam0, finf, active = _random_blocks(seed=1, L=6, P=40)[:5]
    blocks = LightBlocks(u, M0, lam0, finf, active)
    DF1 = blocks.assemble(lam0)
    lam2 = lam0.copy()
    lam2[1] *= 10.0
    lam2[4] *= 100.0
    DF2 = blocks.assemble(lam2)
    diff = DF2 - DF1
    scale = max(1.0, float(np.abs(diff).max()))
    assert np.linalg.eigvalsh(diff).min() >= -1e-12 * scale


# ---------------------------------------------------------------- V-B2
def test_VB2_per_light_derivative():
    """dDeltaF/dlambda_l = u_l K_l Lam0_l K_l u_l^T: PSD and aligned with central
    finite differences of the scalar precision multiplier (rel err <= 1e-5)."""
    u, M0, lam0, finf, active = _random_blocks(seed=2, L=4, P=30)[:5]
    blocks = LightBlocks(u, M0, lam0, finf, active)
    h = 1e-6
    for k in range(4):
        K = sym_inv(M0[k] + lam0[k])
        analytic = (u[k] @ (K @ lam0[k] @ K)) @ u[k].T
        assert np.linalg.eigvalsh(analytic).min() >= -1e-12 * max(
            1.0, float(np.abs(analytic).max()))
        lam_p = lam0.copy()
        lam_m = lam0.copy()
        lam_p[k] = (1.0 + h) * lam0[k]
        lam_m[k] = (1.0 - h) * lam0[k]
        fd = (blocks.assemble(lam_p) - blocks.assemble(lam_m)) / (2.0 * h)
        rel = np.linalg.norm(analytic - fd) / np.linalg.norm(fd)
        assert rel <= 1e-5


# ---------------------------------------------------------------- V-B3
def test_VB3_budget_fairness():
    """At every budget, every policy improves exactly the same NUMBER of
    calibration blocks (prefix construction), and orderings are permutations."""
    state = _state(seed=3, L=12, P=30, n_inactive=2)
    budgets = [0.1, 0.2, 0.4, 0.6, 0.8]
    counts = [round(b * 12) for b in budgets]          # [1, 2, 5, 7, 10]
    rng = np.random.default_rng(20260911)
    for policy in ("mode_aware", "e_opt", "a_opt", "d_opt"):
        if policy == "mode_aware":
            ordering, _ = select_ordering_mode_aware(state)
        else:
            ordering, _ = select_ordering(state, policy)
        assert sorted(ordering) == list(range(12))
        for k in counts:
            scales = budget_scales(ordering, k, 10.0)
            assert int((scales < 1.0).sum()) == k      # same count for every policy
    for p_i in range(5):
        r = np.random.default_rng(20260911 + p_i)
        ordering, _ = select_ordering(state, "random", rng=r)
        for k in counts:
            scales = budget_scales(ordering, k, 10.0)
            assert int((scales < 1.0).sum()) == k


# ---------------------------------------------------------------- V-B4
def test_VB4_leakage_surface():
    """SelectionState exposes the prediction-side inputs only; the selection
    modules contain no reconstruction-side identifiers."""
    names = {f.name for f in fields(SelectionState)}
    assert names == {"u", "M0", "lam0", "finf", "active"}
    import calibinfo.allocation.blocks as blk
    import calibinfo.allocation.policies as pol
    src = inspect.getsource(blk) + inspect.getsource(pol)
    for token in ("rho_hat", "n_hat", "residual", "estimate_albedo"):
        assert token not in src, f"leakage token in selection modules: {token}"


# ---------------------------------------------------------------- V-B5
def test_VB5_determinism():
    """Fixed seed => identical selection sequences (per-light, bit-for-bit)."""
    state = _state(seed=5, L=10, P=30, n_inactive=2)
    for policy in ("mode_aware", "e_opt", "a_opt", "d_opt"):
        if policy == "mode_aware":
            o1, _ = select_ordering_mode_aware(state)
            o2, _ = select_ordering_mode_aware(state)
        else:
            o1, _ = select_ordering(state, policy)
            o2, _ = select_ordering(state, policy)
        assert o1 == o2
    r1, _ = select_ordering(state, "random", rng=np.random.default_rng(7))
    r2, _ = select_ordering(state, "random", rng=np.random.default_rng(7))
    r3, _ = select_ordering(state, "random", rng=np.random.default_rng(8))
    assert r1 == r2 and r1 != r3
    assert sorted(r1) == list(range(10))


# ---------------------------------------------------------------- V-B6
@pytest.mark.parametrize("policy", ["mode_aware", "e_opt", "a_opt", "d_opt"])
def test_VB6_brute_force_reference(policy):
    """Greedy engine == definition-level brute-force reference (orderings and
    per-step objective values) on a small system containing inactive lights."""
    state = _state(seed=6, L=6, P=25, n_inactive=2)
    if policy == "mode_aware":
        ordering, steps = select_ordering_mode_aware(state)
        ref_ordering, ref_steps = _reference_mode_aware(state)
    else:
        ordering, steps = select_ordering(state, policy)
        ref_ordering, ref_steps = _reference_greedy(state, policy)
    assert ordering == ref_ordering
    for st, rs in zip(steps, ref_steps):
        assert st["index"] == rs["index"]
        scale = max(1.0, abs(st["J"]), abs(rs["J"]))
        assert abs(st["J"] - rs["J"]) <= 1e-9 * scale


# ---------------------------------------------------------------- bindings
def test_binding_per_light_assembler_matches_delta_f():
    """The per-light block assembler reduces exactly to the frozen Schur route
    delta_f under a shared Lambda (V1b structure)."""
    u, M0, lam0, finf, active, w, s, B = _random_blocks(seed=7, L=4, P=30)
    Lam = np.diag([0.04, 1.2e-4, 1.2e-4])
    blocks = LightBlocks(u, M0, np.stack([Lam] * 4), finf, active)
    mine = blocks.assemble(np.stack([Lam] * 4))
    ref = np.zeros((30, 30))
    for k in range(4):
        A_k = np.diag(np.sqrt(w[k]) * s[k])
        B_k = np.sqrt(w[k])[:, None] * B[k]
        ref += delta_f(A_k, B_k, Lam)[0]
    assert np.allclose(mine, ref, rtol=0, atol=1e-12 * max(1.0, np.abs(ref).max()))


def test_binding_paired_corruption_matches_generator():
    """paired_corruption with unit scales is bit-identical to the frozen
    CorruptionGenerator.apply on the same rng state."""
    rng = np.random.default_rng(11)
    dirs = rng.normal(size=(6, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    level = 0.5
    gen = CorruptionGenerator("joint", level)
    r1 = np.random.default_rng(123)
    d2a, ga = gen.apply(r1, dirs)
    r2 = np.random.default_rng(123)
    d2b, gb = paired_corruption(
        r2, dirs, gen.sig_logI, np.radians(gen.sig_deg), np.ones(6))
    assert np.allclose(d2a, d2b, rtol=0, atol=0)
    assert np.allclose(ga, gb, rtol=0, atol=0)


def test_engine_zero_gain_tie_break():
    """An active light with numerically-null Fisher presence (u ~ 1e-30) gains
    exactly 0.0: the engine must resolve ties exactly like the full-scan reference
    -- lowest index among the zero-gain holders, inactive lights included."""
    u, M0, lam0, finf, active, _w, _s, _B = _random_blocks(seed=9, L=6, P=20)
    u[2] *= 1e-20                       # active-but-null light at index 2
    active = np.ones(6, bool)
    active[:2] = False                  # inactive lights at LOWER indices 0, 1
    state = SelectionState(u=u, M0=M0, lam0=lam0, finf=finf, active=active)
    for policy in ("e_opt", "a_opt", "d_opt"):
        o1, _ = select_ordering(state, policy)
        o2, _ = _reference_greedy(state, policy)
        assert o1 == o2
        assert o1[-3:] == [0, 1, 2]     # zero-gain tail: ascending index
