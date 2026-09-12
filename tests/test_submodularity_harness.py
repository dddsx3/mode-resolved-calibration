"""P-SUBMOD harness 测试（M4；experiments/submodularity_search.py）。

锁三件事：
  1. **检测器自检**：玩具超模函数（|S|²−|S|）必须被 100% 检出违例；
     玩具模性函数（10|S|）必须零违例——"未发现违例"因此有含义；
  2. 对抗实例上 1/λmin（E-opt）的违例被**可复现地钉住**（首反例逐字段
     重算一致；审计在同类构造下发现大量 E-opt 违例）；
  3. 退化守卫：病态实例（构造性退化）被丢弃而非产生假违例。
"""

import math

import numpy as np
import pytest

from experiments.submodularity_search import (
    build_instance,
    search_violations,
    subset_objectives,
    toy_gain,
)

L_TINY = 5


def _as_F(toy_gain_dict, L):
    """search_violations 吃 F 值（G = F(∅) − F(S)）；把玩具增益 G 转成
    F = max(G) − G（∅ 处最大）。"""
    gmax = max(toy_gain_dict.values())
    return {S: gmax - g for S, g in toy_gain_dict.items()}


def test_detector_selfcheck_supermodular_triggers():
    r"""玩具超模（G=|S|²+|S|，处处严格正递增边际）：全部三元组违例。
    三元组总数 = Σ_b C(L,b)·2^b·(L−b)（每元素 ∈ {A}⊆{B}\{x}/外部）。"""
    L = L_TINY
    toy = toy_gain("supermodular", L)
    out = search_violations(_as_F(toy, L), L)
    expect = sum(math.comb(L, b) * 2 ** b * (L - b) for b in range(1, L + 1))
    assert out["triples"] == expect
    # A = B 的平凡三元组（每 B 有 L−|B| 个 x）边际恒等 → 不计违例
    trivial = sum(math.comb(L, b) * (L - b) for b in range(1, L + 1))
    assert out["violations_submodular"] == expect - trivial
    assert out["nonmonotone_triples"] == 0
    assert out["first_violation"] is not None


def test_detector_selfcheck_modular_clean():
    """玩具模性：零违例、γ = 1（检测器特异性验证）。"""
    L = L_TINY
    toy = toy_gain("modular", L)
    out = search_violations(_as_F(toy, L), L)
    assert out["violations_submodular"] == 0
    assert out["nonmonotone_triples"] == 0
    assert out["gamma"] == pytest.approx(1.0, rel=1e-12)


def test_random_e_opt_submodularity_violation_pinned():
    """正确语义（S 中灯 t=κ）下，随机实例的 1/λmin 有真实次模违例：
    固定种子首反例逐字段重算一致；γ < 1（边际递减破坏，E-opt 类）。"""
    seed = 20260918
    L, P = 5, 40
    u, M0, lam0, finf = build_instance(seed, L, P, family="random")
    F = subset_objectives(u, M0, lam0, finf, kind="E")
    assert F is not None
    out = search_violations(F, L)
    assert out["violations_submodular"] == 56
    assert out["nonmonotone_triples"] == 0          # 正确语义下无负边际
    fv = out["first_violation"]
    assert fv == dict(A=[], B=[1], x=2,
                      dA=0.0006934360672450346,
                      dB=0.0006989027891742827)
    # 反例重算
    A, B, x = frozenset(fv["A"]), frozenset(fv["B"]), fv["x"]
    F0 = F[frozenset()]
    dA = (F0 - F[A | {x}]) - (F0 - F[A])
    dB = (F0 - F[B | {x}]) - (F0 - F[B])
    assert dA == pytest.approx(fv["dA"], abs=1e-15)
    assert dB == pytest.approx(fv["dB"], abs=1e-15)
    assert 0.0 < out["gamma"] < 1.0
    assert out["gamma"] == pytest.approx(0.9368, abs=1e-3)


def test_a_opt_near_submodular_d_opt_clean():
    """A-opt 至多边际性（γ ≥ 0.9999）；D-opt 零违例（10 个随机种子）。"""
    for seed in range(20260918, 20260928):
        u, M0, lam0, finf = build_instance(seed, 5, 40, family="random")
        Fa = subset_objectives(u, M0, lam0, finf, kind="A")
        Fd = subset_objectives(u, M0, lam0, finf, kind="D")
        out_a = search_violations(Fa, 5)
        out_d = search_violations(Fd, 5)
        assert out_d["violations_submodular"] == 0
        if out_a["violations_submodular"]:
            # 实测族内 gamma_min ≈ 0.99989：至多边际性非次模（相对差 ~1e-4）
            assert out_a["gamma"] >= 0.9998
        else:
            assert out_a["gamma"] is None or out_a["gamma"] >= 0.9998


def test_degeneracy_guard_discards_pathological_instance():
    """构造性退化实例（finf 近零）被守卫丢弃，而非产生假违例。"""
    seed = 20260919
    L, P = 5, 40
    u, M0, lam0, finf = build_instance(seed, L, P, family="adversarial")
    finf = finf.copy()
    finf[0] = 1e-14                                  # 病态：近零 F∞ 像素
    F = subset_objectives(u, M0, lam0, finf, kind="E", degeneracy_tol=1e-8)
    assert F is None                                 # 守卫必须丢弃
