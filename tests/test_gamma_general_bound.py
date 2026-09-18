"""γ 数学修正回归：历史候选被否定，谱界须在一般 SPD+PSD 类上有效。

不读取或改写历史 results，不依赖 SymPy。旧 toy/search 测试仍是历史有限
样本回归，不是候选式的定理证明；符号证据由仓库外 verification_v5 独立生成。
"""

import itertools
import warnings

import numpy as np
import pytest

from calibinfo.allocation.alpha_bound import (
    gamma_lower_bound,
    spectral_gamma_lower_bound,
)


def _family(p=1.0, eps=0.1):
    """固定 α=p 的二维非交换族；只使用 eps>0。"""
    A = np.diag([1.0, eps**2])
    return A, [p * np.diag([1.0, 0.0]),
               (p / 2) * np.outer([1.0, eps], [1.0, eps])]


def _alpha(A, updates):
    values, vectors = np.linalg.eigh(A)
    inv_sqrt = (vectors / np.sqrt(values)) @ vectors.T
    return max(float(np.linalg.eigvalsh(inv_sqrt @ W @ inv_sqrt)[-1])
               for W in updates)


def _gain(M, W):
    """独立 Woodbury 计算，避免两个接近的逆迹直接相减。"""
    values, vectors = np.linalg.eigh(W)
    factor = vectors * np.sqrt(np.maximum(values, 0.0))
    solved = np.linalg.solve(M, factor)
    return float(np.trace(np.linalg.solve(
        np.eye(M.shape[0]) + factor.T @ solved, solved.T @ solved)))


def _ratios(A, updates):
    """穷举所有 S⊆T⊆U\\{x}，仅排除零更新，不筛掉小的正边际。"""
    m = len(updates)
    matrices = {}
    for bits in range(1 << m):
        matrices[bits] = A.copy()
        for i, W in enumerate(updates):
            if bits & (1 << i):
                matrices[bits] += W
    for x, W in enumerate(updates):
        if not np.any(W):
            continue
        for T in range(1 << m):
            if T & (1 << x):
                continue
            denominator = _gain(matrices[T], W)
            assert denominator > 0
            for S in range(1 << m):
                if S & T == S:
                    numerator = _gain(matrices[S], W)
                    assert numerator > 0
                    yield numerator / denominator, matrices[S], matrices[T]


def _gamma(A, updates):
    return min((ratio for ratio, _, _ in _ratios(A, updates)), default=1.0)


def test_rational_counterexample_gamma_alpha_and_spectral_bound():
    """全六个三元组给出 γ=14/109、α=1，旧候选确实违反。"""
    A, updates = _family()
    W1, W2 = updates
    assert _alpha(A, updates) == pytest.approx(1.0, abs=1e-14)
    assert _gain(A, W1) == pytest.approx(1 / 2)
    assert _gain(A + W2, W1) == pytest.approx(109 / 28)
    ratios = sorted(ratio for ratio, _, _ in _ratios(A, updates))
    np.testing.assert_allclose(ratios, [14 / 109, 707 / 802, 1, 1, 1, 1])
    assert ratios[0] < gamma_lower_bound(1.0)
    bound = spectral_gamma_lower_bound(A, updates)
    assert isinstance(bound, float)
    assert bound == pytest.approx(1 / 200)
    assert 0 < bound <= ratios[0]
    full_bound = np.linalg.eigvalsh(A)[0] / np.linalg.eigvalsh(A + W1 + W2)[-1]
    assert full_bound < bound
    inverse = np.linalg.inv(A + W2)
    assert np.trace(W1 @ inverse @ inverse) == pytest.approx(109 / 16)


@pytest.mark.parametrize("eps", [1.0, 0.3, 0.1, 0.01, 0.001])
def test_epsilon_family_keeps_alpha_one(eps):
    A, updates = _family(eps=eps)
    d0 = _gain(A, updates[0])
    d2 = _gain(A + updates[1], updates[0])
    expected = 14 * eps**2 / (1 + 9 * eps**2)
    assert _alpha(A, updates) == pytest.approx(1.0)
    assert d0 == pytest.approx(0.5)
    assert d2 == pytest.approx((1 + 9 * eps**2) / (28 * eps**2))
    assert d0 / d2 == pytest.approx(expected)
    gamma = _gamma(A, updates)
    assert 0 < spectral_gamma_lower_bound(A, updates) <= gamma + 1e-13
    assert gamma <= min(1.0, expected) + 1e-13


@pytest.mark.parametrize("p", [0.01, 0.25, 1.0, 3.0, 10.0])
def test_every_fixed_positive_alpha_has_arbitrarily_small_ratios(p):
    """有限点核验精确族公式；趋零的一般结论来自解析式而非随机实验。"""
    threshold = p / (2 * p**2 + 9 * p + 8)
    ratios = []
    for fraction in [0.5, 0.05, 0.0005]:
        eps = np.sqrt(threshold) * fraction
        A, updates = _family(p, eps)
        numerator = _gain(A, updates[0])
        denominator = _gain(A + updates[1], updates[0])
        expected_denominator = (p * (p**2 + eps**2 * (p + 2)**2)
                                / (2 * eps**2 * (p + 1) * (p**2 + 4*p + 2)))
        ratio = 2 * eps**2 * (p**2 + 4*p + 2) / (p**2 + eps**2 * (p + 2)**2)
        assert _alpha(A, updates) == pytest.approx(p)
        assert numerator == pytest.approx(p / (p + 1))
        assert denominator == pytest.approx(expected_denominator)
        assert numerator / denominator == pytest.approx(ratio)
        assert ratio < gamma_lower_bound(p)
        gamma = _gamma(A, updates)
        assert spectral_gamma_lower_bound(A, updates) <= gamma + 1e-13
        assert gamma <= ratio + 1e-13
        ratios.append(ratio)
    assert ratios[0] > ratios[1] > ratios[2] > 0
    assert ratios[2] < 1e-3


@pytest.mark.parametrize("kind", ["empty-list", "empty-array", "empty-generator",
                                  "all-zero"])
def test_empty_or_all_zero_updates_have_explicit_gamma_one_convention(kind):
    A = np.diag([2.0, 5.0])
    updates = {"empty-list": [], "empty-array": np.empty((0, 2, 2)),
               "empty-generator": iter(()), "all-zero": [np.zeros((2, 2))] * 3}[kind]
    assert spectral_gamma_lower_bound(A, updates) == 1.0
    assert _gamma(A, [np.zeros((2, 2))] * 3) == 1.0


def test_zero_candidates_are_excluded_without_weakening_bound():
    A, updates = _family()
    padded = [np.zeros_like(A), *updates, np.zeros_like(A)]
    assert spectral_gamma_lower_bound(A, padded) == spectral_gamma_lower_bound(A, updates)
    assert _gamma(A, padded) == pytest.approx(14 / 109)
    # 唯一非零候选的谱界为基线条件数倒数；错误包含零候选会削弱该界。
    A = np.diag([2.0, 4.0])
    W = np.diag([0.0, 100.0])
    assert spectral_gamma_lower_bound(A, [W, np.zeros_like(A)]) == 0.5
    assert _gamma(A, [W, np.zeros_like(A)]) == 1.0


@pytest.mark.parametrize("rotated", [False, True])
def test_commuting_matrices_have_true_gamma_one_not_necessarily_bound_one(rotated):
    A = np.diag([0.3, 1.0, 4.0])
    updates = [np.diag([1.0, 0.0, 0.4]), np.diag([0.0, 2.0, 0.1]), np.zeros((3, 3))]
    if rotated:
        Q = np.linalg.qr(np.random.default_rng(197).normal(size=(3, 3)))[0]
        A = Q @ A @ Q.T
        updates = [Q @ W @ Q.T for W in updates]
    assert _gamma(A, updates) == pytest.approx(1.0, abs=1e-13)
    assert 0 < spectral_gamma_lower_bound(A, updates) < 1.0


@pytest.mark.parametrize("dimension,seed", list(itertools.product(range(1, 5), range(5))))
def test_random_small_instances_exhaustive_spectral_bounds(dimension, seed):
    """每例四个非零更新、108 个有效三元组；包括秩一与满秩更新。"""
    rng = np.random.default_rng(5900 + 10 * dimension + seed)
    Q = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    A = (Q * 10**rng.uniform(-1.5, 1.5, dimension)) @ Q.T
    updates = []
    for i in range(4):
        factor = rng.normal(size=(dimension, 1 if i % 2 == 0 else dimension))
        updates.append((factor @ factor.T) * 10**rng.uniform(-1, 1))
    bound = spectral_gamma_lower_bound(A, updates)
    full_bound = np.linalg.eigvalsh(A)[0] / np.linalg.eigvalsh(A + sum(updates))[-1]
    assert 0 < full_bound <= bound + 1e-12 <= 1 + 1e-12
    triples = list(_ratios(A, updates))
    assert len(triples) == 4 * 3**3
    for ratio, M, N in triples:
        pairwise = np.linalg.eigvalsh(N)[0] / np.linalg.eigvalsh(M)[-1]
        assert pairwise <= ratio + 1e-10 * max(1, ratio)
        assert bound <= ratio + 1e-10 * max(1, ratio)


@pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
def test_scale_invariance_and_no_input_mutation(scale):
    A, updates = _family()
    A *= scale
    updates = np.stack(updates) * scale
    before_A, before_updates = A.copy(), updates.copy()
    assert spectral_gamma_lower_bound(A, updates) == pytest.approx(1 / 200)
    np.testing.assert_array_equal(A, before_A)
    np.testing.assert_array_equal(updates, before_updates)


def test_generator_and_python_list_inputs():
    A, updates = _family()
    expected = spectral_gamma_lower_bound(A, updates)
    assert spectral_gamma_lower_bound(A.tolist(), (W.tolist() for W in updates)) == expected


def test_large_single_update_does_not_cancel_baseline():
    assert spectral_gamma_lower_bound(np.eye(2), [1e300 * np.eye(2)]) == 1.0
    # 输入均有限，但先求未缩放总和会溢出；共同缩放应安全。
    assert spectral_gamma_lower_bound(
        1e308 * np.eye(2), [1e308 * np.eye(2), 1e308 * np.eye(2)]) == pytest.approx(0.5)


def test_tiny_nonzero_update_is_not_treated_as_zero():
    assert spectral_gamma_lower_bound(
        np.diag([1.0, 2.0]), [np.diag([1e-200, 0.0])]) == 0.5


def test_roundoff_tolerance_is_relative_and_inputs_remain_unchanged():
    W = np.array([[1.0, 1e-15], [0.0, -1e-15]])
    before = W.copy()
    assert spectral_gamma_lower_bound(np.eye(2), [W]) == pytest.approx(1.0)
    np.testing.assert_array_equal(W, before)
    with pytest.raises(ValueError, match="半正定"):
        spectral_gamma_lower_bound(np.eye(2), [np.diag([1e-100, -1e-110])])


@pytest.mark.parametrize("A", [
    None, 1.0, [], np.empty((0, 0)), [1.0, 2.0], np.ones((2, 3)),
    np.ones((1, 2, 2)), [[1.0], [1.0, 2.0]], [["bad"]],
    np.eye(2, dtype=complex), [[1.0, 1e-6], [0.0, 1.0]],
    np.diag([1.0, np.nan]), np.diag([1.0, np.inf]),
    np.zeros((2, 2)), np.diag([1.0, 0.0]), np.diag([1.0, -1e-12]),
])
def test_invalid_baselines_rejected_even_for_no_updates(A):
    with pytest.raises(ValueError):
        spectral_gamma_lower_bound(A, [])


@pytest.mark.parametrize("updates", [
    None, 1.0, np.eye(2), [None], [1.0], [np.ones((2, 3))], [np.eye(3)],
    [np.empty((0, 0))], [[[1.0], [1.0, 2.0]]], [[["bad"]]],
    [np.eye(2, dtype=complex)], [np.diag([1.0, np.nan])],
    [np.diag([1.0, -np.inf])], [np.array([[1.0, 1e-6], [0.0, 1.0]])],
    [np.diag([1.0, -1e-10])], [-np.eye(2)], [-1e-200 * np.eye(2)],
    [np.zeros((2, 2)), -np.eye(2)],
])
def test_invalid_updates_rejected(updates):
    with pytest.raises(ValueError):
        spectral_gamma_lower_bound(np.eye(2), updates)


def test_unrepresentable_positive_spectral_ratio_is_rejected():
    with pytest.raises(ValueError, match="浮点"):
        spectral_gamma_lower_bound(np.diag([1e-300, 1.0]), [1e300 * np.eye(2)])


@pytest.mark.parametrize("alpha", [0.0, 0.187584, 0.514558, 1.0, 3.0,
                                    -0.5, -2.0, float("inf"), float("nan")])
def test_legacy_candidate_preserves_exact_numeric_behavior_without_warnings(alpha):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gamma_lower_bound(alpha)
    expected = 1.0 / (1.0 + alpha)
    assert not caught
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == expected
    assert "已被反例否决" in gamma_lower_bound.__doc__


def test_legacy_candidate_preserves_array_and_zero_division_behavior():
    values = np.array([0.0, 1.0, 3.0])
    np.testing.assert_array_equal(gamma_lower_bound(values), 1.0 / (1.0 + values))
    with pytest.raises(ZeroDivisionError):
        gamma_lower_bound(-1.0)
