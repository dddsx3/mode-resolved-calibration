"""Identifiable-subspace contract, including empty information."""
import numpy as np
import pytest

from calibinfo.information.retention import retention_spectrum, retention_spectrum_dual


@pytest.mark.parametrize("rotated", [False, True])
def test_rank_deficient_spectrum(rotated):
    Q = np.array([[0.6, -0.8], [0.8, 0.6]]) if rotated else np.eye(2)
    Finf = Q @ np.diag([1.0, 0.0]) @ Q.T
    DeltaF = 0.5 * Finf
    out = retention_spectrum(DeltaF, Finf)
    np.testing.assert_allclose(out["rho"], [0.5])
    assert out["n_identifiable"] == 1
    assert out["basis"].shape == out["modes"].shape == (2, 1)
    np.testing.assert_allclose(out["modes"].T @ out["modes"], np.eye(1))
    np.testing.assert_allclose(Finf @ out["modes"], out["modes"], atol=1e-12)


def test_zero_rank_empty_contract():
    for size in [0, 2]:
        out = retention_spectrum(np.zeros((size, size)), np.zeros((size, size)))
        assert out["rho"].shape == (0,)
        assert out["modes"].shape == out["basis"].shape == (size, 0)
        assert out["n_identifiable"] == 0 and out["bounds_ok"]
        assert np.isinf(out["cond_Finf"])
    out = retention_spectrum_dual(np.zeros((2, 2)), np.zeros((2, 2)))
    assert out["dual_rel"] is None
