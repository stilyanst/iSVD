"""isvd.metrics.principal_angle, the one principal-angle implementation.

It is checked against an independent formula (singular values only, without
padding: the former finance.principal_angle). The two SVD calls differ only in
roundoff, so the cosines agree to 1e-14 and the angles to roundoff / sin(angle)
(arccos amplifies roundoff near 0). IncrementalSVD.subspace_principal_angle and
compute_metrics use the same function.
"""
import numpy as np
import pytest

from isvd import IncrementalSVD
from isvd.metrics import compute_metrics, principal_angle


def reference_angle(Q1, Q2):
    k = min(Q1.shape[1], Q2.shape[1])
    s = np.linalg.svd(Q1[:, :k].T @ Q2[:, :k], compute_uv=False)
    return float(np.arccos(max(0.0, min(1.0, float(np.min(s))))))


def _orthonormal(rng, m, k):
    return np.linalg.qr(rng.standard_normal((m, k)))[0]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("perturbation", [0.0, 1e-6, 1e-2, 1.0])
def test_agrees_with_reference_same_dimensions(seed, perturbation):
    rng = np.random.default_rng(seed)
    m, k = 40, int(rng.integers(1, 13))
    Q1 = _orthonormal(rng, m, k)
    Q2 = _orthonormal(rng, m, k) if perturbation == 1.0 else np.linalg.qr(
        Q1 + perturbation * rng.standard_normal((m, k)))[0]
    a = principal_angle(Q1, Q2)
    assert 0.0 <= a <= np.pi / 2
    assert np.cos(a) == pytest.approx(np.cos(reference_angle(Q1, Q2)), abs=1e-14)
    assert a == pytest.approx(reference_angle(Q1, Q2), abs=1e-7, rel=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_agrees_with_reference_different_ranks(seed):
    """Different subspace dimensions: the first min(k1, k2) columns are used."""
    rng = np.random.default_rng(100 + seed)
    Q1 = _orthonormal(rng, 30, 8)
    Q2 = np.linalg.qr(Q1[:, :5] + 0.1 * rng.standard_normal((30, 5)))[0]
    for A, B in ((Q1, Q2), (Q2, Q1)):
        assert principal_angle(A, B) == pytest.approx(reference_angle(A, B), abs=1e-7, rel=1e-12)


def test_rows_appended_pad_with_zeros():
    """A basis taken before rows were appended is compared after zero padding."""
    rng = np.random.default_rng(3)
    Q_old = _orthonormal(rng, 20, 4)
    Q_new = np.vstack((Q_old, np.zeros((5, 4))))
    assert principal_angle(Q_old, Q_new) < 1e-7
    assert principal_angle(Q_old, Q_new) == principal_angle(Q_new, Q_old)


def test_identical_subspaces_and_empty_basis():
    Q = _orthonormal(np.random.default_rng(7), 50, 6)
    assert principal_angle(Q, Q) < 1e-7
    assert np.isnan(principal_angle(Q, Q[:, :0]))


def test_engine_and_compute_metrics_use_principal_angle():
    rng = np.random.default_rng(11)
    inc = IncrementalSVD(rng.standard_normal((25, 15)), rank=4)
    for _ in range(30):
        inc.rank1_update(int(rng.integers(25)), int(rng.integers(15)), float(rng.standard_normal()))
    inc.add_row(rng.standard_normal(15))
    assert inc.subspace_principal_angle() == principal_angle(inc.U0, inc.U)
    U_full = np.linalg.svd(inc.A, full_matrices=False)[0]
    m = compute_metrics(inc, compute_optimal=True)
    assert m["angle"] == principal_angle(inc.U0, inc.U)
    assert m["angle_to_opt"] == principal_angle(U_full[:, :4], inc.U)
