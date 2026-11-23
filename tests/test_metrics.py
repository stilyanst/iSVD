"""isvd.metrics: the optimal rank-k error and the per-step metrics."""
import numpy as np
import pytest

from isvd import IncrementalSVD
from isvd.metrics import compute_metrics, optimal_rank_k_error


@pytest.mark.parametrize("k", [1, 3, 6, 15])
def test_optimal_error_is_norm_of_discarded_singular_values(k):
    rng = np.random.default_rng(k)
    A = rng.standard_normal((20, 15))
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    assert optimal_rank_k_error(A, U, S, Vt, k) == pytest.approx(np.linalg.norm(S[k:]), rel=1e-10, abs=1e-12)


def test_compute_metrics_at_a_full_svd():
    """Right after a full SVD the incremental factors are optimal: ratio 1, angle 0."""
    rng = np.random.default_rng(0)
    inc = IncrementalSVD(rng.standard_normal((20, 15)), rank=4)
    m = compute_metrics(inc, compute_optimal=True)
    assert m["frob_error"] == pytest.approx(m["frob_opt"], rel=1e-12)
    assert m["frob_ratio"] == pytest.approx(1.0, rel=1e-12)
    assert m["error_bound"] == pytest.approx(m["frob_error"], rel=1e-12)
    assert m["angle"] < 1e-7 and m["angle_to_opt"] < 1e-7
    assert m["frob_norm"] == pytest.approx(np.linalg.norm(inc.A))
