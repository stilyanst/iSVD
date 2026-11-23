"""Core operations on a small exact low-rank matrix, and short runs of the synthetic streams."""
import numpy as np
import pandas as pd
import pytest

import streams as ss
from isvd import IncrementalSVD


def test_core_operations_on_exact_low_rank_matrix():
    rng = np.random.default_rng(0)
    m, n, r = 10, 8, 3
    A0 = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
    inc = IncrementalSVD(A0, rank=r)
    assert inc.frobenius_error() < 1e-12
    assert inc.explained_variance_ratio() == pytest.approx(1.0, abs=1e-12)
    assert inc.subspace_principal_angle() < 1e-7

    A = np.vstack((A0, rng.standard_normal(n)))
    inc.add_row(A[-1])
    assert inc.A.shape == (m + 1, n) and np.array_equal(inc.A, A)
    assert inc.frobenius_error() <= inc.error_bound() + 1e-12

    A = np.hstack((A, rng.standard_normal((m + 1, 1))))
    inc.add_col(A[:, -1])
    assert inc.A.shape == (m + 1, n + 1) and np.array_equal(inc.A, A)

    A[2, 1] += 0.5
    inc.rank1_update(2, 1, 0.5)
    assert np.array_equal(inc.A, A)
    assert np.linalg.norm(inc.reconstruct() - A) == pytest.approx(inc.frobenius_error())
    assert inc.frobenius_error() <= inc.error_bound() + 1e-12

    inc.refresh(store_initial_subspace=True)
    s = np.linalg.svd(A, compute_uv=False)
    assert inc.frobenius_error() == pytest.approx(np.linalg.norm(s[r:]), rel=1e-10)
    assert inc.explained_variance_ratio() == pytest.approx(np.sum(s[:r] ** 2) / np.sum(s ** 2))
    assert inc.subspace_principal_angle() < 1e-7


STREAMS = {
    "rank1": (lambda R: ss.run_synthetic_rank1_stream(
        m=30, n=20, true_rank=5, k=5, T=50, delta_scale=0.1, seed=42, refresh_every=R), 50),
    "row": (lambda R: ss.run_synthetic_row_stream(
        m0=20, n=15, true_rank=4, k=4, n_rows_stream=30, noise_std=0.0, seed=0, refresh_every=R), 30),
    "col": (lambda R: ss.run_synthetic_column_stream(
        m=20, n0=15, true_rank=4, k=4, n_cols_stream=30, noise_std=0.0, seed=1, refresh_every=R), 30),
    "mixed": (lambda R: ss.run_synthetic_mixed_stream(
        m0=20, n0=15, true_rank=4, k=4, T=80, row_period=10, col_period=20, delta_scale=0.05,
        noise_std=0.0, seed=123, refresh_every=R), 80),
}


@pytest.mark.parametrize("refresh_every", [None, 10])
@pytest.mark.parametrize("stream", list(STREAMS))
def test_stream_metrics(stream, refresh_every):
    run, T = STREAMS[stream]
    df = pd.DataFrame(run(refresh_every))
    assert list(df["t"]) == list(range(T + 1))
    assert df.loc[0, "update_type"] == "init"
    n_refresh = int(df["update_type"].str.endswith("+refresh").sum())
    assert n_refresh == (0 if refresh_every is None else T // refresh_every)
    for col in ["frob_error", "frob_opt", "error_bound", "frob_norm", "evr", "angle", "angle_to_opt",
                "update_time", "opt_time"]:
        assert np.isfinite(df[col].to_numpy(float)).all(), col
    assert (df["frob_error"] >= df["frob_opt"] - 1e-10).all()
    assert (df["frob_error"] <= df["error_bound"] + 1e-10).all()
