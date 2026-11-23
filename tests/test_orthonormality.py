"""U^T U = I and Vt Vt^T = I after every update type."""
import numpy as np
import pytest

from helpers import apply_update, kinds_for, make_engine

ATOL = 1e-10
N_STEPS = 300


def assert_orthonormal(inc, where):
    m, n = inc.A.shape
    r = len(inc.S)
    assert inc.U.shape == (m, r) and inc.Vt.shape == (r, n), where
    I = np.eye(r)
    dU = np.max(np.abs(inc.U.T @ inc.U - I))
    dV = np.max(np.abs(inc.Vt @ inc.Vt.T - I))
    assert dU <= ATOL, f"{where}: max |U^T U - I| = {dU:.3e}"
    assert dV <= ATOL, f"{where}: max |Vt Vt^T - I| = {dV:.3e}"


@pytest.mark.parametrize("stream", ["rank1", "row", "col", "mixed"])
def test_orthonormal_after_each_incremental_update(stream):
    inc, rng = make_engine(seed=20)
    assert_orthonormal(inc, "init")
    for t, kind in enumerate(kinds_for(stream, N_STEPS, rng), start=1):
        apply_update(inc, rng, kind)
        assert_orthonormal(inc, f"step {t} ({kind})")


@pytest.mark.parametrize("axis", ["row", "col"])
def test_orthonormal_after_append_with_rho_below_tol(axis):
    """Appends that take the rho < tol branch (residual dropped, no new direction)."""
    inc, rng = make_engine(seed=21)
    for t in range(1, 101):
        if axis == "row":
            x = 3.0 * rng.standard_normal(len(inc.S)) @ inc.Vt + 1e-4 * rng.standard_normal(inc.A.shape[1])
            inc.add_row(x, tol=1e-2)
        else:
            y = 3.0 * inc.U @ rng.standard_normal(len(inc.S)) + 1e-4 * rng.standard_normal(inc.A.shape[0])
            inc.add_col(y, tol=1e-2)
        assert_orthonormal(inc, f"step {t} ({axis}, rho < tol)")


def test_orthonormal_after_refresh_and_full_recompute():
    inc, rng = make_engine(seed=22)
    for t, kind in enumerate(kinds_for("mixed", 60, rng), start=1):
        apply_update(inc, rng, kind)
        if t % 20 == 0:
            inc.refresh()
            assert_orthonormal(inc, f"refresh at step {t}")
    m, n = inc.A.shape
    inc.set_entry(0, 0, 1.5, mode="full")
    assert_orthonormal(inc, "set_entry(mode='full')")
    inc.rank1_update(1, 1, 0.7, incremental=False)
    assert_orthonormal(inc, "rank1_update(incremental=False)")
    inc.add_row(rng.standard_normal(n), incremental=False)
    assert_orthonormal(inc, "add_row(incremental=False)")
    inc.add_col(rng.standard_normal(m + 1), incremental=False)
    assert_orthonormal(inc, "add_col(incremental=False)")
