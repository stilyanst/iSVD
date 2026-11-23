"""The rank-1 discarded amount (paper Prop. 1(a)).

eta_s = ||delta (e_i e_j^T - P_U e_i e_j^T P_V)||_F, with P_U = U U^T and
P_V = V V^T taken from the factors before the update, computed here as
explicit m x n matrices and compared with inc.last_eta.
"""
import numpy as np
import pytest

from isvd import policies as rp
from helpers import make_engine, rank1_args

N_STEPS = 300


def discarded_matrix(U, Vt, i, j, delta):
    m, n = U.shape[0], Vt.shape[1]
    E = np.zeros((m, n))
    E[i, j] = 1.0
    P_U = U @ U.T
    P_V = Vt.T @ Vt
    return delta * (E - P_U @ E @ P_V)


@pytest.mark.parametrize("refresh_every", [None, 25], ids=["no-refresh", "periodic25"])
def test_rank1_eta_equals_direct_norm(refresh_every):
    inc, rng = make_engine(seed=3)
    policy = rp.resolve_policy(refresh_every=refresh_every)
    state = rp.new_state()
    for t in range(1, N_STEPS + 1):
        i, j, delta = rank1_args(inc, rng)
        D = discarded_matrix(inc.U, inc.Vt, i, j, delta)
        inc.rank1_update(i, j, delta)
        direct = np.linalg.norm(D, "fro")
        assert inc.last_eta == pytest.approx(direct, rel=1e-10, abs=1e-14 * abs(delta)), f"step {t}"
        rp.apply_refresh_policy(policy, t, inc, state)


def test_rank1_discarded_matrix_is_the_error_increment():
    """E_s - E_{s-1} equals delta (e_i e_j^T - P_U e_i e_j^T P_V) entrywise, so the
    formula describes what the update actually discards (not only its norm)."""
    inc, rng = make_engine(seed=4)
    for t in range(1, N_STEPS + 1):
        E_prev = inc.A - inc.reconstruct()
        i, j, delta = rank1_args(inc, rng)
        D = discarded_matrix(inc.U, inc.Vt, i, j, delta)
        inc.rank1_update(i, j, delta)
        E_new = inc.A - inc.reconstruct()
        np.testing.assert_allclose(E_new - E_prev, D, rtol=0, atol=1e-12, err_msg=f"step {t}")
