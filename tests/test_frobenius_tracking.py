"""N_t (tracked in O(1) per update) equals ||A_t||_F^2 to 1e-8 relative."""
import numpy as np
import pytest

from isvd import policies as rp
from helpers import apply_update, kinds_for, make_engine

RTOL = 1e-8
N_STEPS = 400


def assert_N(inc, t):
    exact = float(np.linalg.norm(inc.A, "fro") ** 2)
    assert inc.N == pytest.approx(exact, rel=RTOL), f"step {t}: N_t = {inc.N!r}, ||A_t||_F^2 = {exact!r}"


@pytest.mark.parametrize("refresh_every", [None, 25], ids=["no-refresh", "periodic25"])
@pytest.mark.parametrize("seed", [10, 11, 12])
def test_N_tracks_frobenius_norm_mixed_stream(seed, refresh_every):
    """Mixed stream of row/column appends and rank-1 updates, plus set_entry(mode='none')
    assignments (the other mutator that changes A without a full SVD)."""
    inc, rng = make_engine(seed=seed)
    policy = rp.resolve_policy(refresh_every=refresh_every)
    state = rp.new_state()
    kinds = kinds_for("mixed", N_STEPS, rng)
    for t, kind in enumerate(kinds, start=1):
        if t % 7 == 0:
            m, n = inc.A.shape
            inc.set_entry(int(rng.integers(m)), int(rng.integers(n)), float(rng.standard_normal()), mode="none")
        else:
            apply_update(inc, rng, kind)
        assert_N(inc, t)
        rp.apply_refresh_policy(policy, t, inc, state)
    assert_N(inc, N_STEPS)
    assert inc.A.shape[0] > 20 and inc.A.shape[1] > 15  # rows and columns were appended
