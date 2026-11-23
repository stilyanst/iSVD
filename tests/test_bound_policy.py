"""The bound(tau) policy refreshes exactly when beta_t / sqrt(N_t) > tau."""
from types import SimpleNamespace

import numpy as np
import pytest

from isvd import policies as rp
from helpers import apply_update, kinds_for, make_engine

N_STEPS = 400


@pytest.mark.parametrize("stream", ["rank1", "row", "col", "mixed"])
@pytest.mark.parametrize("tau_factor", [1.05, 1.5])
def test_bound_policy_refreshes_iff_ratio_exceeds_tau(stream, tau_factor):
    """tau is set relative to the floor beta_0 / ||A_0||_F = E_opt(0) / ||A_0||_F."""
    inc, rng = make_engine(seed=30)
    tau = tau_factor * inc.error_bound() / np.sqrt(inc.N)
    policy = rp.bound(tau)
    state = rp.new_state()
    decisions = []
    for t, kind in enumerate(kinds_for(stream, N_STEPS, rng), start=1):
        apply_update(inc, rng, kind)
        ratio = inc.error_bound() / np.sqrt(inc.N)
        expected = bool(ratio > tau)
        refreshed = rp.apply_refresh_policy(policy, t, inc, state)
        assert refreshed == expected, f"step {t}: ratio = {ratio!r}, tau = {tau!r}, refreshed = {refreshed}"
        if refreshed:
            assert state["last_refresh"] == t
            # after the refresh beta_t restarts at E_opt(t) = E_inc(t)
            assert inc.error_bound() == pytest.approx(inc.frobenius_error(), rel=1e-8)
        decisions.append(refreshed)
    assert state["n_refreshes"] == sum(decisions)
    assert 0 < sum(decisions) < N_STEPS, "stream should exercise both outcomes"


def _fake(beta, N):
    return SimpleNamespace(error_bound=lambda: beta, N=N)


def test_bound_policy_boundary_is_strict():
    """ratio == tau does not refresh; the next float above does. N = 4, beta = 1 -> ratio 0.5 exactly."""
    tau = 0.5
    should = rp.bound(tau)
    assert should(1, _fake(1.0, 4.0), rp.new_state()) is False
    # float(): inc.error_bound() returns a Python float, np.nextafter a NumPy scalar
    assert should(1, _fake(float(np.nextafter(1.0, 2.0)), 4.0), rp.new_state()) is True
    assert should(1, _fake(float(np.nextafter(1.0, 0.0)), 4.0), rp.new_state()) is False


def test_bound_policy_ignores_t_and_state_and_never_fires_at_N_zero():
    should = rp.bound(0.1)
    for t in (1, 7, 1000):
        state = {"last_refresh": t - 1, "n_refreshes": 5}
        assert should(t, _fake(1.0, 4.0), state) is True
        assert should(t, _fake(0.1, 4.0), state) is False
    assert should(1, _fake(1.0, 0.0), rp.new_state()) is False


@pytest.mark.parametrize("tau", [0, -0.1, None])
def test_bound_policy_rejects_nonpositive_tau(tau):
    with pytest.raises(ValueError):
        rp.bound(tau)
