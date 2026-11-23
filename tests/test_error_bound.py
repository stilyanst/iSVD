"""||A_t - A_hat_t||_F <= beta_t + 1e-10 at every step (paper Prop. 1(b))."""
import numpy as np
import pytest

from isvd import policies as rp
from isvd import IncrementalSVD
from helpers import K, M0, N0, apply_update, kinds_for, make_engine

ATOL = 1e-10
N_STEPS = 300
REFRESH = [None, 25]  # (e): no refresh, and periodic refresh every 25 updates


def assert_bound(inc, t):
    err = inc.frobenius_error()
    beta = inc.error_bound()
    assert err <= beta + ATOL, f"step {t}: E_inc = {err!r} > beta_t = {beta!r}"


@pytest.mark.parametrize("refresh_every", REFRESH, ids=["no-refresh", "periodic25"])
@pytest.mark.parametrize("stream", ["rank1", "row", "col", "mixed"])  # (a)-(d)
def test_bound_holds_every_step(stream, refresh_every):
    inc, rng = make_engine(seed=1)
    policy = rp.resolve_policy(refresh_every=refresh_every)
    state = rp.new_state()
    assert_bound(inc, 0)
    n_refresh = 0
    for t, kind in enumerate(kinds_for(stream, N_STEPS, rng), start=1):
        apply_update(inc, rng, kind)
        assert_bound(inc, t)
        if rp.apply_refresh_policy(policy, t, inc, state):
            n_refresh += 1
            assert_bound(inc, t)
    assert n_refresh == (0 if refresh_every is None else N_STEPS // refresh_every)


def _rho_row(inc, x):
    return np.linalg.norm(x - (x @ inc.Vt.T) @ inc.Vt)


def _rho_col(inc, y):
    return np.linalg.norm(y - inc.U @ (inc.U.T @ y))


@pytest.mark.parametrize("refresh_every", REFRESH, ids=["no-refresh", "periodic25"])
@pytest.mark.parametrize("variant", ["in-span", "near-span"])
@pytest.mark.parametrize("axis", ["row", "col"])
def test_bound_holds_when_rho_below_tol(axis, variant, refresh_every):
    """(f) Appends that take the rho <= tol branch (residual dropped, no new direction).

    in-span:   A_0 has exact rank k and every append lies in the current span, so
               rho is at roundoff level (< default tol 1e-10).
    near-span: append = span part + 1e-4 noise, with tol = 1e-2, so a residual of
               norm ~1e-4 is dropped and eta_s = rho is not negligible.
    """
    rng = np.random.default_rng(2)
    if variant == "in-span":
        A0 = rng.standard_normal((M0, K)) @ rng.standard_normal((K, N0))
        tol, noise = 1e-10, 0.0
    else:
        A0 = rng.standard_normal((M0, 6)) @ rng.standard_normal((6, N0)) + 0.05 * rng.standard_normal((M0, N0))
        tol, noise = 1e-2, 1e-4
    inc = IncrementalSVD(A0, rank=K, compute_mask=False)
    policy = rp.resolve_policy(refresh_every=refresh_every)
    state = rp.new_state()
    assert_bound(inc, 0)
    dropped = 0.0
    # fixed scale for the span part, so ||A_t||_F grows linearly (scaling by S would
    # grow it geometrically and drive E_opt / ||A||_F into the cancellation regime
    # of the next test)
    scale = 3.0
    for t in range(1, 101):
        if axis == "row":
            x = scale * rng.standard_normal(K) @ inc.Vt + noise * rng.standard_normal(inc.A.shape[1])
            rho = _rho_row(inc, x)
            assert rho < tol, f"step {t}: rho = {rho} does not take the rho <= tol branch"
            inc.add_row(x, tol=tol)
        else:
            y = scale * inc.U @ rng.standard_normal(K) + noise * rng.standard_normal(inc.A.shape[0])
            rho = _rho_col(inc, y)
            assert rho < tol, f"step {t}: rho = {rho} does not take the rho <= tol branch"
            inc.add_col(y, tol=tol)
        # branch taken: rank stays k with no truncation, so eta_s is exactly the dropped rho
        assert len(inc.S) == K
        assert inc.last_eta == pytest.approx(rho, rel=1e-12, abs=1e-12)
        dropped += rho
        assert_bound(inc, t)
        if rp.apply_refresh_policy(policy, t, inc, state):
            assert_bound(inc, t)
    if variant == "near-span":
        assert dropped > 1e-3  # the bound was exercised by non-negligible dropped residuals


def test_bound_at_full_svd_when_E_opt_is_tiny_relative_to_A():
    """At t0 (initial SVD) E_inc = E_opt(t0) exactly, so beta_t0 must equal it.

    A_0 = rank-k + 1e-9 noise, ||A_0||_F ~ 37, E_opt(0) ~ 1e-8 (ratio ~3e-10).
    beta_0 is ||S_full[k:]||_2; the equal expression sqrt(max(0, N - ||S||^2))
    cancels to 0 in most of the 20 seeds.
    """
    failures = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        A0 = rng.standard_normal((M0, K)) @ rng.standard_normal((K, N0)) + 1e-9 * rng.standard_normal((M0, N0))
        inc = IncrementalSVD(A0, rank=K, compute_mask=False)
        err, beta = inc.frobenius_error(), inc.error_bound()
        if not err <= beta + ATOL:
            failures.append((seed, err, beta))
    assert not failures, f"{len(failures)}/20 seeds: (seed, E_inc, beta_0) = {failures[:3]}"
