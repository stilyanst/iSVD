"""Shared stream generators for the tests (small matrices, fixed seeds)."""
import numpy as np

from isvd import IncrementalSVD

M0, N0, TRUE_RANK, K = 20, 15, 6, 4
NOISE = 0.05


def low_rank_plus_noise(rng, m=M0, n=N0, r=TRUE_RANK, noise=NOISE):
    """m x n matrix of rank r plus Gaussian noise (so E_opt(0) > 0 for k < r)."""
    return rng.standard_normal((m, r)) @ rng.standard_normal((r, n)) + noise * rng.standard_normal((m, n))


def make_engine(seed, k=K, **kw):
    rng = np.random.default_rng(seed)
    inc = IncrementalSVD(low_rank_plus_noise(rng, **kw), rank=k, compute_mask=False)
    return inc, rng


def new_row(inc, rng, noise=NOISE):
    """A row near the current row space: mix of 3 existing rows plus noise."""
    m, n = inc.A.shape
    idx = rng.choice(m, size=3, replace=False)
    return rng.standard_normal(3) @ inc.A[idx] / np.sqrt(3) + noise * rng.standard_normal(n)


def new_col(inc, rng, noise=NOISE):
    """A column near the current column space: mix of 3 existing columns plus noise."""
    m, n = inc.A.shape
    idx = rng.choice(n, size=3, replace=False)
    return inc.A[:, idx] @ rng.standard_normal(3) / np.sqrt(3) + noise * rng.standard_normal(m)


def rank1_args(inc, rng):
    m, n = inc.A.shape
    return int(rng.integers(m)), int(rng.integers(n)), float(rng.standard_normal())


def apply_update(inc, rng, kind):
    """Apply one update of the given kind ('rank1', 'row', 'col') through the engine."""
    if kind == "rank1":
        inc.rank1_update(*rank1_args(inc, rng))
    elif kind == "row":
        inc.add_row(new_row(inc, rng))
    elif kind == "col":
        inc.add_col(new_col(inc, rng))
    else:
        raise ValueError(kind)


def kinds_for(stream, n_steps, rng):
    """Update kinds for a stream: 'rank1', 'row', 'col', or 'mixed' (uniform over the three)."""
    if stream == "mixed":
        return list(rng.choice(["rank1", "row", "col"], size=n_steps))
    return [stream] * n_steps
