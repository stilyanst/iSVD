"""Synthetic streams of the paper's Sec. 4 (ground-truth generators).

Each ``run_synthetic_*`` function draws a low-rank matrix from a seeded
generator, streams updates into an :class:`isvd.IncrementalSVD`, refreshes when
the refresh policy says so, and records :func:`isvd.metrics.compute_metrics`
(against the optimal rank-k SVD of the current matrix) after every step.
``update_time`` includes the refresh, if one was done at that step, and the
update type then ends in ``"+refresh"``.

Refreshes follow ``refresh_policy`` (see :mod:`isvd.policies`);
``refresh_every=R`` is a shortcut for ``periodic(R)``. Neither: no refresh.
"""
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np

from isvd import IncrementalSVD
from isvd import policies as rp
from isvd.metrics import compute_metrics


def run_synthetic_rank1_stream(
    m: int = 50,
    n: int = 40,
    true_rank: int = 5,
    k: int = 5,
    T: int = 200,
    delta_scale: float = 0.1,
    seed: int = 0,
    refresh_every: Optional[int] = None,
    refresh_policy: Optional[rp.RefreshPolicy] = None,
) -> List[Dict[str, Any]]:
    """Rank-1 entry updates of a random low-rank matrix (Secs. 4.1, 4.2, 4.5).

    A_0 = U V^T with standard normal U (m x true_rank) and V (n x true_rank).
    Each of the T steps adds delta ~ N(0, delta_scale^2) to a uniformly random
    entry.

    Parameters
    ----------
    m, n : int
        Matrix size.
    true_rank : int
        Rank of A_0.
    k : int
        Truncation rank.
    T : int
        Number of updates.
    delta_scale : float
        Standard deviation of the increments.
    seed : int
        Seed of the generator.
    refresh_every : int, optional
        Shortcut for ``refresh_policy=periodic(refresh_every)``.
    refresh_policy : RefreshPolicy, optional
        Refresh policy.

    Returns
    -------
    list of dict
        One dict per step t = 0..T (t = 0 is the initial SVD) with the keys of
        :func:`isvd.metrics.compute_metrics` and ``t``.
    """
    policy = rp.resolve_policy(refresh_policy, refresh_every)
    policy_state = rp.new_state()

    rng = np.random.default_rng(seed)

    # 1) Initial low-rank matrix A_0
    U_true = rng.standard_normal((m, true_rank))
    V_true = rng.standard_normal((n, true_rank))
    A0 = U_true @ V_true.T

    # 2) Initialize incremental SVD
    inc = IncrementalSVD(A0, rank=k, compute_mask=False)

    # 3) Storage for metrics
    metrics = []

    # 4) Metrics of the initial SVD (t = 0)
    A_full = A0.copy()
    m0 = compute_metrics(
        inc,
        A_full=A_full,
        update_type="init",
        update_time=0.0,
        compute_optimal=True,
    )
    m0["t"] = 0
    metrics.append(m0)

    # 5) Streaming rank-1 updates
    for t in range(1, T + 1):
        # pick random entry and perturb it
        u = rng.integers(0, m)
        v = rng.integers(0, n)
        delta = delta_scale * rng.standard_normal()

        # update ground-truth matrix
        A_full[u, v] += delta

        # time the incremental update
        tic = perf_counter()
        inc.rank1_update(u, v, delta, update_matrix=True, incremental=True)
        toc = perf_counter()
        dt = toc - tic
        update_type = "rank1"

        # refresh if the policy asks for it
        tic_r = perf_counter()
        if rp.apply_refresh_policy(policy, t, inc, policy_state):
            dt += perf_counter() - tic_r
            update_type += "+refresh"

        m_t = compute_metrics(
            inc,
            A_full=A_full,
            update_type=update_type,
            update_time=dt,
            compute_optimal=True,
        )
        m_t["t"] = t
        metrics.append(m_t)

    return metrics


def run_synthetic_column_stream(
    m: int = 20,
    n0: int = 15,
    true_rank: int = 5,
    k: int = 5,
    n_cols_stream: int = 50,
    noise_std: float = 0.0,
    seed: int = 0,
    refresh_every: Optional[int] = None,
    refresh_policy: Optional[rp.RefreshPolicy] = None,
) -> List[Dict[str, Any]]:
    """Column appends to a low-rank matrix plus noise (Sec. 4.3).

    A_0 = U V_0^T (+ noise) with standard normal U (m x true_rank) and V_0
    (n0 x true_rank). Each new column is U v + noise with a new standard
    normal v; the noise is i.i.d. N(0, noise_std^2).

    Parameters
    ----------
    m, n0 : int
        Size of A_0.
    true_rank : int
        Rank of the noise-free part.
    k : int
        Truncation rank.
    n_cols_stream : int
        Number of appended columns T.
    noise_std : float
        Noise level (0: exact low rank).
    seed : int
        Seed of the generator.
    refresh_every : int, optional
        Shortcut for ``refresh_policy=periodic(refresh_every)``.
    refresh_policy : RefreshPolicy, optional
        Refresh policy.

    Returns
    -------
    list of dict
        One dict per step t = 0..T (t = 0 is the initial SVD) with the keys of
        :func:`isvd.metrics.compute_metrics` and ``t``.
    """
    policy = rp.resolve_policy(refresh_policy, refresh_every)
    policy_state = rp.new_state()

    rng = np.random.default_rng(seed)

    # 1) Underlying low-rank model
    U_true = rng.standard_normal((m, true_rank))
    V_true_0 = rng.standard_normal((n0, true_rank))
    A0 = U_true @ V_true_0.T  # (m, n0)
    if noise_std > 0:
        A0 += noise_std * rng.standard_normal(A0.shape)

    # 2) Initialize incremental SVD
    inc = IncrementalSVD(A0, rank=k, compute_mask=False)

    metrics = []

    # t = 0 (initial)
    A_full = A0.copy()
    m0_full = compute_metrics(
        inc,
        A_full=A_full,
        update_type="init",
        update_time=0.0,
        compute_optimal=True,
    )
    m0_full["t"] = 0
    metrics.append(m0_full)

    # 3) Stream new columns
    for t in range(1, n_cols_stream + 1):
        v_new = rng.standard_normal((1, true_rank))          # (1, r_true)
        new_col = (U_true @ v_new.T).ravel()                 # (m,)
        if noise_std > 0:
            new_col += noise_std * rng.standard_normal(new_col.shape)

        A_full = np.hstack((A_full, new_col.reshape(-1, 1)))

        tic = perf_counter()
        inc.add_col(new_col, incremental=True)
        toc = perf_counter()
        dt = toc - tic
        update_type = "col"

        # refresh if the policy asks for it
        tic_r = perf_counter()
        if rp.apply_refresh_policy(policy, t, inc, policy_state):
            dt += perf_counter() - tic_r
            update_type += "+refresh"

        m_t = compute_metrics(
            inc,
            A_full=A_full,
            update_type=update_type,
            update_time=dt,
            compute_optimal=True,
        )
        m_t["t"] = t
        metrics.append(m_t)

    return metrics


def run_synthetic_row_stream(
    m0: int = 20,
    n: int = 15,
    true_rank: int = 5,
    k: int = 5,
    n_rows_stream: int = 50,
    noise_std: float = 0.0,
    seed: int = 0,
    refresh_every: Optional[int] = None,
    refresh_policy: Optional[rp.RefreshPolicy] = None,
) -> List[Dict[str, Any]]:
    """Row appends to a low-rank matrix plus noise (Sec. 4.3).

    A_0 = U_0 V^T (+ noise) with standard normal U_0 (m0 x true_rank) and V
    (n x true_rank). Each new row is u V^T + noise with a new standard normal
    u; the noise is i.i.d. N(0, noise_std^2).

    Parameters
    ----------
    m0, n : int
        Size of A_0.
    true_rank : int
        Rank of the noise-free part.
    k : int
        Truncation rank.
    n_rows_stream : int
        Number of appended rows T.
    noise_std : float
        Noise level (0: exact low rank).
    seed : int
        Seed of the generator.
    refresh_every : int, optional
        Shortcut for ``refresh_policy=periodic(refresh_every)``.
    refresh_policy : RefreshPolicy, optional
        Refresh policy.

    Returns
    -------
    list of dict
        One dict per step t = 0..T (t = 0 is the initial SVD) with the keys of
        :func:`isvd.metrics.compute_metrics` and ``t``.
    """
    policy = rp.resolve_policy(refresh_policy, refresh_every)
    policy_state = rp.new_state()

    rng = np.random.default_rng(seed)

    # 1) Underlying low-rank model
    U_true_0 = rng.standard_normal((m0, true_rank))
    V_true = rng.standard_normal((n, true_rank))
    A0 = U_true_0 @ V_true.T  # (m0, n)
    if noise_std > 0:
        A0 += noise_std * rng.standard_normal(A0.shape)

    # 2) Initialize incremental SVD
    inc = IncrementalSVD(A0, rank=k, compute_mask=False)

    metrics = []

    # t = 0 (initial state)
    A_full = A0.copy()
    m0_full = compute_metrics(
        inc,
        A_full=A_full,
        update_type="init",
        update_time=0.0,
        compute_optimal=True,
    )
    m0_full["t"] = 0
    metrics.append(m0_full)

    # 3) Stream new rows
    for t in range(1, n_rows_stream + 1):
        # generate a new latent vector and corresponding row
        u_new = rng.standard_normal((1, true_rank))        # (1, r_true)
        new_row = u_new @ V_true.T                         # (1, n)
        if noise_std > 0:
            new_row += noise_std * rng.standard_normal(new_row.shape)
        new_row = new_row.ravel()

        # update ground truth
        A_full = np.vstack((A_full, new_row[None, :]))

        # time the incremental update
        tic = perf_counter()
        inc.add_row(new_row, incremental=True)
        toc = perf_counter()
        dt = toc - tic
        update_type = "row"

        # refresh if the policy asks for it
        tic_r = perf_counter()
        if rp.apply_refresh_policy(policy, t, inc, policy_state):
            dt += perf_counter() - tic_r
            update_type += "+refresh"

        m_t = compute_metrics(
            inc,
            A_full=A_full,
            update_type=update_type,
            update_time=dt,
            compute_optimal=True,
        )
        m_t["t"] = t
        metrics.append(m_t)

    return metrics


def run_synthetic_mixed_stream(
    m0: int = 20,
    n0: int = 15,
    true_rank: int = 5,
    k: int = 5,
    T: int = 200,
    row_period: int = 10,
    col_period: Optional[int] = None,
    delta_scale: float = 0.1,
    noise_std: float = 0.0,
    seed: int = 0,
    refresh_every: Optional[int] = None,
    refresh_policy: Optional[rp.RefreshPolicy] = None,
) -> List[Dict[str, Any]]:
    """Mixed stream of row appends, column appends and rank-1 updates (Sec. 4.4).

    A_0 is drawn as in :func:`run_synthetic_row_stream`. At step t a row is
    appended if t % row_period == 0, else a column if col_period is given and
    t % col_period == 0, else a rank-1 update N(0, delta_scale^2) is applied
    to a random entry. New rows and columns follow the latent factors of all
    rows and columns so far, plus noise.

    Parameters
    ----------
    m0, n0 : int
        Size of A_0.
    true_rank : int
        Rank of the noise-free part.
    k : int
        Truncation rank.
    T : int
        Number of steps.
    row_period : int
        A row is appended every row_period steps.
    col_period : int, optional
        A column is appended every col_period steps (rows take precedence).
    delta_scale : float
        Standard deviation of the rank-1 increments.
    noise_std : float
        Noise level of A_0 and the new rows and columns.
    seed : int
        Seed of the generator.
    refresh_every : int, optional
        Shortcut for ``refresh_policy=periodic(refresh_every)``.
    refresh_policy : RefreshPolicy, optional
        Refresh policy.

    Returns
    -------
    list of dict
        One dict per step t = 0..T (t = 0 is the initial SVD) with the keys of
        :func:`isvd.metrics.compute_metrics` and ``t``.
    """
    policy = rp.resolve_policy(refresh_policy, refresh_every)
    policy_state = rp.new_state()

    rng = np.random.default_rng(seed)

    # 1) Underlying low-rank model
    U_true = rng.standard_normal((m0, true_rank))
    V_true = rng.standard_normal((n0, true_rank))
    A0 = U_true @ V_true.T
    if noise_std > 0:
        A0 += noise_std * rng.standard_normal(A0.shape)

    inc = IncrementalSVD(A0, rank=k, compute_mask=False)

    A_full = A0.copy()
    metrics = []

    # dynamic latent factors (for consistent rows/cols)
    U_true_dyn = U_true.copy()
    V_true_dyn = V_true.copy()

    # t = 0
    m0_full = compute_metrics(
        inc,
        A_full=A_full,
        update_type="init",
        update_time=0.0,
        compute_optimal=True,
    )
    m0_full["t"] = 0
    metrics.append(m0_full)

    for t in range(1, T + 1):
        m_current, n_current = A_full.shape

        # Choose the update type and apply it
        if t % row_period == 0:
            # Add new row
            u_new = rng.standard_normal((1, true_rank))
            U_true_dyn = np.vstack((U_true_dyn, u_new))             # (m+1, r_true)
            new_row = u_new @ V_true_dyn.T                          # (1, n)
            if noise_std > 0:
                new_row += noise_std * rng.standard_normal(new_row.shape)
            new_row = new_row.ravel()

            A_full = np.vstack((A_full, new_row[None, :]))

            tic = perf_counter()
            inc.add_row(new_row, incremental=True)
            toc = perf_counter()
            dt = toc - tic
            update_type = "row"

        elif col_period is not None and t % col_period == 0:
            # Add new column
            v_new = rng.standard_normal((1, true_rank))
            V_true_dyn = np.vstack((V_true_dyn, v_new))             # (n+1, r_true)
            new_col = (U_true_dyn @ v_new.T).ravel()                # (m,)
            if noise_std > 0:
                new_col += noise_std * rng.standard_normal(new_col.shape)

            A_full = np.hstack((A_full, new_col.reshape(-1, 1)))

            tic = perf_counter()
            inc.add_col(new_col, incremental=True)
            toc = perf_counter()
            dt = toc - tic
            update_type = "col"

        else:
            # Rank-1 update at (u, v)
            u_idx = rng.integers(0, A_full.shape[0])
            v_idx = rng.integers(0, A_full.shape[1])
            delta = delta_scale * rng.standard_normal()

            A_full[u_idx, v_idx] += delta

            tic = perf_counter()
            inc.rank1_update(u_idx, v_idx, delta, update_matrix=True, incremental=True)
            toc = perf_counter()
            dt = toc - tic
            update_type = "rank1"

        # refresh if the policy asks for it
        tic_r = perf_counter()
        if rp.apply_refresh_policy(policy, t, inc, policy_state):
            dt += perf_counter() - tic_r
            update_type += "+refresh"

        m_t = compute_metrics(
            inc,
            A_full=A_full,
            update_type=update_type,
            update_time=dt,
            compute_optimal=True,
        )
        m_t["t"] = t
        metrics.append(m_t)

    return metrics
