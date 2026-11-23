"""Financial factor and covariance experiments of the paper's Sec. 5 (figures sec7_*).

Daily log-returns of 60 ETFs (data/prices_etf.csv, see data/README.md) are
streamed row by row into an :class:`isvd.IncrementalSVD` after an initial
window of T0 days. Every EVAL_EVERY days the incremental factors are compared
with a full SVD of the returns so far: factor subspace angle, low-rank
covariance error (eq. (20)) and portfolio risk errors (eq. (22)).

    experiment_sec7_finance                    time series at k = 5 (Secs. 5.2-5.3)
    experiment_sec7_finance_rank_refresh_grid  rank x refresh grid (Sec. 5.3,
                                               Table tab:finance-bound-refresh)

The ``sec7`` prefix follows an earlier section numbering and is kept because
tex/main.tex includes the figures by these names. All settings are the
module-level constants below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_style as ps

from time import perf_counter
import os
from typing import Collection, List, Dict, Tuple, Optional

from isvd import IncrementalSVD
from isvd import policies as rp
from isvd.metrics import optimal_rank_k_error, principal_angle


# ======================================================================
# Experiment settings reported in the paper (Sec. 5)
# ======================================================================

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prices_etf.csv")

# Common to all Sec. 5 runs: the first INITIAL_WINDOW days (T0) build A_0, metrics
# every EVAL_EVERY streamed days, N_RANDOM_PORTFOLIOS random portfolios per
# evaluation drawn from a generator seeded with SEED, no standardization.
INITIAL_WINDOW = 250
EVAL_EVERY = 5
N_RANDOM_PORTFOLIOS = 10
SEED = 0
FINANCE_COMMON = dict(date_col="date", asset_cols=None, T0=INITIAL_WINDOW, eval_every=EVAL_EVERY,
                      n_random_portfolios=N_RANDOM_PORTFOLIOS, standardize=False)

# Returns are centered with the causal running mean (no look-ahead), as in Sec. 5.1.
# "full" (full-sample mean) is available as an explicit option only.
PAPER_DEMEAN = "causal"

# Time series of Secs. 5.2-5.3 (k = 5): no refresh, refresh every 100 and every 20 days.
TIMESERIES_RANK = 5
TIMESERIES_REFRESH = [None, 100, 20]

# Rank x refresh grid (Sec. 5.3 and Table tab:finance-bound-refresh).
GRID_RANKS = [3, 5, 8, 12, 20]
GRID_REFRESH = [None, 30, 50, 100]
# bound(tau) refresh (eq. (16)) at these ranks. Every tau is above the largest
# truncation floor E_opt(t)/||A_t||_F over the stream (0.451 at k = 5, 0.362 at
# k = 8, 0.289 at k = 12).
GRID_BOUND_TAUS = [0.5, 0.6, 0.7]
GRID_BOUND_RANKS = [5, 8, 12]

PAPER_TIMESERIES_FIGURES = (
    "sec7_finance_angle_factor.pdf",
    "sec7_finance_cov_error.pdf",
    "sec7_finance_risk_eq.pdf",
)
# The paper's runtime figure, sec7_finance_runtime_vs_rank_no_leak.pdf, is made
# by timing_figures.py (one BLAS thread, median over repeated runs).
PAPER_FIGURES = PAPER_TIMESERIES_FIGURES + (
    "sec7_finance_rank_vs_risk_eq.pdf",
)


# ======================================================================
# 1. Data loading: prices -> (de-meaned) returns
# ======================================================================

def missing_data_message(path: str) -> str:
    """Error message for a missing price CSV: how to build it (data/README.md)."""
    return (f"{path} not found. The ETF price data (Yahoo Finance) are not distributed with "
            "the code; build the file with 'python experiments/prepare_etf_data.py' "
            "(needs yfinance), see data/README.md.")


def load_price_data_and_returns(
    path: str = DATA_PATH,
    date_col: str = "date",
    asset_cols: Optional[List[str]] = None,
    log_returns: bool = True,
    standardize: bool = False,
    demean: Optional[str] = PAPER_DEMEAN,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load ETF prices and compute (log) returns.

    Parameters
    ----------
    path : str
        CSV with date and one column per ETF (from prepare_etf_data.py).
    date_col : str
        Name of the date column.
    asset_cols : list of str or None
        If None, use all non-date columns as assets.
    log_returns : bool
        If True, compute log(p_t / p_{t-1}); otherwise simple returns.
    standardize : bool
        If True, standardize returns per asset (zero mean, unit std).
        This corresponds to PCA on the correlation matrix.
    demean : {"causal", "full", None}
        Per-asset centering of the returns before they are fed to the SVD
        (PCA on the covariance matrix, the standard choice for factor /
        risk modelling):
          - "causal" (default): subtract the expanding mean
            mu_t = (1/t) sum_{s<=t} r_s, so row t uses only data up to t
            (no look-ahead).
          - "full": subtract the full-sample mean (uses future data).
          - None: no centering.

    Returns
    -------
    R : (T, N) array
        Returns matrix (rows = time, columns = assets), after
        optional de-meaning / standardization.
    dates : (T,) array
        Corresponding dates.
    assets : list of str
        Asset column names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(missing_data_message(path))
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"date_col='{date_col}' not found in CSV columns")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if asset_cols is None:
        asset_cols = [c for c in df.columns if c != date_col]
    if len(asset_cols) == 0:
        raise ValueError("No asset columns found; please specify asset_cols.")

    prices = df[asset_cols].to_numpy(dtype=float)
    if np.any(prices <= 0):
        raise ValueError("Prices must be positive for log-returns; check data.")

    # Compute returns
    if log_returns:
        R = np.log(prices[1:, :] / prices[:-1, :])
    else:
        R = prices[1:, :] / prices[:-1, :] - 1.0

    dates = df[date_col].to_numpy()[1:]

    # Drop any rows with NaNs (e.g., partial history)
    mask = np.isfinite(R).all(axis=1)
    R = R[mask]
    dates = dates[mask]

    # Per-asset de-meaning
    if demean == "causal":
        cumsum = np.cumsum(R, axis=0)
        denom = np.arange(1, R.shape[0] + 1, dtype=float)[:, None]
        running_mean = cumsum / denom
        R = R - running_mean
    elif demean == "full":
        mean = R.mean(axis=0, keepdims=True)
        R = R - mean
    elif demean is not None:
        raise ValueError("demean must be 'causal', 'full' or None")

    # Optional standardization (per asset). Note: this uses the full-sample
    # standard deviation, i.e. it is not causal.
    if standardize:
        std = R.std(axis=0, keepdims=True) + 1e-8
        R = R / std

    return R, dates, asset_cols


# ======================================================================
# 2. Linear algebra helpers
# ======================================================================

def covariance_from_svd(
    V_k: np.ndarray,
    s_k: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Low-rank covariance V_k diag(scale s_k^2) V_k^T (eqs. (18)-(19)).

    If R = U diag(s) V^T is the SVD of the returns (rows = days, columns =
    assets), the sample covariance is (1/(t-1)) V diag(s^2) V^T.

    Parameters
    ----------
    V_k : ndarray, shape (N, k)
        Right singular vectors (assets x factors).
    s_k : ndarray, shape (k,)
        Singular values.
    scale : float
        Scaling factor, 1/(t-1) for t days.

    Returns
    -------
    ndarray, shape (N, N)
    """
    lam = (s_k ** 2) * scale  # eigenvalues
    return V_k @ (lam[:, None] * V_k.T)


# ======================================================================
# 3. Finance metrics (incremental vs full SVD)
# ======================================================================

def compute_finance_metrics(
    inc: IncrementalSVD,
    rank_k: int,
    n_random_portfolios: int = N_RANDOM_PORTFOLIOS,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Compare the incremental factors with a full SVD of the current returns (Sec. 5.3).

    Parameters
    ----------
    inc : IncrementalSVD
        Engine; ``inc.A`` holds the returns of the days seen so far.
    rank_k : int
        Number of factors compared (at most the ranks of both SVDs).
    n_random_portfolios : int
        Number of random long-short portfolios: standard normal weights
        normalized to unit norm, drawn from `rng`.
    rng : numpy.random.Generator, optional
        Generator for the random portfolios; default ``default_rng(0)``.

    Returns
    -------
    dict of str to float
        frob_inc (E_inc), error_bound (beta_t), frob_opt (E_opt), frob_ratio,
        evr_inc and evr_full_k (explained variance of the k incremental and the
        k optimal factors, relative to ||A||_F^2), angle_factor (largest
        principal angle between the asset-side factor spaces V, eq. (6)),
        cov_rel_error (eq. (20)), risk_rel_err_eq (eq. (22), equal-weight
        portfolio), risk_rel_err_rand_mean and risk_rel_err_rand_max (over
        the random portfolios), opt_time (seconds spent in the full SVD).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    A = inc.A
    m, n = A.shape

    # Incremental reconstruction
    A_hat = inc.reconstruct()
    frob_inc = float(np.linalg.norm(A - A_hat, ord="fro"))
    error_bound = inc.error_bound()

    # Full SVD baseline (timed)
    tic = perf_counter()
    U_full, s_full, Vt_full = np.linalg.svd(A, full_matrices=False)
    toc = perf_counter()
    opt_time = float(toc - tic)

    k_full = len(s_full)
    k_inc = len(inc.S)
    k_use = min(rank_k, k_full, k_inc)
    if k_use == 0:
        return {
            "frob_inc": frob_inc,
            "error_bound": error_bound,
            "frob_opt": np.nan,
            "frob_ratio": np.nan,
            "evr_inc": np.nan,
            "evr_full_k": np.nan,
            "angle_factor": np.nan,
            "cov_rel_error": np.nan,
            "risk_rel_err_eq": np.nan,
            "risk_rel_err_rand_mean": np.nan,
            "risk_rel_err_rand_max": np.nan,
            "opt_time": opt_time,
        }

    # Best rank-k approximation from the full SVD
    s_k = s_full[:k_use]
    Vt_k = Vt_full[:k_use, :]
    frob_opt = optimal_rank_k_error(A, U_full, s_full, Vt_full, k_use)
    frob_ratio = frob_inc / frob_opt if frob_opt > 1e-12 else np.nan

    # Explained variance ratios
    total_var = float(np.sum(s_full ** 2))
    evr_full_k = float(np.sum(s_k ** 2) / total_var) if total_var > 0 else np.nan

    s_inc_all = inc.S
    s_inc_k = s_inc_all[:k_use]
    evr_inc = float(np.sum(s_inc_k ** 2) / total_var) if total_var > 0 else np.nan

    # Factor subspace angle (asset side: V)
    V_full_k = Vt_k.T            # (n, k_use)
    V_inc = inc.Vt.T             # (n, k_inc)
    angle_factor = principal_angle(V_full_k, V_inc)

    # Covariance approximation error
    scale = 1.0 / max(1, m - 1)
    Sigma_full = covariance_from_svd(V_full_k, s_k, scale)
    Sigma_inc = covariance_from_svd(V_inc[:, :k_use], s_inc_k, scale)

    cov_diff = Sigma_inc - Sigma_full
    cov_err = float(np.linalg.norm(cov_diff, ord="fro"))
    cov_norm = float(np.linalg.norm(Sigma_full, ord="fro"))
    cov_rel_error = cov_err / cov_norm if cov_norm > 0 else np.nan

    # Portfolio risk metrics
    # Equal-weight portfolio
    w_eq = np.ones(n, dtype=float)
    w_eq /= np.linalg.norm(w_eq)  # unit norm; scaling cancels in rel error

    var_full_eq = float(w_eq @ Sigma_full @ w_eq)
    var_inc_eq = float(w_eq @ Sigma_inc @ w_eq)
    if var_full_eq > 1e-12:
        rf = np.sqrt(max(var_full_eq, 0.0))
        ri = np.sqrt(max(var_inc_eq, 0.0))
        risk_rel_err_eq = abs(ri - rf) / rf
    else:
        risk_rel_err_eq = np.nan

    # Random portfolios
    risk_rel_err_rand_mean = np.nan
    risk_rel_err_rand_max = np.nan

    if n_random_portfolios > 0:
        W = rng.standard_normal((n_random_portfolios, n))
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        W /= norms

        var_full_rand = np.einsum("ij,jk,ik->i", W, Sigma_full, W)
        var_inc_rand = np.einsum("ij,jk,ik->i", W, Sigma_inc, W)

        mask_pos = var_full_rand > 1e-12
        if np.any(mask_pos):
            rf = np.sqrt(np.maximum(var_full_rand[mask_pos], 0.0))
            ri = np.sqrt(np.maximum(var_inc_rand[mask_pos], 0.0))
            rel_errs = np.abs(ri - rf) / rf
            risk_rel_err_rand_mean = float(np.mean(rel_errs))
            risk_rel_err_rand_max = float(np.max(rel_errs))

    return {
        "frob_inc": frob_inc,
        "error_bound": error_bound,
        "frob_opt": frob_opt,
        "frob_ratio": frob_ratio,
        "evr_inc": evr_inc,
        "evr_full_k": evr_full_k,
        "angle_factor": angle_factor,
        "cov_rel_error": cov_rel_error,
        "risk_rel_err_eq": risk_rel_err_eq,
        "risk_rel_err_rand_mean": risk_rel_err_rand_mean,
        "risk_rel_err_rand_max": risk_rel_err_rand_max,
        "opt_time": opt_time,
    }


# ======================================================================
# 4. Streaming engine for finance
# ======================================================================

def run_finance_incremental_policy(
    R: np.ndarray,
    rank_k: int,
    T0: int,
    refresh_every: Optional[int],
    eval_every: int = EVAL_EVERY,
    n_random_portfolios: int = N_RANDOM_PORTFOLIOS,
    seed: int = SEED,
    refresh_policy: Optional[rp.RefreshPolicy] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Stream the rows of a returns matrix under one refresh policy.

    The first T0 rows build A_0; rows T0, ..., T-1 are appended one by one.
    The timed update includes the refresh, if one is done at that step.

    Parameters
    ----------
    R : (T, N) array
        Returns matrix.
    rank_k : int
        Target truncation rank.
    T0 : int
        Initial window length (first T0 rows build A0).
    refresh_every : int or None
        Shortcut for refresh_policy=periodic(refresh_every): refresh every
        this many streaming updates. None: no periodic refresh.
    eval_every : int
        Evaluate metrics every eval_every streaming updates.
    n_random_portfolios : int
        Number of random portfolios per evaluation.
    seed : int
        Seed of the generator of the random portfolios.
    refresh_policy : callable or None
        should_refresh(t, inc, state) -> bool (see isvd.policies).
        Use instead of refresh_every.

    Returns
    -------
    metrics_df : DataFrame
        Time series of metrics at evaluation points.
    summary : dict
        Runtime summary for this policy and n_refreshes (full SVDs after
        the initial one).
    """
    T, N = R.shape
    if T0 <= 0 or T0 >= T:
        raise ValueError("T0 must satisfy 0 < T0 < T.")

    rng = np.random.default_rng(seed)
    policy = rp.resolve_policy(refresh_policy, refresh_every)
    policy_state = rp.new_state()

    # Initialize on first T0 rows
    A0 = R[:T0, :].copy()
    inc = IncrementalSVD(A0, rank=rank_k, compute_mask=False)

    update_times: List[float] = []
    metrics_list: List[Dict[str, float]] = []

    n_stream_events = T - T0
    t_stream = 0

    # Stream rows T0, ..., T-1
    for idx in range(T0, T):
        row = R[idx, :].copy()

        tic = perf_counter()
        inc.add_row(row, incremental=True)
        t_stream += 1

        rp.apply_refresh_policy(policy, t_stream, inc, policy_state)
        toc = perf_counter()
        update_times.append(toc - tic)

        # Evaluate metrics periodically or at the end
        if (t_stream % eval_every == 0) or (idx == T - 1):
            metrics = compute_finance_metrics(
                inc,
                rank_k=rank_k,
                n_random_portfolios=n_random_portfolios,
                rng=rng,
            )
            metrics["t"] = idx + 1               # total rows seen so far
            metrics["n_rows"] = idx + 1
            metrics_list.append(metrics)

    if update_times:
        mean_update_time = float(np.mean(update_times))
        total_update_time = float(np.sum(update_times))
    else:
        mean_update_time = 0.0
        total_update_time = 0.0

    metrics_df = pd.DataFrame(metrics_list)
    summary = {
        "mean_update_time": mean_update_time,
        "total_update_time": total_update_time,
        "n_stream_events": float(n_stream_events),
        "n_refreshes": float(policy_state["n_refreshes"]),  # full SVDs after the initial one
        "last_refresh": float(policy_state["last_refresh"]),  # stream step of the last one (0: none)
    }
    return metrics_df, summary


# ======================================================================
# 5. Plotting helpers
# ======================================================================

def plot_finance_runtime_bars(
    summary_df: pd.DataFrame,
    filename: Optional[str] = None,
) -> None:
    """
    Bar chart of the time per update of each policy (not a paper figure).

    Parameters
    ----------
    summary_df : DataFrame
        Per-policy summary from :func:`experiment_sec7_finance`.
    filename : str or None
        Output path (see :func:`plot_style.save`).
    """
    # Incremental per-update time
    labels = list(summary_df["policy"])
    values = (
        summary_df["total_update_time"] / summary_df["n_stream_events"]
    ).tolist()

    ps.figure(None)
    x = np.arange(len(labels))

    bars = plt.bar(x, values, color="steelblue")

    # Zoom y-axis so small differences are visible
    vmin = min(values)
    vmax = max(values)
    if vmax > vmin:
        plt.ylim(vmin * 0.80, vmax * 1.20)
    else:
        # handle degenerate case where all values are equal
        margin = vmax * 0.20 if vmax != 0 else 1e-6
        plt.ylim(vmax - margin, vmax + margin)

    # Add value labels
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.5f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Time per update (seconds)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    ps.save(filename)



# ======================================================================
# 6. Time series at fixed rank (Secs. 5.2-5.3)
# ======================================================================

def experiment_sec7_finance(
    path: str = DATA_PATH,
    date_col: str = "date",
    asset_cols: Optional[List[str]] = None,
    rank_k: int = TIMESERIES_RANK,
    T0: int = INITIAL_WINDOW,
    refresh_policies: Optional[List[Optional[int]]] = None,
    eval_every: int = EVAL_EVERY,
    n_random_portfolios: int = N_RANDOM_PORTFOLIOS,
    standardize: bool = False,
    demean: Optional[str] = PAPER_DEMEAN,
    outdir: str = ".",
    only: Optional[Collection[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time series of the finance metrics at rank `rank_k` for several refresh periods.

    Paper figures (TIMESERIES_RANK, TIMESERIES_REFRESH): sec7_finance_angle_factor,
    sec7_finance_cov_error and sec7_finance_risk_eq. Also writes the error
    ratio and a runtime bar chart.

    Parameters
    ----------
    path : str
        CSV with date and ETF prices.
    date_col : str
        Name of the date column.
    asset_cols : list of str or None
        If None, use all non-date columns as assets.
    rank_k : int
        Truncation rank for factor model.
    T0 : int
        Initial window length in days.
    refresh_policies : list of {None or int}
        None (no refresh) or a refresh period R ("refresh_R"); default
        TIMESERIES_REFRESH.
    eval_every : int
        Evaluate metrics every eval_every streaming updates.
    n_random_portfolios : int
        Number of random portfolios per evaluation.
    standardize : bool
        If True, standardize returns per asset.
    demean : {"causal", "full", None}
        De-meaning mode, see load_price_data_and_returns.
    outdir : str
        Folder for the figures.
    only : collection of str or None
        If given, save only the figures with these file names.

    Returns
    -------
    metrics_all : DataFrame
        Metrics of :func:`compute_finance_metrics` at every evaluation, with
        columns t, n_rows (days seen) and policy.
    summary_df : DataFrame
        Per-policy summary of :func:`run_finance_incremental_policy`, with
        columns policy and k.
    """
    if refresh_policies is None:
        refresh_policies = TIMESERIES_REFRESH

    R, dates, assets = load_price_data_and_returns(
        path=path,
        date_col=date_col,
        asset_cols=asset_cols,
        log_returns=True,
        standardize=standardize,
        demean=demean,
    )
    T, N = R.shape
    print(f"[Finance] Loaded returns matrix of shape T={T}, N={N}")

    metrics_list: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    for refresh_every in refresh_policies:
        if refresh_every is None:
            policy_label = "no_refresh"
        else:
            policy_label = f"refresh_{refresh_every}"

        print(f"\n[Finance] Running policy: {policy_label}, k={rank_k}")
        metrics_df, summary = run_finance_incremental_policy(
            R=R,
            rank_k=rank_k,
            T0=T0,
            refresh_every=refresh_every,
            eval_every=eval_every,
            n_random_portfolios=n_random_portfolios,
            seed=SEED,
        )
        metrics_df["policy"] = policy_label
        metrics_list.append(metrics_df)

        summary["policy"] = policy_label
        summary["k"] = float(rank_k)
        summaries.append(summary)

    metrics_all = pd.DataFrame(pd.concat(metrics_list, ignore_index=True))
    summary_df = pd.DataFrame(summaries)

    # Plots
    df_list = []
    labels = []
    for refresh_every in refresh_policies:
        if refresh_every is None:
            lab = "no_refresh"
        else:
            lab = f"refresh_{refresh_every}"
        g = metrics_all[metrics_all["policy"] == lab]
        df_list.append(g)
        labels.append(lab)

    ps.line_plot(
        df_list,
        x_col="n_rows",
        y_col="frob_ratio",
        labels=labels,
        xlabel="Number of days",
        ylabel="Frobenius error ratio",
        filename=ps.out_path("sec7_finance_error_ratio.pdf", outdir, only),
    )

    ps.line_plot(
        df_list,
        x_col="n_rows",
        y_col="angle_factor",
        labels=labels,
        xlabel="Number of days",
        ylabel="Factor subspace angle (rad)",
        filename=ps.out_path("sec7_finance_angle_factor.pdf", outdir, only),
    )

    ps.line_plot(
        df_list,
        x_col="n_rows",
        y_col="cov_rel_error",
        labels=labels,
        xlabel="Number of days",
        ylabel="Relative covariance error",
        filename=ps.out_path("sec7_finance_cov_error.pdf", outdir, only),
    )

    ps.line_plot(
        df_list,
        x_col="n_rows",
        y_col="risk_rel_err_eq",
        labels=labels,
        xlabel="Number of days",
        ylabel="Relative risk error",
        filename=ps.out_path("sec7_finance_risk_eq.pdf", outdir, only),
    )

    plot_finance_runtime_bars(
        summary_df,
        filename=ps.out_path("sec7_finance_runtime.pdf", outdir, only),
    )

    return metrics_all, summary_df


# ======================================================================
# 7. Rank x refresh grid (Sec. 5.3, Table tab:finance-bound-refresh)
# ======================================================================

def experiment_sec7_finance_rank_refresh_grid(
    path: str = DATA_PATH,
    date_col: str = "date",
    asset_cols: Optional[List[str]] = None,
    ranks: List[int] = GRID_RANKS,
    refresh_policies: Optional[List[Optional[int]]] = None,
    T0: int = INITIAL_WINDOW,
    eval_every: int = EVAL_EVERY,
    n_random_portfolios: int = N_RANDOM_PORTFOLIOS,
    standardize: bool = False,
    demean: Optional[str] = PAPER_DEMEAN,
    bound_taus: Collection[float] = (),
    bound_ranks: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """
    Final-day accuracy and runtime for every (rank k, refresh policy) pair.

    Paper: sec7_finance_rank_vs_risk_eq (via :func:`plot_finance_error_vs_rank`)
    and Table tab:finance-bound-refresh (via :func:`print_finance_grid_table`),
    with GRID_RANKS, GRID_REFRESH, GRID_BOUND_TAUS and GRID_BOUND_RANKS.

    Parameters
    ----------
    path, date_col, asset_cols, standardize, demean
        Data and pre-processing, see :func:`load_price_data_and_returns`.
    ranks : list of int
        Truncation ranks k.
    refresh_policies : list of {None or int}
        None (no refresh) or a refresh period R ("refresh_R"); default
        GRID_REFRESH.
    T0, eval_every, n_random_portfolios
        See :func:`run_finance_incremental_policy`.
    bound_taus : collection of float
        Also run the bound(tau) policy (refresh when beta_t / ||A_t||_F > tau,
        eq. (16)) for each tau; policy label "bound_<tau>". Default: none.
    bound_ranks : collection of int or None
        Ranks at which the bound(tau) runs are done (None: all of `ranks`).

    Returns
    -------
    grid_df : DataFrame
        One row per (k, policy) pair, with columns:
            ['k', 'refresh_every', 'tau', 'policy',
             'angle_final', 'cov_rel_error_final',
             'risk_eq_rel_final', 'risk_rand_rel_final',
             'mean_update_time', 'total_update_time',
             'n_stream_events', 'n_refreshes', 'steps_since_refresh', 'n_rows']
        steps_since_refresh: streaming steps from the last full SVD (the
        initial one at T0 if there was no refresh) to the final day.
    """
    if refresh_policies is None:
        refresh_policies = GRID_REFRESH

    # Load and pre-process returns ONCE
    R, dates, assets = load_price_data_and_returns(
        path=path,
        date_col=date_col,
        asset_cols=asset_cols,
        log_returns=True,
        standardize=standardize,
        demean=demean,
    )
    T, N = R.shape
    print(f"[finance_grid] Using returns matrix R with shape {R.shape}")

    rows = []

    for k in ranks:
        # (label, refresh_every, refresh_policy, tau) for each run at this k
        runs = []
        for refresh_every in refresh_policies:
            if refresh_every is None:
                runs.append(("no_refresh", None, None, None))
            else:
                runs.append((f"refresh_{refresh_every}", refresh_every, None, None))
        if bound_ranks is None or k in bound_ranks:
            for tau in bound_taus:
                runs.append((f"bound_{tau}", None, rp.bound(tau), tau))

        for policy_label, refresh_every, refresh_policy, tau in runs:
            print(f"\n[finance_grid] k={k}, policy={policy_label}")

            metrics_df, summary = run_finance_incremental_policy(
                R=R,
                rank_k=k,
                T0=T0,
                refresh_every=refresh_every,
                eval_every=eval_every,
                n_random_portfolios=n_random_portfolios,
                seed=SEED,
                refresh_policy=refresh_policy,
            )

            last = metrics_df.iloc[-1]

            angle_final = float(last.get("angle_factor", np.nan))
            cov_final = float(last.get("cov_rel_error", np.nan))
            risk_eq_final = float(last.get("risk_rel_err_eq", np.nan))
            risk_rand_mean_final = float(last.get("risk_rel_err_rand_mean", np.nan))

            row = {
                "k": float(k),
                "refresh_every": float(refresh_every) if refresh_every is not None else np.nan,
                "tau": float(tau) if tau is not None else np.nan,
                "policy": policy_label,
                "angle_final": angle_final,
                "cov_rel_error_final": cov_final,
                "risk_eq_rel_final": risk_eq_final,
                "risk_rand_rel_final": risk_rand_mean_final,
                "mean_update_time": float(summary["mean_update_time"]),
                "total_update_time": float(summary["total_update_time"]),
                "n_stream_events": float(summary["n_stream_events"]),
                "n_refreshes": float(summary["n_refreshes"]),
                "steps_since_refresh": float(summary["n_stream_events"] - summary["last_refresh"]),
                "n_rows": float(last.get("n_rows", T)),
            }
            rows.append(row)

    grid_df = pd.DataFrame(rows)
    return grid_df


def plot_finance_error_vs_rank(
    grid_df: pd.DataFrame,
    metric: str = "risk_eq_rel_final",
    filename: Optional[str] = None,
) -> None:
    """
    Plot a final-day metric of the grid against the rank k, one curve per policy.

    Periodic and no-refresh curves are solid, bound(tau) curves dashed.

    Parameters
    ----------
    grid_df : DataFrame
        Output of :func:`experiment_sec7_finance_rank_refresh_grid`.
    metric : {"risk_eq_rel_final", "risk_rand_rel_final", "cov_rel_error_final", "angle_final"}
        Column to plot.
    filename : str or None
        Output path (see :func:`plot_style.save`).
    """
    ps.figure(filename)
    # Periodic / no-refresh curves first, solid; then bound(tau) curves dashed.
    # Each curve has its own color and marker (plot_style, in plotting order).
    is_bound = grid_df["policy"].str.startswith("bound_")
    i = 0
    for linestyle, sub in (("-", grid_df[~is_bound]), ("--", grid_df[is_bound])):
        for policy, g in sub.groupby("policy"):
            g_sorted = g.sort_values("k")
            if metric not in g_sorted.columns:
                continue
            x = g_sorted["k"].to_numpy(dtype=float)
            y = g_sorted[metric].to_numpy(dtype=float)
            plt.plot(x, y, color=ps.COLORS[i], linestyle=linestyle,
                     marker=ps.MARKERS[i], label=policy)
            i += 1

    y_label_map = {
        "risk_eq_rel_final": "Final relative risk error",
        "risk_rand_rel_final": "Final mean relative risk error (random w)",
        "cov_rel_error_final": "Final relative covariance error",
        "angle_final": "Final angle to full-SVD subspace (rad)",
    }
    plt.xlabel("Truncation rank k")
    plt.ylabel(y_label_map.get(metric, metric))
    plt.legend()
    plt.xticks(sorted(grid_df["k"].unique()))  # ticks at the swept ranks
    ps.save(filename)


def plot_finance_runtime_vs_rank(
    grid_df: pd.DataFrame,
    filename: Optional[str] = None,
) -> None:
    """
    Plot the mean update time against the rank k, one curve per refresh policy.

    Parameters
    ----------
    grid_df : DataFrame
        Grid rows with columns k, policy and mean_update_time.
    filename : str or None
        Output path (see :func:`plot_style.save`).
    """
    ps.figure(filename)
    # Line style and color from the plot_style cycle, one marker per curve.
    for i, (policy, g) in enumerate(grid_df.groupby("policy")):
        g_sorted = g.sort_values("k")
        x = g_sorted["k"].to_numpy(dtype=float)
        y = g_sorted["mean_update_time"].to_numpy(dtype=float)
        plt.plot(x, y, marker=ps.MARKERS[i], label=policy)

    plt.xlabel("Truncation rank k")
    plt.ylabel("Mean update time (s)")
    plt.legend()
    ps.save(filename)


# ======================================================================
# 8. Paper figures, Table tab:finance-bound-refresh and command line
# ======================================================================

def make_paper_figures(outdir: str, path: str = DATA_PATH) -> Dict[str, pd.DataFrame]:
    """Regenerate the four non-timing Sec. 5 figures of the paper (PAPER_FIGURES).

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    path : str
        Price CSV.

    Returns
    -------
    dict of str to DataFrame
        "sec7_finance__metrics_all" and "sec7_finance__summary" (time series,
        see :func:`experiment_sec7_finance`) and "sec7_finance__grid" (the
        rank x refresh grid with the periodic and bound(tau) rows).
    """
    metrics_all, summary_df = experiment_sec7_finance(
        path=path, rank_k=TIMESERIES_RANK, refresh_policies=TIMESERIES_REFRESH,
        demean=PAPER_DEMEAN, outdir=outdir,
        only=PAPER_TIMESERIES_FIGURES, **FINANCE_COMMON,
    )
    grid_df = experiment_sec7_finance_rank_refresh_grid(
        path=path, ranks=GRID_RANKS, refresh_policies=GRID_REFRESH,
        demean=PAPER_DEMEAN, bound_taus=GRID_BOUND_TAUS,
        bound_ranks=GRID_BOUND_RANKS, **FINANCE_COMMON,
    )
    plot_finance_error_vs_rank(
        grid_df, metric="risk_eq_rel_final",
        filename=os.path.join(outdir, "sec7_finance_rank_vs_risk_eq.pdf"),
    )
    return {"sec7_finance__metrics_all": metrics_all,
            "sec7_finance__summary": summary_df,
            "sec7_finance__grid": grid_df}


GRID_TABLE_COLUMNS = ["k", "policy", "n_refreshes", "steps_since_refresh", "angle_final", "cov_rel_error_final",
                      "risk_eq_rel_final", "risk_rand_rel_final"]


def print_finance_grid_table(grid_df: pd.DataFrame, ranks: Collection[int] = GRID_BOUND_RANKS) -> None:
    """Print the final-day metrics and full-SVD counts of Table tab:finance-bound-refresh.

    Parameters
    ----------
    grid_df : DataFrame
        Output of :func:`experiment_sec7_finance_rank_refresh_grid`.
    ranks : collection of int
        Ranks shown.
    """
    sel = grid_df[grid_df["k"].isin([float(k) for k in ranks])][GRID_TABLE_COLUMNS]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print(sel.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Section 5 finance experiments.")
    parser.add_argument("--outdir", default=".", help="folder for the figures")
    parser.add_argument("--demean", default="causal", choices=["causal", "full", "none"])
    parser.add_argument("--data", default=DATA_PATH, help="price CSV")
    args = parser.parse_args()
    demean = None if args.demean == "none" else args.demean
    if not os.path.exists(args.data):
        raise SystemExit(missing_data_message(args.data))

    # Time series at fixed rank, different refresh periods
    metrics_all, summary_df = experiment_sec7_finance(
        path=args.data, rank_k=TIMESERIES_RANK, refresh_policies=TIMESERIES_REFRESH,
        demean=demean, outdir=args.outdir, **FINANCE_COMMON,
    )
    print("\n=== Finance incremental policy summary ===")
    print(summary_df)

    # Rank x refresh grid
    grid_df = experiment_sec7_finance_rank_refresh_grid(
        path=args.data, ranks=GRID_RANKS, refresh_policies=GRID_REFRESH,
        demean=demean, bound_taus=GRID_BOUND_TAUS, bound_ranks=GRID_BOUND_RANKS,
        **FINANCE_COMMON,
    )
    print("\n=== Finance rank × refresh grid ===")
    print(grid_df)
    print("\n=== Final-day metrics and full SVDs, k in GRID_BOUND_RANKS ===")
    print_finance_grid_table(grid_df)

    plot_finance_error_vs_rank(
        grid_df, metric="risk_eq_rel_final",
        filename=os.path.join(args.outdir, "sec7_finance_rank_vs_risk_eq.pdf"),
    )
    plot_finance_runtime_vs_rank(
        grid_df[~grid_df["policy"].str.startswith("bound_")],
        filename=os.path.join(args.outdir, "sec7_finance_runtime_vs_rank.pdf"),
    )
