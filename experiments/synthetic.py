"""Synthetic experiments of the paper's Sec. 4 (figures exp41_* to exp45_*).

Each ``experiment_*`` function saves its figures into `outdir` (only the file
names listed in `only`, if given) and returns its raw metrics: a dict
{config label: DataFrame} for the streaming experiments and a DataFrame of
mean update times for the scaling experiments. The figure file prefixes
(exp41 ... exp45) follow an earlier section numbering and are kept because
tex/main.tex includes the figures by these names:

    experiment_41_rank1_refresh_comparison   Sec. 4.1
    experiment_41_rank_effect                Sec. 4.2
    experiment_42_row_stream, _column_stream Sec. 4.3
    experiment_43_mixed_stream               Sec. 4.4
    experiment_45_error_bound_refresh        Sec. 4.5, Table tab:bound-refresh
    experiment_44_scaling_size, _rank        Sec. 4.6

All settings are the module-level constants below. experiments/reproduce.py
runs the experiments behind the paper figures and tables.
"""

import numpy as np
import pandas as pd
import streams as ss
import matplotlib.pyplot as plt
import plot_style as ps
from isvd import IncrementalSVD
from isvd import policies as rp
from time import perf_counter
from typing import Collection, Dict, Optional


# ======================================================================
# Experiment settings reported in the paper (Sec. 4)
# ======================================================================

# Sec. 4.1: rank-1 entry updates of a 50 x 40 rank-5 matrix, k = 5.
RANK1_STREAM = dict(m=50, n=40, true_rank=5, k=5, T=10000, delta_scale=0.05, seed=0)
RANK1_REFRESH = (("no_refresh", None), ("refresh_1000", 1000), ("refresh_5000", 5000))

# Sec. 4.2: effect of the truncation rank k on the same kind of stream.
RANK_EFFECT_STREAM = dict(m=50, n=40, true_rank=5, T=6000, delta_scale=0.05, seed=1,
                          refresh_every=2000)
RANK_EFFECT_KS = (5, 8, 12)
RANK_EFFECT_PLOT_FROM = 100  # plots and mean update time use t >= 100

# Sec. 4.3: row and column appends to a 50 x 40 rank-6 matrix plus noise, k = 4.
ROW_STREAM = dict(m0=50, n=40, true_rank=6, k=4, n_rows_stream=200, noise_std=0.05, seed=2)
ROW_REFRESH = (("row_no_refresh", None), ("row_refresh_50", 50))
COL_STREAM = dict(m=50, n0=40, true_rank=6, k=4, n_cols_stream=200, noise_std=0.05, seed=3)
COL_REFRESH = (("col_no_refresh", None), ("col_refresh_50", 50))

# Sec. 4.4: mixed stream (a row every 50 steps, a column every 100, rank-1 updates otherwise).
MIXED_STREAM = dict(m0=50, n0=40, true_rank=5, k=8, T=2000, row_period=50, col_period=100,
                    delta_scale=0.05, noise_std=0.05, seed=4)
MIXED_REFRESH = (("mixed_no_refresh", None), ("mixed_refresh_200", 200))

# Sec. 4.5: the Sec. 4.1 stream with these seeds, no refresh, periodic(BOUND_EXP_PERIOD)
# and bound(tau) for each tau (Table tab:bound-refresh).
BOUND_EXP_SEEDS = (0, 1, 2, 3, 4)
BOUND_EXP_TAUS = (0.05, 0.1, 0.2, 0.4)
BOUND_EXP_PERIOD = 1000

# Sec. 4.6: update time against matrix size and against k (rank-1 updates, true rank 5).
SCALING_TRUE_RANK = 5
SCALING_T = 300
SCALING_DELTA_SCALE = 0.05
SCALING_SIZES = ((100, 80), (300, 250), (800, 600))
SCALING_SIZE_RANK = 10
SCALING_SIZE_SEED = 0
SCALING_RANK_SHAPE = (500, 400)
SCALING_RANK_KS = (5, 10, 20, 40)
SCALING_RANK_SEED = 1


# ======================================================================
# Secs. 4.1-4.2: rank-1 entry updates
# ======================================================================

def experiment_41_rank1_refresh_comparison(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.1: rank-1 stream without refresh and with periodic refresh (RANK1_*).

    Paper figures: exp41_rank1_error_ratio (frob_ratio) and exp41_rank1_angle_opt
    (angle_to_opt, the angle to the current optimal subspace). Also writes the
    errors against the optimal error and the angle to the initial subspace.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    dict of str to DataFrame
        Per-step metrics of each run, keyed by the labels of RANK1_REFRESH.
    """
    configs = RANK1_REFRESH

    dfs: list[pd.DataFrame] = []
    labels: list[str] = []

    for label, refresh_every in configs:
        metrics = ss.run_synthetic_rank1_stream(**RANK1_STREAM, refresh_every=refresh_every)
        df = pd.DataFrame(metrics)
        dfs.append(df)
        labels.append(label)

    # incremental vs optimal error
    ps.figure(None)
    for df, lab in zip(dfs, labels):
        inc_valid = df[["t", "frob_error"]].dropna()
        opt_valid = df[["t", "frob_opt"]].dropna()
        plt.plot(inc_valid["t"], inc_valid["frob_error"], label=f"{lab} (inc)")
        plt.plot(opt_valid["t"], opt_valid["frob_opt"], "--", alpha=0.7,
                 label=f"{lab} (opt)")
    plt.xlabel("Time step t")
    plt.ylabel("Frobenius error")
    plt.legend()
    ps.save(ps.out_path("exp41_rank1_error_vs_optimal.pdf", outdir, only))

    # error ratio
    ps.line_plot(
        dfs,
        "t",
        "frob_ratio",
        labels,
        xlabel="Time step t",
        ylabel="Error ratio",
        filename=ps.out_path("exp41_rank1_error_ratio.pdf", outdir, only),
    )

    # drift vs initial
    ps.line_plot(
        dfs,
        "t",
        "angle",
        labels,
        xlabel="Time step t",
        ylabel="Angle to initial (rad)",
        filename=ps.out_path("exp41_rank1_angle_initial.pdf", outdir, only),
    )

    # tracking error vs optimal
    ps.line_plot(
        dfs,
        "t",
        "angle_to_opt",
        labels,
        xlabel="Time step t",
        ylabel="Angle to optimal (rad)",
        filename=ps.out_path("exp41_rank1_angle_opt.pdf", outdir, only),
    )

    # timing
    for df, lab in zip(dfs, labels):
        mean_inc_time = df["update_time"].mean()
        mean_opt_time = df["opt_time"].dropna().mean()
        print(f"[{lab}] mean incremental update_time = {mean_inc_time:.6e} s")
        print(f"[{lab}] mean optimal SVD opt_time    = {mean_opt_time:.6e} s")
        print("-" * 60)

    return dict(zip(labels, dfs))


def experiment_41_rank_effect(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.2: effect of the truncation rank k (RANK_EFFECT_*).

    Paper figure: exp41_rank_effect_error_ratio; the update-time figure of the
    paper is made by timing_figures.py. Plots skip t < RANK_EFFECT_PLOT_FROM.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    dict of str to DataFrame
        Per-step metrics of each run (all steps), keyed "k=<k>".
    """
    ks = list(RANK_EFFECT_KS)
    dfs: list[pd.DataFrame] = []
    labels: list[str] = []

    for k in ks:
        metrics = ss.run_synthetic_rank1_stream(**RANK_EFFECT_STREAM, k=k)
        df = pd.DataFrame(metrics)
        dfs.append(df)
        labels.append(f"k={k}")

    # The plots (and the mean update time) skip the first 100 steps: there
    # E_opt(t) is still close to 0 (k >= true rank 5 and E_opt(0) = 0), so
    # frob_ratio has a start-up spike (up to ~36 for k = 12). The returned
    # metrics keep every step.
    plot_dfs = [df[df["t"] >= RANK_EFFECT_PLOT_FROM] for df in dfs]

    ps.line_plot(
        plot_dfs,
        "t",
        "frob_ratio",
        labels,
        xlabel="Time step t",
        ylabel="Error ratio (inc / opt)",
        filename=ps.out_path("exp41_rank_effect_error_ratio.pdf", outdir, only),
    )

    ps.line_plot(
        plot_dfs,
        "t",
        "angle",
        labels,
        xlabel="Time step t",
        ylabel="Angle to initial (rad)",
        filename=ps.out_path("exp41_rank_effect_angle_initial.pdf", outdir, only),
    )

    ps.line_plot(
        plot_dfs,
        "t",
        "angle_to_opt",
        labels,
        xlabel="Time step t",
        ylabel="Angle to optimal (rad)",
        filename=ps.out_path("exp41_rank_effect_angle_opt.pdf", outdir, only),
    )

    mean_times = [df["update_time"].mean() for df in plot_dfs]
    path = ps.out_path("exp41_rank_effect_update_time.pdf", outdir, only)
    ps.figure(path)
    plt.plot(ks, mean_times, marker="o")
    plt.xlabel("Truncation rank k")
    plt.ylabel("Mean update time (s)")
    ps.save(path)

    return dict(zip(labels, dfs))


# ======================================================================
# Sec. 4.3: structural growth, rows and columns
# ======================================================================

def experiment_42_row_stream(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.3: row appends without and with periodic refresh (ROW_*).

    Paper figure: exp42_row_error_ratio. Also writes both angle figures.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    dict of str to DataFrame
        Per-step metrics of each run, keyed by the labels of ROW_REFRESH.
    """
    configs = ROW_REFRESH
    dfs: list[pd.DataFrame] = []
    labels: list[str] = []

    for label, refresh_every in configs:
        metrics = ss.run_synthetic_row_stream(**ROW_STREAM, refresh_every=refresh_every)
        df = pd.DataFrame(metrics)
        dfs.append(df)
        labels.append(label)

    ps.line_plot(
        dfs,
        "t",
        "frob_ratio",
        labels,
        xlabel="Number of streamed rows (t)",
        ylabel="Error ratio (inc / opt)",
        filename=ps.out_path("exp42_row_error_ratio.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle",
        labels,
        xlabel="Number of streamed rows (t)",
        ylabel="Angle to initial (rad)",
        filename=ps.out_path("exp42_row_angle_initial.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle_to_opt",
        labels,
        xlabel="Number of streamed rows (t)",
        ylabel="Angle to optimal (rad)",
        filename=ps.out_path("exp42_row_angle_opt.pdf", outdir, only),
    )

    for df, lab in zip(dfs, labels):
        print(f"[{lab}] mean update_time = {df['update_time'].mean():.6e} s")

    return dict(zip(labels, dfs))


def experiment_42_column_stream(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.3: column appends without and with periodic refresh (COL_*).

    Paper figure: exp42_col_error_ratio. Also writes both angle figures.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    dict of str to DataFrame
        Per-step metrics of each run, keyed by the labels of COL_REFRESH.
    """
    configs = COL_REFRESH
    dfs: list[pd.DataFrame] = []
    labels: list[str] = []

    for label, refresh_every in configs:
        metrics = ss.run_synthetic_column_stream(**COL_STREAM, refresh_every=refresh_every)
        df = pd.DataFrame(metrics)
        dfs.append(df)
        labels.append(label)

    ps.line_plot(
        dfs,
        "t",
        "frob_ratio",
        labels,
        xlabel="Number of streamed columns (t)",
        ylabel="Error ratio (inc / opt)",
        filename=ps.out_path("exp42_col_error_ratio.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle",
        labels,
        xlabel="Number of streamed columns (t)",
        ylabel="Angle to initial (rad)",
        filename=ps.out_path("exp42_col_angle_initial.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle_to_opt",
        labels,
        xlabel="Number of streamed columns (t)",
        ylabel="Angle to optimal (rad)",
        filename=ps.out_path("exp42_col_angle_opt.pdf", outdir, only),
    )

    for df, lab in zip(dfs, labels):
        print(f"[{lab}] mean update_time = {df['update_time'].mean():.6e} s")

    return dict(zip(labels, dfs))


# ======================================================================
# Sec. 4.4: mixed stream
# ======================================================================

def experiment_43_mixed_stream(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.4: rows, columns and rank-1 updates, without and with refresh (MIXED_*).

    Paper figure: exp43_mixed_error_ratio. Also writes both angle figures.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    dict of str to DataFrame
        Per-step metrics of each run, keyed by the labels of MIXED_REFRESH.
    """
    configs = MIXED_REFRESH
    dfs: list[pd.DataFrame] = []
    labels: list[str] = []

    for label, refresh_every in configs:
        metrics = ss.run_synthetic_mixed_stream(**MIXED_STREAM, refresh_every=refresh_every)
        df = pd.DataFrame(metrics)
        dfs.append(df)
        labels.append(label)

    ps.line_plot(
        dfs,
        "t",
        "frob_ratio",
        labels,
        xlabel="Time step t",
        ylabel="Error ratio (inc / opt)",
        filename=ps.out_path("exp43_mixed_error_ratio.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle",
        labels,
        xlabel="Time step t",
        ylabel="Angle to initial (rad)",
        filename=ps.out_path("exp43_mixed_angle_initial.pdf", outdir, only),
    )

    ps.line_plot(
        dfs,
        "t",
        "angle_to_opt",
        labels,
        xlabel="Time step t",
        ylabel="Angle to optimal (rad)",
        filename=ps.out_path("exp43_mixed_angle_opt.pdf", outdir, only),
    )

    for df, lab in zip(dfs, labels):
        print(f"\n[{lab}] mean update_time by update_type:")
        print(df.groupby("update_type")["update_time"].mean())

    return dict(zip(labels, dfs))


# ======================================================================
# Sec. 4.5: a posteriori error bound and bound-triggered refresh
# ======================================================================

def _rank1_stream_df(seed: int, refresh_policy=None) -> pd.DataFrame:
    """The Sec. 4.1 rank-1 stream (RANK1_STREAM) with the given seed and refresh policy."""
    return pd.DataFrame(ss.run_synthetic_rank1_stream(
        **{**RANK1_STREAM, "seed": seed}, refresh_policy=refresh_policy))


def _bound_run_summary(df: pd.DataFrame) -> Dict[str, float]:
    """Summary of one run over the updates t = 1..T (t = 0 is the initial SVD).

    n_full_svd counts the refreshes (the initial SVD is not counted).
    first_refresh is the step of the first refresh (NaN if none) and
    frob_ratio_max_after_first the max of frob_ratio over the later steps, i.e.
    without the first segment, which starts from E_opt(0) = 0. t_frob_ratio_max
    is the step where the overall max occurs. The *_second_half entries are
    over t > T/2. bound_excess_max is max(E_inc - beta_t) over t = 0..T
    (> 0 only at roundoff level).
    """
    d = df[df["t"] >= 1]
    second = d["t"] > d["t"].max() / 2
    refreshed = d["update_type"].str.endswith("+refresh")
    first = float(d.loc[refreshed, "t"].min()) if refreshed.any() else np.nan
    after_first = d[d["t"] > first] if refreshed.any() else d.iloc[0:0]
    ratio = d["error_bound"] / d["frob_error"]
    return {
        "n_full_svd": int(refreshed.sum()),
        "n_full_svd_second_half": int((refreshed & second).sum()),
        "first_refresh": first,
        "frob_ratio_mean": float(d["frob_ratio"].mean()),
        "frob_ratio_max": float(d["frob_ratio"].max()),
        "t_frob_ratio_max": int(d.loc[d["frob_ratio"].idxmax(), "t"]),
        "frob_ratio_max_after_first": float(after_first["frob_ratio"].max()),
        "frob_ratio_max_second_half": float(d.loc[second, "frob_ratio"].max()),
        "angle_to_opt_mean": float(d["angle_to_opt"].mean()),
        "angle_to_opt_max": float(d["angle_to_opt"].max()),
        "rel_err_max": float((d["frob_error"] / d["frob_norm"]).max()),
        "rel_bound_max": float((d["error_bound"] / d["frob_norm"]).max()),
        "bound_over_err_median": float(ratio.median()),
        "bound_over_err_max": float(ratio.max()),
        "bound_excess_max": float((df["frob_error"] - df["error_bound"]).max()),
    }


def experiment_45_error_bound_refresh(
    outdir: str = ".",
    only: Optional[Collection[str]] = None,
    seeds: Collection[int] = BOUND_EXP_SEEDS,
    taus: Collection[float] = BOUND_EXP_TAUS,
    refresh_every: int = BOUND_EXP_PERIOD,
) -> Dict[str, pd.DataFrame]:
    """Sec. 4.5: the a posteriori bound beta_t (Proposition 1) on the Sec. 4.1 stream.

    Figure exp45_rank1_error_bound: E_inc(t), beta_t and E_opt(t) without
    refresh, for the first seed (the stream of Sec. 4.1). Table
    tab:bound-refresh (printed): no refresh, periodic(refresh_every) and
    bound(tau) (eq. (16)) for each tau, each run on every seed; per-seed
    summaries (:func:`_bound_run_summary`) and their mean over the seeds.

    Parameters
    ----------
    outdir : str
        Folder for the figure.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.
    seeds : collection of int
        Seeds of the rank-1 streams.
    taus : collection of float
        Tolerances of the bound(tau) runs.
    refresh_every : int
        Period of the periodic run.

    Returns
    -------
    dict of str to DataFrame
        "no_refresh": per-step metrics of the figure run; "per_seed": one
        summary row per (policy, seed); "summary": mean over the seeds per
        policy, with the min and max of n_full_svd.
    """
    policies = [("no refresh", lambda: None),
                (f"periodic({refresh_every})", lambda: rp.periodic(refresh_every))]
    policies += [(f"bound({tau:g})", lambda tau=tau: rp.bound(tau)) for tau in taus]

    rows = []
    fig_df = None
    for name, make_policy in policies:
        for seed in seeds:
            df = _rank1_stream_df(seed, refresh_policy=make_policy())
            if name == "no refresh" and fig_df is None:
                fig_df = df
            rows.append({"policy": name, "seed": seed,
                         **_bound_run_summary(df)})
            if name == "no refresh":
                for t in (refresh_every, len(df) - 1):
                    r = df.iloc[t]
                    print(f"[no refresh, seed {seed}] t={t}: E_inc={r.frob_error:.4g} "
                          f"beta_t={r.error_bound:.4g} E_opt={r.frob_opt:.4g} "
                          f"beta_t/E_inc={r.error_bound / r.frob_error:.4g} "
                          f"E_opt/||A||_F={r.frob_opt / r.frob_norm:.4g}")

    per_seed = pd.DataFrame(rows)
    order = [name for name, _ in policies]
    grouped = per_seed.drop(columns="seed").groupby("policy", sort=False)
    summary = grouped.mean().reindex(order)
    summary.insert(1, "n_full_svd_min", grouped["n_full_svd"].min().reindex(order))
    summary.insert(2, "n_full_svd_max", grouped["n_full_svd"].max().reindex(order))

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n[4.5] per seed:")
        print(per_seed.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(f"\n[4.5] mean over seeds {list(seeds)}:")
        print(summary.to_string(float_format=lambda x: f"{x:.4g}"))

    # E_inc vs beta_t (and E_opt for reference), no refresh, t >= 1 (log scale)
    # Line styles from plot_style: bound solid, E_inc dashed, E_opt dotted.
    d = fig_df[fig_df["t"] >= 1]
    path = ps.out_path("exp45_rank1_error_bound.pdf", outdir, only)
    ps.figure(path)
    plt.plot(d["t"], d["error_bound"], label=r"bound $\beta_t$")
    plt.plot(d["t"], d["frob_error"], label=r"incremental $E_{\mathrm{inc}}(t)$")
    plt.plot(d["t"], d["frob_opt"], alpha=0.7, label=r"optimal $E_{\mathrm{opt}}(t)$")
    plt.yscale("log")
    plt.xlabel("Time step t")
    plt.ylabel("Frobenius error")
    plt.legend()
    ps.save(path)

    return {"no_refresh": fig_df, "per_seed": per_seed, "summary": summary}


# ======================================================================
# Sec. 4.6: efficiency and scaling (timing only)
# ======================================================================

def experiment_44_scaling_size(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> pd.DataFrame:
    """Sec. 4.6: mean rank-1 update time against the matrix size (SCALING_*).

    Not a paper figure: writes exp44_scaling_size_update_time.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    DataFrame
        Columns m, n, mean_update_time (seconds).
    """
    sizes = list(SCALING_SIZES)
    mean_times: list[float] = []

    for (m, n) in sizes:
        print(f"\n[scaling_size] Running rank-1 stream for size m={m}, n={n}")
        rng = np.random.default_rng(SCALING_SIZE_SEED)
        true_rank = SCALING_TRUE_RANK
        U_true = rng.standard_normal((m, true_rank))
        V_true = rng.standard_normal((n, true_rank))
        A0 = U_true @ V_true.T

        inc = IncrementalSVD(A0, rank=SCALING_SIZE_RANK, compute_mask=False)
        A_full = A0.copy()

        times = []
        T = SCALING_T
        for _ in range(T):
            u = rng.integers(0, m)
            v = rng.integers(0, n)
            delta = SCALING_DELTA_SCALE * rng.standard_normal()
            A_full[u, v] += delta

            tic = perf_counter()
            inc.rank1_update(u, v, delta, update_matrix=True, incremental=True)
            toc = perf_counter()
            times.append(toc - tic)

        mean_t = float(np.mean(times))
        mean_times.append(mean_t)
        print(f"  mean incremental update_time = {mean_t:.6e} s")

    mn = [m * n for (m, n) in sizes]
    ps.figure(None)
    plt.plot(mn, mean_times, marker="o")
    plt.xlabel("Matrix size m × n")
    plt.ylabel("Mean update time (s)")
    ps.save(ps.out_path("exp44_scaling_size_update_time.pdf", outdir, only))

    return pd.DataFrame({"m": [m for m, _ in sizes], "n": [n for _, n in sizes],
                         "mean_update_time": mean_times})


def experiment_44_scaling_rank(
    outdir: str = ".", only: Optional[Collection[str]] = None
) -> pd.DataFrame:
    """Sec. 4.6: mean rank-1 update time against the truncation rank k (SCALING_RANK_*).

    The paper figure exp44_scaling_rank_update_time is made by
    timing_figures.py from repeated runs of this function.

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    only : collection of str, optional
        File names of the figures to save; None saves all, () none.

    Returns
    -------
    DataFrame
        Columns k, mean_update_time (seconds).
    """
    m, n = SCALING_RANK_SHAPE
    true_rank = SCALING_TRUE_RANK
    ks = list(SCALING_RANK_KS)
    mean_times: list[float] = []

    rng = np.random.default_rng(SCALING_RANK_SEED)
    U_true = rng.standard_normal((m, true_rank))
    V_true = rng.standard_normal((n, true_rank))
    A0 = U_true @ V_true.T

    for k in ks:
        print(f"\n[scaling_rank] Running rank-1 stream for k={k}")
        inc = IncrementalSVD(A0, rank=k, compute_mask=False)
        A_full = A0.copy()

        times = []
        T = SCALING_T
        for _ in range(T):
            u = rng.integers(0, m)
            v = rng.integers(0, n)
            delta = SCALING_DELTA_SCALE * rng.standard_normal()
            A_full[u, v] += delta

            tic = perf_counter()
            inc.rank1_update(u, v, delta, update_matrix=True, incremental=True)
            toc = perf_counter()
            times.append(toc - tic)

        mean_t = float(np.mean(times))
        mean_times.append(mean_t)
        print(f"  [k={k}] mean incremental update_time = {mean_t:.6e} s")

    path = ps.out_path("exp44_scaling_rank_update_time.pdf", outdir, only)
    ps.figure(path)
    plt.plot(ks, mean_times, marker="o")
    plt.xlabel("Truncation rank k")
    plt.ylabel("Mean update time (s)")
    ps.save(path)

    return pd.DataFrame({"k": ks, "mean_update_time": mean_times})


# ======================================================================
# All experiments and the paper figures
# ======================================================================

def run_all_experiments_section4(outdir: str = ".") -> None:
    """Run every Sec. 4 experiment and save all its figures (also those not in the paper).

    Parameters
    ----------
    outdir : str
        Folder for the figures.
    """
    experiment_41_rank1_refresh_comparison(outdir)
    experiment_41_rank_effect(outdir)
    experiment_42_row_stream(outdir)
    experiment_42_column_stream(outdir)
    experiment_43_mixed_stream(outdir)
    experiment_45_error_bound_refresh(outdir)
    experiment_44_scaling_size(outdir)
    experiment_44_scaling_rank(outdir)


# Figures in tex/figs/ produced by each synthetic experiment. The two timing
# figures (exp41_rank_effect_update_time, exp44_scaling_rank_update_time) are
# made by timing_figures.py (one BLAS thread, median over repeated runs).
PAPER_FIGURES = {
    experiment_41_rank1_refresh_comparison: ("exp41_rank1_error_ratio.pdf",
                                             "exp41_rank1_angle_opt.pdf"),
    experiment_41_rank_effect: ("exp41_rank_effect_error_ratio.pdf",),
    experiment_42_row_stream: ("exp42_row_error_ratio.pdf",),
    experiment_42_column_stream: ("exp42_col_error_ratio.pdf",),
    experiment_43_mixed_stream: ("exp43_mixed_error_ratio.pdf",),
    experiment_45_error_bound_refresh: ("exp45_rank1_error_bound.pdf",),
}


# Name of each experiment's metrics in the files written by reproduce.py.
METRICS_NAMES = {
    experiment_41_rank1_refresh_comparison: "exp41_rank1",
    experiment_41_rank_effect: "exp41_rank_effect",
    experiment_42_row_stream: "exp42_row",
    experiment_42_column_stream: "exp42_col",
    experiment_43_mixed_stream: "exp43_mixed",
    experiment_45_error_bound_refresh: "exp45",
}


def make_paper_figures(outdir: str) -> Dict[str, pd.DataFrame]:
    """Regenerate the non-timing Sec. 4 figures of the paper (PAPER_FIGURES).

    Parameters
    ----------
    outdir : str
        Folder for the figures.

    Returns
    -------
    dict of str to DataFrame
        Metrics of every run, keyed "<METRICS_NAMES[experiment]>__<label>",
        e.g. "exp41_rank1__no_refresh" or "exp45__summary".
    """
    metrics = {}
    for experiment, figures in PAPER_FIGURES.items():
        for label, df in experiment(outdir, only=figures).items():
            metrics[f"{METRICS_NAMES[experiment]}__{label}"] = df
    return metrics


def print_bound_table(summary: pd.DataFrame) -> None:
    """Print Table tab:bound-refresh in the units of the paper (angles in 1e-3 rad).

    Parameters
    ----------
    summary : DataFrame
        The "summary" output of :func:`experiment_45_error_bound_refresh`.
    """
    table = pd.DataFrame({
        "SVDs": summary["n_full_svd"],
        "min": summary["n_full_svd_min"],
        "max": summary["n_full_svd_max"],
        "SVDs t>T/2": summary["n_full_svd_second_half"],
        "ratio mean": summary["frob_ratio_mean"],
        "ratio max": summary["frob_ratio_max"],
        "ratio max t>T/2": summary["frob_ratio_max_second_half"],
        "angle mean (1e-3)": 1e3 * summary["angle_to_opt_mean"],
        "angle max (1e-3)": 1e3 * summary["angle_to_opt_max"],
        "beta/E_inc median": summary["bound_over_err_median"],
    })
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(float_format=lambda x: f"{x:.4g}"))
