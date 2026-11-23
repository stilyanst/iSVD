"""Regenerate the three timing figures of the paper.

    exp41_rank_effect_update_time.pdf        (Fig. fig:rank_runtime_comparison(a), Sec. 4.2,
                                              synthetic.experiment_41_rank_effect)
    exp44_scaling_rank_update_time.pdf       (Fig. fig:rank_runtime_comparison(b), Secs. 4.2
                                              and 4.6, synthetic.experiment_44_scaling_rank)
    sec7_finance_runtime_vs_rank_no_leak.pdf (Fig. fig:finance-runtime, Sec. 5.4, periodic
                                              rows of the ETF rank grid)

Timings are made comparable by
  - limiting numpy's BLAS/LAPACK to one thread (OMP_NUM_THREADS,
    OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are set to 1 below, before numpy
    is imported; run this file as a script, not after importing numpy),
  - pinning the process to one CPU (--cpu, default 4: a performance core on the
    i7-13700H used for the paper; on hybrid CPUs the scheduler may otherwise
    move the process between fast and slow cores),
  - running each experiment --repeats times (default 5) and plotting, for every
    point, the median over the runs of the mean per-update time.

Each figure point is a mean over all timed updates of one run (the experiment
code is unchanged); the per-run values and the medians are printed.

Usage (from the repo root, on an otherwise idle machine):
    .venv/bin/python experiments/timing_figures.py [--outdir results] [--repeats 5] [--cpu 4]
        [--synthetic-only]
"""
import os
import sys

if "numpy" in sys.modules:
    raise RuntimeError("timing_figures must be run before numpy is imported "
                       "(the BLAS thread limit is read at import time)")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import platform  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import synthetic  # noqa: E402
import finance as ef  # noqa: E402
import plot_style as ps  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = (
    "exp41_rank_effect_update_time.pdf",
    "exp44_scaling_rank_update_time.pdf",
    "sec7_finance_runtime_vs_rank_no_leak.pdf",
)


def machine_info() -> dict:
    """Describe the machine the timings are measured on.

    Returns
    -------
    dict
        CPU model, CPUs the process may run on, Python and NumPy versions, BLAS
        library and the number of threads of the process after a BLAS call.
    """
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas = f"{blas.get('name')} {blas.get('version')}"
    except Exception:  # older numpy
        blas = "unknown"
    # BLAS threads actually started: run a BLAS call, then count the process threads.
    a = np.ones((500, 500))
    a @ a
    try:
        n_threads = len(os.listdir("/proc/self/task"))
    except OSError:
        n_threads = -1
    return {
        "cpu": cpu,
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "numpy": np.__version__,
        "blas": blas,
        "process_threads": n_threads,
        "python": platform.python_version(),
    }


def _plot_rank_times(ks, times, path):
    """Same plot as synthetic.experiment_41_rank_effect / experiment_44_scaling_rank."""
    ps.figure(path)
    plt.plot(ks, times, marker="o")
    plt.xlabel("Truncation rank k")
    plt.ylabel("Mean update time (s)")
    ps.save(path)


def rank_effect_times(repeats: int) -> pd.DataFrame:
    """Sec. 4.2 rank effect: mean update time over t >= RANK_EFFECT_PLOT_FROM, per run.

    Parameters
    ----------
    repeats : int
        Number of runs.

    Returns
    -------
    DataFrame
        Columns run, k, mean_update_time (seconds).
    """
    rows = []
    for r in range(repeats):
        dfs = synthetic.experiment_41_rank_effect(".", only=())
        for label, df in dfs.items():
            k = int(label.split("=")[1])
            rows.append({"run": r, "k": k, "mean_update_time": df.loc[df["t"] >= synthetic.RANK_EFFECT_PLOT_FROM, "update_time"].mean()})
    return pd.DataFrame(rows)


def scaling_rank_times(repeats: int) -> pd.DataFrame:
    """Sec. 4.6 scaling with k (500 x 400, 300 rank-1 updates per k), per run.

    Parameters
    ----------
    repeats : int
        Number of runs.

    Returns
    -------
    DataFrame
        Columns k, mean_update_time (seconds), run.
    """
    out = []
    for r in range(repeats):
        df = synthetic.experiment_44_scaling_rank(".", only=())
        df["run"] = r
        out.append(df)
    return pd.concat(out, ignore_index=True)


def finance_runtime(repeats: int) -> pd.DataFrame:
    """Sec. 5.4: the ETF rank grid with the periodic and no-refresh policies only, per run.

    Parameters
    ----------
    repeats : int
        Number of runs.

    Returns
    -------
    DataFrame
        Columns k, policy, mean_update_time (seconds), n_refreshes, run.
    """
    out = []
    for r in range(repeats):
        grid = ef.experiment_sec7_finance_rank_refresh_grid(
            path=ef.DATA_PATH, ranks=ef.GRID_RANKS, refresh_policies=ef.GRID_REFRESH,
            demean=ef.PAPER_DEMEAN, **ef.FINANCE_COMMON)
        grid = grid[["k", "policy", "mean_update_time", "n_refreshes"]].copy()
        grid["run"] = r
        out.append(grid)
    return pd.concat(out, ignore_index=True)


def _report(name: str, df: pd.DataFrame, keys) -> pd.DataFrame:
    wide = df.pivot_table(index=keys, columns="run", values="mean_update_time")
    wide["median"] = wide.median(axis=1)
    wide["max/min"] = wide.drop(columns="median").max(axis=1) / wide.drop(columns="median").min(axis=1)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3e}".format):
        print(f"\n=== {name}: mean update time (s) per run, median over runs ===")
        print(wide.to_string())
    return wide["median"].reset_index()


def main() -> None:
    """Command line entry point: measure, print and plot the three timing figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=os.path.join(REPO, "results"),
                        help="output folder (default: results/)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=4, help="CPU to pin to (-1: no pinning)")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="skip the Sec. 5 finance timing figure")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("use at least 3 repeats")
    if args.cpu >= 0 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    os.makedirs(args.outdir, exist_ok=True)

    info = machine_info()
    print("Machine:", info)
    print("Load average before:", os.getloadavg())

    re_med = _report("Sec. 4.2 rank effect", rank_effect_times(args.repeats), ["k"])
    _plot_rank_times(re_med["k"], re_med["median"], os.path.join(args.outdir, FIGURES[0]))

    sr_med = _report("Sec. 4.6 scaling with k", scaling_rank_times(args.repeats), ["k"])
    _plot_rank_times(sr_med["k"], sr_med["median"], os.path.join(args.outdir, FIGURES[1]))

    if not args.synthetic_only:
        fin = finance_runtime(args.repeats)
        fin_med = _report("Sec. 5 finance grid", fin, ["k", "policy"])
        fin_med = fin_med.rename(columns={"median": "mean_update_time"})
        ef.plot_finance_runtime_vs_rank(fin_med, filename=os.path.join(args.outdir, FIGURES[2]))

    print("\nLoad average after:", os.getloadavg())
    print(f"Timing figures written to {args.outdir}")


if __name__ == "__main__":
    main()
