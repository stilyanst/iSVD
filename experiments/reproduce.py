"""Regenerate every figure of the paper and print its two numerical tables.

    python experiments/reproduce.py [--outdir results] [--synthetic-only]
                                    [--figures-only] [--skip-timing]

Writes into --outdir (default: results/ in the repository):

- the 11 non-timing figures (Secs. 4 and 5), as vector PDFs;
- the 3 timing figures, made by timing_figures.py in a separate process (one
  BLAS thread, pinned to one CPU, median of 5 runs; run on an idle machine);
- metrics/<name>.pkl and metrics/<name>.csv: the raw metrics behind every
  non-timing figure and table, one file per run;

and prints Table tab:bound-refresh (Sec. 4.5) and Table
tab:finance-bound-refresh (Sec. 5.3).

To update the figures of the paper's LaTeX source, pass its figure folder
(tex/figs in the author's copy; not part of the public repository). The Sec. 5
experiments need data/prices_etf.csv (see data/README.md).

Options:
    --synthetic-only  only the Sec. 4 (synthetic) experiments
    --figures-only    no tables and no metrics files
    --skip-timing     no timing figures (they take about 4 minutes)
"""
import argparse
import os
import subprocess
import sys
import time
from typing import Dict

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

try:
    import isvd  # noqa: E402,F401
except ImportError:
    sys.exit("isvd is not installed: run `pip install -e .` in the repository root (see README.md)")

import finance  # noqa: E402
import synthetic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def save_metrics(metrics: Dict[str, pd.DataFrame], outdir: str) -> None:
    """Write each DataFrame to <outdir>/<name>.pkl (exact) and <name>.csv (readable).

    Parameters
    ----------
    metrics : dict of str to DataFrame
        Metrics keyed by file name stem.
    outdir : str
        Output folder, created if needed.
    """
    os.makedirs(outdir, exist_ok=True)
    for name, df in metrics.items():
        df.to_pickle(os.path.join(outdir, name + ".pkl"))
        df.to_csv(os.path.join(outdir, name + ".csv"))


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=os.path.join(REPO, "results"),
                        help="output folder (default: results/)")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="only the Sec. 4 synthetic experiments")
    parser.add_argument("--figures-only", action="store_true",
                        help="write the figures only: no tables, no metrics files")
    parser.add_argument("--skip-timing", action="store_true",
                        help="do not regenerate the three timing figures")
    args = parser.parse_args()

    if not args.synthetic_only and not os.path.exists(finance.DATA_PATH):
        sys.exit(finance.missing_data_message(finance.DATA_PATH)
                 + " Or run with --synthetic-only (Sec. 4 only, no data needed).")
    os.makedirs(args.outdir, exist_ok=True)
    start = time.perf_counter()

    metrics = synthetic.make_paper_figures(args.outdir)
    if not args.synthetic_only:
        metrics.update(finance.make_paper_figures(args.outdir))

    if not args.skip_timing:
        cmd = [sys.executable, os.path.join(HERE, "timing_figures.py"), "--outdir", args.outdir]
        if args.synthetic_only:
            cmd.append("--synthetic-only")
        subprocess.run(cmd, check=True)

    if not args.figures_only:
        save_metrics(metrics, os.path.join(args.outdir, "metrics"))
        print("\n=== Table tab:bound-refresh (Sec. 4.5): mean over seeds "
              f"{list(synthetic.BOUND_EXP_SEEDS)} ===")
        synthetic.print_bound_table(metrics["exp45__summary"])
        if not args.synthetic_only:
            print("\n=== Table tab:finance-bound-refresh (Sec. 5.3): final day, "
                  f"k in {finance.GRID_BOUND_RANKS} ===")
            finance.print_finance_grid_table(metrics["sec7_finance__grid"])

    print(f"\nWrote the paper figures to {args.outdir} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
