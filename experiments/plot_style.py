"""Figure style for the paper (Springer Numerical Algorithms artwork guidelines).

- Vector PDF with embedded TrueType fonts (pdf.fonttype 42, never Type 3).
- Sans-serif font (Liberation Sans, metric-compatible with Arial/Helvetica), 8 pt.
- The figure is created at the width it is printed at in tex/main.tex, so LaTeX
  includes it at scale 1 and the 8 pt text stays 8 pt:
    PANEL_WIDTH_PT  = 0.48 * 372 = \\panelfigwidth  (178.56bp, one panel of a pair)
    SINGLE_WIDTH_PT = 0.64 * 372 = \\singlefigwidth (238.08bp, figure on its own)
  where 372 is the sn-jnl text width. Widths are in matplotlib's point (bp,
  1/72 in), as are the macros, so LaTeX includes the PDFs at scale exactly 1.
  Keep them in sync with tex/main.tex and with SINGLE_FIGURES below.
- Lines differ in line style (and markers where the x values are few), not only
  in color, so the figures read in black and white.

Usage: `figure(path)` creates a figure sized for the paper file `path`,
`legend_above(path)` adds a legend above the axes, `save(path)` writes and
closes the figure. `out_path` and `line_plot` are the helpers shared by
synthetic.py and finance.py.
"""
import os
from typing import Collection, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure

TEXTWIDTH_PT = 372.0
PANEL_WIDTH_PT = 0.48 * TEXTWIDTH_PT
SINGLE_WIDTH_PT = 0.64 * TEXTWIDTH_PT
PANEL_ASPECT = 0.85   # height / width
SINGLE_ASPECT = 0.7

# Paper figures printed on their own (\singlefigwidth); all others are panels.
SINGLE_FIGURES = {
    "exp41_rank_effect_error_ratio",
    "exp43_mixed_error_ratio",
    "exp45_rank1_error_bound",
    "sec7_finance_angle_factor",
}

# Matplotlib's default color order, so each series keeps its old color,
# paired with a line style so it can be told apart in black and white.
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (6, 2)), (0, (1, 3))]
# Markers for curves with few x values (one marker per line, in order).
MARKERS = ["o", "^", "v", "D", "s", "P", "X"]

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.sf": "Liberation Sans",
    "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "lines.linewidth": 1.0,         # pt
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "patch.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.25,
    "legend.labelspacing": 0.2,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    "legend.framealpha": 0.9,
    "axes.prop_cycle": cycler(color=COLORS) + cycler(linestyle=LINESTYLES),
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 2 / 72,
    "figure.constrained_layout.w_pad": 2 / 72,
    "savefig.format": "pdf",
}


def apply() -> None:
    """Set the paper style (:data:`RC`) in matplotlib's global rcParams."""
    mpl.rcParams.update(RC)


def figsize(path: Optional[str]) -> Tuple[float, float]:
    """Figure size of the paper file `path`.

    Parameters
    ----------
    path : str or None
        Output path; its stem decides between a single figure
        (:data:`SINGLE_FIGURES`) and a panel (anything else, also None).

    Returns
    -------
    (float, float)
        Width and height in inches.
    """
    stem = os.path.splitext(os.path.basename(path))[0] if path else ""
    if stem in SINGLE_FIGURES:
        w, aspect = SINGLE_WIDTH_PT, SINGLE_ASPECT
    else:
        w, aspect = PANEL_WIDTH_PT, PANEL_ASPECT
    return (w / 72.0, w * aspect / 72.0)


def figure(path: Optional[str]) -> Figure:
    """New figure in the paper style, sized for the paper file `path` (see :func:`figsize`).

    Parameters
    ----------
    path : str or None
        Output path the figure will be saved to.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply()
    return plt.figure(figsize=figsize(path))


def legend_above(path: Optional[str]) -> None:
    """Put the legend of the current figure above the axes, so it hides no data.

    Parameters
    ----------
    path : str or None
        Output path; three legend columns for a single figure, two for a panel.
    """
    ncol = 3 if figsize(path)[0] * 72 > PANEL_WIDTH_PT + 1 else 2
    plt.gcf().legend(loc="outside upper center", ncol=ncol)


def save(path: Optional[str]) -> None:
    """Save the current figure to `path` at its exact size and close it.

    No bbox_inches="tight": that would change the size the figure is printed at.
    CreationDate is omitted so the same data give the same PDF bytes.

    Parameters
    ----------
    path : str or None
        Output path; None only closes the figure.
    """
    if path is not None:
        plt.savefig(path, metadata={"CreationDate": None})
    plt.close()


def out_path(filename: str, outdir: str, only: Optional[Collection[str]]) -> Optional[str]:
    """Output path of a figure, or None if it is not wanted.

    Parameters
    ----------
    filename : str
        File name of the figure.
    outdir : str
        Output folder.
    only : collection of str or None
        File names to save; None saves all.

    Returns
    -------
    str or None
        ``os.path.join(outdir, filename)``, or None if `only` excludes it.
    """
    if only is not None and filename not in only:
        return None
    return os.path.join(outdir, filename)


def line_plot(
    df_list: List[pd.DataFrame],
    x_col: str,
    y_col: str,
    labels: List[str],
    xlabel: str,
    ylabel: str,
    filename: Optional[str] = None,
) -> None:
    """Plot y_col against x_col, one line per DataFrame, and save the figure.

    Rows with NaN in either column are dropped; DataFrames without y_col or
    with no valid rows are skipped. A legend above the axes is added if there
    is more than one label.

    Parameters
    ----------
    df_list : list of DataFrame
        One DataFrame per line.
    x_col, y_col : str
        Columns plotted on the x and y axes.
    labels : list of str
        Legend label of each line.
    xlabel, ylabel : str
        Axis labels.
    filename : str or None
        Output path (see :func:`save`).
    """
    figure(filename)
    for df, lab in zip(df_list, labels):
        if y_col not in df.columns:
            continue
        valid = df[[x_col, y_col]].dropna()
        if valid.empty:
            continue
        plt.plot(valid[x_col], valid[y_col], label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(labels) > 1:
        legend_above(filename)
    save(filename)
