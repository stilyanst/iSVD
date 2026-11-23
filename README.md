# isvd: incremental SVD with a posteriori error bounds and refresh policies

[![DOI](https://zenodo.org/badge/DOI/TODO-ZENODO-DOI.svg)](https://doi.org/TODO-ZENODO-DOI)
[![tests](https://github.com/stilyanst/iSVD/actions/workflows/tests.yml/badge.svg)](https://github.com/stilyanst/iSVD/actions/workflows/tests.yml)

Code of the paper "Incremental SVD for Large-Scale Dynamic Matrices" (arXiv:2605.24514),
version 1.0.0 (see `CHANGELOG.md`).

The package `isvd` maintains a truncated SVD A ≈ U diag(S) Vᵀ of a dense matrix under row
appends, column appends and rank-1 entry updates (Brand-style updates of a small core
matrix), refreshes it with a full SVD when a refresh policy asks for it, and keeps the a
posteriori error bound βₜ ≥ ‖Aₜ − U diag(S) Vᵀ‖_F of Proposition 1 up to date in O(1) per
update. The experiments in `experiments/` reproduce every figure and table of the paper.

## Install

Python 3.10 or newer. The package needs only NumPy:

```bash
pip install -e .                    # the isvd package
pip install -e ".[experiments,dev]" # + pandas, matplotlib (experiments) and pytest
```

To use the exact versions of the paper's runs (Python 3.12.3):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps
```

## Quickstart

```python
import numpy as np
from isvd import IncrementalSVD, bound
from isvd.policies import apply_refresh_policy, new_state

rng = np.random.default_rng(0)
inc = IncrementalSVD(rng.standard_normal((100, 5)) @ rng.standard_normal((5, 40)), rank=5)
policy, state = bound(0.02), new_state()     # refresh when beta_t / ||A_t||_F > 0.02

inc.add_row(rng.standard_normal(40))          # append a row
inc.add_col(rng.standard_normal(101))         # append a column
inc.rank1_update(3, 7, 0.5)                   # A[3, 7] += 0.5
apply_refresh_policy(policy, 3, inc, state)   # full SVD if the policy says so

print(inc.error_bound(), inc.frobenius_error())   # bound beta_t (O(1)) and true error
```

`examples/quickstart.py` runs a longer stream of row appends and rank-1 updates with the
`bound(tau)` policy. The main API:

| | |
|---|---|
| `IncrementalSVD(A, rank)` | truncated SVD of `A`; factors `U`, `S`, `Vt`, the matrix `A` |
| `add_row(x)`, `add_col(y)`, `rank1_update(i, j, delta)` | streaming updates |
| `refresh()` | full truncated SVD of the current matrix |
| `error_bound()` | a posteriori bound βₜ (Proposition 1(b)) |
| `frobenius_error()`, `explained_variance_ratio()`, `subspace_principal_angle()` | accuracy metrics |
| `isvd.periodic(R)`, `isvd.bound(tau)` | refresh policies (eqs. (12) and (16)) |
| `isvd.metrics.compute_metrics(inc, A)` | per-step metrics against the optimal rank-k SVD |

## Data

The synthetic experiments (Sec. 4) generate their data from fixed seeds. The finance
experiments (Sec. 5) need `data/prices_etf.csv`, daily prices of 60 ETFs from Yahoo Finance.
**This file is not included in the repository**: it is third-party data and is not
redistributed. Build it once (needs network access), see `data/README.md` for its contents and
the SHA-256 of the file used for the paper:

```bash
pip install -e ".[experiments,data]"        # yfinance is only needed here
python experiments/prepare_etf_data.py      # writes data/prices_etf.csv
```

Without it, the finance scripts stop with a message pointing to `data/README.md`; use
`reproduce.py --synthetic-only` for Sec. 4 alone. Yahoo Finance revises adjusted prices, so a
file downloaded later can change the Sec. 5 numbers in the last digits.

## Reproduce the paper

Then one command regenerates all 14 figures and prints Tables 2 and 3 (about 5 minutes
on a laptop, about 3 of them without the timing figures):

```bash
python experiments/reproduce.py                    # into results/
python experiments/reproduce.py --synthetic-only   # Sec. 4 only, no data needed
python experiments/reproduce.py --figures-only     # no tables, no metrics files
python experiments/reproduce.py --skip-timing      # no timing figures
```

Besides the figures (vector PDF), `reproduce.py` writes the raw metrics of every run to
`<outdir>/metrics/` (`.pkl` and `.csv`). All experiment settings (sizes, ranks, seeds,
refresh periods, tolerances) are module-level constants at the top of
`experiments/synthetic.py` and `experiments/finance.py`.

The non-timing results are deterministic: with the versions in `requirements.txt` they
come out identical (checked for the synthetic metrics and figures). Other NumPy/LAPACK builds
round differently: with NumPy 2.5.5 the rank-effect runs of Sec. 4.2 at k = 8 and 12 differ by
up to 0.7% (relative), the other synthetic runs by about 1e-8 (principal angles near zero). The three timing figures depend on the machine; `timing_figures.py`
(run by `reproduce.py`) limits BLAS to one thread, pins the process to one CPU and plots the
median of 5 runs, so run it on an otherwise idle machine.

## Layout

| Path | Contents |
|---|---|
| `src/isvd/core.py` | `IncrementalSVD`: updates, refresh, N_t and βₜ tracking |
| `src/isvd/policies.py` | refresh policies `periodic(R)`, `bound(tau)` and the loop helpers |
| `src/isvd/metrics.py` | principal angles, optimal rank-k error, per-step metrics |
| `experiments/streams.py` | synthetic streams (rank-1, rows, columns, mixed) |
| `experiments/synthetic.py` | Sec. 4 experiments and figures |
| `experiments/finance.py` | Sec. 5 ETF factor and covariance experiments and figures |
| `experiments/timing_figures.py` | the three timing figures |
| `experiments/reproduce.py` | all figures and tables of the paper |
| `experiments/plot_style.py` | journal figure style (fonts, sizes, line styles) |
| `experiments/prepare_etf_data.py` | download of `data/prices_etf.csv` |
| `examples/quickstart.py` | usage example |
| `tests/` | pytest tests |
| `data/` | `README.md` on the ETF data (the CSV itself is not included) |

## Tests

```bash
python -m pytest
```

The tests check the bound βₜ ≥ E_inc at every step, the discarded norms ηₛ, the tracking of
N_t = ‖Aₜ‖²_F, orthonormality of the factors, the `bound(tau)` policy, the principal-angle
implementation, short runs of the synthetic streams, the quickstart example and the message
for missing data. GitHub Actions runs them on Python 3.10 and 3.12
(`.github/workflows/tests.yml`).

## Citing

If you use this code, please cite the paper:

```bibtex
@misc{staykov2026isvd,
  author        = {Staykov, Stilyan},
  title         = {Incremental {SVD} for Large-Scale Dynamic Matrices},
  year          = {2026},
  eprint        = {2605.24514},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2605.24514}
}
```

To refer to this exact version of the code, cite the software archived on Zenodo as well:

```bibtex
@software{staykov2026isvd_code,
  author  = {Staykov, Stilyan},
  title   = {isvd: incremental {SVD} with a posteriori error bounds and refresh policies},
  version = {1.0.0},
  year    = {2026},
  doi     = {TODO-ZENODO-DOI},
  url     = {https://doi.org/TODO-ZENODO-DOI}
}
```

Both are in `CITATION.cff`, which GitHub shows as "Cite this repository" and Zenodo uses for
the archive's metadata.

## License

MIT, see `LICENSE`.
