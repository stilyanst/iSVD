"""A missing ETF price CSV gives an error that says how to build it (data/README.md)."""
import os
import subprocess
import sys

import pytest

import finance

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_load_missing_csv_points_to_data_readme(tmp_path):
    missing = str(tmp_path / "prices_etf.csv")
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        finance.load_price_data_and_returns(missing)


def test_finance_cli_missing_csv_exits_with_message(tmp_path):
    missing = str(tmp_path / "prices_etf.csv")
    res = subprocess.run(
        [sys.executable, os.path.join(REPO, "experiments", "finance.py"), "--data", missing,
         "--outdir", str(tmp_path)],
        capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert res.returncode != 0
    assert "data/README.md" in res.stderr and "prepare_etf_data.py" in res.stderr
    assert "Traceback" not in res.stderr
