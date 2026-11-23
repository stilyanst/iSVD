"""examples/quickstart.py runs and its bound holds."""
import os
import runpy

QUICKSTART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "quickstart.py")


def test_quickstart_runs(capsys):
    ns = runpy.run_path(QUICKSTART, run_name="__main__")
    inc, state = ns["inc"], ns["state"]
    assert inc.A.shape == (200, 40)
    assert 0 < state["n_refreshes"] < 500
    assert inc.frobenius_error() <= inc.error_bound() + 1e-10
    assert "a posteriori bound" in capsys.readouterr().out
