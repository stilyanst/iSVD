"""Track a rank-5 SVD of a growing, changing matrix and refresh it when the error bound says so."""
import numpy as np

from isvd import IncrementalSVD, bound
from isvd.policies import apply_refresh_policy, new_state

rng = np.random.default_rng(0)
inc = IncrementalSVD(rng.standard_normal((100, 5)) @ rng.standard_normal((5, 40)), rank=5)
policy, state = bound(0.02), new_state()  # refresh when beta_t / ||A_t||_F > 0.02

for t in range(1, 501):
    if t % 5 == 0:  # append a row
        inc.add_row(rng.standard_normal(5) @ inc.Vt * 3 + 0.01 * rng.standard_normal(40))
    else:           # rank-1 update of one entry
        i, j = rng.integers(inc.A.shape[0]), rng.integers(inc.A.shape[1])
        inc.rank1_update(i, j, 0.1 * rng.standard_normal())
    apply_refresh_policy(policy, t, inc, state)

print(f"A is {inc.A.shape[0]} x {inc.A.shape[1]}, {state['n_refreshes']} refreshes")
print(f"true error ||A - U S V^T||_F = {inc.frobenius_error():.4f}")
print(f"a posteriori bound beta_t    = {inc.error_bound():.4f}  (no SVD needed)")
