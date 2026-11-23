"""Accuracy metrics of an incremental SVD against the optimal rank-k SVD (paper Sec. 2.5)."""
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .core import IncrementalSVD


def principal_angle(Q_ref: ArrayLike, Q: ArrayLike) -> float:
    """Largest principal angle (radians) between span(Q_ref) and span(Q), eq. (6).

    The cosines of the principal angles are the singular values of
    Q_ref^T Q, so the largest angle is arccos(sigma_min(Q_ref^T Q)).

    Parameters
    ----------
    Q_ref, Q : array_like, shapes (m_ref, k_ref) and (m, k)
        Bases with orthonormal columns. If the numbers of rows differ (rows
        appended since Q_ref was taken), the shorter basis is padded with zero
        rows; if the numbers of columns differ, the first min(k_ref, k)
        columns of each are used.

    Returns
    -------
    float
        Angle in [0, pi/2]; NaN if either basis has no column. Near 0 the
        result carries roundoff of order 1e-8 (arccos amplifies roundoff in
        the cosine), so identical subspaces give an angle below about 1e-7.
    """
    U_ref = np.asarray(Q_ref, float)
    U_cur = np.asarray(Q, float)

    # Same number of columns (subspace dimension)
    k_ref = U_ref.shape[1]
    k_cur = U_cur.shape[1]
    k = min(k_ref, k_cur)
    if k == 0:
        return float("nan")

    U_ref = U_ref[:, :k]
    U_cur = U_cur[:, :k]

    # Same number of rows (ambient dimension): pad with zero rows
    m_ref = U_ref.shape[0]
    m_cur = U_cur.shape[0]
    if m_ref < m_cur:
        pad = np.zeros((m_cur - m_ref, k))
        U_ref = np.vstack((U_ref, pad))
    elif m_cur < m_ref:
        pad = np.zeros((m_ref - m_cur, k))
        U_cur = np.vstack((U_cur, pad))

    M = U_ref.T @ U_cur
    _, svals, _ = np.linalg.svd(M, full_matrices=False)
    svals = np.clip(svals, -1.0, 1.0)
    min_cos = float(np.min(svals))
    return float(np.arccos(min_cos))


def optimal_rank_k_error(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> float:
    """Error ||A - A_k||_F of the best rank-k approximation, E_opt in eq. (9).

    A_k = U[:, :k] diag(S[:k]) Vt[:k, :] is formed explicitly from a full SVD
    A = U diag(S) Vt, so the result is the Frobenius norm of a computed
    difference (equal to ||S[k:]||_2 in exact arithmetic).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        The matrix.
    U, S, Vt : ndarray
        Its thin SVD, singular values in decreasing order.
    k : int
        Rank of the approximation.

    Returns
    -------
    float
    """
    A_opt = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return float(np.linalg.norm(A_opt - A, "fro"))


def compute_metrics(
    inc: "IncrementalSVD",
    A_full: Optional[np.ndarray] = None,
    update_type: str = "",
    update_time: Optional[float] = None,
    compute_optimal: bool = False,
) -> Dict[str, Any]:
    """Accuracy metrics of the incremental factors at one step (paper Sec. 2.5).

    Parameters
    ----------
    inc : IncrementalSVD
        Engine after the step.
    A_full : ndarray, shape (m, n), optional
        Ground-truth matrix A_t; default ``inc.A``.
    update_type : str
        Label of the step ("init", "row", "col", "rank1", ..., "+refresh").
    update_time : float, optional
        Seconds spent in the update (and refresh) of this step.
    compute_optimal : bool, default False
        Also compute a full SVD of A_full for the optimal rank-k error and
        subspace (one full SVD per call: small matrices only).

    Returns
    -------
    dict
        t (None, set by the caller), update_type, frob_error (E_inc(t) =
        ||A_full - U diag(S) Vt||_F, eq. (9)), error_bound (beta_t,
        Proposition 1(b), with respect to inc.A), frob_norm (||A_full||_F),
        evr (eq. (11)), angle (to the initial subspace U0, eq. (7)), rank,
        update_time; and, if compute_optimal, frob_opt (E_opt(t), eq. (9)),
        angle_to_opt (to the optimal rank-k subspace, eq. (8)), frob_gap
        (E_inc - E_opt), frob_ratio (E_inc / E_opt, eq. (10); NaN if
        E_opt <= 1e-12) and opt_time (seconds for the full SVD and E_opt),
        otherwise None for these.
    """
    if A_full is None:
        A_full = inc.A

    frob = inc.frobenius_error(A_true=A_full)
    evr = inc.explained_variance_ratio()
    angle = inc.subspace_principal_angle()

    result = {
        "t": None,  # caller sets this
        "update_type": update_type,
        "frob_error": frob,
        "error_bound": inc.error_bound(),
        "frob_norm": float(np.linalg.norm(A_full, "fro")),
        "evr": evr,
        "angle": angle,
        "rank": inc.rank,
        "update_time": update_time,  # incremental update time
        "frob_opt": None,
        "frob_gap": None,
        "frob_ratio": None,
        "opt_time": None,           # time for optimal baseline
    }

    if compute_optimal:
        tic_opt = perf_counter()

        # Full SVD of the current ground-truth matrix
        U_full, S_full, Vt_full = np.linalg.svd(A_full, full_matrices=False)
        k_eff = min(inc.rank, len(S_full))

        frob_opt = optimal_rank_k_error(A_full, U_full, S_full, Vt_full, k_eff)

        toc_opt = perf_counter()
        opt_time = toc_opt - tic_opt

        angle_alg = principal_angle(U_full[:, :k_eff], inc.U)

        result["frob_opt"] = frob_opt
        result["angle_to_opt"] = angle_alg
        result["frob_gap"] = float(frob - frob_opt)
        if frob_opt > 1e-12:
            result["frob_ratio"] = float(frob / frob_opt)
        else:
            result["frob_ratio"] = np.nan
        result["opt_time"] = opt_time

    return result
