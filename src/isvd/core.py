"""The incremental SVD engine (paper Sec. 2 and Appendix A).

:class:`IncrementalSVD` maintains a rank-k truncated SVD A ~ U diag(S) Vt of a
dense matrix under row appends, column appends and rank-1 entry updates
(Brand-style updates of a small core matrix), with full refreshes on demand. It
also tracks N_t = ||A_t||_F^2 and the a posteriori error bound beta_t of
Proposition 1 in O(1) per update.
"""
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .metrics import principal_angle


def _truncate(U, S, Vt, k):
    """Keep the k largest singular triplets (U[:, i], S[i], Vt[i, :]).

    Returns (U, S, Vt, eta) with eta = ||discarded singular values||_2, and the
    inputs unchanged with eta = 0.0 if k is None or there are at most k triplets.
    """
    if k is None or len(S) <= k:
        return U, S, Vt, 0.0
    order = np.argsort(-S)
    idx = order[:k]
    return U[:, idx], S[idx], Vt[idx, :], float(np.linalg.norm(S[order[k:]]))


class IncrementalSVD:
    """Truncated SVD of a streaming matrix, A ~ U diag(S) Vt.

    Supports row appends (:meth:`add_row`), column appends (:meth:`add_col`, via
    the transpose of a row append), rank-1 entry updates A <- A + delta e_u e_v^T
    (:meth:`rank1_update`) and full refreshes (:meth:`refresh`). A dense copy of
    A is kept in :attr:`A`, so that refreshes and accuracy metrics can be
    computed.

    Parameters
    ----------
    A_init : array_like, shape (m, n)
        Initial matrix.
    rank : int
        Target rank k of the truncated SVD.
    compute_mask : bool, default False
        Optional, for recommender data with 0 for missing entries: store the
        boolean mask A != 0 (kept up to date by the update methods) and use it
        in :meth:`rmse`. Not used in the paper.
    store_initial_subspace : bool, default True
        Store the factors of the initial SVD as (U0, S0, Vt0), the reference
        for :meth:`subspace_principal_angle`. U0 is always stored on the first
        SVD; this flag matters only through :meth:`refresh`.

    Attributes
    ----------
    A : ndarray, shape (m, n)
        Current matrix A_t.
    U, S, Vt : ndarray, shapes (m, r), (r,), (r, n)
        Current factors, r <= k.
    rank : int
        Target rank k (reduced to min(k, m, n) at each full SVD).
    U0, S0, Vt0 : ndarray
        Reference factors for drift measurements.
    N : float
        N_t = ||A_t||_F^2, exact at each full SVD and updated in O(1) per update.
    beta : float
        beta_t of Proposition 1(b); read it with :meth:`error_bound`.
    last_eta : float
        eta_s of the most recent update (Proposition 1(a)).
    mask : ndarray of bool or None
        Observed-entry mask if ``compute_mask`` is True, else None.

    Examples
    --------
    >>> import numpy as np
    >>> from isvd import IncrementalSVD
    >>> rng = np.random.default_rng(0)
    >>> inc = IncrementalSVD(rng.standard_normal((20, 15)), rank=4)
    >>> inc.add_row(rng.standard_normal(15))
    >>> inc.rank1_update(0, 3, 0.1)
    >>> bool(inc.frobenius_error() <= inc.error_bound() + 1e-12)
    True
    """

    def __init__(
        self,
        A_init: ArrayLike,
        rank: int,
        compute_mask: bool = False,
        store_initial_subspace: bool = True,
    ) -> None:
        A_init = np.asarray(A_init, dtype=float)
        if A_init.ndim != 2:
            raise ValueError("A_init must be a 2D array")

        self.A = A_init.copy()
        self.rank = int(rank)

        if compute_mask:
            self.mask = (self.A != 0)
        else:
            self.mask = None

        # Full singular spectrum of the last full SVD
        self._S_full = None

        self.last_eta = 0.0

        self._recompute_svd(store_initial=store_initial_subspace)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recompute_svd(self, store_initial=False):
        """Recompute a rank-k truncated SVD of the current matrix A and reset N_t and beta_t."""
        U, S, Vt = np.linalg.svd(self.A, full_matrices=False)

        self._S_full = S.copy()

        # N_t = ||A_t||_F^2, set exactly here and updated in O(1) per update
        self.N = float(np.sum(self.A * self.A))

        k = min(self.rank, len(S))
        self.U = U[:, :k]
        self.S = S[:k]
        self.Vt = Vt[:k, :]
        self.rank = k

        # beta_t (Prop. 1(b)) restarts at E_opt(t0) = ||sigma_{k+1:}(A_t0)||_2,
        # taken from the discarded singular values. (sqrt(N_t0 - ||Sigma_t0||_F^2)
        # is equal in exact arithmetic but cancels when E_opt(t0) << ||A_t0||_F.)
        self.beta = float(np.linalg.norm(S[k:]))

        if store_initial or not hasattr(self, "U0"):
            self.U0 = self.U.copy()
            self.S0 = self.S.copy()
            self.Vt0 = self.Vt.copy()

    def _record_eta(self, eta):
        """Record eta_s, the norm of the part discarded by the update just applied,
        and add it to beta_t = E_opt(t0) + sum_{s > t0} eta_s (Prop. 1(b))."""
        self.last_eta = float(eta)
        self.beta += self.last_eta

    def reconstruct(self) -> np.ndarray:
        """Low-rank approximation U diag(S) Vt.

        Returns
        -------
        ndarray, shape (m, n)
        """
        return self.U @ (self.S[:, None] * self.Vt)

    # ------------------------------------------------------------------
    # Evaluation metrics
    # ------------------------------------------------------------------

    def rmse(self, A_true: Optional[ArrayLike] = None, mask: Optional[np.ndarray] = None) -> float:
        """Root mean squared error of the approximation on selected entries.

        Optional, for recommender data (not used in the paper).

        Parameters
        ----------
        A_true : array_like, shape (m, n), optional
            Reference matrix; default :attr:`A`.
        mask : ndarray of bool, shape (m, n), optional
            Entries to include; default :attr:`mask` if set, else A_true != 0.

        Returns
        -------
        float
            RMSE over the selected entries, NaN if none is selected.
        """
        if A_true is None:
            A_true = self.A
        A_true = np.asarray(A_true, dtype=float)

        if mask is None:
            if self.mask is not None:
                mask = self.mask
            else:
                mask = (A_true != 0)

        A_pred = self.reconstruct()
        diff = (A_pred - A_true)[mask]
        if diff.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(diff**2)))

    def frobenius_error(self, A_true: Optional[ArrayLike] = None) -> float:
        """Frobenius error ||U diag(S) Vt - A_true||_F, E_inc(t) of eq. (9).

        Parameters
        ----------
        A_true : array_like, shape (m, n), optional
            Reference matrix; default :attr:`A`.

        Returns
        -------
        float
        """
        if A_true is None:
            A_true = self.A
        A_true = np.asarray(A_true, dtype=float)
        A_pred = self.reconstruct()
        return float(np.linalg.norm(A_pred - A_true, ord="fro"))

    def error_bound(self) -> float:
        """A posteriori bound beta_t >= ||A_t - U diag(S) Vt||_F (Proposition 1(b)).

        beta_t = E_opt(t0) + sum_{s=t0+1}^{t} eta_s, where t0 is the last full
        SVD (initialization or refresh), E_opt(t0) = ||sigma_{k+1:}(A_t0)||_2
        (the discarded singular values) and eta_s is the norm of the part
        discarded by update s. Costs O(1): no reconstruction or SVD. Valid in
        exact arithmetic when every change to A goes through the update methods
        of this class.

        Returns
        -------
        float
        """
        return float(self.beta)

    def explained_variance_ratio(self) -> float:
        """Explained variance ratio ||Sigma_t||_F^2 / N_t of the maintained factors (eq. (11)).

        N_t = ||A_t||_F^2 is tracked incrementally (exact at each full SVD), so
        the denominator is that of the current matrix, not of the last refresh.

        Returns
        -------
        float
            NaN if N_t == 0.
        """
        num = float(np.sum(self.S**2))
        if self.N == 0:
            return float("nan")
        return num / self.N

    def subspace_principal_angle(self, U_ref: Optional[ArrayLike] = None) -> float:
        """Largest principal angle (radians) between span(U_ref) and span(U), eq. (6).

        Parameters
        ----------
        U_ref : array_like, shape (m_ref, k_ref), optional
            Reference basis with orthonormal columns; default U0, the basis of
            the initial SVD (``angle`` in eq. (7)). If the number of rows
            differs from that of U (appended rows), the shorter basis is padded
            with zero rows; if the number of columns differs, the first
            min(k_ref, k) columns of each are used.

        Returns
        -------
        float
            Angle in [0, pi/2]; NaN if there is no reference or no column.
        """
        if U_ref is None:
            if not hasattr(self, "U0"):
                return float("nan")
            U_ref = self.U0

        return principal_angle(U_ref, self.U)

    # ------------------------------------------------------------------
    # Incremental kernels (Brand-style updates of a small core matrix)
    # ------------------------------------------------------------------

    @staticmethod
    def _incremental_add_row(U, S, Vt, x_row, k=None, tol=1e-10, return_eta=False):
        """Factors of [A; x_row] from those of A ~ U diag(S) Vt (Sec. 2.2.1, Algorithm A1).

        The residual z = x_row - (x_row V) V^T of eq. (4) has norm rho. If
        rho < tol the row is treated as lying in span(V): no new right
        direction is added (the residual is dropped) and the result has r
        instead of r+1 components, so U and Vt stay orthonormal. Otherwise the
        (r+1) x (r+1) core [diag(S) 0; p rho] is diagonalized. The result is
        truncated to k components if k is given.

        If return_eta is True, also returns eta_s of Proposition 1(a): the norm
        of the core singular values removed by the truncation (sigma'_{k+1})
        plus rho when rho < tol.
        """
        U = np.asarray(U, float)
        S = np.asarray(S, float)
        Vt = np.asarray(Vt, float)
        x = np.asarray(x_row, float).reshape(1, -1)  # (1, n)

        m, rdim = U.shape
        rdim = len(S)
        n = Vt.shape[1]
        if x.shape[1] != n:
            raise ValueError("New row must have length n")

        # 1) Project the new row onto span(V)
        p = x @ Vt.T       # (1, r)
        p = p.ravel()

        # 2) Residual orthogonal to span(V), eq. (4)
        r = x - p @ Vt     # (1, n)
        r = r.ravel()
        rho = np.linalg.norm(r)

        eta = 0.0
        if rho < tol:
            # 3a) Row lies in span(V): (r+1) x r core [diag(S); p], no new direction
            eta += float(rho)  # dropped residual
            K = np.vstack((np.diag(S), p.reshape(1, -1)))
            Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)  # Uk: (r+1, r)
            Vt_new = Vkt @ Vt                                    # (r, n)
        else:
            # 3b) Small (r+1) x (r+1) core matrix with the new direction r/rho
            r_unit = r / rho
            K = np.zeros((rdim + 1, rdim + 1))
            K[:rdim, :rdim] = np.diag(S)
            K[-1, :rdim] = p
            K[-1, -1] = rho

            Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

            Vt_aug = np.vstack((Vt, r_unit.reshape(1, -1)))  # (r+1, n)
            Vt_new = Vkt @ Vt_aug                            # (r+1, n)

        # 4) Update U: old rows and the new row
        U_old = U @ Uk[:-1, :]                # (m, r')
        U_new_row = Uk[-1, :].reshape(1, -1)  # (1, r')
        U_new = np.vstack((U_old, U_new_row)) # (m+1, r'), r' = r or r+1

        # 5) Optional truncation: truncated sigma'_{k+1}
        U_new, Sk, Vt_new, eta_trunc = _truncate(U_new, Sk, Vt_new, k)
        eta += eta_trunc

        if return_eta:
            return U_new, Sk, Vt_new, eta
        return U_new, Sk, Vt_new

    @staticmethod
    def _incremental_add_col_via_rows(U, S, Vt, x_col, k=None, tol=1e-10, return_eta=False):
        """Factors of [A, x_col] from those of A ~ U diag(S) Vt (Sec. 2.2.2, Algorithm A2).

        A ~ U S Vt implies A^T ~ V S U^T, so a new column of A is a new row of
        A^T: the row append is applied to A^T and the result transposed back.

        If return_eta is True, also returns eta_s of Proposition 1(a): the
        truncated core singular value(s) plus rho = ||x_col - U U^T x_col||
        when rho < tol.
        """
        U = np.asarray(U, float)
        S = np.asarray(S, float)
        Vt = np.asarray(Vt, float)
        x = np.asarray(x_col, float).ravel()  # (m,)

        m, rdim = U.shape
        n = Vt.shape[1]
        if x.shape[0] != m:
            raise ValueError("New column must have length m")

        # Work on B = A^T, where B ≈ V S U^T
        U_B = Vt.T           # (n, r)
        S_B = S              # (r,)
        Vt_B = U.T           # (r, m)

        # In B, we add new ROW x^T
        b_row = x.reshape(1, -1)  # (1, m)
        U_B_new, S_new, Vt_B_new, eta = IncrementalSVD._incremental_add_row(
            U_B, S_B, Vt_B, b_row, k=None, tol=tol, return_eta=True
        )

        # Now B' ≈ U_B_new S_new Vt_B_new
        # ⇒ A' = (B')^T ≈ (V_B_new) S_new (U_B_new)^T
        V_B_new = Vt_B_new.T       # (m, r')
        U_A_new = V_B_new          # (m, r')
        Vt_A_new = U_B_new.T       # (r', n+1)

        U_A_new, S_new, Vt_A_new, eta_trunc = _truncate(U_A_new, S_new, Vt_A_new, k)
        eta += eta_trunc  # truncated sigma'_{k+1}

        if return_eta:
            return U_A_new, S_new, Vt_A_new, eta
        return U_A_new, S_new, Vt_A_new

    @staticmethod
    def _incremental_rank1(U, S, Vt, u_index, v_index, delta, k=None, return_eta=False):
        """Factors after A <- A + delta e_u e_v^T, from those of A (Sec. 2.3, Algorithm A3).

        Diagonalizes the k x k core K = diag(S) + delta U[u, :]^T V[v, :] in the
        current subspaces, eq. (5). For a truncated SVD this approximates the
        exact update: the part of the update outside span(U) x span(V) is
        discarded, which is the source of the error accumulation studied in the
        experiments.

        If return_eta is True, also returns eta_s of Proposition 1(a), the norm
        of that discarded part: |delta| sqrt(max(0, 1 - a b)) with
        a = ||U[u, :]||^2 and b = ||V[v, :]||^2 before the update.
        """
        U = np.asarray(U, float)
        S = np.asarray(S, float)
        Vt = np.asarray(Vt, float)

        # 1) Coordinates of the update direction in the current subspaces
        s_u = U[u_index, :]      # (k,)
        s_v = Vt[:, v_index]     # (k,)
        ab = float(s_u @ s_u) * float(s_v @ s_v)
        eta = abs(float(delta)) * float(np.sqrt(max(0.0, 1.0 - ab)))

        # 2) Core k x k matrix
        K = np.diag(S) + delta * np.outer(s_u, s_v)
        Uk, Sk, Vkt = np.linalg.svd(K, full_matrices=False)

        # 3) Update U, Vt
        U_new = U @ Uk
        Vt_new = Vkt @ Vt

        U_new, Sk, Vt_new, eta_trunc = _truncate(U_new, Sk, Vt_new, k)
        eta += eta_trunc

        if return_eta:
            return U_new, Sk, Vt_new, eta
        return U_new, Sk, Vt_new

    # ------------------------------------------------------------------
    # Streaming updates
    # ------------------------------------------------------------------

    def add_row(self, x_row: ArrayLike, incremental: bool = True, tol: float = 1e-10) -> None:
        """Append a row to A and update the factors, N_t and beta_t.

        Parameters
        ----------
        x_row : array_like, shape (n,)
            New row.
        incremental : bool, default True
            If True, use the incremental row append; if False, recompute the
            SVD of the new matrix (a refresh).
        tol : float, default 1e-10
            Residual norm below which the row is treated as lying in span(V).

        Raises
        ------
        ValueError
            If the length of x_row is not the number of columns of A.
        """
        x_row = np.asarray(x_row, float).ravel()
        if x_row.shape[0] != self.A.shape[1]:
            raise ValueError("Length of x_row must equal current number of columns (n)")

        self.A = np.vstack((self.A, x_row[None, :]))
        self.N += float(x_row @ x_row)
        if self.mask is not None:
            new_mask_row = (x_row != 0)
            self.mask = np.vstack((self.mask, new_mask_row[None, :]))

        if incremental:
            self.U, self.S, self.Vt, eta = self._incremental_add_row(
                self.U, self.S, self.Vt, x_row, k=self.rank, tol=tol, return_eta=True
            )
            self._record_eta(eta)
        else:
            self._recompute_svd()

    def add_col(self, x_col: ArrayLike, incremental: bool = True, tol: float = 1e-10) -> None:
        """Append a column to A and update the factors, N_t and beta_t.

        Parameters
        ----------
        x_col : array_like, shape (m,)
            New column.
        incremental : bool, default True
            If True, use the incremental column append; if False, recompute the
            SVD of the new matrix (a refresh).
        tol : float, default 1e-10
            Residual norm below which the column is treated as lying in span(U).

        Raises
        ------
        ValueError
            If the length of x_col is not the number of rows of A.
        """
        x_col = np.asarray(x_col, float).ravel()
        if x_col.shape[0] != self.A.shape[0]:
            raise ValueError("Length of x_col must equal current number of rows (m)")

        self.A = np.hstack((self.A, x_col.reshape(-1, 1)))
        self.N += float(x_col @ x_col)
        if self.mask is not None:
            new_mask_col = (x_col != 0).reshape(-1, 1)
            self.mask = np.hstack((self.mask, new_mask_col))

        if incremental:
            self.U, self.S, self.Vt, eta = self._incremental_add_col_via_rows(
                self.U, self.S, self.Vt, x_col, k=self.rank, tol=tol, return_eta=True
            )
            self._record_eta(eta)
        else:
            self._recompute_svd()

    def rank1_update(
        self,
        u: int,
        v: int,
        delta: float,
        update_matrix: bool = True,
        incremental: bool = True,
    ) -> None:
        """Rank-1 entry update A <- A + delta e_u e_v^T (eq. (3)).

        Parameters
        ----------
        u, v : int
            Row and column index of the entry.
        delta : float
            Increment of entry (u, v).
        update_matrix : bool, default True
            If True, also update the stored matrix A[u, v] and N_t. If False,
            A is unchanged and the whole projected change counts as error in
            beta_t.
        incremental : bool, default True
            If True, use the incremental rank-1 update; if False, recompute the
            SVD (a refresh).

        Raises
        ------
        IndexError
            If (u, v) is outside A.
        """
        if not (0 <= u < self.A.shape[0]) or not (0 <= v < self.A.shape[1]):
            raise IndexError("Row or column index out of range")

        if update_matrix:
            # N_t += 2 delta A_uv + delta^2, with the old A_uv
            self.N += 2.0 * float(delta) * float(self.A[u, v]) + float(delta) ** 2
            self.A[u, v] += float(delta)

        if not incremental:
            self._recompute_svd()
        else:
            if not update_matrix:
                # A is unchanged, so the whole projected update
                # delta * P_U e_u e_v^T P_V is error: eta = |delta| sqrt(a b)
                ab = float(self.U[u, :] @ self.U[u, :]) * float(self.Vt[:, v] @ self.Vt[:, v])
                eta_unchanged_A = abs(float(delta)) * float(np.sqrt(ab))
            self.U, self.S, self.Vt, eta = self._incremental_rank1(
                self.U, self.S, self.Vt, u, v, float(delta), k=self.rank, return_eta=True
            )
            self._record_eta(eta if update_matrix else eta_unchanged_A)

    def set_entry(self, u: int, v: int, value: float, mode: str = "full") -> None:
        """Assign A[u, v] = value. Optional; not used in the paper.

        Parameters
        ----------
        u, v : int
            Row and column index of the entry.
        value : float
            New value of the entry (an assignment, not an increment).
        mode : {"full", "none"}, default "full"
            "full" recomputes the SVD; "none" keeps the factors, and the whole
            change |value - old| is added to beta_t.

        Raises
        ------
        IndexError
            If (u, v) is outside A.
        ValueError
            If mode is not "full" or "none".
        """
        if not (0 <= u < self.A.shape[0]) or not (0 <= v < self.A.shape[1]):
            raise IndexError("Row or column index out of range")

        old = float(self.A[u, v])
        self.N += float(value) ** 2 - old ** 2
        self.A[u, v] = float(value)
        if self.mask is not None:
            self.mask[u, v] = (value != 0)

        if mode == "none":
            # factors unchanged: the whole change (value - old) e_u e_v^T is error
            self._record_eta(abs(float(value) - old))
            return
        elif mode == "full":
            self._recompute_svd()
        else:
            raise ValueError("mode must be one of {'full', 'none'}")

    def refresh(self, new_rank: Optional[int] = None, store_initial_subspace: bool = False) -> None:
        """Replace the factors by a truncated SVD of the current matrix A (Sec. 2.4).

        Resets the accumulated error: afterwards beta_t = E_opt(t) and N_t is
        recomputed exactly. Called by the refresh policies (:mod:`isvd.policies`).

        Parameters
        ----------
        new_rank : int, optional
            New target rank k.
        store_initial_subspace : bool, default False
            If True, the refreshed factors become the reference (U0, S0, Vt0)
            for :meth:`subspace_principal_angle`.
        """
        if new_rank is not None:
            self.rank = int(new_rank)
        self._recompute_svd(store_initial=store_initial_subspace)
