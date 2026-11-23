"""Refresh policies for streaming incremental SVD (paper Sec. 3).

A refresh policy decides, after each streaming update, whether the
incremental factors should be replaced by an exact truncated SVD of the
current matrix (:meth:`isvd.IncrementalSVD.refresh`). It is a callable

    should_refresh(t, inc, state) -> bool

where ``t`` is the 1-based index of the update that was just applied, ``inc``
the :class:`~isvd.IncrementalSVD` after that update, and ``state`` a dict owned
by the streaming loop and kept across steps. The loop sets
``state["last_refresh"]`` (t of the last refresh, 0 initially) and
``state["n_refreshes"]``; a policy may store its own entries in it.

Policies:

- :func:`periodic` ``(R)``: refresh after every R-th update (eq. (12)).
- :func:`bound` ``(tau)``: refresh when ``inc.error_bound() / sqrt(inc.N) > tau``,
  i.e. when the a posteriori bound beta_t of Proposition 1 relative to
  ||A_t||_F exceeds tau (eq. (16)). beta_t / ||A_t||_F never drops below the
  truncation floor E_opt(t) / ||A_t||_F, so tau must be above it.

A streaming loop uses :func:`new_state` once and :func:`apply_refresh_policy`
after every update; :func:`resolve_policy` turns the shortcut
``refresh_every=R`` into ``periodic(R)``.
"""
from typing import Any, Callable, Dict, Optional

RefreshPolicy = Callable[[int, Any, Dict[str, Any]], bool]


def periodic(R: int) -> RefreshPolicy:
    """Policy that refreshes after every R-th update (t = R, 2R, 3R, ...).

    Parameters
    ----------
    R : int
        Refresh period, a positive integer.

    Returns
    -------
    RefreshPolicy
        ``should_refresh(t, inc, state)``, true when ``t % R == 0``.

    Raises
    ------
    ValueError
        If R is None or not positive.
    """
    if R is None or R <= 0:
        raise ValueError("refresh_every must be a positive integer or None")

    def should_refresh(t: int, inc: Any, state: Dict[str, Any]) -> bool:
        return t % R == 0

    should_refresh.__name__ = f"periodic({R})"
    return should_refresh


def bound(tau: float) -> RefreshPolicy:
    """Policy that refreshes when the a posteriori relative error bound exceeds tau.

    Refreshes after update t if beta_t / sqrt(N_t) > tau (eq. (16)), where
    beta_t = ``inc.error_bound()`` bounds ||A_t - U S Vt||_F (Proposition 1(b))
    and N_t = ``inc.N`` = ||A_t||_F^2. Costs O(1) per step (no SVD). Never
    refreshes while N_t == 0.

    Parameters
    ----------
    tau : float
        Tolerance on beta_t / ||A_t||_F, a positive number.

    Returns
    -------
    RefreshPolicy
        ``should_refresh(t, inc, state)``.

    Raises
    ------
    ValueError
        If tau is None or not positive.
    """
    if tau is None or not tau > 0:
        raise ValueError("tau must be a positive number")

    def should_refresh(t: int, inc: Any, state: Dict[str, Any]) -> bool:
        if inc.N <= 0:
            return False
        return inc.error_bound() / inc.N ** 0.5 > tau

    should_refresh.__name__ = f"bound({tau})"
    return should_refresh


def resolve_policy(
    refresh_policy: Optional[RefreshPolicy] = None,
    refresh_every: Optional[int] = None,
) -> Optional[RefreshPolicy]:
    """Return the policy to use; ``refresh_every=R`` is a shortcut for ``periodic(R)``.

    Parameters
    ----------
    refresh_policy : RefreshPolicy or None
        Policy to use.
    refresh_every : int or None
        Refresh period R, used instead of ``refresh_policy``.

    Returns
    -------
    RefreshPolicy or None
        None (never refresh) if neither argument is given.

    Raises
    ------
    ValueError
        If both arguments are given.
    """
    if refresh_policy is not None and refresh_every is not None:
        raise ValueError("give either refresh_policy or refresh_every, not both")
    if refresh_every is not None:
        return periodic(refresh_every)
    return refresh_policy


def new_state() -> Dict[str, Any]:
    """Initial policy state of a streaming run.

    Returns
    -------
    dict
        ``{"last_refresh": 0, "n_refreshes": 0}``.
    """
    return {"last_refresh": 0, "n_refreshes": 0}


def apply_refresh_policy(
    policy: Optional[RefreshPolicy],
    t: int,
    inc: Any,
    state: Dict[str, Any],
) -> bool:
    """Ask ``policy`` about update t and refresh ``inc`` if it says so.

    The refresh keeps the initial subspace U0 as the drift reference
    (``store_initial_subspace=False``) and updates ``state``.

    Parameters
    ----------
    policy : RefreshPolicy or None
        Policy to ask; None never refreshes.
    t : int
        1-based index of the update just applied.
    inc : IncrementalSVD
        Engine after update t.
    state : dict
        Policy state from :func:`new_state`, updated in place.

    Returns
    -------
    bool
        True if a refresh was performed.
    """
    if policy is None or not policy(t, inc, state):
        return False
    inc.refresh(store_initial_subspace=False)
    state["last_refresh"] = t
    state["n_refreshes"] += 1
    return True
