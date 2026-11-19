"""Tangent-stiffness backend protocol (CONVENTIONS section 4).

A backend gives the tangent stiffness ``K_tg(U)`` of the internal force
``F_int(U) = K U + g(U) + h(U)`` on the **free DOFs** (clamped DOFs removed, DofMap
order). Displacements ``U`` are always **real** float64 vectors of length ``n_dof``.

Contract
--------
* ``stiffness0()`` must equal ``tangent_many([zeros])[0]``: the same assembly and the
  same sparsity pattern (CONVENTIONS 4.3, "K must be the backend's own K_tg(0)").
* All matrices returned by one backend have the same CSR pattern (explicit zeros kept),
  so the store can share ``indptr``/``indices``.
* ``exact_quad`` / ``exact_cubic`` are ``None`` unless the backend knows the exact
  symmetric polarised forms (Analytic, PySVK). They take complex vectors.
"""

from __future__ import annotations

import abc
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp


@runtime_checkable
class TangentBackend(Protocol):
    kind: str
    n_dof: int
    exact_quad: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    exact_cubic: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None

    def fingerprint(self) -> str: ...

    def stiffness0(self) -> sp.csr_matrix: ...

    def mass(self) -> sp.csr_matrix: ...

    def tangent_many(self, U_list: Sequence[np.ndarray]) -> list[sp.csr_matrix]: ...


class BaseBackend(abc.ABC):
    """Shared validation and bookkeeping for concrete backends."""

    kind: str = "base"
    n_dof: int = 0
    exact_quad = None  # overridden by backends that know the exact forms
    exact_cubic = None

    def __init__(self) -> None:
        self.n_tangent_evals = 0  # number of K_tg(U) evaluations ("jobs") served

    # -- to implement ---------------------------------------------------------------
    @abc.abstractmethod
    def fingerprint(self) -> str: ...

    @abc.abstractmethod
    def stiffness0(self) -> sp.csr_matrix: ...

    @abc.abstractmethod
    def mass(self) -> sp.csr_matrix: ...

    @abc.abstractmethod
    def _tangent_batch(self, U_list: list[np.ndarray]) -> list[sp.csr_matrix]: ...

    # -- public -----------------------------------------------------------------------
    def tangent_many(self, U_list: Sequence[np.ndarray]) -> list[sp.csr_matrix]:
        checked = [self.check_U(U) for U in U_list]
        if not checked:
            return []
        out = self._tangent_batch(checked)
        if len(out) != len(checked):
            raise RuntimeError(f"{self.kind}: backend returned {len(out)} matrices for {len(checked)} inputs")
        self.n_tangent_evals += len(checked)
        return out

    def tangent(self, U: np.ndarray) -> sp.csr_matrix:
        return self.tangent_many([U])[0]

    def check_U(self, U) -> np.ndarray:
        U = np.asarray(U)
        if np.iscomplexobj(U):
            raise TypeError(f"{self.kind}: prescribed displacements must be real")
        U = U.astype(np.float64, copy=False).ravel()
        if U.shape != (self.n_dof,):
            raise ValueError(f"{self.kind}: U must have shape ({self.n_dof},), got {U.shape}")
        if not np.all(np.isfinite(U)):
            raise ValueError(f"{self.kind}: U contains non-finite values")
        return U
