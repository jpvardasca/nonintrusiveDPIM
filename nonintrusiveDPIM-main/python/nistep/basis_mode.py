"""Basis-mode EED-StEP (StEP.tex, nonlinear stiffness coefficients) and Galerkin export.

CONVENTIONS 4.6. For a real basis ``V = [V_0 .. V_{n-1}]`` (N × n; columns are modes, static
derivatives, anything), the engine gives the polarised forms ``Ĝ(V_i,V_j)`` and
``Ĥ(V_i,V_j,V_k)`` for ``i <= j <= k``. From them:

* **Thesis monomial coefficients** (``basis_coefficients``), all N-vectors::

      G̃_ii = Ĝ(V_i,V_i)            G̃_ij = 2 Ĝ(V_i,V_j)
      H̃_iii = Ĥ(V_i,V_i,V_i)       H̃_iij = 3 Ĥ(V_i,V_i,V_j)     H̃_ijj = 3 Ĥ(V_i,V_j,V_j)
      H̃_ijk = 6 Ĥ(V_i,V_j,V_k)

  so that ``g(V q) = Σ_{i<=j} G̃_ij q_i q_j`` and ``h(V q) = Σ_{i<=j<=k} H̃_ijk q_i q_j q_k``.

* **Galerkin tensors** (``galerkin_tensors``): ``g_r[:, i, j] = Vᵀ Ĝ(V_i,V_j)`` and
  ``h_r[:, i, j, k] = Vᵀ Ĥ(V_i,V_j,V_k)``, stored fully symmetric (no multiplicities).
  This reduced model is an approximation (StEP.tex footnote on constraint forces); the
  closure route through MORFE is the reference.

Indices are **0-based** and refer to the columns of ``V``.

Cost: ``n_eed(n) = 2n + C(n-1, 2) = (n² + n + 2)/2`` tangent evaluations (plus ``K_tg(0)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


def n_eed(n: int) -> int:
    """Tangent evaluations of the EED-StEP for an n-vector basis: ``2n + C(n-1, 2)``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return (n * n + n + 2) // 2


def n_ed(n: int) -> int:
    """Static solutions of the ED-StEP, for comparison only: ``(n³ + 6n² + 5n)/6``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return (n**3 + 6 * n * n + 5 * n) // 6


def quad_multiplicity(i: int, j: int) -> int:
    """Monomial coefficient = multiplicity × polarised form (CONVENTIONS 4.2)."""
    return 1 if i == j else 2


def cubic_multiplicity(i: int, j: int, k: int) -> int:
    return {1: 1, 2: 3, 3: 6}[len({i, j, k})]


def _basis(engine, V) -> np.ndarray:
    V = np.asarray(V)
    if np.iscomplexobj(V):
        raise ValueError("basis mode needs a real basis V")
    V = np.array(V, dtype=np.float64)
    if V.ndim == 1:
        V = V[:, None]
    if V.ndim != 2 or V.shape[0] != engine.n or V.shape[1] < 1:
        raise ValueError(f"V must have shape ({engine.n}, n), got {V.shape}")
    if not np.all(np.isfinite(V)):
        raise ValueError("V has non-finite entries")
    if np.any(np.max(np.abs(V), axis=0) == 0.0):
        raise ValueError("V has a zero column")
    return V


def _real(x: np.ndarray) -> np.ndarray:
    if np.max(np.abs(x.imag)) != 0.0:
        raise FloatingPointError("internal: real basis gave a complex result")
    return np.ascontiguousarray(x.real)


def polarised_forms(engine, V, cubic: bool = True, prefetch: bool = True,
                    max_parallel: int | None = None):
    """``({(i,j): Ĝ(V_i,V_j)}, {(i,j,k): Ĥ(V_i,V_j,V_k)})`` for ``i <= j <= k`` (real N-vectors).

    With ``prefetch`` the needed jobs are first collected in record mode and run as one batch
    (``max_parallel`` jobs at a time), so a slow backend can run them in parallel.
    """
    V = _basis(engine, V)
    n = V.shape[1]
    cols = [V[:, i] for i in range(n)]
    engine.register(*cols)  # ids in basis order: probe = lower index, pair = two lowest

    def evaluate():
        G = {(i, j): engine.quad(cols[i], cols[j])
             for i, j in combinations_with_replacement(range(n), 2)}
        H = {}
        if cubic:
            H = {(i, j, k): engine.cubic(cols[i], cols[j], cols[k])
                 for i, j, k in combinations_with_replacement(range(n), 3)}
        return G, H

    if prefetch:
        mode = engine.mode
        engine.set_mode("record")
        try:
            evaluate()
            engine.execute_pending(max_parallel)
        finally:
            engine.set_mode(mode)
    G, H = evaluate()
    if engine.pending_count():
        raise RuntimeError("basis mode: jobs are still pending; call engine.execute_pending()")
    return {k: _real(v) for k, v in G.items()}, {k: _real(v) for k, v in H.items()}


def basis_coefficients(engine, V, cubic: bool = True, prefetch: bool = True,
                       max_parallel: int | None = None) -> dict:
    """Thesis monomial coefficients: keys ``('G', i, j)``, ``('H', i, j, k)``, ``i <= j <= k``."""
    G, H = polarised_forms(engine, V, cubic=cubic, prefetch=prefetch, max_parallel=max_parallel)
    out = {("G", i, j): quad_multiplicity(i, j) * g for (i, j), g in G.items()}
    out.update({("H", i, j, k): cubic_multiplicity(i, j, k) * h for (i, j, k), h in H.items()})
    return out


def save_basis_coefficients(path, coeffs: dict, V) -> Path:
    """Write ``V`` and the coefficients to an npz file (arrays ``G_i_j`` and ``H_i_j_k``)."""
    path = Path(path)
    arrays = {"V": np.asarray(V, dtype=np.float64)}
    for key, val in coeffs.items():
        arrays["_".join(str(x) for x in key)] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)
    return path


def load_basis_coefficients(path) -> tuple[np.ndarray, dict]:
    with np.load(Path(path)) as z:
        V = np.array(z["V"])
        coeffs = {}
        for name in z.files:
            if name == "V":
                continue
            kind, *idx = name.split("_")
            coeffs[(kind, *(int(x) for x in idx))] = np.array(z[name])
    return V, coeffs


@dataclass
class GalerkinModel:
    """Reduced model ``Mr q̈ + Cr q̇ + Kr q + g(q,q) + h(q,q,q) = Vᵀ F`` on the basis ``V``."""

    V: np.ndarray   # (N, n)
    Mr: np.ndarray  # (n, n)
    Kr: np.ndarray  # (n, n)
    Cr: np.ndarray  # (n, n), alpha*Mr + beta*Kr
    g: np.ndarray   # (n, n, n), symmetric in the last two indices
    h: np.ndarray   # (n, n, n, n), symmetric in the last three indices

    def internal_force(self, q) -> np.ndarray:
        """Reduced internal force ``Vᵀ F_int(V q)``."""
        q = np.asarray(q)
        return (self.Kr @ q + np.einsum("rij,i,j->r", self.g, q, q)
                + np.einsum("rijk,i,j,k->r", self.h, q, q, q))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            np.savez(fh, V=self.V, Mr=self.Mr, Kr=self.Kr, Cr=self.Cr, g=self.g, h=self.h)
        return path

    @classmethod
    def load(cls, path) -> "GalerkinModel":
        with np.load(Path(path)) as z:
            return cls(**{k: np.array(z[k]) for k in ("V", "Mr", "Kr", "Cr", "g", "h")})


def galerkin_tensors(engine, V, alpha: float = 0.0, beta: float = 0.0, prefetch: bool = True,
                     max_parallel: int | None = None) -> GalerkinModel:
    """Galerkin projection on the real basis ``V`` (transpose, never conjugate)."""
    V = _basis(engine, V)
    n = V.shape[1]
    G, H = polarised_forms(engine, V, prefetch=prefetch, max_parallel=max_parallel)
    Mr = V.T @ (engine.M() @ V)
    Kr = V.T @ (engine.K() @ V)
    g = np.zeros((n, n, n))
    for (i, j), val in G.items():
        g[:, i, j] = g[:, j, i] = V.T @ val
    h = np.zeros((n, n, n, n))
    for (i, j, k), val in H.items():
        r = V.T @ val
        for a, b, c in {(i, j, k), (i, k, j), (j, i, k), (j, k, i), (k, i, j), (k, j, i)}:
            h[:, a, b, c] = r
    return GalerkinModel(V=V, Mr=Mr, Kr=Kr, Cr=alpha * Mr + beta * Kr, g=g, h=h)
