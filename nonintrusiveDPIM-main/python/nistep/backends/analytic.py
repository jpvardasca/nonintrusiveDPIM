"""Analytic backend: an exact polynomial internal force with known polarised forms.

``F_int(U) = K U + Ĝ(U,U) + Ĥ(U,U,U)`` where (CONVENTIONS 4.2)

* ``Ĝ(a,b)_i   = Σ G[i,j,k]   a_j b_k``      with G symmetric in (j,k)
* ``Ĥ(a,b,c)_i = Σ H[i,j,k,l] a_j b_k c_l``  with H symmetric in (j,k,l)

and the exact tangent is ``K_tg(U) = K + 2 Ĝ(U,·) + 3 Ĥ(U,U,·)``.

Tensors are stored as sparse COO entry lists (:class:`QuadTensor`, :class:`CubicTensor`).
Factories:

* :func:`two_dof_example` is the StEP.tex 2-DoF mass-spring system.
* :func:`duffing` is a 1-DoF oscillator with a closed-form backbone.
* :func:`spring_chain` is a fixed-fixed chain with quadratic and cubic springs.
* :func:`random_sparse_model` uses random "generalised springs" (potential forces, SPD K).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable

import numpy as np
import scipy.sparse as sp

from .._sparse import FixedPattern, coo_entries
from .base import BaseBackend

# --------------------------------------------------------------------------- tensors


def _scatter(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    """out[i] = Σ_{e: idx[e]=i} vals[e]; complex-safe (no conjugation)."""
    if np.iscomplexobj(vals):
        return np.bincount(idx, weights=vals.real, minlength=n) + 1j * np.bincount(
            idx, weights=vals.imag, minlength=n
        )
    return np.bincount(idx, weights=vals, minlength=n).astype(np.float64)


@dataclass(frozen=True)
class QuadTensor:
    """Symmetric polarised quadratic form, entries G[i,j,k] = v (symmetric in j,k)."""

    n: int
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    v: np.ndarray

    def apply(self, a, b) -> np.ndarray:
        a = np.asarray(a)
        b = np.asarray(b)
        return _scatter(self.i, self.v * a[self.j] * b[self.k], self.n)


@dataclass(frozen=True)
class CubicTensor:
    """Symmetric polarised cubic form, entries H[i,j,k,l] = v (symmetric in j,k,l)."""

    n: int
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    l: np.ndarray
    v: np.ndarray

    def apply(self, a, b, c) -> np.ndarray:
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        return _scatter(self.i, self.v * a[self.j] * b[self.k] * c[self.l], self.n)


def _from_dict(n: int, acc: dict[tuple[int, ...], float], order: int):
    if acc:
        keys = np.array(list(acc.keys()), dtype=np.int64)
        vals = np.array(list(acc.values()), dtype=np.float64)
    else:
        keys = np.zeros((0, order + 1), dtype=np.int64)
        vals = np.zeros(0)
    cols = [keys[:, m].copy() for m in range(order + 1)]
    return (QuadTensor if order == 2 else CubicTensor)(n, *cols, vals)


def polarize_quadratic(n: int, monomials: Iterable[tuple[int, int, int, float]]) -> QuadTensor:
    """Monomials ``(i, j, k, coef)`` meaning ``g_i += coef * u_j * u_k`` give the polarised Ĝ."""
    acc: dict[tuple[int, ...], float] = {}
    for i, j, k, coef in monomials:
        perms = set(permutations((int(j), int(k))))
        for p in perms:
            key = (int(i),) + p
            acc[key] = acc.get(key, 0.0) + float(coef) / len(perms)
    return _from_dict(n, acc, 2)


def polarize_cubic(n: int, monomials: Iterable[tuple[int, int, int, int, float]]) -> CubicTensor:
    """Monomials ``(i, j, k, l, coef)`` meaning ``h_i += coef * u_j u_k u_l`` give the polarised Ĥ."""
    acc: dict[tuple[int, ...], float] = {}
    for i, j, k, l, coef in monomials:
        perms = set(permutations((int(j), int(k), int(l))))
        for p in perms:
            key = (int(i),) + p
            acc[key] = acc.get(key, 0.0) + float(coef) / len(perms)
    return _from_dict(n, acc, 3)


def springs_tensors(n: int, supports, coefs, k2, k3) -> tuple[QuadTensor, CubicTensor]:
    """Generalised springs ``e_s = b_s · u`` with potential ``k2/3 e³ + k3/4 e⁴``.

    Force ``F = Σ_s b_s (k2_s e_s² + k3_s e_s³)``, so ``G[i,j,k] = Σ k2 b_i b_j b_k`` and
    ``H[i,j,k,l] = Σ k3 b_i b_j b_k b_l``. Both are fully symmetric.
    """
    gi, gj, gk, gv = [], [], [], []
    hi, hj, hk, hl, hv = [], [], [], [], []
    for idx, b, q2, q3 in zip(supports, coefs, k2, k3):
        idx = np.asarray(idx, dtype=np.int64)
        b = np.asarray(b, dtype=np.float64)
        m = idx.size
        I, J, K_ = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij")
        I, J, K_ = I.ravel(), J.ravel(), K_.ravel()
        if q2 != 0.0:
            gi.append(idx[I]); gj.append(idx[J]); gk.append(idx[K_])
            gv.append(q2 * b[I] * b[J] * b[K_])
        if q3 != 0.0:
            I4, J4, K4, L4 = (x.ravel() for x in np.meshgrid(*(np.arange(m),) * 4, indexing="ij"))
            hi.append(idx[I4]); hj.append(idx[J4]); hk.append(idx[K4]); hl.append(idx[L4])
            hv.append(q3 * b[I4] * b[J4] * b[K4] * b[L4])

    def cat(xs, dtype):
        return np.concatenate(xs).astype(dtype) if xs else np.zeros(0, dtype=dtype)

    G = QuadTensor(n, cat(gi, np.int64), cat(gj, np.int64), cat(gk, np.int64), cat(gv, np.float64))
    H = CubicTensor(n, cat(hi, np.int64), cat(hj, np.int64), cat(hk, np.int64), cat(hl, np.int64),
                    cat(hv, np.float64))
    return G, H


# --------------------------------------------------------------------------- backend


class AnalyticBackend(BaseBackend):
    """Exact polynomial model; serves K_tg(U) with a fixed CSR pattern."""

    kind = "analytic"

    def __init__(self, K, M, G: QuadTensor, H: CubicTensor, name: str = "analytic"):
        super().__init__()
        K = sp.csr_matrix(K, dtype=np.float64)
        M = sp.csr_matrix(M, dtype=np.float64)
        n = K.shape[0]
        if K.shape != (n, n) or M.shape != (n, n) or G.n != n or H.n != n:
            raise ValueError("inconsistent dimensions of K, M, G, H")
        self.name = name
        self.n_dof = n
        self.K_lin = K
        self._M = M
        self.G = G
        self.H = H
        kr, kc, kv = coo_entries(K)
        self._k_vals = kv
        self._n_k = kv.size
        rows = np.concatenate([kr, G.i, H.i])
        cols = np.concatenate([kc, G.k, H.l])
        self._pattern = FixedPattern(rows, cols, (n, n))
        self._fp: str | None = None

    # exact polarised forms (complex-safe)
    def exact_quad(self, a, b) -> np.ndarray:
        return self.G.apply(a, b)

    def exact_cubic(self, a, b, c) -> np.ndarray:
        return self.H.apply(a, b, c)

    def internal_force(self, U) -> np.ndarray:
        U = np.asarray(U)
        return self.K_lin @ U + self.G.apply(U, U) + self.H.apply(U, U, U)

    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256()
            h.update(f"{self.kind}|{self.name}|{self.n_dof}|".encode())
            for arr in (self.K_lin.indptr, self.K_lin.indices, self.K_lin.data, self._M.indptr,
                        self._M.indices, self._M.data, self.G.i, self.G.j, self.G.k, self.G.v,
                        self.H.i, self.H.j, self.H.k, self.H.l, self.H.v):
                h.update(np.ascontiguousarray(arr).tobytes())
            self._fp = h.hexdigest()
        return self._fp

    def stiffness0(self) -> sp.csr_matrix:
        return self._tangent_one(np.zeros(self.n_dof))

    def mass(self) -> sp.csr_matrix:
        return self._M.copy()

    def _tangent_one(self, U: np.ndarray) -> sp.csr_matrix:
        G, H = self.G, self.H
        vals = np.concatenate([
            self._k_vals,
            2.0 * G.v * U[G.j],
            3.0 * H.v * U[H.j] * U[H.k],
        ])
        return self._pattern.assemble(vals)

    def _tangent_batch(self, U_list):
        return [self._tangent_one(U) for U in U_list]


# --------------------------------------------------------------------------- factories


def two_dof_example() -> AnalyticBackend:
    """StEP.tex 2-DoF example: k_g=10, k_c=5, G_{1,1}=G_{2,2}=20, H_{1,2,2}=H_{1,1,2}=10, m=1.

    ``F1 = 15 U1 - 5 U2 + 20 U1² + 10 U1 U2²`` and ``F2 = -5 U1 + 15 U2 + 20 U2² + 10 U1² U2``.
    """
    kg, kc = 10.0, 5.0
    K = np.array([[kg + kc, -kc], [-kc, kg + kc]])
    M = np.eye(2)
    G = polarize_quadratic(2, [(0, 0, 0, 20.0), (1, 1, 1, 20.0)])
    H = polarize_cubic(2, [(0, 0, 1, 1, 10.0), (1, 0, 0, 1, 10.0)])
    return AnalyticBackend(K, M, G, H, name="two_dof")


def duffing(k: float = 1.0, m: float = 1.0, k2: float = 0.3, k3: float = 0.5,
            nl_scale: float = 1.0) -> AnalyticBackend:
    """One DOF: ``m ü + k u + k2 u² + k3 u³ = 0``.

    Its backbone is known in closed form to first order (Nayfeh and Mook): with
    ``ω0² = k/m``, ``α2 = k2/m``, ``α3 = k3/m`` and ``a`` the amplitude of the fundamental,
    ``ω = ω0 + (9 α3 ω0² − 10 α2²) / (24 ω0³) · a²``. Used to check the sign and
    multiplicity conventions of the whole chain against physics.
    """
    K = np.array([[float(k)]])
    M = np.array([[float(m)]])
    G = polarize_quadratic(1, [(0, 0, 0, k2 * nl_scale)])
    H = polarize_cubic(1, [(0, 0, 0, 0, k3 * nl_scale)])
    return AnalyticBackend(K, M, G, H, name="duffing")


def spring_chain(n: int = 20, k: float = 1.0, kg: float = 1.0, k2: float = 0.3, k3: float = 0.5,
                 m: float = 1.0, nl_scale: float = 1.0) -> AnalyticBackend:
    """Fixed-fixed chain of ``n`` masses. Neighbour springs have ``k e + k2 e² + k3 e³``;
    linear ground springs ``kg`` make the dispersion non-linear (no near 1:3 resonance)."""
    supports, coefs = [], []
    for s in range(n + 1):  # spring s joins DOF s-1 and s; -1 and n are walls
        idx, b = [], []
        if s - 1 >= 0:
            idx.append(s - 1); b.append(-1.0)
        if s < n:
            idx.append(s); b.append(1.0)
        supports.append(idx); coefs.append(b)
    S = len(supports)
    rows = np.concatenate([np.full(len(i), s) for s, i in enumerate(supports)])
    cols = np.concatenate([np.asarray(i) for i in supports])
    vals = np.concatenate([np.asarray(b) for b in coefs])
    B = sp.csr_matrix((vals, (rows, cols)), shape=(S, n))
    K = (B.T @ sp.diags(np.full(S, k)) @ B + kg * sp.identity(n)).tocsr()
    M = m * sp.identity(n, format="csr")
    G, H = springs_tensors(n, supports, coefs, np.full(S, k2 * nl_scale), np.full(S, k3 * nl_scale))
    return AnalyticBackend(K, M, G, H, name=f"spring_chain_{n}")


def random_sparse_model(n: int = 40, seed: int = 0, density: float = 0.1,
                        nl_scale: float = 1.0) -> AnalyticBackend:
    """Random SPD model with ``max(n, density*n*(n-1)/2)`` generalised springs on 2 to 3 DOFs.

    K = Bᵀ diag(k1) B + diag(1..2) (SPD), M = diag(0.5..1.5). Potential forces, so G and H
    are fully symmetric and K_tg(U) is symmetric.
    """
    rng = np.random.default_rng(seed)
    S = max(n, int(round(density * n * (n - 1) / 2)))
    supports, coefs = [], []
    for _ in range(S):
        m = int(rng.integers(2, 4)) if n >= 3 else min(n, 2)
        idx = np.sort(rng.choice(n, size=m, replace=False))
        supports.append(idx)
        coefs.append(rng.normal(size=m))
    k1 = rng.uniform(0.5, 1.5, size=S)
    k2 = rng.normal(0.0, 0.5, size=S) * nl_scale
    k3 = rng.uniform(0.1, 1.0, size=S) * nl_scale
    rows = np.concatenate([np.full(len(i), s) for s, i in enumerate(supports)])
    cols = np.concatenate(supports)
    vals = np.concatenate(coefs)
    B = sp.csr_matrix((vals, (rows, cols)), shape=(S, n))
    K = (B.T @ sp.diags(k1) @ B + sp.diags(rng.uniform(1.0, 2.0, size=n))).tocsr()
    M = sp.diags(rng.uniform(0.5, 1.5, size=n)).tocsr()
    G, H = springs_tensors(n, supports, coefs, k2, k3)
    return AnalyticBackend(K, M, G, H, name=f"random_sparse_{n}_{seed}")


def from_config(acfg) -> AnalyticBackend:
    """Build from :class:`nistep.config.AnalyticCfg`."""
    if acfg.model == "two_dof":
        return two_dof_example()
    if acfg.model == "duffing":
        return duffing(nl_scale=acfg.nl_scale)
    if acfg.model == "spring_chain":
        return spring_chain(acfg.n, nl_scale=acfg.nl_scale)
    if acfg.model == "random_sparse":
        return random_sparse_model(acfg.n, seed=acfg.seed, density=acfg.density, nl_scale=acfg.nl_scale)
    raise ValueError(f"unknown analytic model {acfg.model!r}")
