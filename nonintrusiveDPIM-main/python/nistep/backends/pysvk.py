"""PySVK backend: a numpy Total-Lagrangian Saint-Venant-Kirchhoff hex8 element.

It reproduces the Abaqus UEL ``test/uel_nlmech_pk1.for`` on the real mesh, so the whole
pipeline can be tested without Abaqus:

* trilinear hex8, Abaqus C3D8 node order, 2 x 2 x 2 Gauss points;
* ``F = I + grad u``,  ``E = (FᵀF − I)/2``,  ``S = λ tr(E) I + 2μ E``,  ``P = F S``;
* ``f_int = ∫ Bᵀ P dV``,  ``K_tg = ∫ Bᵀ (∂P/∂F) B dV``;
* consistent mass ``∫ ρ NᵀN dV`` (as ``uel_nlmech_pk1_mass_matrix.for``).

The internal force is exactly cubic in the displacement. Writing ``H = grad u``,
``ε(H) = (H + Hᵀ)/2``, ``q̂(Ha,Hb) = (HaᵀHb + HbᵀHa)/4`` and ``C:X = λ tr(X) I + 2μ X``::

    P(H)         = C:ε(H) + [H C:ε(H) + C:q̂(H,H)] + H C:q̂(H,H)
    P2(Ha,Hb)    = [Ha C:ε(Hb) + Hb C:ε(Ha)]/2 + C:q̂(Ha,Hb)                 (polarised quadratic)
    P3(Ha,Hb,Hc) = [Ha C:q̂(Hb,Hc) + Hb C:q̂(Ha,Hc) + Hc C:q̂(Ha,Hb)]/3       (polarised cubic)

These give ``exact_quad`` and ``exact_cubic``, an FE truth independent of the EED-StEP.
They take complex vectors and never conjugate (transposes only).
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.sparse as sp

from ..assembly import Assembler
from ..dofmap import NDOF
from ..inp_io import InpModel, UelProperties, read_inp
from .base import BaseBackend

# Abaqus C3D8 natural coordinates of the 8 nodes
NODE_XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
_G = 1.0 / np.sqrt(3.0)
GAUSS_XI = np.array([[i, j, k] for k in (-_G, _G) for j in (-_G, _G) for i in (-_G, _G)])
GAUSS_W = np.ones(8)
_I3 = np.eye(3)


def shape_functions(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``N[q, a]`` and ``dN[q, a, k] = ∂N_a/∂ξ_k`` at the points ``xi[q, :]``."""
    t = 1.0 + NODE_XI[None, :, :] * xi[:, None, :]  # (q, a, 3)
    N = t.prod(axis=2) / 8.0
    dN = np.empty(t.shape)
    for k in range(3):
        others = [m for m in range(3) if m != k]
        dN[:, :, k] = NODE_XI[None, :, k] * t[:, :, others[0]] * t[:, :, others[1]] / 8.0
    return N, dN


class PySVKBackend(BaseBackend):
    kind = "pysvk"

    def __init__(self, model: InpModel, clamped_nset: str = "Set-2",
                 props: UelProperties | None = None, name: str | None = None):
        super().__init__()
        props = props or model.props
        if props is None:
            raise ValueError("PySVK: the deck has no *UEL PROPERTY and no props were given")
        if model.conn.shape[1] != 8:
            raise ValueError("PySVK: only 8-node hexahedra are supported")
        self.model = model
        self.props = props
        self.clamped_nset = clamped_nset
        self.name = name or (model.path.stem if model.path is not None else "pysvk")
        self.dofmap = model.dofmap(clamped_nset)
        self.asm = Assembler(self.dofmap, model.conn)
        self.n_dof = self.dofmap.n_dof
        self.lam, self.mu, self.rho = props.lam, props.mu, props.rho

        X = model.sorted_coords()[self.dofmap.rank_all(model.conn)]  # (e, a, 3)
        self.N, dN = shape_functions(GAUSS_XI)  # (q, a), (q, a, k)
        J = np.einsum("eai,qak->eqik", X, dN)  # dX_i/dξ_k
        detJ = np.linalg.det(J)
        if np.any(detJ <= 0):
            e, q = np.argwhere(detJ <= 0)[0]
            raise ValueError(f"PySVK: non-positive Jacobian in element {int(model.elem_labels[e])} "
                             f"(Gauss point {int(q) + 1}); check the node ordering")
        self.B = np.einsum("qak,eqkj->eqaj", dN, np.linalg.inv(J))  # ∂N_a/∂X_j
        self.wdet = GAUSS_W[None, :] * detJ  # (e, q)
        self.volume = float(self.wdet.sum())
        self._fp: str | None = None
        self._M: sp.csr_matrix | None = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg) -> "PySVKBackend":
        return cls(read_inp(cfg.model.inp), cfg.model.clamped_nset)

    def fingerprint(self) -> str:
        if self._fp is None:
            h = hashlib.sha256()
            p = self.props
            h.update(f"{self.kind}|{self.n_dof}|{p.E!r}|{p.nu!r}|{p.rho!r}|".encode())
            for arr in (self.model.sorted_coords(), self.model.conn, self.dofmap.nodes,
                        self.dofmap.clamped_nodes):
                h.update(np.ascontiguousarray(arr).tobytes())
            self._fp = h.hexdigest()
        return self._fp

    # ------------------------------------------------------------------- kinematics
    def _grad(self, U) -> np.ndarray:
        """Displacement gradient ``H[e, q, i, j] = ∂u_i/∂X_j`` (dtype of ``U``)."""
        return np.einsum("eai,eqaj->eqij", self.asm.gather(U), self.B)

    def _C(self, X: np.ndarray) -> np.ndarray:
        """``C:X = λ tr(X) I + 2μ X`` for symmetric ``X[e, q, :, :]``."""
        tr = np.einsum("eqii->eq", X)
        return self.lam * tr[..., None, None] * _I3 + 2.0 * self.mu * X

    @staticmethod
    def _eps(H: np.ndarray) -> np.ndarray:
        return 0.5 * (H + np.swapaxes(H, -1, -2))

    @staticmethod
    def _qhat(Ha: np.ndarray, Hb: np.ndarray) -> np.ndarray:
        ab = np.einsum("eqki,eqkj->eqij", Ha, Hb)  # HaᵀHb (transpose, no conjugation)
        return 0.25 * (ab + np.swapaxes(ab, -1, -2))

    def _force(self, P: np.ndarray) -> np.ndarray:
        """``∫ Bᵀ P dV`` assembled on the free DOFs."""
        fe = np.einsum("eq,eqij,eqaj->eai", self.wdet, P, self.B)
        return self.asm.assemble_vector(fe.reshape(self.asm.n_elem, self.asm.nd))

    # ------------------------------------------------------------- exact polarised forms
    def exact_quad(self, a, b) -> np.ndarray:
        Ha, Hb = self._grad(np.asarray(a)), self._grad(np.asarray(b))
        P2 = 0.5 * (Ha @ self._C(self._eps(Hb)) + Hb @ self._C(self._eps(Ha))) \
            + self._C(self._qhat(Ha, Hb))
        return self._force(P2)

    def exact_cubic(self, a, b, c) -> np.ndarray:
        Ha, Hb, Hc = (self._grad(np.asarray(x)) for x in (a, b, c))
        P3 = (Ha @ self._C(self._qhat(Hb, Hc)) + Hb @ self._C(self._qhat(Ha, Hc))
              + Hc @ self._C(self._qhat(Ha, Hb))) / 3.0
        return self._force(P3)

    def internal_force(self, U) -> np.ndarray:
        """``F_int(U) = ∫ Bᵀ P(F) dV`` on the free DOFs."""
        F = _I3 + self._grad(np.asarray(U))
        E = 0.5 * (np.einsum("eqki,eqkj->eqij", F, F) - _I3)
        return self._force(F @ self._C(E))

    def min_det_F(self, U) -> float:
        """Smallest ``det F`` over all Gauss points (element inversion check)."""
        return float(np.linalg.det(_I3 + self._grad(self.check_U(U))).min())

    # ------------------------------------------------------------------- matrices
    def element_tangents(self, U) -> np.ndarray:
        """Element tangent matrices ``(n_elem, 24, 24)`` at the free-DOF displacement ``U``."""
        H = self._grad(np.asarray(U, dtype=np.float64))
        F = _I3 + H
        E = 0.5 * (np.einsum("eqki,eqkj->eqij", F, F) - _I3)
        S = self._C(E)
        FFt = np.einsum("eqim,eqkm->eqik", F, F)
        # A_ijkl = ∂P_ij/∂F_kl = δ_ik S_lj + λ F_ij F_kl + μ (F_il F_kj + (FFᵀ)_ik δ_jl)
        A = (np.einsum("ik,eqlj->eqijkl", _I3, S)
             + self.lam * np.einsum("eqij,eqkl->eqijkl", F, F)
             + self.mu * (np.einsum("eqil,eqkj->eqijkl", F, F)
                          + np.einsum("eqik,jl->eqijkl", FFt, _I3)))
        T = np.einsum("eqaj,eqijkl->eqaikl", self.B, A)
        T *= self.wdet[:, :, None, None, None, None]
        Ke = np.einsum("eqaikl,eqbl->eaibk", T, self.B)
        return Ke.reshape(self.asm.n_elem, self.asm.nd, self.asm.nd)

    def element_masses(self) -> np.ndarray:
        """Consistent element mass matrices ``(n_elem, 24, 24)``."""
        m = self.rho * np.einsum("eq,qa,qb->eab", self.wdet, self.N, self.N)
        Me = np.einsum("eab,ik->eaibk", m, _I3)
        return Me.reshape(self.asm.n_elem, self.asm.nd, self.asm.nd)

    def stiffness0(self) -> sp.csr_matrix:
        return self.asm.assemble(self.element_tangents(np.zeros(self.n_dof)))

    def mass(self) -> sp.csr_matrix:
        if self._M is None:
            self._M = self.asm.assemble(self.element_masses())
        return self._M.copy()

    def _tangent_batch(self, U_list):
        return [self.asm.assemble(self.element_tangents(U)) for U in U_list]


def from_config(cfg) -> PySVKBackend:
    return PySVKBackend.from_config(cfg)


__all__ = ["PySVKBackend", "from_config", "shape_functions", "NODE_XI", "GAUSS_XI", "NDOF"]
