"""Assembly of element matrices into the global free-DOF matrix with a fixed pattern.

The :class:`Assembler` is built once from the DofMap and the element connectivity. Every
``assemble(Ke)`` then returns a CSR matrix with exactly the same pattern (explicit zeros
kept), with the clamped DOFs already eliminated. Local element DOFs are node-major:
local index ``3 * a + (dof - 1)`` for the ``a``-th element node (the Abaqus UEL order).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._sparse import FixedPattern
from .dofmap import NDOF, DofMap


class Assembler:
    def __init__(self, dofmap: DofMap, conn):
        conn = np.asarray(conn, dtype=np.int64)
        if conn.ndim != 2:
            raise ValueError("conn must be an (n_elem, nodes_per_elem) array of node labels")
        self.dofmap = dofmap
        self.conn = conn
        self.n_elem, self.n_en = conn.shape
        self.nd = self.n_en * NDOF  # DOFs per element
        self.n_dof = dofmap.n_dof
        # free-DOF index of every local element DOF (-1 on clamped nodes)
        self.elem_dofs = dofmap.free_indices(conn).reshape(self.n_elem, self.nd)
        rows = np.repeat(self.elem_dofs[:, :, None], self.nd, axis=2)
        cols = np.repeat(self.elem_dofs[:, None, :], self.nd, axis=1)
        keep = (rows >= 0) & (cols >= 0)
        self._keep = np.nonzero(keep.ravel())[0]
        self.pattern = FixedPattern(rows[keep], cols[keep], (self.n_dof, self.n_dof))
        self._vkeep = np.nonzero(self.elem_dofs.ravel() >= 0)[0]
        self._vidx = self.elem_dofs.ravel()[self._vkeep]

    def assemble(self, Ke) -> sp.csr_matrix:
        """Element matrices ``(n_elem, nd, nd)`` -> global free-DOF CSR matrix."""
        Ke = np.asarray(Ke, dtype=np.float64)
        if Ke.shape != (self.n_elem, self.nd, self.nd):
            raise ValueError(f"Ke must have shape {(self.n_elem, self.nd, self.nd)}, got {Ke.shape}")
        return self.pattern.assemble(Ke.ravel()[self._keep])

    def assemble_vector(self, fe) -> np.ndarray:
        """Element vectors ``(n_elem, nd)`` -> global free-DOF vector (complex-safe)."""
        fe = np.asarray(fe)
        if fe.shape != (self.n_elem, self.nd):
            raise ValueError(f"fe must have shape {(self.n_elem, self.nd)}, got {fe.shape}")
        vals = fe.ravel()[self._vkeep]
        if np.iscomplexobj(vals):
            return (np.bincount(self._vidx, weights=vals.real, minlength=self.n_dof)
                    + 1j * np.bincount(self._vidx, weights=vals.imag, minlength=self.n_dof))
        return np.bincount(self._vidx, weights=vals, minlength=self.n_dof)

    def gather(self, U) -> np.ndarray:
        """Free-DOF vector -> element nodal values ``(n_elem, n_en, 3)``, zero on clamped nodes."""
        U = np.asarray(U)
        if U.shape != (self.n_dof,):
            raise ValueError(f"U must have shape ({self.n_dof},), got {U.shape}")
        Ue = np.zeros(self.n_elem * self.nd, dtype=U.dtype)
        Ue[self._vkeep] = U[self._vidx]
        return Ue.reshape(self.n_elem, self.n_en, NDOF)

    def check_element_order(self, labels, nodes, elem_labels) -> np.ndarray:
        """Permutation that puts parsed element matrices in the connectivity order.

        ``labels``/``nodes`` come from a matrix file, ``elem_labels`` from the deck. Raises
        if an element is missing or its node list differs from the deck's connectivity.
        """
        labels = np.asarray(labels, dtype=np.int64)
        elem_labels = np.asarray(elem_labels, dtype=np.int64)
        if labels.size != elem_labels.size or not np.array_equal(np.sort(labels), np.sort(elem_labels)):
            raise ValueError(f"matrix file has {labels.size} elements, the deck has {elem_labels.size} "
                             "(or the labels differ)")
        order = np.argsort(labels, kind="stable")[np.argsort(np.argsort(elem_labels, kind="stable"))]
        if not np.array_equal(np.asarray(nodes)[order], self.conn):
            raise ValueError("element node lists in the matrix file differ from the deck connectivity")
        return order
