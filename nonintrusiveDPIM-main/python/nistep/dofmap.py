"""DofMap: the one DOF ordering used everywhere (CONVENTIONS 4.7).

* Nodes are sorted by **ascending label**. Each node has DOFs 1, 2, 3.
* "Total" index (all nodes):   ``3 * rank_all(node)  + (dof - 1)``.
* "Free" index (clamped nodes removed, duplicates in the clamped set ignored):
  ``3 * rank_free(node) + (dof - 1)``.

All indices are 0-based; ``dof`` is 1, 2 or 3. The same map is used for the boundary
conditions written to the job deck, the assembly of element matrices, the monitor DOF and
every vector exchanged with Julia.
"""

from __future__ import annotations

import numpy as np

NDOF = 3  # DOFs per node


class DofMap:
    def __init__(self, node_labels, clamped_labels):
        nodes = np.asarray(node_labels, dtype=np.int64).ravel()
        if nodes.size == 0:
            raise ValueError("DofMap: no nodes")
        self.nodes = np.unique(nodes)  # sorted
        if self.nodes.size != nodes.size:
            raise ValueError("DofMap: duplicate node labels")
        clamped = np.unique(np.asarray(clamped_labels, dtype=np.int64).ravel())
        missing = np.setdiff1d(clamped, self.nodes)
        if missing.size:
            raise ValueError(f"DofMap: clamped nodes not in the mesh: {missing[:10].tolist()}")
        self.clamped_nodes = clamped
        is_free = ~np.isin(self.nodes, clamped)
        self.free_nodes = self.nodes[is_free]
        if self.free_nodes.size == 0:
            raise ValueError("DofMap: every node is clamped")
        self.n_nodes = int(self.nodes.size)
        self.n_free_nodes = int(self.free_nodes.size)
        self.n_total = NDOF * self.n_nodes
        self.n_dof = NDOF * self.n_free_nodes
        # rank among all nodes -> rank among free nodes (-1 if clamped)
        self._free_rank = np.full(self.n_nodes, -1, dtype=np.int64)
        self._free_rank[is_free] = np.arange(self.n_free_nodes)
        # total DOF index of every free DOF, and its inverse
        all_rank = np.nonzero(is_free)[0]
        self.total_of_free = (NDOF * all_rank[:, None] + np.arange(NDOF)[None, :]).ravel()
        self.free_of_total = np.full(self.n_total, -1, dtype=np.int64)
        self.free_of_total[self.total_of_free] = np.arange(self.n_dof)

    # ------------------------------------------------------------------ node ranks
    def rank_all(self, labels) -> np.ndarray:
        """0-based position of each label among all nodes; raises on unknown labels."""
        labels = np.asarray(labels, dtype=np.int64)
        r = np.searchsorted(self.nodes, labels)
        r = np.minimum(r, self.n_nodes - 1)
        if not np.array_equal(self.nodes[r], labels):
            bad = np.unique(labels[self.nodes[r] != labels])
            raise KeyError(f"unknown node labels: {bad[:10].tolist()}")
        return r

    def rank_free(self, labels) -> np.ndarray:
        """0-based position among the free nodes, or -1 for clamped nodes."""
        return self._free_rank[self.rank_all(labels)]

    # ------------------------------------------------------------------ DOF indices
    @staticmethod
    def _check_dof(dof) -> None:
        if np.any((np.asarray(dof) < 1) | (np.asarray(dof) > NDOF)):
            raise ValueError("dof must be 1, 2 or 3")

    def total_index(self, node: int, dof: int) -> int:
        self._check_dof(dof)
        return int(NDOF * self.rank_all(node) + (dof - 1))

    def free_index(self, node: int, dof: int) -> int:
        """Free-DOF index of ``(node, dof)``; raises ``KeyError`` for a clamped node."""
        self._check_dof(dof)
        r = int(self.rank_free(node))
        if r < 0:
            raise KeyError(f"node {node} is clamped")
        return NDOF * r + (dof - 1)

    def free_indices(self, labels) -> np.ndarray:
        """Array ``labels.shape + (3,)`` of free-DOF indices (-1 where the node is clamped)."""
        r = self.rank_free(labels)
        idx = NDOF * r[..., None] + np.arange(NDOF)
        return np.where(r[..., None] >= 0, idx, -1)

    def node_dof(self, free_index: int) -> tuple[int, int]:
        """``(node label, dof)`` of a free-DOF index."""
        if not 0 <= free_index < self.n_dof:
            raise IndexError("free index out of range")
        return int(self.free_nodes[free_index // NDOF]), int(free_index % NDOF) + 1

    # ----------------------------------------------------------------- vector maps
    def expand(self, U_free) -> np.ndarray:
        """Free-DOF vector -> all-DOF vector (zeros on clamped DOFs). Keeps the dtype."""
        U_free = np.asarray(U_free)
        if U_free.shape != (self.n_dof,):
            raise ValueError(f"expected shape ({self.n_dof},), got {U_free.shape}")
        out = np.zeros(self.n_total, dtype=U_free.dtype)
        out[self.total_of_free] = U_free
        return out

    def restrict(self, U_total) -> np.ndarray:
        """All-DOF vector -> free-DOF vector."""
        U_total = np.asarray(U_total)
        if U_total.shape != (self.n_total,):
            raise ValueError(f"expected shape ({self.n_total},), got {U_total.shape}")
        return U_total[self.total_of_free]

    def restrict_matrix(self, A):
        """All-DOF square sparse matrix -> free-DOF block (CSR)."""
        import scipy.sparse as sp

        A = sp.csr_matrix(A)
        if A.shape != (self.n_total, self.n_total):
            raise ValueError(f"expected shape ({self.n_total}, {self.n_total}), got {A.shape}")
        return A[self.total_of_free][:, self.total_of_free].tocsr()

    def __repr__(self) -> str:
        return (f"DofMap(nodes={self.n_nodes}, clamped={self.clamped_nodes.size}, "
                f"free_dofs={self.n_dof})")
