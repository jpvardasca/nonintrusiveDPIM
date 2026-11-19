"""Fixed-pattern sparse assembly helper.

A :class:`FixedPattern` is built once from COO coordinates ``(rows, cols)``. Every later
``assemble(values)`` returns a CSR matrix with **exactly the same** ``indptr``/``indices``:
duplicates are summed, explicit zeros are kept, and indices are sorted. The cache
(``store.KtgStore``) relies on this stable pattern.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class FixedPattern:
    def __init__(self, rows, cols, shape: tuple[int, int]):
        rows = np.asarray(rows, dtype=np.int64).ravel()
        cols = np.asarray(cols, dtype=np.int64).ravel()
        if rows.shape != cols.shape:
            raise ValueError("rows and cols must have the same length")
        n_rows, n_cols = int(shape[0]), int(shape[1])
        if rows.size and (rows.min() < 0 or rows.max() >= n_rows or cols.min() < 0 or cols.max() >= n_cols):
            raise ValueError("COO index out of range")
        key = rows * n_cols + cols
        uniq, inverse = np.unique(key, return_inverse=True)
        self.shape = (n_rows, n_cols)
        self.n_entries = rows.size
        self.nnz = int(uniq.size)
        self.slot = inverse.astype(np.int64).ravel()  # COO entry -> CSR data slot
        u_rows = uniq // n_cols
        idx_dtype = np.int32 if max(n_rows, n_cols, self.nnz) < 2**31 - 1 else np.int64
        self.indices = (uniq % n_cols).astype(idx_dtype)
        counts = np.bincount(u_rows, minlength=n_rows)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(idx_dtype)

    def data(self, values) -> np.ndarray:
        """Sum COO ``values`` (one per entry, real) into the CSR data array."""
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size != self.n_entries:
            raise ValueError(f"expected {self.n_entries} values, got {values.size}")
        return np.bincount(self.slot, weights=values, minlength=self.nnz)

    def assemble(self, values) -> sp.csr_matrix:
        return self.matrix(self.data(values))

    def matrix(self, data: np.ndarray) -> sp.csr_matrix:
        """Wrap a CSR data array (length nnz) with this pattern (indices are copied)."""
        data = np.asarray(data, dtype=np.float64)
        if data.shape != (self.nnz,):
            raise ValueError(f"data must have shape ({self.nnz},)")
        m = sp.csr_matrix((data, self.indices.copy(), self.indptr.copy()), shape=self.shape)
        m.has_sorted_indices = True
        return m


def coo_entries(A) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rows, cols, values)`` of a sparse or dense matrix, explicit zeros kept."""
    if sp.issparse(A):
        C = sp.coo_matrix(A)
        return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float64)
    A = np.asarray(A, dtype=np.float64)
    r, c = np.nonzero(np.ones_like(A, dtype=bool))
    return r.astype(np.int64), c.astype(np.int64), A[r, c]
