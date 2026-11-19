"""Canonical real directions and vector registration ids (CONVENTIONS 4.4-4.5).

* :func:`canonicalize` writes a real part as ``p = alpha * d``, with ``max|d| = 1`` and the
  sign fixed by a seeded random vector ``w`` (``w·d > 0``).
* :class:`DirectionStore` holds the canonical directions. A new ``d`` matches a stored ``d_k``
  if ``‖d − s·d_k‖_∞ ≤ match_tol`` for ``s = +1`` or ``s = −1``. The ``−1`` case only guards
  against a sign flip when ``w·d ≈ 0``. Job displacements are always built from the
  **stored** direction, so job keys are reproducible.
* :class:`VectorRegistry` gives each complex argument an id at first sight. A vector and its
  complex conjugate share the same id: they span the same real directions, so MORFE's
  conjugate coefficients reuse the probes of the original.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def make_sign_vector(n: int, seed: int) -> np.ndarray:
    """The fixed random vector ``w`` used to fix the sign of canonical directions."""
    return np.random.default_rng(seed).standard_normal(n)


def canonicalize(p: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Return ``(alpha, d)`` with ``p = alpha * d``, ``max|d| = 1`` and ``w·d >= 0``; ``None`` if p == 0."""
    p = np.asarray(p, dtype=np.float64)
    m = float(np.max(np.abs(p))) if p.size else 0.0
    if m == 0.0:
        return None
    d = p / m
    s = 1.0 if float(np.dot(w, d)) >= 0.0 else -1.0
    return s * m, s * d


class DirectionStore:
    """Registry of canonical real directions (id = insertion index)."""

    def __init__(self, n: int, match_tol: float = 1e-9):
        self.n = int(n)
        self.match_tol = float(match_tol)
        self._dirs: list[np.ndarray] = []
        self._norms = np.zeros(0)
        self._saved_count = 0

    def __len__(self) -> int:
        return len(self._dirs)

    def get(self, did: int) -> np.ndarray:
        return self._dirs[did]

    def match(self, d: np.ndarray) -> tuple[int, float] | None:
        """Return ``(id, s)`` with ``d ≈ s * d_id`` (s = ±1), or ``None``."""
        if not self._dirs:
            return None
        nrm = float(np.linalg.norm(d))
        cand = np.nonzero(np.abs(self._norms - nrm) <= self.match_tol * np.sqrt(self.n) + 1e-300)[0]
        for k in cand:
            dk = self._dirs[k]
            if np.max(np.abs(d - dk)) <= self.match_tol:
                return int(k), 1.0
            if np.max(np.abs(d + dk)) <= self.match_tol:
                return int(k), -1.0
        return None

    def add(self, d: np.ndarray) -> int:
        d = np.array(d, dtype=np.float64, copy=True)
        if d.shape != (self.n,):
            raise ValueError(f"direction must have shape ({self.n},)")
        d.flags.writeable = False
        self._dirs.append(d)
        self._norms = np.append(self._norms, np.linalg.norm(d))
        return len(self._dirs) - 1

    def match_or_add(self, d: np.ndarray) -> tuple[int, float]:
        hit = self.match(d)
        if hit is not None:
            return hit
        return self.add(d), 1.0

    # -- persistence (npz, atomic) ---------------------------------------------------
    @property
    def dirty(self) -> bool:
        return len(self._dirs) != self._saved_count

    def save(self, path: str | Path, seed: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        D = np.stack(self._dirs) if self._dirs else np.zeros((0, self.n))
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, D=D, seed=np.int64(seed), n=np.int64(self.n))
        os.replace(tmp, path)
        self._saved_count = len(self._dirs)

    def load(self, path: str | Path, seed: int) -> bool:
        """Load stored directions (only into an empty store). Returns False on mismatch."""
        path = Path(path)
        if not path.is_file() or self._dirs:
            return False
        with np.load(path) as z:
            if int(z["seed"]) != int(seed) or int(z["n"]) != self.n:
                return False
            D = np.array(z["D"], dtype=np.float64)
        for d in D:
            self.add(d)
        self._saved_count = len(self._dirs)
        return True


class VectorRegistry:
    """Ids for complex argument vectors, in order of first sight (conjugates share an id)."""

    def __init__(self, match_tol: float = 1e-9):
        self.match_tol = float(match_tol)
        self._vecs: list[np.ndarray] = []
        self._scale = np.zeros(0)

    def __len__(self) -> int:
        return len(self._vecs)

    def id_of(self, a: np.ndarray) -> int:
        a = np.asarray(a, dtype=np.complex128)
        s = float(np.max(np.abs(a))) if a.size else 0.0
        if self._vecs:
            tol = self.match_tol * np.maximum(self._scale, s)
            cand = np.nonzero(np.abs(self._scale - s) <= tol)[0]
            for k in cand:
                ak = self._vecs[k]
                t = tol[k]
                if np.max(np.abs(a - ak)) <= t or np.max(np.abs(a - np.conj(ak))) <= t:
                    return int(k)
        self._vecs.append(np.array(a, copy=True))
        self._scale = np.append(self._scale, s)
        return len(self._vecs) - 1
