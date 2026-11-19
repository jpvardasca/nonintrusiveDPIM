"""Replay backend: serve tangent matrices recorded by an earlier run, and nothing else.

A run with a disk cache (``[cache] dir``) leaves, under ``<cache dir>/<fp16>/``, every
tangent matrix it computed plus ``backend_info.npz`` (fingerprint, size and mass matrix of
the model). Copy that folder from the Abaqus host to any machine and set::

    [backend]
    kind = "replay"
    [cache]
    dir = "<the copied cache dir>"

The engine then finds every recorded matrix in its store and never calls the backend. A
request for a state that was not recorded raises :class:`ReplayMiss`. Use it to debug or
post-process a production run without Abaqus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .base import BaseBackend

INFO_FILE = "backend_info.npz"


class ReplayMiss(RuntimeError):
    """The requested tangent matrix was not recorded."""

    def __init__(self, key: str, max_abs_u: float):
        self.key = key
        self.max_abs_u = max_abs_u
        super().__init__(f"replay: no recorded tangent matrix for job key {key} "
                         f"(max|U| = {max_abs_u:.6g}); it was not computed in the recorded run")


def save_backend_info(root: Path, backend, M: sp.csr_matrix) -> Path:
    """Write ``backend_info.npz`` (once) so the cache folder can be replayed later."""
    path = Path(root) / INFO_FILE
    if path.is_file():
        return path
    M = sp.csr_matrix(M, dtype=np.float64)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, fingerprint=np.array(backend.fingerprint()), kind=np.array(backend.kind),
                 n_dof=np.array(int(backend.n_dof)), M_data=M.data, M_indices=M.indices,
                 M_indptr=M.indptr)
    tmp.replace(path)
    return path


class ReplayBackend(BaseBackend):
    kind = "replay"

    def __init__(self, cache_dir, fingerprint: str | None = None):
        super().__init__()
        cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"replay: cache directory not found: {cache_dir}")
        found = sorted(d for d in cache_dir.iterdir() if (d / INFO_FILE).is_file())
        if fingerprint:
            found = [d for d in found if d.name == fingerprint[:16]]
        if len(found) != 1:
            names = [d.name for d in found]
            raise ValueError(f"replay: expected exactly one recorded model in {cache_dir}, found "
                             f"{names or 'none'}; set [backend] replay_fingerprint to choose")
        self.root = found[0]
        with np.load(self.root / INFO_FILE) as z:
            self._fp = str(z["fingerprint"])
            self.recorded_kind = str(z["kind"])
            self.n_dof = int(z["n_dof"])
            self._M = sp.csr_matrix((np.array(z["M_data"]), np.array(z["M_indices"]),
                                     np.array(z["M_indptr"])), shape=(self.n_dof, self.n_dof))

    def fingerprint(self) -> str:
        return self._fp  # the recorded model's fingerprint, so the job keys match

    def _key(self, U: np.ndarray) -> str:
        h = hashlib.sha256(self._fp.encode())
        h.update(np.ascontiguousarray(U, dtype=np.float64).tobytes())
        return h.hexdigest()

    def stiffness0(self) -> sp.csr_matrix:
        raise ReplayMiss(self._key(np.zeros(self.n_dof)), 0.0)

    def mass(self) -> sp.csr_matrix:
        return self._M.copy()

    def _tangent_batch(self, U_list):
        U = U_list[0]
        raise ReplayMiss(self._key(U), float(np.max(np.abs(U))) if U.size else 0.0)
