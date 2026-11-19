"""KtgStore: cache of tangent stiffness matrices keyed by the prescribed displacement.

Key: ``sha256(fingerprint ‖ U.tobytes())`` (CONVENTIONS 4.5), exact and reproducible.

* In memory: an LRU of CSR matrices sharing one read-only ``indptr``/``indices`` pattern.
  Without a disk cache nothing is evicted (an evicted matrix would mean re-running a job).
* On disk (optional): ``<cache_dir>/<fp16>/``, where ``fp16`` is the first 16 hex
  characters of the fingerprint, containing
    - ``pattern.npz``: the shared CSR pattern;
    - ``jobs/<key>.npz``: ``data`` and ``U``, plus ``indptr``/``indices`` only if the
      matrix's pattern differs from the shared one;
    - ``manifest.jsonl``: one line per stored job (key, time, nnz, max|U|).
  All writes are atomic (``*.tmp`` then ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


class KtgStore:
    def __init__(self, fingerprint: str, cache_dir: str | Path | None = None, lru_items: int = 64):
        if lru_items < 1:
            raise ValueError("lru_items must be >= 1")
        self.fingerprint = str(fingerprint)
        self._fp_bytes = self.fingerprint.encode()
        self.lru_items = int(lru_items)
        self._lru: OrderedDict[str, sp.csr_matrix] = OrderedDict()
        self._pattern: tuple[np.ndarray, np.ndarray, tuple[int, int]] | None = None
        self.root: Path | None = None
        self._disk_keys: set[str] = set()
        self.n_mem_hits = 0
        self.n_disk_hits = 0
        self.n_misses = 0
        self.n_puts = 0
        self.n_pattern_mismatch = 0
        if cache_dir is not None:
            self.root = Path(cache_dir) / self.fingerprint[:16]
            (self.root / "jobs").mkdir(parents=True, exist_ok=True)
            self._disk_keys = {p.stem for p in (self.root / "jobs").glob("*.npz")}
            pat = self.root / "pattern.npz"
            if pat.is_file():
                with np.load(pat) as z:
                    self._set_pattern(z["indptr"], z["indices"], tuple(int(x) for x in z["shape"]))

    # ----------------------------------------------------------------------------- keys
    def key(self, U: np.ndarray) -> str:
        U = np.ascontiguousarray(U, dtype=np.float64)
        h = hashlib.sha256(self._fp_bytes)
        h.update(U.tobytes())
        return h.hexdigest()

    def contains(self, key: str) -> bool:
        return key in self._lru or key in self._disk_keys

    def __len__(self) -> int:
        return len(set(self._lru) | self._disk_keys)

    # ---------------------------------------------------------------------------- pattern
    def _set_pattern(self, indptr, indices, shape) -> None:
        indptr = np.array(indptr, copy=True)
        indices = np.array(indices, copy=True)
        indptr.flags.writeable = False
        indices.flags.writeable = False
        self._pattern = (indptr, indices, (int(shape[0]), int(shape[1])))

    def _same_pattern(self, K: sp.csr_matrix) -> bool:
        if self._pattern is None:
            return False
        indptr, indices, shape = self._pattern
        return (K.shape == shape and K.indptr.shape == indptr.shape and K.indices.shape == indices.shape
                and np.array_equal(K.indptr, indptr) and np.array_equal(K.indices, indices))

    def _wrap(self, data: np.ndarray) -> sp.csr_matrix:
        indptr, indices, shape = self._pattern
        M = sp.csr_matrix((data, indices, indptr), shape=shape, copy=False)
        M.has_sorted_indices = True
        return M

    # ------------------------------------------------------------------------------ get/put
    def get(self, key: str, U: np.ndarray | None = None) -> sp.csr_matrix | None:
        K = self._lru.get(key)
        if K is not None:
            self._lru.move_to_end(key)
            self.n_mem_hits += 1
            return K
        if key in self._disk_keys:
            K = self._load(key, U)
            if K is not None:
                self.n_disk_hits += 1
                self._remember(key, K)
                return K
        self.n_misses += 1
        return None

    def put(self, U: np.ndarray, K: sp.csr_matrix, key: str | None = None) -> str:
        U = np.ascontiguousarray(U, dtype=np.float64)
        key = key or self.key(U)
        K = sp.csr_matrix(K, dtype=np.float64)
        if not K.has_canonical_format:
            K = K.copy()
            K.sum_duplicates()
        if self._pattern is None:
            self._set_pattern(K.indptr, K.indices, K.shape)
            if self.root is not None:
                indptr, indices, shape = self._pattern
                _atomic_savez(self.root / "pattern.npz", indptr=indptr, indices=indices,
                              shape=np.array(shape, dtype=np.int64))
        shared = self._same_pattern(K)
        if shared:
            K = self._wrap(np.array(K.data, dtype=np.float64, copy=True))
        else:
            self.n_pattern_mismatch += 1
            log.warning("KtgStore: matrix pattern differs from the shared pattern; storing full matrix")
        if self.root is not None:
            arrays = {"data": K.data, "U": U}
            if not shared:
                arrays.update(indptr=K.indptr, indices=K.indices, shape=np.array(K.shape, dtype=np.int64))
            _atomic_savez(self.root / "jobs" / f"{key}.npz", **arrays)
            self._disk_keys.add(key)
            with open(self.root / "manifest.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "time": time.time(), "nnz": int(K.nnz),
                                     "max_abs_U": float(np.max(np.abs(U))) if U.size else 0.0}) + "\n")
        self.n_puts += 1
        self._remember(key, K)
        return key

    def _remember(self, key: str, K: sp.csr_matrix) -> None:
        self._lru[key] = K
        self._lru.move_to_end(key)
        if self.root is not None:  # only evict when the disk holds a copy
            while len(self._lru) > self.lru_items:
                self._lru.popitem(last=False)

    def _load(self, key: str, U: np.ndarray | None) -> sp.csr_matrix | None:
        path = self.root / "jobs" / f"{key}.npz"
        try:
            with np.load(path) as z:
                data = np.array(z["data"], dtype=np.float64)
                U_stored = np.array(z["U"], dtype=np.float64)
                full = "indptr" in z.files
                if full:
                    K = sp.csr_matrix((data, np.array(z["indices"]), np.array(z["indptr"])),
                                      shape=tuple(int(x) for x in z["shape"]))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("KtgStore: unreadable cache file %s (%s); treating as miss", path, exc)
            self._disk_keys.discard(key)
            return None
        if U is not None and not np.array_equal(U_stored, np.ascontiguousarray(U, dtype=np.float64)):
            log.warning("KtgStore: stored U differs for key %s; treating as miss", key)
            return None
        if full:
            return K
        if self._pattern is None:
            log.warning("KtgStore: missing pattern.npz for %s; treating as miss", key)
            return None
        return self._wrap(data)

    def manifest_keys(self) -> list[str]:
        """Keys in ``manifest.jsonl`` in write order (duplicates mean a job was re-run)."""
        if self.root is None or not (self.root / "manifest.jsonl").is_file():
            return []
        keys = []
        with open(self.root / "manifest.jsonl", encoding="utf-8") as fh:
            for line in fh:
                try:
                    keys.append(json.loads(line)["key"])
                except (ValueError, KeyError, TypeError):  # blank, or torn by a killed process
                    continue
        return keys

    def stats(self) -> dict:
        nnz = int(self._pattern[1].size) if self._pattern is not None else 0
        return {
            "store_items_memory": len(self._lru),
            "store_items_disk": len(self._disk_keys),
            "store_mem_hits": self.n_mem_hits,
            "store_disk_hits": self.n_disk_hits,
            "store_misses": self.n_misses,
            "store_puts": self.n_puts,
            "store_pattern_mismatch": self.n_pattern_mismatch,
            "store_nnz": nnz,
            "store_ram_estimate_MB": len(self._lru) * nnz * 8 / 2**20,
            "store_MB_per_matrix": nnz * 8 / 2**20,
            # with a disk cache at most lru_items matrices stay in memory; without one, all do
            "store_ram_limit_MB": (self.lru_items * nnz * 8 / 2**20) if self.root is not None else None,
        }

    def memory_report(self) -> str:
        """One line on the memory used by cached tangent matrices (PLAN.md P9.4)."""
        s = self.stats()
        per = s["store_MB_per_matrix"]
        if self.root is not None:
            limit = (f"at most {self.lru_items} kept in memory ({s['store_ram_limit_MB']:.1f} MB), "
                     f"the rest on disk ({s['store_items_disk']} stored)")
        else:
            limit = "no disk cache, so every matrix stays in memory"
        return (f"tangent matrices: {s['store_nnz']} stored entries = {per:.3f} MB each; "
                f"{s['store_items_memory']} in memory now ({s['store_ram_estimate_MB']:.1f} MB); {limit}")
