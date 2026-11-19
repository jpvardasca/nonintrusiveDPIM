"""EEDEngine: closure evaluator of the symmetric polarised forms Ĝ(a,b) and Ĥ(a,b,c).

Implements CONVENTIONS 4.3-4.5 exactly. Outputs have the **internal-force sign (+)**; the
minus sign for MORFE's right-hand side is applied only in the Julia closure.

EED primitives for a canonical real direction ``d`` (``max|d| = 1``), with ``c = eps·d_ref``:

    K⁺(d) = K_tg(+c d),  K⁻(d) = K_tg(−c d)
    G_op[d]    @ v = (K⁺ − K⁻) v / (4c)                              = Ĝ(d, v)
    H_op[d,d]  @ v = (K⁺ + K⁻ − 2K) v / (6c²)                         = Ĥ(d, d, v)
    H_op[d1,d2]@ v = (K_tg(c d1 + s c d2) − K_tg(c d1) − K_tg(s c d2) + K) v / (6 s c²)
                                                                      = Ĥ(d1, d2, v)
    s = +1 if max|c d1 + c d2| >= c else −1   (legacy rule; d1 is the direction with the lower id)

Complex arguments: ``a = Σ κ_x d_x`` (real/imag parts on canonical directions).

* quad:  ``Ĝ(a,b) = Σ_x κ_x G_op[d_x] @ b``; the probe is the argument with the lower id.
* cubic: ``Ĥ(a,b,c) = Σ_x Σ_y κ_x λ_y H_op[d_x,d_y] @ c``. The pair is a repeated argument
  if ids repeat, otherwise the two lowest ids.

Modes: ``"lazy"`` runs missing jobs immediately, as one batch per call. ``"record"`` queues
missing jobs in ``pending`` and returns zeros; ``execute_pending()`` runs them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .registry import DirectionStore, VectorRegistry, canonicalize, make_sign_vector
from .store import KtgStore

log = logging.getLogger(__name__)

MODES = ("lazy", "record")


@dataclass(frozen=True)
class Job:
    key: str
    U: np.ndarray
    kind: str  # "probe" | "pair" | "k0"


class EEDEngine:
    def __init__(self, backend, step, store: KtgStore, mode: str = "lazy", debug: bool = False,
                 debug_tol: float = 1e-6):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.backend = backend
        self.step = step
        self.store = store
        self.n = int(backend.n_dof)
        self.c = float(step.c)
        self.mode = mode
        self.debug = bool(debug)
        self.debug_tol = float(debug_tol)
        self.config = None  # set by nistep.build_engine
        self._w = make_sign_vector(self.n, step.seed)
        self.directions = DirectionStore(self.n, step.match_tol)
        self.vectors = VectorRegistry(step.match_tol)
        self._probe_jobs: dict[int, tuple[Job, Job]] = {}
        self._pair_jobs: dict[tuple[int, int], tuple[Job, float]] = {}
        self._pending: dict[str, Job] = {}
        self._K: sp.csr_matrix | None = None
        self._M: sp.csr_matrix | None = None
        self._stats = dict(calls_quad=0, calls_cubic=0, calls_zero=0, calls_incomplete=0,
                           jobs_probe=0, jobs_pair=0, jobs_k0=0, batches=0, time_backend_s=0.0,
                           debug_max_rel_err_quad=0.0, debug_max_rel_err_cubic=0.0)
        self._dir_file = None
        if store.root is not None:
            self._dir_file = store.root / f"directions_seed{step.seed}.npz"
            if self.directions.load(self._dir_file, step.seed):
                log.info("EEDEngine: loaded %d stored directions", len(self.directions))

    # ============================================================== linear operators
    def K(self) -> sp.csr_matrix:
        """Linear stiffness = the backend's own K_tg(0) (cached as a job with U = 0)."""
        if self._K is None:
            U0 = np.zeros(self.n)
            key = self.store.key(U0)
            K = self.store.get(key, U0) if self.store.contains(key) else None
            if K is None:
                t0 = time.perf_counter()
                K = self.backend.stiffness0()
                self._stats["time_backend_s"] += time.perf_counter() - t0
                self._stats["jobs_k0"] += 1
                self.store.put(U0, K, key=key)
                K = self.store.get(key, U0)
            self._K = K
            log.info("EEDEngine: %s", self.store.memory_report())
        return self._K

    def M(self) -> sp.csr_matrix:
        if self._M is None:
            self._M = sp.csr_matrix(self.backend.mass(), dtype=np.float64)
            if self.store.root is not None:  # lets the cache folder be replayed later
                from .backends.replay import save_backend_info

                save_backend_info(self.store.root, self.backend, self._M)
        return self._M

    # ================================================================ public closures
    def quad(self, a, b) -> np.ndarray:
        """Ĝ(a, b): complex (N,) vector, internal-force sign."""
        self._stats["calls_quad"] += 1
        a, b = self._vec(a), self._vec(b)
        if self._is_zero(a) or self._is_zero(b):
            self._stats["calls_zero"] += 1
            return np.zeros(self.n, dtype=np.complex128)
        ia, ib = self.vectors.id_of(a), self.vectors.id_of(b)
        probe, other = (a, b) if ia <= ib else (b, a)
        terms = self._decompose(probe)
        jobs: list[Job] = []
        for _, d in terms:
            jobs.extend(self._probe(d))
        if not self._ensure(jobs):
            return np.zeros(self.n, dtype=np.complex128)
        out = np.zeros(self.n, dtype=np.complex128)
        for kappa, d in terms:
            out += kappa * self._G_apply(d, other)
        if self.debug:
            self._check(out, "quad", None if self.backend.exact_quad is None else
                        (lambda: self.backend.exact_quad(a, b)))
        return out

    def cubic(self, a, b, c) -> np.ndarray:
        """Ĥ(a, b, c): complex (N,) vector, internal-force sign."""
        self._stats["calls_cubic"] += 1
        args = [self._vec(a), self._vec(b), self._vec(c)]
        if any(self._is_zero(x) for x in args):
            self._stats["calls_zero"] += 1
            return np.zeros(self.n, dtype=np.complex128)
        ids = [self.vectors.id_of(x) for x in args]
        p, q, r = self._choose_pair(ids)
        X = self._decompose(args[p])
        Y = self._decompose(args[q])
        v = args[r]
        combos = [(kx * ky, dx, dy) for kx, dx in X for ky, dy in Y]
        jobs: list[Job] = []
        for _, dx, dy in combos:
            jobs.extend(self._cubic_jobs(dx, dy))
        if not self._ensure(jobs):
            return np.zeros(self.n, dtype=np.complex128)
        out = np.zeros(self.n, dtype=np.complex128)
        for f, dx, dy in combos:
            out += f * self._H_apply(dx, dy, v)
        if self.debug:
            self._check(out, "cubic", None if self.backend.exact_cubic is None else
                        (lambda: self.backend.exact_cubic(*args)))
        return out

    def register(self, *vectors) -> list[int]:
        """Give ids to ``vectors`` in the given order (fixes the probe and pair choices)."""
        ids = []
        for x in vectors:
            a = self._vec(x)
            if self._is_zero(a):
                raise ValueError("cannot register a zero vector")
            ids.append(self.vectors.id_of(a))
        return ids

    # ------------------------------------------------------------- mode / batching API
    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def pending_count(self) -> int:
        return len(self._pending)

    def execute_pending(self, max_parallel: int | None = None) -> int:
        """Run all queued jobs (in batches of at most ``max_parallel``); return how many ran."""
        jobs = [j for j in self._pending.values() if not self.store.contains(j.key)]
        self._pending.clear()
        size = len(jobs) if not max_parallel else max(1, int(max_parallel))
        for i in range(0, len(jobs), max(size, 1)):
            self._run(jobs[i:i + size])
        return len(jobs)

    def required_jobs(self, kind: str, *args) -> list[np.ndarray]:
        """Displacements needed by ``quad(*args)`` or ``cubic(*args)``. Registers ids, runs nothing."""
        vecs = [self._vec(x) for x in args]
        if any(self._is_zero(x) for x in vecs):
            return []
        ids = [self.vectors.id_of(x) for x in vecs]
        jobs: list[Job] = []
        if kind == "quad":
            probe = vecs[0] if ids[0] <= ids[1] else vecs[1]
            for _, d in self._decompose(probe):
                jobs.extend(self._probe(d))
        elif kind == "cubic":
            p, q, _ = self._choose_pair(ids)
            for _, dx in self._decompose(vecs[p]):
                for _, dy in self._decompose(vecs[q]):
                    jobs.extend(self._cubic_jobs(dx, dy))
        else:
            raise ValueError("kind must be 'quad' or 'cubic'")
        return [j.U.copy() for j in self._unique(jobs)]

    def stats(self) -> dict:
        s = dict(self._stats)
        s.update(mode=self.mode, n_dof=self.n, c=self.c, n_vectors=len(self.vectors),
                 n_directions=len(self.directions), pending=len(self._pending),
                 jobs_total=s["jobs_probe"] + s["jobs_pair"])
        s.update(self.store.stats())
        return s

    def save_state(self) -> None:
        if self._dir_file is not None and self.directions.dirty:
            self.directions.save(self._dir_file, self.step.seed)

    def with_d_ref(self, d_ref: float) -> "EEDEngine":
        """A new engine with another ``d_ref``, on the same backend, store and directions.

        A job is stored under its displacement, so the engines share every matrix they have
        in common (``K_tg(0)`` at least). The new engine has its own vector ids and counters.
        Used by the ``d_ref`` sweeps (PLAN.md P10.1).
        """
        from dataclasses import replace

        d_ref = float(d_ref)
        if not (d_ref > 0 and np.isfinite(d_ref)):
            raise ValueError(f"d_ref must be > 0, got {d_ref!r}")
        step = replace(self.step, d_ref=d_ref)
        other = EEDEngine(self.backend, step, self.store, mode=self.mode, debug=self.debug,
                          debug_tol=self.debug_tol)
        other.directions = self.directions
        other._K, other._M = self._K, self._M
        if self.config is not None:
            other.config = replace(self.config, step=step)
        return other

    # ====================================================================== internals
    def _vec(self, x) -> np.ndarray:
        a = np.array(x, dtype=np.complex128, copy=True).ravel()
        if a.shape != (self.n,):
            raise ValueError(f"argument must have length {self.n}, got {a.shape}")
        return a

    @staticmethod
    def _amax(a: np.ndarray) -> float:
        return max(float(np.max(np.abs(a.real))), float(np.max(np.abs(a.imag))))

    def _is_zero(self, a: np.ndarray) -> bool:
        return self._amax(a) == 0.0

    def _decompose(self, a: np.ndarray) -> list[tuple[complex, int]]:
        """``a = Σ κ d`` over canonical directions (real and imag parts merged if parallel)."""
        amax = self._amax(a)
        terms: dict[int, complex] = {}
        for part, unit in ((a.real, 1.0), (a.imag, 1j)):
            if float(np.max(np.abs(part))) <= self.step.zero_tol * amax:
                continue
            alpha, d = canonicalize(part, self._w)
            did, s = self.directions.match_or_add(d)
            terms[did] = terms.get(did, 0.0) + unit * alpha * s
        return [(k, d) for d, k in terms.items()]

    @staticmethod
    def _choose_pair(ids: list[int]) -> tuple[int, int, int]:
        if ids[0] == ids[1]:
            return 0, 1, 2
        if ids[0] == ids[2]:
            return 0, 2, 1
        if ids[1] == ids[2]:
            return 1, 2, 0
        o = sorted(range(3), key=lambda t: ids[t])
        return o[0], o[1], o[2]

    # -- jobs -------------------------------------------------------------------------
    def _make_job(self, U: np.ndarray, kind: str) -> Job:
        U = np.ascontiguousarray(U, dtype=np.float64)
        U.flags.writeable = False
        return Job(self.store.key(U), U, kind)

    def _probe(self, d: int) -> tuple[Job, Job]:
        jobs = self._probe_jobs.get(d)
        if jobs is None:
            D = self.directions.get(d)
            jobs = (self._make_job(self.c * D, "probe"), self._make_job(-self.c * D, "probe"))
            self._probe_jobs[d] = jobs
        return jobs

    def _pair(self, d1: int, d2: int) -> tuple[Job, float]:
        """Mixed job for directions d1 < d2 (by id) and its sign s."""
        key = (d1, d2)
        hit = self._pair_jobs.get(key)
        if hit is None:
            D1, D2 = self.directions.get(d1), self.directions.get(d2)
            c = self.c
            s = 1.0 if float(np.max(np.abs(c * D1 + c * D2))) >= c else -1.0
            hit = (self._make_job(c * D1 + (s * c) * D2, "pair"), s)
            self._pair_jobs[key] = hit
        return hit

    def _cubic_jobs(self, dx: int, dy: int) -> list[Job]:
        if dx == dy:
            return list(self._probe(dx))
        d1, d2 = min(dx, dy), max(dx, dy)
        mix, s = self._pair(d1, d2)
        p2, m2 = self._probe(d2)
        return [self._probe(d1)[0], p2 if s > 0 else m2, mix]

    @staticmethod
    def _unique(jobs: list[Job]) -> list[Job]:
        seen: dict[str, Job] = {}
        for j in jobs:
            seen.setdefault(j.key, j)
        return list(seen.values())

    def _ensure(self, jobs: list[Job]) -> bool:
        """Make sure every job is in the store. Record mode queues misses and returns False."""
        if any(j.kind != "k0" for j in jobs):
            self.K()  # H formulas need K; also fixes the shared pattern first
        missing = [j for j in self._unique(jobs) if not self.store.contains(j.key)]
        if not missing:
            return True
        if self.mode == "record":
            for j in missing:
                self._pending.setdefault(j.key, j)
            self._stats["calls_incomplete"] += 1
            return False
        self._run(missing)
        return True

    def _run(self, jobs: list[Job]) -> None:
        if not jobs:
            return
        t0 = time.perf_counter()
        mats = self.backend.tangent_many([j.U for j in jobs])
        self._stats["time_backend_s"] += time.perf_counter() - t0
        self._stats["batches"] += 1
        for j, Kt in zip(jobs, mats):
            self.store.put(j.U, Kt, key=j.key)
            self._stats["jobs_" + j.kind] += 1
        self.save_state()

    def _mat(self, job: Job) -> sp.csr_matrix:
        K = self.store.get(job.key, job.U)
        if K is None:  # evicted from memory and no disk cache, or unreadable file
            if self.mode == "record":
                raise RuntimeError("internal: matrix vanished from the store in record mode")
            self._run([job])
            K = self.store.get(job.key, job.U)
        return K

    # -- operator applications ---------------------------------------------------------
    def _G_apply(self, d: int, v: np.ndarray) -> np.ndarray:
        jp, jm = self._probe(d)
        return (self._mat(jp) @ v - self._mat(jm) @ v) / (4.0 * self.c)

    def _H_apply(self, dx: int, dy: int, v: np.ndarray) -> np.ndarray:
        c = self.c
        K0v = self.K() @ v
        if dx == dy:
            jp, jm = self._probe(dx)
            return (self._mat(jp) @ v + self._mat(jm) @ v - 2.0 * K0v) / (6.0 * c * c)
        d1, d2 = min(dx, dy), max(dx, dy)
        mix, s = self._pair(d1, d2)
        p1 = self._probe(d1)[0]
        p2, m2 = self._probe(d2)
        K2 = self._mat(p2 if s > 0 else m2)
        return (self._mat(mix) @ v - self._mat(p1) @ v - K2 @ v + K0v) / (6.0 * s * c * c)

    # -- debug ---------------------------------------------------------------------------
    def _check(self, out: np.ndarray, what: str, exact_fn) -> None:
        if out.shape != (self.n,) or not np.all(np.isfinite(out)):
            raise FloatingPointError(f"EEDEngine.{what}: non-finite or mis-shaped result")
        if exact_fn is None:
            return
        ex = np.asarray(exact_fn(), dtype=np.complex128)
        scale = max(float(np.max(np.abs(ex))), 1e-300)
        rel = float(np.max(np.abs(out - ex))) / scale
        k = f"debug_max_rel_err_{what}"
        self._stats[k] = max(self._stats[k], rel)
        if rel > self.debug_tol:
            log.warning("EEDEngine.%s: relative error vs exact = %.3e", what, rel)
