"""Abaqus backend: tangent stiffness matrices from the Abaqus UEL, through a runner.

For each prescribed displacement ``U`` (free DOFs, DofMap order):

1. render a job deck from the base deck (``inp_io.JobDeck``): every free DOF prescribed,
   element-matrix output named after the job;
2. run it through the runner (local Abaqus, or a worker on the Abaqus host);
3. parse the element-matrix file (``mtx_io.parse_element_matrices``);
4. check the elements against the deck and assemble on the free DOFs with a fixed sparsity
   pattern (``assembly.Assembler``).

``stiffness0()`` is the same pipeline at ``U = 0`` (CONVENTIONS 4.3: K must be the backend's
own ``K_tg(0)``). ``mass()`` is a job with the mass UEL, or the repository's processed mass
matrix when ``mass_from = "processed"``.

The job name is ``j`` + 12 hex characters of ``sha256(fingerprint, library, U)``: the same
state always gives the same name, different states never collide (legacy bug B8).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..assembly import Assembler
from ..inp_io import InpModel, JobDeck, read_inp
from ..mtx_io import load_processed_ijv, parse_element_matrices
from ..runners.base import AbaqusJob, Runner
from .base import BaseBackend

log = logging.getLogger(__name__)

# a stiffness matrix gives |Ke t| / |Ke| at round-off level (about 1e-14, or 1e-7 with 8-digit
# output) for the rigid translations t; a consistent mass gives a ratio of order 1
TRANSLATION_TOL = 1e-2


def translation_ratio(mats: np.ndarray) -> float:
    """Largest ``max|Ke t| / max|Ke|`` over the elements and the three rigid translations ``t``.

    ``mats`` is ``(n_elem, nd, nd)`` with node-major local DOFs (1, 2, 3 per node). Total-
    Lagrangian internal forces do not change under a rigid translation, so ``K_tg(U) t = 0``
    for every ``U``; the mass matrix maps ``t`` to the element's mass distribution.
    """
    nd = mats.shape[1]
    T = np.zeros((nd, 3))
    for d in range(3):
        T[d::3, d] = 1.0
    num = np.abs(mats @ T).max(axis=(1, 2))
    den = np.abs(mats).max(axis=(1, 2))
    return float(np.max(num / np.maximum(den, np.finfo(float).tiny)))


class AbaqusBackend(BaseBackend):
    kind = "abaqus"

    def __init__(self, model: InpModel, runner: Runner, clamped_nset: str = "Set-2",
                 mass_from: str = "uel", mass_mtx: Path | None = None,
                 provenance: Sequence[str | Path] = (), name: str | None = None):
        super().__init__()
        if mass_from not in ("uel", "processed"):
            raise ValueError("mass_from must be 'uel' or 'processed'")
        if mass_from == "processed" and mass_mtx is None:
            raise ValueError("mass_from='processed' needs mass_mtx")
        self.model = model
        self.runner = runner
        self.clamped_nset = clamped_nset
        self.mass_from = mass_from
        self.mass_mtx = Path(mass_mtx) if mass_mtx is not None else None
        self.provenance = tuple(str(p) for p in provenance)
        self.name = name or (model.path.stem if model.path is not None else "abaqus")
        self.dofmap = model.dofmap(clamped_nset)
        self.asm = Assembler(self.dofmap, model.conn)
        self.deck = JobDeck(model, self.dofmap, clamped_nset)
        self.n_dof = self.dofmap.n_dof
        self._fp: str | None = None
        self._M: sp.csr_matrix | None = None
        self.n_jobs = 0
        self.time_run_s = 0.0

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg, runner: Runner | None = None) -> "AbaqusBackend":
        q = cfg.abaqus
        if runner is None:
            runner = build_runner(cfg)
        provenance = [x for x in (q.user_lib_dir, q.mass_lib_dir, q.user_source, q.mass_source) if x]
        return cls(read_inp(cfg.model.inp), runner, cfg.model.clamped_nset, mass_from=q.mass_from,
                   mass_mtx=cfg.model.mass_mtx, provenance=provenance)

    def fingerprint(self) -> str:
        """Identity of the model: deck, clamped set, UEL provenance and mass source.

        Local files named in ``provenance`` (e.g. the Fortran sources) are hashed by content;
        host paths that do not exist here are hashed by name.
        """
        if self._fp is None:
            h = hashlib.sha256()
            h.update(f"{self.kind}|{self.clamped_nset}|{self.mass_from}|".encode())
            # the boundary block and the matrix file name change per job: hash the model data
            m = self.model
            for arr in (m.sorted_coords(), m.conn, m.elem_labels, self.dofmap.nodes,
                        self.dofmap.clamped_nodes):
                h.update(np.ascontiguousarray(arr).tobytes())
            h.update(repr(m.props).encode())
            for p in self.provenance:
                path = Path(p)
                h.update(path.read_bytes() if path.is_file() else f"|{p}|".encode())
            if self.mass_from == "processed":
                h.update(self.mass_mtx.read_bytes())
            self._fp = h.hexdigest()
        return self._fp

    # ------------------------------------------------------------------------ jobs
    def job_name(self, U: np.ndarray, library: str) -> str:
        h = hashlib.sha256(f"{self.fingerprint()}|{library}|".encode())
        h.update(np.ascontiguousarray(U, dtype=np.float64).tobytes())
        return "j" + h.hexdigest()[:12]

    def make_job(self, U: np.ndarray, library: str = "stiffness") -> AbaqusJob:
        name = self.job_name(U, library)
        output = "mass" if library == "mass" else "stiffness"
        return AbaqusJob(name=name, inp_text=self.deck.render(U, name, output), matrix_file=name,
                         n_elements=self.model.n_elem, library=library)

    def _matrices(self, U_list: list[np.ndarray], library: str) -> list[sp.csr_matrix]:
        import time

        jobs = [self.make_job(U, library) for U in U_list]
        t0 = time.perf_counter()
        results = self.runner.run(jobs)
        self.time_run_s += time.perf_counter() - t0
        self.n_jobs += len(jobs)
        if len(results) != len(jobs):
            raise RuntimeError(f"runner returned {len(results)} results for {len(jobs)} jobs")
        expected = "MASS" if library == "mass" else "STIFFNESS"
        out = []
        for job, res in zip(jobs, results):
            if res.name != job.name:
                raise RuntimeError(f"runner returned job {res.name}, expected {job.name}")
            em = parse_element_matrices(res.mtx_text)
            if em.kind != expected:
                raise RuntimeError(f"job {job.name}: matrix type {em.kind}, expected {expected} "
                                   "(the *Element Matrix Output request of the deck)")
            if em.dofs != (1, 2, 3):
                raise RuntimeError(f"job {job.name}: unexpected element DOFs {em.dofs}")
            # Abaqus labels the output by the deck's request, not by the UEL library, so check
            # the content: a stiffness matrix maps rigid translations to zero, a mass does not.
            r = translation_ratio(em.matrices)
            looks = "STIFFNESS" if r < TRANSLATION_TOL else "MASS"
            if looks != expected:
                raise RuntimeError(f"job {job.name}: the element matrices look like {looks.lower()} "
                                   f"matrices (rigid-translation ratio {r:.1e}), expected "
                                   f"{expected.lower()}; wrong UEL library?")
            order = self.asm.check_element_order(em.labels, em.nodes, self.model.elem_labels)
            out.append(self.asm.assemble(em.matrices[order]))
        return out

    # -------------------------------------------------------------------- matrices
    def stiffness0(self) -> sp.csr_matrix:
        return self._matrices([np.zeros(self.n_dof)], "stiffness")[0]

    def mass(self) -> sp.csr_matrix:
        if self._M is None:
            if self.mass_from == "processed":
                full = load_processed_ijv(self.mass_mtx, self.dofmap.n_total)
                self._M = self.dofmap.restrict_matrix(full)
            else:
                self._M = self._matrices([np.zeros(self.n_dof)], "mass")[0]
        return self._M.copy()

    def _tangent_batch(self, U_list):
        return self._matrices(U_list, "stiffness")


def build_runner(cfg) -> Runner:
    """The runner described by ``[abaqus]`` in the config."""
    kind = cfg.abaqus.runner
    if kind == "local":
        from ..runners.local import LocalRunner

        return LocalRunner.from_config(cfg)
    if kind == "spool":
        from ..runners.spool import SpoolRunner

        return SpoolRunner.from_config(cfg)
    raise NotImplementedError(f"runner {kind!r} is not implemented (use 'local' or 'spool')")
