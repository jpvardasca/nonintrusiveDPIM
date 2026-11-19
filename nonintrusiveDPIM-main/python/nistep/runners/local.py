"""LocalRunner: run Abaqus jobs on this machine, several at a time.

Each job gets its own directory ``<work_dir>/<job name>/`` holding ``<job>.inp`` and, when
a precompiled UEL library is configured, an ``abaqus_v6.env`` that points to it. The checks
that decide success are in :mod:`nistep.runners.abaqus_exec`.

Failed jobs keep their directory (for inspection) and raise :class:`RunnerError` with the
tail of the Abaqus logs. Successful jobs are deleted unless ``keep="all"``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

from . import abaqus_exec
from .base import AbaqusJob, JobResult, RunnerError


class LocalRunner:
    def __init__(self, work_dir, command: Sequence[str] = ("abaqus",), max_parallel: int = 1,
                 lib_dirs: dict[str, str | None] | None = None,
                 user_sources: dict[str, str | None] | None = None,
                 timeout_s: float = 900.0, retries: int = 1, keep: str = "failed",
                 env: dict[str, str] | None = None, extra_args: Sequence[str] = ()):
        if keep not in ("failed", "all", "none"):
            raise ValueError("keep must be 'failed', 'all' or 'none'")
        self.work_dir = Path(work_dir)
        self.command = tuple(command)
        self.max_parallel = max(1, int(max_parallel))
        self.lib_dirs = dict(lib_dirs or {})
        self.user_sources = dict(user_sources or {})
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.keep = keep
        self.env = env
        self.extra_args = tuple(extra_args)
        self.n_jobs_run = 0
        self.n_attempts = 0
        self.wall_s = 0.0

    @classmethod
    def from_config(cls, cfg) -> "LocalRunner":
        q = cfg.abaqus
        return cls(q.work_dir, q.command, q.max_parallel,
                   lib_dirs={"stiffness": q.user_lib_dir, "mass": q.mass_lib_dir},
                   user_sources={"stiffness": q.user_source, "mass": q.mass_source},
                   timeout_s=q.timeout_s, retries=q.retries, keep=q.keep_job_dirs)

    def run(self, jobs: Sequence[AbaqusJob]) -> list[JobResult]:
        jobs = list(jobs)
        if not jobs:
            return []
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(jobs))) as pool:
            outcomes = list(pool.map(self._run_one, jobs))  # every job finishes, even if one fails
        errors = [o for o in outcomes if isinstance(o, RunnerError)]
        if errors:
            raise errors[0]
        return outcomes

    def _run_one(self, job: AbaqusJob):
        job_dir = self.work_dir / job.name
        r = abaqus_exec.run_job_with_retries(
            retries=self.retries, job_dir=job_dir, name=job.name, inp_text=job.inp_text,
            matrix_file=job.matrix_file, n_elements=job.n_elements, command=self.command,
            lib_dir=self.lib_dirs.get(job.library), user_source=self.user_sources.get(job.library),
            timeout_s=self.timeout_s, extra_args=self.extra_args, env=self.env)
        self.n_jobs_run += 1
        self.n_attempts += r["attempts"]
        self.wall_s += r["wall_s"]
        if r["status"] != "ok":
            if self.keep == "none":
                abaqus_exec.remove_job_dir(job_dir)
                return RunnerError(job.name, r["reason"], r["log_tail"])
            return RunnerError(job.name, r["reason"], r["log_tail"], job_dir)
        text = Path(r["mtx_path"]).read_text(encoding="utf-8", errors="replace")
        if self.keep != "all":
            abaqus_exec.remove_job_dir(job_dir)
        return JobResult(job.name, text, r["wall_s"], r["attempts"],
                         job_dir if self.keep == "all" else None)
