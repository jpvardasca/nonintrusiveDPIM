"""SpoolRunner: send jobs to a worker on the Abaqus host through a shared folder.

Folder layout (``spool_dir``, visible from both machines)::

    inbox/    <job>.inp, <job>.json     submitted by the client; the .json is written last
    running/  <job>.inp, <job>.json     claimed by the worker (atomic rename)
    outbox/   <job>.mtx, <job>.json     written by the worker; the .json is written last
    worker.alive                         heartbeat, rewritten by the worker every few seconds

Every file is written under a temporary name and renamed, so a reader never sees a partial
file. The client reads ``outbox/<job>.mtx`` only after ``outbox/<job>.json`` exists.

The worker is ``scripts/abaqus_worker.py`` (standard library only). The client decides that
the worker is down when the *content* of ``worker.alive`` has not changed for
``heartbeat_stale_s`` seconds; it never compares clocks of the two machines.

A finished result left in ``outbox`` by an interrupted client run is reused: job names are
content hashes, so the same name always means the same job.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Sequence

from .base import AbaqusJob, JobResult, RunnerError


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


class SpoolRunner:
    def __init__(self, spool_dir, max_parallel: int = 1, timeout_s: float = 900.0,
                 retries: int = 1, poll_s: float = 0.5, heartbeat_stale_s: float = 120.0):
        self.spool_dir = Path(spool_dir)
        self.max_parallel = max(1, int(max_parallel))
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.poll_s = float(poll_s)
        self.heartbeat_stale_s = float(heartbeat_stale_s)
        self.inbox = self.spool_dir / "inbox"
        self.outbox = self.spool_dir / "outbox"
        self.heartbeat = self.spool_dir / "worker.alive"
        self.n_jobs_run = 0
        self.n_reused = 0

    @classmethod
    def from_config(cls, cfg) -> "SpoolRunner":
        q = cfg.abaqus
        return cls(q.spool_dir, q.max_parallel, q.timeout_s, q.retries, q.poll_s,
                   q.heartbeat_stale_s)

    # ------------------------------------------------------------------ helpers
    def _read_heartbeat(self) -> str | None:
        try:
            return self.heartbeat.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _read_result(self, name: str) -> dict | None:
        marker = self.outbox / f"{name}.json"
        if not marker.is_file():
            return None
        try:
            return json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # not complete yet on a slow share; try again on the next poll

    def _discard_result(self, name: str) -> None:
        for suffix in (".json", ".mtx"):
            try:
                (self.outbox / f"{name}{suffix}").unlink()
            except OSError:
                pass

    def _submit(self, job: AbaqusJob) -> None:
        _atomic_write(self.inbox / f"{job.name}.inp", job.inp_text)
        meta = {"name": job.name, "library": job.library, "matrix_file": job.matrix_file,
                "n_elements": job.n_elements, "timeout_s": self.timeout_s,
                "retries": self.retries}
        _atomic_write(self.inbox / f"{job.name}.json", json.dumps(meta))  # last: "ready" marker

    # --------------------------------------------------------------------- run
    def run(self, jobs: Sequence[AbaqusJob]) -> list[JobResult]:
        jobs = list(jobs)
        if not jobs:
            return []
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.outbox.mkdir(parents=True, exist_ok=True)

        done: dict[str, JobResult] = {}
        failed: dict[str, RunnerError] = {}
        pending: dict[str, AbaqusJob] = {}
        for job in jobs:
            if job.name in pending or job.name in done:
                continue
            old = self._read_result(job.name)
            if old is not None and old.get("status") == "ok":
                done[job.name] = self._collect(job, old)  # left by an interrupted run
                self.n_reused += 1
                continue
            if old is not None:
                self._discard_result(job.name)  # an old failure: run it again
            self._submit(job)
            pending[job.name] = job

        beat = self._read_heartbeat()
        beat_changed = time.monotonic()
        # generous overall limit: queueing time on the host is unknown
        rounds = (len(pending) + self.max_parallel - 1) // self.max_parallel + 1
        deadline = time.monotonic() + rounds * self.timeout_s * (self.retries + 1) + 60.0
        while pending:
            for name in list(pending):
                res = self._read_result(name)
                if res is None:
                    continue
                job = pending.pop(name)
                if res.get("status") == "ok":
                    done[name] = self._collect(job, res)
                else:
                    failed[name] = RunnerError(name, res.get("reason", "unknown failure"),
                                               res.get("log_tail", ""), res.get("work_dir"))
                    self._discard_result(name)
            if not pending:
                break
            now_beat = self._read_heartbeat()
            if now_beat != beat:
                beat, beat_changed = now_beat, time.monotonic()
            elif time.monotonic() - beat_changed > self.heartbeat_stale_s:
                self._withdraw(pending)
                state = "missing" if beat is None else "not updated"
                raise RunnerError(next(iter(pending)), f"worker down: {self.heartbeat} {state} for "
                                  f"{self.heartbeat_stale_s:g} s. Start scripts/abaqus_worker.py "
                                  "on the Abaqus host.")
            if time.monotonic() > deadline:
                self._withdraw(pending)
                raise RunnerError(next(iter(pending)), "timed out waiting for the worker")
            time.sleep(self.poll_s)

        self.n_jobs_run += len(done) + len(failed)
        if failed:
            raise next(iter(failed.values()))
        return [done[j.name] for j in jobs]

    def _collect(self, job: AbaqusJob, res: dict) -> JobResult:
        mtx = self.outbox / f"{job.name}.mtx"
        try:
            text = mtx.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RunnerError(job.name, f"result marker present but {mtx} is unreadable: {exc}") from None
        self._discard_result(job.name)
        return JobResult(job.name, text, float(res.get("wall_s", 0.0)), int(res.get("attempts", 1)))

    def _withdraw(self, pending: dict[str, AbaqusJob]) -> None:
        """Remove jobs the worker has not claimed, so a later run starts clean."""
        for name in pending:
            for suffix in (".json", ".inp"):  # the marker first: the worker claims on it
                try:
                    (self.inbox / f"{name}{suffix}").unlink()
                except OSError:
                    pass
