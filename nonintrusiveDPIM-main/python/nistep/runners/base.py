"""Runner interface: turn job decks into element-matrix files.

A runner takes :class:`AbaqusJob` objects (the text of a job deck plus what to expect back)
and returns one :class:`JobResult` per job, in the same order, holding the text of the
element-matrix file. It raises :class:`RunnerError` if any job fails after its retries.

Runners: ``local.LocalRunner`` (Abaqus on this machine) and ``spool.SpoolRunner`` (a worker
on the Abaqus host, through a shared folder).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

LIBRARIES = ("stiffness", "mass")


@dataclass(frozen=True)
class AbaqusJob:
    name: str  # job name: letters, digits, underscore; unique for the content
    inp_text: str  # the complete job deck
    matrix_file: str  # base name of the element-matrix output (no extension)
    n_elements: int  # element blocks expected in the matrix file
    library: str = "stiffness"  # which UEL: "stiffness" (K_tg) or "mass"

    def __post_init__(self):
        if self.library not in LIBRARIES:
            raise ValueError(f"library must be one of {LIBRARIES}")
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"bad job name {self.name!r}")


@dataclass
class JobResult:
    name: str
    mtx_text: str
    wall_s: float = 0.0
    attempts: int = 1
    work_dir: Path | None = None


class RunnerError(RuntimeError):
    """A job failed. ``log_tail`` holds the end of the Abaqus logs; ``work_dir`` is kept."""

    def __init__(self, name: str, reason: str, log_tail: str = "", work_dir=None):
        self.name = name
        self.reason = reason
        self.log_tail = log_tail
        self.work_dir = work_dir
        msg = f"Abaqus job {name} failed: {reason}"
        if work_dir is not None:
            msg += f"\n  job directory kept: {work_dir}"
        if log_tail:
            msg += "\n" + log_tail
        super().__init__(msg)


class Runner(Protocol):
    max_parallel: int

    def run(self, jobs: Sequence[AbaqusJob]) -> list[JobResult]: ...
