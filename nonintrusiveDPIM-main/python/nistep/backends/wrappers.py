"""Backend wrappers used to test batching and resume (PLAN.md P9).

:class:`ThrottledBackend` makes any backend slow, or makes it fail after a number of jobs,
so that a run can be interrupted at a known point. It does not change the results, and it
keeps the fingerprint of the wrapped backend, so both share one cache.
"""

from __future__ import annotations

import time


class SimulatedFailure(RuntimeError):
    """Raised by :class:`ThrottledBackend` when ``fail_after`` jobs have run."""


class ThrottledBackend:
    """Wrap ``inner``: sleep ``delay_s`` per tangent job; fail once ``fail_after`` jobs ran.

    Everything else (``n_dof``, ``fingerprint``, ``stiffness0``, ``mass``, ``exact_quad``, …)
    is forwarded to ``inner``. ``jobs_done`` counts the tangent jobs that completed.
    A failure in the middle of a batch loses that whole batch, as a crash would.
    """

    def __init__(self, inner, delay_s: float = 0.0, fail_after: int | None = None):
        if delay_s < 0:
            raise ValueError("delay_s must be >= 0")
        self.inner = inner
        self.delay_s = float(delay_s)
        self.fail_after = None if fail_after is None else int(fail_after)
        self.jobs_done = 0

    def __getattr__(self, name):  # only called for attributes not set on the wrapper
        return getattr(self.inner, name)

    def tangent_many(self, U_list):
        out = []
        for U in U_list:
            if self.fail_after is not None and self.jobs_done >= self.fail_after:
                raise SimulatedFailure(f"simulated failure after {self.jobs_done} jobs")
            if self.delay_s > 0:
                time.sleep(self.delay_s)
            out.extend(self.inner.tangent_many([U]))
            self.jobs_done += 1
        return out

    def tangent(self, U):
        return self.tangent_many([U])[0]
