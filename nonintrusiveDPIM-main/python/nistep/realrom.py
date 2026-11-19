"""The real form of a ROM written by MORFEStEP's ``export_real_rom`` (PLAN.md P11.2).

The file ``real_rom.npz`` holds polynomials in the real variables ``x1, y1, ..., xm, ym``
(master pair ``p``: ``z_p = x_p + i y_p``) and, for a forced ROM, ``c, s`` (the forcing:
``c = level cos(omega t)``, ``s = level sin(omega t)``; ``level = 1`` is the load the ROM was
built with). Every polynomial is stored as one row of coefficients on the monomials given by
the rows of ``exponents``.

    rom = RealROM.load("runs/mems_forced/real_rom.npz")
    dq = rom.rhs(q, t, omega)                 # dq/dt of the reduced state q = (x1, y1, ...)
    u = rom.displacement(q, t, omega)         # displacement of each exported DOF

``rhs`` is the right-hand side of a real first-order ODE with ``2m`` states: it can be
given to a time integrator or a harmonic balance solver directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RealROM:
    variables: tuple[str, ...]
    exponents: np.ndarray  # (L, n_variables), int
    f: np.ndarray  # (2m, L): d(x1)/dt, d(y1)/dt, ...
    dofs: np.ndarray  # (n,) 1-based free DOF indices
    displacement_coeffs: np.ndarray  # (n, L)
    velocity_coeffs: np.ndarray  # (n, L) or (0, L)
    omega0: float  # forcing frequency the ROM was built at (nan without forcing)

    @classmethod
    def load(cls, path: str | Path) -> RealROM:
        with np.load(Path(path)) as d:
            names = bytes(d["variables"].astype(np.uint8)).decode("utf-8").split(",")
            exps = np.asarray(d["exponents"], dtype=np.int64)
            rom = cls(
                variables=tuple(names),
                exponents=exps.reshape(-1, len(names)),
                f=np.atleast_2d(np.asarray(d["f"], dtype=float)),
                dofs=np.atleast_1d(np.asarray(d["dofs"], dtype=np.int64)),
                displacement_coeffs=np.asarray(d["displacement"], dtype=float).reshape(-1, exps.shape[0]),
                velocity_coeffs=np.asarray(d["velocity"], dtype=float).reshape(-1, exps.shape[0]),
                omega0=float(d["omega0"]),
            )
        rom._check()
        return rom

    def _check(self) -> None:
        L, nv = self.exponents.shape
        m2 = self.f.shape[0]
        if self.f.shape != (m2, L) or m2 % 2 or nv not in (m2, m2 + 2):
            raise ValueError(f"inconsistent real ROM: f {self.f.shape}, exponents {self.exponents.shape}")
        if self.displacement_coeffs.shape != (len(self.dofs), L):
            raise ValueError("displacement rows do not match the DOFs")

    @property
    def n_states(self) -> int:
        return self.f.shape[0]

    @property
    def forced(self) -> bool:
        return self.exponents.shape[1] == self.n_states + 2

    def _variables(self, q, t, omega, level) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.n_states:
            raise ValueError(f"the state has {self.n_states} entries, got {q.shape[0]}")
        if not self.forced:
            return q
        if omega is None:
            raise ValueError("a forced ROM needs the forcing frequency omega")
        t = np.asarray(t, dtype=float)
        c = level * np.cos(omega * t) * np.ones(q.shape[1:])
        s = level * np.sin(omega * t) * np.ones(q.shape[1:])
        return np.concatenate([q, c[None, ...], s[None, ...]], axis=0)

    def monomials(self, q, t=0.0, omega=None, level: float = 1.0) -> np.ndarray:
        """Values of the ``L`` monomials; ``q`` may be ``(2m,)`` or ``(2m, n_points)``
        (then ``t`` is a scalar or ``(n_points,)``)."""
        v = self._variables(q, t, omega, level)
        e = self.exponents.reshape(self.exponents.shape + (1,) * (v.ndim - 1))
        return np.prod(v[None, ...] ** e, axis=1)

    def rhs(self, q, t=0.0, omega=None, level: float = 1.0) -> np.ndarray:
        """``dq/dt`` of the reduced state ``q = (x1, y1, ..., xm, ym)``."""
        return self.f @ self.monomials(q, t, omega, level)

    def displacement(self, q, t=0.0, omega=None, level: float = 1.0) -> np.ndarray:
        """Displacement of each exported DOF."""
        return self.displacement_coeffs @ self.monomials(q, t, omega, level)

    def velocity(self, q, t=0.0, omega=None, level: float = 1.0) -> np.ndarray:
        """Velocity of each exported DOF (empty if it was not exported)."""
        return self.velocity_coeffs @ self.monomials(q, t, omega, level)
