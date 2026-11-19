"""Diagnostics: checks that a tangent-stiffness backend behaves as the EED-StEP assumes.

* :func:`matrix_rel_diff` compares two sparse matrices.
* :func:`cubic_exactness` checks that ``K_tg(t d)`` is exactly quadratic in ``t``. A UEL with
  an active F-bar formulation (legacy bug B9) or any other non-polynomial behaviour fails it.
* :func:`smoke_test` runs the P8.6 checks of an Abaqus backend against the numpy element
  (PySVK) built from the same deck.
* :func:`d_ref_sweep` evaluates ``Ĝ(φ,φ)`` and ``Ĥ(φ,φ,φ)`` at several ``d_ref`` (P10.1), to
  find the range in which the result does not depend on ``d_ref``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def matrix_rel_diff(A, B) -> float:
    """``max|A − B| / max|B|`` for sparse or dense matrices of the same shape."""
    A, B = sp.csr_matrix(A), sp.csr_matrix(B)
    if A.shape != B.shape:
        raise ValueError(f"shapes differ: {A.shape} and {B.shape}")
    scale = abs(B).max()
    diff = abs(A - B).max()
    return 0.0 if diff == 0 else float(diff / scale) if scale > 0 else float("inf")


def symmetry_error(A) -> float:
    """``max|A − Aᵀ| / max|A|``."""
    A = sp.csr_matrix(A)
    scale = abs(A).max()
    return float(abs(A - A.T).max() / scale) if scale > 0 else 0.0


def cubic_exactness(backend, d: np.ndarray, c: float, matrices=None) -> float:
    """Third difference of ``K_tg`` along ``d``, relative to the first difference.

    ``K_tg(t d) = K + 2t Ĝ(d,·) + 3t² Ĥ(d,d,·)`` is a quadratic in ``t``, so
    ``K(3c) − 3K(2c) + 3K(c) − K(0)`` must vanish. Returns
    ``max|third difference| / max|K(c) − K(0)|``; round-off gives about 1e-13.
    Pass ``matrices = [K(0), K(c d), K(2c d), K(3c d)]`` to reuse matrices already computed.
    """
    d = np.asarray(d, dtype=np.float64)
    if matrices is None:
        matrices = backend.tangent_many([t * c * d for t in range(4)])
    K0, K1, K2, K3 = (sp.csr_matrix(K) for K in matrices)
    first = abs(K1 - K0).max()
    if first == 0:
        raise ValueError("K(c d) equals K(0): the direction or the amplitude is zero")
    return float(abs(K3 - 3 * K2 + 3 * K1 - K0).max() / first)


def shortest_edge(model) -> float:
    """Shortest element edge of a hex8 mesh (``model`` is an ``InpModel``)."""
    X = model.sorted_coords()
    rank = np.searchsorted(np.sort(model.node_labels), model.conn)
    Xe = X[rank]  # (e, 8, 3)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return float(min(np.linalg.norm(Xe[:, a] - Xe[:, b], axis=1).min() for a, b in edges))


def smoke_test(backend, reference, amplitude: float | None = None, seed: int = 0,
               tol: float = 1e-8, processed_K=None, processed_M=None) -> dict:
    """P8.6 checks of ``backend`` (Abaqus) against ``reference`` (PySVK on the same deck).

    Runs six jobs: ``K_tg`` at 0, at a random state, at ``c d``, ``2c d``, ``3c d``, and the
    mass matrix. ``amplitude`` is the largest prescribed displacement of the random state and
    ``c`` of the exactness test (default: 10 % of the shortest element edge).

    Returns a dict with one entry per check: ``value``, ``tol`` and ``ok``; plus ``ok`` (all
    passed), ``jobs`` and ``seconds_per_job``.
    """
    import time

    n = backend.n_dof
    if reference.n_dof != n:
        raise ValueError("backend and reference have different sizes")
    if amplitude is None:
        amplitude = 0.1 * shortest_edge(reference.model)
    rng = np.random.default_rng(seed)
    U = rng.uniform(-1.0, 1.0, size=n)
    U *= amplitude / np.max(np.abs(U))
    d = rng.uniform(-1.0, 1.0, size=n)
    d /= np.max(np.abs(d))
    states = [np.zeros(n), U, amplitude * d, 2 * amplitude * d, 3 * amplitude * d]

    t0 = time.perf_counter()
    Ks = backend.tangent_many(states)
    M = backend.mass()
    seconds = time.perf_counter() - t0
    refs = reference.tangent_many(states)

    checks: dict[str, dict] = {}

    def add(name, value, limit, note=""):
        checks[name] = {"value": float(value), "tol": float(limit), "ok": bool(value <= limit),
                        "note": note}

    add("a_K0_vs_pysvk", matrix_rel_diff(Ks[0], refs[0]), tol, "K_tg(0): Abaqus against PySVK")
    add("b_KU_vs_pysvk", matrix_rel_diff(Ks[1], refs[1]), tol,
        f"K_tg(U), max|U| = {amplitude:.4g}: Abaqus against PySVK")
    add("c_cubic_exactness", cubic_exactness(backend, d, amplitude, [Ks[0], Ks[2], Ks[3], Ks[4]]),
        1e-9, "third difference of K_tg along a direction (0 for a polynomial UEL)")
    add("c_cubic_exactness_pysvk", cubic_exactness(reference, d, amplitude,
                                                   [refs[0], refs[2], refs[3], refs[4]]),
        1e-9, "same test on PySVK (the round-off floor)")
    add("mass_vs_pysvk", matrix_rel_diff(M, reference.mass()), tol, "mass: Abaqus against PySVK")
    add("K0_symmetry", symmetry_error(Ks[0]), 1e-10, "max|K − Kᵀ| / max|K|")
    add("nonlinear_part", 1.0 / max(matrix_rel_diff(Ks[1], Ks[0]), 1e-300), 1e6,
        "1 / (relative size of K_tg(U) − K): the test state must be nonlinear enough")
    if processed_K is not None:
        add("d_K0_vs_processed", matrix_rel_diff(Ks[0], processed_K), tol,
            "K_tg(0) against the repository's processed stiffness matrix")
    if processed_M is not None:
        add("mass_vs_processed", matrix_rel_diff(M, processed_M), tol,
            "mass against the repository's processed mass matrix")
    jobs = len(states) + 1
    return {"checks": checks, "ok": all(c["ok"] for c in checks.values()), "jobs": jobs,
            "seconds_per_job": seconds / jobs, "amplitude": float(amplitude), "n_dof": n}


def reference_backend(backend):
    """A backend with ``min_det_F`` and exact forms for the model of ``backend``, or ``None``.

    PySVK is its own reference. For an Abaqus backend the numpy element (PySVK) is built from
    the same deck. Analytic backends have exact forms but no mesh; they are returned as is.
    """
    if getattr(backend, "min_det_F", None) is not None or getattr(backend, "exact_quad", None) is not None:
        return backend
    model = getattr(backend, "model", None)
    if model is None:
        return None
    from .backends.pysvk import PySVKBackend

    return PySVKBackend(model, getattr(backend, "clamped_nset", "Set-2"))


def d_ref_sweep(engine, phi, d_refs, reference=None, det_f_min: float | None = 0.2,
                plateau_tol: float = 1e-6) -> dict:
    """``Ĝ(φ,φ)`` and ``Ĥ(φ,φ,φ)`` of a real vector ``φ`` at each ``d_ref`` (PLAN.md P10.1).

    For an exactly cubic internal force the two forms do not depend on ``d_ref``. What changes
    is the error: the rounding of the tangent matrices is divided by ``c`` (in ``Ĝ``) and by
    ``c²`` (in ``Ĥ``), ``c = eps·d_ref``, so a small ``d_ref`` amplifies it, while a large one
    distorts the elements. The usable range is where successive values agree.

    Each ``d_ref`` costs two jobs (``K_tg(±c d)``, ``d = φ / max|φ|``) on an engine made with
    ``engine.with_d_ref``; all of them share the matrix store of ``engine``.

    * ``reference``: a backend of the same model that is exact (default:
      :func:`reference_backend`, i.e. PySVK for a deck). If it has ``min_det_F``, the smallest
      ``det F`` over the Gauss points is computed for both prescribed states, and a ``d_ref``
      with ``min det F <= det_f_min`` is refused: its jobs do not run (``det_f_min = None``
      switches the refusal off). If it has ``exact_quad``/``exact_cubic``, the errors against
      them are reported.
    * A backend failure (an Abaqus job that fails) is recorded and the sweep goes on.

    Returns ``{"rows": [...], "exact": bool, "n_dof", "eps", "plateau_tol"}``. Each row has
    ``d_ref``, ``c``, ``status`` (``ok`` | ``refused`` | ``failed``), ``reason``, ``min_det_F``,
    ``jobs``, ``G_max`` and ``H_max`` (largest entry), ``phi_G`` and ``phi_H`` (``φ·Ĝ``,
    ``φ·Ĥ``), ``change_G`` and ``change_H`` (``max|x − x_prev| / max|x_prev|`` against the
    previous ``ok`` row), ``err_G`` and ``err_H`` (against the exact forms of the reference)
    and ``plateau`` (both changes ``<= plateau_tol``).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.shape != (engine.n,):
        raise ValueError(f"phi must have length {engine.n}, got {phi.shape}")
    scale = float(np.max(np.abs(phi)))
    if scale == 0:
        raise ValueError("phi is zero")
    d_refs = [float(x) for x in d_refs]
    if not d_refs or any(not x > 0 for x in d_refs):
        raise ValueError("d_refs must be a non-empty list of positive numbers")
    if reference is None:
        reference = reference_backend(engine.backend)
    if reference is not None and reference.n_dof != engine.n:
        raise ValueError("the reference backend has a different size")
    geometry = reference if getattr(reference, "min_det_F", None) is not None else None
    exact_g = exact_h = None
    if getattr(reference, "exact_quad", None) is not None:
        exact_g = np.asarray(reference.exact_quad(phi, phi)).real
        exact_h = np.asarray(reference.exact_cubic(phi, phi, phi)).real

    def rel(x, ref):
        m = float(np.max(np.abs(ref)))
        return float(np.max(np.abs(x - ref)) / m) if m > 0 else float("nan")

    rows, prev = [], None
    for d_ref in d_refs:
        eng = engine.with_d_ref(d_ref)
        row = dict(d_ref=d_ref, c=eng.c, status="ok", reason="", min_det_F=None, jobs=0,
                   G_max=None, H_max=None, phi_G=None, phi_H=None, change_G=None, change_H=None,
                   err_G=None, err_H=None, plateau=False)
        rows.append(row)
        if geometry is not None:
            U = (eng.c / scale) * phi
            row["min_det_F"] = min(geometry.min_det_F(U), geometry.min_det_F(-U))
            if det_f_min is not None and row["min_det_F"] <= det_f_min:
                row.update(status="refused",
                           reason=f"min det F = {row['min_det_F']:.3g} <= {det_f_min:g}")
                continue
        try:
            g = np.asarray(eng.quad(phi, phi)).real
            h = np.asarray(eng.cubic(phi, phi, phi)).real
        except Exception as exc:  # a failed job must not end the sweep
            row.update(status="failed", reason=(str(exc).splitlines() or [type(exc).__name__])[0][:200])
            continue
        finally:
            row["jobs"] = int(eng.stats()["jobs_total"])
        row.update(G_max=float(np.max(np.abs(g))), H_max=float(np.max(np.abs(h))),
                   phi_G=float(phi @ g), phi_H=float(phi @ h))
        if exact_g is not None:
            row.update(err_G=rel(g, exact_g), err_H=rel(h, exact_h))
        if prev is not None:
            row.update(change_G=rel(g, prev[0]), change_H=rel(h, prev[1]))
            row["plateau"] = bool(row["change_G"] <= plateau_tol and row["change_H"] <= plateau_tol)
        prev = (g, h)
    return {"rows": rows, "exact": exact_g is not None, "n_dof": int(engine.n),
            "eps": float(engine.step.eps), "plateau_tol": float(plateau_tol)}


def format_d_ref_sweep(report: dict) -> str:
    def num(x, fmt="10.3e"):
        return format(x, fmt) if x is not None else " " * 9 + "-"

    lines = [f"d_ref sweep: {report['n_dof']} free DOFs, c = {report['eps']:g} d_ref, two jobs per d_ref",
             "     d_ref          c   min det F      phi.G       phi.H    change G   change H"
             + ("      err G      err H" if report["exact"] else "")]
    for r in report["rows"]:
        line = (f"{r['d_ref']:10.4g} {r['c']:10.4g}  {num(r['min_det_F'], '10.4f')} "
                f"{num(r['phi_G'], '11.4e')} {num(r['phi_H'], '11.4e')} "
                f"{num(r['change_G'])} {num(r['change_H'])}")
        if report["exact"]:
            line += f" {num(r['err_G'])} {num(r['err_H'])}"
        if r["status"] != "ok":
            line += f"   {r['status'].upper()}: {r['reason']}"
        elif r["plateau"]:
            line += "   plateau"
        lines.append(line)
    lines.append(f"'change' compares a row with the previous one that ran; 'plateau' means both "
                 f"changes are <= {report['plateau_tol']:g}.")
    return "\n".join(lines)


def format_smoke_report(report: dict) -> str:
    lines = [f"Abaqus smoke test: {report['n_dof']} free DOFs, {report['jobs']} jobs, "
             f"{report['seconds_per_job']:.2f} s per job, test amplitude {report['amplitude']:.4g}"]
    for name, c in report["checks"].items():
        lines.append(f"  [{'ok' if c['ok'] else 'FAIL'}] {name:26s} {c['value']:10.3e}  "
                     f"(limit {c['tol']:.0e})  {c['note']}")
    lines.append("RESULT: " + ("all checks passed" if report["ok"] else "FAILED"))
    return "\n".join(lines)
