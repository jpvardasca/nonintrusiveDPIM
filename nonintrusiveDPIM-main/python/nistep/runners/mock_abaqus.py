"""A stand-in for the ``abaqus`` command, for tests on a machine without Abaqus.

Called like Abaqus, inside the job directory::

    python -m nistep.runners.mock_abaqus job=<name> input=<name>.inp double=both ...

It reads the job deck, takes the prescribed displacements from its ``*Boundary`` block,
evaluates the element matrices with the numpy element (PySVK) and writes them in the raw
Abaqus element-matrix layout, plus ``<job>.sta``, ``<job>.msg`` and ``<job>.dat``.

Which matrices it writes mirrors the real set-up: the mass UEL is selected through the
library directory in ``abaqus_v6.env`` (a path whose last folder contains "mass"), or
through a ``user=`` source whose file name contains "mass". As in Abaqus, the label of the
output (``*MATRIX,TYPE=MASS`` or ``STIFFNESS``) comes from the deck's output request
(``Mass=Yes`` or ``Stiffness=Yes``), not from the library: the UEL only fills AMATRX.

``NISTEP_MOCK_MODE`` injects faults (comma-separated): ``fail`` (exit code 1), ``no_mtx``,
``truncated`` (one element block missing), ``no_sta``, ``bad_sta``, ``symmetric`` (lower
triangle storage), ``sleep=<seconds>``, ``fail_once=<marker file>`` (fail on the first
call only), ``nonpoly`` (the tangent is scaled by ``1 + max|U|^3``, so it is no longer a
quadratic polynomial of ``U``; this imitates a UEL with the F-bar flag switched on),
``digits=<n>`` (values written with ``n`` significant digits instead of all 18; the
repository's processed matrices show that a real ``.mtx`` file carries about 15).
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path


def _args(argv: list[str]) -> dict[str, str]:
    out = {}
    for a in argv:
        k, _, v = a.partition("=")
        out[k.lower()] = v
    return out


def _wants_mass(args: dict[str, str]) -> bool:
    if "mass" in Path(args.get("user", "")).name.lower():
        return True
    env = Path("abaqus_v6.env")
    if env.is_file():
        m = re.search(r"usub_lib_dir\s*=\s*r?['\"](.+?)['\"]", env.read_text(encoding="utf-8"))
        if m and "mass" in Path(m.group(1).rstrip("/\\")).name.lower():
            return True
    return False


def main(argv: list[str]) -> int:
    args = _args(argv)
    job = args.get("job")
    if not job:
        print("mock_abaqus: missing job=<name>")
        return 2
    modes = dict((m.partition("=")[0], m.partition("=")[2])
                 for m in os.environ.get("NISTEP_MOCK_MODE", "").split(",") if m)
    if "sleep" in modes:
        time.sleep(float(modes["sleep"]))
    Path(f"{job}.dat").write_text("mock abaqus data file\n", encoding="utf-8")
    if "fail_once" in modes:
        marker = Path(modes["fail_once"])
        if not marker.exists():
            marker.write_text("failed once\n", encoding="utf-8")
            modes["fail"] = ""
    if "fail" in modes:
        Path(f"{job}.msg").write_text("***ERROR: mock failure requested\n", encoding="utf-8")
        Path(f"{job}.sta").write_text(" THE ANALYSIS HAS NOT BEEN COMPLETED\n", encoding="utf-8")
        print("Abaqus Error: mock failure")
        return 1

    from nistep.backends.pysvk import PySVKBackend
    from nistep.inp_io import _find_boundary_block, read_inp, read_prescribed
    from nistep.mtx_io import format_element_matrices

    text = Path(args.get("input", f"{job}.inp")).read_text(encoding="utf-8")
    model = read_inp(text)
    lines = text.splitlines()
    start, end = _find_boundary_block(lines)
    clamped = next(ln.split(",")[0].strip() for ln in lines[start + 1:end]
                   if "ENCASTRE" in ln.upper())
    be = PySVKBackend(model, clamped)
    if _wants_mass(args):
        Ke = be.element_masses()
    else:
        U = read_prescribed(text, be.dofmap)
        Ke = be.element_tangents(U)
        if "nonpoly" in modes:
            Ke = Ke * (1.0 + float(abs(U).max()) ** 3)
    request = next((ln for ln in lines
                    if ln.strip().upper().startswith("*ELEMENT MATRIX OUTPUT")), "")
    kind = "MASS" if re.search(r"\bmass\s*=\s*yes\b", request, re.IGNORECASE) else "STIFFNESS"
    labels, conn = model.elem_labels, model.conn
    if "truncated" in modes:
        labels, conn, Ke = labels[:-1], conn[:-1], Ke[:-1]
    if "no_mtx" not in modes:
        out = format_element_matrices(labels, conn, Ke, kind=kind, symmetric="symmetric" in modes,
                                      sig_digits=int(modes.get("digits", 18)))
        Path(f"{model.matrix_file_name}.mtx").write_text(out, encoding="utf-8")
    Path(f"{job}.msg").write_text("mock abaqus message file\n", encoding="utf-8")
    if "no_sta" not in modes:
        ok = " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n" if "bad_sta" not in modes else " ABORTED\n"
        Path(f"{job}.sta").write_text(" SUMMARY OF JOB INFORMATION:\n" + ok, encoding="utf-8")
    print("Abaqus JOB", job, "COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
