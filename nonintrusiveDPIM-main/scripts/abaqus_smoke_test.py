#!/usr/bin/env python
"""Smoke test of an Abaqus set-up (PLAN.md P8.6). Run it once per deck before any ROM.

    python scripts/abaqus_smoke_test.py configs/<model>_abaqus.toml [--amplitude X] [--tol 1e-8]

The config must have ``[backend] kind = "abaqus"``. The script runs six Abaqus jobs through
the configured runner (local, or the spool worker on the Abaqus host) and compares them with
the numpy element (PySVK) built from the same deck:

  (a) K_tg(0)                      Abaqus against PySVK
  (b) K_tg(U) at a random state    Abaqus against PySVK
  (c) cubic exactness              K_tg(t d) must be exactly quadratic in t; a UEL with the
                                   F-bar flag switched on by accident (bug B9) fails here
  (d) K_tg(0) against the repository's processed stiffness, if [model] stiffness_mtx is set
      mass matrix                  Abaqus against PySVK (and [model] mass_mtx if set)
  (e) wall time per job

Exit code 0 if every check passes. Needs Python >= 3.11 with numpy and scipy (the same
environment as the rest of ``nistep``).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import nistep  # noqa: E402
from nistep.backends.pysvk import PySVKBackend  # noqa: E402
from nistep.diagnostics import format_smoke_report, smoke_test  # noqa: E402
from nistep.mtx_io import load_processed_ijv  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", help="TOML config with [backend] kind = \"abaqus\"")
    p.add_argument("--amplitude", type=float, default=None,
                   help="largest prescribed displacement (default: 10%% of the shortest edge)")
    p.add_argument("--tol", type=float, default=1e-8, help="tolerance of the matrix comparisons")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write the report to this JSON file")
    args = p.parse_args(argv)

    cfg = nistep.load_config(args.config)
    if cfg.backend.kind != "abaqus":
        p.error(f"[backend] kind is {cfg.backend.kind!r}; this test needs \"abaqus\"")
    backend = nistep.build_backend(cfg)
    reference = PySVKBackend(backend.model, cfg.model.clamped_nset)
    dm = backend.dofmap
    K_proc = M_proc = None
    if cfg.model.stiffness_mtx is not None:
        K_proc = dm.restrict_matrix(load_processed_ijv(cfg.model.stiffness_mtx, dm.n_total))
    if cfg.model.mass_mtx is not None:
        M_proc = dm.restrict_matrix(load_processed_ijv(cfg.model.mass_mtx, dm.n_total))

    print(f"config: {cfg.path}\nrunner: {cfg.abaqus.runner}, command: {' '.join(cfg.abaqus.command)}")
    report = smoke_test(backend, reference, amplitude=args.amplitude, seed=args.seed,
                        tol=args.tol, processed_K=K_proc, processed_M=M_proc)
    print(format_smoke_report(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
