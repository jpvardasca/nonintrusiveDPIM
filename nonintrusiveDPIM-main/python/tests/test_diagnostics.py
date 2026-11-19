"""Tests of the diagnostics (PLAN.md P8.6) and of the host-side helper scripts.

Abaqus is replaced by ``nistep.runners.mock_abaqus``. The same checks run against a real
Abaqus through ``scripts/abaqus_smoke_test.py`` on the Abaqus host.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from decks import DECKS, LEGACY, backend, deck_path, model
from nistep.backends.abaqus import AbaqusBackend
from nistep.diagnostics import (cubic_exactness, format_smoke_report, matrix_rel_diff,
                                shortest_edge, smoke_test, symmetry_error)
from nistep.mtx_io import load_processed_ijv
from nistep.runners.local import LocalRunner

MOCK = (sys.executable, "-m", "nistep.runners.mock_abaqus")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def mock_backend(tmp_path, name="single", mode=""):
    env = dict(os.environ)
    env["NISTEP_MOCK_MODE"] = mode
    lib = {"stiffness": str(tmp_path / "uel_lib" / "stiffness"), "mass": str(tmp_path / "uel_lib" / "mass")}
    runner = LocalRunner(tmp_path / "work", MOCK, lib_dirs=lib, env=env, retries=0, max_parallel=3)
    return AbaqusBackend(model(name), runner), runner


class Warped:
    """A backend that is not polynomial: ``K_tg(U) · (1 + max|U|³)``."""

    def __init__(self, inner):
        self.inner, self.n_dof, self.model = inner, inner.n_dof, inner.model

    def tangent_many(self, U_list):
        return [K * (1.0 + np.max(np.abs(U)) ** 3) for K, U in zip(self.inner.tangent_many(U_list), U_list)]

    def mass(self):
        return self.inner.mass()


# ------------------------------------------------------------------ matrix helpers

def test_matrix_helpers():
    A = sp.csr_matrix(np.array([[2.0, 1.0], [1.0, 4.0]]))
    assert matrix_rel_diff(A, A) == 0.0
    B = A.copy()
    B[0, 1] = 1.5
    assert matrix_rel_diff(B, A) == pytest.approx(0.5 / 4.0)
    assert symmetry_error(A) == 0.0 and symmetry_error(B) == pytest.approx(0.5 / 4.0)
    assert matrix_rel_diff(A, sp.csr_matrix((2, 2))) == float("inf")
    with pytest.raises(ValueError, match="shapes differ"):
        matrix_rel_diff(A, sp.csr_matrix((3, 3)))
    # different sparsity patterns are compared entry by entry
    C = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 4.0]]))
    assert matrix_rel_diff(C, A) == pytest.approx(1.0 / 4.0)


def test_shortest_edge():
    assert shortest_edge(model("single")) == pytest.approx(1.0)
    m = model("mems")
    X = m.sorted_coords()
    assert 0 < shortest_edge(m) <= np.ptp(X, axis=0).max()


# ------------------------------------------------------------------ cubic exactness

@pytest.mark.parametrize("name,c", [("single", 0.1), ("mems", 1.0)])
def test_cubic_exactness_of_pysvk(name, c):
    be = backend(name)
    d = np.random.default_rng(1).uniform(-1, 1, size=be.n_dof)
    assert cubic_exactness(be, d, c) <= 1e-11


def test_cubic_exactness_detects_non_polynomial_backend():
    be = Warped(backend("single"))
    d = np.random.default_rng(1).uniform(-1, 1, size=be.n_dof)
    assert cubic_exactness(be, d, 0.1) > 1e-4
    with pytest.raises(ValueError, match="direction or the amplitude is zero"):
        cubic_exactness(be, np.zeros(be.n_dof), 0.1)


# ----------------------------------------------------------------------- smoke test

def test_smoke_test_with_mock_abaqus(tmp_path):
    ab, runner = mock_backend(tmp_path)
    ref = backend("single")
    n_total = ab.dofmap.n_total
    K_proc = ab.dofmap.restrict_matrix(load_processed_ijv(LEGACY / DECKS["single"][1], n_total))
    M_proc = ab.dofmap.restrict_matrix(load_processed_ijv(LEGACY / DECKS["single"][2], n_total))
    report = smoke_test(ab, ref, processed_K=K_proc, processed_M=M_proc)
    assert report["ok"], format_smoke_report(report)
    assert report["jobs"] == 6 and runner.n_jobs_run == 6
    assert report["amplitude"] == pytest.approx(0.1) and report["n_dof"] == 12
    c = report["checks"]
    assert set(c) == {"a_K0_vs_pysvk", "b_KU_vs_pysvk", "c_cubic_exactness", "c_cubic_exactness_pysvk",
                      "mass_vs_pysvk", "K0_symmetry", "nonlinear_part", "d_K0_vs_processed",
                      "mass_vs_processed"}
    assert c["a_K0_vs_pysvk"]["value"] == 0.0 and c["b_KU_vs_pysvk"]["value"] == 0.0
    assert c["c_cubic_exactness"]["value"] <= 1e-12
    assert c["d_K0_vs_processed"]["value"] <= 1e-10 and c["mass_vs_processed"]["value"] <= 1e-10
    text = format_smoke_report(report)
    assert "all checks passed" in text and "FAIL" not in text and text.count("[ok]") == 9
    json.dumps(report)  # the report is plain data


def test_smoke_test_on_real_mesh(tmp_path):
    ab, runner = mock_backend(tmp_path, "mems")
    report = smoke_test(ab, backend("mems"), seed=3)
    assert report["ok"], format_smoke_report(report)
    assert report["n_dof"] == 1521 and runner.n_jobs_run == 6
    assert "d_K0_vs_processed" not in report["checks"]


def test_smoke_test_fails_on_non_polynomial_uel(tmp_path):
    ab, _ = mock_backend(tmp_path, mode="nonpoly")
    report = smoke_test(ab, backend("single"))
    c = report["checks"]
    assert not report["ok"]
    assert c["a_K0_vs_pysvk"]["ok"] and c["mass_vs_pysvk"]["ok"]  # U = 0 is unaffected
    assert not c["c_cubic_exactness"]["ok"] and c["c_cubic_exactness"]["value"] > 1e-4
    assert not c["b_KU_vs_pysvk"]["ok"]
    assert c["c_cubic_exactness_pysvk"]["ok"]
    text = format_smoke_report(report)
    assert "RESULT: FAILED" in text and "[FAIL] c_cubic_exactness" in text


def test_smoke_test_rejects_mismatched_reference(tmp_path):
    ab, runner = mock_backend(tmp_path)
    with pytest.raises(ValueError, match="different sizes"):
        smoke_test(ab, backend("mems"))
    assert runner.n_jobs_run == 0


def test_smoke_test_flags_a_linear_test_state(tmp_path):
    ab, _ = mock_backend(tmp_path)
    report = smoke_test(ab, backend("single"), amplitude=1e-9)
    assert not report["checks"]["nonlinear_part"]["ok"]  # K_tg(U) − K is lost in round-off


# ------------------------------------------------------ scripts/abaqus_smoke_test.py

def write_config(tmp_path, kind="abaqus"):
    cmd = ", ".join(f'"{c}"' for c in MOCK).replace("\\", "/")
    cfg = tmp_path / "single_abaqus.toml"
    cfg.write_text(f'''
name = "single_abaqus"
[model]
inp = "{deck_path("single").as_posix()}"
stiffness_mtx = "{(LEGACY / DECKS["single"][1]).as_posix()}"
mass_mtx = "{(LEGACY / DECKS["single"][2]).as_posix()}"
[step]
d_ref = 1.0
[backend]
kind = "{kind}"
[abaqus]
runner = "local"
command = [{cmd}]
max_parallel = 2
user_lib_dir = "{(tmp_path / "uel" / "stiffness").as_posix()}"
mass_lib_dir = "{(tmp_path / "uel" / "mass").as_posix()}"
work_dir = "{(tmp_path / "work").as_posix()}"
retries = 0
''')
    return cfg


def run_script(args, mode=""):
    env = dict(os.environ)
    env.update(NISTEP_MOCK_MODE=mode, PYTHONUTF8="1")
    return subprocess.run([sys.executable, "-B", str(SCRIPTS / "abaqus_smoke_test.py"), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)


def test_smoke_script_passes_and_writes_json(tmp_path):
    out = tmp_path / "report.json"
    r = run_script([write_config(tmp_path), "--json", out])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "RESULT: all checks passed" in r.stdout and "runner: local" in r.stdout
    report = json.loads(out.read_text())
    assert report["ok"] and report["jobs"] == 6
    assert {"d_K0_vs_processed", "mass_vs_processed"} <= set(report["checks"])


def test_smoke_script_exit_code_on_failure(tmp_path):
    r = run_script([write_config(tmp_path)], mode="nonpoly")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[FAIL] c_cubic_exactness" in r.stdout and "RESULT: FAILED" in r.stdout


def test_smoke_script_needs_an_abaqus_config(tmp_path):
    r = run_script([write_config(tmp_path, kind="pysvk")])
    assert r.returncode == 2 and "this test needs" in r.stderr


# ----------------------------------------------------------- scripts/make_flat_uel.py

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("source", ["uel_nlmech_pk1.for", "uel_nlmech_pk1_mass_matrix.for"])
def test_make_flat_uel(tmp_path, source, capsys):
    flat = load_script("make_flat_uel")
    src = LEGACY / source
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = tmp_path / "flat" / "uel_flat.for"
    assert flat.main([str(src), str(dst)]) == 0
    assert "wrote" in capsys.readouterr().out
    text = dst.read_text(errors="replace")
    original = src.read_text(errors="replace")
    includes = [ln for ln in text.splitlines() if flat.INCLUDE.match(ln)]
    assert includes and all("ABA_PARAM.INC" in ln.upper() for ln in includes)  # Abaqus provides it
    assert "../module" not in "\n".join(ln for ln in text.splitlines() if not ln.startswith("!"))
    n_modules = sum(1 for ln in original.splitlines()
                    if flat.INCLUDE.match(ln) and "ABA_PARAM" not in ln.upper())
    assert n_modules > 0 and text.count(">>> begin include") >= n_modules
    assert len(text.splitlines()) > len(original.splitlines())
    assert "subroutine uel" in text.lower()
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before  # the original is untouched


def test_make_flat_uel_errors(tmp_path, capsys):
    flat = load_script("make_flat_uel")
    assert flat.main([]) == 2
    capsys.readouterr()
    bad = tmp_path / "bad.for"
    bad.write_text("      include 'missing.for'\n")
    with pytest.raises(RuntimeError, match="does not exist"):
        flat.flatten(str(bad))
    a, b = tmp_path / "a.for", tmp_path / "b.for"
    a.write_text("      include 'b.for'\n")
    b.write_text("      include 'a.for'\n")
    with pytest.raises(RuntimeError, match="circular include"):
        flat.flatten(str(a))
