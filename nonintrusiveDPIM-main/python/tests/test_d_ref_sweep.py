"""Tests of the ``d_ref`` sweep (PLAN.md P10.1): ``diagnostics.d_ref_sweep`` and
``EEDEngine.with_d_ref``."""

import json
import os
import sys

import numpy as np
import pytest
import scipy.linalg

from decks import backend, model
from nistep import build_engine
from nistep.backends.abaqus import AbaqusBackend
from nistep.backends.analytic import random_sparse_model
from nistep.config import StepCfg
from nistep.diagnostics import d_ref_sweep, format_d_ref_sweep, reference_backend
from nistep.engine import EEDEngine
from nistep.mtx_io import format_element_matrices, parse_element_matrices
from nistep.runners.local import LocalRunner
from nistep.store import KtgStore

MOCK = (sys.executable, "-m", "nistep.runners.mock_abaqus")


def engine_of(be, d_ref=1.0, cache_dir=None):
    return EEDEngine(be, StepCfg(d_ref=d_ref), KtgStore(be.fingerprint(), cache_dir=cache_dir))


def first_mode(be, k=0):
    """Mode ``k`` (0 = lowest) of a small model, mass-normalised (dense solver)."""
    w2, V = scipy.linalg.eigh(be.stiffness0().toarray(), be.mass().toarray())
    assert w2[0] > 0
    return V[:, k]


def mock_engine(tmp_path, mode, d_ref=1.0):
    env = dict(os.environ)
    env["NISTEP_MOCK_MODE"] = mode
    lib = {"stiffness": str(tmp_path / "uel_lib" / "stiffness"), "mass": str(tmp_path / "uel_lib" / "mass")}
    runner = LocalRunner(tmp_path / "work", MOCK, lib_dirs=lib, env=env, retries=0, max_parallel=2)
    return engine_of(AbaqusBackend(model("single"), runner), d_ref), runner


# ------------------------------------------------------------------------ with_d_ref

def test_with_d_ref_shares_backend_store_and_directions(tmp_path):
    be = random_sparse_model(12, seed=3)
    e1 = engine_of(be, 1.0, tmp_path)
    a = np.random.default_rng(0).uniform(-1, 1, 12)
    g1 = e1.quad(a, a)
    e2 = e1.with_d_ref(4.0)
    assert e2.c == pytest.approx(4 * e1.c) and e2.step.eps == e1.step.eps
    assert e2.backend is e1.backend and e2.store is e1.store and e2.directions is e1.directions
    assert e2.stats()["jobs_total"] == 0 and e1.step.d_ref == 1.0  # the original is unchanged
    g2 = e2.quad(a, a)
    assert e2.stats()["jobs_total"] == 2          # new displacements, so new jobs
    assert e2.stats()["n_directions"] == 1        # but the same direction
    assert np.max(np.abs(g2 - g1)) <= 1e-9 * np.max(np.abs(g1))
    # K_tg(0) is one stored matrix for both engines
    assert e1.K() is e2.K() or np.array_equal(e1.K().data, e2.K().data)
    assert e1.stats()["jobs_k0"] + e2.stats()["jobs_k0"] == 1
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="d_ref must be > 0"):
            e1.with_d_ref(bad)


def test_with_d_ref_updates_the_config(tmp_path):
    cfg = tmp_path / "chain.toml"
    cfg.write_text('[step]\nd_ref = 2.0\n[backend]\nkind = "analytic"\n[analytic]\n'
                   'model = "spring_chain"\nn = 6\n', encoding="utf-8")
    e = build_engine(cfg)
    e5 = e.with_d_ref(5.0)
    assert e5.config.step.d_ref == 5.0 and e.config.step.d_ref == 2.0
    assert e5.config.backend == e.config.backend and e5.config.rom == e.config.rom


# ----------------------------------------------------------------------------- sweep

def test_sweep_on_an_exact_backend():
    be = backend("single")
    phi = first_mode(be)
    eng = engine_of(be)
    d_refs = [1e-7, 1e-4, 0.1, 0.5, 1.0]
    rep = d_ref_sweep(eng, phi, d_refs)
    rows = rep["rows"]
    assert rep["exact"] and rep["n_dof"] == 12 and rep["eps"] == 0.24
    assert [r["d_ref"] for r in rows] == d_refs and all(r["status"] == "ok" for r in rows)
    assert all(r["jobs"] == 2 for r in rows)                 # K_tg(±c d) only
    assert [r["c"] for r in rows] == pytest.approx([0.24 * d for d in d_refs])
    assert rows[0]["change_G"] is None and rows[0]["plateau"] is False
    # the forms do not depend on d_ref; only the round-off does, and it grows as d_ref shrinks
    assert rows[0]["err_H"] > 1e3 * rows[-1]["err_H"]
    assert rows[0]["err_G"] > rows[-1]["err_G"]
    for r in rows[2:]:
        assert r["err_G"] <= 1e-10 and r["err_H"] <= 1e-9
        assert r["phi_G"] == pytest.approx(rows[-1]["phi_G"], rel=1e-9, abs=1e-9 * abs(rows[-1]["phi_H"]))
        assert r["phi_H"] == pytest.approx(rows[-1]["phi_H"], rel=1e-9)
    assert rows[-1]["plateau"] and rows[-2]["plateau"]
    assert rows[1]["plateau"] is False                       # the step from 1e-7 is not converged
    # the element stays far from inversion at these amplitudes
    assert all(0.2 < r["min_det_F"] <= 1.0 + 1e-12 for r in rows)
    assert rows[0]["min_det_F"] > rows[-1]["min_det_F"]
    exact = be.exact_cubic(phi, phi, phi)
    assert rows[-1]["H_max"] == pytest.approx(np.max(np.abs(exact)), rel=1e-9)
    assert rows[-1]["phi_H"] == pytest.approx(phi @ exact, rel=1e-9)
    json.dumps(rep)                                          # plain data
    text = format_d_ref_sweep(rep)
    assert "err G" in text and text.count("plateau") >= 3 and "REFUSED" not in text


def test_sweep_refuses_a_d_ref_that_inverts_elements():
    be = backend("single")
    # The fourth mode stretches and squeezes the element. (The three lowest ones bend or
    # shear it, and for those det F does not fall as the amplitude grows.)
    phi = first_mode(be, 3)
    eng = engine_of(be)
    rep = d_ref_sweep(eng, phi, [1.0, 5.0, 2.0])
    r1, r5, r2 = rep["rows"]
    assert r1["status"] == "ok" and r5["status"] == "refused" and r5["jobs"] == 0
    assert r1["min_det_F"] > r2["min_det_F"] > 0.2 >= r5["min_det_F"] and "min det F" in r5["reason"]
    assert r5["phi_G"] is None and r5["err_H"] is None
    assert r2["change_G"] is not None                        # compared with the last row that ran
    assert "REFUSED: min det F" in format_d_ref_sweep(rep)
    assert eng.store.stats()["store_puts"] == 1 + 2 * 2      # K_tg(0) and two jobs per row that ran
    # without the limit the row runs (the numpy element is polynomial, so it is still exact)
    rep = d_ref_sweep(eng, phi, [5.0], det_f_min=None)
    assert rep["rows"][0]["status"] == "ok" and rep["rows"][0]["err_H"] <= 1e-9


def test_sweep_on_an_analytic_backend_has_no_det_f():
    be = random_sparse_model(15, seed=5)
    assert reference_backend(be) is be
    phi = np.random.default_rng(2).uniform(-1, 1, 15)
    rep = d_ref_sweep(engine_of(be), phi, [0.01, 1.0, 100.0])
    assert rep["exact"] and all(r["min_det_F"] is None and r["status"] == "ok" for r in rep["rows"])
    assert rep["rows"][-1]["err_G"] <= 1e-10 and rep["rows"][-1]["err_H"] <= 1e-10


def test_sweep_input_checks():
    be = random_sparse_model(8, seed=1)
    eng = engine_of(be)
    with pytest.raises(ValueError, match="length 8"):
        d_ref_sweep(eng, np.ones(7), [1.0])
    with pytest.raises(ValueError, match="phi is zero"):
        d_ref_sweep(eng, np.zeros(8), [1.0])
    for bad in ([], [1.0, 0.0], [-2.0]):
        with pytest.raises(ValueError, match="positive numbers"):
            d_ref_sweep(eng, np.ones(8), bad)
    with pytest.raises(ValueError, match="different size"):
        d_ref_sweep(eng, np.ones(8), [1.0], reference=random_sparse_model(9, seed=1))


# ------------------------------------------------- rounded matrices (the .mtx noise floor)

def test_rounded_matrix_output():
    Ke = np.random.default_rng(0).uniform(-1e11, 1e11, size=(1, 24, 24))
    nodes = np.arange(1, 9)[None, :]
    full = parse_element_matrices(format_element_matrices([1], nodes, Ke))
    assert np.array_equal(full.matrices, Ke)                 # 18 digits keep every double
    rounded = parse_element_matrices(format_element_matrices([1], nodes, Ke, sig_digits=8))
    err = np.max(np.abs(rounded.matrices - Ke) / np.abs(Ke))
    assert 1e-9 < err <= 5.0001e-8
    for bad in (0, 19):
        with pytest.raises(ValueError, match="sig_digits"):
            format_element_matrices([1], nodes, Ke, sig_digits=bad)


def test_sweep_shows_the_noise_floor_of_rounded_matrices(tmp_path):
    """With 8-digit matrices the error falls as d_ref grows: 1/c for Ĝ and 1/c² for Ĥ."""
    eng, runner = mock_engine(tmp_path, "digits=8")
    ref = reference_backend(eng.backend)
    assert type(ref).__name__ == "PySVKBackend" and ref.n_dof == 12
    phi = first_mode(ref)
    rep = d_ref_sweep(eng, phi, [1e-3, 1e-2, 1e-1, 1.0])
    rows = rep["rows"]
    assert rep["exact"] and all(r["status"] == "ok" and r["jobs"] == 2 for r in rows)
    assert runner.n_jobs_run == 1 + 2 * 4
    errG, errH = [r["err_G"] for r in rows], [r["err_H"] for r in rows]
    assert errH[0] > 1e-3 and errH[-1] < 1e-5               # useless at 1e-3, fine at 1
    assert errH[0] > 1e4 * errH[-1] and errG[0] > 1e1 * errG[-1]
    assert all(h >= g for g, h in zip(errG, errH))           # Ĥ needs the larger d_ref
    assert not rows[1]["plateau"] and not rows[2]["plateau"]
    # the same sweep with full-precision output is at round-off level
    eng18, _ = mock_engine(tmp_path / "full", "")
    rows18 = d_ref_sweep(eng18, phi, [1e-1, 1.0])["rows"]
    assert all(r["err_H"] <= 1e-9 for r in rows18) and rows18[-1]["plateau"]
    assert errH[-1] > 1e3 * rows18[-1]["err_H"]


def test_sweep_records_a_failed_job_and_goes_on(tmp_path):
    marker = tmp_path / "failed_once"
    eng, _ = mock_engine(tmp_path, f"fail_once={marker.as_posix()}")
    eng.backend.runner.max_parallel = 1
    phi = first_mode(reference_backend(eng.backend))
    rep = d_ref_sweep(eng, phi, [0.5, 1.0, 2.0])
    r0, r1, r2 = rep["rows"]
    assert r0["status"] == "failed" and r0["reason"] and r0["phi_G"] is None
    assert r1["status"] == "ok" and r1["change_G"] is None   # nothing to compare with yet
    assert r2["status"] == "ok" and r2["plateau"]
    assert "FAILED:" in format_d_ref_sweep(rep)
