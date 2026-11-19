"""P8.3, P8.4, P8.7: LocalRunner, AbaqusBackend and ReplayBackend, with a mock ``abaqus``.

The mock (``nistep.runners.mock_abaqus``) is called exactly like Abaqus: it reads the job
deck in the job directory and writes the raw element-matrix file, computed with PySVK.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import nistep
from decks import LEGACY, DECKS, backend, deck_path, model
from nistep.backends.abaqus import AbaqusBackend
from nistep.backends.analytic import random_sparse_model
from nistep.backends.replay import ReplayBackend, ReplayMiss
from nistep.config import StepCfg
from nistep.engine import EEDEngine
from nistep.runners.base import AbaqusJob, RunnerError
from nistep.runners.local import LocalRunner
from nistep.store import KtgStore

MOCK = (sys.executable, "-m", "nistep.runners.mock_abaqus")


def mock_env(mode=""):
    env = dict(os.environ)
    env["NISTEP_MOCK_MODE"] = mode
    return env


def make(tmp_path, name="single", mode="", **kw):
    lib = {"stiffness": str(tmp_path / "uel_lib" / "stiffness"), "mass": str(tmp_path / "uel_lib" / "mass")}
    kw.setdefault("retries", 0)
    runner = LocalRunner(tmp_path / "work", MOCK, lib_dirs=lib, env=mock_env(mode), **kw)
    return AbaqusBackend(model(name), runner), runner


def rel(x, y):
    return np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300)


def rand_u(be, seed=0, scale=0.05):
    return np.random.default_rng(seed).uniform(-scale, scale, size=be.n_dof)


# ----------------------------------------------------------------------- happy path

def test_tangent_matches_pysvk_exactly(tmp_path):
    ab, runner = make(tmp_path)
    ref = backend("single")
    U = rand_u(ab)
    K = ab.tangent(U)
    Kref = ref.tangent(U)
    assert np.array_equal(K.indices, Kref.indices) and np.array_equal(K.data, Kref.data)
    K0 = ab.stiffness0()
    assert np.array_equal(K0.data, ref.stiffness0().data)
    assert np.array_equal(K0.indptr, K.indptr)  # one fixed pattern
    assert not any((tmp_path / "work").iterdir())  # successful job directories are removed
    assert runner.n_jobs_run == 2 and runner.n_attempts == 2 and ab.n_jobs == 2


def test_mass_from_uel_library_and_from_processed_file(tmp_path):
    ab, _ = make(tmp_path)
    ref = backend("single")
    assert np.array_equal(ab.mass().data, ref.mass().data)
    runner = LocalRunner(tmp_path / "w2", MOCK, env=mock_env())
    ab2 = AbaqusBackend(model("single"), runner, mass_from="processed",
                        mass_mtx=LEGACY / DECKS["single"][2])
    M2 = ab2.mass()
    assert abs(M2 - ref.mass()).max() <= 1e-12 * abs(ref.mass()).max()
    assert runner.n_jobs_run == 0  # no Abaqus job for the mass
    assert ab2.fingerprint() != ab.fingerprint()


def test_job_names_and_decks(tmp_path):
    ab, _ = make(tmp_path)
    U = rand_u(ab, 1)
    j1, j2, jm = ab.make_job(U), ab.make_job(U.copy()), ab.make_job(U, "mass")
    assert j1.name == j2.name and j1.inp_text == j2.inp_text  # content-addressed
    assert jm.name != j1.name and ab.make_job(-U).name != j1.name
    assert len(j1.name) == 13 and j1.name.startswith("j") and j1.matrix_file == j1.name
    assert f"File Name={j1.name}," in j1.inp_text and j1.n_elements == 1
    with pytest.raises(ValueError):
        AbaqusJob("bad name", "x", "m", 1)
    with pytest.raises(ValueError):
        AbaqusJob("ok", "x", "m", 1, library="other")


def test_parallel_batch_keeps_order_and_symmetric_storage(tmp_path):
    ab, runner = make(tmp_path, mode="symmetric", max_parallel=3)
    ref = backend("single")
    Us = [rand_u(ab, s) for s in range(5)]
    Ks = ab.tangent_many(Us)
    for U, K in zip(Us, Ks):  # tangent of a hyperelastic element is symmetric
        assert rel(K.toarray(), ref.tangent(U).toarray()) <= 1e-13
    assert runner.n_jobs_run == 5


def test_real_mesh_job(tmp_path):
    ab, _ = make(tmp_path, "cc")
    ref = backend("cc")
    U = rand_u(ab, 2, 1e-4)
    assert np.array_equal(ab.tangent(U).data, ref.tangent(U).data)


# ------------------------------------------------------------------------- failures

@pytest.mark.parametrize("mode,reason", [
    ("fail", "exit code 1"),
    ("no_mtx", "no element-matrix file"),
    ("truncated", "0 element blocks, expected 1"),
    ("no_sta", "no status file"),
    ("bad_sta", "status file lacks"),
])
def test_failed_job_raises_and_keeps_directory(tmp_path, mode, reason):
    ab, runner = make(tmp_path, mode=mode, retries=1)
    with pytest.raises(RunnerError) as exc:
        ab.tangent(rand_u(ab))
    err = exc.value
    assert reason in err.reason and err.name.startswith("j")
    assert err.work_dir is not None and err.work_dir.is_dir()  # kept for inspection
    assert (err.work_dir / f"{err.name}.inp").is_file()
    assert runner.n_attempts == 2  # retried once
    if mode == "fail":
        assert "mock failure" in err.log_tail and "***ERROR" in str(err)


def test_retry_succeeds_after_one_failure(tmp_path):
    marker = tmp_path / "failed_once"
    ab, runner = make(tmp_path, mode=f"fail_once={marker}", retries=1)
    K = ab.tangent(rand_u(ab))
    assert marker.exists() and runner.n_attempts == 2
    assert np.array_equal(K.data, backend("single").tangent(rand_u(ab)).data)


def test_timeout(tmp_path):
    ab, _ = make(tmp_path, mode="sleep=20", timeout_s=2.0)
    with pytest.raises(RunnerError, match="timeout after 2 s"):
        ab.tangent(rand_u(ab))


def test_wrong_library_detected(tmp_path):
    lib = {"stiffness": str(tmp_path / "uel_lib" / "mass")}  # the mass UEL by mistake
    runner = LocalRunner(tmp_path / "work", MOCK, lib_dirs=lib, env=mock_env())
    ab = AbaqusBackend(model("single"), runner)
    with pytest.raises(RuntimeError, match="matrix type MASS, expected STIFFNESS"):
        ab.stiffness0()


def test_missing_command(tmp_path):
    runner = LocalRunner(tmp_path / "work", ("no_such_abaqus_command_xyz",), retries=0)
    ab = AbaqusBackend(model("single"), runner)
    with pytest.raises(RunnerError):
        ab.stiffness0()


def test_keep_modes(tmp_path):
    ab, _ = make(tmp_path, keep="all")
    ab.stiffness0()
    dirs = list((tmp_path / "work").iterdir())
    assert len(dirs) == 1 and (dirs[0] / "abaqus_v6.env").read_text().startswith("usub_lib_dir = r'")
    ab2, _ = make(tmp_path / "b", mode="fail", keep="none")
    with pytest.raises(RunnerError) as exc:
        ab2.stiffness0()
    assert exc.value.work_dir is None and not any((tmp_path / "b" / "work").iterdir())


# --------------------------------------------------------------- engine on the backend

def test_engine_on_abaqus_backend(tmp_path):
    ab, runner = make(tmp_path, max_parallel=4)
    ref = backend("single")
    engine = EEDEngine(ab, StepCfg(d_ref=1.0), KtgStore(ab.fingerprint(), cache_dir=tmp_path / "cache"))
    rng = np.random.default_rng(3)
    a = rng.normal(size=ab.n_dof) + 1j * rng.normal(size=ab.n_dof)
    b = rng.normal(size=ab.n_dof) + 1j * rng.normal(size=ab.n_dof)
    assert rel(engine.quad(a, b), ref.exact_quad(a, b)) <= 1e-8
    assert rel(engine.cubic(a, a, b), ref.exact_cubic(a, a, b)) <= 1e-8
    n = engine.stats()["jobs_total"]
    assert runner.n_jobs_run == n + 1  # + K_tg(0)
    assert ab.exact_quad is None
    # resume in a "new process" with the same disk cache: no Abaqus job runs again
    ab2, runner2 = make(tmp_path / "second")
    assert ab2.fingerprint() == ab.fingerprint()
    engine2 = EEDEngine(ab2, StepCfg(d_ref=1.0), KtgStore(ab2.fingerprint(), cache_dir=tmp_path / "cache"))
    assert np.array_equal(engine2.quad(a, b), engine.quad(a, b))
    assert np.array_equal(engine2.cubic(a, a, b), engine.cubic(a, a, b))
    assert runner2.n_jobs_run == 0 and engine2.stats()["jobs_total"] == 0
    # the UEL provenance is part of the model identity
    src = tmp_path / "uel.for"
    src.write_text("      subroutine uel\n")
    ab3 = AbaqusBackend(model("single"), runner, provenance=[src])
    ab4 = AbaqusBackend(model("single"), runner, provenance=["D:/host/uel_lib/stiffness"])
    assert len({ab.fingerprint(), ab3.fingerprint(), ab4.fingerprint()}) == 3
    src.write_text("      subroutine uel2\n")
    assert AbaqusBackend(model("single"), runner, provenance=[src]).fingerprint() != ab3.fingerprint()


def test_build_engine_from_config_with_mock(tmp_path):
    cfg = tmp_path / "single_abaqus.toml"
    cmd = ", ".join(f'"{c}"' for c in MOCK).replace("\\", "/")
    cfg.write_text(f'''
name = "single_abaqus"
[model]
inp = "{deck_path("single").as_posix()}"
[step]
d_ref = 1.0
[backend]
kind = "abaqus"
[abaqus]
runner = "local"
command = [{cmd}]
max_parallel = 2
mass_lib_dir = "{(tmp_path / "uel" / "mass").as_posix()}"
work_dir = "{(tmp_path / "work").as_posix()}"
retries = 0
[cache]
dir = "{(tmp_path / "cache").as_posix()}"
''')
    os.environ["NISTEP_MOCK_MODE"] = ""
    engine = nistep.build_engine(cfg)
    ref = backend("single")
    assert engine.backend.kind == "abaqus" and engine.n == 12
    assert np.array_equal(engine.K().data, ref.stiffness0().data)
    assert np.array_equal(engine.M().data, ref.mass().data)
    v = np.random.default_rng(0).normal(size=12)
    assert rel(engine.quad(v, v), ref.exact_quad(v, v)) <= 1e-9

    # P8.7: replay the recorded cache without any backend
    replay_cfg = tmp_path / "replay.toml"
    replay_cfg.write_text(f'''
[step]
d_ref = 1.0
[backend]
kind = "replay"
[cache]
dir = "{(tmp_path / "cache").as_posix()}"
''')
    rep = nistep.build_engine(replay_cfg)
    assert rep.backend.kind == "replay" and rep.backend.recorded_kind == "abaqus"
    assert rep.backend.fingerprint() == engine.backend.fingerprint()
    assert np.array_equal(rep.K().data, engine.K().data) and np.array_equal(rep.M().data, engine.M().data)
    assert np.array_equal(rep.quad(v, v), engine.quad(v, v))
    assert rep.stats()["jobs_total"] == 0 and rep.stats()["jobs_k0"] == 0
    # a new vector contracted with the recorded probe needs no new job ...
    w = np.random.default_rng(9).normal(size=12)
    assert rel(rep.quad(w, v), ref.exact_quad(w, v)) <= 1e-9
    # ... but probing along the new vector does, and that state was never recorded
    with pytest.raises(ReplayMiss) as exc:
        rep.quad(w, w)
    assert len(exc.value.key) == 64


def test_job_files_do_not_depend_on_the_locale_encoding(tmp_path):
    """The decks hold a UTF-8 'ν' in a comment. With Python's UTF-8 mode off (as inside
    Julia's embedded Python) the Windows default encoding cannot write it."""
    assert "\u03bd" in deck_path("single").read_text(encoding="utf-8")
    code = '''
import sys
from pathlib import Path
import numpy as np
from nistep.backends.abaqus import AbaqusBackend
from nistep.backends.pysvk import PySVKBackend
from nistep.inp_io import read_inp
from nistep.runners.local import LocalRunner
assert not sys.flags.utf8_mode
deck, tmp = Path(sys.argv[1]), Path(sys.argv[2])
mock = (sys.executable, "-X", "utf8=0", "-m", "nistep.runners.mock_abaqus")
lib = {"stiffness": str(tmp / "uel" / "stiffness"), "mass": str(tmp / "uel" / "mass")}
runner = LocalRunner(tmp / "work", mock, lib_dirs=lib, retries=0, keep="all")
model = read_inp(deck)
ab, ref = AbaqusBackend(model, runner), PySVKBackend(model)
U = np.random.default_rng(0).uniform(-0.05, 0.05, size=ab.n_dof)
assert np.array_equal(ab.tangent(U).data, ref.tangent(U).data)
assert np.array_equal(ab.mass().data, ref.mass().data)
decks = list((tmp / "work").glob("*/*.inp"))
assert len(decks) == 2 and all(b"\\xce\\xbd" in p.read_bytes() for p in decks)
print("LOCALE-OK")
'''
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "NISTEP_MOCK_MODE")}
    r = subprocess.run([sys.executable, "-X", "utf8=0", "-B", "-c", code, str(deck_path("single")),
                        str(tmp_path)], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "LOCALE-OK" in r.stdout, r.stdout + r.stderr


# ---------------------------------------------------------------------------- replay

def test_replay_backend_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayBackend(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="found none"):
        ReplayBackend(tmp_path / "empty")
    # two recorded models: the fingerprint selects one
    for seed in (1, 2):
        be = random_sparse_model(10, seed=seed)
        e = EEDEngine(be, StepCfg(d_ref=1.0), KtgStore(be.fingerprint(), cache_dir=tmp_path / "two"))
        e.M()
        e.K()
    with pytest.raises(ValueError, match="exactly one"):
        ReplayBackend(tmp_path / "two")
    be = random_sparse_model(10, seed=2)
    rb = ReplayBackend(tmp_path / "two", be.fingerprint())
    assert rb.n_dof == 10 and rb.fingerprint() == be.fingerprint()
    assert np.array_equal(rb.mass().toarray(), be.mass().toarray())
    with pytest.raises(ReplayMiss):
        rb.tangent(np.ones(10))
