"""Resume after an interruption (PLAN.md P9.2) and the memory estimate (P9.4).

The hard test kills a separate Python process in the middle of its tangent jobs, restarts
it on the same cache folder and checks that no job ran twice. The MORFE-level version of the
same check is in ``julia/MORFEStEP/test/prefetch.jl``.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import nistep
from nistep.backends.analytic import random_sparse_model
from nistep.backends.wrappers import SimulatedFailure, ThrottledBackend
from nistep.basis_mode import n_eed, polarised_forms
from nistep.config import ConfigError, StepCfg, config_from_dict
from nistep.engine import EEDEngine
from nistep.store import KtgStore
from resume_driver import basis, forms_as_arrays

DRIVER = Path(__file__).with_name("resume_driver.py")
N_DOF, N_VEC = 30, 6


def write_config(tmp_path, delay_s=0.0, cache=True):
    cfg = tmp_path / ("slow.toml" if delay_s else "fast.toml")
    cfg.write_text(f'''
[step]
d_ref = 1.0
[backend]
kind = "analytic"
delay_s = {delay_s}
[analytic]
model = "random_sparse"
n = {N_DOF}
seed = 3
''' + (f'[cache]\ndir = "{(tmp_path / "cache").as_posix()}"\n' if cache else ""))
    return cfg


def cache_root(tmp_path) -> Path:
    roots = [p for p in (tmp_path / "cache").iterdir() if p.is_dir()]
    assert len(roots) == 1
    return roots[0]


def manifest_lines(tmp_path) -> int:
    try:
        return len((cache_root(tmp_path) / "manifest.jsonl").read_text().splitlines())
    except (OSError, AssertionError):
        return 0


def run_driver(cfg, out, wait=True):
    env = dict(os.environ, PYTHONUTF8="1")
    cmd = [sys.executable, "-B", str(DRIVER), str(cfg), str(N_VEC), str(out)]
    if wait:
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------- hard kill + resume

def test_killed_run_resumes_without_repeating_a_job(tmp_path):
    cfg = write_config(tmp_path, delay_s=0.15)
    total = n_eed(N_VEC) + 1  # tangent jobs + the K_tg(0) job
    out = tmp_path / "forms.npz"

    proc = run_driver(cfg, out, wait=False)
    try:
        deadline = time.monotonic() + 120
        while manifest_lines(tmp_path) < 6:
            assert proc.poll() is None, "the driver ended before it could be interrupted"
            assert time.monotonic() < deadline, "the driver did not start its jobs"
            time.sleep(0.02)
    finally:
        proc.kill()
        proc.wait()
    root = cache_root(tmp_path)
    store = KtgStore(root.name)  # only used to read the manifest
    store.root = root
    keys_before = store.manifest_keys()
    on_disk_before = len(list((root / "jobs").glob("*.npz")))
    assert 6 <= len(keys_before) < total and not out.exists()
    assert len(set(keys_before)) == len(keys_before)

    done = run_driver(cfg, out)
    assert done.returncode == 0, done.stdout + done.stderr
    keys = store.manifest_keys()
    assert len(set(keys)) == len(keys), "a job ran twice"
    assert keys[:len(keys_before)] == keys_before
    assert len(list((root / "jobs").glob("*.npz"))) == total
    # a kill between writing a job file and its manifest line leaves at most one line missing
    assert total - 1 <= len(keys) <= total
    with np.load(out) as z:
        resumed = {k: z[k] for k in z.files}
    assert int(resumed.pop("jobs_k0")) == 0  # K_tg(0) came from the cache
    assert int(resumed.pop("jobs_total")) == total - on_disk_before

    # the resumed result equals a clean run without cache
    engine = nistep.build_engine(write_config(tmp_path, cache=False))
    G, H = polarised_forms(engine, basis(N_DOF, N_VEC))
    assert engine.stats()["jobs_total"] == n_eed(N_VEC)
    clean = forms_as_arrays(G, H)
    assert set(clean) == set(resumed)
    for k, v in clean.items():
        assert np.max(np.abs(resumed[k] - v)) <= 1e-12 * np.max(np.abs(v)), k

    # a third run finds everything in the cache
    again = run_driver(cfg, tmp_path / "again.npz")
    assert again.returncode == 0
    with np.load(tmp_path / "again.npz") as z:
        assert int(z["jobs_total"]) == 0 and int(z["jobs_k0"]) == 0
        for k, v in resumed.items():
            assert np.array_equal(z[k], v), k
    assert store.manifest_keys() == keys


# ------------------------------------------------------------------- simulated failure

def make_engine(tmp_path, fail_after=None, delay_s=0.0):
    inner = random_sparse_model(N_DOF, seed=3)
    be = ThrottledBackend(inner, delay_s=delay_s, fail_after=fail_after)
    store = KtgStore(be.fingerprint(), cache_dir=tmp_path / "cache")
    return EEDEngine(be, StepCfg(d_ref=1.0), store), be


def test_failure_mid_batch_then_resume(tmp_path):
    V = basis(N_DOF, 4)
    engine, be = make_engine(tmp_path, fail_after=5)
    with pytest.raises(SimulatedFailure, match="after 5 jobs"):
        polarised_forms(engine, V, max_parallel=2)
    # batches of 2: the fifth job's batch-mate was lost with the failure
    assert be.jobs_done == 5 and engine.stats()["jobs_total"] == 4
    keys_before = engine.store.manifest_keys()
    assert len(keys_before) == 4 + 1

    engine2, be2 = make_engine(tmp_path)
    assert len(engine2.directions) == len(engine.directions) > 0  # loaded from disk
    G, H = polarised_forms(engine2, V, max_parallel=2)
    keys = engine2.store.manifest_keys()
    assert len(keys) == len(set(keys)) == n_eed(4) + 1
    assert engine2.stats()["jobs_total"] == be2.jobs_done == n_eed(4) - 4
    ref = random_sparse_model(N_DOF, seed=3)
    for (i, j), g in G.items():
        ex = ref.exact_quad(V[:, i], V[:, j])
        assert np.max(np.abs(g - ex)) <= 1e-9 * np.max(np.abs(ex))
    for (i, j, k), h in H.items():
        ex = ref.exact_cubic(V[:, i], V[:, j], V[:, k])
        assert np.max(np.abs(h - ex)) <= 1e-9 * np.max(np.abs(ex))


def test_throttled_backend_forwards_and_delays():
    inner = random_sparse_model(8, seed=1)
    be = ThrottledBackend(inner, delay_s=0.05)
    assert be.n_dof == 8 and be.fingerprint() == inner.fingerprint() and be.kind == inner.kind
    U = np.random.default_rng(0).normal(size=8) * 0.1
    t0 = time.perf_counter()
    Ks = be.tangent_many([U, -U])
    assert time.perf_counter() - t0 >= 0.1 and be.jobs_done == 2
    assert abs(Ks[0] - inner.tangent(U)).max() == 0 and abs(be.tangent(-U) - Ks[1]).max() == 0
    assert abs(be.stiffness0() - inner.stiffness0()).max() == 0
    v = np.ones(8)
    assert np.array_equal(be.exact_quad(v, v), inner.exact_quad(v, v))
    with pytest.raises(ValueError):
        ThrottledBackend(inner, delay_s=-1)
    with pytest.raises(AttributeError):
        be.no_such_attribute


# ------------------------------------------------------------------------- config

def base_dict(**sections):
    d = {"step": {"d_ref": 1.0}}
    d.update(sections)
    return d


def test_config_delay_and_prefetch(tmp_path):
    cfg = config_from_dict(base_dict(backend={"kind": "analytic", "delay_s": 0.01}))
    be = nistep.build_backend(cfg)
    assert isinstance(be, ThrottledBackend) and be.delay_s == 0.01
    assert not isinstance(nistep.build_backend(config_from_dict(base_dict())), ThrottledBackend)

    assert config_from_dict(base_dict()).use_prefetch is False  # auto: off for analytic
    assert config_from_dict(base_dict(engine={"prefetch": "on"})).use_prefetch is True
    deck = Path(__file__).resolve().parents[2] / "test" / "Meu_input_file.inp"
    abq = base_dict(backend={"kind": "abaqus"}, model={"inp": str(deck)})
    assert config_from_dict(abq).use_prefetch is True  # auto: on for abaqus
    abq["engine"] = {"prefetch": "off"}
    assert config_from_dict(abq).use_prefetch is False
    with pytest.raises(ConfigError, match="prefetch"):
        config_from_dict(base_dict(engine={"prefetch": "yes"}))
    with pytest.raises(ConfigError, match="delay_s"):
        config_from_dict(base_dict(backend={"delay_s": -1.0}))


# ------------------------------------------------------------------- memory estimate

def test_memory_report(tmp_path):
    be = random_sparse_model(N_DOF, seed=3)
    V = basis(N_DOF, 3)
    # with a disk cache the LRU bounds the memory
    store = KtgStore(be.fingerprint(), cache_dir=tmp_path / "cache", lru_items=3)
    engine = EEDEngine(be, StepCfg(d_ref=1.0), store)
    polarised_forms(engine, V)
    s = engine.stats()
    nnz = engine.K().nnz
    assert s["store_nnz"] == nnz and s["store_MB_per_matrix"] == pytest.approx(nnz * 8 / 2**20)
    assert s["store_items_memory"] == 3 and s["store_items_disk"] == n_eed(3) + 1
    assert s["store_ram_limit_MB"] == pytest.approx(3 * nnz * 8 / 2**20)
    assert s["store_ram_estimate_MB"] <= s["store_ram_limit_MB"]
    text = store.memory_report()
    assert "at most 3 kept in memory" in text and f"{nnz} stored entries" in text
    # without one, every matrix stays in memory
    store2 = KtgStore(be.fingerprint(), lru_items=3)
    engine2 = EEDEngine(be, StepCfg(d_ref=1.0), store2)
    polarised_forms(engine2, V)
    s2 = engine2.stats()
    assert s2["store_items_memory"] == n_eed(3) + 1 and s2["store_ram_limit_MB"] is None
    assert "no disk cache" in store2.memory_report()


def test_manifest_tolerates_a_torn_line(tmp_path):
    be = random_sparse_model(8, seed=1)
    store = KtgStore(be.fingerprint(), cache_dir=tmp_path / "cache")
    U = np.zeros(8)
    key = store.put(U, be.stiffness0())
    with open(store.root / "manifest.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"key": "abc", "ti')  # a process killed in the middle of a write
    assert store.manifest_keys() == [key]
