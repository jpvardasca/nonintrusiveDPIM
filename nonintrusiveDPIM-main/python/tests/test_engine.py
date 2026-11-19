"""P2.4: EEDEngine against exact analytic polarised forms."""

from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

import nistep
from nistep.backends.analytic import random_sparse_model, two_dof_example
from nistep.config import StepCfg
from nistep.engine import EEDEngine
from nistep.store import KtgStore

N = 40


def make_engine(backend=None, mode="lazy", cache_dir=None, d_ref=1.0, debug=False, lru=8):
    backend = backend or random_sparse_model(N, seed=11, density=0.1)
    store = KtgStore(backend.fingerprint(), cache_dir=cache_dir, lru_items=lru)
    return EEDEngine(backend, StepCfg(d_ref=d_ref), store, mode=mode, debug=debug)


def rel(x, y):
    return np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300)


def cvec(rng, n=N):
    return rng.normal(size=n) + 1j * rng.normal(size=n)


def jobs(e):
    return e.stats()["jobs_total"]


def test_c_is_eps_dref():
    e = make_engine(d_ref=1.0)
    assert e.c == pytest.approx(0.24)


def test_a_real_quad_exact():
    e = make_engine()
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=N), rng.normal(size=N)
    assert rel(e.quad(a, b), e.backend.exact_quad(a, b)) <= 1e-10
    assert jobs(e) == 2


def test_b_complex_quad_and_cubic_exact():
    e = make_engine()
    rng = np.random.default_rng(1)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    assert rel(e.quad(a, b), e.backend.exact_quad(a, b)) <= 1e-9
    assert rel(e.cubic(a, b, c), e.backend.exact_cubic(a, b, c)) <= 1e-9
    assert rel(e.cubic(a, a, c), e.backend.exact_cubic(a, a, c)) <= 1e-9
    assert rel(e.cubic(a, a, a), e.backend.exact_cubic(a, a, a)) <= 1e-9


def test_c_symmetry():
    e = make_engine()
    rng = np.random.default_rng(2)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    q = e.quad(a, b)
    assert rel(e.quad(b, a), q) <= 1e-9
    h = e.cubic(a, b, c)
    for p in permutations((a, b, c)):
        assert rel(e.cubic(*p), h) <= 1e-9


def test_d_conjugate_reuse():
    e = make_engine()
    rng = np.random.default_rng(3)
    a, b = cvec(rng), cvec(rng)
    e.quad(a, b)
    n0 = jobs(e)
    assert n0 == 4
    out = e.quad(np.conj(a), b)
    assert jobs(e) == n0
    assert rel(out, e.backend.exact_quad(np.conj(a), b)) <= 1e-9


def test_e_proportional_parts_two_jobs():
    e = make_engine()
    p = np.random.default_rng(4).normal(size=N)
    b = cvec(np.random.default_rng(5))
    a = (1 + 2j) * p
    out = e.quad(a, b)
    assert jobs(e) == 2
    assert rel(out, e.backend.exact_quad(a, b)) <= 1e-9


def test_f_zero_vectors_and_parts():
    e = make_engine()
    rng = np.random.default_rng(6)
    z = np.zeros(N, dtype=complex)
    a = cvec(rng)
    assert np.all(e.quad(z, a) == 0) and np.all(e.cubic(a, z, a) == 0)
    assert jobs(e) == 0 and len(e.vectors) == 0
    real = rng.normal(size=N) + 0j  # zero imaginary part
    e.quad(real, a)
    assert jobs(e) == 2
    tiny_imag = rng.normal(size=N) + 1e-15j * rng.normal(size=N)  # below zero_tol
    e2 = make_engine()
    e2.quad(tiny_imag, a)
    assert jobs(e2) == 2


def test_g_record_mode_then_execute():
    e = make_engine(mode="record")
    rng = np.random.default_rng(7)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    assert np.all(e.quad(a, b) == 0)
    assert np.all(e.cubic(a, b, c) == 0)
    assert e.pending_count() > 0 and jobs(e) == 0
    n = e.execute_pending(max_parallel=3)
    # 4 probes of a (Re/Im, ±) + 4 mixed jobs + one s-signed probe per direction of b
    # (2..4 distinct jobs, depending on the pair signs)
    assert n == jobs(e) and 10 <= n <= 12
    assert e.stats()["jobs_pair"] == 4
    assert e.pending_count() == 0
    q = e.quad(a, b)
    h = e.cubic(a, b, c)
    assert jobs(e) == n
    assert rel(q, e.backend.exact_quad(a, b)) <= 1e-9
    assert rel(h, e.backend.exact_cubic(a, b, c)) <= 1e-9


def test_h_two_dof_example():
    e = make_engine(backend=two_dof_example())
    V1 = np.ones(2)
    np.testing.assert_allclose(e.quad(V1, V1).real, [20, 20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(e.cubic(V1, V1, V1).real, [10, 10], rtol=0, atol=1e-12)
    np.testing.assert_allclose((e.K() @ V1), [10, 10])
    assert jobs(e) == 2  # ±c V1


def test_i_linearity():
    e = make_engine()
    rng = np.random.default_rng(8)
    a, a2, b = cvec(rng), cvec(rng), cvec(rng)
    lhs = e.quad(2 * a + 3 * a2, b)
    rhs = 2 * e.quad(a, b) + 3 * e.quad(a2, b)
    assert rel(lhs, rhs) <= 1e-9


def test_job_counts_cubic():
    e = make_engine()
    rng = np.random.default_rng(9)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    e.cubic(a, a, c)  # repeated complex arg: 4 probes + 1 pair (Re a, Im a)
    assert e.stats()["jobs_probe"] == 4 and e.stats()["jobs_pair"] == 1
    e2 = make_engine()
    e2.cubic(a, b, c)  # distinct: 4 pairs; + probes of a (Re, Im) and s-signed probes of b
    assert e2.stats()["jobs_pair"] == 4 and 4 <= e2.stats()["jobs_probe"] <= 6
    e3 = make_engine()
    r = rng.normal(size=N)
    e3.cubic(r, r, c)  # real repeated: diagonal only
    assert jobs(e3) == 2


def test_pair_negative_sign_branch():
    e = make_engine()
    w = e._w
    pos = np.nonzero(w > 0)[0]
    i, j = next((i, j) for i in pos for j in pos if i != j and 0.6 < w[i] / w[j] < 1.6)
    p1 = np.zeros(N); p1[i], p1[j] = 1.0, -0.5
    p2 = np.zeros(N); p2[i], p2[j] = -0.5, 1.0
    v = cvec(np.random.default_rng(10))
    out = e.cubic(p1, p2, v)
    signs = [s for _, s in e._pair_jobs.values()]
    assert signs == [-1.0]
    assert rel(out, e.backend.exact_cubic(p1, p2, v)) <= 1e-9


def test_probe_choice_deterministic_lowest_id():
    e = make_engine()
    rng = np.random.default_rng(12)
    a, b = cvec(rng), cvec(rng)
    e.quad(a, b)  # probe a (id 0)
    n0 = jobs(e)
    e.quad(b, a)  # still probe a
    assert jobs(e) == n0


def test_required_jobs_lists_without_running():
    e = make_engine()
    rng = np.random.default_rng(13)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    Us = e.required_jobs("cubic", a, b, c)
    assert 8 <= len(Us) <= 10 and jobs(e) == 0  # 4 pairs + 2 probes of a + 2..4 of b
    assert all(np.max(np.abs(U)) <= 2 * e.c + 1e-15 for U in Us)
    assert len(e.required_jobs("quad", a, b)) == 4


def test_debug_mode_records_error():
    e = make_engine(debug=True)
    rng = np.random.default_rng(14)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    e.quad(a, b)
    e.cubic(a, b, c)
    s = e.stats()
    assert 0 < s["debug_max_rel_err_quad"] <= 1e-9
    assert 0 < s["debug_max_rel_err_cubic"] <= 1e-9


def test_disk_cache_resume_zero_jobs(tmp_path):
    rng = np.random.default_rng(15)
    a, b, c = cvec(rng), cvec(rng), cvec(rng)
    e1 = make_engine(cache_dir=tmp_path, lru=2)
    q1, h1 = e1.quad(a, b), e1.cubic(a, b, c)
    n1 = jobs(e1)
    assert 10 <= n1 <= 12
    e1.save_state()
    e2 = make_engine(cache_dir=tmp_path, lru=2)  # "new process"
    assert len(e2.directions) == len(e1.directions)
    q2, h2 = e2.quad(a, b), e2.cubic(a, b, c)
    assert jobs(e2) == 0 and e2.stats()["jobs_k0"] == 0
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(h1, h2)
    keys = e2.store.manifest_keys()
    assert len(keys) == len(set(keys)) == n1 + 1  # jobs + K0, none repeated


def test_lru_eviction_with_disk_still_exact(tmp_path):
    e = make_engine(cache_dir=tmp_path, lru=1)
    rng = np.random.default_rng(16)
    vs = [cvec(rng) for _ in range(4)]
    for x in vs:
        for y in vs:
            assert rel(e.quad(x, y), e.backend.exact_quad(x, y)) <= 1e-9
    assert e.stats()["store_items_memory"] == 1


def test_build_engine_from_config():
    cfg = Path(__file__).resolve().parents[2] / "configs" / "analytic_random.toml"
    e = nistep.build_engine(cfg)
    assert e.n == 50 and e.config.name == "analytic_random"
    rng = np.random.default_rng(17)
    a, b = cvec(rng, 50), cvec(rng, 50)
    assert rel(e.quad(a, b), e.backend.exact_quad(a, b)) <= 1e-9
    assert e.stats()["debug_max_rel_err_quad"] <= 1e-9


def test_rejects_bad_shapes():
    e = make_engine()
    with pytest.raises(ValueError):
        e.quad(np.ones(3), np.ones(N))
