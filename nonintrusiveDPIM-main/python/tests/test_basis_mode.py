"""P3.2: basis-mode EED (StEP.tex) and Galerkin export."""

from itertools import combinations_with_replacement

import numpy as np
import pytest

from nistep.backends.analytic import random_sparse_model, two_dof_example
from nistep.basis_mode import (GalerkinModel, basis_coefficients, cubic_multiplicity,
                               galerkin_tensors, load_basis_coefficients, n_ed, n_eed,
                               quad_multiplicity, save_basis_coefficients)
from nistep.config import StepCfg
from nistep.engine import EEDEngine
from nistep.store import KtgStore

N = 30


def make_engine(backend=None, cache_dir=None):
    backend = backend or random_sparse_model(N, seed=21, density=0.15)
    store = KtgStore(backend.fingerprint(), cache_dir=cache_dir)
    return EEDEngine(backend, StepCfg(d_ref=1.0), store)


def basis(n, seed=0):
    return np.random.default_rng(seed).normal(size=(N, n))


def rel(x, y):
    return np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300)


def test_d_formulas():
    assert n_eed(10) == 56 and n_ed(10) == 275
    assert [n_eed(n) for n in (1, 2, 3)] == [2, 4, 7]
    for n in range(1, 30):
        assert n_eed(n) == 2 * n + (n - 1) * (n - 2) // 2
    with pytest.raises(ValueError):
        n_eed(0)


def test_multiplicities():
    assert quad_multiplicity(1, 1) == 1 and quad_multiplicity(0, 1) == 2
    assert cubic_multiplicity(2, 2, 2) == 1
    assert cubic_multiplicity(0, 0, 1) == cubic_multiplicity(0, 1, 1) == 3
    assert cubic_multiplicity(0, 1, 2) == 6


@pytest.mark.parametrize("n", range(1, 9))
def test_a_job_count(n):
    e = make_engine()
    basis_coefficients(e, basis(n, seed=n))
    s = e.stats()
    assert s["jobs_total"] == n_eed(n)
    assert s["jobs_probe"] == 2 * n and s["jobs_pair"] == (n - 1) * (n - 2) // 2
    assert s["jobs_k0"] == 1


@pytest.mark.parametrize("prefetch", [True, False])
def test_b_coefficients_match_exact(prefetch):
    e = make_engine()
    n = 4
    V = basis(n, seed=3)
    co = basis_coefficients(e, V, prefetch=prefetch)
    be = e.backend
    for i, j in combinations_with_replacement(range(n), 2):
        ex = quad_multiplicity(i, j) * be.exact_quad(V[:, i], V[:, j])
        assert rel(co["G", i, j], ex) <= 1e-9
    for i, j, k in combinations_with_replacement(range(n), 3):
        ex = cubic_multiplicity(i, j, k) * be.exact_cubic(V[:, i], V[:, j], V[:, k])
        assert rel(co["H", i, j, k], ex) <= 1e-9
    assert len(co) == n * (n + 1) // 2 + n * (n + 1) * (n + 2) // 6
    assert all(v.dtype == np.float64 and v.shape == (N,) for v in co.values())


def test_b_monomial_sums_reproduce_forces():
    """g(Vq) = Σ_{i<=j} G̃_ij q_i q_j and h(Vq) = Σ_{i<=j<=k} H̃_ijk q_i q_j q_k."""
    e = make_engine()
    n = 3
    V = basis(n, seed=5)
    co = basis_coefficients(e, V)
    q = np.array([0.7, -1.3, 0.4])
    u = V @ q
    g = sum(co["G", i, j] * q[i] * q[j] for i, j in combinations_with_replacement(range(n), 2))
    h = sum(co["H", i, j, k] * q[i] * q[j] * q[k]
            for i, j, k in combinations_with_replacement(range(n), 3))
    assert rel(g, e.backend.exact_quad(u, u)) <= 1e-9
    assert rel(h, e.backend.exact_cubic(u, u, u)) <= 1e-9


def test_c_two_dof_example():
    e = make_engine(backend=two_dof_example())
    co = basis_coefficients(e, np.ones((2, 1)))
    np.testing.assert_allclose(co["G", 0, 0], [20, 20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(co["H", 0, 0, 0], [10, 10], rtol=0, atol=1e-12)
    assert e.stats()["jobs_total"] == 2


def test_two_dof_full_basis():
    # g = [20 U1², 20 U2²], h = [10 U1 U2², 10 U1² U2] with V = identity
    e = make_engine(backend=two_dof_example())
    co = basis_coefficients(e, np.eye(2))
    expect = {("G", 0, 0): [20, 0], ("G", 0, 1): [0, 0], ("G", 1, 1): [0, 20],
              ("H", 0, 0, 0): [0, 0], ("H", 0, 0, 1): [0, 10], ("H", 0, 1, 1): [10, 0],
              ("H", 1, 1, 1): [0, 0]}
    assert set(co) == set(expect)
    for key, val in expect.items():
        np.testing.assert_allclose(co[key], val, rtol=0, atol=1e-11)
    assert e.stats()["jobs_total"] == n_eed(2) == 4


def test_galerkin_model(tmp_path):
    e = make_engine()
    n = 3
    V = basis(n, seed=7)
    rom = galerkin_tensors(e, V, alpha=0.01, beta=0.002)
    be = e.backend
    np.testing.assert_allclose(rom.Mr, V.T @ (be.mass() @ V), rtol=1e-13)
    np.testing.assert_allclose(rom.Kr, V.T @ (be.stiffness0() @ V), rtol=1e-13)
    np.testing.assert_allclose(rom.Cr, 0.01 * rom.Mr + 0.002 * rom.Kr, rtol=1e-13)
    # symmetry of the stored tensors
    assert np.array_equal(rom.g, rom.g.transpose(0, 2, 1))
    assert np.array_equal(rom.h, rom.h.transpose(0, 2, 1, 3))
    assert np.array_equal(rom.h, rom.h.transpose(0, 3, 2, 1))
    # reduced internal force = projection of the full internal force
    q = np.array([0.3, -0.2, 0.5])
    assert rel(rom.internal_force(q), V.T @ be.internal_force(V @ q)) <= 1e-9
    assert e.stats()["jobs_total"] == n_eed(n)
    back = GalerkinModel.load(rom.save(tmp_path / "rom.npz"))
    assert np.array_equal(back.h, rom.h) and np.array_equal(back.V, V)


def test_save_load_roundtrip(tmp_path):
    e = make_engine()
    V = basis(2, seed=9)
    co = basis_coefficients(e, V)
    V2, co2 = load_basis_coefficients(save_basis_coefficients(tmp_path / "nsc.npz", co, V))
    assert np.array_equal(V, V2) and set(co) == set(co2)
    assert all(np.array_equal(co[k], co2[k]) for k in co)


def test_prefetch_restores_mode_and_runs_one_batch():
    e = make_engine()
    basis_coefficients(e, basis(4, seed=11))
    assert e.mode == "lazy" and e.stats()["batches"] == 1 and e.pending_count() == 0


def test_rejects_bad_basis():
    e = make_engine()
    with pytest.raises(ValueError):
        basis_coefficients(e, np.ones((N, 2)) * 1j)
    with pytest.raises(ValueError):
        basis_coefficients(e, np.zeros((N, 1)))
    with pytest.raises(ValueError):
        basis_coefficients(e, np.ones((N + 1, 1)))
