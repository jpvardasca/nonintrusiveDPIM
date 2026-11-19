"""P1.5: AnalyticBackend checks (finite-difference tangent, StEP.tex 2-DoF, symmetry)."""

from itertools import permutations

import numpy as np
import pytest

from nistep.backends.analytic import (
    AnalyticBackend,
    polarize_cubic,
    polarize_quadratic,
    random_sparse_model,
    spring_chain,
    two_dof_example,
)
from nistep.backends.base import TangentBackend

MODELS = {
    "two_dof": two_dof_example,
    "chain": lambda: spring_chain(12),
    "random": lambda: random_sparse_model(30, seed=3, density=0.1),
}


def fd_jacobian(f, U, h=1e-6):
    n = U.size
    J = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = h
        J[:, k] = (f(U + e) - f(U - e)) / (2 * h)
    return J


@pytest.mark.parametrize("name", MODELS)
def test_protocol(name):
    be = MODELS[name]()
    assert isinstance(be, TangentBackend)
    assert len(be.fingerprint()) == 64
    assert be.fingerprint() == MODELS[name]().fingerprint()  # deterministic


@pytest.mark.parametrize("name", MODELS)
def test_tangent_equals_fd_jacobian(name):
    be = MODELS[name]()
    rng = np.random.default_rng(7)
    for _ in range(3):
        U = rng.normal(size=be.n_dof) * 0.3
        Kt = be.tangent_many([U])[0].toarray()
        J = fd_jacobian(be.internal_force, U)
        assert np.max(np.abs(Kt - J)) <= 1e-6 * max(1.0, np.max(np.abs(J)))


@pytest.mark.parametrize("name", MODELS)
def test_fixed_pattern_and_stiffness0(name):
    be = MODELS[name]()
    K0 = be.stiffness0()
    rng = np.random.default_rng(1)
    Ks = be.tangent_many([np.zeros(be.n_dof), rng.normal(size=be.n_dof)])
    for Kt in Ks:
        np.testing.assert_array_equal(Kt.indptr, K0.indptr)
        np.testing.assert_array_equal(Kt.indices, K0.indices)
    np.testing.assert_array_equal(Ks[0].data, K0.data)
    np.testing.assert_allclose(K0.toarray(), be.K_lin.toarray())
    assert be.n_tangent_evals == 2


def test_two_dof_tangent_matches_steptex():
    """StEP.tex lines 309-318: K_tg = [[15+40U1+10U2², -5+20U1U2], [-5+20U1U2, 15+40U2+10U1²]]."""
    be = two_dof_example()
    rng = np.random.default_rng(0)
    for _ in range(5):
        U1, U2 = rng.normal(size=2)
        expected = np.array([
            [15 + 40 * U1 + 10 * U2**2, -5 + 20 * U1 * U2],
            [-5 + 20 * U1 * U2, 15 + 40 * U2 + 10 * U1**2],
        ])
        np.testing.assert_allclose(be.tangent(np.array([U1, U2])).toarray(), expected, rtol=1e-14, atol=1e-12)
    V1 = np.ones(2)
    np.testing.assert_allclose(be.exact_quad(V1, V1), [20, 20])
    np.testing.assert_allclose(be.exact_cubic(V1, V1, V1), [10, 10])
    np.testing.assert_allclose(be.K_lin @ V1, [10, 10])


@pytest.mark.parametrize("name", MODELS)
def test_polarised_forms_symmetric(name):
    be = MODELS[name]()
    rng = np.random.default_rng(5)
    a, b, c = (rng.normal(size=be.n_dof) + 1j * rng.normal(size=be.n_dof) for _ in range(3))
    q = be.exact_quad(a, b)
    assert np.max(np.abs(q - be.exact_quad(b, a))) <= 1e-14 * max(1, np.max(np.abs(q)))
    h = be.exact_cubic(a, b, c)
    for p in permutations((a, b, c)):
        assert np.max(np.abs(h - be.exact_cubic(*p))) <= 1e-14 * max(1, np.max(np.abs(h)))


@pytest.mark.parametrize("name", ["chain", "random"])
def test_potential_forms_fully_symmetric(name):
    """Potential models: c·Ĝ(a,b) is symmetric in (a,b,c), so K_tg(U) is symmetric."""
    be = MODELS[name]()
    rng = np.random.default_rng(2)
    a, b, c, d = (rng.normal(size=be.n_dof) for _ in range(4))
    assert np.dot(c, be.exact_quad(a, b)) == pytest.approx(np.dot(a, be.exact_quad(c, b)), rel=1e-12)
    assert np.dot(d, be.exact_cubic(a, b, c)) == pytest.approx(np.dot(a, be.exact_cubic(d, b, c)), rel=1e-12)
    Kt = be.tangent(rng.normal(size=be.n_dof)).toarray()
    np.testing.assert_allclose(Kt, Kt.T, atol=1e-12)


def test_polarize_reproduces_monomials():
    n = 3
    G = polarize_quadratic(n, [(0, 0, 1, 2.0), (2, 2, 2, -1.5)])
    H = polarize_cubic(n, [(1, 0, 1, 2, 6.0), (0, 0, 0, 1, 3.0)])
    u = np.array([0.7, -1.3, 0.4])
    np.testing.assert_allclose(G.apply(u, u), [2.0 * u[0] * u[1], 0.0, -1.5 * u[2] ** 2])
    np.testing.assert_allclose(H.apply(u, u, u), [3.0 * u[0] ** 2 * u[1], 6.0 * u[0] * u[1] * u[2], 0.0])


def test_complex_forms_not_conjugated():
    be = random_sparse_model(10, seed=1)
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=10), rng.normal(size=10)
    z = (1 + 2j) * a
    np.testing.assert_allclose(be.exact_quad(z, b), (1 + 2j) * be.exact_quad(a, b))


def test_check_U_rejects_bad_input():
    be = two_dof_example()
    with pytest.raises(TypeError):
        be.tangent_many([np.array([1j, 0])])
    with pytest.raises(ValueError):
        be.tangent_many([np.zeros(3)])
    with pytest.raises(ValueError):
        be.tangent_many([np.array([np.nan, 0.0])])
