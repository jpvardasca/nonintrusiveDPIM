"""P6.4-P6.6: assembly and the numpy TL-SVK hex8 backend."""

from itertools import permutations

import numpy as np
import pytest
import scipy.sparse as sp

from decks import DECKS, LEGACY, backend, model
from nistep.assembly import Assembler
from nistep.backends.pysvk import GAUSS_XI, NODE_XI, shape_functions
from nistep.config import StepCfg
from nistep.engine import EEDEngine
from nistep.mtx_io import format_element_matrices, load_processed_ijv, parse_element_matrices
from nistep.store import KtgStore


def rel(x, y):
    return np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300)


def smooth_field(be, seed, scale):
    """A smooth real displacement field (free DOFs) with max |u| = scale."""
    rng = np.random.default_rng(seed)
    X = be.model.sorted_coords()[be.dofmap.rank_all(be.dofmap.free_nodes)]
    L = np.ptp(be.model.coords, axis=0).max()
    A, b = rng.normal(size=(3, 3)), rng.normal(size=3)
    u = np.sin(X @ A.T / L + b) + 0.3 * rng.normal(size=X.shape)
    u = u.ravel()
    return scale * u / np.max(np.abs(u))


def cfield(be, seed, scale):
    return smooth_field(be, seed, scale) + 1j * smooth_field(be, seed + 100, scale)


# ------------------------------------------------------------------------- element basics

def test_shape_functions():
    N, dN = shape_functions(NODE_XI)
    np.testing.assert_allclose(N, np.eye(8), atol=1e-15)  # N_a(node b) = δ_ab
    N, dN = shape_functions(GAUSS_XI)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-15)  # partition of unity
    np.testing.assert_allclose(dN.sum(axis=1), 0.0, atol=1e-15)
    h = 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        fd = (shape_functions(GAUSS_XI + d)[0] - shape_functions(GAUSS_XI - d)[0]) / (2 * h)
        np.testing.assert_allclose(dN[:, :, k], fd, atol=1e-9)


def test_volumes():
    assert backend("single").volume == pytest.approx(1.0 * 1.1 * 1.2, rel=1e-13)
    assert backend("mems").volume > 0 and backend("cantilever").volume > 0


# ---------------------------------------------------------------------------- assembly

def test_assembler_matches_dense_loop_and_keeps_pattern():
    be = backend("mems")
    asm, dm = be.asm, be.dofmap
    rng = np.random.default_rng(0)
    Ke = rng.normal(size=(asm.n_elem, 24, 24))
    A = asm.assemble(Ke)
    dense = np.zeros((dm.n_total, dm.n_total))
    for e in range(asm.n_elem):
        g = (3 * dm.rank_all(asm.conn[e])[:, None] + np.arange(3)).ravel()
        dense[np.ix_(g, g)] += Ke[e]
    ref = dense[np.ix_(dm.total_of_free, dm.total_of_free)]
    np.testing.assert_allclose(A.toarray(), ref, rtol=0, atol=1e-12)
    B = asm.assemble(np.zeros_like(Ke))  # explicit zeros kept: same pattern
    assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
    assert B.nnz == A.nnz and not B.data.any()
    # vectors: gather then assemble counts how many elements touch each DOF
    ones = asm.assemble_vector(np.ones((asm.n_elem, 24)))
    assert ones.min() >= 1 and ones.sum() == (asm.elem_dofs >= 0).sum()
    u = rng.normal(size=dm.n_dof) + 1j * rng.normal(size=dm.n_dof)
    ue = asm.gather(u)
    k = asm.elem_dofs.reshape(asm.n_elem, 8, 3)
    assert np.array_equal(ue[k >= 0], u[k[k >= 0]]) and not ue[k < 0].any()


def test_raw_mtx_pipeline_reproduces_assembly():
    """PySVK element matrices -> raw Abaqus text -> parser -> assembler (mock of the Abaqus path)."""
    be = backend("cc")
    m = be.model
    U = smooth_field(be, 3, 1e-3)
    Ke = be.element_tangents(U)
    shuffle = np.random.default_rng(1).permutation(m.n_elem)  # file order differs from the deck
    text = format_element_matrices(m.elem_labels[shuffle], m.conn[shuffle], Ke[shuffle])
    em = parse_element_matrices(text)
    order = be.asm.check_element_order(em.labels, em.nodes, m.elem_labels)
    K = be.asm.assemble(em.matrices[order])
    ref = be.tangent(U)
    assert np.array_equal(K.indices, ref.indices) and np.array_equal(K.data, ref.data)
    with pytest.raises(ValueError):
        be.asm.check_element_order(em.labels[:-1], em.nodes[:-1], m.elem_labels)
    bad = em.nodes.copy()
    bad[0, [0, 1]] = bad[0, [1, 0]]
    with pytest.raises(ValueError):
        be.asm.check_element_order(em.labels, bad, m.elem_labels)


# -------------------------------------------------------- P6.6 (a): repository matrices

@pytest.mark.parametrize("name", list(DECKS))
def test_a_stiffness_and_mass_match_repository(name):
    _, kf, mf, n_nodes, _, _, _ = DECKS[name]
    be = backend(name)
    dm = be.dofmap
    K, M = be.stiffness0(), be.mass()
    Kref = dm.restrict_matrix(load_processed_ijv(LEGACY / kf, 3 * n_nodes))
    Mref = dm.restrict_matrix(load_processed_ijv(LEGACY / mf, 3 * n_nodes))
    assert abs(K - Kref).max() <= 1e-10 * abs(Kref).max()
    assert abs(M - Mref).max() <= 1e-10 * abs(Mref).max()
    assert abs(K - K.T).max() <= 1e-13 * abs(K).max()
    assert abs(M - M.T).max() <= 1e-13 * abs(M).max()
    # total mass: rigid translation u_x = 1 on ALL nodes (use element matrices, no elimination)
    Me = be.element_masses()
    tx = np.tile([1.0, 0.0, 0.0], 8)
    assert np.einsum("i,eij,j->", tx, Me, tx) == pytest.approx(be.rho * be.volume, rel=1e-12)


def test_stiffness0_is_tangent_at_zero_with_same_pattern():
    be = backend("mems")
    K0 = be.stiffness0()
    Kt = be.tangent(smooth_field(be, 0, 1.0))
    assert np.array_equal(K0.indptr, Kt.indptr) and np.array_equal(K0.indices, Kt.indices)
    assert np.array_equal(be.tangent(np.zeros(be.n_dof)).data, K0.data)


# -------------------------------------------------------------- P6.6 (b): tangent vs FD

def test_b_tangent_is_jacobian_single_element():
    be = backend("single")
    U = smooth_field(be, 1, 0.1)
    K = be.tangent(U).toarray()
    h = 1e-6
    J = np.zeros_like(K)
    for j in range(be.n_dof):
        e = np.zeros(be.n_dof)
        e[j] = h
        J[:, j] = (be.internal_force(U + e) - be.internal_force(U - e)) / (2 * h)
    assert rel(K, J) <= 1e-6


@pytest.mark.parametrize("name,scale", [("mems", 2.0), ("cantilever", 0.01)])
def test_b_tangent_directional_derivative(name, scale):
    be = backend(name)
    U = smooth_field(be, 2, scale)
    K = be.tangent(U)
    for seed in (5, 6):
        v = smooth_field(be, seed, 1.0)
        h = 1e-6 * scale
        fd = (be.internal_force(U + h * v) - be.internal_force(U - h * v)) / (2 * h)
        assert rel(K @ v, fd) <= 1e-6


# ------------------------------------------------- polynomial structure and exact forms

@pytest.mark.parametrize("name,scale", [("single", 0.2), ("mems", 5.0)])
def test_polynomial_structure(name, scale):
    be = backend(name)
    K = be.stiffness0()
    U = smooth_field(be, 7, scale)
    v = smooth_field(be, 8, scale)
    # F_int = K U + g(U) + h(U)
    f = K @ U + be.exact_quad(U, U) + be.exact_cubic(U, U, U)
    assert rel(f, be.internal_force(U)) <= 1e-11
    # K_tg(U) v = K v + 2 Ĝ(U, v) + 3 Ĥ(U, U, v)
    Kv = K @ v + 2 * be.exact_quad(U, v) + 3 * be.exact_cubic(U, U, v)
    assert rel(be.tangent(U) @ v, Kv) <= 1e-11
    # K_tg is exactly quadratic in the load factor: the third difference vanishes
    d = U / np.max(np.abs(U))
    c = 0.24 * scale
    Ks = [be.tangent(t * c * d) for t in range(4)]
    third = Ks[3] - 3 * Ks[2] + 3 * Ks[1] - Ks[0]
    assert abs(third).max() <= 1e-11 * abs(Ks[1] - Ks[0]).max()


def test_exact_forms_symmetric_complex_and_homogeneous():
    be = backend("cc")
    a, b, c = cfield(be, 1, 1e-3), cfield(be, 2, 1e-3), cfield(be, 3, 1e-3)
    q = be.exact_quad(a, b)
    assert q.dtype == np.complex128 and rel(be.exact_quad(b, a), q) <= 1e-13
    h = be.exact_cubic(a, b, c)
    for p in permutations((a, b, c)):
        assert rel(be.exact_cubic(*p), h) <= 1e-12
    # bilinear without conjugation: Ĝ(i a, b) = i Ĝ(a, b)
    assert rel(be.exact_quad(1j * a, b), 1j * q) <= 1e-13
    assert rel(be.exact_cubic(2 * a, 3 * b, 1j * c), 6j * h) <= 1e-13


# ------------------------------------------------------ P6.6 (c): engine on a real mesh

@pytest.mark.parametrize("name,d_ref", [("mems", 20.0), ("cantilever", 10.0), ("single", 1.0)])
def test_c_engine_on_pysvk_matches_exact_forms(name, d_ref):
    be = backend(name)
    engine = EEDEngine(be, StepCfg(d_ref=d_ref), KtgStore(be.fingerprint()))
    a, b, c = cfield(be, 11, 1.0), cfield(be, 12, 1.0), cfield(be, 13, 1.0)
    assert rel(engine.quad(a, b), be.exact_quad(a, b)) <= 1e-8
    assert rel(engine.cubic(a, b, c), be.exact_cubic(a, b, c)) <= 1e-8
    assert rel(engine.cubic(a, a, c), be.exact_cubic(a, a, c)) <= 1e-8
    assert rel(engine.quad(np.conj(a), b), be.exact_quad(np.conj(a), b)) <= 1e-8
    assert engine.stats()["jobs_k0"] == 1
    # every prescribed displacement stays within the overlap rule 2c = 0.48 d_ref
    for U in engine.required_jobs("cubic", a, b, c):
        assert np.max(np.abs(U)) <= 0.48 * d_ref * (1 + 1e-12)


def test_min_det_F_and_bad_input():
    be = backend("single")
    assert be.min_det_F(np.zeros(be.n_dof)) == pytest.approx(1.0)
    assert be.min_det_F(smooth_field(be, 0, 0.05)) > 0.5
    with pytest.raises(TypeError):
        be.tangent(np.zeros(be.n_dof) * 1j)
    with pytest.raises(ValueError):
        be.tangent(np.zeros(be.n_dof + 1))


def test_fingerprint_depends_on_model():
    assert backend("mems").fingerprint() != backend("cc").fingerprint()
    assert len(backend("mems").fingerprint()) == 64
    assert sp.issparse(backend("mems").mass())
