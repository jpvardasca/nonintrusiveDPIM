import numpy as np

from nistep.registry import DirectionStore, VectorRegistry, canonicalize, make_sign_vector


def test_canonicalize_basic():
    w = make_sign_vector(5, 12345)
    p = np.array([0.5, -2.0, 1.0, 0.0, 0.3])
    alpha, d = canonicalize(p, w)
    np.testing.assert_allclose(alpha * d, p, rtol=0, atol=1e-15)
    assert np.max(np.abs(d)) == 1.0
    assert np.dot(w, d) >= 0
    assert canonicalize(np.zeros(5), w) is None


def test_canonicalize_proportional_parts_same_direction():
    w = make_sign_vector(6, 1)
    p = np.random.default_rng(0).normal(size=6)
    _, d1 = canonicalize(p, w)
    _, d2 = canonicalize(-3.7 * p, w)
    np.testing.assert_allclose(d1, d2, atol=1e-15)


def test_direction_store_match_with_tolerance_and_sign():
    S = DirectionStore(4, match_tol=1e-9)
    d = np.array([1.0, 0.2, -0.3, 0.5])
    i = S.add(d)
    assert S.match(d + 1e-12) == (i, 1.0)
    assert S.match(-d) == (i, -1.0)
    assert S.match(d + 1e-6) is None
    j, s = S.match_or_add(np.array([0.0, 1.0, 0.0, 0.0]))
    assert j == 1 and s == 1.0 and len(S) == 2


def test_direction_store_persistence(tmp_path):
    S = DirectionStore(3)
    S.add(np.array([1.0, 0.0, 0.5]))
    S.add(np.array([0.1, 1.0, 0.0]))
    assert S.dirty
    S.save(tmp_path / "d.npz", seed=7)
    assert not S.dirty
    T = DirectionStore(3)
    assert T.load(tmp_path / "d.npz", seed=7)
    assert len(T) == 2 and np.array_equal(T.get(1), S.get(1))
    assert not DirectionStore(3).load(tmp_path / "d.npz", seed=8)  # seed mismatch


def test_vector_registry_ids_and_conjugates():
    R = VectorRegistry(1e-9)
    rng = np.random.default_rng(0)
    a = rng.normal(size=5) + 1j * rng.normal(size=5)
    b = rng.normal(size=5) + 1j * rng.normal(size=5)
    assert R.id_of(a) == 0
    assert R.id_of(b) == 1
    assert R.id_of(a * (1 + 1e-12)) == 0
    assert R.id_of(np.conj(a)) == 0  # conjugate shares the id
    assert R.id_of(2 * a) == 2  # scaled copy is a new vector
    assert len(R) == 3
