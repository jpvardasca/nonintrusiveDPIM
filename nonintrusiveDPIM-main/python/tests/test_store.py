import numpy as np
import scipy.sparse as sp

from nistep.backends.analytic import random_sparse_model
from nistep.store import KtgStore


def _mats(n=3):
    be = random_sparse_model(12, seed=2)
    rng = np.random.default_rng(0)
    Us = [rng.normal(size=12) for _ in range(n)]
    return be, Us, be.tangent_many(Us)


def test_memory_store_roundtrip_shares_pattern():
    be, Us, Ks = _mats()
    st = KtgStore(be.fingerprint(), lru_items=1)
    keys = [st.put(U, K) for U, K in zip(Us, Ks)]
    # no disk -> nothing evicted despite lru_items=1
    for k, U, K in zip(keys, Us, Ks):
        assert st.contains(k)
        G = st.get(k, U)
        np.testing.assert_array_equal(G.toarray(), K.toarray())
    A, B = st.get(keys[0]), st.get(keys[1])
    assert A.indices is B.indices or np.shares_memory(A.indices, B.indices)
    assert st.stats()["store_pattern_mismatch"] == 0


def test_key_is_exact_and_fingerprinted():
    st1, st2 = KtgStore("a" * 64), KtgStore("b" * 64)
    U = np.array([0.1, 0.2])
    assert st1.key(U) == st1.key(U.copy())
    assert st1.key(U) != st1.key(np.array([0.1, np.nextafter(0.2, 1.0)]))  # one ulp apart
    assert st1.key(U) != st2.key(U)


def test_disk_cache_resume_and_lru(tmp_path):
    be, Us, Ks = _mats(4)
    st = KtgStore(be.fingerprint(), cache_dir=tmp_path, lru_items=2)
    keys = [st.put(U, K) for U, K in zip(Us, Ks)]
    assert st.stats()["store_items_memory"] == 2  # evicted to disk
    np.testing.assert_array_equal(st.get(keys[0], Us[0]).toarray(), Ks[0].toarray())
    assert st.n_disk_hits == 1
    # new process: everything comes back from disk
    st2 = KtgStore(be.fingerprint(), cache_dir=tmp_path, lru_items=2)
    assert len(st2) == 4
    for k, U, K in zip(keys, Us, Ks):
        np.testing.assert_array_equal(st2.get(k, U).toarray(), K.toarray())
    assert st2.manifest_keys() == keys
    assert not list(tmp_path.rglob("*.tmp"))


def test_mismatched_pattern_stored_in_full(tmp_path):
    st = KtgStore("f" * 64, cache_dir=tmp_path)
    A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    B = sp.csr_matrix(np.array([[1.0, 3.0], [0.0, 2.0]]))
    ka = st.put(np.zeros(2), A)
    kb = st.put(np.ones(2), B)
    assert st.n_pattern_mismatch == 1
    st2 = KtgStore("f" * 64, cache_dir=tmp_path)
    np.testing.assert_array_equal(st2.get(kb).toarray(), B.toarray())
    np.testing.assert_array_equal(st2.get(ka).toarray(), A.toarray())


def test_stored_U_verified(tmp_path):
    st = KtgStore("e" * 64, cache_dir=tmp_path)
    K = sp.identity(2, format="csr")
    k = st.put(np.zeros(2), K)
    st2 = KtgStore("e" * 64, cache_dir=tmp_path)
    assert st2.get(k, np.ones(2)) is None  # U mismatch -> miss
