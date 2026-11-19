import numpy as np
import pytest
import scipy.sparse as sp

from nistep._sparse import FixedPattern, coo_entries


def test_fixed_pattern_sums_duplicates_and_keeps_zeros():
    rows = [0, 0, 1, 2, 0]
    cols = [0, 2, 1, 2, 0]
    P = FixedPattern(rows, cols, (3, 3))
    assert P.nnz == 4
    A = P.assemble([1.0, 2.0, 0.0, 3.0, 4.0])
    assert A.nnz == 4  # explicit zero at (1,1) kept
    np.testing.assert_array_equal(A.toarray(), [[5, 0, 2], [0, 0, 0], [0, 0, 3]])
    B = P.assemble([0, 0, 7, 0, 0])
    np.testing.assert_array_equal(A.indptr, B.indptr)
    np.testing.assert_array_equal(A.indices, B.indices)
    assert A.has_sorted_indices


def test_fixed_pattern_rejects_bad_input():
    with pytest.raises(ValueError):
        FixedPattern([0, 3], [0, 0], (3, 3))
    P = FixedPattern([0], [0], (1, 1))
    with pytest.raises(ValueError):
        P.assemble([1.0, 2.0])


def test_coo_entries_roundtrip():
    A = sp.random(6, 6, density=0.4, random_state=1, format="csr")
    r, c, v = coo_entries(A)
    B = FixedPattern(r, c, A.shape).assemble(v)
    np.testing.assert_allclose(B.toarray(), A.toarray())
