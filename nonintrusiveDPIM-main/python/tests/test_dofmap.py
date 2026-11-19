"""P6.2: DofMap (CONVENTIONS 4.7)."""

import numpy as np
import pytest

from decks import DECKS, model
from nistep.dofmap import DofMap


def test_small_map_ordering():
    dm = DofMap([10, 3, 7, 5], [7, 7])  # unsorted labels, duplicated clamped node
    assert dm.nodes.tolist() == [3, 5, 7, 10]
    assert dm.free_nodes.tolist() == [3, 5, 10] and dm.clamped_nodes.tolist() == [7]
    assert (dm.n_nodes, dm.n_total, dm.n_dof) == (4, 12, 9)
    assert dm.free_index(3, 1) == 0 and dm.free_index(5, 3) == 5 and dm.free_index(10, 2) == 7
    assert dm.total_index(7, 1) == 6 and dm.total_index(10, 3) == 11
    assert dm.node_dof(7) == (10, 2)
    with pytest.raises(KeyError):
        dm.free_index(7, 1)
    with pytest.raises(KeyError):
        dm.free_index(4, 1)
    with pytest.raises(ValueError):
        dm.free_index(3, 4)
    idx = dm.free_indices(np.array([[3, 7], [10, 5]]))
    assert idx.shape == (2, 2, 3)
    assert idx[0, 0].tolist() == [0, 1, 2] and idx[0, 1].tolist() == [-1, -1, -1]
    assert idx[1, 0].tolist() == [6, 7, 8]


def test_expand_restrict_roundtrip():
    dm = DofMap([1, 2, 3, 4], [2])
    u = np.arange(9.0) + 1
    full = dm.expand(u)
    assert full.tolist() == [1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 9]
    assert np.array_equal(dm.restrict(full), u)
    z = dm.expand(u * (1 + 2j))
    assert z.dtype == np.complex128 and z[4] == 0
    assert dm.free_of_total.tolist() == [0, 1, 2, -1, -1, -1, 3, 4, 5, 6, 7, 8]
    A = np.arange(144.0).reshape(12, 12)
    R = dm.restrict_matrix(A).toarray()
    assert R.shape == (9, 9) and R[3, 0] == A[6, 0] and R[0, 8] == A[0, 11]


def test_errors():
    with pytest.raises(ValueError):
        DofMap([1, 1, 2], [])
    with pytest.raises(ValueError):
        DofMap([1, 2], [3])
    with pytest.raises(ValueError):
        DofMap([1, 2], [1, 2])


@pytest.mark.parametrize("name", list(DECKS))
def test_repository_decks(name):
    _, _, _, n_nodes, n_elem, n_clamped, n_free = DECKS[name]
    m = model(name)
    dm = m.dofmap("Set-2")
    assert m.n_nodes == n_nodes and m.n_elem == n_elem
    assert dm.clamped_nodes.size == n_clamped and dm.n_dof == n_free
    assert dm.n_dof == 3 * (n_nodes - n_clamped)
    assert np.all(np.diff(dm.free_nodes) > 0)
    # free index i <-> (node, dof) is consistent
    for i in (0, 1, 2, dm.n_dof // 2, dm.n_dof - 1):
        node, dof = dm.node_dof(i)
        assert dm.free_index(node, dof) == i


def test_mems_clamped_set_has_duplicates():
    m = model("mems")
    assert m.nset("Set-2").size == 20 and np.unique(m.nset("set-2")).size == 18
