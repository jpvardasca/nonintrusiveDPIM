"""P6.3: matrix files (raw Abaqus element matrices, processed "i j v" matrices)."""

import numpy as np
import pytest

from decks import DECKS, LEGACY, model
from nistep.mtx_io import (MtxError, format_element_matrices, load_processed_ijv,
                           parse_element_matrices)


def _random_elements(n_elem=3, seed=0, symmetric=False):
    rng = np.random.default_rng(seed)
    labels = np.array([7, 2, 11][:n_elem])
    nodes = rng.permutation(40)[: 8 * n_elem].reshape(n_elem, 8) + 1
    mats = rng.normal(size=(n_elem, 24, 24)) * 1e9
    if symmetric:
        mats = mats + mats.transpose(0, 2, 1)
    return labels, nodes, mats


def test_raw_roundtrip_full():
    labels, nodes, mats = _random_elements()
    text = "** some header line\n" + format_element_matrices(labels, nodes, mats)
    em = parse_element_matrices(text)
    assert em.labels.tolist() == labels.tolist() and em.dofs == (1, 2, 3)
    assert np.array_equal(em.nodes, nodes)
    assert np.array_equal(em.matrices, mats)  # 17 digits: exact
    assert em.kind == "STIFFNESS"
    d = em.as_dict()
    assert np.array_equal(d[2][2], mats[1])
    s = em.sorted_by_label()
    assert s.labels.tolist() == [2, 7, 11] and np.array_equal(s.matrices[0], mats[1])


def test_raw_roundtrip_symmetric_storage():
    labels, nodes, mats = _random_elements(symmetric=True)
    em = parse_element_matrices(format_element_matrices(labels, nodes, mats, symmetric=True))
    assert np.array_equal(em.matrices, mats)


def test_raw_mass_kind_and_d_exponent():
    labels, nodes, mats = _random_elements(1)
    text = format_element_matrices(labels, nodes, mats, kind="MASS").replace("E+", "D+")
    em = parse_element_matrices(text)
    assert em.kind == "MASS" and np.array_equal(em.matrices, mats)


def test_raw_malformed_raises():
    labels, nodes, mats = _random_elements(2)
    good = format_element_matrices(labels, nodes, mats)
    lines = good.splitlines()
    with pytest.raises(MtxError, match="expected 576"):
        parse_element_matrices("\n".join(lines[:-1]) + "\n")  # last data line missing
    with pytest.raises(MtxError, match="no element blocks"):
        parse_element_matrices("garbage\n1, 2, 3\n")
    with pytest.raises(MtxError, match="no \\*MATRIX"):
        parse_element_matrices("\n".join(lines[:5]) + "\n")
    with pytest.raises(MtxError, match="duplicate"):
        parse_element_matrices(good + format_element_matrices(labels[:1], nodes[:1], mats[:1]))
    with pytest.raises(MtxError, match="non-numeric"):
        parse_element_matrices(good.replace(lines[6], lines[6] + ", abc"))


@pytest.mark.parametrize("name", list(DECKS))
def test_processed_repository_matrices(name):
    _, kf, mf, n_nodes, _, _, _ = DECKS[name]
    n_total = 3 * n_nodes
    for f in (kf, mf):
        A = load_processed_ijv(LEGACY / f, n_total)
        assert A.shape == (n_total, n_total)
        assert abs(A - A.T).max() <= 1e-12 * abs(A).max()  # both triangles present
        assert A.diagonal().min() > 0
    dm = model(name).dofmap("Set-2")
    K = dm.restrict_matrix(load_processed_ijv(LEGACY / kf, n_total))
    assert K.shape == (dm.n_dof, dm.n_dof)


def test_processed_errors(tmp_path):
    p = tmp_path / "bad.mtx"
    p.write_text("1 2\n")
    with pytest.raises(MtxError):
        load_processed_ijv(p)
    p.write_text("0 1 2.0\n")
    with pytest.raises(MtxError):
        load_processed_ijv(p)
    p.write_text("1 5 2.0\n")
    with pytest.raises(MtxError):
        load_processed_ijv(p, 3)
    p.write_text("1 2 3.0\n2 1 3.0\n1 2 1.0\n")
    A = load_processed_ijv(p)
    assert A.shape == (2, 2) and A[0, 1] == 4.0  # duplicates summed
