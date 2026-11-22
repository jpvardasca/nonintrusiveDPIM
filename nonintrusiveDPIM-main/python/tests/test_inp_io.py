"""P6.1: reading base decks and writing job decks."""

import hashlib

import numpy as np
import pytest

from decks import DECKS, deck_path, model
from nistep.inp_io import (InpError, JobDeck, prepare_base_inp, read_inp, read_prescribed,
                           render_job_inp)

SMALL = """*Heading
** two elements
*Node
1, 0., 0., 0.
2, 1., 0., 0.
3, 1., 1., 0.
4, 0., 1., 0.
5, 0., 0., 1.
6, 1., 0., 1.
7, 1., 1., 1.
8, 0., 1., 1.
 9, 0., 0., 2.
10, 1., 0., 2.
11, 1., 1., 2.
12, 0., 1., 2.
*Nset, nset=All, generate
1, 12, 1
*Nset, nset=Base
1, 2, 3,
4
*User Element,Type=U3,Nodes=8,Coordinates=3,Properties=3,Iproperties=4, Unsym
1, 2, 3
*Element, type=U3, elset=allUEL
1, 1, 2, 3, 4, 5, 6, 7, 8
2, 5, 6, 7, 8,
 9, 10, 11, 12
*UEL PROPERTY, ELSET=allUEL
** E, nu, rho, nInt, fbar, matID, nPost
100., 0.25, 2., 8, 0, 1, 12
*Element, type=C3D8
100001, 1, 2, 3, 4, 5, 6, 7, 8
*Step, name=LoadStep, nlgeom=YES, inc=1
*Static, direct
1., 1.,
** Boundary conditions
*Boundary
Base, ENCASTRE
** Output requests
*Element Matrix Output, Elset=allUEL, File Name=old_name, Output File=User Defined, Stiffness=Yes
*End Step
"""


def test_read_small_deck():
    m = read_inp(SMALL)
    assert m.n_nodes == 12 and m.n_elem == 2
    assert m.conn[1].tolist() == [5, 6, 7, 8, 9, 10, 11, 12]  # continuation line
    assert m.nset("ALL").tolist() == list(range(1, 13))  # generate, case-insensitive
    assert m.nset("base").tolist() == [1, 2, 3, 4]
    assert m.props.E == 100.0 and m.props.nu == 0.25 and m.props.rho == 2.0
    assert (m.props.n_int, m.props.fbar, m.props.mat_id, m.props.n_post) == (8, 0, 1, 12)
    assert m.props.mu == pytest.approx(40.0) and m.props.lam == pytest.approx(40.0)
    assert m.matrix_file_name == "old_name"
    assert not m.has_node_sets()
    with pytest.raises(InpError):
        m.nset("nope")


@pytest.mark.parametrize("name", list(DECKS))
def test_read_repository_decks(name):
    m = model(name)
    _, _, _, n_nodes, n_elem, _, _ = DECKS[name]
    assert m.n_nodes == n_nodes and m.n_elem == n_elem
    assert m.conn.shape == (n_elem, 8)
    assert m.has_node_sets()
    assert m.props.n_int == 8 and m.props.fbar == 0 and m.props.mat_id == 1
    assert m.nset("All").size == n_nodes
    assert m.matrix_file_name


def test_render_job_deck_roundtrip_and_base_untouched():
    path = deck_path("mems")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    m = model("mems")
    dm = m.dofmap("Set-2")
    deck = JobDeck(m, dm, "Set-2")
    U = np.random.default_rng(0).normal(size=dm.n_dof) * 1e-3
    U[5] = 0.0
    U[7] = 1.0 / 3.0
    text = deck.render(U, "job_abc123")
    assert np.array_equal(read_prescribed(text, dm), U)  # 17 significant digits: exact
    assert "File Name=job_abc123," in text and m.matrix_file_name not in text
    assert text.count("*Boundary") == 1 and "Set-2, ENCASTRE" in text
    # everything outside the boundary block and the file name is unchanged
    m2 = read_inp(text)
    assert np.array_equal(m2.conn, m.conn) and np.array_equal(m2.coords, m.coords)
    assert m2.props == m.props
    # first prescribed line belongs to the first free node, DOF 1
    node, dof = dm.node_dof(0)
    assert f"node{node}, 1, 1, {U[0]:.17e}" in text
    # a second render is independent of the first
    text2 = deck.render(np.zeros(dm.n_dof), "job_zero")
    assert np.all(read_prescribed(text2, dm) == 0) and "job_abc123" not in text2
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before  # legacy bug B2


def test_render_rejects_bad_input():
    m = model("single")
    dm = m.dofmap("Set-2")
    deck = JobDeck(m, dm, "Set-2")
    with pytest.raises(ValueError):
        deck.render(np.zeros(dm.n_dof + 1), "j")
    with pytest.raises(ValueError):
        deck.render(np.full(dm.n_dof, np.nan), "j")
    with pytest.raises(ValueError):
        deck.render(np.zeros(dm.n_dof), "bad name.mtx")
    with pytest.raises(ValueError):
        deck.render(np.zeros(dm.n_dof), "j", output="damping")


def test_render_mass_output_request():
    # Abaqus labels the element matrices by the output request, not by the UEL library:
    # a mass job must ask for Mass=Yes (found on the Abaqus host, P8.6).
    m = model("single")
    dm = m.dofmap("Set-2")
    deck = JobDeck(m, dm, "Set-2")
    U = np.zeros(dm.n_dof)
    request = lambda text: next(ln for ln in text.splitlines()
                                if ln.upper().startswith("*ELEMENT MATRIX OUTPUT"))
    stiff = request(deck.render(U, "jk"))
    mass = request(deck.render(U, "jm", output="mass"))
    assert "Stiffness=Yes" in stiff and "Mass=" not in stiff
    assert "Mass=Yes" in mass and "Stiffness=" not in mass
    assert "File Name=jm," in mass
    assert mass.replace("jm", "jk").replace("Mass=Yes", "Stiffness=Yes") == stiff


def test_prepare_base_inp_adds_node_sets():
    m = read_inp(SMALL)
    dm = m.dofmap("Base")
    with pytest.raises(InpError):
        JobDeck(m, dm, "Base")  # no per-node sets yet
    prepared = prepare_base_inp(SMALL, "Base")
    mp = read_inp(prepared)
    assert mp.has_node_sets() and mp.nset("node12").tolist() == [12]
    assert prepare_base_inp(prepared, "Base") == prepared  # idempotent
    U = np.linspace(-1, 1, dm.n_dof)
    text = render_job_inp(mp, dm, U, "j1", clamped_nset="Base")
    assert np.array_equal(read_prescribed(text, dm), U)
    assert "node5, 1, 1, " in text and "node1, 1, 1" not in text  # clamped nodes not prescribed
    # label style needs no node sets
    text_l = render_job_inp(m, dm, U, "j1", clamped_nset="Base", bc_style="label")
    assert np.array_equal(read_prescribed(text_l, dm), U) and "\n5, 1, 1, " in text_l


def test_malformed_decks():
    with pytest.raises(InpError):
        read_inp("*Heading\n*Node\n1, 0., 0.\n")  # 2-D node
    with pytest.raises(InpError):
        read_inp(SMALL.replace("2, 5, 6, 7, 8,\n 9, 10, 11, 12\n", "2, 5, 6, 7\n"))
    with pytest.raises(InpError):
        read_inp(SMALL.replace("9, 10, 11, 12\n*UEL", "9, 10, 11, 99\n*UEL"))  # unknown node
    m = read_inp(prepare_base_inp(SMALL.replace("*Boundary\nBase, ENCASTRE\n", ""), "Base"))
    with pytest.raises(InpError):
        JobDeck(m, m.dofmap("Base"), "Base")  # no *Boundary block
