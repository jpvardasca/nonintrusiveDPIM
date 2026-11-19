"""Matrix files: Abaqus element-matrix output and the repository's processed matrices.

Raw Abaqus element-matrix output
--------------------------------
``*Element Matrix Output, ..., Output File=User Defined, Stiffness=Yes`` writes one block
per element (the layout the legacy parser ``Organize_the_matrix_right_order.py`` reads)::

    ** ELEMENT NUMBER <label> STEP NUMBER 1 INCREMENT NUMBER 1 TYPE U3
    *USER ELEMENT, NODES=8, LINEAR, UNSYMM
    ** ELEMENT NODES
    ** n1, n2, ..., n8
    1, 2, 3
    *MATRIX,TYPE=STIFFNESS
    v, v, v, v
    ...

The matrix has ``(nodes x dofs)^2`` values, row by row, for an unsymmetric element, or the
lower triangle row by row for a symmetric one. Local DOFs are node-major: local index
``3 * a + (dof - 1)`` for the ``a``-th element node.

Unlike the legacy code, :func:`parse_element_matrices` **raises** on a malformed block
(legacy bug B4) and keeps every value, zeros included (B5).

Processed matrices
------------------
The repository files ``*_stiffness_matrix.mtx`` / ``*_mass_matrix.mtx`` hold ``i j v`` per
line, 1-based, over the DOFs of **all** nodes: ``i = 3 * (node - 1) + dof`` with nodes
numbered 1..n.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


class MtxError(ValueError):
    """Raised for a malformed matrix file."""


@dataclass
class ElementMatrices:
    labels: np.ndarray  # (n_elem,) element labels, file order
    nodes: np.ndarray  # (n_elem, n_nodes_per_elem) node labels, element order
    dofs: tuple[int, ...]  # DOFs per node, e.g. (1, 2, 3)
    matrices: np.ndarray  # (n_elem, nd, nd) with nd = n_nodes_per_elem * len(dofs)
    kind: str = "STIFFNESS"

    def __len__(self) -> int:
        return int(self.labels.size)

    def as_dict(self) -> dict[int, tuple[np.ndarray, tuple[int, ...], np.ndarray]]:
        return {int(lab): (self.nodes[k], self.dofs, self.matrices[k])
                for k, lab in enumerate(self.labels)}

    def sorted_by_label(self) -> "ElementMatrices":
        o = np.argsort(self.labels, kind="stable")
        return ElementMatrices(self.labels[o], self.nodes[o], self.dofs, self.matrices[o], self.kind)


def _ints(line: str, what: str) -> list[int]:
    body = line.replace("*", " ")
    try:
        return [int(tok) for tok in body.replace(",", " ").split()]
    except ValueError:
        raise MtxError(f"{what}: cannot read integers from {line!r}") from None


def parse_element_matrices(text: str) -> ElementMatrices:
    """Parse raw Abaqus element-matrix output. Raises :class:`MtxError` if anything is off."""
    labels: list[int] = []
    nodes: list[list[int]] = []
    mats: list[np.ndarray] = []
    dofs_ref: tuple[int, ...] | None = None
    kind_ref: str | None = None

    cur: dict | None = None

    def flush() -> None:
        nonlocal dofs_ref, kind_ref
        if cur is None:
            return
        lab = cur["label"]
        ints = cur["int_lines"]
        if not ints:
            raise MtxError(f"element {lab}: no node list")
        if len(ints) >= 2:
            el_nodes = [v for ln in ints[:-1] for v in ln]
            el_dofs = tuple(ints[-1])
        else:  # no DOF line: default translational DOFs
            el_nodes, el_dofs = ints[0], (1, 2, 3)
        if not cur["has_matrix"]:
            raise MtxError(f"element {lab}: no *MATRIX block")
        nd = len(el_nodes) * len(el_dofs)
        try:
            vals = np.array(" ".join(cur["data"]).replace(",", " ").replace("D", "E")
                            .replace("d", "E").split(), dtype=np.float64)
        except ValueError as exc:
            raise MtxError(f"element {lab}: non-numeric matrix data ({exc})") from None
        if vals.size == nd * nd:
            M = vals.reshape(nd, nd)
        elif vals.size == nd * (nd + 1) // 2:
            M = np.zeros((nd, nd))
            il = np.tril_indices(nd)
            M[il] = vals  # lower triangle, row by row
            M = M + np.tril(M, -1).T
        else:
            raise MtxError(f"element {lab}: expected {nd * nd} (full) or {nd * (nd + 1) // 2} "
                           f"(symmetric) values for {len(el_nodes)} nodes x {len(el_dofs)} DOFs, "
                           f"got {vals.size}")
        if not np.all(np.isfinite(M)):
            raise MtxError(f"element {lab}: non-finite matrix values")
        if dofs_ref is None:
            dofs_ref, kind_ref = el_dofs, cur["kind"]
        elif el_dofs != dofs_ref:
            raise MtxError(f"element {lab}: DOF list {el_dofs} differs from {dofs_ref}")
        elif cur["kind"] != kind_ref:
            raise MtxError(f"element {lab}: matrix type {cur['kind']} differs from {kind_ref}")
        if nodes and len(el_nodes) != len(nodes[0]):
            raise MtxError(f"element {lab}: {len(el_nodes)} nodes, expected {len(nodes[0])}")
        labels.append(lab)
        nodes.append(el_nodes)
        mats.append(M)

    state = None
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        up = s.upper()
        if up.startswith("** ELEMENT NUMBER"):
            flush()
            m = re.search(r"ELEMENT NUMBER\s+(\d+)", up)
            if m is None:
                raise MtxError(f"cannot read the element number from {s!r}")
            cur = {"label": int(m.group(1)), "int_lines": [], "data": [], "has_matrix": False,
                   "kind": "STIFFNESS"}
            state = None
        elif cur is None:
            continue  # file header
        elif up.startswith("** ELEMENT NODES"):
            state = "ints"
        elif up.startswith("*MATRIX"):
            if cur["has_matrix"]:
                raise MtxError(f"element {cur['label']}: more than one *MATRIX block")
            cur["has_matrix"] = True
            m = re.search(r"TYPE\s*=\s*([A-Z_ ]+)", up)
            cur["kind"] = m.group(1).strip() if m else "STIFFNESS"
            state = "matrix"
        elif state == "ints":
            cur["int_lines"].append(_ints(s, f"element {cur['label']}"))
        elif state == "matrix":
            if s.startswith("*"):
                state = None  # a keyword or comment ends the matrix data
            else:
                cur["data"].append(s)
    flush()

    if not labels:
        raise MtxError("no element blocks found ('** ELEMENT NUMBER' lines)")
    if len(set(labels)) != len(labels):
        raise MtxError("duplicate element labels in the matrix file")
    return ElementMatrices(np.asarray(labels, dtype=np.int64), np.asarray(nodes, dtype=np.int64),
                           dofs_ref, np.stack(mats), kind_ref or "STIFFNESS")


def format_element_matrices(labels, nodes, matrices, dofs=(1, 2, 3), kind: str = "STIFFNESS",
                            symmetric: bool = False, per_line: int = 4, sig_digits: int = 18) -> str:
    """Write element matrices in the raw Abaqus layout (used by tests and the mock runner).

    ``sig_digits`` is the number of significant digits of each value. The default, 18, keeps
    every double exactly; a smaller number imitates the rounding of a real ``.mtx`` file.
    """
    if not 1 <= int(sig_digits) <= 18:
        raise ValueError("sig_digits must be between 1 and 18")
    prec = int(sig_digits) - 1
    out: list[str] = []
    matrices = np.asarray(matrices, dtype=np.float64)
    for lab, el_nodes, M in zip(labels, nodes, matrices):
        out.append(f"** ELEMENT NUMBER {int(lab):9d} STEP NUMBER     1 INCREMENT NUMBER     1 TYPE U3")
        out.append(f"*USER ELEMENT, NODES={len(el_nodes):6d}, LINEAR" + ("" if symmetric else ", UNSYMM"))
        out.append("** ELEMENT NODES")
        out.append("** " + ", ".join(f"{int(v):9d}" for v in el_nodes))
        out.append(", ".join(str(int(d)) for d in dofs))
        out.append(f"*MATRIX,TYPE={kind}")
        vals = M[np.tril_indices(M.shape[0])] if symmetric else M.ravel()
        for k in range(0, vals.size, per_line):
            out.append(", ".join(f"{v: .{prec}E}" for v in vals[k:k + per_line]))
    return "\n".join(out) + "\n"


def load_processed_ijv(path, n_total: int | None = None) -> sp.csr_matrix:
    """Load a processed ``i j v`` file (1-based) as a CSR matrix over the DOFs of all nodes."""
    path = Path(path)
    try:
        data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise MtxError(f"cannot read {path}: {exc}") from None
    if data.shape[1] != 3 or data.shape[0] == 0:
        raise MtxError(f"{path}: expected lines 'i j value'")
    i, j = data[:, 0], data[:, 1]
    if np.any(i != np.floor(i)) or np.any(j != np.floor(j)) or i.min() < 1 or j.min() < 1:
        raise MtxError(f"{path}: indices must be positive integers")
    i = i.astype(np.int64) - 1
    j = j.astype(np.int64) - 1
    n = int(max(i.max(), j.max())) + 1
    if n_total is None:
        n_total = n
    elif n > n_total:
        raise MtxError(f"{path}: index {n} exceeds the expected size {n_total}")
    A = sp.coo_matrix((data[:, 2], (i, j)), shape=(n_total, n_total)).tocsr()
    A.sum_duplicates()
    return A
