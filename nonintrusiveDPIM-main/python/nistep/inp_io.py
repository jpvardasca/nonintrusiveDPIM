"""Abaqus input decks: read the base deck and write per-job decks.

Reading (:func:`read_inp`) extracts what the FE side needs from a base deck:

* ``*Node`` (labels and coordinates),
* ``*Element, type=U3`` (the user-element connectivity),
* every ``*Nset`` (including ``generate``),
* ``*UEL PROPERTY`` (``E, nu, rho, nInt, fbar, matID, nPostVars``),
* the ``*Element Matrix Output`` file name.

Writing (:class:`JobDeck`, :func:`render_job_inp`) returns the **text** of a job deck in
which every free DOF has a prescribed displacement and the element-matrix output file has
a given name. The base file is never modified (legacy bug B2).

The prescribed-displacement block replaces the ``*Boundary`` block of the step::

    *Boundary
    <clamped set>, ENCASTRE
    node<label>, <dof>, <dof>, <value with 17 significant digits>
    ...                       (free DOFs in DofMap order)

``node<label>`` is a node set holding one node; the repository decks define one per node.
:func:`prepare_base_inp` adds them to a deck that lacks them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dofmap import NDOF, DofMap


class InpError(ValueError):
    """Raised when a deck cannot be parsed or lacks something we need."""


def _is_keyword(s: str) -> bool:
    return s.startswith("*") and not s.startswith("**")


def _keyword(line: str) -> tuple[str, dict[str, str]]:
    """``*Element Matrix Output, File Name=x`` -> ``("ELEMENT MATRIX OUTPUT", {"FILE NAME": "x"})``."""
    parts = [p.strip() for p in line.strip()[1:].split(",")]
    name = " ".join(parts[0].upper().split())
    params: dict[str, str] = {}
    for p in parts[1:]:
        if not p:
            continue
        if "=" in p:
            k, v = p.split("=", 1)
            params[" ".join(k.upper().split())] = v.strip()
        else:
            params[" ".join(p.upper().split())] = ""
    return name, params


def _numbers(lines: list[str], what: str, cast=float) -> list:
    out = []
    for ln in lines:
        for tok in ln.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(cast(tok))
            except ValueError:
                raise InpError(f"{what}: cannot read {tok!r} in line {ln!r}") from None
    return out


@dataclass(frozen=True)
class UelProperties:
    E: float
    nu: float
    rho: float
    n_int: int = 8
    fbar: int = 0
    mat_id: int = 1
    n_post: int = 12

    @property
    def mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self) -> float:
        return self.nu * self.E / ((1.0 - 2.0 * self.nu) * (1.0 + self.nu))


@dataclass
class InpModel:
    text: str
    node_labels: np.ndarray  # (n_nodes,), file order
    coords: np.ndarray  # (n_nodes, 3), file order
    elem_labels: np.ndarray  # (n_elem,)
    conn: np.ndarray  # (n_elem, 8) node labels, Abaqus C3D8 order
    nsets: dict[str, np.ndarray]  # lower-case name -> node labels in file order
    nset_names: dict[str, str]  # lower-case name -> name as written in the deck
    props: UelProperties | None
    matrix_file_name: str | None
    path: Path | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.node_labels.size)

    @property
    def n_elem(self) -> int:
        return int(self.elem_labels.size)

    def nset(self, name: str) -> np.ndarray:
        try:
            return self.nsets[name.lower()]
        except KeyError:
            raise InpError(f"node set {name!r} not found; available: "
                           f"{[n for n in self.nset_names.values() if not _is_node_set(n)]}") from None

    def nset_name(self, name: str) -> str:
        self.nset(name)
        return self.nset_names[name.lower()]

    def has_node_sets(self) -> bool:
        """True if every node ``i`` has a set ``node<i>`` containing exactly that node."""
        for lab in self.node_labels:
            s = self.nsets.get(f"node{int(lab)}")
            if s is None or s.size != 1 or int(s[0]) != int(lab):
                return False
        return True

    def sorted_coords(self) -> np.ndarray:
        """Coordinates with nodes in ascending label order (the DofMap order)."""
        return self.coords[np.argsort(self.node_labels, kind="stable")]

    def dofmap(self, clamped_nset: str) -> DofMap:
        return DofMap(self.node_labels, self.nset(clamped_nset))


def _is_node_set(name: str) -> bool:
    return re.fullmatch(r"node\d+", name.lower()) is not None


def read_inp(path_or_text, uel_type: str = "U3") -> InpModel:
    """Parse a base deck given as a path, or as text (a string containing a newline)."""
    path = None
    if isinstance(path_or_text, Path) or "\n" not in str(path_or_text):
        path = Path(path_or_text)
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        text = str(path_or_text)
    lines = text.splitlines()

    node_labels: list[int] = []
    coords: list[list[float]] = []
    elem_rows: list[int] = []
    nsets: dict[str, np.ndarray] = {}
    nset_names: dict[str, str] = {}
    props = None
    matrix_file_name = None

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not _is_keyword(s):
            i += 1
            continue
        name, params = _keyword(s)
        data = []
        j = i + 1
        while j < n:
            t = lines[j].strip()
            if _is_keyword(t):
                break
            if t and not t.startswith("**"):
                data.append(t)
            j += 1

        if name == "NODE":
            for ln in data:
                vals = _numbers([ln], "*Node")
                if len(vals) != 4:
                    raise InpError(f"*Node: expected 'label, x, y, z', got {ln!r}")
                if vals[0] != int(vals[0]):
                    raise InpError(f"*Node: non-integer label in {ln!r}")
                node_labels.append(int(vals[0]))
                coords.append(vals[1:])
        elif name == "ELEMENT" and params.get("TYPE", "").upper() == uel_type.upper():
            elem_rows.extend(_numbers(data, "*Element", int))
        elif name == "NSET":
            nm = params.get("NSET")
            if not nm:
                raise InpError(f"*Nset without a name: {s!r}")
            vals = _numbers(data, f"*Nset {nm}", int)
            if "GENERATE" in params:
                out: list[int] = []
                for ln in data:
                    t = _numbers([ln], f"*Nset {nm}", int)
                    if len(t) not in (2, 3):
                        raise InpError(f"*Nset {nm}, generate: expected 'first, last[, step]', got {ln!r}")
                    step = t[2] if len(t) == 3 else 1
                    if step <= 0 or t[1] < t[0]:
                        raise InpError(f"*Nset {nm}, generate: bad range {ln!r}")
                    out.extend(range(t[0], t[1] + 1, step))
                vals = out
            key = nm.lower()
            arr = np.asarray(vals, dtype=np.int64)
            nsets[key] = np.concatenate([nsets[key], arr]) if key in nsets else arr
            nset_names.setdefault(key, nm)
        elif name == "UEL PROPERTY":
            vals = _numbers(data, "*UEL PROPERTY")
            if len(vals) < 3:
                raise InpError("*UEL PROPERTY: expected at least E, nu, rho")
            ints = [int(v) for v in vals[3:7]]
            props = UelProperties(float(vals[0]), float(vals[1]), float(vals[2]), *ints)
        elif name == "ELEMENT MATRIX OUTPUT":
            matrix_file_name = params.get("FILE NAME")
        i = j

    if not node_labels:
        raise InpError("no *Node block found")
    if not elem_rows:
        raise InpError(f"no '*Element, type={uel_type}' block found")
    if len(elem_rows) % 9:
        raise InpError(f"*Element, type={uel_type}: expected 9 integers per element "
                       f"(label + 8 nodes), got {len(elem_rows)} in total")
    elem = np.asarray(elem_rows, dtype=np.int64).reshape(-1, 9)
    labels = np.asarray(node_labels, dtype=np.int64)
    if np.unique(labels).size != labels.size:
        raise InpError("*Node: duplicate node labels")
    if np.unique(elem[:, 0]).size != elem.shape[0]:
        raise InpError("*Element: duplicate element labels")
    unknown = np.setdiff1d(elem[:, 1:], labels)
    if unknown.size:
        raise InpError(f"*Element: nodes not defined in *Node: {unknown[:10].tolist()}")
    return InpModel(text=text, node_labels=labels, coords=np.asarray(coords, dtype=np.float64),
                    elem_labels=elem[:, 0].copy(), conn=elem[:, 1:].copy(), nsets=nsets,
                    nset_names=nset_names, props=props, matrix_file_name=matrix_file_name,
                    path=path)


# --------------------------------------------------------------------------- writing

_FILE_NAME = re.compile(r"(file\s*name\s*=\s*)([^,\r\n]+)", re.IGNORECASE)


def _find_boundary_block(lines: list[str]) -> tuple[int, int]:
    """Line range ``[start, end)`` of the single ``*Boundary`` block inside the step."""
    step = next((k for k, ln in enumerate(lines)
                 if _is_keyword(ln.strip()) and _keyword(ln)[0] == "STEP"), None)
    if step is None:
        raise InpError("no *Step found in the deck")
    starts = [k for k in range(step, len(lines))
              if _is_keyword(lines[k].strip()) and _keyword(lines[k])[0] == "BOUNDARY"]
    if len(starts) != 1:
        raise InpError(f"expected exactly one *Boundary block in the step, found {len(starts)}")
    start = starts[0]
    end = start + 1
    while end < len(lines) and not lines[end].strip().startswith("*"):
        end += 1
    return start, end


class JobDeck:
    """Renders job decks for one base deck. Build once, call :meth:`render` per job."""

    def __init__(self, model: InpModel, dofmap: DofMap, clamped_nset: str,
                 bc_style: str = "nset"):
        if bc_style not in ("nset", "label"):
            raise ValueError("bc_style must be 'nset' or 'label'")
        if bc_style == "nset" and not model.has_node_sets():
            raise InpError("the deck has no per-node sets 'node<i>'; run prepare_base_inp() "
                           "first, or use bc_style='label'")
        lines = model.text.splitlines()
        start, end = _find_boundary_block(lines)
        matrix_lines = [k for k, ln in enumerate(lines)
                        if _is_keyword(ln.strip()) and _keyword(ln)[0] == "ELEMENT MATRIX OUTPUT"]
        if len(matrix_lines) != 1 or not _FILE_NAME.search(lines[matrix_lines[0]]):
            raise InpError("expected exactly one '*Element Matrix Output, ..., File Name=...' line")
        self._head = "\n".join(lines[:start])
        tail = lines[end:]
        self._matrix_line = matrix_lines[0] - end
        if self._matrix_line < 0:
            raise InpError("*Element Matrix Output must come after the *Boundary block")
        self._tail = tail
        self._clamped = model.nset_name(clamped_nset)
        pre = "node" if bc_style == "nset" else ""
        self._prefix = [f"{pre}{int(node)}, {d}, {d}, "
                        for node in dofmap.free_nodes for d in range(1, NDOF + 1)]
        self.n_dof = dofmap.n_dof

    def render(self, U, matrix_file_name: str, output: str = "stiffness") -> str:
        """Job deck. ``output`` selects the matrix Abaqus writes: "stiffness" (Stiffness=Yes,
        the deck's own setting) or "mass" (Mass=Yes; needed with the mass UEL library, which
        otherwise writes its matrix labelled as stiffness)."""
        U = np.asarray(U, dtype=np.float64).ravel()
        if U.shape != (self.n_dof,):
            raise ValueError(f"U must have shape ({self.n_dof},), got {U.shape}")
        if not np.all(np.isfinite(U)):
            raise ValueError("U contains non-finite values")
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", matrix_file_name):
            raise ValueError(f"matrix file name must be alphanumeric, got {matrix_file_name!r}")
        if output not in ("stiffness", "mass"):
            raise ValueError(f"output must be 'stiffness' or 'mass', got {output!r}")
        bc = ["*Boundary", f"{self._clamped}, ENCASTRE"]
        bc.extend(p + format(u, ".17e") for p, u in zip(self._prefix, U))
        tail = list(self._tail)
        line = _FILE_NAME.sub(lambda m: m.group(1) + matrix_file_name, tail[self._matrix_line],
                              count=1)
        if output == "mass":
            if "Stiffness=Yes" not in line:
                raise InpError("expected 'Stiffness=Yes' in the *Element Matrix Output line")
            line = line.replace("Stiffness=Yes", "Mass=Yes")
        tail[self._matrix_line] = line
        return "\n".join([self._head, *bc, *tail]) + "\n"


def render_job_inp(model: InpModel, dofmap: DofMap, U, matrix_file_name: str,
                   clamped_nset: str = "Set-2", bc_style: str = "nset") -> str:
    """Text of the job deck with prescribed displacements ``U`` (free DOFs, DofMap order)."""
    return JobDeck(model, dofmap, clamped_nset, bc_style).render(U, matrix_file_name)


def read_prescribed(text: str, dofmap: DofMap) -> np.ndarray:
    """Inverse of :meth:`JobDeck.render`: the prescribed free-DOF vector of a job deck."""
    lines = text.splitlines()
    start, end = _find_boundary_block(lines)
    U = np.zeros(dofmap.n_dof)
    seen = np.zeros(dofmap.n_dof, dtype=bool)
    for ln in lines[start + 1:end]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue  # "<set>, ENCASTRE"
        m = re.fullmatch(r"(?:node)?(\d+)", parts[0], re.IGNORECASE)
        if m is None:
            raise InpError(f"unexpected boundary line {ln!r}")
        d1, d2 = int(parts[1]), int(parts[2])
        if d1 != d2:
            raise InpError(f"boundary line with a DOF range is not supported: {ln!r}")
        k = dofmap.free_index(int(m.group(1)), d1)
        U[k] = float(parts[3])
        seen[k] = True
    if not seen.all():
        raise InpError(f"{int((~seen).sum())} free DOFs have no prescribed value")
    return U


def prepare_base_inp(text: str, clamped_nset: str = "Set-2") -> str:
    """Add the per-node sets ``node<i>`` after the clamped set, if the deck lacks them."""
    model = read_inp(text if "\n" in text else text + "\n")
    missing = [int(lab) for lab in np.sort(model.node_labels)
               if f"node{int(lab)}" not in model.nsets]
    if not missing:
        return text
    lines = text.splitlines()
    target = model.nset_name(clamped_nset).lower()
    at = None
    for k, ln in enumerate(lines):
        s = ln.strip()
        if _is_keyword(s):
            name, params = _keyword(s)
            if name == "NSET" and params.get("NSET", "").lower() == target:
                at = k + 1
                while at < len(lines) and not lines[at].strip().startswith("*"):
                    at += 1
    if at is None:
        raise InpError(f"node set {clamped_nset!r} not found")
    new = []
    for lab in missing:
        new.extend([f"*Nset, nset=node{lab}", f"{lab}"])
    return "\n".join(lines[:at] + new + lines[at:]) + "\n"
