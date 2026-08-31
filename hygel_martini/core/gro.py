"""One GRO reader for the whole package.

The format is fixed-column, not whitespace-delimited::

    %5d%-5s%5s%5d%8.3f%8.3f%8.3f
    resnr residue atom  index    x       y       z

The package previously carried three readers.  Two were partial --- one took
only atom names, one only coordinates and the box --- and the third split the
tail of each line on whitespace.  That third one fails on valid input: a
coordinate of ``-100.000`` fills its eight columns exactly, so it abuts its
neighbour and the fields stop being separable by whitespace.  Large boxes and
templates with negative coordinates both reach that case, and the failure is
an exception rather than a wrong number, which is the one mercy in it.

This reader takes the columns, inferring the coordinate field width rather
than assuming three decimals so that high-precision files are read correctly,
and parses the box line in both its three-value orthorhombic and nine-value
triclinic forms.  Whitespace splitting is kept only as a fallback for records
that do not sit on the standard columns -- the tracked example structures
shift the atom index right by one column, and the previous lenient reader had
been accepting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

__all__ = ["GroAtom", "GroFrame", "read_gro", "read_gro_atoms", "read_gro_atom_names"]

# resnr, residue name, atom name, atom index each occupy five columns
_INDEX_END = 20


@dataclass
class GroAtom:
    """One atom record from a GRO file."""

    index: int
    residue_number: int
    residue_name: str
    atom_name: str
    position: np.ndarray


@dataclass
class GroFrame:
    """A parsed GRO file: title, atoms, and the periodic cell if present."""

    title: str
    atoms: List[GroAtom] = field(default_factory=list)
    #: 3x3 cell with rows as cell vectors, or ``None`` if the box line was absent.
    box: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def positions(self) -> np.ndarray:
        if not self.atoms:
            return np.empty((0, 3), dtype=float)
        return np.array([atom.position for atom in self.atoms], dtype=float)

    @property
    def atom_names(self) -> List[str]:
        return [atom.atom_name for atom in self.atoms]


def _parse_tail(line: str, path: str, line_number: int) -> tuple[int, np.ndarray]:
    """Atom index and position from the part of a record after column 15.

    Fixed columns are tried first, because that is the format and because
    whitespace splitting fails on valid input: a coordinate of ``-100.000``
    fills its eight columns exactly and abuts its neighbour.  Files that are
    off by a column -- the tracked example structures shift the index field
    right by one -- do not parse that way, so whitespace splitting remains as a
    fallback rather than as the primary rule.  Both are exact when they apply;
    neither guesses a value.
    """
    tail = line[15:].rstrip()

    # 1. fixed columns: %5d then three equal-width floating point fields
    index_field = line[15:_INDEX_END]
    body = line[_INDEX_END:].rstrip()
    if index_field.strip():
        for fields in (3, 6):
            width, remainder = divmod(len(body), fields)
            if remainder or not 8 <= width <= 20:
                continue
            try:
                return int(index_field), np.array(
                    [
                        float(body[step * width : (step + 1) * width])
                        for step in range(3)
                    ],
                    dtype=float,
                )
            except ValueError:
                break

    # 2. whitespace fallback for records that do not sit on the standard columns
    parts = tail.split()
    if len(parts) >= 4:
        try:
            return int(parts[0]), np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])], dtype=float
            )
        except ValueError:
            pass

    raise ValueError(
        f"{path}:{line_number}: could not read an atom index and position from "
        f"{line.rstrip()!r} by fixed columns or by whitespace"
    )


def read_gro(path: str | Path) -> GroFrame:
    """Parse a GRO file into title, atom records and periodic cell."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: GRO file needs a title, a count and a box line")

    title = lines[0].strip()
    try:
        count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"{path}:2: atom count is not an integer") from exc
    if count < 0:
        raise ValueError(f"{path}:2: atom count is negative")
    if len(lines) < count + 3:
        raise ValueError(
            f"{path}: declares {count} atoms but the file holds only "
            f"{max(len(lines) - 3, 0)}"
        )

    atoms: List[GroAtom] = []
    for offset in range(count):
        line_number = offset + 3
        line = lines[offset + 2]
        if len(line) < _INDEX_END:
            raise ValueError(
                f"{path}:{line_number}: atom record is shorter than the "
                f"{_INDEX_END}-column header"
            )
        try:
            residue_number = int(line[0:5])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: residue number field is not an integer "
                f"in {line.rstrip()!r}"
            ) from exc
        atom_index, position = _parse_tail(line, str(path), line_number)
        atoms.append(
            GroAtom(
                index=atom_index,
                residue_number=residue_number,
                residue_name=line[5:10].strip(),
                atom_name=line[10:15].strip(),
                position=position,
            )
        )

    box = _parse_box(lines[count + 2], str(path), count + 3)
    return GroFrame(title=title, atoms=atoms, box=box)


def _parse_box(line: str, path: str, line_number: int) -> np.ndarray | None:
    """Parse a GRO box line, orthorhombic or triclinic."""
    values: Sequence[float]
    try:
        values = [float(value) for value in line.split()]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: malformed box line") from exc
    if not values:
        return None
    if len(values) == 3:
        box = np.diag(values)
    elif len(values) == 9:
        # GROMACS order: xx yy zz xy xz yx yz zx zy, rows are cell vectors
        box = np.array(
            [
                [values[0], values[3], values[4]],
                [values[5], values[1], values[6]],
                [values[7], values[8], values[2]],
            ],
            dtype=float,
        )
    else:
        raise ValueError(
            f"{path}:{line_number}: GRO box line must hold 3 or 9 values, "
            f"found {len(values)}"
        )
    if float(np.linalg.det(box)) <= 0.0:
        raise ValueError(f"{path}:{line_number}: periodic box has non-positive volume")
    return box


def read_gro_atoms(path: str | Path) -> List[GroAtom]:
    """Atom records only, for callers that ignore the periodic cell."""
    return read_gro(path).atoms


def read_gro_atom_names(path: str | Path) -> List[str]:
    """Atom names in file order."""
    return read_gro(path).atom_names
