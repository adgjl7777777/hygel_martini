"""
Gro file parsing utilities.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GroAtom:
    index: int
    residue_number: int
    residue_name: str
    atom_name: str
    position: np.ndarray


def read_gro_atoms(path: str) -> List[GroAtom]:
    """
    Parse a .gro file and return atom entries (box vector is ignored).
    """
    atoms: List[GroAtom] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        handle.readline()  # title
        count = int(handle.readline().strip())
        for idx in range(count):
            line = handle.readline()
            if not line:
                raise ValueError(f"GRO file {path} ended prematurely at atom {idx+1}")
            resnr_str = line[0:5].strip()
            if not resnr_str:
                raise ValueError(f"GRO line '{line.rstrip()}'에서 residue number를 파싱할 수 없습니다.")
            resnr = int(resnr_str)
            residu = line[5:10].strip()
            atom = line[10:15].strip()
            remainder = line[15:].split()
            if len(remainder) < 4:
                raise ValueError(f"GRO line '{line.rstrip()}'가 예상되는 필드 수보다 적습니다.")
            atom_id = int(remainder[0])
            x = float(remainder[1])
            y = float(remainder[2])
            z = float(remainder[3])
            atoms.append(
                GroAtom(
                    index=atom_id,
                    residue_number=resnr,
                    residue_name=residu,
                    atom_name=atom,
                    position=np.array([x, y, z], dtype=np.float64),
                )
            )
        # skip box line
    return atoms
