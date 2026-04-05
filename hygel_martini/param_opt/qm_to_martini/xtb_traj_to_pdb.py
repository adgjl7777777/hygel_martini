from __future__ import annotations

import sys
from pathlib import Path


def parse_frames_streaming(path: Path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        while True:
            line = f.readline()
            while line and not line.strip():
                line = f.readline()
            if not line:
                break
            try:
                natoms = int(line.strip())
            except ValueError:
                break
            f.readline()
            atom_lines = []
            for _ in range(natoms):
                atom_line = f.readline()
                if not atom_line:
                    break
                atom_lines.append(atom_line.strip())
            if len(atom_lines) == natoms:
                yield atom_lines
            else:
                break


def pdb_atom_line(atom_index: int, symbol: str, x: float, y: float, z: float) -> str:
    atom_name = symbol[:2].upper().rjust(2)
    return (
        f"ATOM  {atom_index:5d} {atom_name:<4} MOL A{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol[:2].upper():>2}\n"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print('Usage: xtb_traj_to_pdb.py input.xyztraj output.pdb', file=sys.stderr)
        return 1
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    with output_path.open('w', encoding='utf-8') as out_f:
        for model_index, atom_lines in enumerate(parse_frames_streaming(input_path), start=1):
            out_f.write(f'MODEL     {model_index}\n')
            for atom_index, raw in enumerate(atom_lines, start=1):
                parts = raw.split()
                if len(parts) < 4:
                    continue
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    out_f.write(pdb_atom_line(atom_index, symbol, x, y, z))
                except ValueError:
                    continue
            out_f.write('ENDMDL\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
