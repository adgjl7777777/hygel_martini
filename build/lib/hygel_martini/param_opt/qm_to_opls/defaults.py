from __future__ import annotations

from typing import Any, Dict

from ..polymer_maker.maker import DEFAULT_MONOMER_FILES


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "base_dir": ".",
        "out_root": "output",
        "orca_path": "/opt/orca/orca",
    },
    "monomers": dict(DEFAULT_MONOMER_FILES),
    "system": {
        "n_torsion_mode": "repeat",
    },
    "dft": {
        "method": "B3LYP",
        "basis_set": "def2-TZVP",
        "dispersion": "D3BJ",
        "solvent": "CPCM(Water)",
        "extra_flags": "TightSCF KeepDens",
        "nprocs": 1,
        "charge": 0,
        "multiplicity": 1,
        "max_iter": 300,
    },
}
