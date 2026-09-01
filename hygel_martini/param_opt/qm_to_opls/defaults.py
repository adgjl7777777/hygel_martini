"""Default configuration for the stage-01 (QM -> OPLS) workflow.

Holds the single ``DEFAULT_CONFIG`` dict that ``cli.main`` can dump as a
starting template and that ``generator.run_qm_to_opls`` merges under the
user's config file (user values win).  Per the param_opt layout rules,
workflow defaults live only inside this package.
"""

from __future__ import annotations

from typing import Any, Dict

from ..polymer_maker.maker import DEFAULT_MONOMER_FILES


#: Template config merged beneath the user's YAML/JSON config.
#: - paths: base_dir (resolve monomer files against), out_root (where
#:   per-sequence ORCA input dirs are created), orca_path (ORCA binary).
#: - monomers: symbol -> xyz file mapping, copied from polymer_maker.
#: - system.n_torsion_mode: "repeat" uses one torsion step per monomer
#:   (see orca_runner.generate_orca_inputs).
#: - dft: ORCA keyword-line pieces plus %pal/%geom settings; nprocs
#:   defaults to 1 to avoid oversubscribing shared nodes.
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
