from __future__ import annotations

from typing import Any, Dict

from ..structure.maker import DEFAULT_MONOMER_FILES

NM_TO_ANGSTROM = 10.0

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "base_dir": ".",
        "out_root": "constructor_output",
        "gmxrc_path": "/opt/gromacs/2026/bin/GMXRC",
    },
    "monomers": dict(DEFAULT_MONOMER_FILES),
    "system": {
        # Optional explicit sequence list. If set, this overrides symbols/lengths.
        # Examples:
        #   ["S,D,C", "S C S", "SDC"]
        #   [["SBMA", "DMAPS", "CBA"]]
        "sequences": None,
        "symbols": ["S"],
        "lengths": [1, 2, 3, 4],
        "replicas": 3,
        "temperature_c": 37.0,
        "cutoff_nm": 1.1,
        "min_box_safety_nm": 0.1,
        "n_torsion_mode": "repeat",
        "solvate_tool": "gromacs",
    },
    "sampling": {
        "dt_ps": 0.002,
        "sample_nsteps": 10_000_000,
    },
    "mdp": {
        "em_nsteps": 50000,
        "emtol": 1000.0,
        "emstep": 0.01,
        "nvt_nsteps": 250000,
        "npt_nsteps": 500000,
        "nstxout_compressed": 1000,
        "cutoff_scheme": "Verlet",
        "coulombtype": "PME",
        "rcoulomb_nm": 1.1,
        "rvdw_nm": 1.1,
        "tcoupl": "V-rescale",
        "tc_grps": "System",
        "tau_t_ps": 0.1,
        "tau_p_ps": 2.0,
        "ref_p_bar": 1.0,
        "compressibility_bar_inv": 4.5e-5,
        "constraints": "h-bonds",
        "pbc": "xyz",
        "npt_pcoupltype": "isotropic",
    },
    "water": {
        "molar_mass_g_per_mol": 18.01528,
        "avogadro": 6.02214076e23,
        "packmol_tolerance": 2.0,
        "packmol_water_structure": "water.xyz",
        "gromacs_water_model": "spc216.gro",
    },
    "runtime": {
        "default_run_mode": "none",
        "cpu_omp_threads": 40,
        "gpu_omp_threads": 5,
        "none_omp_threads": 1,
        "random_seed_fallback": 123456789,
    },
    "topology": {
        "forcefield_include": "oplsaa.ff/forcefield.itp",
        "water_include": "oplsaa.ff/tip4p.itp",
        "polymer_itp": "polymer.itp",
        "system_name": "polymer in water",
        "molecule_name": "POLYMER",
        "molecule_count": 1,
        "water_molecule_name": "SOL",
        "water_molecule_count": 0,
    },
}
