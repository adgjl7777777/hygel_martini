from __future__ import annotations

from typing import Any, Dict

from ..polymer_maker.maker import DEFAULT_MONOMER_FILES

NM_TO_ANGSTROM = 10.0

DEFAULT_CONFIG: Dict[str, Any] = {
    "workflow": {
        # constructor: legacy OPLS input / GROMACS setup generator.
        # existing_data_fit: consume existing OPLS/GROMACS data and prepare
        # trim + Bartender refit jobs without launching an MD production run.
        "mode": "constructor",
    },
    "paths": {
        "base_dir": ".",
        "out_root": "constructor_output",
        "gmxrc_path": "/opt/gromacs/2026/bin/GMXRC",
        "postprocess_mirror_root": "opls_bartender_runs",
        "postprocess_output_root": "postprocessing_result",
    },
    "tools": {
        "gmx": "gmx_mpi",
        "gmxrc_path": "/opt/gromacs/2026/bin/GMXRC",
    },
    "monomers": dict(DEFAULT_MONOMER_FILES),
    "system": {
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
        "cpu_omp_threads": 1,
        "gpu_omp_threads": 1,
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
    "opls_data": {
        "cases": [],
        "trim": {
            "auto_trim": True,
            "method": "pymbar",
            "skip_frames": 0,
            "nskip": 1,
            "max_trim_fraction": 1.0,
            "fast": True,
            "ref_fraction": 0.2,
            "threshold_sigma": 1.0,
            "energy_term": "Potential",
            "write_plots": True,
            "trjconv_selections": ["System"],
            "trjconv_extra": [],
        },
        "execution": {
            "mode": "",
            "run_trim": False,
            "run_bartender": False,
        },
    },
    "bartender_pipeline": {
        # 02 names the existing trajectory source "md" instead of "xtb".
        # Aliases accepted by the runner: existing, gromacs, bartender-noxtb.
        #   md:       prepare/trim existing MD trajectory, then run Bartender.
        #   md_notrim: convert/use existing MD trajectory without auto-trim.
        #   trim:     prepare/trim trajectory only; no Bartender script.
        #   off:      metadata/case scaffolding only.
        "md": "md",
        "bartender": {
            "enabled": True,
            "root": "",
            "env_script": "",
            "binary": "bartender",
            "cpus": 1,
            "charge": 0,
            "skip": 1,
            "output_dirname": "bartender_job",
        },
        "postprocess": {
            "screening": {
                "enabled": False,
                "potentials": {
                    "angles": "bartender",
                    "dihedrals": "bartender",
                    "impropers": "bartender",
                },
                "bond_constraint_mode": "bartender",
                "candidate_source": "active",
                "show_all_info": True,
                "multi_constant_metric": "max_abs",
                "write_plots": True,
                "thresholds": {
                    "force_metric_min_mode": "absolute",
                    "force_metric_min": {
                        "bonds": 0.0,
                        "constraints": 0.0,
                        "angles": 0.0,
                        "dihedrals": 0.0,
                        "impropers": 0.0,
                    },
                    "rmsd_max": 10.0,
                },
            },
        },
    },
}
