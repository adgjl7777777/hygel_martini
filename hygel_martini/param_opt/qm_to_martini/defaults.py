"""Built-in default configuration for the stage-03 (qm_to_martini) pipeline.

This module owns ``DEFAULT_CONFIG``, the complete default settings tree
that ``cli.main --dump-default-config`` writes out and that user YAML
configs are conceptually diffs against. Keys mirror the sections consumed
by ``config.resolve_*`` (``paths``, ``system``, ``monomers``,
``bartender_pipeline`` with its xtb/orca/bartender/postprocess subtrees).

Callers: ``cli`` (config dump) imports ``DEFAULT_CONFIG``; the monomer
defaults are derived from ``polymer_maker.maker.DEFAULT_MONOMER_FILES``
so the two stay in sync.

Units follow the consuming tools: xTB MD uses K/ps/fs (``temp_k``,
``time_ps``, ``dump_fs``, ``step_fs``); Bartender uses ps/K. Values here
are pipeline defaults, not scientific recommendations.
"""

from __future__ import annotations

from typing import Any, Dict

from ..polymer_maker.maker import DEFAULT_MONOMER_FILES


def _default_monomers() -> Dict[str, Dict[str, Any]]:
    """Derive default per-monomer entries from the built-in monomer library.

    Each library XYZ ``<stem>.xyz`` gets an init-template default of
    ``<stem>_init.inp`` plus a neutral singlet electronic state.

    Returns:
        Monomer token -> {xyz, init_template, charge, multiplicity}.
    """
    monomers: Dict[str, Dict[str, Any]] = {}
    for symbol, xyz_name in DEFAULT_MONOMER_FILES.items():
        stem = xyz_name[:-4] if xyz_name.endswith(".xyz") else xyz_name
        monomers[symbol] = {
            "xyz": xyz_name,
            "init_template": f"{stem}_init.inp",
            "charge": 0,
            "multiplicity": 1,
        }
    return monomers


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "base_dir": ".",
        "out_root": "runs",
    },
    "system": {
        "sequences": ["S"],
        "n_torsion_mode": "repeat",
    },
    "monomers": _default_monomers(),
    "bartender_pipeline": {
        "allow_invalid": False,
        "relaxation": "xtb",
        "md": "bartender",
        "md_traj": "",
        "workdir_name": "relax_xtb_geoopt",
        "execution": {
            "run_relaxation": False,
            "run_bartender": False,
            "shell": "bash",
        },
        "logs": {
            "enabled": True,
            "dirname": "logs",
            "write_validation": True,
            "capture_runtime": True,
        },
        "electronic_state": {
            "charge": None,
            "uhf": None,
            "multiplicity": None,
        },
        "xtb": {
            "env_script": "",
            "binary": "xtb",
            "gfn": 2,
            "parallel": 1,
            "opt_level": "normal",
            "opt_cycles": 10000,
            "acc": 1.0,
            "etemp": 300.0,
            "solvent_model": "alpb",
            "solvent": "water",
            "solvent_reference": "",
            "md_input_template_path": "",
            "md": {
                "temp_k": 310.0,
                "time_ps": 5000,
                "dump_fs": 50.0,
                "step_fs": 4.0,
                "velo": False,
                "hmass": 4,
                "shake": 2,
                "sccacc": 2.0,
                "restart": False,
            },
        },
        "orca": {
            "binary": "orca",
            "nprocs": 1,
            "method_line": "r2scan-3c CPCM(water) Opt TightSCF",
            "max_iter": 300,
            "input_template_path": "",
        },
        "bartender": {
            "enabled": True,
            "root": "",
            "env_script": "",
            "binary": "bartender",
            "cpus": 1,
            "charge": None,
            "time_ps": 5000,
            "temperature_k": 310.0,
            "solvent": "h2o",
            "dcd_save": "",
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
