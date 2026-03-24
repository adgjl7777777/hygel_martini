from __future__ import annotations

from typing import Any, Dict

from ..polymer_maker.maker import DEFAULT_MONOMER_FILES


def _default_monomers() -> Dict[str, Dict[str, Any]]:
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
        "workdir_name": "relax_xtb_geoopt",
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
            "execute": False,
            "output_dirname": "bartender_job",
        },
        "postprocess": {
            "collect": True,
            "merge": False,
            "label_map_path": None,
            "collect_json": "bartender_summary.json",
            "merged_itp": "merged_forcefield.itp",
            "merged_json": "merged_forcefield.json",
        },
    },
}
