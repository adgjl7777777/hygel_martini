from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ase.io import read, write

from ..polymer_maker.maker import build_polymer, load_monomer_library

from hygel_martini.core.physics import estimate_water_molecules, water_density_g_cm3
from hygel_martini.core.utils import ensure_min_box_nm, parse_csv_list
from .defaults import NM_TO_ANGSTROM
from .writers import (
    write_gromacs_mdp_templates,
    write_packmol_input,
    write_pipeline_script,
    write_text,
    write_topol_stub,
)


def _sequence_stem(tokens: List[str]) -> str:
    if all(len(tok) == 1 for tok in tokens):
        return "".join(tokens)
    return "_".join(tokens)


def _parse_sequence_entry(entry: Any, monomer_keys: set[str]) -> List[str]:
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            raise ValueError("Empty sequence entry is not allowed")
        if "," in text:
            tokens = parse_csv_list(text)
        elif " " in text:
            tokens = [tok for tok in text.split() if tok]
        elif text in monomer_keys:
            tokens = [text]
        else:
            # Backward-compatible shorthand for one-letter symbol sequences (e.g. "SDC").
            tokens = list(text)
    elif isinstance(entry, (list, tuple)):
        tokens = [str(tok).strip() for tok in entry if str(tok).strip()]
    else:
        raise TypeError(f"Unsupported sequence entry type: {type(entry)!r}")

    if not tokens:
        raise ValueError("Sequence entry produced no tokens")
    return tokens


def _build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str]) -> List[List[str]]:
    explicit_sequences = system_cfg.get("sequences")
    if explicit_sequences is not None:
        if not isinstance(explicit_sequences, list):
            raise ValueError("system.sequences must be a list when provided")
        jobs = [_parse_sequence_entry(entry, monomer_keys) for entry in explicit_sequences]
        if not jobs:
            raise ValueError("system.sequences is empty")
        return jobs

    symbols: List[str] = list(system_cfg["symbols"])
    lengths: List[int] = list(system_cfg["lengths"])
    jobs: List[List[str]] = []
    for symbol in symbols:
        for n_repeat in lengths:
            jobs.append([symbol] * int(n_repeat))
    return jobs


def build_cases(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_root = Path(cfg["paths"]["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    library = load_monomer_library(cfg["monomers"], base_dir=base_dir)

    system_cfg = cfg["system"]
    water_cfg = cfg["water"]

    replicas: int = int(system_cfg["replicas"])
    cutoff_nm = float(system_cfg["cutoff_nm"])
    safety_nm = float(system_cfg["min_box_safety_nm"])
    temp_c = float(system_cfg["temperature_c"])
    solvate_tool = system_cfg["solvate_tool"]

    sequence_jobs = _build_sequence_jobs(system_cfg, set(library.keys()))

    rho = water_density_g_cm3(temp_c)

    all_cases = []
    for seq_tokens in sequence_jobs:
        for token in seq_tokens:
            if token not in library:
                raise KeyError(f"Unknown monomer token: {token}. Available: {sorted(library.keys())}")

        seq_name = _sequence_stem(seq_tokens)
        n_repeat = len(seq_tokens)
        n_torsion = n_repeat if system_cfg["n_torsion_mode"] == "repeat" else max(1, n_repeat - 1)

        case_dir = out_root / seq_name
        case_dir.mkdir(parents=True, exist_ok=True)
        polymer_xyz = case_dir / "polymer.xyz"
        build_polymer(
            seq_tokens,
            monomer_dict=library,
            n_torsion=n_torsion,
            output_filename=polymer_xyz.name,
            output_dir=case_dir,
        )

        atoms = read(polymer_xyz)

        mins = atoms.positions.min(axis=0)
        maxs = atoms.positions.max(axis=0)
        span_ang = (maxs - mins).tolist()
        max_span_ang = max(span_ang)

        box_len_nm = (max_span_ang / NM_TO_ANGSTROM) + (2.0 * cutoff_nm) + safety_nm
        box_nm = ensure_min_box_nm([box_len_nm, box_len_nm, box_len_nm], cutoff_nm, safety_nm)
        box_ang = [value * NM_TO_ANGSTROM for value in box_nm]

        n_waters = estimate_water_molecules(
            box_ang,
            rho,
            molar_mass=float(water_cfg["molar_mass_g_per_mol"]),
            avogadro=float(water_cfg["avogadro"]),
        )

        # GROMACS editconf cannot read .xyz directly; keep a PDB copy for gmx input.
        write(case_dir / "polymer.pdb", atoms)

        # If polymer_itp points to an existing file under base_dir, copy it into the
        # case directory and include it with a local filename.
        case_cfg = cfg
        polymer_itp_ref = str(cfg["topology"].get("polymer_itp", "polymer.itp"))
        polymer_itp_src = Path(polymer_itp_ref)
        if polymer_itp_src.is_absolute():
            raise ValueError(
                "topology.polymer_itp must be a paths.base_dir-relative path, not an absolute path"
            )
        polymer_itp_src = (base_dir / polymer_itp_ref).resolve()
        if polymer_itp_src.exists() and base_dir in polymer_itp_src.parents:
            local_itp_name = Path(polymer_itp_ref).name
            shutil.copyfile(polymer_itp_src, case_dir / local_itp_name)
            case_cfg = copy.deepcopy(cfg)
            case_cfg["topology"]["polymer_itp"] = local_itp_name

        write_gromacs_mdp_templates(case_dir, case_cfg)
        write_topol_stub(case_dir / "topol.top", case_cfg)

        replicas_info = []
        for rep_idx in range(1, replicas + 1):
            rep_dir = case_dir / f"replica_{rep_idx:02d}"
            rep_dir.mkdir(parents=True, exist_ok=True)

            seed = 1000 + rep_idx + (n_repeat * 100)
            if solvate_tool == "packmol":
                write_packmol_input(
                    path=rep_dir / "packmol.inp",
                    polymer_xyz=polymer_xyz,
                    output_xyz="solvated_init.xyz",
                    box_ang=box_ang,
                    n_waters=n_waters,
                    seed=seed,
                    cfg=cfg,
                )
            write_pipeline_script(rep_dir, box_nm=box_nm, cfg=case_cfg)

            replicas_info.append(
                {
                    "replica": rep_idx,
                    "path": str(rep_dir),
                    "packmol_seed": seed,
                }
            )

        all_cases.append(
            {
                "sequence": seq_name,
                "sequence_tokens": seq_tokens,
                "symbol": seq_tokens[0] if len(set(seq_tokens)) == 1 else None,
                "repeat": n_repeat,
                "polymer_path": str(polymer_xyz),
                "span_angstrom": span_ang,
                "box_angstrom": box_ang,
                "box_nm": box_nm,
                "cutoff_nm": cutoff_nm,
                "target_temp_c": temp_c,
                "water_density_g_cm3": rho,
                "estimated_water_molecules": n_waters,
                "replicas": replicas_info,
            }
        )

    result = {
        "settings": cfg,
        "cases": all_cases,
    }
    write_text(out_root / "summary.json", json.dumps(result, indent=2, ensure_ascii=False))
    return result
