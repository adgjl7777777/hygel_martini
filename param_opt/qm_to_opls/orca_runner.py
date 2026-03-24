from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import itertools
from ase.io import read, write

from ..polymer_maker.maker import _sequence_output_stem, load_monomer_library

def generate_orca_inputs(cfg: Dict[str, Any]) -> None:
    """
    Generates ORCA input files for N-mers (up to 200 atoms).
    Automatically caps Br with H using existing maker.py logic.
    """
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Load monomer library
    monomer_dict = load_monomer_library(cfg["monomers"], base_dir=base_dir)
    symbols = list(monomer_dict.keys())
    
    # DFT settings
    dft_cfg = cfg.get("dft", {
        "method": "B3LYP", "basis_set": "def2-TZVP", "dispersion": "D3BJ",
        "solvent": "CPCM(Water)", "extra_flags": "TightSCF KeepDens",
        "nprocs": 1, "charge": 0, "multiplicity": 1, "max_iter": 300
    })
    
    # 2. Explore N-mer combinations up to 200 atoms
    max_atoms = 200
    
    for length in range(1, 11):  # Max 10-mer safety break
        found_valid_in_length = False
        for seq_tuple in itertools.product(symbols, repeat=length):
            seq = list(seq_tuple)
            
            # Rough atom count estimation
            total_atoms_est = sum(len(monomer_dict[s]) for s in seq)
            if total_atoms_est > max_atoms + 20: 
                continue
            
            seq_name = _sequence_output_stem(seq)
            
            try:
                n_torsion = length if cfg.get("system", {}).get("n_torsion_mode") == "repeat" else max(1, length - 1)
                
                # Build ASE Atoms directly
                atoms = _build_atoms_for_dft(seq, monomer_dict, n_torsion)
                
                if len(atoms) > max_atoms:
                    continue
                
                found_valid_in_length = True
                
                # Write ORCA Input
                _write_orca_file(seq_name, atoms, dft_cfg, out_root)
                print(f"[ORCA Runner] Generated input for {seq_name} ({len(atoms)} atoms)")
                
            except Exception as e:
                print(f"[Error] Failed to build {seq_name}: {e}")
                
        if not found_valid_in_length:
            break

def _build_atoms_for_dft(sequence, monomer_dict, n_torsion):
    """
    Internal helper that mirrors maker.build_polymer but returns ASE Atoms.
    Used for Phase 1 DFT input generation.
    """
    import numpy as np
    from ..polymer_maker.maker import cap_ends_with_hydrogen, get_connection_info
    
    angle_increment = max(90, 360 / n_torsion)
    current_angle = 0  
    
    chain = monomer_dict[sequence[0]].copy()
    _, c1_curr, _, bc_tail_curr = get_connection_info(chain)
    current_tail_c_idx = c1_curr
    current_tail_bc_idx = bc_tail_curr

    for symbol in sequence[1:]:
        new_monomer = monomer_dict[symbol].copy()
        c0_new, c1_new, bc_head_new, bc_tail_new = get_connection_info(new_monomer)
        
        v_target = chain.positions[current_tail_bc_idx] - chain.positions[current_tail_c_idx]
        v_source = new_monomer.positions[c0_new] - new_monomer.positions[bc_head_new]
        new_monomer.rotate(v_source, v_target)
        
        current_angle += angle_increment 
        new_monomer.rotate(current_angle, v_target, center=new_monomer.positions[c0_new])

        target_pos = chain.positions[current_tail_c_idx] + (v_target / np.linalg.norm(v_target)) * 1.513
        translation_vec = target_pos - new_monomer.positions[c0_new]
        new_monomer.translate(translation_vec)
        
        del chain[current_tail_bc_idx]
        del new_monomer[bc_head_new]
        
        index_shift = len(chain)
        bc_tail_adjusted = bc_tail_new - 1 if bc_tail_new > bc_head_new else bc_tail_new
        c1_adjusted = c1_new - 1 if c1_new > bc_head_new else c1_new
        
        chain += new_monomer
        current_tail_c_idx = index_shift + c1_adjusted
        current_tail_bc_idx = index_shift + bc_tail_adjusted

    return cap_ends_with_hydrogen(chain)

def _write_orca_file(name, atoms, dft_cfg, out_root):
    header = f"! {dft_cfg.get('method', '')} {dft_cfg.get('basis_set', '')} {dft_cfg.get('dispersion', '')} {dft_cfg.get('solvent', '')} {dft_cfg.get('extra_flags', '')}\n"
    header += "%pal\n"
    header += f"   nprocs {dft_cfg.get('nprocs', 1)}\n"
    header += "end\n"
    header += "%geom\n"
    header += f"   MaxIter {dft_cfg.get('max_iter', 300)}\n"
    header += "end\n"
    header += f"* xyz {dft_cfg.get('charge', 0)} {dft_cfg.get('multiplicity', 1)}\n"
    
    coords = ""
    for atom in atoms:
        coords += f"{atom.symbol:2s} {atom.position[0]:12.6f} {atom.position[1]:12.6f} {atom.position[2]:12.6f}\n"
    footer = "*\n"
    
    target_dir = out_root / name
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{name}_opt.inp").write_text(header + coords + footer)
    write(target_dir / f"{name}_capped.xyz", atoms)
