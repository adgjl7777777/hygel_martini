import subprocess
import os
import sys
import shutil
import numpy as np
from hygel_martini.hydrogel_builder.config_params.config import Config


def _normalize_box_lengths(box_lengths_nm, fallback=None):
    """
    Returns a [lx, ly, lz] list in nm. Accepts scalar or iterable input.
    """
    if box_lengths_nm is None:
        if fallback is None:
            return None
        try:
            val = float(fallback)
        except (TypeError, ValueError):
            return None
        return [val, val, val]

    if isinstance(box_lengths_nm, (list, tuple)):
        values = [float(v) for v in box_lengths_nm if v is not None]
    else:
        try:
            val = float(box_lengths_nm)
            values = [val]
        except (TypeError, ValueError):
            return None

    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0], values[0]]
    if len(values) >= 3:
        return [values[0], values[1], values[2]]
    # If only two values are provided, duplicate the second for z
    return [values[0], values[1], values[1]]


def _gro_to_pdb_manual(gro_path, pdb_path):
    try:
        with open(gro_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        num_atoms = int(lines[1].strip())
        atom_lines = lines[2:2 + num_atoms]
        with open(pdb_path, 'w', encoding='utf-8') as out:
            out.write("HEADER    Converted manually from GRO\n")
            for idx, line in enumerate(atom_lines, start=1):
                resid = int(line[0:5])
                resname = line[5:10].strip() or "RES"
                atom_name = line[10:15].strip() or "X"
                atom_id = idx
                x = float(line[20:28]) * 10.0
                y = float(line[28:36]) * 10.0
                z = float(line[36:44]) * 10.0
                element = atom_name[0].upper() if atom_name else 'X'
                out.write(f"ATOM  {atom_id:5d} {atom_name:<4} {resname:>3} A{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}\n")
            out.write("TER\nEND\n")
        print(f"[Fallback] Converted {gro_path} to {pdb_path} without GROMACS.")
        Config.debug_log(f"[Fallback] Converted {gro_path} to {pdb_path} manually.")
        return pdb_path
    except Exception as exc:
        print(f"[Fallback] Failed to manually convert {gro_path} to {pdb_path}: {exc}", file=sys.stderr)
        Config.debug_log(f"[Fallback] Failed manual GRO->PDB for {gro_path}: {exc}")
        raise


def convert_gro_to_pdb(gro_path, pdb_path, gmx_path):
    """Converts a .gro file to a .pdb file using gmx editconf or a manual fallback."""
    command = [gmx_path, 'editconf', '-f', gro_path, '-o', pdb_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully converted {gro_path} to {pdb_path}")
        Config.debug_log(f"editconf gro->pdb success: {gro_path} -> {pdb_path}")
        return pdb_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[WARN] Unable to run '{' '.join(command)}' ({e}). Falling back to manual conversion.", file=sys.stderr)
        Config.debug_log(f"[WARN] editconf gro->pdb failed for {gro_path}: {e}")
        return _gro_to_pdb_manual(gro_path, pdb_path)

def convert_pdb_to_gro(pdb_path, gro_path, gmx_path, box_lengths_nm=None):
    """Converts a .pdb file to a .gro file using gmx editconf or a manual fallback."""
    lengths_nm = _normalize_box_lengths(box_lengths_nm, None)
    command = [gmx_path, 'editconf', '-f', pdb_path, '-o', gro_path]
    if lengths_nm:
        command.extend(['-box', f"{lengths_nm[0]:.5f}", f"{lengths_nm[1]:.5f}", f"{lengths_nm[2]:.5f}"])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully converted {pdb_path} to {gro_path}")
        Config.debug_log(f"editconf pdb->gro success: {pdb_path} -> {gro_path} box={lengths_nm}")
        return gro_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[WARN] Unable to run '{' '.join(command)}' ({e}). Falling back to manual conversion.", file=sys.stderr)
        Config.debug_log(f"[WARN] editconf pdb->gro failed for {pdb_path}: {e}")
        return _pdb_to_gro_manual(pdb_path, gro_path, lengths_nm)


def _pdb_to_gro_manual(pdb_path, gro_path, box_lengths_nm=None):
    lengths_nm = _normalize_box_lengths(box_lengths_nm, 10.0)
    lx, ly, lz = lengths_nm if lengths_nm else (10.0, 10.0, 10.0)
    atoms = []
    try:
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_id = int(line[6:11])
                    atom_name = line[12:16].strip() or "X"
                    resname = line[17:20].strip() or "RES"
                    resid = int(line[22:26])
                    x = float(line[30:38]) / 10.0
                    y = float(line[38:46]) / 10.0
                    z = float(line[46:54]) / 10.0
                    atoms.append((resid, resname, atom_name, atom_id, x, y, z))
        with open(gro_path, 'w', encoding='utf-8') as out:
            out.write("Converted manually from PDB\n")
            out.write(f"{len(atoms):5d}\n")
            for resid, resname, atom_name, atom_id, x, y, z in atoms:
                out.write(f"{resid%100000:5d}{resname:<5}{atom_name:<5}{atom_id%100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
            out.write(f"{lx:10.5f}{ly:10.5f}{lz:10.5f}\n")
        print(f"[Fallback] Converted {pdb_path} to {gro_path} without GROMACS.")
        Config.debug_log(f"[Fallback] Converted {pdb_path} to {gro_path} manually with box {lx,ly,lz}")
        return gro_path
    except Exception as exc:
        print(f"[Fallback] Failed to manually convert {pdb_path} to {gro_path}: {exc}", file=sys.stderr)
        Config.debug_log(f"[Fallback] Failed manual PDB->GRO for {pdb_path}: {exc}")
        raise

def convert_xyz_to_gro(xyz_path, gro_path, gmx_path, molecule_name="MOL"):

    """Converts a .xyz file to a .gro file by manually parsing it."""

    try:

        with open(xyz_path, 'r', encoding='utf-8-sig') as f_xyz:

            lines = f_xyz.readlines()

        

        num_atoms = int(lines[0].strip())

        comment = lines[1].strip()

        atom_lines = lines[2:2+num_atoms]



        with open(gro_path, 'w') as f_gro:

            f_gro.write(f"{comment}\n")

            f_gro.write(f" {num_atoms}\n")

            

            for i, line in enumerate(atom_lines):

                parts = line.split()

                atom_name = parts[0]

                x = float(parts[1]) / 10.0

                y = float(parts[2]) / 10.0

                z = float(parts[3]) / 10.0

                

                f_gro.write(f"{'1':>5s}{molecule_name:<5s}{atom_name:<5s}{i+1:>5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")



            f_gro.write("   10.00000   10.00000   10.00000\n")



        print(f"Successfully converted {xyz_path} to {gro_path} manually.")

        return gro_path



    except Exception as e:

        print(f"Error manually converting {xyz_path} to {gro_path}:", file=sys.stderr)

        print(e, file=sys.stderr)

def _read_gro_atom_names(gro_path):
    try:
        with open(gro_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        atom_count = int(lines[1].strip())
        atom_lines = lines[2:2 + atom_count]
        names = [line[10:15].strip() for line in atom_lines]
        return names
    except Exception as exc:
        print(f"[WARN] Failed to read atom names from {gro_path}: {exc}", file=sys.stderr)
        Config.debug_log(f"[WARN] Failed to read atom names from {gro_path}: {exc}")
        return []

def _restore_atom_names_from_sources(output_gro, base_gro, molecules_to_add):
    expected_names = []
    base_names = _read_gro_atom_names(base_gro)
    if not base_names:
        return False
    expected_names.extend(base_names)

    for mol in molecules_to_add:
        mol_names = _read_gro_atom_names(mol['file'])
        if not mol_names:
            return False
        count = int(mol.get('number', 0))
        for _ in range(count):
            expected_names.extend(mol_names)

    try:
        with open(output_gro, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        atom_count = int(lines[1].strip())
        if atom_count != len(expected_names):
            msg = (
                f"[WARN] Atom count mismatch for {output_gro}: "
                f"gro={atom_count}, expected={len(expected_names)}"
            )
            print(msg, file=sys.stderr)
            Config.debug_log(msg)
            return False
        atom_lines = lines[2:2 + atom_count]
        updated = []
        for idx, line in enumerate(atom_lines):
            atom_name = expected_names[idx]
            updated.append(f"{line[:10]}{atom_name:<5}{line[15:]}")
        lines[2:2 + atom_count] = updated
        with open(output_gro, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        Config.debug_log(f"[INFO] Restored atom names in {output_gro}")
        return True
    except Exception as exc:
        print(f"[WARN] Failed to restore atom names in {output_gro}: {exc}", file=sys.stderr)
        Config.debug_log(f"[WARN] Failed to restore atom names in {output_gro}: {exc}")
        return False

        raise

def run_packmol(packmol_path, inp_filename, output_dir, sim_params=None):
    """
    Generates a Packmol input file and runs Packmol.
    """
    print("\n--- Running Packmol... ---")
    print(f"Input file generated at: {inp_filename}")

    command = [packmol_path]

    try:
        with open(inp_filename, 'r') as f:
            result = subprocess.run(
                command,
                stdin=f,
                check=True,
                capture_output=True,
                text=True,
                cwd=output_dir
            )
        print("Packmol execution successful.")
        print(result.stdout)
        if result.stderr:
            print("Packmol stderr:", result.stderr, file=sys.stderr)
        Config.debug_log(f"Packmol success for {inp_filename}")

        if "ERROR" in result.stdout or "Could not pack the molecules in the required number of tries." in result.stdout:
            raise subprocess.CalledProcessError(1, command, stdout=result.stdout, stderr=result.stderr)

    except FileNotFoundError:
        print(f"Error: Packmol executable not found: '{packmol_path}'", file=sys.stderr)
        # In test mode, allow skipping without raising
        if sim_params and sim_params.get('test_mode'):
            Config.debug_log(f"Packmol not found, test_mode skip for {inp_filename}")
            return None
        raise
    except subprocess.CalledProcessError as e:
        print("Error: Packmol execution failed.", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        Config.debug_log(f"Packmol failed for {inp_filename}: {e}")
        raise

def pack_system_with_molecules(step_name, base_structure_gro, molecules_to_add, final_output_gro, box_lengths_nm, sim_params):
    """
    Runs a full packing step: converts to PDB, generates packmol input, runs packmol, and converts back to GRO.
    Intermediate files are saved with step-specific names for debugging.

    Args:
        step_name (str): The name of the current packing step (e.g., "Add_Polymer").
        base_structure_gro (str): Path to the base .gro file (e.g., initial hydrogel).
        molecules_to_add (list): A list of dictionaries, where each dict contains 'file' (path to .gro) and 'number'.
        final_output_gro (str): Path for the final, packed .gro file.
        box_lengths_nm (Sequence[float]): The box lengths (nm) along x/y/z.
        sim_params (dict): Dictionary of simulation parameters from the config.
    """
    print(f"\n--- pack_system_with_molecules: Packing into {final_output_gro} ---")
    output_dir = sim_params['output_dir']
    gmx_path = sim_params.get('gromacs_executable_path') or 'gmx_mpi'
    packmol_path = sim_params['packmol_path']
    packmol_threshold = sim_params['packmol_threshold']
    resolved_lengths_nm = _normalize_box_lengths(box_lengths_nm, sim_params.get('box_size_nm'))
    if not resolved_lengths_nm:
        raise ValueError("Box lengths could not be determined for packmol packing step.")

    # 1. Convert all source .gro files to .pdb for packmol, using step-specific names
    base_pdb_name = f"{step_name}_base.pdb"
    base_structure_pdb = convert_gro_to_pdb(
        base_structure_gro,
        os.path.join(output_dir, base_pdb_name),
        gmx_path
    )
    # Compute centroid of base pdb to avoid unwanted translation
    base_centroid = None
    try:
        coords = []
        with open(base_structure_pdb, 'r') as pdb_f:
            for line in pdb_f:
                if line.startswith(("ATOM", "HETATM")):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
        if coords:
            arr = np.array(coords, dtype=float)
            base_centroid = arr.mean(axis=0)
            Config.debug_log(f"Computed base centroid for {base_structure_pdb}: {base_centroid}")
    except Exception as exc:
        print(f"[WARN] Failed to compute centroid for {base_structure_pdb}: {exc}", file=sys.stderr)
        Config.debug_log(f"[WARN] Centroid computation failed for {base_structure_pdb}: {exc}")
        base_centroid = None
    molecules_pdb_to_add = []
    for mol in molecules_to_add:
        mol_name = os.path.splitext(os.path.basename(mol['file']))[0]
        pdb_name = f"{step_name}_{mol_name}.pdb"
        pdb_path = convert_gro_to_pdb(
            mol['file'],
            os.path.join(output_dir, pdb_name),
            gmx_path
        )
        molecules_pdb_to_add.append({"file": pdb_path, "number": mol['number']})

    # 2. Generate packmol input content with step-specific filenames
    box_lengths_angstrom = [val * 10.0 for val in resolved_lengths_nm]
    box_center = [val / 2.0 for val in box_lengths_angstrom]
    packed_output_pdb = os.path.join(output_dir, f"{step_name}_packed.pdb")
    packmol_inp_filename = os.path.join(output_dir, f"{step_name}_packmol.inp")

    lines = [
        f"tolerance {packmol_threshold}",
        "filetype pdb",
        f"output {os.path.abspath(packed_output_pdb)}",
        "",
        f"structure {os.path.abspath(base_structure_pdb)}",
        "  number 1",
        (f"  fixed {base_centroid[0]:.4f} {base_centroid[1]:.4f} {base_centroid[2]:.4f} 0. 0. 0."
         if base_centroid is not None else
         f"  fixed {box_center[0]:.4f} {box_center[1]:.4f} {box_center[2]:.4f} 0. 0. 0."),
        "end structure",
        ""
    ]
    for mol in molecules_pdb_to_add:
        if mol['number'] > 0:
            lines.extend([
                f"structure {os.path.abspath(mol['file'])}",
                f"  number {mol['number']}",
                f"  inside box 0. 0. 0. {box_lengths_angstrom[0]:.4f} {box_lengths_angstrom[1]:.4f} {box_lengths_angstrom[2]:.4f}",
                "end structure",
                ""
            ])
    packmol_inp_content = "\n".join(lines)
    
    with open(packmol_inp_filename, 'w') as f:
        f.write(packmol_inp_content)

    # 3. Run packmol
    packmol_result = run_packmol(packmol_path, packmol_inp_filename, output_dir, sim_params=sim_params)
    if packmol_result is None and sim_params.get('test_mode'):
        print(f"[TEST_MODE] Packmol skipped; copying base structure to {final_output_gro} without adding molecules.")
        shutil.copy(base_structure_gro, final_output_gro)
        return final_output_gro, False

    # 4. Convert the packmol output PDB to the final GRO file
    convert_pdb_to_gro(packed_output_pdb, final_output_gro, gmx_path, resolved_lengths_nm)
    _restore_atom_names_from_sources(final_output_gro, base_structure_gro, molecules_to_add)
    
    print(f"--- Packing step complete. Final system at: {final_output_gro} ---")
    print(f"--- Intermediate files saved with prefix '{step_name}_' in {output_dir} ---")
    return final_output_gro, True
