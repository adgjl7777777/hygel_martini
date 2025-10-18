import subprocess
import os
import sys

def convert_gro_to_pdb(gro_path, pdb_path, gmx_path):
    """Converts a .gro file to a .pdb file using gmx editconf."""
    command = [gmx_path, 'editconf', '-f', gro_path, '-o', pdb_path]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully converted {gro_path} to {pdb_path}")
        return pdb_path
    except FileNotFoundError:
        print(f"Error: '{gmx_path}' command not found. Is GROMACS installed and in your PATH?", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error converting {gro_path} to {pdb_path}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

def convert_pdb_to_gro(pdb_path, gro_path, gmx_path, box_size_nm=None):
    """Converts a .pdb file to a .gro file using gmx editconf."""
    command = [gmx_path, 'editconf', '-f', pdb_path, '-o', gro_path]
    if box_size_nm:
        command.extend(['-box', str(box_size_nm), str(box_size_nm), str(box_size_nm)])
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Successfully converted {pdb_path} to {gro_path}")
        return gro_path
    except FileNotFoundError:
        print(f"Error: '{gmx_path}' command not found. Is GROMACS installed and in your PATH?", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error converting {pdb_path} to {gro_path}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
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

        raise

def run_packmol(packmol_path, input_content, output_dir):
    """
    Generates a Packmol input file and runs Packmol.
    """
    inp_filename = os.path.join(output_dir, "packmol_input.inp")
    with open(inp_filename, 'w') as f:
        f.write(input_content)

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

        if "ERROR" in result.stdout or "Could not pack the molecules in the required number of tries." in result.stdout:
            raise subprocess.CalledProcessError(1, command, stdout=result.stdout, stderr=result.stderr)

    except FileNotFoundError:
        print(f"Error: Packmol executable not found: '{packmol_path}'", file=sys.stderr)
        raise
    except subprocess.CalledProcessError as e:
        print("Error: Packmol execution failed.", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

def pack_system_with_molecules(base_structure_gro, molecules_to_add, final_output_gro, box_size_nm, sim_params):
    """
    Runs a full packing step: converts to PDB, generates packmol input, runs packmol, and converts back to GRO.

    Args:
        base_structure_gro (str): Path to the base .gro file (e.g., initial hydrogel).
        molecules_to_add (list): A list of dictionaries, where each dict contains 'file' (path to .gro) and 'number'.
        final_output_gro (str): Path for the final, packed .gro file.
        box_size_nm (float): The box size in nanometers.
        sim_params (dict): Dictionary of simulation parameters from the config.
    """
    print(f"\n--- pack_system_with_molecules: Packing into {final_output_gro} ---")
    output_dir = sim_params['output_dir']
    gmx_path = sim_params.get('gromacs_executable_path') or 'gmx_mpi'
    packmol_path = sim_params['packmol_path']
    packmol_threshold = sim_params['packmol_threshold']

    # 1. Convert all source .gro files to .pdb for packmol
    base_structure_pdb = convert_gro_to_pdb(
        base_structure_gro,
        os.path.join(output_dir, f"{os.path.splitext(os.path.basename(base_structure_gro))[0]}.pdb"),
        gmx_path
    )
    
    molecules_pdb_to_add = []
    for mol in molecules_to_add:
        pdb_path = convert_gro_to_pdb(
            mol['file'],
            os.path.join(output_dir, f"{os.path.splitext(os.path.basename(mol['file']))[0]}.pdb"),
            gmx_path
        )
        molecules_pdb_to_add.append({"file": pdb_path, "number": mol['number']})

    # 2. Generate packmol input content
    box_size_angstrom = box_size_nm * 10
    box_center = box_size_angstrom / 2.0
    temp_output_pdb = os.path.join(output_dir, "temp_packed_output.pdb")

    lines = [
        f"tolerance {packmol_threshold}",
        "filetype pdb",
        f"output {os.path.abspath(temp_output_pdb)}",
        "",
        f"structure {os.path.abspath(base_structure_pdb)}",
        "  number 1",
        f"  fixed {box_center:.4f} {box_center:.4f} {box_center:.4f} 0. 0. 0.",
        "end structure",
        ""
    ]
    for mol in molecules_pdb_to_add:
        if mol['number'] > 0:
            lines.extend([
                f"structure {os.path.abspath(mol['file'])}",
                f"  number {mol['number']}",
                f"  inside box 0. 0. 0. {box_size_angstrom:.4f} {box_size_angstrom:.4f} {box_size_angstrom:.4f}",
                "end structure",
                ""
            ])
    packmol_inp_content = "\n".join(lines)

    # 3. Run packmol
    run_packmol(packmol_path, packmol_inp_content, output_dir)

    # 4. Convert the packmol output PDB to the final GRO file
    convert_pdb_to_gro(temp_output_pdb, final_output_gro, gmx_path, box_size_nm)
    
    print(f"--- Packing step complete. Final system at: {final_output_gro} ---")
    return final_output_gro