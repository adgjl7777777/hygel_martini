import subprocess
import sys
import os
import re
from config_params.config import Config
import numpy as np
import random
def run_genion_for_neutralization(input_gro, output_gro, topology_file, sim_params, ion_params, solvent_name):
    """
    Runs the GROMACS genion tool to add ions and neutralize the system.

    Args:
        input_gro (str): Path to the input .gro file.
        output_gro (str): Path for the final output .gro file.
        topology_file (str): Path to the topology file (.top).
        sim_params (dict): Simulation parameters from config.
        ion_params (dict): Ion parameters from config.
    """
    print("="*50)
    print("Running GROMACS genion to add ions and neutralize the system...")
    print("="*50)

    output_dir = sim_params['output_dir']
    gmx_exec_path = sim_params.get('gromacs_executable_path', 'gmx_mpi')
    gmx_include_path = sim_params.get('gromacs_include_path')

    # 2. Run gmx grompp to create a .tpr file
    env = os.environ.copy()
    if gmx_include_path:
        env['GMX_INCLUDE'] = gmx_include_path

    ion_list = ion_params.get('ions')
    cation_list = []
    anion_list = []
    additional_cation_list = []
    additional_anion_list = []
    total_charge = 0
    for i in range(len(ion_list)):
        total_charge += ion_list[i].get("charge") * ion_list[i].get("number")
        if ion_list[i].get("charge")>0:
            if ion_list[i].get("additional_add",False):
                additional_cation_list.append(ion_list[i])
            else:
                cation_list.append(ion_list[i])
        elif ion_list[i].get("charge")<0:
            if ion_list[i].get("additional_add",False):
                additional_anion_list.append(ion_list[i])
            else:
                anion_list.append(ion_list[i])
    if not additional_cation_list:
        if len(cation_list) == 0:
            print("Check cation list!")
            return 1
        additional_cation_list.append(cation_list.pop(0))
    if not additional_anion_list:
        if len(anion_list) == 0:
            print("Check cation list!")
            return 1
        additional_anion_list.append(cation_list.pop(0))
    if abs(total_charge) > 1e-6 and total_charge>0:
        additional_ion_ratio = []
        additional_ion_charge = []
        for i in range(len(additional_anion_list)):
            additional_ion_ratio.append(additional_anion_list[i].get("number"))
            additional_ion_charge.append(additional_anion_list[i].get("charge"))
        additional_ion_ratio = np.array(additional_ion_ratio)
        additional_ion_charge = np.array(additional_ion_charge)
        basic_charge = np.dot(additional_ion_ratio,additional_ion_charge)
        total_mul = np.abs(total_charge / basic_charge)
        first_solution = np.floor(additional_ion_ratio * total_mul)
        first_charge = np.abs(np.dot(first_solution,additional_ion_charge))
        remained_charge = np.abs(total_charge) - first_charge
        if abs(remained_charge)>1e-6:
            max_checker= np.max(np.abs(additional_ion_charge))
            final_solution = []
            if len(additional_ion_charge) == 1:
                for i in range(max_checker):
                    if abs(abs(np.dot(np.array([i]),additional_ion_charge)) - remained_charge)<1e-6:
                        final_solution.append([i])
            elif len(additional_ion_charge) == 2:
                for i in range(max_checker):
                    for j in range(max_checker):
                        if abs(abs(np.dot(np.array([i,j]),additional_ion_charge)) - remained_charge)<1e-6:
                            final_solution.append([i,j])
            else:
                for i in range(max_checker):
                    for j in range(max_checker):
                        for k in range(max_checker):
                            if abs(abs(np.dot(np.array([i,j,k]),additional_ion_charge[:3])) - remained_charge)<1e-6:
                                final_solution.append([i,j,k])
            random.seed(Config.get_param("simulation_parameters").get("random_seed",0))
            if len(final_solution) != 0:
                selected_solution = random.choice(final_solution)
                for i in range(len(selected_solution)):
                    first_solution[i] += selected_solution[i]
        for i in range(len(first_solution)):
            additional_anion_list[i]['number'] += first_solution[i]

    elif abs(total_charge) > 1e-6 and total_charge<0:
        additional_ion_ratio = []
        additional_ion_charge = []
        for i in range(len(additional_cation_list)):
            additional_ion_ratio.append(additional_cation_list[i].get("number"))
            additional_ion_charge.append(additional_cation_list[i].get("charge"))
        additional_ion_ratio = np.array(additional_ion_ratio)
        additional_ion_charge = np.array(additional_ion_charge)
        basic_charge = np.dot(additional_ion_ratio,additional_ion_charge)
        total_mul = np.abs(total_charge / basic_charge)
        first_solution = np.floor(np.additional_ion_ratio * total_mul)
        first_charge = np.abs(np.dot(first_solution,additional_ion_charge))
        remained_charge = np.abs(total_charge) - first_charge
        if abs(remained_charge)>1e-6:
            max_checker= np.max(np.abs(additional_ion_charge))
            final_solution = []
            if len(additional_ion_charge) == 1:
                for i in range(max_checker):
                    if abs(abs(np.dot(np.array([i]),additional_ion_charge)) - remained_charge)<1e-6:
                        final_solution.append([i])
            elif len(additional_ion_charge) == 2:
                for i in range(max_checker):
                    for j in range(max_checker):
                        if abs(abs(np.dot(np.array([i,j]),additional_ion_charge)) - remained_charge)<1e-6:
                            final_solution.append([i,j])
            else:
                for i in range(max_checker):
                    for j in range(max_checker):
                        for k in range(max_checker):
                            if abs(abs(np.dot(np.array([i,j,k]),additional_ion_charge[:3])) - remained_charge)<1e-6:
                                final_solution.append([i,j,k])
            random.seed(Config.get_param("simulation_parameters").get("random_seed",0))
            if len(final_solution) != 0:
                selected_solution = random.choice(final_solution)
                for i in range(len(selected_solution)):
                    first_solution[i] += selected_solution[i]
        for i in range(len(first_solution)):
            additional_cation_list[i]['number'] += first_solution[i]
    additional_anion_list.reverse()
    additional_cation_list.reverse()
    anion_list.extend(additional_anion_list)
    cation_list.extend(additional_cation_list)
    genion_count = 0
    for i in range(len(anion_list)-1):
        temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
        temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

        # 1. Create a minimal .mdp file for grompp
        with open(temp_mdp_file, 'w') as f:
            f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

        # 2. Run gmx grompp to create a .tpr file

        grompp_command = [
            gmx_exec_path, 'grompp',
            '-f', os.path.basename(temp_mdp_file),
            '-c', os.path.basename(input_gro),
            '-p', os.path.basename(topology_file),
            '-o', os.path.basename(temp_tpr_file),
            '-maxwarn', '2' # Allow warnings for box size and charge
        ]
        print(f"\nRunning grompp: {' '.join(grompp_command)}")
        try:
            subprocess.run(grompp_command, check=True, capture_output=True, text=True, cwd=output_dir, env=env)
        except subprocess.CalledProcessError as e:
            print("--- GROMACS grompp command failed ---", file=sys.stderr)
            print(f"Return code: {e.returncode}", file=sys.stderr)
            print("\n--- stdout ---", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print("\n--- stderr ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            print("------------------------------------", file=sys.stderr)
            raise
        
        # 3. Run gmx genion
        genion_command = [
            gmx_exec_path, 'genion',
            '-s', os.path.basename(temp_tpr_file),
            '-o', os.path.basename(f"{output_gro}_{genion_count}.gro"),
            '-p', os.path.basename(topology_file),
            '-nname', anion_list[i]["ion_name"],
            '-nn', str(int(anion_list[i]["number"])),
            '-nq', str(int(anion_list[i]["charge"])),
        ]

        print(f"\nRunning genion: {' '.join(map(str, genion_command))}")
        # Provide the replacement group (e.g., 'W') to stdin
        result = subprocess.run(genion_command, check=True, cwd=output_dir, capture_output=True, text=True, input=solvent_name)
        print(result.stdout)
        if result.stderr:
            print("genion stderr:", result.stderr, file=sys.stderr)
        # 4. Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        genion_count += 1
    for i in range(len(cation_list)-1):
        temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
        temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

        # 1. Create a minimal .mdp file for grompp
        with open(temp_mdp_file, 'w') as f:
            f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

        # 2. Run gmx grompp to create a .tpr file

        grompp_command = [
            gmx_exec_path, 'grompp',
            '-f', os.path.basename(temp_mdp_file),
            '-c', os.path.basename(input_gro),
            '-p', os.path.basename(topology_file),
            '-o', os.path.basename(temp_tpr_file),
            '-maxwarn', '2' # Allow warnings for box size and charge
        ]
        print(f"\nRunning grompp: {' '.join(grompp_command)}")
        try:
            subprocess.run(grompp_command, check=True, capture_output=True, text=True, cwd=output_dir, env=env)
        except subprocess.CalledProcessError as e:
            print("--- GROMACS grompp command failed ---", file=sys.stderr)
            print(f"Return code: {e.returncode}", file=sys.stderr)
            print("\n--- stdout ---", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print("\n--- stderr ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            print("------------------------------------", file=sys.stderr)
            raise
            
        # 3. Run gmx genion
        genion_command = [
            gmx_exec_path, 'genion',
            '-s', os.path.basename(temp_tpr_file),
            '-o', os.path.basename(f"{output_gro}_{genion_count}.gro"),
            '-p', os.path.basename(topology_file),
            '-pname', cation_list[i]["ion_name"],
            '-np', str(int(cation_list[i]["number"])),
            '-pq', str(int(cation_list[i]["charge"])),
        ]

        print(f"\nRunning genion: {' '.join(map(str, genion_command))}")
        # Provide the replacement group (e.g., 'W') to stdin
        result = subprocess.run(genion_command, check=True, capture_output=True, text=True, cwd=output_dir, input=solvent_name)
        print(result.stdout)
        if result.stderr:
            print("genion stderr:", result.stderr, file=sys.stderr)
        input_gro = f"{output_gro}_{genion_count}.gro"
        # 4. Clean up temporary files
        print("\nCleaning up temporary files...")
        for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        genion_count += 1
    temp_mdp_file = os.path.join(output_dir, "temp_for_genion.mdp")
    temp_tpr_file = os.path.join(output_dir, "temp_for_genion.tpr")

    # 1. Create a minimal .mdp file for grompp
    with open(temp_mdp_file, 'w') as f:
        f.write("title       = Minimal MDP for grompp\nintegrator  = steep\nnsteps      = 0\n")

    grompp_command = [
        gmx_exec_path, 'grompp',
        '-f', os.path.basename(temp_mdp_file),
        '-c', os.path.basename(input_gro),
        '-p', os.path.basename(topology_file),
        '-o', os.path.basename(temp_tpr_file),
        '-maxwarn', '2' # Allow warnings for box size and charge
    ]
    print(f"\nRunning grompp: {' '.join(grompp_command)}")
    try:
        subprocess.run(grompp_command, check=True, capture_output=True, text=True, cwd=output_dir, env=env)
    except subprocess.CalledProcessError as e:
        print("--- GROMACS grompp command failed ---", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("\n--- stdout ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- stderr ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("------------------------------------", file=sys.stderr)
        raise
    final_gro = f"{output_gro}_end.gro"
    # 3. Run gmx genion
    genion_command = [
        gmx_exec_path, 'genion',
        '-s', os.path.basename(temp_tpr_file),
        '-o', os.path.basename(final_gro),
        '-p', os.path.basename(topology_file),
        '-nname', anion_list[-1]["ion_name"],
        '-nn', str(int(anion_list[-1]["number"])),
        '-nq', str(int(anion_list[-1]["charge"])),
        '-pname', cation_list[-1]["ion_name"],
        '-np', str(int(cation_list[-1]["number"])),
        '-pq', str(int(cation_list[-1]["charge"])),
        '-neutral'
    ]

    print(f"\nRunning genion: {' '.join(map(str, genion_command))}")
    # Provide the replacement group (e.g., 'W') to stdin
    result = subprocess.run(genion_command, check=True, capture_output=True, text=True, cwd=output_dir, input=solvent_name)
    print(result.stdout)
    if result.stderr:
        print("genion stderr:", result.stderr, file=sys.stderr)

    # Post-process the gro file to set residue name for all ions to 'ION'
    final_gro_path = os.path.join(output_dir, final_gro)
    all_ion_names = {ion['ion_name'] for ion in ion_params.get('ions', [])}

    try:
        with open(final_gro_path, 'r') as f:
            lines = f.readlines()

        if len(lines) >= 3:
            new_lines = lines[:2]  # Title and atom count

            for line in lines[2:-1]:  # Atom lines
                if len(line) > 20:
                    resname = line[5:10].strip()
                    atomname = line[10:15].strip()
                    if resname in all_ion_names or atomname in all_ion_names:
                        new_line = line[:5] + 'ION  ' + line[10:]
                        new_lines.append(new_line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)

            new_lines.append(lines[-1])  # Box vectors

            with open(final_gro_path, 'w') as f:
                f.writelines(new_lines)

            print(f"\nResidue names for all ions in '{final_gro}' have been standardized to 'ION'.")

    except FileNotFoundError:
        print(f"Warning: Output file '{final_gro_path}' not found for post-processing.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during .gro file post-processing: {e}", file=sys.stderr)

    # 4. Clean up temporary files
    print("\nCleaning up temporary files...")
    for temp_file in [temp_mdp_file, temp_tpr_file, os.path.join(output_dir, "mdout.mdp")]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("\n" + "="*50)
    print(f"Ion addition complete. Final file: '{output_gro}'")
    print("="*50)
