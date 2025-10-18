import json
import glob
import os
import sys
import shutil
import random
from core_utils import packer, topology_updater
from core_utils.writer import write_to_gro, write_to_itp
from add_series import add_small_ion

from config_params.config import Config

def replace_in_file(file_path, old_str, new_str,margin="left"):
    # 파일 읽어들이기
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()
    changestr = new_str
    if margin == "left":
        if len(new_str)<3:
            changestr = " " * (3-len(new_str)) + new_str
    else:
        if len(new_str)<3:
            changestr = new_str + " " * (3-len(new_str))
    # old_str -> new_str 치환
    fw = open(file_path, 'w')
    for line in lines:
        fw.write(line.replace(old_str, changestr))
    fw.close()

def execute_mode():
    mode = Config.get_param('mode')
    print(f"\n--- 실행 모드: {mode} ---")

    if mode == 'all':
        _execute_all_mode()
    else:
        print(f"알 수 없는 모드 또는 이 스크립트에서 직접 실행 미지원: {mode}")

def _run_packing_step(step_name, base_structure_gro, molecules_to_add, final_output_gro, box_size_nm, sim_params):
    """
    A helper function to run a single step of packing with packmol.
    """
    print(f"\n--- Packmol 실행 단계: {step_name} ---")
    
    # Use the centralized packer function
    packer.pack_system_with_molecules(
        base_structure_gro=base_structure_gro,
        molecules_to_add=molecules_to_add,
        final_output_gro=final_output_gro,
        box_size_nm=box_size_nm,
        sim_params=sim_params
    )
    
    print(f"단계 '{step_name}' 완료. 결과 파일: {final_output_gro}")
    return final_output_gro

def _create_single_ion_gro(ion_name, output_dir):
    """
    Creates a .gro file for a single ion.
    """
    gro_path = os.path.join(output_dir, f"{ion_name}.gro")
    content = f"Generated single ion file for {ion_name}\n    1\n1ION    {ion_name:<5}    1   0.000   0.000   0.000\n   0.00000   0.00000   0.00000\n"
    with open(gro_path, 'w') as f:
        f.write(content)
    print(f"생성된 단일 이온 파일: {gro_path}")
    return gro_path

def _execute_all_mode():
    """
    Executes the full workflow with sequential packing and genion.
    """
    from config_params import build_hydrogel, make_polymer_only
    
    print("\n--- 하이드로젤 구성 및 추가 물질 삽입 시작 ---")
    sim_params = Config.get_param('simulation_parameters')
    output_dir = sim_params['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initial hydrogel generation
    hydrogel_world = build_hydrogel.main()
    current_gro_file = os.path.join(output_dir, "initial_hydrogel.gro")
    initial_itp = os.path.join(output_dir, "initial_hydrogel.itp")
    write_to_gro(hydrogel_world, filename=current_gro_file)
    write_to_itp(hydrogel_world, filename=initial_itp, moleculetype_name="HYDROGEL")
    print(f"성공적으로 초기 하이드로젤 파일 생성: {current_gro_file}, {initial_itp}")

    try:
        itp_files_to_include = glob.glob(os.path.join(Config.get_param('simulation_parameters').get('martini_itp_default_directory'), "**", "*.itp"), recursive=True)
    except KeyError:
        itp_files_to_include = []

    try:
        itp_files_to_include.append(Config.get_param('additional_itp_files'))
    except KeyError:
        itp_files_to_include.append(itp_files_to_include)
        
    itp_files_to_include.append(initial_itp)
    molecule_counts_for_top = {"HYDROGEL": 1}
    box_size_nm = hydrogel_world.box_length
    sim_params['box_size_nm'] = box_size_nm # Save for later steps
    
    
    try:
        add_series_params = Config.get_param('add_series_parameters')
    except:
        add_series_params = []

    # --- Sequential Packing Steps ---

    # 2. Add Polymer
    if 'add_polymer' in add_series_params and add_series_params['add_polymer'].get('num_polymers', 0) > 0:
        poly_params = add_series_params['add_polymer']
        if poly_params.get('generation_mode') == 'generate':
            print("\n--- 고분자 사전 생성 시작 ---")
            generated_gro_paths, generated_itp_paths = make_polymer_only.generate_polymer_only_from_config(sim_params, poly_params, Config.get_param('monomer_definitions'))
            poly_params['polymer_source_file'] = generated_gro_paths[0]
            itp_files_to_include.extend(generated_itp_paths)

        poly_source_gro = poly_params['polymer_source_file']
        poly_dest_gro = os.path.join(output_dir, os.path.basename(poly_source_gro))
        if os.path.abspath(poly_source_gro) != os.path.abspath(poly_dest_gro):
            shutil.copy(poly_source_gro, poly_dest_gro)
        
        molecules_to_add = [{"file": poly_dest_gro, "number": poly_params['num_polymers']}]
        packed_after_poly_gro = os.path.join(output_dir, "packed_after_polymer.gro")
        
        current_gro_file = _run_packing_step(
            "Add_Polymer", current_gro_file, molecules_to_add, packed_after_poly_gro, box_size_nm, sim_params
        )
        molecule_counts_for_top[poly_params['molecule_name']]= molecule_counts_for_top.get(poly_params['molecule_name'], 0) + poly_params['num_polymers']

    # 3. Add Molecule
    if 'add_molecule' in add_series_params and add_series_params['add_molecule'].get('num_molecules', 0) > 0:
        mol_params = add_series_params['add_molecule']
        mol_source = mol_params['molecule_gro']
        
        if mol_source.endswith('.xyz'):
            gro_filename = f"{os.path.splitext(os.path.basename(mol_source))[0]}.gro"
            mol_source_gro = packer.convert_xyz_to_gro(mol_source, os.path.join(output_dir, gro_filename), sim_params.get('gromacs_executable_path') or 'gmx_mpi', molecule_name=mol_params['molecule_name'])
        else:
            mol_source_gro = mol_source

        mol_dest_gro = os.path.join(output_dir, os.path.basename(mol_source_gro))
        if os.path.abspath(mol_source_gro) != os.path.abspath(mol_dest_gro):
            shutil.copy(mol_source_gro, mol_dest_gro)

        # Find and include the ITP file for the molecule
        itp_path_to_add = None
        if 'molecule_itp' in mol_params:
            # Case 1: ITP path is explicitly provided in the config
            itp_path_to_add = mol_params['molecule_itp']
            print(f"설정 파일에서 분자 ITP 경로를 사용합니다: {itp_path_to_add}")
        else:
            # Case 2: Auto-detect ITP path by replacing the extension of the source file
            base_path, _ = os.path.splitext(mol_source)
            potential_itp_path = base_path + ".itp"
            if os.path.exists(potential_itp_path):
                itp_path_to_add = potential_itp_path
                print(f"자동으로 분자 ITP 파일을 감지했습니다: {itp_path_to_add}")

        if itp_path_to_add:
            itp_dest = os.path.join(output_dir, os.path.basename(itp_path_to_add))
            if os.path.abspath(itp_path_to_add) != os.path.abspath(itp_dest):
                shutil.copy(itp_path_to_add, itp_dest)
            itp_files_to_include.append(itp_dest)
            print(f"분자 ITP 파일 추가: {itp_dest}")
        else:
            print(f"경고: 분자 '{mol_params['molecule_name']}'의 ITP 파일을 찾을 수 없습니다. grompp 단계에서 오류가 발생할 수 있습니다.")

        molecules_to_add = [{"file": mol_dest_gro, "number": mol_params['num_molecules']}]
        packed_after_mol_gro = os.path.join(output_dir, "packed_after_molecule.gro")

        current_gro_file = _run_packing_step(
            "Add_Molecule", current_gro_file, molecules_to_add, packed_after_mol_gro, box_size_nm, sim_params
        )
        molecule_counts_for_top[mol_params['molecule_name']] = molecule_counts_for_top.get(mol_params['molecule_name'], 0) + mol_params['num_molecules']
    watername = "W"
    # 4. Add Water
    if 'add_water' in add_series_params:
        water_params = add_series_params['add_water']
        watername = water_params.get('molecule_name',"W")
        # Use calculate_water_molecules if available, otherwise use a default
        try:
            from add_series.add_water import calculate_water_molecules
            n_water = calculate_water_molecules(water_params.get('mode', 'full'))
        except (ImportError, KeyError):
            n_water = water_params.get('number_of_water', 10000) # Fallback
            print(f"Could not calculate water molecules, using fallback value: {n_water}")

        water_source_gro = os.path.join(os.path.dirname(__file__), '..', 'add_series', 'water.gro')
        water_dest_gro = os.path.join(output_dir, f'{watername}.gro') # <--- This is the problem for GRO
        if os.path.abspath(water_source_gro) != os.path.abspath(water_dest_gro):
            shutil.copy(water_source_gro, water_dest_gro)
        replace_in_file(water_dest_gro,"***",watername,margin="right")
        molecules_to_add = [{"file": water_dest_gro, "number": n_water}]
        packed_after_water_gro = os.path.join(output_dir, "packed_after_water.gro")

        current_gro_file = _run_packing_step(
            "Add_Water", current_gro_file, molecules_to_add, packed_after_water_gro, box_size_nm, sim_params
        )
        molecule_counts_for_top[water_params['molecule_name']] = n_water

        # Reconstruct and add the water ITP file path to the list of includes for the final topology
        water_source_itp = os.path.join(os.path.dirname(__file__), '..', 'add_series', 'water.itp')
        water_dest_itp = os.path.join(output_dir, f'{watername}.itp') # <--- This is the problem for GRO
        if os.path.abspath(water_source_itp) != os.path.abspath(water_dest_itp):
            shutil.copy(water_source_itp, water_dest_itp)
        replace_in_file(water_dest_itp,"***",watername,margin="right")
        replace_in_file(water_dest_itp,"&&&",watername,margin="left")

        if watername != "W":
            itp_files_to_include.append(water_dest_itp)

    # 5. Create topology for genion AND final topology
    if 'add_small_ion' in add_series_params and add_series_params['add_small_ion'].get('additional_ion_itp_files'):
        itp_ion_add_list = add_series_params['add_small_ion'].get('additional_ion_itp_files')
        if len(itp_ion_add_list) > 0:
            itp_files_to_include.extend(itp_ion_add_list)
    final_itp = []
    for i in range(len(itp_files_to_include)):
        if itp_files_to_include[i] != []:
            final_itp.append(itp_files_to_include[i])
    final_top_path = os.path.join(output_dir, "system.top")
    print(f"\n--- 최종 토폴로지 파일 생성 중: {final_top_path} ---")
    print(f"ITP files to include in topology: {final_itp}")
    
    topology_updater.create_system_topology(output_dir, final_top_path, final_itp)
    
    topology_updater.update_topology_molecules(final_top_path, molecule_counts_for_top)
    print("이온 추가 전 토폴로지 업데이트 완료.")

    # 6. Add Ions using GROMACS genion
    if 'add_small_ion' in add_series_params and add_series_params['add_small_ion'].get('ions'):
        ion_config = add_series_params['add_small_ion']
        # The function expects a flat dictionary, so we prepare one.
        ion_params_for_function = ion_config.copy()

        final_gro_path = os.path.join(output_dir, "final_system")

        # Call the refactored genion function
        add_small_ion.run_genion_for_neutralization(
            input_gro=current_gro_file,
            output_gro=final_gro_path,
            topology_file=final_top_path, # genion will read and update this file
            sim_params=sim_params,
            ion_params=ion_params_for_function,
            solvent_name=watername
        )
        current_gro_file = final_gro_path
    else:
        # If no ions are added, the last .gro file is the final one.
        shutil.copy(current_gro_file, os.path.join(output_dir, "final_system.gro"))

    print("\n--- 모든 작업 완료 ---")