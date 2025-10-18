import os
import sys
import json # Added for temporary config handling

from config_params.read_json import Config
from core_utils import polymer_generator

def generate_polymer_only_from_config(sim_params, poly_gen_params, monomer_definitions):
    print(f"\n--- 단일 고분자 .gro 및 .itp 파일 생성 시작 (전달된 파라미터 기반) ---\n")

    random_seed = sim_params['random_seed']
    output_dir = sim_params['output_dir']
    num_polymers_to_generate = poly_gen_params['num_polymers']
    length = poly_gen_params['length']
    output_gro_filename = poly_gen_params['polymer_output_gro_filename']
    output_itp_filename = poly_gen_params['polymer_output_itp_filename'] # New: Get ITP filename
    moleculetype_name = poly_gen_params['molecule_name'] # New: Get moleculetype_name

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    generated_gro_paths = []
    generated_itp_paths = []

    for i in range(num_polymers_to_generate):
        print(f"고분자 {i+1}/{num_polymers_to_generate} 생성 중...")
        
        # 파일명에 고분자 인덱스 추가
        if num_polymers_to_generate > 1:
            base_gro, ext_gro = os.path.splitext(output_gro_filename)
            indexed_output_gro_path = os.path.join(output_dir, f"{base_gro}_{i+1}{ext_gro}")
            
            base_itp, ext_itp = os.path.splitext(output_itp_filename)
            indexed_output_itp_path = os.path.join(output_dir, f"{base_itp}_{i+1}{ext_itp}")
        else:
            indexed_output_gro_path = os.path.join(output_dir, output_gro_filename)
            indexed_output_itp_path = os.path.join(output_dir, output_itp_filename)

        # polymer_generator.generate_single_polymer_gro는 output_filename에서 itp_filename을 유추하므로
        # output_gro_filename만 전달하면 됩니다.
        polymer_generator.generate_single_polymer_gro(
            p_mon_num=length,
            output_filename=indexed_output_gro_path,
            mean_sep=sim_params['mean_sep'], # Pass mean_sep
            random_seed=random_seed + i,
            include_chemical_detail=True,
            include_angles=True,
            moleculetype_name=moleculetype_name # Pass moleculetype_name
        )
        generated_gro_paths.append(indexed_output_gro_path)
        generated_itp_paths.append(indexed_output_itp_path) # Add generated ITP path

    print("\n" + "="*50)
    print("고분자 생성 작업이 완료되었습니다.")
    print("="*50)
    return generated_gro_paths, generated_itp_paths # Return both GRO and ITP paths

if __name__ == "__main__":
    # Example usage (for testing) - this part needs to be updated if run directly
    print("This script is intended to be imported as a module and called with parameters.")
    print("Direct execution is for testing purposes and requires manual parameter setup.")