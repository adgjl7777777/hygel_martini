import os
import shutil
import subprocess
import sys

# 프로젝트 모듈 임포트
from config_params.read_json import Config
from main_components.Universe import World, initialize_world
from main_components import Hydrogel
from core_utils import writer
from main_components import Attributes

def main():
    World.reset() # 클래스 상태 초기화
    '''
      하이드로젤 구조를 생성하는 메인 실행 함수.
      기존 01_run_initializer.sh와 original.py의 역할을 수행합니다.
    '''
    print("="*50)
    print("하이드로젤 구조 생성을 시작합니다.")
    print("="*50)

    # 1. 출력 디렉토리 설정 및 생성
    output_dir = Config.get_param('simulation_parameters', 'output_dir')

    # 2. World 초기화
    print("\n--- World 초기화 중... ---")
    segment_length = Config.get_param('simulation_parameters', 'segment_length')
    mean_sep = Config.get_param('simulation_parameters', 'mean_sep')
    initialize_world(segment_length, mean_sep)
    
    # 3. 하이드로젤 생성
    print("\n--- 하이드로젤 구성 중... ---")
    
    # 다이아몬드 네트워크는 짝수 개의 셀을 가져야 함
    num_cells = Config.get_param('simulation_parameters', 'number_of_cells')
    assert num_cells % 2 == 0, "Diamond network must have even number of cells in one side"

    # World 및 Hydrogel 객체 생성
    world = World()
    world.make_hydrogel(
        False, 
        nx=num_cells, 
        ny=num_cells, 
        nz=num_cells
    )
    hd = world.hydrogels[0]

    # 구조 생성 단계
    hd.construct_atoms()
    pbc = Config.get_param('simulation_parameters', 'pbc_true_or_false')
    hd.construct_bonds(pbc, num_cells, output_dir)

    num_slices = Config.get_param('simulation_parameters', 'number_of_slices')
    if num_slices > 0:
        print("네트워크 절단을 수행합니다...")
        random_seed = Config.get_param('simulation_parameters', 'random_seed')
        hd.cutter(num_slices, random_seed)

    print("화학적 상세 구조를 구성합니다...")
    hd.construct_chemical_detail()
    
    print("각도(angle)를 구성합니다...")
    hd.construct_angles()

    if num_slices > 0:
        print("절단을 적용합니다...")
        hd.cut()

    world.update_hydrogel_attributes(hd)
    print("하이드로젤 구성 완료.")

    return world


if __name__ == "__main__":
    # Attributes 초기화 (필요 시)
    Attributes.initialize()
    main()