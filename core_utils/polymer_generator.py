import os

from main_components.Universe import World, initialize_world
from main_components import Attributes
from core_utils import writer
from main_components import Polymer

def generate_single_polymer_gro(
    p_mon_num: int,
    output_filename: str,
    mean_sep: float, # New parameter
    random_seed: int = 2024,
    include_chemical_detail: bool = True,
    include_angles: bool = True,
    moleculetype_name: str = 'HDGEL' # New parameter
):
    """
    단일 고분자 사슬의 .gro 파일을 생성합니다.

    Args:
        p_mon_num (int): 고분자 사슬을 구성하는 단량체의 총 개수 (길이).
        output_filename (str): 생성될 .gro 파일의 이름.
        mean_sep (float): 비드 간의 평균 거리.
        random_seed (int): 고분자 구조 생성을 위한 무작위 시드.
        include_chemical_detail (bool): 곁사슬을 포함할지 여부.
        include_angles (bool): 각도 정보를 포함할지 여부.
    """
    print(f"\n--- 단일 고분자(길이: {p_mon_num}) .gro 파일 생성 중... ---")
    
    # World 상태를 리셋하고 고분자 생성에 맞게 재초기화
    World.reset()
    initialize_world(0, mean_sep) # segment_length는 고분자 생성에 필요 없으므로 0으로 설정

    # 이 함수 내에서만 사용할 임시 World 객체 생성
    world = World()
    Attributes.initialize() # 원자, 결합 등 카운터 초기화

    # 고분자 길이 계산 및 생성
    p_length = (p_mon_num - 1) * World.mean_sep
    world.make_polymer(p_mon_num, p_length)
    pm = world.polymers[-1]
    
    # 고분자 구조 구성
    pm.construct_atoms(random_seed)
    if include_chemical_detail:
        pm.construct_chemical_detail()
    if include_angles:
        pm.construct_angles()

    # 단일 고분자 시스템을 .gro 파일로 저장
    class MockDNAsys: # writer.write_to_gro가 요구하는 DNA 객체 흉내
        def __init__(self):
            self.dna_atoms_list = []
    
    writer.write_to_gro(world, filename=output_filename)
    print(f"성공적으로 단일 고분자 구조 파일 '{output_filename}'을 생성했습니다.")

    itp_filename = os.path.splitext(output_filename)[0] + ".itp"
    writer.write_to_itp(world, filename=itp_filename, moleculetype_name=moleculetype_name)
    print(f"성공적으로 단일 고분자 토폴로지 파일 '{itp_filename}'을 생성했습니다.")
