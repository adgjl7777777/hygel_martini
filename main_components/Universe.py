# 이 파일은 World.py의 리팩토링된 버전입니다.
# 템플릿 플레이스홀더 대신 `initialize_world` 함수를 통해 파라미터를 설정합니다.

import numpy as np
import collections
import sys

def initialize_world(segment_length_from_config, mean_sep_from_config):
    '''
      설정 파일로부터 받은 파라미터로 World 클래스 변수를 초기화합니다.
    '''
    World.segment_length = segment_length_from_config
    World.mean_sep = mean_sep_from_config

    # 단일 고분자 생성 시나리오 (segment_length가 0)에서는 ubox_length 계산이 불필요하고 오류를 유발할 수 있으므로 건너뜁니다.
    if segment_length_from_config == 0:
        World.ubox_length = 0
        print("세그먼트 길이가 0이므로, 단위 박스 길이(ubox_length) 계산을 건너뜁니다. (단일 고분자 생성 모드)")
        return

    # ubox_length 계산 로직
    a1 = 3/4
    a2 = -3 * World.mean_sep
    a3 = 9 * World.mean_sep**2 - np.square(World.mean_sep * (2 + World.segment_length))
    roots = np.roots([a1, a2, a3])

    # 여러 해 중에서 유효한(양의 실수) 값을 선택
    valid_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if not valid_roots:
        raise ValueError("방정식에서 유효한(양의 실수) 박스 길이를 찾을 수 없습니다.")
    
    World.ubox_length = valid_roots[0]
    print(f"세그먼트 길이 (Segment Length): {World.segment_length}")
    print(f"계산된 단위 박스 길이 (Unit Box Length): {World.ubox_length}")

# World 클래스는 시뮬레이션 시스템 전체의 상태와 데이터를 담는 전역 컨테이너 역할을 합니다.
class World:
    # --- 클래스 변수: 시뮬레이션의 전역 파라미터 및 데이터 저장소 ---

    # 원자 간 평균 거리 (nm)
    mean_sep = 0.24

    # 이 값들은 initialize_world 함수에 의해 설정됩니다.
    ubox_length = 0.0
    segment_length = 0

    # 전체 시뮬레이션 박스의 길이 (단위체 반복에 의해 결정됨)
    box_length = 0.0
    
    # 생성된 하이드로젤/고분자 개수
    number_of_hydrogels = 0
    number_of_polymers = 0

    # 생성된 하이드로젤/고분자 객체를 담는 리스트
    hydrogels = []
    polymers = []

    # 하이드로젤의 구성 요소 개수 (업데이트용)
    number_of_hydrogel_atoms = 0
    number_of_hydrogel_bonds = 0
    number_of_hydrogel_angles = 0
    number_of_hydrogel_dihedrals = 0

    # 고분자의 구성 요소 개수 (업데이트용)
    number_of_polymer_atoms = 0
    number_of_polymer_bonds = 0
    number_of_polymer_angles = 0
    number_of_polymer_dihedrals = 0

    # 전체 원자 수
    number_of_atoms = 0

    # --- 시스템의 모든 구성 요소를 저장하는 딕셔너리 ---
    Atoms = collections.defaultdict(list)
    Bonds = collections.defaultdict(list)
    Network_bonds = collections.defaultdict(list)
    Constraints = collections.defaultdict(list)
    Exclusions = collections.defaultdict(list)
    Angles = collections.defaultdict(list)
    Dihedrals = collections.defaultdict(list)

    @classmethod
    def reset(cls):
        """모든 클래스 변수를 초기 기본값으로 재설정합니다."""
        cls.mean_sep = 0.24
        cls.ubox_length = 0.0
        cls.segment_length = 0
        cls.box_length = 0.0
        cls.number_of_hydrogels = 0
        cls.number_of_polymers = 0
        cls.hydrogels = []
        cls.polymers = []
        cls.number_of_hydrogel_atoms = 0
        cls.number_of_hydrogel_bonds = 0
        cls.number_of_hydrogel_angles = 0
        cls.number_of_hydrogel_dihedrals = 0
        cls.number_of_polymer_atoms = 0
        cls.number_of_polymer_bonds = 0
        cls.number_of_polymer_angles = 0
        cls.number_of_polymer_dihedrals = 0
        cls.number_of_atoms = 0
        cls.Atoms = collections.defaultdict(list)
        cls.Bonds = collections.defaultdict(list)
        cls.Network_bonds = collections.defaultdict(list)
        cls.Constraints = collections.defaultdict(list)
        cls.Exclusions = collections.defaultdict(list)
        cls.Angles = collections.defaultdict(list)
        cls.Dihedrals = collections.defaultdict(list)
        print("World state has been reset.")

    def __init__(self):
        print('World Created!')

    def make_hydrogel(self, fix_dna, nx=6, ny=6, nz=6):
        '''
          하이드로젤 객체를 생성하고 World에 추가합니다.
        '''
        from main_components.Hydrogel import Hydrogel

        if fix_dna:
            World.number_of_hydrogels = 0
            World.hydrogels = []
            World.number_of_hydrogel_atoms = 0
            World.number_of_hydrogel_bonds = 0
            World.number_of_hydrogel_angles = 0
            World.number_of_hydrogel_dihedrals = 0
            World.number_of_atoms = 0
            World.Atoms.clear()
            World.Bonds.clear()
            World.Network_bonds.clear()
            World.Constraints.clear()
            World.Exclusions.clear()
            World.Angles.clear()
            World.Dihedrals.clear()
        
        World.number_of_hydrogels += 1
        World.hydrogels.append(Hydrogel(nx, ny, nz))

    def make_polymer(self, p_mon_num, p_length):
        '''
          고분자 객체를 생성하고 World에 추가합니다. (이 프로젝트에서는 사용되지 않을 수 있음)
        '''
        from main_components.Polymer import Polymer
        World.number_of_polymers += 1
        World.polymers.append(Polymer(p_mon_num, p_length))

    def update_hydrogel_attributes(self, hydrogel):
        '''
          Hydrogel 객체에 저장된 최종 원자/결합/각도 개수를 World에 업데이트합니다.
        '''
        self.number_of_hydrogel_atoms = hydrogel.num_HDG_atoms
        self.number_of_hydrogel_bonds = hydrogel.num_HDG_bonds
        self.number_of_hydrogel_angles = hydrogel.num_HDG_angles
        self.number_of_hydrogel_dihedrals = hydrogel.num_HDG_dihedrals
        print('Hydrogel information updated')

    def update_polymer_attributes(self, polymer):
        '''
          Polymer 객체의 정보를 World에 업데이트합니다.
        '''
        self.number_of_polymer_atoms = polymer.num_PLM_atoms
        self.number_of_polymer_bonds = polymer.num_PLM_bonds
        self.number_of_polymer_angles = polymer.num_PLM_angles
        self.number_of_polymer_dihedrals = polymer.num_PLM_dihedrals
        print('Polymer information updated')
