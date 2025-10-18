import numpy as np
from random import Random
from main_components import Attributes
from itertools import product as pd
from core_utils.utility import interp3D, dij_sq, normal_tetrahedral_vector, not_self, is_overlap, random_normal_vector
from config_params import read_json as p

class Polymer():
    '''
    Polymer 클래스는 단일 고분자 사슬의 구조를 생성하고 관리하는 역할을 합니다.
    이 클래스는 주로 선형 고분자 사슬을 구성하는 원자(atoms), 결합(bonds), 각도(angles) 등의
    토폴로지 정보를 설정하고, 시뮬레이션 환경 내에서 고분자의 초기 공간적 배치를 결정합니다.
    생성된 고분자 정보는 World 객체에 통합되어 전체 시스템의 일부가 됩니다.
    '''
    # 클래스 변수: 생성된 고분자 사슬의 총 원자, 결합, 각도, 이면각 수를 추적합니다.
    # 이 값들은 시뮬레이션 전체의 통계 및 검증에 사용될 수 있습니다.
    num_PLM_atoms = 0
    num_PLM_bonds = 0
    num_PLM_angles = 0
    num_PLM_dihedrals = 0

    def __init__(self, p_mon_num, p_length):
        '''
        Polymer 객체를 초기화하고 고분자 사슬의 기본 매개변수를 설정합니다.
        또한, 시뮬레이션 박스의 크기를 고분자 길이에 맞춰 조정합니다.

        Args:
            p_mon_num (int): 고분자 사슬을 구성하는 단량체(monomer)의 총 개수입니다.
                             이 값은 고분자 사슬의 길이를 결정하는 주요 인자입니다.
            p_length (float): 고분자 사슬의 전체 길이(예: 나노미터 단위)입니다.
                              이 길이는 고분자 백본의 물리적 확장 범위를 나타냅니다.
        '''
        from main_components.Universe import World

        self.p_length = p_length # 고분자 사슬의 정의된 전체 길이
        self.p_mon_num = p_mon_num # 고분자 사슬을 구성하는 단량체의 수

        # 시뮬레이션 박스 길이 설정:
        # 고분자 사슬의 길이를 기반으로 전체 시뮬레이션 박스의 한 변 길이를 설정합니다.
        # 일반적으로 고분자 길이가 박스 크기에 영향을 미치므로, 여기서는 고분자 길이의 2배로 설정하여
        # 고분자가 박스 내에 충분히 포함될 수 있도록 합니다.
        # 주석 처리된 'if not World.box_length:' 부분은 World.box_length가 한 번만 설정되도록
        # 의도되었을 수 있으나, 현재는 매 초기화마다 덮어쓰고 있습니다.
        World.box_length = self.p_length * 2 # 고분자 길이의 두 배로 시뮬레이션 박스 길이 설정

    def make_lines(self, random_seed):
        '''
        시스템 내에 고분자 단량체 원자들이 배치될 가상 선(경로)들을 생성합니다.
        이 메서드는 고분자 사슬의 시작점과 끝점을 무작위로 정의하고,
        그 사이를 선형 보간하여 고분자 단량체들이 위치할 3D 좌표들을 생성합니다.
        이는 고분자 사슬의 초기 형태를 결정하는 중요한 단계입니다.

        Args:
            random_seed (int): 무작위 값 생성을 위한 시드(seed)입니다.
                               동일한 시드를 사용하면 재현 가능한 고분자 구조를 생성할 수 있습니다.

        Returns:
            np.array: 고분자 단량체들이 위치할 3D 좌표들의 배열입니다.
                      각 행은 [x, y, z] 형태의 단량체 위치를 나타냅니다.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.

        # 1. 고분자 사슬의 중간 지점을 무작위로 결정합니다.
        # 시뮬레이션 박스 내에서 고분자 길이가 p_length인 고분자가 배치될 수 있도록
        # 0.5 * p_length ~ 1.5 * p_length 범위 내에서 중간 지점을 설정합니다.
        pm_middle_point = [0.5 * self.p_length + Random(random_seed).random() * self.p_length, \
                            0.5 * self.p_length + Random(random_seed-1).random() * self.p_length, \
                            0.5 * self.p_length + Random(random_seed-2).random() * self.p_length]

        # 2. 고분자 사슬의 방향 벡터를 무작위로 생성합니다。
        # x, y, z 방향 성분의 제곱합이 1이 되도록 정규화된 무작위 벡터를 생성합니다.
        # 각 성분은 -1 또는 1의 부호를 가질 수 있어 다양한 방향성을 부여합니다.
        x_direct = Random(random_seed-3).random()
        y_direct = (1-x_direct)*Random(random_seed-4).random()
        z_direct = 1-x_direct-y_direct
        direct = [Random(random_seed-5).choice([-1, 1])*x_direct**0.5, \
                   Random(random_seed-6).choice([-1, 1])*y_direct**0.5, \
                   Random(random_seed-7).choice([-1, 1])*z_direct**0.5]

        # 3. 중간 지점과 방향 벡터를 이용하여 고분자 사슬의 시작점과 끝점을 계산합니다.
        # 고분자 길이가 p_length이므로, 중간 지점에서 방향 벡터의 절반 길이만큼 이동하여 시작점과 끝점을 정의합니다.
        pm_start_point = np.array(pm_middle_point) - np.array(direct) * self.p_length * 0.5
        pm_last_point = np.array(pm_middle_point) + np.array(direct) * self.p_length * 0.5

        # 4. interp3D 함수를 사용하여 시작점과 끝점 사이에 단량체 수만큼의 점들을 보간합니다.
        # 이 보간된 점들이 각 단량체 원자의 초기 위치가 됩니다.
        return interp3D(self.p_mon_num, pm_start_point, pm_last_point)

    def construct_atoms(self, random_seed):
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.

        # World에 고분자가 1개만 있는 경우 (즉, 현재 생성 중인 고분자가 첫 번째 고분자인 경우)
        if World.number_of_polymers == 1: 
             # make_lines 메서드를 호출하여 고분자 단량체들의 3D 좌표 리스트를 생성합니다.
             pm_crd_list = self.make_lines(random_seed) 
             for i, ii in enumerate(pm_crd_list):
                 _tmp = Attributes.Atom() # 새로운 원자 객체를 생성합니다.
                 _tmp.atom_type = p.Config.get_param('polymer_components', 'backbone', 'atom_type') # 원자 타입 설정 (예: Martini C1 타입, coarse-grained 모델에서 사용).
                 _tmp.residue_number = p.Config.get_param('polymer_components', 'backbone', 'residue_number') # 잔기(residue) 번호 설정. 모든 단량체를 동일한 잔기로 간주합니다.
                 _tmp.residue_name = p.Config.get_param('polymer_components', 'backbone', 'residue_name') # 잔기 이름 설정: Hydrogel.
                 _tmp.atom_name = p.Config.get_param('polymer_components', 'backbone', 'atom_name') # 원자 이름 설정: Segment (고분자 사슬의 한 단위).
                 _tmp.cgnr = p.Config.get_param('polymer_components', 'backbone', 'cgnr') # 전하 그룹 번호 설정.
                 _tmp.mass = p.Config.get_param('polymer_components', 'backbone', 'mass') # 원자의 질량 설정.
                 _tmp.charge = p.Config.get_param('polymer_components', 'backbone', 'charge') # 원자의 전하 설정.
                 _tmp.position = ii # make_lines에서 얻은 3D 좌표를 원자의 위치로 설정합니다.

                 # 고분자 사슬의 터미널 원자(시작과 끝)와 중간 원자를 구분하여 처리합니다.
                 # 특별한 두 개의 터미널 원자는 결합 부분에서 다시 처리될 수 있습니다.
                 if i == self.p_mon_num -1:  # 고분자 사슬의 마지막 원자인 경우
                     _tmp.end_tag = 1 # end_tag를 1로 설정하여 터미널 원자임을 표시합니다.
                     # 이전 원자와의 결합을 생성합니다. (백본 결합)
                     if _tmp.atom_id > 0:
                         _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                         _tmp2.bond_funct = p.Config.get_param('polymer_components', 'backbone', 'bond_funct') # 결합 함수 타입 (예: 조화 포텐셜).
                         _tmp2.bond_c0 = p.Config.get_param('polymer_components', 'backbone', 'bond_c0') # 평형 결합 거리 (nm).
                         _tmp2.bond_c1 = p.Config.get_param('polymer_components', 'backbone', 'bond_c1') # 힘 상수 (kJ/mol/nm^2).
                     # self.terminals[_tmp.end_tag].append(_tmp) # 터미널 원자를 저장하는 로직 (현재 주석 처리됨).
                 elif i == 0:  # 고분자 사슬의 첫 번째 원자인 경우
                     _tmp.end_tag = 1 # end_tag를 1로 설정하여 터미널 원자임을 표시합니다.
                     # self.terminals[_tmp.end_tag].append(_tmp) # 터미널 원자를 저장하는 로직 (현재 주석 처리됨).
                 else: # 중간 원자
                    if _tmp.atom_id > 0:
                        _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                        _tmp2.bond_funct = p.Config.get_param('polymer_components', 'backbone', 'bond_funct')
                        _tmp2.bond_c0 = p.Config.get_param('polymer_components', 'backbone', 'bond_c0')
                        _tmp2.bond_c1 = p.Config.get_param('polymer_components', 'backbone', 'bond_c1')

        # World에 고분자가 1개보다 많은 경우 (즉, 여러 고분자가 시스템에 존재할 수 있는 경우)
        # 이 경우, 새로 추가되는 고분자가 기존 고분자들과 겹치는지 확인하는 겹침 테스트가 필요합니다.
        elif World.number_of_polymers > 1: 
            print("겹침 테스트가 필요합니다. 아직 구현되지 않았습니다. (현재는 첫 번째 고분자만 처리)")

    def construct_chemical_detail(self):
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.
        overlap_check_limit = p.Config.get_param('simulation_parameters', 'overlap_check_limit')

        # 루프 도중 World.Atoms 컬렉션이 변경되는 것을 방지하기 위해 복사본을 사용합니다.
        _World_Atoms = [*World.Atoms]
        print(len(_World_Atoms), " World에 현재 존재하는 원자의 총 개수")

        for _id in _World_Atoms:
            atom = World.Atoms[_id][0]
            # 현재 원자에 결합된 첫 번째 원자를 찾습니다.
            b1 = not_self(atom, atom.bonded_atoms[0])
            # 현재 원자에 결합된 두 번째 원자가 있다면 찾습니다.
            if len(atom.bonded_atoms) > 1:
                b2 = not_self(atom, atom.bonded_atoms[1])
            
            side_atom = Attributes.Atom() # 새로운 측쇄 원자 객체를 생성합니다。

            # 측쇄 원자의 속성을 설정합니다।
            side_atom.atom_type = p.Config.get_param('polymer_components', 'side_chain', 'atom_type') # 측쇄 원자의 타입 (예: Martini Nda 타입).
            side_atom.residue_number = p.Config.get_param('polymer_components', 'side_chain', 'residue_number') # 잔기 번호.
            side_atom.residue_name = p.Config.get_param('polymer_components', 'side_chain', 'residue_name') # 잔기 이름: Hydrogel.
            side_atom.atom_name = p.Config.get_param('polymer_components', 'side_chain', 'atom_name')  # 원자 이름: Hydrogel Side Chain (하이드로젤 측쇄).
            side_atom.cgnr = p.Config.get_param('polymer_components', 'side_chain', 'cgnr') # 전하 그룹 번호.
            side_atom.mass = p.Config.get_param('polymer_components', 'side_chain', 'mass') # 질량.
            side_atom.charge = p.Config.get_param('polymer_components', 'side_chain', 'charge') # 전하.
            
            # 백본 원자와 측쇄 원자 사이의 결합을 생성합니다.
            bond = Attributes.Bond(atom.atom_id, side_atom.atom_id)
            bond.bond_funct = p.Config.get_param('polymer_components', 'side_chain', 'bond_funct') # 결합 함수 타입.
            bond.bond_c0 = p.Config.get_param('polymer_components', 'side_chain', 'bond_c0') # 평형 결합 거리 (nm).
            bond.bond_c1 = p.Config.get_param('polymer_components', 'side_chain', 'bond_c1') # 힘 상수 (kJ/mol/nm^2).

            if atom.number_of_bonds == 4: # 최종 결합 수 4: 가교 지점 원자 (Original bonds: 3)
                # 사면체 구조를 만들기 위해, 결합된 3개의 이웃 원자 위치를 기반으로 4번째 위치를 계산합니다.
                batom_positions = []
                for bond_ in atom.bonded_atoms[:3]:  # 곁사슬을 제외한 3개의 이웃
                    batom_positions.append(
                        not_self(atom, bond_).
                            position)
                rij = normal_tetrahedral_vector(atom.position,
                                                batom_positions[0],
                                                batom_positions[1],
                                                batom_positions[2],
                                                World.box_length)

                side_atom.position = atom.position + rij * World.mean_sep
                print('4가 진행중, innoculated site')

                # 겹침 테스트를 위해 주변 2단계까지의 원자 리스트를 준비합니다.
                depth_2_atoms = [atom.atom_id, b1.atom_id, b2.atom_id] 
                for _ in range(2):
                    for i in depth_2_atoms:
                        atom_ = World.Atoms[i][0]
                        for bond_ in atom_.bonded_atoms:
                            depth_2_atoms.append(bond_.bond_atom_1.atom_id)
                            depth_2_atoms.append(bond_.bond_atom_2.atom_id)
                        depth_2_atoms = list(set(depth_2_atoms))
                # 자신과 새로 결합된 곁사슬 원자는 겹침 테스트에서 제외합니다.
                depth_2_atoms.remove(bond.bond_atom_1.atom_id)
                depth_2_atoms.remove(bond.bond_atom_2.atom_id)

                position_testers = np.zeros((len(depth_2_atoms), 3))
                for i, atom_id in enumerate(depth_2_atoms):
                    position_testers[i, :] = World.Atoms[atom_id][0].position
                
                # 겹치지 않는 위치를 찾을 때까지 위치 재선정
                test_result = True
                counter = 0
                while test_result is True:
                    # b1, b2를 기준으로 평면을 정의하고 그 법선벡터 방향으로 곁사슬 위치를 정합니다.
                    side_atom.position = atom.position + random_normal_vector(b1.position,
                                                              atom.position,
                                                              b2.position,
                                                              World.mean_sep,
                                                              World.box_length
                                                              )
                    # 생성된 위치가 주변 원자들과 겹치는지 확인합니다.
                    test_result = is_overlap(side_atom.position,
                                             position_testers,
                                             World.mean_sep,
                                             World.box_length)
                    counter += 1
                    if counter > overlap_check_limit: # 무한 루프 방지
                        print(f"경고: 원자 {atom.atom_id}의 4가 측쇄 위치를 최적화하지 못했습니다. 약간의 겹침이 있을 수 있습니다.")
                        break

            elif atom.number_of_bonds == 3: # 최종 결합 수 3: 중간 백본 원자 (Original bonds: 2)
                # 겹침을 피하기 위해, 현재 원자로부터 2단계까지 떨어진 이웃 원자들의 리스트를 만듭니다.
                depth_2_atoms = [atom.atom_id, b1.atom_id, b2.atom_id] 
                for _ in range(2):
                    for i in depth_2_atoms:
                        atom_ = World.Atoms[i][0]
                        for bond_ in atom_.bonded_atoms:
                            depth_2_atoms.append(bond_.bond_atom_1.atom_id)
                            depth_2_atoms.append(bond_.bond_atom_2.atom_id)
                        depth_2_atoms = list(set(depth_2_atoms))
                # 테스트 목록에서 자기 자신과, 방금 연결된 곁사슬 원자는 제외합니다.
                depth_2_atoms.remove(bond.bond_atom_1.atom_id)
                depth_2_atoms.remove(bond.bond_atom_2.atom_id)

                # 겹침 테스트 대상 원자들의 3D 좌표를 준비합니다.
                position_testers = np.zeros((len(depth_2_atoms), 3))
                for i, atom_id in enumerate(depth_2_atoms):
                    position_testers[i, :] = World.Atoms[atom_id][0].position
                
                # 겹치지 않는 위치를 찾을 때까지 반복합니다.
                test_result = True
                counter = 0
                while test_result is True:
                    # b1, atom, b2로 정의된 평면의 법선 벡터 방향으로 곁사슬을 배치합니다.
                    side_atom.position = atom.position + random_normal_vector(b1.position,
                                                              atom.position,
                                                              b2.position,
                                                              World.mean_sep,
                                                              World.box_length
                                                              )
                    # 생성된 위치가 주변 원자들과 겹치는지 확인합니다.
                    test_result = is_overlap(side_atom.position,
                                             position_testers,
                                             World.mean_sep,
                                             World.box_length)
                    counter += 1
                    if counter > overlap_check_limit: # 무한 루프 방지
                        print(f"경고: 원자 {atom.atom_id}의 3가(중간) 측쇄 위치를 최적화하지 못했습니다.")
                        break

            elif atom.number_of_bonds == 2: # 최종 결합 수 2: 말단 원자 (Original bonds: 1)
                # 겹침을 피하기 위해, 현재 원자로부터 2단계까지 떨어진 이웃 원자들의 리스트를 만듭니다.
                depth_1_atoms = [atom.atom_id, b1.atom_id]
                for _ in range(2):
                    for i in depth_1_atoms:
                        atom_ = World.Atoms[i][0]
                        for bond_ in atom_.bonded_atoms:
                            depth_1_atoms.append(bond_.bond_atom_1.atom_id)
                            depth_1_atoms.append(bond_.bond_atom_2.atom_id)
                        depth_1_atoms = list(set(depth_1_atoms))
                # 테스트 목록에서 자기 자신과, 방금 연결된 곁사슬 원자는 제외합니다.
                depth_1_atoms.remove(bond.bond_atom_1.atom_id)
                depth_1_atoms.remove(bond.bond_atom_2.atom_id)

                # 겹침 테스트 대상 원자들의 3D 좌표를 준비합니다.
                position_testers = np.zeros((len(depth_1_atoms), 3))
                for i, atom_id in enumerate(depth_1_atoms):
                    position_testers[i, :] = World.Atoms[atom_id][0].position
                
                # 겹치지 않는 위치를 찾을 때까지 반복합니다.
                test_result = True
                counter = 0
                while test_result is True:
                    # b1, atom, 그리고 가상의 점을 이용해 정의된 평면의 법선 벡터 방향으로 곁사슬을 배치합니다.
                    side_atom.position = atom.position + random_normal_vector(b1.position,
                                                              atom.position,
                                                              atom.position + (atom.position - b1.position),
                                                              World.mean_sep,
                                                              World.box_length
                                                              )
                    # 생성된 위치가 주변 원자들과 겹치는지 확인합니다.
                    test_result = is_overlap(side_atom.position,
                                         position_testers,
                                         World.mean_sep,
                                         World.box_length)
                    counter += 1
                    if counter > overlap_check_limit: # 무한 루프 방지
                        print(f"경고: 원자 {atom.atom_id}의 2가(말단) 측쇄 위치를 최적화하지 못했습니다.")
                        break

            elif atom.number_of_bonds == 1: # 최종 결합 수 1: 예외 상황 (Original bonds: 0)
                # 겹침 테스트를 위한 주변 원자 리스트를 준비합니다.
                depth_1_atoms = [atom.atom_id, b1.atom_id]
                depth_1_atoms = list(set(depth_1_atoms))
                depth_1_atoms.remove(atom.atom_id)

                position_testers = np.zeros((len(depth_1_atoms), 3))
                for i, atom_id in enumerate(depth_1_atoms):
                    position_testers[i, :] = World.Atoms[atom_id][0].position

                # 겹치지 않는 위치를 찾을 때까지 반복합니다.
                test_result = True
                counter = 0
                while test_result is True:
                    # b1, atom, 그리고 가상의 점을 이용해 정의된 평면의 법선 벡터 방향으로 곁사슬을 배치합니다.
                    side_atom.position = atom.position + random_normal_vector(b1.position,
                                                              atom.position,
                                                              atom.position + (atom.position - b1.position),
                                                              World.mean_sep,
                                                              World.box_length
                                                              )
                    # 생성된 위치가 주변 원자들과 겹치는지 확인합니다.
                    test_result = is_overlap(side_atom.position,
                                             position_testers,
                                             World.mean_sep,
                                             World.box_length)
                    counter += 1
                    if counter > overlap_check_limit: # 무한 루프 방지
                        print(f"경고: 원자 {atom.atom_id}의 1가 측쇄 위치를 최적화하지 못했습니다.")
                        break

            else:
                # 예상치 못한 결합 수를 가진 원자가 발견될 경우, 명확한 오류 메시지를 출력합니다.
                print(f"원자 {atom.atom_id}의 결합 상태가 올바르지 않습니다. 최종 결합 수: {atom.number_of_bonds}")


        # 현재 World에 존재하는 총 원자 및 결합 수를 업데이트합니다.
        self.num_PLM_atoms = len(World.Atoms)
        # print('World.Atoms', len(World.Atoms), self.num_HDG_atoms) # 디버깅용 주석 처리된 라인
        self.num_PLM_bonds = len(World.Bonds)

    def construct_angles(self):
        '''
        고분자 내의 각도(angles) 상호작용을 구성합니다.
        이 메서드는 이미 정의된 원자들과 결합 정보를 바탕으로,
        세 개의 원자로 이루어진 각도 상호작용을 식별하고 `Attributes.Angle` 객체를 생성합니다.
        JSON 설정 파일에 정의된 `specific_angles` 규칙에 따라 다른 힘 상수를 적용하여
        분자 역학 시뮬레이션에서 정확한 각도 거동을 모델링할 수 있도록 합니다.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.
        
        _World_Bonds = World.Bonds
        _atom = World.Atoms
        
        angle_configs = p.Config.get_param('polymer_components', 'angles')
        default_params = angle_configs['default_angle']
        specific_params_list = angle_configs.get('specific_angles', [])

        atom1_bond = [key[0] for key in _World_Bonds]
        atom2_bond = [key[1] for key in _World_Bonds]
        
        b11 = np.array(atom1_bond)
        b22 = np.array(atom2_bond)
        
        # Find central atoms for angles
        pos = list(set(b11) & set(b22))
        pos.sort()
        
        for cen_atom_id in pos:
            near_cen_atom_ids = []
            for j in np.where(b11 == cen_atom_id)[0]:
                near_cen_atom_ids.append(b22[j])
            for k in np.where(b22 == cen_atom_id)[0]:
                near_cen_atom_ids.append(b11[k])
            near_cen_atom_ids.sort()
            
            for i in range(len(near_cen_atom_ids)):
                for j in range(i + 1, len(near_cen_atom_ids)):
                    side_atom1_id = near_cen_atom_ids[i]
                    side_atom2_id = near_cen_atom_ids[j]
                    
                    angle = Attributes.Angle(side_atom1_id, cen_atom_id, side_atom2_id)
                    
                    atom_types_in_angle = {
                        _atom[side_atom1_id][0].atom_type, 
                        _atom[cen_atom_id][0].atom_type, 
                        _atom[side_atom2_id][0].atom_type
                    }
                    
                    applied_specific = False
                    for specific_rule in specific_params_list:
                        # Check for intersection between atom types in angle and rule
                        if not atom_types_in_angle.isdisjoint(specific_rule['atom_types']):
                            params = specific_rule['parameters']
                            angle.angle_funct = params['angle_funct']
                            angle.angle_c0 = params['angle_c0']
                            angle.angle_c1 = params['angle_c1']
                            applied_specific = True
                            break # First matching rule wins
                    
                    if not applied_specific:
                        angle.angle_funct = default_params['angle_funct']
                        angle.angle_c0 = default_params['angle_c0']
                        angle.angle_c1 = default_params['angle_c1']

        self.num_PLM_angles = len(World.Angles)