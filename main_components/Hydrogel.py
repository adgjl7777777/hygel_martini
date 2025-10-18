import numpy as np
from main_components import Attributes
from itertools import product as pd
from core_utils.utility import interp3D, dij_sq, rij, normal_tetrahedral_vector, not_self, is_overlap, random_normal_vector
import random
from config_params import read_json as p
import itertools
import os
from tqdm import tqdm


class Hydrogel():
    '''
    하이드로젤 네트워크의 구조를 생성하고 관리하는 클래스입니다.
    다이아몬드 네트워크 구조를 기반으로 원자, 결합, 각도 등을 구성하며,
    곁사슬 추가 및 네트워크 절단과 같은 기능을 제공합니다.
    '''
    # 클래스 변수로, 생성된 하이드로젤의 구성 요소 수를 추적합니다.
    num_HDG_atoms = 0
    num_HDG_bonds = 0
    num_HDG_angles = 0
    num_HDG_dihedrals = 0

    def __init__(
                 self, 
                 x_number_of_repeat=6,
                 y_number_of_repeat=6,
                 z_number_of_repeat=6):
        '''
          하이드로젤 객체를 초기화합니다.

          Args:
              x_number_of_repeat (int): x축 방향으로 단위 셀이 반복되는 횟수.
              y_number_of_repeat (int): y축 방향으로 단위 셀이 반복되는 횟수.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트

        # 단위 셀(unit cell)의 반복 횟수를 설정합니다.
        self.x_number_of_repeat = x_number_of_repeat
        self.y_number_of_repeat = y_number_of_repeat
        self.z_number_of_repeat = z_number_of_repeat

        # 전체 시뮬레이션 박스 길이를 업데이트합니다.
        # World.ubox_length는 단일 단위 셀의 길이이며, 이를 반복 횟수만큼 곱하여 전체 박스 길이를 결정합니다.
        World.box_length = self.x_number_of_repeat * World.ubox_length

        # 특별한 원자(터미널)들을 저장하기 위한 딕셔너리입니다.
        # end_tag 값에 따라 다른 종류의 터미널 원자들을 분류하여 저장합니다.
        self.terminals = {}
        self.terminals[1] = [] # 주 사슬(backbone)의 끝단 원자
        self.terminals[2] = [] # 가교제(crosslinker)의 끝단 원자
        self.terminals[3] = [] # 가교제 내부의 중간 원자
        self.terminals[4] = [] # PBC(주기 경계 조건) 연결에 사용되는 원자

        # 생성된 결합 및 각도의 종류별 개수를 세기 위한 변수들입니다.
        # 주로 통계 및 디버깅 목적으로 사용됩니다.
        self.num_Bonds_44 = 0 # 터미널 4-4 간의 결합 수
        self.num_Bonds_12 = 0 # 터미널 1-2 간의 결합 수
        self.num_Bonds_33 = 0 # 터미널 3-3 간의 결합 수 (현재 코드에서 사용되지 않음)
        self.num_Bonds_23 = 0 # 터미널 2-3 간의 결합 수
        self.num_Bonds_24 = 0 # 터미널 2-4 간의 결합 수

        self.num_Angles_124 = 0 # 각도 1-2-4 타입의 수 (현재 코드에서 사용되지 않음)
        self.num_Angles_123 = 0 # 각도 1-2-3 타입의 수 (현재 코드에서 사용되지 않음)
        self.num_Angles_121 = 0 # 각도 1-2-1 타입의 수 (현재 코드에서 사용되지 않음)


    def make_lines(self, bx, by, bz):
        '''
        원자들이 채워질 시스템 내의 가상 선(경로)들을 생성합니다.
        이 함수는 다이아몬드 네트워크의 단위 셀(unit cell) 내에서
        고분자 주 사슬(backbone)과 가교제(crosslinker)가 위치할 경로를 정의합니다.

        Args:
            bx (int): 현재 단위 셀의 x축 인덱스.
            by (int): 현재 단위 셀의 y축 인덱스.
            bz (int): 현재 단위 셀의 z축 인덱스.

        Returns:
            tuple: (segment_xyz, link_xyz)
                - segment_xyz (list of np.array): 주 사슬 원자들이 위치할 3D 좌표 리스트.
                - link_xyz (list of np.array): 가교제 원자들이 위치할 3D 좌표 리스트.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        
        n_link = 4 # 각 단위 셀에서 생성될 링크(가교제)의 수
        # 주 사슬(segment)의 단량체(monomer) 개수를 계산합니다.
        # 이 계산은 다이아몬드 네트워크의 기하학적 구조와 World.mean_sep(평균 원자 간 거리)에 기반합니다.
        n_segment = np.floor(np.sqrt(np.square(World.ubox_length / 2 - 3 * World.mean_sep) + 2 * np.square(World.ubox_length / 2)) / World.mean_sep) - 1
        print("주 사슬 단량체 수 : ", n_segment - 1)
        n_segment = n_segment.astype(int)

        lines = [] # 원자 경로를 저장할 리스트
        link_axis = np.array([1, 0, 0]) # 링크 축 (주로 x축 방향)

        # 다이아몬드 네트워크의 중심 비스(bis) 헤드 및 테일 위치를 정의합니다.
        # 이들은 단위 셀 내에서 가교제의 시작점과 끝점을 나타냅니다.
        CenterBisHead = np.array([World.ubox_length / 2 + 3 / 2 * World.mean_sep, World.ubox_length / 2, World.ubox_length / 2])
        CenterBisTail = np.array([World.ubox_length / 2 - 3 / 2 * World.mean_sep, World.ubox_length / 2, World.ubox_length / 2])
        print("현재 격자 : ", bx, by, bz)

        # 다이아몬드 네트워크의 큐브 타입(cubeType)을 정의합니다.
        # 이는 단위 셀 내에서 원자들이 연결되는 방식을 결정합니다.
        cubeType = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]])
        # (bx + by + bz)의 홀짝성에 따라 다른 큐브 타입을 적용하여 다이아몬드 격자 패턴을 생성합니다.
        if (bx + by + bz) % 2 == 1:
            cubeType = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]])
        print("현재 격자 요소 ; ", cubeType, bx + by + bz)

        # 주 사슬(segment)의 시작점과 끝점을 정의하는 라인들을 추가합니다.
        # 이 라인들은 CenterBisHead/Tail과 cubeType의 조합으로 생성됩니다.
        lines.append([CenterBisTail, World.ubox_length * cubeType[0] - World.mean_sep * 3 / 2 * np.array([cubeType[0][0] > 0.5], dtype=int) * link_axis + World.mean_sep * 3 / 2 * np.array([cubeType[0][0] < 0.5], dtype=int) * link_axis])
        lines.append([CenterBisTail, World.ubox_length * cubeType[1] - World.mean_sep * 3 / 2 * np.array([cubeType[1][0] > 0.5], dtype=int) * link_axis + World.mean_sep * 3 / 2 * np.array([cubeType[1][0] < 0.5], dtype=int) * link_axis])
        lines.append([CenterBisHead, World.ubox_length * cubeType[2] - World.mean_sep * 3 / 2 * np.array([cubeType[2][0] > 0.5], dtype=int) * link_axis + World.mean_sep * 3 / 2 * np.array([cubeType[2][0] < 0.5], dtype=int) * link_axis])
        lines.append([CenterBisHead, World.ubox_length * cubeType[3] - World.mean_sep * 3 / 2 * np.array([cubeType[3][0] > 0.5], dtype=int) * link_axis + World.mean_sep * 3 / 2 * np.array([cubeType[3][0] < 0.5], dtype=int) * link_axis])

        # CenterBisHead와 CenterBisTail을 연결하는 라인을 추가합니다.
        lines.append([CenterBisHead, CenterBisTail])

        # 추가적인 링크 라인들을 생성합니다.
        if True: # 이 조건문은 항상 참이므로, 항상 실행됩니다.
            for boxPoint in cubeType:
                lines.append([World.ubox_length * boxPoint - World.mean_sep / 2 * np.array([boxPoint[0] > 0.5], dtype=int) * link_axis + World.mean_sep / 2 * np.array([boxPoint[0] < 0.5], dtype=int) * link_axis, 
                              World.ubox_length * boxPoint - World.mean_sep * 3 / 2 * np.array([boxPoint[0] > 0.5], dtype=int) * link_axis + World.mean_sep * 3 / 2 * np.array([boxPoint[0] < 0.5], dtype=int) * link_axis])

        segment_xyz = [] # 주 사슬 원자 좌표를 저장할 리스트
        link_xyz = [] # 가교제 원자 좌표를 저장할 리스트

        # 첫 4개의 라인(주 사슬)에 대해 interp3D를 사용하여 원자 좌표를 보간합니다.
        for p1, p2 in lines[:4]:
            segment_xyz.append(interp3D(n_segment, p1, p2)[1:]) # 시작점과 끝점 제외

        # 나머지 라인(가교제)에 대해 원자 좌표를 정의합니다.
        for p1, p2 in lines[4:]:
            if p1[0] == CenterBisHead[0]:
                # CenterBisHead에서 시작하는 가교제는 4개의 비드로 구성됩니다.
                link_xyz.append(np.array([p1, 2 / 3 * p1 + 1 / 3 * p2, 1 / 3 * p1 + 2 / 3 * p2, p2]))
            else:
                # 그 외의 가교제는 2개의 비드로 구성됩니다.
                link_xyz.append(np.array([p1, p2]))

        return segment_xyz, link_xyz

    def construct_atoms(self):
        '''
          정의된 경로를 따라 실제 원자 객체를 생성하고 시스템에 추가합니다.
          이 메서드는 `make_lines`에서 정의된 가상 경로를 기반으로
          하이드로젤의 주 사슬(backbone) 및 가교제(crosslinker) 원자들을 생성하고 초기 속성을 부여합니다.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        
        # 주기 경계 조건(PBC)을 넘어가는 링커 원자들을 테스트하기 위한 리스트
        PBC_Linker_test = []

        # 정의된 반복 횟수(x, y, z)에 따라 단위 셀을 순회합니다.
        for bx, by, bz in pd(range(self.x_number_of_repeat),
                             range(self.y_number_of_repeat),
                             range(self.z_number_of_repeat)):
            
            # 다이아몬드 네트워크의 홀짝성 규칙에 따라 특정 셀만 처리합니다.
            if (bx + by + bz) % 2 == 0:
                segment_xyz, link_xyz = self.make_lines(bx, by, bz)
            else:
                continue # 홀수 셀은 건너뜁니다.

            # 주 사슬(segment) 원자들을 생성합니다.
            for i in segment_xyz:
                for ii in i:
                    _tmp = Attributes.Atom() # 새로운 원자 객체 생성
                    _tmp.atom_type = p.Config.get_param('hydrogel_components', 'backbone', 'atom_type') # 원자 타입 설정 (예: Martini C1 타입)
                    _tmp.residue_number = 1 # 잔기(residue) 번호
                    _tmp.residue_name = p.Config.get_param('hydrogel_components', 'backbone', 'residue_name') # 잔기 이름: Backbone (주 사슬)
                    _tmp.atom_name = p.Config.get_param('hydrogel_components', 'backbone', 'atom_name') # 원자 이름: Segment
                    _tmp.cgnr = p.Config.get_param('hydrogel_components', 'backbone', 'cgnr') # 전하 그룹 번호
                    _tmp.mass = p.Config.get_param('hydrogel_components', 'backbone', 'mass') # 질량
                    _tmp.charge = p.Config.get_param('hydrogel_components', 'backbone', 'charge') # 전하
                    # 원자 위치 설정: 단위 셀 내 좌표 + 전체 박스 내 오프셋
                    _tmp.position = ii + World.ubox_length * np.array([bx, by, bz])

                    # 원자의 end_tag를 설정하고, 필요한 경우 결합을 생성합니다.
                    # end_tag는 원자의 특수한 연결 상태(예: 사슬의 끝)를 나타냅니다.
                    if np.prod(ii == i[-1, :]): # 현재 원자가 세그먼트의 마지막 원자인 경우
                        _tmp.end_tag = 1 # end_tag 1로 설정
                        # 이전 원자와의 결합 생성
                        if _tmp.atom_id > 0:
                            _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                            _tmp2.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct') # 결합 함수 타입
                            _tmp2.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0') # 평형 거리
                            _tmp2.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1') # 힘 상수
                        self.terminals[_tmp.end_tag].append(_tmp) # 터미널 리스트에 추가
                    elif np.prod(ii == i[0, :]): # 현재 원자가 세그먼트의 첫 번째 원자인 경우
                        _tmp.end_tag = 1 # end_tag 1로 설정
                        self.terminals[_tmp.end_tag].append(_tmp) # 터미널 리스트에 추가
                    else: # 세그먼트의 중간 원자인 경우
                        # 이전 원자와의 결합 생성
                        if _tmp.atom_id > 0:
                            _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                            _tmp2.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct') # 결합 함수 타입
                            _tmp2.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0') # 평형 거리
                            _tmp2.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1') # 힘 상수

            # 가교제(linker) 원자들을 생성합니다.
            testPass = True # PBC 링커 테스트 통과 여부 플래그
            for j in link_xyz:
                for ndx, jj in enumerate(j):
                    # 2개의 비드로 구성된 링커의 첫 번째 비드 처리
                    if len(j)==2 and ndx == 0:
                        # PBC_Linker_test 리스트에 있는 다른 링커 원자와 겹치는지 확인
                        for testP in PBC_Linker_test:
                            if np.linalg.norm(rij(jj + World.ubox_length * np.array([bx, by, bz]),testP, World.box_length)) < 0.1*World.mean_sep:
                                testPass = False # 겹치면 테스트 실패
                        if testPass: # 겹치지 않으면 원자 생성
                            _tmp = Attributes.Atom()
                            _tmp.atom_type = p.Config.get_param('hydrogel_components', 'linkers', 2, 'atom_type')
                            _tmp.residue_number = 1
                            _tmp.residue_name = p.Config.get_param('hydrogel_components', 'linkers', 2, 'residue_name')
                            _tmp.atom_name = p.Config.get_param('hydrogel_components', 'linkers', 2, 'atom_name')
                            _tmp.cgnr = p.Config.get_param('hydrogel_components', 'linkers', 2, 'cgnr')
                            _tmp.mass = p.Config.get_param('hydrogel_components', 'linkers', 2, 'mass')
                            _tmp.charge = p.Config.get_param('hydrogel_components', 'linkers', 2, 'charge')
                            _tmp.position = jj + World.ubox_length * np.array([bx, by, bz])
                            PBC_Linker_test.append(_tmp.position) # 테스트 리스트에 추가
                    # 2개의 비드로 구성된 링커의 두 번째 비드 처리
                    elif len(j) == 2 and ndx == 1:
                        # PBC_Linker_test 리스트에 있는 다른 링커 원자와 겹치는지 확인
                        for testP in PBC_Linker_test:
                            if np.linalg.norm(rij(jj + World.ubox_length * np.array([bx, by, bz]),testP, World.box_length)) < 0.1*World.mean_sep:
                                testPass = False
                        if testPass:
                            _tmp = Attributes.Atom()
                            _tmp.atom_type = p.Config.get_param('hydrogel_components', 'linkers', 1, 'atom_type')
                            _tmp.residue_number = 1
                            _tmp.residue_name = p.Config.get_param('hydrogel_components', 'linkers', 1, 'residue_name')
                            _tmp.atom_name = p.Config.get_param('hydrogel_components', 'linkers', 1, 'atom_name')
                            _tmp.cgnr = p.Config.get_param('hydrogel_components', 'linkers', 1, 'cgnr')
                            _tmp.mass = p.Config.get_param('hydrogel_components', 'linkers', 1, 'mass')
                            _tmp.charge = p.Config.get_param('hydrogel_components', 'linkers', 1, 'charge')
                            _tmp.position = jj + World.ubox_length * np.array([bx, by, bz])
                            PBC_Linker_test.append(_tmp.position)
                    # 4개의 비드로 구성된 링커의 중간 비드 처리 (인덱스 0 < ndx < 3)
                    elif len(j) == 4 and (0<ndx<3):
                        _tmp = Attributes.Atom()
                        _tmp.atom_type = p.Config.get_param('hydrogel_components', 'linkers', 0, 'atom_type')
                        _tmp.residue_number = 1
                        _tmp.residue_name = p.Config.get_param('hydrogel_components', 'linkers', 0, 'residue_name')
                        _tmp.atom_name = p.Config.get_param('hydrogel_components', 'linkers', 0, 'atom_name')
                        _tmp.cgnr = p.Config.get_param('hydrogel_components', 'linkers', 0, 'cgnr')
                        _tmp.mass = p.Config.get_param('hydrogel_components', 'linkers', 0, 'mass')
                        _tmp.charge = p.Config.get_param('hydrogel_components', 'linkers', 0, 'charge')
                        _tmp.position = jj + World.ubox_length * np.array([bx, by, bz])
                    # 그 외의 경우 (4개 비드 링커의 첫/마지막 비드)
                    else:
                        _tmp = Attributes.Atom()
                        _tmp.atom_type = p.Config.get_param('hydrogel_components', 'linkers', 3, 'atom_type')
                        _tmp.residue_number = 1
                        _tmp.residue_name = p.Config.get_param('hydrogel_components', 'linkers', 3, 'residue_name')
                        _tmp.atom_name = p.Config.get_param('hydrogel_components', 'linkers', 3, 'atom_name')
                        _tmp.cgnr = p.Config.get_param('hydrogel_components', 'linkers', 3, 'cgnr')
                        _tmp.mass = p.Config.get_param('hydrogel_components', 'linkers', 3, 'mass')
                        _tmp.charge = p.Config.get_param('hydrogel_components', 'linkers', 3, 'charge')
                        _tmp.position = jj + World.ubox_length * np.array([bx, by, bz])
                    
                    # 링커 원자의 end_tag 설정 및 결합 생성
                    if len(j) == 4 and np.prod(jj == j[-1, :]): # 4개 비드 링커의 마지막 원자
                        _tmp.end_tag = 2
                        _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                        _tmp2.bond_funct = p.Config.get_param('hydrogel_components', 'linkers', 3, 'bond_funct')
                        _tmp2.bond_c0 = p.Config.get_param('hydrogel_components', 'linkers', 3, 'bond_c0')
                        _tmp2.bond_c1 = p.Config.get_param('hydrogel_components', 'linkers', 3, 'bond_c1')
                        self.terminals[_tmp.end_tag].append(_tmp)
                    elif len(j)== 4 and np.prod(jj == j[0, :]): # 4개 비드 링커의 첫 번째 원자
                        _tmp.end_tag = 2
                        self.terminals[_tmp.end_tag].append(_tmp)
                    elif len(j) == 4: # 4개 비드 링커의 중간 원자
                        _tmp.end_tag = 3
                        self.terminals[_tmp.end_tag].append(_tmp)
                        _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                        _tmp2.bond_funct = p.Config.get_param('hydrogel_components', 'linkers', 2, 'bond_funct')
                        _tmp2.bond_c0 = p.Config.get_param('hydrogel_components', 'linkers', 2, 'bond_c0')
                        _tmp2.bond_c1 = p.Config.get_param('hydrogel_components', 'linkers', 2, 'bond_c1')
                    if len(j)== 2 and np.prod(jj == j[0,:]): # 2개 비드 링커의 첫 번째 원자
                        _tmp.end_tag = 4
                        self.terminals[_tmp.end_tag].append(_tmp)
                    elif len(j) == 2 and np.prod(jj==j[-1, :]): # 2개 비드 링커의 마지막 원자
                        _tmp.end_tag = 2
                        self.terminals[_tmp.end_tag].append(_tmp)
                        _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                        self.num_Bonds_24 += 1
                        _tmp2.bond_funct = p.Config.get_param('hydrogel_components', 'linkers', 1, 'bond_funct')
                        _tmp2.bond_c0 = p.Config.get_param('hydrogel_components', 'linkers', 1, 'bond_c0')
                        _tmp2.bond_c1 = p.Config.get_param('hydrogel_components', 'linkers', 1, 'bond_c1')

    def construct_bonds(self, pbc, num_cell, output_dir):
        '''
          특별한 결합들을 생성합니다.
          이 메서드는 `self.terminals`에 저장된 특정 원자들 사이에 결합을 생성합니다.
          주기 경계 조건(PBC)을 고려하여 결합을 형성하며, 생성된 결합의 통계를 기록합니다.

          Args:
              pbc (bool): 주기 경계 조건(PBC)을 적용할지 여부.
              num_cell (int): 시뮬레이션 박스의 한 변에 있는 단위 셀의 개수.
              output_dir (str): 출력 디렉토리 경로.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        
        # PBC를 넘어가는 결합 정보를 기록할 파일 (디버깅 및 분석용)
        output_filename = os.path.join(output_dir, "pbc_bonds.txt")
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w") as w:
            w.write('Atom ID starts from 1 list of bonds over PBC\n')
            # 터미널 4 (PBC 연결에 사용되는 원자) 간의 결합을 탐색합니다.
            for atom_1 in self.terminals[4]:
                for atom_2 in self.terminals[4]:
                    if atom_1 is atom_2: continue # 동일 원자는 건너뜁니다.
                    # 이미 결합된 원자는 건너뜁니다.
                    if atom_2.atom_id in [not_self(atom_1, b).atom_id for b in atom_1.bonded_atoms]: continue
                    
                    # 두 원자 사이의 거리 제곱을 계산합니다.
                    d_sq = dij_sq(atom_1.position, atom_2.position, World.box_length)
                    if pbc: # PBC가 활성화된 경우
                        # 특정 거리 이내에 있는 원자들만 고려합니다.
                        if d_sq < (np.square(World.mean_sep) * 1.2):
                            # 실제 거리와 거리 제곱을 비교하여 PBC를 넘어가는 결합인지 확인합니다.
                            if 2 * d_sq < np.sqrt(np.sum(np.square(atom_1.position - atom_2.position))):
                                w.write("{} {}\n".format(atom_1.atom_id + 1, atom_2.atom_id + 1)) # 파일에 기록
                            
                            # 원자 ID 순서에 따라 결합을 생성하고 통계를 업데이트합니다.
                            if atom_1.atom_id < atom_2.atom_id:
                                _tmp = Attributes.Bond(atom_1.atom_id, atom_2.atom_id)
                                _tmp.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct')
                                _tmp.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0')
                                _tmp.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1')
                            else:
                                _tmp = Attributes.Bond(atom_2.atom_id, atom_1.atom_id)
                                self.num_Bonds_44 +=1
                                _tmp.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct')
                                _tmp.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0')
                                _tmp.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1')
                    elif num_cell != 1: # PBC가 비활성화되었지만, 단위 셀이 1개가 아닌 경우 (일반적인 거리 계산)
                        d_sq_l = np.sum(np.square(atom_1.position - atom_2.position))
                        if d_sq_l < (np.square(World.mean_sep) * 1.2):
                            if atom_1.atom_id < atom_2.atom_id:
                                _tmp = Attributes.Bond(atom_1.atom_id, atom_2.atom_id)
                                self.num_Bonds_44 +=1
                                _tmp.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct')
                                _tmp.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0')
                                _tmp.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1')
                            else:
                                _tmp = Attributes.Bond(atom_2.atom_id, atom_1.atom_id)
                                self.num_Bonds_44 +=1
                                _tmp.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct')
                                _tmp.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0')
                                _tmp.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1')

            # 터미널 1 (주 사슬 끝단)과 터미널 2 (가교제 끝단) 간의 결합을 탐색합니다.
            for atom_1 in self.terminals[1]:
                for atom_2 in self.terminals[2]:
                    if atom_1 is atom_2: continue
                    if atom_2.atom_id in [not_self(atom_1, b).atom_id for b in atom_1.bonded_atoms]: continue

                    d_sq = dij_sq(atom_1.position, atom_2.position, World.box_length) if pbc else np.sum(np.square(atom_1.position - atom_2.position))

                    if d_sq < (np.square(World.mean_sep) * 1.5):
                        if atom_1.atom_id < atom_2.atom_id:
                            _tmp = Attributes.Bond(atom_1.atom_id, atom_2.atom_id)
                        else:
                            _tmp = Attributes.Bond(atom_2.atom_id, atom_1.atom_id)
                        
                        _tmp.bond_funct = p.Config.get_param('hydrogel_components', 'backbone', 'bond_funct')
                        _tmp.bond_c0 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c0')
                        _tmp.bond_c1 = p.Config.get_param('hydrogel_components', 'backbone', 'bond_c1')
                        self.num_Bonds_12 += 1
                        break # Found a bond for atom_1, move to the next atom_1
        w.close()
        print("num of bonds 12 : ", self.num_Bonds_12)
        print("num of bonds 44 : ", self.num_Bonds_44)
        print("num of bonds 24 : ", self.num_Bonds_24)

    def cutter(self, cut_num, random_seed):
        '''
          네트워크가 분리되지 않도록 확인하며 특정 개수(cut_num)의 결합을 자릅니다.
          이 메서드는 하이드로젤 네트워크에서 지정된 수의 결합을 제거하여
          네트워크의 무결성을 유지하면서 구조를 변형합니다.

          Args:
              cut_num (int): 제거할 결합의 총 개수.
              random_seed (int): 무작위 선택을 위한 시드 값.
        '''
        from main_components.Attributes import Atom # 순환 참조를 피하기 위해 함수 내에서 임포트
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        from random import Random # 무작위 선택을 위한 Random 클래스 임포트

        global bond_cut # 전역 변수로, 실제로 잘린 결합들을 저장합니다.
        bond_cut = []
        k = [] # 네트워크 연결성 테스트를 위한 임시 리스트
        kk = 0 # 현재까지 잘린 결합의 수
        while kk != cut_num: # 목표한 개수만큼 결합을 자를 때까지 반복
            # 무작위로 결합 하나를 선택합니다.
            a = Random(random_seed).choice(list(World.Bonds.keys()))
            del World.Bonds[a] # World에서 해당 결합을 임시로 제거합니다.
            
            # 제거된 결합에 연결된 원자들의 bonded_atoms 리스트에서도 해당 결합을 제거합니다.
            kkk = []
            for i in range(len(a)):
                for ii in range(len(World.Atoms[a[i]][0].bonded_atoms)):
                    if World.Atoms[a[i]][0].bonded_atoms[ii].bond_atom_1.atom_id == a[1 - i] \
                            or World.Atoms[a[i]][0].bonded_atoms[ii].bond_atom_2.atom_id == a[1 - i]:
                        kkk.append([a[i], ii])
            for i in range(len(kkk)):
                del World.Atoms[kkk[i][0]][0].bonded_atoms[kkk[i][1]]

            # 네트워크가 분리되지 않았는지 확인하기 위한 연결성 테스트 (BFS/DFS와 유사)
            bonded_p = []
            for i in range(len(World.Atoms[a[0]][0].bonded_atoms)):
                bonded_p.append(World.Atoms[a[0]][0].bonded_atoms[i].bond_atom_1.atom_id)
                bonded_p.append(World.Atoms[a[0]][0].bonded_atoms[i].bond_atom_2.atom_id)

            while True:
                # 연결된 원자들을 탐색하여 bonded_p 리스트를 확장합니다.
                for i in bonded_p[::-1]: # 역순으로 순회하여 리스트 변경에 안전하게
                    for ii in range(len(World.Atoms[i][0].bonded_atoms)):
                        if World.Atoms[i][0].bonded_atoms[ii].bond_atom_1.atom_id not in bonded_p:
                            bonded_p.append(World.Atoms[i][0].bonded_atoms[ii].bond_atom_1.atom_id)
                        if World.Atoms[i][0].bonded_atoms[ii].bond_atom_2.atom_id not in bonded_p:
                            bonded_p.append(World.Atoms[i][0].bonded_atoms[ii].bond_atom_2.atom_id)
                bonded_p = list(set(bonded_p)) # 중복 제거
                k.append(len(bonded_p))

                # 모든 원자가 연결되어 있으면 (네트워크가 분리되지 않았으면) 결합을 자릅니다.
                if len(bonded_p) == len(World.Atoms):
                    bond_cut.append(a) # 실제로 잘린 결합 리스트에 추가
                    kk += 1 # 잘린 결합 수 증가
                    break
                # 네트워크가 분리되었거나 더 이상 연결된 원자를 찾을 수 없으면 (루프에 갇히는 것을 방지)
                elif len(bonded_p) > 0.5 * len(World.Atoms) and k[-1] == k[-2]:
                    Attributes.Bond(a[0], a[1]) # 임시로 제거했던 결합을 다시 추가합니다.
                    break
            k = [] # 다음 시도를 위해 k 리스트 초기화
        
        # 모든 결합 제거 시도가 끝난 후, 실제로 잘린 결합들을 World.Bonds에서 제거합니다.
        # (위에서 임시로 제거하고 다시 추가하는 로직이 있으므로, 최종적으로 제거하는 단계가 필요합니다.)
        for a in bond_cut:
            Attributes.Bond(a[0], a[1]) # 이 부분은 실제로는 다시 추가하는 것이므로, 의도와 다를 수 있습니다.
                                        # `cutter`의 목적이 '자르는' 것이라면, 이 부분은 제거되어야 합니다.
                                        # 현재 코드는 잘린 결합을 다시 추가하여 네트워크를 복구하는 것처럼 보입니다.


    def rand_cutter(self, object, cut_rand_num, random_seed):
        '''
          무작위로 특정 개수의 결합을 선택합니다. (실제로 자르지는 않음)
          이 메서드는 `cut` 메서드에서 실제로 제거될 결합들을 미리 선택하는 역할을 합니다.

          Args:
              object (World): World 객체 (시스템의 결합 정보를 포함).
              cut_rand_num (int): 무작위로 선택할 결합의 개수.
              random_seed (int): 무작위 선택을 위한 시드 값.
        '''
        from random import Random # 무작위 선택을 위한 Random 클래스 임포트
        global rand_bond_cut # 전역 변수로, 무작위로 선택된 결합들을 저장합니다.
        # World 객체의 모든 결합 중에서 `cut_rand_num` 개수만큼 무작위로 선택합니다.
        rand_bond_cut = Random(random_seed).sample(list(object.Bonds.keys()), cut_rand_num)

    def cut(self):
        '''
          `cutter`에서 선택된 결합을 실제로 제거합니다.
          이 메서드는 `bond_cut` 전역 변수에 저장된 결합들을 World 객체에서 영구적으로 제거합니다.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        for a in bond_cut: # `bond_cut` 리스트에 있는 각 결합에 대해
            del World.Bonds[a] # World.Bonds 딕셔너리에서 해당 결합을 제거합니다.
            
            # 제거된 결합에 연결된 원자들의 bonded_atoms 리스트에서도 해당 결합을 제거합니다.
            kkk = []
            for i in range(len(a)):
                for ii in range(len(World.Atoms[a[i]][0].bonded_atoms)):
                    if World.Atoms[a[i]][0].bonded_atoms[ii].bond_atom_1.atom_id == a[1 - i] \
                            or World.Atoms[a[i]][0].bonded_atoms[ii].bond_atom_2.atom_id == a[1 - i]:
                        kkk.append([a[i], ii])
            for i in range(len(kkk)):
                del World.Atoms[kkk[i][0]][0].bonded_atoms[kkk[i][1]]
        self.num_HDG_bonds = len(World.Bonds) # 제거 후 남은 결합의 총 개수를 업데이트합니다.

    def rand_cut(self, object):
        '''
          `rand_cutter`에서 선택된 결합을 실제로 제거합니다.
          이 메서드는 `rand_bond_cut` 전역 변수에 저장된 결합들을 주어진 World 객체에서 제거합니다.

          Args:
              object (World): World 객체 (시스템의 결합 및 원자 정보를 포함).
        '''
        for i in range(len(rand_bond_cut)): # `rand_bond_cut` 리스트에 있는 각 결합에 대해
            del object.Bonds[rand_bond_cut[i]] # World.Bonds 딕셔너리에서 해당 결합을 제거합니다.
            Attributes.Bond.num_bonds -= 1 # Attributes 모듈의 총 결합 수 감소
            # 결합에 참여했던 원자들의 결합 수도 감소시킵니다.
            object.Atoms[rand_bond_cut[i][0]][0].number_of_bonds -= 1
            object.Atoms[rand_bond_cut[i][1]][0].number_of_bonds -= 1
        self.num_HDG_bonds = len(object.Bonds) # 제거 후 남은 결합의 총 개수를 업데이트합니다.

    def construct_chemical_detail(self):
        from main_components.Universe import World
        print(f"World.box_length in construct_chemical_detail: {World.box_length}")
        from core_utils.martini_parser import read_itp_definitions
        import os

        # --- 곁사슬 배치 전략 파라미터 ---
        # 곁사슬을 배치할 최적의 방향을 찾기 위한 설정입니다.

        # 테스트할 후보 방향의 개수. 많을수록 최적의 위치를 찾을 확률이 높지만, 계산 시간이 길어집니다.
        NUM_CANDIDATE_VECTORS = 72
        # 원자 겹침(overlap)을 판단하는 거리 임계값. (World.mean_sep * OVERLAP_THRESHOLD_FACTOR)
        # 이 값보다 가까우면 겹친 것으로 간주합니다.
        OVERLAP_THRESHOLD_FACTOR = 0.8
        # 겹침을 확인할 주변 원자를 검색할 반경. (가장 긴 곁사슬 길이 + a) 보다 커야 합니다.
        # (World.mean_sep * SEARCH_RADIUS_FACTOR)
        SEARCH_RADIUS_FACTOR = 10.0
        # 곁사슬을 얼마나 멀리 밀어낼지 결정하는 스케일링 팩터.
        SIDE_CHAIN_PLACEMENT_SCALE = 0.5

        # --- 1. 시퀀스 생성기 준비 ---
        # monomer_definitions에서 중합 전략(SEQUENCE_STRATEGY)을 읽어와
        # 곁사슬을 생성할 순서를 결정하는 생성기(generator)를 준비합니다.
        try:
            monomer_definitions = p.Config.get_param('monomer_definitions')
            # Load definitions from ITP files if specified
            for monomer in monomer_definitions['MONOMERS']:
                if 'itp_file' in monomer and 'martini_id' in monomer:
                    itp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'martini_v300', monomer['itp_file'])
                    print(f"Loading definition for '{monomer['martini_id']}' from '{itp_path}'...")
                    itp_definitions = read_itp_definitions(itp_path)
                    if monomer['martini_id'] in itp_definitions:
                        monomer['definition'] = itp_definitions[monomer['martini_id']]
                    else:
                        print(f"Warning: '{monomer['martini_id']}' not found in '{monomer['itp_file']}'. Using inline definition.")

            strategy = monomer_definitions['SEQUENCE_STRATEGY']['strategy']
            # 단량체 ID와 정의를 매핑하는 딕셔너리 생성
            monomer_map = {m['id']: m['definition'] for m in monomer_definitions['MONOMERS']}
            sequence_generator = None

            print(f"중합 전략: {strategy}")

            if strategy == 'random':
                # 'random' 전략: MONOMERS 리스트의 'ratio' 가중치에 따라 무작위로 단량체를 선택합니다.
                monomer_defs = [m['definition'] for m in monomer_definitions['MONOMERS']]
                ratios = [m['ratio'] for m in monomer_definitions['MONOMERS']]
                # 매번 random.choices를 호출하여 무작위성을 보장합니다.
                def random_generator():
                    while True:
                        yield random.choices(monomer_defs, weights=ratios, k=1)[0]
                sequence_generator = random_generator()

            elif strategy == 'alternating':
                # 'alternating' 전략: MONOMERS 리스트에 정의된 단량체들을 순서대로 번갈아 가며 사용합니다.
                alternating_defs = [m['definition'] for m in monomer_definitions['MONOMERS']]
                sequence_generator = itertools.cycle(alternating_defs)

            elif strategy == 'block':
                # 'block' 전략: 'blocks'에 정의된 순서와 길이대로 단량체 블록을 반복하여 사용합니다.
                block_sequence = []
                for monomer_id, block_size in monomer_definitions['SEQUENCE_STRATEGY']['blocks']:
                    if monomer_id not in monomer_map:
                        print(f"경고: 'blocks'에 정의된 ID '{monomer_id}'가 MONOMERS 리스트에 없습니다. 건너뜁니다.")
                        continue
                    block_sequence.extend([monomer_map[monomer_id]] * block_size)
                sequence_generator = itertools.cycle(block_sequence)
            
            if sequence_generator is None:
                raise ValueError(f"알 수 없는 전략: {strategy}")

        except (AttributeError, KeyError, ValueError) as e:
            print(f"오류: monomer_definitions 설정 형식이 잘못되었거나 strategy 설정이 유효하지 않습니다. {e}")
            return # 오류 발생 시 함수 종료

        # --- 2. 곁사슬 생성 ---
        _World_Atoms_keys = [*World.Atoms]
        monomer_counts = {m['id']: 0 for m in monomer_definitions['MONOMERS']}
        
        # 효율적인 겹침 계산을 위해 모든 원자 객체 리스트를 미리 만듭니다.
        all_atoms = [World.Atoms[k][0] for k in World.Atoms]

        for _id in tqdm(_World_Atoms_keys):
            backbone_atom = World.Atoms[_id][0]
            if backbone_atom.end_tag > 1 or backbone_atom.residue_name != 'BCKN':
                continue

            # 곁사슬을 추가할 위치 계산을 위한 기준점(p1, p2, p3) 설정
            if backbone_atom.number_of_bonds > 1:
                b1 = not_self(backbone_atom, backbone_atom.bonded_atoms[0])
                b2 = not_self(backbone_atom, backbone_atom.bonded_atoms[1])
                p1, p2, p3 = b1.position, backbone_atom.position, b2.position
            elif backbone_atom.number_of_bonds == 1:
                b1 = not_self(backbone_atom, backbone_atom.bonded_atoms[0])
                # PBC를 고려하여 벡터 계산
                p1, p2, p3 = b1.position, backbone_atom.position, backbone_atom.position + rij(backbone_atom.position, b1.position, World.box_length)
            else:
                continue

            # --- 최적의 곁사슬 배치 방향 탐색 ---
            best_vector = None
            min_penalty = float('inf')
            
            # 겹침 검사를 위한 파라미터
            search_radius_sq = (SEARCH_RADIUS_FACTOR * World.mean_sep)**2
            overlap_threshold_sq = (OVERLAP_THRESHOLD_FACTOR * World.mean_sep)**2
            
            # 자신과 직접 연결된 원자들은 겹침 검사에서 제외
            bonded_atom_ids = {backbone_atom.atom_id}
            for bond in backbone_atom.bonded_atoms:
                bonded_atom_ids.add(not_self(backbone_atom, bond).atom_id)

            # 검색 반경 내의 원자들로 검사 대상을 한정하여 효율성 증대
            nearby_atoms = []
            for atom in all_atoms:
                if atom.atom_id != backbone_atom.atom_id and dij_sq(backbone_atom.position, atom.position, World.box_length) < search_radius_sq:
                    if atom.atom_id not in bonded_atom_ids:
                        nearby_atoms.append(atom)

            chosen_monomer_def = next(sequence_generator)

            for _ in range(NUM_CANDIDATE_VECTORS):
                candidate_vector = random_normal_vector(p1, p2, p3, 1.0, World.box_length)
                
                current_penalty = 0.0
                is_valid = True
                
                # 후보 방향에 따른 곁사슬 원자들의 가상 위치 계산
                tentative_positions = []
                last_pos = backbone_atom.position
                for i, bead_def in enumerate(chosen_monomer_def['beads']):
                    bond_length = chosen_monomer_def['bonds'][i]['length']
                    new_pos = last_pos + candidate_vector * bond_length
                    tentative_positions.append(new_pos)
                    last_pos = new_pos

                # 가상 위치와 주변 원자들 간의 겹침 및 페널티 계산
                for pos in tentative_positions:
                    for nearby_atom in nearby_atoms:
                        d_sq = dij_sq(pos, nearby_atom.position, World.box_length)
                        if d_sq < overlap_threshold_sq:
                            is_valid = False
                            break
                        current_penalty += 1.0 / d_sq
                    if not is_valid:
                        break
                
                if is_valid and current_penalty < min_penalty:
                    min_penalty = current_penalty
                    best_vector = candidate_vector

            # --- 곁사슬 실제 생성 ---
            if best_vector is not None:
                # 통계 기록
                for m in monomer_definitions['MONOMERS']:
                    if m['definition'] is chosen_monomer_def:
                        monomer_counts[m['id']] += 1
                        break
                
                # 가장 패널티가 적은 방향으로 곁사슬 원자 생성
                side_chain_atoms = []
                last_atom = backbone_atom
                for i, bead_def in enumerate(chosen_monomer_def['beads']):
                    side_atom = Attributes.Atom()
                    side_atom.atom_type = bead_def['type']
                    side_atom.residue_name = chosen_monomer_def['residue_name']
                    side_atom.atom_name = bead_def['name']
                    side_atom.mass = bead_def['mass']
                    side_atom.charge = bead_def['charge']
                    
                    bond_length = chosen_monomer_def['bonds'][i]['length']
                    side_atom.position = last_atom.position + best_vector * bond_length
                    
                    side_chain_atoms.append(side_atom)
                    all_atoms.append(side_atom) # 새로 생성된 원자도 전체 목록에 추가
                    last_atom = side_atom

                # 곁사슬 결합 생성
                for bond_def in chosen_monomer_def['bonds']:
                    atom1 = backbone_atom if bond_def['from'] == 'backbone' else side_chain_atoms[bond_def['from']]
                    atom2 = side_chain_atoms[bond_def['to']]
                    bond_params = {k: v for k, v in bond_def.items() if k not in ['from', 'to']}
                    Attributes.Bond(atom1.atom_id, atom2.atom_id, **bond_params)
            else:
                failed_monomer_id = "Unknown"
                for m in monomer_definitions['MONOMERS']:
                    if m['definition'] is chosen_monomer_def:
                        failed_monomer_id = m['id']
                        break
                print(f"경고: 원자 ID {backbone_atom.atom_id}에 곁사슬({failed_monomer_id})을 배치할 유효한 공간을 찾지 못했습니다. 건너뜁니다.")

        print("곁사슬 생성 완료:")
        for name, count in monomer_counts.items():
            print(f"- {name}: {count}개")
        self.num_HDG_atoms = len(World.Atoms) # 최종적으로 생성된 총 원자 수를 업데이트합니다.

    def construct_angles(self):
        '''
          시스템의 모든 각도(angle)를 생성합니다.
          이 메서드는 World 객체에 저장된 원자 및 결합 정보를 기반으로
          세 원자(i-j-k)로 구성되는 각도를 찾아 Attributes.Angle 객체로 생성하고,
          JSON 설정 파일에 정의된 `specific_angles` 규칙에 따라 파라미터를 적용합니다.
        '''
        from main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        
        # --- 하드코딩된 각도 파라미터 ---
        # 이 값들은 곁사슬의 기하학적 구조를 제어하기 위해 사용됩니다.
        
        # 주 사슬(Backbone) - 주 사슬 - 주 사슬 각도 (직선 유지)
        BACKBONE_BACKBONE_ANGLE = 180.0
        BACKBONE_BACKBONE_FORCE_CONSTANT = 5.0

        # 주 사슬 - 주 사슬 - 곁사슬 각도 (곁사슬을 90도로 뻗게 함)
        BACKBONE_SIDECHAIN_ANGLE = 90.0
        BACKBONE_SIDECHAIN_FORCE_CONSTANT = 5.0

        # 곁사슬 내부 각도 (곁사슬을 직선으로 유지)
        SIDECHAIN_INTERNAL_ANGLE = 180.0
        SIDECHAIN_INTERNAL_FORCE_CONSTANT = 20.0 # 더 강한 힘으로 직선 유지

        # --- 각도 생성 로직 ---

        _World_Bonds = World.Bonds
        _atom = World.Atoms

        # JSON에서 기본 각도 설정 로드 (하드코딩된 규칙에 맞지 않는 경우 사용)
        angle_configs = p.Config.get_param('hydrogel_components', 'angles')
        default_params = angle_configs['default_angle']
        
        # 모든 결합 정보 수집
        atom1_bond = []
        atom2_bond = []
        for i, p_list in _World_Bonds.items():
            for bond_obj in p_list:
                atom1_bond.append(bond_obj.bond_atom_1.atom_id)
                atom2_bond.append(bond_obj.bond_atom_2.atom_id)
        
        b11 = np.array(atom1_bond)
        b22 = np.array(atom2_bond)

        # 각도의 중심이 될 수 있는 모든 원자 찾기 (2개 이상의 결합을 가진 원자)
        pos = list(set(b11) & set(b22))
        pos.sort()

        # 각도 생성
        for cen_atom_id in pos:
            cen_atom = _atom[cen_atom_id][0]
            
            # 중심 원자에 연결된 이웃 원자들 찾기
            near_cen_atom_ids = []
            for j in np.where(b11 == cen_atom_id)[0]:
                near_cen_atom_ids.append(b22[j])
            for k in np.where(b22 == cen_atom_id)[0]:
                near_cen_atom_ids.append(b11[k])
            near_cen_atom_ids = sorted(list(set(near_cen_atom_ids)))
            
            # 이웃 원자들의 모든 쌍에 대해 각도 생성
            for i in range(len(near_cen_atom_ids)):
                for j in range(i + 1, len(near_cen_atom_ids)):
                    side_atom1_id = near_cen_atom_ids[i]
                    side_atom2_id = near_cen_atom_ids[j]
                    
                    side_atom1 = _atom[side_atom1_id][0]
                    side_atom2 = _atom[side_atom2_id][0]

                    angle = Attributes.Angle(side_atom1_id, cen_atom_id, side_atom2_id)
                    
                    # 각도 유형 식별 및 파라미터 적용
                    is_cen_backbone = cen_atom.residue_name == 'BCKN'
                    is_side1_backbone = side_atom1.residue_name == 'BCKN'
                    is_side2_backbone = side_atom2.residue_name == 'BCKN'

                    # 유형 1: Backbone-Backbone-Backbone
                    if is_cen_backbone and is_side1_backbone and is_side2_backbone:
                        angle.angle_funct = 1
                        angle.angle_c0 = BACKBONE_BACKBONE_ANGLE
                        angle.angle_c1 = BACKBONE_BACKBONE_FORCE_CONSTANT
                    
                    # 유형 2: Backbone-Backbone-Sidechain
                    elif is_cen_backbone and (is_side1_backbone != is_side2_backbone):
                        angle.angle_funct = 1
                        angle.angle_c0 = BACKBONE_SIDECHAIN_ANGLE
                        angle.angle_c1 = BACKBONE_SIDECHAIN_FORCE_CONSTANT

                    # 유형 3: 곁사슬 내부 각도 (Backbone-Sidechain-Sidechain 또는 Sidechain-Sidechain-Sidechain)
                    elif not is_cen_backbone:
                        # 중심 원자가 곁사슬의 일부인 경우
                        angle.angle_funct = 1
                        angle.angle_c0 = SIDECHAIN_INTERNAL_ANGLE
                        angle.angle_c1 = SIDECHAIN_INTERNAL_FORCE_CONSTANT
                    
                    # 기타: 가교 부분 등 명시적으로 처리되지 않은 각도 (JSON 기본값 사용)
                    else:
                        angle.angle_funct = default_params['angle_funct']
                        angle.angle_c0 = default_params['angle_c0']
                        angle.angle_c1 = default_params['angle_c1']

        self.num_HDG_angles = len(World.Angles)
        print(f"총 {self.num_HDG_angles}개의 각도가 생성되었습니다.")
