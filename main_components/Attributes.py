# numpy 라이브러리를 np라는 이름으로 가져옵니다. 주로 배열 및 수학적 연산에 사용됩니다.
import numpy as np

# 시뮬레이션 환경을 초기화하는 함수입니다.
def initialize():
    # 각 클래스에 정의된 총 개수 변수들을 0으로 초기화합니다.
    Atom.num_atoms = 0  # 총 원자 수
    Bond.num_bonds = 0  # 총 결합 수
    Angle.num_angles = 0  # 총 각도 수
    Network_bond.num_network_bonds = 0  # 총 네트워크 결합 수
    Constraint.num_constraints = 0  # 총 제약조건 수
    Exclusion.num_exclustions = 0  # 총 제외 수
    Dihedral.num_dihedrals = 0  # 총 이면각 수

# 원자(Atom)의 속성을 정의하는 클래스입니다.
class Atom():

    # 클래스 변수로, 생성된 모든 원자의 수를 추적합니다.
    num_atoms = 0

    # 원자 객체가 생성될 때 호출되는 초기화 메서드입니다.
    def __init__(self):
        from main_components.Universe import World

        # MARTINI 원자를 위한 고유 ID를 부여합니다.
        self.atom_id = Atom.num_atoms

        # 총 원자 수를 1 증가시킵니다.
        Atom.num_atoms += 1

        # MARTINI 원자 타입 (예: SC1, P5 등)을 지정합니다.
        # 정수형으로 받아 나중에 사전(dictionary)을 통해 문자열로 변환될 수 있습니다.
        self.atom_type = 'C1'

        # MARTINI 레지듀 번호를 지정합니다.
        self.residue_number = 1
        
        # MARTINI 레지듀 이름을 지정합니다.
        self.residue_name = 'HDG'

        # MARTINI 원자 이름을 지정합니다.
        self.atom_name = 'AT'

        # MARTINI 차지 그룹 번호(Charge Group NumbeR)를 지정합니다.
        self.cgnr = 0
        
        # MARTINI 비드(bead)의 질량을 지정합니다.
        self.mass = 72.0

        # MARTINI 비드(bead)의 전하를 지정합니다.
        self.charge = 0.0

        # 원자의 3D 좌표 (x, y, z)를 numpy 배열로 저장합니다.
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 이 원자와 결합(bond)된 다른 원자들의 리스트입니다.
        self.bonded_atoms = []

        # 이 원자와 네트워크 결합(network bond)된 다른 원자들의 리스트입니다.
        self.network_bonded_atoms = []

        # 이 원자에 제약조건(constraint)이 걸린 다른 원자들의 리스트입니다.
        self.constrained_atoms = []

        # 이 원자와 상호작용에서 제외(exclusion)된 다른 원자들의 리스트입니다.
        self.excluded_atoms = []

        # 이 원자가 포함된 각도(angle) 정보에 사용되는 다른 원자들의 리스트입니다.
        self.angle_atoms = []

        # 이 원자가 형성하는 결합의 수입니다.
        self.number_of_bonds = 0

        # 이 원자가 형성하는 네트워크 결합의 수입니다.
        self.number_of_network_bonds = 0

        # 이 원자가 포함된 각도의 수입니다.
        self.number_of_angles = 0

        # 특별한 터미널(terminal)을 표기하기 위한 태그입니다.
        # 터미널은 결합되어야 하므로, 효율성을 위해 이 태그를 사용합니다.
        self.end_tag = 0
        
        # 생성된 원자 객체를 World 클래스의 Atoms 딕셔너리에 추가합니다.
        World.Atoms[self.atom_id].append(self)


# 결합(Bond)의 속성을 정의하는 클래스입니다.
class Bond():

    # 클래스 변수로, 생성된 모든 결합의 수를 추적합니다.
    num_bonds = 0

    # 결합 객체가 생성될 때 호출되는 초기화 메서드입니다. i와 j는 결합을 형성하는 두 원자의 ID입니다.
    def __init__(self, i, j, **kwargs):
        from main_components.Universe import World

        # Enforce canonical order for atom IDs to prevent duplicate bonds
        if i > j:
            i, j = j, i

        # Check if this bond already exists
        if World.Bonds.get((i, j)):
            return

        # 고유한 결합 ID를 부여합니다.
        self.bond_id = Bond.num_bonds

        # 총 결합 수를 1 증가시킵니다.
        Bond.num_bonds += 1

        # 결합의 기능(function) 타입을 지정합니다. (GROMACS 토폴로지 형식)
        self.bond_funct = kwargs.get('funct', 1)

        # 결합의 첫 번째 원자 객체를 World에서 찾아 할당합니다.
        self.bond_atom_1 = World.Atoms[i][0]
        # 해당 원자의 bonded_atoms 리스트에 이 결합 객체를 추가합니다.
        World.Atoms[i][0].bonded_atoms.append(self)
        # 해당 원자의 결합 수를 1 증가시킵니다.
        World.Atoms[i][0].number_of_bonds += 1

        # 결합의 두 번째 원자 객체를 World에서 찾아 할당합니다.
        self.bond_atom_2 = World.Atoms[j][0]
        World.Atoms[j][0].bonded_atoms.append(self)
        World.Atoms[j][0].number_of_bonds += 1

        # 결합 길이(equilibrium distance) 파라미터 (c0) 입니다. 단위: nm
        self.bond_c0 = kwargs.get('c0', kwargs.get('length', 0.249))

        # 결합 강도(force constant) 파라미터 (c1) 입니다. 단위: kJ/mol/nm^2
        self.bond_c1 = kwargs.get('c1', kwargs.get('fc', 10000.0))

        # 생성된 결합 객체를 World 클래스의 Bonds 딕셔너리에 추가합니다.
        # 두 원자 ID의 튜플을 키로 사용합니다.
        World.Bonds[(i, j)].append(self)


# 네트워크 결합(Network_bond)의 속성을 정의하는 클래스입니다. 일반 결합과 다른 파라미터를 가질 수 있습니다.
class Network_bond():

    # 클래스 변수로, 생성된 모든 네트워크 결합의 수를 추적합니다.
    num_network_bonds = 0

    # 네트워크 결합 객체 초기화 메서드입니다.
    def __init__(self, i, j):
        from main_components.Universe import World
        # 고유한 네트워크 결합 ID를 부여합니다.
        self.network_bond_id = Network_bond.num_network_bonds

        # 총 네트워크 결합 수를 1 증가시킵니다.
        Network_bond.num_network_bonds += 1

        # 네트워크 결합의 기능 타입을 지정합니다.
        self.network_bond_funct = 1

        # 네트워크 결합의 첫 번째 원자입니다.
        self.network_bond_atom_1 = World.Atoms[i][0]
        World.Atoms[i][0].network_bonded_atoms.append(self)
        World.Atoms[i][0].number_of_network_bonds += 1

        # 네트워크 결합의 두 번째 원자입니다.
        self.network_bond_atom_2 = World.Atoms[j][0]
        World.Atoms[j][0].network_bonded_atoms.append(self)
        World.Atoms[j][0].number_of_network_bonds += 1

        # 네트워크 결합 길이 파라미터입니다.
        self.network_bond_c0 = 0.249

        # 네트워크 결합 강도 파라미터입니다.
        self.network_bond_c1 = 500.0

        # 생성된 객체를 World의 Network_bonds 딕셔너리에 추가합니다.
        World.Network_bonds[
            (self.network_bond_atom_1.atom_id,
             self.network_bond_atom_2.atom_id)
        ].append(self)


# 제약조건(Constraint)의 속성을 정의하는 클래스입니다. 두 원자 사이의 거리를 고정시킵니다.
class Constraint():

    # 클래스 변수로, 생성된 모든 제약조건의 수를 추적합니다.
    num_constraints = 0

    # 제약조건 객체 초기화 메서드입니다.
    def __init__(self, i, j):
        from main_components.Universe import World

        # 고유한 제약조건 ID를 부여합니다.
        self.constraint_id = Constraint.num_constraints

        # 총 제약조건 수를 1 증가시킵니다.
        Constraint.num_constraints += 1

        # 제약조건의 첫 번째 원자입니다.
        self.constraint_atom_1 = World.Atoms[i][0]
        World.Atoms[i][0].constrained_atoms.append(self)
        #World.Atoms[i][0].number_of_bonds += 1 # 주석 처리됨

        # 제약조건의 두 번째 원자입니다.
        self.constraint_atom_2 = World.Atoms[j][0]
        World.Atoms[j][0].constrained_atoms.append(self)
        #World.Atoms[j][0].number_of_bonds += 1 # 주석 처리됨

        # 제약조건의 기능 타입을 지정합니다.
        self.constraint_funct = 1

        # 제약조건 거리 파라미터입니다.
        self.constraint_c0 = 0.249

        # 생성된 객체를 World의 Constraints 딕셔너리에 추가합니다.
        World.Constraints[
            (self.constraint_atom_1.atom_id,
             self.constraint_atom_2.atom_id)
        ].append(self)

# 제외(Exclusion)의 속성을 정의하는 클래스입니다. 특정 원자 쌍 간의 비결합 상호작용을 무시하도록 설정합니다.
class Exclusion():

    # 클래스 변수로, 생성된 모든 제외 항목의 수를 추적합니다.
    num_exclustions = 0

    # 제외 객체 초기화 메서드입니다.
    def __init__(self, i, j):
        from main_components.Universe import World
        # 고유한 제외 ID를 부여합니다.
        self.exclustion_id = Exclusion.num_exclustions

        # 총 제외 수를 1 증가시킵니다.
        Exclusion.num_exclustions += 1

        # 제외의 첫 번째 원자입니다.
        self.exclusion_atom_1 = World.Atoms[i][0]
        World.Atoms[i][0].excluded_atoms.append(self)
        # World.Atoms[i][0].number_of_bonds += 1 # 주석 처리됨

        # 제외의 두 번째 원자입니다.
        self.exclusion_atom_2 = World.Atoms[j][0]
        World.Atoms[j][0].excluded_atoms.append(self)
        # World.Atoms[j][0].number_of_bonds += 1 # 주석 처리됨

        # 생성된 객체를 World의 Exclusions 딕셔너리에 추가합니다.
        World.Exclusions[
            (self.exclusion_atom_1.atom_id,
             self.exclusion_atom_2.atom_id)
        ].append(self)

# 각도(Angle)의 속성을 정의하는 클래스입니다. 세 원자(i-j-k)가 이루는 각도에 대한 포텐셜을 정의합니다.
class Angle():
    
    # 클래스 변수로, 생성된 모든 각도의 수를 추적합니다.
    num_angles = 0

    # 각도 객체 초기화 메서드입니다. i, j, k는 각도를 형성하는 세 원자의 ID입니다.
    def __init__(self, i, j, k):
        from main_components.Universe import World
        
        # 고유한 각도 ID를 부여합니다.
        self.angle_id = Angle.num_angles

        # 총 각도 수를 1 증가시킵니다.
        Angle.num_angles += 1

        # 각도의 기능 타입을 지정합니다.
        self.angle_funct = 0

        # 각도의 첫 번째 원자 (i) 입니다.
        self.angle_atom_1 = World.Atoms[i][0]
        World.Atoms[i][0].angle_atoms.append(self)
        World.Atoms[i][0].number_of_angles += 1
        
        # 각도의 두 번째 원자 (j, 중심 원자) 입니다.
        self.angle_atom_2 = World.Atoms[j][0]
        World.Atoms[j][0].angle_atoms.append(self)
        World.Atoms[j][0].number_of_angles += 1

        # 각도의 세 번째 원자 (k) 입니다.
        self.angle_atom_3 = World.Atoms[k][0]
        World.Atoms[k][0].angle_atoms.append(self)
        World.Atoms[k][0].number_of_angles += 1

        # 평형 각도(equilibrium angle) 파라미터 (c0) 입니다. 단위: 도(degree)
        self.angle_c0 = 180.000
        
        # 각도 강도(force constant) 파라미터 (c1) 입니다. 단위: kJ/mol/rad^2
        self.angle_c1 = 75

        # 생성된 객체를 World의 Angles 딕셔너리에 추가합니다.
        World.Angles[
                    (self.angle_atom_1.atom_id,
                     self.angle_atom_2.atom_id,
                     self.angle_atom_3.atom_id)
                    ].append(self)


# 이면각(Dihedral)의 속성을 정의하는 클래스입니다. 네 원자(i-j-m-n)가 이루는 이면각에 대한 포텐셜을 정의합니다.
class Dihedral():

    # 클래스 변수로, 생성된 모든 이면각의 수를 추적합니다.
    num_dihedrals = 0

    # 이면각 객체 초기화 메서드입니다. i, j, m, n은 이면각을 형성하는 네 원자의 ID이며, c0는 위상(phase) 각도입니다.
    def __init__(self, i, j, m, n, c0):
        from main_components.Universe import World
        
        # 고유한 이면각 ID를 부여합니다.
        self.dihedral_id = Dihedral.num_dihedrals

        # 총 이면각 수를 1 증가시킵니다.
        Dihedral.num_dihedrals += 1
        
        # 이면각의 기능 타입을 지정합니다.
        self.dihedral_funct = 0
        
        # 이면각의 첫 번째 원자 (i) 입니다.
        self.dihedral_atom_1 = World.Atoms[i][0]
        
        # 이면각의 두 번째 원자 (j) 입니다.
        self.dihedral_atom_2 = World.Atoms[j][0]
        
        # 이면각의 세 번째 원자 (m) 입니다.
        self.dihedral_atom_3 = World.Atoms[m][0]

        # 이면각의 네 번째 원자 (n) 입니다.
        self.dihedral_atom_4 = World.Atoms[n][0]

        # 위상 각도(phase angle) 파라미터 (c0) 입니다.
        self.dihedral_c0 = c0

        # 이면각 강도(force constant) 파라미터 (c1) 입니다. 여기서는 사용되지 않아 None으로 설정됩니다.
        self.dihedral_c1 = None

        # 주기성(multiplicity) 파라미터 (c2) 입니다. 여기서는 사용되지 않아 None으로 설정됩니다.
        self.dihedral_c2 = None
        
        # 생성된 객체를 World의 Dihedrals 딕셔너리에 추가합니다.
        World.Dihedrals[
                    (self.dihedral_atom_1.atom_id,
                     self.dihedral_atom_2.atom_id,
                     self.dihedral_atom_3.atom_id,
                     self.dihedral_atom_4.atom_id,
                     self.dihedral_c0)
                ].append(self)