# 작성자: Seunghyok Rho, Chongyong Nam, Sebin Kim


##### 수정 이력 (2021/04/20) ######
# .gro 파일 수정: atom type -> atom name

import numpy as np
import os


def write_to_xyz(object, filename='xyz.xyz'):
    '''
      시스템의 원자 좌표를 간단한 .xyz 파일 형식으로 저장합니다.
      시각화 프로그램에서 구조를 빠르게 확인하는 데 유용합니다.

      Args:
          object (World): 원자 정보를 포함하는 World 객체
          filename (str): 저장할 파일 이름
    '''
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')
    f.write('{:d}\n'.format(len(object.Atoms))) # 첫 줄: 전체 원자 수
    f.write('\n') # 둘째 줄: 주석 (비워둠)
    for at in object.Atoms:
        # 셋째 줄부터: 원소 기호, x, y, z 좌표
        f.write('{:d}  {:.8f}  {:.8f}  {:.8f}\n'.format(np.random.randint(1000), # 원소 기호 대신 임의의 정수 사용
            float(object.Atoms[at][0].position[0]),
            float(object.Atoms[at][0].position[1]),
            float(object.Atoms[at][0].position[2]))
               )
    f.close()


def write_to_lammps(object, filename='lammps.data'):
    '''
      시스템 정보를 LAMMPS 데이터 파일 형식으로 저장합니다.

      Args:
          object (World): 시스템 정보를 포함하는 World 객체
          filename (str): 저장할 파일 이름
    '''
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')

    f.write('AuPS initial configuration\n\n') # 주석

    # 헤더 정보: 원자, 결합, 각도 등의 개수와 타입 개수
    f.write('{:d} atoms\n'.format(len(object.Atoms)))
    f.write('{:d} bonds\n'.format(len(object.Bonds)))
    f.write('{:d} angles\n'.format(len(object.Angles)))
    f.write('{:d} atom types\n'.format(2)) # 원자 타입 개수 (하드코딩)
    f.write('{:d} bond types\n'.format(2)) # 결합 타입 개수 (하드코딩)
    f.write('{:d} angle types\n\n'.format(2)) # 각도 타입 개수 (하드코딩)

    # 시뮬레이션 박스 크기 정보
    xlo, xhi = 0.0, object.box_length
    ylo, yhi = 0.0, object.box_length
    zlo, zhi = 0.0, object.box_length

    f.write('{:.8f} {:.8f} xlo xhi\n'.format(xlo, xhi))
    f.write('{:.8f} {:.8f} ylo yhi\n'.format(ylo, yhi))
    f.write('{:.8f} {:.8f} zlo zhi\n\n'.format(zlo, zhi))

    # 질량 정보
    f.write('Masses\n\n')
    f.write('1 {:.8f}\n'.format(1.0))
    f.write('2 {:.8f}\n\n'.format(1.0))

    # 원자 정보: atom-ID, molecule-ID, atom-type, charge, x, y, z
    f.write('Atoms\n\n')
    for i in object.Atoms:
        atom = object.Atoms[i][0]
        tmp = 1 # molecule-ID (임시로 1로 설정)
        f.write('{:<d}   {:d}    {:d}    {:.8}    {:.8f}   {:.8f}  {:.8f}\n'.
                format(atom.atom_id + 1, tmp, 1, atom.charge, # atom-type도 임시로 1로 설정
                       atom.position[0], atom.position[1], atom.position[2]))
    
    # 결합 정보: bond-ID, bond-type, atom1-ID, atom2-ID
    f.write('\nBonds\n\n')
    for j, b in enumerate(object.Bonds):
        bond = object.Bonds[b][0]
        f.write('{:d} {:d} {:d} {:d}\n'.
                format(j + 1, bond.bond_type + 1, b[0] + 1, b[1] + 1))

    # 각도 정보: angle-ID, angle-type, atom1-ID, atom2-ID, atom3-ID
    f.write('\nAngles\n\n')
    for k, z in enumerate(object.Angles):
        angle = object.Angles[z][0]
        f.write('{:d} {:d} {:d} {:d} {:d}\n'.format(k + 1, angle.angle_type + 1, z[0] + 1, z[1] + 1, z[2] + 1))

    # 이면각 정보
    f.write('\nDihedrals\n\n')
    for p, q in enumerate(object.Dihedrals):
        f.write('{} {} {} {} {} {}\n'.format(p + 1, 1, q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1))

    f.close()


def write_to_gro(object, filename='gromacs.gro'):
    '''
      시스템 정보를 GROMACS .gro 파일 형식으로 저장합니다.

      Args:
          object (World): 시스템 정보를 포함하는 World 객체
          DNA (DNAimport): DNA 정보 객체 (여기서는 사용되지 않음)
          filename (str): 저장할 파일 이름
    '''
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')
    f.write('Gromacs.gro file\n') # 주석
    f.write(4 * ' ' + '{}\n'.format(len(object.Atoms))) # 전체 원자 수
    
    # 원자 정보: res-num, res-name, atom-name, atom-num, x, y, z
    # GROMACS .gro 파일은 고정된 자릿수 형식을 따릅니다.
    for i in object.Atoms:
        atom = object.Atoms[i][0]
        f.write('{:>5d}{:<5}{:<5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n'.format(
            int(atom.residue_number) % 100000, # 레지듀 번호
            atom.residue_name, # 레지듀 이름
            atom.atom_name, # 원자 이름
            (atom.atom_id + 1) % 100000, # 원자 번호
            atom.position[0], # x 좌표 (nm)
            atom.position[1], # y 좌표 (nm)
            atom.position[2]  # z 좌표 (nm)
        ))
        
    # 마지막 줄: 시뮬레이션 박스 벡터 (x, y, z)
    f.write('   {:.5f}    {:.5f}    {:.5f}\n'.format(object.box_length, object.box_length, object.box_length))
    f.close()
    return 1

def write_to_itp(object, filename='gromacs.itp', moleculetype_name='HDGEL'):
    '''
      시스템의 토폴로지 정보를 GROMACS .itp 파일 형식으로 저장합니다.
      이 파일은 분자 내 상호작용(결합, 각도 등)을 정의합니다.
    '''
    # --- 파일 작성 전, 특정 원자 쌍들의 ID를 분석하여 별도의 파일로 저장하는 부분 (분석용) ---
    # (PAAM 사슬의 양 끝단 쌍, MBA 가교의 특정 원자 쌍 등을 찾아 저장)
    # 이 부분은 주석 처리가 길어지므로 상세 설명은 생략합니다.
    # ...

    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w')
    f.write(';Gromacs.itp file\n')
    
    # --- [ moleculetype ] 섹션 ---
    # 분자 이름과 비결합 상호작용 제외 규칙(nrexcl)을 정의합니다.
    f.write('[ moleculetype ]\n')
    f.write('; name  nrexcl\n')
    f.write(f'{moleculetype_name}           1\n') # 분자 이름: HDGEL, nrexcl: 1 (1-2 상호작용 제외)
    f.write('#define RUBBER_BANDS\n\n') # 전처리기 지시문: Elastic network(고무줄) 결합을 활성화

    # --- [ atoms ] 섹션 ---
    # 시스템의 모든 원자 정보를 정의합니다.
    f.write('[ atoms ]\n')
    f.write(';   nr    type    resnr   residu    atom    cgnr  charge  mass\n')
    for i in range(len(object.Atoms)):
        atom = object.Atoms[i][0]
        f.write('{:<7d}{:<6}{:<6d}{:<6}{:<6}{:<6d}{:<8.4f}{:<8.4f}\n'.format(
            atom.atom_id + 1, atom.atom_type, int(atom.residue_number),
            atom.residue_name, atom.atom_name, int(atom.cgnr),
            float(atom.charge), float(atom.mass)))

    # --- [ bonds ] 섹션 ---
    # 일반적인 결합 정보를 정의합니다.
    f.write('\n[ bonds ]\n\n')
    for j, b in enumerate(object.Bonds):
        bond = object.Bonds[b][0]
        f.write('{:d}  {:d}   {:d}  {:f} {:f}\n'.format(
            bond.bond_atom_1.atom_id + 1, bond.bond_atom_2.atom_id + 1, 
            bond.bond_funct, bond.bond_c0, bond.bond_c1))
            
    # Elastic network (RUBBER_BANDS) 결합 정보를 정의합니다.
    f.write('#ifdef RUBBER_BANDS\n')
    f.write('#ifndef RUBBER_FC\n')
    f.write('#define RUBBER_FC 500.000000\n') # 힘 상수를 정의
    f.write('#endif\n')
    for i, n in enumerate(object.Network_bonds):
        network_bond = object.Network_bonds[n][0]
        f.write('{:d}  {:d}   {:d}  {:f} {:s}\n'.format(
            network_bond.network_bond_atom_1.atom_id + 1, 
            network_bond.network_bond_atom_2.atom_id + 1, 
            network_bond.network_bond_funct, 
            network_bond.network_bond_c0, 
            network_bond.network_bond_c1))
    f.write('#endif\n')

    # --- [ constraints ] 섹션 ---
    # 두 원자 사이의 거리를 고정하는 제약조건 정보를 정의합니다.
    f.write('\n[ constraints ]\n\n')
    for i, c in enumerate(object.Constraints):
        constraint = object.Constraints[c][0]
        f.write('{:d}  {:d}  {:d}  {:f}\n'.format(
            constraint.constraint_atom_1.atom_id + 1, constraint.constraint_atom_2.atom_id + 1, 
            constraint.constraint_funct, constraint.constraint_c0))

    # --- [ exclusions ] 섹션 ---
    # 비결합 상호작용 계산에서 제외할 원자 쌍을 정의합니다.
    f.write('\n[ exclusions ]\n\n')
    for i, e in enumerate(object.Exclusions):
        exclusion = object.Exclusions[e][0]
        f.write('{:d}  {:d}\n'.format(
            exclusion.exclusion_atom_1.atom_id + 1, exclusion.exclusion_atom_2.atom_id + 1))

    # --- [ angles ] 섹션 ---
    # 세 원자가 이루는 각도에 대한 포텐셜 정보를 정의합니다.
    f.write('\n[ angles ]\n\n')
    for k, z in enumerate(object.Angles):
        angle = object.Angles[z][0]
        f.write('{:5d}  {:5d}  {:5d}  {:5d}  {:f}  {:f}\n'.format(
            angle.angle_atom_1.atom_id + 1, angle.angle_atom_2.atom_id + 1, angle.angle_atom_3.atom_id + 1, 
            angle.angle_funct, angle.angle_c0, angle.angle_c1))

    # --- [ dihedrals ] 섹션 ---
    # 네 원자가 이루는 이면각에 대한 포텐셜 정보를 정의합니다.
    f.write('\n[ dihedrals ]\n\n')
    for p, q in enumerate(object.Dihedrals):
        dihedral = object.Dihedrals[q][0]
        if dihedral.dihedral_c2 == ' ':
            f.write('{} {} {} {} {} {} {}\n'.format(
                dihedral.dihedral_atom_1.atom_id + 1, dihedral.dihedral_atom_2.atom_id + 1, 
                dihedral.dihedral_atom_3.atom_id + 1, dihedral.dihedral_atom_4.atom_id + 1, 
                dihedral.dihedral_funct, dihedral.dihedral_c0, dihedral.dihedral_c1))
        else:
            f.write('{} {} {} {} {} {} {} {}\n'.format(
                dihedral.dihedral_atom_1.atom_id + 1, dihedral.dihedral_atom_2.atom_id + 1, 
                dihedral.dihedral_atom_3.atom_id + 1, dihedral.dihedral_atom_4.atom_id + 1, 
                dihedral.dihedral_funct, dihedral.dihedral_c0, dihedral.dihedral_c1, dihedral.dihedral_c2))
    f.close()

    return 1