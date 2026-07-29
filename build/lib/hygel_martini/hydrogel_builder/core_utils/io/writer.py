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
    with open(filename, 'w') as f:
        f.write('{:d}\n'.format(len(object.Atoms))) # 첫 줄: 전체 원자 수
        f.write('\n') # 둘째 줄: 주석 (비워둠)
        for at in object.Atoms:
            # 셋째 줄부터: 원소 기호, x, y, z 좌표
            f.write('{:d}  {:.8f}  {:.8f}  {:.8f}\n'.format(np.random.randint(1000), # 원소 기호 대신 임의의 정수 사용
                float(object.Atoms[at][0].position[0]),
                float(object.Atoms[at][0].position[1]),
                float(object.Atoms[at][0].position[2]))
                   )


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
    with open(filename, 'w') as f:
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
        box_vec = getattr(object, 'box_vector', None)
        if box_vec is None or not np.any(box_vec):
            box_vec = np.array([object.box_length, object.box_length, object.box_length], dtype=float)
        f.write('   {:.5f}    {:.5f}    {:.5f}\n'.format(float(box_vec[0]), float(box_vec[1]), float(box_vec[2])))
    return 1

def write_to_itp(object, filename='gromacs.itp', moleculetype_name='HDGEL'):
    '''
      시스템의 토폴로지 정보를 GROMACS .itp 파일 형식으로 저장합니다.
      이 파일은 분자 내 상호작용(결합, 각도 등)을 정의합니다.
    '''
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(';Gromacs.itp file\n')

        # --- [ moleculetype ] 섹션 ---
        f.write('[ moleculetype ]\n')
        f.write('; name  nrexcl\n')
        f.write(f'{moleculetype_name}           1\n') # 분자 이름: HDGEL, nrexcl: 1 (1-2 상호작용 제외)
        f.write('#define RUBBER_BANDS\n\n') # 전처리기 지시문: Elastic network(고무줄) 결합을 활성화

        # --- [ atoms ] 섹션 ---
        f.write('[ atoms ]\n')
        f.write(';   nr    type    resnr   residu    atom    cgnr  charge  mass\n')
        for i in range(len(object.Atoms)):
            atom = object.Atoms[i][0]
            f.write('{:<7d}{:<6}{:<6d}{:<6}{:<6}{:<6d}{:<8.4f}{:<8.4f}\n'.format(
                atom.atom_id + 1, atom.atom_type, int(atom.residue_number),
                atom.residue_name, atom.atom_name, int(atom.cgnr),
                float(atom.charge), float(atom.mass)))

        # --- [ bonds ] 섹션 ---
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
        f.write('\n[ constraints ]\n\n')
        for i, c in enumerate(object.Constraints):
            constraint = object.Constraints[c][0]
            f.write('{:d}  {:d}  {:d}  {:f}\n'.format(
                constraint.constraint_atom_1.atom_id + 1, constraint.constraint_atom_2.atom_id + 1,
                constraint.constraint_funct, constraint.constraint_c0))

        # --- [ exclusions ] 섹션 ---
        f.write('\n[ exclusions ]\n\n')
        for i, e in enumerate(object.Exclusions):
            exclusion = object.Exclusions[e][0]
            f.write('{:d}  {:d}\n'.format(
                exclusion.exclusion_atom_1.atom_id + 1, exclusion.exclusion_atom_2.atom_id + 1))

        # --- [ angles ] 섹션 ---
        f.write('\n[ angles ]\n\n')
        for k, z in enumerate(object.Angles):
            for angle in object.Angles[z]:
                f.write('{:5d}  {:5d}  {:5d}  {:5d}  {:f}  {:f}\n'.format(
                    angle.angle_atom_1.atom_id + 1, angle.angle_atom_2.atom_id + 1, angle.angle_atom_3.atom_id + 1,
                    angle.angle_funct, angle.angle_c0, angle.angle_c1))

        # --- [ dihedrals ] 섹션 ---
        f.write('\n[ dihedrals ]\n\n')
        for p, q in enumerate(object.Dihedrals):
            for dihedral in object.Dihedrals[q]:
                d_params = getattr(dihedral, "dihedral_params", None)
                ids_str = '{} {} {} {} {}'.format(
                    dihedral.dihedral_atom_1.atom_id + 1, dihedral.dihedral_atom_2.atom_id + 1,
                    dihedral.dihedral_atom_3.atom_id + 1, dihedral.dihedral_atom_4.atom_id + 1,
                    dihedral.dihedral_funct)
                if d_params:
                    f.write('{} {}\n'.format(ids_str, ' '.join(str(p) for p in d_params)))
                else:
                    c2 = dihedral.dihedral_c2
                    if c2 in (None, ' ', ''):
                        f.write('{} {} {}\n'.format(ids_str, dihedral.dihedral_c0, dihedral.dihedral_c1))
                    else:
                        f.write('{} {} {} {}\n'.format(ids_str, dihedral.dihedral_c0, dihedral.dihedral_c1, c2))

        # --- 추가 섹션 (파서에서 보존된 항목) ---
        extras = getattr(object, "OtherSections", {}) or {}

        def _write_simple_section(sec_name, rows):
            if not rows:
                return
            f.write(f"\n[ {sec_name} ]\n")
            for row in rows:
                f.write(row + "\n")

        # constraints / pairs / exclusions
        if extras.get("constraints"):
            f.write("\n[ constraints ]\n")
            for c in extras["constraints"]:
                params = " ".join(str(x) for x in c.get("params", []))
                f.write(f"{int(c['i'])+1:5d} {int(c['j'])+1:5d} {int(c.get('funct',1)):3d} {params}\n")

        if extras.get("pairs"):
            f.write("\n[ pairs ]\n")
            for p_def in extras["pairs"]:
                params = " ".join(str(x) for x in p_def.get("params", []))
                f.write(f"{int(p_def['i'])+1:5d} {int(p_def['j'])+1:5d} {int(p_def.get('funct',1)):3d} {params}\n")

        if extras.get("exclusions"):
            f.write("\n[ exclusions ]\n")
            for ex in extras["exclusions"]:
                lst = [str(int(ex['atom'])+1)] + [str(int(x)+1) if isinstance(x, (int,float)) else str(x) for x in ex.get("exclude",[])]
                f.write(" ".join(lst) + "\n")

        # virtual sites (section별로 그룹)
        if extras.get("virtual_sites"):
            vs_by_sec = {}
            for vs in extras["virtual_sites"]:
                sec = vs.get("section", "virtual_sites")
                vs_by_sec.setdefault(sec, []).append(vs)
            for sec, vs_list in vs_by_sec.items():
                f.write(f"\n[ {sec} ]\n")
                for vs in vs_list:
                    line = vs.get("line")
                    if line:
                        f.write(line + "\n")
                    else:
                        parts = vs.get("parts", [])
                        f.write(" ".join(str(p) for p in parts) + "\n")

        # restraints & others (raw values)
        for sec, rows in extras.items():
            if sec in ("constraints","pairs","exclusions","virtual_sites","cmaptypes","polarization"):
                continue
            if sec.endswith("_restraints") or "restraint" in sec:
                f.write(f"\n[ {sec} ]\n")
                for r in rows:
                    vals = r.get("values") or r.get("line") or r
                    if isinstance(vals, list):
                        f.write(" ".join(str(v) for v in vals) + "\n")
                    else:
                        f.write(str(vals) + "\n")

        if extras.get("cmaptypes"):
            f.write("\n[ cmaptypes ]\n")
            for cm in extras["cmaptypes"]:
                row = cm.get("row") if isinstance(cm, dict) else cm
                if isinstance(row, list):
                    f.write(" ".join(str(x) for x in row) + "\n")
                else:
                    f.write(str(row) + "\n")

        if extras.get("polarization"):
            f.write("\n[ polarization ]\n")
            for pol in extras["polarization"]:
                params = " ".join(str(p) for p in pol.get("params", []))
                f.write(f"{int(pol['atom'])+1:5d} {params}\n")

        # 기타 raw 섹션
        if extras:
            for sec, rows in extras.items():
                if sec in ("constraints","pairs","exclusions","virtual_sites","cmaptypes","polarization") or sec.endswith("_restraints") or "restraint" in sec:
                    continue
                f.write(f"\n[ {sec} ]\n")
                for r in rows:
                    if isinstance(r, dict) and "line" in r:
                        f.write(r["line"] + "\n")
                    else:
                        f.write(str(r) + "\n")

    return 1


def write_combined_itp(world, filename, moleculetype_name):
    """
    월드의 Atoms/Bonds/Angles/Dihedrals/Constraints/Exclusions 및 OtherSections에 저장된
    추가 섹션을 모두 포함한 단일 ITP를 작성합니다. 외부 ITP를 건드리지 않고,
    월드에서 생성된 분자에 대해서만 사용합니다.
    """
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('; Combined Gromacs.itp file generated by write_combined_itp\n')
        f.write('[ moleculetype ]\n; name  nrexcl\n')
        f.write(f'{moleculetype_name}           1\n\n')

        # atoms
        f.write('[ atoms ]\n')
        f.write(';   nr    type    resnr   residu    atom    cgnr  charge  mass\n')
        for i in range(len(world.Atoms)):
            atom = world.Atoms[i][0]
            f.write('{:<7d}{:<6}{:<6d}{:<6}{:<6}{:<6d}{:<8.4f}{:<8.4f}\n'.format(
                atom.atom_id + 1, atom.atom_type, int(atom.residue_number),
                atom.residue_name, atom.atom_name, int(atom.cgnr),
                float(atom.charge), float(atom.mass)))

        # bonds
        f.write('\n[ bonds ]\n')
        for b in world.Bonds:
            bond = world.Bonds[b][0]
            f.write('{:d}  {:d}   {:d}  {:f} {:f}\n'.format(
                bond.bond_atom_1.atom_id + 1, bond.bond_atom_2.atom_id + 1,
                bond.bond_funct, bond.bond_c0, bond.bond_c1))

        # constraints
        f.write('\n[ constraints ]\n')
        for c in world.Constraints:
            constraint = world.Constraints[c][0]
            f.write('{:d}  {:d}  {:d}  {:f}\n'.format(
                constraint.constraint_atom_1.atom_id + 1, constraint.constraint_atom_2.atom_id + 1,
                constraint.constraint_funct, constraint.constraint_c0))

        # exclusions
        f.write('\n[ exclusions ]\n')
        for e in world.Exclusions:
            exclusion = world.Exclusions[e][0]
            f.write('{:d}  {:d}\n'.format(
                exclusion.exclusion_atom_1.atom_id + 1, exclusion.exclusion_atom_2.atom_id + 1))

        # angles
        f.write('\n[ angles ]\n')
        for z in world.Angles:
            for angle in world.Angles[z]:
                f.write('{:5d}  {:5d}  {:5d}  {:5d}  {:f}  {:f}\n'.format(
                    angle.angle_atom_1.atom_id + 1, angle.angle_atom_2.atom_id + 1, angle.angle_atom_3.atom_id + 1,
                    angle.angle_funct, angle.angle_c0, angle.angle_c1))

        # impropers는 현재 OtherSections로만 전달되며, emit_rich_itp_sections=true일 때만 기록됨

        # OtherSections 그대로 append (옵션으로 on/off)
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            emit_rich = Config.get_param("simulation_parameters").get("emit_rich_itp_sections", True)
        except Exception:
            emit_rich = False

        extras = getattr(world, "OtherSections", {}) or {}
        if emit_rich and extras:
            try:
                raw_counts = {k: len(v) for k, v in extras.items() if v}
                Config.debug_log(f"[rich-itp-raw-counts] {raw_counts}")
            except Exception:
                pass
            try:
                from hygel_martini.hydrogel_builder.core_utils.templates.rich_itp_validator import validate_and_filter_other_sections
                try:
                    strict = Config.get_param("simulation_parameters").get("strict_rich_itp_validation", False)
                except Exception:
                    strict = False
                extras, warnings = validate_and_filter_other_sections(
                    extras, atom_count=len(world.Atoms), strict=strict
                )
                try:
                    filt_counts = {k: len(v) for k, v in extras.items() if v}
                    Config.debug_log(f"[rich-itp-filtered-counts] {filt_counts}")
                except Exception:
                    pass
                for wmsg in warnings:
                    try:
                        Config.debug_log(f"[rich-itp-skip] {wmsg}")
                    except Exception:
                        pass
            except Exception:
                raise

        # dihedrals (avoid duplicates: if rich dihedrals will be emitted, skip World.Dihedrals)
        if not (emit_rich and extras.get("dihedrals")):
            f.write('\n[ dihedrals ]\n')
            for q in world.Dihedrals:
                for dihedral in world.Dihedrals[q]:
                    try:
                        funct = int(dihedral.dihedral_funct)
                    except Exception:
                        continue
                    if funct <= 0 or funct > 50:
                        continue
                    d_params = getattr(dihedral, "dihedral_params", None)
                    ids_str = '{} {} {} {} {}'.format(
                        dihedral.dihedral_atom_1.atom_id + 1, dihedral.dihedral_atom_2.atom_id + 1,
                        dihedral.dihedral_atom_3.atom_id + 1, dihedral.dihedral_atom_4.atom_id + 1,
                        funct)
                    if d_params:
                        f.write('{} {}\n'.format(ids_str, ' '.join(str(p) for p in d_params)))
                    else:
                        c2 = getattr(dihedral, "dihedral_c2", None)
                        if c2 in (None, ' ', ''):
                            f.write('{} {} {}\n'.format(ids_str, dihedral.dihedral_c0, dihedral.dihedral_c1))
                        else:
                            f.write('{} {} {} {}\n'.format(ids_str, dihedral.dihedral_c0, dihedral.dihedral_c1, c2))

            def _write_section(sec, rows):
                if not rows:
                    return
                f.write(f"\n[ {sec} ]\n")
                for r in rows:
                    if isinstance(r, dict) and "line" in r:
                        f.write(r["line"] + "\n")
                    elif isinstance(r, dict) and "values" in r:
                        f.write(" ".join(str(v) for v in r["values"]) + "\n")
                    elif isinstance(r, dict) and sec in ("constraints", "pairs"):
                        i = r.get("i"); j = r.get("j"); funct = r.get("funct", 1)
                        params = r.get("params", [])
                        if i is not None and j is not None:
                            # Convert 0-based memory index to GROMACS 1-based index
                            f.write(" ".join(str(v) for v in [int(i)+1, int(j)+1, funct, *params]) + "\n")
                        else:
                            f.write(str(r) + "\n")
                    elif isinstance(r, dict) and sec == "exclusions":
                        atom = r.get("atom"); excl = r.get("exclude", [])
                        if atom is not None:
                            # Convert 0-based memory index to GROMACS 1-based index
                            excl_1based = [int(x)+1 if isinstance(x, (int, float)) else x for x in excl]
                            f.write(" ".join(str(v) for v in [int(atom)+1, *excl_1based]) + "\n")
                        else:
                            f.write(str(r) + "\n")
                    elif isinstance(r, dict) and sec == "polarization":
                        atom = r.get("atom"); params = r.get("params", [])
                        if atom is not None:
                            # Convert 0-based memory index to GROMACS 1-based index
                            f.write(" ".join(str(v) for v in [int(atom)+1, *params]) + "\n")
                        else:
                            f.write(str(r) + "\n")
                    elif isinstance(r, dict) and "row" in r:
                        row = r["row"]
                        if isinstance(row, list):
                            f.write(" ".join(str(v) for v in row) + "\n")
                        else:
                            f.write(str(row) + "\n")
                    else:
                        f.write(str(r) + "\n")

            # virtual_sites는 실제 섹션명(virtual_sites2/3/4/n)으로 풀어쓰기
            vs_entries = extras.get("virtual_sites") or []
            if vs_entries:
                vs_by_sec = {}
                for vs in vs_entries:
                    sec = vs.get("section", "virtual_sites2")
                    vs_by_sec.setdefault(sec, []).append(vs)
                for sec, rows in vs_by_sec.items():
                    _write_section(sec, rows)

            for sec, rows in extras.items():
                if sec == "virtual_sites":
                    continue
                _write_section(sec, rows)

    print(f"[INFO] Combined ITP written to {filename}")
    return 1
