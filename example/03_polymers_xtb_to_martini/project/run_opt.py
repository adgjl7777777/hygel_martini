import os
from ase.io import read, write
from xtb.ase.calculator import XTB
from ase.optimize import BFGS
from ase.constraints import FixAtoms

# ================= 설정 부분 =================
basename= os.environ.get("dirname")
input_file = f"{basename}.xyz"        # 입력 파일 이름
output_file = f"{basename}_out.xyz"   # 결과 저장 파일 이름

# 고정할 원자 번호 (1부터 시작하는 번호 그대로 입력하세요)
# 예: 2번과 44번을 고정하고 싶을 때
fix_indices_1based = [2, 44] if basename != "S2" else [2,41]

# XTB 계산 설정 (GFN2, 물 용매 등)
xtb_method = "GFN2-xTB"
solvent = "water"
charge = 0
# ============================================

def main():
    print(f"Reading structure from {input_file}...")
    atoms = read(input_file)

    # 1. 계산기(Calculator) 설정
    # ASE가 설치된 환경에 'xtb' 명령어가 PATH에 잡혀 있어야 합니다.
    atoms.calc = XTB(method=xtb_method, solvent=solvent, charge=charge)

    # 2. 제약 조건(Constraint) 설정
    if fix_indices_1based:
        # 1-based index를 0-based index로 변환
        fix_indices_0based = [i - 1 for i in fix_indices_1based]
        
        # 범위 체크
        max_idx = len(atoms) - 1
        valid_indices = [i for i in fix_indices_0based if 0 <= i <= max_idx]
        
        if len(valid_indices) != len(fix_indices_0based):
            print("Warning: Some indices are out of range and will be ignored.")

        constraint = FixAtoms(indices=valid_indices)
        atoms.set_constraint(constraint)
        print(f"Constraints applied on atoms (1-based): {fix_indices_1based}")
    
    # 3. 형상 최적화 (Geometry Optimization) 수행
    # trajectory='opt.traj'를 지정하면 최적화 과정을 저장합니다.
    print("Starting BFGS optimization...")
    opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')
    
    # fmax: 힘이 0.05 eV/Angstrom 이하가 될 때까지 (Tight 옵션과 유사)
    opt.run(fmax=0.05)

    # 4. 결과 저장
    print(f"Optimization finished. Saving to {output_file}...")
    write(output_file, atoms)

if __name__ == "__main__":
    main()