import os
from pathlib import Path

import numpy as np
from ase.io import read, write

BASE_CONNECTOR = "Br"
CH_LENGTH = 1.094
CONNECTOR_CUTOFF = 2.2
DEFAULT_MONOMER_FILES = {
    "S": "NEW_SBMA.xyz",
    "D": "NEW_DMAPS.xyz",
    "C": "NEW_CBA.xyz",
}

def get_connection_info(atoms):
    """
    모노머에서 연결 정보를 추출합니다.
    return:
        c0_idx: Head Carbon index (0)
        c1_idx: Tail Carbon index (1)
        bc_head_idx: C0에 연결된 BASE_CONNECTOR 인덱스
        bc_tail_idx: C1에 연결된 BASE_CONNECTOR 인덱스
    """
    c0_idx = 0
    c1_idx = 1
    
    bc_head_idx = None
    bc_tail_idx = None
    
    # 거리 기반으로 연결된 BASE_CONNECTOR 찾기
    # 인덱스 순서가 섞이지 않도록 주의
    for i, atom in enumerate(atoms):
        if atom.symbol == BASE_CONNECTOR:
            d0 = atoms.get_distance(c0_idx, i)
            d1 = atoms.get_distance(c1_idx, i)
            
            if d0 <= CONNECTOR_CUTOFF:
                bc_head_idx = i
            elif d1 <= CONNECTOR_CUTOFF:
                bc_tail_idx = i
                
    if bc_head_idx is None or bc_tail_idx is None:
        raise ValueError(f"모노머 {atoms.info.get('name', '')}에서 연결용 {BASE_CONNECTOR}을 찾을 수 없습니다.")
        
    return c0_idx, c1_idx, bc_head_idx, bc_tail_idx

def cap_ends_with_hydrogen(atoms):
    """
    양 끝단에 남아있는 BASE_CONNECTOR을 H로 치환하고 길이를 1.094A로 조정합니다.
    """
    # BASE_CONNECTOR 원자 인덱스 찾기 (역순으로 처리하여 인덱스 밀림 방지)
    bc_indices = [atom.index for atom in atoms if atom.symbol == BASE_CONNECTOR]
    
    # BASE_CONNECTOR와 연결된 탄소 찾기
    for bc_idx in sorted(bc_indices, reverse=True):
        connected_c_idx = None
        nearest_dist = float("inf")
        for i, atom in enumerate(atoms):
            if i == bc_idx or atom.symbol == BASE_CONNECTOR:
                continue
            dist = atoms.get_distance(bc_idx, i)
            if atom.symbol == "C" and dist <= CONNECTOR_CUTOFF and dist < nearest_dist:
                connected_c_idx = i
                nearest_dist = dist
        
        if connected_c_idx is not None:
            # 벡터 계산 (C -> BASE_CONNECTOR)
            vec = atoms.positions[bc_idx] - atoms.positions[connected_c_idx]
            vec_norm = vec / np.linalg.norm(vec)
            
            # BASE_CONNECTOR을 H로 변경
            atoms[bc_idx].symbol = 'H'
            
            # 위치 조정 (C-H 길이 CH_LENGTH A)
            new_pos = atoms.positions[connected_c_idx] + vec_norm * CH_LENGTH
            atoms.positions[bc_idx] = new_pos
            
    return atoms

def normalize_sequence(sequence):
    """
    시퀀스 입력을 토큰 리스트로 정규화합니다.
    - ["S", "D", "B"] 형태를 권장
    - "S,D,B" / "S D B" 문자열도 허용
    - "SDB" 문자열은 한 글자 토큰으로 처리
    """
    if isinstance(sequence, str):
        seq = sequence.strip()
        if "," in seq:
            tokens = [tok.strip() for tok in seq.split(",") if tok.strip()]
        elif " " in seq:
            tokens = [tok for tok in seq.split() if tok]
        else:
            tokens = list(seq)
    else:
        tokens = [str(tok).strip() for tok in sequence if str(tok).strip()]

    if not tokens:
        raise ValueError("시퀀스가 비어 있습니다.")
    return tokens


def load_monomer_library(monomer_files, base_dir=None):
    """
    monomer_files: {"S": "NEW_SBMA.xyz", "D": "NEW_DMAPS.xyz"} 형태
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    else:
        base_dir = Path(base_dir)

    monomer_dict = {}
    for symbol, filepath in monomer_files.items():
        path_obj = Path(filepath)
        if not path_obj.is_absolute():
            path_obj = base_dir / path_obj
        if not path_obj.exists():
            raise FileNotFoundError(f"모노머 파일이 없습니다: {path_obj}")

        atoms = read(path_obj)
        atoms.info["name"] = symbol
        monomer_dict[symbol] = atoms

    return monomer_dict


def _sequence_output_stem(sequence_tokens):
    if all(len(tok) == 1 for tok in sequence_tokens):
        return "".join(sequence_tokens)
    return "_".join(sequence_tokens)


def build_polymer(sequence, monomer_dict, n_torsion, output_filename=None):
    """
    Args:
        sequence: ["S", "D", "D", "S", "B"] 또는 "S,D,D,S,B"
        monomer_dict: {"S": ase.Atoms(...), "D": ase.Atoms(...)} 형태
        n_torsion: 비틀림 각도를 결정하는 정수 (1, 2, 3, 4...)
        output_filename: 저장할 파일명
    """
    sequence_tokens = normalize_sequence(sequence)
    for symbol in sequence_tokens:
        if symbol not in monomer_dict:
            raise KeyError(f"시퀀스 심볼 '{symbol}'이 monomer_dict에 없습니다.")
    
    # 1. 기본 회전 단위 (Increment) 설정
    # n=4라면 90도입니다.
    angle_increment = max(90, 360 / n_torsion)
    
    # 첫 모노머는 0도 기준, 이후 붙을 때마다 각도가 더해집니다.
    current_angle = 0  

    if output_filename is None:
        output_filename = f"{_sequence_output_stem(sequence_tokens)}.xyz"

    print(f"Build info: Sequence={sequence_tokens}, n={n_torsion}, Angle Increment={angle_increment} deg")

    # 첫 모노머 초기화
    chain = monomer_dict[sequence_tokens[0]].copy()
    _, c1_curr, _, bc_tail_curr = get_connection_info(chain)
    
    current_tail_c_idx = c1_curr
    current_tail_bc_idx = bc_tail_curr

    # 루프 시작
    for symbol in sequence_tokens[1:]:
        new_monomer = monomer_dict[symbol].copy()
        c0_new, c1_new, bc_head_new, bc_tail_new = get_connection_info(new_monomer)
        
        # --- [Step A] 벡터 정렬 (Alignment) ---
        v_target = chain.positions[current_tail_bc_idx] - chain.positions[current_tail_c_idx]
        v_source = new_monomer.positions[c0_new] - new_monomer.positions[bc_head_new]
        new_monomer.rotate(v_source, v_target)
        
        # --- [Step B] 각도 누적 및 회전 (Twist) ---
        # 매 루프마다 각도를 증가시킴

        current_angle += angle_increment 
    
        # 축을 기준으로 회전
        new_monomer.rotate(current_angle, v_target, center=new_monomer.positions[c0_new])

        # --- [Step C] 이동 (Translation) ---
        target_pos = chain.positions[current_tail_c_idx] + (v_target / np.linalg.norm(v_target)) * 1.513
        translation_vec = target_pos - new_monomer.positions[c0_new]
        new_monomer.translate(translation_vec)
        
        # --- [Step D] 병합 및 인덱스 정리 ---
        del chain[current_tail_bc_idx]
        del new_monomer[bc_head_new]
        
        index_shift = len(chain)
        
        bc_tail_adjusted = bc_tail_new - 1 if bc_tail_new > bc_head_new else bc_tail_new
        c1_adjusted = c1_new - 1 if c1_new > bc_head_new else c1_new
        
        chain += new_monomer
        
        current_tail_c_idx = index_shift + c1_adjusted
        current_tail_bc_idx = index_shift + bc_tail_adjusted

    final_polymer = cap_ends_with_hydrogen(chain)
    os.makedirs("output",exist_ok=True)
    write(os.path.join("output",output_filename), final_polymer)
    print(f"Done! Saved to {output_filename}")

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    # 심볼(한 글자/두 글자 이상 모두 가능)과 xyz 파일 매핑
    monomer_files = dict(DEFAULT_MONOMER_FILES)
    # monomer_files["SB"] = "your_sb_monomer.xyz"  # 두 글자 이상 심볼도 가능
    library = load_monomer_library(monomer_files)

    # 원하는 시퀀스를 리스트로 입력
    sequences = [
        ["S", "D", "D", "S", "C"],
        ["C", "S", "S", "D"],
        ["S", "C", "D"],
    ]

    for seq in sequences:
        build_polymer(seq, monomer_dict=library, n_torsion=len(seq), output_filename=None)
    
