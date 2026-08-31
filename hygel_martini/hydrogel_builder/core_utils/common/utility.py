"""Numerical helpers, geometry utilities, and text-normalization helpers."""

from collections import defaultdict
import heapq
import os
import shutil
import subprocess

import numba
import numpy as np

# Numba JIT(Just-In-Time) 컴파일러를 사용하여 함수를 최적화합니다.
# fastmath=True: 엄격한 IEEE 754 규칙을 완화하여 속도를 높입니다.
# cache=True: 컴파일 결과를 캐시에 저장하여 다음 실행 시 빠르게 로드합니다.
# nogil=True: Python의 GIL(Global Interpreter Lock)을 해제하여 병렬 처리를 가능하게 합니다.
def interp3D(n, A, B):
    '''
      두 3D 점 A와 B 사이에 n개의 점을 등간격으로 보간하여 생성합니다.
      반환되는 점들은 A와 B 사이의 선분을 n+1개의 구간으로 나눈 점들입니다.

      Args:
          n (int): 생성할 점의 개수
          A (np.array): 시작점 좌표 [x, y, z]
          B (np.array): 끝점 좌표 [x, y, z]

      Returns:
          np.array: n개의 보간된 점들을 담은 (n, 3) 크기의 배열
    '''
    return np.array([A + i*(B-A)/(n+1) for i in range(1, n+1)])


@numba.jit(fastmath=True, cache=True, nogil=True)
def rij(position_i, position_j, L):
    '''
    주기 경계 조건(Periodic Boundary Conditions, PBC)을 고려하여
    원자 i에서 원자 j로 향하는 벡터(r_ij)를 계산합니다.
    가장 가까운 이미지(minimum image convention)를 사용합니다.

    Args:
        position_i (np.array): 원자 i의 좌표
        position_j (np.array): 원자 j의 좌표
        L (float): 시뮬레이션 박스의 한 변 길이

    Returns:
        np.array: PBC를 고려한 i에서 j로의 벡터
    '''
    # NOTE: L is a single scalar, so this is the CUBIC minimum image only.
    # It stays hand-rolled because numba compiles it in the inner overlap loop;
    # anything that can take a general cell must use hygel_martini.core.pbc.
    r_ij = np.zeros(3)
    for t in range(3): # x, y, z 각 축에 대해 계산
        sij = (position_j[t] - position_i[t]) / L
        # 벡터를 박스 길이로 나눈 후, 가장 가까운 정수로 반올림한 값을 빼서 최소 이미지를 찾습니다.
        r_ij[t] = (sij - np.round(sij)) * L

    return r_ij


@numba.jit(fastmath=True, cache=True, nogil=True)
def dij_sq(position_i, position_j, L):
    '''
    PBC를 고려하여 두 원자 i와 j 사이의 거리의 제곱(d_ij^2)을 계산합니다.
    제곱근 계산을 피하여 연산 속도를 높입니다.

    Args:
        position_i (np.array): 원자 i의 좌표
        position_j (np.array): 원자 j의 좌표
        L (float): 시뮬레이션 박스의 한 변 길이

    Returns:
        float: PBC를 고려한 두 원자 사이의 거리의 제곱
    '''
    d_ij_sq = 0.0
    for t in range(3):
        sij = (position_j[t] - position_i[t]) / L
        d_ij_sq += ((sij - np.round(sij)) * L) ** 2

    return d_ij_sq


@numba.jit(fastmath=True, cache=True, nogil=True)
def normal_to_3vectors(position_i, position_j, position_k, L):
    '''
    세 점 i, j, k가 이루는 평면의 법선 벡터를 계산합니다.

    Args:
        position_i (np.array): 기준점 i의 좌표
        position_j (np.array): 점 j의 좌표
        position_k (np.array): 점 k의 좌표
        L (float): 시뮬레이션 박스 길이

    Returns:
        np.array: 평면의 단위 법선 벡터
    '''
    # 기준점 i로부터 두 벡터 r_ij와 r_ik를 계산합니다.
    r1 = rij(position_i, position_j, L)
    r2 = rij(position_i, position_k, L)

    # 두 벡터의 외적(cross product)을 통해 평면에 수직인 벡터를 구합니다.
    cross = np.cross(r1, r2)

    # 벡터의 크기를 1로 만들어 단위 벡터로 반환합니다.
    normal_cross = cross / np.sqrt(np.sum(np.square(cross)))

    return normal_cross


@numba.jit(fastmath=True, cache=True, nogil=True)
def normal_tetrahedral_vector(position_1, position_2, position_3, position_4, L):
    '''
    중심 원자(1)와 세 개의 이웃 원자(2, 3, 4)가 주어졌을 때,
    사면체(tetrahedral) 구조에서 네 번째 결합이 향해야 할 방향 벡터를 계산합니다.

    Args:
        position_1 (np.array): 중심 원자의 좌표
        position_2,3,4 (np.array): 이웃 원자들의 좌표
        L (float): 시뮬레이션 박스 길이

    Returns:
        np.array: 정사면체의 중심에서 꼭짓점으로 향하는 단위 벡터
    '''
    # 중심에서 각 이웃으로 향하는 세 벡터를 계산합니다.
    r12 = rij(position_1, position_2, L)
    r13 = rij(position_1, position_3, L)
    r14 = rij(position_1, position_4, L)
    
    # 세 벡터의 합의 반대 방향이 네 번째 꼭짓점을 향하는 벡터가 됩니다.
    r_tetra = -(r12 + r13 + r14)
    # 단위 벡터로 만들어 반환합니다.
    r_tetra /= np.sqrt(np.sum(np.square(r_tetra)))

    return r_tetra


def not_self(i, obj):
    '''
    결합(bond) 객체(obj)와 그 결합에 속한 원자(i) 하나를 입력받아,
    그 결합에 속한 다른 원자를 반환하는 헬퍼 함수입니다.

    Args:
        i (Atom): 원자 객체
        obj (Bond): 결합 객체

    Returns:
        Atom: 결합의 상대방 원자 객체
    '''
    if i is obj.bond_atom_1:
        return obj.bond_atom_2
    else:
        return obj.bond_atom_1


@numba.jit(fastmath=True, cache=True, nogil=True)
def is_overlap(A, B, d, L):
    '''
    점 A가 점들의 배열 B에 있는 어떤 점과 거리 d 미만으로 겹치는지 확인합니다.

    Args:
        A (np.array): 확인할 점의 좌표
        B (np.array): 다른 점들의 좌표 배열 (N, 3)
        d (float): 겹침을 판단할 기준 거리
        L (float): 시뮬레이션 박스 길이

    Returns:
        bool: 겹치면 True, 겹치지 않으면 False
    '''
    d_sq = d * d  # 기준 거리의 제곱
    n = B.shape[0] # 점들의 개수
    d_sq_min = 10000000.0 # 최소 거리 제곱을 큰 값으로 초기화
    
    # 모든 점들과의 거리를 확인합니다.
    for i in range(n):
        d_sq_curr = dij_sq(A, B[i, :], L) # 현재 점과의 거리 제곱 계산
        if d_sq_curr < d_sq_min:
            d_sq_min = d_sq_curr
            
    # 계산된 최소 거리가 기준 거리보다 작으면 겹치는 것으로 판단합니다.
    if d_sq_min < d_sq:
        return True  # 겹침
    else:
        return False # 겹치지 않음


@numba.jit(fastmath=True, cache=True, nogil=True)
def random_normal_vector(A, B, C, r, L):
    '''
    A-B-C로 연결된 구조에서 중심 원자 B에 대해,
    두 결합(A-B, C-B)이 이루는 평면에 거의 수직인 방향으로
    길이가 r인 무작위 벡터를 생성합니다.
    곁사슬(side chain)을 생성할 때 사용됩니다.

    Args:
        A, B, C (np.array): 원자 A, B(중심), C의 좌표
        r (float): 생성할 벡터의 길이
        L (float): 시뮬레이션 박스 길이

    Returns:
        np.array: 무작위 방향 벡터
    '''
    # A->B 벡터와 C->B 벡터를 계산합니다.
    r1 = rij(A, B, L)
    r2 = rij(B, C, L)
    
    # 두 벡터의 평균 방향에 약간의 무작위성을 더합니다.
    direction_vector = (r1 + r2) / 2
    norm = np.linalg.norm(direction_vector)
    if norm > 1e-9:
        direction_vector /= norm
    direction_vector += (np.random.random(3) - 0.5) * 0.02

    # direction_vector에 수직인 임의의 벡터를 찾습니다.
    # (v . x = 0 방정식을 푸는 과정)
    x1, y1 = 2 * (np.random.random(2) - 0.5)
    x1 /= direction_vector[0]
    y1 /= direction_vector[1]
    z1 = -(direction_vector[0] * x1 + direction_vector[1] * y1) / (direction_vector[2] + 1e-9)
    
    # 찾은 벡터의 길이를 r로 맞추어 반환합니다.
    norm = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    x1 = x1 / norm * r
    y1 = y1 / norm * r
    z1 = z1 / norm * r
    return np.array([x1, y1, z1])


def find_minimum_distances(positions, box_length, top_n=10, cell_size=None):
    """
    Return the smallest inter-particle distances using a simple cell list search.

    Args:
        positions (np.ndarray): (N, 3) 좌표 배열.
        box_length (float): 시뮬레이션 박스 길이 (정육면체 가정).
        top_n (int): 리포트할 최소 거리 쌍의 개수.
        cell_size (float, optional): 셀 리스트의 크기. 기본은 box_length/10.

    Returns:
        list[tuple]: (거리[nm], index_i, index_j) 튜플 리스트.
    """
    if positions is None:
        return []
    positions = np.asarray(positions, dtype=np.float64)
    n_atoms = positions.shape[0]
    if n_atoms < 2:
        return []

    if box_length is None or box_length <= 0:
        span = np.max(positions, axis=0) - np.min(positions, axis=0)
        box_length = float(np.max(span) + 1e-6)

    if cell_size is None:
        cell_size = max(box_length / 10.0, 0.5)

    cell_count = max(1, int(np.ceil(box_length / cell_size)))
    wrapped = np.mod(positions, box_length)
    cell_indices = np.floor(wrapped / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, cell_count - 1)

    cells = defaultdict(list)
    for idx, cell_idx in enumerate(cell_indices):
        cells[tuple(cell_idx)].append(idx)

    neighbor_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if (dx > 0) or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    neighbor_offsets.append((dx, dy, dz))

    heap = []

    def consider_pair(i, j):
        if i == j:
            return
        if i > j:
            i, j = j, i
        d_sq = dij_sq(positions[i], positions[j], box_length)
        if len(heap) < top_n:
            heapq.heappush(heap, (-d_sq, i, j))
        else:
            if d_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-d_sq, i, j))

    try:
        from hygel_martini.hydrogel_builder.config_params.config import Config
        progress = Config.get_runtime("progress_tracker")
    except Exception:
        progress = None
    total_cells = len(cells)
    report_interval = max(1, total_cells // 100) if total_cells else 1
    processed_cells = 0

    for cell_key, atom_indices in cells.items():
        n_local = len(atom_indices)
        for i in range(n_local):
            for j in range(i + 1, n_local):
                consider_pair(atom_indices[i], atom_indices[j])

        for dx, dy, dz in neighbor_offsets:
            neighbor_key = (
                (cell_key[0] + dx) % cell_count,
                (cell_key[1] + dy) % cell_count,
                (cell_key[2] + dz) % cell_count,
            )
            neighbor_atoms = cells.get(neighbor_key)
            if not neighbor_atoms:
                continue
            for i_idx in atom_indices:
                for j_idx in neighbor_atoms:
                    consider_pair(i_idx, j_idx)
        processed_cells += 1
        if progress and (processed_cells % report_interval == 0 or processed_cells == total_cells):
            progress.stage_tick(processed_cells / float(total_cells), "min_distance")

    results = []
    while heap:
        neg_d_sq, i, j = heapq.heappop(heap)
        distance = float(np.sqrt(-neg_d_sq))
        results.append((distance, i, j))

    results.sort(key=lambda item: item[0])
    return results

def run_dos2unix_on_inputs(config_data):
    """
    Normalize line endings for all configured input structure files.

    Args:
        config_data (dict): Fully merged configuration dictionary.
    """
    files_to_process = []

    # 1. additional_itp_files
    if 'additional_itp_files' in config_data:
        files_to_process.extend(config_data['additional_itp_files'])

    # 2. Monomer definitions
    if 'monomer_definitions' in config_data and 'MONOMERS' in config_data['monomer_definitions']:
        for monomer in config_data['monomer_definitions']['MONOMERS']:
            if 'gro' in monomer:
                files_to_process.append(monomer['gro'])
            if 'itp' in monomer:
                files_to_process.append(monomer['itp'])

    # 3. Add series parameters
    if 'add_series_parameters' in config_data:
        add_series = config_data['add_series_parameters']
        if 'add_polymer' in add_series and 'monomer_definitions' in add_series['add_polymer']:
            monomer_defs = add_series['add_polymer']['monomer_definitions']
            if 'MONOMERS' in monomer_defs:
                for monomer in monomer_defs['MONOMERS']:
                    if 'gro' in monomer:
                        files_to_process.append(monomer['gro'])
                    if 'itp' in monomer:
                        files_to_process.append(monomer['itp'])
            if 'TERMINALS' in monomer_defs:
                for terminal in monomer_defs['TERMINALS']:
                    if 'gro' in terminal:
                        files_to_process.append(terminal['gro'])
                    if 'itp' in terminal:
                        files_to_process.append(terminal['itp'])
        if 'add_molecule' in add_series:
            add_mol = add_series['add_molecule']
            if 'molecule_gro' in add_mol:
                files_to_process.append(add_mol['molecule_gro'])
            if 'molecule_itp' in add_mol:
                files_to_process.append(add_mol['molecule_itp'])

    # 4. Linker definitions
    linker_defs = (
        config_data.get('linker_definitions', {})
        or config_data.get('hydrogel_components', {}).get('linker_definitions', {})
    )
    if 'LINKERS' in linker_defs:
        for linker in linker_defs['LINKERS']:
            if 'gro' in linker:
                files_to_process.append(linker['gro'])
            if 'itp' in linker:
                files_to_process.append(linker['itp'])

    # 5. Additional ion ITP files
    additional_ion_itps = (
        config_data.get('add_series_parameters', {})
        .get('add_small_ion', {})
        .get('additional_ion_itp_files', [])
    )
    files_to_process.extend(additional_ion_itps)

    # Exclude files from gromacs_include_path
    gromacs_include_path = config_data.get('simulation_parameters', {}).get('gromacs_include_path', '')
    if gromacs_include_path:
        files_to_process = [f for f in files_to_process if not f.startswith(gromacs_include_path)]

    dos2unix_path = shutil.which("dos2unix")
    warned_about_fallback = False

    # Run dos2unix on all collected files
    for file_path in set(files_to_process):
        if not os.path.exists(file_path):
            print(f"Warning: File not found, skipping dos2unix: {file_path}")
            continue
        if dos2unix_path:
            try:
                print(f"Running dos2unix on: {file_path}")
                subprocess.run([dos2unix_path, file_path], check=True, capture_output=True, text=True)
                continue
            except subprocess.CalledProcessError as exc:
                print(f"Error running dos2unix on {file_path}: {exc.stderr}")
                continue

        if not warned_about_fallback:
            print("dos2unix is unavailable; falling back to an in-Python CRLF normalizer.")
            warned_about_fallback = True

        with open(file_path, 'rb') as src:
            original = src.read()
        normalized = original.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        if normalized != original:
            with open(file_path, 'wb') as dst:
                dst.write(normalized)
