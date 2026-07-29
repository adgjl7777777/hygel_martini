"""Hydrogel network construction primitives.

This module still contains a large amount of legacy logic, but the core data
flow is now easier to follow:

1. build backbone and linker beads from a geometric blueprint,
2. add proto-level crosslinks,
3. expand backbone beads into detailed monomer side chains,
4. derive angles and dihedrals from template metadata and fallbacks.

The implementation remains stateful because newly created atoms and topology
records register themselves in ``World``.
"""

import numpy as np
import sys
from typing import List
from hygel_martini.hydrogel_builder.main_components import Attributes
from itertools import product as pd
from hygel_martini.hydrogel_builder.core_utils.common.utility import interp3D, dij_sq, rij, not_self, random_normal_vector
from hygel_martini.hydrogel_builder.core_utils.common.sequence_strategy import TemplateStrategyIterator, StrategyRecord
from hygel_martini.hydrogel_builder.core_utils.layout.template_placement import compute_template_positions, resolve_sidechain_placement_tuning
from hygel_martini.hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates
import random
from hygel_martini.hydrogel_builder.config_params import read_json as p
import itertools
from tqdm import tqdm


class Hydrogel():
    """Construct and enrich a hydrogel network stored in ``World``.

    The class is responsible for translating a geometric lattice plan into
    concrete atoms and topology terms. The implementation mixes newer
    template-driven logic with older fallback behavior, so detailed docstrings
    are used to make the stage boundaries explicit.
    """
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
        """Initialize lattice repetition counts and terminal registries.

        Args:
            x_number_of_repeat: Number of unit-cell repetitions along the
                x axis.
            y_number_of_repeat: Number of unit-cell repetitions along the
                y axis.
            z_number_of_repeat: Number of unit-cell repetitions along the
                z axis.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트

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
        """Generate the line segments used to populate one lattice cell.

        Args:
            bx: Unit-cell x index.
            by: Unit-cell y index.
            bz: Unit-cell z index.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: Interpolated backbone
            coordinates and linker coordinates for the selected cell.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        
        max_linker_span = getattr(World, 'max_linker_span', 0.0)
        # 주 사슬(segment)의 단량체(monomer) 개수를 계산합니다.
        # 이 계산은 다이아몬드 네트워크의 기하학적 구조와 World.mean_sep(평균 원자 간 거리)에 기반합니다.
        n_segment = np.floor(np.sqrt(np.square(World.ubox_length / 2 - 3 * World.mean_sep) + 2 * np.square(World.ubox_length / 2)) / World.mean_sep) - 1
        print("주 사슬 단량체 수 : ", n_segment - 1)
        n_segment = n_segment.astype(int)

        lines = [] # 원자 경로를 저장할 리스트
        link_axis = np.array([1.0, 0.0, 0.0], dtype=float) # 링크 축 (주로 x축 방향)

        if max_linker_span > 0 and World.mean_sep > 0:
            center_shift_factor = max_linker_span / (2 * World.mean_sep)
            link_extension_factor = max_linker_span / (2 * World.mean_sep)
            link_midpoint_factor = max_linker_span / (4 * World.mean_sep)
        else:
            center_shift_factor = 1.5
            link_extension_factor = 1.5
            link_midpoint_factor = 0.5

        center_shift = max(center_shift_factor * World.mean_sep, 1.5 * World.mean_sep)
        extension_shift = max(link_extension_factor * World.mean_sep, 1.5 * World.mean_sep)
        midpoint_shift = max(link_midpoint_factor * World.mean_sep, 0.5 * World.mean_sep)

        def axis_shift(box_point, magnitude):
            sign = -1.0 if box_point[0] > 0.5 else 1.0
            return sign * magnitude * link_axis

        # 다이아몬드 네트워크의 중심 비스(bis) 헤드 및 테일 위치를 정의합니다.
        # 이들은 단위 셀 내에서 가교제의 시작점과 끝점을 나타냅니다.
        CenterBisHead = np.array([World.ubox_length / 2 + center_shift, World.ubox_length / 2, World.ubox_length / 2])
        CenterBisTail = np.array([World.ubox_length / 2 - center_shift, World.ubox_length / 2, World.ubox_length / 2])
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
        lines.append([CenterBisTail, World.ubox_length * cubeType[0] + axis_shift(cubeType[0], extension_shift)])
        lines.append([CenterBisTail, World.ubox_length * cubeType[1] + axis_shift(cubeType[1], extension_shift)])
        lines.append([CenterBisHead, World.ubox_length * cubeType[2] + axis_shift(cubeType[2], extension_shift)])
        lines.append([CenterBisHead, World.ubox_length * cubeType[3] + axis_shift(cubeType[3], extension_shift)])

        # CenterBisHead와 CenterBisTail을 연결하는 라인을 추가합니다.
        lines.append([CenterBisHead, CenterBisTail])

        # 추가적인 링크 라인들을 생성합니다.
        if True: # 이 조건문은 항상 참이므로, 항상 실행됩니다.
            for boxPoint in cubeType:
                start_point = World.ubox_length * boxPoint + axis_shift(boxPoint, midpoint_shift)
                end_point = World.ubox_length * boxPoint + axis_shift(boxPoint, extension_shift)
                lines.append([start_point, end_point])

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
        """Materialize backbone and linker atoms from the geometric layout.

        The algorithm is template driven:

        1. load backbone definitions and choose a sequence strategy,
        2. walk each lattice cell and generate backbone beads,
        3. expand linker templates bead-by-bead,
        4. remap template-local constraints, exclusions, and related sections
           into global ``World`` indices.

        Fail-fast behavior is intentional because malformed backbone or linker
        template configuration usually produces much harder-to-debug errors in
        later GROMACS stages.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트
        from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import load_linker_templates
        
        # 주기 경계 조건(PBC)을 넘어가는 링커 원자들을 테스트하기 위한 리스트
        # --- Backbone 정의 및 생성기 준비 ---
        try:
            backbone_definitions = p.Config.get_param('hydrogel_components', 'backbone_definitions')
            backbone_list = backbone_definitions['BACKBONES']
            bb_strategy = backbone_definitions.get('SEQUENCE_STRATEGY', {'strategy': 'random'})

            if not backbone_list:
                raise ValueError("BACKBONES 리스트가 비어있습니다.")

            def get_backbone_generator_factory(strategy_info, backbones):
                from hygel_martini.hydrogel_builder.core_utils.common.sequence_strategy import TemplateStrategyIterator, StrategyRecord
                records = [StrategyRecord(template=b, ratio=b.get("ratio", 1.0), template_id=b.get("id")) for b in backbones]
                def create_generator():
                    iterator = TemplateStrategyIterator(records, strategy_info)
                    while True:
                        yield iterator.next()
                return create_generator

            backbone_generator_factory = get_backbone_generator_factory(bb_strategy, backbone_list)

            try:
                sim_params = p.Config.get_param("simulation_parameters")
                block_settings = p.resolve_block_copolymer_settings(sim_params)
            except KeyError:
                block_settings = {
                    "chain_orientation_policy": "random",
                    "linker_terminal_compensation": {},
                    "terminal_compensation_enabled": False,
                    "legacy_policy": "",
                    "warnings": [],
                }

            for warning in block_settings.get("warnings", []):
                print(warning)

            terminal_comp = block_settings.get("linker_terminal_compensation", {}) or {}
            terminal_enabled = block_settings.get("terminal_compensation_enabled", False)
            if terminal_enabled or block_settings.get("chain_orientation_policy") != "random":
                print("[INFO] block copolymer generic settings are active.")
                print(f"[INFO] chain_orientation_policy={block_settings.get('chain_orientation_policy')}")
                if terminal_enabled:
                    target = terminal_comp.get("target_backbone") or terminal_comp.get("target_monomer_type")
                    beads = terminal_comp.get("compensation_beads", 0)
                    print(f"[INFO] linker_terminal_compensation: target={target}, compensation_beads={beads}")
                print("[INFO] (한계점/구현 범위) 현재 버전에서는 다이아몬드 단위 격자의 기하 배치(CenterBis -> CubeType Vertex)에 따라 물리 좌표 순서대로 시퀀스가 빌드됩니다.")
                print("[INFO] 그래프 매칭 결과(사이클/경로)의 방향성을 완전하게 추적하여 체인별로 sequence를 자동 반전(reverse)하는 로직은 미구현 상태이며, 기하 구조 방향으로 일관되게 배치됩니다.")

        except (KeyError, ValueError) as e:
            print(f"오류: backbone_definitions 설정이 잘못되었습니다. {e}", file=sys.stderr)
            sys.exit(1)

        def _map_extra_sections(template_like, idx_map):
            """Map template-local optional topology sections into ``World``.

            Args:
                template_like: Linker/backbone template metadata dictionary.
                idx_map: Mapping from template-local atom indices to global
                    ``World`` atom identifiers.
            """
            from hygel_martini.hydrogel_builder.main_components.Universe import World
            def _add(sec, payload):
                World.OtherSections[sec].append(payload)
            def get_list(name):
                return template_like.get(name, []) if isinstance(template_like, dict) else []
            # constraints
            for c in get_list("constraints"):
                i = idx_map.get(c.get("i"))
                j = idx_map.get(c.get("j"))
                if i and j:
                    _add("constraints", {"i": i, "j": j, "funct": c.get("funct", 1), "params": c.get("params", [])})
            # pairs
            for p_def in get_list("pairs"):
                i = idx_map.get(p_def.get("i"))
                j = idx_map.get(p_def.get("j"))
                if i and j:
                    _add("pairs", {"i": i, "j": j, "funct": p_def.get("funct", 1), "params": p_def.get("params", [])})
            # exclusions
            for ex_def in get_list("exclusions"):
                atom_idx = idx_map.get(ex_def.get("atom"))
                if atom_idx:
                    mapped = [idx_map.get(x, x) for x in ex_def.get("exclude", [])]
                    _add("exclusions", {"atom": atom_idx, "exclude": mapped})
            # virtual sites
            for vs in get_list("virtual_sites"):
                sec = vs.get("section", "virtual_sites")
                parts = vs.get("parts", vs.get("line", "").split()) if isinstance(vs, dict) else []
                mapped_parts = []
                for token in parts:
                    try:
                        val = int(token)
                        mapped_parts.append(str(idx_map.get(val, val)))
                    except ValueError:
                        mapped_parts.append(token)
                line = " ".join(mapped_parts) if mapped_parts else vs.get("line", "")
                _add("virtual_sites", {"section": sec, "parts": mapped_parts, "line": line})
            # restraints/cmap/polarization
            for rst in get_list("restraints"):
                sec = rst.get("section", "restraints")
                vals = rst.get("values", [])
                mapped = []
                for v in vals:
                    if isinstance(v, int):
                        mapped.append(idx_map.get(v, v))
                    else:
                        mapped.append(v)
                _add(sec, {"values": mapped})
            for cm in get_list("cmaptypes"):
                _add("cmaptypes", {"row": cm})
            for pol in get_list("polarization"):
                atom_idx = idx_map.get(pol.get("atom"))
                if atom_idx:
                    _add("polarization", {"atom": atom_idx, "params": pol.get("params", [])})
            # impropers/dihedrals (원본 ITP 인덱스 기준, 매핑 가능한 경우만)
            for imp in (get_list("impropers_full") or get_list("impropers")):
                gi = idx_map.get(imp.get("i"))
                gj = idx_map.get(imp.get("j"))
                gk = idx_map.get(imp.get("k"))
                gl = idx_map.get(imp.get("l"))
                if gi and gj and gk and gl:
                    vals = [gi + 1, gj + 1, gk + 1, gl + 1, int(imp.get("funct", 2))]
                    vals.extend(imp.get("params", []))
                    _add("impropers", {"values": vals})
            if isinstance(template_like, dict):
                for sec, lines in template_like.get("other_sections", {}).items():
                    for ln in lines:
                        _add(sec, {"line": ln})

        # 정의된 반복 횟수(x, y, z)에 따라 단위 셀을 순회합니다.
        chain_global_idx = 0
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
                current_chain_idx = chain_global_idx
                chain_global_idx += 1
                chain_backbone_generator = backbone_generator_factory()
                for ii in i:
                    # 각 원자(모노머)에 대해 백본 타입을 랜덤하게 선택합니다.
                    chosen_backbone = next(chain_backbone_generator)
                    chosen_bb_def = chosen_backbone['definition']
                    chosen_bb_id = chosen_backbone['id']

                    _tmp = Attributes.Atom() # 새로운 원자 객체 생성
                    _tmp.source_template = chosen_backbone
                    _tmp.source_index = 0
                    # 선택된 backbone 타입의 속성을 사용합니다.
                    _tmp.atom_type = chosen_bb_def['atom_type']
                    _tmp.residue_number = chosen_bb_def.get('residue_number', 1)
                    
                    # Handle list residue_name for alternating chains
                    raw_res_name = chosen_bb_def['residue_name']
                    if isinstance(raw_res_name, list) and len(raw_res_name) > 0:
                        _tmp.residue_name = raw_res_name[current_chain_idx % len(raw_res_name)]
                    else:
                        _tmp.residue_name = raw_res_name
                        
                    _tmp.atom_name = chosen_bb_def['atom_name']
                    _tmp.cgnr = chosen_bb_def['charge_group_number']
                    _tmp.mass = chosen_bb_def['mass']
                    _tmp.charge = chosen_bb_def['charge']
                    _tmp.backbone_type = chosen_bb_id # <<< backbone 타입 ID 저장
                    _tmp.chain_index = current_chain_idx # Store chain index
                    
                    # 원자 위치 설정: 단위 셀 내 좌표 + 전체 박스 내 오프셋
                    _tmp.position = ii + World.ubox_length * np.array([bx, by, bz])

                    # 원자의 end_tag를 설정하고, 필요한 경우 결합을 생성합니다.
                    # (기존 로직과 동일)
                    if np.prod(ii == i[-1, :]): # 현재 원자가 세그먼트의 마지막 원자인 경우
                        _tmp.end_tag = 1 # end_tag 1로 설정
                        # 이전 원자와의 결합 생성
                        if _tmp.atom_id > 0:
                            _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                            _tmp2.bond_funct = chosen_bb_def.get('bond_funct', 1)
                            _tmp2.bond_c0 = chosen_bb_def.get('bond_c0', 0.25)
                            _tmp2.bond_c1 = chosen_bb_def.get('bond_c1', 10000)
                        self.terminals[_tmp.end_tag].append(_tmp) # 터미널 리스트에 추가
                    elif np.prod(ii == i[0, :]): # 현재 원자가 세그먼트의 첫 번째 원자인 경우
                        _tmp.end_tag = 1 # end_tag 1로 설정
                        self.terminals[_tmp.end_tag].append(_tmp) # 터미널 리스트에 추가
                    else: # 세그먼트의 중간 원자인 경우
                        # 이전 원자와의 결합 생성
                        if _tmp.atom_id > 0:
                            _tmp2 = Attributes.Bond(_tmp.atom_id - 1, _tmp.atom_id)
                            _tmp2.bond_funct = chosen_bb_def.get('bond_funct', 1)
                            _tmp2.bond_c0 = chosen_bb_def.get('bond_c0', 0.25)
                            _tmp2.bond_c1 = chosen_bb_def.get('bond_c1', 10000)

                    try:
                        # 백본 템플릿 부가 섹션 매핑 (인덱스는 1개 비드 기준)
                        idx_map = {1: _tmp.atom_id}
                        _map_extra_sections(chosen_bb_def, idx_map)
                    except Exception as exc:
                        print(f"[경고] 백본 부가 섹션 매핑 실패: {exc}")

            # 가교제(linker) 원자들을 생성합니다. (REFACTORED LOGIC)
            try:
                linker_entries = p.Config.get_param('linker_definitions', 'LINKERS')
                backbone_defs = p.Config.get_param('hydrogel_components', 'backbone_definitions', 'BACKBONES')
                linker_library = load_linker_templates(linker_entries, backbone_defs)
                
                linker_strategy_cfg = p.Config.get_param('linker_definitions', 'SEQUENCE_STRATEGY', default={'strategy': 'random'})

                def get_linker_generator(strategy_info, library):
                    if not library.records:
                        return None
                    strategy = strategy_info.get('strategy', 'random')
                    if strategy == 'random':
                        templates = [rec.template for rec in library.records]
                        ratios = [rec.ratio for rec in library.records]
                        def random_generator():
                            while True:
                                yield random.choices(templates, weights=ratios, k=1)[0]
                        return random_generator()
                    else: # cycle
                        templates = [rec.template for rec in library.records]
                        return itertools.cycle(templates)

                linker_generator = get_linker_generator(linker_strategy_cfg, linker_library)
                self._linker_library = linker_library
            except (KeyError, IndexError, ValueError) as e:
                print(f"경고: 'linker_definitions'를 찾을 수 없거나 설정이 잘못되어 링커를 생성하지 않습니다. 오류: {e}")
                linker_generator = None

            if linker_generator:
                for j in link_xyz: # j는 make_lines에서 온 링커 경로 좌표를 포함합니다.
                    
                    chosen_linker = next(linker_generator) # This is now a LinkerTemplate object
                    try:
                        from hygel_martini.hydrogel_builder.config_params.config import Config
                        Config.debug_log(
                            f"[choose-linker] id={chosen_linker.id} beads={len(chosen_linker.beads)} "
                            f"constraints={len(chosen_linker.constraints)} pairs={len(chosen_linker.pairs)} "
                            f"exclusions={len(chosen_linker.exclusions)} vs={len(chosen_linker.virtual_sites)} "
                            f"restraints={len(chosen_linker.restraints)} dih_full={len(getattr(chosen_linker,'dihedrals_full',[]))} "
                            f"imp_full={len(getattr(chosen_linker,'impropers_full',[]))}"
                        )
                    except Exception:
                        pass
                    
                    p_start = j[0] + World.ubox_length * np.array([bx, by, bz])
                    p_end = j[-1] + World.ubox_length * np.array([bx, by, bz])
                    
                    path_vector = rij(p_end, p_start, World.box_length)
                    path_length = np.linalg.norm(path_vector)
                    path_unit_vector = path_vector / path_length if path_length > 0 else np.array([1.0, 0, 0])
                    if path_length < 1e-6:
                        print("경고: 링커 경로 길이가 0에 가깝습니다. 해당 링커 배치를 건너뜁니다.")
                        continue

                    scale_factor = path_length / chosen_linker.span_length if chosen_linker.span_length > 0 else 1.0
                    
                    # 링커의 로컬 좌표를 월드 좌표로 변환하기 위한 회전 행렬 계산
                    basis_z = path_unit_vector
                    ref_vec = np.array([0.0, 0.0, 1.0])
                    if abs(np.dot(basis_z, ref_vec)) > 0.9:
                        ref_vec = np.array([0.0, 1.0, 0.0])
                    basis_x = np.cross(ref_vec, basis_z)
                    basis_x /= np.linalg.norm(basis_x)
                    basis_y = np.cross(basis_z, basis_x)
                    rotation = np.column_stack((basis_x, basis_y, basis_z))

                    linker_atoms = []
                    new_atom_ids = []
                    
                    # 링커 원자 생성 및 배치
                    linker_map_entries = []
                    
                    # Pre-calculate stub indices for quick lookup
                    left_stub_indices = {s[0] for s in chosen_linker.stub_bonds_left}
                    right_stub_indices = {s[0] for s in chosen_linker.stub_bonds_right}

                    for i, bead_template in enumerate(chosen_linker.beads):
                        raw_res_name = bead_template.residue_name
                        final_res_name = raw_res_name
                        
                        if isinstance(raw_res_name, list):
                            if i in left_stub_indices:
                                final_res_name = raw_res_name[0]
                            elif i in right_stub_indices:
                                final_res_name = raw_res_name[1] if len(raw_res_name) > 1 else raw_res_name[0]
                            else:
                                final_res_name = raw_res_name[0] # Default

                        atom = Attributes.Atom(
                            source_template=chosen_linker,
                            source_index=i,
                            source_residue_name=final_res_name
                        )
                        atom.atom_type = bead_template.atom_type
                        atom.residue_name = final_res_name
                        atom.atom_name = bead_template.name
                        atom.mass = bead_template.mass
                        atom.charge = bead_template.charge
                        atom.residue_number = bead_template.residue_number
                        atom.cgnr = bead_template.cgnr
                        
                        # 로컬 좌표를 월드 좌표로 변환
                        local_coord_scaled = bead_template.coord * scale_factor
                        atom.position = p_start + rotation @ local_coord_scaled

                        linker_atoms.append(atom)
                        new_atom_ids.append(atom.atom_id)
                        linker_map_entries.append({
                            "template_id": chosen_linker.id,
                            "original_index": getattr(bead_template, "original_index", i + 1),
                            "source_index": i,
                            "global_atom_id": atom.atom_id
                        })

                    # 링커 내부 결합 생성
                    for idx_i, idx_j, params in chosen_linker.internal_bonds:
                        Attributes.Bond(new_atom_ids[idx_i], new_atom_ids[idx_j], **params)

                    # 외부 연결(stub)을 위한 터미널 설정
                    # 이 부분은 이제 동적 cross-linking 로직에서 처리되므로,
                    # 여기서는 터미널 원자에 필요한 정보만 저장합니다.
                    for stub_idx, params in chosen_linker.stub_bonds_left:
                        atom = linker_atoms[stub_idx]
                        atom.end_tag = 2 # Backbone에 연결될 터미널
                        atom.target_bb = params['target']
                        atom.target_backbone = params['target']
                        atom.stub_from_bead = True # 이 원자가 stub임을 표시
                        atom.stub_bond_params = params
                        self.terminals[2].append(atom)

                    for stub_idx, params in chosen_linker.stub_bonds_right:
                        atom = linker_atoms[stub_idx]
                        atom.end_tag = 2 # Backbone에 연결될 터미널
                        atom.target_bb = params['target']
                        atom.target_backbone = params['target']
                        atom.stub_from_bead = True # 이 원자가 stub임을 표시
                        atom.stub_bond_params = params
                        self.terminals[2].append(atom)
                    # 부가 섹션을 World에 반영
                    try:
                        from hygel_martini.hydrogel_builder.main_components.Universe import World
                        idx_map = {e["original_index"]: e["global_atom_id"] for e in linker_map_entries}
                        def _add(sec, payload):
                            World.OtherSections[sec].append(payload)
                        for c in chosen_linker.constraints:
                            i = idx_map.get(c.get("i"))
                            j = idx_map.get(c.get("j"))
                            if i and j:
                                _add("constraints", {"i": i, "j": j, "funct": c.get("funct", 1), "params": c.get("params", [])})
                        for p_def in chosen_linker.pairs:
                            i = idx_map.get(p_def.get("i"))
                            j = idx_map.get(p_def.get("j"))
                            if i and j:
                                _add("pairs", {"i": i, "j": j, "funct": p_def.get("funct", 1), "params": p_def.get("params", [])})
                        for ex_def in chosen_linker.exclusions:
                            atom_idx = idx_map.get(ex_def.get("atom"))
                            if atom_idx:
                                mapped = [idx_map.get(x, x) for x in ex_def.get("exclude", [])]
                                _add("exclusions", {"atom": atom_idx, "exclude": mapped})
                        for vs in chosen_linker.virtual_sites:
                            sec = vs.get("section", "virtual_sites")
                            parts = vs.get("parts", vs.get("line", "").split())
                            mapped_parts = []
                            for token in parts:
                                try:
                                    val = int(token)
                                    mapped_parts.append(str(idx_map.get(val, val)))
                                except ValueError:
                                    mapped_parts.append(token)
                            _add("virtual_sites", {"section": sec, "parts": mapped_parts, "line": " ".join(mapped_parts)})
                        for rst in chosen_linker.restraints:
                            sec = rst.get("section", "restraints")
                            vals = rst.get("values", [])
                            mapped = []
                            for v in vals:
                                if isinstance(v, int):
                                    mapped.append(idx_map.get(v, v))
                                else:
                                    mapped.append(v)
                            _add(sec, {"values": mapped})
                        for cm in chosen_linker.cmaptypes:
                            _add("cmaptypes", {"row": cm})
                        for pol in chosen_linker.polarization:
                            atom_idx = idx_map.get(pol.get("atom"))
                            if atom_idx:
                                _add("polarization", {"atom": atom_idx, "params": pol.get("params", [])})
                        # 링커 impropers는 현재 World.Dihedrals에 생성되지 않으므로 OtherSections로 전달
                        for imp in (getattr(chosen_linker, "internal_impropers", []) +
                                    getattr(chosen_linker, "impropers_full", [])):
                            gi = idx_map.get(imp.get("i"))
                            gj = idx_map.get(imp.get("j"))
                            gk = idx_map.get(imp.get("k"))
                            gl = idx_map.get(imp.get("l"))
                            if gi and gj and gk and gl:
                                vals = [gi + 1, gj + 1, gk + 1, gl + 1, int(imp.get("funct", 2))]
                                vals.extend(imp.get("params", []))
                                _add("impropers", {"values": vals})
                            else:
                                try:
                                    from hygel_martini.hydrogel_builder.config_params.config import Config
                                    Config.debug_log(f"[linker-rich-skip] improper refs missing local idx (likely stub): {imp}")
                                except Exception:
                                    pass
                        for sec, lines in chosen_linker.other_sections.items():
                            sec_lower = str(sec).lower()
                            if sec_lower in (
                                "constraints","pairs","exclusions","dihedrals","impropers",
                                "cmaptypes","polarization"
                            ) or sec_lower.startswith("virtual_sites") or sec_lower.endswith("_restraints") or "restraint" in sec_lower:
                                continue
                            for ln in lines:
                                _add(sec_lower, {"line": ln})
                        try:
                            from hygel_martini.hydrogel_builder.config_params.config import Config
                            Config.debug_log(
                                f"[linker-rich] {chosen_linker.id}: constraints={len(chosen_linker.constraints)}, "
                                f"pairs={len(chosen_linker.pairs)}, exclusions={len(chosen_linker.exclusions)}, "
                                f"virtual_sites={len(chosen_linker.virtual_sites)}, restraints={len(chosen_linker.restraints)}, "
                                f"impropers_full={len(getattr(chosen_linker,'impropers_full',[]))}"
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        print(f"[경고] 링커 부가 섹션 매핑 실패: {exc}")

    def _construct_proto_bonds(self, output_dir):
        """Attach linker stubs to the nearest allowed backbone terminals.

        Args:
            output_dir: Output directory kept for interface compatibility with
                older callers. The current implementation only uses runtime
                state already stored in ``World``.

        Returns:
            bool: ``True`` when the proto-bond pass finishes without a fatal
            configuration error.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            Config.debug_log("[stage] _construct_proto_bonds start")
        except Exception:
            pass
        box_len = World.box_length if World.box_length else float(np.max(World.box_vector))
        if box_len <= 0:
            box_len = float(np.max(World.box_vector)) if np.any(World.box_vector) else 100.0

        stub_atoms = [atom for atom in self.terminals[2] if hasattr(atom, 'stub_from_bead')]
        if not stub_atoms:
            return True

        backbone_atoms = self.terminals[1]
        if not backbone_atoms:
            print("[경고] proto 연결에 사용할 backbone 터미널이 없습니다.")
            return True

        axis_index = {"x": 0, "y": 1, "z": 2}
        used_backbones = set()
        success = 0

        grouped = {}
        for stub in stub_atoms:
            chain_id = getattr(stub, 'linker_chain_index', None)
            grouped.setdefault(chain_id, []).append(stub)

        for chain_id, stubs in grouped.items():
            if not stubs:
                continue
            local_idx = getattr(stubs[0], 'linker_local_index', None)
            axis = getattr(stubs[0], 'linker_axis', None) or "x"
            axis_idx = axis_index.get(str(axis).lower(), 0)

            def _pos(atom):
                return getattr(atom, 'pre_compress_position', atom.position)

            def _pick_backbone(stub, avoid_ids=None):
                target_type = getattr(stub, 'target_bb', None)
                best_atom = None
                best_dist = None
                for candidate in backbone_atoms:
                    if avoid_ids and candidate.atom_id in avoid_ids:
                        continue
                    if target_type and getattr(candidate, 'backbone_type', None) != target_type:
                        continue
                    dist = dij_sq(_pos(stub), _pos(candidate), box_len)
                    if best_atom is None or dist < best_dist:
                        best_atom = candidate
                        best_dist = dist
                return best_atom

            if local_idx == 0:
                target_stub = max(stubs, key=lambda s: _pos(s)[axis_idx])
                best_atom = _pick_backbone(target_stub)
                if best_atom is None:
                    continue
                params = getattr(target_stub, 'stub_bond_params', {})
                target_stub.position = best_atom.position.copy()
                Attributes.Bond(target_stub.atom_id, best_atom.atom_id,
                                funct=params.get('funct', 1),
                                c0=params.get('length', getattr(target_stub, 'external_bond_length', World.mean_sep)),
                                c1=params.get('fc', 56000.0))
                used_backbones.add(best_atom.atom_id)
                success += 1
            else:
                used_local = set()
                for stub in stubs:
                    best_atom = _pick_backbone(stub, avoid_ids=used_local)
                    if best_atom is None:
                        continue
                    params = getattr(stub, 'stub_bond_params', {})
                    stub.position = best_atom.position.copy()
                    Attributes.Bond(stub.atom_id, best_atom.atom_id,
                                    funct=params.get('funct', 1),
                                    c0=params.get('length', getattr(stub, 'external_bond_length', World.mean_sep)),
                                    c1=params.get('fc', 56000.0))
                    used_local.add(best_atom.atom_id)
                    used_backbones.add(best_atom.atom_id)
                    success += 1

        self.num_Bonds_12 += success
        print(f"[INFO] Proto bonds 생성: {success}개")
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            Config.debug_log(f"[stage] _construct_proto_bonds done: {success}")
        except Exception:
            pass
        return True

    def construct_bonds(self, pbc, num_cell, output_dir):
        """Compatibility wrapper for proto-bond construction.

        Args:
            pbc: Unused legacy flag kept for call-site compatibility.
            num_cell: Unused legacy cell count kept for compatibility.
            output_dir: Output directory associated with the current run.
        """
        return self._construct_proto_bonds(output_dir)

    def construct_chemical_detail(self):
        """Expand backbone beads into detailed monomer side chains.

        The placement procedure is heuristic but systematic:

        1. load monomer templates grouped by backbone type,
        2. for each backbone bead, determine a local tangent direction from
           neighboring backbone bonds,
        3. sample candidate normal vectors around the backbone,
        4. build trial coordinates for the chosen monomer template,
        5. score candidate placements using overlap rejection plus a soft
           inverse-distance penalty,
        6. commit the best valid placement and remap extra topology sections.

        Runtime tuning knobs such as the number of candidate vectors and the
        overlap threshold are now exposed through
        ``simulation_parameters.sidechain_*`` settings.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World
        print(f"World.box_length in construct_chemical_detail: {World.box_length}")
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            Config.debug_log("[stage] construct_chemical_detail enter")
        except Exception:
            pass

        # Side-chain placement is intentionally parameterized because the search
        # quality/performance tradeoff changes dramatically with system size.
        sim_params = {}
        try:
            sim_params = p.Config.get_param('simulation_parameters') or {}
        except Exception:
            sim_params = {}

        tuning = resolve_sidechain_placement_tuning(sim_params, len(World.Atoms))
        NUM_CANDIDATE_VECTORS = int(tuning["num_candidate_vectors"])
        OVERLAP_THRESHOLD_FACTOR = float(tuning["overlap_threshold_factor"])
        SEARCH_RADIUS_FACTOR = float(tuning["search_radius_factor"])
        NEARBY_ATOM_LIMIT = int(tuning["nearby_atom_limit"])

        # Prepare one sequence iterator per backbone chemistry so mixed
        # hydrogels can attach different monomers on different backbones.
        try:
            monomer_config = p.Config.get_param('monomer_definitions')
            backbone_defs = p.Config.get_param('hydrogel_components', 'backbone_definitions', 'BACKBONES')
            template_library = load_monomer_templates(monomer_config['MONOMERS'], backbone_defs)
            strategy_cfg = monomer_config.get('SEQUENCE_STRATEGY', {})

            backbone_ids = {bb['id'] for bb in backbone_defs}
            sequence_generators = {}
            for bb_id in backbone_ids:
                records = template_library.by_backbone.get(bb_id, [])
                strategy_records = [
                    StrategyRecord(
                        template=rec.template,
                        ratio=rec.ratio,
                        template_id=getattr(rec.template, 'id', None)
                    )
                    for rec in records
                ]
                sequence_generators[bb_id] = TemplateStrategyIterator(strategy_records, strategy_cfg)
        except (AttributeError, KeyError, ValueError) as e:
            print(f"오류: Monomer 또는 Backbone 설정 형식이 잘못되었습니다. {e}", file=sys.stderr)
            return

        # Grow side chains bead-by-bead while avoiding immediate clashes with
        # nearby atoms already present in the world state.
        _World_Atoms_keys = [*World.Atoms]
        monomer_counts = {rec.template.id: 0 for rec in template_library.records}
        monomer_global_maps = []
        
        # 효율적인 겹침 계산을 위해 모든 원자 객체 리스트를 미리 만듭니다.
        all_atoms = [World.Atoms[k][0] for k in World.Atoms]
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            progress = Config.get_runtime("progress_tracker")
        except Exception:
            progress = None
        total_keys = len(_World_Atoms_keys)
        report_interval = max(1, total_keys // 100) if total_keys else 1
        processed_keys = 0

        for _id in tqdm(_World_Atoms_keys):
            processed_keys += 1
            if progress and (processed_keys % report_interval == 0 or processed_keys == total_keys):
                progress.stage_tick(processed_keys / float(total_keys), "construct_chemical_detail")
            backbone_atom = World.Atoms[_id][0]
            if not hasattr(backbone_atom, 'backbone_type'):
                continue # Backbone 원자가 아니면 건너뜁니다.
            
            current_bb_type = backbone_atom.backbone_type
            iterator = sequence_generators.get(current_bb_type)

            if not iterator:
                continue # 이 backbone에 해당하는 monomer가 없습니다.

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
            best_positions = None
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
            nearby_distances = []
            for atom in all_atoms:
                if atom.atom_id != backbone_atom.atom_id and dij_sq(backbone_atom.position, atom.position, World.box_length) < search_radius_sq:
                    if atom.atom_id not in bonded_atom_ids:
                        nearby_atoms.append(atom)
                        nearby_distances.append(
                            dij_sq(backbone_atom.position, atom.position, World.box_length)
                        )
            if NEARBY_ATOM_LIMIT > 0 and len(nearby_atoms) > NEARBY_ATOM_LIMIT:
                idxs = np.argpartition(np.array(nearby_distances), NEARBY_ATOM_LIMIT)[:NEARBY_ATOM_LIMIT]
                nearby_atoms = [nearby_atoms[i] for i in idxs]

            chosen_template = iterator.next()
            if chosen_template is None:
                continue
            tangent_vec = rij(p1, p3, World.box_length)
            if np.linalg.norm(tangent_vec) < 1e-8:
                tangent_vec = np.array([0.0, 0.0, 1.0])

            for _ in range(NUM_CANDIDATE_VECTORS):
                candidate_vector = random_normal_vector(p1, p2, p3, 1.0, World.box_length)
                positions = compute_template_positions(
                    chosen_template.coords,
                    backbone_atom.position,
                    candidate_vector,
                    tangent_vec,
                )
                if positions is None:
                    continue

                current_penalty = 0.0
                is_valid = True
                
                # 가상 위치와 주변 원자들 간의 겹침 및 페널티 계산
                for pos in positions:
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
                    best_positions = positions

            # --- 곁사슬 실제 생성 ---
            if best_vector is not None and best_positions is not None:
                # 통계 기록
                monomer_counts[chosen_template.id] += 1

                new_atom_ids: List[int] = []
                for i, (bead_def, pos) in enumerate(zip(chosen_template.beads, best_positions)):
                    side_atom = Attributes.Atom(
                        source_template=chosen_template,
                        source_index=i,
                        source_residue_name=bead_def.residue_name
                    )
                    side_atom.atom_type = bead_def.atom_type
                    side_atom.residue_name = bead_def.residue_name
                    side_atom.atom_name = bead_def.name
                    side_atom.mass = bead_def.mass
                    side_atom.charge = bead_def.charge
                    side_atom.residue_number = bead_def.residue_number
                    side_atom.cgnr = bead_def.cgnr
                    side_atom.position = pos
                    all_atoms.append(side_atom)
                    new_atom_ids.append(side_atom.atom_id)
                    monomer_global_maps.append({
                        "template_id": chosen_template.id,
                        "original_index": getattr(bead_def, "original_index", i + 1),
                        "source_index": i,
                        "global_atom_id": side_atom.atom_id
                    })

                for idx_i, idx_j, params in chosen_template.internal_bonds:
                    Attributes.Bond(new_atom_ids[idx_i], new_atom_ids[idx_j], **params)
                for bead_idx, params in chosen_template.backbone_bonds:
                    Attributes.Bond(backbone_atom.atom_id, new_atom_ids[bead_idx], **params)
            else:
                print(f"경고: 원자 ID {backbone_atom.atom_id}에 곁사슬({chosen_template.id})을 배치할 유효한 공간을 찾지 못했습니다. 건너뜁니다.")

        print("곁사슬 생성 완료:")
        for name, count in monomer_counts.items():
            print(f"- {name}: {count}개")
        try:
            from hygel_martini.hydrogel_builder.config_params.config import Config
            Config.debug_log("[stage] construct_chemical_detail exit")
        except Exception:
            pass
        self.num_HDG_atoms = len(World.Atoms) # 최종적으로 생성된 총 원자 수를 업데이트합니다.
        # --- 곁사슬 템플릿 섹션을 World에 반영 ---
        try:
            template_lookup = template_library.lookup
            mapped_by_template = {}
            for entry in monomer_global_maps:
                mapped_by_template.setdefault(entry["template_id"], []).append(entry)

            def _add_other(section, payload):
                from hygel_martini.hydrogel_builder.main_components.Universe import World as W
                W.OtherSections[section].append(payload)

            for temp_id, entries in mapped_by_template.items():
                template = template_lookup.get(temp_id)
                if not template:
                    continue
                idx_map = {e["original_index"]: e["global_atom_id"] for e in entries}
                # constraints
                for c in template.constraints:
                    i = idx_map.get(c.get("i"))
                    j = idx_map.get(c.get("j"))
                    if i and j:
                        _add_other("constraints", {"i": i, "j": j, "funct": c.get("funct", 1), "params": c.get("params", [])})
                # pairs
                for p_def in template.pairs:
                    i = idx_map.get(p_def.get("i"))
                    j = idx_map.get(p_def.get("j"))
                    if i and j:
                        _add_other("pairs", {"i": i, "j": j, "funct": p_def.get("funct", 1), "params": p_def.get("params", [])})
                # exclusions
                for ex_def in template.exclusions:
                    atom = idx_map.get(ex_def.get("atom"))
                    if atom:
                        mapped = [idx_map.get(x, x) for x in ex_def.get("exclude", [])]
                        _add_other("exclusions", {"atom": atom, "exclude": mapped})
                # virtual sites
                for vs in template.virtual_sites:
                    parts = vs.get("parts", [])
                    section = vs.get("section", "virtual_sites")
                    mapped_parts = []
                    for token in parts:
                        try:
                            val = int(token)
                            mapped_parts.append(str(idx_map.get(val, val)))
                        except ValueError:
                            mapped_parts.append(token)
                    _add_other("virtual_sites", {"section": section, "parts": mapped_parts, "line": " ".join(mapped_parts)})
                # restraints (raw)
                for rst in template.restraints:
                    section = rst.get("section", "restraints")
                    values = rst.get("values", [])
                    mapped_vals = []
                    for v in values:
                        if isinstance(v, int):
                            mapped_vals.append(idx_map.get(v, v))
                        else:
                            mapped_vals.append(v)
                    _add_other(section, {"values": mapped_vals})
                # cmaptypes / polarization
                for cm in template.cmaptypes:
                    _add_other("cmaptypes", {"row": cm})
                for pol in template.polarization:
                    atom = idx_map.get(pol.get("atom"))
                    if atom:
                        _add_other("polarization", {"atom": atom, "params": pol.get("params", [])})
                # 기타
                for sec, lines in template.other_sections.items():
                    sec_lower = str(sec).lower()
                    if sec_lower in (
                        "constraints","pairs","exclusions","dihedrals","impropers",
                        "cmaptypes","polarization"
                    ) or sec_lower.startswith("virtual_sites") or sec_lower.endswith("_restraints") or "restraint" in sec_lower:
                        continue
                    for ln in lines:
                        _add_other(sec_lower, {"line": ln})
        except Exception as exc:
            print(f"[경고] 곁사슬 추가 섹션 매핑 중 오류: {exc}")

    def construct_angles(self):
        """Generate angle terms from template metadata and fallback rules.

        The priority order is:

        1. internal template angles explicitly defined in the source ITP,
        2. coarse structural heuristics based on backbone/sidechain role,
        3. configured default angle parameters.
        """
        from hygel_martini.hydrogel_builder.main_components.Universe import World

        angle_configs = {}
        try:
            angle_configs = p.Config.get_param('hydrogel_components', 'backbone_definitions', 'angles')
        except Exception:
            angle_configs = {}
        if angle_configs and 'default_angle' in angle_configs:
            default_params = angle_configs['default_angle']
        else:
            try:
                default_params = p.Config.get_param('simulation_parameters', 'default_angle')
            except Exception:
                default_params = {'angle_funct': 1, 'angle_c0': 180.0, 'angle_c1': 25.0}

        _atom = World.Atoms
        
        # 모든 결합 정보 수집
        bonds_by_atom = {i: [] for i in _atom.keys()}
        for bond_obj in World.Bonds.values():
            bond = bond_obj[0]
            bonds_by_atom[bond.bond_atom_1.atom_id].append(bond.bond_atom_2.atom_id)
            bonds_by_atom[bond.bond_atom_2.atom_id].append(bond.bond_atom_1.atom_id)

        # 각도 생성
        for cen_atom_id, neighbors in bonds_by_atom.items():
            if len(neighbors) < 2:
                continue

            cen_atom = _atom[cen_atom_id][0]

            # 이웃 원자들의 모든 쌍에 대해 각도 생성
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    side_atom1_id = neighbors[i]
                    side_atom2_id = neighbors[j]
                    
                    side_atom1 = _atom[side_atom1_id][0]
                    side_atom2 = _atom[side_atom2_id][0]

                    # --- 1. 내부 각도 (ITP) 우선 적용 ---
                    is_internal = (
                        cen_atom.source_template is not None and
                        cen_atom.source_template is side_atom1.source_template and
                        cen_atom.source_template is side_atom2.source_template
                    )

                    if is_internal:
                        template = cen_atom.source_template
                        s_idx_cen = cen_atom.source_index
                        s_idx_1 = side_atom1.source_index
                        s_idx_2 = side_atom2.source_index
                        
                        found_internal_angle = False
                        for angle_def in template.internal_angles:
                            # Check both permutations (i-j-k and k-j-i)
                            if (angle_def['center'] == s_idx_cen and
                                ((angle_def['from'] == s_idx_1 and angle_def['to'] == s_idx_2) or
                                 (angle_def['from'] == s_idx_2 and angle_def['to'] == s_idx_1))):
                                
                                angle = Attributes.Angle(side_atom1_id, cen_atom_id, side_atom2_id)
                                angle.angle_funct = angle_def['funct']
                                angle.angle_c0 = angle_def['params'][0]
                                angle.angle_c1 = angle_def['params'][1]
                                found_internal_angle = True
                                break
                        
                        if found_internal_angle:
                            continue # 내부 각도를 찾았으면 다음 조합으로

                    # --- 2. 분자 간 각도 규칙 적용 ---
                    angle = Attributes.Angle(side_atom1_id, cen_atom_id, side_atom2_id)
                    
                    is_cen_backbone = hasattr(cen_atom, 'backbone_type')
                    is_side1_backbone = hasattr(side_atom1, 'backbone_type')
                    is_side2_backbone = hasattr(side_atom2, 'backbone_type')

                    # 유형 1: Backbone-Backbone-Backbone
                    if is_cen_backbone and is_side1_backbone and is_side2_backbone:
                        angle.angle_funct = 1
                        angle.angle_c0 = 180.0
                        angle.angle_c1 = 5.0
                    
                    # 유형 2: Backbone-Backbone-Sidechain
                    elif is_cen_backbone and (is_side1_backbone != is_side2_backbone):
                        angle.angle_funct = 1
                        angle.angle_c0 = 90.0
                        angle.angle_c1 = 25.0

                    # 유형 3: 곁사슬 내부 -> Backbone-Sidechain-Sidechain
                    elif is_side1_backbone and not is_cen_backbone and not is_side2_backbone:
                        angle.angle_funct = 1
                        angle.angle_c0 = 120.0
                        angle.angle_c1 = 20.0
                    
                    # --- 3. 기본값(Fallback) 적용 ---
                    else:
                        angle.angle_funct = default_params['angle_funct']
                        angle.angle_c0 = default_params['angle_c0']
                        angle.angle_c1 = default_params['angle_c1']

        self.num_HDG_angles = len(World.Angles)
        print(f"총 {self.num_HDG_angles}개의 각도가 생성되었습니다.")

    def construct_dihedrals(self):
        """Generate internal dihedrals directly from template definitions."""
        from hygel_martini.hydrogel_builder.main_components.Universe import World
        
        processed_templates = set()
        all_atoms_list = [a[0] for a in World.Atoms.values()]

        for atom in all_atoms_list:
            if atom.source_template is None or id(atom.source_template) in processed_templates:
                continue
            
            template = atom.source_template
            processed_templates.add(id(template))

            # 템플릿에 속한 실제 원자들을 찾기 위한 맵
            template_atoms = [a for a in all_atoms_list if a.source_template is template]
            source_idx_to_atom_id = {a.source_index: a.atom_id for a in template_atoms}

            for dihedral_def in template.internal_dihedrals:
                try:
                    atom_i_id = source_idx_to_atom_id[dihedral_def['i']]
                    atom_j_id = source_idx_to_atom_id[dihedral_def['j']]
                    atom_k_id = source_idx_to_atom_id[dihedral_def['k']]
                    atom_l_id = source_idx_to_atom_id[dihedral_def['l']]
                    
                    # GROMACS는 c0를 init의 인자로 받지 않으므로, 생성 후 설정
                    dihedral = Attributes.Dihedral(atom_i_id, atom_j_id, atom_k_id, atom_l_id, 0)
                    dihedral.dihedral_funct = dihedral_def['funct']
                    dihedral.dihedral_c0 = dihedral_def['params'][0]
                    dihedral.dihedral_c1 = dihedral_def['params'][1]
                    if len(dihedral_def['params']) > 2:
                        dihedral.dihedral_c2 = dihedral_def['params'][2]

                except (KeyError, IndexError):
                    # ITP에 정의가 없거나 잘못된 경우, 요청대로 그냥 넘어갑니다.
                    continue
        
        self.num_HDG_dihedrals = len(World.Dihedrals)
        print(f"총 {self.num_HDG_dihedrals}개의 내부 이면각이 생성되었습니다.")

    def construct_impropers(self):
        """Placeholder for explicit improper handling.

        Improper-like records are currently carried through the shared
        dihedral machinery and auxiliary topology sections. A dedicated
        ``Improper`` object could replace this in a future cleanup pass.
        """
        # 현재 Dihedral 클래스를 공유하므로 로직은 동일합니다.
        # 만약 Improper 클래스가 별도로 생긴다면 여기에 구현합니다.
        pass
