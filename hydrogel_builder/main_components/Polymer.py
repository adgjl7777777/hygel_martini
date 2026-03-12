"""Standalone polymer construction logic.

This module supports both a legacy single-backbone polymer path and a newer
template-driven path that can mix backbone chemistries, side chains, and
terminal groups. The newer path is the one aligned with the current cleanup.
"""

import numpy as np
import random
from random import Random
from collections import deque
from hydrogel_builder.main_components import Attributes
from hydrogel_builder.core_utils.common.utility import interp3D, dij_sq, normal_tetrahedral_vector, not_self, is_overlap, random_normal_vector, rij
from hydrogel_builder.config_params import read_json as p
from hydrogel_builder.core_utils.common.sequence_strategy import TemplateStrategyIterator, StrategyRecord
from hydrogel_builder.core_utils.layout.template_placement import build_alignment_basis, compute_template_positions, place_template_coords
from hydrogel_builder.core_utils.templates.monomer_loader import load_monomer_templates


def _build_polymer_bond_lookup(bond_rules, fallback_length):
    """Create a fast lookup table for backbone-to-backbone bond parameters."""
    lookup = {}
    if not bond_rules:
        return lookup
    for rule in bond_rules:
        between = rule.get('between', [])
        if len(between) != 2:
            continue
        key = tuple(sorted(between))
        lookup[key] = {
            'bond_funct': rule.get('bond_funct', rule.get('funct', 1)),
            'bond_c0': rule.get('bond_c0', rule.get('length', fallback_length)),
            'bond_c1': rule.get('bond_c1', rule.get('fc', 56000.0))
        }
    return lookup

class Polymer():
    """Construct a polymer chain and register it into ``World``.

    The class exposes a template-first workflow but still retains a legacy
    fallback for historical input sets that only define a single generic
    backbone bead type.
    """
    # 클래스 변수: 생성된 고분자 사슬의 총 원자, 결합, 각도, 이면각 수를 추적합니다.
    # 이 값들은 시뮬레이션 전체의 통계 및 검증에 사용될 수 있습니다.
    num_PLM_atoms = 0
    num_PLM_bonds = 0
    num_PLM_angles = 0
    num_PLM_dihedrals = 0
    _polymer_config = None
    _backbone_iterator: TemplateStrategyIterator | None = None
    _backbone_defs = []
    _backbone_lookup = {}
    _bond_lookup = {}
    _sidechain_library = None
    _sidechain_iterators = {}
    _sidechain_strategy_cfg = {}
    _terminal_records = []
    _terminal_strategy = {}
    _terminal_random = random.Random()

    def __init__(self, p_mon_num, p_length):
        """Store polymer dimensions and initialize the working box size.

        Args:
            p_mon_num: Number of backbone monomers.
            p_length: End-to-end length used for the initial straight-chain
                coordinate interpolation.
        """
        from hydrogel_builder.main_components.Universe import World

        self.p_length = p_length # 고분자 사슬의 정의된 전체 길이
        self.p_mon_num = p_mon_num # 고분자 사슬을 구성하는 단량체의 수
        self._backbone_atom_ids: list[int] = []

        # 시뮬레이션 박스 길이 설정:
        # 고분자 사슬의 길이를 기반으로 전체 시뮬레이션 박스의 한 변 길이를 설정합니다.
        # 일반적으로 고분자 길이가 박스 크기에 영향을 미치므로, 여기서는 고분자 길이의 2배로 설정하여
        # 고분자가 박스 내에 충분히 포함될 수 있도록 합니다.
        # 주석 처리된 'if not World.box_length:' 부분은 World.box_length가 한 번만 설정되도록
        # 의도되었을 수 있으나, 현재는 매 초기화마다 덮어쓰고 있습니다.
        World.box_length = self.p_length * 2 # 고분자 길이의 두 배로 시뮬레이션 박스 길이 설정

    @classmethod
    def configure(cls, config: dict | None):
        """Cache template libraries and strategy iterators for polymer builds."""
        cls._polymer_config = config or {}
        if not config:
            cls._backbone_iterator = None
            cls._backbone_defs = []
            cls._backbone_lookup = {}
            cls._bond_lookup = {}
            cls._sidechain_library = None
            cls._sidechain_iterators = {}
            cls._terminal_records = []
            cls._terminal_strategy = {}
            cls._terminal_random = random.Random()
            return

        backbone_defs = config.get('BACKBONES', [])
        cls._backbone_defs = backbone_defs
        cls._backbone_lookup = {bb.get('id'): bb for bb in backbone_defs if bb.get('id')}
        backbone_records = [
            StrategyRecord(
                template=bb,
                ratio=bb.get('ratio', 1.0),
                template_id=bb.get('id')
            )
            for bb in backbone_defs if bb.get('id')
        ]
        backbone_strategy_cfg = config.get('BACKBONE_SEQUENCE_STRATEGY',
                                           config.get('SEQUENCE_STRATEGY', {}))
        cls._backbone_iterator = TemplateStrategyIterator(backbone_records, backbone_strategy_cfg) if backbone_records else None

        fallback_length = config.get('default_bond_length', 0.24)
        cls._bond_lookup = _build_polymer_bond_lookup(config.get('BONDS'), fallback_length)

        # Side-chain templates
        cls._sidechain_library = None
        cls._sidechain_iterators = {}
        monomer_entries = config.get('MONOMERS', [])
        if monomer_entries:
            try:
                cls._sidechain_library = load_monomer_templates(monomer_entries, backbone_defs)
                monomer_strategy_cfg = config.get('MONOMER_SEQUENCE_STRATEGY',
                                                  config.get('SEQUENCE_STRATEGY', {}))
                for bb_id, records in cls._sidechain_library.by_backbone.items():
                    strat_records = [
                        StrategyRecord(
                            template=rec.template,
                            ratio=rec.ratio,
                            template_id=getattr(rec.template, 'id', None)
                        )
                        for rec in records
                    ]
                    cls._sidechain_iterators[bb_id] = TemplateStrategyIterator(strat_records, monomer_strategy_cfg)
            except Exception as exc:
                print(f"[경고] Polymer 곁사슬 템플릿 로딩 실패: {exc}")
                cls._sidechain_library = None
                cls._sidechain_iterators = {}

        # Terminal templates
        cls._terminal_records = []
        cls._terminal_strategy = config.get('TERMINAL_STRATEGY', {'strategy': 'random'})
        cls._terminal_random = random.Random(cls._terminal_strategy.get('seed'))
        terminal_entries = config.get('TERMINALS', [])
        if terminal_entries:
            try:
                terminal_library = load_monomer_templates(terminal_entries, backbone_defs)
                cls._terminal_records = [
                    StrategyRecord(
                        template=rec.template,
                        ratio=rec.ratio,
                        template_id=getattr(rec.template, 'id', None)
                    )
                    for rec in terminal_library.records
                ]
            except Exception as exc:
                print(f"[경고] Polymer terminal 템플릿 로딩 실패: {exc}")
                cls._terminal_records = []

    def make_lines(self, random_seed):
        """Generate a reproducible straight-chain backbone path.

        Args:
            random_seed: Seed used to choose the chain center and orientation.

        Returns:
            np.ndarray: Interpolated backbone coordinates.
        """
        from hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.

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
        """Dispatch to the template-driven or legacy atom-construction path."""
        from hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.

        if self._backbone_iterator is not None and self._backbone_defs:
            return self._construct_atoms_from_templates(random_seed)
        return self._legacy_construct_atoms(random_seed)

    def _legacy_construct_atoms(self, random_seed):
        """Construct a polymer using the historical single-backbone settings."""
        from hydrogel_builder.main_components.Universe import World
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
        return

    def _next_backbone_definition(self):
        """Return the next backbone template according to the configured strategy."""
        template = self._backbone_iterator.next() if self._backbone_iterator else None
        if template:
            return template
        if self._backbone_defs:
            return self._backbone_defs[0]
        return {}

    def _construct_atoms_from_templates(self, random_seed):
        """Construct backbone beads from template metadata.

        Each interpolated backbone position receives a template-selected bead.
        Consecutive bead pairs are then connected using chemistry-specific bond
        parameters when available, falling back to the configured default
        backbone bond otherwise.
        """
        from hydrogel_builder.main_components.Universe import World
        coords = self.make_lines(random_seed)
        prev_atom_id = None
        prev_backbone_id = None
        self._backbone_atom_ids = []

        for idx, position in enumerate(coords):
            template = self._next_backbone_definition()
            definition = template.get('definition', template)
            backbone_id = template.get('id') or definition.get('id') or f"POLY_{idx}"

            atom = Attributes.Atom()
            atom.source_template = template
            atom.source_index = idx
            atom.atom_type = definition.get('atom_type', 'C1')
            atom.residue_name = definition.get('residue_name', 'POL')
            atom.residue_number = definition.get('residue_number', idx + 1)
            atom.atom_name = definition.get('atom_name', f"PB{idx}")
            atom.cgnr = definition.get('charge_group_number', definition.get('cgnr', idx + 1))
            atom.mass = definition.get('mass', 56.0)
            atom.charge = definition.get('charge', 0.0)
            atom.position = position
            atom.backbone_type = backbone_id
            if idx == 0 or idx == len(coords) - 1:
                atom.end_tag = 1

            if prev_atom_id is not None:
                bond_key = tuple(sorted((prev_backbone_id, backbone_id)))
                params = self._bond_lookup.get(bond_key, {})
                bond = Attributes.Bond(prev_atom_id, atom.atom_id)
                bond.bond_funct = int(params.get('bond_funct', params.get('funct', 1)))
                bond.bond_c0 = float(params.get('bond_c0', params.get('length', World.mean_sep)))
                bond.bond_c1 = float(params.get('bond_c1', params.get('fc', 56000.0)))

            prev_atom_id = atom.atom_id
            prev_backbone_id = backbone_id
            self._backbone_atom_ids.append(atom.atom_id)
            # TODO: collect backbone template sections when template carries them

        self.num_PLM_atoms = len(World.Atoms)
        self.num_PLM_bonds = len(World.Bonds)
        self._attach_terminals(World)

    def _select_terminal_templates(self):
        """Choose left and right terminal templates according to strategy."""
        if not self._terminal_records:
            return None, None
        strategy = (self._terminal_strategy.get('strategy') or 'random').lower()

        def pick(exclude_id=None):
            candidates = self._terminal_records
            if exclude_id is not None:
                filtered = [rec for rec in candidates if rec.template_id != exclude_id]
                if filtered:
                    candidates = filtered
            weights = [max(float(rec.ratio), 0.0) or 1.0 for rec in candidates]
            return self._terminal_random.choices(candidates, weights=weights, k=1)[0]

        left_record = pick()
        right_record = pick(left_record.template_id) if strategy == 'semi_random' else pick()
        return left_record.template if left_record else None, right_record.template if right_record else None

    def _alignment_basis(self, axis):
        """Build an orthonormal basis whose x-axis follows ``axis``."""
        return build_alignment_basis(axis)

    def _place_template(self, template, origin, axis_vector):
        """Rotate and translate template coordinates onto an axis-aligned frame."""
        return place_template_coords(template.coords, origin, axis_vector)

    def _compute_template_positions(self, template, origin, normal_vector, tangent_vector):
        """Build a local side-chain frame from normal and tangent vectors."""
        return compute_template_positions(template.coords, origin, normal_vector, tangent_vector)

    def _create_template_atoms(self, template, positions, residue_override=None):
        """Instantiate atoms for a placed template and return their IDs."""
        atom_ids = []
        for bead, pos in zip(template.beads, positions):
            new_atom = Attributes.Atom()
            new_atom.atom_type = bead.atom_type
            new_atom.residue_name = residue_override or bead.residue_name
            new_atom.residue_number = bead.residue_number
            new_atom.atom_name = bead.name
            new_atom.cgnr = bead.cgnr
            new_atom.mass = bead.mass
            new_atom.charge = bead.charge
            new_atom.position = pos
            atom_ids.append(new_atom.atom_id)
        return atom_ids

    def _connect_template_bonds(self, template, created_atom_ids, backbone_atom_id):
        """Transfer template-local topology terms to global polymer indices."""
        from hydrogel_builder.main_components.Universe import World
        idx_map = {i: created_atom_ids[i] for i in range(len(created_atom_ids))}
        # original_index -> global atom id (backbone 포함)
        orig_to_global = {}
        for i, bead in enumerate(getattr(template, "beads", [])):
            orig_idx = getattr(bead, "original_index", None)
            if orig_idx is not None and i < len(created_atom_ids):
                orig_to_global[orig_idx] = created_atom_ids[i]
        bck_orig = getattr(template, "backbone_original_index", None)
        if isinstance(bck_orig, int):
            orig_to_global[bck_orig] = backbone_atom_id

        template_graph = {}

        def _add_template_edge(i, j):
            if not isinstance(i, int) or not isinstance(j, int):
                return
            template_graph.setdefault(i, set()).add(j)
            template_graph.setdefault(j, set()).add(i)

        for idx_i, idx_j, _params in getattr(template, "internal_bonds", []):
            if idx_i < len(template.beads) and idx_j < len(template.beads):
                orig_i = getattr(template.beads[idx_i], "original_index", None)
                orig_j = getattr(template.beads[idx_j], "original_index", None)
                _add_template_edge(orig_i, orig_j)
        for bead_idx, _params in getattr(template, "backbone_bonds", []):
            if bead_idx < len(template.beads):
                other_orig = getattr(template.beads[bead_idx], "original_index", None)
                _add_template_edge(bck_orig, other_orig)
        for c in getattr(template, "constraints", []):
            _add_template_edge(c.get("i"), c.get("j"))

        def _template_path_length(start_idx, end_idx):
            if start_idx == end_idx:
                return 0
            seen = {start_idx}
            queue = deque([(start_idx, 0)])
            while queue:
                node, depth = queue.popleft()
                for nxt in template_graph.get(node, ()):
                    if nxt == end_idx:
                        return depth + 1
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
            return None

        def _add_other(sec, payload):
            World.OtherSections[sec].append(payload)
        for idx_i, idx_j, params in template.internal_bonds:
            if idx_i < len(created_atom_ids) and idx_j < len(created_atom_ids):
                bond = Attributes.Bond(created_atom_ids[idx_i], created_atom_ids[idx_j])
                bond.bond_funct = int(params.get('funct', 1))
                bond.bond_c0 = float(params.get('c0', World.mean_sep))
                bond.bond_c1 = float(params.get('c1', 56000.0))

        for bead_idx, params in template.backbone_bonds:
            if bead_idx >= len(created_atom_ids):
                continue
            bond = Attributes.Bond(backbone_atom_id, created_atom_ids[bead_idx])
            bond.bond_funct = int(params.get('funct', 1))
            bond.bond_c0 = float(params.get('c0', World.mean_sep))
            bond.bond_c1 = float(params.get('c1', 56000.0))

        # 추가 섹션 매핑 (constraints/pairs/exclusions/vsites/restraints/cmap/polarization + dihedrals/impropers)
        try:
            for c in getattr(template, "constraints", []):
                i = orig_to_global.get(c.get("i"))
                j = orig_to_global.get(c.get("j"))
                if i is not None and j is not None:
                    con = Attributes.Constraint(i, j)
                    con.constraint_funct = int(c.get("funct", 1))
                    params = c.get("params", [])
                    if params:
                        con.constraint_c0 = float(params[0])
            # pairs는 현재 단계에서 World로 옮기지 않음(사용자 요청)
            for p_def in getattr(template, "pairs", []):
                _add_other("pairs", p_def)
            for ex_def in getattr(template, "exclusions", []):
                atom_orig = ex_def.get("atom")
                atom_idx = orig_to_global.get(atom_orig)
                if atom_idx is not None:
                    for x in ex_def.get("exclude", []):
                        gx = orig_to_global.get(x)
                        if gx is None:
                            continue
                        # Test-only 1-4 exclusions from side-chain templates can
                        # stretch beyond the cutoff once the branch is attached to
                        # a polymer backbone bead.
                        path_len = _template_path_length(atom_orig, x)
                        if bck_orig in (atom_orig, x) and path_len is not None and path_len >= 3:
                            continue
                        Attributes.Exclusion(atom_idx, gx)
            for vs in getattr(template, "virtual_sites", []):
                sec = vs.get("section", "virtual_sites")
                parts = vs.get("parts", vs.get("line", "").split())
                mapped_parts = []
                for token in parts:
                    try:
                        val = int(token)
                        gx = orig_to_global.get(val)
                        mapped_parts.append(str(gx + 1) if gx is not None else str(val))
                    except ValueError:
                        mapped_parts.append(token)
                _add_other("virtual_sites", {"section": sec, "parts": mapped_parts, "line": " ".join(mapped_parts)})
            for rst in getattr(template, "restraints", []):
                sec = rst.get("section", "restraints")
                vals = rst.get("values", [])
                mapped = []
                for v in vals:
                    if isinstance(v, int):
                        gx = orig_to_global.get(v)
                        mapped.append((gx + 1) if gx is not None else v)
                    else:
                        mapped.append(v)
                _add_other(sec, {"values": mapped})
            for cm in getattr(template, "cmaptypes", []):
                _add_other("cmaptypes", {"row": cm})
            for pol in getattr(template, "polarization", []):
                _add_other("polarization", pol)
            # full dihedrals/impropers (원본 ITP 인덱스 기준)
            for dih in getattr(template, "dihedrals_full", []):
                gi = orig_to_global.get(dih.get("i"))
                gj = orig_to_global.get(dih.get("j"))
                gk = orig_to_global.get(dih.get("k"))
                gl = orig_to_global.get(dih.get("l"))
                if gi and gj and gk and gl:
                    dih_obj = Attributes.Dihedral(gi, gj, gk, gl, 0)
                    dih_obj.dihedral_funct = int(dih.get("funct", 1))
                    params = dih.get("params", [])
                    if dih_obj.dihedral_funct == 1 and len(params) == 2:
                        params = list(params) + [1.0]
                    if dih_obj.dihedral_funct == 1 and len(params) < 3:
                        raise ValueError(f"Proper dihedral params 부족: {dih}")
                    if dih_obj.dihedral_funct != 1 and len(params) < 2:
                        raise ValueError(f"Dihedral params 부족: {dih}")
                    if len(params) > 0:
                        dih_obj.dihedral_c0 = float(params[0])
                    if len(params) > 1:
                        dih_obj.dihedral_c1 = float(params[1])
                    if len(params) > 2:
                        dih_obj.dihedral_c2 = float(params[2])
                else:
                    try:
                        from hydrogel_builder.config_params.config import Config
                        Config.debug_log(f"[polymer-rich-skip] dihedral refs missing local idx: {dih}")
                    except Exception:
                        pass
            for imp in getattr(template, "impropers_full", []):
                gi = orig_to_global.get(imp.get("i"))
                gj = orig_to_global.get(imp.get("j"))
                gk = orig_to_global.get(imp.get("k"))
                gl = orig_to_global.get(imp.get("l"))
                if gi and gj and gk and gl:
                    imp_obj = Attributes.Dihedral(gi, gj, gk, gl, 0)
                    imp_obj.dihedral_funct = int(imp.get("funct", 2))
                    params = imp.get("params", [])
                    if imp_obj.dihedral_funct == 1 and len(params) == 2:
                        params = list(params) + [1.0]
                    if imp_obj.dihedral_funct == 1 and len(params) < 3:
                        raise ValueError(f"Proper improper-dihedral params 부족: {imp}")
                    if imp_obj.dihedral_funct != 1 and len(params) < 2:
                        raise ValueError(f"Improper params 부족: {imp}")
                    if len(params) > 0:
                        imp_obj.dihedral_c0 = float(params[0])
                    if len(params) > 1:
                        imp_obj.dihedral_c1 = float(params[1])
                    if len(params) > 2:
                        imp_obj.dihedral_c2 = float(params[2])
                else:
                    try:
                        from hydrogel_builder.config_params.config import Config
                        Config.debug_log(f"[polymer-rich-skip] improper refs missing local idx: {imp}")
                    except Exception:
                        pass
            for sec, lines in getattr(template, "other_sections", {}).items():
                # already-handled rich sections should not be duplicated via raw other_sections
                sec_lower = str(sec).lower()
                if sec_lower in (
                    "constraints","pairs","exclusions","dihedrals","impropers",
                    "cmaptypes","polarization"
                ) or sec_lower.startswith("virtual_sites") or sec_lower.endswith("_restraints") or "restraint" in sec_lower:
                    continue
                for ln in lines:
                    _add_other(sec_lower, {"line": ln})
        except Exception as exc:
            print(f"[경고] 폴리머 템플릿 부가 섹션 매핑 실패: {exc}")

    def _attach_terminals(self, World):
        """Attach terminal templates to the first and last backbone beads."""
        if not self._terminal_records or not getattr(self, '_backbone_atom_ids', None):
            return
        left_template, right_template = self._select_terminal_templates()
        if left_template and len(self._backbone_atom_ids) >= 1:
            left_atom = World.Atoms[self._backbone_atom_ids[0]][0]
            if len(self._backbone_atom_ids) > 1:
                tangent = World.Atoms[self._backbone_atom_ids[1]][0].position - left_atom.position
            else:
                tangent = np.array([1.0, 0.0, 0.0])
            positions = self._place_template(left_template, left_atom.position, tangent)
            created_atoms = self._create_template_atoms(left_template, positions)
            self._connect_template_bonds(left_template, created_atoms, left_atom.atom_id)

        if right_template and len(self._backbone_atom_ids) >= 1:
            right_atom = World.Atoms[self._backbone_atom_ids[-1]][0]
            if len(self._backbone_atom_ids) > 1:
                tangent = right_atom.position - World.Atoms[self._backbone_atom_ids[-2]][0].position
            else:
                tangent = np.array([-1.0, 0.0, 0.0])
            positions = self._place_template(right_template, right_atom.position, tangent)
            created_atoms = self._create_template_atoms(right_template, positions)
            self._connect_template_bonds(right_template, created_atoms, right_atom.atom_id)

    def _construct_sidechains_from_templates(self):
        """Attach polymer side-chain templates while avoiding local clashes."""
        from hydrogel_builder.main_components.Universe import World
        if not self._sidechain_iterators or not self._sidechain_library:
            return

        NUM_CANDIDATE_VECTORS = 72
        OVERLAP_THRESHOLD_FACTOR = 0.8
        SEARCH_RADIUS_FACTOR = 10.0
        overlap_check_limit = p.Config.get_param('simulation_parameters', 'overlap_check_limit')

        all_atoms = [World.Atoms[_id][0] for _id in World.Atoms]
        for backbone_atom in all_atoms:
            backbone_type = getattr(backbone_atom, 'backbone_type', None)
            if backbone_type is None:
                continue
            iterator = self._sidechain_iterators.get(backbone_type)
            if not iterator:
                continue
            template = iterator.next()
            if not template or not template.beads:
                continue

            if len(backbone_atom.bonded_atoms) == 0:
                continue
            b1 = not_self(backbone_atom, backbone_atom.bonded_atoms[0])
            if len(backbone_atom.bonded_atoms) > 1:
                b2 = not_self(backbone_atom, backbone_atom.bonded_atoms[1])
            else:
                b2 = b1

            if backbone_atom.number_of_bonds > 1:
                p1, p2, p3 = b1.position, backbone_atom.position, b2.position
            else:
                p1, p2, p3 = b1.position, backbone_atom.position, backbone_atom.position + rij(backbone_atom.position, b1.position, World.box_length)

            search_radius_sq = (SEARCH_RADIUS_FACTOR * World.mean_sep) ** 2
            overlap_threshold_sq = (OVERLAP_THRESHOLD_FACTOR * World.mean_sep) ** 2
            bonded_atom_ids = {backbone_atom.atom_id}
            for bond in backbone_atom.bonded_atoms:
                bonded_atom_ids.add(not_self(backbone_atom, bond).atom_id)
            nearby_atoms = []
            for atom in all_atoms:
                if atom.atom_id != backbone_atom.atom_id and dij_sq(backbone_atom.position, atom.position, World.box_length) < search_radius_sq:
                    if atom.atom_id not in bonded_atom_ids:
                        nearby_atoms.append(atom)

            tangent_vec = rij(p1, p3, World.box_length)
            if np.linalg.norm(tangent_vec) < 1e-8:
                tangent_vec = np.array([0.0, 0.0, 1.0])

            best_positions = None
            min_penalty = float('inf')
            for _ in range(NUM_CANDIDATE_VECTORS):
                candidate_vector = random_normal_vector(p1, p2, p3, 1.0, World.box_length)
                positions = self._compute_template_positions(template, backbone_atom.position, candidate_vector, tangent_vec)
                if positions is None:
                    continue
                current_penalty = 0.0
                is_valid = True
                for pos in positions:
                    for neighbor in nearby_atoms:
                        d_sq = dij_sq(pos, neighbor.position, World.box_length)
                        if d_sq < overlap_threshold_sq:
                            is_valid = False
                            break
                        current_penalty += 1.0 / max(d_sq, 1e-6)
                    if not is_valid:
                        break
                if is_valid and current_penalty < min_penalty:
                    min_penalty = current_penalty
                    best_positions = positions

            if best_positions is None:
                continue
            created_atoms = self._create_template_atoms(template, best_positions)
            self._connect_template_bonds(template, created_atoms, backbone_atom.atom_id)

    def construct_chemical_detail(self):
        """Expand the polymer backbone into a chemically detailed topology.

        The preferred path uses configured side-chain and terminal templates.
        If template libraries are unavailable, the method falls back to the
        older geometric side-chain placement routine.
        """
        if self._sidechain_iterators:
            return self._construct_sidechains_from_templates()
        from hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.
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
        """Generate polymer angle terms from specific and default rules.

        Angle assignment first checks chemistry-specific overrides declared in
        the polymer configuration. If no override matches the atom-type set,
        the method applies the configured default angle parameters.
        """
        from hydrogel_builder.main_components.Universe import World # 순환 참조를 피하기 위해 함수 내에서 임포트합니다.
        
        _World_Bonds = World.Bonds
        _atom = World.Atoms
        
        if self._polymer_config and 'angles' in self._polymer_config:
            angle_configs = self._polymer_config['angles']
        else:
            try:
                angle_configs = p.Config.get_param('polymer_components', 'angles')
            except KeyError:
                angle_configs = {
                    'default_angle': {
                        'angle_funct': 1,
                        'angle_c0': 180.0,
                        'angle_c1': 25.0
                    },
                    'specific_angles': []
                }
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
