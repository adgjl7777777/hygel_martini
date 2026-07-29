"""Populate Hydrogel atoms/bonds from a proto layout blueprint."""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from hygel_martini.hydrogel_builder.core_utils.layout.layout_executor import LayoutBlueprint, ChainBlueprint
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.main_components.Universe import World


def _ordered_chain_entries(chain_key: Tuple[str, int],
                           chain_atom_map: Dict[Tuple[str, int], List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    entries = chain_atom_map.get(chain_key, [])
    entries.sort(key=lambda item: item[0])
    return entries


def _mark_backbone_terminals(hydrogel, atom_ids: List[int]):
    if not atom_ids:
        return
    head = World.Atoms[atom_ids[0]][0]
    tail = World.Atoms[atom_ids[-1]][0]

    for atom in (head, tail):
        if atom not in hydrogel.terminals[1]:
            atom.end_tag = 1
            hydrogel.terminals[1].append(atom)


def _mark_linker_terminals(hydrogel, chain: ChainBlueprint, bead_atom_ids: List[int]):
    ext_bonds = chain.definition.get('external_bonds', []) or []
    if not ext_bonds or not bead_atom_ids:
        return

    for ext in ext_bonds:
        bead_idx = ext.get('from_bead')
        if bead_idx is None or bead_idx < 0 or bead_idx >= len(bead_atom_ids):
            continue
        atom = World.Atoms[bead_atom_ids[bead_idx]][0]
        target_bb = ext.get('to_backbone')
        if target_bb == 'dummy_id':
            target_bb = None
        atom.end_tag = 2
        atom.target_bb = target_bb
        atom.external_bond_length = float(ext.get('length', World.mean_sep))
        hydrogel.terminals[2].append(atom)


def _create_backbone_bonds(chain: ChainBlueprint, atom_ids: List[int]):
    if len(atom_ids) < 2:
        return
    metadata = chain.metadata or {}
    sequence = metadata.get('sequence', [])
    bond_lookup = metadata.get('bond_lookup', {}) or {}
    fallback = float(metadata.get('mean_sep', World.mean_sep))
    default_params = chain.definition or {}
    for idx, (first, second) in enumerate(zip(atom_ids[:-1], atom_ids[1:])):
        entry_a = sequence[idx] if idx < len(sequence) else None
        entry_b = sequence[idx + 1] if (idx + 1) < len(sequence) else None
        key = None
        if entry_a and entry_b:
            ida = entry_a.get('id')
            idb = entry_b.get('id')
            if ida and idb:
                key = tuple(sorted((ida, idb)))
        params = bond_lookup.get(key, {})
        funct = params.get('bond_funct', params.get('funct', default_params.get('bond_funct', 1)))
        c0 = params.get('bond_c0', params.get('length', default_params.get('bond_c0', fallback)))
        c1 = params.get('bond_c1', params.get('fc', default_params.get('bond_c1', 56000.0)))
        Attributes.Bond(first, second, funct=int(funct), c0=float(c0), c1=float(c1))


def _create_linker_bonds(chain: ChainBlueprint, atom_ids: List[int]):
    bonds = chain.definition.get('bonds', []) or []
    if not bonds or len(atom_ids) < 2:
        return
    for bond_def in bonds:
        idx1 = bond_def.get('from')
        idx2 = bond_def.get('to')
        if idx1 is None or idx2 is None:
            continue
        if min(idx1, idx2) < 0 or max(idx1, idx2) >= len(atom_ids):
            continue
        params = {k: v for k, v in bond_def.items() if k not in {'from', 'to'}}
        Attributes.Bond(atom_ids[idx1], atom_ids[idx2], **params)


def _finalize_counts(hydrogel):
    hydrogel.num_HDG_atoms = Attributes.Atom.num_atoms
    hydrogel.num_HDG_bonds = Attributes.Bond.num_bonds


def populate_hydrogel_from_blueprint(hydrogel, blueprint: LayoutBlueprint):
    """
    Construct Atom objects and backbone/linker internal bonds directly from a proto blueprint.
    """
    chain_atom_map: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    stub_meta_by_chain = defaultdict(list)
    # original_index(ITP, 1-based) -> global atom_id(0-based)
    orig_to_global_by_chain: Dict[Tuple[str, int], Dict[int, int]] = defaultdict(dict)
    template_by_chain: Dict[Tuple[str, int], object] = {}
    chain_meta_by_key = {(c.chain_type, c.chain_index): (c.metadata or {}) for c in blueprint.chains}

    for atom_bp in blueprint.atoms:
        # Pass source_template directly to constructor if available
        source_template = atom_bp.extra.get('source_template') if atom_bp.extra else None
        atom = Attributes.Atom(source_template=source_template)
        
        atom.atom_type = atom_bp.atom_type
        atom.residue_number = atom_bp.residue_number
        atom.residue_name = atom_bp.residue_name
        atom.atom_name = atom_bp.atom_name
        atom.cgnr = atom_bp.charge_group_number
        atom.mass = atom_bp.mass
        atom.charge = atom_bp.charge
        atom.chain_type = atom_bp.chain_type
        if atom_bp.backbone_type and atom_bp.backbone_type != 'dummy_id':
            atom.backbone_type = atom_bp.backbone_type
        atom.position = np.array(atom_bp.position, dtype=np.float64)
        atom.chain_index = atom_bp.chain_index  # Track chain index for all atoms
        chain_key = (atom_bp.chain_type, atom_bp.chain_index)
        chain_meta = chain_meta_by_key.get(chain_key, {})
        if atom_bp.chain_type == "linker":
            atom.linker_axis = chain_meta.get("linker_axis")
            atom.linker_local_index = chain_meta.get("local_linker_index")
            atom.linker_axis_dir = chain_meta.get("axis_direction")
            atom.linker_anchor = chain_meta.get("anchor_position")
            atom.linker_chain_index = atom_bp.chain_index

        # Set stub_type and other extra attributes
        if atom_bp.extra:
            atom.stub_type = atom_bp.extra.get('stub_type')
            atom.target_backbone = atom_bp.extra.get('target_backbone')
            if atom.target_backbone == 'dummy_id':
                atom.target_backbone = None
            if atom.target_backbone and not atom.target_bb:
                atom.target_bb = atom.target_backbone
            if 'stub_from_bead' in atom_bp.extra:
                params = atom_bp.extra.get('external_params', {})
                atom.stub_from_bead = atom_bp.extra.get('stub_from_bead')
                atom.stub_bond_params = params
            orig_idx = atom_bp.extra.get('original_index')
            if isinstance(orig_idx, int):
                chain_key = (atom_bp.chain_type, atom_bp.chain_index)
                orig_to_global_by_chain[chain_key][orig_idx] = atom.atom_id
                if source_template is not None:
                    template_by_chain[chain_key] = source_template
            pre_pos = atom_bp.extra.get('pre_compress_position')
            if pre_pos is not None:
                atom.pre_compress_position = np.array(pre_pos, dtype=np.float64)

        chain_key = (atom_bp.chain_type, atom_bp.chain_index)
        chain_atom_map.setdefault(chain_key, []).append((atom_bp.bead_index, atom.atom_id))
        if atom_bp.extra and 'stub_from_bead' in atom_bp.extra:
            stub_meta_by_chain[chain_key].append({
                'atom_id': atom.atom_id,
                'from_bead': atom_bp.extra.get('stub_from_bead'),
                'bond_params': atom_bp.extra.get('external_params', {})
            })

    for chain in blueprint.chains:
        chain_key = (chain.chain_type, chain.chain_index)
        entries = _ordered_chain_entries(chain_key, chain_atom_map)
        atom_ids = [atom_id for _, atom_id in entries]
        if chain.chain_type == 'backbone':
            _mark_backbone_terminals(hydrogel, atom_ids)
            _create_backbone_bonds(chain, atom_ids)
        elif chain.chain_type == 'linker':
            bead_atom_ids = [atom_id for bead_idx, atom_id in entries if bead_idx >= 0]
            _mark_linker_terminals(hydrogel, chain, bead_atom_ids)
            _create_linker_bonds(chain, bead_atom_ids)
            bead_map = {bead_idx: atom_id for bead_idx, atom_id in entries if bead_idx >= 0}
            for stub_meta in stub_meta_by_chain.get(chain_key, []):
                from_idx = stub_meta.get('from_bead')
                source_id = bead_map.get(from_idx)
                if source_id is None:
                    continue
                params = stub_meta.get('bond_params', {})
                Attributes.Bond(source_id, stub_meta['atom_id'], **params)

    # --- Map linker rich sections into World.OtherSections (stub 포함 idx_map 사용) ---
    try:
        from hygel_martini.hydrogel_builder.config_params.config import Config
    except Exception:
        Config = None  # type: ignore

    def _add_other(sec: str, payload: Dict):
        World.OtherSections[sec].append(payload)

    for chain in blueprint.chains:
        if chain.chain_type != "linker":
            continue
        chain_key = (chain.chain_type, chain.chain_index)
        template = template_by_chain.get(chain_key)
        if template is None:
            continue
        idx_map = orig_to_global_by_chain.get(chain_key, {})
        mapped_counts = defaultdict(int)
        skipped_counts = defaultdict(int)
        stub_original_indices = {
            bead.get("nr")
            for bead in (getattr(template, "stub_definitions", []) or [])
            if bead.get("nr") is not None
        }

        def _map_constraints(rows: List[Dict]):
            for row in rows or []:
                i_local = row.get("i")
                j_local = row.get("j")
                i = idx_map.get(i_local)
                j = idx_map.get(j_local)
                if i is None or j is None:
                    skipped_counts["constraints"] += 1
                    continue
                constraint = Attributes.Constraint(i, j)
                constraint.constraint_funct = int(row.get("funct", 1))
                params = row.get("params", [])
                if params:
                    constraint.constraint_c0 = float(params[0])
                mapped_counts["constraints"] += 1

        _map_constraints(getattr(template, "constraints", []))

        # pairs는 현재 단계에서 World로 옮기지 않음(사용자 요청)
        if getattr(template, "pairs", []):
            skipped_counts["pairs"] += len(getattr(template, "pairs", []))

        for ex_def in getattr(template, "exclusions", []) or []:
            atom_local = ex_def.get("atom")
            atom_idx = idx_map.get(atom_local)
            if atom_idx is None:
                skipped_counts["exclusions"] += 1
                continue
            for x in ex_def.get("exclude", []):
                # Stub-to-stub exclusions become long-ranged after the two
                # linker terminals are attached to different backbone ends.
                if atom_local in stub_original_indices and x in stub_original_indices:
                    skipped_counts["exclusions"] += 1
                    continue
                gx = idx_map.get(x)
                if gx is None:
                    skipped_counts["exclusions"] += 1
                    continue
                Attributes.Exclusion(atom_idx, gx)
                mapped_counts["exclusions"] += 1

        # 나머지 rich 섹션은 OtherSections로만 전달(기본 스킵 정책)
        for vs in getattr(template, "virtual_sites", []) or []:
            _add_other("virtual_sites", vs)
            skipped_counts["virtual_sites"] += 1
        for rst in getattr(template, "restraints", []) or []:
            _add_other(rst.get("section", "restraints"), rst)
            skipped_counts[rst.get("section", "restraints")] += 1
        for cm in getattr(template, "cmaptypes", []) or []:
            _add_other("cmaptypes", {"row": cm})
            skipped_counts["cmaptypes"] += 1
        for pol in getattr(template, "polarization", []) or []:
            _add_other("polarization", pol)
            skipped_counts["polarization"] += 1

        # full dihedrals/impropers
        for dih in getattr(template, "dihedrals_full", []) or []:
            gi = idx_map.get(dih.get("i"))
            gj = idx_map.get(dih.get("j"))
            gk = idx_map.get(dih.get("k"))
            gl = idx_map.get(dih.get("l"))
            if None in (gi, gj, gk, gl):
                skipped_counts["dihedrals"] += 1
                continue
            dihedral = Attributes.Dihedral(gi, gj, gk, gl, 0)
            dihedral.dihedral_funct = int(dih.get("funct", 1))
            params = dih.get("params", [])
            # GROMACS funct=1 proper dihedral requires 3 params; if template provides 2, assume multiplicity=1.
            if dihedral.dihedral_funct == 1 and len(params) == 2:
                params = list(params) + [1.0]
            if dihedral.dihedral_funct == 1 and len(params) < 3:
                skipped_counts["dihedrals"] += 1
                continue
            if dihedral.dihedral_funct != 1 and len(params) < 2:
                skipped_counts["dihedrals"] += 1
                continue
            if len(params) > 0:
                dihedral.dihedral_c0 = float(params[0])
            if len(params) > 1:
                dihedral.dihedral_c1 = float(params[1])
            if len(params) > 2:
                dihedral.dihedral_c2 = float(params[2])
            mapped_counts["dihedrals"] += 1

        for imp in getattr(template, "impropers_full", []) or []:
            gi = idx_map.get(imp.get("i"))
            gj = idx_map.get(imp.get("j"))
            gk = idx_map.get(imp.get("k"))
            gl = idx_map.get(imp.get("l"))
            if None in (gi, gj, gk, gl):
                skipped_counts["impropers"] += 1
                continue
            dihedral = Attributes.Dihedral(gi, gj, gk, gl, 0)
            dihedral.dihedral_funct = int(imp.get("funct", 2))
            params = imp.get("params", [])
            if dihedral.dihedral_funct == 1 and len(params) == 2:
                params = list(params) + [1.0]
            if dihedral.dihedral_funct == 1 and len(params) < 3:
                skipped_counts["impropers"] += 1
                continue
            if dihedral.dihedral_funct != 1 and len(params) < 2:
                skipped_counts["impropers"] += 1
                continue
            if len(params) > 0:
                dihedral.dihedral_c0 = float(params[0])
            if len(params) > 1:
                dihedral.dihedral_c1 = float(params[1])
            if len(params) > 2:
                dihedral.dihedral_c2 = float(params[2])
            mapped_counts["impropers"] += 1

        for sec, lines in getattr(template, "other_sections", {}).items():
            sec_lower = str(sec).lower()
            if sec_lower in (
                "constraints","pairs","exclusions","dihedrals","impropers",
                "cmaptypes","polarization"
            ) or sec_lower.startswith("virtual_sites") or sec_lower.endswith("_restraints") or "restraint" in sec_lower:
                continue
            for ln in lines:
                _add_other(sec_lower, {"line": ln})
                mapped_counts[sec_lower] += 1

        if Config is not None:
            Config.debug_log(
                f"[proto-linker-rich] chain={chain.chain_index} template={getattr(template,'id',None)} "
                f"mapped={dict(mapped_counts)} skipped={dict(skipped_counts)}"
            )

    _finalize_counts(hydrogel)
