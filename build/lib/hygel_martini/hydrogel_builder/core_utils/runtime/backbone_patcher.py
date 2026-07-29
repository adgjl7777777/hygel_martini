import yaml
import os
from hygel_martini.hydrogel_builder.main_components.Universe import World
from hygel_martini.hydrogel_builder.main_components.Attributes import Angle, Dihedral

def patch_backbone_topology(config_path, sections=('bonds', 'angles', 'dihedrals')):
    """
    Patches the World's Bonds, Angles and Dihedrals based on backbone.yaml rules.
    sections: which sections to process. Default is all three.
              Pass sections=('bonds',) for an early bonds-only pass before finalize_hydrogel.
    """
    if not os.path.exists(config_path):
        print(f"[INFO] Backbone patch config not found at {config_path}. Skipping.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Identify backbone residue names from Config (for precise wildcard filtering)
    # We load real backbone residue names to avoid classifying linker stubs as backbones.
    backbone_res_names = set()
    linker_res_names = set()
    sidechain_res_names = set()

    try:
        from hygel_martini.hydrogel_builder.config_params.read_json import Config
        bb_defs = Config.get_param('hydrogel_components', 'backbone_definitions', 'BACKBONES') or []
        for bb in bb_defs:
            res_name = bb.get('definition', {}).get('residue_name')
            if res_name:
                backbone_res_names.add(res_name)

        # Retrieve Linker & Sidechain residue names to check for overlaps
        linker_defs = Config.get_param('hydrogel_components', 'linker_definitions', 'LINKERS') or []
        for lk in linker_defs:
            res_name = lk.get('linker_residue_name')
            if res_name:
                linker_res_names.add(res_name)
            bb_res = lk.get('backbone_residue_name')
            if isinstance(bb_res, list):
                linker_res_names.update(bb_res)
            elif bb_res:
                linker_res_names.add(bb_res)

        monomer_defs = Config.get_param('monomer_definitions', 'MONOMERS') or []
        for mon in monomer_defs:
            res_name = mon.get('residue_name') or mon.get('id')
            if res_name:
                sidechain_res_names.add(res_name)
    except Exception as e:
        print(f"[WARNING] Failed to load component definitions from Config: {e}. Falling back to rule-based parsing.")

    # Fallback to rule-based gathering if Config was not loaded or returned no backbones
    if not backbone_res_names:
        for rule_type in ['bonds', 'angles', 'dihedrals']:
            for rule in config.get(rule_type, []):
                for res in rule['residue_name']:
                    if res not in ["*", "-"]:
                        backbone_res_names.add(res)

    # Overlap checking and warnings
    if backbone_res_names:
        overlap_bb_lk = backbone_res_names.intersection(linker_res_names)
        overlap_bb_sc = backbone_res_names.intersection(sidechain_res_names)
        if overlap_bb_lk:
            print(f"[WARNING] 백본과 가교제(Linker)의 잔기 이름이 중복됩니다: {overlap_bb_lk}. "
                  f"이로 인해 와일드카드 '*' 매칭이 정상 작동하지 않을 수 있습니다.")
        if overlap_bb_sc:
            print(f"[WARNING] 백본과 사이드체인(Sidechain/Monomer)의 잔기 이름이 중복됩니다: {overlap_bb_sc}. "
                  f"이로 인해 와일드카드 '*' 매칭이 정상 작동하지 않을 수 있습니다.")


    # Build adjacency list
    adj = {}
    for (i, j), bonds in World.Bonds.items():
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    def matches(atom, res_name_rule, bead_type_rule):
        if res_name_rule == "-" or bead_type_rule == "-":
            return True
        if res_name_rule == "*":
            if atom.residue_name in backbone_res_names:
                return False
            if bead_type_rule != "*" and atom.atom_type != bead_type_rule:
                return False
            return True
        if res_name_rule != "*" and atom.residue_name != res_name_rule:
            return False
        if bead_type_rule != "*" and atom.atom_type != bead_type_rule:
            return False
        return True

    def find_paths(current_path, res_rules, type_rules):
        if len(current_path) == len(res_rules):
            yield current_path
            return
        current_idx = current_path[-1]
        next_rule_idx = len(current_path)
        for neighbor in adj.get(current_idx, []):
            if neighbor in current_path:
                continue
            neighbor_atom = World.Atoms[neighbor][0]
            if matches(neighbor_atom, res_rules[next_rule_idx], type_rules[next_rule_idx]):
                yield from find_paths(current_path + [neighbor], res_rules, type_rules)

    def count_wildcards(rule):
        return rule.get('residue_name', []).count('*') + rule.get('bead_type', []).count('*')

    def normalize_param(value):
        if value in (None, ' ', ''):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    def dihedral_signature_from_rule(rule):
        funct = normalize_param(rule['funct'])
        if 'params' in rule:
            return (funct, tuple(normalize_param(param) for param in rule['params']))
        return (
            funct,
            normalize_param(rule.get('c0')),
            normalize_param(rule.get('c1')),
            normalize_param(rule.get('multiplicity')),
        )

    def dihedral_signature(dihedral):
        funct = normalize_param(dihedral.dihedral_funct)
        d_params = getattr(dihedral, "dihedral_params", None)
        if d_params is not None:
            return (funct, tuple(normalize_param(param) for param in d_params))
        return (
            funct,
            normalize_param(dihedral.dihedral_c0),
            normalize_param(dihedral.dihedral_c1),
            normalize_param(dihedral.dihedral_c2),
        )

    def has_dihedral_signature(existing_dihedrals, d_key, signature):
        return any(dihedral_signature(dih) == signature for dih in existing_dihedrals.get(d_key, []))

    # Process Bonds
    if 'bonds' in sections:
        bonds_rules = sorted(config.get('bonds', []), key=count_wildcards, reverse=True)
        for rule in bonds_rules:
            res_rules = rule['residue_name']
            type_rules = rule['bead_type']
            if len(res_rules) != 2:
                continue
            for atom_id in World.Atoms:
                atom = World.Atoms[atom_id][0]
                if matches(atom, res_rules[0], type_rules[0]):
                    for neighbor_id in adj.get(atom_id, []):
                        neighbor = World.Atoms[neighbor_id][0]
                        if matches(neighbor, res_rules[1], type_rules[1]):
                            a, b = (atom_id, neighbor_id) if atom_id < neighbor_id else (neighbor_id, atom_id)
                            if (a, b) in World.Bonds:
                                for bond in World.Bonds[(a, b)]:
                                    bond.bond_funct = rule['funct']
                                    bond.bond_c0 = rule['c0']
                                    bond.bond_c1 = rule['c1']

    # Process Angles
    if 'angles' in sections:
        existing_angles = {}
        for key, angles in World.Angles.items():
            a1, a2, a3 = key
            if a1 > a3: a1, a3 = a3, a1
            existing_angles.setdefault((a1, a2, a3), []).extend(angles)

        angles_rules = sorted(config.get('angles', []), key=count_wildcards, reverse=True)
        for rule in angles_rules:
            res_rules = rule['residue_name']
            type_rules = rule['bead_type']
            allow_dup = rule.get('allow_duplicate', False)

            include_indices = [i for i, r in enumerate(res_rules) if r != "-"]
            if len(include_indices) != 3:
                continue

            for atom_id in World.Atoms:
                atom = World.Atoms[atom_id][0]
                if matches(atom, res_rules[0], type_rules[0]):
                    for path in find_paths([atom_id], res_rules, type_rules):
                        idx1, idx2, idx3 = path[include_indices[0]], path[include_indices[1]], path[include_indices[2]]
                        lookup_idx1, lookup_idx3 = (idx1, idx3) if idx1 < idx3 else (idx3, idx1)
                        angle_key = (lookup_idx1, idx2, lookup_idx3)

                        if angle_key in existing_angles and not allow_dup:
                            for ang in existing_angles[angle_key]:
                                ang.angle_funct = rule['funct']
                                ang.angle_c0 = rule['c0']
                                ang.angle_c1 = rule['c1']
                        else:
                            new_ang = Angle(idx1, idx2, idx3)
                            new_ang.angle_funct = rule['funct']
                            new_ang.angle_c0 = rule['c0']
                            new_ang.angle_c1 = rule['c1']
                            existing_angles.setdefault(angle_key, []).append(new_ang)

        # Update World.Angles to include newly added/modified angles
        World.Angles.clear()
        for key, angles in existing_angles.items():
            World.Angles[key] = angles

    # Process Dihedrals
    if 'dihedrals' in sections:
        existing_dihedrals = {}
        for key, dihedrals in World.Dihedrals.items():
            a1, a2, a3, a4 = key
            d_key = (a4, a3, a2, a1) if a1 > a4 else (a1, a2, a3, a4)
            existing_dihedrals.setdefault(d_key, []).extend(dihedrals)

        dihedrals_rules = sorted(config.get('dihedrals', []), key=count_wildcards, reverse=True)
        for rule in dihedrals_rules:
            res_rules = rule['residue_name']
            type_rules = rule['bead_type']
            allow_dup = rule.get('allow_duplicate', False)
            rule_signature = dihedral_signature_from_rule(rule)

            include_indices = [i for i, r in enumerate(res_rules) if r != "-"]
            if len(include_indices) != 4:
                continue

            for atom_id in World.Atoms:
                atom = World.Atoms[atom_id][0]
                if matches(atom, res_rules[0], type_rules[0]):
                    for path in find_paths([atom_id], res_rules, type_rules):
                        ids = [path[i] for i in include_indices]
                        d_key = (ids[3], ids[2], ids[1], ids[0]) if ids[0] > ids[3] else (ids[0], ids[1], ids[2], ids[3])

                        if d_key in existing_dihedrals and not allow_dup:
                            for dih in existing_dihedrals[d_key]:
                                dih.dihedral_funct = rule['funct']
                                if 'params' in rule:
                                    dih.dihedral_params = list(rule['params'])
                                else:
                                    dih.dihedral_c0 = rule['c0']
                                    dih.dihedral_c1 = rule['c1']
                                    if 'multiplicity' in rule:
                                        dih.dihedral_c2 = rule['multiplicity']
                        else:
                            if allow_dup and has_dihedral_signature(existing_dihedrals, d_key, rule_signature):
                                continue
                            new_dih = Dihedral(ids[0], ids[1], ids[2], ids[3])
                            new_dih.dihedral_funct = rule['funct']
                            if 'params' in rule:
                                new_dih.dihedral_params = list(rule['params'])
                            else:
                                new_dih.dihedral_c0 = rule['c0']
                                new_dih.dihedral_c1 = rule['c1']
                                if 'multiplicity' in rule:
                                    new_dih.dihedral_c2 = rule['multiplicity']
                            existing_dihedrals.setdefault(d_key, []).append(new_dih)

        # Update World.Dihedrals to include newly added/modified dihedrals
        World.Dihedrals.clear()
        for key, dihedrals in existing_dihedrals.items():
            World.Dihedrals[key] = dihedrals

    print(f"Backbone topology patching complete (sections={sections}).")
