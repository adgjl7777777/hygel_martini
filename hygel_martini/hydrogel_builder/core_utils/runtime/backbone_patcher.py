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

    # Identify backbone residue names from rules (for wildcard filtering)
    backbone_res_names = set()
    for rule_type in ['bonds', 'angles', 'dihedrals']:
        for rule in config.get(rule_type, []):
            for res in rule['residue_name']:
                if res not in ["*", "-"]:
                    backbone_res_names.add(res)

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

    print(f"Backbone topology patching complete (sections={sections}).")
