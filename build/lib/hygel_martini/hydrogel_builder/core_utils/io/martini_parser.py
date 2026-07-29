
import re

def read_atom_types(itp_file_path):
    """
    Parses an .itp file and extracts only the [ atomtypes ] section.
    Returns a dictionary mapping atom type names to their mass.
    """
    atom_types = {}
    section = None
    if not itp_file_path or not isinstance(itp_file_path, str):
        return {}
    try:
        with open(itp_file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                if ';' in line:
                    line = line.split(';', 1)[0].rstrip()
                    if not line:
                        continue
                match = re.match(r'\[\s*(\w+)\s*\]', line)
                if match:
                    section = match.group(1).lower()
                    continue
                if section == 'atomtypes':
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type_name = parts[0]
                        try:
                            mass = float(parts[1])
                            atom_types[atom_type_name] = {'mass': mass}
                        except (ValueError, IndexError):
                            # Ignore lines that are not valid atomtype definitions
                            pass
    except FileNotFoundError:
        print(f"Warning: Atom types file not found at {itp_file_path}. Masses will not be loaded from here.")
    return atom_types

def read_itp_definitions(itp_file_path, atom_type_masses=None, prefer_explicit_masses=False):
    """
    Parses a Martini .itp file and extracts molecule definitions.

    Args:
        itp_file_path (str): The path to the .itp file.
        atom_type_masses (dict, optional): A lookup table for atom type masses.
        prefer_explicit_masses (bool): If true, the mass column in the
            molecule's [ atoms ] section takes precedence over atomtype mass.

    Returns:
        dict: A dictionary where keys are molecule names and values are their
              definitions (including beads, bonds, angles, etc.). Sections that
              are not explicitly parsed are collected under 'other_sections'
              as raw lines for future use.
    """
    definitions = {}
    current_molecule = None
    section = None
    # Helper to stash unknown/raw lines for later use.
    # NOTE: known/parsed rich sections should NOT be stashed, otherwise they
    # will be duplicated when emitted via OtherSections.
    KNOWN_SECTIONS = {
        "moleculetype","atoms","bonds","angles","dihedrals","impropers",
        "constraints","pairs","exclusions","cmaptypes","polarization",
        "position_restraints","distance_restraints","angle_restraints",
        "dihedral_restraints","orientation_restraints",
    }
    def stash_raw(sec_name, line):
        if not current_molecule or not sec_name:
            return
        sec_lower = sec_name.lower()
        if sec_lower in KNOWN_SECTIONS:
            return
        if sec_lower.startswith("virtual_sites"):
            return
        if sec_lower.endswith("_restraints") or "restraint" in sec_lower:
            return
        other = current_molecule.setdefault('other_sections', {})
        other.setdefault(sec_lower, []).append(line)

    with open(itp_file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if ';' in line:
                line = line.split(';', 1)[0].rstrip()
                if not line:
                    continue

            match = re.match(r'\[\s*(\w+)\s*\]', line)
            if match:
                section = match.group(1).lower()
                if section == 'moleculetype':
                    current_molecule = None
                continue

            if section == 'moleculetype':
                parts = line.split()
                if len(parts) >= 1:
                    molecule_name = parts[0]
                    current_molecule = {
                        'name': molecule_name,
                        'beads': [],
                        'bonds': [],
                        'angles': [],
                        'dihedrals': [],
                        'impropers': [],
                        'constraints': [],
                        'pairs': [],
                        'exclusions': [],
                        'virtual_sites': [],
                        'restraints': [],
                        'cmaptypes': [],
                        'polarization': []
                    }
                    definitions[molecule_name] = current_molecule

            elif section == 'atoms' and current_molecule:
                parts = line.split()
                if len(parts) >= 5:
                    atom_type = parts[1]
                    mass = 0.0
                    if prefer_explicit_masses and len(parts) > 7:
                        mass = float(parts[7]) # Fallback for itp files with explicit mass
                    elif atom_type_masses and atom_type in atom_type_masses:
                        mass = atom_type_masses[atom_type]['mass']
                    elif len(parts) > 7:
                        mass = float(parts[7]) # Fallback for itp files with explicit mass
                    
                    if mass == 0.0:
                        raise ValueError(
                            f"Mass for atom type '{atom_type}' in molecule '{current_molecule['name']}' "
                            f"from file '{itp_file_path}' could not be determined. "
                            "Please ensure it is defined in the base_itp_file or explicitly in the molecule's ITP."
                        )

                    bead = {
                        'nr': int(parts[0]),
                        'type': atom_type,
                        'resnr': int(parts[2]),
                        'residue': parts[3],
                        'atom': parts[4],
                        'cgnr': int(parts[5]),
                        'charge': float(parts[6]) if len(parts) > 6 else 0.0,
                        'mass': mass
                    }
                    current_molecule['beads'].append(bead)

            elif section == 'bonds' and current_molecule:
                parts = line.split()
                if len(parts) >= 4:
                    bond = {
                        'from': int(parts[0]),
                        'to': int(parts[1]),
                        'funct': int(parts[2]),
                        'params': [float(p) for p in parts[3:]]
                    }
                    current_molecule['bonds'].append(bond)
            
            elif section == 'angles' and current_molecule:
                parts = line.split()
                if len(parts) >= 5:
                    angle = {
                        'from': int(parts[0]),
                        'center': int(parts[1]),
                        'to': int(parts[2]),
                        'funct': int(parts[3]),
                        'params': [float(p) for p in parts[4:]]
                    }
                    current_molecule['angles'].append(angle)

            elif section == 'dihedrals' and current_molecule:
                parts = line.split()
                if len(parts) >= 6:
                    dihedral = {
                        'i': int(parts[0]),
                        'j': int(parts[1]),
                        'k': int(parts[2]),
                        'l': int(parts[3]),
                        'funct': int(parts[4]),
                        'params': [float(p) for p in parts[5:]]
                    }
                    current_molecule['dihedrals'].append(dihedral)

            elif section == 'impropers' and current_molecule:
                parts = line.split()
                if len(parts) >= 6:
                    improper = {
                        'i': int(parts[0]),
                        'j': int(parts[1]),
                        'k': int(parts[2]),
                        'l': int(parts[3]),
                        'funct': int(parts[4]),
                        'params': [float(p) for p in parts[5:]]
                    }
                    current_molecule['impropers'].append(improper)
            elif section == 'constraints' and current_molecule:
                parts = line.split()
                if len(parts) >= 3:
                    constraint = {
                        'i': int(parts[0]),
                        'j': int(parts[1]),
                        'funct': int(parts[2]),
                        'params': [float(p) for p in parts[3:]]
                    }
                    current_molecule['constraints'].append(constraint)
            elif section == 'pairs' and current_molecule:
                parts = line.split()
                if len(parts) >= 3:
                    pair = {
                        'i': int(parts[0]),
                        'j': int(parts[1]),
                        'funct': int(parts[2]),
                        'params': [float(p) for p in parts[3:]]
                    }
                    current_molecule['pairs'].append(pair)
            elif section == 'exclusions' and current_molecule:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom = int(parts[0])
                        excluded = [int(x) for x in parts[1:]]
                        current_molecule['exclusions'].append({'atom': atom, 'exclude': excluded})
                    except ValueError:
                        stash_raw(section, line)
            elif section and section.startswith('virtual_sites') and current_molecule:
                # Capture any virtual site definition; store raw + parsed ints if possible
                parts = line.split()
                try:
                    entry = {'section': section, 'line': line, 'parts': parts}
                    current_molecule['virtual_sites'].append(entry)
                except ValueError:
                    stash_raw(section, line)
            elif section in ('position_restraints', 'distance_restraints',
                             'angle_restraints', 'dihedral_restraints',
                             'orientation_restraints') and current_molecule:
                # Keep raw; optional parse to ints/floats
                stash_raw(section, line)
                parts = line.split()
                try:
                    nums = [float(x) if '.' in x or 'e' in x.lower() else int(x) for x in parts]
                    current_molecule['restraints'].append({'section': section, 'values': nums})
                except ValueError:
                    pass
            elif section == 'cmaptypes' and current_molecule:
                stash_raw(section, line)
                parts = line.split()
                if len(parts) >= 7:
                    current_molecule['cmaptypes'].append(parts)
            elif section == 'polarization' and current_molecule:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom = int(parts[0])
                        pol_values = [float(p) if '.' in p or 'e' in p.lower() else int(p) for p in parts[1:]]
                        current_molecule['polarization'].append({'atom': atom, 'params': pol_values})
                    except ValueError:
                        stash_raw(section, line)
            else:
                # Keep unknown sections raw for later debugging/extension
                if section and current_molecule:
                    stash_raw(section, line)

    return definitions


__all__ = ["read_atom_types", "read_itp_definitions"]
