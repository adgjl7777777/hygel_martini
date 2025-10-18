
import re

def read_itp_definitions(itp_file_path):
    """
    Parses a Martini .itp file and extracts molecule definitions.

    Args:
        itp_file_path (str): The path to the .itp file.

    Returns:
        dict: A dictionary where keys are molecule names and values are their
              definitions (including beads and bonds).
    """
    definitions = {}
    current_molecule = None
    section = None

    with open(itp_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # Check for section headers like [ moleculetype ], [ atoms ], etc.
            match = re.match(r'\[\s*(\w+)\s*\]', line)
            if match:
                section = match.group(1).lower()
                if section == 'moleculetype':
                    current_molecule = None # Reset on new moleculetype
                continue

            if section == 'moleculetype':
                parts = line.split()
                if len(parts) >= 1:
                    molecule_name = parts[0]
                    current_molecule = {
                        'name': molecule_name,
                        'beads': [],
                        'bonds': []
                    }
                    definitions[molecule_name] = current_molecule

            elif section == 'atoms' and current_molecule:
                parts = line.split()
                if len(parts) >= 5:
                    bead = {
                        'nr': int(parts[0]),
                        'type': parts[1],
                        'resnr': int(parts[2]),
                        'residue': parts[3],
                        'atom': parts[4],
                        'cgnr': int(parts[5]),
                        'charge': float(parts[6]) if len(parts) > 6 else 0.0,
                        'mass': float(parts[7]) if len(parts) > 7 else 0.0
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

    return definitions

if __name__ == '__main__':
    # Example usage:
    # Replace with a real path to a martini .itp file for testing
    # itp_file = '../martini_v300/martini_v3.0.0_small_molecules_v1.itp'
    # definitions = read_itp_definitions(itp_file)
    # import json
    # print(json.dumps(definitions, indent=2))
    pass
