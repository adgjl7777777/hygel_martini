"""Reject template semantics the current expansion cannot faithfully preserve.

The generated network uses nrexcl=1 and rule-generated boundary angles.
This gate prevents silently claiming template-faithful transfer for unsupported
inputs. It does not change or fit any interaction parameters.
"""

def require_supported_template(definition, backbone_indices=(), exclusion_policy=None):
    name = definition.get("name", "template")
    junction_override = (exclusion_policy == "network_nrexcl_one"
                         and len(definition.get("beads", [])) == 2
                         and len(definition.get("bonds", [])) == 1
                         and {definition["bonds"][0].get("from"), definition["bonds"][0].get("to")} == {b["nr"] for b in definition["beads"]}
                         and not any(definition.get(k) for k in ("angles", "dihedrals", "impropers")))
    if definition.get("nrexcl", 1) != 1 and not junction_override:
        raise ValueError(f"Unsupported template {name}: nrexcl must be 1; per-instance exclusion transfer is not implemented")
    adjacency = {}
    for bond in definition.get("bonds", []):
        a, b = bond.get("from"), bond.get("to")
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    for angle in definition.get("angles", []):
        a, b, c = angle.get("from"), angle.get("center"), angle.get("to")
        if a not in adjacency.get(b, ()) or c not in adjacency.get(b, ()):
            raise ValueError(f"Unsupported template {name}: non-path angle {(a,b,c)} would not be retained")
        if set((a, b, c)) & set(backbone_indices):
            raise ValueError(f"Unsupported template {name}: template angle involving a replaced backbone bead requires explicit instance mapping")
    for section in ("dihedrals", "impropers"):
        if definition.get(section):
            raise ValueError(f"Unsupported template {name}: per-instance {section} transfer is not certified; use registered assembly rules for supported models")
    # Auxiliary records are currently assembled through a template-level map;
    # repeated instances need a distinct map before these can be certified.
    for section in ("constraints", "pairs", "exclusions", "virtual_sites", "restraints", "polarization", "cmaptypes", "other_sections"):
        if definition.get(section):
            raise ValueError(f"Unsupported template {name}: per-instance {section} transfer is not certified")
