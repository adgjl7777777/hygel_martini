"""
Rich ITP section validator.

When users enable `simulation_parameters.emit_rich_itp_sections=true`,
invalid test entries (duplicate virtual sites, out-of-range indices, bad funct)
can make grompp fail. This module validates and filters those sections before
they are written out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _try_int(token: Any):
    try:
        return int(token)
    except Exception:
        return None


def _in_range(idx: int, atom_count: int) -> bool:
    return 1 <= idx <= atom_count


def validate_and_filter_other_sections(
    extras: Dict[str, List[Dict[str, Any]]],
    atom_count: int,
    strict: bool = False,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """
    Validate rich sections stored in World.OtherSections.

    Returns:
        (filtered_extras, warnings)
    """
    if not extras:
        return {}, []

    warnings: List[str] = []
    filtered: Dict[str, List[Dict[str, Any]]] = {}

    # Only emit a minimal safe allowlist of rich sections by default.
    # User requested to keep only dihedrals/impropers/exclusions/constraints.
    ALLOWED = {"constraints", "exclusions", "dihedrals"}

    # ---- constraints / pairs ----
    # pairs are always skipped under the minimal allowlist policy.
    for sec in ("constraints", "pairs"):
        out_rows: List[Dict[str, Any]] = []
        for row in extras.get(sec, []) or []:
            i = row.get("i")
            j = row.get("j")
            if not isinstance(i, int) or not isinstance(j, int) or not _in_range(i, atom_count) or not _in_range(j, atom_count):
                msg = f"{sec}: out-of-range or invalid indices i={i}, j={j}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            if sec == "pairs":
                msg = "pairs: skipped by minimal allowlist policy"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            out_rows.append(row)
        if out_rows:
            filtered[sec] = out_rows

    # ---- exclusions ----
    out_excl: List[Dict[str, Any]] = []
    for row in extras.get("exclusions", []) or []:
        atom = row.get("atom")
        excl = row.get("exclude", [])
        if not isinstance(atom, int) or not _in_range(atom, atom_count):
            msg = f"exclusions: invalid atom={atom}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        mapped_excl = []
        bad = False
        for e in excl:
            if isinstance(e, int) and _in_range(e, atom_count):
                mapped_excl.append(e)
            else:
                bad = True
        if bad:
            msg = f"exclusions: some exclude indices invalid for atom={atom}"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
        out_excl.append({"atom": atom, "exclude": mapped_excl})
    if out_excl:
        filtered["exclusions"] = out_excl

    # ---- polarization ----
    # GROMACS topology에서는 [ polarization ] 지시자가 일반적으로 지원되지 않으므로
    # strict=true일 때만 에러로 처리하고, 기본(strict=false)에서는 스킵합니다.
    if extras.get("polarization"):
        msg = "polarization: directive not supported by standard grompp, skipped"
        if strict:
            raise ValueError(msg)
        warnings.append(msg)

    # ---- cmaptypes, restraints, other raw sections ----
    for sec, rows in extras.items():
        if sec in filtered or sec in ("constraints", "pairs", "exclusions", "polarization", "virtual_sites"):
            continue
        if not rows:
            continue
        # Under minimal allowlist, drop everything else early.
        sec_lower = sec.lower()
        if sec_lower not in ALLOWED and sec_lower != "impropers":
            warnings.append(f"{sec_lower}: skipped by minimal allowlist policy")
            continue
        # Restraint 섹션은 grompp에서 별도 include/define이 필요하고
        # 포맷이 다양해 자동 매핑이 위험하므로 기본(strict=false)에서는 스킵합니다.
        if sec_lower == "cmaptypes":
            msg = "cmaptypes: forcefield-level directive skipped by default"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        if sec_lower.endswith("_restraints") or "restraint" in sec_lower:
            msg = f"{sec_lower}: restraint directive skipped by default"
            if strict:
                raise ValueError(msg)
            warnings.append(msg)
            continue
        # Validate dihedrals/impropers minimal parameter counts for common funct types.
        if sec in ("dihedrals", "impropers"):
            out_rows: List[Dict[str, Any]] = []
            for row in rows:
                vals = row.get("values") if isinstance(row, dict) else None
                if not isinstance(vals, list) or len(vals) < 5:
                    msg = f"{sec}: invalid values payload {row}"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue
                funct = _try_int(vals[4]) or 0
                param_count = len(vals) - 5
                # Proper dihedral funct=1 expects phi,k,multiplicity (3 params) in GROMACS.
                if funct == 1 and param_count < 3:
                    msg = f"{sec}: funct=1 requires >=3 params, got {param_count} ({vals})"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue
                out_rows.append(row)

            if out_rows:
                # impropers 지시자는 grompp에서 invalid이므로 dihedrals로 합칩니다.
                target = "dihedrals" if sec == "impropers" else "dihedrals"
                filtered.setdefault(target, []).extend(out_rows)
            continue
        # Any remaining non-allowed section is dropped.
        if sec_lower in ALLOWED:
            filtered[sec_lower] = list(rows)

    # ---- virtual sites (any virtual_sites2/3/4/n captured under extras["virtual_sites"]) ----
    vs_rows = extras.get("virtual_sites") or []
    if vs_rows:
        # virtual sites require massless site beads and careful topology setup.
        # To keep default runs stable, skip virtual_sites unless strict.
        msg = "virtual_sites: directive skipped by default"
        if strict:
            # continue with validation+emit
            pass
        else:
            warnings.append(msg)
            vs_rows = []

    if vs_rows:
        vs_by_sec: Dict[str, List[Dict[str, Any]]] = {}
        seen_sites: Dict[str, set] = {}
        for row in vs_rows:
            sec_name = row.get("section", "virtual_sites2")
            parts = row.get("parts") or row.get("line", "").split()
            if not parts:
                msg = f"{sec_name}: empty virtual site line"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            # collect integer tokens and validate range
            int_tokens: List[int] = []
            for t in parts:
                i = _try_int(t)
                if i is not None:
                    int_tokens.append(i)

            if any(not _in_range(i, atom_count) for i in int_tokens):
                msg = f"{sec_name}: out-of-range indices in {parts}"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            # minimal integer-token count by virtual site type
            # - virtual_sites2: i j k funct a  -> >=3 integer indices
            # - virtual_sites3: i j k l funct a -> >=4 integer indices
            # - virtual_sites4: i j k l m funct a -> >=5 integer indices
            if sec_name.startswith("virtual_sites2") and len(int_tokens) < 3:
                msg = f"{sec_name}: requires >=3 index tokens, got {len(int_tokens)} ({parts})"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            if sec_name.startswith("virtual_sites3") and len(int_tokens) < 4:
                msg = f"{sec_name}: requires >=4 index tokens, got {len(int_tokens)} ({parts})"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue
            if sec_name.startswith("virtual_sites4") and len(int_tokens) < 5:
                msg = f"{sec_name}: requires >=5 index tokens, got {len(int_tokens)} ({parts})"
                if strict:
                    raise ValueError(msg)
                warnings.append(msg)
                continue

            # funct별 파라미터 최소 개수 (가장 흔한 GROMACS 규칙)
            try:
                if sec_name.startswith("virtual_sites2") and len(parts) >= 4:
                    funct = _try_int(parts[3]) or 0
                    param_count = len(parts) - 4
                elif sec_name.startswith("virtual_sites3") and len(parts) >= 5:
                    funct = _try_int(parts[4]) or 0
                    param_count = len(parts) - 5
                elif sec_name.startswith("virtual_sites4") and len(parts) >= 6:
                    funct = _try_int(parts[5]) or 0
                    param_count = len(parts) - 6
                else:
                    funct = 0
                    param_count = 0
                # virtual_sites3 funct=1 in GROMACS requires 2 params (site definition),
                # others use >=1 param.
                if funct == 1:
                    need = 2 if sec_name.startswith("virtual_sites3") else 1
                    if param_count < need:
                        msg = f"{sec_name}: funct=1 requires >={need} params, got {param_count} ({parts})"
                        if strict:
                            raise ValueError(msg)
                        warnings.append(msg)
                        continue
                elif funct != 0 and param_count < 2:
                    msg = f"{sec_name}: funct={funct} requires >=2 params, got {param_count} ({parts})"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue
            except Exception:
                pass

            # duplicate site id check (first integer token)
            site_id = int_tokens[0] if int_tokens else None
            if site_id is not None:
                seen_sites.setdefault(sec_name, set())
                if site_id in seen_sites[sec_name]:
                    msg = f"{sec_name}: duplicate site id {site_id}"
                    if strict:
                        raise ValueError(msg)
                    warnings.append(msg)
                    continue
                seen_sites[sec_name].add(site_id)

            vs_by_sec.setdefault(sec_name, []).append(row)

        if vs_by_sec:
            filtered["virtual_sites"] = []
            for sec_name, rows in vs_by_sec.items():
                filtered["virtual_sites"].extend(rows)

    # Final allowlist enforcement (defensive)
    filtered = {k: v for k, v in filtered.items() if k in ALLOWED and v}
    return filtered, warnings
