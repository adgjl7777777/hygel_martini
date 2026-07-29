from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from ..config import (
    ParamLine, 
    TypedRecord, 
    MergedVariant, 
    write_text,
    RMSD_RE
)
from .core import _sorted_pair, _build_graph, shortest_path_len

def split_main_and_comment(raw: str) -> Tuple[str, str]:
    stripped = raw.lstrip()
    while stripped.startswith(";"):
        stripped = stripped[1:].lstrip()
    if ";" in stripped:
        main, comment = stripped.split(";", 1)
        return main.strip(), comment.strip()
    return stripped.strip(), ""

def parse_param_line(raw: str, section: str, n_idx: int) -> Optional[ParamLine]:
    stripped = raw.strip()
    if not stripped:
        return None
    main, comment = split_main_and_comment(raw)
    if not main or not main[0].isdigit():
        return None
    parts = main.split()
    if len(parts) < n_idx + 1:
        return None

    rmsd = None
    if comment:
        match = RMSD_RE.search(comment)
        if match:
            try:
                rmsd = float(match.group(1))
            except ValueError:
                rmsd = None

    try:
        indices = tuple(int(parts[idx]) for idx in range(n_idx))
    except ValueError:
        return None

    return ParamLine(
        section=section,
        indices=indices,
        tokens=tuple(parts[n_idx:]),
        commented=stripped.startswith(";"),
        inline_comment=comment,
        rmsd=rmsd,
        raw=raw.rstrip("\n"),
    )

def parse_gmx_out_itp(path: Path) -> Dict[str, List[ParamLine]]:
    header_map = {
        "bonds": "bonds",
        "bondtypes": "bonds",
        "constraints": "constraints",
        "constrainttypes": "constraints",
        "angles": "angles",
        "angletypes": "angles",
        "dihedrals": "dihedrals",
        "dihedraltypes": "dihedrals",
        "impropers": "impropers",
        "impropertypes": "impropers",
    }
    parsed: Dict[str, List[ParamLine]] = {
        section: [] for section in ("bonds", "constraints", "angles", "dihedrals", "impropers")
    }
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped.strip("[]").strip().lower()
            current = header_map.get(header)
            continue
        if current is None:
            continue
        n_idx = 2 if current in {"bonds", "constraints"} else 3 if current == "angles" else 4
        line = parse_param_line(raw, current, n_idx)
        if line is not None:
            parsed[current].append(line)
    return parsed

def summarize_itp(path: Path) -> Dict[str, object]:
    parsed = parse_gmx_out_itp(path)

    def _payload(line: ParamLine) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "indices": list(line.indices),
            "params": list(line.tokens),
            "commented": line.commented,
            "comment": line.inline_comment,
        }
        if line.rmsd is not None:
            payload["rmsd"] = line.rmsd
        return payload

    return {
        "path": str(path),
        "counts": {section: len(lines) for section, lines in parsed.items()},
        "bonds": [_payload(line) for line in parsed["bonds"]],
        "constraints": [_payload(line) for line in parsed["constraints"]],
        "angles": [_payload(line) for line in parsed["angles"]],
        "dihedrals": [_payload(line) for line in parsed["dihedrals"]],
        "impropers": [_payload(line) for line in parsed["impropers"]],
    }

def choose_best_rmsd_uncomment(lines: List[ParamLine]) -> List[ParamLine]:
    grouped: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for index, line in enumerate(lines):
        grouped[line.indices].append(index)

    updated = list(lines)
    for positions in grouped.values():
        best_index = None
        best_value = math.inf
        for position in positions:
            value = updated[position].rmsd if updated[position].rmsd is not None else math.inf
            if value < best_value:
                best_value = value
                best_index = position
        if best_index is None or math.isinf(best_value):
            continue
        for position in positions:
            line = updated[position]
            updated[position] = ParamLine(
                section=line.section,
                indices=line.indices,
                tokens=line.tokens,
                commented=position != best_index,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                raw=line.raw,
            )
    return updated

def typed_records_for_result(
    itp_path: Path,
    case_path: Path,
    label_overrides: Dict[str, Dict[str, List[str]]],
) -> List[TypedRecord]:
    from .loader import build_bead_maps
 # Circular import avoidance if needed, but pipeline is main
    case = json.loads(case_path.read_text(encoding="utf-8"))
    parsed = parse_gmx_out_itp(itp_path)
    label_map, type_map, backbone_beads = build_bead_maps(case, label_overrides)

    graph = _build_graph(
        {_sorted_pair(*line.indices) for line in parsed["bonds"]}
        | {_sorted_pair(*line.indices) for line in parsed["constraints"]}
    )
    angle_lines = choose_best_rmsd_uncomment(parsed["angles"])
    source_tag = f"{case.get('sequence_stem', case_path.parent.name)}:{itp_path.parent.name}"

    def category(indices: Tuple[int, ...]) -> str:
        return "WITH_BACKBONE" if any(index in backbone_beads for index in indices) else "WITHOUT_BACKBONE"

    def map_labels(indices: Tuple[int, ...]) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        try:
            display = tuple(label_map[index] for index in indices)
            types = tuple(type_map[index] for index in indices)
        except KeyError as exc:
            raise KeyError(f"{itp_path}: bead index {exc.args[0]} is not present in the case bead map.") from exc
        return display, types

    section_map = {
        "bonds": "bondtypes",
        "constraints": "constrainttypes",
        "angles": "angletypes",
        "dihedrals": "dihedraltypes",
        "impropers": "impropertypes",
    }
    records: List[TypedRecord] = []

    for section_name in ("bonds", "constraints"):
        for line in parsed[section_name]:
            display, types = map_labels(line.indices)
            records.append(
                TypedRecord(
                    section=section_map[section_name],
                    category=category(line.indices),
                    angle_dist="",
                    type_names=types,
                    display_labels=display,
                    indices=line.indices,
                    tokens=line.tokens,
                    commented=line.commented,
                    inline_comment=line.inline_comment,
                    rmsd=line.rmsd,
                    source_tag=source_tag,
                    source_path=str(itp_path),
                )
            )

    for line in angle_lines:
        display, types = map_labels(line.indices)
        endpoint_dist = shortest_path_len(graph, line.indices[0], line.indices[2])
        records.append(
            TypedRecord(
                section="angletypes",
                category=category(line.indices),
                angle_dist="DIST_LE2" if endpoint_dist is not None and endpoint_dist <= 2 else "DIST_GE3",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )

    for line in parsed["dihedrals"]:
        display, types = map_labels(line.indices)
        records.append(
            TypedRecord(
                section="dihedraltypes",
                category=category(line.indices),
                angle_dist="",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )
    for line in parsed["impropers"]:
        display, types = map_labels(line.indices)
        records.append(
            TypedRecord(
                section="impropertypes",
                category=category(line.indices),
                angle_dist="",
                type_names=types,
                display_labels=display,
                indices=line.indices,
                tokens=line.tokens,
                commented=line.commented,
                inline_comment=line.inline_comment,
                rmsd=line.rmsd,
                source_tag=source_tag,
                source_path=str(itp_path),
            )
        )
    return records

def merge_records(records: List[TypedRecord]) -> Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]]:
    grouped: Dict[Tuple[str, str, str, Tuple[str, ...]], List[TypedRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.section, record.category, record.angle_dist, record.type_names)].append(record)

    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]] = {}
    for key, group in grouped.items():
        variants_by_signature: Dict[Tuple[Tuple[str, ...], bool, str], List[TypedRecord]] = defaultdict(list)
        for record in group:
            variants_by_signature[(record.tokens, record.commented, record.inline_comment.strip())].append(record)

        items = []
        for records_in_variant in variants_by_signature.values():
            sample = records_in_variant[0]
            items.append(
                {
                    "sample": sample,
                    "display_labels": sorted({record.display_labels for record in records_in_variant}),
                    "sources": sorted({record.source_tag for record in records_in_variant}),
                    "indices_examples": sorted({record.indices for record in records_in_variant}),
                    "inline_comments": sorted({record.inline_comment.strip() for record in records_in_variant if record.inline_comment.strip()}),
                    "rmsd_values": [record.rmsd for record in records_in_variant if record.rmsd is not None],
                }
            )

        def score(item: Dict[str, Any]) -> Tuple[float, int, float, str]:
            sample = item["sample"]
            if sample.section == "angletypes":
                rmsd = min(item["rmsd_values"]) if item["rmsd_values"] else math.inf
                return (0 if not sample.commented else 1, 0 if item["rmsd_values"] else 1, rmsd, sample.source_tag)
            return (0 if not sample.commented else 1, 0, 0.0, sample.source_tag)

        primary_item = min(items, key=score)
        variants: List[MergedVariant] = []
        for item in sorted(items, key=score):
            sample = item["sample"]
            variants.append(
                MergedVariant(
                    section=sample.section,
                    category=sample.category,
                    angle_dist=sample.angle_dist,
                    type_names=sample.type_names,
                    display_labels=item["display_labels"],
                    tokens=sample.tokens,
                    commented=sample.commented if item is primary_item else True,
                    sources=item["sources"],
                    indices_examples=item["indices_examples"],
                    inline_comments=item["inline_comments"],
                    rmsd_values=item["rmsd_values"],
                    primary=item is primary_item,
                )
            )
        merged[key] = variants

    return merged

def _format_type_names(type_names: Tuple[str, ...], widths: Tuple[int, ...]) -> str:
    return " ".join(f"{value:<{width}}" for value, width in zip(type_names, widths))

def line_from_variant(variant: MergedVariant) -> str:
    widths = (8, 8, 8, 8)
    prefix = _format_type_names(variant.type_names, widths[: len(variant.type_names)]).rstrip()
    main = f"{';' if variant.commented else ''}{prefix} {' '.join(variant.tokens)}".rstrip()
    comment_parts = []
    if variant.display_labels:
        comment_parts.append("labels=" + " | ".join(" ".join(entry) for entry in variant.display_labels))
    if variant.inline_comments:
        comment_parts.append("comments=" + " | ".join(variant.inline_comments))
    if variant.rmsd_values:
        comment_parts.append("rmsd=" + ",".join(f"{value:.3f}" for value in sorted(set(variant.rmsd_values))))
    if variant.sources:
        comment_parts.append("sources=" + ",".join(variant.sources))
    if variant.indices_examples:
        examples = " | ".join("-".join(str(value) for value in indices) for indices in variant.indices_examples[:5])
        comment_parts.append(f"indices={examples}")
    return main + (" ; " + " ; ".join(comment_parts) if comment_parts else "")

def write_merged_forcefield(
    path: Path,
    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]],
    root: Path,
    label_map_path: Optional[Path],
) -> None:
    lines = [
        "; Auto-generated merged Bartender forcefield summary",
        f"; root = {root}",
        f"; label_map = {label_map_path if label_map_path else '(default token-based labels)'}",
        "; The first uncommented line per type key is the selected representative.",
        "",
    ]

    section_order = ("bondtypes", "constrainttypes", "angletypes", "dihedraltypes", "impropertypes")
    category_order = ("WITH_BACKBONE", "WITHOUT_BACKBONE")

    for section in section_order:
        lines.append(f"[ {section} ]")
        for category in category_order:
            if section == "angletypes":
                for angle_dist in ("DIST_LE2", "DIST_GE3"):
                    keys = [key for key in merged if key[0] == section and key[1] == category and key[2] == angle_dist]
                    if not keys:
                        continue
                    lines.append(f"; {category} / {angle_dist}")
                    for key in sorted(keys, key=lambda item: item[3]):
                        for variant in merged[key]:
                            lines.append(line_from_variant(variant))
                    lines.append("")
            else:
                keys = [key for key in merged if key[0] == section and key[1] == category]
                if not keys:
                    continue
                lines.append(f"; {category}")
                for key in sorted(keys, key=lambda item: item[3]):
                    for variant in merged[key]:
                        lines.append(line_from_variant(variant))
                lines.append("")
        lines.append("")

    write_text(path, "\n".join(lines).rstrip() + "\n")

def merged_summary_payload(
    root: Path,
    merged: Dict[Tuple[str, str, str, Tuple[str, ...]], List[MergedVariant]],
    skipped: List[Dict[str, str]],
) -> Dict[str, object]:
    groups = []
    for key, variants in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        groups.append(
            {
                "section": key[0],
                "category": key[1],
                "angle_dist": key[2],
                "type_names": list(key[3]),
                "variant_count": len(variants),
                "selected_variant": next(index for index, variant in enumerate(variants) if variant.primary),
                "variants": [
                    {
                        "primary": variant.primary,
                        "commented": variant.commented,
                        "tokens": list(variant.tokens),
                        "display_labels": [list(entry) for entry in variant.display_labels],
                        "sources": list(variant.sources),
                        "indices_examples": [list(entry) for entry in variant.indices_examples],
                        "inline_comments": list(variant.inline_comments),
                        "rmsd_values": list(variant.rmsd_values),
                    }
                    for variant in variants
                ],
            }
        )

    return {"root": str(root), "group_count": len(groups), "groups": groups, "skipped": skipped}
