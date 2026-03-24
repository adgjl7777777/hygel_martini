from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.utils import parse_csv_list
from ..polymer_maker.maker import build_polymer as build_polymer_structure
from ..polymer_maker.maker import load_monomer_library


SECTION_HEADERS = {
    "BEADS",
    "BONDS",
    "CONSTRAINTS",
    "ANGLES",
    "DIHEDRALS",
    "IMPROPERS",
}
CONNECTION_CUTOFF = 2.2
RMSD_RE = re.compile(r"rmsd:\s*([0-9]*\.?[0-9]+)")


@dataclass(frozen=True)
class WeightedAtomRef:
    atom_index: int
    denominator: int = 1

    @property
    def weight(self) -> Fraction:
        return Fraction(1, self.denominator)

    def format(self) -> str:
        if self.denominator == 1:
            return str(self.atom_index)
        return f"{self.atom_index}/{self.denominator}"


@dataclass
class ValidationReport:
    target: str
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [f"Target: {self.target}", f"Status: {'OK' if self.ok else 'FAILED'}"]
        if self.problems:
            lines.append("Problems:")
            lines.extend(f"- {item}" for item in self.problems)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in self.warnings)
        return "\n".join(lines) + "\n"


@dataclass
class MonomerTemplate:
    path: Path
    preamble: List[str]
    beads: "OrderedDict[int, List[WeightedAtomRef]]"
    bonds: List[Tuple[int, int]]
    constraints: List[Tuple[int, int]]
    angles: List[Tuple[int, int, int]]
    dihedrals: List[Tuple[int, int, int, int]]
    impropers: List[Tuple[int, int, int, int]]

    @property
    def bead_count(self) -> int:
        return len(self.beads)

    @property
    def atom_count(self) -> int:
        if not self.beads:
            return 0
        return max(ref.atom_index for refs in self.beads.values() for ref in refs)


@dataclass(frozen=True)
class ConnectionMetadata:
    head_carbon: int
    tail_carbon: int
    head_br: int
    tail_br: int
    left_connection_bead: int
    right_connection_bead: int


@dataclass
class PolymerInputBundle:
    base: MonomerTemplate
    augmented: MonomerTemplate
    base_text: str
    augmented_text: str
    base_report: ValidationReport
    augmented_report: ValidationReport
    connection_bonds: List[Tuple[int, int]]
    connection_beads: List[int]


@dataclass(frozen=True)
class ParamLine:
    section: str
    indices: Tuple[int, ...]
    tokens: Tuple[str, ...]
    commented: bool
    inline_comment: str
    rmsd: Optional[float]
    raw: str


@dataclass(frozen=True)
class TypedRecord:
    section: str
    category: str
    angle_dist: str
    type_names: Tuple[str, ...]
    display_labels: Tuple[str, ...]
    indices: Tuple[int, ...]
    tokens: Tuple[str, ...]
    commented: bool
    inline_comment: str
    rmsd: Optional[float]
    source_tag: str
    source_path: str


@dataclass
class MergedVariant:
    section: str
    category: str
    angle_dist: str
    type_names: Tuple[str, ...]
    display_labels: List[Tuple[str, ...]]
    tokens: Tuple[str, ...]
    commented: bool
    sources: List[str]
    indices_examples: List[Tuple[int, ...]]
    inline_comments: List[str]
    rmsd_values: List[float]
    primary: bool

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def shell_assign(name: str, value: str) -> str:
    return f'{name}={shlex.quote(value)}'


def resolve_under_base(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def default_workdir_name(relaxation: str, md: str) -> str:
    if md == "bartender":
        if relaxation == "xtb":
            return "relax_xtb_geoopt"
        if relaxation == "orca":
            return "relax_orca_geoopt"
        return "relax_input_geometry"
    if md == "off":
        if relaxation == "xtb":
            return "relax_xtb_geoopt_only"
        if relaxation == "orca":
            return "relax_orca_geoopt_only"
        return "polymer_geometry_only"
    if md == "xtb":
        if relaxation == "xtb":
            return "relax_xtb_geoopt_xtb_nvt"
        if relaxation == "orca":
            return "relax_orca_geoopt_xtb_nvt"
        return "relax_xtb_nvt"
    raise ValueError(f"Unsupported md mode: {md}")


def _normalize_pipeline_mode(value: Any, default: str, field_name: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        if value is False:
            return "off"
        raise ValueError(f"{field_name} must be one of the documented string modes, not boolean true")
    return str(value).strip().lower()


def resolve_pipeline_modes(pipeline_cfg: Dict[str, Any]) -> Dict[str, str]:
    top_relaxation = pipeline_cfg.get("relaxation")
    top_md = pipeline_cfg.get("md")
    if (
        "workdir_name" in pipeline_cfg
        or top_md is not None
        or (top_relaxation is not None and not isinstance(top_relaxation, dict))
    ):
        relaxation = _normalize_pipeline_mode(
            top_relaxation,
            "xtb",
            "bartender_pipeline.relaxation",
        )
        md = _normalize_pipeline_mode(top_md, "bartender", "bartender_pipeline.md")
        workdir_name = str(
            pipeline_cfg.get("workdir_name") or default_workdir_name(relaxation, md)
        ).strip()
    else:
        mode_cfg = pipeline_cfg.get("mode")
        if isinstance(mode_cfg, dict):
            relaxation = _normalize_pipeline_mode(
                mode_cfg.get("relaxation"),
                "xtb",
                "bartender_pipeline.mode.relaxation",
            )
            md = _normalize_pipeline_mode(
                mode_cfg.get("md"),
                "bartender",
                "bartender_pipeline.mode.md",
            )
            workdir_name = str(
                mode_cfg.get("workdir_name") or default_workdir_name(relaxation, md)
            ).strip()
        else:
            legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
            if not isinstance(legacy_relax_cfg, dict):
                legacy_relax_cfg = {}
            legacy_bartender_cfg = pipeline_cfg.get("bartender", {})
            if not isinstance(legacy_bartender_cfg, dict):
                legacy_bartender_cfg = {}

            backend = _normalize_pipeline_mode(
                legacy_relax_cfg.get("backend"),
                "xtb",
                "bartender_pipeline.relaxation.backend",
            )
            if backend == "xtb":
                relaxation = "xtb"
            elif backend in {"orca", "orca_then_xtb"}:
                relaxation = "orca"
            elif backend == "off":
                relaxation = "off"
            else:
                raise ValueError(f"Unsupported legacy relaxation backend: {backend}")

            geometry_source = str(
                legacy_bartender_cfg.get("geometry_source", "polymer_xyz")
            ).strip()
            md = "xtb" if geometry_source == "relaxation_output" else "bartender"
            workdir_name = str(
                legacy_relax_cfg.get("workdir_name")
                or default_workdir_name(relaxation, md)
            ).strip()

    if relaxation not in {"xtb", "orca", "off"}:
        raise ValueError("bartender_pipeline.relaxation must be one of: xtb, orca, off")
    if md not in {"bartender", "xtb", "off"}:
        raise ValueError("bartender_pipeline.md must be one of: bartender, xtb, off")

    return {
        "relaxation": relaxation,
        "md": md,
        "workdir_name": workdir_name or default_workdir_name(relaxation, md),
    }


def resolve_spin_state(
    uhf_value: Any,
    multiplicity_value: Any,
    *,
    label: str,
) -> Tuple[int, int]:
    uhf = None if uhf_value is None else int(uhf_value)
    multiplicity = None if multiplicity_value is None else int(multiplicity_value)

    if uhf is None and multiplicity is None:
        return 0, 1
    if uhf is None:
        uhf = multiplicity - 1
    if multiplicity is None:
        multiplicity = uhf + 1

    if uhf < 0:
        raise ValueError(f"{label}: uhf must be >= 0")
    if multiplicity < 1:
        raise ValueError(f"{label}: multiplicity must be >= 1")
    if multiplicity != uhf + 1:
        raise ValueError(
            f"{label}: multiplicity ({multiplicity}) must equal uhf + 1 ({uhf + 1})"
        )
    return uhf, multiplicity


def normalize_monomer_configs(
    raw_monomers: Dict[str, Any],
    legacy_init_templates: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for token, raw_entry in raw_monomers.items():
        if isinstance(raw_entry, str):
            entry: Dict[str, Any] = {"xyz": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
        else:
            raise TypeError(f"monomers.{token} must be a string or mapping")

        xyz = entry.get("xyz")
        if not xyz:
            raise ValueError(f"monomers.{token}.xyz is required")
        uhf, multiplicity = resolve_spin_state(
            entry.get("uhf"),
            entry.get("multiplicity"),
            label=f"monomers.{token}",
        )
        normalized[token] = {
            "xyz": str(xyz),
            "init_template": entry.get("init_template", legacy_init_templates.get(token)),
            "charge": int(entry.get("charge", 0)),
            "uhf": uhf,
            "multiplicity": multiplicity,
        }
    return normalized


def resolve_case_electronic_state(
    tokens: Sequence[str],
    monomer_cfg: Dict[str, Dict[str, Any]],
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, int]:
    inferred_charge = sum(int(monomer_cfg[token]["charge"]) for token in tokens)
    inferred_uhf = sum(int(monomer_cfg[token]["uhf"]) for token in tokens)
    state_cfg = pipeline_cfg.get("electronic_state")
    if state_cfg is None and isinstance(pipeline_cfg.get("relaxation"), dict):
        state_cfg = pipeline_cfg["relaxation"]
    if state_cfg is None:
        state_cfg = {}
    if not isinstance(state_cfg, dict):
        raise TypeError("bartender_pipeline.electronic_state must be a mapping")

    charge_raw = state_cfg.get("charge")
    charge = inferred_charge if charge_raw is None else int(charge_raw)

    if state_cfg.get("uhf") is None and state_cfg.get("multiplicity") is None:
        uhf = inferred_uhf
        multiplicity = inferred_uhf + 1
    else:
        uhf, multiplicity = resolve_spin_state(
            state_cfg.get("uhf"),
            state_cfg.get("multiplicity"),
            label="bartender_pipeline.electronic_state",
        )

    return {
        "charge": charge,
        "uhf": uhf,
        "multiplicity": multiplicity,
        "inferred_charge": inferred_charge,
        "inferred_uhf": inferred_uhf,
    }


def resolve_optional_path(base_dir: Path, raw_value: Any) -> Optional[Path]:
    value = str(raw_value or "").strip()
    if not value:
        return None
    return resolve_under_base(base_dir, value)


def resolve_xtb_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
    if not isinstance(legacy_relax_cfg, dict):
        legacy_relax_cfg = {}

    xtb_cfg = pipeline_cfg.get("xtb")
    if xtb_cfg is None:
        xtb_cfg = legacy_relax_cfg.get("xtb", {})
    if not isinstance(xtb_cfg, dict):
        raise TypeError("bartender_pipeline.xtb must be a mapping")
    md_cfg = xtb_cfg.get("md", {})
    if not isinstance(md_cfg, dict):
        raise TypeError("bartender_pipeline.xtb.md must be a mapping")

    solvent_model = str(xtb_cfg.get("solvent_model", "alpb")).strip().lower() or "off"
    if solvent_model not in {"off", "alpb", "gbsa"}:
        raise ValueError("bartender_pipeline.xtb.solvent_model must be one of: off, alpb, gbsa")

    return {
        "env_script": str(
            xtb_cfg.get("env_script", legacy_relax_cfg.get("xtb_env_script", ""))
        ).strip(),
        "binary": str(xtb_cfg.get("binary", legacy_relax_cfg.get("xtb_binary", "xtb"))).strip(),
        "gfn": int(xtb_cfg.get("gfn", 2)),
        "parallel": int(xtb_cfg.get("parallel", legacy_relax_cfg.get("nprocs", 32))),
        "opt_level": str(xtb_cfg.get("opt_level", "normal")).strip(),
        "opt_cycles": int(xtb_cfg.get("opt_cycles", 10000)),
        "acc": float(xtb_cfg.get("acc", 1.0)),
        "etemp": float(xtb_cfg.get("etemp", 300.0)),
        "solvent_model": solvent_model,
        "solvent": str(xtb_cfg.get("solvent", legacy_relax_cfg.get("solvent", "water"))).strip(),
        "solvent_reference": str(xtb_cfg.get("solvent_reference", "")).strip(),
        "md_input_template_path": str(xtb_cfg.get("md_input_template_path", "")).strip(),
        "md_temp_k": float(md_cfg.get("temp_k", legacy_relax_cfg.get("temp_k", 310.0))),
        "md_time_ps": float(md_cfg.get("time_ps", legacy_relax_cfg.get("time_ps", 5000))),
        "md_dump_fs": float(md_cfg.get("dump_fs", 50.0)),
        "md_step_fs": float(md_cfg.get("step_fs", 4.0)),
        "md_velo": parse_bool(md_cfg.get("velo", False)),
        "md_hmass": int(md_cfg.get("hmass", 4)),
        "md_shake": int(md_cfg.get("shake", 2)),
        "md_sccacc": float(md_cfg.get("sccacc", 2.0)),
        "md_restart": parse_bool(md_cfg.get("restart", False)),
    }


def resolve_orca_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    legacy_relax_cfg = pipeline_cfg.get("relaxation", {})
    if not isinstance(legacy_relax_cfg, dict):
        legacy_relax_cfg = {}

    orca_cfg = pipeline_cfg.get("orca")
    if orca_cfg is None:
        orca_cfg = legacy_relax_cfg.get("orca", {})
    if not isinstance(orca_cfg, dict):
        raise TypeError("bartender_pipeline.orca must be a mapping")
    return {
        "binary": str(orca_cfg.get("binary", legacy_relax_cfg.get("orca_binary", "orca"))).strip(),
        "nprocs": int(orca_cfg.get("nprocs", legacy_relax_cfg.get("nprocs", 32))),
        "method_line": str(
            orca_cfg.get(
                "method_line",
                f"{legacy_relax_cfg.get('orca_method', 'r2scan-3c')} CPCM({legacy_relax_cfg.get('solvent', 'water')}) Opt TightSCF",
            )
        ).strip(),
        "max_iter": int(orca_cfg.get("max_iter", 300)),
        "input_template_path": str(orca_cfg.get("input_template_path", "")).strip(),
    }


def render_xtb_md_input(md_mode: str, xtb_cfg: Dict[str, Any], template_text: Optional[str]) -> str:
    if template_text is not None:
        return template_text.rstrip() + "\n"
    if md_mode != "nvt":
        raise ValueError("xTB MD input generation only supports md_mode 'nvt'")
    return (
        "$md\n"
        f" temp={xtb_cfg['md_temp_k']:.3f}\n"
        f" time={xtb_cfg['md_time_ps']:.3f}\n"
        f" dump={xtb_cfg['md_dump_fs']:.3f}\n"
        f" step={xtb_cfg['md_step_fs']:.3f}\n"
        f" velo={'true' if xtb_cfg['md_velo'] else 'false'}\n"
        " nvt=true\n"
        f" hmass={xtb_cfg['md_hmass']}\n"
        f" shake={xtb_cfg['md_shake']}\n"
        f" sccacc={xtb_cfg['md_sccacc']:.3f}\n"
        f" restart={'true' if xtb_cfg['md_restart'] else 'false'}\n"
        "$end\n"
    )


def render_orca_input(
    local_xyz_name: str,
    state: Dict[str, Any],
    orca_cfg: Dict[str, Any],
    template_text: Optional[str],
) -> str:
    if template_text is not None:
        return template_text.rstrip() + "\n\n" + (
            f"* xyzfile {int(state['charge'])} {int(state['multiplicity'])} {local_xyz_name}\n"
        )
    method_line = orca_cfg["method_line"].strip()
    if not method_line.startswith("!"):
        method_line = f"! {method_line}"
    return (
        f"{method_line}\n"
        f"%pal nprocs {int(orca_cfg['nprocs'])} end\n\n"
        "%geom\n"
        f"   MaxIter {int(orca_cfg['max_iter'])}\n"
        "end\n\n"
        f"* xyzfile {int(state['charge'])} {int(state['multiplicity'])} {local_xyz_name}\n"
    )


def render_xyz_traj_to_pdb_converter() -> str:
    return (
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def parse_frames(path: Path):\n"
        "    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()\n"
        "    index = 0\n"
        "    while index < len(lines):\n"
        "        while index < len(lines) and not lines[index].strip():\n"
        "            index += 1\n"
        "        if index >= len(lines):\n"
        "            break\n"
        "        natoms = int(lines[index].strip())\n"
        "        start = index + 2\n"
        "        stop = start + natoms\n"
        "        atom_lines = lines[start:stop]\n"
        "        if len(atom_lines) != natoms:\n"
        "            raise SystemExit(f'Malformed trajectory frame near line {index + 1}')\n"
        "        yield atom_lines\n"
        "        index = stop\n"
        "\n"
        "\n"
        "def pdb_atom_line(atom_index: int, symbol: str, x: float, y: float, z: float) -> str:\n"
        "    atom_name = symbol[:2].upper().rjust(2)\n"
        "    return (\n"
        "        f\"ATOM  {atom_index:5d} {atom_name:<4} MOL A{1:4d}    \"\n"
        "        f\"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {symbol[:2].upper():>2}\"\n"
        "    )\n"
        "\n"
        "\n"
        "def main(argv: list[str]) -> int:\n"
        "    if len(argv) != 3:\n"
        "        raise SystemExit('Usage: xtb_traj_to_pdb.py input.xyztraj output.pdb')\n"
        "    input_path = Path(argv[1])\n"
        "    output_path = Path(argv[2])\n"
        "    out_lines = []\n"
        "    for model_index, atom_lines in enumerate(parse_frames(input_path), start=1):\n"
        "        out_lines.append(f'MODEL     {model_index}')\n"
        "        for atom_index, raw in enumerate(atom_lines, start=1):\n"
        "            parts = raw.split()\n"
        "            if len(parts) < 4:\n"
        "                raise SystemExit(f'Malformed atom line: {raw}')\n"
        "            symbol = parts[0]\n"
        "            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])\n"
        "            out_lines.append(pdb_atom_line(atom_index, symbol, x, y, z))\n"
        "        out_lines.append('ENDMDL')\n"
        "    output_path.write_text('\\n'.join(out_lines) + '\\n', encoding='utf-8')\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(sys.argv))\n"
    )


def normalize_sequence(sequence: Sequence[str] | str) -> List[str]:
    if isinstance(sequence, str):
        text = sequence.strip()
        if not text:
            raise ValueError("Sequence is empty.")
        if "," in text:
            tokens = [token.strip() for token in text.split(",") if token.strip()]
        elif " " in text:
            tokens = [token for token in text.split() if token]
        else:
            tokens = list(text)
    else:
        tokens = [str(token).strip() for token in sequence if str(token).strip()]
    if not tokens:
        raise ValueError("Sequence produced no tokens.")
    return tokens


def sequence_stem(tokens: Sequence[str]) -> str:
    if all(len(token) == 1 for token in tokens):
        return "".join(tokens)
    return "_".join(tokens)


def parse_sequence_entry(entry: Any, monomer_keys: set[str]) -> List[str]:
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            raise ValueError("Empty sequence entry is not allowed")
        if "," in text:
            tokens = parse_csv_list(text)
        elif " " in text:
            tokens = [token for token in text.split() if token]
        elif text in monomer_keys:
            tokens = [text]
        else:
            tokens = list(text)
    elif isinstance(entry, (list, tuple)):
        tokens = [str(token).strip() for token in entry if str(token).strip()]
    else:
        raise TypeError(f"Unsupported sequence entry type: {type(entry)!r}")

    if not tokens:
        raise ValueError("Sequence entry produced no tokens")
    return tokens


def build_sequence_jobs(system_cfg: Dict[str, Any], monomer_keys: set[str]) -> List[List[str]]:
    explicit_sequences = system_cfg.get("sequences")
    if explicit_sequences is not None:
        if not isinstance(explicit_sequences, list):
            raise ValueError("system.sequences must be a list when provided")
        jobs = [parse_sequence_entry(entry, monomer_keys) for entry in explicit_sequences]
        if not jobs:
            raise ValueError("system.sequences is empty")
        return jobs

    symbols = list(system_cfg["symbols"])
    lengths = list(system_cfg["lengths"])
    jobs: List[List[str]] = []
    for symbol in symbols:
        for repeat in lengths:
            jobs.append([symbol] * int(repeat))
    return jobs


def parse_xyz(path: Path) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty xyz: {path}")
    natoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ {path} declares {natoms} atoms but contains {len(atom_lines)} coordinates.")
    symbols: List[str] = []
    coords: List[Tuple[float, float, float]] = []
    for raw in atom_lines:
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed xyz atom line in {path}: {raw}")
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _split_csv(raw: str) -> List[str]:
    return [token.strip() for token in re.split(r"\s*,\s*", raw.strip()) if token.strip()]


def _parse_weighted_atom(token: str) -> WeightedAtomRef:
    match = re.fullmatch(r"(\d+)(?:/(\d+))?", token)
    if not match:
        raise ValueError(f"Malformed BEADS atom token: {token}")
    denominator = int(match.group(2)) if match.group(2) else 1
    if denominator < 1:
        raise ValueError(f"Invalid denominator in BEADS atom token: {token}")
    return WeightedAtomRef(atom_index=int(match.group(1)), denominator=denominator)


def _parse_section_ints(path: Path, line: str, expected: int) -> Tuple[int, ...]:
    values = tuple(int(token) for token in _split_csv(line))
    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} integers in line '{line}'")
    return values


def parse_bartender_inp(path: Path) -> MonomerTemplate:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    preamble: List[str] = []
    sections: Dict[str, List[str]] = {header: [] for header in SECTION_HEADERS}
    current: Optional[str] = None
    seen_header = False

    for raw in lines:
        stripped = raw.strip()
        if not seen_header:
            if stripped.upper() in SECTION_HEADERS:
                seen_header = True
                current = stripped.upper()
                continue
            preamble.append(raw)
            continue

        if not stripped or stripped.startswith("#"):
            continue

        header = stripped.upper()
        if header in SECTION_HEADERS:
            current = header
            continue
        if current is None:
            continue
        sections[current].append(stripped)

    if not sections["BEADS"]:
        raise ValueError(f"{path} has no BEADS section.")

    beads: "OrderedDict[int, List[WeightedAtomRef]]" = OrderedDict()
    for line in sections["BEADS"]:
        match = re.match(r"^(\d+)\s+(.+)$", line)
        if not match:
            raise ValueError(f"{path}: malformed BEADS line '{line}'")
        bead_id = int(match.group(1))
        refs = [_parse_weighted_atom(token) for token in _split_csv(match.group(2))]
        beads[bead_id] = refs

    bonds = [tuple(_parse_section_ints(path, line, 2)) for line in sections["BONDS"]]
    constraints = [tuple(_parse_section_ints(path, line, 2)) for line in sections["CONSTRAINTS"]]
    angles = [tuple(_parse_section_ints(path, line, 3)) for line in sections["ANGLES"]]
    dihedrals = [tuple(_parse_section_ints(path, line, 4)) for line in sections["DIHEDRALS"]]
    impropers = [tuple(_parse_section_ints(path, line, 4)) for line in sections["IMPROPERS"]]

    return MonomerTemplate(
        path=path,
        preamble=preamble,
        beads=beads,
        bonds=[(int(a), int(b)) for a, b in bonds],
        constraints=[(int(a), int(b)) for a, b in constraints],
        angles=[(int(a), int(b), int(c)) for a, b, c in angles],
        dihedrals=[(int(a), int(b), int(c), int(d)) for a, b, c, d in dihedrals],
        impropers=[(int(a), int(b), int(c), int(d)) for a, b, c, d in impropers],
    )


def _weighted_atom_owners(template: MonomerTemplate) -> Dict[int, List[WeightedAtomRef]]:
    owners: Dict[int, List[WeightedAtomRef]] = defaultdict(list)
    for refs in template.beads.values():
        for ref in refs:
            owners[ref.atom_index].append(ref)
    return owners


def validate_template(template: MonomerTemplate, xyz_path: Path) -> ValidationReport:
    symbols, _ = parse_xyz(xyz_path)
    natoms = len(symbols)
    report = ValidationReport(target=str(template.path))

    owners = _weighted_atom_owners(template)
    if template.atom_count != natoms:
        report.problems.append(
            f"Template max atom index is {template.atom_count}, but xyz atom count is {natoms}."
        )

    for atom_index, refs in owners.items():
        if atom_index < 1 or atom_index > natoms:
            report.problems.append(f"Atom index {atom_index} is out of range for {xyz_path.name}.")
            continue
        total_weight = sum((ref.weight for ref in refs), Fraction(0, 1))
        if total_weight != Fraction(1, 1):
            report.problems.append(f"Atom {atom_index} has total bead weight {total_weight} instead of 1.")
        if len(refs) == 1 and refs[0].denominator != 1:
            report.problems.append(
                f"Atom {atom_index} appears once but uses fractional token {refs[0].format()}."
            )
        if len(refs) > 1 and any(ref.denominator != len(refs) for ref in refs):
            tokens = ", ".join(ref.format() for ref in refs)
            report.problems.append(
                f"Atom {atom_index} is duplicated but tokens do not match the n-way fractional rule: {tokens}"
            )

    missing_atoms = [atom_index for atom_index in range(1, natoms + 1) if atom_index not in owners]
    if missing_atoms:
        report.problems.append(
            f"Template misses atom indices: {missing_atoms[:20]}{'...' if len(missing_atoms) > 20 else ''}"
        )

    br_indices = [index for index, symbol in enumerate(symbols, start=1) if symbol == "Br"]
    if len(br_indices) < 2:
        report.problems.append(f"{xyz_path.name} must contain Br connectors, found {len(br_indices)}.")
    for br_index in br_indices:
        if br_index not in owners:
            report.problems.append(f"Br atom {br_index} is not assigned to any bead.")

    bead_ids = set(template.beads.keys())
    adjacency: Dict[int, set[int]] = {bead_id: set() for bead_id in bead_ids}
    for a, b in list(template.bonds) + list(template.constraints):
        if a not in bead_ids or b not in bead_ids:
            report.problems.append(f"Bond/constraint references unknown bead ids: {a},{b}")
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)

    isolated = sorted(bead_id for bead_id, neighbors in adjacency.items() if not neighbors)
    if isolated:
        report.problems.append(
            f"Each bead must participate in at least one bond/constraint. Isolated beads: {isolated}"
        )

    if bead_ids:
        start = next(iter(bead_ids))
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        disconnected = sorted(bead_id for bead_id in bead_ids if bead_id not in seen)
        if disconnected:
            report.problems.append(f"Bead graph is disconnected. Unreachable beads: {disconnected}")

    return report


def infer_connection_metadata(template: MonomerTemplate, xyz_path: Path) -> ConnectionMetadata:
    symbols, coords = parse_xyz(xyz_path)
    if len(symbols) < 2:
        raise ValueError(f"{xyz_path.name} must contain at least two atoms.")

    head_carbon = 1
    tail_carbon = 2
    head_br: Optional[int] = None
    tail_br: Optional[int] = None

    for atom_index, symbol in enumerate(symbols, start=1):
        if symbol != "Br":
            continue
        d_head = _distance(coords[head_carbon - 1], coords[atom_index - 1])
        d_tail = _distance(coords[tail_carbon - 1], coords[atom_index - 1])
        if d_head <= CONNECTION_CUTOFF:
            head_br = atom_index
        elif d_tail <= CONNECTION_CUTOFF:
            tail_br = atom_index

    if head_br is None or tail_br is None:
        raise ValueError(
            f"{xyz_path.name}: could not infer head/tail Br atoms near atoms 1 and 2 with cutoff {CONNECTION_CUTOFF} A."
        )

    def owner(atom_index: int, label: str) -> int:
        owners = [
            bead_id
            for bead_id, refs in template.beads.items()
            if any(ref.atom_index == atom_index for ref in refs)
        ]
        if len(owners) != 1:
            raise ValueError(
                f"{xyz_path.name}: expected exactly one bead for {label} atom {atom_index}, found {owners or 'none'}."
            )
        return owners[0]

    return ConnectionMetadata(
        head_carbon=head_carbon,
        tail_carbon=tail_carbon,
        head_br=head_br,
        tail_br=tail_br,
        left_connection_bead=owner(head_br, "head Br"),
        right_connection_bead=owner(tail_br, "tail Br"),
    )


def _sorted_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _canon_angle(i: int, j: int, k: int) -> Tuple[int, int, int]:
    return (i, j, k) if i <= k else (k, j, i)


def _build_graph(edges: Iterable[Tuple[int, int]]) -> Dict[int, set[int]]:
    graph: Dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _compute_within3(bead_ids: Sequence[int], graph: Dict[int, set[int]]) -> Dict[int, set[int]]:
    within3: Dict[int, set[int]] = {}
    for center in bead_ids:
        one = graph.get(center, set())
        two: set[int] = set()
        for node in one:
            two.update(graph.get(node, set()))
        three: set[int] = set()
        for node in two:
            three.update(graph.get(node, set()))
        values = set(one) | two | three
        values.discard(center)
        within3[center] = values
    return within3


def _bfs_dists(graph: Dict[int, set[int]], start: int, max_depth: int = 6) -> Dict[int, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if dist[current] >= max_depth:
            continue
        for neighbor in graph.get(current, set()):
            if neighbor not in dist:
                dist[neighbor] = dist[current] + 1
                queue.append(neighbor)
    return dist


def _compute_dist_map(bead_ids: Sequence[int], graph: Dict[int, set[int]]) -> Dict[int, Dict[int, int]]:
    return {bead_id: _bfs_dists(graph, bead_id, max_depth=6) for bead_id in bead_ids}


def _angle_core_filters(
    i: int,
    j: int,
    k: int,
    adjacency: set[Tuple[int, int]],
    constraints: set[Tuple[int, int]],
    within3: Dict[int, set[int]],
    dist_map: Dict[int, Dict[int, int]],
) -> bool:
    if i == j or j == k or i == k:
        return False

    ij = _sorted_pair(i, j)
    jk = _sorted_pair(j, k)
    ij_bonded = ij in adjacency
    jk_bonded = jk in adjacency

    if ij in constraints and jk in constraints:
        return False
    if not (ij_bonded or jk_bonded):
        return False
    if i not in within3.get(j, set()) or k not in within3.get(j, set()):
        return False

    dik = dist_map.get(i, {}).get(k)
    dij = dist_map.get(i, {}).get(j)
    djk = dist_map.get(j, {}).get(k)
    if dik is None or dij is None or djk is None:
        return False
    return dik == dij + djk


def _remove_angles_in_dihedrals_new_only(
    new_angles: List[Tuple[int, int, int]],
    dihedrals: Iterable[Tuple[int, int, int, int]],
    bead_count: int,
) -> List[Tuple[int, int, int]]:
    if bead_count < 5:
        return new_angles
    drop: set[Tuple[int, int, int]] = set()
    for a, b, c, d in dihedrals:
        drop.add(_canon_angle(a, b, c))
        drop.add(_canon_angle(b, c, d))
    return [angle for angle in new_angles if angle not in drop]


def _generate_new_angles(
    inp_data: MonomerTemplate,
    priority_bonds: Iterable[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    bead_ids = sorted(inp_data.beads.keys())
    if len(bead_ids) <= 2:
        return []

    bonds = {_sorted_pair(a, b) for a, b in inp_data.bonds}
    constraints = {_sorted_pair(a, b) for a, b in inp_data.constraints}
    adjacency = bonds | constraints
    graph = _build_graph(adjacency)
    within3 = _compute_within3(bead_ids, graph)
    dist_map = _compute_dist_map(bead_ids, graph)
    atom_count = sum(len(refs) for refs in inp_data.beads.values())
    priority_set = {_sorted_pair(a, b) for a, b in priority_bonds}

    angle_list = list(inp_data.angles)

    def duplicate_membership(i: int, j: int, k: int) -> bool:
        for ai, aj, ak in angle_list:
            if i in (ai, aj, ak) and j in (ai, aj, ak) and k in (ai, aj, ak):
                return True
        return False

    def fan_suppressed(i: int, j: int, k: int) -> bool:
        if atom_count <= 15:
            return False
        for ai, aj, ak in angle_list:
            if aj != j or i not in (ai, ak):
                continue
            for bi, bj, bk in angle_list:
                if bj == j and k in (bi, bk):
                    return True
        return False

    def prio_edge(a: int, b: int) -> int:
        return 0 if _sorted_pair(a, b) in priority_set else 1

    candidates0: List[Tuple[int, int, int, int, int]] = []
    for i in bead_ids:
        for j in bead_ids:
            for k in bead_ids:
                ij = _sorted_pair(i, j)
                jk = _sorted_pair(j, k)
                if not _angle_core_filters(i, j, k, adjacency, constraints, within3, dist_map):
                    continue
                if ij in adjacency and jk in adjacency:
                    candidates0.append((prio_edge(i, j), prio_edge(j, k), i, j, k))

    candidates0.sort(key=lambda item: (item[0], item[1], item[3], item[2], item[4]))
    for _, _, i, j, k in candidates0:
        if duplicate_membership(i, j, k) or fan_suppressed(i, j, k):
            continue
        angle_list.append((i, j, k))

    candidates1: List[Tuple[int, int, Tuple[int, int], int, int, int]] = []
    for i in bead_ids:
        for j in bead_ids:
            for k in bead_ids:
                ij = _sorted_pair(i, j)
                jk = _sorted_pair(j, k)
                if not _angle_core_filters(i, j, k, adjacency, constraints, within3, dist_map):
                    continue
                if ij in adjacency and jk in adjacency:
                    continue
                if ij in adjacency:
                    stretched = dist_map.get(j, {}).get(k)
                    if stretched is None:
                        continue
                    anchor_key = (j, i)
                    priority = 0 if ij in priority_set else 2
                elif jk in adjacency:
                    stretched = dist_map.get(j, {}).get(i)
                    if stretched is None:
                        continue
                    anchor_key = (j, k)
                    priority = 1 if jk in priority_set else 2
                else:
                    continue
                candidates1.append((priority, stretched, anchor_key, i, j, k))

    candidates1.sort(key=lambda item: (item[0], item[1], item[4], item[3], item[5]))
    used_anchor: set[Tuple[int, int]] = set()
    for _, _, anchor_key, i, j, k in candidates1:
        if anchor_key in used_anchor:
            continue
        if duplicate_membership(i, j, k) or fan_suppressed(i, j, k):
            continue
        angle_list.append((i, j, k))
        used_anchor.add(anchor_key)

    existing_canon = {_canon_angle(i, j, k) for i, j, k in inp_data.angles}
    all_canon = {_canon_angle(i, j, k) for i, j, k in angle_list}
    new_canon = [angle for angle in all_canon if angle not in existing_canon]
    new_canon.sort(key=lambda angle: (angle[1], angle[0], angle[2]))
    return _remove_angles_in_dihedrals_new_only(new_canon, inp_data.dihedrals, len(inp_data.beads))


def format_inp(template: MonomerTemplate) -> str:
    lines: List[str] = []
    if template.preamble:
        lines.extend(template.preamble)
    lines.append("BEADS")
    for bead_id, refs in template.beads.items():
        lines.append(f"{bead_id} " + ",".join(ref.format() for ref in refs))
    lines.append("BONDS")
    for a, b in template.bonds:
        lines.append(f"{a},{b}")
    if template.constraints:
        lines.append("CONSTRAINTS")
        for a, b in template.constraints:
            lines.append(f"{a},{b}")
    lines.append("ANGLES")
    for a, b, c in template.angles:
        lines.append(f"{a},{b},{c}")
    lines.append("DIHEDRALS")
    for a, b, c, d in template.dihedrals:
        lines.append(f"{a},{b},{c},{d}")
    lines.append("IMPROPERS")
    for a, b, c, d in template.impropers:
        lines.append(f"{a},{b},{c},{d}")
    return "\n".join(lines) + "\n"


def validate_generated_input(
    inp_data: MonomerTemplate,
    xyz_path: Path,
    terminal_cap_indices: Optional[Sequence[int]] = None,
) -> ValidationReport:
    symbols, _ = parse_xyz(xyz_path)
    natoms = len(symbols)
    report = ValidationReport(target=str(xyz_path))
    owners = _weighted_atom_owners(inp_data)

    missing_atoms = [atom_index for atom_index in range(1, natoms + 1) if atom_index not in owners]
    if missing_atoms:
        report.problems.append(
            f"Generated input misses atom indices: {missing_atoms[:20]}{'...' if len(missing_atoms) > 20 else ''}"
        )

    for atom_index, refs in owners.items():
        if atom_index < 1 or atom_index > natoms:
            report.problems.append(f"Generated input references atom {atom_index} outside 1..{natoms}.")
            continue
        total_weight = sum((ref.weight for ref in refs), Fraction(0, 1))
        if total_weight != Fraction(1, 1):
            report.problems.append(
                f"Generated input atom {atom_index} has total bead weight {total_weight} instead of 1."
            )

    if terminal_cap_indices:
        wrong_caps = [
            (atom_index, symbols[atom_index - 1])
            for atom_index in sorted(set(terminal_cap_indices))
            if 1 <= atom_index <= natoms and symbols[atom_index - 1] != "H"
        ]
        if wrong_caps:
            report.problems.append(f"Expected terminal cap atoms to be H, found {wrong_caps}")

    return report


def build_polymer_input(
    sequence: Sequence[str] | str,
    polymer_xyz_path: Path,
    templates: Dict[str, MonomerTemplate],
    metadata: Dict[str, ConnectionMetadata],
) -> PolymerInputBundle:
    tokens = normalize_sequence(sequence)
    symbols, _ = parse_xyz(polymer_xyz_path)
    natoms = len(symbols)

    preamble = [f"# Auto-generated polymer Bartender input for {sequence_stem(tokens)}"]
    beads: "OrderedDict[int, List[WeightedAtomRef]]" = OrderedDict()
    bonds: List[Tuple[int, int]] = []
    constraints: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    dihedrals: List[Tuple[int, int, int, int]] = []
    impropers: List[Tuple[int, int, int, int]] = []

    report = ValidationReport(target=f"{sequence_stem(tokens)} base inp")
    atom_offset = 0
    bead_offset = 0
    connection_bonds: List[Tuple[int, int]] = []
    connection_beads: List[int] = []
    terminal_cap_indices: List[int] = []
    previous_right_bead: Optional[int] = None
    expected_atoms = 0

    for block_index, token in enumerate(tokens):
        if token not in templates or token not in metadata:
            raise KeyError(f"Unknown monomer token '{token}'")
        template = templates[token]
        meta = metadata[token]

        removed = set()
        if len(tokens) > 1:
            if block_index > 0:
                removed.add(meta.head_br)
            if block_index < len(tokens) - 1:
                removed.add(meta.tail_br)

        local_to_global: Dict[int, int] = {}
        for local_atom_index in range(1, template.atom_count + 1):
            if local_atom_index in removed:
                continue
            local_to_global[local_atom_index] = atom_offset + len(local_to_global) + 1

        if meta.head_br in local_to_global:
            terminal_cap_indices.append(local_to_global[meta.head_br])
        if meta.tail_br in local_to_global:
            terminal_cap_indices.append(local_to_global[meta.tail_br])

        for bead_id, refs in template.beads.items():
            global_bead = bead_offset + bead_id
            mapped = [
                WeightedAtomRef(atom_index=local_to_global[ref.atom_index], denominator=ref.denominator)
                for ref in refs
                if ref.atom_index in local_to_global
            ]
            if not mapped:
                report.problems.append(
                    f"Block {block_index + 1} token {token} produced empty bead {global_bead} after connector removal."
                )
            beads[global_bead] = mapped

        bonds.extend([(bead_offset + a, bead_offset + b) for a, b in template.bonds])
        constraints.extend([(bead_offset + a, bead_offset + b) for a, b in template.constraints])
        angles.extend([(bead_offset + a, bead_offset + b, bead_offset + c) for a, b, c in template.angles])
        dihedrals.extend(
            [(bead_offset + a, bead_offset + b, bead_offset + c, bead_offset + d) for a, b, c, d in template.dihedrals]
        )
        impropers.extend(
            [(bead_offset + a, bead_offset + b, bead_offset + c, bead_offset + d) for a, b, c, d in template.impropers]
        )

        left_bead = bead_offset + meta.left_connection_bead
        right_bead = bead_offset + meta.right_connection_bead
        connection_beads.extend([left_bead, right_bead])
        if previous_right_bead is not None:
            bond = _sorted_pair(previous_right_bead, left_bead)
            bonds.append(bond)
            connection_bonds.append(bond)
        previous_right_bead = right_bead

        kept_atoms = template.atom_count - len(removed)
        expected_atoms += kept_atoms
        atom_offset += kept_atoms
        bead_offset += template.bead_count

    if natoms != expected_atoms:
        report.problems.append(f"Polymer xyz atom count is {natoms}, but template-based build expects {expected_atoms}.")

    base = MonomerTemplate(
        path=polymer_xyz_path.with_suffix(".inp"),
        preamble=preamble,
        beads=beads,
        bonds=[(int(a), int(b)) for a, b in bonds],
        constraints=[(int(a), int(b)) for a, b in constraints],
        angles=[(int(a), int(b), int(c)) for a, b, c in angles],
        dihedrals=[(int(a), int(b), int(c), int(d)) for a, b, c, d in dihedrals],
        impropers=[(int(a), int(b), int(c), int(d)) for a, b, c, d in impropers],
    )

    base_check = validate_generated_input(base, polymer_xyz_path, terminal_cap_indices)
    report.problems.extend(base_check.problems)
    report.warnings.extend(base_check.warnings)

    new_angles = _generate_new_angles(base, connection_bonds)
    augmented = MonomerTemplate(
        path=polymer_xyz_path.with_suffix(".inp"),
        preamble=list(base.preamble),
        beads=base.beads,
        bonds=list(base.bonds),
        constraints=list(base.constraints),
        angles=list(base.angles) + new_angles,
        dihedrals=list(base.dihedrals),
        impropers=list(base.impropers),
    )
    augmented_report = validate_generated_input(augmented, polymer_xyz_path, terminal_cap_indices)

    return PolymerInputBundle(
        base=base,
        augmented=augmented,
        base_text=format_inp(base),
        augmented_text=format_inp(augmented),
        base_report=report,
        augmented_report=augmented_report,
        connection_bonds=connection_bonds,
        connection_beads=sorted(set(connection_beads)),
    )


def default_bead_spec(token: str, bead_count: int) -> dict[str, list[str]]:
    labels = [f"{token}{index}" for index in range(1, bead_count + 1)]
    return {"labels": labels, "types": list(labels)}


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
    parsed: Dict[str, List[ParamLine]] = {section: [] for section in ("bonds", "constraints", "angles", "dihedrals")}
    current: Optional[str] = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped.strip("[]").strip().lower()
            current = header if header in parsed else None
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
    return {
        "path": str(path),
        "counts": {section: len(lines) for section, lines in parsed.items()},
        "bonds": [
            {"indices": line.indices, "params": list(line.tokens), "commented": line.commented, "comment": line.inline_comment}
            for line in parsed["bonds"]
        ],
        "angles": [
            {
                "indices": line.indices,
                "params": list(line.tokens),
                "commented": line.commented,
                "comment": line.inline_comment,
                "rmsd": line.rmsd,
            }
            for line in parsed["angles"]
        ],
    }


def find_case_json(start: Path) -> Optional[Path]:
    current = start.resolve()
    for _ in range(6):
        candidate = current / "case.json"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def resolve_case_artifact(case_dir: Path, case: Dict[str, object], key: str) -> Path:
    candidates: List[Path] = []
    artifacts = case.get("artifacts", {})
    if isinstance(artifacts, dict) and key in artifacts:
        candidates.append(case_dir / str(artifacts[key]))
    if key in case:
        raw = Path(str(case[key]))
        if raw.is_absolute():
            candidates.append(raw)
            candidates.append(case_dir / raw.name)
        else:
            candidates.append(case_dir / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve case artifact '{key}' from {case_dir}. Tried: {', '.join(str(path) for path in candidates)}"
    )


def normalize_label_spec(token: str, bead_count: int, raw_spec: Any) -> Dict[str, List[str]]:
    if raw_spec is None:
        return default_bead_spec(token, bead_count)
    if isinstance(raw_spec, list):
        labels = [str(value) for value in raw_spec]
        if len(labels) != bead_count:
            raise ValueError(f"Label override for token {token} has {len(labels)} entries, expected {bead_count}.")
        return {"labels": labels, "types": [label.split("(", 1)[0] if "(" in label else label for label in labels]}
    if isinstance(raw_spec, dict):
        labels = raw_spec.get("labels")
        types = raw_spec.get("types")
        if labels is None:
            raise ValueError(f"Label override for token {token} must contain 'labels'.")
        labels = [str(value) for value in labels]
        if len(labels) != bead_count:
            raise ValueError(f"Label override for token {token} has {len(labels)} labels, expected {bead_count}.")
        if types is None:
            types = [label.split("(", 1)[0] if "(" in label else label for label in labels]
        else:
            types = [str(value) for value in types]
        if len(types) != bead_count:
            raise ValueError(f"Type override for token {token} has {len(types)} entries, expected {bead_count}.")
        return {"labels": labels, "types": types}
    raise TypeError(f"Unsupported label specification for token {token}: {type(raw_spec)!r}")


def load_label_map(path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label map JSON must be an object.")
    overrides: Dict[str, Dict[str, List[str]]] = {}
    for token, spec in data.items():
        if isinstance(spec, dict):
            entry = {"labels": [str(value) for value in spec.get("labels", [])]}
            if "types" in spec:
                entry["types"] = [str(value) for value in spec.get("types", [])]
            overrides[str(token)] = entry
        elif isinstance(spec, list):
            overrides[str(token)] = {"labels": [str(value) for value in spec], "types": []}
        else:
            raise ValueError(f"Unsupported label map entry for token {token}: {type(spec)!r}")
    return overrides


def build_bead_maps(
    case: Dict[str, object],
    overrides: Dict[str, Dict[str, List[str]]],
) -> tuple[Dict[int, str], Dict[int, str], set[int]]:
    monomers = case.get("monomers")
    tokens = case.get("sequence_tokens")
    if not isinstance(monomers, dict) or not isinstance(tokens, list):
        raise ValueError("case.json must contain 'monomers' and 'sequence_tokens'.")

    case_specs = case.get("bead_specs", {})
    if not isinstance(case_specs, dict):
        case_specs = {}

    label_map: Dict[int, str] = {}
    type_map: Dict[int, str] = {}
    connection_beads = set(int(value) for value in case.get("connection_beads", []))
    offset = 0
    for token in tokens:
        if token not in monomers:
            raise KeyError(f"Token {token} is not present in case['monomers'].")
        bead_count = int(monomers[token]["bead_count"])
        spec = normalize_label_spec(token, bead_count, overrides.get(token) or case_specs.get(token))
        for local_index in range(1, bead_count + 1):
            global_index = offset + local_index
            label_map[global_index] = spec["labels"][local_index - 1]
            type_map[global_index] = spec["types"][local_index - 1]
        if not connection_beads:
            connection_beads.add(offset + int(monomers[token].get("left_connection_bead", 1)))
        offset += bead_count
    return label_map, type_map, connection_beads


def shortest_path_len(graph: Dict[int, set[int]], start: int, goal: int) -> Optional[int]:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor == goal:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None


def choose_best_rmsd_uncomment(lines: List[ParamLine]) -> List[ParamLine]:
    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
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
    case = json.loads(case_path.read_text(encoding="utf-8"))
    parsed = parse_gmx_out_itp(itp_path)
    label_map, type_map, connection_beads = build_bead_maps(case, label_overrides)

    graph = build_graph({_sorted_pair(*line.indices) for line in parsed["bonds"]} | {_sorted_pair(*line.indices) for line in parsed["constraints"]})
    angle_lines = choose_best_rmsd_uncomment(parsed["angles"])
    source_tag = f"{case.get('sequence_stem', case_path.parent.name)}:{itp_path.parent.name}"

    def category(indices: Tuple[int, ...]) -> str:
        return "WITH_BACKBONE" if any(index in connection_beads for index in indices) else "WITHOUT_BACKBONE"

    def map_labels(indices: Tuple[int, ...]) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        try:
            display = tuple(label_map[index] for index in indices)
            types = tuple(type_map[index] for index in indices)
        except KeyError as exc:
            raise KeyError(f"{itp_path}: bead index {exc.args[0]} is not present in the case bead map.") from exc
        return display, types

    section_map = {"bonds": "bondtypes", "constraints": "constrainttypes", "angles": "angletypes", "dihedrals": "dihedraltypes"}
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

    section_order = ("bondtypes", "constrainttypes", "angletypes", "dihedraltypes")
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


def prepare_relaxation_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    pipeline_cfg: Dict[str, Any],
    base_dir: Path,
) -> Optional[Path]:
    if flow["relaxation"] == "off" and flow["md"] != "xtb":
        case["relaxation"] = {
            "mode": flow["relaxation"],
            "md": flow["md"],
            "workdir": None,
            "run_script": None,
            "geometry_hint": str(case["artifacts"]["polymer_xyz"]),
            "trajectory_hint": None,
            "geometry_opt": False,
        }
        return None

    workdir_name = flow["workdir_name"]
    workdir = case_dir / workdir_name
    workdir.mkdir(parents=True, exist_ok=True)

    xyz_name = str(case["artifacts"]["polymer_xyz"])
    polymer_xyz = case_dir / xyz_name
    local_xyz = workdir / polymer_xyz.name
    local_xyz.write_text(polymer_xyz.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        raise TypeError("case.electronic_state must be a mapping")
    charge = int(state.get("charge", 0))
    uhf = int(state.get("uhf", 0))
    xtb_cfg = resolve_xtb_settings(pipeline_cfg)
    orca_cfg = resolve_orca_settings(pipeline_cfg)
    xtb_env_script = xtb_cfg["env_script"]
    xtb_parallel = int(xtb_cfg["parallel"])
    xtb_parallel_expr = f"${{SLURM_NTASKS:-{xtb_parallel}}}"
    xtb_solvent_flags = ""
    if xtb_cfg["solvent_model"] != "off":
        xtb_solvent_flags = f" --{xtb_cfg['solvent_model']} {shlex.quote(xtb_cfg['solvent'])}"
        if xtb_cfg["solvent_reference"]:
            xtb_solvent_flags += f" {shlex.quote(xtb_cfg['solvent_reference'])}"
    xtb_common_flags = (
        f"--gfn {xtb_cfg['gfn']} "
        f"--chrg {charge} "
        f"--uhf {uhf} "
        f"--acc {xtb_cfg['acc']:.3f} "
        f"--etemp {xtb_cfg['etemp']:.3f}"
        f"{xtb_solvent_flags} "
        f"--parallel {xtb_parallel_expr}"
    )

    needs_xtb = flow["relaxation"] == "xtb" or flow["md"] == "xtb"
    needs_orca = flow["relaxation"] == "orca"
    geometry_hint = local_xyz.name
    trajectory_hint: Optional[str] = None
    geometry_opt = flow["relaxation"] == "xtb"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -J relax_{polymer_xyz.stem}",
        "#SBATCH -p gpupart",
        "#SBATCH -N 1",
        f"#SBATCH -n {xtb_parallel if needs_xtb else max(1, int(orca_cfg['nprocs']))}",
        "#SBATCH -t 100:00:00",
        "",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        "export OMP_NUM_THREADS=${SLURM_NTASKS:-1}",
        "export MKL_NUM_THREADS=${SLURM_NTASKS:-1}",
        "export OMP_STACKSIZE=1G",
    ]
    if xtb_env_script and needs_xtb:
        lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(xtb_env_script)} ]; then",
                f"  source {shlex.quote(xtb_env_script)}",
                "fi",
                "set -u",
            ]
        )
    if needs_orca:
        lines.append(shell_assign("ORCA_BIN", orca_cfg["binary"]))
    if needs_xtb:
        lines.append(shell_assign("XTB_BIN", xtb_cfg["binary"]))

    if flow["relaxation"] == "xtb":
        lines.append(
            f"$XTB_BIN {shlex.quote(local_xyz.name)} {xtb_common_flags} --opt {shlex.quote(xtb_cfg['opt_level'])} --cycles {xtb_cfg['opt_cycles']} > xtb_opt.out"
        )
        geometry_hint = "xtbopt.xyz"
    elif flow["relaxation"] == "orca":
        orca_template_path = resolve_optional_path(base_dir, orca_cfg["input_template_path"])
        orca_template_text = (
            orca_template_path.read_text(encoding="utf-8", errors="replace")
            if orca_template_path is not None
            else None
        )
        write_text(workdir / "relax.inp", render_orca_input(local_xyz.name, state, orca_cfg, orca_template_text))
        lines.append("$ORCA_BIN relax.inp > relax.out")
        geometry_hint = "relax.xyz"
    elif flow["relaxation"] != "off":
        raise ValueError(f"Unsupported relaxation mode: {flow['relaxation']}")

    if flow["md"] == "xtb":
        md_template_path = resolve_optional_path(base_dir, xtb_cfg["md_input_template_path"])
        md_template_text = (
            md_template_path.read_text(encoding="utf-8", errors="replace")
            if md_template_path is not None
            else None
        )
        write_text(workdir / "gochem.inp", render_xtb_md_input("nvt", xtb_cfg, md_template_text))
        write_text(workdir / "xtb_traj_to_pdb.py", render_xyz_traj_to_pdb_converter())
        lines.append(
            f"$XTB_BIN {shlex.quote(geometry_hint)} {xtb_common_flags} --md --input gochem.inp > xtb_md.out"
        )
        lines.append("python3 xtb_traj_to_pdb.py xtb.trj xtb_traj.pdb")
        trajectory_hint = "xtb_traj.pdb"

    script_path = workdir / "run_relax.sh"
    write_text(script_path, "\n".join(lines) + "\n")
    script_path.chmod(0o755)
    case["relaxation"] = {
        "mode": flow["relaxation"],
        "md": flow["md"],
        "workdir": workdir_name,
        "run_script": script_path.name,
        "geometry_hint": geometry_hint,
        "trajectory_hint": trajectory_hint,
        "raw_trajectory_hint": "xtb.trj" if flow["md"] == "xtb" else None,
        "geometry_opt": geometry_opt,
    }
    return workdir


def prepare_bartender_job(
    case_dir: Path,
    case: Dict[str, Any],
    flow: Dict[str, str],
    bartender_cfg: Dict[str, Any],
) -> Optional[Path]:
    if flow["md"] == "off":
        case["bartender"] = {
            "mode": "off",
            "workdir": None,
            "run_script": None,
            "geometry_source": None,
            "trajectory_source": None,
        }
        return None

    if not bartender_cfg.get("enabled", True):
        return None

    polymer_geometry = case_dir / str(case["artifacts"]["polymer_xyz"])
    relax = case.get("relaxation", {})
    if not isinstance(relax, dict):
        relax = {}
    if flow["relaxation"] == "off":
        geometry = polymer_geometry
        geometry_source = "polymer_xyz"
    else:
        relax_workdir = relax.get("workdir")
        relax_geometry = relax.get("geometry_hint")
        if not relax_workdir or not relax_geometry:
            raise ValueError("Relaxation metadata is incomplete; cannot determine Bartender geometry input.")
        geometry = case_dir / str(relax_workdir) / str(relax_geometry)
        geometry_source = "relaxation_output"

    trajectory: Optional[Path] = None
    if flow["md"] == "xtb":
        relax_workdir = relax.get("workdir")
        relax_trajectory = relax.get("trajectory_hint")
        if not relax_workdir or not relax_trajectory:
            raise ValueError("xTB reuse mode requires relaxation.trajectory_hint metadata.")
        trajectory = case_dir / str(relax_workdir) / str(relax_trajectory)

    inp = case_dir / str(case["artifacts"]["bartender_inp"])
    if not inp.exists():
        raise FileNotFoundError(f"Bartender inp does not exist: {inp}")

    outdir = case_dir / str(bartender_cfg.get("output_dirname", "bartender_job"))
    outdir.mkdir(parents=True, exist_ok=True)
    local_inp = outdir / inp.name
    local_inp.write_text(inp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    geometry_arg = os.path.relpath(str(geometry), start=str(outdir))
    command = [str(bartender_cfg.get("binary"))]
    command.extend(["-cpus", str(int(bartender_cfg.get("cpus", 1)))])

    state = case.get("electronic_state", {})
    if not isinstance(state, dict):
        state = {}
    bartender_charge = bartender_cfg.get("charge")
    if bartender_charge is None:
        bartender_charge = int(state.get("charge", 0))
    command.extend(["-charge", str(int(bartender_charge))])

    skip = int(bartender_cfg.get("skip", 1))
    if flow["md"] == "bartender":
        command.extend(
            [
                "-method",
                "gfn2",
                "-time",
                str(int(bartender_cfg.get("time_ps", 5000))),
            ]
        )
        command.extend(["-temperature", f"{float(bartender_cfg.get('temperature_k', 310.0)):.3f}"])
        solvent = str(bartender_cfg.get("solvent", "h2o")).strip()
        if solvent:
            command.extend(["-solvent", solvent])
        dcd_save = str(bartender_cfg.get("dcd_save", "")).strip()
        if dcd_save:
            command.extend(["-dcdSave", dcd_save])
        if skip > 1:
            command.extend(["-skip", str(skip)])
    elif flow["md"] == "xtb":
        if trajectory is None:
            raise ValueError("xTB reuse mode requires a trajectory path.")
        trajectory_arg = os.path.relpath(str(trajectory), start=str(outdir))
        command.extend(["-owntraj", trajectory_arg, "-refit"])
        if skip > 1:
            command.extend(["-skip", str(skip)])
    else:
        raise ValueError(f"Unsupported md mode: {flow['md']}")

    command.extend([geometry_arg, local_inp.name])
    bt_root = str(bartender_cfg.get("root", "")).strip()
    bt_env_script = str(bartender_cfg.get("env_script", "")).strip()
    script_path = outdir / "run_bartender.sh"
    script_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
    ]
    if bt_root:
        script_lines.append(f"export {shell_assign('BTROOT', bt_root)}")
    if bt_env_script:
        script_lines.extend(
            [
                "set +u",
                f"if [ -f {shlex.quote(bt_env_script)} ]; then",
                f"  source {shlex.quote(bt_env_script)}",
                "fi",
                "set -u",
            ]
        )
    script_lines.append(" ".join(shlex.quote(part) for part in command))
    write_text(
        script_path,
        "\n".join(script_lines) + "\n",
    )
    script_path.chmod(0o755)

    geometry_exists = geometry.exists()
    trajectory_exists = trajectory.exists() if trajectory is not None else None
    if bartender_cfg.get("execute", False):
        if not geometry_exists:
            raise FileNotFoundError(f"Bartender geometry source does not exist yet: {geometry}")
        if trajectory is not None and not trajectory_exists:
            raise FileNotFoundError(f"Bartender owntraj source does not exist yet: {trajectory}")

    manifest = {
        "mode": flow["md"],
        "geometry": geometry_arg,
        "inp": local_inp.name,
        "trajectory": os.path.relpath(str(trajectory), start=str(outdir)) if trajectory is not None else None,
        "command": command,
        "outdir": str(outdir),
        "geometry_source": geometry_source,
        "geometry_exists": geometry_exists,
        "trajectory_exists": trajectory_exists,
    }
    write_text(outdir / "bartender_job.json", json.dumps(manifest, indent=2))

    case.setdefault("bartender", {})
    case["bartender"]["job_dir"] = outdir.name
    case["bartender"]["run_script"] = script_path.name
    case["bartender"]["mode"] = flow["md"]
    case["bartender"]["geometry_source"] = geometry_source
    case["bartender"]["geometry_path"] = str(geometry)
    case["bartender"]["trajectory_path"] = str(trajectory) if trajectory is not None else None

    if bartender_cfg.get("execute", False):
        result = subprocess.run(["bash", script_path.name], cwd=outdir, text=True, capture_output=True)
        write_text(outdir / "bartender.stdout", result.stdout)
        write_text(outdir / "bartender.stderr", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Bartender failed with exit code {result.returncode}")
        gmx_out = outdir / "gmx_out.itp"
        if gmx_out.exists():
            write_text(outdir / "gmx_out_summary.json", json.dumps(summarize_itp(gmx_out), indent=2))

    return outdir


def collect_results(root: Path, output: Path) -> Dict[str, Any]:
    records = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        summary = summarize_itp(itp_path)
        case_json = find_case_json(itp_path.parent)
        if case_json:
            case = json.loads(case_json.read_text(encoding="utf-8"))
            summary["case"] = {
                "sequence_stem": case.get("sequence_stem"),
                "sequence_tokens": case.get("sequence_tokens"),
                "case_json": str(case_json),
            }
        records.append(summary)
    payload = {"root": str(root), "count": len(records), "records": records}
    write_text(output, json.dumps(payload, indent=2))
    return payload


def merge_results(root: Path, output_itp: Path, output_json: Path, label_map_path: Optional[Path]) -> Dict[str, Any]:
    label_overrides = load_label_map(label_map_path) if label_map_path else {}
    records: List[TypedRecord] = []
    skipped: List[Dict[str, str]] = []
    for itp_path in sorted(root.rglob("gmx_out.itp")):
        case_path = find_case_json(itp_path.parent)
        if case_path is None:
            skipped.append({"path": str(itp_path), "reason": "case.json not found in parent chain"})
            continue
        try:
            records.extend(typed_records_for_result(itp_path, case_path, label_overrides))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(itp_path), "reason": str(exc)})

    merged = merge_records(records)
    write_merged_forcefield(output_itp, merged, root=root, label_map_path=label_map_path)
    payload = merged_summary_payload(root, merged, skipped)
    write_text(output_json, json.dumps(payload, indent=2))
    return payload


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_dir = Path(cfg["paths"]["base_dir"]).resolve()
    out_root = Path(cfg["paths"]["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = cfg["bartender_pipeline"]
    flow = resolve_pipeline_modes(pipeline_cfg)
    legacy_init_templates = pipeline_cfg.get("init_templates", {})
    if legacy_init_templates is None:
        legacy_init_templates = {}
    if not isinstance(legacy_init_templates, dict):
        raise TypeError("bartender_pipeline.init_templates must be a mapping when provided")

    monomer_cfg = normalize_monomer_configs(cfg["monomers"], legacy_init_templates)
    monomer_paths = {
        token: str(resolve_under_base(base_dir, entry["xyz"]))
        for token, entry in monomer_cfg.items()
    }
    library = load_monomer_library(monomer_paths, base_dir=base_dir)
    monomer_keys = set(library.keys())

    init_templates = {
        token: resolve_under_base(base_dir, entry["init_template"])
        for token, entry in monomer_cfg.items()
        if entry.get("init_template")
    }
    sequence_jobs = build_sequence_jobs(cfg["system"], monomer_keys)

    template_cache: Dict[str, MonomerTemplate] = {}
    metadata_cache: Dict[str, ConnectionMetadata] = {}
    validation_cache: Dict[str, ValidationReport] = {}

    cases: List[Dict[str, Any]] = []
    for tokens in sequence_jobs:
        for token in tokens:
            if token not in monomer_paths:
                raise KeyError(f"Unknown monomer token: {token}")
            if token not in init_templates:
                raise KeyError(f"Missing init template for monomer token: {token}")
            if token not in template_cache:
                template = parse_bartender_inp(init_templates[token])
                template_cache[token] = template
                validation_cache[token] = validate_template(template, Path(monomer_paths[token]))
                metadata_cache[token] = infer_connection_metadata(template, Path(monomer_paths[token]))

        stem = sequence_stem(tokens)
        case_dir = out_root / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        torsion_mode = str(cfg["system"].get("n_torsion_mode", "repeat"))
        torsion = len(tokens) if torsion_mode == "repeat" else max(1, len(tokens) - 1)

        builder_tmp = case_dir / "_builder_tmp"
        if builder_tmp.exists():
            shutil.rmtree(builder_tmp, ignore_errors=True)
        builder_tmp.mkdir(parents=True, exist_ok=True)
        build_polymer_structure(
            tokens,
            monomer_dict=library,
            n_torsion=torsion,
            output_filename=f"{stem}.xyz",
            output_dir=builder_tmp,
        )

        built_xyz = builder_tmp / f"{stem}.xyz"
        final_xyz = case_dir / f"{stem}.xyz"
        if not built_xyz.exists():
            raise FileNotFoundError(f"param_opt builder did not produce {built_xyz}")
        shutil.copyfile(built_xyz, final_xyz)
        if builder_tmp.exists():
            shutil.rmtree(builder_tmp, ignore_errors=True)

        bundle = build_polymer_input(tokens, final_xyz, template_cache, metadata_cache)
        base_inp = case_dir / f"{stem}_base.inp"
        final_inp = case_dir / f"{stem}_bartender.inp"
        write_text(base_inp, bundle.base_text)
        write_text(final_inp, bundle.augmented_text)

        monomer_validation_text = []
        has_failure = False
        for token in sorted(set(tokens)):
            report = validation_cache[token]
            monomer_validation_text.append(f"[{token}] {report.target}")
            monomer_validation_text.append(report.render().strip())
            monomer_validation_text.append("")
            if not report.ok:
                has_failure = True

        write_text(case_dir / "monomer_validation.txt", "\n".join(monomer_validation_text).rstrip() + "\n")
        write_text(case_dir / "polymer_base_validation.txt", bundle.base_report.render())
        write_text(case_dir / "polymer_augmented_validation.txt", bundle.augmented_report.render())
        if not bundle.base_report.ok or not bundle.augmented_report.ok:
            has_failure = True

        electronic_state = resolve_case_electronic_state(tokens, monomer_cfg, pipeline_cfg)

        case: Dict[str, Any] = {
            "sequence_tokens": tokens,
            "sequence_stem": stem,
            "torsion": torsion,
            "workflow_mode": flow,
            "artifacts": {
                "polymer_xyz": final_xyz.name,
                "base_inp": base_inp.name,
                "bartender_inp": final_inp.name,
            },
            "polymer_xyz": str(final_xyz),
            "base_inp": str(base_inp),
            "bartender_inp": str(final_inp),
            "electronic_state": electronic_state,
            "bead_specs": {
                token: default_bead_spec(token, template_cache[token].bead_count)
                for token in sorted(set(tokens))
            },
            "monomers": {
                token: {
                    "xyz": monomer_paths[token],
                    "init_inp": str(init_templates[token]),
                    "head_br": metadata_cache[token].head_br,
                    "tail_br": metadata_cache[token].tail_br,
                    "left_connection_bead": metadata_cache[token].left_connection_bead,
                    "right_connection_bead": metadata_cache[token].right_connection_bead,
                    "bead_count": template_cache[token].bead_count,
                    "charge": monomer_cfg[token]["charge"],
                    "uhf": monomer_cfg[token]["uhf"],
                    "multiplicity": monomer_cfg[token]["multiplicity"],
                }
                for token in sorted(set(tokens))
            },
            "connection_bonds": bundle.connection_bonds,
            "connection_beads": bundle.connection_beads,
            "reports": {
                "monomer_validation_ok": all(validation_cache[token].ok for token in sorted(set(tokens))),
                "base_validation_ok": bundle.base_report.ok,
                "augmented_validation_ok": bundle.augmented_report.ok,
            },
        }

        if has_failure and not pipeline_cfg.get("allow_invalid", False):
            write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))
            raise ValueError(f"Validation failed for case {stem}. See {case_dir}.")

        relax_dir = prepare_relaxation_job(case_dir, case, flow, pipeline_cfg, base_dir)
        bartender_dir = prepare_bartender_job(case_dir, case, flow, pipeline_cfg["bartender"])
        write_text(case_dir / "case.json", json.dumps(case, indent=2, ensure_ascii=False))

        cases.append(
            {
                "sequence_stem": stem,
                "sequence_tokens": tokens,
                "case_dir": str(case_dir),
                "polymer_xyz": str(final_xyz),
                "bartender_inp": str(final_inp),
                "relaxation_dir": str(relax_dir) if relax_dir else None,
                "bartender_dir": str(bartender_dir) if bartender_dir else None,
            }
        )

    summary: Dict[str, Any] = {"settings": cfg, "cases": cases}

    post_cfg = pipeline_cfg["postprocess"]
    if post_cfg.get("collect", True):
        collect_path = resolve_under_base(out_root, str(post_cfg.get("collect_json", "bartender_summary.json")))
        summary["collect"] = collect_results(out_root, collect_path)
    if post_cfg.get("merge", False):
        label_map_value = post_cfg.get("label_map_path")
        label_map_path = resolve_under_base(base_dir, str(label_map_value)) if label_map_value else None
        output_itp = resolve_under_base(out_root, str(post_cfg.get("merged_itp", "merged_forcefield.itp")))
        output_json = resolve_under_base(out_root, str(post_cfg.get("merged_json", "merged_forcefield.json")))
        summary["merge"] = merge_results(out_root, output_itp, output_json, label_map_path)

    write_text(out_root / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
