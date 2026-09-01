"""Configuration resolution and shared datatypes for the stage-03 pipeline.

This module owns the shared vocabulary of the ``qm_to_martini`` (stage 03)
workflow: frozen dataclasses describing monomer templates, parsed Bartender
ITP lines, connection metadata, and merged force-field variants, plus the
functions that normalize the ``bartender_pipeline`` section of the user YAML
config into validated setting dictionaries.

Position in the pipeline: ``pipeline.run_pipeline`` (and ``generator`` /
``cli``) call the ``resolve_*`` helpers here to interpret the raw config
before building per-sequence cases; ``workflow_logic.loader`` and
``workflow_logic.merger`` consume the dataclasses defined here.

Key conventions and gotchas:

- Atom indices inside templates and dataclasses are 1-based (Bartender
  convention); user-facing ``backbone_atoms`` YAML entries are 0-based and
  converted on input (``_normalize_index_list``) and output
  (``export_backbone_atom_config``).
- Several resolvers accept both the current flat config layout and legacy
  nested layouts (``relaxation``/``bartender`` mappings); the legacy branch
  is kept for backward compatibility with Series-01 configs.
- Distances passed to connection detection are in Angstrom; xTB MD settings
  use ps/fs/K as named in their keys.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from fractions import Fraction

from hygel_martini.core.utils import parse_csv_list

# Default head/tail connector detection radius in Angstrom (C-Br bond ~1.9 A).
CONNECTION_CUTOFF = 2.2
# Extracts the "rmsd: <float>" annotation Bartender writes into ITP comments.
RMSD_RE = re.compile(r"rmsd:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

@dataclass(frozen=True)
class ConnectionDetectionConfig:
    """How monomer connection (capping) atoms are located in an XYZ file.

    Fields:
        indicator: element symbol that marks a connector atom (default "Br").
        cutoff: maximum connector-to-backbone-atom distance in Angstrom.
    """
    indicator: str
    cutoff: float

@dataclass(frozen=True)
class TermGenerationConfig:
    """Resolved ``bartender_pipeline.term_generation`` settings.

    Fields:
        mode: normalized candidate-term generation mode; one of init_only,
            all_unique, polymer_backbone, topology_n, topology_swap_n,
            polymer_n, polymer_swap_n.
        n: non-negative extension budget for the ``*_n`` modes (0 otherwise).
        main_itp_dir: directory of reference main ITPs; required by the
            polymer_n / polymer_swap_n modes, None otherwise.
        candidates_tsv_dir: directory of candidate TSV tables; required by
            the polymer_n / polymer_swap_n modes, None otherwise.
    """
    mode: str
    n: int
    main_itp_dir: Optional[str] = None
    candidates_tsv_dir: Optional[str] = None

@dataclass(frozen=True)
class WeightedAtomRef:
    """One atom's (possibly fractional) membership in a CG bead.

    Fields:
        atom_index: 1-based atom index in the monomer/polymer XYZ file.
        denominator: n-way split factor; an atom shared by n beads appears
            in each with denominator n so the total weight sums to 1.
    """
    atom_index: int
    denominator: int = 1

    @property
    def weight(self) -> Fraction:
        """Exact fractional weight (1/denominator) of this reference."""
        return Fraction(1, self.denominator)

    def format(self) -> str:
        """Render the Bartender BEADS token, e.g. "12" or "12/2"."""
        if self.denominator == 1:
            return str(self.atom_index)
        return f"{self.atom_index}/{self.denominator}"

@dataclass
class ValidationReport:
    """Accumulated problems/warnings from validating one template or input.

    Fields:
        target: label of the validated object (usually a file path).
        problems: fatal findings; any entry makes ``ok`` False.
        warnings: non-fatal findings kept for the report only.
    """
    target: str
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no fatal problem was recorded (warnings are allowed)."""
        return not self.problems

    def render(self) -> str:
        """Format the report as a human-readable multi-line status block."""
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
    """Parsed Bartender ``.inp`` mapping/topology template for one monomer.

    Produced by ``workflow_logic.loader.parse_bartender_inp``. All indices
    are 1-based. Bonded-term entries are bead-index tuples.

    Fields:
        path: source ``.inp`` file.
        preamble: verbatim lines that precede the first section header.
        beads: bead id -> weighted atom references composing that bead.
        bonds/constraints: 2-tuples of bead ids.
        angles: 3-tuples of bead ids.
        dihedrals/impropers: 4-tuples of bead ids.
    """
    path: Path
    preamble: List[str]
    beads: Dict[int, List[WeightedAtomRef]]
    bonds: List[Tuple[int, int]]
    constraints: List[Tuple[int, int]]
    angles: List[Tuple[int, int, int]]
    dihedrals: List[Tuple[int, int, int, int]]
    impropers: List[Tuple[int, int, int, int]]

    @property
    def bead_count(self) -> int:
        """Number of CG beads defined by the template."""
        return len(self.beads)

    @property
    def atom_count(self) -> int:
        """Highest atom index referenced by any bead (0 if no beads).

        Used as a proxy for the expected XYZ atom count during validation.
        """
        if not self.beads:
            return 0
        return max(ref.atom_index for refs in self.beads.values() for ref in refs)

@dataclass
class PolymerInputBundle:
    """Everything built for one polymer's Bartender input.

    Produced by ``workflow_logic.builder.build_polymer_input`` from the
    per-monomer templates.

    Fields:
        base: concatenated polymer template with only the terms taken
            directly from the monomer templates plus connection terms.
        augmented: base template extended with generated candidate terms
            according to the term-generation mode.
        base_text/augmented_text: rendered ``.inp`` file contents.
        base_report/augmented_report: validation results for each template.
        connection_bonds: inter-monomer bead-bead bonds (global bead ids).
        connection_beads: global ids of beads that carry a connection.
        backbone_beads: global ids of beads flagged as backbone.
    """
    base: MonomerTemplate
    augmented: MonomerTemplate
    base_text: str
    augmented_text: str
    base_report: ValidationReport
    augmented_report: ValidationReport
    connection_bonds: List[Tuple[int, int]]
    connection_beads: List[int]
    backbone_beads: List[int]

@dataclass(frozen=True)
class ParamLine:
    """One parsed bonded-parameter line from a Bartender ``gmx_out.itp``.

    Fields:
        section: normalized section name (bonds/constraints/angles/
            dihedrals/impropers).
        indices: bead indices of the term (2-4 integers, 1-based).
        tokens: remaining whitespace-separated tokens (funct + parameters),
            kept as raw strings.
        commented: True when the whole line was commented out (leading ";"),
            i.e. a candidate that Bartender did not select.
        inline_comment: text after the first inline ";" separator.
        rmsd: fit RMSD parsed from the inline comment, if present.
        raw: original line without the trailing newline.
    """
    section: str
    indices: Tuple[int, ...]
    tokens: Tuple[str, ...]
    commented: bool
    inline_comment: str
    rmsd: Optional[float]
    raw: str

@dataclass(frozen=True)
class TypedRecord:
    """A ``ParamLine`` lifted to bead-type space for cross-case merging.

    Produced by ``workflow_logic.merger.typed_records_for_result``; grouping
    on (section, category, angle_dist, type_names) merges equivalent terms
    coming from different sequences.

    Fields:
        section: GROMACS type-section name (bondtypes/constrainttypes/
            angletypes/dihedraltypes/impropertypes).
        category: "WITH_BACKBONE" when any bead is a backbone bead,
            otherwise "WITHOUT_BACKBONE".
        angle_dist: for angles only, "DIST_LE2"/"DIST_GE3" by bond-graph
            distance between the outer beads; empty for other sections.
        type_names: per-bead type names used as the merge key.
        display_labels: per-bead human-readable labels.
        indices: original bead indices in the source case.
        tokens: funct + parameter tokens as raw strings.
        commented: comment state of the source line.
        inline_comment: inline comment text of the source line.
        rmsd: fit RMSD from the source line, if present.
        source_tag: "<sequence_stem>:<job dirname>" provenance tag.
        source_path: absolute path of the source ITP.
    """
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

@dataclass(frozen=True)
class ConnectionMetadata:
    """Head/tail connection geometry inferred for one monomer.

    Produced by ``workflow_logic.loader.infer_connection_metadata``. Atom
    and bead indices are 1-based within the monomer.

    Fields:
        head_carbon/tail_carbon: first configured backbone atom on each end
            (0 when that end has no configured atoms).
        head_br/tail_br: connector ("Br"-indicator) atom nearest each end.
        left_connection_bead/right_connection_bead: bead owning each
            connector atom; these beads are bonded across monomer joints.
        backbone_beads: beads containing any configured backbone atom.
    """
    head_carbon: int
    tail_carbon: int
    head_br: Optional[int]
    tail_br: Optional[int]
    left_connection_bead: int
    right_connection_bead: int
    backbone_beads: Tuple[int, ...]

@dataclass
class MergedVariant:
    """One distinct parameter variant within a merged type group.

    Produced by ``workflow_logic.merger.merge_records``: records sharing the
    same (tokens, commented, inline comment) signature collapse into one
    variant; exactly one variant per group is marked ``primary``.

    Fields:
        section/category/angle_dist/type_names: the group key (see
            ``TypedRecord``).
        display_labels: distinct per-source label tuples observed.
        tokens: funct + parameter tokens of the variant.
        commented: written comment state; non-primary variants are forced
            to True so only the primary line is active in the merged ITP.
        sources: source tags contributing this variant.
        indices_examples: example bead-index tuples from the sources.
        inline_comments: distinct non-empty inline comments observed.
        rmsd_values: RMSD values observed across sources.
        primary: True for the selected representative of the group.
    """
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
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def shell_assign(name: str, value: str) -> str:
    """Render a shell-safe ``name=value`` assignment line."""
    return f'{name}={shlex.quote(value)}'

def resolve_under_base(base_dir: Path, value: str | Path) -> Path:
    """Resolve a possibly relative path against ``base_dir``.

    Absolute paths are returned unchanged (not resolved); relative paths are
    joined to ``base_dir`` and fully resolved.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

def parse_bool(value: Any, default: bool = False) -> bool:
    """Coerce a YAML/JSON scalar to bool.

    None yields ``default``; numbers use truthiness; strings accept
    "1"/"true"/"yes"/"on" (case-insensitive) as True, anything else False.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def resolve_connection_detection_config(pipeline_cfg: Dict[str, Any]) -> ConnectionDetectionConfig:
    """Read connector-atom detection settings from the pipeline config.

    Args:
        pipeline_cfg: the ``bartender_pipeline`` mapping; keys
            ``connection_indicator`` (element symbol, default "Br") and
            ``connection_cutoff`` (Angstrom, default 2.2) are honored.

    Raises:
        ValueError: when the cutoff is not positive.
    """
    indicator = str(pipeline_cfg.get("connection_indicator", "Br")).strip() or "Br"
    cutoff = float(pipeline_cfg.get("connection_cutoff", CONNECTION_CUTOFF))
    if cutoff <= 0:
        raise ValueError("bartender_pipeline.connection_cutoff must be > 0")
    return ConnectionDetectionConfig(
        indicator=indicator,
        cutoff=cutoff,
    )

def resolve_term_generation_config(pipeline_cfg: Dict[str, Any]) -> TermGenerationConfig:
    """Normalize ``bartender_pipeline.term_generation`` into a config object.

    Accepts either a bare mode string or a mapping with ``mode``/``n`` and
    the optional ``main_itp_dir``/``candidates_tsv_dir`` keys. Mode aliases
    "exhaustive"/"all" -> "all_unique" and "original" -> "init_only" are
    applied before validation.

    Raises:
        TypeError: when the raw value is neither a string nor a mapping.
        ValueError: on an unsupported mode, a negative budget ``n``, or a
            polymer_n/polymer_swap_n mode missing its required directories.
    """
    raw_cfg = pipeline_cfg.get("term_generation", {})
    if raw_cfg is None:
        raw_cfg = {}
    if isinstance(raw_cfg, str):
        mode = raw_cfg
        n = 0
    elif isinstance(raw_cfg, dict):
        mode = raw_cfg.get("mode", "all_unique")
        n = raw_cfg.get("n", 0)
    else:
        raise TypeError("bartender_pipeline.term_generation must be a mapping or string when provided")

    aliases = {
        "exhaustive": "all_unique",
        "original": "init_only",
        "all": "all_unique",
    }
    normalized_mode = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    supported = {
        "init_only", "all_unique", "polymer_backbone",
        "topology_n", "topology_swap_n",
        "polymer_n", "polymer_swap_n",
    }
    if normalized_mode not in supported:
        raise ValueError(
            "bartender_pipeline.term_generation.mode must be one of "
            f"{sorted(supported)} (or alias exhaustive/original), got {mode!r}"
        )

    budget = int(n)
    if budget < 0:
        raise ValueError("bartender_pipeline.term_generation.n must be >= 0")

    main_itp_dir = raw_cfg.get("main_itp_dir") if isinstance(raw_cfg, dict) else None
    candidates_tsv_dir = raw_cfg.get("candidates_tsv_dir") if isinstance(raw_cfg, dict) else None

    if normalized_mode in {"polymer_n", "polymer_swap_n"}:
        if not main_itp_dir:
            raise ValueError(
                "bartender_pipeline.term_generation.main_itp_dir is required for polymer_n / polymer_swap_n mode"
            )
        if not candidates_tsv_dir:
            raise ValueError(
                "bartender_pipeline.term_generation.candidates_tsv_dir is required for polymer_n / polymer_swap_n mode"
            )

    return TermGenerationConfig(
        mode=normalized_mode,
        n=budget,
        main_itp_dir=main_itp_dir,
        candidates_tsv_dir=candidates_tsv_dir,
    )

_WORKDIR_NAMES: Dict[Tuple[str, str], str] = {
    # (md, relaxation) -> workdir name
    ("bartender",       "xtb"):  "relax_xtb_geoopt",
    ("bartender",       "orca"): "relax_orca_geoopt",
    ("bartender",       "off"):  "relax_input_geometry",
    ("existing",        "xtb"):  "relax_xtb_geoopt_existing_traj",
    ("existing",        "orca"): "relax_orca_geoopt_existing_traj",
    ("existing",        "off"):  "existing_traj_refit",
    ("off",             "xtb"):  "relax_xtb_geoopt_only",
    ("off",             "orca"): "relax_orca_geoopt_only",
    ("off",             "off"):  "polymer_geometry_only",
    ("xtb_nobartender", "xtb"):  "relax_xtb_geoopt_xtb_nvt_only",
    ("xtb_nobartender", "orca"): "relax_orca_geoopt_xtb_nvt_only",
    ("xtb_nobartender", "off"):  "relax_xtb_nvt_only",
    ("xtb",             "xtb"):  "relax_xtb_geoopt_xtb_nvt",
    ("xtb",             "orca"): "relax_orca_geoopt_xtb_nvt",
    ("xtb",             "off"):  "relax_xtb_nvt",
}

def default_workdir_name(relaxation: str, md: str) -> str:
    """Return the conventional relaxation workdir name for a mode pair.

    Args:
        relaxation: geometry-relaxation mode ("xtb", "orca", or "off").
        md: MD/trajectory-source mode (see ``resolve_pipeline_modes``).

    Returns:
        A descriptive directory name from ``_WORKDIR_NAMES``; unknown
        relaxation values fall back to the (md, "off") entry.

    Raises:
        ValueError: when ``md`` has no entry at all.
    """
    key = (md, relaxation)
    name = _WORKDIR_NAMES.get(key)
    if name is not None:
        return name
    # md known but relaxation unexpected: fall back to off-relaxation default
    fallback = _WORKDIR_NAMES.get((md, "off"))
    if fallback is not None:
        return fallback
    raise ValueError(f"Unsupported md mode: {md}")

def _normalize_pipeline_mode(value: Any, default: str, field_name: str) -> str:
    """Lowercase a mode string; map None -> default and False -> "off".

    Boolean True is rejected because it does not name a concrete mode.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        if value is False:
            return "off"
        raise ValueError(f"{field_name} must be one of the documented string modes, not boolean true")
    return str(value).strip().lower()

def resolve_pipeline_modes(pipeline_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Resolve the workflow "flow" triple: relaxation mode, md mode, workdir.

    Three config layouts are accepted, tried in order:

    1. Flat keys ``relaxation``/``md``/``workdir_name`` at the top of
       ``bartender_pipeline`` (current layout).
    2. A nested ``mode`` mapping with the same three keys.
    3. Legacy layout: ``relaxation`` as a mapping with a ``backend`` key
       plus ``bartender.geometry_source`` deciding the md mode.

    Returns:
        Dict with keys ``relaxation`` ("xtb"/"orca"/"off"), ``md`` (one of
        bartender, xtb, existing, existing_notrim, xtb_nobartender,
        xtb_nobartender_notrim, trim, off), and ``workdir_name``.

    Raises:
        ValueError: when either mode is outside its allowed set.
    """
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
    _VALID_MD = {
        "bartender", "xtb", "existing", "existing_notrim",
        "xtb_nobartender", "xtb_nobartender_notrim", "trim", "off",
    }
    if md not in _VALID_MD:
        raise ValueError(f"bartender_pipeline.md must be one of: {', '.join(sorted(_VALID_MD))}")

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
    """Reconcile the (uhf, multiplicity) pair describing a spin state.

    Either value may be None; the missing one is derived from the relation
    multiplicity = uhf + 1 (uhf = number of unpaired electrons). Both None
    means a closed-shell singlet (0, 1).

    Args:
        uhf_value: raw uhf entry from the config, or None.
        multiplicity_value: raw multiplicity entry, or None.
        label: config location used in error messages.

    Returns:
        (uhf, multiplicity) as validated non-negative integers.

    Raises:
        ValueError: on negative uhf, multiplicity < 1, or an inconsistent
            pair (multiplicity != uhf + 1).
    """
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

def _normalize_index_list(raw: Any, *, label: str) -> List[int]:
    """Convert user 0-based atom indices to a deduplicated 1-based list.

    Accepts a single int/str or a sequence; preserves first-seen order.

    Raises:
        TypeError: on an unsupported container type.
        ValueError: on a negative (invalid 0-based) index.
    """
    if raw is None:
        return []
    if isinstance(raw, (int, str)):
        values = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        values = list(raw)
    else:
        raise TypeError(f"{label} must be an integer or a list of integers")

    normalized: List[int] = []
    seen: set[int] = set()
    for value in values:
        index = int(value)
        if index < 0:
            raise ValueError(f"{label} must contain 0-based atom indices")
        converted = index + 1
        if converted not in seen:
            normalized.append(converted)
            seen.add(converted)
    return normalized

def resolve_backbone_atom_config(raw: Any, *, label: str) -> Dict[str, List[int]]:
    """Normalize a monomer's ``backbone_atoms`` mapping.

    Args:
        raw: user mapping with optional 0-based ``head``/``tail``/``body``
            atom lists, or None for the legacy default (atoms 0 and 1, i.e.
            1-based head=[1], tail=[2]).
        label: config location used in error messages.

    Returns:
        Dict with 1-based ``head``/``tail``/``body`` index lists.

    Raises:
        TypeError: when ``raw`` is not a mapping.
        ValueError: when neither head nor tail is defined.
    """
    if raw is None:
        return {"head": [1], "tail": [2], "body": []}
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must be a mapping with optional head/body/tail atom lists")

    head = _normalize_index_list(raw.get("head"), label=f"{label}.head")
    tail = _normalize_index_list(raw.get("tail"), label=f"{label}.tail")
    body = _normalize_index_list(raw.get("body"), label=f"{label}.body")
    if not head and not tail:
        raise ValueError(f"{label} must define at least one of head or tail")
    return {"head": head, "tail": tail, "body": body}

def export_backbone_atom_config(cfg: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Convert an internal 1-based backbone config back to 0-based indices."""
    return {
        key: [int(value) - 1 for value in cfg.get(key, [])]
        for key in ("head", "tail", "body")
    }

def normalize_monomer_configs(
    raw_monomers: Dict[str, Any],
    legacy_init_templates: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Normalize the top-level ``monomers`` config section.

    Each entry may be a bare XYZ path string or a mapping with ``xyz``
    (required), ``init_template``, ``charge``, ``uhf``/``multiplicity``, and
    ``backbone_atoms``. Missing init templates fall back to the legacy
    ``bartender_pipeline.init_templates`` mapping.

    Args:
        raw_monomers: token -> raw monomer entry.
        legacy_init_templates: token -> init template path fallback.

    Returns:
        token -> normalized dict with keys xyz, init_template, charge, uhf,
        multiplicity, backbone_atoms (1-based).

    Raises:
        TypeError/ValueError: on malformed entries or a missing xyz path.
    """
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
            "backbone_atoms": resolve_backbone_atom_config(
                entry.get("backbone_atoms"),
                label=f"monomers.{token}.backbone_atoms",
            ),
        }
    return normalized

def resolve_case_electronic_state(
    tokens: Sequence[str],
    monomer_cfg: Dict[str, Dict[str, Any]],
    pipeline_cfg: Dict[str, Any],
) -> Dict[str, int]:
    """Determine the total charge and spin state of one polymer case.

    Charge and uhf are inferred by summing the per-monomer values over the
    sequence tokens; an explicit ``bartender_pipeline.electronic_state``
    mapping (or, legacy, a ``relaxation`` mapping) overrides them.

    Returns:
        Dict with charge, uhf, multiplicity plus the purely inferred
        ``inferred_charge``/``inferred_uhf`` for provenance.

    Raises:
        TypeError: when the electronic_state config is not a mapping.
        ValueError: propagated from ``resolve_spin_state`` on inconsistent
            explicit uhf/multiplicity.
    """
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
    """Resolve an optional config path; empty/None becomes None."""
    value = str(raw_value or "").strip()
    if not value:
        return None
    return resolve_under_base(base_dir, value)

def resolve_xtb_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``bartender_pipeline.xtb`` (plus legacy fallbacks) to a dict.

    Falls back to the legacy ``relaxation`` mapping for env_script, binary,
    parallel count, solvent, MD temperature, and MD length when the modern
    ``xtb`` mapping omits them.

    Returns:
        Dict of xTB settings. Units follow the key names: ``etemp`` and
        ``md_temp_k`` in K, ``md_time_ps`` in ps, ``md_dump_fs`` and
        ``md_step_fs`` in fs. The ``trim_*`` keys parameterize the
        equilibration-trimming step applied to the xTB MD trajectory before
        Bartender refitting (method "pymbar" or "energy_threshold").

    Raises:
        TypeError: when the xtb or xtb.md config is not a mapping.
        ValueError: on an unsupported solvent_model (off/alpb/gbsa).
    """
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
        "md_skip_frames": int(xtb_cfg.get("md_skip_frames", md_cfg.get("skip_frames", 0))),
        "trim_nskip": int(xtb_cfg.get("trim_nskip", 1)),
        "trim_max_fraction": float(xtb_cfg.get("trim_max_fraction", 1.0)),
        "trim_detrend": parse_bool(xtb_cfg.get("trim_detrend", False), False),
        "trim_fast": parse_bool(xtb_cfg.get("trim_fast", True), True),
        "trim_method": str(xtb_cfg.get("trim_method", "pymbar")),
        "trim_ref_fraction": float(xtb_cfg.get("trim_ref_fraction", 0.2)),
        "trim_threshold_sigma": float(xtb_cfg.get("trim_threshold_sigma", 1.0)),
    }

def resolve_orca_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``bartender_pipeline.orca`` (plus legacy fallbacks) to a dict.

    Returns:
        Dict with binary, nprocs, method_line (default builds
        "<orca_method> CPCM(<solvent>) Opt TightSCF" from legacy keys),
        max_iter, and input_template_path.

    Raises:
        TypeError: when the orca config is not a mapping.
    """
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

def _inspect_configured_executable(base_dir: Path, raw_value: Any) -> Dict[str, Any]:
    """Report how a configured executable resolves (path vs PATH lookup).

    Returns:
        Dict with ``configured`` (raw string), ``resolved`` (final path or
        None), ``exists`` (bool), and ``lookup`` describing the strategy:
        "PATH" for bare names, "path" for explicit paths, "path->PATH" when
        a missing explicit path fell back to a PATH lookup of its basename,
        "missing" for an empty entry.
    """
    configured = str(raw_value or "").strip()
    if not configured:
        return {
            "configured": configured,
            "resolved": None,
            "exists": False,
            "lookup": "missing",
        }
    if "/" in configured or configured.startswith("."):
        path = resolve_under_base(base_dir, configured)
        fallback = None
        if not path.exists():
            fallback = shutil.which(path.name)
        return {
            "configured": configured,
            "resolved": fallback or str(path),
            "exists": path.exists() or fallback is not None,
            "lookup": "path->PATH" if fallback else "path",
        }
    found = shutil.which(configured)
    return {
        "configured": configured,
        "resolved": found,
        "exists": found is not None,
        "lookup": "PATH",
    }

def resolve_executable_command(base_dir: Path, raw_value: Any) -> str:
    """Return the resolved executable path, or the raw string if unresolved.

    Falling back to the configured string keeps generated scripts honest:
    they fail loudly at run time instead of silently dropping the tool.
    """
    payload = _inspect_configured_executable(base_dir, raw_value)
    resolved = str(payload.get("resolved") or "").strip()
    if resolved:
        return resolved
    return str(payload.get("configured") or "").strip()

def _inspect_optional_file(base_dir: Path, raw_value: Any) -> Dict[str, Any]:
    """Report existence of an optional file entry; empty counts as OK."""
    configured = str(raw_value or "").strip()
    if not configured:
        return {
            "configured": configured,
            "resolved": None,
            "exists": True,
            "lookup": "optional-empty",
        }
    path = resolve_under_base(base_dir, configured)
    return {
        "configured": configured,
        "resolved": str(path),
        "exists": path.exists(),
        "lookup": "path",
    }

def check_configured_tools(cfg: Dict[str, Any], requested: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Verify that the configured external tool binaries can be found.

    Backs the ``--check-tools`` CLI option: inspects the xtb, orca, and
    bartender entries (or the subset in ``requested``) without running them.

    Args:
        cfg: full resolved config (needs ``paths.base_dir`` and
            ``bartender_pipeline``).
        requested: tool names to check; defaults to all three.

    Returns:
        Dict with overall ``ok`` (True only if every requested binary
        exists), ``base_dir``, and a per-tool ``tools`` list of inspection
        payloads (binary plus optional env_script/root entries).

    Raises:
        TypeError: when pipeline/bartender config sections are not mappings.
    """
    requested_tools = [str(value).strip().lower() for value in (requested or ("xtb", "orca", "bartender"))]
    base_dir = Path(str(cfg["paths"]["base_dir"])).resolve()
    pipeline_cfg = cfg.get("bartender_pipeline", {})
    if not isinstance(pipeline_cfg, dict):
        raise TypeError("bartender_pipeline must be a mapping")

    xtb_cfg = resolve_xtb_settings(pipeline_cfg)
    orca_cfg = resolve_orca_settings(pipeline_cfg)
    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        raise TypeError("bartender_pipeline.bartender must be a mapping")

    tools: List[Dict[str, Any]] = []
    if "xtb" in requested_tools:
        tools.append(
            {
                "name": "xtb",
                "binary": _inspect_configured_executable(base_dir, xtb_cfg.get("binary")),
                "env_script": _inspect_optional_file(base_dir, xtb_cfg.get("env_script")),
            }
        )
    if "orca" in requested_tools:
        tools.append(
            {
                "name": "orca",
                "binary": _inspect_configured_executable(base_dir, orca_cfg.get("binary")),
            }
        )
    if "bartender" in requested_tools:
        tools.append(
            {
                "name": "bartender",
                "binary": _inspect_configured_executable(base_dir, bartender_cfg.get("binary")),
                "env_script": _inspect_optional_file(base_dir, bartender_cfg.get("env_script")),
                "root": _inspect_optional_file(base_dir, bartender_cfg.get("root")),
            }
        )

    ok = True
    for tool in tools:
        ok = ok and bool(tool["binary"]["exists"])
    return {
        "ok": ok,
        "base_dir": str(base_dir),
        "tools": tools,
    }

def resolve_execution_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``bartender_pipeline.execution`` into a settings dict.

    Returns:
        Dict with run_relaxation, run_bartender (legacy fallback:
        ``bartender.execute``), shell, slurm, use_srun. ``use_srun=true``
        implies ``slurm=true``.

    Raises:
        TypeError: when the execution config is not a mapping.
    """
    exec_cfg = pipeline_cfg.get("execution", {})
    if exec_cfg is None:
        exec_cfg = {}
    if not isinstance(exec_cfg, dict):
        raise TypeError("bartender_pipeline.execution must be a mapping")

    bartender_cfg = pipeline_cfg.get("bartender", {})
    if not isinstance(bartender_cfg, dict):
        bartender_cfg = {}

    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    use_srun = parse_bool(exec_cfg.get("use_srun", False), False)
    if use_srun and not slurm_enabled:
        slurm_enabled = True

    return {
        "run_relaxation": parse_bool(exec_cfg.get("run_relaxation", False)),
        "run_bartender": parse_bool(exec_cfg.get("run_bartender", bartender_cfg.get("execute", False))),
        "shell": str(exec_cfg.get("shell", "bash")).strip() or "bash",
        "slurm": slurm_enabled,
        "use_srun": use_srun,
    }

def _get_slurm_cpu_count() -> int:
    """Return SLURM_CPUS_PER_TASK as an int >= 1, or 0 when unset/invalid."""
    val = str(os.environ.get("SLURM_CPUS_PER_TASK", "")).strip()
    if val:
        try:
            return max(1, int(val))
        except ValueError:
            return 0
    return 0

def resolve_log_settings(pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten ``bartender_pipeline.logs`` into a settings dict.

    Returns:
        Dict with enabled, dirname, write_validation, capture_runtime.

    Raises:
        TypeError: when the logs config is not a mapping.
    """
    log_cfg = pipeline_cfg.get("logs", {})
    if log_cfg is None:
        log_cfg = {}
    if not isinstance(log_cfg, dict):
        raise TypeError("bartender_pipeline.logs must be a mapping")

    return {
        "enabled": parse_bool(log_cfg.get("enabled", True), True),
        "dirname": str(log_cfg.get("dirname", "logs")).strip() or "logs",
        "write_validation": parse_bool(log_cfg.get("write_validation", True), True),
        "capture_runtime": parse_bool(log_cfg.get("capture_runtime", True), True),
    }

def ensure_case_logs_dir(case_dir: Path, log_cfg: Dict[str, Any]) -> Optional[Path]:
    """Create and return the case logs directory, or None when disabled."""
    if not log_cfg.get("enabled", True):
        return None
    logs_dir = case_dir / str(log_cfg["dirname"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def execute_case_script(
    label: str,
    script_path: Path,
    cwd: Path,
    exec_cfg: Dict[str, Any],
    logs_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run one generated case script and optionally tee its output to logs.

    When capture is enabled, stdout/stderr are streamed line-by-line to the
    live terminal *and* to ``<label>.stdout``/``<label>.stderr`` inside
    ``logs_dir`` using drain threads (so neither pipe can block). When
    ``exec_cfg`` enables slurm+use_srun, the command is wrapped in
    ``srun --export=ALL --ntasks=1``.

    Args:
        label: log-file stem, e.g. "relaxation" or "bartender".
        script_path: script to execute (its basename is run via the shell).
        cwd: working directory for the subprocess.
        exec_cfg: resolved execution settings (shell/slurm/use_srun/...).
        logs_dir: log directory, or None to disable capture.

    Returns:
        A manifest dict (script, cwd, shell, slurm flags, full command,
        returncode, stdout/stderr log names).

    Raises:
        RuntimeError: on a non-zero exit (message carries the last 20
            captured stderr lines when available) or when srun is requested
            but not on PATH.
    """
    capture_runtime = bool(logs_dir) and bool(exec_cfg.get("capture_runtime", True))
    command = [str(exec_cfg.get("shell", "bash")), script_path.name]
    slurm_enabled = parse_bool(exec_cfg.get("slurm", False), False)
    use_srun = parse_bool(exec_cfg.get("use_srun", False), False)
    if slurm_enabled and use_srun:
        if not shutil.which("srun"):
             raise RuntimeError("execution.use_srun=true but 'srun' was not found in PATH")
        srun_command = ["srun", "--export=ALL", "--ntasks=1"]
        slurm_cpus = str(os.environ.get("SLURM_CPUS_PER_TASK", "")).strip()
        if slurm_cpus:
            srun_command.extend(["--cpus-per-task", slurm_cpus])
        command = srun_command + command

    stdout_name = None
    stderr_name = None
    stderr_lines: List[str] = []

    if capture_runtime and logs_dir is not None:
        stdout_name = f"{label}.stdout"
        stderr_name = f"{label}.stderr"
        stdout_path = logs_dir / stdout_name
        stderr_path = logs_dir / stderr_name

        def _drain(src, terminal, log_fh, sink=None):
            for ln in src:
                terminal.write(ln)
                terminal.flush()
                log_fh.write(ln)
                log_fh.flush()
                if sink is not None:
                    sink.append(ln)

        with open(stdout_path, "w", encoding="utf-8") as out_fh, \
             open(stderr_path, "w", encoding="utf-8") as err_fh, \
             subprocess.Popen(
                 command, cwd=cwd, text=True,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
             ) as proc:
            t_out = threading.Thread(target=_drain, args=(proc.stdout, sys.stdout, out_fh))
            t_err = threading.Thread(target=_drain, args=(proc.stderr, sys.stderr, err_fh, stderr_lines))
            t_out.start()
            t_err.start()
            proc.wait()
            t_out.join()
            t_err.join()
        returncode = proc.returncode
    else:
        result = subprocess.run(command, cwd=cwd, text=True)
        returncode = result.returncode

    if returncode != 0:
        err_tail = "".join(stderr_lines[-20:]).strip() if stderr_lines else f"exit code {returncode}"
        raise RuntimeError(err_tail)

    return {
        "script": script_path.name,
        "cwd": str(cwd),
        "shell": str(exec_cfg.get("shell", "bash")),
        "slurm": slurm_enabled,
        "use_srun": use_srun,
        "command": command,
        "returncode": returncode,
        "stdout": stdout_name,
        "stderr": stderr_name,
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
        nprocs = int(orca_cfg.get("nprocs", 1))
        if nprocs > 1:
            if re.search(r"%pal\s+nprocs\s+\d+\s+end", template_text, re.IGNORECASE):
                template_text = re.sub(
                    r"%pal\s+nprocs\s+\d+\s+end",
                    f"%pal nprocs {nprocs} end",
                    template_text,
                    flags=re.IGNORECASE,
                )
            elif not re.search(r"%pal", template_text, re.IGNORECASE):
                template_text = f"%pal nprocs {nprocs} end\n{template_text}"

        return template_text.rstrip() + "\n\n" + (
            f"* xyzfile {int(state['charge'])} {int(state.get('multiplicity', 1))} {local_xyz_name}\n"
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
    symbols = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return symbols, coords
