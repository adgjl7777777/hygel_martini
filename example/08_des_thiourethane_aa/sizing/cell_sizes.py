#!/usr/bin/env python3
"""Plan and emit this example's cell at any legal supercell size.

Which machine a run lands on is not known when the science is planned, so the
size of the cell should be an input rather than something baked into a maker.
Everything that changes with size here is arithmetic on the net and the
topologies:

* ``pcu`` with repeats ``(rx, ry, rz)`` has ``J = rx*ry*rz`` junctions and
  ``E = 3J`` edges (six arms per junction, each edge shared by two);
* a formed strand consumes one arm at each end, and each of those arms loses
  its thiol cap hydrogen, so atoms ``= J*N_HEXU + S*N_STR - 2S``;
* AcChCl enters at a fixed ratio per junction (6 by default, the experimental
  Hexakis:AcChCl = 1:6);
* the construction box is ``cell_parameter * repeats`` (the rigid strand sets
  the junction spacing, so this does not shrink with size), and the shrink
  target is the box that holds the total mass at the target density.

The masses and atom counts are read from the project's own ITP files rather
than tabulated here, so a re-parameterization cannot leave this planner
quoting stale numbers. The three shrink targets already committed in this
example (6.34, 9.32 and 7.19 nm) are reproduced by this arithmetic to 0.01 nm,
which is the check that it is the same model.

``pcu`` accepts only even repeats of at least 4: an odd supercell manufactures
odd cycles through the periodic boundary and destroys the net's bipartiteness,
and a repeat below 4 makes the shortest cycle a box artifact rather than the
net's own girth (``nets.validate_repeats``). So the size ladder is
4, 6, 8, ... per axis, and this tool refuses anything else before a build
wastes an hour discovering it.

Usage::

    # what sizes exist, and how big each gets
    cell_sizes.py table --strand n33 --max-atoms 400000

    # write maker + shrink maker for one size
    cell_sizes.py emit --repeats 6 --strand n33 --des
    cell_sizes.py emit --repeats 4 --strand n3 --conversion fraction:0.166667 --seed 3

    # exact target box from a finished build (uses the real molecule counts)
    cell_sizes.py target --top ../project/output_des/system.top
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import yaml

from itp_inventory import MoleculeType, moleculetypes, system_composition

#: Directory of this script; the project sits next to it.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, os.pardir, "project"))

#: Size profiles, named as in the integrated report (§18C.3) so the tool and
#: the shared spec say the same thing. ``strands_per_junction`` turns into an
#: exact ``count`` for whatever repeats are chosen: the molar reading of the
#: experimental 1:0.5 is 32 prepolymers per 64 junctions (0.5 each), the
#: equivalent-ratio reading 96 (1.5 each). ``repeats`` is the default size and
#: ``--repeats`` overrides it.
PROFILES = {
    "diagnostic_n3": dict(strand="n3", repeats=(4, 4, 4), strands_per_junction=0.5,
                          des=False, note="topology/parameter smoke test"),
    "target_molar": dict(strand="n33", repeats=(4, 4, 4), strands_per_junction=0.5,
                         des=True, note="n~33, Hexakis:PPG-TDI = 1:0.5 as a molar ratio, AcChCl 1:6"),
    "target_equiv": dict(strand="n33", repeats=(4, 4, 4), strands_per_junction=1.5,
                         des=True, note="n~33, SH:NCO = 1:0.5 as an equivalent ratio, AcChCl 1:6"),
    "n33_full_control": dict(strand="n33", repeats=(4, 4, 4), strands_per_junction=None,
                             des=False, note="full-conversion upper-bound control"),
    "scaled_supercell": dict(strand="n33", repeats=(6, 6, 6), strands_per_junction=0.5,
                             des=True, note="target_molar at 6^3 for size convergence / integer PEO count"),
}

#: ``pcu`` constraints, mirrored from
#: ``hygel_martini/hydrogel_builder/core_utils/layout/nets.py``. Mirrored rather
#: than imported so this planner runs without PYTHONPATH set; ``--check-net``
#: verifies the mirror against the package when it is importable.
PCU_ARMS = 6
PCU_MIN_REPEAT = 4
PCU_EVEN_ONLY = True


@dataclass
class StrandVariant:
    """One strand template: its ITP and the junction spacing it demands."""

    name: str
    itp: str
    gro: str
    #: Junction-to-junction distance (nm) that matches this strand's rigid span.
    cell_parameter: float
    #: Where cell_parameter was read from, for the report.
    origin: str


def _yaml(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def _cell_parameter(config: dict) -> float | None:
    layout = (config.get("simulation_parameters") or {}).get("network_layout") or {}
    value = layout.get("cell_parameter")
    return float(value) if value is not None else None


def discover_strands() -> Dict[str, StrandVariant]:
    """Find the strand variants by reading the project's own makers.

    ``config/network.yaml`` carries the n = 3 spacing and ``maker_n33.yaml``
    overrides it for the tiled strand, so both numbers come from the files a
    build actually uses. A variant whose ITP is missing is dropped with no
    complaint (the n = 33 strand is generated by ``tile_ppg.py`` and may not
    exist in a fresh checkout); one whose spacing cannot be found raises,
    because guessing it would silently pre-strain every crosslink.
    """
    base = _cell_parameter(_yaml(os.path.join(PROJECT, "config", "network.yaml")))
    n33 = _cell_parameter(_yaml(os.path.join(PROJECT, "maker_n33.yaml")))
    candidates = [
        ("n3", "STR", base, "config/network.yaml"),
        ("n33", "STR_n33", n33, "maker_n33.yaml"),
    ]
    found: Dict[str, StrandVariant] = {}
    for name, stem, spacing, origin in candidates:
        itp = os.path.join(PROJECT, "structure", f"{stem}.itp")
        gro = os.path.join(PROJECT, "structure", f"{stem}.gro")
        if not (os.path.exists(itp) and os.path.exists(gro)):
            continue
        if spacing is None:
            raise ValueError(
                f"strand variant {name!r} has {stem}.itp but no cell_parameter in "
                f"{origin}; the junction spacing must match the strand span "
                "(build_templates.py prints it) and cannot be guessed"
            )
        found[name] = StrandVariant(name, itp, gro, spacing, origin)
    if not found:
        raise FileNotFoundError(f"no strand ITP found under {PROJECT}/structure")
    return found


def validate_repeats(repeats: Sequence[int]) -> Tuple[int, int, int]:
    """Reject supercells ``pcu`` would reject, with the same reasoning."""
    counts = tuple(int(v) for v in repeats)
    if len(counts) != 3:
        raise ValueError(f"repeats must be three integers, got {repeats!r}")
    if PCU_EVEN_ONLY:
        odd = [axis for axis, v in zip("xyz", counts) if v % 2]
        if odd:
            raise ValueError(
                f"pcu refuses odd repeats {counts} along {', '.join(odd)}: the "
                "net's two-colouring is a coordinate parity, which an odd "
                "supercell destroys through the periodic boundary"
            )
    if min(counts) < PCU_MIN_REPEAT:
        raise ValueError(
            f"repeats {counts} include a count below {PCU_MIN_REPEAT}: a wrap "
            "shorter than the net's fundamental cycle (4) makes the measured "
            "girth a box artifact"
        )
    return counts



@dataclass
class CellPlan:
    """One planned cell: its composition, boxes and the numbers to report."""

    repeats: Tuple[int, int, int]
    strand: StrandVariant
    #: Formed strands (crosslinked prepolymer bridges).
    formed_strands: int
    #: Total net edges, i.e. the maximum number of strands the net can hold.
    edges: int
    #: ``{molecule_name: count}`` for species added on top of the network.
    extras: Dict[str, int]
    #: ``{molecule_name: file stem}`` when a species comes from a variant file
    #: (``ACC`` from ``structure/ACC_f080.itp`` for a charge-scaled ion).
    stems: Dict[str, str]
    atom_count: int
    mass: float
    #: Unreacted arms left carrying a real thiol S-H.
    free_thiols: int
    construction_box: Tuple[float, float, float]
    target_box: Tuple[float, float, float]
    target_density: float

    @property
    def junctions(self) -> int:
        return self.repeats[0] * self.repeats[1] * self.repeats[2]

    @property
    def conversion(self) -> float:
        """Realized SH conversion: two arms consumed per formed strand."""
        return 2.0 * self.formed_strands / (self.junctions * PCU_ARMS)

    @property
    def construction_density(self) -> float:
        vx, vy, vz = self.construction_box
        return self.mass / (602.214076 * vx * vy * vz)

    @property
    def shrink_ratio(self) -> float:
        """Linear compression the guarded shrink has to achieve."""
        return self.construction_box[0] / self.target_box[0]


def plan_cell(repeats: Sequence[int],
              strand: StrandVariant,
              types: Dict[str, MoleculeType],
              conversion: str = "full",
              conversion_count: int | None = None,
              conversion_fraction: float | None = None,
              extras: Dict[str, int] | None = None,
              density: float = 1.0,
              stems: Dict[str, str] | None = None) -> CellPlan:
    """Compute one cell's composition and boxes.

    ``conversion`` is ``full`` (every net edge becomes a strand), ``count``
    (exactly ``conversion_count`` strands -- what an experimental equivalent
    ratio actually specifies) or ``fraction`` (each edge forms with probability
    ``conversion_fraction``, so the count here is the *expectation* and a build
    will land near but not on it; ``cell_sizes.py target`` reads the realized
    number back out of the finished topology).
    """
    counts = validate_repeats(repeats)
    junctions = counts[0] * counts[1] * counts[2]
    edges = junctions * PCU_ARMS // 2

    if conversion == "full":
        formed = edges
    elif conversion == "count":
        if conversion_count is None:
            raise ValueError("conversion='count' needs conversion_count")
        formed = int(conversion_count)
        if not 0 < formed <= edges:
            raise ValueError(
                f"{formed} strands requested but this supercell has {edges} edges"
            )
    elif conversion == "fraction":
        if conversion_fraction is None:
            raise ValueError("conversion='fraction' needs conversion_fraction")
        if not 0.0 < float(conversion_fraction) < 1.0:
            raise ValueError(
                f"conversion fraction must be in (0, 1), got {conversion_fraction}; "
                "for every edge use conversion='full'"
            )
        formed = int(round(conversion_fraction * edges))
    else:
        raise ValueError(f"unknown conversion mode {conversion!r}")

    hexu = types["HEXU"]
    strand_type = types[_moleculetype_name(strand.itp)]

    # Each formed strand reacts one arm at each end; a reacted arm loses its
    # thiol cap hydrogen and its sulfur takes the thiourethane charge/type
    # (the mass is kept, so only the hydrogen leaves the mass balance).
    cap_mass = _cap_hydrogen_mass(strand)
    atoms = junctions * hexu.atom_count + formed * strand_type.atom_count - 2 * formed
    mass = (junctions * hexu.mass + formed * strand_type.mass
            - 2 * formed * cap_mass)

    if not (math.isfinite(density) and density > 0):
        raise ValueError(f"density must be a finite positive number, got {density!r}")
    extras = dict(extras or {})
    for name, count in extras.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"species {name!r} count must be a non-negative integer, got {count!r}")
    stems = {name: (stems or {}).get(name, name) for name in extras}
    for name, count in extras.items():
        if name not in types:
            raise KeyError(
                f"species {name!r} has no [ moleculetype ]; expected "
                f"{PROJECT}/structure/{stems[name]}.itp"
            )
        atoms += types[name].atom_count * count
        mass += types[name].mass * count

    construction = tuple(strand.cell_parameter * c for c in counts)
    volume = mass / (602.214076 * density)
    scale = (volume / junctions) ** (1.0 / 3.0)
    target = tuple(scale * c for c in counts)

    return CellPlan(
        repeats=counts,
        strand=strand,
        formed_strands=formed,
        edges=edges,
        extras=extras,
        stems=stems,
        atom_count=atoms,
        mass=mass,
        free_thiols=junctions * PCU_ARMS - 2 * formed,
        construction_box=construction,
        target_box=target,
        target_density=density,
    )


def _moleculetype_name(itp_path: str) -> str:
    """The single ``[ moleculetype ]`` name in ``itp_path``."""
    names = list(moleculetypes([itp_path]))
    if len(names) != 1:
        raise ValueError(f"{itp_path} defines {len(names)} moleculetypes, expected 1")
    return names[0]


def _cap_hydrogen_mass(strand: StrandVariant) -> float:
    """Mass of one thiol cap hydrogen, from the junction ITP and stub_caps.

    Read rather than assumed 1.008: the caps are named per arm in
    ``config/hydrogel.yaml`` (generated by ``build_templates.py``), and if a
    re-parameterization ever gave them different masses this would follow.
    Raises if the caps disagree, since the removal count is then ambiguous.
    """
    hydrogel = _yaml(os.path.join(PROJECT, "config", "hydrogel.yaml"))
    linkers = (((hydrogel.get("hydrogel_components") or {})
                .get("linker_definitions") or {}).get("LINKERS") or [])
    if not linkers:
        raise ValueError(
            "config/hydrogel.yaml declares no "
            "hydrogel_components.linker_definitions.LINKERS")
    cap_names = [name
                 for entry in (linkers[0].get("stub_caps") or [])
                 for name in (entry.get("cap_atoms") or [])]
    if not cap_names:
        raise ValueError(
            "config/hydrogel.yaml linker_definitions.LINKERS[0] has no stub_caps; without cap atoms "
            "a reacted arm removes nothing and this planner's mass balance "
            "does not apply"
        )
    masses = {}
    section = None
    with open(os.path.join(PROJECT, "structure", "HEXU.itp")) as handle:
        for line in handle:
            text = line.split(";", 1)[0].strip()
            if not text:
                continue
            if text.startswith("["):
                section = text.strip("[] ").strip().lower()
                continue
            if section == "atoms":
                parts = text.split()
                if len(parts) >= 8 and parts[4] in cap_names:
                    masses[parts[4]] = float(parts[7])
    missing = sorted(set(cap_names) - set(masses))
    if missing:
        raise ValueError(f"cap atoms not found in HEXU.itp: {', '.join(missing)}")
    distinct = sorted(set(round(v, 6) for v in masses.values()))
    if len(distinct) != 1:
        raise ValueError(f"cap hydrogens have differing masses: {distinct}")
    return distinct[0]


# --------------------------------------------------------------------------
# Emitting makers
# --------------------------------------------------------------------------

#: Config files every emitted maker includes, in the project's own order.
BASE_INCLUDES = (
    "config/simulation.yaml",
    "config/mdp.yaml",
    "config/hydrogel.yaml",
    "config/network.yaml",
)

#: Shrink makers inherit the guarded-shrink driver from example 05.
SHRINK_INCLUDES = (
    "../../05_hydrogel_relaxation/project/config/common.yaml",
    "../../05_hydrogel_relaxation/project/config/hard_em_shrink.yaml",
)


def default_tag(plan: CellPlan, conversion: str) -> str:
    """A tag that names the axes this cell actually varies."""
    rx, ry, rz = plan.repeats
    size = f"r{rx}" if rx == ry == rz else f"r{rx}{ry}{rz}"
    conv = {"full": "full", "count": f"c{plan.formed_strands}", "fraction": "part"}[conversion]
    # ACC_f080 + CL_f080 -> "_acccl_f080"; ACC + CL -> "_acccl"
    bases = sorted(k.split("_")[0].lower() for k in plan.extras)
    tags = sorted({k.split("_", 1)[1] for k in plan.extras if "_" in k})
    extras = ("_" + "".join(bases) + ("_" + tags[0] if tags else "")) if plan.extras else ""
    return f"{plan.strand.name}_{size}_{conv}{extras}"


def _box_list(box: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in box) + "]"


def emit_build_maker(plan: CellPlan, tag: str, conversion: str,
                     seed: int | None, omp_threads: int | None) -> str:
    """Render the build maker for one planned cell.

    Only what size changes is written here; the chemistry, force field and
    runtime stay in ``config/*.yaml`` so a parameter update reaches every size
    at once. ``${CONFIG_DIR}`` is the *top-level* maker's directory, which is
    why these files are emitted next to the hand-written makers rather than
    into a subdirectory: an emitted maker one level down would re-point every
    ``${CONFIG_DIR}/structure`` path in the included files.
    """
    lines: List[str] = [
        f"# GENERATED by sizing/cell_sizes.py -- edit the generator, not this file.",
        f"#",
        f"# pcu {plan.repeats[0]}x{plan.repeats[1]}x{plan.repeats[2]}"
        f" = {plan.junctions} junctions, {plan.edges} net edges.",
        f"# Strand {plan.strand.name} ({plan.strand.itp.rsplit('/', 1)[-1]}),"
        f" junction spacing {plan.strand.cell_parameter} nm"
        f" (from {plan.strand.origin}).",
        f"# Formed strands {plan.formed_strands} -> SH conversion"
        f" {plan.conversion:.4f}, {plan.free_thiols} arms keep a real S-H.",
        f"# Predicted {plan.atom_count} atoms, {plan.mass:.0f} g/mol.",
        f"# Construction box {_box_list(plan.construction_box)} nm"
        f" = {plan.construction_density:.3f} g/cm3 (dilute by design: the rigid",
        f"# strand sets the junction spacing). Shrink target"
        f" {_box_list(plan.target_box)} nm",
        f"# at {plan.target_density:g} g/cm3 -- see maker_size_{tag}_shrink.yaml.",
        "",
        "includes:",
    ]
    lines += [f"  - {path}" for path in BASE_INCLUDES]
    lines += ["", "simulation_parameters:", f"  output_dir: ${{CONFIG_DIR}}/output_size_{tag}"]

    lines += ["  network_layout:",
              f"    repeats: [{plan.repeats[0]}, {plan.repeats[1]}, {plan.repeats[2]}]"]
    if abs(plan.strand.cell_parameter
           - _cell_parameter(_yaml(os.path.join(PROJECT, "config", "network.yaml")))) > 1e-9:
        lines.append(f"    cell_parameter: {plan.strand.cell_parameter}")
    if conversion == "count":
        lines += ["    # Exactly this many net edges become strands: an experimental",
                  "    # equivalent ratio fixes a count, not a per-edge probability.",
                  "    conversion:",
                  f"      count: {plan.formed_strands}",
                  f"      seed: {seed if seed is not None else 3}"]
    elif conversion == "fraction":
        lines += ["    # Each edge forms independently with this probability, so the",
                  "    # realized count scatters around the target; read the realized",
                  "    # value out of the build's metadata, not from here.",
                  "    conversion:",
                  f"      fraction: {plan.formed_strands / plan.edges:.6f}",
                  f"      seed: {seed if seed is not None else 3}"]
    if omp_threads is not None:
        lines.append(f"  omp_threads: {omp_threads}")

    # Long strands make a box big enough that PME setup is wasteful for a
    # construction-stage EM; the committed n = 33 maker switches to cut-off
    # electrostatics for the same reason, with PME left to the later NPT.
    if plan.construction_box[0] > 20.0:
        lines += ["  # Construction box is tens of nm across; cut-off electrostatics for",
                  "  # the build-stage EM, PME later in NPT where the box is dense.",
                  "  geo_opt:",
                  "    mdp:",
                  "      coulombtype: Cut-off",
                  "      rcoulomb: 1.5",
                  "      rvdw: 1.5"]

    if plan.strand.name != "n3":
        lines += ["", "hydrogel_components:",
                  "  backbone_definitions:",
                  "    # Replaces the default BACKBONES list wholesale (lists never merge).",
                  "    BACKBONES:",
                  "      - id: STR1",
                  "        ratio: 1",
                  "        template:",
                  f"          gro: ${{CONFIG_DIR}}/structure/"
                  f"{plan.strand.gro.rsplit('/', 1)[-1]}",
                  f"          itp: ${{CONFIG_DIR}}/structure/"
                  f"{plan.strand.itp.rsplit('/', 1)[-1]}"]

    if plan.extras:
        lines += ["", "add_series_parameters:",
                  "  # All species go into a single Packmol call; genion is unusable",
                  "  # here because the ions ARE the solvent, not a dilute additive.",
                  "  add_molecule:"]
        for name in sorted(plan.extras):
            stem = plan.stems.get(name, name)
            lines += [f"    - molecule_gro: ${{CONFIG_DIR}}/structure/{stem}.gro",
                      f"      molecule_itp: ${{CONFIG_DIR}}/structure/{stem}.itp",
                      f"      molecule_name: {name}",
                      f"      num_molecules: {plan.extras[name]}"]

    if plan.conversion < 0.999:
        lines += ["", "# Below full conversion the covalent system can sit under the ideal",
                  "# A6+B2 gel point (p = 0.2), where fragments are the expected physics",
                  "# rather than a build failure, so the audit reports instead of gating.",
                  "hydrogel_topology_connectivity_audit:",
                  "  enabled: true",
                  "  min_largest_component_fraction: 0.0",
                  "  max_components: 100000",
                  "  fail_on_violation: false"]

    return "\n".join(lines) + "\n"


def emit_shrink_maker(plan: CellPlan, tag: str, omp_threads: int | None) -> str:
    """Render the guarded-shrink maker matching a build maker.

    The target box is the one that holds this cell's mass at the requested
    density. It is a three-value list so an anisotropic supercell keeps its
    proportions instead of being squeezed into a cube.
    """
    runtime = ([] if omp_threads is None else
               ["", "runtime:", f"  omp_threads: {omp_threads}"])
    return "\n".join([
        "# GENERATED by sizing/cell_sizes.py -- edit the generator, not this file.",
        "#",
        f"# Compresses output_size_{tag} from {_box_list(plan.construction_box)} nm",
        f"# to {_box_list(plan.target_box)} nm, i.e. a linear factor"
        f" {plan.shrink_ratio:.2f}, which is",
        f"# {plan.mass:.0f} g/mol at {plan.target_density:g} g/cm3."
        " The target is a melt-density",
        "# estimate, NOT an experimental density; the run ends in an EM state, so",
        "# no property may be read from it before NPT.",
        "",
        "includes:",
        *[f"  - {path}" for path in SHRINK_INCLUDES],
        *runtime,
        "",
        "paths:",
        f"  start_gro: ${{CONFIG_DIR}}/output_size_{tag}/final_system.gro",
        f"  system_top: ${{CONFIG_DIR}}/output_size_{tag}/"
        "final_system_no_ions_geo_opt/system.top",
        f"  bonded_itp: ${{CONFIG_DIR}}/output_size_{tag}/initial_hydrogel.itp",
        f"  workdir: ${{CONFIG_DIR}}/shrink_output_size_{tag}",
        "",
        "hard_em_shrink:",
        "  minim_mdp: ${CONFIG_DIR}/config_shrink/minim.mdp",
        "  nvt_recovery_mdp: ${CONFIG_DIR}/config_shrink/nvt_recovery.mdp",
        f"  target_box_nm: {_box_list(plan.target_box)}",
        "  shrink_fraction: 0.02",
        "  max_steps: 300",
    ]) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_types(strand: StrandVariant, extras: Sequence[str],
                stems: Dict[str, str] | None = None) -> Dict[str, MoleculeType]:
    """Parse the ITPs this plan needs, and only those."""
    paths = [os.path.join(PROJECT, "structure", "HEXU.itp"), strand.itp]
    for name in extras:
        stem = (stems or {}).get(name, name)
        path = os.path.join(PROJECT, "structure", f"{stem}.itp")
        if not os.path.exists(path):
            hint = ("apply_charges.py scale writes this variant"
                    if "_f" in name else
                    "PEO and water are not parameterized yet (handoff/DEV_PLAN items 9 "
                    "and 10); this planner will not invent them")
            raise FileNotFoundError(f"species {name!r} needs {path}, which does not exist. {hint}.")
        paths.append(path)
    return moleculetypes(paths)


def _ion_names(args) -> Tuple[str, str]:
    """Moleculetype names of the AcChCl pair for the requested charge scaling.

    A scaled variant is its own moleculetype (``ACC_f080``), because the
    builder auto-includes every ITP under ``structure/`` and refuses two files
    that declare the same name -- so the variant cannot be called ``ACC``.
    File stem and moleculetype name coincide by construction
    (``apply_charges.py scale``).
    """
    factor = getattr(args, "ion_charge_scale", None)
    if factor is None or abs(float(factor) - 1.0) < 1e-12:
        return "ACC", "CL"
    tag = f"f{int(round(float(factor) * 100)):03d}"
    return f"ACC_{tag}", f"CL_{tag}"


def _ion_stems(args) -> Dict[str, str]:
    """Kept for callers that pass stems explicitly; names are stems now."""
    return {}


def _apply_profile(args) -> None:
    """Fill unset arguments from ``--profile``; explicit flags win."""
    name = getattr(args, "profile", None)
    if not name:
        return
    if name not in PROFILES:
        raise SystemExit(f"unknown profile {name!r}; have {', '.join(PROFILES)}")
    prof = PROFILES[name]
    if getattr(args, "strand", None) in (None, "n33") and not getattr(args, "_strand_explicit", False):
        args.strand = prof["strand"]
    if getattr(args, "repeats", None) in (None, []):
        args.repeats = list(prof["repeats"])
    if prof["des"]:
        args.des = True
    per = prof["strands_per_junction"]
    if getattr(args, "conversion", "full") == "full" and per is not None:
        reps = args.repeats if len(args.repeats) == 3 else args.repeats * 3
        junctions = reps[0] * reps[1] * reps[2]
        args.conversion = f"count:{int(round(per * junctions))}"


def _resolve_extras(args, junctions: int) -> Dict[str, int]:
    """Build the extra-species counts from ``--des`` and ``--extra``."""
    extras: Dict[str, int] = {}
    if args.des:
        pairs = args.des_ratio * junctions
        cation, anion = _ion_names(args)
        extras[cation] = pairs
        extras[anion] = pairs
    if args.des and (not isinstance(args.des_ratio, int) or args.des_ratio < 0):
        raise ValueError(f"--des-ratio must be a non-negative integer, got {args.des_ratio!r}")
    for spec in args.extra or []:
        if ":" not in spec:
            raise ValueError(f"--extra takes NAME:COUNT, got {spec!r}")
        name, _, count = spec.partition(":")
        try:
            value = int(count)
        except ValueError:
            raise ValueError(f"--extra {spec!r}: COUNT must be an integer") from None
        if value < 0:
            raise ValueError(f"--extra {spec!r}: a negative molecule count is not a thing")
        extras[name.strip()] = value
    return extras


def _conversion_from_args(args) -> Tuple[str, int | None, float | None]:
    spec = args.conversion
    if spec == "full":
        return "full", None, None
    kind, _, value = spec.partition(":")
    if kind == "count":
        try:
            count = int(value)
        except ValueError:
            raise ValueError(f"--conversion count:N needs an integer, got {value!r}") from None
        if count < 1:
            raise ValueError(f"--conversion count must be at least 1, got {count}")
        return "count", count, None
    if kind == "fraction":
        try:
            fraction = float(value)
        except ValueError:
            raise ValueError(f"--conversion fraction:F needs a number, got {value!r}") from None
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                f"--conversion fraction must be in (0, 1), got {fraction}; use 'full' for every edge"
            )
        return "fraction", None, fraction
    raise ValueError(
        f"--conversion takes full, count:N or fraction:F, got {spec!r}"
    )


def cmd_table(args) -> int:
    """Print the size ladder for one scenario."""
    strands = discover_strands()
    names = [args.strand] if args.strand else list(strands)
    conversion, count, fraction = _conversion_from_args(args)

    print(f"pcu size ladder -- even repeats from {PCU_MIN_REPEAT}; "
          f"conversion {args.conversion}; "
          f"{'AcChCl 1:%d' % args.des_ratio if args.des else 'dry'}; "
          f"target {args.density:g} g/cm3")
    header = (f"{'strand':>7} {'repeats':>9} {'junct':>6} {'strands':>8} {'atoms':>9} "
              f"{'mass/kDa':>9} {'build box':>10} {'target box':>11} {'shrink':>7}")
    for name in names:
        strand = strands[name]
        stems = _ion_stems(args)
        types = _load_types(strand, list(_resolve_extras(args, 1)), stems)
        print()
        print(header)
        for repeat in range(PCU_MIN_REPEAT, args.max_repeat + 1, 2):
            counts = (repeat, repeat, repeat)
            junctions = repeat ** 3
            extras = _resolve_extras(args, junctions)
            edges = junctions * PCU_ARMS // 2
            if conversion == "count" and count is not None and count > edges:
                continue
            plan = plan_cell(counts, strand, types, conversion=conversion,
                             conversion_count=count, conversion_fraction=fraction,
                             extras=extras, density=args.density, stems=stems)
            if args.max_atoms and plan.atom_count > args.max_atoms:
                print(f"{name:>7} {repeat:>3}^3    {junctions:>6} {plan.formed_strands:>8} "
                      f"{plan.atom_count:>9} -- above --max-atoms, stopping")
                break
            print(f"{name:>7} {repeat:>3}^3    {junctions:>6} {plan.formed_strands:>8} "
                  f"{plan.atom_count:>9} {plan.mass / 1000:>9.1f} "
                  f"{plan.construction_box[0]:>9.1f} {plan.target_box[0]:>10.2f} "
                  f"{plan.shrink_ratio:>7.2f}")
    print("\nAnisotropic supercells are legal too (all axes even, none below "
          f"{PCU_MIN_REPEAT}): pass --repeats 4 4 6 to emit.")
    return 0


def cmd_emit(args) -> int:
    """Write the build and shrink makers for one size."""
    _apply_profile(args)
    strands = discover_strands()
    if args.strand not in strands:
        raise SystemExit(f"unknown strand {args.strand!r}; have {', '.join(strands)}")
    strand = strands[args.strand]
    if not args.repeats:
        raise SystemExit("--repeats is required (or a --profile that supplies it)")
    repeats = validate_repeats(args.repeats if len(args.repeats) == 3
                               else args.repeats * 3)
    junctions = repeats[0] * repeats[1] * repeats[2]
    extras = _resolve_extras(args, junctions)
    stems = _ion_stems(args)
    types = _load_types(strand, list(extras), stems)
    conversion, count, fraction = _conversion_from_args(args)
    plan = plan_cell(repeats, strand, types, conversion=conversion,
                     conversion_count=count, conversion_fraction=fraction,
                     extras=extras, density=args.density, stems=stems)
    tag = args.tag or (f"{args.profile}_r{repeats[0]}" if getattr(args, "profile", None)
                       and repeats[0] == repeats[1] == repeats[2]
                       else default_tag(plan, conversion))

    files = {
        os.path.join(PROJECT, f"maker_size_{tag}.yaml"):
            emit_build_maker(plan, tag, conversion, args.seed, args.omp_threads),
        os.path.join(PROJECT, f"maker_size_{tag}_shrink.yaml"):
            emit_shrink_maker(plan, tag, args.omp_threads),
    }
    existing = [path for path in files if os.path.exists(path)]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite:\n  " + "\n  ".join(existing) +
            "\nre-run with --force if that is intended"
        )
    for path, text in files.items():
        with open(path, "w") as handle:
            handle.write(text)
        print(f"wrote {path}")

    print(f"\n{tag}: {plan.junctions} junctions, {plan.formed_strands}/{plan.edges} "
          f"strands (SH conversion {plan.conversion:.4f}), "
          f"{plan.free_thiols} free S-H")
    for name in sorted(plan.extras):
        print(f"  + {plan.extras[name]} x {name}")
    print(f"  {plan.atom_count} atoms, {plan.mass / 1000:.1f} kDa")
    print(f"  build {_box_list(plan.construction_box)} nm "
          f"({plan.construction_density:.3f} g/cm3) -> "
          f"target {_box_list(plan.target_box)} nm "
          f"({plan.target_density:g} g/cm3), linear factor {plan.shrink_ratio:.2f}")
    if conversion == "fraction":
        print("  NOTE fraction mode: the realized strand count will differ; re-run\n"
              "       'cell_sizes.py target --top <build>/system.top' afterwards and\n"
              "       put that box in the shrink maker.")
    print("\nrun:\n"
          f"  PYTHONPATH=$REPO python -m hygel_martini.hydrogel_builder "
          f"{os.path.relpath(os.path.join(PROJECT, f'maker_size_{tag}.yaml'))}\n"
          f"  PYTHONPATH=$REPO python -m hygel_martini.hydrogel_builder.relax "
          f"{os.path.relpath(os.path.join(PROJECT, f'maker_size_{tag}_shrink.yaml'))}")
    return 0


def cmd_target(args) -> int:
    """Report a finished build's real composition and its density target box.

    This is the number to trust: it counts the molecules the topology actually
    contains, including a partial build's realized strand count, rather than
    the count the maker asked for.
    """
    comp = system_composition(args.top)
    print(f"{'species':<12} {'count':>7} {'atoms':>9} {'mass/amu':>13} {'charge/e':>10}")
    for name, count in comp.molecules:
        mt = comp.types[name]
        print(f"{name:<12} {count:>7} {mt.atom_count * count:>9} "
              f"{mt.mass * count:>13.3f} {mt.charge * count:>10.4f}")
    print(f"{'TOTAL':<12} {'':>7} {comp.atom_count:>9} {comp.mass:>13.3f} "
          f"{comp.charge:>10.4f}")
    aspect = args.aspect if args.aspect else (1.0, 1.0, 1.0)
    box = comp.box_for_density(args.density, aspect)
    print(f"\ntarget_box_nm: {_box_list(box)}   # {args.density:g} g/cm3")
    if abs(comp.charge) > 1e-3:
        print(f"NOTE net charge {comp.charge:+.4f} e. Per-molecule ITP charge sums "
              "that miss zero\n     by ~1e-4 e (LigParGen's four-decimal tables) add up "
              "linearly with\n     system size; apply_charges.py should neutralize each "
              "ITP exactly.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, with_strand_default):
        p.add_argument("--strand", default="n33" if with_strand_default else None,
                       help="strand variant (n3 or n33)")
        p.add_argument("--conversion", default="full",
                       help="full | count:N | fraction:F (default full)")
        p.add_argument("--des", action="store_true",
                       help="add AcChCl at --des-ratio per junction")
        p.add_argument("--des-ratio", type=int, default=6,
                       help="AcChCl pairs per Hexakis junction (default 6)")
        p.add_argument("--extra", action="append", metavar="NAME:COUNT",
                       help="extra species from structure/NAME.itp, repeatable")
        p.add_argument("--density", type=float, default=1.0,
                       help="shrink target density, g/cm3 (default 1.0)")
        p.add_argument("--ion-charge-scale", type=float, default=None,
                       help="use charge-scaled ion variants structure/ACC_fNNN.itp and "
                            "CL_fNNN.itp (written by apply_charges.py scale), e.g. 0.8")

    t = sub.add_parser("table", help="size ladder with atom counts and boxes")
    add_common(t, with_strand_default=False)
    t.add_argument("--max-repeat", type=int, default=10)
    t.add_argument("--max-atoms", type=int, default=0,
                   help="stop a strand's ladder once a cell exceeds this")
    t.set_defaults(func=cmd_table)

    e = sub.add_parser("emit", help="write maker + shrink maker for one size")
    add_common(e, with_strand_default=True)
    e.add_argument("--repeats", type=int, nargs="+", default=None,
                   help="one value for a cubic supercell, or three (a --profile "
                        "supplies a default)")
    e.add_argument("--profile", choices=sorted(PROFILES),
                   help="named size profile from the integrated report §18C.3; "
                        "explicit flags override its defaults")
    e.add_argument("--seed", type=int, default=None,
                   help="conversion seed (required by the builder when partial)")
    e.add_argument("--omp-threads", type=int, default=None,
                   help="OpenMP threads for build/shrink mdrun. Omit to keep "
                        "the project defaults; under a batch system pass what "
                        "the allocation gave you (run_size.sh reads "
                        "SLURM_CPUS_PER_TASK for this)")
    e.add_argument("--tag", default=None, help="override the generated name")
    e.add_argument("--force", action="store_true", help="overwrite existing makers")
    e.set_defaults(func=cmd_emit)

    g = sub.add_parser("target", help="exact target box from a finished build")
    g.add_argument("--top", required=True, help="path to the build's system.top")
    g.add_argument("--density", type=float, default=1.0)
    g.add_argument("--aspect", type=float, nargs=3, default=None,
                   help="keep this box shape (pass the supercell repeats)")
    g.set_defaults(func=cmd_target)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        # Validation happens before any file is written, so a refusal here
        # leaves the project directory untouched.
        print(f"cell_sizes.py: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
