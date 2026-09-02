"""Junction-crossing dihedrals and 1-4 pairs for all-atom mode.

The builder already enumerates every angle. What all-atom force fields add is
the next shell out: dihedrals along three-bond paths through builder-created
bonds, and the 1-4 pairs that nrexcl=3 excludes. Both are emitted
parameterless, so the force field's [ *types ] tables own the numbers and the
builder stays force-field-agnostic. Paths internal to one template are the
template's own business and are never regenerated -- for pairs that would
double-count 1-4 energy, silently.
"""

from __future__ import annotations

import pytest

from hygel_martini.hydrogel_builder.core_utils.io.writer import write_combined_itp
from hygel_martini.hydrogel_builder.core_utils.runtime.aa_bonded import (
    generate_junction_bonded_terms,
)
from hygel_martini.hydrogel_builder.main_components import Attributes
from hygel_martini.hydrogel_builder.main_components.Universe import World


class _Template:
    """Stand-in for a source template; only identity matters here."""


def _fresh_world():
    World.Atoms.clear()
    World.Bonds.clear()
    World.Angles.clear()
    World.Dihedrals.clear()
    # World is a class-level registry, so generated pairs from one test would
    # leak into the next -- exactly the shape of bug this suite exists to
    # catch elsewhere.
    World.generated_pairs = None
    Attributes.initialize()
    return World


def _chain(template, count, start_mass=72.0):
    ids = []
    for _ in range(count):
        atom = Attributes.Atom(source_template=template)
        atom.mass = start_mass
        ids.append(atom.atom_id)
    for left, right in zip(ids, ids[1:]):
        Attributes.Bond(left, right, funct=1, c0=0.15, c1=1000.0)
    return ids


def test_a_crosslink_bond_generates_its_crossing_dihedrals_and_pairs() -> None:
    # Two three-atom chains from different templates, joined by one new bond:
    #   a0-a1-a2  x  b0-b1-b2   (crosslink a2-b0)
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.15, c1=1000.0)

    added, pairs, _ = generate_junction_bonded_terms(world)

    # Crossing torsions: a0-a1-a2-b0, a1-a2-b0-b1, a2-b0-b1-b2.
    assert added == 3
    keys = {tuple(key) for key in world.Dihedrals}
    canonical = {min(key, tuple(reversed(key))) for key in keys}
    assert canonical == {
        (chain_a[0], chain_a[1], chain_a[2], chain_b[0]),
        (chain_a[1], chain_a[2], chain_b[0], chain_b[1]),
        (chain_a[2], chain_b[0], chain_b[1], chain_b[2]),
    }
    # 1-4 pairs are those torsions' endpoints, each at bond distance three.
    assert pairs == [
        (chain_a[0], chain_b[0]),
        (chain_a[1], chain_b[1]),
        (chain_a[2], chain_b[2]),
    ]
    assert world.generated_pairs == pairs


def test_template_internal_paths_are_never_regenerated() -> None:
    # One template owns the whole chain: every three-bond path is internal, so
    # regenerating its dihedrals or pairs would double-count what the template
    # ITP already declares (or deliberately omits).
    world = _fresh_world()
    _chain(_Template(), 5)

    added, pairs, _ = generate_junction_bonded_terms(world)

    assert added == 0
    assert pairs == []


def test_ring_closures_are_kept_out_of_the_one_four_list() -> None:
    # A four-ring built from two templates: every endpoint pair of a
    # three-bond path is ALSO a 1-2 neighbour through the other side, so no
    # pair qualifies as 1-4, while the crossing torsions still exist.
    world = _fresh_world()
    left = _Template()
    right = _Template()
    a0 = Attributes.Atom(source_template=left)
    a1 = Attributes.Atom(source_template=left)
    b0 = Attributes.Atom(source_template=right)
    b1 = Attributes.Atom(source_template=right)
    for i, j in ((a0.atom_id, a1.atom_id), (a1.atom_id, b0.atom_id),
                 (b0.atom_id, b1.atom_id), (b1.atom_id, a0.atom_id)):
        Attributes.Bond(i, j, funct=1, c0=0.15, c1=1000.0)

    added, pairs, _ = generate_junction_bonded_terms(world)

    assert added > 0
    assert pairs == []


def test_existing_template_dihedrals_are_not_duplicated() -> None:
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.15, c1=1000.0)

    # Suppose an earlier stage already registered one of the crossing torsions
    # (with parameters). The walk must keep it and not add a twin.
    existing = Attributes.Dihedral(chain_a[1], chain_a[2], chain_b[0], chain_b[1], 0)
    existing.dihedral_funct = 1
    existing.dihedral_c0 = 180.0
    existing.dihedral_c1 = 5.0

    added, _, _ = generate_junction_bonded_terms(world)

    assert added == 2  # the other two crossing torsions only


def test_the_writer_emits_nrexcl_pairs_and_parameterless_terms(tmp_path) -> None:
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.15, c1=1000.0)
    generate_junction_bonded_terms(world)

    path = tmp_path / "aa.itp"
    write_combined_itp(world, filename=str(path), moleculetype_name="AAMOL", nrexcl=3)
    text = path.read_text()

    assert "AAMOL           3" in text
    # generated pairs present, 1-based
    assert "[ pairs ]" in text
    assert f"{chain_a[0] + 1}  {chain_b[0] + 1}   1" in text
    # parameterless dihedral line: five integers, nothing after funct
    dihedral_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and line.split() and "dihedrals" not in line
    ]
    parameterless = [
        line for line in dihedral_lines
        if len(line.split()) == 5 and line.split()[4] == "3"
    ]
    assert len(parameterless) == 3
    # and the round trip parses: parameterless entries survive the shared parser
    from hygel_martini.core.itp import read_itp_definitions

    definition = read_itp_definitions(str(path), require_mass=False)["AAMOL"]
    assert len(definition["dihedrals"]) == 3
    assert all(d["params"] == [] for d in definition["dihedrals"])
    assert len(definition["pairs"]) == 3


def test_martini_defaults_are_untouched(tmp_path) -> None:
    # No junction_bonded_generation config, no generated pairs: nrexcl stays 1
    # and no [ pairs ] section appears.
    world = _fresh_world()
    _chain(_Template(), 3)

    path = tmp_path / "cg.itp"
    write_combined_itp(world, filename=str(path), moleculetype_name="CGMOL")
    text = path.read_text()

    assert "CGMOL           1" in text
    assert "[ pairs ]" not in text


def test_two_instances_of_one_template_still_generate_crossing_terms() -> None:
    # Ownership is per molecule instance. Two chains stamped from the SAME
    # template object, joined by a builder bond, must still get their crossing
    # dihedrals -- identity-based comparison silently skipped exactly this.
    world = _fresh_world()
    shared = _Template()
    chain_a = _chain(shared, 3)
    chain_b = _chain(shared, 3)
    for atom_id in chain_a:
        world.Atoms[atom_id][0].chain_type = "linker"
        world.Atoms[atom_id][0].chain_index = 0
    for atom_id in chain_b:
        world.Atoms[atom_id][0].chain_type = "linker"
        world.Atoms[atom_id][0].chain_index = 1
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.15, c1=1000.0)

    added, pairs, _ = generate_junction_bonded_terms(world)

    assert added == 3
    assert len(pairs) == 3


def test_generated_pairs_merge_with_preexisting_template_pairs() -> None:
    # The populator registers template-internal 1-4s on the same World list;
    # the junction walk must union with them, not overwrite them.
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.15, c1=1000.0)
    world.generated_pairs = [(97, 99)]  # as if copied from a template

    _, crossing, _ = generate_junction_bonded_terms(world)

    assert (97, 99) in world.generated_pairs
    assert set(crossing) <= set(world.generated_pairs)
    assert len(world.generated_pairs) == len(crossing) + 1


def test_written_charges_survive_the_round_trip(tmp_path) -> None:
    # The writer used to print four decimals, which is lossless for Martini
    # beads and systematically lossy for all-atom charges: 57600 atoms sharing
    # a five-decimal value each lost ~4e-5 e in the same direction, giving a
    # network topology with -2.33 e that no template accounted for.
    world = _fresh_world()
    template = _Template()
    charges = [0.03574, -0.28376, 0.11642, -0.35694, 0.09374]
    ids = []
    for charge in charges:
        atom = Attributes.Atom(source_template=template)
        atom.mass = 12.011
        atom.charge = charge
        ids.append(atom.atom_id)
    for left, right in zip(ids, ids[1:]):
        Attributes.Bond(left, right, funct=1, c0=0.15, c1=1000.0)

    path = tmp_path / "charged.itp"
    write_combined_itp(world, filename=str(path), moleculetype_name="CHG")

    from hygel_martini.core.itp import read_itp_definitions

    definition = read_itp_definitions(str(path), require_mass=False)["CHG"]
    written = [bead["charge"] for bead in definition["beads"]]
    assert written == pytest.approx(charges, abs=1e-9)
    assert sum(written) == pytest.approx(sum(charges), abs=1e-9)


def test_impropers_complete_the_centre_the_new_bond_made(tmp_path) -> None:
    # A thiourethane carbonyl carbon is bonded to O and N in the strand
    # template and only becomes three-coordinate when the junction's sulfur
    # arrives, so its planarity improper exists in neither template. Model
    # that: chain A's last atom carries two extra neighbours, and the builder
    # bond to chain B completes it.
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    extra = [Attributes.Atom(source_template=world.Atoms[chain_a[0]][0].source_template)
             for _ in range(1)]
    oxygen = Attributes.Atom(source_template=None)
    oxygen.mass = 16.0
    Attributes.Bond(chain_a[2], oxygen.atom_id, funct=1, c0=0.12, c1=1000.0)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.17, c1=1000.0)

    added, pairs, impropers = generate_junction_bonded_terms(
        world, generate_impropers=True, improper_funct=4,
        improper_params=[180.0, 43.932, 2],
    )

    # chain_a[2] now has three neighbours (chain_a[1], oxygen, chain_b[0]);
    # chain_b[0] has two (chain_a[2], chain_b[1]) and gets none. Two of the
    # bonds touching the centre are builder bonds, and it still gets exactly
    # one improper -- two would assert the same planarity twice.
    assert impropers == 1
    written = [
        d for key in world.Dihedrals for d in world.Dihedrals[key]
        if int(d.dihedral_funct) == 4
    ]
    assert len(written) == 1
    improper = written[0]
    # Central atom second, the convention the OPLS templates themselves use.
    assert improper.dihedral_atom_2.atom_id == chain_a[2]
    assert improper.dihedral_params == [180.0, 43.932, 2.0]


def test_impropers_are_off_unless_requested() -> None:
    world = _fresh_world()
    chain_a = _chain(_Template(), 3)
    oxygen = Attributes.Atom(source_template=None)
    oxygen.mass = 16.0
    Attributes.Bond(chain_a[2], oxygen.atom_id, funct=1, c0=0.12, c1=1000.0)
    chain_b = _chain(_Template(), 3)
    Attributes.Bond(chain_a[2], chain_b[0], funct=1, c0=0.17, c1=1000.0)

    _, _, impropers = generate_junction_bonded_terms(world)

    assert impropers == 0
    assert not [
        d for key in world.Dihedrals for d in world.Dihedrals[key]
        if int(d.dihedral_funct) == 4
    ]
