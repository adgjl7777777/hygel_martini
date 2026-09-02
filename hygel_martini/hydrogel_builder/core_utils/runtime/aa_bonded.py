"""Junction-crossing dihedrals and 1-4 pairs for all-atom force fields.

The builder already enumerates every angle in the final bond graph, crosslink
bonds included. What an all-atom topology additionally needs, and Martini
never did, are the terms one step further out: proper dihedrals along every
three-bond path through a newly formed bond, and the ``[ pairs ]`` entries
that reintroduce the scaled 1-4 interaction those paths' endpoints lose to
``nrexcl = 3``.

Two decisions define the scope:

**Junction-crossing only.** A path whose four atoms all belong to one template
*instance* (same template object, same chain) is that molecule's own business:
its ITP either declares the term or deliberately does not, and regenerating it
here would double-count it --- for pairs that means double-counted 1-4 energy,
which is silent and wrong. Only paths spanning instances (equivalently,
crossing a bond the builder itself created) are generated. The populator is
the counterpart: it copies template-internal pairs into the combined topology
whenever ``topology_nrexcl >= 2``, so between the two every 1-4 appears exactly
once.

**Parameterless.** Entries are written as ``i j k l funct`` with no inline
parameters, which is complete GROMACS: grompp resolves them from the force
field's ``[ dihedraltypes ]``/``[ pairtypes ]`` tables. This module therefore
needs to know nothing about any particular force field --- the same reason the
rest of the builder does not.

A 1-4 pair is generated only for endpoint pairs at bond distance exactly
three: a path of three bonds whose ends are also 1-2 or 1-3 neighbours (a
ring) is excluded, matching how force fields define the 1-4 set.

**Impropers are opt-in and explicit.** A builder-created bond can complete a
planar centre that neither template could declare on its own -- a thiourethane
carbonyl carbon is bonded to O and N in the strand template and only becomes
three-coordinate once the junction's sulfur arrives, so its planarity improper
exists in neither file. Unlike a proper dihedral, whose presence follows from
connectivity, an improper asserts a *geometry*, and asserting the wrong one is
silent. So this module generates impropers only when asked to, only on an
endpoint that ends up exactly three-coordinate, and with the parameters the
caller supplies (or parameterless, resolved from ``[ dihedraltypes ]``).

Atom order follows the convention the templates themselves use --- verified
across every improper in this project's OPLS templates: funct 4 with the
*central atom second*. For the planarity form (phase 180, multiplicity 2) the
energy is even in the dihedral angle, so the order of the three outer atoms
does not change it; that is stated rather than assumed because it stops being
true for other phases.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from hygel_martini.hydrogel_builder.main_components import Attributes

__all__ = ["generate_junction_bonded_terms"]


def _adjacency(world) -> Dict[int, Set[int]]:
    """Undirected bond adjacency over every atom currently in ``world``."""
    adjacency: Dict[int, Set[int]] = {atom_id: set() for atom_id in world.Atoms}
    for bond_entry in world.Bonds.values():
        bond = bond_entry[0]
        left = bond.bond_atom_1.atom_id
        right = bond.bond_atom_2.atom_id
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _same_template(world, atom_ids) -> bool:
    """Whether one template *instance* owns the whole path.

    Ownership is per molecule instance, not per template object: two
    molecules stamped from the same template that a builder bond later joins
    are different owners, and the path crossing that bond must be generated.
    Comparing template identity alone silently skipped exactly that case.
    An atom without a source template can never claim ownership, so its path
    is treated as crossing.
    """
    owners = set()
    for atom_id in atom_ids:
        atom = world.Atoms[atom_id][0]
        if atom.source_template is None:
            return False
        owners.add((
            id(atom.source_template),
            getattr(atom, "chain_type", None),
            getattr(atom, "chain_index", None),
        ))
    return len(owners) == 1


def _builder_bonds(world, adjacency) -> List[Tuple[int, int]]:
    """Bonds joining two different template instances.

    These are exactly the bonds the builder formed: within a template both
    endpoints share an owner, across one they do not.
    """
    crossing = []
    for bond_entry in world.Bonds.values():
        bond = bond_entry[0]
        left = bond.bond_atom_1.atom_id
        right = bond.bond_atom_2.atom_id
        if not _same_template(world, (left, right)):
            crossing.append((left, right))
    return crossing


def generate_junction_bonded_terms(
    world,
    dihedral_funct: int = 3,
    generate_dihedrals: bool = True,
    generate_pairs: bool = True,
    generate_impropers: bool = False,
    improper_funct: int = 4,
    improper_params: List[float] | None = None,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Enumerate junction-crossing dihedrals and 1-4 pairs on ``world``.

    Returns ``(dihedrals_added, pairs, impropers_added)``; ``pairs`` is a
    sorted list of 0-based atom-id tuples. Dihedrals are registered as
    parameterless :class:`Attributes.Dihedral` objects with the given
    ``funct`` (OPLS-AA convention is 3, Ryckaert-Bellemans); pairs are also
    stored on ``world.generated_pairs`` for the topology writer.

    Args:
        generate_impropers: Emit a planarity improper on each endpoint of a
            builder-created bond that ends up three-coordinate. Off by
            default: an improper asserts a geometry, so it is requested
            rather than inferred.
        improper_funct: GROMACS dihedral function type for those entries
            (4 = periodic improper, the OPLS convention these templates use).
        improper_params: Inline parameters, e.g. ``[180.0, 43.932, 2]``;
            omit for parameterless entries resolved from
            ``[ dihedraltypes ]``.
    """
    adjacency = _adjacency(world)

    existing: Set[Tuple[int, ...]] = set()
    for key in world.Dihedrals:
        forward = tuple(key)
        existing.add(min(forward, tuple(reversed(forward))))

    # 1-2 and 1-3 neighbour sets, to keep ring closures out of the 1-4 list.
    one_two: Set[frozenset] = {
        frozenset((left, right))
        for left, neighbours in adjacency.items()
        for right in neighbours
    }
    one_three: Set[frozenset] = set()
    for centre, neighbours in adjacency.items():
        ordered = sorted(neighbours)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                one_three.add(frozenset((first, second)))

    dihedrals_added = 0
    pairs: Set[Tuple[int, int]] = set()

    for j in sorted(adjacency):
        for k in sorted(adjacency[j]):
            if k <= j:
                continue  # each central bond once
            for i in sorted(adjacency[j] - {k}):
                for l in sorted(adjacency[k] - {j}):
                    if l == i:
                        continue  # a three-membered ring, not a torsion
                    path = (i, j, k, l)
                    if _same_template(world, path):
                        continue

                    if generate_dihedrals:
                        canonical = min(path, tuple(reversed(path)))
                        if canonical not in existing:
                            existing.add(canonical)
                            dihedral = Attributes.Dihedral(i, j, k, l, 0)
                            dihedral.dihedral_funct = int(dihedral_funct)
                            # Parameterless on purpose: grompp resolves the
                            # term from [ dihedraltypes ].
                            dihedral.dihedral_c0 = None
                            dihedral.dihedral_c1 = None
                            dihedral.dihedral_params = None
                            dihedrals_added += 1

                    if generate_pairs:
                        ends = frozenset((i, l))
                        if ends not in one_two and ends not in one_three:
                            pairs.add((min(i, l), max(i, l)))

    impropers_added = 0
    if generate_impropers:
        # One improper per endpoint of a builder-created bond that ends up
        # three-coordinate: that is the centre the new bond completed. A
        # two-coordinate endpoint (the junction sulfur here) has no improper
        # to complete, and a four-coordinate one is not planar.
        # Collected per CENTRE, not per (bond, endpoint): a centre touched by
        # two builder bonds would otherwise receive two impropers asserting
        # the same planarity, which double-counts its energy.
        centres = set()
        for left, right in _builder_bonds(world, adjacency):
            for centre in (left, right):
                if len(adjacency[centre]) == 3:
                    centres.add(centre)
        for centre in sorted(centres):
            first, second, third = sorted(adjacency[centre])
            # Central atom second, the convention this project's OPLS
            # templates use throughout; outer atoms sorted so the row is
            # deterministic. For the planarity form (phase 180,
            # multiplicity 2) the energy is even in the angle, so their
            # order does not change it.
            path = (first, centre, second, third)
            canonical = min(path, tuple(reversed(path)))
            if canonical in existing:
                continue
            existing.add(canonical)
            improper = Attributes.Dihedral(*path, 0)
            improper.dihedral_funct = int(improper_funct)
            if improper_params:
                improper.dihedral_params = [float(x) for x in improper_params]
            else:
                improper.dihedral_c0 = None
                improper.dihedral_c1 = None
                improper.dihedral_params = None
            impropers_added += 1

    sorted_pairs = sorted(pairs)
    if generate_pairs:
        # Merge with pairs already registered (template-internal 1-4s copied
        # by the populator); overwriting would discard them.
        existing = getattr(world, "generated_pairs", None) or []
        world.generated_pairs = sorted(set(map(tuple, existing)) | pairs)
    return dihedrals_added, sorted_pairs, impropers_added
