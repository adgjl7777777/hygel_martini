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
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from hygel_martini.hydrogel_builder.main_components import Attributes

__all__ = ["generate_junction_bonded_terms"]


def _adjacency(world) -> Dict[int, Set[int]]:
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


def generate_junction_bonded_terms(
    world,
    dihedral_funct: int = 3,
    generate_dihedrals: bool = True,
    generate_pairs: bool = True,
) -> Tuple[int, List[Tuple[int, int]]]:
    """Enumerate junction-crossing dihedrals and 1-4 pairs on ``world``.

    Returns ``(dihedrals_added, pairs)`` where ``pairs`` is a sorted list of
    0-based atom-id tuples. Dihedrals are registered as parameterless
    :class:`Attributes.Dihedral` objects with the given ``funct`` (OPLS-AA
    convention is 3, Ryckaert-Bellemans); pairs are also stored on
    ``world.generated_pairs`` for the topology writer.
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

    sorted_pairs = sorted(pairs)
    if generate_pairs:
        # Merge with pairs already registered (template-internal 1-4s copied
        # by the populator); overwriting would discard them.
        existing = getattr(world, "generated_pairs", None) or []
        world.generated_pairs = sorted(set(map(tuple, existing)) | pairs)
    return dihedrals_added, sorted_pairs
