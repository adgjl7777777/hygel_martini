"""Periodic net definitions used as construction seeds.

A *net* here is a topologically regular periodic graph in the sense of
reticular chemistry: junctions on lattice sites, strands on the edges between
them.  Nets are named by their RCSR three-letter symbols (O'Keeffe et al.,
Acc. Chem. Res. 2008, 41, 1782), which fixes what ``dia`` and ``pcu`` mean
without further argument.

Two nets are defined:

``dia``
    The diamond net.  Four-connected, fundamental cycle size 6.  This is the
    seed the tetrafunctional builder has always used.

``pcu``
    The primitive cubic net.  Six-connected, fundamental cycle size 4.  This
    is the seed for hexafunctional crosslinkers such as a six-arm thiol.

Each definition carries the published coordination number, fundamental cycle
size, and bipartiteness alongside the geometry, and
:func:`build_periodic_net` returns a graph that the topology audit can check
against them.  A seed that does not reproduce its own net's published
invariants is a bug in this module, not a discovery.

Both nets are bipartite, so a construction that leaves strands on net edges
has strictly even loop orders and never the odd-order cycles real networks
contain.  That is a property of the seed, not of the chemistry, and it is why
a rewiring step exists.

How a periodic supercell interacts with that bipartiteness differs per net and
is recorded per definition rather than assumed.  ``dia`` is two-coloured by its
A/B sublattice and every bond joins the two, so no repeat count can break it.
``pcu`` is two-coloured by the parity of the site index, which an odd repeat
count destroys through the boundary.  Independently, a supercell can be small
enough that a walk wraps the box in fewer bonds than the net's own fundamental
cycle, at which point the measured girth is an artifact; one step along a
``pcu`` axis costs one bond while ``dia`` needs two, so the threshold is also
per net.  :func:`validate_repeats` applies both checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np

from hygel_martini.hydrogel_builder.core_utils.layout.local_matching import LocalVertex

__all__ = [
    "NetDefinition",
    "DIA",
    "PCU",
    "NETS",
    "get_net",
    "validate_repeats",
    "build_periodic_net",
]

CellOffset = Tuple[int, int, int]
ArmLabel = Tuple[int, int]  # (bond index within the definition, +1 out / -1 in)


@dataclass(frozen=True)
class NetDefinition:
    """A periodic net: lattice, basis, edges, and its published invariants."""

    name: str
    description: str
    #: Rows are the lattice vectors, in units of the cell parameter.
    lattice_vectors: Tuple[Tuple[float, float, float], ...]
    #: Cartesian basis-site positions, in the same units.
    basis: Tuple[Tuple[float, float, float], ...]
    #: ``(site_i, site_j, offset)`` with the offset in lattice-vector units.
    bonds: Tuple[Tuple[int, int, CellOffset], ...]
    #: Published RCSR values, asserted by the test suite rather than assumed.
    coordination: int
    fundamental_cycle_size: int
    bipartite: bool
    #: Bonds in the shortest walk that translates by one lattice vector.  This
    #: sets how long a wrap-around cycle is, which differs per net: one step
    #: along a pcu axis is a single bond, while dia needs two (A to B to A).
    cell_traversal_bonds: int
    #: True when the two-colouring survives any periodic identification.  A net
    #: whose bipartition is the basis-site index (dia) keeps it for any repeat
    #: count; one whose bipartition is a coordinate parity (pcu) loses it on an
    #: odd supercell.
    bipartition_survives_odd_repeats: bool

    def arms_of_site(self, site: int) -> Tuple[ArmLabel, ...]:
        """Stable arm labels for one basis site, in definition order."""
        outgoing = [(index, 1) for index, (i, _, _) in enumerate(self.bonds) if i == site]
        incoming = [(index, -1) for index, (_, j, _) in enumerate(self.bonds) if j == site]
        return tuple(outgoing + incoming)

    def validate(self) -> None:
        for site in range(len(self.basis)):
            arms = self.arms_of_site(site)
            if len(arms) != self.coordination:
                raise ValueError(
                    f"Net {self.name!r}: basis site {site} has {len(arms)} arms "
                    f"but the net is {self.coordination}-connected"
                )
        if self.coordination % 2:
            raise ValueError(
                f"Net {self.name!r} is {self.coordination}-connected; the "
                "transition-system planner needs even coordination"
            )

    def bond_vectors(self) -> List[np.ndarray]:
        """Cartesian vector of each bond, for checking the embedding."""
        cell = np.asarray(self.lattice_vectors, dtype=float)
        basis = np.asarray(self.basis, dtype=float)
        return [
            basis[j] + np.asarray(offset, dtype=float) @ cell - basis[i]
            for i, j, offset in self.bonds
        ]


DIA = NetDefinition(
    name="dia",
    description="diamond net; four-connected, fundamental cycle size 6",
    # FCC primitive vectors: with the basis offset below, all four bonds come
    # out the same length, i.e. this is diamond geometrically and not merely a
    # graph that happens to be four-connected.
    lattice_vectors=(
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ),
    basis=(
        (0.0, 0.0, 0.0),
        (-0.25, -0.25, -0.25),
    ),
    bonds=(
        (0, 1, (0, 0, 0)),
        (0, 1, (1, 0, 0)),
        (0, 1, (0, 1, 0)),
        (0, 1, (0, 0, 1)),
    ),
    coordination=4,
    fundamental_cycle_size=6,
    bipartite=True,
    cell_traversal_bonds=2,
    # dia's two-colouring is the A/B sublattice and every bond joins the two
    # sublattices, so no periodic identification can break it.
    bipartition_survives_odd_repeats=True,
)

PCU = NetDefinition(
    name="pcu",
    description="primitive cubic net; six-connected, fundamental cycle size 4",
    lattice_vectors=(
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    basis=((0.0, 0.0, 0.0),),
    bonds=(
        (0, 0, (1, 0, 0)),
        (0, 0, (0, 1, 0)),
        (0, 0, (0, 0, 1)),
    ),
    coordination=6,
    fundamental_cycle_size=4,
    bipartite=True,
    cell_traversal_bonds=1,
    # pcu's two-colouring is the parity of (ix + iy + iz), which an odd repeat
    # count destroys through the boundary.
    bipartition_survives_odd_repeats=False,
)

NETS: Mapping[str, NetDefinition] = {net.name: net for net in (DIA, PCU)}


def get_net(name: str) -> NetDefinition:
    """Look up a net by RCSR symbol."""
    key = str(name).strip().lower()
    if key not in NETS:
        raise ValueError(
            f"Unknown net {name!r}; available: {', '.join(sorted(NETS))}"
        )
    net = NETS[key]
    net.validate()
    return net


def validate_repeats(net: NetDefinition, repeats: Sequence[int]) -> None:
    """Reject supercells whose shortest cycles come from the box, not the net.

    Two failure modes, both of which produce a network whose loop-order
    spectrum is an artifact:

    * an odd repeat count closes odd walks through the periodic boundary,
      destroying the bipartiteness the net is supposed to guarantee;
    * a repeat count so small that a walk wraps the box in fewer steps than
      the net's own fundamental cycle size.
    """
    counts = tuple(int(value) for value in repeats)
    if len(counts) != 3 or any(value < 1 for value in counts):
        raise ValueError(f"repeats must be three positive integers, got {repeats!r}")

    if net.bipartite and not net.bipartition_survives_odd_repeats:
        odd = [axis for axis, value in zip("xyz", counts) if value % 2]
        if odd:
            raise ValueError(
                f"Net {net.name!r} is bipartite, but repeats {counts} are odd "
                f"along {', '.join(odd)}. This net's two-colouring is a "
                "coordinate parity, which an odd supercell destroys through "
                "the periodic boundary, manufacturing odd cycles that are box "
                "artifacts rather than chemistry. Use even repeats."
            )

    # One step along a lattice vector costs ``cell_traversal_bonds`` bonds, so
    # a wrap costs that many times the repeat count.  The box must not undercut
    # the net's own shortest cycle or the measured girth is an artifact.
    shortest_wrap = net.cell_traversal_bonds * min(counts)
    if shortest_wrap < net.fundamental_cycle_size:
        minimum = -(-net.fundamental_cycle_size // net.cell_traversal_bonds)
        if net.bipartite and not net.bipartition_survives_odd_repeats and minimum % 2:
            minimum += 1
        raise ValueError(
            f"Repeats {counts} give a wrap-around cycle of length "
            f"{shortest_wrap}, shorter than the fundamental cycle size "
            f"{net.fundamental_cycle_size} of net {net.name!r}; the measured "
            "girth would be set by the box rather than the net. Increase the "
            f"smallest repeat to at least {minimum}."
        )


def build_periodic_net(
    net: NetDefinition | str,
    repeats: Sequence[int] | int,
    cell_parameter: float = 1.0,
    check_repeats: bool = True,
) -> Tuple[List[LocalVertex], List[Tuple[Hashable, Hashable]], Dict[Hashable, np.ndarray]]:
    """Materialize a periodic net as planner input.

    Returns ``(vertices, strands, positions)``:

    * ``vertices`` are :class:`LocalVertex` objects, one per junction, keyed by
      ``(ix, iy, iz, site)`` and carrying one endpoint per arm;
    * ``strands`` are endpoint pairs, one per net edge;
    * ``positions`` maps each junction to its Cartesian coordinate.

    Endpoint identifiers are ``(junction_key, arm_label)``, which is stable
    across runs so a plan can be compared or replayed.
    """
    definition = net if isinstance(net, NetDefinition) else get_net(net)
    definition.validate()
    if isinstance(repeats, int):
        counts = (repeats, repeats, repeats)
    else:
        counts = tuple(int(value) for value in repeats)
    if check_repeats:
        validate_repeats(definition, counts)

    cell = np.asarray(definition.lattice_vectors, dtype=float) * float(cell_parameter)
    basis = np.asarray(definition.basis, dtype=float) * float(cell_parameter)
    n_sites = len(definition.basis)

    cells = [
        (ix, iy, iz)
        for ix in range(counts[0])
        for iy in range(counts[1])
        for iz in range(counts[2])
    ]

    vertices: List[LocalVertex] = []
    positions: Dict[Hashable, np.ndarray] = {}
    for cell_index in cells:
        origin = np.asarray(cell_index, dtype=float) @ cell
        for site in range(n_sites):
            key = cell_index + (site,)
            positions[key] = origin + basis[site]
            vertices.append(
                LocalVertex(
                    key,
                    {arm: (key, arm) for arm in definition.arms_of_site(site)},
                )
            )

    strands: List[Tuple[Hashable, Hashable]] = []
    for bond_index, (site_i, site_j, offset) in enumerate(definition.bonds):
        for cell_index in cells:
            target_cell = tuple(
                (cell_index[axis] + offset[axis]) % counts[axis] for axis in range(3)
            )
            head = (cell_index + (site_i,), (bond_index, 1))
            tail = (target_cell + (site_j,), (bond_index, -1))
            strands.append((head, tail))

    for vertex in vertices:
        vertex.validate()
    return vertices, strands, positions
