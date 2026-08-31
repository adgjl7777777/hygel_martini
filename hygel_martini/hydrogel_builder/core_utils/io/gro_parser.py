"""GRO parsing for the builder.

The implementation lives in :mod:`hygel_martini.core.gro`, which is shared with
the analysis package.  This module previously carried its own parser that split
the tail of each record on whitespace; GRO is a fixed-column format, so that
failed on valid input whenever adjacent fields filled their columns exactly --
a coordinate of ``-100.000`` is eight characters and abuts its neighbour.
"""

from hygel_martini.core.gro import GroAtom, GroFrame, read_gro, read_gro_atoms

__all__ = ["GroAtom", "GroFrame", "read_gro", "read_gro_atoms"]
