"""GROMACS topology parsing for the builder.

The implementation lives in :mod:`hygel_martini.core.itp`, shared with the
analysis package.  The name here is historical: the parser was never
Martini-specific, and the analysis package had grown a second partial parser of
the same format.
"""

from hygel_martini.core.itp import read_atom_types, read_itp_definitions

__all__ = ["read_atom_types", "read_itp_definitions"]
