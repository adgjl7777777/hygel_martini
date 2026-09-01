"""Stage 01 workflow package: QM-side inputs to OPLS preparation.

Owns the first parameterization stage of ``param_opt``: from a monomer
library and DFT settings it generates ORCA optimization inputs (and the
LigParGen-based OPLS preparation helpers).  Invoked either as
``python -m param_opt.qm_to_opls`` (via ``__main__``/``cli``) or
programmatically through :func:`run_qm_to_opls`.

Re-exports the two public entry points so callers can import them
directly from the package root.
"""

from .generator import run_qm_to_opls
from .cli import main

__all__ = ["main", "run_qm_to_opls"]
