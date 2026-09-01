"""Stage 02 workflow package: OPLS/GROMACS data to Martini/Bartender fitting.

Owns the second parameterization stage of ``param_opt``.  Two workflow
modes are supported (selected by ``workflow.mode`` in the config):

- ``constructor``: generate legacy OPLS/GROMACS setup cases (polymer
  build, solvation, EM/NVT/NPT/MD pipeline scripts) via ``builder``.
- ``existing_data_fit``: reuse existing OPLS/GROMACS trajectories and
  prepare trim + Bartender refit jobs via ``fitting``.

Invoked as ``python -m param_opt.opls_to_martini`` (``__main__``/``cli``)
or programmatically through :func:`run_opls_to_martini`.
"""

from .builder import build_cases
from .cli import main
from .generator import run_opls_to_martini

__all__ = ["build_cases", "main", "run_opls_to_martini"]
