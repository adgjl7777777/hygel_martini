"""Relaxation workflows that run after hydrogel_builder system construction.

Owns the staged post-build relaxation of the coarse-grained hydrogel:
``soft_em`` (iterative energy minimization with a bonded-strength ramp
and pressure-driven box updates), ``soft_md`` (a single settling MD
run), and ``hard_em_shrink`` (guarded fixed-increment box compression
toward a target box).  One stage per invocation, selected by
``workflow.mode`` in the relax config; run via the ``hygel-relax`` CLI,
``python -m hygel_martini.hydrogel_builder.relax``, or
:func:`run_relax_workflow` (a lazy wrapper so importing the package does
not pull in the GROMACS-driving modules).
"""

from __future__ import annotations


def run_relax_workflow(*args, **kwargs):
    """Lazily import and invoke ``generator.run_relax_workflow``."""
    from .generator import run_relax_workflow as _run_relax_workflow

    return _run_relax_workflow(*args, **kwargs)


__all__ = ["run_relax_workflow"]
