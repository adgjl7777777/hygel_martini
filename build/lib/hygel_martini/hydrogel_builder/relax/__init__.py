"""Relaxation workflows that run after hydrogel_builder system construction."""

from __future__ import annotations


def run_relax_workflow(*args, **kwargs):
    from .generator import run_relax_workflow as _run_relax_workflow

    return _run_relax_workflow(*args, **kwargs)


__all__ = ["run_relax_workflow"]
