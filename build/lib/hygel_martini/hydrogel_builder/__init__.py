"""Hydrogel construction package namespace."""

from __future__ import annotations


def run_hydrogel_builder(*args, **kwargs):
    from .generator import run_hydrogel_builder as _run_hydrogel_builder

    return _run_hydrogel_builder(*args, **kwargs)


__all__ = ["run_hydrogel_builder"]
