"""Compatibility layer for hydrogel builder helpers.

The package is now organized into subpackages:

- ``core_utils.common``: math helpers and generic utilities
- ``core_utils.io``: GRO/ITP readers and writers
- ``core_utils.templates``: monomer/linker template loaders
- ``core_utils.layout``: proto planning and blueprint generation
- ``core_utils.runtime``: GROMACS, packmol, and topology orchestration
- ``core_utils.generators``: standalone structure generators

Top-level modules remain as thin wrappers so existing imports continue to work
while callers migrate gradually.
"""
