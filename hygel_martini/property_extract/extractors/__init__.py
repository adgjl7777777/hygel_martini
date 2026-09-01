"""Extractor plugin package: adapters from analyzers to the manifest runner.

Importing this package is what populates
:data:`._registry.EXTRACTOR_REGISTRY` — every extractor module below is
imported for its ``@register_extractor`` side effect, so the
manifest-driven runner (:mod:`..validation_manifest`) can resolve the
``extractor`` names declared in a validation manifest.  Each module
adapts exactly one analysis from the parent package into the
:class:`._registry.BaseExtractor` contract (gated inputs in, a
:class:`..result.PropertyResult` out).
"""
from ._registry import EXTRACTOR_REGISTRY, register_extractor, BaseExtractor

# Imported for registration side effects: each module's
# @register_extractor call adds its class to EXTRACTOR_REGISTRY.

from . import composition
from . import swelling
from . import pore_size
from . import rheology_nemd
from . import topology
from . import mechanics
from . import clearance

__all__ = [
    "BaseExtractor",
    "EXTRACTOR_REGISTRY",
    "clearance",
    "composition",
    "mechanics",
    "pore_size",
    "register_extractor",
    "rheology_nemd",
    "swelling",
    "topology",
]
