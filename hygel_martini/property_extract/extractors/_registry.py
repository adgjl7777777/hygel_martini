"""Plugin registry and base class for manifest-driven extractors.

Owns :data:`EXTRACTOR_REGISTRY` (extractor name -> class), the
:func:`register_extractor` decorator that populates it, and
:class:`BaseExtractor`, the adapter contract every ``extractors/*``
module implements.  The manifest-driven runner
(:mod:`..validation_manifest`) looks extractors up here by the
``extractor`` string declared in a validation manifest entry, then
calls ``can_compute``/``missing_inputs_list`` before ``compute``.

Registration happens as an import side effect: ``extractors/__init__``
imports every extractor module, and each ``@register_extractor(...)``
class definition inserts itself into the registry.  Extractors never
raise on missing inputs at the gate level — the runner uses the
``can_compute`` gate to refuse computation and report the missing
input keys instead.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..result import PropertyResult

# Global name -> extractor class mapping, filled by @register_extractor
# at import time.  Keys are dotted names like "swelling.volume_from_energy".
EXTRACTOR_REGISTRY: dict[str, type] = {}


def register_extractor(name: str):
    """Return a class decorator that registers an extractor under ``name``.

    Args:
        name: Registry key the validation manifest uses to select this
            extractor (conventionally ``"<domain>.<method>"``).

    Returns:
        A decorator that inserts the class into
        :data:`EXTRACTOR_REGISTRY` and returns the class unchanged.
    """
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator


class BaseExtractor:
    """Adapter contract between the manifest runner and one analyzer.

    Subclasses declare which named inputs they need
    (``required_inputs``) and implement :meth:`compute`, which wraps a
    concrete analyzer from the parent package and always returns a
    :class:`..result.PropertyResult` (never a bare value).  The default
    input gate treats an input as present only when the manifest
    supplied a non-empty path that exists on disk; subclasses with
    non-file inputs (e.g. glob patterns) override the gate methods.

    Attributes:
        extractor_name: Registry key, mirrored on the class for
            introspection (set by each subclass).
        required_inputs: Manifest input keys that must resolve to
            existing paths before :meth:`compute` may run.
    """
    extractor_name: str = ""
    required_inputs: list[str] = []

    def can_compute(self, inputs: dict[str, str | None]) -> bool:
        """Return True when every required input path exists on disk.

        Args:
            inputs: Manifest-supplied mapping of input key to path
                (values may be None or empty when not provided).
        """
        return all(
            inputs.get(k) and os.path.exists(str(inputs[k]))
            for k in self.required_inputs
        )

    def missing_inputs_list(self, inputs: dict[str, str | None]) -> list[str]:
        """Return the required input keys that are absent or nonexistent.

        The runner reports these keys verbatim when it refuses to run
        the extractor, so the list mirrors :meth:`can_compute` exactly.
        """
        return [
            k for k in self.required_inputs
            if not (inputs.get(k) and os.path.exists(str(inputs[k] or "")))
        ]

    def compute(self, inputs: dict[str, str | None], params: dict) -> "PropertyResult":
        """Run the analysis; subclasses must override.

        Args:
            inputs: Manifest input key -> path mapping (already gated
                by :meth:`can_compute`).
            params: Free-form ``parameters`` block from the manifest
                entry (units and defaults are extractor-specific).

        Returns:
            A PropertyResult carrying value, status, and metadata.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.compute() 미구현")
