from __future__ import annotations
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..result import PropertyResult

EXTRACTOR_REGISTRY: dict[str, type] = {}


def register_extractor(name: str):
    def decorator(cls):
        EXTRACTOR_REGISTRY[name] = cls
        return cls
    return decorator


class BaseExtractor:
    extractor_name: str = ""
    required_inputs: list[str] = []

    def can_compute(self, inputs: dict[str, str | None]) -> bool:
        return all(
            inputs.get(k) and os.path.exists(str(inputs[k]))
            for k in self.required_inputs
        )

    def missing_inputs_list(self, inputs: dict[str, str | None]) -> list[str]:
        return [
            k for k in self.required_inputs
            if not (inputs.get(k) and os.path.exists(str(inputs[k] or "")))
        ]

    def compute(self, inputs: dict[str, str | None], params: dict) -> "PropertyResult":
        raise NotImplementedError(f"{self.__class__.__name__}.compute() 미구현")
