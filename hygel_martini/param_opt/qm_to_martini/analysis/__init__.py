"""Analysis helpers for qm_to_martini postprocessing."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "summarize": "summarizer",
    "plot": "plotter",
    "organize_logs": "log_manager",
    "trim_summary": "trim_summary",
    "compare_sweeps": "compare_sweeps",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    return module.main
