"""Sealed, evidence-gated parameterization protocol.

This package is deliberately independent of a particular chemistry or fitting
program.  It records *why* a bonded term may advance from a generated
candidate to a tested-domain release, while preserving data roles, frozen
thresholds, checksums, and non-pass outcomes.
"""

from .engine import (
    ProtocolError,
    evaluate_evidence,
    initialize_project,
    new_iteration,
    project_status,
    seal_iteration,
    validate_project,
)

__all__ = [
    "ProtocolError",
    "evaluate_evidence",
    "initialize_project",
    "new_iteration",
    "project_status",
    "seal_iteration",
    "validate_project",
]

