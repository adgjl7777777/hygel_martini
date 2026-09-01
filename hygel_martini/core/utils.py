"""Small text-parsing and box-sizing helpers shared across hygel_martini.

Owns the comma/semicolon token parsers used by ``core.config`` when applying
CLI overrides, plus the minimum-box guard used by the opls_to_martini case
builder. Pure functions with no I/O; callers pass plain strings/sequences.
"""

from __future__ import annotations

from typing import List, Sequence


def parse_csv_list(text: str) -> List[str]:
    """Split comma-separated text into stripped, non-empty tokens."""
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_semicolon_list(text: str) -> List[str]:
    """Split semicolon-separated text into stripped, non-empty tokens."""
    return [token.strip() for token in text.split(";") if token.strip()]


def parse_int_csv(text: str) -> List[int]:
    """Parse a comma-separated list of integers.

    Args:
        text: Comma-separated integer tokens (whitespace tolerated).

    Returns:
        The parsed integers, in input order.

    Raises:
        ValueError: If no token remains after stripping (empty list), or if
            a token is not a valid integer.
    """
    values = []
    for token in parse_csv_list(text):
        values.append(int(token))
    if not values:
        raise ValueError("lengths is empty")
    return values


def sequence_name(symbol: str, n_repeat: int) -> str:
    """Return the homopolymer sequence label (symbol repeated n_repeat times)."""
    return symbol * n_repeat


def ensure_min_box_nm(box_nm: Sequence[float], cutoff_nm: float, safety_nm: float) -> List[float]:
    """Clamp each box edge to the minimum-image-safe minimum length.

    Args:
        box_nm: Proposed box edge lengths in nm.
        cutoff_nm: Nonbonded interaction cutoff in nm.
        safety_nm: Extra margin in nm added on top of twice the cutoff.

    Returns:
        Edge lengths in nm, each at least ``2 * cutoff_nm + safety_nm``.
    """
    min_len = (2.0 * cutoff_nm) + safety_nm
    return [max(value, min_len) for value in box_nm]
