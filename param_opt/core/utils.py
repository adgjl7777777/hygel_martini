from __future__ import annotations

from typing import List, Sequence


def parse_csv_list(text: str) -> List[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_semicolon_list(text: str) -> List[str]:
    return [token.strip() for token in text.split(";") if token.strip()]


def parse_int_csv(text: str) -> List[int]:
    values = []
    for token in parse_csv_list(text):
        values.append(int(token))
    if not values:
        raise ValueError("lengths is empty")
    return values


def sequence_name(symbol: str, n_repeat: int) -> str:
    return symbol * n_repeat


def ensure_min_box_nm(box_nm: Sequence[float], cutoff_nm: float, safety_nm: float) -> List[float]:
    min_len = (2.0 * cutoff_nm) + safety_nm
    return [max(value, min_len) for value in box_nm]
