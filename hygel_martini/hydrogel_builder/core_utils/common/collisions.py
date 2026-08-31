"""Detect declarations that silently overwrite one another.

The builder assembles many lookup tables keyed by identifiers the user
supplies: component ids, residue names, molecule names in ITP files, atom
types, bond rules.  Every one of them was built with a plain dictionary
assignment, so declaring the same key twice discarded one declaration without
a word.  The build then succeeded, and the loss showed up -- if at all -- as a
wrong mass, a wrong bond parameter, or a component that never appeared.

Two policies cover the cases that arise:

``require_unique``
    A key may be declared once.  A second declaration is an error, because
    there is no defensible way to choose between two components that claim the
    same identity.

``require_consistent``
    A key may be declared repeatedly as long as the declarations agree.  This
    fits records that legitimately appear more than once -- the same molecule
    named in two topology files, the same atom type listed twice -- where
    agreement means nothing was lost and disagreement means something was.

Both name the offending key and both values, so the message says what to fix
rather than that something went wrong.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple

__all__ = [
    "DuplicateDeclaration",
    "require_unique",
    "require_consistent",
    "find_duplicates",
]


class DuplicateDeclaration(ValueError):
    """Two declarations claimed the same key."""


def _format(value: Any, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def find_duplicates(keys: Iterable[Hashable]) -> Dict[Hashable, int]:
    """Keys appearing more than once, with their counts."""
    counts: Dict[Hashable, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return {key: count for key, count in counts.items() if count > 1}


def require_unique(
    items: Iterable[Tuple[Hashable, Any]],
    what: str,
    key_name: str = "identifier",
    source: str | None = None,
) -> Dict[Hashable, Any]:
    """Build a lookup, refusing any key declared twice.

    ``what`` names the kind of thing being indexed, for the message: passing
    ``"linker"`` produces ``Duplicate linker identifier 'LNK1'``.
    """
    lookup: Dict[Hashable, Any] = {}
    duplicates: List[Tuple[Hashable, Any, Any]] = []
    for key, value in items:
        if key in lookup:
            duplicates.append((key, lookup[key], value))
            continue
        lookup[key] = value

    if duplicates:
        where = f" in {source}" if source else ""
        detail = "; ".join(
            f"{key!r} declared as {_format(first)} and again as {_format(second)}"
            for key, first, second in duplicates[:3]
        )
        more = "" if len(duplicates) <= 3 else f" (and {len(duplicates) - 3} more)"
        raise DuplicateDeclaration(
            f"Duplicate {what} {key_name}{where}: {detail}{more}. "
            f"Each {what} needs its own {key_name}; one of these declarations "
            "would otherwise be discarded silently."
        )
    return lookup


def require_consistent(
    items: Iterable[Tuple[Hashable, Any]],
    what: str,
    key_name: str = "identifier",
    source: str | None = None,
    equal: Callable[[Any, Any], bool] | None = None,
) -> Dict[Hashable, Any]:
    """Build a lookup, allowing repeats only when the values agree.

    Use where a record may legitimately be declared more than once, so that a
    repeat is harmless but a *conflicting* repeat means one declaration is
    being thrown away.
    """
    same = equal or (lambda first, second: first == second)
    lookup: Dict[Hashable, Any] = {}
    conflicts: List[Tuple[Hashable, Any, Any]] = []
    for key, value in items:
        if key in lookup:
            if not same(lookup[key], value):
                conflicts.append((key, lookup[key], value))
            continue
        lookup[key] = value

    if conflicts:
        where = f" in {source}" if source else ""
        detail = "; ".join(
            f"{key!r} is {_format(first)} but also {_format(second)}"
            for key, first, second in conflicts[:3]
        )
        more = "" if len(conflicts) <= 3 else f" (and {len(conflicts) - 3} more)"
        raise DuplicateDeclaration(
            f"Conflicting {what} {key_name}{where}: {detail}{more}. "
            "Repeated declarations are allowed only when they agree; here the "
            "first would silently win."
        )
    return lookup
