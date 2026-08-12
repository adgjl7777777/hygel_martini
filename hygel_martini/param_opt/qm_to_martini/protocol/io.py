"""Small, deterministic I/O helpers used by the protocol engine."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a JSON-compatible object for stable hashing."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"YAML document is empty: {path}")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace one file without exposing a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    atomic_write_text(path, text)


def safe_project_path(root: Path, relative: str) -> Path:
    """Resolve a project-relative path and reject traversal outside the root."""

    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("artifact path must be a non-empty string")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path escapes project root: {relative!r}") from error
    return candidate

