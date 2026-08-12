"""Hash-chained, append-only verification ledger."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .io import canonical_json_bytes, sha256_bytes


GENESIS_HASH = "0" * 64


def _hash_event(event_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(event_without_hash))


def validate_records(records: List[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    previous_hash = GENESIS_HASH
    for index, record in enumerate(records, start=1):
        location = f"ledger row {index}"
        if record.get("sequence") != index:
            errors.append(f"{location}: sequence must equal {index}")
        if record.get("previous_hash") != previous_hash:
            errors.append(f"{location}: previous_hash does not match the prior row")
        supplied_hash = record.get("event_hash")
        content = dict(record)
        content.pop("event_hash", None)
        observed_hash = _hash_event(content)
        if supplied_hash != observed_hash:
            errors.append(f"{location}: event_hash mismatch")
        if isinstance(supplied_hash, str):
            previous_hash = supplied_hash
        else:
            previous_hash = observed_hash
    return errors


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"ledger line {line_number} is invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"ledger line {line_number} must contain a JSON object")
            records.append(record)
    return records


def validate_ledger(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        records = read_ledger(path)
    except ValueError as error:
        return [], [str(error)]
    return records, validate_records(records)


def append_event(
    path: Path,
    *,
    event_type: str,
    iteration_id: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Append one event under an advisory lock and fsync it before returning."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="\n") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - supported platforms are POSIX HPC hosts
            fcntl = None  # type: ignore[assignment]
        handle.seek(0)
        records: List[Dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"ledger line {line_number} is invalid JSON: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"ledger line {line_number} must be a JSON object")
            records.append(record)
        ledger_errors = validate_records(records)
        if ledger_errors:
            raise ValueError("refusing to append to invalid ledger: " + "; ".join(ledger_errors))
        previous_hash = records[-1]["event_hash"] if records else GENESIS_HASH
        event: Dict[str, Any] = {
            "sequence": len(records) + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_type": event_type,
            "iteration_id": iteration_id,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        event["event_hash"] = _hash_event(event)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return event


def events_for_iteration(
    records: List[Mapping[str, Any]], iteration_id: str
) -> List[Mapping[str, Any]]:
    return [record for record in records if record.get("iteration_id") == iteration_id]


def latest_decision(
    records: List[Mapping[str, Any]], iteration_id: str
) -> Optional[Mapping[str, Any]]:
    decisions = [
        record
        for record in records
        if record.get("iteration_id") == iteration_id
        and record.get("event_type") == "DECISION"
    ]
    return decisions[-1] if decisions else None
