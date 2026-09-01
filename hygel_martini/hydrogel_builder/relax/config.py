"""Config loading for the relax workflows (separate from core.config).

Owns the relax-specific config machinery used by ``generator``:

- YAML (with recursive ``includes:``, later/outer values winning) or
  JSON (includes rejected) loading with cyclic-include detection,
- normalization of path-like string values anywhere in the tree:
  ``${CONFIG_DIR}`` / ``${REPO_ROOT}`` substitution, env-var and ``~``
  expansion, and resolution of relative paths against the config
  file directory.

Which keys count as paths is decided by name (_PATH_KEYS plus the
_PATH_SUFFIXES endings), so stage runners can trust that those values
are absolute.  No schema validation happens here.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict


# Exact key names whose string values are treated as filesystem paths.
_PATH_KEYS = {
    "system_top",
    "bonded_itp",
    "start_gro",
    "mdp",
    "minim_mdp",
    "workdir",
    "log_dir",
}
# Key-name endings that also mark a value as a path.
_PATH_SUFFIXES = ("_path", "_file", "_dir", "_gro", "_itp", "_root", "_mdp")
# List-valued keys passed through verbatim as CLI argument strings.
_PATH_LIST_KEYS = {"grompp_extra", "mdrun_extra"}
# Keys never path-resolved even if the name looks path-like ("gmx" may
# be a bare command name that must stay resolvable via PATH).
_LITERAL_KEYS = {"gmx"}


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``incoming`` into ``base`` in place (nested dicts recursively).

    Non-dict values replace wholesale (deep-copied); returns ``base``.
    """
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load one YAML file whose root must be a mapping (empty -> {}).

    Raises:
        ImportError: If PyYAML is not installed.
        TypeError: If the document root is not a mapping.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required for hydrogel_builder.relax.") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Relax config root must be a mapping: {path}")
    return data


def _load_with_includes(path: Path, seen: set[Path] | None = None) -> Dict[str, Any]:
    """Load a config file, expanding YAML ``includes:`` recursively.

    Include entries resolve relative to the including file; later
    includes override earlier ones, and the including file overrides
    all of its includes.  JSON files are loaded directly and must not
    carry an ``includes`` key.

    Raises:
        ValueError: On cyclic includes or ``includes`` in JSON.
        TypeError: If any loaded root is not a mapping.
    """
    if seen is None:
        seen = set()
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic include detected in relax config: {resolved}")
    seen.add(resolved)

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = _load_yaml_file(resolved)
        merged: Dict[str, Any] = {}
        includes = data.pop("includes", []) or []
        for inc in includes:
            inc_path = Path(inc)
            if not inc_path.is_absolute():
                inc_path = resolved.parent / inc_path
            _deep_merge(merged, _load_with_includes(inc_path, seen))
        _deep_merge(merged, data)
        return merged

    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Relax config root must be a mapping: {resolved}")
    if "includes" in data:
        raise ValueError(
            f"'includes' is not supported in JSON config files: {resolved}\n"
            "Convert to .yaml/.yml to use includes."
        )
    return data


def _build_context(config_path: Path) -> Dict[str, str]:
    """Substitution context: CONFIG_DIR (config file dir) and REPO_ROOT.

    REPO_ROOT is the installable package root two levels above this
    module (the directory containing ``hygel_martini``).
    """
    config_dir = str(config_path.resolve().parent)
    repo_root = str(Path(__file__).resolve().parents[2])
    return {"CONFIG_DIR": config_dir, "REPO_ROOT": repo_root}


def _looks_like_path_key(key: str | None) -> bool:
    """Decide by key name whether a string value should be path-resolved."""
    if not isinstance(key, str):
        return False
    if key in _PATH_KEYS:
        return True
    return key.endswith(_PATH_SUFFIXES)


def _resolve_path_value(value: str, context: Dict[str, str]) -> str:
    """Expand env vars, ~, and ${TOKEN}s, then absolutize vs CONFIG_DIR."""
    expanded = os.path.expanduser(os.path.expandvars(value))
    for token, replacement in context.items():
        expanded = expanded.replace(f"${{{token}}}", replacement)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(os.path.join(context["CONFIG_DIR"], expanded))
    return expanded


def _normalize_tree(node: Any, context: Dict[str, str], parent_key: str | None = None) -> Any:
    """Recursively normalize a config tree (see module docstring).

    Path-like string values are resolved; lists under _PATH_LIST_KEYS
    are stringified verbatim; everything else passes through unchanged.
    """
    if isinstance(node, dict):
        return {
            key: _normalize_tree(value, context, key)
            for key, value in node.items()
        }
    if isinstance(node, list):
        if parent_key in _PATH_LIST_KEYS:
            return [str(item) for item in node]
        return [_normalize_tree(item, context) for item in node]
    if isinstance(node, str) and _looks_like_path_key(parent_key) and parent_key not in _LITERAL_KEYS:
        return _resolve_path_value(node, context)
    return node


def load_relax_config(config_path: str | Path) -> Dict[str, Any]:
    """Load, include-merge, and path-normalize a relax config file.

    Args:
        config_path: .yaml/.yml/.json file; ``~`` is expanded.

    Returns:
        The normalized config dict with absolute path values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Relax config not found: {path}")
    context = _build_context(path)
    data = _load_with_includes(path)
    return _normalize_tree(data, context)
