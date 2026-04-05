from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict


_PATH_KEYS = {
    "system_top",
    "bonded_itp",
    "start_gro",
    "mdp",
    "minim_mdp",
    "workdir",
    "log_dir",
}
_PATH_SUFFIXES = ("_path", "_file", "_dir", "_gro", "_itp", "_root", "_mdp")
_PATH_LIST_KEYS = {"grompp_extra", "mdrun_extra"}
_LITERAL_KEYS = {"gmx"}


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml_file(path: Path) -> Dict[str, Any]:
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
    config_dir = str(config_path.resolve().parent)
    repo_root = str(Path(__file__).resolve().parents[2])
    return {"CONFIG_DIR": config_dir, "REPO_ROOT": repo_root}


def _looks_like_path_key(key: str | None) -> bool:
    if not isinstance(key, str):
        return False
    if key in _PATH_KEYS:
        return True
    return key.endswith(_PATH_SUFFIXES)


def _resolve_path_value(value: str, context: Dict[str, str]) -> str:
    expanded = os.path.expanduser(os.path.expandvars(value))
    for token, replacement in context.items():
        expanded = expanded.replace(f"${{{token}}}", replacement)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(os.path.join(context["CONFIG_DIR"], expanded))
    return expanded


def _normalize_tree(node: Any, context: Dict[str, str], parent_key: str | None = None) -> Any:
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
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Relax config not found: {path}")
    context = _build_context(path)
    data = _load_with_includes(path)
    return _normalize_tree(data, context)
