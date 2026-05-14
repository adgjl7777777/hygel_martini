from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from .utils import parse_csv_list, parse_int_csv, parse_semicolon_list


REPO_ROOT = Path(__file__).resolve().parents[2]
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required to load .yaml/.yml config files. Install with: pip install pyyaml"
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _load_single_config(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _load_yaml(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return data
    raise ValueError(f"Unsupported config extension: {path}. Use .yaml/.yml or .json")


def _load_with_includes(path: Path, seen: List[Path] | None = None) -> Dict[str, Any]:
    if seen is None:
        seen = []

    rpath = path.resolve()
    if rpath in seen:
        chain = " -> ".join(str(p) for p in (seen + [rpath]))
        raise ValueError(f"Circular includes detected: {chain}")

    data = _load_single_config(rpath)
    includes = data.pop("includes", [])
    if includes is None:
        includes = []
    if not isinstance(includes, list):
        raise ValueError(f"'includes' must be a list: {rpath}")

    merged: Dict[str, Any] = {}
    for item in includes:
        if not isinstance(item, str):
            raise ValueError(f"Include path must be a string in {rpath}: {item}")
        inc_path = (rpath.parent / item).resolve()
        inc_cfg = _load_with_includes(inc_path, seen + [rpath])
        merged = deep_update(merged, inc_cfg)

    merged = deep_update(merged, data)
    return merged


def _resolve_path_value(value: str, config_dir: Path) -> str:
    resolved = os.path.expanduser(os.path.expandvars(value))
    resolved = resolved.replace("${CONFIG_DIR}", str(config_dir))
    resolved = resolved.replace("${REPO_ROOT}", str(REPO_ROOT))
    path_obj = Path(resolved)
    if not path_obj.is_absolute():
        path_obj = config_dir / path_obj
    return str(path_obj.resolve())


def _normalize_paths(cfg: Dict[str, Any], config_path: Path | None) -> Dict[str, Any]:
    if not config_path or not config_path.exists():
        return cfg

    result = copy.deepcopy(cfg)
    config_dir = config_path.resolve().parent
    path_section = result.get("paths", {})
    if not isinstance(path_section, dict):
        return result

    for key, value in list(path_section.items()):
        if isinstance(value, str) and key.endswith(("_dir", "_root", "_path", "_glob")):
            path_section[key] = _resolve_path_value(value, config_dir)
        elif isinstance(value, list) and key.endswith(("_dirs", "_roots", "_paths", "_globs")):
            path_section[key] = [
                _resolve_path_value(item, config_dir) if isinstance(item, str) else item
                for item in value
            ]
    return result


def _parse_override_value(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if _INT_RE.match(value):
        try:
            return int(value)
        except ValueError:
            pass
    if _FLOAT_RE.match(value):
        try:
            return float(value)
        except ValueError:
            pass

    if value[0] in "[{\"'":
        try:
            import yaml  # type: ignore

            return yaml.safe_load(value)
        except Exception:
            try:
                return json.loads(value)
            except Exception:
                return value

    return value


def _apply_set_override(cfg: Dict[str, Any], expr: str) -> None:
    if "=" not in expr:
        raise ValueError(f"Invalid --set override: {expr!r}. Expected key.path=value")

    path_text, raw_value = expr.split("=", 1)
    keys = [part.strip() for part in path_text.split(".") if part.strip()]
    if not keys:
        raise ValueError(f"Invalid --set override path: {expr!r}")

    value = _parse_override_value(raw_value)
    current: Dict[str, Any] = cfg
    for key in keys[:-1]:
        next_value = current.get(key)
        if next_value is None:
            current[key] = {}
            next_value = current[key]
        if not isinstance(next_value, dict):
            raise TypeError(
                f"Cannot apply --set {expr!r}: {'.'.join(keys[:-1])!r} is not a mapping"
            )
        current = next_value
    current[keys[-1]] = value


def load_config(config_path: Path | None, default_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(default_config or {})
    if config_path and config_path.exists():
        user_cfg = _load_with_includes(config_path)
        cfg = deep_update(cfg, user_cfg)
    return _normalize_paths(cfg, config_path)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    result = copy.deepcopy(cfg)

    if getattr(args, "symbols", None) is not None:
        result.setdefault("system", {})
        result["system"]["symbols"] = parse_csv_list(args.symbols)
    if getattr(args, "sequences", None) is not None:
        result.setdefault("system", {})
        result["system"]["sequences"] = parse_semicolon_list(args.sequences)
    if getattr(args, "lengths", None) is not None:
        result.setdefault("system", {})
        result["system"]["lengths"] = parse_int_csv(args.lengths)
    if getattr(args, "replicas", None) is not None:
        result.setdefault("system", {})
        result["system"]["replicas"] = args.replicas
    if getattr(args, "cutoff_nm", None) is not None:
        result.setdefault("system", {})
        result["system"]["cutoff_nm"] = args.cutoff_nm
    if getattr(args, "min_box_safety_nm", None) is not None:
        result.setdefault("system", {})
        result["system"]["min_box_safety_nm"] = args.min_box_safety_nm
    if getattr(args, "temp_c", None) is not None:
        result.setdefault("system", {})
        result["system"]["temperature_c"] = args.temp_c
    if getattr(args, "out", None) is not None:
        result.setdefault("paths", {})
        result["paths"]["out_root"] = args.out
    if getattr(args, "solvate_tool", None) is not None:
        result.setdefault("system", {})
        result["system"]["solvate_tool"] = args.solvate_tool
    if getattr(args, "n_torsion_mode", None) is not None:
        result.setdefault("system", {})
        result["system"]["n_torsion_mode"] = args.n_torsion_mode
    if getattr(args, "sample_nsteps", None) is not None:
        result.setdefault("sampling", {})
        result["sampling"]["sample_nsteps"] = args.sample_nsteps
    if getattr(args, "gmxrc_path", None) is not None:
        result.setdefault("paths", {})
        result["paths"]["gmxrc_path"] = args.gmxrc_path
    if getattr(args, "gromacs_water_model", None) is not None:
        result.setdefault("water", {})
        result["water"]["gromacs_water_model"] = args.gromacs_water_model
    if getattr(args, "cpu_omp_threads", None) is not None:
        result.setdefault("runtime", {})
        result["runtime"]["cpu_omp_threads"] = args.cpu_omp_threads
    if getattr(args, "gpu_omp_threads", None) is not None:
        result.setdefault("runtime", {})
        result["runtime"]["gpu_omp_threads"] = args.gpu_omp_threads
    if getattr(args, "default_run_mode", None) is not None:
        result.setdefault("runtime", {})
        result["runtime"]["default_run_mode"] = args.default_run_mode
    if getattr(args, "set_values", None):
        for expr in args.set_values:
            _apply_set_override(result, expr)

    return result


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="maker.yaml",
        help="Config path (.yaml/.yml/.json). Supports includes for YAML/JSON.",
    )
    parser.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Write workflow defaults to --config path and exit (JSON format)",
    )


def add_sequence_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", default=None, help="Override: comma-separated symbols")
    parser.add_argument(
        "--sequences",
        default=None,
        help="Override explicit sequences separated by ';' (each sequence: token1,token2 or token1 token2 or compact one-letter form)",
    )
    parser.add_argument("--lengths", default=None, help="Override: comma-separated lengths")
    parser.add_argument("--n-torsion-mode", choices=["repeat", "bonds"], default=None)


def add_opls_to_martini_cli_args(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    add_sequence_override_args(parser)

    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--cutoff-nm", type=float, default=None)
    parser.add_argument("--min-box-safety-nm", type=float, default=None)
    parser.add_argument("--temp-c", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--solvate-tool", choices=["gromacs", "packmol"], default=None)
    parser.add_argument("--sample-nsteps", type=int, default=None)
    parser.add_argument("--gmxrc-path", default=None)
    parser.add_argument("--gromacs-water-model", default=None)
    parser.add_argument("--cpu-omp-threads", type=int, default=None)
    parser.add_argument("--gpu-omp-threads", type=int, default=None)
    parser.add_argument("--default-run-mode", choices=["none", "cpu", "gpu"], default=None)


def add_qm_to_martini_cli_args(parser: argparse.ArgumentParser) -> None:
    add_config_args(parser)
    add_sequence_override_args(parser)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=None,
        help=(
            "Override a nested config value with key.path=value. "
            "Repeatable. Example: --set bartender_pipeline.md=off "
            "--set system.sequences='[S,D,D,S]'"
        ),
    )
