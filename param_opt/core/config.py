from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .defaults import DEFAULT_CONFIG
from .utils import parse_csv_list, parse_int_csv, parse_semicolon_list


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        if isinstance(value, str) and key.endswith(("_dir", "_root", "_path")):
            path_section[key] = _resolve_path_value(value, config_dir)
    return result


def load_config(config_path: Path | None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path and config_path.exists():
        user_cfg = _load_with_includes(config_path)
        cfg = deep_update(cfg, user_cfg)
    return _normalize_paths(cfg, config_path)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    result = copy.deepcopy(cfg)

    if args.symbols is not None:
        result["system"]["symbols"] = parse_csv_list(args.symbols)
    if args.sequences is not None:
        result["system"]["sequences"] = parse_semicolon_list(args.sequences)
    if args.lengths is not None:
        result["system"]["lengths"] = parse_int_csv(args.lengths)
    if args.replicas is not None:
        result["system"]["replicas"] = args.replicas
    if args.cutoff_nm is not None:
        result["system"]["cutoff_nm"] = args.cutoff_nm
    if args.min_box_safety_nm is not None:
        result["system"]["min_box_safety_nm"] = args.min_box_safety_nm
    if args.temp_c is not None:
        result["system"]["temperature_c"] = args.temp_c
    if args.out is not None:
        result["paths"]["out_root"] = args.out
    if args.solvate_tool is not None:
        result["system"]["solvate_tool"] = args.solvate_tool
    if args.n_torsion_mode is not None:
        result["system"]["n_torsion_mode"] = args.n_torsion_mode
    if args.sample_nsteps is not None:
        result["sampling"]["sample_nsteps"] = args.sample_nsteps
    if args.gmxrc_path is not None:
        result["paths"]["gmxrc_path"] = args.gmxrc_path
    if args.gromacs_water_model is not None:
        result["water"]["gromacs_water_model"] = args.gromacs_water_model
    if args.cpu_omp_threads is not None:
        result["runtime"]["cpu_omp_threads"] = args.cpu_omp_threads
    if args.gpu_omp_threads is not None:
        result["runtime"]["gpu_omp_threads"] = args.gpu_omp_threads
    if args.default_run_mode is not None:
        result["runtime"]["default_run_mode"] = args.default_run_mode

    return result


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="maker.yaml",
        help="Config path (.yaml/.yml/.json). Supports includes for YAML/JSON.",
    )

    parser.add_argument("--symbols", default=None, help="Override: comma-separated symbols")
    parser.add_argument(
        "--sequences",
        default=None,
        help="Override explicit sequences separated by ';' (each sequence: token1,token2 or token1 token2 or compact one-letter form)",
    )
    parser.add_argument("--lengths", default=None, help="Override: comma-separated lengths")
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--cutoff-nm", type=float, default=None)
    parser.add_argument("--min-box-safety-nm", type=float, default=None)
    parser.add_argument("--temp-c", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--solvate-tool", choices=["gromacs", "packmol"], default=None)
    parser.add_argument("--n-torsion-mode", choices=["repeat", "bonds"], default=None)
    parser.add_argument("--sample-nsteps", type=int, default=None)

    parser.add_argument("--gmxrc-path", default=None)
    parser.add_argument("--gromacs-water-model", default=None)
    parser.add_argument("--cpu-omp-threads", type=int, default=None)
    parser.add_argument("--gpu-omp-threads", type=int, default=None)
    parser.add_argument("--default-run-mode", choices=["none", "cpu", "gpu"], default=None)

    parser.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Write default config to --config path and exit (JSON format)",
    )
