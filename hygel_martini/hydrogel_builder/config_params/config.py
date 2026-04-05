"""Configuration loader with JSON/YAML include support.

The project still uses a global configuration singleton because many legacy
modules expect process-wide mutable state. This wrapper centralizes file
loading, deep-merging of YAML includes, runtime scratch values, and optional
debug logging.
"""

import copy
import json
import os

class Config:
    """Singleton-style access to configuration and runtime metadata."""
    _data = None
    _file_path = None
    _runtime_state = {}
    _debug_file = None
    _debug_enabled = False
    _PATH_LIST_KEYS = {"additional_itps", "additional_itp_files", "additional_tabulated_tables"}
    _LITERAL_COMMAND_KEYS = {"gromacs_executable_path", "packmol_path"}
    _NON_PATH_KEYS = {"output_dir_suffix"}
    _PATH_SUFFIXES = ("_path", "_file", "_dir", "_gro", "_itp", "_root")

    @classmethod
    def load_config(cls, file_path):
        """Load a JSON or YAML maker file into the global config cache."""
        if cls._data is None or cls._file_path != file_path:
            abs_file_path = os.path.abspath(file_path)
            path_context = cls._build_path_context(abs_file_path)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".yaml", ".yml"):
                cls._data = cls._load_yaml_with_includes(abs_file_path)
                cls._file_path = abs_file_path
            else:
                try:
                    with open(abs_file_path, 'r', encoding='utf-8') as f:
                        cls._data = json.load(f)
                        cls._file_path = abs_file_path
                except FileNotFoundError:
                    raise FileNotFoundError(f"Configuration file not found at {file_path}")
                except json.JSONDecodeError:
                    raise ValueError(f"Error decoding JSON from {file_path}")
            cls._data = cls._normalize_path_tree(cls._data, path_context)
            cls._runtime_state["config_dir"] = path_context["CONFIG_DIR"]
            cls._runtime_state["repo_root"] = path_context["REPO_ROOT"]
        return cls._data

    @classmethod
    def get_param(cls, *keys, file_path=None):
        """Read a nested value from the loaded configuration tree."""
        if file_path and (cls._file_path != file_path or cls._data is None):
            cls.load_config(file_path)
        elif cls._data is None:
            raise ValueError("Configuration not loaded. Call load_config(file_path) first.")
        
        current_level = cls._data
        for key in keys:
            try:
                if isinstance(key, int) and isinstance(current_level, list):
                    current_level = current_level[key]
                elif isinstance(current_level, dict) and key in current_level:
                    current_level = current_level[key]
                else:
                    str_keys = [str(k) for k in keys]
                    raise KeyError(f"Key '{key}' not found in configuration at path {'.'.join(str_keys)}")
            except (KeyError, IndexError):
                str_keys = [str(k) for k in keys]
                raise KeyError(f"Key '{key}' not found in configuration at path {'.'.join(str_keys)}")
        return current_level

    @classmethod
    def set_param(cls, value, *keys):
        """Write a nested value into the live configuration tree."""
        if cls._data is None:
            raise ValueError("Configuration not loaded. Call load_config(file_path) first.")
        current_level = cls._data
        for i, key in enumerate(keys[:-1]):
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value

    @classmethod
    def set_runtime(cls, key, value):
        """Store ephemeral runtime state that should not live in the config."""
        cls._runtime_state[key] = value

    @classmethod
    def get_runtime(cls, key, default=None):
        """Read ephemeral runtime state with an optional default."""
        return cls._runtime_state.get(key, default)

    @classmethod
    def enable_debug_logging(cls, file_path):
        """Enable debug logging to a file (overwrite on enable)."""
        cls._debug_enabled = True
        cls._debug_file = file_path
        try:
            with open(cls._debug_file, 'w', encoding='utf-8') as f:
                f.write("=== Debug Log Start ===\n")
        except Exception:
            cls._debug_enabled = False
            cls._debug_file = None

    @classmethod
    def disable_debug_logging(cls):
        """Disable file-backed debug logging."""
        cls._debug_enabled = False
        cls._debug_file = None

    @classmethod
    def debug_log(cls, message):
        """Append a debug message with basic timestamp if debug logging is enabled."""
        if not cls._debug_enabled or not cls._debug_file:
            return
        try:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(cls._debug_file, 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] {message}\n")
        except Exception:
            pass

    # ---------- internal helpers for YAML + include support ----------
    @classmethod
    def _deep_merge(cls, base, incoming):
        """Recursively merge dict incoming into base (mutates base)."""
        for k, v in incoming.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                cls._deep_merge(base[k], v)
            else:
                base[k] = copy.deepcopy(v)
        return base

    @classmethod
    def _load_yaml_file(cls, path):
        """Read a single YAML file without processing includes."""
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML이 필요합니다. `pip install pyyaml` 후 다시 시도하세요.") from e
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {path}")
        return data if isinstance(data, dict) else {}

    @classmethod
    def _load_yaml_with_includes(cls, path, seen=None):
        """Load a YAML file and recursively merge its ``includes`` chain."""
        if seen is None:
            seen = set()
        abspath = os.path.abspath(path)
        if abspath in seen:
            raise ValueError(f"Cyclic include detected for {path}")
        seen.add(abspath)

        data = cls._load_yaml_file(abspath)
        merged = {}
        includes = data.pop("includes", []) or []
        base_dir = os.path.dirname(abspath)
        for inc in includes:
            inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
            inc_data = cls._load_yaml_with_includes(inc_path, seen)
            cls._deep_merge(merged, inc_data)
        cls._deep_merge(merged, data)
        return merged

    @classmethod
    def _build_path_context(cls, file_path):
        config_dir = os.path.dirname(os.path.abspath(file_path))
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return {"CONFIG_DIR": config_dir, "REPO_ROOT": repo_root}

    @classmethod
    def _looks_like_path_key(cls, key):
        if not isinstance(key, str) or key in cls._NON_PATH_KEYS:
            return False
        if key in cls._PATH_LIST_KEYS:
            return True
        return key in {"gro", "itp", "molecule_gro", "molecule_itp"} or key.endswith(cls._PATH_SUFFIXES)

    @classmethod
    def _resolve_path_value(cls, value, path_context):
        expanded = os.path.expanduser(os.path.expandvars(value))
        for token, replacement in path_context.items():
            expanded = expanded.replace(f"${{{token}}}", replacement)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(os.path.join(path_context["CONFIG_DIR"], expanded))
        return expanded

    @classmethod
    def _should_resolve_scalar_path(cls, key, value):
        if key in cls._LITERAL_COMMAND_KEYS:
            return (
                value.startswith(".")
                or value.startswith("~")
                or "${" in value
                or "/" in value
                or "\\" in value
            )
        return True

    @classmethod
    def _normalize_path_tree(cls, node, path_context, parent_key=None):
        if isinstance(node, dict):
            return {
                key: cls._normalize_path_tree(value, path_context, key)
                for key, value in node.items()
            }
        if isinstance(node, list):
            if parent_key in cls._PATH_LIST_KEYS:
                return [
                    cls._resolve_path_value(item, path_context) if isinstance(item, str) else item
                    for item in node
                ]
            return [cls._normalize_path_tree(item, path_context) for item in node]
        if (
            isinstance(node, str)
            and cls._looks_like_path_key(parent_key)
            and cls._should_resolve_scalar_path(parent_key, node)
        ):
            return cls._resolve_path_value(node, path_context)
        return node
