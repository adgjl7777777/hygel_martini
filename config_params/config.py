import json

class Config:
    _data = None
    _file_path = None

    @classmethod
    def load_config(cls, file_path):
        if cls._data is None or cls._file_path != file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cls._data = json.load(f)
                    cls._file_path = file_path
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found at {file_path}")
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON from {file_path}")
        return cls._data

    @classmethod
    def get_param(cls, *keys, file_path=None):
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
        if cls._data is None:
            raise ValueError("Configuration not loaded. Call load_config(file_path) first.")
        current_level = cls._data
        for i, key in enumerate(keys[:-1]):
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value
