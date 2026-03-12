"""param_opt package for polymer constructor parameter optimization workflows."""


def main() -> None:
    from .cli import main as _main

    _main()


__all__ = ["main"]
