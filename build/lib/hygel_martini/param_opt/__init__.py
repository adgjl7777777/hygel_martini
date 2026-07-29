"""Explicit workflow packages for 01/02/03 polymer parameter generation."""


def main() -> None:
    from .cli import main as _main

    _main()


__all__ = ["main"]
