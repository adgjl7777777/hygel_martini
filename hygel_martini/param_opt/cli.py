from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Use an explicit workflow module instead of `python -m param_opt`:\n"
        "  python -m param_opt.qm_to_opls --config ...\n"
        "  python -m param_opt.opls_to_martini --config ...\n"
        "  python -m param_opt.qm_to_martini --config ...\n"
        "  python -m param_opt.bead_generator\n"
    )
