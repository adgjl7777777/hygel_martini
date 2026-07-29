"""Thin workflow entry helper for post-build hydrogel relaxation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .config import load_relax_config
from .soft_em import run_soft_em
from .soft_md import run_soft_md
from .hard_em_shrink import run_hard_em_shrink


def run_relax_workflow(config_path: str | Path) -> Dict[str, Any]:
    resolved_path = Path(config_path).expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Config not found: {resolved_path}")

    cfg = load_relax_config(resolved_path)
    workflow = cfg.get("workflow", {})
    mode = str(workflow.get("mode", "soft_em")).strip().lower()
    name = resolved_path.name
    print(f"\n--- hydrogel relaxation 실행 중 ({name}) ---")

    if mode == "soft_em":
        final_path = run_soft_em(cfg)
    elif mode == "soft_md":
        final_path = run_soft_md(cfg)
    elif mode == "hard_em_shrink":
        final_path = run_hard_em_shrink(cfg)
    else:
        raise ValueError("workflow.mode must be one of: soft_em, soft_md, hard_em_shrink")

    result = {"config_path": str(resolved_path.resolve()), "mode": mode, "final_path": str(final_path)}
    print("\n--- hydrogel relaxation 완료 ---")
    return result
