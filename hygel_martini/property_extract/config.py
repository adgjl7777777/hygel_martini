"""
config.py — 분석 실행에 필요한 세 YAML을 한 번에 로드하는 편의 모듈.
각 로더의 실제 구현은 개별 모듈에 있다.
"""
from __future__ import annotations
import os

from .analysis_jobs import load_analysis_jobs, AnalysisJob
from .validation_manifest import load_manifest, ManifestTarget


def load_all(
    analysis_path: str,
    manifest_path: str | None = None,
    requirements_path: str | None = None,
) -> tuple[list[AnalysisJob], str, list[ManifestTarget] | None, str | None]:
    """
    analysis_jobs.yaml, validation_manifest.yaml, md_requirements.yaml을 한 번에 로드.

    반환값:
        (jobs, base_dir, manifest_targets_or_None, requirements_path_or_None)
    """
    jobs, base_dir = load_analysis_jobs(analysis_path)

    if manifest_path is None:
        candidate = os.path.join(base_dir, "validation_manifest.yaml")
        if os.path.exists(candidate):
            manifest_path = candidate

    if requirements_path is None:
        candidate = os.path.join(base_dir, "md_requirements.yaml")
        if os.path.exists(candidate):
            requirements_path = candidate

    manifest_targets = load_manifest(manifest_path) if manifest_path else None

    return jobs, base_dir, manifest_targets, requirements_path
