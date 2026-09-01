"""One-shot loader for the three YAML files that drive an analysis run.

Convenience module used by the CLI (``__main__``) and scripts: it wraps
:func:`analysis_jobs.load_analysis_jobs` and
:func:`validation_manifest.load_manifest`, and auto-discovers the two
companion YAMLs (``validation_manifest.yaml`` and
``md_requirements.yaml``) next to the analysis-jobs file when their
paths are not given explicitly.  The actual parsing/validation logic
lives in the individual modules; nothing here adds gates of its own.
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
    """Load analysis jobs plus optional manifest and requirements YAMLs.

    When ``manifest_path`` / ``requirements_path`` are omitted, the
    default filenames ``validation_manifest.yaml`` and
    ``md_requirements.yaml`` are looked up in the directory containing
    ``analysis_path`` and used only if they exist.

    Args:
        analysis_path: Path to the analysis_jobs YAML file.
        manifest_path: Optional explicit path to the validation
            manifest YAML.
        requirements_path: Optional explicit path to the MD
            requirements YAML.  Returned as-is; not parsed here.

    Returns:
        Tuple ``(jobs, base_dir, manifest_targets_or_None,
        requirements_path_or_None)`` where ``base_dir`` is the directory
        of ``analysis_path`` (used to resolve relative input paths).
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
