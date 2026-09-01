"""Time-series stability gate for equilibration judgment.

Owns :func:`check_stability`, which decides whether a scalar MD time
series (volume, energy, ...) has stopped drifting by fitting a line to
its final window, and the still-unimplemented
:func:`find_equilibration_time` placeholder.  Reports through
:class:`.result.PropertyResult` (property ``volume_stability``, no
experimental comparison): fewer than 10 points refuses with
``insufficient_data`` instead of guessing, and the verdict plus drift
diagnostics always land in metadata.
"""
import numpy as np
from .result import PropertyResult


def check_stability(data, threshold=0.01, window=0.2) -> PropertyResult:
    """Judge stability from the linear drift of the trailing window.

    Fits a straight line to the last ``window`` fraction of ``data``
    and compares the total drift over that segment (slope times
    segment length), relative to the segment mean, against
    ``threshold``.  A zero segment mean is treated as trivially stable
    (relative drift undefined but no signal to drift).

    Args:
        data: 1-D series (e.g. volume in nm^3, energy in kJ/mol);
            units cancel in the relative drift.
        threshold: Maximum allowed |relative drift| (default 1%).
        window: Trailing fraction of the series to fit (default 20%).

    Returns:
        PropertyResult ``volume_stability`` — computed with a boolean
        value and mean/std/drift metadata, or ``insufficient_data``
        when the series has fewer than 10 points.
    """
    if len(data) < 10:
        return PropertyResult.insufficient_data(
            "volume_stability",
            reason="stability 판단에는 최소 10개 이상의 데이터 포인트가 필요합니다.",
            metadata={
                "mean": float(np.mean(data)) if len(data) > 0 else float("nan"),
                "std": float(np.std(data)) if len(data) > 0 else float("nan"),
                "drift": float("nan"),
                "is_stable": False,
            },
        )

    n = len(data)
    start_idx = int(n * (1 - window))
    segment = data[start_idx:]

    mean = float(np.mean(segment))
    std = float(np.std(segment))

    if mean == 0:
        return PropertyResult(
            property="volume_stability",
            value=True,
            status="computed",
            direct_experiment_comparison_allowed=False,
            validation_role="",
            metadata={"mean": mean, "std": std, "drift": 0.0, "is_stable": True},
        )

    x = np.arange(len(segment))
    slope, _ = np.polyfit(x, segment, 1)
    total_drift = slope * len(segment)
    rel_drift = float(total_drift / mean)
    is_stable = abs(rel_drift) < threshold

    return PropertyResult(
        property="volume_stability",
        value=is_stable,
        status="computed",
        direct_experiment_comparison_allowed=False,
        validation_role="",
        metadata={
            "mean": mean,
            "std": std,
            "drift": rel_drift,
            "is_stable": is_stable,
            "threshold": threshold,
            "window_fraction": window,
        },
    )


def find_equilibration_time(times, data, threshold=0.01, window_size=100):
    """Placeholder: automatic equilibration-time detection.

    Raises:
        NotImplementedError: Always — a rolling mean/std implementation
            is still needed; nothing is estimated silently.
    """
    raise NotImplementedError(
        "equilibration time 자동 탐색 미구현. rolling mean/std 기반 구현 필요."
    )
