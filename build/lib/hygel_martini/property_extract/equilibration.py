import numpy as np
from .result import PropertyResult


def check_stability(data, threshold=0.01, window=0.2) -> PropertyResult:
    """
    마지막 window 구간의 linear drift 로 안정성 판단.

    data      : 1D array (예: volume, energy)
    threshold : 허용 최대 상대 drift (기본 1%)
    window    : drift 계산에 사용할 끝 구간 비율 (기본 20%)
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
    raise NotImplementedError(
        "equilibration time 자동 탐색 미구현. rolling mean/std 기반 구현 필요."
    )
