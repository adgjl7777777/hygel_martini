import numpy as np
from .gmx_utils import parse_xvg


def calculate_viscosity_green_kubo(energy_xvg, temperature, volume_nm3, dt_ps):
    """
    Green-Kubo 관계식을 이용한 전단 점도 계산.
    eta = (V / kT) * integral <Pxy(0)Pxy(t)> dt

    미구현: GROMACS 'gmx energy -vis' 또는 별도 ACF 적분 구현 필요.
    """
    raise NotImplementedError(
        "Green-Kubo 점도 계산 미구현. "
        "GROMACS 'gmx energy -vis' 또는 별도 ACF 적분 구현 필요."
    )


def analyze_shear_rate_viscosity(energy_xvgs, shear_rates_ps_inv):
    """
    NEMD 전단 점도 분석.

    energy_xvgs     : xvg 파일 경로 목록
    shear_rates_ps_inv : 전단 속도 목록 [ps^-1]
    반환값          : 점도 배열 [Pa·s]

    단위 변환:
        Pxy [bar] → Pa (*1e5)
        shear rate [ps^-1] → s^-1 (*1e12)
    """
    viscosities = []
    for xvg, sr in zip(energy_xvgs, shear_rates_ps_inv):
        data = parse_xvg(xvg)
        pxy_val = data.get('Pres-XY', data.get('Pres-YX'))
        if pxy_val is None:
            raise ValueError(
                f"stress component (Pres-XY / Pres-YX) not found in {xvg}"
            )
        pxy_pa = np.abs(np.mean(pxy_val)) * 1e5   # bar → Pa
        sr_s1 = sr * 1e12                          # ps^-1 → s^-1

        if sr_s1 == 0:
            raise ValueError(f"shear rate가 0입니다: {sr} ps^-1")

        viscosities.append(pxy_pa / sr_s1)
    return np.array(viscosities)
