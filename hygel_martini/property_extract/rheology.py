"""Shear viscosity analysis from GROMACS energy XVGs.

Owns the implemented NEMD steady-shear estimator
(:func:`analyze_shear_rate_viscosity`, one viscosity per shear-rate
run from the mean off-diagonal pressure) and the unimplemented
Green-Kubo placeholder (:func:`calculate_viscosity_green_kubo`), which
refuses by raising instead of returning a guess.  Called by the NEMD
extractor adapter (:mod:`.extractors.rheology_nemd`), which supplies
the trend-only claim boundary; this module itself returns bare arrays
and raises ``ValueError`` on missing stress columns or zero rates.
"""
import numpy as np
from .gmx_utils import parse_xvg


def calculate_viscosity_green_kubo(energy_xvg, temperature, volume_nm3, dt_ps):
    """Placeholder: Green-Kubo shear viscosity.

    Intended relation: ``eta = (V / kT) * integral <Pxy(0) Pxy(t)> dt``.

    Raises:
        NotImplementedError: Always — needs GROMACS ``gmx energy -vis``
            or a dedicated stress-ACF integration; nothing is computed
            silently.
    """
    raise NotImplementedError(
        "Green-Kubo 점도 계산 미구현. "
        "GROMACS 'gmx energy -vis' 또는 별도 ACF 적분 구현 필요."
    )


def analyze_shear_rate_viscosity(energy_xvgs, shear_rates_ps_inv):
    """Compute NEMD steady-shear viscosities, one per shear-rate run.

    For each (xvg, rate) pair: ``eta = |<Pxy>| / shear_rate`` using the
    ``Pres-XY`` column (``Pres-YX`` as fallback).  Unit conversions:
    Pxy bar -> Pa (*1e5); shear rate ps^-1 -> s^-1 (*1e12).  Pairing is
    positional via ``zip``, so both lists must be ordered consistently.

    Args:
        energy_xvgs: Energy XVG paths, one per shear-rate run.
        shear_rates_ps_inv: Applied shear rates in ps^-1, same order.

    Returns:
        Array of viscosities in Pa*s, one per input pair.

    Raises:
        ValueError: No Pres-XY/Pres-YX column in a file, or a zero
            shear rate (division undefined).
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
