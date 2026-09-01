"""Water-related physical property helpers for solvation box setup.

Used by the opls_to_martini case builder to estimate how many water
molecules fill a simulation box at a target temperature. No state; the
CoolProp dependency is optional and falls back to an analytic fit.
"""

from __future__ import annotations

from typing import Sequence


def water_density_g_cm3(temp_c: float) -> float:
    """
    Return water density in g/cm^3 at the target temperature.
    Prefer CoolProp. Fallback to Kell equation approximation (0-100C).

    Args:
        temp_c: Temperature in degrees Celsius.

    Returns:
        Liquid water density in g/cm^3 at 1 atm. CoolProp (IAPWS) when
        importable; otherwise the Kell-form polynomial, which is only an
        approximation and intended for the 0-100 C range.
    """
    try:
        from CoolProp.CoolProp import PropsSI  # type: ignore

        rho_kg_m3 = PropsSI("D", "T", temp_c + 273.15, "P", 101325.0, "Water")
        return rho_kg_m3 / 1000.0
    except Exception:
        t = temp_c
        rho_kg_m3 = 1000.0 * (
            1.0
            - ((t + 288.9414) / (508929.2 * (t + 68.12963))) * (t - 3.9863) ** 2
        )
        return rho_kg_m3 / 1000.0


def estimate_water_molecules(
    box_ang: Sequence[float], density_g_cm3: float, molar_mass: float, avogadro: float
) -> int:
    """Estimate how many water molecules fill a rectangular box.

    Args:
        box_ang: Box edge lengths in Angstrom (converted internally to cm^3).
        density_g_cm3: Water density in g/cm^3 at the target temperature.
        molar_mass: Water molar mass in g/mol.
        avogadro: Avogadro constant in 1/mol.

    Returns:
        Non-negative rounded molecule count for a box filled entirely with
        water (solute volume is not subtracted).
    """
    volume_ang3 = box_ang[0] * box_ang[1] * box_ang[2]
    volume_cm3 = volume_ang3 * 1.0e-24
    water_mass_g = density_g_cm3 * volume_cm3
    n_mol = water_mass_g / molar_mass
    n_molecules = int(round(n_mol * avogadro))
    return max(0, n_molecules)
