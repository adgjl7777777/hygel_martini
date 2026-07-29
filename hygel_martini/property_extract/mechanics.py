"""Small-deformation mechanics primitives with explicit claim boundaries.

These functions prepare an instantaneous, volume-preserving simple-shear step
and combine matched positive/negative pressure traces. They do not by themselves
estimate an equilibrium, plateau, or experimental storage modulus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SHEAR_PLANES = ("xy", "xz", "yz")
UNIAXIAL_AXES = ("x", "y", "z")
BOLTZMANN_J_PER_K = 1.380649e-23


def classical_network_modulus_bounds(
    n_strands: int,
    n_junctions: int,
    volume_nm3: float,
    temperature_k: float,
) -> dict[str, float]:
    """Return affine and classical phantom-network shear-modulus estimates.

    The affine estimate is ``n_strands * kBT / V``.  The phantom estimate is
    ``(n_strands - n_junctions) * kBT / V``, equivalent to the usual
    ``(1 - 2/f)`` correction for a regular network with functionality ``f``.

    These are topology-density reference scales, not fitted MD moduli.  They
    assume that all supplied strands are elastically active and therefore do
    not correct for loops, dangling strands, incomplete conversion, spatial
    heterogeneity, finite extensibility, or rate-dependent solvent coupling.
    """
    if not isinstance(n_strands, (int, np.integer)) or n_strands <= 0:
        raise ValueError("n_strands must be a positive integer")
    if not isinstance(n_junctions, (int, np.integer)) or n_junctions < 0:
        raise ValueError("n_junctions must be a non-negative integer")
    if n_junctions >= n_strands:
        raise ValueError("n_junctions must be smaller than n_strands")
    if volume_nm3 <= 0:
        raise ValueError("volume_nm3 must be positive")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")

    kbt_j = BOLTZMANN_J_PER_K * float(temperature_k)
    volume_m3 = float(volume_nm3) * 1.0e-27
    affine_kpa = n_strands * kbt_j / volume_m3 / 1.0e3
    phantom_kpa = (n_strands - n_junctions) * kbt_j / volume_m3 / 1.0e3
    mean_functionality = 2.0 * n_strands / n_junctions if n_junctions else np.inf
    return {
        "affine_shear_modulus_kpa": float(affine_kpa),
        "phantom_shear_modulus_kpa": float(phantom_kpa),
        "mean_functionality": float(mean_functionality),
        "phantom_to_affine_ratio": float(phantom_kpa / affine_kpa),
        "strand_number_density_nm3": float(n_strands / volume_nm3),
    }


def harmonic_strain_energy_kbt(
    modulus_kpa: float,
    strain: float,
    volume_nm3: float,
    temperature_k: float,
) -> float:
    """Return ``(1/2) * modulus * strain**2 * volume`` in units of ``kBT``.

    This diagnostic exposes whether an equilibrium stress signal is expected
    to be thermally resolvable in a finite simulation cell.  It is applicable
    only within a harmonic small-strain interpretation and does not constitute
    a sampling-error estimate.
    """
    if modulus_kpa < 0:
        raise ValueError("modulus_kpa must be non-negative")
    if volume_nm3 <= 0:
        raise ValueError("volume_nm3 must be positive")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    energy_j = (
        0.5
        * float(modulus_kpa)
        * 1.0e3
        * float(strain) ** 2
        * float(volume_nm3)
        * 1.0e-27
    )
    return float(energy_j / (BOLTZMANN_J_PER_K * float(temperature_k)))


def volume_preserving_uniaxial(
    coordinates: np.ndarray,
    stretch: float,
    axis: str = "x",
) -> np.ndarray:
    """Apply a homogeneous incompressible uniaxial deformation.

    The selected axis is scaled by ``stretch`` and both lateral axes by
    ``stretch**(-1/2)``.  This is an affine periodic-cell deformation, not a
    free-surface compression protocol.
    """
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    if stretch <= 0:
        raise ValueError("stretch must be positive")
    if axis not in UNIAXIAL_AXES:
        raise ValueError(f"axis must be one of {UNIAXIAL_AXES}")
    factors = np.full(3, float(stretch) ** -0.5)
    factors[UNIAXIAL_AXES.index(axis)] = float(stretch)
    return xyz * factors


def volume_preserving_uniaxial_box(
    lengths: np.ndarray,
    stretch: float,
    axis: str = "x",
) -> np.ndarray:
    """Return row-wise box vectors after incompressible uniaxial deformation."""
    box = np.asarray(lengths, dtype=float)
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("lengths must contain three positive orthorhombic lengths")
    return np.diag(volume_preserving_uniaxial(box[np.newaxis, :], stretch, axis)[0])


def uniaxial_nominal_stress_from_pressure(
    axial_pressure: np.ndarray,
    lateral_pressure_1: np.ndarray,
    lateral_pressure_2: np.ndarray,
    stretch: float,
    pressure_to_stress: float = 0.1,
) -> np.ndarray:
    """Convert GROMACS pressure components to nominal uniaxial stress.

    GROMACS pressure has the opposite sign from Cauchy normal stress.  The
    axial-minus-mean-lateral Cauchy stress removes the unknown hydrostatic
    contribution.  Dividing that stress difference by ``stretch`` converts it
    to first-Piola/nominal (engineering) stress for comparison with an
    incompressible uniaxial stress--stretch equation.  The default factor
    converts bar to MPa.
    """
    if stretch <= 0:
        raise ValueError("stretch must be positive")
    axial, lateral_1, lateral_2 = np.broadcast_arrays(
        np.asarray(axial_pressure, dtype=float),
        np.asarray(lateral_pressure_1, dtype=float),
        np.asarray(lateral_pressure_2, dtype=float),
    )
    lateral_mean = 0.5 * (lateral_1 + lateral_2)
    return -(axial - lateral_mean) / float(stretch) * pressure_to_stress


def affine_simple_shear(
    coordinates: np.ndarray,
    gamma: float,
    plane: str = "xy",
) -> np.ndarray:
    """Return coordinates after a simple engineering-shear step."""
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    if plane not in SHEAR_PLANES:
        raise ValueError(f"plane must be one of {SHEAR_PLANES}")
    sheared = xyz.copy()
    if plane == "xy":
        sheared[:, 0] += gamma * xyz[:, 1]
    elif plane == "xz":
        sheared[:, 0] += gamma * xyz[:, 2]
    else:
        sheared[:, 1] += gamma * xyz[:, 2]
    return sheared


def simple_shear_box(
    lengths: np.ndarray,
    gamma: float,
    plane: str = "xy",
) -> np.ndarray:
    """Return three GROMACS box vectors as rows after volume-preserving shear."""
    box = np.asarray(lengths, dtype=float)
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError("lengths must contain three positive orthorhombic lengths")
    if plane not in SHEAR_PLANES:
        raise ValueError(f"plane must be one of {SHEAR_PLANES}")
    vectors = np.diag(box)
    if plane == "xy":
        vectors[1, 0] = gamma * box[1]
    elif plane == "xz":
        vectors[2, 0] = gamma * box[2]
    else:
        vectors[2, 1] = gamma * box[2]
    return vectors


def gromacs_box_values(box_vectors: np.ndarray) -> np.ndarray:
    """Convert three row-wise box vectors to the nine-value GRO ordering."""
    vectors = np.asarray(box_vectors, dtype=float)
    if vectors.shape != (3, 3):
        raise ValueError("box_vectors must have shape (3, 3)")
    v1, v2, v3 = vectors
    return np.array(
        [v1[0], v2[1], v3[2], v1[1], v1[2], v2[0], v2[2], v3[0], v3[1]]
    )


def write_step_sheared_gro(
    source: str | Path,
    destination: str | Path,
    gamma: float,
    plane: str = "xy",
    orthorhombic_tolerance: float = 1.0e-8,
) -> None:
    """Write a sheared GRO while preserving atom fields and velocities.

    The input must be orthorhombic. Coordinates and the periodic box receive the
    same affine transform, which preserves fractional coordinates and volume.
    Velocities are deliberately unchanged for an instantaneous step strain.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    lines = source_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file is too short")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError("invalid GRO atom-count line") from exc
    if len(lines) != atom_count + 3:
        raise ValueError(
            f"expected {atom_count + 3} GRO lines, found {len(lines)}"
        )

    raw_box = np.asarray([float(value) for value in lines[-1].split()])
    if raw_box.size not in (3, 9):
        raise ValueError("GRO box must contain 3 or 9 values")
    if raw_box.size == 9 and np.any(np.abs(raw_box[3:]) > orthorhombic_tolerance):
        raise ValueError("input GRO must be orthorhombic")
    lengths = raw_box[:3]
    if np.any(lengths <= 0):
        raise ValueError("box lengths must be positive")

    coordinates = np.empty((atom_count, 3), dtype=float)
    for index, line in enumerate(lines[2:-1]):
        if len(line) < 44:
            raise ValueError(f"atom line {index + 1} is shorter than 44 columns")
        coordinates[index] = (
            float(line[20:28]),
            float(line[28:36]),
            float(line[36:44]),
        )
    sheared = affine_simple_shear(coordinates, gamma, plane)

    output = [f"{lines[0]} | affine {plane} gamma={gamma:+.6f}", lines[1]]
    for line, (x, y, z) in zip(lines[2:-1], sheared):
        output.append(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}{line[44:]}")
    gro_box = gromacs_box_values(simple_shear_box(lengths, gamma, plane))
    output.append("".join(f"{value:10.5f}" for value in gro_box))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("\n".join(output) + "\n")


def write_uniaxially_deformed_gro(
    source: str | Path,
    destination: str | Path,
    stretch: float,
    axis: str = "x",
    orthorhombic_tolerance: float = 1.0e-8,
) -> None:
    """Write an affinely deformed constant-volume GRO file.

    Coordinates and the periodic box receive the same incompressible uniaxial
    transform, preserving fractional coordinates and volume.  Atom fields and
    any velocity columns are retained verbatim apart from the coordinates.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    lines = source_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file is too short")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError("invalid GRO atom-count line") from exc
    if len(lines) != atom_count + 3:
        raise ValueError(
            f"expected {atom_count + 3} GRO lines, found {len(lines)}"
        )

    raw_box = np.asarray([float(value) for value in lines[-1].split()])
    if raw_box.size not in (3, 9):
        raise ValueError("GRO box must contain 3 or 9 values")
    if raw_box.size == 9 and np.any(np.abs(raw_box[3:]) > orthorhombic_tolerance):
        raise ValueError("input GRO must be orthorhombic")
    lengths = raw_box[:3]
    if np.any(lengths <= 0):
        raise ValueError("box lengths must be positive")

    coordinates = np.empty((atom_count, 3), dtype=float)
    for index, line in enumerate(lines[2:-1]):
        if len(line) < 44:
            raise ValueError(f"atom line {index + 1} is shorter than 44 columns")
        coordinates[index] = (
            float(line[20:28]),
            float(line[28:36]),
            float(line[36:44]),
        )
    deformed = volume_preserving_uniaxial(coordinates, stretch, axis)

    output = [
        f"{lines[0]} | incompressible {axis} lambda={stretch:.6f}",
        lines[1],
    ]
    for line, (x, y, z) in zip(lines[2:-1], deformed):
        output.append(f"{line[:20]}{x:8.3f}{y:8.3f}{z:8.3f}{line[44:]}")
    box_vectors = volume_preserving_uniaxial_box(lengths, stretch, axis)
    output.append("".join(f"{value:10.5f}" for value in np.diag(box_vectors)))
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("\n".join(output) + "\n")


def paired_step_shear_response(
    baseline_pressure: np.ndarray,
    positive_pressure: np.ndarray,
    negative_pressure: np.ndarray,
    gamma: float,
    pressure_to_modulus: float = 0.1,
) -> dict[str, np.ndarray]:
    """Decompose matched +/- pressure traces into odd and even responses.

    ``pressure_to_modulus=0.1`` converts pressure in bar to apparent modulus in
    MPa. GROMACS pressure has the opposite sign from Cauchy shear stress, hence
    ``G = -(P_plus-P_minus)/(2*gamma)``.
    """
    if gamma == 0:
        raise ValueError("gamma must be non-zero")
    p0, pp, pm = np.broadcast_arrays(
        np.asarray(baseline_pressure, dtype=float),
        np.asarray(positive_pressure, dtype=float),
        np.asarray(negative_pressure, dtype=float),
    )
    odd_pressure = 0.5 * (pp - pm)
    even_residual = 0.5 * (pp + pm) - p0
    return {
        "odd_pressure": odd_pressure,
        "even_residual_pressure": even_residual,
        "apparent_modulus": -odd_pressure / gamma * pressure_to_modulus,
        "positive_apparent_modulus": -(pp - p0) / gamma * pressure_to_modulus,
        "negative_apparent_modulus": (pm - p0) / gamma * pressure_to_modulus,
    }
