"""Scaling-law bridges from simulation observables to experiment.

Small, dependency-free formulas that translate simulated quantities
(volume fraction, loading ratio) toward experimentally reported ones:
a semi-dilute mesh-size scaling (:func:`calculate_mesh_size`), a
water-uptake conversion (:func:`calculate_water_uptake`), and an
unimplemented rubber-elasticity modulus placeholder
(:func:`estimate_elastic_modulus`) that refuses by raising rather than
guessing.  These are approximate mappings, not gated analyses — they
return bare floats, and their assumptions are stated per function.
"""
def calculate_mesh_size(phi, strand_length_nm):
    """Estimate the mesh size xi from a semi-dilute scaling law.

    Uses ``xi ~ strand_length_nm * phi**(-3/4)`` (semi-dilute,
    athermal-solvent assumption).

    Caution: this is a bare scaling relation and may differ from
    literature forms such as Peppas-Merrill; calibrate the prefactor
    and exponent against the validation target before quantitative use.

    Args:
        phi: Polymer volume fraction (dimensionless).
        strand_length_nm: Strand contour length in nm.

    Returns:
        Estimated mesh size in nm; 0.0 for non-positive ``phi``
        (the scaling law is undefined there).
    """
    if phi <= 0:
        return 0.0
    return strand_length_nm * (phi ** (-0.75))


def estimate_elastic_modulus(phi, temp_k, crosslink_density, strand_molar_mass,
                              polymer_density_g_cm3, functionality):
    """Placeholder: rubber-elasticity shear modulus estimate.

    Declared inputs (all currently unused): crosslink density in
    nm^-3, mean strand molar mass in g/mol, polymer density in g/cm^3,
    temperature in K, and junction functionality.

    Raises:
        NotImplementedError: Always — no modulus is estimated silently.
    """
    raise NotImplementedError(
        "Elastic modulus 계산 미구현. "
        "필요 입력: crosslink_density, strand_molar_mass, polymer_density_g_cm3, "
        "temperature, functionality."
    )


def calculate_water_uptake(loading_qm):
    """Convert the loading ratio Qm to a water-uptake percentage.

    ``uptake% = (1 - 1/Qm) * 100`` — the solvent mass fraction of the
    wet gel, given Qm = m_wet / m_dry.

    Args:
        loading_qm: Mass loading ratio (wet/dry, dimensionless, > 0).

    Returns:
        Water uptake as a percentage of wet mass.
    """
    return (1.0 - (1.0 / loading_qm)) * 100.0
