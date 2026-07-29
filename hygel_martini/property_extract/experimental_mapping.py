def calculate_mesh_size(phi, strand_length_nm):
    """
    스케일링 법칙으로 mesh size (xi) 추정.
    xi ≈ strand_length_nm * phi^(-3/4)  (semi-dilute/athermal 가정)

    phi             : polymer volume fraction
    strand_length_nm: strand contour length [nm]

    주의: 이 식은 단순 스케일링이며 Peppas-Merrill 등 논문식과 다를 수 있음.
          real_step.md의 검증 목표식과 비교 후 상수/지수를 보정할 것.
    """
    if phi <= 0:
        return 0.0
    return strand_length_nm * (phi ** (-0.75))


def estimate_elastic_modulus(phi, temp_k, crosslink_density, strand_molar_mass,
                              polymer_density_g_cm3, functionality):
    """
    고무 탄성론 기반 전단 탄성률 추정. 미구현.

    필요 입력:
        crosslink_density    : crosslink 개수/부피 [nm^-3]
        strand_molar_mass    : strand 평균 분자량 [g/mol]
        polymer_density_g_cm3: polymer 밀도 [g/cm^3]
        functionality        : crosslink 기능기 수
    """
    raise NotImplementedError(
        "Elastic modulus 계산 미구현. "
        "필요 입력: crosslink_density, strand_molar_mass, polymer_density_g_cm3, "
        "temperature, functionality."
    )


def calculate_water_uptake(loading_qm):
    """
    Water uptake percentage from loading Qm.
    """
    return (1.0 - (1.0 / loading_qm)) * 100.0
