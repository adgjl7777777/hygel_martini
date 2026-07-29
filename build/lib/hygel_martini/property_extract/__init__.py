# 안정 API (검증 중)
from .analyzer import HydrogelAnalyzer
from .swelling import SwellingAnalyzer
from .equilibration import check_stability
from .result import PropertyResult
from .validation_manifest import (
    find_target_properties_for_simulation_property,
    get_target_property,
    load_manifest,
    ManifestTarget,
    ManifestProperty,
)
from .requirements import check_requirements, check_all_requirements, RequirementStatus
from .analysis_jobs import load_analysis_jobs, run_analysis, AnalysisJob
from .extractors import EXTRACTOR_REGISTRY
from .config import load_all
from .timeseries import read_xvg, select_time_window, block_statistics, linear_drift
from .geometry import (
    minimum_image_displacement,
    unwrap_ordered_chain,
    gyration_metrics,
    bond_orientation_metrics,
)
from .aggregation import ContactGraphResult, chain_contact_graph
from .spatial import (
    phase_randomized_field,
    periodic_field_correlation,
    summarize_voxel_counts,
    voxel_counts,
)
from .diffusion import unwrap_trajectory, multi_origin_msd, fit_diffusion_coefficient
from .structure_factor import (
    cic_density_grid,
    fft_structure_factor,
    radial_bin_structure_factor,
    reciprocal_axis_structure_factor,
)
from .pore_size import (
    calculate_periodic_clearance_distribution,
    periodic_clearance_grid,
    periodic_component_summary,
    summarize_periodic_clearance,
)
from .mechanics import (
    affine_simple_shear,
    classical_network_modulus_bounds,
    gromacs_box_values,
    harmonic_strain_energy_kbt,
    paired_step_shear_response,
    simple_shear_box,
    uniaxial_nominal_stress_from_pressure,
    volume_preserving_uniaxial,
    volume_preserving_uniaxial_box,
    write_step_sheared_gro,
    write_uniaxially_deformed_gro,
)
from .mechanics_analysis import (
    analyze_cycle_blocks,
    analyze_paired_ramp,
    holm_adjust,
    paired_step_window_summary,
    paired_step_xvg_summary,
    summarize_equal_realizations,
)
from .network_topology import audit_reduced_network

# 실험적 API (초안 수준, 사용 전 확인 필요)
from .pore_size import get_peak_pore_size, parse_gro_coords
from .polymer_stats import PolymerStats
from .rheology import analyze_shear_rate_viscosity
from .experimental_mapping import calculate_mesh_size, calculate_water_uptake
from .parametric import ParametricAnalyzer
from .simulation_protocols import MDProtocolGenerator

__all__ = [
    "AnalysisJob",
    "ContactGraphResult",
    "EXTRACTOR_REGISTRY",
    "HydrogelAnalyzer",
    "MDProtocolGenerator",
    "ManifestProperty",
    "ManifestTarget",
    "ParametricAnalyzer",
    "PolymerStats",
    "PropertyResult",
    "RequirementStatus",
    "SwellingAnalyzer",
    "affine_simple_shear",
    "analyze_cycle_blocks",
    "analyze_paired_ramp",
    "analyze_shear_rate_viscosity",
    "audit_reduced_network",
    "block_statistics",
    "bond_orientation_metrics",
    "calculate_mesh_size",
    "calculate_periodic_clearance_distribution",
    "calculate_water_uptake",
    "chain_contact_graph",
    "check_all_requirements",
    "check_requirements",
    "check_stability",
    "cic_density_grid",
    "classical_network_modulus_bounds",
    "fft_structure_factor",
    "find_target_properties_for_simulation_property",
    "fit_diffusion_coefficient",
    "get_peak_pore_size",
    "get_target_property",
    "gromacs_box_values",
    "gyration_metrics",
    "harmonic_strain_energy_kbt",
    "holm_adjust",
    "linear_drift",
    "load_all",
    "load_analysis_jobs",
    "load_manifest",
    "minimum_image_displacement",
    "multi_origin_msd",
    "paired_step_shear_response",
    "paired_step_window_summary",
    "paired_step_xvg_summary",
    "parse_gro_coords",
    "periodic_clearance_grid",
    "periodic_component_summary",
    "periodic_field_correlation",
    "phase_randomized_field",
    "radial_bin_structure_factor",
    "read_xvg",
    "reciprocal_axis_structure_factor",
    "run_analysis",
    "select_time_window",
    "simple_shear_box",
    "summarize_equal_realizations",
    "summarize_periodic_clearance",
    "summarize_voxel_counts",
    "uniaxial_nominal_stress_from_pressure",
    "unwrap_ordered_chain",
    "unwrap_trajectory",
    "volume_preserving_uniaxial",
    "volume_preserving_uniaxial_box",
    "voxel_counts",
    "write_step_sheared_gro",
    "write_uniaxially_deformed_gro",
]

# 미구현 — 공개 API에서 제외:
#   calculate_viscosity_green_kubo  (rheology.py)
#   estimate_elastic_modulus        (experimental_mapping.py)
#   find_equilibration_time         (equilibration.py)
#   plot_phase_diagram              (parametric.py)
