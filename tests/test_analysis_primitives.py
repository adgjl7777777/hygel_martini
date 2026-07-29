import tempfile
import unittest
from pathlib import Path

import numpy as np

from hygel_martini.property_extract.aggregation import chain_contact_graph
from hygel_martini.property_extract.diffusion import (
    fit_diffusion_coefficient,
    multi_origin_msd,
    unwrap_trajectory,
)
from hygel_martini.property_extract.geometry import (
    bond_orientation_metrics,
    gyration_metrics,
    minimum_image_displacement,
    unwrap_ordered_chain,
)
from hygel_martini.property_extract.mechanics import (
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
from hygel_martini.property_extract.pore_size import (
    calculate_periodic_clearance_distribution,
    periodic_clearance_grid,
    periodic_component_summary,
)
from hygel_martini.property_extract.spatial import (
    phase_randomized_field,
    periodic_field_correlation,
    summarize_voxel_counts,
    voxel_counts,
)
from hygel_martini.property_extract.structure_factor import (
    cic_density_grid,
    fft_structure_factor,
    radial_bin_structure_factor,
    reciprocal_axis_structure_factor,
)
from hygel_martini.property_extract.timeseries import (
    block_statistics,
    linear_drift,
    read_xvg,
    select_time_window,
)


class TimeSeriesTests(unittest.TestCase):
    def test_xvg_window_blocks_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "energy.xvg"
            path.write_text('@ s0 legend "Volume"\n0 10\n1 11\n2 12\n3 13\n')
            legends, data = read_xvg(path)
        self.assertEqual(legends, ["Volume"])
        times, values = select_time_window(data[:, 0], data[:, 1], 1, 3)
        self.assertEqual(values.tolist(), [11.0, 12.0, 13.0])
        self.assertAlmostEqual(block_statistics(values, 3)["mean"], 12.0)
        self.assertAlmostEqual(linear_drift(times, values)["slope_per_time"], 1.0)


class GeometryTests(unittest.TestCase):
    def test_minimum_image_chain_and_rg(self):
        box = np.array([10.0, 10.0, 10.0])
        delta = minimum_image_displacement(np.array([9.8, 0.0, 0.0]), box)
        np.testing.assert_allclose(delta, [-0.2, 0.0, 0.0])
        wrapped = np.array([[9.8, 0, 0], [0.1, 0, 0], [0.4, 0, 0]])
        chain = unwrap_ordered_chain(wrapped, box)
        np.testing.assert_allclose(np.diff(chain[:, 0]), [0.3, 0.3])
        metrics = gyration_metrics(chain)
        self.assertAlmostEqual(metrics["end_to_end"], 0.6)
        self.assertAlmostEqual(metrics["relative_shape_anisotropy"], 1.0)

    def test_bond_orientation_tensor(self):
        aligned = bond_orientation_metrics(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        np.testing.assert_allclose(aligned["eigenvalues"], [-0.5, -0.5, 1.0])
        self.assertAlmostEqual(aligned["largest_eigenvalue"], 1.0)
        isotropic_axes = bond_orientation_metrics(np.eye(3))
        np.testing.assert_allclose(isotropic_axes["orientation_tensor"], np.zeros((3, 3)))
        self.assertAlmostEqual(isotropic_axes["largest_eigenvalue"], 0.0)


class AggregationTests(unittest.TestCase):
    def test_periodic_contact_components(self):
        chains = [
            np.array([[0.1, 0.1, 0.1]]),
            np.array([[9.9, 0.1, 0.1]]),
            np.array([[5.0, 5.0, 5.0]]),
        ]
        result = chain_contact_graph(chains, np.array([10.0, 10.0, 10.0]), 0.3)
        self.assertEqual(result.components, ((0, 1), (2,)))
        self.assertEqual(result.edge_contact_counts[(0, 1)], 1)


class SpatialTests(unittest.TestCase):
    def test_voxel_count_conservation(self):
        positions = np.array([[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]])
        counts, spacing = voxel_counts(positions, np.array([2.0, 2.0, 2.0]), 1.0)
        self.assertEqual(int(counts.sum()), 2)
        np.testing.assert_allclose(spacing, [1.0, 1.0, 1.0])
        self.assertEqual(summarize_voxel_counts(counts)["n_voxels"], 8)

    def test_periodic_field_translation_alignment(self):
        reference = np.zeros((8, 8, 8), dtype=float)
        reference[1, 2, 3] = 2.0
        reference[5, 6, 1] = 1.0
        shifted = np.roll(reference, shift=(2, -1, 3), axis=(0, 1, 2))
        result = periodic_field_correlation(reference, shifted)
        self.assertLess(result["zero_shift_correlation"], 0.5)
        self.assertAlmostEqual(
            result["translation_aligned_correlation"], 1.0, places=12
        )

    def test_phase_randomized_field_preserves_mean_and_spectrum(self):
        rng = np.random.default_rng(1234)
        field = rng.normal(loc=2.5, scale=0.7, size=(8, 10, 12))
        surrogate = phase_randomized_field(field, np.random.default_rng(5678))
        self.assertAlmostEqual(float(np.mean(surrogate)), float(np.mean(field)))
        original_amplitude = np.abs(np.fft.fftn(field - np.mean(field)))
        surrogate_amplitude = np.abs(
            np.fft.fftn(surrogate - np.mean(surrogate))
        )
        np.testing.assert_allclose(
            surrogate_amplitude, original_amplitude, rtol=1e-12, atol=1e-10
        )
        correlation = float(np.corrcoef(field.ravel(), surrogate.ravel())[0, 1])
        self.assertLess(abs(correlation), 0.2)


class PoreClearanceTests(unittest.TestCase):
    def test_periodic_clearance_wraps_and_is_chunk_invariant(self):
        box = np.array([2.0, 2.0, 2.0])
        obstacles = [(np.array([[0.1, 0.5, 0.5]]), 0.1)]
        full, spacing = periodic_clearance_grid(
            obstacles, box, grid_spacing=0.5, chunk_size=10_000
        )
        chunked, _ = periodic_clearance_grid(
            obstacles, box, grid_spacing=0.5, chunk_size=3
        )
        np.testing.assert_allclose(full, chunked)
        np.testing.assert_allclose(spacing, [0.5, 0.5, 0.5])
        # Cell centres at x=0.25 and x=1.75 are both close through PBC.
        self.assertLess(full[0, 0, 0], 0.4)
        self.assertLess(full[-1, 0, 0], 0.6)

    def test_mixed_obstacles_reduce_probe_admissible_volume(self):
        box = np.array([3.0, 3.0, 3.0])
        polymer = np.array([[1.5, 1.5, 1.5]])
        water = np.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]])
        _, _, polymer_only = calculate_periodic_clearance_distribution(
            [(polymer, 0.24)],
            box,
            grid_spacing=0.3,
            probe_radius=0.1657,
            bins=20,
            chunk_size=50,
        )
        _, _, occupied = calculate_periodic_clearance_distribution(
            [(polymer, 0.24), (water, 0.14)],
            box,
            grid_spacing=0.3,
            probe_radius=0.1657,
            bins=20,
            chunk_size=50,
        )
        self.assertLess(
            occupied["probe_admissible_fraction"],
            polymer_only["probe_admissible_fraction"],
        )

    def test_periodic_components_merge_across_faces(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[0, 1, 1] = True
        mask[-1, 1, 1] = True
        mask[2, 3, 3] = True
        result = periodic_component_summary(mask)
        self.assertEqual(result["n_periodic_components"], 2)
        self.assertEqual(result["largest_component_voxels"], 2)


class MechanicsTests(unittest.TestCase):
    def test_classical_network_bounds_and_signal_scale(self):
        bounds = classical_network_modulus_bounds(
            n_strands=128,
            n_junctions=64,
            volume_nm3=8000.0,
            temperature_k=310.0,
        )
        self.assertAlmostEqual(bounds["mean_functionality"], 4.0)
        self.assertAlmostEqual(bounds["phantom_to_affine_ratio"], 0.5)
        self.assertAlmostEqual(
            bounds["affine_shear_modulus_kpa"],
            2.0 * bounds["phantom_shear_modulus_kpa"],
        )
        signal = harmonic_strain_energy_kbt(
            modulus_kpa=36.3,
            strain=0.02,
            volume_nm3=8000.0,
            temperature_k=310.0,
        )
        self.assertGreater(signal, 0.01)
        self.assertLess(signal, 0.02)

    def test_affine_shear_preserves_volume_and_fractional_mapping(self):
        coordinates = np.array([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]])
        sheared = affine_simple_shear(coordinates, 0.01, "xy")
        np.testing.assert_allclose(
            sheared[:, 0], coordinates[:, 0] + 0.01 * coordinates[:, 1]
        )
        box = simple_shear_box(np.array([10.0, 20.0, 30.0]), 0.01, "xy")
        self.assertAlmostEqual(float(np.linalg.det(box)), 6000.0)
        np.testing.assert_allclose(
            gromacs_box_values(box),
            [10.0, 20.0, 30.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        )

    def test_step_sheared_gro_preserves_velocity_columns(self):
        text = (
            "toy\n"
            "1\n"
            "    1PEG     B    1   1.000   2.000   3.000  0.1000  0.2000  0.3000\n"
            "   10.00000   20.00000   30.00000\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.gro"
            destination = Path(tmp) / "out.gro"
            source.write_text(text)
            write_step_sheared_gro(source, destination, 0.01, "xy")
            output = destination.read_text().splitlines()
        self.assertAlmostEqual(float(output[2][20:28]), 1.02, places=3)
        self.assertEqual(output[2][44:], text.splitlines()[2][44:])

    def test_uniaxial_transform_preserves_volume_and_fractional_mapping(self):
        lengths = np.array([10.0, 20.0, 30.0])
        coordinates = np.array([[2.5, 5.0, 7.5]])
        stretch = 0.64
        deformed = volume_preserving_uniaxial(coordinates, stretch, "x")
        box = volume_preserving_uniaxial_box(lengths, stretch, "x")
        self.assertAlmostEqual(float(np.linalg.det(box)), float(np.prod(lengths)))
        np.testing.assert_allclose(
            deformed / np.diag(box), coordinates / lengths, atol=1e-15
        )
        np.testing.assert_allclose(np.diag(box), [6.4, 25.0, 37.5])

    def test_uniaxial_gro_and_nominal_stress_conversion(self):
        text = (
            "toy\n"
            "1\n"
            "    1PEG     B    1   1.000   2.000   3.000  0.1000  0.2000  0.3000\n"
            "   10.00000   20.00000   30.00000\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "in.gro"
            destination = Path(tmp) / "out.gro"
            source.write_text(text)
            write_uniaxially_deformed_gro(source, destination, 0.25, "x")
            output = destination.read_text().splitlines()
        np.testing.assert_allclose(
            [float(output[2][20:28]), float(output[2][28:36]), float(output[2][36:44])],
            [0.25, 4.0, 6.0],
        )
        self.assertEqual(output[2][44:], text.splitlines()[2][44:])
        np.testing.assert_allclose(
            [float(value) for value in output[-1].split()],
            [2.5, 40.0, 60.0],
        )
        # Pressure is minus Cauchy stress. At lambda=0.5, a 10 MPa
        # compressive nominal stress corresponds to +50 bar axial pressure
        # relative to the lateral pressure.
        nominal = uniaxial_nominal_stress_from_pressure(
            np.array([50.0]), np.array([0.0]), np.array([0.0]), 0.5
        )
        np.testing.assert_allclose(nominal, [-10.0])

    def test_paired_pressure_sign_and_even_residual(self):
        response = paired_step_shear_response(
            np.array([10.0]),
            np.array([-210.0]),
            np.array([230.0]),
            gamma=0.01,
        )
        np.testing.assert_allclose(response["odd_pressure"], [-220.0])
        np.testing.assert_allclose(response["even_residual_pressure"], [0.0])
        np.testing.assert_allclose(response["apparent_modulus"], [2200.0])


class DiffusionTests(unittest.TestCase):
    def test_unwrap_msd_and_fit(self):
        wrapped = np.array([
            [[9.8, 0.0, 0.0]],
            [[0.1, 0.0, 0.0]],
            [[0.4, 0.0, 0.0]],
            [[0.7, 0.0, 0.0]],
        ])
        unwrapped = unwrap_trajectory(wrapped, np.array([10.0, 10.0, 10.0]))
        np.testing.assert_allclose(unwrapped[:, 0, 0], [9.8, 10.1, 10.4, 10.7])
        times, msd, counts = multi_origin_msd(unwrapped, 1.0, max_lag_frames=3)
        np.testing.assert_allclose(msd, [0.0, 0.09, 0.36, 0.81], atol=1e-12)
        self.assertEqual(counts.tolist(), [4, 3, 2, 1])
        fit = fit_diffusion_coefficient(times, msd, 1.0, 3.0)
        self.assertGreater(fit["diffusion_coefficient_coordinate2_per_time"], 0.0)


class StructureFactorTests(unittest.TestCase):
    def test_cic_conservation_and_axis_extinction(self):
        box = np.array([8.0, 8.0, 8.0])
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        density = cic_density_grid(positions, box, (16, 16, 16))
        self.assertAlmostEqual(float(density.sum()), 2.0)

        axis = reciprocal_axis_structure_factor(positions, box, max_mode=2)
        self.assertAlmostEqual(float(axis["S_x"][0]), 0.0, places=12)
        self.assertAlmostEqual(float(axis["S_x"][1]), 2.0, places=12)
        np.testing.assert_allclose(axis["S_y"], [2.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(axis["S_z"], [2.0, 2.0], atol=1e-12)

    def test_fft_modes_and_radial_bins(self):
        box = np.array([8.0, 8.0, 8.0])
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        q, structure = fft_structure_factor(
            positions, box, 16, q_max=2.0, deconvolve_cic=True,
        )
        self.assertEqual(q.shape, structure.shape)
        self.assertTrue(np.all(np.isfinite(structure)))
        edges = np.arange(0.0, 2.0 + 0.25, 0.25)
        binned = radial_bin_structure_factor(q, structure, edges)
        self.assertEqual(len(binned["q_center"]), len(edges) - 1)
        self.assertGreater(int(np.sum(binned["n_modes"])), 0)


if __name__ == "__main__":
    unittest.main()
