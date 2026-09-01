"""Sweep-level analysis across state points (T, P, wt%).

Owns :class:`ParametricAnalyzer`, which runs one
:class:`.analyzer.HydrogelAnalyzer` per registered state-point
directory and gathers the flattened results for cross-point plots.
Points share a fixed file-naming convention inside each directory
(top/itp/gro/edr filenames set at construction).

Failure handling is collect-and-report rather than raise: a point with
a missing top file lands in ``missing_files``, a point whose analysis
throws lands in ``failed_points``, and only clean points enter
``results`` — so one broken directory never aborts a sweep.  Plots are
saved as PNGs into ``workspace_dir``.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from .analyzer import HydrogelAnalyzer
from .result import PropertyResult


def _property_value(value):
    """Return the scalar payload of a PropertyResult, else pass through."""
    if isinstance(value, PropertyResult):
        return value.value
    return value


def _flatten_property_results(results):
    """Flatten a name->PropertyResult mapping into plain plot-ready values.

    Each PropertyResult contributes its bare ``value`` under its name;
    the full serialized results are preserved side-by-side under the
    ``_property_results`` key so status/metadata survive flattening.
    Non-PropertyResult entries pass through unchanged.
    """
    flat = {}
    raw = {}
    for name, result in results.items():
        if isinstance(result, PropertyResult):
            flat[name] = result.value
            raw[name] = result.to_dict()
        else:
            flat[name] = result
    if raw:
        flat['_property_results'] = raw
    return flat


class ParametricAnalyzer:
    """
    Analyzes hydrogel properties across different state points (T, P, wt%).
    """

    def __init__(
        self,
        workspace_dir,
        top_filename='system.top',
        itp_filename='initial_hydrogel.itp',
        gro_filename='production.gro',
        edr_filename='production.edr',
        start_time_ps=100000,
    ):
        """Configure the sweep's file conventions and output location.

        Args:
            workspace_dir: Directory where plots are saved.
            top_filename: Topology filename inside each point directory.
            itp_filename: Polymer itp filename inside each point
                directory.
            gro_filename: Structure filename inside each point
                directory.
            edr_filename: Energy-file name inside each point directory
                (optional per point; enables energy extraction).
            start_time_ps: Equilibrated-window start time (ps) passed
                to every per-point analysis.
        """
        self.workspace_dir = workspace_dir
        self.top_filename = top_filename
        self.itp_filename = itp_filename
        self.gro_filename = gro_filename
        self.edr_filename = edr_filename
        self.start_time_ps = start_time_ps
        self.points = []

    def add_point(self, temp, press, wt, dir_path):
        """Register one state point (T in K, P, wt%) and its directory."""
        self.points.append({'T': temp, 'P': press, 'wt': wt, 'path': dir_path})

    def collect_properties(self):
        """Run the full analysis on every registered point.

        Per point: refuse (record under ``missing_files``) when the top
        file is absent; otherwise build a HydrogelAnalyzer, extract
        ``energy.xvg`` from the edr when one exists, run ``analyze``,
        flatten the results, and merge in the state-point labels.  Any
        exception is caught and recorded under ``failed_points`` so the
        sweep continues.

        Returns:
            Dict with ``results`` (flattened per-point property dicts),
            ``missing_files`` (points skipped for a missing top), and
            ``failed_points`` (points whose analysis raised, with the
            error string).
        """
        results = []
        missing_files = []
        failed_points = []

        for pt in self.points:
            top = os.path.join(pt['path'], self.top_filename)
            itp = os.path.join(pt['path'], self.itp_filename)
            gro = os.path.join(pt['path'], self.gro_filename)
            edr = os.path.join(pt['path'], self.edr_filename)

            if not os.path.exists(top):
                missing_files.append({'point': pt, 'missing': top})
                continue

            try:
                analyzer = HydrogelAnalyzer(top, itp, gro)
                if os.path.exists(edr):
                    xvg_path = os.path.join(pt['path'], "energy.xvg")
                    analyzer.extract_energy_from_edr(edr, output_xvg=xvg_path)

                res = _flatten_property_results(
                    analyzer.analyze(start_time_ps=self.start_time_ps)
                )
                res.update(pt)
                results.append(res)
            except Exception as e:
                failed_points.append({'point': pt, 'error': str(e)})

        return {
            'results': results,
            'missing_files': missing_files,
            'failed_points': failed_points,
        }

    def plot_temperature_sensitivity(self, results, target_property='loading_qm'):
        """Plot one property against temperature and save the PNG.

        Points lacking the property or carrying non-finite/non-numeric
        values are dropped before plotting; the figure is saved as
        ``<target_property>_vs_T.png`` in ``workspace_dir``.

        Args:
            results: Flattened per-point dicts from
                :meth:`collect_properties`.
            target_property: Property key to plot on the y axis.

        Raises:
            ValueError: No point provides a plottable numeric value.
        """
        data = sorted(
            [r for r in results if target_property in r],
            key=lambda x: x['T'],
        )
        ts = [d['T'] for d in data]
        props = [_property_value(d[target_property]) for d in data]

        valid = [
            (t, p) for t, p in zip(ts, props)
            if isinstance(p, (int, float, np.number)) and np.isfinite(p)
        ]
        if not valid:
            raise ValueError(f"plot 가능한 numeric 값이 없습니다: {target_property}")

        ts, props = zip(*valid)

        plt.figure()
        plt.plot(ts, props, 'o-')
        plt.xlabel("Temperature (K)")
        plt.ylabel(target_property)
        plt.title(f"{target_property} vs Temperature")
        plt.savefig(os.path.join(self.workspace_dir, f"{target_property}_vs_T.png"))

    def plot_phase_diagram(self, results, prop='loading_qm'):
        """Placeholder: 2-D property heatmap over T and wt%.

        Raises:
            NotImplementedError: Always — the interpolation scheme is
                undecided, so no diagram is produced silently.
        """
        raise NotImplementedError("2D phase diagram 미구현. interpolation 방식 결정 후 구현 필요.")
