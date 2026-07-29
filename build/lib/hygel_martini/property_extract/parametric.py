import numpy as np
import os
import matplotlib.pyplot as plt
from .analyzer import HydrogelAnalyzer
from .result import PropertyResult


def _property_value(value):
    if isinstance(value, PropertyResult):
        return value.value
    return value


def _flatten_property_results(results):
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
        """
        workspace_dir  : 결과 그래프를 저장할 디렉터리
        top_filename   : 각 point 디렉터리 내의 top 파일 이름
        itp_filename   : 각 point 디렉터리 내의 itp 파일 이름
        gro_filename   : 각 point 디렉터리 내의 gro 파일 이름
        edr_filename   : 각 point 디렉터리 내의 edr 파일 이름
        start_time_ps  : 평형 구간 시작 시간 [ps]
        """
        self.workspace_dir = workspace_dir
        self.top_filename = top_filename
        self.itp_filename = itp_filename
        self.gro_filename = gro_filename
        self.edr_filename = edr_filename
        self.start_time_ps = start_time_ps
        self.points = []

    def add_point(self, temp, press, wt, dir_path):
        self.points.append({'T': temp, 'P': press, 'wt': wt, 'path': dir_path})

    def collect_properties(self):
        """
        등록된 모든 point에 대해 분석을 실행한다.
        반환값: {'results': [...], 'missing_files': [...], 'failed_points': [...]}
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
        """Plots a property vs Temperature."""
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
        """Plots 2D heatmap if multiple T and wt% exist."""
        raise NotImplementedError("2D phase diagram 미구현. interpolation 방식 결정 후 구현 필요.")
