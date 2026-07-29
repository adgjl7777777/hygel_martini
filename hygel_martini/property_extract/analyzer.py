import os
from .swelling import SwellingAnalyzer
from .pore_size import parse_gro_coords, get_peak_pore_size
from .equilibration import check_stability
from .gmx_utils import run_gmx
from .result import PropertyResult


class HydrogelAnalyzer:
    def __init__(
        self,
        top_file,
        itp_file,
        gro_file=None,
        energy_xvg=None,
        # --- polymer/solvent identity ---
        pore_selection_residues=None,
        polymer_residue_name=None,
        polymer_atom_name=None,
        solvent_molecule_names='W',
        # --- mass / volume ---
        polymer_bead_mass=45.0,
        solvent_bead_mass=72.0,
        polymer_bead_vol_nm3=0.065,
        # --- pore size grid ---
        pore_grid_spacing_nm=0.2,
        pore_bead_radius_nm=0.24,
        pore_bins=50,
        # --- equilibration ---
        equilibration_threshold=0.01,
        equilibration_window=0.2,
        start_time_ps=0,
    ):
        """
        top_file                : system.top 경로
        itp_file                : hydrogel itp 경로
        gro_file                : 참조 .gro 경로
        energy_xvg              : 기존 energy.xvg 경로
        pore_selection_residues : pore 분석에 사용할 residue 이름 목록 (None → {"PEO","HYDROGEL"})
        polymer_residue_name    : ITP 파싱 시 residue 필터 (None = 전체)
        polymer_atom_name       : ITP 파싱 시 atom 이름 필터 (None = 전체)
        solvent_molecule_names   : top [ molecules ]에서 solvent 이름 (str 또는 list[str])
        polymer_bead_mass       : polymer bead 질량 [amu]. 기본값 45.0 (M3 PEO SN3r)
        solvent_bead_mass       : solvent bead 질량 [amu]. 기본값 72.0 (Martini W)
        polymer_bead_vol_nm3    : polymer bead 부피 [nm³]. 기본값 0.065 (FF 의존)
        pore_grid_spacing_nm    : pore grid 간격 [nm]
        pore_bead_radius_nm     : pore bead 유효 반경 [nm]
        pore_bins               : pore histogram bin 수
        equilibration_threshold : 안정성 판정 최대 상대 drift
        equilibration_window    : drift 계산에 사용할 끝 부분 비율
        start_time_ps           : 평균 계산에 사용할 시작 시간 [ps]
        """
        self.top_file = top_file
        self.itp_file = itp_file
        self.gro_file = gro_file
        self.energy_xvg = energy_xvg
        self.pore_selection_residues = pore_selection_residues
        self.pore_grid_spacing_nm = pore_grid_spacing_nm
        self.pore_bead_radius_nm = pore_bead_radius_nm
        self.pore_bins = pore_bins
        self.equilibration_threshold = equilibration_threshold
        self.equilibration_window = equilibration_window
        self.start_time_ps = start_time_ps

        self.swelling_analyzer = SwellingAnalyzer.from_files(
            top_file,
            itp_file,
            polymer_bead_mass=polymer_bead_mass,
            solvent_bead_mass=solvent_bead_mass,
            polymer_bead_vol_nm3=polymer_bead_vol_nm3,
            polymer_residue_name=polymer_residue_name,
            polymer_atom_name=polymer_atom_name,
            solvent_molecule_names=solvent_molecule_names,
        )

    @classmethod
    def from_config(cls, config_path):
        """
        YAML 설정 파일로부터 HydrogelAnalyzer를 생성한다.
        파일 경로는 config_path 기준 상대 경로로 해석된다.
        """
        import yaml

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg_dir = os.path.dirname(os.path.abspath(config_path))

        def resolve(p):
            if not p:
                return None
            return p if os.path.isabs(str(p)) else os.path.join(cfg_dir, str(p))

        files = cfg.get('files', {})
        top_file = resolve(files.get('top', 'system.top'))
        itp_file = resolve(files.get('itp', 'initial_hydrogel.itp'))
        gro_file = resolve(files.get('gro'))
        energy_xvg = resolve(files.get('energy_xvg'))

        # GROMACS executable — YAML > 환경 변수, 하드코딩 없음
        gmx_cfg = cfg.get('gromacs', {})
        gmx_exe = gmx_cfg.get('executable')
        if gmx_exe:
            os.environ['GMX_BIN'] = str(gmx_exe)

        # polymer / solvent identity
        polymer_cfg = cfg.get('components', {}).get('polymer', {})
        solvent_cfg = cfg.get('components', {}).get('solvent', {})

        residue_names = polymer_cfg.get('residue_names') or []
        atom_names = polymer_cfg.get('atom_names') or []
        polymer_residue_name = residue_names[0] if len(residue_names) == 1 else None
        polymer_atom_name = atom_names[0] if len(atom_names) == 1 else None

        solvent_molecule_names = solvent_cfg.get('molecule_names') or ['W']

        # mass — null은 명시적 오류 (ITP mass 컬럼 읽기 미구현, 값을 직접 지정해야 함)
        mass_cfg = cfg.get('mass', {})
        _poly_mass_raw = mass_cfg.get('default_polymer_bead_mass')
        if _poly_mass_raw is None:
            raise ValueError(
                "mass.default_polymer_bead_mass 가 null입니다. "
                "ITP mass 컬럼 읽기는 미구현입니다. 값을 직접 지정하세요. "
                "(예: default_polymer_bead_mass: 45.0)"
            )
        polymer_bead_mass = float(_poly_mass_raw)

        _sol_mass_raw = mass_cfg.get('water_bead_mass')
        if _sol_mass_raw is None:
            raise ValueError(
                "mass.water_bead_mass 가 null입니다. "
                "(예: water_bead_mass: 72.0  # Martini W = 4 × 18)"
            )
        solvent_bead_mass = float(_sol_mass_raw)

        # volume fraction — bead_volume method에서 null은 오류
        vf_cfg = cfg.get('volume_fraction', {})
        vf_method = vf_cfg.get('method', 'bead_volume')
        if vf_method != 'bead_volume':
            raise ValueError(
                f"volume_fraction.method={vf_method!r} 는 아직 지원하지 않습니다. "
                "현재 구현은 bead_volume만 계산할 수 있습니다."
            )
        _bvol_raw = vf_cfg.get('bead_volume_nm3')
        if _bvol_raw is None:
            raise ValueError(
                "volume_fraction.method 가 bead_volume인데 bead_volume_nm3 가 null입니다. "
                "FF에 맞는 값을 지정하세요. "
                "(예: bead_volume_nm3: 0.065  # M3 SN3r 추정값)"
            )
        polymer_bead_vol_nm3 = float(_bvol_raw)

        # pore size
        pore_cfg = cfg.get('pore_size', {})
        pore_selection_residues = (
            pore_cfg.get('selection_residues')
            or polymer_cfg.get('include_residues')
            or residue_names
            or None
        )
        pore_grid_spacing_nm = float(pore_cfg.get('grid_spacing_nm', 0.2))
        pore_bins = int(pore_cfg.get('bins', 50))

        bead_radius_cfg = pore_cfg.get('bead_radius_nm', 0.24)
        if isinstance(bead_radius_cfg, dict):
            pore_bead_radius_nm = float(next(iter(bead_radius_cfg.values()), 0.24))
        else:
            pore_bead_radius_nm = float(bead_radius_cfg)

        # equilibration
        eq_cfg = cfg.get('equilibration', {})
        start_time_ps = float(eq_cfg.get('start_time_ps', 0))
        equilibration_threshold = float(eq_cfg.get('drift_threshold', 0.01))
        equilibration_window = float(eq_cfg.get('last_fraction', 0.2))

        instance = cls(
            top_file=top_file,
            itp_file=itp_file,
            gro_file=gro_file,
            energy_xvg=energy_xvg,
            pore_selection_residues=pore_selection_residues,
            polymer_residue_name=polymer_residue_name,
            polymer_atom_name=polymer_atom_name,
            solvent_molecule_names=solvent_molecule_names,
            polymer_bead_mass=polymer_bead_mass,
            solvent_bead_mass=solvent_bead_mass,
            polymer_bead_vol_nm3=polymer_bead_vol_nm3,
            pore_grid_spacing_nm=pore_grid_spacing_nm,
            pore_bead_radius_nm=pore_bead_radius_nm,
            pore_bins=pore_bins,
            equilibration_threshold=equilibration_threshold,
            equilibration_window=equilibration_window,
            start_time_ps=start_time_ps,
        )
        instance._config = cfg
        instance._config_path = config_path
        return instance

    def extract_energy_from_edr(
        self,
        edr_file,
        output_xvg='energy.xvg',
        terms=None,
    ):
        """
        gmx energy로 edr 파일에서 Volume 등을 xvg로 추출.
        cwd는 edr_file 위치 기준으로 자동 설정.
        """
        if terms is None:
            terms = ['Volume', 'Potential', 'Density']

        edr_dir = os.path.dirname(os.path.abspath(edr_file))
        input_text = "\n".join(terms) + "\n\n"

        try:
            run_gmx(
                ['energy', '-f', edr_file, '-o', output_xvg],
                input_text=input_text,
                cwd=edr_dir,
            )
            self.energy_xvg = output_xvg
            return output_xvg
        except Exception as e:
            raise RuntimeError(f"gmx energy 실행 실패: {e}") from e

    def analyze(self, start_time_ps=None) -> dict[str, PropertyResult]:
        """
        사용 가능한 파일로 분석 수행.
        start_time_ps: None 이면 생성자에서 설정한 값 사용.

        반환값: {property_name: PropertyResult}
        """
        if start_time_ps is None:
            start_time_ps = self.start_time_ps

        results: dict[str, PropertyResult] = {}

        # 0. 초기 조성 요약 — energy_xvg 없이도 항상 계산
        comp = self.swelling_analyzer.composition_summary()
        results[comp.property] = comp

        # 1. Polymer volume fraction (energy xvg 필요)
        if self.energy_xvg and os.path.exists(self.energy_xvg):
            try:
                phi_result = self.swelling_analyzer.analyze_trajectory(
                    self.energy_xvg, start_time_ps=start_time_ps
                )
                results[phi_result.property] = phi_result

                # 안정성 체크
                from .gmx_utils import parse_xvg
                xvg_data = parse_xvg(self.energy_xvg)
                vol_key = next((k for k in xvg_data if 'Volume' in k), None)
                if vol_key:
                    times = xvg_data['time']
                    mask = times >= start_time_ps
                    if mask.sum() >= 10:
                        stab = check_stability(
                            xvg_data[vol_key][mask],
                            threshold=self.equilibration_threshold,
                            window=self.equilibration_window,
                        )
                        results[stab.property] = stab
                    else:
                        results['volume_stability'] = PropertyResult.insufficient_data(
                            'volume_stability',
                            reason=(
                                f"start_time_ps={start_time_ps} 이후 volume frame이 "
                                f"{int(mask.sum())}개뿐입니다. 최소 10개가 필요합니다."
                            ),
                        )
                else:
                    results['volume_stability'] = PropertyResult.invalid_input(
                        'volume_stability',
                        reason=f"Volume 컬럼을 찾을 수 없습니다: {self.energy_xvg}",
                        inputs=[self.energy_xvg],
                    )
            except ValueError as e:
                results['polymer_volume_fraction'] = PropertyResult.invalid_input(
                    'polymer_volume_fraction',
                    reason=str(e),
                    inputs=[self.energy_xvg or 'energy.xvg'],
                    validation_role='direct',
                )
            except Exception as e:
                results['polymer_volume_fraction'] = PropertyResult.analysis_failed(
                    'polymer_volume_fraction',
                    error=str(e),
                    inputs=[self.energy_xvg or 'energy.xvg'],
                    validation_role='direct',
                )
        else:
            results['polymer_volume_fraction'] = PropertyResult.missing(
                'polymer_volume_fraction',
                missing_inputs=[self.energy_xvg or 'energy.xvg (not set)'],
                validation_role='direct',
            )

        # 2. Pore size (GRO 필요)
        if self.gro_file and os.path.exists(self.gro_file):
            try:
                coords, box = parse_gro_coords(
                    self.gro_file,
                    selection_residues=self.pore_selection_residues,
                )
                if len(coords) > 0:
                    pore_result = get_peak_pore_size(
                        coords, box,
                        grid_spacing=self.pore_grid_spacing_nm,
                        bead_radius=self.pore_bead_radius_nm,
                        bins=self.pore_bins,
                    )
                    results[pore_result.property] = pore_result
                else:
                    results['pore_size_single_frame_grid'] = PropertyResult.invalid_input(
                        'pore_size_single_frame_grid',
                        reason='polymer atom 0개 선택됨 — pore_selection_residues 확인 필요',
                        inputs=[self.gro_file],
                        validation_role='proxy',
                        metadata={'target_aliases': ['pore_diameter_nm']},
                    )
            except ValueError as e:
                results['pore_size_single_frame_grid'] = PropertyResult.invalid_input(
                    'pore_size_single_frame_grid',
                    reason=str(e),
                    inputs=[self.gro_file],
                    validation_role='proxy',
                    metadata={'target_aliases': ['pore_diameter_nm']},
                )
            except Exception as e:
                results['pore_size_single_frame_grid'] = PropertyResult.analysis_failed(
                    'pore_size_single_frame_grid',
                    error=str(e),
                    inputs=[self.gro_file],
                    validation_role='proxy',
                    metadata={'target_aliases': ['pore_diameter_nm']},
                )
        else:
            results['pore_size_single_frame_grid'] = PropertyResult.missing(
                'pore_size_single_frame_grid',
                missing_inputs=[self.gro_file or 'gro file (not set)'],
                validation_role='proxy',
                metadata={'target_aliases': ['pore_diameter_nm']},
            )

        return results

    def report(self, results: dict[str, PropertyResult], targets=None):
        print("\n" + "=" * 50)
        print("         Hydrogel Property Analysis")
        print("=" * 50)

        for name, pr in results.items():
            if pr.status == 'missing_required_md':
                print(f"  {name:<36} [MISSING MD]")
                for inp in pr.missing_required_inputs:
                    print(f"    required: {inp}")
                continue
            if pr.status != 'computed':
                print(f"  {name:<36} [{_format_status(pr.status)}]")
                if pr.metadata.get('reason'):
                    print(f"    reason: {pr.metadata['reason']}")
                if pr.metadata.get('error'):
                    print(f"    error: {pr.metadata['error']}")
                for inp in pr.missing_required_inputs:
                    print(f"    input: {inp}")
                continue
            if pr.value is None:
                continue

            val_str = f"{pr.value:.4f}" if isinstance(pr.value, float) else str(pr.value)
            role_tag = f"  role={pr.validation_role}" if pr.validation_role else ""
            print(f"  {name:<36} {val_str}{role_tag}")

            # metadata 중 주요 항목만 출력
            meta = pr.metadata
            for key in ('phi_std', 'vol_avg_nm3', 'drift', 'note', 'method'):
                if key in meta:
                    print(f"    {key}: {meta[key]}")

        if targets:
            print("\n  [타겟 비교]")
            _report_targets(results, targets)

        print("=" * 50 + "\n")


def _find_result_for_target(results: dict[str, PropertyResult], target_key: str):
    pr = results.get(target_key)
    if pr is not None:
        return pr, target_key

    for result_key, result in results.items():
        aliases = result.metadata.get('target_aliases', [])
        if target_key in aliases:
            return result, result_key

    return None, None


def _format_status(status: str) -> str:
    return status.replace('_', ' ').upper()


def _report_targets(results: dict, targets: dict):
    """
    targets dict 의 각 property 를 results 의 PropertyResult 와 비교.
    direct_experiment_comparison_allowed=False 인 property 는 비교를 블록한다.
    """
    for target_key, target_spec in targets.items():
        if not target_spec:
            continue

        pr, result_key = _find_result_for_target(results, target_key)

        # property 가 계산되지 않은 경우
        if pr is None:
            print(f"    {target_key:<32} [결과 없음]")
            continue

        mapped = "" if result_key == target_key else f"  -> {result_key}"

        if pr.status != 'computed' or pr.value is None:
            print(f"    {target_key:<32} [SKIP — status={pr.status}]{mapped}")
            continue

        # 비교 불가 gate
        if not pr.direct_experiment_comparison_allowed:
            print(
                f"    {target_key:<32} [비교 불가]  "
                f"validation_role={pr.validation_role}{mapped}"
            )
            if pr.metadata.get('note'):
                print(f"      이유: {pr.metadata['note']}")
            continue

        # 비교 수행
        val = float(pr.value)
        if 'value' in target_spec:
            t = float(target_spec['value'])
            tol = float(target_spec.get('tolerance', 0))
            ok = abs(val - t) <= tol
            print(f"    {target_key:<32} {val:.4f} vs {t} ± {tol}  [{'OK' if ok else 'MISS'}]")
        elif 'min' in target_spec or 'max' in target_spec:
            lo = float(target_spec.get('min', float('-inf')))
            hi = float(target_spec.get('max', float('inf')))
            ok = lo <= val <= hi
            print(f"    {target_key:<32} {val:.4f} in [{lo}, {hi}]  [{'OK' if ok else 'MISS'}]")
