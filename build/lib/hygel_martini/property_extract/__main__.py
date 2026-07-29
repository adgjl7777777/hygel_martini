import argparse
import json
import os
import sys


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m hygel_martini.property_extract",
        description="hygel_martini 물성 추출 — YAML 설정 기반",
    )

    # Legacy Phase-A command path. Kept for backward compatibility.
    parser.add_argument("--config", required=False, metavar="YAML",
                        help="legacy property_extract.yaml 경로")
    parser.add_argument("--check-gmx", action="store_true",
                        help="GROMACS 실행 가능 여부 확인 후 종료")
    parser.add_argument("--extract-only", action="store_true",
                        help="gmx energy로 xvg 추출만 수행하고 종료")

    subparsers = parser.add_subparsers(dest="command")

    ana = subparsers.add_parser(
        "analyze",
        help="analysis_jobs.yaml을 실행해 property를 추출",
    )
    ana.add_argument("--analysis", required=True, metavar="YAML",
                     help="analysis_jobs.yaml 경로")
    ana.add_argument("--manifest", required=False, metavar="YAML",
                     help="validation_manifest.yaml 경로 (비교 게이트에 사용)")
    ana.add_argument("--requirements", required=False, metavar="YAML",
                     help="md_requirements.yaml 경로")

    req = subparsers.add_parser(
        "requirements",
        help="analysis_jobs.yaml과 md_requirements.yaml의 충족 여부를 확인",
    )
    req.add_argument("--analysis", required=True, metavar="YAML",
                     help="analysis_jobs.yaml 경로")
    req.add_argument("--requirements", required=False, metavar="YAML",
                     help="md_requirements.yaml 경로 (기본: analysis 파일 옆)")
    req.add_argument("--strict", action="store_true",
                     help="미충족 requirement가 있으면 non-zero로 종료")

    man = subparsers.add_parser(
        "manifest",
        help="validation_manifest.yaml에서 simulation property mapping 확인",
    )
    man.add_argument("--manifest", required=True, metavar="YAML",
                     help="validation_manifest.yaml 경로")
    man.add_argument("--property", required=True, metavar="NAME",
                     help="simulation property 이름")

    topology = subparsers.add_parser(
        "topology",
        help="ITP/GRO에서 reduced junction--strand graph를 검증",
    )
    topology.add_argument("--itp", required=True)
    topology.add_argument("--gro")
    topology.add_argument("--junction-residue", default="BCK")
    topology.add_argument("--expected-junctions", type=int)
    topology.add_argument("--expected-strands", type=int)
    topology.add_argument("--expected-winding-rank", type=int)
    topology.add_argument("--output")

    mechanics = subparsers.add_parser(
        "mechanics-step",
        help="aligned baseline/+/- XVG에서 finite-rate paired-step response 계산",
    )
    mechanics.add_argument("--baseline-xvg", required=True)
    mechanics.add_argument("--positive-xvg", required=True)
    mechanics.add_argument("--negative-xvg", required=True)
    mechanics.add_argument("--component", required=True, help="예: Pres-XY")
    mechanics.add_argument("--gamma", required=True, type=float)
    mechanics.add_argument("--window-start-ps", required=True, type=float)
    mechanics.add_argument("--window-end-ps", required=True, type=float)
    mechanics.add_argument("--output")

    clearance = subparsers.add_parser(
        "clearance-frame",
        help="single GRO frame의 periodic local-clearance proxy 계산",
    )
    clearance.add_argument("--gro", required=True)
    clearance.add_argument(
        "--selection-residue",
        action="append",
        dest="selection_residues",
        help="obstacle residue; repeatable (default: PEO, HYDROGEL)",
    )
    clearance.add_argument("--bead-radius-nm", type=float, default=0.24)
    clearance.add_argument("--probe-radius-nm", type=float, default=0.1657)
    clearance.add_argument("--grid-spacing-nm", type=float, default=0.2)
    clearance.add_argument("--bins", type=int, default=50)
    clearance.add_argument("--chunk-size", type=int, default=250_000)
    clearance.add_argument("--output")

    return parser


def _legacy_main(args, parser):

    if args.check_gmx:
        from .gmx_utils import run_gmx
        out = run_gmx(["--version"])
        print(out)
        return

    if not args.config:
        parser.error("--config YAML 이 필요합니다. (--check-gmx 없이 실행 시)")

    import yaml

    print(
        "[warn] --config/property_extract.yaml 경로는 legacy 모드입니다. "
        "논문 target 비교는 validation_manifest.yaml 기반 compare 경로로 이전하세요.",
        file=sys.stderr,
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_dir = os.path.dirname(os.path.abspath(args.config))

    def resolve(p):
        if not p:
            return None
        return p if os.path.isabs(str(p)) else os.path.join(cfg_dir, str(p))

    from .analyzer import HydrogelAnalyzer
    analyzer = HydrogelAnalyzer.from_config(args.config)

    # edr → xvg 추출 (edr 파일이 있을 때만)
    files = cfg.get("files", {})
    edr_file = resolve(files.get("edr"))
    if edr_file and os.path.exists(edr_file):
        xvg_out = resolve(files.get("energy_xvg") or "energy.xvg")
        print(f"[info] gmx energy: {edr_file} → {xvg_out}")
        analyzer.extract_energy_from_edr(edr_file, output_xvg=xvg_out)
    elif edr_file:
        print(f"[warn] edr 파일 없음 (아직 MD 미실행?): {edr_file}", file=sys.stderr)

    if args.extract_only:
        return

    results = analyzer.analyze()
    targets = cfg.get("targets", {})
    if targets:
        print(
            "[warn] property_extract.yaml의 targets 섹션은 deprecated입니다. "
            "validation_manifest.yaml로 옮겨야 합니다.",
            file=sys.stderr,
        )
    analyzer.report(results, targets=targets if targets else None)

    # 오류 있으면 비정상 종료
    if results.get("errors"):
        sys.exit(1)


def _analyze_main(args):
    from .analysis_jobs import run_analysis

    results = run_analysis(
        analysis_path=args.analysis,
        md_requirements_path=getattr(args, "requirements", None),
    )

    manifest_targets = None
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        from .validation_manifest import load_manifest
        manifest_targets = load_manifest(manifest_path)

    print("\n" + "=" * 60)
    print("         Property Analysis Results")
    print("=" * 60)
    print(f"analysis: {os.path.abspath(args.analysis)}")
    print("")

    for job_id, pr in results.items():
        if pr.status == "missing_required_md":
            print(f"  {job_id:<28} [{pr.property}]  [MISSING MD]")
            for inp in pr.missing_required_inputs:
                print(f"    required: {inp}")
        elif pr.status != "computed":
            tag = pr.status.replace("_", " ").upper()
            print(f"  {job_id:<28} [{pr.property}]  [{tag}]")
            if pr.metadata.get("reason"):
                print(f"    reason: {pr.metadata['reason']}")
            if pr.metadata.get("error"):
                print(f"    error: {pr.metadata['error']}")
        else:
            val_str = f"{pr.value:.4f}" if isinstance(pr.value, float) else str(pr.value)
            print(f"  {job_id:<28} {pr.property:<36} {val_str}")
            print(f"    validation_role: {pr.validation_role}  "
                  f"direct_compare: {pr.direct_experiment_comparison_allowed}")
            for key in ("method", "note", "phi_std", "drift"):
                if key in pr.metadata:
                    print(f"    {key}: {pr.metadata[key]}")

    if manifest_targets:
        print("\n  [manifest 비교 게이트]")
        from .validation_manifest import find_target_properties_for_simulation_property
        for job_id, pr in results.items():
            matches = find_target_properties_for_simulation_property(
                manifest_targets, pr.property
            )
            for target, prop, relation in matches:
                comparable_by_manifest = relation in ("same_property", "comparable_to")
                if not comparable_by_manifest:
                    tag = "비교 금지 (방법론 불일치)"
                elif pr.status != "computed" or pr.value is None:
                    tag = f"대기 중 (status={pr.status})"
                elif pr.direct_experiment_comparison_allowed:
                    tag = "비교 가능"
                else:
                    tag = "비교 불가 (validation_role 제한)"
                print(f"    {pr.property:<36} [{tag}]  "
                      f"relation={relation}  target={target.target_id}.{prop.name}")

    print("=" * 60 + "\n")


def _requirements_main(args):
    from .analysis_jobs import load_analysis_jobs
    from .requirements import check_requirements

    jobs, base_dir = load_analysis_jobs(args.analysis)
    req_path = args.requirements or os.path.join(base_dir, "md_requirements.yaml")

    print("\n" + "=" * 60)
    print("         Property Extraction Requirements")
    print("=" * 60)
    print(f"analysis     : {os.path.abspath(args.analysis)}")
    print(f"requirements : {os.path.abspath(req_path)}")
    print("")

    all_satisfied = True
    for job in jobs:
        inputs = job.resolved_inputs(base_dir)
        status = check_requirements(job.property, inputs, req_path, base_dir=base_dir)
        all_satisfied = all_satisfied and status.satisfied
        tag = "OK" if status.satisfied else "MISSING"
        print(f"{job.job_id:<28} {job.property:<36} [{tag}]")
        if job.requires_md_job:
            print(f"  analysis requires_md_job: {job.requires_md_job}")
        if status.required_md_jobs:
            print(f"  required_md_jobs: {', '.join(status.required_md_jobs)}")
        for item in status.missing_required_inputs:
            print(f"  missing: {item}")
        for item in status.missing_md_jobs:
            print(f"  missing_job: {item}")
        for item in status.invalid_inputs:
            print(f"  invalid: {item}")
        if status.notes:
            print(f"  notes: {status.notes}")

    print("=" * 60 + "\n")
    if args.strict and not all_satisfied:
        sys.exit(1)


def _manifest_main(args):
    from .validation_manifest import (
        find_target_properties_for_simulation_property,
        load_manifest,
    )

    targets = load_manifest(args.manifest)
    matches = find_target_properties_for_simulation_property(targets, args.property)
    if not matches:
        print(f"{args.property}: manifest mapping 없음")
        sys.exit(1)

    for target, prop, relation in matches:
        allowed = relation in ("same_property", "comparable_to")
        print(
            f"{args.property} -> {target.target_id}.{prop.name} "
            f"relation={relation} direct_compare_allowed={allowed}"
        )


def _write_json_result(payload, output_path=None):
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_path:
        output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


def _topology_main(args):
    from .extractors.topology import ReducedNetworkTopologyExtractor

    result = ReducedNetworkTopologyExtractor().compute(
        {"itp": args.itp, "gro": args.gro},
        {
            "junction_residue": args.junction_residue,
            "expected_junction_count": args.expected_junctions,
            "expected_strand_count": args.expected_strands,
            "expected_winding_rank": args.expected_winding_rank,
        },
    )
    _write_json_result(result.to_dict(), args.output)
    if result.status != "computed" or result.value is False:
        raise SystemExit(2)


def _mechanics_step_main(args):
    from .mechanics_analysis import paired_step_xvg_summary

    result = paired_step_xvg_summary(
        args.baseline_xvg,
        args.positive_xvg,
        args.negative_xvg,
        component=args.component,
        gamma=args.gamma,
        window_start_ps=args.window_start_ps,
        window_end_ps=args.window_end_ps,
    )
    _write_json_result(result, args.output)


def _clearance_frame_main(args):
    from .extractors.clearance import PeriodicClearanceExtractor

    result = PeriodicClearanceExtractor().compute(
        {"gro": args.gro},
        {
            "selection_residues": args.selection_residues,
            "bead_radius_nm": args.bead_radius_nm,
            "probe_radius_nm": args.probe_radius_nm,
            "grid_spacing_nm": args.grid_spacing_nm,
            "bins": args.bins,
            "chunk_size": args.chunk_size,
        },
    )
    _write_json_result(result.to_dict(), args.output)
    if result.status != "computed":
        raise SystemExit(2)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "analyze":
            _analyze_main(args)
            return
        if args.command == "requirements":
            _requirements_main(args)
            return
        if args.command == "manifest":
            _manifest_main(args)
            return
        if args.command == "topology":
            _topology_main(args)
            return
        if args.command == "mechanics-step":
            _mechanics_step_main(args)
            return
        if args.command == "clearance-frame":
            _clearance_frame_main(args)
            return

        # Backward-compatible no-subcommand mode.
        _legacy_main(args, parser)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
