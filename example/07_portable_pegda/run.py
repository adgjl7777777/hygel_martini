#!/usr/bin/env python3
"""Small dry PEGDA build from installed HyGel plus separately supplied Martini.

No minimization, packing, production MD, or property validation is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--martini-dir', type=Path, required=True)
    parser.add_argument('--gmx', default='gmx')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are refused')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    ff = args.martini_dir.resolve() / 'martini_v3.0.0.itp'
    gmx = shutil.which(args.gmx)
    if not ff.is_file() or not gmx:
        parser.error('Supply martini_v3.0.0.itp and an executable GROMACS command')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = '1'
    env['MPLCONFIGDIR'] = str(root/'cache/matplotlib')
    env['NUMBA_CACHE_DIR'] = str(root/'cache/numba')
    env['TMPDIR'] = str(root/'tmp')
    (root/'tmp').mkdir()
    result = {'status': 'FAIL', 'scope': 'dry PEGDA construction and preprocessing; no MD',
              'commands': [], 'forcefield': {'name': ff.name, 'sha256': sha(ff)}}
    start = time.monotonic()
    def command(cmd, log):
        result['commands'].append(cmd)
        with (root/log).open('w') as handle:
            completed = subprocess.run(cmd, cwd=root, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, timeout=args.timeout)
        if completed.returncode:
            raise RuntimeError(f'Command failed ({completed.returncode}); see {log}')
    try:
        import yaml
        import hygel_martini
        from hygel_martini.tools.audit_hydrogel_topology import parse_itp, connected_components
        from hygel_martini.property_extract.network_topology import audit_reduced_network
        result['package_version'] = hygel_martini.__version__
        result['package_location'] = str(Path(hygel_martini.__file__).resolve())
        here = Path(__file__).resolve().parent
        inputs = root/'inputs'
        inputs.mkdir()
        shutil.copytree(here/'structure', inputs/'structure')
        shutil.copy2(here/'bonded_parameters.yaml', inputs/'bonded_parameters.yaml')
        cfg = yaml.safe_load((here/'maker_template.yaml').read_text())
        cfg['simulation_parameters'].update(gromacs_executable_path=gmx,
            gromacs_include_path=str(ff.parent), base_itp_file=str(ff), output_dir=str(root/'build'))
        maker = inputs/'maker.yaml'
        maker.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command([gmx, '--version'], 'gromacs_version.txt')
        command([sys.executable, '-m', 'hygel_martini.hydrogel_builder', str(maker)], 'builder.log')
        out = root/'build'
        gro, itp = out/'initial_hydrogel.gro', out/'initial_hydrogel.itp'
        plan = json.loads((out/'planned_crosslinks.json').read_text())
        guards = [json.loads((out/(name+'.itp.plan_audit.json')).read_text())
                  for name in ['initial_backbone', 'initial_hydrogel']]
        if not all(r['status'] == 'PASS' for r in guards):
            raise RuntimeError('Written-plan guards did not pass')
        atoms, bonds, _, _ = parse_itp(itp)
        components, _ = connected_components(atoms, bonds)
        # Analytic expectations: 16 strands of 10 beads and 16 linker beads;
        # 144 internal strand bonds, 32 attachments and 8 internal linker bonds.
        observed = {'atoms': len(atoms), 'bonds': len(bonds), 'components': len(components),
                    'planned_attachments': len(plan['pairs'])}
        expected = {'atoms': 176, 'bonds': 184, 'components': 1, 'planned_attachments': 32}
        result.update(observed=observed, expected=expected)
        if observed != expected:
            raise RuntimeError(f'Unexpected build counts: {observed}')
        audit = audit_reduced_network(itp, gro)
        (root/'network_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        # A preprocessing check of the actual saved dry model, at maxwarn=0.
        top = root/'smoke.top'
        top.write_text(f'#include "{ff}"\n#include "{itp}"\n\n[ system ]\nDry PEGDA smoke\n'
                       '\n[ molecules ]\nHYDROGEL 1\n')
        mdp = root/'smoke.mdp'
        mdp.write_text('integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n'
                       'coulombtype = Cut-off\nrcoulomb = 1.1\nvdwtype = Cut-off\n'
                       'rvdw = 1.1\npbc = xyz\nnstlist = 20\n')
        command([gmx, 'grompp', '-f', str(mdp), '-c', str(gro), '-p', str(top),
                 '-o', str(root/'smoke.tpr'), '-maxwarn', '0'], 'grompp.log')
        if not (root/'smoke.tpr').is_file():
            raise RuntimeError('GROMACS did not write the preprocessed system')
        result['artifacts'] = {str(p.relative_to(root)): sha(p) for p in
            [gro, itp, out/'planned_crosslinks.json', root/'network_audit.json', root/'smoke.tpr', maker]}
        result['status'] = 'PASS'
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        (root/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
