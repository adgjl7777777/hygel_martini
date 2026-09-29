"""Render a routing counterexample using actual HyGel assignments.

This is an illustrative endpoint-routing fixture, not a material structure or
a comparison with another software package. Run with an installed HyGel wheel.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def route(explicit):
    from hygel_martini.hydrogel_builder.core_utils.runtime.dynamic_crosslink import plan_dynamic_crosslinks
    x = {'a': 0., 'b': 9., 'c': 1., 'd': 10.}
    ends = {i: [SimpleNamespace(atom_id=i, position=(x[n], 0., 0.),
             chain_index=i, planned_endpoint_id=n, backbone_type=None)]
            for i, n in enumerate('abcd')}
    stubs = [SimpleNamespace(atom_id=10+i, position=(p, 0., 0.),
             planned_endpoint_edges=(('a', 'b'), ('c', 'd')) if explicit else None,
             stub_type='backbone_'+str(i+1), target_backbone=None,
             linker_chain_index=7, backbone_type=None) for i, p in enumerate((.2, 9.8))]
    assignments, notes = plan_dynamic_crosslinks({7: stubs}, ends, None,
        candidate_limit=64, targets_per_stub=2)
    groups = {}
    for a in assignments[7]:
        groups.setdefault(a.stub_atom.atom_id, []).append(a.backbone_atom.planned_endpoint_id)
    return {'pairs': sorted(sorted(v) for v in groups.values()), 'notes': notes,
            'total_stub_endpoint_distance': sum(a.distance for a in assignments[7])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output/'.mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    import hygel_martini
    requested = [['a', 'b'], ['c', 'd']]
    legacy, planned = route(False), route(True)
    if planned['pairs'] != requested or legacy['pairs'] == requested:
        raise RuntimeError('Routing fixture no longer exhibits the expected distinction')
    data = {'scope': 'illustrative routing fixture, not a physical benchmark',
            'package_version': hygel_martini.__version__, 'requested': requested,
            'geometry_only': legacy, 'explicit_plan': planned}
    (args.output/'endpoint_identity.json').write_text(json.dumps(data, indent=2)+'\n')
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8))
    coords = {'a': 0., 'c': 1., 'b': 9., 'd': 10.}
    cases = [('A  Requested pairing', requested, '#236a9b', 'a–b  /  c–d'),
             ('B  Geometry-only routing', legacy['pairs'], '#b65c22', 'a–c  /  b–d'),
             ('C  Explicit-plan routing', planned['pairs'], '#236a9b', 'a–b  /  c–d')]
    for ax, (title, pairs, color, label) in zip(axes, cases):
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=15)
        for index, (a, b) in enumerate(pairs):
            ax.add_patch(FancyArrowPatch((coords[a], 0), (coords[b], 0),
                connectionstyle='arc3,rad='+str(.45 if index == 0 else -.45),
                arrowstyle='-', color=color, linewidth=2.3))
        for name, x in coords.items():
            ax.scatter([x], [0], s=220, c='white', edgecolors='#20313f', zorder=3, linewidths=1.4)
            ax.text(x, 0, name, ha='center', va='center', fontweight='bold', fontsize=10, zorder=4)
        ax.text(5, -2.55, label, ha='center', color=color, fontsize=12)
        ax.set_xlim(-1, 11); ax.set_ylim(-3.2, 3.2); ax.set_aspect('equal'); ax.axis('off')
    fig.text(.5, .09, 'The same endpoint positions can support different pairings.\n'
             'Explicit-plan routing retains the requested identity.', ha='center', fontsize=11)
    fig.subplots_adjust(left=.025, right=.99, top=.84, bottom=.23, wspace=.15)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(args.output/('endpoint_identity.'+ext), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
