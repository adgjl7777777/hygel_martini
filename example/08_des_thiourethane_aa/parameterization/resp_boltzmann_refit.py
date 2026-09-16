#!/usr/bin/env python3
"""Re-solve the collaborators' RESP fits with Boltzmann conformer weights.

Why this exists
---------------
The shipped multi-conformer RESP fits weight every conformer equally. For most
fragments that is harmless because their conformers sit within ~1.4 kcal/mol of
each other. For F3 (the thiourethane) it is not: its second conformer is
5.17 kcal/mol above the minimum, a Boltzmann population of 1.7e-4 at 300 K, and
it carries **half** the weight of the fit. In that conformer the N-H donor is
intramolecularly buried, so half the fit was constraining a site that no
chloride can reach. The result is q(N) dragged from -0.66 to -0.43, and the
thiourethane-vs-urethane ordering -- the measurement this project exists to
make -- is lost.

Re-solving with weights w_i = exp(-dE_i / RT) restores the ordering using **no
new quantum chemistry**: every input already exists.

What this does NOT do
---------------------
It does not touch anything under the collaborators' directory: every file there
is opened read-only and all output is written inside this package. It does not
claim to fix the absolute interaction depth, which carries polarization and
charge transfer that no fixed-charge model represents.

Weighting convention
--------------------
Weights are normalised so that sum(w_i) = n_conf, which keeps the total amount
of ESP data -- and therefore the restraint/data balance inside the fit -- the
same as the unweighted fit it replaces. A conformer with negligible population
then contributes essentially nothing instead of contributing 1/n.

Usage
-----
    PYTHONPATH=<package> python3 resp_boltzmann_refit.py [--temperature 300]
                                                        [--outdir raw/resp_boltzmann]
                                                        [--fragments F3_thiouret_p ...]

Writes per fragment:  <outdir>/<fragment>.qout          one charge per line
                      <outdir>/<fragment>.provenance.json
and one summary:      <outdir>/refit_summary.json       consumed by the gate test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
sys.path.insert(0, str(EXAMPLE / "validation"))
import ff_pair_benchmark as ffb  # noqa: E402

#: The collaborators' DFT tree. Read-only. Never written to by this script.
DFT_ROOT = Path("/nas_3/active/soohki/27.des/dft")

BOHR = 0.529177210903          # Angstrom per bohr
HARTREE_KCAL = 627.509474
KB_KCAL = 0.0019872041         # kcal/mol/K

#: Fragments, and the chloride complex each one is scored against where one
#: exists. Fragments without a complex are still refitted; they simply get no
#: Cl- score. The ITP is the classical topology used by the pair benchmark.
FRAGMENTS = {
    "F1_thiol":         dict(complex="C1_thiol_Cl",     itp="F1.itp"),
    "F2_urethane_p":    dict(complex="C2_urethane_Cl",  itp="F2.itp"),
    "F3_thiouret_p":    dict(complex="C3_thiouret_Cl",  itp="F3.itp"),
    "F3x_thiouret_ext": dict(complex="C3x_thiouretext_Cl", itp=None),
    "F4_ppg1":          dict(complex=None,              itp=None),
    "F4x_ppg2":         dict(complex=None,              itp=None),
    "F5_acch":          dict(complex=None,              itp=None),
}

#: The N-H axial cone used to score donors, from the ESP-cone analysis.
CONE_HALF_ANGLE_DEG = 25.0
CONE_RMIN_A = 1.9
CONE_RMAX_A = 2.3


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def read_esp_blocks(path: Path):
    """Parse a multi-conformer RESP ESP file into [(atom_xyz_bohr, points)]."""
    lines = [s for s in path.read_text().splitlines() if s.strip()]
    i, blocks = 0, []
    while i < len(lines):
        natoms, npoints = int(lines[i][:5]), int(lines[i][5:10])
        i += 1
        xyz = np.array([list(map(float, s.split())) for s in lines[i:i + natoms]])
        i += natoms
        pts = np.array([list(map(float, s.split())) for s in lines[i:i + npoints]])
        i += npoints
        blocks.append((xyz, pts))
    return blocks


def read_vpot_block(path: Path):
    """Parse a single-conformer .vpot file (used for the held-out conformer)."""
    lines = [s for s in path.read_text().splitlines() if s.strip()]
    natoms, npoints = int(lines[0].split()[0]), int(lines[0].split()[1])
    xyz = np.array([list(map(float, s.split())) for s in lines[1:1 + natoms]])
    pts = np.array([list(map(float, s.split())) for s in lines[1 + natoms:1 + natoms + npoints]])
    return xyz, pts


def read_respin(path: Path, natoms: int):
    """Restraint weight, total charge and the per-atom (element, equiv) spec."""
    text = path.read_text()
    qwt = float(re.search(r"qwt\s*=\s*([\d.]+)", text).group(1))
    body = text.split("&end")[1].strip().splitlines()
    charge, nat = map(int, body[2].split())
    if nat != natoms:
        raise ValueError(f"{path}: respin says {nat} atoms, ESP says {natoms}")
    spec = np.array([list(map(int, s.split())) for s in body[3:3 + natoms]])
    return qwt, charge, spec


def read_conformer_energies(fragment: str):
    """CREST relative conformer energies in kcal/mol, in CREST's own order."""
    path = DFT_ROOT / "05_resp" / fragment / "crest.energies"
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()]
    return [float(r[1]) for r in rows]


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
def design_matrix(block):
    """Coulomb design matrix A (npoints x natoms) and the target potential."""
    xyz, pts = block
    A = 1.0 / np.linalg.norm(pts[:, None, 1:] - xyz[None, :, :], axis=2)
    return A, pts[:, 0]


def solve_resp(designs, spec, total_charge, restraint, initial, n_structures):
    """One RESP stage: hyperbolic restraint on heavy atoms, linear constraints.

    ``n_structures`` scales the restraint so its balance against the data does
    not change when conformers are reweighted.
    """
    n = len(spec)
    rows, rhs = [np.ones(n)], [total_charge]
    for i, (_, flag) in enumerate(spec):
        row = np.zeros(n)
        if flag < 0:                      # frozen at its stage-1 value
            row[i] = 1.0
            rows.append(row); rhs.append(initial[i])
        elif flag > 0:                    # equivalenced to atom flag-1
            row[i] = 1.0; row[flag - 1] = -1.0
            rows.append(row); rhs.append(0.0)
    C, d = np.array(rows), np.array(rhs)
    particular = np.linalg.lstsq(C, d, rcond=None)[0]
    kernel = null_space(C)

    H = sum(A.T @ A for A, _ in designs)
    g = sum(A.T @ y for A, y in designs)
    heavy = (spec[:, 0] != 1).astype(float)

    q = initial.copy()
    for _ in range(20000):
        diag = n_structures * restraint * heavy / np.sqrt(q * q + 0.01)
        M = H + np.diag(diag)
        new = particular + kernel @ np.linalg.solve(kernel.T @ M @ kernel,
                                                   kernel.T @ (g - M @ particular))
        if np.max(np.abs(new - q)) < 1e-11:
            q = new
            break
        q = new
    rrms = float(np.sqrt(sum(np.sum((A @ q - y) ** 2) for A, y in designs)
                         / sum(y @ y for A, y in designs)))
    return q, rrms


def two_stage_fit(fragment: str, blocks, weights=None):
    """Full two-stage RESP, optionally with per-conformer weights."""
    natoms = blocks[0][0].shape[0]
    designs = [design_matrix(b) for b in blocks]
    if weights is not None:
        w = np.asarray(weights, float)
        w = w / w.sum() * len(w)
        designs = [(A * np.sqrt(wi), y * np.sqrt(wi)) for (A, y), wi in zip(designs, w)]
    base = DFT_ROOT / "05_resp" / fragment
    qwt1, charge, spec1 = read_respin(base / "mol.respin1", natoms)
    qwt2, _, spec2 = read_respin(base / "mol.respin2", natoms)
    q1, _ = solve_resp(designs, spec1, charge, qwt1, np.zeros(natoms), len(blocks))
    q2, rrms = solve_resp(designs, spec2, charge, qwt2, q1, len(blocks))
    return q2, rrms


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def rrms_against(q, block):
    A, y = design_matrix(block)
    return float(np.sqrt(np.sum((A @ q - y) ** 2) / np.sum(y * y)))


def axial_cone(block, q, elements):
    """QM and model potential averaged in the cone along the N-H axis.

    Returns (qm, model, n_points) in kcal/mol, or None when the fragment has no
    nitrogen (nothing to score).
    """
    nitrogens = [i for i, e in enumerate(elements) if e == "N"]
    if not nitrogens:
        return None
    xyz_bohr, pts = block
    xyz = xyz_bohr * BOHR
    n_idx = nitrogens[0]
    h_idx = min((i for i, e in enumerate(elements) if e == "H"),
                key=lambda i: np.linalg.norm(xyz[i] - xyz[n_idx]))
    axis = xyz[h_idx] - xyz[n_idx]
    axis /= np.linalg.norm(axis)

    P = pts[:, 1:] * BOHR
    v = P - xyz[h_idx]
    dist = np.linalg.norm(v, axis=1)
    cosine = (v @ axis) / dist
    sel = ((cosine > np.cos(np.radians(CONE_HALF_ANGLE_DEG)))
           & (dist >= CONE_RMIN_A) & (dist < CONE_RMAX_A))
    if not sel.any():
        return None
    R = np.linalg.norm(P[:, None, :] - xyz[None, :, :], axis=2) / BOHR
    qm = -pts[sel, 0].mean() * HARTREE_KCAL
    model = -((1.0 / R) @ q)[sel].mean() * HARTREE_KCAL
    return float(qm), float(model), int(sel.sum())


def chloride_scores(fragment: str, complex_name: str, itp: Path, q):
    """Cl- interaction energy at the DFT geometry and at the model's own minimum."""
    atoms, bonds = ffb.read_itp(str(itp), ffb.read_atomtypes(str(itp)))
    ref_el, ref_xyz = ffb.read_xyz(str(DFT_ROOT / "05_resp" / fragment / "start.xyz"))
    mapping = ffb.match_graphs([a.element for a in atoms], bonds,
                               ref_el, ffb.bonds_from_geometry(ref_el, ref_xyz))
    charged = [replace(a, charge=float(q[mapping[i]])) for i, a in enumerate(atoms)]

    el, xyz = ffb.read_xyz(str(DFT_ROOT / "03_complexes" / complex_name / "opt.xyz"))
    frag_idx, cl_idx = ffb.split_complex(el, xyz)
    frag_el = [el[i] for i in frag_idx]
    frag_xyz = [xyz[i] for i in frag_idx]
    cmap = ffb.match_graphs([a.element for a in atoms], bonds,
                            frag_el, ffb.bonds_from_geometry(frag_el, frag_xyz))
    placed = [frag_xyz[i] for i in cmap]
    h_idx, *_ = ffb.donor_hydrogen(el, xyz, frag_idx, cl_idx)
    chloride = ffb.Atom("Cl", **ffb.CHLORIDE)

    at_dft = ffb.interaction_energy(charged, placed, [chloride], [xyz[cl_idx]])[0] / 4.184
    scan = ffb.scan_along_axis(charged, placed, chloride, xyz[cl_idx], xyz[h_idx],
                               span=3.0, step=0.005)
    r_min, e_min = min(scan, key=lambda t: t[1])
    return float(at_dft), float(r_min), float(e_min / 4.184)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def refit_fragment(fragment: str, temperature: float):
    base = DFT_ROOT / "05_resp" / fragment
    blocks = read_esp_blocks(base / "all.esp")
    elements, _ = ffb.read_xyz(str(base / "start.xyz"))
    natoms = blocks[0][0].shape[0]

    rel = read_conformer_energies(fragment)
    if len(rel) < len(blocks):
        raise ValueError(f"{fragment}: {len(blocks)} ESP blocks but only "
                         f"{len(rel)} CREST energies")
    rel_used = rel[:len(blocks)]
    rt = KB_KCAL * temperature
    weights = [float(np.exp(-e / rt)) for e in rel_used]
    populations = [w / sum(weights) for w in weights]

    shipped_q = np.array(list(map(float, (base / "qout_stage2").read_text().split()))[:natoms])
    refit_q, refit_rrms = two_stage_fit(fragment, blocks, weights=weights)
    control_q, control_rrms = two_stage_fit(fragment, blocks, weights=None)

    record = {
        "fragment": fragment,
        "n_atoms": natoms,
        "n_conformers_in_fit": len(blocks),
        "conformer_rel_energy_kcal": rel_used,
        "boltzmann_population_300K": populations,
        "temperature_K": temperature,
        "effective_conformers": float(1.0 / sum(p * p for p in populations)),
        "rrms_shipped_on_training": rrms_against(shipped_q, blocks[0]),
        "rrms_refit_training": refit_rrms,
        "rrms_unweighted_control": control_rrms,
        "reproduces_shipped": bool(np.max(np.abs(control_q - shipped_q)) < 5e-3),
        "max_abs_dev_control_vs_shipped": float(np.max(np.abs(control_q - shipped_q))),
    }

    # held-out conformer, excluded from every fit by the collaborators
    cross_path = base / "confcross.vpot"
    if cross_path.exists():
        cross = read_vpot_block(cross_path)
        record["rrms_shipped_heldout"] = rrms_against(shipped_q, cross)
        record["rrms_refit_heldout"] = rrms_against(refit_q, cross)
        record["heldout_rrms_change"] = (record["rrms_refit_heldout"]
                                         - record["rrms_shipped_heldout"])

    for label, q in (("shipped", shipped_q), ("refit", refit_q)):
        cone = axial_cone(blocks[0], q, elements)
        if cone is not None:
            qm, model, npts = cone
            record[f"axial_qm_kcal"] = qm
            record[f"axial_{label}_kcal"] = model
            record[f"axial_{label}_error_kcal"] = model - qm
            record["axial_n_points"] = npts

    meta = FRAGMENTS[fragment]
    if meta["complex"] and meta["itp"]:
        itp = EXAMPLE / "validation" / "fragments" / meta["itp"]
        if itp.exists() and (DFT_ROOT / "03_complexes" / meta["complex"] / "opt.xyz").exists():
            for label, q in (("shipped", shipped_q), ("refit", refit_q)):
                at_dft, r_min, e_min = chloride_scores(fragment, meta["complex"], itp, q)
                record[f"cl_at_dft_{label}_kcal"] = at_dft
                record[f"cl_own_min_r_{label}_nm"] = r_min
                record[f"cl_own_min_{label}_kcal"] = e_min

    return record, refit_q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temperature", type=float, default=300.0,
                    help="temperature for the Boltzmann weights, K (default 300)")
    ap.add_argument("--outdir", default="raw/resp_boltzmann",
                    help="output directory, relative to this script")
    ap.add_argument("--fragments", nargs="*", default=sorted(FRAGMENTS),
                    help="fragments to refit (default: all)")
    args = ap.parse_args(argv)

    if not DFT_ROOT.exists():
        print(f"DFT data not reachable at {DFT_ROOT}; nothing to do.", file=sys.stderr)
        return 2

    outdir = (HERE / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temperature_K": args.temperature,
        "dft_root": str(DFT_ROOT),
        "weighting": "w_i = exp(-dE_i/RT), normalised so sum(w) = n_conformers",
        "fragments": {},
    }

    for fragment in args.fragments:
        if fragment not in FRAGMENTS:
            print(f"unknown fragment {fragment!r}", file=sys.stderr)
            return 2
        record, q = refit_fragment(fragment, args.temperature)
        summary["fragments"][fragment] = record

        (outdir / f"{fragment}.qout").write_text(
            "\n".join(f"{v: .6f}" for v in q) + "\n")
        (outdir / f"{fragment}.provenance.json").write_text(
            json.dumps(record, indent=2) + "\n")

        eff = record["effective_conformers"]
        line = (f"{fragment:<18} n_conf={record['n_conformers_in_fit']} "
                f"eff={eff:4.2f}  RRMS {record['rrms_unweighted_control']:.4f} -> "
                f"{record['rrms_refit_training']:.4f}")
        if "axial_refit_kcal" in record:
            line += (f"  | axial QM {record['axial_qm_kcal']:6.2f} "
                     f"shipped {record['axial_shipped_kcal']:6.2f} "
                     f"refit {record['axial_refit_kcal']:6.2f}")
        if "cl_own_min_refit_kcal" in record:
            line += (f"  | Cl- own-min {record['cl_own_min_shipped_kcal']:6.2f} -> "
                     f"{record['cl_own_min_refit_kcal']:6.2f}")
        print(line)
        if not record["reproduces_shipped"]:
            print(f"    NOTE: unweighted control deviates from the shipped charges by "
                  f"{record['max_abs_dev_control_vs_shipped']:.4f} e -- the fit "
                  f"reproduction is not exact for this fragment.")

    # the ordering that the project exists to measure
    f2 = summary["fragments"].get("F2_urethane_p")
    f3 = summary["fragments"].get("F3_thiouret_p")
    if f2 and f3 and "axial_refit_kcal" in f2 and "axial_refit_kcal" in f3:
        summary["ordering"] = {
            "axial_delta_qm": f3["axial_qm_kcal"] - f2["axial_qm_kcal"],
            "axial_delta_shipped": f3["axial_shipped_kcal"] - f2["axial_shipped_kcal"],
            "axial_delta_refit": f3["axial_refit_kcal"] - f2["axial_refit_kcal"],
            "axial_error_match_shipped": abs(f3["axial_shipped_error_kcal"]
                                             - f2["axial_shipped_error_kcal"]),
            "axial_error_match_refit": abs(f3["axial_refit_error_kcal"]
                                           - f2["axial_refit_error_kcal"]),
        }
        if "cl_own_min_refit_kcal" in f2 and "cl_own_min_refit_kcal" in f3:
            summary["ordering"]["cl_own_min_delta_shipped"] = (
                f3["cl_own_min_shipped_kcal"] - f2["cl_own_min_shipped_kcal"])
            summary["ordering"]["cl_own_min_delta_refit"] = (
                f3["cl_own_min_refit_kcal"] - f2["cl_own_min_refit_kcal"])
        print("\nF3 - F2 (negative = thiourethane preferred, the DFT ordering)")
        for k, v in summary["ordering"].items():
            print(f"  {k:<28} {v:+7.2f}")

    (outdir / "refit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {outdir}/refit_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
