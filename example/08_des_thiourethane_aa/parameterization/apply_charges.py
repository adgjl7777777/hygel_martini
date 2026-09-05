#!/usr/bin/env python3
"""Put a charge set into an ITP without touching anything else, and prove it.

Charges are the one part of this example's force field that is expected to
change: the LigParGen 1.14*CM1A-LBCC values validate construction, and the DFT
side is preparing RESP candidates (integrated report §18B.2). Swapping them by
hand across 93-, 73- and 373-atom molecules is how a topology quietly ends up
with one wrong atom, so this does it mechanically and leaves a record.

Four jobs, one file:

``neutralize``
    Keep the charges, fix the sum. Every LigParGen ITP here misses zero by
    -1.0e-4 e (four-decimal tables), and that adds up linearly with cell size:
    -0.0256 e for the n=3 full network, -0.0096 e for the count:32 cell. PME
    absorbs it into a uniform background, grompp warns every time, and the
    report (§18C.4) asks for exactly zero per neutral molecule instead.
``apply``
    Replace charges from a release CSV (``molecule,atom_name,charge``), then
    neutralize the remainder the same way.
``convert``
    Turn a DFT ``.pc_resp`` (ORCA point charges, one per line in the QM atom
    order) plus a ``dft_index,atom_name`` mapping into that release CSV --
    the mapping is the DFT team's deliverable (§18B.2 item 5), and without it
    nothing here will guess which QM atom is which ITP atom.
``scale``
    Multiply every charge of an ion by a factor and write a variant file --
    the ion-only charge-scaling candidates (1.0 / 0.8 / 0.69) the report wants
    frozen as reproducible files (§18C.2), with the neutral network untouched.

What "prove it" means here:

* every line outside ``[ atoms ]`` is written back byte-for-byte, and every
  ``[ atoms ]`` row keeps all fields but the charge;
* the written charges sum to the target integer **exactly at the written
  precision** (six decimals), not "to within rounding" -- the residual after
  rounding lands on one named anchor atom so the sum is checkable by adding
  up the file;
* a provenance block at the top of the ITP and a JSON sidecar record the
  source file hashes, the rule, the before/after totals, every atom whose
  charge moved and by how much;
* a release whose own total is not the target integer (a fragment fitted at
  the wrong charge, a mapping that dropped an atom) is refused, not repaired.

The junction has one coupling this tool has to honour: a reacted arm's
sulfur override in ``config/hydrogel.yaml`` is ``q(S) + q(cap H)`` so a
junction stays neutral at any conversion (``build_templates.py``). Change the
junction's charges and those overrides are stale. ``--stub-caps`` recomputes
them from the new charges and rewrites the YAML in place, reporting each one.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Itp", "neutralize", "apply_release", "scale_charges", "convert_resp"]

#: Charges are written with this many decimals; exact-sum bookkeeping is done
#: in Decimal at this precision so the file itself adds up.
PRECISION = 6
QUANTUM = Decimal(1).scaleb(-PRECISION)


def _sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class Itp:
    """One ITP file, held as lines, with its ``[ atoms ]`` rows indexed.

    Only the atom rows are ever interpreted; every other line is opaque text
    and is written back exactly. That is deliberate: a charge tool that
    re-serializes bonded sections is a charge tool that can corrupt them.
    """

    def __init__(self, path: str):
        self.path = path
        with open(path) as handle:
            self.lines: List[str] = handle.read().split("\n")
        self.atom_rows: List[int] = []          # line indices of atom rows
        self.names: List[str] = []
        self.masses: List[float] = []
        self.moleculetype: Optional[str] = None
        section = None
        pending_name = False
        for index, line in enumerate(self.lines):
            text = line.split(";", 1)[0].strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("["):
                section = text.strip("[] ").strip().lower()
                pending_name = section == "moleculetype"
                continue
            if pending_name:
                self.moleculetype = text.split()[0]
                pending_name = False
                continue
            if section == "atoms":
                parts = text.split()
                if len(parts) < 7:
                    raise ValueError(f"{path}:{index + 1}: [ atoms ] row too short: {line!r}")
                self.atom_rows.append(index)
                self.names.append(parts[4])
                self.masses.append(float(parts[7]) if len(parts) >= 8 else 0.0)
        if not self.atom_rows:
            raise ValueError(f"{path}: no [ atoms ] rows")
        if len(set(self.names)) != len(self.names):
            dupes = sorted({n for n in self.names if self.names.count(n) > 1})
            raise ValueError(f"{path}: duplicate atom names {dupes}; a name-keyed release "
                             "cannot address them")

    def charges(self) -> List[Decimal]:
        return [Decimal(self.lines[i].split(";", 1)[0].split()[6]) for i in self.atom_rows]

    def set_charges(self, values: Sequence[Decimal]) -> None:
        """Rewrite the charge column only, keeping the row's other fields."""
        if len(values) != len(self.atom_rows):
            raise ValueError("charge count does not match atom count")
        for index, value in zip(self.atom_rows, values):
            line = self.lines[index]
            body, _, comment = line.partition(";")
            parts = body.split()
            parts[6] = f"{value:.{PRECISION}f}"
            mass = f"  {parts[7]}" if len(parts) >= 8 else ""
            rebuilt = (f"{int(parts[0]):6d}  {parts[1]:<8s} {int(parts[2]):4d}  {parts[3]:<5s} "
                       f"{parts[4]:<6s} {int(parts[5]):4d} {parts[6]:>10s}{mass}")
            self.lines[index] = rebuilt + (f" ;{comment}" if comment else "")

    def write(self, out_path: str, provenance: Sequence[str]) -> None:
        """Write with a ``;`` provenance block ahead of the first section."""
        first = next(i for i, l in enumerate(self.lines) if l.strip().startswith("["))
        header = [f"; {line}" if line else ";" for line in provenance]
        with open(out_path, "w") as handle:
            handle.write("\n".join(self.lines[:first] + header + self.lines[first:]))


def _rule_indices(itp: Itp, rule: str) -> List[int]:
    """Which atoms absorb a neutralization residual.

    ``uniform`` spreads it over every atom, ``heavy`` over non-hydrogens (so a
    zero-LJ polar hydrogen's charge is never what gets adjusted), ``atom:NAME``
    puts all of it on one atom -- what ``build_templates.py`` did with the
    strand's cap charge, onto the carbonyl carbon.
    """
    if rule == "uniform":
        return list(range(len(itp.names)))
    if rule == "heavy":
        heavy = [i for i, m in enumerate(itp.masses) if m > 1.5]
        if not heavy:
            raise ValueError("rule 'heavy' but the ITP has no atom heavier than 1.5 amu")
        return heavy
    if rule.startswith("atom:"):
        name = rule[5:]
        if name not in itp.names:
            raise ValueError(f"rule {rule!r}: no atom named {name!r} in {itp.path}")
        return [itp.names.index(name)]
    raise ValueError(f"unknown neutralization rule {rule!r}; use uniform, heavy or atom:NAME")


def neutralize(charges: Sequence[Decimal], indices: Sequence[int],
               target: int) -> Tuple[List[Decimal], Dict[int, Decimal]]:
    """Adjust ``charges[indices]`` so the sum is exactly ``target`` at PRECISION.

    The residual is split evenly over the chosen atoms and quantized; whatever
    is left after quantization (at most one quantum per atom) goes onto the
    first chosen atom, so the final sum is exact -- verified before returning.
    Returns the new charges and ``{atom_index: delta}`` for the record.
    """
    values = [Decimal(q).quantize(QUANTUM) for q in charges]
    residual = Decimal(target) - sum(values)
    if residual == 0:
        return values, {}
    share = (residual / len(indices)).quantize(QUANTUM)
    deltas: Dict[int, Decimal] = {}
    for i in indices:
        values[i] += share
        deltas[i] = share
    leftover = Decimal(target) - sum(values)
    if leftover != 0:
        anchor = indices[0]
        values[anchor] += leftover
        deltas[anchor] = deltas.get(anchor, Decimal(0)) + leftover
    assert sum(values) == Decimal(target), "neutralization did not close"
    return values, {k: v for k, v in deltas.items() if v != 0}


def _provenance(itp: Itp, action: str, rule: str, target: int, before: Sequence[Decimal],
                after: Sequence[Decimal], deltas: Dict[int, Decimal],
                extra: Sequence[str] = ()) -> Tuple[List[str], dict]:
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    max_change = max((abs(b - a) for a, b in zip(before, after)), default=Decimal(0))
    stamp = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        f"apply_charges.py {action} -- {stamp}",
        f"source itp: {os.path.basename(itp.path)} sha256 {_sha256(itp.path)}",
        *extra,
        f"target total {target:+d} e; before {sum(before):+.{PRECISION}f}, after "
        f"{sum(after):+.{PRECISION}f} (exact at {PRECISION} decimals)",
        f"neutralization rule {rule}: {len(deltas)} atom(s) adjusted, "
        f"max |dq| over all atoms {max_change:.{PRECISION}f} e",
    ]
    if deltas:
        # A uniform spread over dozens of atoms is one fact, not dozens; list
        # atoms individually only when they differ.
        distinct = sorted(set(deltas.values()))
        if len(deltas) > 8 and len(distinct) <= 2:
            common = max(distinct, key=lambda v: sum(1 for d in deltas.values() if d == v))
            odd = {i: d for i, d in deltas.items() if d != common}
            text = f"{len(deltas)} atom(s) {common:+.{PRECISION}f} each"
            if odd:
                text += "; " + ", ".join(f"{itp.names[i]} {d:+.{PRECISION}f}"
                                         for i, d in sorted(odd.items()))
            lines.append("adjustments: " + text)
        else:
            lines.append("adjustments: " + ", ".join(
                f"{itp.names[i]} {d:+.{PRECISION}f}" for i, d in sorted(deltas.items())))
    record = {
        "action": action, "timestamp": stamp, "source_itp": itp.path,
        "source_sha256": _sha256(itp.path), "moleculetype": itp.moleculetype,
        "target_total": target, "total_before": str(sum(before)),
        "total_after": str(sum(after)), "rule": rule,
        "changed_atoms": [{"name": itp.names[i], "before": str(before[i]),
                           "after": str(after[i])} for i in changed],
        "neutralization": {itp.names[i]: str(d) for i, d in deltas.items()},
        "extra": list(extra),
    }
    return lines, record


def _finish(itp: Itp, out: str, lines: List[str], record: dict) -> None:
    itp.write(out, lines)
    # A variant written under a new stem needs its coordinates next to it,
    # because the makers address a species as <stem>.itp + <stem>.gro.
    src_gro = os.path.splitext(itp.path)[0] + ".gro"
    dst_gro = os.path.splitext(out)[0] + ".gro"
    if os.path.abspath(src_gro) != os.path.abspath(dst_gro) and os.path.exists(src_gro) \
            and not os.path.exists(dst_gro):
        import shutil
        shutil.copy2(src_gro, dst_gro)
        record["gro_copied_from"] = src_gro
    with open(out + ".charges.json", "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {out}")
    print(f"      {record['moleculetype']}: total {record['total_before']} -> "
          f"{record['total_after']} e, {len(record['changed_atoms'])} atom(s) changed")


def cmd_neutralize(itp: Itp, out: str, rule: str, target: int) -> dict:
    before = itp.charges()
    after, deltas = neutralize(before, _rule_indices(itp, rule), target)
    itp.set_charges(after)
    lines, record = _provenance(itp, "neutralize", rule, target, before, after, deltas)
    _finish(itp, out, lines, record)
    return record


def read_release(path: str, molecule: str) -> Dict[str, Decimal]:
    """``molecule,atom_name,charge`` rows for one molecule; duplicates refused."""
    out: Dict[str, Decimal] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["molecule"].strip() != molecule:
                continue
            name = row["atom_name"].strip()
            if name in out:
                raise ValueError(f"{path}: atom {name!r} listed twice for {molecule}")
            out[name] = Decimal(row["charge"].strip())
    if not out:
        raise ValueError(f"{path}: no rows for molecule {molecule!r}")
    return out


def apply_release(itp: Itp, release: Dict[str, Decimal], target: int,
                  tolerance: Decimal = Decimal("0.01")) -> Tuple[List[Decimal], List[Decimal]]:
    """Substitute charges by atom name; the release must cover every atom.

    A partial release is refused rather than merged with the old charges: a
    molecule half on RESP and half on LBCC is exactly the inconsistency the
    report warns against (§18B.2 item 8). So is a release whose own sum is
    not the target integer within ``tolerance`` -- that is a fitting or
    mapping error upstream, and neutralizing it away would hide it.
    """
    missing = sorted(set(itp.names) - set(release))
    unknown = sorted(set(release) - set(itp.names))
    if missing or unknown:
        raise ValueError(
            f"release does not match {itp.path}: "
            f"{len(missing)} atom(s) without a charge {missing[:6]}{'...' if len(missing) > 6 else ''}; "
            f"{len(unknown)} name(s) not in the ITP {unknown[:6]}{'...' if len(unknown) > 6 else ''}")
    before = itp.charges()
    proposed = [release[n] for n in itp.names]
    total = sum(proposed)
    if abs(total - Decimal(target)) > tolerance:
        raise ValueError(
            f"release sums to {total:+.6f} e but the molecule's charge is {target:+d}; "
            f"that is a fitting or mapping error, not a rounding one, and will not be "
            f"neutralized away (tolerance {tolerance})")
    return before, proposed


def cmd_apply(itp: Itp, out: str, rule: str, target: int, release_path: str) -> dict:
    release = read_release(release_path, itp.moleculetype)
    before, proposed = apply_release(itp, release, target)
    after, deltas = neutralize(proposed, _rule_indices(itp, rule), target)
    itp.set_charges(after)
    lines, record = _provenance(itp, "apply", rule, target, before, after, deltas,
                                extra=[f"release: {os.path.basename(release_path)} "
                                       f"sha256 {_sha256(release_path)}"])
    _finish(itp, out, lines, record)
    return record


def scale_charges(charges: Sequence[Decimal], factor: Decimal) -> List[Decimal]:
    return [(q * factor).quantize(QUANTUM) for q in charges]


def rename_moleculetype(itp: Itp, new_name: str) -> str:
    """Rename the ``[ moleculetype ]`` in place; returns the old name.

    The builder auto-includes every ITP under its include path and refuses two
    files declaring the same moleculetype (a guard worth keeping: the later
    one would silently win). So a variant of ``ACC`` cannot also be called
    ``ACC`` if it is to live beside the original; it becomes ``ACC_f080`` and
    is addressed by that name in makers and recipes. Residue names inside the
    molecule are untouched, so analyses keyed on ``resname ACC`` still work.
    """
    section = None
    pending = False
    for index, line in enumerate(itp.lines):
        text = line.split(";", 1)[0].strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("["):
            section = text.strip("[] ").strip().lower()
            pending = section == "moleculetype"
            continue
        if pending:
            parts = line.split(";", 1)
            fields = parts[0].split()
            old = fields[0]
            fields[0] = new_name
            itp.lines[index] = "  " + "  ".join(fields) + (f" ;{parts[1]}" if len(parts) > 1 else "")
            itp.moleculetype = new_name
            return old
    raise ValueError(f"{itp.path}: no [ moleculetype ] section")


def cmd_scale(itp: Itp, out: str, factor: str, total: int, rule: str,
              moleculetype: Optional[str] = None) -> dict:
    """Scale an ion's charges; the sum becomes ``total * factor`` exactly.

    Scaling is applied to ions only (report §18C.2/§18C.4); for a neutral
    molecule ``total`` is 0 and scaling is meaningless, so it is refused. The
    variant is written under a new moleculetype name (default
    ``<name>_fNNN``) so it can sit beside the original -- see
    ``rename_moleculetype``.
    """
    if total == 0:
        raise ValueError("scale is for ions; a neutral molecule has nothing to scale")
    f = Decimal(factor)
    new_name = moleculetype or f"{itp.moleculetype}_f{int(round(float(f) * 100)):03d}"
    old_name = rename_moleculetype(itp, new_name)
    before = itp.charges()
    scaled = scale_charges(before, f)
    # Exact target after scaling: total*f, quantized -- e.g. -0.8 for Cl-.
    exact = (Decimal(total) * f).quantize(QUANTUM)
    residual = exact - sum(scaled)
    deltas: Dict[int, Decimal] = {}
    if residual != 0:
        indices = _rule_indices(itp, rule)
        share = (residual / len(indices)).quantize(QUANTUM)
        for i in indices:
            scaled[i] += share
            deltas[i] = share
        leftover = exact - sum(scaled)
        if leftover != 0:
            scaled[indices[0]] += leftover
            deltas[indices[0]] = deltas.get(indices[0], Decimal(0)) + leftover
    assert sum(scaled) == exact
    itp.set_charges(scaled)
    lines, record = _provenance(itp, f"scale x{f}", rule, 0, before, scaled, deltas,
                                extra=[f"ion charge scaling factor {f}: formal {total:+d} e -> "
                                       f"{exact:+.{PRECISION}f} e",
                                       f"moleculetype {old_name} -> {new_name} (a variant must "
                                       "not share the original's name in the include path)"])
    record["moleculetype_from"] = old_name
    # _provenance was told target 0 (its integer slot); the ion's real target
    # is the scaled value, recorded here and in the extra line above.
    record["target_total"] = str(exact)
    _finish(itp, out, lines, record)
    return record


def convert_resp(pc_resp: str, mapping_csv: str, molecule: str, out_csv: str) -> int:
    """``.pc_resp`` + ``dft_index,atom_name`` mapping -> release CSV rows.

    The ``.pc_resp`` format is ORCA's: first line atom count, second a comment,
    then ``Q x y z charge`` per atom in QM order. Every QM atom must be mapped
    to exactly one name and vice versa; a mapping that is off by one atom is
    the classic way to put a hydrogen's charge on a carbon, so both directions
    are checked and the total is reported for the caller to compare with the
    molecule's formal charge.
    """
    with open(pc_resp) as handle:
        raw = handle.read().split("\n")
    count = int(raw[0].split()[0])
    rows = [l.split() for l in raw[2:2 + count]]
    if len(rows) != count or any(len(r) < 5 for r in rows):
        raise ValueError(f"{pc_resp}: expected {count} 'Q x y z q' rows")
    charges = [Decimal(r[4]) for r in rows]

    mapping: Dict[int, str] = {}
    with open(mapping_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["dft_index"])
            if idx in mapping:
                raise ValueError(f"{mapping_csv}: dft_index {idx} mapped twice")
            mapping[idx] = row["atom_name"].strip()
    expected = set(range(1, count + 1))
    if set(mapping) != expected:
        raise ValueError(f"{mapping_csv}: must map dft_index 1..{count} exactly once each; "
                         f"missing {sorted(expected - set(mapping))}, "
                         f"extra {sorted(set(mapping) - expected)}")
    names = list(mapping.values())
    if len(set(names)) != len(names):
        raise ValueError(f"{mapping_csv}: two QM atoms map to the same atom name")

    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule", "atom_name", "charge"])
        for idx in range(1, count + 1):
            writer.writerow([molecule, mapping[idx], f"{charges[idx - 1]:.6f}"])
    print(f"wrote {out_csv}: {count} atoms for {molecule}, total {sum(charges):+.6f} e "
          f"(source {os.path.basename(pc_resp)} sha256 {_sha256(pc_resp)[:12]}...)")
    return count


STUB_CAP_RE = re.compile(
    r"(\{cap_atoms: \[(?P<cap>\w+)\], reacted: \{(?P<sulfur>\w+): \{charge: )"
    r"(?P<q>-?\d+\.\d+)(\, type: \w+\}\}\})")


def refresh_stub_caps(yaml_path: str, itp: Itp) -> List[Tuple[str, str, str, str]]:
    """Recompute ``reacted`` sulfur charges as q(S)+q(cap H) from ``itp``.

    Rewrites the matching lines of the YAML in place and returns
    ``(sulfur, cap, old, new)`` per arm. The file is edited textually so its
    comments survive; only the number inside each ``reacted:`` mapping moves.
    """
    charges = dict(zip(itp.names, itp.charges()))
    with open(yaml_path) as handle:
        text = handle.read()
    changes: List[Tuple[str, str, str, str]] = []

    def repl(match: "re.Match[str]") -> str:
        cap, sulfur = match.group("cap"), match.group("sulfur")
        if cap not in charges or sulfur not in charges:
            raise ValueError(f"{yaml_path}: stub cap {cap}/{sulfur} not in {itp.path}")
        new = (charges[sulfur] + charges[cap]).quantize(QUANTUM)
        changes.append((sulfur, cap, match.group("q"), f"{new:.{PRECISION}f}"))
        return f"{match.group(1)}{new:.{PRECISION}f}{match.group(5)}"

    new_text, n = STUB_CAP_RE.subn(repl, text)
    if n == 0:
        raise ValueError(f"{yaml_path}: no 'stub_caps' reacted entries found to refresh")
    with open(yaml_path, "w") as handle:
        handle.write(new_text)
    for sulfur, cap, old, new in changes:
        print(f"      stub cap {sulfur}+{cap}: reacted charge {old} -> {new}")
    return changes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("itp", help="input ITP")
        p.add_argument("--out", help="output ITP (default: overwrite the input)")
        p.add_argument("--rule", default="uniform",
                       help="uniform | heavy | atom:NAME (default uniform)")
        p.add_argument("--total", type=int, default=0,
                       help="the molecule's formal charge (default 0; +1 for ACC, -1 for CL)")
        p.add_argument("--stub-caps", metavar="HYDROGEL_YAML",
                       help="also refresh the reacted-sulfur overrides in this YAML "
                            "(junction ITP only)")

    n = sub.add_parser("neutralize", help="keep charges, make the sum exact")
    common(n)
    a = sub.add_parser("apply", help="replace charges from a release CSV")
    common(a)
    a.add_argument("--release", required=True, help="molecule,atom_name,charge CSV")
    s = sub.add_parser("scale", help="ion-only charge scaling variant")
    common(s)
    s.add_argument("--factor", required=True, help="e.g. 0.8")
    s.add_argument("--moleculetype", default=None,
                   help="name for the variant (default <name>_fNNN)")
    c = sub.add_parser("convert", help="pc_resp + mapping -> release CSV")
    c.add_argument("--pc-resp", required=True)
    c.add_argument("--mapping", required=True, help="dft_index,atom_name CSV")
    c.add_argument("--molecule", required=True, help="moleculetype name (HEXU, STR, ACC...)")
    c.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "convert":
        convert_resp(args.pc_resp, args.mapping, args.molecule, args.out)
        return 0

    itp = Itp(args.itp)
    out = args.out or args.itp
    if args.command == "neutralize":
        cmd_neutralize(itp, out, args.rule, args.total)
    elif args.command == "apply":
        cmd_apply(itp, out, args.rule, args.total, args.release)
    elif args.command == "scale":
        cmd_scale(itp, out, args.factor, args.total, args.rule, args.moleculetype)
    if args.stub_caps:
        refresh_stub_caps(args.stub_caps, Itp(out))
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
