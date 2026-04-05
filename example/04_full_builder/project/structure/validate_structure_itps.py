"""
Quick validator for real_test/structure/*.itp

Checks:
 - indices within atom count per molecule
 - duplicate virtual site IDs
 - self-pairs

Run:
  python real_test/structure/validate_structure_itps.py
"""

import glob
import os
from collections import defaultdict

from hygel_martini.core_utils.io.martini_parser import read_itp_definitions


def main():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.itp")))
    print("Validating:", len(files), "files")
    ok = True
    for path in files:
        defs = read_itp_definitions(path, atom_type_masses=None)
        for name, d in defs.items():
            atom_count = len(d.get("beads", []))
            if atom_count == 0:
                continue
            # constraints/pairs
            for sec in ("constraints", "pairs"):
                for row in d.get(sec, []):
                    i = row.get("i")
                    j = row.get("j")
                    if i is None or j is None:
                        continue
                    if i < 1 or j < 1 or i > atom_count or j > atom_count:
                        print(f"[BAD] {os.path.basename(path)}:{name}:{sec} out-of-range i={i} j={j} atom_count={atom_count}")
                        ok = False
                    if sec == "pairs" and i == j:
                        print(f"[BAD] {os.path.basename(path)}:{name}:{sec} self-pair i=j={i}")
                        ok = False
            # exclusions
            for row in d.get("exclusions", []):
                a = row.get("atom")
                if a < 1 or a > atom_count:
                    print(f"[BAD] {os.path.basename(path)}:{name}:exclusions atom out-of-range {a}")
                    ok = False
            # virtual sites duplicates
            seen = defaultdict(set)
            for vs in d.get("virtual_sites", []):
                sec = vs.get("section", "virtual_sites")
                parts = vs.get("parts") or vs.get("line","").split()
                if not parts:
                    continue
                try:
                    site = int(parts[0])
                except Exception:
                    continue
                if site in seen[sec]:
                    print(f"[BAD] {os.path.basename(path)}:{name}:{sec} duplicate site {site}")
                    ok = False
                seen[sec].add(site)
    print("OK" if ok else "FOUND ISSUES")


if __name__ == "__main__":
    main()
