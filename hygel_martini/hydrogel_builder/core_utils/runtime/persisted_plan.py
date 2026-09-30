"""Check planned crosslink attachments in a written (unprocessed) builder ITP.

This checks endpoint identity, not force-field validity or physical stability.
The builder writes crosslink bonds unconditionally; GROMACS preprocessing is a
separate check. Internal linker bonds and chemical side branches are excluded
from the attachment comparison.

Why a connectivity audit is not enough: a plan of a-b, c-d written as a-c,
b-d leaves the graph connected with the same component count, and deleting
one bond of a ring leaves it connected too. Only comparing the written stub
to endpoint pairs against the plan catches those. The parser below reads the
file that was written, never the in-memory bond registry, so it cannot agree
with the writer by construction.

Ported verbatim from the Series-01 reliability copy 0.1.1.dev0
(/nas_0/software_backup/hygel_builder_reliability_20260929, commit 50192af;
docs/BUILDER_IMPROVEMENTS_20260930_KO.md section 3). The one-based indices it
expects are what omni's writer emits: ``atom.atom_id + 1`` for atoms and
bonds alike, and per-stub cap deletion happens at layout time, before ids
are assigned, so ids recorded at planning time are the ids in the file.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path


def guard_persisted_plan(itp_path, expected_pairs, report_path):
    """Require each planned stub-to-endpoint bond exactly once; raise on failure.

    ``expected_pairs`` uses one-based ITP indices, with the stub first. The
    independent file parser never reads the mutable in-memory bond registry.
    """
    itp_path, report_path = Path(itp_path), Path(report_path)
    report = {"status": "FAIL", "itp": itp_path.name, "scope": "written crosslink attachments"}
    try:
        if not expected_pairs:
            raise ValueError("Missing persisted crosslink plan")
        pairs = [tuple(pair) for pair in expected_pairs]
        if any(len(pair) != 2 or any(type(i) is not int or i < 1 for i in pair) for pair in pairs):
            raise ValueError("Plan requires positive integer ITP indices")
        stubs = {a for a, _ in pairs}
        endpoints = {b for _, b in pairs}
        if stubs & endpoints or len(endpoints) != len(pairs):
            raise ValueError("Overlapping stub/end identities or reused planned endpoint")
        expected = Counter(tuple(sorted(pair)) for pair in pairs)
        raw = itp_path.read_bytes()
        atoms, bonds, section = [], [], None
        for line in raw.decode().splitlines():
            line = line.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[] ").lower()
                continue
            fields = line.split()
            if section == "atoms":
                atoms.append(int(fields[0]))
            elif section == "bonds":
                a, b = int(fields[0]), int(fields[1])
                if (a in stubs and b in endpoints) or (b in stubs and a in endpoints):
                    bonds.append(tuple(sorted((a, b))))
        if len(atoms) != len(set(atoms)) or not stubs.union(endpoints) <= set(atoms):
            raise ValueError("Missing or duplicate atom identities in written ITP")
        actual = Counter(bonds)
        missing = list((expected - actual).elements())
        extra = list((actual - expected).elements())
        report.update(itp_sha256=hashlib.sha256(raw).hexdigest(),
                      expected_attachments=len(pairs), observed_attachments=len(bonds),
                      missing=missing, unexpected_or_duplicate=extra)
        if missing or extra:
            raise ValueError(f"Written crosslink plan mismatch: missing={missing}, unexpected/duplicate={extra}")
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"Persisted crosslink guard failed: {exc}") from exc
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
