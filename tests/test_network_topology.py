import tempfile
import unittest
from pathlib import Path

from hygel_martini.property_extract.network_topology import audit_reduced_network


class ReducedNetworkTopologyTests(unittest.TestCase):
    def test_parallel_strands_and_periodic_winding(self):
        itp_text = """\
[ moleculetype ]
TOY 1

[ atoms ]
1 T 1 BCK J1 1 0 1
2 T 1 BCK J2 1 0 1
3 T 2 BCK J1 2 0 1
4 T 2 BCK J2 2 0 1
5 T 3 PEO E1 3 0 1
6 T 3 PEO E2 3 0 1
7 T 4 PEO E1 4 0 1
8 T 4 PEO E2 4 0 1

[ bonds ]
1 2 1
3 4 1
1 5 1
5 6 1
6 3 1
2 7 1
7 8 1
8 4 1
"""
        coordinates = [
            (1.0, 1.0, 1.0),
            (1.1, 1.0, 1.0),
            (4.0, 1.0, 1.0),
            (4.1, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            (3.0, 1.0, 1.0),
            (0.2, 1.0, 1.0),
            (9.0, 1.0, 1.0),
        ]
        gro_lines = ["toy", str(len(coordinates))]
        for index, (x, y, z) in enumerate(coordinates, start=1):
            # Deliberately repeat the fixed-width serial field.  Large GRO
            # files wrap this field after 99,999 atoms, so topology mapping
            # must follow coordinate-line order rather than serial identity.
            wrapped_serial = 1
            gro_lines.append(
                f"{index:5d}{'TOY':<5}{'B':>5}{wrapped_serial:5d}"
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
            )
        gro_lines.append("  10.00000  10.00000  10.00000")

        with tempfile.TemporaryDirectory() as directory:
            itp = Path(directory) / "toy.itp"
            gro = Path(directory) / "toy.gro"
            itp.write_text(itp_text)
            gro.write_text("\n".join(gro_lines) + "\n")
            result = audit_reduced_network(itp, gro)

        self.assertEqual(result["junction_count"], 2)
        self.assertEqual(result["valid_strand_count"], 2)
        self.assertEqual(result["junction_degree_distribution"], {2: 2})
        self.assertEqual(result["parallel_junction_pair_count"], 1)
        self.assertEqual(result["parallel_strand_excess"], 1)
        self.assertEqual(result["self_loop_count"], 0)
        self.assertEqual(result["bridge_strand_count"], 0)
        self.assertEqual(result["two_core_strand_count"], 2)
        self.assertEqual(result["cycle_rank"], 1)
        self.assertEqual(result["junction_attachment_bond_count"], 4)
        self.assertEqual(len(result["atom_bond_connectivity_sha256"]), 64)
        self.assertEqual(len(result["junction_attachment_sha256"]), 64)
        self.assertEqual(result["periodic"]["winding_rank"], 1)
        self.assertTrue(result["periodic"]["spans_x"])
        self.assertFalse(result["periodic"]["spans_y"])
        self.assertFalse(result["periodic"]["spans_z"])


if __name__ == "__main__":
    unittest.main()
