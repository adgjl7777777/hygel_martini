# Bonded-parameter decision protocol example

This is a complete, synthetic E0--E6 replay.  It complements the trajectory
generation example in `../project/`: that example creates QM/xTB/Bartender
candidates, while this example determines whether one registered candidate may
advance to a tested-domain release and then to a declared transfer domain.

The numerical values and thresholds here are illustrative.  Copy the schema
and workflow, not the numbers.

## One-command replay

Install the package from the repository root, then run:

```bash
cd /path/to/hygel_martini
python -m pip install -e .
bash example/03_qm_to_martini/protocol_project/run_demo.sh
```

The script creates an isolated project under `work/synthetic_bond_demo`, seals
its prospective contract, commits E0 through E6, validates the hash chain and
all artifacts, and prints the final state.  It refuses to overwrite an
existing work directory.  To choose another output location:

```bash
bash example/03_qm_to_martini/protocol_project/run_demo.sh /tmp/my_protocol_demo
```

## Files to read

- `configure_demo.py`: constructs the exact scientific identity, data roles,
  candidate ladder, numeric thresholds, and evidence records.
- `run_demo.sh`: shows the complete command order with no hidden orchestration.
- generated `protocol.yaml`: project-level invariants.
- generated `iterations/v001/contract.yaml`: frozen decision rules.
- generated `evidence/E0.yaml` through `E6.yaml`: observations only; thresholds
  remain in the contract.
- generated `ledger.jsonl`: immutable decision history linked by SHA-256.

## Adapting it to a real chemistry

1. Initialize one bounded decision track for a bond family, angle block,
   dihedral block, or complete-topology replay.
2. Replace the synthetic mapping, topology graph, bead/nonbonded parent,
   exclusions, and independent-family registries.
3. Register omission or the exact upstream topology as the predecessor.
4. Replace every illustrative threshold with a scientifically justified,
   prospectively frozen value and retain sensitivity outputs as checksummed
   artifacts.
5. Keep genuinely unopened families in the sealed confirmation role until E5.
6. Treat length, single-chain, dilute-solution, and hydrogel tests as E6.  Do
   not feed them back into E5 coefficients under the same version.

The detailed schema and iteration rules are in
`docs/PARAMETERIZATION_PROTOCOL.md`.

