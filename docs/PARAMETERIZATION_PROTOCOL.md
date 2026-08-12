# Reproducible QM/xTB-to-Martini parameterization protocol

## 1. Purpose and claim boundary

`hygel-parameter-protocol` turns a bonded-parameter study into a sealed,
replayable decision process.  It does not assume that a Bartender fit, an ORCA
termination, a GROMACS topology, or a finite trajectory is itself a final
parameter.  Instead, each candidate advances through cumulative evidence
levels, and every non-pass outcome remains an explicit terminal.

The protocol is chemistry-independent.  A project supplies its own mapping,
bead/nonbonded parent, candidate functions, independent-family definitions,
objectives, thresholds, and transfer domain.  The package supplies the state
machine, schema checks, role separation, checksum seal, weakest-link decision,
and verification ledger.

The strongest automatic claim is deliberately bounded:

- E5 may authorize a topology only inside the sealed tested domain.
- E6 may qualify specified length, single-chain, dilute-solution, or material
  domains.
- E6 data may not be used to alter the E5 parameter under the same version.
- No E-level creates a universal mapping, bead, or transferability claim.

## 2. Relationship to the existing package workflow

The package components have separate responsibilities:

1. `hygel-qm-to-martini` generates polymers, QM/xTB trajectories, Bartender
   candidates, and postprocessed term tables.
2. `hygel-qm-reference-audit` checks reference-energy agreement, stationarity,
   endpoint-family consistency, and reweighting overlap without fitting a CG
   force field.
3. Project-specific analysis produces grouped fit, identifiability,
   selection, GROMACS realization, and transfer metrics.
4. `hygel-parameter-protocol` decides which evidence is eligible, whether a
   frozen gate passes, what claim is allowed, and whether a new iteration is
   required.

The analysis program may be replaced.  Its output cannot replace the sealed
contract: the contract fixes the criterion IDs, operators, thresholds, data
roles, candidate ladder, grouping, and terminal mapping before the result is
opened.

## 3. Scientific sequence

The intended scientific order is:

```text
mapping + bead/nonbonded parent
            |
candidate generation (including explicit omission/predecessor)
            |
qualified reference and correction design
            |
bond -> angle -> proper dihedral
            |
single complete-topology replay
            |
unopened one-shot confirmation = E5 tested-domain release
            |
length / single-chain / dilute solution / hydrogel = E6 domain qualification
```

Angles may not silently repair a failed bond, and dihedrals may not silently
repair a failed bond or angle.  E4 therefore contains both a local target gate
and an upstream non-regression gate.  Omission is a real candidate state, not
missing output.

## 4. Evidence levels

| Level | Required evidence | What a pass permits | What it cannot prove |
|---|---|---|---|
| E0 | exact inputs, sources, markers, software, manifests, checksums | evidence eligibility | a correct parameter |
| E1 | graph, rank, condition, leverage, support, physical admissibility | analytic candidacy | realized CG behavior |
| E2 | family-grouped effect and complete nested selection | grouped intrinsic qualification | GROMACS realization |
| E3 | exact topology, force/sign verification, `grompp -maxwarn 0`, EM, finite MD | numerical realization | distribution agreement |
| E4 | family-balanced target distribution, support, upstream non-regression, complete replay | local tested-model decision | unopened confirmation |
| E5 | one-shot genuinely unopened confirmation without retuning | tested-domain release | broad transferability |
| E6 | declared length/chain/solution/material tests | stated domain qualification | universality or coefficient feedback |

The engine evaluates levels in this exact order.  A FAIL or INCONCLUSIVE
terminal blocks every later level in the iteration.

## 5. One decision track per project

A protocol project should correspond to one bounded scientific decision:

- one symmetry-equivalent bond family;
- one angle block after its upstream bond vector is frozen;
- one proper-dihedral block after bond and angle decisions are frozen; or
- one complete-topology release replay.

Separate chemical species or scientifically independent candidate branches
should use sibling projects.  This prevents a large heterogeneous contract
from hiding which candidate, predecessor, data groups, or terminal applies.

## 6. Initialize and populate a project

```bash
hygel-parameter-protocol init my_bond_protocol \
  --project-id speciesX_backbone_bond \
  --title "Species X backbone-bond decision" \
  --claim-domain "mapping m1, Martini parent p1, 310 K"
```

The command refuses a non-empty target directory.  It creates:

```text
my_bond_protocol/
├── README.md
├── protocol.yaml
├── ledger.jsonl
├── evidence_template.yaml
├── inputs/
│   ├── mapping.yaml
│   ├── topology_graph.yaml
│   ├── bead_model.yaml
│   ├── nonbonded_parent.yaml
│   └── exclusions.yaml
├── data/
│   ├── development_groups.tsv
│   ├── validation_groups.tsv
│   ├── stress_groups.tsv
│   └── confirmation_groups.tsv
├── evidence/
├── artifacts/
└── iterations/v001/
    ├── contract.yaml
    └── decisions/
```

Replace every placeholder.  Artifact paths must remain inside the project;
path traversal and symlinks resolving outside it are rejected.  Set
`placeholder: false` only after the exact file has been installed, then write
the observed hashes:

```bash
hygel-parameter-protocol hash-inputs my_bond_protocol --write
hygel-parameter-protocol validate my_bond_protocol
```

Draft placeholder warnings are allowed by `validate`, but `seal` rejects every
placeholder.

## 7. Freeze the scientific contract

The contract contains five immutable scientific-identity artifacts:

- mapping;
- topology graph;
- bead model, including type and charge;
- nonbonded parent;
- exclusions.

It also freezes:

- data-group IDs and roles;
- sealed confirmation groups;
- coordinate and explicit predecessor/omission;
- candidate function ladder and maximum complexity;
- primary and sensitivity objectives;
- independent grouping unit;
- stop rule and claim ceiling;
- E0--E6 criteria, operators, thresholds, and outcome labels;
- permitted implementation repairs and prohibited post-seal changes.

Seal only before the relevant evidence is opened:

```bash
hygel-parameter-protocol seal my_bond_protocol
```

`seal.json` records the canonical contract hash, scientific-identity hash,
every input/data checksum, timestamp, and sealed confirmation IDs.  Later
editing of the contract or any referenced file invalidates the project.

## 8. Criterion operators

Each criterion has an ID, description, operator, optional expected value, and
non-pass labels.  The supported operators are:

| Operator | Meaning |
|---|---|
| `truthy` | observed value must be the YAML boolean `true` |
| `status` | independently verified external result must be `PASS`, `FAIL`, or `INCONCLUSIVE` |
| `eq`, `ne` | exact equality/inequality |
| `lt`, `le`, `gt`, `ge` | finite numeric comparison |
| `between` | inclusive finite interval `[lower, upper]` |
| `in` | membership in a frozen list |

Prefer direct numeric criteria when the analysis exposes the metric.  Use
`status` for a separately validated multivariate check, and retain its
checksummed machine-readable artifact.  The evidence file carries only the
observation; it cannot supply or override a threshold.

Example:

```yaml
- id: exact_outer_selection_frequency
  description: Exact model identity recurs in the required outer families.
  operator: ge
  expected: 0.6666667
  on_fail: SELECTION_LIMITED
  on_inconclusive: DATA_LIMITED
```

These numbers are study-specific frozen engineering guardrails, never
universal Martini constants.

## 9. Data roles and leakage prevention

Each independent group belongs to exactly one role.  The same source path and
checksum cannot be registered twice under different names.

- `development`: opened fitting, candidate design, and method development;
- `validation`: opened internal/grouped validation fixed by the contract;
- `stress`: length, boundary, or transfer tests, especially E6;
- `confirmation`: genuinely unopened E5 data.

Only confirmation groups may have `sealed: true`.  E0--E4 evidence that names
a sealed group is rejected.  E5 accepts only sealed confirmation groups.  E6
must include a stress/transfer group.

Once an E5 record is committed, the confirmation group is opened even if the
result is unfavorable or inconclusive.  If a new iteration is created, the
engine reclassifies it as development.  A fresh E5 claim therefore requires a
new sealed confirmation group.

## 10. Evaluate and commit evidence

Store both the evidence YAML and every referenced artifact in the project.
The evidence schema is:

```yaml
schema_version: '1.0'
project_id: speciesX_backbone_bond
iteration_id: v001
gate: E1
evidence_id: rank_support_verification_v1
data_group_ids: [development_groups]
artifacts:
  - id: rank_support_json
    path: artifacts/rank_support_v1.json
    sha256: <64-lowercase-hex-digest>
observations:
  graph_and_canonicalization: true
  design_rank_fraction: 1.0
  maximum_condition_number: 8200.0
notes: Frozen analysis script version and concise interpretation.
```

Preview first; preview is read-only:

```bash
hygel-parameter-protocol evaluate my_bond_protocol \
  my_bond_protocol/evidence/E1.yaml
```

Commit only after checking the preview:

```bash
hygel-parameter-protocol evaluate my_bond_protocol \
  my_bond_protocol/evidence/E1.yaml --commit
hygel-parameter-protocol status my_bond_protocol
```

The evidence must contain exactly the criterion IDs registered for the current
gate.  Missing and extra IDs are rejected.  Every referenced artifact hash is
verified.  The gate result uses weakest-link aggregation:

1. any criterion FAIL -> gate FAIL;
2. otherwise any criterion INCONCLUSIVE -> gate INCONCLUSIVE;
3. otherwise -> PASS.

A scientific non-pass is a valid protocol result, not a crashed command.  The
ledger stores the gate result, exact terminal, criterion-specific non-pass
diagnoses, claim ceiling, and next permitted action.

## 11. Iteration classes

Do not edit a failed terminal.  Create a new draft:

```bash
hygel-parameter-protocol new-iteration my_bond_protocol \
  --id v002 \
  --class TYPE_III \
  --failure-mechanism REPRESENTABILITY_FAILURE
```

| Class | Intended change | Enforced boundary |
|---|---|---|
| `TYPE_I` | next candidate already in a predeclared ladder | identity, design, and gates unchanged |
| `TYPE_II` | parser/sign/runtime implementation repair | identity, design, gates, manifests, and roles unchanged |
| `TYPE_III` | new coupled coordinate/function/support model | mapping/bead/nonbonded identity unchanged |
| `TYPE_IV` | changed estimand, threshold, role boundary, mapping, bead, or nonbonded parent | new confirmation required |

The engine limits repeated Type-I/II corrections for one unchanged mechanism
according to `max_correction_iterations_per_mechanism`.  At the cap, classify
the limitation or prospectively justify a Type-III/IV change.

Any output that motivated a changed function, threshold, grouping, estimand,
or model is development evidence in the new version.  It is not retroactive
confirmation.

## 12. Verification and recovery

Run at any time:

```bash
hygel-parameter-protocol validate my_bond_protocol
hygel-parameter-protocol status my_bond_protocol
```

Validation checks:

- protocol and contract schema;
- exact E0--E6 ordering and criterion uniqueness;
- data-role/source exclusivity;
- artifact existence and checksum;
- contract and scientific-identity seals;
- transition-class restrictions;
- ledger sequence, previous hash, and event hash;
- evidence and analysis-artifact immutability;
- no decision before a seal;
- no gate after a non-pass terminal.

The ledger is append-only and hash chained.  Preserve every failed and
superseded artifact.  A damaged ledger or changed committed evidence is not
silently repaired; restore the preserved file or start an explicitly documented
recovery version.

## 13. Complete runnable example

The repository contains an end-to-end synthetic example at:

```text
example/03_qm_to_martini/protocol_project/
```

It demonstrates frozen numeric operators, the omission predecessor, exact
data roles, E0--E6 evidence files, and a one-command isolated replay.  The
numbers are intentionally synthetic and must not be copied into a scientific
study without justification.
