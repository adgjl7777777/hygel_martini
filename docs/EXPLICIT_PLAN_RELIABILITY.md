# Explicit-plan construction and written-attachment guard

Introduced in **0.1.1.dev0** and included in release **0.1.1**. The Series-01 manuscript snapshot remains
`2472810c94303dfd447d17c5ccba6296ade08ece` (0.1.0); the following behavior
belongs to the subsequent development version.

```yaml
simulation_parameters:
  linker_orientation_strategy: connectivity_aware
  dynamic_crosslink_targets_per_stub: 2
  require_explicit_crosslink_plan: true
```

With this option enabled, absence of all plan metadata is an error. Partial
metadata continues to be rejected. Missing linkers or backbone ends also
stop the build. The option must be a YAML boolean. Existing configurations
that omit it keep their previous behavior, including legacy geometric routing.

Before bonding, the explicit route resolves its expected attachments to
one-based ITP atom indices. `planned_crosslinks.json` records these pairs and
the plan/materialization hash notes. After writing each backbone and chemically
expanded ITP, an independent parser checks that every planned attachment is
present exactly once and that there are no extra bonds between the registered
stubs and endpoints. Missing atom identities, duplicates, missing plans, or an
unreadable ITP stop this route. The check runs before the next geometry step
or formulation assembly; a `*.plan_audit.json` records its decision.

The reference is the planned assignment, not the writer's mutable bond list.
One removed cycle attachment can leave the graph connected; this check still
rejects it. Internal linker bonds and side-chain bonds outside the registered
endpoint sets are not attachment comparisons. It checks the builder's
unconditionally written crosslink records, not an arbitrary preprocessed ITP.
GROMACS preprocessing, complete chemical audits, relaxation acceptance, and
physical validation remain separate checks. Editing an ITP after the builder
returns is not monitored automatically.

The dry example also exposed an existing formulation-switch issue: a present
`add_water` block ran even when `enabled: false`. Explicit false values now
disable water, ion, molecule, or polymer addition before the stages are selected.
Absent switches retain the original key/count semantics; non-boolean switches
raise an error. This changes only configurations that explicitly disable a
stage or supply an invalid boolean. The existing manuscript inputs are not
modified or rerun.

See [the portable PEGDA example](../example/07_portable_pegda/README.md) for a
small complete construction/preprocessing run. CI builds a wheel and runs the
test suite against that wheel from outside the checkout. Installing the wheel
cannot silently borrow unbundled files from the working tree in that test.

## Template exclusion policy

From 0.1.1 the template contract rejects linker or monomer templates whose
`nrexcl` differs from 1, because per-instance exclusion transfer is not
implemented. A two-bead, one-bond, angle-free junction template (such as the
PEGDA `LNK` linker) may declare

```yaml
template_exclusion_policy: network_nrexcl_one
```

in its `linker_definitions` entry. The assembled network then uses the
network-wide `nrexcl = 1` convention. Templates outside that scope are
rejected before building. The historical examples `example/04_full_builder`
and `example/04_1_example_system` have not been migrated to this contract.
