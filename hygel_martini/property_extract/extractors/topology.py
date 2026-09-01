"""Extractor adapter: reduced junction-strand network audit gate.

Registers ``topology.reduced_network``, which wraps
:func:`..network_topology.audit_reduced_network` for the manifest
runner and turns the raw audit into named pass/fail checks against the
expected counts declared in the manifest.  Requires only ``itp``; an
optional ``gro`` input additionally enables the periodic winding audit
(and is required when ``expected_winding_rank`` is declared).
``validation_role="structural_audit"``: this gate certifies bonded-
graph construction only, never force-field quality or equilibrium
mechanics.
"""
from __future__ import annotations

from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("topology.reduced_network")
class ReducedNetworkTopologyExtractor(BaseExtractor):
    """Audit a junction--strand graph without chemistry-specific defaults."""

    extractor_name = "topology.reduced_network"
    required_inputs = ["itp"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
        """Run the audit and evaluate the manifest's expected counts.

        Only expectations actually declared in ``params`` become
        checks; each check is named ``<field>_equals_<value>`` so the
        report is self-describing.  A ``max_malformed_strands`` bound
        (default 0) is always checked.

        Args:
            inputs: ``itp`` (bonded graph); optional ``gro`` for the
                periodic winding audit.
            params: ``junction_residue`` (default ``BCK``) plus any of
                ``expected_junction_count``, ``expected_strand_count``,
                ``expected_self_loop_count``,
                ``expected_parallel_strand_excess``,
                ``expected_bridge_strand_count``,
                ``expected_winding_rank``, ``max_malformed_strands``.

        Returns:
            Computed PropertyResult whose value is the boolean gate
            verdict; metadata keeps per-check outcomes, the full audit
            dict, and the claim boundary.

        Raises:
            ValueError: ``expected_winding_rank`` declared without a
                ``gro`` input (the periodic audit needs coordinates).
        """
        from ..network_topology import audit_reduced_network

        audit = audit_reduced_network(
            str(inputs["itp"]),
            str(inputs["gro"]) if inputs.get("gro") else None,
            junction_residue=str(params.get("junction_residue", "BCK")),
        )
        checks: dict[str, bool] = {}
        expected = {
            "junction_count": params.get("expected_junction_count"),
            "valid_strand_count": params.get("expected_strand_count"),
            "self_loop_count": params.get("expected_self_loop_count"),
            "parallel_strand_excess": params.get("expected_parallel_strand_excess"),
            "bridge_strand_count": params.get("expected_bridge_strand_count"),
        }
        for field, value in expected.items():
            if value is not None:
                checks[f"{field}_equals_{int(value)}"] = (
                    int(audit[field]) == int(value)
                )
        winding = params.get("expected_winding_rank")
        if winding is not None:
            if "periodic" not in audit:
                raise ValueError(
                    "expected_winding_rank requires inputs.gro for periodic audit"
                )
            checks[f"winding_rank_equals_{int(winding)}"] = (
                int(audit["periodic"]["winding_rank"]) == int(winding)
            )
        malformed_limit = int(params.get("max_malformed_strands", 0))
        checks[f"malformed_strands_at_most_{malformed_limit}"] = (
            int(audit["malformed_strand_component_count"]) <= malformed_limit
        )
        gate_pass = all(checks.values())
        return PropertyResult(
            property="reduced_network_topology_audit",
            value=gate_pass,
            status="computed",
            direct_experiment_comparison_allowed=False,
            validation_role="structural_audit",
            metadata={
                "checks": checks,
                "gate_pass": gate_pass,
                "audit": audit,
                "claim_boundary": (
                    "bonded-graph construction audit; not force-field or "
                    "equilibrium-mechanics validation"
                ),
            },
        )
