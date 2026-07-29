from __future__ import annotations

from ._registry import BaseExtractor, register_extractor
from ..result import PropertyResult


@register_extractor("topology.reduced_network")
class ReducedNetworkTopologyExtractor(BaseExtractor):
    """Audit a junction--strand graph without chemistry-specific defaults."""

    extractor_name = "topology.reduced_network"
    required_inputs = ["itp"]

    def compute(self, inputs: dict, params: dict) -> PropertyResult:
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
