from pathlib import Path

import pytest
import yaml

from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import _extract_definition
from hygel_martini.hydrogel_builder.core_utils.templates.template_contract import require_supported_template

EXAMPLE = Path(__file__).resolve().parents[1] / "example" / "07_portable_pegda"


def _linker_entry():
    cfg = yaml.safe_load((EXAMPLE / "maker_template.yaml").read_text())

    def find(node):
        if isinstance(node, dict):
            if "linker_definitions" in node:
                return node["linker_definitions"]["LINKERS"][0]
            for value in node.values():
                hit = find(value)
                if hit is not None:
                    return hit
        return None

    return find(cfg)


def test_portable_pegda_linker_passes_template_contract():
    entry = _linker_entry()
    definition = _extract_definition(str(EXAMPLE / "structure" / "LNK.itp"), entry["molecule_name"])
    require_supported_template(definition, exclusion_policy=entry.get("template_exclusion_policy"))


def test_portable_pegda_linker_rejected_without_policy():
    entry = _linker_entry()
    definition = _extract_definition(str(EXAMPLE / "structure" / "LNK.itp"), entry["molecule_name"])
    with pytest.raises(ValueError):
        require_supported_template(definition, exclusion_policy=None)
