from __future__ import annotations

import subprocess
import sys
from importlib.resources import files

import hygel_martini


def test_version_is_exposed() -> None:
    assert hygel_martini.__version__ == "0.1.0"


def test_builder_resources_are_packaged() -> None:
    package_root = files("hygel_martini")
    assert package_root.joinpath(
        "hydrogel_builder", "add_series", "water.gro"
    ).is_file()
    assert package_root.joinpath(
        "hydrogel_builder", "add_series", "water.itp"
    ).is_file()
    assert package_root.joinpath(
        "bash_settings", "hydrogel_builder", "run_full_builder.sh"
    ).is_file()


def test_primary_module_clis_expose_help() -> None:
    modules = (
        "hygel_martini.hydrogel_builder",
        "hygel_martini.hydrogel_builder.relax",
        "hygel_martini.param_opt.qm_to_opls",
        "hygel_martini.param_opt.opls_to_martini",
        "hygel_martini.param_opt.qm_to_martini",
        "hygel_martini.param_opt.qm_to_martini.analysis.reference_qualification",
        "hygel_martini.param_opt.qm_to_martini.protocol",
        "hygel_martini.property_extract",
    )
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout
