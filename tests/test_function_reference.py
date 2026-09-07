"""The generated function reference must match the source it describes.

`docs/archive/README_detailed_for_llm_series01.md` is what happens without
this test: a per-function reference, generated once, that ended up with zero
mentions of `net_layout`, `nets`, `rewire`, `local_matching`, `strand_loader`,
`aa_bonded` and `hard_em_shrink` -- every module the current branch is about.
A reference that lies is worse than none, so staleness is a test failure here
rather than something a reader discovers.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "tools", "gen_function_reference.py")
DOC = os.path.join(REPO, "docs", "FUNCTION_REFERENCE.md")


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_function_reference", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_function_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_reference_is_current(gen):
    """If this fails: `tools/gen_function_reference.py --write`, then re-read the diff."""
    assert os.path.exists(DOC), "docs/FUNCTION_REFERENCE.md is missing"
    assert open(DOC, encoding="utf-8").read() == gen.build([gen.DEFAULT_ROOT]), (
        "docs/FUNCTION_REFERENCE.md is stale; regenerate with "
        "tools/gen_function_reference.py --write"
    )


def test_check_mode_exits_nonzero_on_a_stale_file(gen, tmp_path):
    stale = tmp_path / "stale.md"
    stale.write_text("# not the reference\n")
    rc = subprocess.run([sys.executable, SCRIPT, "--check", "--out",
                         os.path.relpath(stale, REPO)],
                        capture_output=True, text=True, cwd=REPO).returncode
    assert rc == 1


def test_it_documents_the_modules_this_branch_added(gen):
    """The exact blind spot of the archived reference."""
    text = open(DOC, encoding="utf-8").read()
    for module in ("layout/nets.py", "layout/net_layout.py", "layout/rewire.py",
                   "layout/local_matching.py", "templates/strand_loader.py",
                   "runtime/aa_bonded.py", "relax/hard_em_shrink.py"):
        assert module in text, f"{module} missing from the function reference"


def test_undocumented_functions_are_marked_not_invented(gen, tmp_path):
    """`(no docstring)` rather than generated prose -- the old doc's worst habit."""
    sample = tmp_path / "m.py"
    sample.write_text('"""Mod."""\n\n\ndef bare(a, b=2):\n    return a\n')
    import ast
    tree = ast.parse(sample.read_text())
    fn = tree.body[-1]
    rendered = "\n".join(gen.render_def(fn, str(sample), False, "####"))
    assert "*(no docstring)*" in rendered
    assert "`bare(a, b=2)`" in rendered
    assert "returns: `a`" in rendered


def test_effect_hints_fire_on_subprocess_and_filesystem(gen, tmp_path):
    import ast
    src = ('def run():\n'
           '    subprocess.check_call(["ls"])\n'
           '    open("x").read()\n')
    fn = ast.parse(src).body[0]
    info = gen.analyse_body(fn)
    assert "subprocess" in info["effects"]
    assert "filesystem" in info["effects"]
