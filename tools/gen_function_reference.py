#!/usr/bin/env python3
"""Generate the package function reference from source, so it cannot go stale.

`docs/archive/README_detailed_for_llm_series01.md` was this document's
ancestor: a per-function reference written for a language model at Series-01.
It rotted, because it was generated once by hand-run analysis and then the
package grew a whole branch's worth of modules it had never heard of --
`net_layout`, `nets`, `rewire`, `local_matching`, `strand_loader`,
`aa_bonded`, `hard_em_shrink`, zero mentions of any of them.

So this is a script rather than a document, and `tests/` runs it in
`--check` mode: if the source and `docs/FUNCTION_REFERENCE.md` disagree, a
test fails and says to regenerate. A reference that lies is worse than no
reference, and the only reliable way to keep one honest is to make it
derived.

    tools/gen_function_reference.py --write        # regenerate the doc
    tools/gen_function_reference.py --check        # exit 1 if the doc is stale

Everything here is static analysis over `ast`. That has honest limits, and
the generated header says so: dynamic imports, `getattr` dispatch and
runtime-registered callables are invisible to it, and "effects" are *hints*
read off call names, not a proof of what a function does. Where the old
document had no docstring to quote it invented confident prose ("`main`
함수입니다...") -- this one prints `(no docstring)` instead, because a
reference whose descriptions are generated filler teaches nothing and hides
which functions are actually undocumented.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_ROOT = "hygel_martini"
DEFAULT_OUT = os.path.join("docs", "FUNCTION_REFERENCE.md")

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}
#: Call-name fragments that hint at a side effect, and the label to report.
EFFECT_HINTS = (
    (("subprocess.", "check_call", "check_output", "Popen", "os.system"), "subprocess"),
    (("open", "shutil.", "os.makedirs", "os.remove", "os.rename", "os.replace",
      "write_text", "mkdir", "unlink"), "filesystem"),
    (("Config.get", "Config.set", "Config.load", "get_runtime", "set_runtime"), "Config/runtime state"),
    (("World.", "Universe.", "register", "reset"), "global registry"),
    (("print",), "stdout"),
)
#: Third-party/stdlib prefixes not worth listing as "calls".
CALL_NOISE = {"len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
              "sorted", "enumerate", "zip", "range", "isinstance", "getattr", "min",
              "max", "sum", "abs", "round", "any", "all", "map", "filter", "repr"}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def python_files(root: str) -> List[str]:
    """Every ``.py`` under ``root``, sorted, skipping caches."""
    out: List[str] = []
    for base, dirs, files in os.walk(os.path.join(REPO, root)):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for name in sorted(files):
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(base, name), REPO))
    return sorted(out)


def first_line(text: Optional[str]) -> str:
    """The summary line of a docstring, collapsed to one line."""
    if not text:
        return ""
    for para in text.strip().split("\n\n"):
        line = " ".join(w for w in para.split())
        if line:
            return line
    return ""


def signature(fn: ast.AST) -> str:
    """Render a def's parameter list the way the source wrote it."""
    try:
        rendered = ast.unparse(fn.args)  # type: ignore[attr-defined]
    except Exception:                     # pragma: no cover - very old grammar
        rendered = "..."
    return f"{fn.name}({rendered})"       # type: ignore[attr-defined]


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def analyse_body(fn: ast.AST) -> Dict[str, object]:
    """Returns, raises, effect hints and call names for one def."""
    returns: List[str] = []
    raises: List[str] = []
    calls: List[str] = []
    bare_return = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            if node.value is None:
                bare_return = True
            else:
                try:
                    returns.append(ast.unparse(node.value))
                except Exception:
                    returns.append("?")
        elif isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            name = _call_name(target)
            if name:
                raises.append(name)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name and name not in CALL_NOISE:
                calls.append(name)
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            returns.append("<generator>")

    effects = []
    joined = " ".join(calls)
    for fragments, label in EFFECT_HINTS:
        if any(f in joined for f in fragments) and label not in effects:
            effects.append(label)

    return {
        "returns": _dedupe(returns),
        "bare_return": bare_return,
        "raises": _dedupe(raises),
        "effects": effects,
        "calls": _dedupe(calls),
    }


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def kind_of(fn: ast.AST, in_class: bool) -> str:
    decorators = {_call_name(d) or "" for d in getattr(fn, "decorator_list", [])}
    if "classmethod" in decorators:
        base = "classmethod"
    elif "staticmethod" in decorators:
        base = "staticmethod"
    elif "property" in decorators:
        base = "property"
    elif in_class:
        base = "method"
    else:
        base = "function"
    if isinstance(fn, ast.AsyncFunctionDef):
        base = "async " + base
    flags = [base]
    if fn.name.startswith("_") and not fn.name.startswith("__"):  # type: ignore[attr-defined]
        flags.append("internal")
    if fn.name == "main":                                          # type: ignore[attr-defined]
        flags.append("CLI entry")
    return ", ".join(flags)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render_def(fn: ast.AST, path: str, in_class: bool, indent: str) -> List[str]:
    info = analyse_body(fn)
    doc = first_line(ast.get_docstring(fn))
    lines = [f"{indent} `{signature(fn)}` — line {fn.lineno}"]  # type: ignore[attr-defined]
    lines.append(f"- {doc}" if doc else "- *(no docstring)*")
    lines.append(f"- kind: {kind_of(fn, in_class)}")
    if info["returns"]:
        shown = "; ".join(f"`{r}`" for r in info["returns"][:3])   # type: ignore[index]
        more = "" if len(info["returns"]) <= 3 else f" (+{len(info['returns']) - 3} more)"  # type: ignore[arg-type]
        lines.append(f"- returns: {shown}{more}")
    elif info["bare_return"]:
        lines.append("- returns: `None` (bare return)")
    if info["raises"]:
        lines.append("- raises: " + ", ".join(f"`{r}`" for r in info["raises"][:6]))  # type: ignore[index]
    if info["effects"]:
        lines.append("- effects: " + ", ".join(info["effects"]))    # type: ignore[arg-type]
    if info["calls"]:
        shown = ", ".join(f"`{c}`" for c in info["calls"][:12])     # type: ignore[index]
        more = "" if len(info["calls"]) <= 12 else f", +{len(info['calls']) - 12} more"  # type: ignore[arg-type]
        lines.append(f"- calls: {shown}{more}")
    lines.append("")
    return lines


def render_module(path: str) -> Tuple[List[str], int, int]:
    """Markdown for one module, plus its class and function counts."""
    with open(os.path.join(REPO, path), encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ([f"### `{path}`", "", f"**Does not parse**: {exc}", ""], 0, 0)

    lines = [f"### `{path}`", ""]
    doc = first_line(ast.get_docstring(tree))
    lines.append(doc if doc else "*(no module docstring)*")
    lines.append("")

    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    method_count = sum(
        1 for c in classes for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )

    for cls in classes:
        bases = ", ".join(_call_name(b) or "?" for b in cls.bases)
        lines.append(f"#### class `{cls.name}`" + (f"({bases})" if bases else "") +
                     f" — line {cls.lineno}")
        cdoc = first_line(ast.get_docstring(cls))
        lines.append(cdoc if cdoc else "*(no docstring)*")
        lines.append("")
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines += render_def(node, path, True, "#####")
    for fn in functions:
        lines += render_def(fn, path, False, "####")
    return lines, len(classes), len(functions) + method_count


def build(roots: Sequence[str]) -> str:
    files: List[str] = []
    for root in roots:
        files += python_files(root)

    by_package: Dict[str, List[str]] = {}
    for path in files:
        parts = path.split(os.sep)
        package = os.sep.join(parts[:2]) if len(parts) > 2 else parts[0]
        by_package.setdefault(package, []).append(path)

    body: List[str] = []
    total_classes = total_functions = 0
    toc: List[str] = []
    for package in sorted(by_package):
        toc.append(f"- [`{package}`](#{_anchor(package)})")
        body.append(f"## `{package}`")
        body.append("")
        for path in by_package[package]:
            rendered, ncls, nfn = render_module(path)
            total_classes += ncls
            total_functions += nfn
            body += rendered

    header = [
        "# Function reference",
        "",
        "**Generated — do not edit.** Regenerate with",
        "`tools/gen_function_reference.py --write`; `tests/test_function_reference.py`",
        "fails when this file and the source disagree.",
        "",
        f"Covers {len(files)} modules, {total_classes} classes, {total_functions} "
        "functions and methods under " + ", ".join(f"`{r}`" for r in roots) + ".",
        "",
        "For orientation — architecture, the config system, how to apply each",
        "example series, the invariants that break silently, and what a build does",
        "and does not license you to claim — read [`../README_FOR_LLM.md`](../README_FOR_LLM.md)",
        "first. This file is the index, not the map.",
        "",
        "### How to read it, and what it cannot tell you",
        "",
        "Everything below is static analysis over the syntax tree:",
        "",
        "- **`calls`** lists distinct names called in the body, in first-appearance",
        "  order, capped at twelve. Dynamic imports, `getattr` dispatch and",
        "  runtime-registered callables are invisible here.",
        "- **`effects`** are *hints* inferred from call names (`subprocess`,",
        "  `filesystem`, `Config/runtime state`, `global registry`, `stdout`), not a",
        "  proof. A function with no hint may still mutate state through a helper.",
        "- **`returns`** shows up to three distinct returned expressions verbatim, so",
        "  a function returning several shapes is visible as such.",
        "- **`(no docstring)`** means exactly that. It is never replaced with",
        "  generated prose, so this doubles as the list of undocumented functions.",
        "- Line numbers are from the generation commit. If they are off, regenerate.",
        "",
        "Two design facts worth knowing before reading any single function, both",
        "detailed in `README_FOR_LLM.md`: `main_components/Universe.World` keeps",
        "class-level registries, so constructing an `Atom` or `Bond` *is* a mutation;",
        "and `config_params/config.Config` holds the loaded config, its path and",
        "runtime state as class variables. Reading one function in isolation will",
        "mislead you about both.",
        "",
        "## Contents",
        "",
        *toc,
        "",
    ]
    return "\n".join(header + body).rstrip("\n") + "\n"


def _anchor(text: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation, keep underscores.

    Underscores survive and slashes do not, so ``hygel_martini/core`` anchors
    as ``hygel_martinicore``. Getting this wrong silently produces a table of
    contents whose links all miss.
    """
    keep = [c for c in text.lower() if c.isalnum() or c in "_- "]
    return "".join(keep).strip().replace(" ", "-")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", default=None,
                        help=f"package root to document (default {DEFAULT_ROOT})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"output path (default {DEFAULT_OUT})")
    parser.add_argument("--write", action="store_true", help="write the file")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the file on disk differs from freshly generated")
    args = parser.parse_args(argv)

    roots = args.root or [DEFAULT_ROOT]
    text = build(roots)
    target = os.path.join(REPO, args.out)

    if args.check:
        if not os.path.exists(target):
            print(f"{args.out} does not exist; run --write", file=sys.stderr)
            return 1
        current = open(target, encoding="utf-8").read()
        if current != text:
            print(f"{args.out} is stale; regenerate with "
                  "tools/gen_function_reference.py --write", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0
    if args.write:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.out} ({text.count(chr(10)) + 1} lines)")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
