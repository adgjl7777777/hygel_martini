# HyGel Builder manuscript release checklist

This checklist separates technical release readiness from institutional and
submission decisions. It does not authorize a software license, mint an archive
identifier, or claim that paper-specific trajectories are publicly distributed.

## Technical gates

- [x] Package version is exposed as `0.1.0` and agrees with `CITATION.cff`.
- [x] Generated `build/` output is excluded from version control.
- [x] PoreBlazer is registered as an external Git submodule rather than copied
      into the Python distribution.
- [x] The public repository contains executable builder, relaxation, topology
      audit, and property-analysis entry points.
- [x] Regression tests cover packaging, graph planning, corrupted construction
      records, topology, analysis primitives, and command-line help routes.
- [x] GitHub Actions installs the package, runs the regression suite, builds the
      source/wheel distributions, and checks the distribution metadata.
- [ ] Confirm that the first public CI run passes from a clean GitHub runner.
- [ ] Re-run one tracked, small builder example in a clean documented
      environment and retain its command, configuration, output hashes, and
      audit report.

## Institutional and archival gates

- [ ] The author, principal investigator, and applicable Seoul National
      University/R&DB office approve an open-source license. Until then,
      `LICENSING.md` remains the authoritative notice.
- [ ] After approval, add the exact `LICENSE`, update `pyproject.toml` license
      metadata, and add the license identifier to `CITATION.cff`.
- [ ] Freeze the manuscript software commit after all code/documentation edits.
- [ ] Create a versioned release tag only after the license gate passes.
- [ ] Archive that exact release and record a DOI only after one actually exists.
- [ ] Replace manuscript placeholders with the final commit, release tag, archive
      DOI, and access date; never predict or invent these identifiers.

## Manuscript-package gates

- [ ] Build `main.tex` and `si.tex` without missing citations or references.
- [ ] Verify all values against the frozen analysis tables and
      `submission_manifest.json`.
- [ ] Keep paper-specific production provenance separate from the small public
      reproduction example.
- [ ] Submit the 150-word abstract, approximately 75-word short summary, TOC
      graphic, cover letter, main PDF, SI PDF, source files, and code/archive
      statement required by the selected Journal of Computational Chemistry
      article type.
