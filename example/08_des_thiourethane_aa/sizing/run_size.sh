#!/usr/bin/env bash
# Build one cell at a requested size, then shrink it.
#
# Why a wrapper rather than a maker per size: the makers have to be generated
# for the size, and the shrink target for a partial build is only known once
# the build has decided which strands formed. So this emits the pair
# (cell_sizes.py emit), builds, recomputes the target from the finished
# topology, and shrinks.
#
# Cores are whatever the batch allocation gave: --omp N, else
# SLURM_CPUS_PER_TASK, else OMP_NUM_THREADS, else the project defaults. This
# script does not survey the node or second-guess the scheduler. GPUs are left
# to GROMACS, which honours the CUDA_VISIBLE_DEVICES the scheduler sets.
#
# The import path is set explicitly: this checkout must shadow the frozen
# Series-01 install, which is what an unset PYTHONPATH would import instead.
#
# Usage:
#   ./run_size.sh --repeats 6 --strand n33 --des
#   ./run_size.sh --repeats 4 --strand n3 --conversion count:32 --seed 3
#   ./run_size.sh --repeats 4 --strand n3 --build-only --force
#
# Under Slurm, nothing extra is needed -- SLURM_CPUS_PER_TASK is picked up:
#   #SBATCH --cpus-per-task=16
#   srun ./run_size.sh --repeats 6 --strand n33 --des
#
# Every argument except the flags below is passed through to
# "cell_sizes.py emit". Extra flags: --density D (also used to retarget the
# shrink), --recipe R (composition audit after the build; a FAIL blocks the
# shrink), --omp N, --build-only, --shrink-only, --tag T.
#
# Fail-closed by construction (independent review 2026-09-05, Major 1): a
# refused emit, a force field matching no frozen release, a failed build, a
# failed recipe audit, drifted inputs or a failed retarget each stop the next
# stage, and the ledger records which.

set -uo pipefail
# NOT set -e. A stage that fails is the run the ledger most needs to record --
# the guarded shrink refusing to compress further is a result, not an accident
# -- so failures are captured per stage and reported at the end instead of
# aborting the script before anything is written down.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"          # the package root
PROJECT="$(cd "$HERE/../project" && pwd)"
LEDGER="$HERE/run_ledger.tsv"

emit_args=()
omp=""
density="1.0"
recipe=""
do_build=1
do_shrink=1
tag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)  do_shrink=0; shift ;;
    --shrink-only) do_build=0; shift ;;
    --tag)         tag="$2"; emit_args+=("--tag" "$2"); shift 2 ;;
    --omp)         omp="$2"; shift 2 ;;
    --density)     density="$2"; emit_args+=("--density" "$2"); shift 2 ;;
    --recipe)      recipe="$2"; shift 2 ;;
    *)             emit_args+=("$1"); shift ;;
  esac
done

# 0. Preflight. The builder shells out to gmx and Packmol, and a missing one
# surfaces halfway through as a confusing failure, so it is checked up front.
# GMXRC is sourced automatically when GROMACS is installed but not on PATH
# (it only prepends GROMACS's own paths); Packmol lives in a conda env here,
# and activating someone's environment behind their back is not this script's
# business, so that one is reported rather than fixed.
if ! command -v gmx_mpi >/dev/null 2>&1; then
  for rc in /opt/gromacs/2026/bin/GMXRC /opt/gromacs/bin/GMXRC; do
    if [[ -r "$rc" ]]; then
      # shellcheck disable=SC1090
      source "$rc"
      echo "sourced $rc"
      break
    fi
  done
fi
missing=()
command -v gmx_mpi >/dev/null 2>&1 || missing+=("gmx_mpi (source /opt/gromacs/2026/bin/GMXRC)")
command -v packmol >/dev/null 2>&1 || missing+=("packmol (conda activate hygel)")
if [[ ${#missing[@]} -gt 0 ]]; then
  printf 'missing on PATH:\n' >&2
  printf '  - %s\n' "${missing[@]}" >&2
  exit 1
fi

# The force-field files must match a frozen release, or the run has no
# parameter provenance. Fail closed rather than build on unnamed parameters.
if ! python3 "$HERE/../parameterization/release.py" current; then
  echo "refusing to run: the force-field files match no frozen release" >&2
  echo "  (parameterization/release.py freeze <name> after reviewing the diff)" >&2
  exit 1
fi

# 1. Cores: the allocation decides, not this script.
if [[ -z "$omp" ]]; then omp="${SLURM_CPUS_PER_TASK:-}"; fi
if [[ -z "$omp" ]]; then omp="${OMP_NUM_THREADS:-}"; fi
if [[ -n "$omp" ]]; then
  echo "cores: $omp (from --omp / SLURM_CPUS_PER_TASK / OMP_NUM_THREADS)"
  emit_args+=("--omp-threads" "$omp")
else
  echo "cores: not specified; keeping the project defaults"
  omp="-"
fi

# 2. Generate the makers for this size.
#
# Not in --shrink-only: re-emitting there would rewrite the makers and then
# re-write the run manifest, so the "inputs unchanged since the build" check
# below would compare a manifest against itself and pass vacuously. Resuming
# a build means resuming ITS inputs, so that mode requires an existing tag
# and an existing manifest and touches neither.
if [[ $do_build -eq 0 ]]; then
  [[ -n "$tag" ]] || { echo "--shrink-only needs --tag (which build to resume)" >&2; exit 1; }
  if [[ ! -f "$HERE/manifests/run_manifest_${tag}.json" ]]; then
    echo "no run manifest for tag '$tag': a build made before manifests existed" >&2
    echo "cannot be resumed under the audit trail -- rebuild it" >&2
    exit 1
  fi
  echo "resuming tag '$tag' against the manifest its build wrote (no re-emit)"
  # Whether those inputs still match is decided by the manifest check below,
  # which runs in both modes.
else

echo
echo "== emit =="
emit_tmp="$(mktemp)"
python3 "$HERE/cell_sizes.py" emit "${emit_args[@]}" > "$emit_tmp" 2>&1
emit_status=$?
emit_log="$(cat "$emit_tmp")"; rm -f "$emit_tmp"
echo "$emit_log"
if [[ $emit_status -ne 0 ]]; then
  echo "emit refused the inputs (status $emit_status); nothing built" >&2
  exit 1
fi
if [[ -z "$tag" ]]; then
  # cell_sizes.py prints "wrote .../maker_size_<tag>.yaml" first.
  tag="$(sed -n 's#^wrote .*/maker_size_\(.*\)\.yaml$#\1#p' <<<"$emit_log" | head -1)"
fi
[[ -n "$tag" ]] || { echo "could not determine the tag from emit output" >&2; exit 1; }

# 2b. Freeze everything this run reads -- makers, configs, every ITP in the
# include closure, recipe, mdps, seeds, HEAD, binaries -- so a later stage can
# prove it ran on the same inputs (independent review, Major 3).
echo
echo "== run manifest =="
manifest_args=(--tag "$tag"); [[ -n "$recipe" ]] && manifest_args+=(--recipe "$recipe")
if ! python3 "$HERE/run_manifest.py" write "${manifest_args[@]}"; then
  echo "refusing to run without a complete run manifest" >&2
  exit 1
fi
fi   # end of the build-mode emit/manifest block

build_maker="$PROJECT/maker_size_${tag}.yaml"
shrink_maker="$PROJECT/maker_size_${tag}_shrink.yaml"
outdir="$PROJECT/output_size_${tag}"

# /usr/bin/time -v gives peak RSS, which is the number that decides whether a
# bigger cell fits on this node. Fall back to the shell builtin if absent.
timer=(env)
if [[ -x /usr/bin/time ]]; then timer=(/usr/bin/time -v); fi

build_wall="-"; build_rss="-"; shrink_wall="-"; shrink_rss="-"; atoms="-"
build_status="skipped"; shrink_status="skipped"

_stat() {  # _stat <time -v log> <"Elapsed"|"Maximum">
  awk -v key="$2" '$0 ~ key {print $NF; exit}' "$1" 2>/dev/null || true
}

if [[ $do_build -eq 1 ]]; then
  echo
  echo "== build $tag =="
  log="$HERE/log_build_${tag}.txt"
  PYTHONPATH="$REPO" "${timer[@]}" python3 -m hygel_martini.hydrogel_builder \
      "$build_maker" 2>&1 | tee "$log"
  build_status="${PIPESTATUS[0]}"
  build_wall="$(_stat "$log" 'Elapsed .wall clock. time')"
  build_rss="$(_stat "$log" 'Maximum resident set size')"
  if [[ "$build_status" != "0" ]]; then
    echo "build failed (status $build_status); not shrinking" >&2
    do_shrink=0
  else
    python3 "$HERE/run_manifest.py" copy --tag "$tag" --into "$outdir" || true
    if [[ -n "$recipe" ]]; then
      echo
      echo "== composition audit against $(basename "$recipe") =="
      top="$outdir/final_system_no_ions_geo_opt/system.top"
      [[ -f "$top" ]] || top="$outdir/system.top"
      if ! python3 "$HERE/composition_audit.py" --top "$top" --recipe "$recipe"; then
        echo "composition audit FAILED; the build does not match its recipe -- not shrinking" >&2
        build_status="audit-failed"
        do_shrink=0
      fi
    fi
  fi
fi

if [[ $do_shrink -eq 1 ]]; then
  # 3a. Nothing this run reads may have changed since the manifest was written.
  # In --shrink-only mode this is what stops a stale build from being shrunk
  # against edited inputs under the same tag.
  echo
  echo "== run manifest check =="
  if ! python3 "$HERE/run_manifest.py" verify --tag "$tag" --ignore-shrink-maker; then
    echo "inputs changed since this tag was emitted; refusing to shrink a stale build" >&2
    shrink_status="stale-inputs"
    do_shrink=0
  fi
  if [[ ! -f "$outdir/system.top" ]]; then
    echo "no build output for tag $tag ($outdir/system.top); nothing to shrink" >&2
    shrink_status="no-build"
    do_shrink=0
  fi
fi

if [[ $do_shrink -eq 1 ]]; then
  # 3b. The realized composition -- not the requested one -- sets the target box.
  echo
  echo "== target box from the built topology (density $density g/cm3) =="
  top="$outdir/final_system_no_ions_geo_opt/system.top"
  [[ -f "$top" ]] || top="$outdir/system.top"
  python3 - "$top" "$shrink_maker" "$density" <<'PY'
import re, sys
sys.path.insert(0, "/nas_1/transcendence/2026_2/omni_hygel_package/package/example/08_des_thiourethane_aa/sizing")
from itp_inventory import system_composition

top, maker, density = sys.argv[1], sys.argv[2], float(sys.argv[3])
comp = system_composition(top)
text = open(maker).read()
aspect = [float(v) for v in re.search(r"target_box_nm: \[([^\]]+)\]", text).group(1).split(",")]
box = comp.box_for_density(density, aspect)
new = "[" + ", ".join(f"{v:.3f}" for v in box) + "]"
old = re.search(r"target_box_nm: (\[[^\]]+\])", text).group(1)
print(f"{comp.atom_count} atoms, {comp.mass:.0f} g/mol, net charge {comp.charge:+.4f} e")
print(f"target_box_nm {old} -> {new}")
if old != new:
    open(maker, "w").write(text.replace(f"target_box_nm: {old}", f"target_box_nm: {new}"))
    print("shrink maker updated to the realized composition")
PY
  retarget_status=$?
  if [[ $retarget_status -ne 0 ]]; then
    echo "target recalculation failed (status $retarget_status); not shrinking" >&2
    shrink_status="retarget-failed"
    do_shrink=0
  fi
fi

if [[ $do_shrink -eq 1 ]]; then
  echo
  echo "== shrink $tag =="
  log="$HERE/log_shrink_${tag}.txt"
  PYTHONPATH="$REPO" "${timer[@]}" python3 -m hygel_martini.hydrogel_builder.relax \
      "$shrink_maker" 2>&1 | tee "$log"
  shrink_status="${PIPESTATUS[0]}"
  shrink_wall="$(_stat "$log" 'Elapsed .wall clock. time')"
  shrink_rss="$(_stat "$log" 'Maximum resident set size')"
  # The guard stops rather than producing a broken cell, so a non-zero status
  # here means "this cell would not compress to that box", which is a finding.
  # last_valid.gro holds the last structure that passed every guard.
  if [[ "$shrink_status" != "0" ]]; then
    echo "shrink stopped early (status $shrink_status); last valid structure:" >&2
    echo "  $PROJECT/shrink_output_size_${tag}/last_valid.gro" >&2
    grep -E '^\[step' "$log" | tail -3 >&2 || true
  fi
fi

# 4. A ledger, so the next size can be chosen from what runs actually cost
# here rather than from a guess.
if [[ ! -f "$LEDGER" ]]; then
  printf 'when\thost\ttag\tomp\tatoms\tbuild_wall\tbuild_peak_kb\tbuild_status\tshrink_wall\tshrink_peak_kb\tshrink_status\n' > "$LEDGER"
fi
# Atom count for the ledger, read from whichever topology exists.
for candidate in "$outdir/final_system_no_ions_geo_opt/system.top" "$outdir/system.top"; do
  if [[ -f "$candidate" ]]; then
    atoms="$(PYTHONPATH="$HERE" python3 -c \
      'import sys; from itp_inventory import system_composition; print(system_composition(sys.argv[1]).atom_count)' \
      "$candidate" 2>/dev/null || echo -)"
    break
  fi
done
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -Is)" "$(hostname)" "$tag" "$omp" "${atoms:--}" \
  "$build_wall" "$build_rss" "$build_status" \
  "$shrink_wall" "$shrink_rss" "$shrink_status" >> "$LEDGER"
echo
echo "ledger: $LEDGER"
tail -2 "$LEDGER"

# Exit non-zero if any stage did, so a caller in a loop notices.
[[ "$build_status" == "0" || "$build_status" == "skipped" ]] || exit 1
[[ "$shrink_status" == "0" || "$shrink_status" == "skipped" ]] || exit 1
exit 0
