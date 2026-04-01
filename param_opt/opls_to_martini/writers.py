from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_packmol_input(
    path: Path,
    polymer_xyz: Path,
    output_xyz: str,
    box_ang: Sequence[float],
    n_waters: int,
    seed: int,
    cfg: Dict[str, Any],
) -> None:
    polymer_ref = Path("..") / polymer_xyz.name
    wcfg = cfg["water"]
    text = f"""tolerance {wcfg['packmol_tolerance']}
filetype xyz
output {output_xyz}
seed {seed}

structure {polymer_ref.as_posix()}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure

structure {wcfg['packmol_water_structure']}
  number {n_waters}
  inside box 0.0 0.0 0.0 {box_ang[0]:.4f} {box_ang[1]:.4f} {box_ang[2]:.4f}
end structure
"""
    write_text(path, text)


def write_gromacs_mdp_templates(case_dir: Path, cfg: Dict[str, Any]) -> None:
    temp_k = cfg["system"]["temperature_c"] + 273.15
    mdp_cfg = cfg["mdp"]
    sampling_cfg = cfg["sampling"]

    em = f"""integrator  = steep
nsteps      = {mdp_cfg['em_nsteps']}
emtol       = {mdp_cfg['emtol']}
emstep      = {mdp_cfg['emstep']}
cutoff-scheme = {mdp_cfg['cutoff_scheme']}
coulombtype = {mdp_cfg['coulombtype']}
rcoulomb    = {mdp_cfg['rcoulomb_nm']}
rvdw        = {mdp_cfg['rvdw_nm']}
pbc         = {mdp_cfg['pbc']}
"""

    nvt = f"""integrator  = md
dt          = {sampling_cfg['dt_ps']}
nsteps      = {mdp_cfg['nvt_nsteps']}
nstxout-compressed = {mdp_cfg['nstxout_compressed']}
cutoff-scheme = {mdp_cfg['cutoff_scheme']}
coulombtype = {mdp_cfg['coulombtype']}
rcoulomb    = {mdp_cfg['rcoulomb_nm']}
rvdw        = {mdp_cfg['rvdw_nm']}
tcoupl      = {mdp_cfg['tcoupl']}
tc-grps     = {mdp_cfg['tc_grps']}
tau-t       = {mdp_cfg['tau_t_ps']}
ref-t       = {temp_k:.2f}
pcoupl      = no
gen-vel     = yes
gen-temp    = {temp_k:.2f}
gen-seed    = -1
constraints = {mdp_cfg['constraints']}
pbc         = {mdp_cfg['pbc']}
"""

    npt = f"""integrator  = md
dt          = {sampling_cfg['dt_ps']}
nsteps      = {mdp_cfg['npt_nsteps']}
nstxout-compressed = {mdp_cfg['nstxout_compressed']}
cutoff-scheme = {mdp_cfg['cutoff_scheme']}
coulombtype = {mdp_cfg['coulombtype']}
rcoulomb    = {mdp_cfg['rcoulomb_nm']}
rvdw        = {mdp_cfg['rvdw_nm']}
tcoupl      = {mdp_cfg['tcoupl']}
tc-grps     = {mdp_cfg['tc_grps']}
tau-t       = {mdp_cfg['tau_t_ps']}
ref-t       = {temp_k:.2f}
pcoupl      = Parrinello-Rahman
pcoupltype  = {mdp_cfg['npt_pcoupltype']}
tau-p       = {mdp_cfg['tau_p_ps']}
ref-p       = {mdp_cfg['ref_p_bar']}
compressibility = {mdp_cfg['compressibility_bar_inv']}
constraints = {mdp_cfg['constraints']}
pbc         = {mdp_cfg['pbc']}
"""

    md = f"""integrator  = md
dt          = {sampling_cfg['dt_ps']}
nsteps      = {sampling_cfg['sample_nsteps']}
nstxout-compressed = {mdp_cfg['nstxout_compressed']}
cutoff-scheme = {mdp_cfg['cutoff_scheme']}
coulombtype = {mdp_cfg['coulombtype']}
rcoulomb    = {mdp_cfg['rcoulomb_nm']}
rvdw        = {mdp_cfg['rvdw_nm']}
tcoupl      = {mdp_cfg['tcoupl']}
tc-grps     = {mdp_cfg['tc_grps']}
tau-t       = {mdp_cfg['tau_t_ps']}
ref-t       = {temp_k:.2f}
pcoupl      = no
constraints = {mdp_cfg['constraints']}
pbc         = {mdp_cfg['pbc']}
"""

    mdp_dir = case_dir / "mdp"
    write_text(mdp_dir / "em.mdp", em)
    write_text(mdp_dir / "nvt.mdp", nvt)
    write_text(mdp_dir / "npt.mdp", npt)
    write_text(mdp_dir / "md.mdp", md)


def write_topol_stub(path: Path, cfg: Dict[str, Any]) -> None:
    top_cfg = cfg["topology"]
    water_include = top_cfg.get("water_include", "")
    water_name = top_cfg.get("water_molecule_name", "SOL")
    water_count = top_cfg.get("water_molecule_count", 0)

    include_lines = [f'#include "{top_cfg["forcefield_include"]}"']
    if water_include:
        include_lines.append(f'#include "{water_include}"')
    include_lines.append(f'#include "{top_cfg["polymer_itp"]}"')

    solvate_tool = cfg.get("system", {}).get("solvate_tool", "gromacs")
    molecules_lines = [f"{top_cfg['molecule_name']}   {top_cfg['molecule_count']}"]
    # In gromacs-solvate mode, let `gmx solvate -p` append/update SOL entries.
    if not (solvate_tool == "gromacs" and int(water_count) == 0):
        molecules_lines.append(f"{water_name}   {water_count}")

    include_block = "\n".join(include_lines)
    molecules_block = "\n".join(molecules_lines)

    content = f"""; Fill this topology for your force field and polymer itp.
; Update include paths and molecule names/count as needed.

{include_block}

[ system ]
{top_cfg['system_name']}

[ molecules ]
{molecules_block}
"""
    write_text(path, content)


def write_pipeline_script(replica_dir: Path, box_nm: Sequence[float], cfg: Dict[str, Any]) -> None:
    lbox = f"{box_nm[0]:.4f}"
    runtime_cfg = cfg["runtime"]
    water_cfg = cfg["water"]
    gmxrc_path = cfg["paths"]["gmxrc_path"]

    script = f"""#!/usr/bin/env bash
set -euo pipefail

# GROMACS command bootstrap (works in nohup/non-interactive shells)
GMXRC_PATH="${{GMXRC_PATH:-{gmxrc_path}}}"
if ! command -v gmx >/dev/null 2>&1 && ! command -v gmx_mpi >/dev/null 2>&1; then
  if [[ -f "$GMXRC_PATH" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$GMXRC_PATH"
    set -u
  fi
fi
if command -v gmx >/dev/null 2>&1; then
  GMX_BIN="gmx"
elif command -v gmx_mpi >/dev/null 2>&1; then
  GMX_BIN="gmx_mpi"
else
  echo "[ERROR] GROMACS command not found (gmx/gmx_mpi)." >&2
  echo "[ERROR] Set PATH or GMXRC_PATH." >&2
  exit 127
fi

# Runtime controls:
#   RUN_MODE=none|cpu|gpu (default from config)
#   OMP_THREADS=<int>
#   GPU_ID=<id[,id...]>
#   NTMPI_GPU=<int>
#   SEED=<int>
RUN_MODE="${{RUN_MODE:-{runtime_cfg['default_run_mode']}}}"
POLYMER_ITP_REL="../{cfg['topology']['polymer_itp']}"
TOPOL_WORK="../topol.run.top"
REPLICA_NAME="$(basename "$PWD")"
REPLICA_NUM="${{REPLICA_NAME##*_}}"
REPLICA_NUM="${{REPLICA_NUM#0}}"
if [[ -z "${{REPLICA_NUM}}" ]]; then REPLICA_NUM=1; fi
DEFAULT_GPU_ID=$((REPLICA_NUM - 1))
GPU_ID="${{GPU_ID:-$DEFAULT_GPU_ID}}"

if [[ "${{RUN_MODE}}" == "cpu" ]]; then
  OMP_THREADS="${{OMP_THREADS:-{runtime_cfg['cpu_omp_threads']}}}"
elif [[ "${{RUN_MODE}}" == "gpu" ]]; then
  OMP_THREADS="${{OMP_THREADS:-{runtime_cfg['gpu_omp_threads']}}}"
else
  OMP_THREADS="${{OMP_THREADS:-{runtime_cfg['none_omp_threads']}}}"
fi

if [[ -z "${{SEED:-}}" ]]; then
  SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
fi
if [[ -z "${{SEED}}" ]]; then
  SEED={runtime_cfg['random_seed_fallback']}
fi
ROTX=$(( SEED % 360 ))
ROTY=$(( (SEED / 7) % 360 ))
ROTZ=$(( (SEED / 13) % 360 ))

"$GMX_BIN" editconf -f ../polymer.pdb -o polymer_box.gro -c -box {lbox} {lbox} {lbox} -rotate $ROTX $ROTY $ROTZ
cp ../topol.top "$TOPOL_WORK"
"$GMX_BIN" solvate -cp polymer_box.gro -cs {water_cfg['gromacs_water_model']} -o solvated_init.gro -p "$TOPOL_WORK"

if [[ "$RUN_MODE" == "none" ]]; then
  echo "[INFO] RUN_MODE=none -> stopped after solvated_init.gro generation"
  exit 0
fi

if [[ ! -f "$POLYMER_ITP_REL" ]]; then
  echo "[ERROR] Missing polymer ITP include: $POLYMER_ITP_REL" >&2
  echo "[ERROR] Provide topology.polymer_itp as an existing paths.base_dir-relative path." >&2
  exit 3
fi

cp ../mdp/nvt.mdp nvt_local.mdp
sed -i "s/^gen-seed.*/gen-seed    = $SEED/" nvt_local.mdp

run_mdrun() {{
  local stage="$1"
  if [[ "$RUN_MODE" == "cpu" ]]; then
    "$GMX_BIN" mdrun -deffnm "$stage" -ntmpi 1 -ntomp "$OMP_THREADS" -pin on
  elif [[ "$RUN_MODE" == "gpu" ]]; then
    IFS=',' read -r -a GPU_ARR <<< "$GPU_ID"
    GPU_COUNT="${{#GPU_ARR[@]}}"
    if (( GPU_COUNT < 1 )); then GPU_COUNT=1; fi
    NTMPI_GPU="${{NTMPI_GPU:-$GPU_COUNT}}"
    "$GMX_BIN" mdrun -deffnm "$stage" -ntmpi "$NTMPI_GPU" -ntomp "$OMP_THREADS" -nb gpu -pme cpu -gpu_id "$GPU_ID" -pin on
  else
    echo "[INFO] Unsupported RUN_MODE=$RUN_MODE"
    exit 2
  fi
}}

"$GMX_BIN" grompp -f ../mdp/em.mdp  -c solvated_init.gro -p "$TOPOL_WORK" -o em.tpr
run_mdrun em
"$GMX_BIN" grompp -f nvt_local.mdp  -c em.gro -r em.gro -p "$TOPOL_WORK" -o nvt.tpr
run_mdrun nvt
"$GMX_BIN" grompp -f ../mdp/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p "$TOPOL_WORK" -o npt.tpr
run_mdrun npt
"$GMX_BIN" grompp -f ../mdp/md.mdp  -c npt.gro -t npt.cpt -p "$TOPOL_WORK" -o md.tpr
run_mdrun md
"""
    write_text(replica_dir / "run_pipeline.sh", script)
    (replica_dir / "run_pipeline.sh").chmod(0o755)