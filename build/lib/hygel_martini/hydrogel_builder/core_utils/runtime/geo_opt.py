"""
Geometry optimization helpers (grompp/mdrun) with consistent stdout/stderr logging.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from hygel_martini.hydrogel_builder.config_params.config import Config


def mdrun_gpu_flags(gpu_id: Optional[int]) -> Tuple[List[str], Dict[str, str]]:
    """Return (extra_args, env_overrides) for GPU/CPU mdrun control (single-process).

    Kept for backward compatibility with callers that don't use mpirun.
    For mpirun support use build_mdrun_cmd() instead.
    """
    if gpu_id is None:
        return ["-nb", "cpu", "-update", "cpu"], {}
    return [], {"CUDA_VISIBLE_DEVICES": str(gpu_id)}


def build_mdrun_cmd(
    gmx: str,
    base_args: List[str],
    gpu_id: Optional[str],
    mpi_np: Optional[int],
    mpi_args: List[str],
    extra: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Build the full mdrun command and required env overrides.

    No MPI (mpi_np=None):
      gpu_id=None    → single process, CPU-only flags (-nb cpu -update cpu)
      gpu_id="N"     → single process, GPU via CUDA_VISIBLE_DEVICES=N
    With MPI (mpi_np=N):
      gpu_id=None    → mpirun -np N [mpi_args] gmx mdrun ...
      gpu_id="0123"  → mpirun -np N [mpi_args] gmx mdrun ... -gpu_id 0123
      CUDA_VISIBLE_DEVICES is NOT set with MPI; ranks select GPU via -gpu_id.
    """
    env_extra: Dict[str, str] = {}
    mdrun = [gmx, "mdrun"] + base_args
    if mpi_np is None:
        if gpu_id is None:
            mdrun += ["-nb", "cpu", "-update", "cpu"]
        else:
            env_extra["CUDA_VISIBLE_DEVICES"] = gpu_id
        return mdrun + extra, env_extra
    else:
        if gpu_id is not None:
            mdrun += ["-gpu_id", gpu_id]
        return ["mpirun", "-np", str(mpi_np)] + mpi_args + mdrun + extra, env_extra


def _print_process_output(label: str, stdout: Optional[str], stderr: Optional[str]) -> None:
    """
    Print stdout/stderr blocks for a subprocess result using a common format.
    """
    stdout_content = stdout if stdout else ""
    stderr_content = stderr if stderr else ""

    print(f"--- {label} stdout ---")
    print(stdout_content.rstrip() if stdout_content.strip() else "(empty)")
    print(f"--- {label} stderr ---")
    print(stderr_content.rstrip() if stderr_content.strip() else "(empty)")


def _run_with_logs(cmd, label, log_path=None, cwd=None, env=None, input_text=None):
    """
    Run a subprocess, print stdout/stderr in a consistent block, and optionally
    append to a log file. Debug logging goes to Config.debug_log when enabled.
    """
    pretty = " ".join(map(str, cmd))
    print(f"\nRunning {label}: {pretty}")
    Config.debug_log(f"{label} CMD: {pretty}")

    proc = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd, env=env, input=input_text)
    _print_process_output(label, proc.stdout, proc.stderr)

    if log_path:
        try:
            with open(log_path, "w") as f:
                if proc.stdout:
                    f.write(proc.stdout)
                if proc.stderr:
                    f.write(proc.stderr)
        except Exception as e:
            Config.debug_log(f"{label} log write failed: {e}")

    Config.debug_log(f"{label} STDOUT:\n{proc.stdout or ''}")
    Config.debug_log(f"{label} STDERR:\n{proc.stderr or ''}")
    return proc

def _create_mdp_file(
    directory: str,
    cell_opt: bool = False,
    em_tol: float = 1000.0,
    nsteps: int = 5000,
    mdp_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Creates a gromacs .mdp file for energy minimization in the specified directory.
    """
    # Regarding cell_opt, for energy minimization with 'steep' integrator, the box is not changed.
    # An NPT run would be needed for cell equilibration. The user's request for
    # cell_opt is noted, but for this minimization script, it does not alter the MDP file.
    mdp_defaults = {
        "integrator": "steep",
        "nsteps": nsteps,
        "emtol": em_tol,
        "emstep": 0.01,
        "coulombtype": "PME",
        "rcoulomb": 1.1,
        "vdw_type": "cutoff",
        "rvdw": 1.1,
    }
    define_line = None

    if mdp_overrides:
        for key, value in mdp_overrides.items():
            if value is not None:
                if key == "define":
                    define_line = str(value)
                else:
                    mdp_defaults[key] = value

    mdp_content = f"""
; MDP file for energy minimization
integrator  = {mdp_defaults['integrator']}
nsteps      = {mdp_defaults['nsteps']}
emtol       = {mdp_defaults['emtol']}
emstep      = {mdp_defaults['emstep']}
"""
    if define_line:
        mdp_content += f"define      = {define_line}\n"

    mdp_content += f"""
; Parameters for interactions
coulombtype = {mdp_defaults['coulombtype']}
rcoulomb    = {mdp_defaults['rcoulomb']}
vdw_type    = {mdp_defaults['vdw_type']}
rvdw        = {mdp_defaults['rvdw']}
"""

    mdp_filepath = os.path.join(directory, "minim.mdp")
    with open(mdp_filepath, "w") as f:
        f.write(mdp_content)
    return mdp_filepath


def run_geo_opt(
    structure_file: str,
    topology_file: str,
    output_dir: str,
    cell_opt: bool = False,
    gmx_executable: str = "gmx",
    em_tol: float = 1000.0,
    nsteps: int = 5000,
    maxwarn: int = 1,
    mdp_overrides: Optional[Dict[str, Any]] = None,
    deffnm_prefix: str = "em",
    mdrun_extra_args: Optional[List[str]] = None,
    gpu_id: Optional[int] = None,
    ntomp: Optional[int] = None,
    mpi_np: Optional[int] = None,
    mpi_args: Optional[List[str]] = None,
):
    """
    Performs geometry optimization (energy minimization) for a given structure using Gromacs.

    Args:
        structure_file (str): Absolute path to the input structure file (.gro, .pdb).
        topology_file (str): Absolute path to the input topology file (.top).
        output_dir (str): Absolute path to the directory where optimization will be run and results stored.
        cell_opt (bool): If True, allows for cell optimization. (Note: For energy minimization with 'steep',
                         this has no effect. Cell optimization typically requires an NPT simulation).
        gmx_executable (str): The command to run gromacs (e.g., 'gmx' or 'gmx_mpi').
        em_tol (float): The energy minimization tolerance (emtol in .mdp).
        nsteps (int): Maximum number of steps for minimization (nsteps in .mdp).
        maxwarn (int): Number of warnings to ignore with gmx grompp.
        mdrun_extra_args (list[str] | None): Extra args appended to mdrun.

    Returns:
        str: The absolute path to the optimized structure file (.gro), or None if it failed.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Use absolute paths for input files to avoid issues
    structure_file = os.path.abspath(structure_file)
    topology_file = os.path.abspath(topology_file)

    mdp_file = _create_mdp_file(output_dir, cell_opt, em_tol, nsteps, mdp_overrides)

    tpr_file = os.path.join(output_dir, f"{deffnm_prefix}.tpr")
    
    # Run gmx grompp
    grompp_cmd = [
        gmx_executable, "grompp",
        "-f", mdp_file,
        "-c", structure_file,
        "-p", topology_file,
        "-o", tpr_file,
        "-po", os.path.join(output_dir, "mdout.mdp"),
    ]
    if mdp_overrides and str(mdp_overrides.get("define", "")).find("POSRES") >= 0:
        grompp_cmd.extend(["-r", structure_file])
    if maxwarn >= 0:
        grompp_cmd.extend(["-maxwarn", str(maxwarn)])

    grompp_log = os.path.join(output_dir, "grompp.log")
    process = _run_with_logs(grompp_cmd, "grompp", log_path=grompp_log)

    if process.returncode != 0:
        print(f"ERROR: gmx grompp failed. Check log: {grompp_log}")
        return None

    # Run gmx mdrun
    mdrun_log = os.path.join(output_dir, "mdrun.log")
    deffnm_path = os.path.join(output_dir, deffnm_prefix)

    # Thread count: explicit ntomp > SLURM env > let GROMACS auto-detect.
    # OMP_NUM_THREADS must match -ntomp; GROMACS 2026 is strict about this.
    resolved_ntomp: Optional[str] = None
    if ntomp is not None:
        resolved_ntomp = str(ntomp)
    else:
        slurm_threads = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_NTASKS")
        if slurm_threads:
            resolved_ntomp = str(slurm_threads)

    base_args = ["-v", "-deffnm", deffnm_path]
    if resolved_ntomp is not None:
        base_args += ["-ntomp", resolved_ntomp]

    gpu_id_str: Optional[str] = str(gpu_id) if gpu_id is not None else None
    mdrun_cmd, env_extra = build_mdrun_cmd(
        gmx_executable,
        base_args,
        gpu_id_str,
        mpi_np,
        mpi_args or [],
        mdrun_extra_args or [],
    )

    run_env = os.environ.copy()
    run_env.update(env_extra)
    if resolved_ntomp is not None:
        run_env["OMP_NUM_THREADS"] = resolved_ntomp
        run_env["GMX_OPENMP_MAX_THREADS"] = resolved_ntomp
    process = _run_with_logs(mdrun_cmd, "mdrun", log_path=mdrun_log, env=run_env)
    
    optimized_structure = f"{deffnm_path}.gro"
    
    if process.returncode != 0:
        print(f"ERROR: gmx mdrun failed. Check log: {mdrun_log}")
        if os.path.exists(optimized_structure):
             print(f"WARNING: mdrun failed, but an output structure was found at {optimized_structure}. It may be usable.")
             return optimized_structure
        return None
    
    if os.path.exists(optimized_structure):
        print(f"Successfully optimized structure. Output: {optimized_structure}")
        return optimized_structure
    else:
        print(f"ERROR: Optimization finished but the output file was not found: {optimized_structure}")
        return None
