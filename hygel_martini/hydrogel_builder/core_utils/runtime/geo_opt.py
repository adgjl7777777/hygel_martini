"""
Geometry optimization helpers (grompp/mdrun) with consistent stdout/stderr logging.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional
from hygel_martini.hydrogel_builder.config_params.config import Config


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
    mdrun_cmd = [gmx_executable, "mdrun", "-v", "-deffnm", deffnm_path]
    
    # Auto-detect number of threads from Slurm environment
    threads = os.environ.get("SLURM_CPUS_PER_TASK")
    if threads is None:
        threads = os.environ.get("SLURM_NTASKS")
    
    if threads:
        mdrun_cmd.extend(["-ntomp", str(threads)])
    if mdrun_extra_args:
        mdrun_cmd.extend(mdrun_extra_args)

    process = _run_with_logs(mdrun_cmd, "mdrun", log_path=mdrun_log)
    
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
