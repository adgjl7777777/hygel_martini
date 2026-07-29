from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _run(cmd: List[str], *, cwd: Path, env: Dict[str, str]) -> None:
    process = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={process.returncode}): {' '.join(map(shlex.quote, cmd))}\n"
            f"--- output ---\n{process.stdout}"
        )
    if process.stdout:
        print(process.stdout.rstrip())


def _string_list(value: Iterable[Any]) -> List[str]:
    return [str(item) for item in value]


def _build_mdrun_cmd(
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


def run_soft_md(cfg: Dict[str, Any]) -> Path:
    tools = cfg.get("tools", {})
    runtime = cfg.get("runtime", {})
    paths = cfg.get("paths", {})
    soft_md = cfg.get("soft_md", {})

    gmx = str(tools.get("gmx", "gmx_mpi"))
    system_top = Path(str(paths["system_top"])).resolve()
    start_gro = Path(str(paths["start_gro"])).resolve()
    mdp = Path(str(soft_md["mdp"])).resolve()
    workdir = Path(str(paths["workdir"])).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    for path in (system_top, start_gro, mdp):
        if not path.exists():
            raise FileNotFoundError(path)

    deffnm = str(soft_md.get("deffnm", "soft_md"))
    maxwarn = int(soft_md.get("maxwarn", 1))
    ntomp = int(runtime.get("omp_threads", 1))
    gpu_id_raw = runtime.get("gpu_id")
    gpu_id: Optional[str] = str(gpu_id_raw) if gpu_id_raw is not None else None
    mpi_np_raw = runtime.get("mpi_np")
    mpi_np: Optional[int] = int(mpi_np_raw) if mpi_np_raw is not None else None
    mpi_args: List[str] = [str(a) for a in runtime.get("mpi_args", [])]

    env = os.environ.copy()
    # OMP_NUM_THREADS must exactly match -ntomp (per-rank threads).
    env["OMP_NUM_THREADS"] = str(ntomp)
    env["GMX_OPENMP_MAX_THREADS"] = str(ntomp)

    tpr = workdir / f"{deffnm}.tpr"
    grompp_cmd = [
        gmx,
        "grompp",
        "-f",
        str(mdp),
        "-c",
        str(start_gro),
        "-p",
        str(system_top),
        "-o",
        str(tpr),
        "-maxwarn",
        str(maxwarn),
    ]
    grompp_cmd.extend(_string_list(soft_md.get("grompp_extra", [])))
    print(f"Running soft_md grompp: {' '.join(map(shlex.quote, grompp_cmd))}")
    _run(grompp_cmd, cwd=workdir, env=env)

    mdrun_cmd, env_extra = _build_mdrun_cmd(
        gmx,
        ["-deffnm", deffnm, "-ntomp", str(ntomp)],
        gpu_id,
        mpi_np,
        mpi_args,
        _string_list(soft_md.get("mdrun_extra", [])),
    )
    env.update(env_extra)
    print(f"Running soft_md mdrun: {' '.join(map(shlex.quote, mdrun_cmd))}")
    _run(mdrun_cmd, cwd=workdir, env=env)
    return workdir / f"{deffnm}.gro"
