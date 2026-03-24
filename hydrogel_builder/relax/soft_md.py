from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
    ntomp = int(soft_md.get("ntomp", runtime.get("omp_threads", 1)))

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(runtime.get("omp_threads", ntomp))
    env["GMX_OPENMP_MAX_THREADS"] = str(runtime.get("omp_threads", ntomp))

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

    mdrun_cmd = [gmx, "mdrun", "-deffnm", deffnm, "-ntomp", str(ntomp)]
    mdrun_cmd.extend(_string_list(soft_md.get("mdrun_extra", [])))
    print(f"Running soft_md mdrun: {' '.join(map(shlex.quote, mdrun_cmd))}")
    _run(mdrun_cmd, cwd=workdir, env=env)
    return workdir / f"{deffnm}.gro"
