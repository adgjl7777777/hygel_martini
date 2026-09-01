"""Thin GROMACS process and XVG parsing helpers.

Owns the subprocess wrapper :func:`run_gmx` (binary discovery via the
``GMX_BIN`` env var, then ``gmx_mpi``/``gmx`` on PATH) and the legacy
dict-style XVG parser :func:`parse_xvg` used by the requirement gate's
column checks and older analyzers.  Newer code reading XVGs for
statistics uses :func:`timeseries.read_xvg`, which returns the raw
array instead of a column dict.

Gotcha: :func:`parse_xvg` keys the first column as ``"time"``
regardless of what the file actually plots on the x axis.
"""
import subprocess
import os
import re
import numpy as np

def run_gmx(cmd, input_text=None, cwd=None):
    """Run a GROMACS command and return its stdout.

    The executable is resolved in this order: ``GMX_BIN`` environment
    variable, then ``gmx_mpi`` on PATH, then ``gmx`` on PATH.  A
    leading ``gmx``/``gmx_mpi`` token in ``cmd`` is replaced by the
    resolved binary; otherwise the binary is prepended.

    Args:
        cmd: Command tokens (list); the caller's list is not mutated.
        input_text: Optional text piped to stdin (e.g. group selection
            for ``gmx energy``).
        cwd: Optional working directory for the subprocess.

    Returns:
        Captured stdout as text.

    Raises:
        RuntimeError: If no GROMACS binary can be found, the binary is
            missing at exec time, or the command exits non-zero (the
            message includes stdout and stderr).
    """
    # Order of preference for gmx binary:
    # 1. Environment variable GMX_BIN
    # 2. gmx_mpi in PATH (shutil.which)
    # 3. gmx in PATH (shutil.which)
    import shutil
    gmx_exe = os.environ.get("GMX_BIN")
    if not gmx_exe:
        gmx_exe = shutil.which("gmx_mpi") or shutil.which("gmx")
    if not gmx_exe:
        raise RuntimeError(
            "GROMACS 실행 파일을 찾을 수 없습니다. "
            "GMX_BIN 환경 변수를 설정하거나 gmx/gmx_mpi를 PATH에 추가하세요."
        )
    
    cmd = list(cmd)  # caller의 list를 변경하지 않도록 복사
    if cmd[0] in ["gmx", "gmx_mpi"]:
        cmd[0] = gmx_exe
    elif cmd[0] != gmx_exe:
        cmd.insert(0, gmx_exe)

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, input=input_text, cwd=cwd)
    except FileNotFoundError:
        raise RuntimeError(f"GROMACS executable not found: {gmx_exe}. Please set GMX_BIN.")

    if proc.returncode != 0:
        raise RuntimeError(f"GROMACS command failed: {' '.join(cmd)}\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
    return proc.stdout

def parse_xvg(xvg_file):
    """Parse a GROMACS .xvg file into a column dictionary.

    Legend labels are read from ``@ s<i> legend "..."`` lines and used
    as keys for data columns after the first; unlabeled columns get
    ``"col<i>"`` names.  The first column is always keyed ``"time"``
    (whatever the file's x axis actually is).

    Args:
        xvg_file: Path to the .xvg file.

    Returns:
        Dict mapping ``"time"`` and each legend label to a 1-D float
        numpy array.

    Raises:
        ValueError: If no numeric rows are found, or the legend count
            exceeds the number of data columns (a sign the file and
            legends are inconsistent).
    """
    data = []
    labels = []
    with open(xvg_file, 'r') as f:
        for line in f:
            if line.startswith('@'):
                if 's' in line and 'legend' in line:
                    match = re.search(r'legend "(.*)"', line)
                    if match:
                        labels.append(match.group(1))
            elif not line.startswith(('#', '@')):
                parts = line.split()
                if parts:
                    data.append([float(x) for x in parts])
    
    if len(data) == 0:
        raise ValueError(f"No numeric data found in {xvg_file}")
    data = np.array(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    result = {'time': data[:, 0]}
    n_cols = data.shape[1] - 1
    for i in range(n_cols):
        label = labels[i] if i < len(labels) else f"col{i+1}"
        result[label] = data[:, i + 1]
    if len(labels) > n_cols:
        raise ValueError(f"xvg legend ({len(labels)} entries) exceeds data columns ({n_cols})")
    return result
