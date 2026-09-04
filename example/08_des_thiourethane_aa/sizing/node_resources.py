#!/usr/bin/env python3
"""Ask the machine what is free, and recommend how much of it to take.

A build or a shrink is launched wherever there is room, which is not known
when the maker is written. This reports what the node has, what is already in
use, and a thread count and GPU that would not displace someone else's work,
so ``run_size.sh`` can fill those in at launch instead of a maker hardcoding
them.

The policy is deliberately conservative, and each rule exists for a reason:

* **Never more than a fraction of the node.** Default half. A build that grabs
  every core makes the node unusable for everyone else, and the builder's
  Python stages are single-threaded anyway, so the extra cores buy little.
* **Subtract what is already running.** ``loadavg`` over one minute is used as
  the count of busy cores, which is the cheapest honest estimate available.
* **Leave a reserve free.** Default two cores, so an interactive shell on the
  node stays responsive.
* **Back off hard on a node with busy GPUs.** Someone else's GPU job needs CPU
  cores to feed it; taking them starves a job that is far more expensive than
  ours. On such a node the recommendation is capped low regardless of the
  idle-core count.
* **Only claim a GPU that is genuinely idle.** Both near-zero utilization and
  near-empty memory, because either alone is a common false negative.

Nothing here changes any file. It reads ``/proc``, ``os``, and -- if present
-- ``nvidia-smi``; a machine without GPUs simply reports none.

Usage::

    node_resources.py                 # human-readable report
    node_resources.py --json          # machine-readable, for scripts
    node_resources.py --omp-only      # just the thread count, for $(...)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

#: Never recommend more than this fraction of the node's cores.
DEFAULT_MAX_FRACTION = 0.5
#: Always leave at least this many cores unclaimed.
DEFAULT_RESERVE_CORES = 2
#: Cap on a node where someone else's GPU work is running.
BUSY_GPU_NODE_THREAD_CAP = 4
#: A GPU counts as busy above either of these.
GPU_BUSY_UTIL_PERCENT = 10.0
GPU_BUSY_MEMORY_FRACTION = 0.05


@dataclass
class Gpu:
    index: int
    name: str
    memory_used_mib: float
    memory_total_mib: float
    utilization_percent: float

    @property
    def memory_fraction(self) -> float:
        return self.memory_used_mib / self.memory_total_mib if self.memory_total_mib else 0.0

    @property
    def busy(self) -> bool:
        return (self.utilization_percent > GPU_BUSY_UTIL_PERCENT
                or self.memory_fraction > GPU_BUSY_MEMORY_FRACTION)


@dataclass
class NodeReport:
    hostname: str
    total_cores: int
    #: One-minute load average, used as the count of busy cores.
    load1: float
    #: Cores visible to this process (cgroup/taskset affinity), which on a
    #: scheduler-managed node is the real budget and may be far below
    #: ``total_cores``.
    affinity_cores: int
    memory_total_gib: float
    memory_available_gib: float
    gpus: List[Gpu] = field(default_factory=list)
    #: Recommendation and the rule that produced it.
    omp_threads: int = 1
    gpu_id: int | None = None
    reasons: List[str] = field(default_factory=list)


def _read_meminfo() -> Dict[str, float]:
    """Total and available memory in GiB, from /proc/meminfo (kB values)."""
    values = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = float(rest.split()[0]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return values


def query_gpus() -> List[Gpu]:
    """Read GPU state from ``nvidia-smi``; empty list if there is none.

    Failures are swallowed on purpose: the absence of a GPU, a driver
    mismatch, and a node where ``nvidia-smi`` hangs should all end in "assume
    no GPU" rather than in a traceback that blocks a CPU-only build.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    query = "index,name,memory.used,memory.total,utilization.gpu"
    try:
        out = subprocess.run(
            [binary, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus: List[Gpu] = []
    for line in out.strip().splitlines():
        parts = [field.strip() for field in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(Gpu(index=int(parts[0]), name=parts[1],
                            memory_used_mib=float(parts[2]),
                            memory_total_mib=float(parts[3]),
                            utilization_percent=float(parts[4])))
        except ValueError:
            continue
    return gpus


def survey(max_fraction: float = DEFAULT_MAX_FRACTION,
           reserve_cores: int = DEFAULT_RESERVE_CORES) -> NodeReport:
    """Measure the node and apply the policy in this module's docstring."""
    total = os.cpu_count() or 1
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        affinity = total
    try:
        load1 = os.getloadavg()[0]
    except OSError:  # pragma: no cover
        load1 = 0.0
    mem = _read_meminfo()
    gpus = query_gpus()

    report = NodeReport(
        hostname=os.uname().nodename,
        total_cores=total,
        load1=load1,
        affinity_cores=affinity,
        memory_total_gib=mem.get("MemTotal", 0.0),
        memory_available_gib=mem.get("MemAvailable", 0.0),
        gpus=gpus,
    )

    budget = min(affinity, total)
    ceiling = max(1, int(budget * max_fraction))
    report.reasons.append(
        f"{budget} core(s) in this process's budget; {max_fraction:.0%} cap -> {ceiling}")

    idle = int(max(0.0, budget - load1)) - reserve_cores
    if idle < ceiling:
        report.reasons.append(
            f"load1 {load1:.1f} and a {reserve_cores}-core reserve leave {max(idle, 0)}")
    threads = max(1, min(ceiling, idle))

    busy = [gpu for gpu in gpus if gpu.busy]
    if busy:
        capped = min(threads, BUSY_GPU_NODE_THREAD_CAP)
        report.reasons.append(
            f"GPU(s) {', '.join(str(gpu.index) for gpu in busy)} busy -- someone "
            f"else's GPU job needs CPU feeders, so capping at "
            f"{BUSY_GPU_NODE_THREAD_CAP} -> {capped}")
        threads = capped
    report.omp_threads = threads

    free = [gpu for gpu in gpus if not gpu.busy]
    if free:
        report.gpu_id = free[0].index
        report.reasons.append(
            f"GPU {free[0].index} idle ({free[0].utilization_percent:.0f}% util, "
            f"{free[0].memory_fraction:.1%} memory) -> usable")
    elif gpus:
        report.reasons.append("every GPU is in use -> CPU only")
    else:
        report.reasons.append("no GPU detected -> CPU only")
    return report


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-fraction", type=float, default=DEFAULT_MAX_FRACTION)
    parser.add_argument("--reserve-cores", type=int, default=DEFAULT_RESERVE_CORES)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--omp-only", action="store_true",
                        help="print only the recommended thread count")
    args = parser.parse_args(argv)

    report = survey(args.max_fraction, args.reserve_cores)
    if args.omp_only:
        print(report.omp_threads)
        return 0
    if args.json:
        print(json.dumps(asdict(report), indent=2))
        return 0

    print(f"node        {report.hostname}")
    print(f"cores       {report.total_cores} total, {report.affinity_cores} in "
          f"affinity, load1 {report.load1:.2f}")
    print(f"memory      {report.memory_available_gib:.1f} of "
          f"{report.memory_total_gib:.1f} GiB available")
    if report.gpus:
        for gpu in report.gpus:
            state = "BUSY" if gpu.busy else "idle"
            print(f"gpu {gpu.index}       {gpu.name}: {state}, "
                  f"{gpu.utilization_percent:.0f}% util, "
                  f"{gpu.memory_used_mib:.0f}/{gpu.memory_total_mib:.0f} MiB")
    else:
        print("gpu         none detected")
    print(f"\nrecommend   omp_threads={report.omp_threads} "
          f"gpu_id={'null' if report.gpu_id is None else report.gpu_id}")
    for reason in report.reasons:
        print(f"            - {reason}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
