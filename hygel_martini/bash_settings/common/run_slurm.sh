#!/bin/bash
#SBATCH -J qm_to_martini
#SBATCH -p gpupart
#SBATCH -o ./out/%x.o.%j.out
#SBATCH -e ./out/%x.e.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
##SBATCH --nodelist=nanode04
#SBATCH -t 100000:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
DEFAULT_CONFIG_REL="${DEFAULT_CONFIG_REL:-config_common/common.yaml}"
USE_STAGE_SRUN="${USE_STAGE_SRUN:-0}"

# One qm_to_martini case is a single Slurm task that uses threads.
# Adjust partition / time / cpus-per-task above for your cluster.
# Set DEFAULT_CONFIG_REL or pass a config path argument for a different default config.

CONFIG_ARG="${1:-$DEFAULT_CONFIG_REL}"
if [ $# -gt 0 ]; then
  shift
fi

if [[ "$CONFIG_ARG" = /* ]]; then
  CONFIG_PATH="$CONFIG_ARG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG_ARG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/out"
cd "$SCRIPT_DIR"

# Default behavior is to stay inside the batch allocation and let the
# generated stage scripts inherit SLURM_CPUS_PER_TASK. Set USE_STAGE_SRUN=1
# if you explicitly want each generated stage to re-enter through srun.
bash ./run_qm_to_martini.sh "$CONFIG_PATH" \
  --set bartender_pipeline.execution.slurm=true \
  --set "bartender_pipeline.execution.use_srun=${USE_STAGE_SRUN}" \
  --set bartender_pipeline.execution.shell=bash \
  "$@"
