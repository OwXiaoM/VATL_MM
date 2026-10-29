#!/bin/bash
#SBATCH --job-name=sdf_validation
#SBATCH --account=vnc@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=sdf_validation_%j.log
set -eo pipefail
SDF_REPO="${SDF_REPO:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM}"
cd "$SDF_REPO"
export SDF_GPU_ARCH=h100
source scripts/decomposition/environment.sh
: "${SDF_CHECKPOINT:?Set SDF_CHECKPOINT to an atlas-rigid v2 checkpoint}"
CHECKPOINT="$SDF_CHECKPOINT"
srun --ntasks=1 python scripts/decomposition/validate.py \
  --checkpoint "$CHECKPOINT" \
  --output "${SDF_OUTPUT:-../training_outputs/validation_${SLURM_JOB_ID}}" "$@"
