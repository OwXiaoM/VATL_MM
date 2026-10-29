#!/bin/bash
#SBATCH --job-name=vascular_atlas
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=vascular_atlas_%j.log
set -euo pipefail
SDF_REPO="${SDF_REPO:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM}"
cd "$SDF_REPO"
source scripts/decomposition/environment.sh
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1
: "${SDF_REG_STUDY:?Set SDF_REG_STUDY to isolated prepared directory}"
srun --ntasks=1 python "${SDF_REG_PIPELINE:-$SDF_REG_STUDY/source/pipeline.py}" --study "$SDF_REG_STUDY" "$@"
