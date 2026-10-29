#!/bin/bash
#SBATCH --job-name=sdf_decomposition
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=sdf_decomposition_%j.log
set -eo pipefail
source "$HOME/.bashrc1"
conda activate torch
cd "${SDF_REPO:-${SLURM_SUBMIT_DIR}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --ntasks=1 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  train_decomposition.py --config "${SDF_CONFIG:-configs/decomposition/full.yaml}" \
  --output "${SDF_OUTPUT:-../training_outputs/decomposition_${SLURM_JOB_ID}}" "$@"
