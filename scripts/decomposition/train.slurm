#!/bin/bash
#SBATCH --job-name=sdf_decomposition
#SBATCH --account=vnc@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=sdf_decomposition_%j.log
set -eo pipefail
SDF_REPO="${SDF_REPO:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM}"
cd "$SDF_REPO"
export SDF_GPU_ARCH=h100
source scripts/decomposition/environment.sh
python scripts/jean_zay/check_gpu.py --expected-arch h100 --gpus 4
srun --ntasks=1 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  train_decomposition.py --config "${SDF_CONFIG:-configs/decomposition/full.yaml}" \
  --output "${SDF_OUTPUT:-../training_outputs/decomposition_${SLURM_JOB_ID}}" "$@"
