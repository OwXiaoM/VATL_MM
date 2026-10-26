#!/bin/bash
#SBATCH -p l40s
#SBATCH --time=6:00:00
#SBATCH --job-name=sdf_ablation_A
#SBATCH --output=sdf_ablation_A_%j.log
#SBATCH --error=sdf_ablation_A_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

# n4k2 means 4 MoE experts with top-2 routing; every preset uses 4 GPUs.
SDF_NUM_EXPERTS="${SDF_NUM_EXPERTS:-4}"
SDF_GRAPH_MODE="${SDF_GRAPH_MODE:-ce}"
SDF_RADIUS="${SDF_RADIUS:-False}"
source /users/zhangx/VATL_MM/training_script/common.sh \
    --experiment_config /users/zhangx/VATL_MM/configs/ablations/A.yaml \
    --output_dir /medias/medical/caravel/zhangx/training_outputs/ablations/A "$@"
