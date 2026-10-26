#!/bin/bash
#SBATCH -p l40s
#SBATCH --time=24:00:00
#SBATCH --job-name=sdf_n4_graph_sdf_radius
#SBATCH --output=sdf_n4_graph_sdf_radius_%j.log
#SBATCH --error=sdf_n4_graph_sdf_radius_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

# n4k2 means 4 MoE experts with top-2 routing; every preset uses 4 GPUs.
SDF_NUM_EXPERTS="${SDF_NUM_EXPERTS:-4}"
SDF_GRAPH_MODE="${SDF_GRAPH_MODE:-sdf}"
SDF_RADIUS="${SDF_RADIUS:-True}"
source /users/zhangx/VATL_MM/training_script/common.sh "$@"
