#!/bin/bash
#SBATCH -p l40s
#SBATCH --time=24:00:00
#SBATCH --job-name=sdf_n8_graph_both
#SBATCH --output=sdf_n8_graph_both_%j.log
#SBATCH --error=sdf_n8_graph_both_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

# n8k2 means 8 MoE experts with top-2 routing; every preset uses 4 GPUs.
SDF_NUM_EXPERTS="${SDF_NUM_EXPERTS:-8}"
SDF_GRAPH_MODE="${SDF_GRAPH_MODE:-both}"
SDF_RADIUS="${SDF_RADIUS:-False}"
source /users/zhangx/VATL_MM/training_script/common.sh "$@"
