#!/bin/bash
#SBATCH --account=vnc@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --time=20:00:00
#SBATCH --job-name=sdf_n4_graph_ce
#SBATCH --output=sdf_n4_graph_ce_%j.log
#SBATCH --error=sdf_n4_graph_ce_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48

set -euo pipefail

# n4k2 means 4 MoE experts with top-2 routing; every preset uses 4 GPUs.
SDF_NUM_EXPERTS="${SDF_NUM_EXPERTS:-4}"
SDF_GRAPH_MODE="${SDF_GRAPH_MODE:-ce}"
SDF_RADIUS="${SDF_RADIUS:-False}"
source /lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM/training_script/common.sh "$@"
