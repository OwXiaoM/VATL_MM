#!/bin/bash
#SBATCH --job-name=sdf_diagnostic
#SBATCH --account=vnc@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=sdf_diagnostic_%j.log
set -eo pipefail
SDF_REPO="${SDF_REPO:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM}"
cd "$SDF_REPO"
export SDF_GPU_ARCH=h100
source scripts/decomposition/environment.sh
: "${SDF_STUDY:?Set SDF_STUDY to a prepared study directory}"
experiment="${1:?Pass an experiment name or existing_train}"
python scripts/jean_zay/check_gpu.py --expected-arch h100 --gpus 1
python -c 'import json,os; p=json.load(open(os.path.join(os.environ["SDF_STUDY"], "study.json"))); assert p.get("architecture")=="vatl_sdf_atlas_rigid_v2", "Prepare a NEW atlas-rigid v2 study; historical study outputs must stay unchanged"'
[[ "$experiment" =~ ^[a-z0-9_]+$ ]] || { echo 'Invalid experiment name'; exit 1; }
config="$SDF_STUDY/configs/$experiment.yaml"
[[ -f "$config" ]] || { echo "Missing config: $config"; exit 1; }
srun --ntasks=1 python train_decomposition.py --config "$config" --output "$SDF_STUDY/$experiment"
