#!/bin/bash
#SBATCH --job-name=atlas_coverage_fit
#SBATCH --account=vnc@h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=atlas_coverage_%j.log
set -euo pipefail
: "${SDF_FOLLOWUP:?Set isolated follow-up directory}"
: "${SDF_REPO:?Use the frozen repository snapshot}"
cd "$SDF_REPO"
export SDF_GPU_ARCH=h100
source scripts/decomposition/environment.sh
python scripts/jean_zay/check_gpu.py --expected-arch h100 --gpus 1
phase="${1:?Pass existing, prealigned, or evaluate}"
python -c 'import json,os; p=json.load(open(os.path.join(os.environ["SDF_FOLLOWUP"],"preflight/gate.json"))); assert p["passed"], "Coverage/preservation gate closed"'
case "$phase" in
 existing|prealigned)
  srun --ntasks=1 python train_decomposition.py --config "$SDF_FOLLOWUP/configs/$phase.yaml" --output "$SDF_FOLLOWUP/fits/$phase"
  ;;
 evaluate)
  srun --ntasks=1 python scripts/decomposition/registration_atlas/next_steps/evaluate_pair.py --study "$SDF_FOLLOWUP"
  ;;
 *) echo 'Unknown phase'; exit 2 ;;
esac
