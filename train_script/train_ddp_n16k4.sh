#!/bin/bash
#SBATCH --job-name=vatl_a100_train
#SBATCH --account=vnc@a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_MM/logs/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_MM/logs/%x_%j.err

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0

source $WORK/envs/vatl_torch/bin/activate

cd $WORK/projects/VATL_MM

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

nvidia-smi

$WORK/envs/vatl_torch/bin/python -u -c "import torch; import ants; print('torch:', torch.__version__); print('ants ok'); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0))"

$WORK/envs/vatl_torch/bin/python -m torch.distributed.run --standalone \
    --nproc_per_node=1 \
    run_ddp.py \
    --config_data mra_atlas \
    --use_moe True \
    --num_experts 16 \
    --moe_k 4