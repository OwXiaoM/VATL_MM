#!/bin/bash
# Shared launcher sourced by the SLURM presets in this directory.
set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Submit a train_ddp_*.sh preset with sbatch; common.sh is a shared launcher." >&2
    exit 1
fi

SDF_REPO="${SDF_REPO:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM}"
SDF_NPROC="${SDF_NPROC:-4}"
SDF_NUM_EXPERTS="${SDF_NUM_EXPERTS:-4}"
SDF_GRAPH_MODE="${SDF_GRAPH_MODE:-off}"
SDF_RADIUS="${SDF_RADIUS:-False}"

if [[ ! "$SDF_NUM_EXPERTS" =~ ^[1-9][0-9]*$ ]] || (( SDF_NUM_EXPERTS < 2 )); then
    echo "SDF_NUM_EXPERTS must be an integer >= 2 for top-2 routing." >&2
    exit 1
fi
case "${SDF_RADIUS,,}" in
    true|1|yes) SDF_RADIUS=True ;;
    false|0|no) SDF_RADIUS=False ;;
    *) echo "SDF_RADIUS must be True or False." >&2; exit 1 ;;
esac
case "$SDF_GRAPH_MODE" in
    off) SDF_GRAPH_ACTIVE=False; SDF_LOSS_MODE=sdf ;;
    ce|sdf|both) SDF_GRAPH_ACTIVE=True; SDF_LOSS_MODE="$SDF_GRAPH_MODE" ;;
    *) echo "SDF_GRAPH_MODE must be off, ce, sdf, or both." >&2; exit 1 ;;
esac
if [[ "$SDF_GRAPH_MODE" == off && "$SDF_RADIUS" == True ]]; then
    echo "Radius supervision requires SDF_GRAPH_MODE=ce, sdf, or both." >&2
    exit 1
fi

SDF_EXPERIMENT="n${SDF_NUM_EXPERTS}k2/graph_${SDF_GRAPH_MODE}"
if [[ "$SDF_RADIUS" == True ]]; then
    SDF_EXPERIMENT="${SDF_EXPERIMENT}_radius"
fi
SDF_OUTPUT_ROOT="${SDF_OUTPUT_ROOT:-/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/training_outputs/sdf}"

SDF_COMMAND=(
    srun --ntasks=1 python -m torch.distributed.run
    --standalone --nnodes=1 --nproc_per_node="$SDF_NPROC"
    run_ddp.py
    --config_data mra_atlas_synthseg_label_t1_fast_sdf
    --use_moe True --num_experts "$SDF_NUM_EXPERTS" --moe_k 2
    --inr_decoder__sdf_dim 1
    --optimizer__sdf_weight 1.0
    --optimizer__sdf_trunc_mm 5.0
    --dataset__sdf_sampling__num_points 500000
    --dataset__sdf_sampling__surface_band_mm 2.0
    --graph_supervision__activate "$SDF_GRAPH_ACTIVE"
    --graph_supervision__mode "$SDF_LOSS_MODE"
    --graph_supervision__points_per_subject 256
    --graph_supervision__weight 0.1
    --graph_supervision__sdf_weight 0.1
    --graph_supervision__radius__activate "$SDF_RADIUS"
    --graph_supervision__radius__weight 0.05
    --graph_supervision__radius__scale_to_mm 1.0
    --graph_supervision__radius__max_sdf_disagreement_mm 0.5
    --output_dir "${SDF_OUTPUT_ROOT}/${SDF_EXPERIMENT}"
    "$@"
)

echo "SDF experiment: $SDF_EXPERIMENT"
echo "Graph mode: $SDF_GRAPH_MODE; radius: $SDF_RADIUS; GPUs: $SDF_NPROC"
printf 'Command:'
printf ' %q' "${SDF_COMMAND[@]}"
printf '\n'

# Inspect commands without loading the training environment or launching work.
if [[ "${SDF_DRY_RUN:-0}" == 1 ]]; then
    exit 0
fi

# Shared module setup selects H100, or A100 for an already queued A100 job.
source "$SDF_REPO/scripts/decomposition/environment.sh"
export SDF_DATA_ROOT="${SDF_DATA_ROOT:-/lustre/fsn1/projects/rech/vnc/uil92qd/datasets/data/v1}"
if [[ ! -d "$SDF_DATA_ROOT" ]]; then
    echo "Dataset directory missing: $SDF_DATA_ROOT" >&2
    exit 1
fi
cd "$SDF_REPO"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PYTHONUNBUFFERED=1

echo "Job: ${SLURM_JOB_ID:-interactive}; node: $(hostname)"
"${SDF_COMMAND[@]}"
