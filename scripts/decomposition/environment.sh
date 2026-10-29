#!/bin/bash
# Source on Jean Zay. Preserve queued A100 jobs by detecting their Slurm account.
if [[ -z "${SDF_GPU_ARCH:-}" ]]; then
    case "${SLURM_JOB_ACCOUNT:-}" in
        *@a100) SDF_GPU_ARCH=a100 ;;
        *) SDF_GPU_ARCH=h100 ;;
    esac
fi
case "$SDF_GPU_ARCH" in
    a100|h100) ;;
    *) echo "SDF_GPU_ARCH must be a100 or h100" >&2; return 1 ;;
esac
export SDF_GPU_ARCH
SDF_ENV="${SDF_ENV:-/lustre/fswork/projects/rech/vnc/uil92qd/envs/vatl_torch}"
module purge
module load "arch/${SDF_GPU_ARCH}"
module load pytorch-gpu/py3/2.8.0
if [[ "$SDF_GPU_ARCH" == h100 ]]; then
    # The existing venv's interpreter is pinned to the A100 base installation.
    # H100's MKL 2024 cannot load that torch build (iJIT_NotifyEvent failure).
    # Use the H100 module interpreter, adding only project-local dependencies.
    SDF_MODULE_PYTHON="$(command -v python)"
    if [[ "$SDF_MODULE_PYTHON" == "$SDF_ENV/bin/"* ]]; then
        echo "H100 requires the module Python, not the A100 venv interpreter" >&2
        return 1
    fi
    if [[ -d "$SDF_ENV/lib/python3.12/site-packages/torch" ]]; then
        echo "SDF_ENV contains its own torch; use a dependency-only environment for H100" >&2
        return 1
    fi
    export PATH="$(dirname "$SDF_MODULE_PYTHON"):$PATH"
    export PYTHONPATH="$SDF_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
else
    export PATH="$SDF_ENV/bin:$PATH"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
