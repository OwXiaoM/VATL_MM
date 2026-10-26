#!/bin/bash
# CPU only: no sbatch, GPU allocation, or volume generation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set +eu
source "$HOME/.bashrc1"
conda activate torch
activation_status=$?
set -euo pipefail
(( activation_status == 0 )) || exit "$activation_status"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vatl_mm_mpl"
python -m unittest discover -s tests -p 'test_experiment_e.py' -v
