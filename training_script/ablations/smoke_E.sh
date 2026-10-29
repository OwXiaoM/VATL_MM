#!/bin/bash
# CPU only: no sbatch, GPU allocation, or volume generation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/decomposition/environment.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/vatl_mm_mpl"
python -m unittest discover -s tests -p 'test_experiment_e.py' -v
