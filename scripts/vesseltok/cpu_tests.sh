#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest vesseltok.tests vesseltok.test_gap_sampling vesseltok.test_gap16 vesseltok.test_break_audit vesseltok.test_partial_volume -v
python -m compileall -q vesseltok diagnostics
