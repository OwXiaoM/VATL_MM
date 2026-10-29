#!/bin/bash
# Default submits the one-subject diagnostic. --ablations is explicit.
set -eo pipefail
cd "$(dirname "$0")/../../.."
: "${SDF_STUDY:?Set SDF_STUDY to a prepared study directory}"
export SDF_STUDY
python -c 'import json,os; p=json.load(open(os.path.join(os.environ["SDF_STUDY"], "study.json"))); assert p.get("architecture")=="vatl_sdf_atlas_rigid_v2", "Prepare a NEW atlas-rigid v2 study"'
mode="${1:---initial}"
case "$mode" in
    --initial) experiments=(single_sdf) ;;
    --ablations) experiments=(single_aux small_reference small_uniform50 small_less_interior small_rotvec small_no_eikonal) ;;
    *) echo 'Usage: submit.sh [--initial|--ablations]'; exit 2 ;;
esac
for experiment in "${experiments[@]}"; do
    [[ ! -e "$SDF_STUDY/$experiment" ]] || { echo "Output already exists: $experiment"; exit 1; }
    if [[ -f "$SDF_STUDY/jobs.tsv" ]] && awk -F '\t' -v name="$experiment" '$1==name {found=1} END {exit !found}' "$SDF_STUDY/jobs.tsv"; then
        echo "Already submitted: $experiment (see jobs.tsv)"; exit 1
    fi
done
for experiment in "${experiments[@]}"; do
    job_id="$(sbatch --parsable --job-name="sdf_$experiment" --output="$SDF_STUDY/${experiment}_%j.log" scripts/decomposition/ablations/run.slurm "$experiment")"
    printf '%s\t%s\n' "$experiment" "$job_id" | tee -a "$SDF_STUDY/jobs.tsv"
done
