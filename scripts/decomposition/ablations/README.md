# Atlas and rigid registration diagnostics

The active network is a shared SDF atlas plus physical rigid registration.
The subject residual network, subject codes, their losses and their experiments
were removed at the user's request on 2026-10-28. Training now optimizes atlas
and rigid transforms jointly; there is no residual fitting stage.

The 20261027 study is historical. Its non-residual-specific results remain for
comparison; residual-focused configurations, outputs and logs have been deleted.
`jobs.tsv` and `CONTEXT.md` retain submission provenance and measured findings.
Do not append new runs to that study. New checkpoints use
`vatl_sdf_atlas_rigid_v2`; residual-era checkpoints cannot be resumed or evaluated
by the current model. They have not been silently converted to a different model.

## Prepare a fresh study

```bash
source scripts/decomposition/environment.sh
python scripts/decomposition/ablations/prepare.py \
  --checkpoint ../training_outputs/decomposition_1590276/checkpoint_epoch_99.pth \
  --output ../training_outputs/diagnostic_study_atlas_rigid_NEW --epochs 500
export SDF_STUDY="$(realpath ../training_outputs/diagnostic_study_atlas_rigid_NEW)"
bash scripts/decomposition/ablations/submit.sh --initial
```

The source checkpoint supplies cohort and configuration only. All prepared fits
start from scratch. `--initial` submits single_sdf only. Inspect it before further
submission. Each job uses one H100 in Slurm and creates a new output directory.
A longer epoch budget can be selected with `--epochs`.

| Experiment | Comparison |
|---|---|
| single_sdf | One-subject SDF capacity control, registration disabled |
| single_aux | Adds image and tissue losses to single_sdf |
| small_reference | Four subjects, joint atlas/rigid, original auxiliary terms |
| small_uniform50 | Half of SDF samples uniform within the valid crop |
| small_less_interior | Interior/surface/tissue/background fractions .05/.35/.40/.20 |
| small_rotvec | Rotation vector instead of legacy ZXZ Euler |
| small_no_eikonal | Removes Eikonal loss |

Use the NEW small_reference as the comparator for small_* runs. Old residual-era
reference scores are historical context, not a matched control. Uniform sampling
changes the objective and does not implement inverse probability weighting.

Submit one experiment after inspecting the preceding results:

```bash
sbatch --output="$SDF_STUDY/small_reference_%j.log" \
  scripts/decomposition/ablations/run.slurm small_reference
python scripts/decomposition/ablations/summarize.py --study "$SDF_STUDY"
```

The batch `--ablations` option remains available for an explicitly chosen broad
screen; the recommended next investigation is in `../NEXT_STEPS.md`.

Full-grid training reconstruction is monitored after epoch 1, every 50 epochs,
and at the final epoch. Truth outside crop is counted. Tables and curves report
Dice, precision, recall and predicted/reference volume ratio. Best-training-Dice
checkpoints are capacity diagnostics, not validation-selected models.

Held-out evaluation fits rigid parameters to ground-truth SDF samples:

```bash
python scripts/decomposition/ablations/evaluate.py \
  --checkpoint NEW_V2_CHECKPOINT --split val --subjects 3 --steps 100 \
  --output NEW_OUTPUT
```

This is an oracle diagnostic, not image-only inference. The current adapter does
not optimize a loss on MRA intensities. Keep final test subjects separate from
tuning. Legacy Euler remains the default; never reinterpret saved Euler angles
as rotation vectors.

## Verification

```bash
python -m unittest discover -s tests -p 'test*decomposition*.py'
python -m unittest discover -s tests -p 'test_validation_pipeline.py'
python scripts/decomposition/smoke_pipeline.py --diagnostics --training-only --resume-check
```
