# Decomposition checkpoint validation

The decomposition training loop currently has no validation call. The original
pipeline's evaluate_checkpoint.py cannot read decomposition-format checkpoints.
These scripts evaluate them after training without updating population weights.

From VATL_MM, submit on H100:

```bash
SDF_CHECKPOINT=/absolute/path/NEW_V2_CHECKPOINT sbatch scripts/decomposition/validate.slurm
```

Set SDF_CHECKPOINT to a vatl_sdf_atlas_rigid_v2 checkpoint. Old residual-era
checkpoints require their original implementation.
The default run selects the first 3 eligible subjects in the checkpoint's ordered
validation split, rejects overlap with the checkpoint's training IDs, and fits
a new rigid transform using 8192 observed SDF points for 100
optimization steps per subject, with seed 42. This is held-out reconstruction
AFTER subject-specific fitting to ground-truth SDF, not label-free prediction.
Population network weights stay frozen. Adaptation steps are not training epochs.

For more subjects or fitting steps:

```bash
SDF_CHECKPOINT=/absolute/path/NEW_V2_CHECKPOINT sbatch scripts/decomposition/validate.slurm --subjects 10 --adapt-steps 300
```

Use validate_a100.slurm instead for A100. To change the checkpoint/output:

```bash
SDF_CHECKPOINT=/absolute/path/checkpoint.pth SDF_OUTPUT=/absolute/path/new_validation SDF_CHECKPOINT=/absolute/path/NEW_V2_CHECKPOINT sbatch scripts/decomposition/validate.slurm
```

The output directory must be new. The default is
../training_outputs/validation_JOBID; the Slurm log is sdf_validation_JOBID.log
in the submission directory. No jobs are submitted by the Python command itself.

Outputs:
- val/subject_ids.json: exact held-out cohort. Reuse via --subject-manifest PATH
  together with the same --subjects count for future comparisons.
- protocol.json and validation_metadata.tsv: checkpoint, cohort and fitting settings.
- metrics.csv and metrics_summary.json: per-subject values and finite means/stds/counts.
- subject_000/, etc.: registered atlas/final SDF, vessel mask, valid crop and adaptation state.

Vessel metrics compare the final SDF-zero mask against the reference segmentation's
Vessel class on the FULL segmentation reference grid. Outside the model crop is
background; valid_crop_fraction records crop coverage on the SDF output grid.
Dice, physical HD95 and ASD support general affines. For orthogonal grids the
command also computes clDice, surface Dice (1 mm), centerline recall, connected
components and Euler errors. Empty masks produce explicit status and undefined
surface distances; summary counts should be checked. NIfTI spatial units must be
known. This command does not evaluate decoder segmentation logits or intensity
PSNR/SSIM. Exported continuous SDF has NaNs outside the modeled crop; mask export
uses background there. The infer_decomposition.py --metrics option compares against the full
SDF reference rather than the segmentation reference; this validator deliberately does not
use that option.

A run that fails partway retains completed subject outputs and metrics. Choose a
new output directory when retrying. Start with 3 subjects to assess runtime and
memory before increasing the cohort. This does not add validation during training.
