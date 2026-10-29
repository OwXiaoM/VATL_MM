# Reproducible native-space validation

The existing single-GPU `AtlasBuilder` and distributed `AtlasBuilderDDP` now use
one metric implementation (`native_evaluation.py`, through the public functions
in `utils.py`). Training losses, model architecture, learned STN mappings and the
latent/test-time adaptation algorithm are unchanged. The optional missing
`subject_to_atlas` exporter is imported only when enabled, allowing the existing
single-GPU builder to import without that optional module.

## Cohort definition

`data_loading/dataset.py` now treats `val` specially. It applies the existing
metadata/modality/constraint checks, additionally requires its target files to
exist, restores the **predefined validation split's ordering**, and takes the
first `n_subjects.val` eligible IDs. Metadata TSV row order and random seed cannot
change the selection. It does not use age-stratified or random sampling for val.
Training sampling is unchanged. Ambiguous duplicate IDs and insufficient eligible
subjects fail explicitly; `n_subjects.val` is not silently reduced.

The ordered ID list is logged and written as a JSON array to
`<experiment>/val/subject_ids.json`, even when only image export is requested.
Set `evaluation.subject_manifest` to a previous run's JSON to enforce an exact
ordered match. A missing/ineligible required ID, changed N/order, or an attempt to
overwrite an existing experiment manifest with another cohort fails clearly.
Ordinary eligibility filtering can skip candidates; a manifest prevents a changed
file/metadata availability from silently substituting different comparison subjects.
Use the same manifest for SDF/non-SDF variants, since their target requirements
can differ.

## Geometry and image metrics

The native-grid generator is unchanged and still applies the learned STN during
inference. No additional registration is estimated. `save_subject()` no longer
calls ANTs at epoch zero; it exports untouched native references with the same
filenames. The legacy `reg_imgs()` tuple interface is retained but now performs
only geometry-based resampling and rejects estimated-transform dictionaries.

Matching shapes/affines are compared directly. Otherwise prediction values are
resampled onto the **complete reference grid**, using nearest neighbour for hard
segmentation and linear interpolation for intensities and continuous SDF values.
There is no reference-foreground or vessel bounding-box crop. Values outside the
prediction grid use zero intensity/background or a positive SDF (outside vessel).
The full modality reference grid is the evaluation domain. This cannot measure
anatomy outside the reference field of view; keep the same reference data across
experiments. Existing model-side reconstruction masking remains unchanged and
its setting is included in the saved experiment configuration.

MRA and T1 have separately named PSNR/SSIM columns. The default
`evaluation.intensity_normalization: none` compares stored reference values with
decoder values without independently rescaling either image. Use the same
preprocessed reference scale and fixed `data_range` (default 1.0) across models.
The explicit `independent_minmax` option retains legacy per-volume normalization
when required for old datasets, but can hide scale errors and is a different
protocol. Old scores using independent normalization and foreground cropping
should be recomputed before comparison with this protocol.

SSIM uses an odd window up to 7 voxels; a dimension below 3 makes SSIM undefined.
Exactly identical images have positive-infinite PSNR, reported in CSV and with
`positive_infinite_count` in the summary; infinities are excluded from finite
means/stds. Do not interpret a null finite mean as failed reconstruction in that
case. Invalid/nonfinite images and geometry receive explicit status/error fields.

Spatial units are converted from NIfTI mm/metre/micron into mm before resampling
and distances. Unknown units are rejected by default. Only use
`evaluation.assume_mm_if_unknown: true` when the upstream affine units have been
independently established as mm; that decision is recorded in the configuration.

## Segmentation and distance metrics

The existing multiclass Dice definition is retained: macro average over the union
of present non-background labels. Empty foreground unions are now undefined,
not perfect. The vessel ID is looked up by `label_names.index('Vessel')`, not
hardcoded to one.

The segmentation-channel mask and, when present, continuous `SDF < 0` mask are
evaluated independently against the **same reference vessel segmentation**:

- `Seg_Vessel_Dice`, `Seg_Vessel_HD95_mm`, `Seg_Vessel_ASD_mm`
- `SDF_Vessel_Dice`, `SDF_Vessel_HD95_mm`, `SDF_Vessel_ASD_mm`

Surface distances use boundary voxel centers mapped through the full NIfTI
affine and nearest-neighbour searches in physical mm. They support oblique and
sheared grids. HD95 is the maximum of the two directed 95th percentiles; ASD is
the mean of both directed surface-point distance arrays weighted by point count.
These are voxel-surface measures, not triangle-mesh surface-area measures.

Both masks empty: Dice/HD95/ASD are null with `both_empty` status. One empty:
Dice=0, distances null with prediction/reference-empty status. Absent or missing
SDF channels and invalid geometry are explicitly marked, not replaced by perfect
scores. `utils.compute_dice()` follows the same empty-foreground policy.

## Timing, output and provenance

Validation metrics are computed once after the existing `epochs.val` adaptation
loop, not on optimization iterations. `save_imgs.val` and `compute_metrics` are
independent: either or both can be enabled. Existing NIfTI names/checkpoint formats
are preserved. Rank zero alone reconstructs, writes manifests/metrics, and logs
metric summaries; other DDP ranks retain the existing synchronization behavior.

Each evaluated split contains:

- `subject_ids.json`: ordered subjects.
- `metrics.csv`: one row per subject, named metrics, statuses, checkpoint and
  adaptation/training epoch information.
- `metrics_summary.json`: finite mean, population std (`ddof=0`), valid/undefined/
  nonfinite counts, full experiment configuration and evaluation protocol.
- `epoch_NNNN/`: per-epoch subject list/CSV/summary copies to retain
  measurements from previous validation checkpoints. The split-root files are
  the most recent evaluation, and existing NIfTI names retain their old behavior.
- Historical `<split>_metrics_ep=N.json` and `<split>_df.csv` for compatibility.

A checkpoint path/existence flag is recorded, with an optional SHA256 controlled
by `evaluation.hash_checkpoint`. If no model file was saved, provenance explicitly
says the weights were in memory rather than claiming a checkpoint exists. The
standalone evaluation command below records the exact input checkpoint path
and retains the original checkpoint experiment configuration separately from
the evaluation overrides (including its original training seed).

The new policy makes cohorts and evaluation geometry deterministic. GPU kernel
bitwise determinism is not asserted. Compare identical adaptation seeds, epochs,
reference preprocessing and protocol settings; all are recorded. The standalone
command defaults to seed 42 for adaptation independently of the training seed.
The separate decomposition-specific inference command retains its own explicitly
selected-subject workflow; this cohort workflow targets the existing original
single-GPU/DDP validation builders.

## Commands

From the current VATL_MM checkout, lightweight CPU checks:

```bash
source scripts/decomposition/environment.sh
python -m unittest discover -s tests -p 'test_*.py' -v
# Metadata/file checks only; no patient volumes or weights loaded:
python scripts/evaluation/preview_subjects.py --subjects 3 \
  --output /tmp/vatl_cohort_preview
```

On Helix, activate its existing environment with `source ~/.bashrc1` and
`conda activate torch` instead of the Jean Zay module script.

To enable evaluation during existing DDP training, add the flat overrides file
`--experiment_config configs/evaluation.yaml` to `run_ddp.py` or a training preset:

```bash
sbatch training_script/train_ddp_n4k2_t1_sdf.sh \
  --experiment_config configs/evaluation.yaml \
  --evaluation__subject_manifest /path/reference_run/val/subject_ids.json
```

`configs/config_atlas.yaml` keeps `compute_metrics: false` as the baseline default;
`configs/evaluation.yaml` enables it. That file configures N=3, five adaptation
epochs and metrics without image saving. Override with `--n_subjects__val N`,
`--epochs__val K`, or `--save_imgs__val True`. The single-GPU builder uses the same
`compute_metrics`, `evaluation`, `n_subjects.val` and `save_imgs.val` YAML settings.
Its original CLI does not support the flat experiment-file flag.

For a saved original single-GPU/DDP checkpoint, this command restores its saved
architecture/data configuration, optimizes validation latents using the existing
algorithm, and evaluates without population-model training. **Run inside an
allocated GPU node**, from the repo root:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  scripts/evaluation/evaluate_checkpoint.py \
  --checkpoint /path/run/checkpoint_epoch_100.pth \
  --output /path/new_evaluation_run \
  --subjects 3 --adaptation-epochs 5 \
  --subject-manifest /path/reference_run/val/subject_ids.json
```

Omit `--subject-manifest` for the first experiment, then reuse its output manifest.
Add `--save-images` to export reconstruction NIfTIs. Output directories must be
new; checkpoints retain the `checkpoint_epoch_N.pth` filename. Decomposition-format
checkpoints are deliberately rejected by this original-pipeline entry point.
No GPU evaluations or large jobs were submitted during implementation.
