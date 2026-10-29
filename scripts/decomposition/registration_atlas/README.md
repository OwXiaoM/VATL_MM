# Four-subject vascular registration and population atlas experiment

The research objective is a continuous population-conditioned cerebrovascular
atlas, with separate evaluation of correspondence, population statistics,
individual reconstruction and future normative modelling. The simplified neural
model is unchanged. No residual, subject code, MRA encoder or graph model is added.

## Reference and differences

Xu et al., *Age-Specific Cerebral Vessel Templates Across the Lifespan of Healthy
Adults*, Scientific Data (2026), https://doi.org/10.1038/s41597-026-07710-5.
The downloaded open-access reference PDF is retained here. Structural alignment
uses T1-guided ANTs SyN; the paper's vascular refinement uses ANIMAL with nominal
16/8/4/2-mm steps, not SyN throughout. The paper uses FOV-aware probabilities,
group-wise updates with mean deformation correction, and optional re-segmentation
of warped MRA to recover thin vessels. These motivate the controls below, not a
claim of exact reproduction.

Our initial baseline uses available ANTsPy 0.6.3, conservative SyN and vascular
features. Optimization is at 2-mm resolution with four pyramid levels (nominal
16/8/4/2-mm image spacings); these are not ANIMAL control-grid step sizes. Final
products are on a shared 1-mm grid covering the union of centered native FOVs.
Grid effects are quantified separately and may justify a finer follow-up.

## Coordinate and transform conventions

- Input images retain voxel orientation and physical spacing. Each native affine's
  translation is shifted by `-affine(shape/2)`, matching existing model centering.
  This is a coordinate change, not an anatomical registration.
- R0 is this centered physical frame before learned neural registration. It does
  not use incompatible historical checkpoint transforms. Header-world anatomical
  standardization is not established by RAS axis labels.
- NIfTI uses RAS mm. Files are read by ANTs, which handles ITK's LPS mm convention.
  Never pass NIfTI RAS vectors directly as ANTs LPS displacement vectors.
- ANTs fwdtransforms are image-resampling lists from moving into fixed geometry;
  their sampling coordinates map target locations into the source. Inverse lists
  reverse the image resampling direction. Explicit inversion flags are recorded.
- For pull maps `old(increment(x))`, the ANTs list is `[increment, *old]`.
  Noncommuting affine tests validate list order and inverse order.
- R1 uses paired forward/inverse composites. R2 estimates a separate increment
  against prealigned optimization features and composes it explicitly with both
  structural directions. This avoids a missing inverse when ANTs loads a nonlinear
  H5 initializer. Incremental reapplication is checked against warpedmovout; final
  labels/intensities are resampled directly from original centered images.
- Full forward/inverse sampling displacement fields and unclamped physical
  Jacobian determinants are saved. Fold fraction uses `det <= 0`.

Coverage is a warped native image-of-ones, conservatively thresholded at >.999.
This is geometric FOV support. Original acquisition/padding masks and upstream
transform provenance are not supplied, so it cannot recover hidden missing data.
Probability is NaN where coverage is zero. Missing coverage is never a vessel
absence. Pairwise metrics are evaluated only in observed shared FOV; compare.py
also fixes coverage regions across all compared stages.

## Reproducible execution

All substantial image processing runs in CPU Slurm allocations. No GPU is requested.
Inputs and historical outputs are read-only. Each study and each subject phase
must use a new output directory. The fixed four training IDs are in prepare.py;
all are approximately age 20, insufficient for lifespan or sex-effect modelling.

```bash
python scripts/decomposition/registration_atlas/prepare.py \
  --output ../training_outputs/registration_atlas_NEW
export SDF_REG_STUDY="$(realpath ../training_outputs/registration_atlas_NEW)"
sbatch --time=00:20:00 --output="$SDF_REG_STUDY/audit_%j.log" \
  scripts/decomposition/registration_atlas/cpu.slurm audit
```

Inspect audit/audit.json, audit/synthetic_rigid.json, R0/metrics.json and overlays.
Then run native-image overlays and identity-resampling controls using qc.py as
SDF_REG_PIPELINE, through the same CPU launcher. Keep that script in a study code
snapshot when submitting. Record every returned Slurm ID in jobs.tsv.

After inspecting the audit:

```bash
python scripts/decomposition/registration_atlas/submit.py --study "$SDF_REG_STUDY" --stage R1
# Inspect R1/metrics.json, deformation, round-trip and anatomical overlays.
python scripts/decomposition/registration_atlas/submit.py --study "$SDF_REG_STUDY" --stage R2 --base R1
# If R1 worsens correspondence, use --base R0 instead and document the decision.
```

R1 uses paired slab-grid T1 images. Full native T1 availability is not presumed.
If paired files are absent the stage explicitly skips. R2 uses two positive
vessel-distance features (sigma 2 and 5 mm), starting from the chosen baseline.
Binary and Gaussian density controls are available via --feature binary/density,
but should be submitted only to answer a measured uncertainty, not as a blind sweep.
Transforms apply directly to original centered masks/MRA/T1, one final interpolation.
Masks use nearest neighbor; intensities and coverage use linear interpolation.
Aligned SDFs are recomputed from aligned masks with physical spacing.

```bash
source scripts/decomposition/environment.sh
python scripts/decomposition/registration_atlas/compare.py --study "$SDF_REG_STUDY" \
  --stages R0 R1 R2 --output "$SDF_REG_STUDY/comparison_R0_R1_R2"
```

Each submission snapshots its code and appends job IDs to jobs.tsv. Do not edit
code snapshots after submission; later phases get their own snapshots.

## Conditional group-wise experiment

After R2 inspection, the groupwise command runs fixed-original-observation
registration to updated FOV-weighted population proximity fields (2 and 5 mm),
with occupancy probability saved separately. It estimates a mean
incremental sampling displacement, inverts a damped version, applies the same shape
correction to every subject, updates coverage and probability, and repeats. This
is an approximate small-deformation mean-bias correction, not an exact unbiased
Fréchet atlas or a reproduction of the paper's ANIMAL procedure. Inspect Jacobians
and measured residual bias at every iteration. All masks resample from originals.

```bash
python scripts/decomposition/registration_atlas/submit.py --study "$SDF_REG_STUDY" \
  --stage groupwise --iterations 2
# For this completed study, the successful vascular stage is R2_incremental;
# use --vascular-stage R2_incremental when preparing another gated comparison.
```

Use --reference ID for an alternative probability-template initialization and a
new output. This tests initialization sensitivity within the existing structural
frame; it does not change R1's fixed structural reference. Convergence requires
both a small vessel-mass-normalized probability change and mean incremental warp
RMS <.05 mm, or stops at the specified iteration budget. No anatomical validity
claim follows from convergence alone.

## Evaluation and decision gates

- Registration: pairwise Dice/clDice, symmetric centerline distances; full and
  vessel-neighborhood Jacobians/displacements; inverse consistency; native
  round-trip mask recall and component/endpoint/bifurcation-cluster proxies.
- Atlas: geometric coverage counts, contributing vessel counts, FOV-weighted
  probability, centerline probability, concentration, disagreement and entropy.
  Concentration is also compared on identical all-stage/all-subject coverage.
- Preservation: identity resampling vs registration round-trip controls and native
  MRA/label overlays. Component proxies are not homologous branch counts.
- Anatomical QC: inspect slices as well as MIPs. Circle of Willis ROI and homologous
  bifurcation annotation require expert labeling; no arbitrary z slab is called CoW.
- Optional re-segmentation is skipped unless a compatible model is verified.
- A fresh matched v2 shared-SDF prealignment experiment is justified only after
  measurable correspondence gains pass deformation/preservation QC. An old 44.15%
  checkpoint is not a v2 baseline. No old checkpoint is resumed.
- Neural occupancy should fit Bernoulli observations with FOV masking and correct
  sampling weights, evaluated by likelihood/Brier/calibration and classical-map
  agreement, not only individual Dice. These neural phases remain gated.

Four subjects demonstrate registration behavior, not healthy population
probabilities, lifespan effects, normative deviations or preserved graph topology.

## Checks

```bash
source scripts/decomposition/environment.sh
python -m unittest discover -s tests -p test_registration_atlas.py
```

Results and pending decisions are in REPORT.md and the study output, with updates
recorded in CONTEXT.md. The current model implementation and training losses are
not modified by these registration experiments.

## Finer-grid export and neural training gate

`regrid.py --study SOURCE --output NEW_DIRECTORY --spacing .8` is a prepared
fixed-transform export for the original observations. It also writes multiclass
labels, recomputed SDFs, Coverage and per-stage metadata. It has not been run in
the initial study. Use the CPU launcher with this script as SDF_REG_PIPELINE and
the source study as SDF_REG_STUDY; choose a new output every time.

The existing trainer does not yet honor a Coverage column. Merely exporting
aligned SDFs is NOT sufficient to launch a valid controlled fit: all SDF, image,
tissue and graph sampling/losses must exclude unobserved support. That adaptation
is a pending prerequisite, not implemented by these registration scripts.
`training_gate.json` explicitly records this limitation.

Additional checks: `python scripts/decomposition/registration_atlas/smoke_groupwise.py`
uses tiny synthetic tubes in a temporary directory, including the mean correction
and inverse composition. It is a plumbing check; degenerate skeletons produce
explicit undefined metrics rather than NaNs.

## Authorized follow-up implementation

The finer-grid/coverage-aware follow-up now has an isolated driver and exact
submission order in [next_steps/ORDER.md](next_steps/ORDER.md). The new training
adapter is opt-in through `dataset.coverage_key`; it uses CPU-prepared trusted
point caches and never adds coverage as a target channel. Existing data-loader
behavior stays available when coverage_key is absent. The network architecture
has not changed. Read ORDER.md and the latest CONTEXT.md before submitting; the
historical statements above that Coverage support was pending describe the prior
registration turn, not the new opt-in implementation.
