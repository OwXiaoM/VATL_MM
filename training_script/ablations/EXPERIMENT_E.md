# Experiment E: independent image/tissue and vessel-SDF sampling

E tests whether dense geometric supervision can coexist with global multimodal
reconstruction. It is a hypothesis test; smoothness alone is not success.

## Audited baseline and implementation

Saved C reference:
`/medias/medical/caravel/zhangx/training_outputs/ablations/C/mra_atlas_synthseg_label_t1_fast_sdf_20261025_202737_701/config_final.yaml`.
Saved D reference:
`/medias/medical/caravel/zhangx/training_outputs/ablations/D/mra_atlas_synthseg_label_t1_fast_sdf_20261026_022819_702/config_final.yaml`.
The effective E launcher configuration was compared against saved C: only
`task_specific_sampling`, experiment filename and output directory differ.
The configuration regression test also compares the resolved C/E launchers.
Recheck saved configurations if defaults or subject split files change later.

| Experiment | Image/tissue coordinates | Voxel SDF coordinates | Weight |
|---|---|---|---|
| C | Original, 500,000/subject | Same 500,000 | 0.1 |
| D | Existing SDF-aware, 500,000/subject | Same 500,000 | 0.1 |
| E | Original, 500,000/subject | Independent vessel-aware, 50,000 | 0.1 |

Dataset items in E contain separate `image` and `sdf` tuples. Each carries its
own coordinates, targets, conditions and subject indices. One augmentation is
shared. The existing image sampler and NumPy RNG consumption are preserved;
SDF sampling/shuffling use separate local RNGs. Cross-branch coordinate overlap
is allowed: independence does not require excluding image locations from SDF.
Uniqueness statistics are per subject and branch, not across the whole epoch.

The collator shuffles the two branches independently. Training partitions the
SDF batch proportionally across the existing image microsteps, concatenates
image/SDF/graph coordinates, conditions and subject indices in that order, and
uses the same indexed latent and transformation lookup. Graph sampling still
uses the subjects in the image microstep. Its normalization, class label and
weight are unchanged. There is one decoder forward and one backward per step.

Only image points receive MRA/T1 L1 and tissue CE. Only independent SDF points
receive voxel SDF L1. Graph points receive the existing graph CE. Output-gradient
tests verify all three boundaries. All loss terms have separate mean reduction;
SDF loss is not divided by the combined coordinate count. SDF gradients still
reach shared SIREN/MoE, modulation, transformations and subject latents.
Architecture, checkpoint tensors, exports, mean-latent aggregation and masking
are unchanged. Disabling `task_specific_sampling.enabled` retains A–D behavior.
Validation remains the existing image-only latent adaptation, never E sampling.

Saved C settings retained: four experts/top-2; age/sex conditioning; batch size
20; image microstep 100,000; seed 42; 1000 training epochs; five validation
adaptation epochs; cosine schedule with eta_min 2.5e-5; decoder LR 1e-4;
latent LR 5e-4; transformation LR 1e-4; tissue weight 1; voxel SDF weight 0.1;
graph CE weight 0.1; MoE weight 0.01; transformation penalty weight **0.0**;
SDF truncation 5 mm; graph SDF/radius off; augmentation off.
Do not shorten the scheduler horizon to 30: select checkpoint 30 from the same
1000-epoch schedule. Existing checkpoint numbering is zero-based; use the
literal `checkpoint_epoch_30.pth` consistently.

## Disjoint SDF pools and fallback

Physical-mm distances define mutually exclusive pools:

- Interior: SDF < 0, including thin-vessel interiors (25%).
- Surface: 0 <= SDF <= 2 mm, the **outside** zero-level band (50%).
- Exterior: SDF > 2 mm, including tissue and background (25%).

This deliberately uses a one-sided surface band so negative near-surface
voxels cannot be counted twice. A symmetric surface band plus interior below
-2 mm was rejected after real-data inspection: the first two subjects have
only 308/585 deep-interior voxels, incompatible with a unique 12,500-point
quota. All negative voxels are therefore prioritized as interior.

All pools are cropped with the same physical center/bounding box as the image
and graph samplers. Sample each quota without replacement; if a pool is small,
use every available voxel, then fill from all unused in-bounds voxels. Only
when the entire valid volume is exhausted is repetition allowed. Fallback can
change the requested fractions; actual counts and pool sizes are logged.
The surface band is not a topology or centerline guarantee.

All three checked subjects provided the exact 25,000 surface / 12,500 interior /
12,500 exterior counts with 100% SDF uniqueness. Image uniqueness was
99.3762–99.4742%, with original label distributions unchanged. Surface pools
contained 89,322–100,586 voxels; interior pools 18,198–19,503.

Three real-subject CPU QC results are stored in
`diagnostics/E_sampling_qc.json`: image anatomical label histograms, exact unique
counts, pool sizes, requested/actual SDF region counts. These are representative
checks, not certification of the entire cohort.

## Budget and diagnostics

50,000 is an initial recommendation, not a measured GPU optimum. It adds 10%
voxel coordinates while retaining C's full image coverage. Per usual microstep:
100,000 image + 10,000 SDF + unchanged graph points. It does **not** double the
forward size. Relative to C, SDF supervision changes from 500,000 broadly sampled
points to 50,000 targeted points per subject; report that reduction too.
Increase only after inspecting real GPU memory, iteration time and surface/inside
pool sizes. Counts are independently configurable. Each image microstep must
receive at least one SDF point; insufficient budgets fail explicitly.

Rank-zero `loss_components.jsonl` includes actual per-subject branch composition,
uniqueness, repeats, microstep task counts and losses at the configured interval.
`task_sampling_cost.jsonl` records post-update step seconds and CUDA lifetime
peak allocated memory. Time includes sampled-step logging/synchronization,
excludes DataLoader waiting, and is not an epoch-throughput measurement. CUDA
peaks are rank-zero allocator peaks, not total device memory. Use Slurm elapsed
and MaxRSS, per-rank GPU monitoring and equal-step timings for production cost.
No actual E GPU memory/time measurement is available from CPU validation.

The extra points also enter the existing MoE auxiliary loss, so the weight is
unchanged but its population changes. Shared gradients can still conflict.
Extra compute, fewer SDF supervision points, the new disjoint pool definition,
and stochastic/DDP effects are confounders when comparing E directly with C/D.
An optional compute-matched control should retain E's 500k+50k routing/counts but
sample its SDF-only points with the original image sampler. Compare that control
to E to isolate geometric sampling, and to C to assess the split/budget change.
That optional control is a proposed separate experiment, not implemented here.

## Commands (nothing submitted automatically)

On the current Eurecom cluster:

```bash
cd /users/zhangx/VATL_MM
bash training_script/ablations/smoke_E.sh
SDF_DRY_RUN=1 bash training_script/ablations/E.sh
sbatch --test-only training_script/ablations/E.sh
sbatch training_script/ablations/E.sh
```

The supplied six-hour limit matches A–D but is a wall-time cap, not evidence of
1000-epoch completion. Ensure epoch 30 exists. If needed, use an adequate common
wall time for comparable future runs; keep scheduler horizon 1000.

Leonardo four-GPU variant:

```bash
# Run from the checked-out repository on Leonardo; use your actual allocation.
sbatch --account=YOUR_LEONARDO_ACCOUNT --partition=YOUR_GPU_PARTITION \
  training_script/ablations/E_leonardo.sh
```

The Leonardo wrapper uses one task launching four torchrun processes. Its account
and partition cannot be validated from this Eurecom login node. It requires the
same bashrc/conda setup and accessible data/output paths; adjust `SDF_REPO` and
CLI dataset paths on that cluster. All output roots in the supplied launchers
remain under `/medias/medical/caravel/zhangx/training_outputs` as required by this
workspace. Do not claim remote readiness if those mounts are unavailable.

Optional sample counts (do not change image count for the main comparison):

```bash
sbatch training_script/ablations/E.sh \
  --task_specific_sampling__sdf__num_points 50000
python -m diagnostics.task_sampling_qc --subjects 3 --output /tmp/new_E_qc.json
```

## Evaluation after E training

Use C, D and E **checkpoint_epoch_30.pth**. Verify saved subject IDs/splits,
architecture, seed, conditioning, resolution and 1000-epoch LR horizon. Use the
same subject IDs, not merely the same integer indices if ordering differs.
Use deterministic native full grids, or one frozen set of evaluation coordinates
and targets shared across checkpoints. Training losses on different samplers
are not comparable image-fidelity or SDF-accuracy scores.

For each checkpoint, native subject metrics and paired raw/clamped/masked outputs
are already supported by `diagnostics.compare_masking`:

```bash
python -m diagnostics.compare_masking --checkpoint /path/to/checkpoint_epoch_30.pth \
  --subject-index 0 --native --age ACTUAL_SUBJECT_AGE --sex F --device cuda \
  --data-root /medias/medical/data/Atlas_training/v1 \
  --output-dir /medias/medical/caravel/zhangx/training_outputs/diagnostics/E_ep30_subject0
```

Use actual recorded age/sex so native reference metrics are enabled. Repeat for
C/D with new output directories and the same subjects. The tool reports MRA/T1
MAE, PSNR, SSIM on fixed range 1 for clamped/unmasked and masked outputs, SDF MAE
within the 5-mm truncation region, SDF vessel Dice and component counts. Raw
linear predictions are exported for separate inspection; do not renormalize them.
Report foreground/vascular subregions too: whole-volume image scores can be
dominated by background. Compare multiple subjects and retain all results.

Mean-latent atlases must be evaluated separately:

```bash
python -m diagnostics.compare_masking --checkpoint /path/to/checkpoint_epoch_30.pth \
  --age 20 --sex F --spacing 1 1 1 --device cuda \
  --output-dir /medias/medical/caravel/zhangx/training_outputs/diagnostics/E_ep30_mean_F20
```

Repeat sex M and identical C/D settings. Saved C/D atlas spacing is 1 mm, age 20,
both sexes. An atlas has no direct single-subject voxelwise ground truth; assess
continuity, cortical detail, vessel topology and registration-matched population
references rather than treating one subject as the atlas target.

Predefine near-zero SDF bands (e.g. |GT SDF| <= 1 and <= 2 mm), thin-vessel strata,
component-size distributions, physical vessel volume and centerline/clDice if a
validated skeletonization tool is available. The existing diagnostic reports
truncation-band MAE and component totals, but does not yet automate those finer
strata, physical volume, component-size histograms or clDice. Do not substitute
smooth appearance for those measurements. Check raw versus masked outputs even
though prior investigation found masking was not the principal cause.

## Validation and limits

CPU tests cover A–D regression, effective C/E configurations, disjoint pools,
unique fallback/repetition, shared augmentation, exact C image coordinates,
mean loss invariance to count, two-subject conditions/transformation alignment,
image/SDF/graph output-gradient isolation and real tiny-model optimizer steps.
No full training, four-GPU E smoke or full-volume inference was submitted.
Four-GPU memory, AMP behavior at scale, runtime, convergence and superiority
over C remain unestablished. Existing checkpoint format is preserved; resuming
C into E would change the experiment and is not the prescribed from-seed test.

Validation completed: eight existing A–D/diagnostic regression tests and six E
CPU tests passed (including the shell smoke in the activated torch environment).
Python compilation, shell syntax, launcher dry runs and `git diff --check` passed.
E.sh also passed `sbatch --test-only` on Eurecom; Leonardo was not contacted.
