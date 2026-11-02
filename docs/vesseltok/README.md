# VesselTok-inspired graph-conditioned training and validation

Standalone Stage1 graph encoder/implicit occupancy decoder,isolated from the
existing VATL SIREN baseline. Inputs are world-mm graph centerlines; FPS anchors,
cross/self-attention and compact tokens condition arbitrary-coordinate queries.
Targets are union-of-edge capsules with a configurable pseudo radius (default
1mm),not measured anatomical radii or a true vessel SDF. Image geometry supplies
native coordinates/FOV,not MRA/T1 intensity inference. Age/sex are recorded;
provided configs disable demographic conditioning and all learned transforms.

## Data contract

Data/checkpoints/results are deliberately external and not shipped. Only load
trusted NetworkX graph pickles: node `pos` is a finite3-vector in RAS world-mm,
edges are straight source segments; hidden polyline fields are rejected. Images
must be scalar3D NIfTI with mm units and a valid affine. Metadata is a TSV with
`subject_id,MRA,Graph,scan_age,sex` columns (tab separated),sexM/F. Optional
anatomical validation requires `Seg,SDF` paths and exactly matching native grids;
training/preparation/native_geometry_v1 do not consume those annotations.

Split YAML format: one top-level dataset key,containing
`subject_ids: {train: [...], val: [...], test: [...]}`. Canonical subject IDs may
not overlap across partitions. The current cohort builder draws training people
using seed42,uses the first configured validation-list people as evaluation and
the following people for checkpoint selection. Inspect and freeze generated
splits.json BEFORE fitting. Keep final evaluation people untouched by tuning.
No real subject manifest is included.

## CPU checks and preparation

From this checkout,activate the environment described in DEPENDENCIES.md.
Run `bash scripts/vesseltok/cpu_tests.sh` (synthetic data,CPU only).
Edit a copy of configs/vesseltok/stage1.yaml to point to your external metadata
and split manifest. Prepare a NEW external cache:

```bash
python -m vesseltok.train --config configs/vesseltok/stage1.yaml   --prepared /external/cache --output /external/unused --prepare-only
```

The standard config samples8192 input points directly. The paired gap16 configs
retain the original `nested_4096` policy and require a matching precomputed shared
cache and frozen graph-only positive/gap pools; they are not interchangeable with
a freshly prepared standard-policy cache. No cache/pool/checkpoint is bundled.
Training rejects mismatched geometry,cache hashes,cohorts and pool signatures.
The optional gap adapter changes only eligible far-negative queries; both arms
keep targeted positives and private shadow RNG. `gap_mode` is the sole difference
between the paired configs. Sampler generation and eligibility must be frozen
before fitting and must never mine final evaluation outcomes.

## Allocated GPU training and validation-only selection

```bash
sbatch --account=YOUR_ACCOUNT scripts/vesseltok/train.sh   --config configs/vesseltok/stage1.yaml --prepared /external/cache   --output /external/new_run
sbatch --account=YOUR_ACCOUNT scripts/vesseltok/render_validation.sh   --retained-only --run /external/new_run --output /external/new_run/native_validation
```

Submit rendering only after successful training (e.g.,Slurm afterok dependency).
The retained checkpoints are500/1000/1500/2000 with the provided configs.
Run selection on CPU after rendering,passing the candidate directories listed
in native_validation/candidate_roots.json:

```bash
python -m vesseltok.select_checkpoint   --candidates /external/new_run/native_validation/step_000500     /external/new_run/native_validation/step_001000     /external/new_run/native_validation/step_001500     /external/new_run/native_validation/step_002000   --output /external/new_run/selection
```

native_geometry_v1 freezes threshold.5 and macro per-person harmonic mean of
pseudoDice and complete native source-path support. Ties favor fewer cross-source
connections,then earlier step/hash. Only exact saved checkpoint validation
subjects are accepted; no Seg/SDF or final evaluation data selects checkpoints.
BCE-best is saved separately and does not masquerade as geometry-selected.
Selection emits immutable provenance and selected.pt alias. No expensive training
is launched by this repository or its tests.

Inference utilities are included because native validation renders probabilities
with the same decoder and coordinate convention. Anatomical scoring utilities
are shared regression-test dependencies; they do not alter prediction or select
models from anatomical GT. No experimental reports,results,logs,meshes,data,
checkpoints or caches are included in this branch.

See DEPENDENCIES.md for attribution and environment details; root context.md
records the implementation scope and constraints.
