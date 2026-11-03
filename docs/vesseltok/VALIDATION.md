# Read-only validation tools

These modules contain validation CODE,not experiment outputs. Supply your own
external data,models,caches and immutable plan files. No runtime artifacts or
subject manifests are included. Activate the environment in DEPENDENCIES.md.

## Break localization

`python -m vesseltok.break_prepare --A RUN_A --B RUN_B --gap-pool GAP_POOL.npz
--plan FROZEN_PLAN.md --output NEW_ROOT` (single command). The sibling
pool_manifest.json must declare the external prepared cache and positive_pool.
This pilot audits the first6saved training subjects and both saved selection-
validation subjects. Exact sampler replay must match stored streams and final
CPU RNG;no model forward/backward or optimizer is used in replay. It requires
the deterministic2000-step16-gap configuration and selected2000 checkpoints.

On an allocated Slurm GPU,run `python -m vesseltok.break_fields --prepared ROOT
--arm A` and B. Training fields use cached input clouds;validation fields reuse
saved step2000 validation tokens/probabilities. Model parameters remain frozen.
Then run `python -m vesseltok.break_report --prepared ROOT` on CPU,followed by
`python -m vesseltok.break_supplement --prepared ROOT` for explicitly exploratory
teacher-raster/common-stratum checks. Source-node pair counts are correlated;
bridge footprints are diagnostic witnesses,not independent anatomical false edges.
Do not use these diagnostics to tune on final evaluation subjects.

## Physical voxel-fraction targets

`python -m vesseltok.partial_volume_synthetic --output NEW_DIRECTORY` validates
fixed geometric stress scenes on CPU without any training/data files.
`python -m vesseltok.partial_volume_validate --config CONFIG.yaml --splits
EXTERNAL_SPLITS.json --plan FROZEN_PLAN.md --case INDEX --output NEW_DIRECTORY`
(single command) runs case0/1(first2train) or2/3(selection validation). Metadata
must supply trusted Graph files and MRA native geometry. No Seg/SDF is consumed.
The scientific configuration must specify pseudo_radius_mm=1 and a valid edge
spacing. Voxel fractions use the physical affine basis,not normalized bbox axes;
the values measure volume fill,not point-presence uncertainty. Numerical lower/
upper bounds bracket integration,not probabilistic confidence. Threshold stays.5.

`python -m vesseltok.partial_volume_report --cases CASE0 CASE1 CASE2 CASE3
--synthetic SYNTHETIC_ROOT --plan FROZEN_PLAN.md --output NEW_DIRECTORY` collects
checks. This report is intentionally gated for the present negative-validation
workflow:it never starts training and rejects interpreting failed numerical
checks as a scientific representation conclusion. A future positive gate would
require review of a separately frozen training protocol. No anatomy/radius claim
follows from fixed1mm graph tubes. Use `partial_volume_plot` for standalone plots.

The local-positive sampler experiment is a distinct binary-target sampling
control;its adapter is pending. Do not treat its protocol YAML as runnable
training code. CPU regressions cover geometry/affines,quadrature certificates,
naive/adaptive agreement,path/partition arithmetic and paired sampling controls.
