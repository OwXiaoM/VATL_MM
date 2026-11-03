# VesselTok implementation context

This branch isolates graph-conditioned Stage1 training/native validation from
the pre-existing VATL_MM baseline. It includes source,portable example configs,
dependency/attribution documentation and synthetic CPU regression tests only.
Experiment results,datasets,checkpoints,logs and caches are intentionally absent.

Scientific target: graph-derived1mm pseudo occupancy,not true anatomical radius
or SDF. Encoder uses centerline point embeddings,FPS latent queries,cross/self
attention and compact tokens; implicit decoder accepts arbitrary world-mm query
coordinates via an isotropic normalized frame. VAE support is optional; supplied
baseline configs use deterministic tokens. No diffusion/flow/nonlinear warp.

Native affine/FOV conventions and physically rigid-mm utilities are tested.
Learned transforms and demographics are disabled in the supplied controls;
recorded age/sex remain fixed metadata. Training labels/sampling use graph
geometry only. Validation selection is fixed native_geometry_v1 on the exact
saved selection-validation cohort,threshold.5. Final evaluation data must not
choose samples,thresholds or checkpoints. Graph reconstruction is not image-only
inference. Pseudo and anatomical assessments remain separate.

Training supports optional frozen positive-inter-anchor and gap-negative pools.
The paired16-query configs hold everything else fixed. Private generators/shadow
plans preserve global RNG and unchanged supervision. Pools must match training
subject order/cache hashes; validation never receives them. The included standard
config can prepare fresh graph-only caches; nested paired configs require their
own matching external cache/pools. See docs/vesseltok/README.md.

Core tests cover physical transforms,exact capsules,FPS/padding/attention,
query balance,token decoding,native validation provenance/leakage guards,
connectivity and paired samplers. All runtime outputs belong outside version
control. Use allocated Slurm compute for GPU work. Preserve existing VATL source
and user data; do not launch expensive scaling as part of setup or validation.


## Current implementation checkpoint — 2026-11-03

The branch now includes read-only break-localization helpers/drivers and physical
partial-volume target validation,with their synthetic regression tests. External
cache/model/plan paths are explicit CLI inputs;private experiment files are not
bundled. The original training model,activation,BCE and checkpoint policy are
unchanged. Break audits keep training/selection-validation scope separate from
final evaluation and distinguish realized negative queries from candidate geometry.
They preserve cached training inputs and saved validation fields,verify sampler
replay/RNG and frozen parameters,and report point/native path transitions.

Voxel occupancy fraction is deterministic physical cell fill,not uncertainty
about vessel presence. The implementation uses full native affine vectors and
subcell integration bounds. Validation found that majority thresholding can erase
subvoxel vessels or join narrow gaps;partial-volume soft-target training was NOT
launched. Numerical sampling refinement was required and validated. Raw experiment
measurements,subject lists,figures,logs and data stay outside this branch.

## Local-positive protection implementation — 2026-11-03

The previously pending adapter is now implemented as LocalPositiveSet,with paired
private shadow plans and exact binary native-voxel partner pools. Both arms retain
16gap negatives and64inter-anchor positive plans;only B reallocates up to16other
positive slots to graph-positive native voxel centers within3mm of gap queries.
Missing slots/partners preserve subjects and original queries. Inputs,query/class
budget,initialization,BCE,optimizer,model and2000-step budget stay fixed;transforms
and demographics are disabled. No partial-volume training is introduced.

LocalPositiveStreams records matched inputs,negative/inter-anchor/gap queries,
labels,protection proposals/partners and retained queries. Preparation freezes
training-only graph/header pools and an unused confirmation cohort before fit.
Preflight verifies original control sampler hashes,initial weights,final CPU RNG,
strict partner distances and native-grid binary labels. Native_geometry_v1 and
threshold.5 remain unchanged. Original AdamW parameter groups are preserved,
including its intentionally frozen variance head;unexpected frozen/missing/
duplicate parameters are rejected. Encoder/decoder gradients and short bitwise
optimizer replay are tested. The control also reproduced the reference final
model in the completed fixed-budget local pilot.

Both arms completed the authorized2000steps and validation-only checkpoint
selection. Inference/scoring and detailed results were recorded only in the
research worktree,not this branch. No confirmation-guided rule/threshold/
checkpoint change or additional training was performed. A mixed scientific
outcome does not make the sampler a universally improved default. No anatomical
radius/SDF,image-only deployment or population-level claim is justified.

This source checkout passes73focused CPU tests. Local full-pipeline regression
and actual-data preflight records are external. Raw pools,cohort lists,predictions,
metrics,tables,figures,logs,checkpoints and cached data are excluded from Git.
See docs/vesseltok/LOCAL_POSITIVE.md for portable preparation/preflight/training
commands. New data require their own immutable provenance and unused cohorts;
passing synthetic tests alone never authorizes bypassing dataset-specific checks.
