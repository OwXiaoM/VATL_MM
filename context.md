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

NEXT AUTHORIZED TASK (pending implementation at this push):implement the local-
positive protection sampler and tests,then run a controlled two-arm2000-step pilot
on allocated Slurm GPUs only after checks pass. Keep16gap negatives and64inter-
anchor positives identical;only B may reallocate up to16other positives to exact
binary graph-positive native voxel centers within3mm of gap queries. Keep inputs,
query/class budget,initialization,BCE,optimizer and model fixed;no transforms or
demographic conditioning. Freeze rule/unused confirmation cohort before fitting;
unchanged native_geometry_v1 and threshold.5. No confirmation-guided tuning.
The adapter did not exist and no such training was submitted before this snapshot.
configs/vesseltok/local_positive_protocol.yaml records the prepared rule and
required checks;it is not a loadable training config or a completed experiment.
Do not start partial-volume training,increase budget or publish experiment outputs.
