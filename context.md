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
