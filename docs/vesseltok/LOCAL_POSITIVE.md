# Binary local-positive protection control

This is a sampling adapter,not partial-volume training and not a new model/loss.
Both arms retain exactly the existing16gap-negative and up-to64inter-anchor
positive plans,including original eligibility fallbacks. Only B reallocates up
to16OTHER original interior-positive slots to graph-positive native voxel centers
within3mm of the corresponding gap queries. All targets remain exact0/1.

LocalPositiveSet wraps GapPatchSet without changing its RNG or query draw.
Private generator61103 creates identical shadow plans in A/B. Original interior
slots already assigned to the inter-anchor plan cannot be protected. Partnerable
gap slots are processed in ascending slot order;uniform native-positive partners
are drawn from frozen CSR lists. Missing partners/slots keep original queries
and subjects;actual counts are logged. No negatives are removed or relocated.

The pool uses trusted TRAINING graphs plus MRA header geometry only. Native centers
must be in observed FOV/model cube,exact graph1mm point-positive after float32
coordinate roundtrip,and within3mm of a frozen gap candidate. CSR adjacency is
checked strictly at3mm using the actual reconstructed world coordinates. No
Seg/SDF or model error maps influence candidates. The radius is pseudo geometry,
not a measurement of anatomical vessel radii.

## External data preparation and checks

Use a compatible shared cache and frozen gap/positive pools;none are shipped.
Provide metadata/split paths in copies of configs/vesseltok/local_positive_A.yaml
and B. The configs differ ONLY by local_positive_mode;all other model/optimizer/
query settings remain fixed. The copied protocol YAML is a rule document,not
an executable training config.

```bash
python -m vesseltok.local_positive_prepare   --config CONFIG_A.yaml --prepared EXTERNAL_CACHE   --gap-pool-root EXTERNAL_GAP_POOL_ROOT --prediction-root PRIOR_PREDICTION_ROOT   --confirmation-start 21 --plan FROZEN_PLAN.md --output NEW_POOL_ROOT
python -m vesseltok.local_positive_preflight   --pool NEW_POOL_ROOT --reference PRIOR_GAP16_RUN   --config-A CONFIG_A.yaml --config-B CONFIG_B.yaml
```

Preparation reserves3canonical unused confirmation people from the declared
manifest positions,checks all prior prediction manifests and freezes the
cohort/pools/plan before fitting. It reads no confirmation GT contents. Do not
change the positions based on evaluation quality. Preflight replays2000sampler
steps with no optimizer/model updates,requiring the reference initial-model hash,
original control streams and final CPU RNG. It verifies every proposed native
positive label,pair distance,FOV and inverse-grid position. Synthetic tests also
verify3optimizer updates bitwise against the unwrapped baseline and correct
encoder/decoder gradients. Original AdamW includes the deliberately frozen
variance head;that head must stay frozen and must not be silently removed from
parameter groups. Generic training checks missing/duplicate/unexpected frozen
parameters and records parameter names.

## Training after ALL gates pass

On allocated Slurm GPUs only,invoke the existing train.sh for each arm with:
`--config CONFIG_A_OR_B.yaml --prepared EXTERNAL_CACHE --sampling-pool
EXTERNAL_POSITIVE_POOL.npz --gap-pool NEW_POOL_ROOT/gap_pool.npz
--local-positive-pool NEW_POOL_ROOT/local_positive_pool.npz --output NEW_RUN`
(single command). Execute only after CPU tests and preflight pass;verify its
critical source/config/pool hashes have not changed. No helper here silently
submits jobs. The protected experiment enforces2000steps,16gap/64inter-anchor/
16protection caps,condition_dim0 and binary targets. No transform is introduced.

Use unchanged native_geometry_v1 validation-only selection,threshold.5 and
candidate steps500/1000/1500/2000. Do not choose samples,thresholds or checkpoints
from confirmation outcomes. Keep the new cohort untouched until predictions are
frozen. Model A must replay the reference gap16 B final parameters. Assess
continuous/native complete paths and false/missing connections together;Dice or
component count alone is insufficient. Anatomical metrics remain separate.

This branch includes CODE/configs/tests/context only. Runtime manifests,pools,
cohort lists,predictions,metrics,figures,logs and checkpoints remain external.
