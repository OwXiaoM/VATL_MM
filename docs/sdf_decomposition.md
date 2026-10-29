# SDF atlas / subject decomposition

This is an opt-in implementation. `run_ddp.py`, `build_atlas_ddp.py`,
`models/inr_decoder.py`, `models/siren.py`, the original configs and training presets
are unchanged by this feature. The dirty working tree already contained the Jean
Zay path migration, so no branch was created, reset or committed. The pre-feature
tracked-file hashes and existing patch are in `scripts/decomposition/`.

## Inspection and conventions

The original INR decoder interpolates a CNN-modulated **per-subject** latent grid
and concatenates age/sex before modulated MoE-SIREN evaluation. Population atlas
export uses weighted mean subject grids. This can blur population anatomy; this
implementation does not assert that this is the sole cause of missing vessels.

`utils.embed2affine` implements Euler rotations plus translations (6 parameters),
optionally scaling (9), and shear (12). The default `tf_dim: 6` is rigid in the
**normalized coordinate metric**, not necessarily in physical space. Scale can
be zero or negative; the optional affine variants have no invertibility guarantee.
There is no SVF, deformation field, STN image encoder, or diffeomorphic model.
Transforms are directly optimized per subject.

Dataset coordinates are
`u = (affine @ voxel - affine @ (shape / 2)) / (world_bbox / 2)`.
The center is `shape/2`, deliberately not `(shape-1)/2`. The default IXI box is
240 x 240 x 120 mm. A header-only check of IXI002-Guys-0828 found a 256 x 256 x 100 SDF,
float32, mm spatial units, and spacing approximately 0.9375 x 0.9375 x 0.799992 mm.
Source voxel spacing and orientation come from the NIfTI
sform/qform affine; they must not be replaced by atlas export spacing (1 mm by
default). `grid_sample` uses `align_corners=True`, border padding and the existing
latent-grid axis order; this implementation retains that convention.

The original decoder applies `R u + t` **before sampling the atlas**, so its stored
transform is the subject-to-atlas inverse sampling map `psi = phi^-1`.
The new `SubjectRegistration` reuses those rotation/inverse routines, but applies
them to centered **millimetre** coordinates, then normalizes again. Translation
is now mm and rotation is radians. The physical rotation has determinant +1 and
an analytic inverse. It is only a global rigid map, not a deformable registration
or a topology-changing warp. No Jacobian penalty is appropriate here. A future
registration module can replace the two explicit map methods.

SDF files are read as floating-point physical mm (the upstream SDF-generation
script is not present). Repository convention is **negative inside** vessels.
The baseline target is `clip(sdf_mm, -tau, tau)/tau`, tau=5 mm. The feature exposes
physical-mm predictions and converts only the decoder's SDF channel by tau.
There is no SDF intensity normalization or softmax. A binary mask is `d < 0`.
Verify upstream distance units/sign before using a new dataset; voxel spacing
alone cannot establish stored SDF distance units.

GraphStore loads NetworkX node `pos` world coordinates, normalizes using the same
center/box, and optionally samples the NIfTI SDF and graph radius. Existing graph
losses support vessel-class CE, sampled SDF and consistency-gated radius. The new
trainer reuses GraphStore and the sampled-SDF/CE loss helper; shipped configs use
sampled SDF only. Radius supervision is not enabled in this first trainer.
This is sparse centerline supervision, **not a connectivity or topology guarantee**.
No differentiable clDice loss is added for unordered samples. clDice is an optional
evaluation metric on reconstructed grids.

## Current architecture and optimization (2026-10-28)

The additive subject residual and subject-code embeddings have been removed.
`SharedAtlasSDF` uses one population grid, coordinates and demographic conditions.
The prediction is `d_i(u) = tau * decoder_sdf(psi_i(u), age, sex)`.
Only the atlas and rigid transform parameter groups are optimized, jointly.
There is no subject-specific branch generator and no spatial image encoder.
This remains a population-template model, not an image-to-segmentation network.

Losses include physical SDF fitting, optional near-surface/Eikonal terms and rigid
regularization. Existing image/tissue and sparse graph-SDF supervision remain.
These auxiliary losses do not turn inference into image-driven adaptation.
Legacy Euler is the default; rotvec is opt-in for fresh checkpoints.

Checkpoint format is `vatl_sdf_atlas_rigid_v2`. Residual-era v1 checkpoints cannot
be resumed or evaluated by the current network and are explicitly rejected by
entry points. Original baseline initialization still copies decoder weights and
the mean latent grid, without reinterpreting saved rigid parameters.

Inference modes remain atlas_only, subject_reconstruction and atlas_and_components.
Components are registered atlas SDF and final SDF (identical in this architecture).
There is no residual export or zero-residual option. An unseen subject's existing
adapter optimizes only its rigid transform against observed ground-truth SDF.
That is an oracle fitting diagnostic, not label-free vessel segmentation.

See [current study commands](../scripts/decomposition/ablations/README.md),
[inference investigation plan](../scripts/decomposition/NEXT_STEPS.md), and
[measured experiment history](../CONTEXT.md). Training reconstruction monitoring
includes truth outside the model crop; its best checkpoint is selected on training
Dice, not held-out performance. The historical 20261027 study is closed to new runs.
