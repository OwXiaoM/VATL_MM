> Historical plan, superseded by the population-atlas objective in AGENTS.md
> and the current NEXT_STEPS.md. Retained as conversation history only.

# Next investigation: accurate vessel inference without an additive residual

2026-10-28. This plan distinguishes measured results, architectural constraints
and proposed experiments. No accuracy improvement from the removal is claimed.

## What the completed runs establish

- Original training reconstruction mean Dice 0.0832 and held-out SDF-supervised
  reconstruction mean Dice 0.0685: the problem already exists in fitting training
  subjects. More test-time optimization alone cannot be assumed to solve it.
- One-subject fits reach Dice 0.8391 (SDF only) and 0.8464 (with image/tissue
  losses), still improving at 500 epochs. Real-data fitting works substantially
  better than the population fit, but the >0.90 debugging target is not reached.
- Four-subject reference Dice 0.4415, recall 0.9513, precision 0.2892 and volume
  ratio 3.378: excessive positive volume, not primarily missed vessel coverage.
- Removing the residual penalty gave Dice 0.4534 and only 0.000382 direct Dice
  improvement over its own atlas-only prediction. This branch did not solve the
  failure. Its artifacts are deleted; measurements remain in CONTEXT.md.
- These comparisons do not isolate a unique cause. In particular, the one-subject
  versus four-subject comparison changes registration and losses as well as cohort.

## Why the inference protocol needs to change

SharedAtlasSDF receives coordinates and demographic conditions. Image values are
training targets, not local subject-image inputs. The shared modulation grid is
also identical across subjects. With no additive residual, subject specificity
comes from rigid transforms and conditions. For equal conditions, a rigid transform
can reposition a template tree but cannot create different branch connectivity.
This is an architectural limit, not proof that registration explains all error.

The existing adapt_subject minimizes SDF error against supplied subject truth.
It is an oracle reconstruction diagnostic. It is not a deployment procedure for
an unlabeled MRA. Auxiliary image reconstruction during training does not make
this adapter image-driven. Increasing adaptation steps does not resolve that gap.

Assumption for the deployment plan below: an MRA volume is available, but vessel
segmentation/SDF and graphs derived from them are unavailable. If deployment is
instead atlas generation from age/sex alone, precise patient-specific branches
are underdetermined; that is a different objective from individual segmentation.

## Ordered experiments and decision gates

1. **Finish establishing single-subject capacity.** Run a fresh v2 SDF-only fit
   with a longer budget, for example 1500 epochs, monitoring full-grid Dice,
   precision, recall, volume ratio and surface distances every 50 epochs. Inspect
   several centerline-aligned and cross-sectional views. If it plateaus below the
   debugging target, investigate decoder resolution and sampling before population
   scaling. The 1500-epoch budget is proposed, not a measured optimum. A new fit
   must not be labeled a resumed residual-era run.

2. **Check alignment independently of population anatomy.** First test two
   synthetic copies of the same real subject with a known physical rigid transform
   (consistent image/SDF geometry); compare learned transform recovery and Dice
   with identity and known-transform controls. Then audit the four real subjects'
   image affines, orientations, spatial units and overlays. Independently recompute
   spacing-aware signed distances for all four training labels, checking inside
   and outside bands; the earlier numerical audit covered three validation files,
   not every training target. Quantify queries mapped outside the atlas box and
   whether border-padded feature sampling coincides with false positives. These
   are unmeasured checks, not confirmed bugs. Centering volumes is
   not anatomical registration. Compare fixed image-based prealignment with learned
   rigid registration on exactly the same cohort, objective and update budget.
   Use rotvec only in new checkpoints and include matched legacy-Euler control.
   Do not register predictions to truth solely to inflate reported Dice.

3. **Test the excessive-volume failure directly.** On the fresh four-subject v2
   reference compare the prepared uniform50 and less_interior samplers separately.
   Record sample distributions and separate SDF errors inside vessels, within
   0–2 mm outside, and farther away, plus full-grid false positives and Dice.
   Existing sampling yields about 30% vessel points versus 0.23–0.36% native
   occupancy. This mismatch is a plausible contributor, not an established sole
   cause. If needed, subsequently test explicit false-positive penalties or
   hard-negative sampling on training data, one change at a time. Do not alter
   the tissue-class head and assume it directly calibrates the separate SDF mask.
   Threshold calibration belongs on validation subjects only, with a frozen
   threshold for test; the old truth-tuned threshold sweep was insufficient.

4. **Give deployment access to individual anatomy.** Establish an image-to-vessel
   supervised baseline using MRA patches and a 3D segmentation network, with
   patient-level train/validation/test separation. Use the same full-grid physical
   evaluation and include small-vessel recall/precision and clDice. This is a
   proposed implementation, not something already present or validated here.
   If retaining the INR is essential, use an image encoder with spatial feature
   sampling at each query, predicting the SDF directly, without an additive
   subject residual. Train on labeled subjects, infer from MRA alone. A global
   age/sex condition cannot substitute for these spatial image features.
   Keep the atlas as an optional prior; first measure the image-only baseline.

5. **Evaluate the actual inference route.** No SDF, segmentation or label-derived
   graph may be used during image-only fitting or selection. A ground-truth image
   geometry can be replaced with the input MRA geometry. Select model/threshold
   on validation subjects, report frozen test metrics, per-subject values and
   volume bias. Compare training reconstruction, held-out oracle SDF fitting and
   held-out image-only predictions as separate protocols. Expand the cohort and
   repeat seeds only after the small-cohort gates are understood.

Image-driven nonrigid registration is another possible atlas-transfer experiment,
but it must be evaluated as such; an invertible deformation of one tree does not
supply missing anatomical branches. It should not be assumed sufficient for thin
vessels. The immediate priority is capacity/alignment/sampling diagnostics and an
image-conditioned segmentation route, not another residual optimization sweep.

## Implementation status

Implemented: residual and code removal; atlas/rigid v2 checkpoints; fresh study
preparation without residual variants; updated metrics/exports/smoke checks.
Prepared but not run: v2 sampler, rotation and Eikonal variants.
Proposed, not implemented: longer real fit, copy/transform control, image-based
prealignment comparison, spatial image encoder, segmentation baseline and an
image-only deployment adapter. No new GPU job was submitted by this change.
