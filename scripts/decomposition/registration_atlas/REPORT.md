# Four-subject registration and population atlas investigation

Study: `../training_outputs/registration_atlas_20261028_01` (relative to VATL_MM).
Status is updated as completed results arrive. No major neural training launched.
The simplified atlas-rigid network and old checkpoints/results are unchanged.

## Research objective and interpretation

The user clarified that population anatomy, continuous age/sex-conditioned vessel
statistics and future normative modelling are primary. The earlier assumption
that the project should become conventional image segmentation was incorrect
and has been superseded. Individual reconstruction Dice is a fitting diagnostic,
not the sole definition of a useful population atlas.

## Completed audit

Slurm CPU job 1661345 completed, exit 0:0, elapsed 3 minutes. Fixed IDs: IXI425,
IXI070, IXI278, IXI202 (full IDs in study.json). Demographic range 19.98–20.70
with two females/two males; no lifespan/sex conclusions can be drawn.

- Actual MRA, T1, segmentation, SDF and graph paths exist in the v1 data tree.
- Each subject's modalities have matching affines/shapes and RAS orientation.
  Native images remain oblique; RAS labels do not mean axis-aligned voxels.
- IXI425 spacing is approximately 1.055/1.055/0.800 mm, shape 256/256/92;
  the others approximately .938/.938/.800 mm, shape 256/256/100.
- All four negative SDF regions exactly match vessel label 1. Independently
  recomputed spacing-aware EDT agrees inside vessels and within the 5-mm outside
  band at saved float32 precision (measured MAE and maximum 0).
- All four have 100% vessel coverage in the model's 240/240/120-mm crop.
  The analysis grid covers the native centered FOV union, not only that model crop.
- Code audit: the loader subtracts affine(shape/2) and divides by half the world
  box. The simplified registration acts on centered mm, subject-to-atlas as an
  inverse sampling map, with analytic inverse. No nonlinear anatomical alignment
  occurs in that loader. Feature grids use border padding beyond their box.
- Known synthetic rigid point correspondences recovered with RMS error 5.11e-5 mm
  (legacy Euler), 4.20e-7 mm (rotvec); inverse max error 2.15e-5 mm. This validates
  physical point-map conventions, not convergence of an image-registration loss.
- T1 availability is limited to the provided MRA slab grids. Full-brain native T1
  coverage and upstream preprocessing transformations were not found in the
  supplied v1 tree. Original padding/acquisition support masks are unavailable.
  Their absence limits registration provenance and exact FOV reconstruction.

No gross sign, unit, cross-modality geometry or vessel-crop failure was confirmed
on this cohort. Inadequate anatomical correspondence is a measured weakness;
its contribution to historical shared-SDF fitting is not yet isolated.

## Completed R0: current centered alignment

Common 1-mm grid, shape 290/292/136, union of centered native FOVs plus padding.
Probability maps use geometric FOV masks, with NaN outside zero coverage.
Pairwise metrics use each pair's shared FOV.

| Mean over six pairs | R0 |
|---|---:|
| Vessel Dice | 0.05715 |
| clDice | 0.03408 |
| Symmetric mean centerline distance | 6.074 mm |
| Occupancy concentration within all-four FOV | 0.29440 |

The representative is IXI278-HH-1771, selected before registration by the largest
mean R0 pairwise Dice (0.06828); fixed ID order breaks ties. This is a documented
medoid-like criterion, not the paper's standardized-difference prototype rule.
R0 probability/coverage/centerline maps and overlays are saved in the study.
Inspected MIPs show grossly similar global vascular layout but large local offsets.
MIP sharpness alone is not used to infer topology or adequate correspondence.

## Completed structural and vascular baselines

R1 structural SyN and corrected R2 incremental vascular registration completed on
CPU with exit 0:0. The initial direct-H5 R2 attempt failed to export inverses;
its outputs are retained but excluded. R2_incremental estimates and composes a
separate forward/inverse refinement, validated against the optimizer output.

The comparison below uses **identical per-pair coverage regions across all three
stages**, and atlas concentration uses their common all-subject FOV. R0_ants uses
exactly the same ANTs resampling backend as R1/R2; it is an identity-only control.
Consequently its values differ slightly from the initial SciPy-resampled R0 audit.

| Stage | Pairwise Dice | clDice | Mean centerline distance mm | Occupancy concentration |
|---|---:|---:|---:|---:|
| R0, matched ANTs identity resampling | 0.05440 | 0.03379 | 6.010 | 0.2901 |
| R1, slab-T1 structural SyN | 0.12015 | 0.07559 | 4.547 | 0.3386 |
| R2, vascular distance-feature refinement | 0.16892 | 0.10285 | 3.969 | 0.3748 |

No folding was detected in covered regions at the analysis resolution. Total
sampling displacement p95 reaches 32.4 mm and includes the initial structural
rigid/affine/nonlinear normalization; it must not be attributed entirely to the
vascular increment. Vascular-only incremental p95 displacements are 2.99, 4.66
and 3.25 mm for non-reference subjects. Their minimum incremental Jacobians are
0.755, 0.517 and 0.623, with zero detected folding. Total R2 Jacobian minima reach
0.432; small fractions (<0.054%) of covered voxels have determinant <0.5.

Native-to-atlas vessel volume ratios range from 1.01 to 1.42 after normalization.
These include true geometric scaling and interpolation and cannot be called
branch preservation or loss. R1 inverse consistency p95 is <0.012 mm on sampled
vessel-neighborhood points. All per-subject fields/statistics are saved.

These results support a measurable correspondence benefit beyond structural
alignment. They do **not** isolate registration as the principal cause of the old
neural failure, demonstrate improved neural fitting, or establish preservation of
all biological variation. Considerable local mismatch remains after R2.

## Completed small-vessel preservation controls

Identity native-to-canonical-grid-to-native resampling using the same ANTs backend:

| Grid spacing | Round-trip Dice range | Centerline retention range | Components |
|---|---:|---:|---|
| 1.0 mm | 0.8693–0.9384 | 0.8614–0.9469 | Marked fragmentation, e.g. 24 to 107 |
| 0.8 mm | 0.9882–0.9977 | 0.9892–0.9989 | Close to native counts |
| 0.5 mm | 1.0000 | 1.0000 | Original counts recovered in this control |

Total vessel volume is approximately conserved despite 1-mm fragmentation.
Volume retention alone is therefore inadequate QC. The 0.5-mm result is a specific
identity-resampling control, not a promise of perfect preservation under arbitrary
nonlinear warps or recovery of anatomy absent from the original scan.

The current registration comparison intentionally remains on its controlled 1-mm
grid. Before choosing aligned neural supervision or a final population template,
re-export the original observations through fixed transforms at finer resolution
and recheck native inverse-space preservation. Do not repeatedly warp masks.
No compatible re-segmentation model was verified; no re-segmentation was run.

A second known-transform control perturbed a real vessel image's physical affine
without altering its voxels. Classical rigid recovery improved Dice from 0.11415
to 1.0, with mean point error 2.20e-5 mm and p95 4.19e-5 mm. Together with the neural
point-map test this argues against a gross transform-direction/unit implementation
error in these tested paths; it does not validate arbitrary subject registration.

## Completed two-iteration group-wise comparison

Job **1662464** completed, exit 0:0, elapsed 10:16. Group comparison/bias QC
**1662785** completed, exit 0:0. All scores below use the same conservative
per-pair FOV intersection across all FIVE compared stages; this differs slightly
from the three-stage comparison above.

| Stage | Pairwise Dice | clDice | Centerline distance mm | Occupancy concentration |
|---|---:|---:|---:|---:|
| Current alignment, ANTs identity | 0.05429 | 0.03387 | 5.981 | 0.2902 |
| Structural R1 | 0.11986 | 0.07561 | 4.541 | 0.3386 |
| Fixed-reference vascular R2 | 0.16874 | 0.10272 | 3.954 | 0.3748 |
| Group iteration 1 | 0.17473 | 0.10722 | 3.796 | 0.3787 |
| Group iteration 2 | 0.21934 | 0.13892 | 3.223 | 0.4102 |

The group target is a FOV-weighted population of proximity features at 2 and 5 mm,
while occupancy probabilities are saved separately. Mean incremental sampling
bias is corrected by applying a damped inverse mean field consistently to every
subject. On a common eroded FOV, incremental mean-field RMS decreases from
1.887 to 1.511 mm (iteration 1), and 0.542 to 0.433 mm (iteration 2), approximately
the expected 20% reduction for the specified shape step. This validates the tested
correction direction/composition, not a globally unbiased population template.
Structural-reference bias is not automatically removed.

**The overlap gain has a preservation/regularity cost.** No folding was detected,
but the minimum total Jacobian fell to **0.248**, and up to **0.489%** of covered
voxels have determinant <0.5. Native round-trip Dice spans **0.852–0.932**, worse
than R2's **0.882–0.960**; maximum sampled inverse-consistency p95 is 0.0372 mm.
These effects combine deformation-induced geometry changes and discrete label
interpolation. They do not by themselves establish biological branch loss or
suppression of true variation, but they prevent treating higher overlap or a
sharper MIP as sufficient success.

The run stopped at its two-iteration budget, **not convergence**: the second
vessel-mass-normalized probability change is 0.824, far above tolerance. We did
not expand to more rounds, alternative references or a full cohort. Initial-
reference sensitivity, homologous branch/bifurcation QC and CoW annotation remain
pending. Fixed-z slice figures are not validated CoW ROIs.

The group implementation is an ANTs-based exploratory, first-order mean-bias
correction, not an exact unbiased diffeomorphic mean or the paper's ANIMAL method.

## Completed anatomical/header and preservation summary

Supplemental preservation retry **1662712** completed. Original qform/sform
matrices differ by at most 1.44e-6 in their stored affine entries. The first summary
attempt rejected this tiny serialization roundoff; the retry explicitly uses
1e-4-mm absolute affine tolerance, with unchanged data and a new output folder.
Native-space TP/FP/FN additions/losses, centerline retention and component counts
are saved for R0/R1/R2. These voxel differences quantify round-trip interpolation,
not true anatomical branch correspondence.

Structural T1 overlays were inspected alongside vessel slices/MIPs. R1 improves
central structural correspondence, while slab-boundary differences remain clear.
This does not certify anatomical plausibility or homologous vascular alignment.
Original full-brain T1 and pre-padding support provenance remain limitations.

## Decision: hold neural expansion at the preservation/FOV gate

Correspondence is demonstrably improvable on this cohort. Whether that is a major
cause of shared-SDF fitting failure remains **unmeasured** until a fresh matched
neural experiment is performed. Neither historical checkpoint Dice nor classical
registration Dice is a v2 model performance measurement.

The next justified work is finer-grid re-export through fixed transforms and
native-space preservation QC, then an explicit coverage-aware training adapter.
The current neural loader does not consume Coverage. Training directly on a
zero-padded common grid would incorrectly treat unobserved support as background
for SDF/image/tissue losses, invalidating the comparison. `regrid.py` is prepared,
not executed; it writes multiclass labels, physical SDFs, Coverage and metadata
without changing registration optimization. Its training_gate.json records the
remaining requirements. Graph supervision must be consistently transformed or
disabled in both arms.

No major neural training, occupancy fitting, encoder or graph generative model
was launched. A classical vs continuous Bernoulli occupancy baseline and canonical
SDF comparison remain the next representation questions, after the data gates.
The four-subject empirical probability maps are descriptive pooled frequencies,
not reliable age/sex-conditioned healthy-population or normative distributions.

## Reference and methodological boundaries

[ Xu et al., Scientific Data 2026 ](https://doi.org/10.1038/s41597-026-07710-5)
uses structural ANTs SyN and a separate ANIMAL vascular-refinement stage, with
iterative map/deformation updates and FOV-weighted probabilities. We downloaded
and inspected the reference PDF. Our ANTsPy vascular baseline is inspired by these
principles, not a reproduction of ANIMAL or the 1,288-subject study.

Pipeline invariants tested: missing FOV vs background; physical signed distances;
unclamped Jacobian determinant on rotated anisotropic grids including fold
detection; noncommuting ANTs transform composition and inverse order. All five
new tests passed, including explicit undefined centerline metrics when skeletons
are empty. See README for reproducible commands and transform conventions.
The group-wise script is implemented and synthetic-tested; inspect real-data
results before drawing conclusions.

## Supported findings, remaining hypotheses and pending work

**Supported by completed measurements:** vascular correspondence in the current
centered frame is poor; structural registration improves it; vascular refinement
adds further benefit; group updates can further improve overlap; 1-mm label
resampling fragments thin vessels; finer identity grids strongly reduce that
artifact; FOV-aware maps and transform conventions work in the tested controls.

**Not established:** registration as the sole/major cause of neural failure;
accurate shared-SDF fitting after prealignment; preservation of biological graph
variants; unbiased group geometry; calibrated lifespan-conditioned probabilities;
healthy normative deviation scores; superiority to the paper or other registration
methods. Sampling, convergence, objective conflicts and deterministic-versus-
probabilistic population representation remain active explanations.

**Pending, in order:** finer-grid inverse-space preservation; coverage-aware
sampling/loss support; conservative registration/feature sensitivity and expert
branch/CoW QC; fresh matched v2 SDF comparison; minimal Bernoulli field against
classical probability; larger demographically appropriate cohort. Alternative
initial references and repeated seeds should follow preservation validation.
Architecture proposals, including correlated SDFs and possible graph/SDF models,
remain technical proposals only in ARCHITECTURE_PROPOSAL.md.

Five registration tests, 16 decomposition tests and 5 validation tests passed.
Synthetic group-wise and network training/resume integration passed. Verification
logs, source/input hashes and CPU job IDs are retained in the study. All remaining
submitted jobs have completed or are explicitly recorded as failed/cancelled;
none remain queued. Dataset, historical results and simplified network preserved.
