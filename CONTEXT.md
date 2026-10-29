# Conversation and experiment context

Updated 2026-10-28 (Europe/Paris). This is a conversation handoff: user requests
are recorded verbatim below; assistant responses are condensed with their findings
and commitments. It is not an export of hidden reasoning or tool transcripts.

## Conversation record

1. User: "projects/VATL_SDF/VATL_MM/scripts/decomposition/validate.slurm  /lustre/fswork/projects/rech/vnc/uil92qd/ is finish runing ananlysis the results"

   Assistant inspected the Slurm script, job accounting, validation logs, metrics,
   exported component volumes and inference code. Reported that job 1619397
   completed successfully but segmentation was poor. It evaluated epoch 99 on
   three held-out subjects with 100 SDF-supervised adaptation steps, 8192 points,
   seed 42. Mean Dice 0.0685275; HD95 16.8704 mm; ASD 6.1665 mm; surface Dice
   0.07906; clDice 0.06386. Predicted vessel volumes were 15.7–27.0 times truth.
   Atlas-only results were almost identical to full predictions; crop coverage
   included essentially all reference vessel voxels. Suggested investigation.

2. User: "can you help me check the problem, and propose useful solutions, I feel very sad about it since I have stuggled very much for solving this problem"

   Assistant acknowledged the frustration and investigated targets, sampling,
   architecture, adaptation, loss history and physical coordinate handling.
   Found consistent validation SDF signs and outside-vessel millimetre distances,
   excessive atlas prediction regions and ineffective residuals during both
   training and validation. Proposed a one-real-subject capacity test, existing
   training-subject reconstruction, controlled residual/sampling/registration
   ablations, and full-grid monitoring before scaling up. Explained that the
   leading explanation remains a hypothesis until those experiments are run.

3. User: "ok do as you propose, and give abation study scripts in the folder, and save our conversation in the agent md and context md"

   Assistant accepted implementation, diagnostic runs, ablation scripts, and
   `AGENTS.md`/`CONTEXT.md`. Prepared the study below. GPU submission and result
   status must be read from the execution-status section and `jobs.tsv`.

## Source run and measured evidence

- Original checkpoint: `../training_outputs/decomposition_1590276/checkpoint_epoch_99.pth`.
- Validation: `../training_outputs/validation_1619397`.
- Original job: COMPLETED, exit 0:0, elapsed 00:02:03.
- Validation IDs: IXI403-Guys-0965, IXI037-Guys-0704, IXI050-Guys-0711.
- Dice: 0.0872154, 0.0675556, 0.0508115. Atlas without residual: 0.0870479,
  0.0674912, 0.0505920. Voxel precision 0.0263–0.0464; recall 0.689–0.726.
- SDF-negative regions exactly match vessel label 1 for these three subjects.
  Outside distances within 5 mm match independently recomputed spacing-aware EDT
  to about 7e-8 mm mean absolute error. This does not audit every training file.
- 79.7–87.8% of false positives are >2 mm outside reference vessels; 30.4–53.6%
  are >5 mm outside. Mean |residual| is 0.0021–0.0095 mm; maxima 0.0083–0.0309 mm.
- The configured sampler gives about 30% vessel samples on the three subjects,
  versus 0.23–0.36% native in-crop vessel voxels. Pool overlap is intentional.
- Epoch-99 rank-0 logged-step mean atlas SDF MAE 1.07184 mm; final MAE 1.07185 mm;
  mean |residual| 0.00167 mm. The residual was ineffective during training too.
- Legacy rotations use ZXZ Euler angles initialized at zero. The rotation-matrix
  Jacobian at identity has rank 2 (singular values about 2, 1.414, 0). This is a
  confirmed local singularity; its contribution to segmentation error is unproven.
- Threshold -0.75 mm gives mean validation Dice 0.11148: insufficient improvement.
  This sweep used validation truth and is diagnostic, not an unbiased result.
- Adaptation objectives decreased modestly; no wholesale divergence was observed.
- Detailed original analysis and figures: `../training_outputs/validation_1619397/diagnostics/REPORT.md`.

## Hypotheses and experiment order

Leading hypothesis: a broad shared atlas combines imperfectly aligned vessel
locations, while the rigid transform and weak smooth residual cannot reconstruct
individual branches. Oversampling can amplify this population fitting bias;
oversampling alone is not necessarily wrong for a single-subject fit.

First establish a strong real-data one-subject fit (debugging target Dice >0.90),
and compare the old model's training reconstruction to its held-out reconstruction.
Then test residual capacity, residual scale/LR/penalty, sampling mixture, and
rotation parameterization individually on the same four training subjects.
Use multiple seeds and held-out tuning only after screening. Keep test data separate.

## Implementation in this session

- Added full segmentation-grid evaluation in `models/decomposition_evaluation.py`.
  Reports Dice, precision, recall, volume ratio, atlas Dice, residual magnitude,
  physical surface distances, and optional vascular metrics. Outside crop counts
  as background; reference truth is not cropped.
- Added optional single-process training monitoring via
  `decomposition.debug_evaluation`, preserving RNG states around evaluation.
  Writes `reconstruction_metrics.jsonl` and `best_training_reconstruction.pth`.
  This best checkpoint uses TRAINING Dice, not held-out validation.
- Added residual/code gradient logging to training.
- Added opt-in rotation-vector registration with finite identity derivatives.
  Legacy checkpoint behavior stays the default.
- Added opt-in residual Fourier features and uniform-crop sampling mixture.
  The latter changes the training distribution; no inverse weighting is claimed.
- Added scripts in `scripts/decomposition/ablations/`: prepare, evaluate, submit,
  Slurm runner, summarize and README. See README for the complete matrix.
- Added meaningful regression tests for rotation Jacobian/inverse/recovery,
  checkpoint compatibility, Fourier gradients, sampler endpoints/cropping,
  and full-reference metrics including crop misses.
- Extended synthetic smoke pipeline to exercise diagnostics and resume.

## Prepared study

Directory: `../training_outputs/diagnostic_study_20261027`.
Source checkpoint above; 500 epochs per training experiment, single H100, FP32,
original decoder size, seed 42, full-grid metrics after epoch 1/every 50/final.

Pinned four training IDs from the source checkpoint:

- IXI425-IOP-0988 (also the one-subject control)
- IXI070-Guys-0767
- IXI278-HH-1771
- IXI202-HH-1526

Initial jobs: `existing_train` (first three saved training subjects, no adaptation)
and `single_sdf`. Follow-up variants: `single_aux`, `small_reference`,
`small_no_residual`, `small_alpha1`, `small_residual_lr`,
`small_no_residual_penalty`, `small_fourier`, `small_uniform50`,
`small_less_interior`, `small_rotvec`, `small_no_eikonal`.

Each small-cohort experiment starts from scratch with stages disabled. Compare
each small_* variant to small_reference, not directly to the original large run.
The single_aux control adds image and tissue losses together to single_sdf.

## Remaining work and limits

- Initial jobs are complete; see the results section below. Next compare single_aux
  against single_sdf before interpreting the four-subject ablations.
- The broad ablation launcher is prepared; inspect initial diagnostics before use.
- Frozen-atlas residual-only fitting and adaptive hard-negative mining remain
  follow-ups if the screening supports them.
- Adapter resampling, independent rotation/translation/code learning rates and
  best-monitoring-state selection are proposed follow-ups, not implemented here.
  The evaluation CLI can vary steps using the unchanged adapter for comparability.
- New opt-in features do not establish improved accuracy until measured.
- Preserve the substantial unrelated pre-existing uncommitted changes.

## Execution status

Prepared all 12 resolved experiment YAMLs and verified target files exist for the
four pinned subjects. Unit/regression checks: 15 decomposition tests and 5
validation-pipeline tests passed. Synthetic diagnostics integration passed, including
stages A/B/C, full-grid monitoring, Fourier features, rotation vectors, uniform
sampling, checkpoint save/load and matched resumed training losses.

Submitted at approximately 14:36 Paris time:

- `existing_train`: Slurm job **1624893**, COMPLETED, exit 0:0, elapsed 00:00:59.
- `single_sdf`: Slurm job **1624894**, COMPLETED, exit 0:0, elapsed 00:13:08.

Both use `scripts/decomposition/ablations/run.slurm` and write logs into the study
directory. The next single_aux job is recorded below. Other ablations have not
been submitted by the assistant. Do not
resubmit these names: `jobs.tsv` records the jobs.
Verification logs and source SHA256 manifest are saved under `verification/`.
Run `summarize.py --study ../training_outputs/diagnostic_study_20261027` as results arrive.

## Initial real-data diagnostic results

The user asked which scripts to run one by one; the assistant provided the order:
inspect initial jobs, summarize, single_aux, small_reference, then individual
small-cohort variants. The user then reported the two initial jobs finished and
asked the assistant to check their accounting/results.

- Existing model training Dice: IXI425 0.0838607; IXI070 0.0632965; IXI278 0.1023660;
  mean 0.0831744. Training precision is only 3.29–5.45%, with predicted vessel
  volumes 15.6–25.7 times the reference. The failure is present on training data,
  so it is not only a held-out adaptation/generalization problem.
- Single-subject SDF-only fit on IXI425, 500 epochs: Dice 0.8391211, precision
  0.7325139, recall 0.9820439, volume ratio 1.34065, HD95 1.05468 mm,
  ASD 0.20107 mm. Its old-model Dice on the same subject was 0.0838607.
- Dice improves throughout: epoch 100 0.33850; 200 0.60178; 300 0.72635;
  400 0.79281; 450 0.80974; 500 0.83912. Best monitored epoch is the final epoch.
  The curve is still improving, and the >0.90 debugging target was not reached.
- This demonstrates useful real-data fitting capacity in the basic pipeline.
  It does not establish population performance or identify a unique cause: the
  control changes cohort size, epoch budget, auxiliary terms and decomposition.
- Independently checked single-subject vessel crop coverage is 100%, despite
  only 79.37% of total volume being in crop. The remaining fitting error is not
  caused by excluded reference vessels. Final sampled SDF MAE is 0.14409 mm.
- Comparison tables and learning curves were refreshed. Further measurements
  are in `initial_results.json`; see `INITIAL_RESULTS.md` for the interpretation.
- Recommended next existing script: single_aux (matched single-subject auxiliary
  loss comparison), followed by small_reference. No new jobs were submitted in
  the results-inspection turn.

## Single-auxiliary-loss comparison submitted

User: "so do single_aux".

Submitted `single_aux` as Slurm job **1625823** using the prepared configuration
and the existing single-H100 runner. It adds image and tissue losses to the
one-subject SDF control, with the same subject, seed, architecture and 500 epochs.
Output: `../training_outputs/diagnostic_study_20261027/single_aux`.
Log: `../training_outputs/diagnostic_study_20261027/single_aux_1625823.log`.
Submission is recorded in `jobs.tsv`. Compare its full-grid learning curve with
single_sdf (final Dice 0.8391211). No accuracy result is available at submission.

## Single_aux completed and inspected

User: "sdf_sing is finished". Inspected job 1625823: COMPLETED, exit 0:0,
elapsed 00:14:26. Final single_aux Dice 0.8464377 versus single_sdf 0.8391211
(+0.7317 percentage points). Precision 0.7430614 versus 0.7325139; recall
0.9832262 versus 0.9820439; volume ratio 1.32321 versus 1.34065. HD95 is
essentially unchanged at 1.05468 mm; ASD is slightly worse (0.20595 vs 0.20107 mm).
Both use IXI425, seed 42, 500 epochs, no residual or registration. The curve is
still improving, with final epoch the best monitored single_aux epoch.

This single-subject, single-seed comparison gives no evidence that image/tissue
objectives are the principal cause of the original failure. Their final Dice
effect is small and positive here; this does not exclude interactions with
population fitting, other regularizers or different subjects. Next recommended
prepared experiment is small_reference (four subjects, original objective terms,
registration/residual enabled, stages disabled). No new GPU job was submitted
in this inspection turn. Comparison outputs and learning curves were refreshed;
`single_aux_comparison.json` records metrics and resolved config differences.

## Four-subject small_reference submitted

User: "ok run small_reference: four subjects with registration".

Submitted **1626673** (`sdf_small_reference`) using the prepared small_reference
configuration, one H100, 500 epochs. Confirmed registration, residual, image/tissue,
surface/Eikonal and sampled graph-SDF supervision are enabled; staged freezing is
disabled. Cohort: IXI425-IOP-0988, IXI070-Guys-0767, IXI278-HH-1771,
IXI202-HH-1526. All image, segmentation, SDF and graph input files exist.

Output: `../training_outputs/diagnostic_study_20261027/small_reference`.
Log: `../training_outputs/diagnostic_study_20261027/small_reference_1626673.log`.
Submission recorded in jobs.tsv. Full-grid reconstruction metrics cover all four
subjects after epoch 1, every 50 epochs, and the final epoch. Compare this run
with the small_* variants; its comparison with single_aux changes several factors,
so a decrease alone cannot isolate registration or population size as the cause.
No results were available at submission time.

## Four-subject small_reference completed and inspected (2026-10-28)

User asked to read context and inspect the current four-subject results. Slurm
job **1626673** COMPLETED, exit 0:0, elapsed 00:59:27; all 500 epochs finished.
Protocol is full-grid training reconstruction without adaptation, not held-out
image-only segmentation. Final epoch is the best monitored mean Dice.

- Final mean Dice **0.4414607**. Per-subject Dice: IXI425 0.4478681;
  IXI070 0.4178041; IXI202 0.3693399; IXI278 0.5308306.
- Predicted/reference volume ratios respectively 3.1871, 3.7101, 3.9842,
  2.6295. Precision is 0.2310–0.3664; recall is 0.9204–0.9839.
- Mean Dice curve: epoch 100 0.11484, 200 0.23499, 300 0.34838,
  400 0.40066, 450 0.43504, 500 0.44146. Improvement continues but slows.
- Atlas-only Dice equals final Dice for every subject. Full-grid mean absolute
  residual is approximately 1.51e-6 mm, maximum approximately 2.75e-6 mm.
  The enabled residual is effectively inactive in this run.
- IXI425 Dice is lower than its matched-subject single_aux score (0.84644),
  but cohort, registration and residual change together, so this does not
  isolate registration or establish a unique cause. The broad atlas / ineffective
  subject-specific correction hypothesis remains consistent with measurements.
- Refreshed comparison.csv, comparison.json and learning_curves.png in the study
  directory. No new jobs submitted. A useful next prepared diagnostic is
  small_no_residual against small_reference; subsequent residual ablations can
  test why correction contributes so little. No fix has yet been demonstrated.

## Residual-collapse investigation authorized and submitted (2026-10-28)

User: "record your results in context and run the experiments".

The next-step plan prioritizes auditing residual optimization and the matched
small_no_residual_penalty experiment. small_reference has mean precision 0.28919,
recall 0.95133, predicted/truth volume ratio 3.37772, HD95 9.49289 mm and ASD
2.17841 mm. Its residual contributes no measured Dice improvement.

Submitted **1655863** (`small_no_residual_penalty`), one H100, 500 epochs, same four
subjects and seed as small_reference. Verified resolved YAML differs ONLY in
residual magnitude penalty weight (0.01 to zero). Output: `/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/training_outputs/diagnostic_study_20261027/small_no_residual_penalty`.
Submission recorded in jobs.tsv; this is not an accuracy result.

Compare residual magnitude, Dice, precision, volume ratio and learning curves
against small_reference. If residual remains inactive, next test small_residual_lr;
if active but inadequate, consider small_fourier. These conditional follow-ups
have not been submitted. No full ablation matrix was launched.

Residual static/checkpoint audit: residual and subject codes are separate Adam
parameter groups, enabled throughout stage C; backward and optimizer.step are
called. Head weights initialize at std 1e-5 and residual output is scaled by 0.1.
Final checkpoint optimizer/code measurements:

```json
{
  "residual": {
    "lr": 0.0001,
    "parameters_with_state": 8,
    "step_min": 8000.0,
    "step_max": 8000.0
  },
  "latent": {
    "lr": 0.0005,
    "parameters_with_state": 1,
    "step_min": 8000.0,
    "step_max": 8000.0
  },
  "codes": {
    "rms": 0.0007411822443827987,
    "max_pair_distance": 0.006940906401723623
  }
}
```

These confirm optimizer state advancement and distinct final subject codes, not
useful subject-specific learning. Per-loss gradient competition has not yet been
measured; the penalty ablation tests its practical effect.

## No-residual-penalty result inspected (2026-10-28)

User: "read the context and check the latest result".

Slurm job **1655863**, small_no_residual_penalty, COMPLETED, exit 0:0,
elapsed 01:03:48. All 500 epochs completed; final epoch is the best monitored
mean Dice. This is four-subject full-grid training reconstruction without
adaptation, not held-out image-only segmentation.

- Final mean Dice **0.4534255**, versus reference **0.4414607**: +1.1965
  percentage points. Per-subject Dice: IXI425 0.4733089; IXI070 0.4274128;
  IXI202 0.3837912; IXI278 0.5291891. Three improve; IXI278 decreases slightly.
- Mean absolute residual rises from 1.51e-6 mm to **0.0031692 mm**;
  maximum across subjects is 0.0033146 mm. Removing the penalty substantially
  increases its magnitude, but correction remains very small in physical units.
- Mean atlas-only Dice is **0.4530434**: adding the residual improves mean Dice
  by only approximately **0.0382 percentage points** within this checkpoint.
  The between-run Dice gain cannot all be attributed to direct residual correction;
  the atlas/registration also follow a different optimization trajectory.
- Mean predicted/truth volume ratio remains approximately **3.339**, with high
  recall and low precision. The broad false-positive region remains unresolved.
- Mean Dice curve: epoch 100 0.11550; 200 0.25673; 300 0.36733;
  400 0.43355; 450 0.44201; 500 0.45343. Still improving at the end.

Interpretation: the penalty suppresses residual magnitude in this controlled
single-seed comparison, but removing it alone does not produce useful
subject-specific reconstruction or solve the population fit. A prepared next
optimization diagnostic is small_residual_lr against small_reference (its
configuration must be checked before submission). No new job was submitted in
this inspection turn. Comparison tables and learning curves were refreshed.

## User-directed removal and next inference plan (2026-10-28)

User: "I think you now delete all the experiments related to residual, and delete
the residual part in the network, we should not still to this as this is not the
real problem, analysis what we can do next so that during inference I will not
have such bad dice of the vessel".

Implemented removal of SubjectResidualSDF, subject code embeddings, residual/code
optimizer groups, magnitude/smoothness/latent losses, residual gradient logging,
residual export and the zero-residual CLI switch. Unseen adaptation now fits only
rigid parameters to supplied SDF samples. Atlas/rigid training is joint; the old
residual-only freezing stage is removed. Legacy Euler remains the default.

Deleted residual-specific variants small_no_residual, small_alpha1,
small_residual_lr, small_no_residual_penalty and small_fourier from the active
study matrix, and deleted their existing generated YAMLs, logs and run folders
(including the completed penalty-ablation checkpoints). Deleted residual_only,
no_latent and no_residual_regularization repository configs. Preserved original
source checkpoints, mixed reference/single-subject controls, jobs.tsv and this
measured history. The old study is closed to new runs; its remaining comparison
outputs are refreshed. Prepare a NEW study before further GPU experiments.

New checkpoint format: vatl_sdf_atlas_rigid_v2. Current resume/inference/validation
reject residual-era v1 checkpoints; they are not silently reinterpreted. Study
preparation can read v1 cohort/config metadata and strips obsolete settings,
starting fresh v2 fits. Existing source results remain historical measurements;
there is no measured real-data Dice improvement from removing the branch.

Analysis from code: the shared decoder takes coordinates and demographic
conditions, not local MRA features. A shared grid plus rigid map cannot represent
different branch connectivity at identical conditions. Training image reconstruction
is an auxiliary objective, not an image-conditioned inference mechanism. The
current adaptation loss uses ground-truth SDF, so its held-out scores are oracle
reconstruction diagnostics rather than image-only segmentation performance.
These are architectural/protocol observations, not proof of a unique failure cause.

Assuming deployment provides MRA but no vessel labels/SDF, next priorities:
longer single-subject fit to establish convergence; known-rigid-transform copies
and anatomical alignment checks; matched v2 sampling experiments targeting excess
positive volume; an image-to-vessel baseline or spatial MRA encoder feeding the
SDF decoder directly. No additive residual is proposed. Nonrigid image registration
may help correspondence but cannot be assumed to recover missing branches.
Detailed proposed experiments and decision gates: scripts/decomposition/NEXT_STEPS.md.
These follow-ups are proposed/prepared, not completed. No GPU job submitted.

Verification after removal: 16 decomposition tests and 5 validation-pipeline tests
passed in the project H100 module environment (CPU execution). Full synthetic
integration passed training, checkpoint resume, component/atlas/mask exports and
unseen SDF-supervised rigid fitting. A second training-only integration explicitly
compared resumed loss records with uninterrupted training and passed exactly.
Fresh-study preparation from synthetic residual-era checkpoint metadata passed:
all seven generated configs instantiated v2 models without residual/code parameters.
Artifact/summary checks confirm all five retired variant names are absent from
remaining comparison rows and their generated configs/run folders/logs are gone.
Python compilation and shell syntax checks passed. These are implementation
checks, not measured improvements on real subjects.


## Population-atlas clarification and registration study (2026-10-28)

The user clarified that the long-term objective is continuous, age/sex-conditioned
population cerebrovascular anatomy and normative analysis, not primarily an
MRA-to-vessel segmentation project. Population occupancy, canonical/individual
geometry, and normative modelling must be evaluated separately. This supersedes
the earlier image-only-deployment assumption in NEXT_STEPS.md. No residual or
subject code is reintroduced; no old checkpoint is resumed.

Read and downloaded Xu et al., Scientific Data 2026,
https://doi.org/10.1038/s41597-026-07710-5. The paper uses structural ANTs SyN
and separate ANIMAL vascular refinement; the available ANTsPy baseline is not
an ANIMAL reproduction. Full methods were inspected from the reference PDF.

Created scripts/decomposition/registration_atlas/ and isolated outputs
../training_outputs/registration_atlas_20261028_01. Each submitted phase has a
code snapshot, source hashes and jobs.tsv entries. Dataset and historical outputs
are read-only. Substantial processing runs in CPU Slurm allocations, no GPUs.

Completed audit job 1661345: actual paired MRA/T1/Seg/SDF/Graph paths exist for
all four subjects. Modalities match geometry; oblique RAS images have different
spacing/FOV. T1 files are on the MRA slab grids, not verified whole-brain native
T1. Upstream transform provenance and original padding/support masks were not
found in the supplied v1 tree. SDF sign exactly matches Vessel label 1 and
independent physical EDT agrees inside and within 5 mm outside at float32
precision. Model crop includes 100% of each subject's vessel voxels.

Current centered-alignment R0: six-pair mean Dice 0.05715, clDice 0.03408,
centerline distance 6.074 mm. These are existing alignment metrics, NOT neural
reconstruction. Selected IXI278-HH-1771 by highest mean R0 pairwise Dice before
registration. All four subjects are near age 20, unsuitable for lifespan modelling.
Known physical rigid point recovery passed (Euler RMS 5.11e-5 mm; rotvec 4.20e-7).
A real-vessel known-affine perturbation control, job 1661938, recovered Dice 1.0
from 0.11415 and mean coordinate error 2.20e-5 mm.

Completed R1 structural jobs 1661793–1661796, aggregate 1661797; completed corrected
R2_incremental jobs 1662170/1662172/1662176/1662179, aggregate 1662184. R0_ants
identity control 1661948 and matched-FOV/deformation comparison 1662292 completed.
On IDENTICAL coverage regions with the same interpolation backend:

- R0_ants: mean pairwise Dice 0.05440, clDice 0.03379, centerline 6.010 mm;
  common-FOV occupancy concentration 0.2901.
- R1 structural: Dice 0.12015, clDice 0.07559, centerline 4.547 mm;
  concentration 0.3386.
- R2 incremental vascular: Dice 0.16892, clDice 0.10285, centerline 3.969 mm;
  concentration 0.3748.

No detected folding; vascular-only p95 displacements 2.99–4.66 mm for non-reference
subjects, incremental minimum Jacobians 0.517–0.755. Total normalization includes
large structural affine/pose changes, p95 up to 32.4 mm, and registered/native
vessel volume ratios up to 1.42. These are not proof of branch preservation.
Correspondence improves, but no causal contribution to neural fitting has yet
been measured; no new v2 neural fit was run. Old Dice scores remain historical.

Identity resampling QC 1661798 and resolution control 1662041 completed. At 1 mm,
native round-trip Dice is 0.869–0.938 with centerline retention 0.861–0.947 and
increased connected-component counts despite approximately conserved volume.
At 0.8 mm, Dice is 0.988–0.998 and centerline retention 0.989–0.999. At 0.5 mm,
this identity control recovers all native masks and component counts exactly.
This supports finer-grid preservation checks before aligned neural supervision,
not a claim of perfect nonlinear resampling. No re-segmentation model was verified.

Failed R2 attempt 1662004/1662005/1662007 produced no inverse composite when
initializing from nonlinear H5; outputs retained but excluded, aggregate 1662008
cancelled. The corrected independent-increment forward/inverse composition passed
reapplication checks and completed in NEW R2_incremental outputs. A supplemental
preservation summary job 1662659 rejected tiny ITK/NIfTI affine roundoff under a
relative-only tolerance; retry 1662712 uses explicit 1e-4-mm absolute tolerance
and a new output. No failed attempt is an accuracy result.

Five new physical/FOV/transform regression tests passed. Group-wise synthetic
integration passed. Existing 16 decomposition and 5 validation tests plus synthetic
training/resume checks passed. A two-iteration four-subject group-wise comparison
1662464 is running; its real outcomes/reference sensitivity are not yet known.
No neural occupancy, encoder, graph generative model or full-cohort training ran.
Report/proposals: registration_atlas/REPORT.md and ARCHITECTURE_PROPOSAL.md.


Registration study completion update: two-iteration group-wise CPU job **1662464**
COMPLETED 0:0 in 10:16; comparison/bias job **1662785** COMPLETED 0:0. On identical
per-pair FOV across all FIVE stages, mean Dice/clDice/centerline mm/concentration:
R0_ants 0.05429/0.03387/5.981/0.2902; R1 0.11986/0.07561/4.541/0.3386;
R2_incremental 0.16874/0.10272/3.954/0.3748; group iteration 1
0.17473/0.10722/3.796/0.3787; group iteration 2
0.21934/0.13892/3.223/0.4102. These are registration metrics, not neural Dice.
The changed matched coverage explains small differences from earlier tables.

Measured mean-bias correction reduces incremental sampling-field RMS by about 20%
as configured: 1.887→1.511 mm and 0.542→0.433 mm on a shared eroded FOV. This is
approximate incremental bias control, not proof of an unbiased atlas. No detected
folding, but group2 minimum total Jacobian 0.248 and up to 0.489% covered voxels
below 0.5. Native round-trip Dice worsens to 0.852–0.932 vs R2 0.882–0.960.
Maximum inverse-consistency p95 remains 0.0372 mm. The two-iteration run did NOT
converge: normalized probability change 0.824. Higher overlap therefore has a
preservation/compression tradeoff. No further rounds/reference sensitivity or
large-cohort runs were launched.

Preservation summary retry **1662712** completed, including native TP/FP/FN,
centerline/component metrics and structural T1 overlays. Original qform/sform
entry difference max 1.44e-6, consistent with float serialization. No gross
structural error was inferred; slab boundaries remain a limitation. Biological
branch preservation and CoW homologous correspondence require further QC.

The next data gate is finer-grid export through fixed transforms followed by
coverage-aware training sampling/losses. Crucially, the current neural loader does
not consume Coverage, so aligned common-grid background cannot yet be used as
unmasked supervision. regrid.py is prepared but not run; no aligned metadata is
claimed training-ready. Fresh matched v2 SDF and neural occupancy baselines remain
pending these preservation/FOV gates; no improvement in v2 neural performance or
normative validity is claimed. No residual, subject code or MRA encoder introduced.

Five registration + 16 decomposition + 5 validation tests and synthetic groupwise
and network-resume checks passed. Input SHA256 manifest covers 22 files; code
snapshots/hashes, verification logs, reports and jobs.tsv retained. All submitted
jobs are finished or explicitly failed/cancelled; none remain pending/running.

## Follow-up scripts and execution authorized (2026-10-28)

User: "so do the next steps, prepare the scipts, and give me the submited order,
record our coversation".

Prepared scripts/decomposition/registration_atlas/next_steps/ with start.py,
submit_next.py, preflight.py, gpu.slurm, evaluate_pair.py, smoke_coverage.py and
ORDER.md. Current isolated follow-up:
../training_outputs/registration_followup_20261028_01.

Submitted CPU finer-grid export **1665634**, COMPLETED exit 0:0 in 16:18.
It reuses fixed R0_ants/R1/R2_incremental transforms, resamples originals once onto
0.8-mm geometry, exports multiclass labels and recomputes physical-mm SDFs.
Original observations/transform studies/checkpoints are preserved. Group2 is not
selected for the first training pair because of its preservation/compression cost.

Submitted CPU preflight/cache job **1666128**, dependent on successful export;
currently running. It validates round-trip Dice/centerlines, Jacobians/inverses,
uncropped inverse-SDF interpolation controls and trusted vessel supervision.
Engineering thresholds are declared in followup.json and preflight source;
failed checks block GPU submission and are not relaxed automatically.

Implemented opt-in data_loading/coverage_dataset.py. The standard Data loader is
unchanged. train_decomposition.py selects CoverageData only when coverage_key is
configured. CPU caches retain read-only/mmap target arrays and valid point pools;
source/cache fingerprint changes are rejected. All image/tissue/SDF targets share
observed, trusted points; six physical Eikonal stencil positions remain inside
observed support and model crop. Trust uses padded physical EDT clearance beyond
min(abs(raw SDF),tau)+stencil step+half-voxel diagonal. Target SDF is clipped in mm
before every SDF term in both arms; raw files remain unchanged. Coverage is not an
extra target channel. Graph terms and augmentation are disabled in both arms.

Updated decomposition_evaluation.py only for opt-in coverage: unobserved support
is excluded from prediction/truth metrics, while observed truth outside model
crop still counts as missed. Default no-coverage behavior remains intact.
The shared network architecture, absence of residual/codes, and v2 checkpoint
format are unchanged. No legacy checkpoint is resumed.

Prepared matched four-subject 500-epoch/seed42 fits: existing alignment R0_ants vs
fixed nonlinear R2_incremental. Learned rigid and graph supervision are disabled
in BOTH arms. Resolved configs differ only in dataset.tsv_file; metadata/cache
contents reflect the alignment treatment. Monitoring uses observed training
reconstruction. Final paired evaluation maps level sets back through saved inverse
transforms to full original native references, reporting Dice/precision/recall,
volume ratio, HD95/ASD and clDice. Inverse-warped values are not assumed to remain
native SDF distances. No held-out/image-only/normative claims are made.

Submission order: export → CPU preflight/cache → GPU existing-alignment control →
GPU fixed-prealignment fit (afterok existing) → GPU paired evaluation (afterok
prealigned). GPU phases require inspection of a completed passing gate. Occupancy,
reference/regularization sensitivity and larger cohorts remain gated on the paired
results; no blanket full study is launched. Each phase uses a frozen code snapshot
and appends jobs.tsv; outputs refuse overwrite.

Validation: four new coverage tests passed (including poisoned missing-FOV targets,
physical stencil support, cache/config rejection, observed evaluation and crop
misses). Tiny coverage-aware all-head/Eikonal training and exact resume passed.
The full 20 decomposition tests and 5 validation tests plus original diagnostics
training/resume smoke passed. A newly named queue.py initially shadowed Python's
standard queue module; it was renamed submit_next.py before GPU submission and
the integration re-ran successfully. No failed smoke is an accuracy result.


### Follow-up preflight outcome and finer candidate submitted

CPU preflight **1666128** completed exit 0:0, but its SCIENTIFIC GATE FAILED.
All observed-point/trusted-vessel, fold, point inverse, NN mask and centerline gates
passed at 0.8 mm. The additional pure inverse-SDF linear-interpolation control
failed Dice >=.95 in four arm/subject cases: existing IXI070 0.935167, IXI278
0.937620, IXI202 0.938925; prealigned reference IXI278 0.937620. NN preservation
alone therefore did not certify the level-set inverse path. No GPU training was
submitted and no threshold was relaxed. This is an interpolation diagnostic, not
poor learned-network performance.

Following the predeclared finer-grid option, prepared a NEW candidate
../training_outputs/registration_followup_20261028_02 with 0.5-mm exports directly
from ORIGINAL observations and the SAME fixed R0/R1/R2 transforms. No 0.8-mm
resampling is reused. Source `_01` and all old studies remain preserved. The first
0.5-mm sbatch call was rejected before allocation because Jean Zay forbids --mem;
corrected to CPU-core-based memory allocation (16 cores), with up to two hours.
Safe --submit-prepared recovery requires no existing job/output and records a
new actual ID; no failed submission is claimed as a completed experiment.

Submitted `_02` CPU export **1666316**, followed by CPU preflight/cache **1666319**
with afterok dependency. They are the current active jobs; their quantitative
outcome is pending. The 0.5-mm full-grid export is larger and may take substantially
longer. Model architecture, transforms, cohort, future fit seed/epochs and both
controlled-arm policies are unchanged.

ORDER.md now contains exact commands and actual submitted order, including the
failed 0.8-mm gate and pending 0.5-mm jobs. Prepared GPU existing → prealigned →
evaluate sequence is NOT submitted: must inspect a completed passing gate first.
0.5-mm fits have a four-hour single-H100 wall cap to accommodate dense monitoring.
Occupancy and larger-scale experiments remain conditional on paired results.
All 20 decomposition tests (including the four new coverage tests), 5 validation
tests, coverage training/resume smoke and unchanged original diagnostics/resume
smoke passed. No accuracy fix is claimed from tests or submission.

## GitHub branch snapshot requested (2026-10-29)

User: "push your current work to the github branch, give a new name called regis_dev".

Created local branch regis_dev from the existing main checkout to publish the
current workspace, including existing project changes, the simplified model,
registration/coverage follow-up scripts, tests, reports and handoff files.
Added ignore rules for generated Python caches and notebook autosaves. Experiment
volumes and checkpoints remain in sibling training_outputs, outside this Git
repository. The snapshot does not claim new experimental results.
