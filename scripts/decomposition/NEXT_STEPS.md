# Population cerebrovascular atlas: current investigation

The user's clarified objective supersedes the earlier assumption of image-only
individual segmentation. Population occupancy P(V(x)=1 | age, sex), canonical
continuous geometry, and individualized reconstruction are different targets.
Normative modelling requires population variability, held-out likelihood and
calibration, not just training Dice. No MRA encoder or subject residual is being
introduced. Historical results remain measurements of earlier architectures.

Current priority: the four-subject coordinate/SDF audit, existing-alignment R0,
structural T1-guided nonlinear R1 where paired data exist, and vascular-feature
nonlinear R2. Construct classical FOV-aware probability maps and quantify
correspondence, deformation and thin-vessel resampling before neural expansion.

Implementation, conventions and commands:
[registration_atlas/README.md](registration_atlas/README.md).
Verified/pending results: [registration_atlas/REPORT.md](registration_atlas/REPORT.md).

Completed: R0, structural R1, vessel-feature R2 and two group-wise iterations,
plus classical probability maps, inverse-space/deformation controls and multiresolution
resampling diagnostics. See the report for matched-FOV measurements. These are
registration results, not performance of a newly trained neural model.

The group-wise run improved correspondence but increased compression and worsened
native round-trip label preservation; it did not converge within two iterations.
The data gates now take precedence over further registration or neural expansion:

1. Re-export fixed transforms from original observations on a finer grid using
   regrid.py, then validate native inverse-space preservation. The script is
   prepared but has not been run. Its outputs are explicitly not training-ready.
2. Add Coverage-aware point selection and loss masking for every SDF/image/tissue
   pathway. The existing neural loader does not consume Coverage; common-grid
   padding must not be treated as observed absence. Match graph policy in both arms.
3. Test conservative registration/feature and initial-reference sensitivity with
   anatomical branch/CoW QC before treating a sharper atlas as preserved variation.
4. Run a fresh matched v2 shared-SDF comparison, existing vs fixed nonlinear
   prealignment, with the same cohort, architecture, objective, sampling budget,
   optimizer and seed. No incompatible checkpoint resume.
5. Implement the minimal Bernoulli neural occupancy field against the classical
   FOV-aware probability baseline, evaluating likelihood, Brier and calibration,
   separately from individual Dice. Use a larger appropriate cohort before
   estimating age/sex effects or normative distributions.

No MRA encoder, residual, subject code, large graph/generative model, or full-cohort
training is introduced. Architecture directions are proposals only. The previous
next-steps document is retained in registration_atlas for historical context, with
its segmentation assumption explicitly superseded here.


## Active authorized follow-up

Scripts and exact gated submission order are now prepared in
[registration_atlas/next_steps/ORDER.md](registration_atlas/next_steps/ORDER.md).
The 0.8-mm export completed (1665634); CPU preflight/caches were submitted
(1666128). The opt-in coverage-aware adapter is implemented and regression/smoke
checked; standard Data remains unchanged. Follow the current jobs.tsv and
CONTEXT.md for later GPU IDs. The previously listed missing Coverage support has
now been addressed for fixed-prealignment, graph-disabled training. Paired model
results are still required before expanding to neural occupancy or a larger cohort.


The 0.8-mm follow-up's NN checks passed, but its inverse-SDF interpolation check
failed the .95 threshold; no GPUs were launched. Active candidate `_02` is now
0.5 mm (CPU export 1666316; dependent preflight 1666319). See the actual ledger in
ORDER.md/jobs.tsv. Inspect a completed PASS before existing/prealigned GPU fits.
