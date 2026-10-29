# Population representation proposal and experimental gates

This is a technical proposal, not an implemented or validated model.

## Statistical target

After a fixed, documented spatial normalization, learn marginal vessel occupancy
p_theta(x, age, sex). The normalized reference defines what variation remains in
this distribution: aggressive vessel normalization may remove geometry relevant
to normative analysis. Preserve transformation-derived shape measurements and
native anatomy separately. Report how much variation registration removes.

Classical baseline: sum(C_i V_i)/sum(C_i), with an accompanying count map and
undefined probabilities outside coverage. With only four near-age-20 subjects,
probabilities are demonstration-level empirical frequencies, not estimates of a
healthy lifespan distribution. Segmentation uncertainty, site effects, FOV and
registration uncertainty are not inter-subject biological variability.

## Minimal continuous occupancy baseline, when the registration gate passes

Reuse coordinate normalization and a small SIREN, with a single Bernoulli logit
output and optional normalized age and sex input. No residual, individual code,
image encoder or new neural registration module is needed. Fit registered binary
observations with their valid coverage. Spatial-query inputs remain continuous.

For an empirical per-voxel objective,

L = sum_x sum_i C_i(x) BCEWithLogits(l_theta(x), V_i(x)) / sum_x sum_i C_i(x).

For the alternative equal-location objective, first average subjects at each
location then average valid locations. These differ when coverage varies; declare
which is used. The classical probability is the pointwise optimum of either at
each location, but finite-capacity fitting weights differ. At identical conditions
and FOV the model should reproduce the classical probability field. If vessel-
balanced queries are used, weight the loss by inverse query inclusion probability
(or explicitly stratified volume weights). Do not use a class-weighted BCE and
interpret its raw sigmoid as calibrated occurrence probability. Calibrate only
on held-out data; keep final test subjects untouched.

Validation: agreement to the classical map, coverage-stratified Brier/log loss,
reliability curves, vessel-support likelihood, occupancy distribution and spatial
features. A uniform-background predictor can have deceptively good global Brier
loss for sparse vessels; report informative stratified metrics alongside the
proper population-weighted score. A held-out subject likelihood remains affected
by registration estimated using its own labels; identify such oracle alignment
and evaluate image-based alignment separately before normative deployment.

The deterministic SDF remains a canonical continuous geometry comparator. The isolated pointwise L1 distance
data term has a weighted conditional-median optimum under its actual sampling;
network capacity and the auxiliary terms can alter that optimum. It does not
estimate occupancy probability. Neither mean distance nor sigmoid of
an arbitrary distance scale is automatically calibrated vessel occurrence.

## Continuous lifespan atlas after the small-cohort gates

Use a much larger healthy cohort with age coverage, sex/site balance, reliable
coverage and standardized segmentation/QC. Fit smooth age effects with validated
capacity (e.g. explicit low-order splines feeding the probability field), and test
sex interactions against simpler models. Quantify subject-level/bootstrap
uncertainty. Four subjects cannot identify age effects or separate scanner/site
and biology. Separate template construction/training, tuning and frozen testing;
fit population preprocessing/reference selection only in the training cohort.

Normative deviations should combine predeclared native/registered regional vessel
burden, radius, density, topology/graph features and deformation statistics, with
held-out calibration. A sum of voxel Bernoulli surprises is a composite score:
spatially correlated voxels make a naive independence interpretation incorrect.
Do not call it a subject abnormality probability without empirical calibration.

## Later correlated geometry and topology proposals

A spatially correlated SDF distribution could model geometric variation, but must
specify correlation, sampling and resulting topology. Independent mean/variance
outputs do not generate valid connected vascular networks.

A possible later factorization is p(G,S | age,sex) = p(G | age,sex) p(S | G,age,sex).
G would describe homologous branch identities, presence and connectivity; S would
condition continuous radius/centerline surfaces on G. Prerequisites are stable
branch naming, graph correspondence, uncertain/absent-branch labels, and explicit
anatomical validity evaluation. Diffusion or large generative models are premature
before those data and classical baselines exist. No graph model is implemented.

Individual inference, if later required, needs individual observations and a
separate evaluation protocol. An MRA encoder is one option, not a prerequisite
for a population probability atlas.

The four observed ages are not exactly identical. A sufficiently flexible
conditional network could memorize these four cases through age itself; therefore
poor four-subject fitting is not proved mathematically inevitable by the absence
of individual codes. Smooth population effects and case memorization must be
distinguished when expanding the cohort.

## Controlled shared-SDF registration test

Only after correspondence improves without unacceptable deformations or branch
loss, prepare a fresh v2 matched pair using existing alignment vs fixed nonlinear
prealignment. Same cohort, SIREN/grid, loss terms, graph supervision policy,
sampling, optimizer budget and seed. Recompute aligned SDF from aligned masks in
mm; transform graphs consistently or disable graph terms in BOTH arms. Otherwise
a supposed registration-only comparison changes the supervision. New common-grid products require an explicit Coverage-aware sampler and loss
mask for SDF, image and tissue objectives. The current loader does not yet consume
Coverage; do not silently train on zero-padded missing FOV. Freeze learned
rigid registration in BOTH arms initially so prealignment is the only coordinate
mapping difference. If joint rigid refinement is later tested, match it in both.

Evaluate canonical predictions in aligned space and inverse-resampled original
subject space; include interpolation-only controls and full reference truth.
Distinguish fit loss, Dice/precision/recall/volume/HD95/ASD/clDice and population
probability metrics. The historical 44.15% run is not a simplified-v2 baseline.
Better registration would support a correspondence contribution; unchanged neural
fitting would motivate convergence/objective/representation diagnostics rather
than automatically escalating deformation complexity.
