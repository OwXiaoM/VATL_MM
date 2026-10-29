# Project handoff: VATL SDF decomposition

Read `CONTEXT.md` for the conversation record, investigation, experiment status,
and unresolved questions. The user asked to preserve this context on 2026-10-27.

## Current authorized work

The research objective is a continuous population-conditioned cerebrovascular
atlas and eventual normative modelling, not primarily image segmentation. Separate
population occupancy, canonical geometry, individual reconstruction and normative
validity. No residual network or subject-specific code should be reintroduced.

The four-subject CPU registration study is completed through R0/R1/R2 and two
group-wise iterations. Read `scripts/decomposition/registration_atlas/REPORT.md`
for measured correspondence gains and deformation/resampling tradeoffs. Next are
finer-grid preservation and Coverage-aware sampling/losses before fresh matched
neural experiments. Current common-grid exports are not yet training-ready. Do
not launch a full 497-subject run or resume incompatible historical checkpoints.

## Repository and environment

- Repository: `/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM`.
- Outputs: sibling `../training_outputs/`; never overwrite an experiment directory.
- H100: source `scripts/decomposition/environment.sh`; use its module Python.
  The existing A100 venv interpreter is incompatible with the H100 module stack.
- GPU work belongs in Slurm allocations. The small study uses one H100 per job.
- The working tree contains substantial pre-existing uncommitted work. Preserve it.
- Negative SDF means vessel; model coordinates are centered world mm divided by
  half the configured world box. Keep SDF separate from intensity normalization.
- Default rotation remains `legacy_euler` for checkpoint compatibility. `rotvec`
  is opt-in and must not reinterpret old saved Euler parameters.
- Full segmentation-grid evaluation includes truth outside the model crop.
  Training reconstruction and SDF-supervised held-out adaptation are different
  protocols; neither should be presented as image-only segmentation performance.

## Historical diagnostic study entry points

- `scripts/decomposition/ablations/README.md`: experiment matrix and commands.
- `prepare.py`: resolved configs, fixed subject cohort, study protocol.
- `submit.sh --initial`: fresh v2 one-subject fit only; the old study is closed.
- `submit.sh --ablations`: remaining small-cohort screening experiments.
- `summarize.py`: partial/completed comparison tables and learning curves.
- Study directory: `../training_outputs/diagnostic_study_20261027`.
- `jobs.tsv` in that directory is the source of truth for submitted job IDs.

## Checks

Run in the project environment:

```bash
python -m unittest discover -s tests -p 'test*decomposition*.py'
python -m unittest discover -s tests -p 'test_validation_pipeline.py'
python scripts/decomposition/smoke_pipeline.py --diagnostics --training-only --resume-check
```

Update `CONTEXT.md` when results arrive. Clearly distinguish measured findings,
hypotheses, queued jobs and completed experiments. Do not claim a fix from scripts
or a successful Slurm submission alone.

## Updated direction (2026-10-28)

The user explicitly requested deletion of residual experiments and removal of the
residual network. The active model is now shared atlas plus rigid registration,
without subject codes. New checkpoints are `vatl_sdf_atlas_rigid_v2`; do not resume
or silently reinterpret old v1 checkpoints. The 20261027 study is historical and
closed to new submissions. Residual-specific artifacts were deleted; jobs.tsv and
CONTEXT.md retain provenance/results. Prepare a NEW study for remaining variants.
See `scripts/decomposition/NEXT_STEPS.md` for the prioritized capacity, alignment,
sampling and image-conditioned inference plan. Image-only segmentation is the
assumed deployment goal; no image encoder or image-only adapter is implemented.
Do not restart residual ablations. No new GPU jobs were submitted for this change.

## Population-atlas objective clarification (latest user instruction)

The primary goal is population-conditioned cerebrovascular anatomy and normative
modelling, NOT conventional image segmentation. This supersedes the earlier
image-only-deployment assumption. Separate correspondence, population statistics,
canonical geometry and individual reconstruction. Work is authorized on the four
pinned subjects through the staged registration/representation gates, starting
with audit and CPU classical registration. Scripts and current report live in
`scripts/decomposition/registration_atlas/`; output study is
`../training_outputs/registration_atlas_20261028_01`. Keep every phase reproducible,
record jobs.tsv, preserve old results, and inspect quantitative outcomes before
expanding. Do not add residuals/codes/encoders or large graph/generative models.


Registration study 20261028_01 is complete through R0/R1/R2 and two group-wise
iterations. Read registration_atlas/REPORT.md and CONTEXT.md for measured gains
and compression/resampling tradeoffs. All CPU jobs are finished. No neural fits
were launched: finer-grid preservation plus Coverage-aware sampling/losses are
required first. Do not treat common-grid padding as observed background. The
current neural loader does not consume Coverage. regrid.py is prepared, not run.

## Active registration study entry points

- `scripts/decomposition/registration_atlas/README.md`: commands and conventions.
- `REPORT.md`: confirmed results, hypotheses, limitations and pending gates.
- `ARCHITECTURE_PROPOSAL.md`: probabilistic atlas and normative design proposals.
- `prepare.py`, `pipeline.py`, `submit.py`, `cpu.slurm`: isolated CPU studies.
- `compare.py`, `groupwise_compare.py`: common-FOV quantitative comparisons.
- `resolution_control.py`, `preservation_summary.py`: interpolation/preservation QC.
- `regrid.py`: prepared finer-grid export; NOT executed or training-ready yet.
- Current output: `../training_outputs/registration_atlas_20261028_01`.
- `jobs.tsv` and per-phase source snapshots/hashes are submission provenance.

## Current follow-up (latest)

User authorized proceeding with the next scripts and submission order and asked
that conversation be recorded. See registration_atlas/next_steps/ORDER.md.
Output: ../training_outputs/registration_followup_20261028_01; jobs.tsv tracks
actual submissions. Export 1665634 completed; preflight/cache 1666128 was submitted.
Coverage-aware sampling is now IMPLEMENTED as an opt-in CoverageData adapter;
standard Data behavior remains unchanged. All supervised heads share trusted
observed points and physical Eikonal margins. SDF targets are clipped in mm in
both controlled arms. Do not bypass preflight gates or reuse stale caches.
Fresh paired fits use fixed preprocessing, learned rigid OFF and graphs OFF in
BOTH arms, seed42/500 epochs. Check the latest CONTEXT.md/jobs.tsv for GPU status.
The old statements that Coverage was unimplemented describe the previous turn.
Do not reintroduce residuals/codes or launch a full cohort. Inspect paired results
before occupancy or further architecture/registration expansion.


Follow-up status update: `_01` 0.8-mm export/preflight finished, but inverse-SDF
interpolation Dice fell below the declared .95 gate (0.935–0.939 in four cases).
No GPU was submitted. Active candidate is
`../training_outputs/registration_followup_20261028_02` at 0.5 mm, export **1666316**
and dependent CPU preflight **1666319**. Keep `_01` as failed-gate evidence. Inspect
`_02/preflight/gate.json` before any GPU phase. All actual IDs are in each jobs.tsv.
The prepared GPU order is existing → prealigned → evaluate; still unsubmitted.
