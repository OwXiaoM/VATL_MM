# Submission order: finer grid → observed-FOV control → matched neural fits

Run from VATL_MM. The active follow-up is:
`../training_outputs/registration_followup_20261028_02`.
`jobs.tsv` is the source of truth: do not resubmit an existing phase name.
Old experiments/checkpoints are preserved; all neural fits start from scratch.

## 1. CPU finer-grid export

The active 0.5-mm export job is **1666316**, followed by CPU preflight
**1666319** (afterok export). The previous 0.8-mm export **1665634** and
preflight **1666128** completed in follow-up `_01`, but the inverse-SDF Dice
gate failed for four arm/subject cases (0.935–0.939 < .95). No GPU jobs were
submitted for that failed candidate; its data and report are preserved.

The new export uses R0 identity, structural
R1 and fixed-reference vascular R2 onto a new 0.5-mm grid, directly from original
observations. Group-wise iteration 2 is excluded from the training pair because
its overlap gain had a compression/preservation tradeoff.

For a NEW independent follow-up only:

```bash
python scripts/decomposition/registration_atlas/next_steps/start.py \
  --source ../training_outputs/registration_atlas_20261028_01 \
  --output ../training_outputs/registration_followup_NEW \
  --spacing .5 --epochs 500
```

## 2. CPU preservation gate and sampling caches

```bash
export FOLLOWUP="$(realpath ../training_outputs/registration_followup_20261028_02)"
python scripts/decomposition/registration_atlas/next_steps/submit_next.py \
  --study "$FOLLOWUP" --phase preflight
```

This phase depends on successful export. It checks native round-trip masks,
centerline retention, folds and inverse consistency; additionally it inverse-warps
ideal aligned SDFs with exactly the level-set interpolation used in neural
assessment. It creates read-only/memory-mapped observed-point caches and two
resolved YAMLs. No GPU is requested. Check `preflight/gate.json` and `SUMMARY.md`.

Predeclared engineering gates (not anatomical/normative validation thresholds):
identity Dice/centerline recall ≥.98; prealigned Dice/centerline recall ≥.95;
ideal inverse-SDF Dice ≥.95; zero detected folds; inverse p95 ≤.1 mm; at least
80% of observed vessel voxels remain eligible for trusted supervision. A failure
blocks GPU submission; thresholds are not relaxed automatically. If .8-mm
nonlinear preservation fails, inspect it and try a NEW .5-mm follow-up rather
than overwriting outputs.

Coverage is geometric support, not proof of original unpadded acquisition FOV.
Trusted points satisfy physical clearance beyond the truncated distance plus the
finite-difference offset and half-voxel-diagonal uncertainty. All stencil points
also remain within the model crop. SDF/image/tissue losses share these observed
points. Distances are clipped in mm before ALL SDF terms in both arms; no intensity
normalization is applied to SDFs. Graph terms and augmentation are disabled.

## 3. Fresh shared-SDF control: existing alignment

Only after inspecting a passing preflight:

```bash
python scripts/decomposition/registration_atlas/next_steps/submit_next.py \
  --study "$FOLLOWUP" --phase existing
```

One H100, four fixed subjects, 500 epochs, seed 42. The 0.5-mm fits have
a four-hour wall-clock cap to allow dense monitoring; this is not a claim of runtime. Same shared decoder/grid,
image/tissue/SDF/Eikonal objectives and point budget in both arms. Learned rigid
registration is disabled in BOTH arms, so fixed preprocessing is the treatment.
No residual, subject code, encoder or incompatible checkpoint resume.

## 4. Fresh shared-SDF fit: fixed nonlinear prealignment

```bash
python scripts/decomposition/registration_atlas/next_steps/submit_next.py \
  --study "$FOLLOWUP" --phase prealigned
```

One H100; this job waits for successful completion of `existing`. Resolved fit
YAMLs differ only in `dataset.tsv_file`. Dataset contents differ because of the
precomputed geometric alignment; pool sizes/sample streams need not be identical.
No old 44.15% checkpoint is treated as the fresh baseline.

## 5. Frozen paired reconstruction assessment

```bash
python scripts/decomposition/registration_atlas/next_steps/submit_next.py \
  --study "$FOLLOWUP" --phase evaluate
```

Waits for the two fits. Uses their final matched epoch, not a truth-tuned threshold
or held-out-selected best checkpoint. Reports observed-aligned and full-original-
native Dice, precision, recall, volume ratio, HD95, ASD and clDice. Inverse-warped
level sets are used only for geometry/masks, not assumed to be native SDF distances.
Observed truth outside the model crop remains a false negative. Missing FOV is
excluded from aligned evaluation. This is TRAINING reconstruction without
adaptation, not unlabeled-MRA segmentation or held-out normative generalization.

## 6. Inspect before further model submissions

Review `evaluation/REPORT.md`, `comparison.json`, learning curves and native
outputs. Only then decide whether the next experiment is longer convergence,
registration regularization/reference sensitivity, sampling/objective controls,
or the minimal Bernoulli occupancy field versus the classical probability map.
No occupancy, encoder, graph generative model or full-cohort run is automatically
submitted by these scripts. Four near-age-20 subjects cannot support lifespan or
normative inference.

## Reproducibility and checks

`submit_next.py` freezes Python/config/shell sources under `source/repo`, records
hashes, and runs CPU/GPU phases from that snapshot. Per-phase IDs append to
jobs.tsv; outputs refuse overwrite. GPU launch requires a completed PASS gate.

```bash
source scripts/decomposition/environment.sh
python -m unittest discover -s tests -p test_coverage_decomposition.py
python scripts/decomposition/registration_atlas/next_steps/smoke_coverage.py
```

The smoke includes deliberately poisoned unobserved labels/SDFs, all supervised
heads/Eikonal, observed-only evaluation and exact checkpoint resume.

## Actual submission ledger for this session

| Order | Candidate | Phase | Slurm ID | State when recorded |
|---|---|---|---:|---|
| 1 | 0.8 mm, `_01` | CPU export | 1665634 | Completed |
| 2 | 0.8 mm, `_01` | CPU preflight/cache | 1666128 | Completed; scientific gate FAIL |
| 3 | 0.5 mm, `_02` | CPU export | 1666316 | Submitted |
| 4 | 0.5 mm, `_02` | CPU preflight/cache | 1666319 | Submitted after 1666316 |
| 5 | Passing candidate only | GPU existing control | — | NOT submitted |
| 6 | Passing candidate only | GPU prealigned fit | — | NOT submitted |
| 7 | Both fits complete | GPU paired evaluation | — | NOT submitted |

Refresh actual states with `sacct`/`squeue` and check each candidate's jobs.tsv.
The fine-grid gate must be inspected before GPU phases. Do not equate Slurm exit
0 with gate PASS: preflight reports a completed diagnostic even when it closes
the GPU gate. The gate JSON is authoritative for scientific eligibility.

Jean Zay allocates memory with CPU cores and rejects explicit --mem options. The
0.5-mm CPU export requests 16 cores and up to two hours; no GPU is used. A rejected
submission before a job ID can be retried with start.py --submit-prepared, but
only when no export job/output exists. Already submitted phases cannot duplicate.
