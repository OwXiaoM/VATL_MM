Validation on 2026-10-26, Jean Zay, existing vatl_torch environment.

- All 26 discovered CPU tests passed (including the previous 11 decomposition tests).
- New tests cover first-N ordering independent of RNG, file eligibility, manifests,
  unmodified training sampler dispatch, direct identical-grid comparison,
  affine-only interpolation, vessel Dice/physical HD95/ASD, empty/invalid/SDF-
  absent cases, full-grid false positives, per-modality summaries, rank-zero
  writes, both builders' save/metrics switches, unchanged learned transformations,
  and evaluation occurring after adaptation. ANTs registration is patched to fail
  if called during evaluation/export/legacy resampling.
- Real metadata preview found 55 eligible validation subjects and selected:
  IXI403-Guys-0965, IXI037-Guys-0704, IXI050-Guys-0711.
  Only metadata and file existence were inspected; no patient volumes were read.
- Python syntax and git diff whitespace checks passed.

No GPU checkpoint evaluation, actual distributed GPU evaluation, or real-data
quantitative experiment was run. Both builder paths were exercised with tiny CPU
synthetic inputs and a stub decoder; rank writing was checked with rank-one
simulation. No training jobs were submitted and no changes were committed/pushed.
