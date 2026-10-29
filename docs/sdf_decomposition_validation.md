# Atlas-rigid validation checks

The current regression suite checks physical rigid transforms and inverses,
finite gradients, SDF signs, physical Eikonal scaling, baseline initialization,
checkpoint/optimizer round trips, absence of residual/code parameters, rotation
Jacobians, crop-aware full-reference metrics and sampler behavior.

The synthetic CPU integration checks training, exact resume, component and mask
exports, and unseen-subject SDF-supervised rigid fitting. It is a plumbing check,
not evidence of real-data accuracy.

```bash
source scripts/decomposition/environment.sh
python -m unittest discover -s tests -p 'test*decomposition*.py'
python -m unittest discover -s tests -p 'test_validation_pipeline.py'
python scripts/decomposition/smoke_pipeline.py --diagnostics --resume-check
```

Real-data results and protocol limits are recorded in CONTEXT.md. Next experiments
are in scripts/decomposition/NEXT_STEPS.md. No image-only inference accuracy is
established by the SDF-supervised adapter.
