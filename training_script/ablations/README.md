# Fragmentation ablations on the Eurecom cluster

These launchers have been adapted from Leonardo to `l40s`, four GPUs, 16 CPUs,
128 GB RAM, and `source ~/.bashrc1; conda activate torch` through `../common.sh`.
They have passed effective-configuration tests and SLURM `--test-only` validation.
No ablation training jobs have been submitted by this audit.

| Experiment | SDF head | SDF-aware sampling | Voxel SDF weight | Graph loss |
| --- | --- | --- | --- | --- |
| A | 1 channel | off | 0.0 | CE, weight 0.1 |
| B | 1 channel | on | 0.0 | CE, weight 0.1 |
| C | 1 channel | off | 0.1 | CE, weight 0.1 |
| D | 1 channel | on | 0.1 | CE, weight 0.1 |

All use seed 42, four experts/top-2, 500,000 coordinates per training subject,
identical subject splits, model, batch size, epochs, and optimizer defaults.
Graph SDF and radius losses are off. The SDF target and output remain present in
A/B; their voxel SDF gradients are zero. The original Graph-only checkpoint is
an external reference, not interchangeable with an A/B checkpoint.

The flat YAML overrides live in `configs/ablations/{A,B,C,D}.yaml`.
`run_ddp.py --experiment_config FILE` expands them at that position in the
command, so the YAML overrides earlier common defaults and later CLI flags can
still override the experiment. `config_final.yaml` is saved for every run.
Each run gets a unique child directory under
`/medias/medical/caravel/zhangx/training_outputs/ablations/<letter>`.

From `/users/zhangx/VATL_MM`, launch only when ready:

```bash
sbatch training_script/ablations/A.sh
sbatch training_script/ablations/B.sh
sbatch training_script/ablations/C.sh
sbatch training_script/ablations/D.sh
```

The inherited time limit is six hours. Set the same larger limit for all four
with `sbatch --time=24:00:00 ...` if necessary; compare equal completed epochs /
optimizer steps rather than equal wall time. Monitor memory for full subject
batches. The small CPU checks do not establish full-scale GPU memory/runtime.

Inspect commands without launching:

```bash
SDF_DRY_RUN=1 bash training_script/ablations/A.sh
sbatch --test-only training_script/ablations/A.sh
```

Printouts include defaults followed by `--experiment_config`; use the saved
`config_final.yaml` or the effective-configuration unit test to confirm the final
values, rather than reading only the first occurrence of a command-line flag.
Do not override architecture/optimizer/step-count settings differently among A–D.

Experiment E adds independent voxel SDF coordinates while retaining C's original
image/tissue sampler. See [EXPERIMENT_E.md](EXPERIMENT_E.md) for the exact routing,
disjoint pool definitions, budgets, validation, launcher variants and evaluation.
