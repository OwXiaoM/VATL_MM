Current Jean Zay presets request four H100 GPUs. See [H100 environment details](../docs/h100_setup.md).

Jean Zay setup and validation: [scripts/jean_zay/README.md](../scripts/jean_zay/README.md).

The notes below describe the original machine and include historical paths.

# SDF training on this cluster

Run from `/users/zhangx/VATL_MM`. All ten training presets request one node,
four GPUs, 16 CPUs and 128 GB RAM on the `l40s` partition for 24 hours.
`n4k2` and `n8k2` mean four/eight MoE experts with top-2 routing.

The launcher sources `~/.bashrc1` and activates `conda activate torch`.
Data paths in `configs/config_data.yaml` and the dataset TSVs point to
`/medias/medical/data/Atlas_training/v1`. Original TSVs are preserved alongside
as `*.tsv.leonardo.bak`.

Submit the smoke test (four GPUs, four subjects, two epochs, original model size):

```bash
sbatch training_script/smoke_test.sh
```

This exercises distributed training, image/tissue/SDF losses, graph SDF and
radius supervision, validation control flow, and checkpoint saving. Full-volume
reconstruction, atlas generation and held-out adaptation are disabled for this
short check. Logs are `training_script/smoke_<jobid>.log` and `.err`.

Submit full training with one of:

```bash
sbatch training_script/train_ddp_n4k2_t1_sdf.sh
sbatch training_script/train_ddp_n4k2_t1_sdf_graph_sdf.sh
sbatch training_script/train_ddp_n4k2_t1_sdf_graph_sdf_radius.sh
```

Other endings are `graph_ce` and `graph_both`; each is also available for n8k2.
The full presets keep the YAML training, reconstruction and adaptation settings.
Reverse subject-to-atlas export is disabled because this checkout does not
contain the optional `subject_to_atlas.py` module.

Outputs default to `/medias/medical/caravel/zhangx/training_outputs/sdf` (or
`outputs/smoke` for smoke runs), grouped by experts, graph mode and unique run.
Each run saves `config_final.yaml` and `checkpoint_epoch_<epoch>.pth`.

Print a command without loading the environment or launching training:

```bash
SDF_DRY_RUN=1 bash training_script/smoke_test.sh
```

Extra command-line options override defaults. Environment overrides include
`SDF_OUTPUT_ROOT`, `SDF_REPO`, `SDF_CONDA_ENV`, `SDF_NPROC`, `SDF_NUM_EXPERTS`,
`SDF_GRAPH_MODE=off|ce|sdf|both`, and `SDF_RADIUS=True|False`. Match `SDF_NPROC`
to the GPU allocation when overriding `sbatch --gres`.

To smoke-test all ten presets (at most two concurrent four-GPU jobs):

```bash
sbatch training_script/smoke_matrix.sh
```

The matrix uses the same reduced settings and original model architecture.
Its array index follows the alphabetically sorted `train_ddp_*.sh` filenames.
`requirements-cluster.txt` records the installed conda environment's pip packages.
