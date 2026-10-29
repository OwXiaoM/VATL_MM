Current launchers default to H100. See [H100 setup](../../docs/h100_setup.md). The A100 notes below describe the earlier migration.

Jean Zay setup uses the existing `vatl_torch` environment with `arch/a100` and
`pytorch-gpu/py3/2.8.0`. Training presets request four A100 GPUs under
`vnc@a100`, with the 20 hour QoS limit.

From VATL_MM:

```bash
python scripts/jean_zay/check_paths.py
SDF_DRY_RUN=1 bash training_script/train_ddp_n4k2_t1_sdf.sh
sbatch training_script/train_ddp_n4k2_t1_sdf.sh
```

Data defaults to `/lustre/fsn1/projects/rech/vnc/uil92qd/datasets/data/v1`.
Set `SDF_DATA_ROOT` to override it. Paths containing `/v1/` in metadata are
relocated in memory; shared TSVs are unchanged. Outputs default to
`/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/training_outputs`.
`SDF_OUTPUT_ROOT` overrides the SDF launcher output root.

The ablation launchers require `configs/ablations/*.yaml`, which were missing
from this checkout. The single GPU entry point also imports the missing
`subject_to_atlas.py`. Use the regular DDP presets for this checkout.
