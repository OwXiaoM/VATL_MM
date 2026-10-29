# Jean Zay H100 launchers

The default decomposition launcher and original `training_script/train_ddp_*.sh`
presets now request `vnc@h100`, `qos_gpu_h100-t3`, partition `gpu_p6`, constraint
`h100`, and four typed `gpu:h100` resources on one node for 20 hours. They request
48 CPUs per task. Explicit memory directives are prohibited by Jean Zay and were
removed; host memory follows the partition's CPU allocation policy. Slurm dry
validation accepted the decomposition and representative original launcher.
A dry validation does not submit a job or guarantee its predicted start time.

## Important interpreter correction

The existing `vatl_torch` virtual environment is pinned to the **A100** base Python
under `/lustre/fshomisc/sup/hpe/`. Loading `arch/h100` and then running that venv
interpreter failed with `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`
because the H100 module uses MKL 2024 while that torch build expects another setup.

The shared `scripts/decomposition/environment.sh` now loads `arch/h100` and
`pytorch-gpu/py3/2.8.0`, uses the **H100 module interpreter** under
`/lustre/fshomisc/sup/pub/`, and adds the existing dependency-only venv site-packages
through PYTHONPATH. This supplies ANTs and compatible SciPy without replacing
H100 torch. No global environment, package installation, or venv files were changed.
It rejects an overlay containing its own torch to avoid silently selecting the
wrong torch binary.

Verified on the login node without GPU execution:

- H100 module PyTorch 2.8.0, CUDA 12.8, compiled CUDA architecture `sm_90`.
- Imports of torch, ANTs, pandas, YAML, nibabel, TorchIO and wandb.
- Shell syntax and Slurm `--test-only` resource/account validation.
- CPU model/evaluation regression tests under the H100 Python environment.

The model uses standard torch linear/sine/SiLU, MoE, grid sampling, linear algebra,
autograd and NCCL DDP, with no custom architecture-specific CUDA extensions.
The existing FP32 setting (`amp: false`) and scientific/model hyperparameters
were preserved. GPU forward/backward, multi-GPU NCCL and memory/runtime remain
to be tested on an allocated H100 node. The default decomposition launcher runs
`scripts/jean_zay/check_gpu.py` before training to check four visible H100s and a
tiny CUDA forward/backward on each device. Do not run GPU tests on a login node.

## Run

From VATL_SDF:

```bash
sbatch VATL_MM/scripts/decomposition/train.slurm
```

The same ablation configuration switch works:

```bash
SDF_CONFIG=configs/decomposition/registration_only.yaml \
  sbatch VATL_MM/scripts/decomposition/train.slurm
```

To deliberately retain A100:

```bash
sbatch VATL_MM/scripts/decomposition/train_a100.slurm
```

An already queued A100 job retains the batch script/resources saved by Slurm.
The environment helper detects its `SLURM_JOB_ACCOUNT` and selects A100 when
`SDF_GPU_ARCH` is unset. Job 1589281 was left unchanged; no jobs were cancelled,
resubmitted or newly submitted during this migration. If you submit H100 as well,
the two jobs are independent and may both run.

For interactive CPU tests, H100 is now the default:

```bash
cd VATL_MM
source scripts/decomposition/environment.sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

Use `SDF_GPU_ARCH=a100` before sourcing to explicitly select the A100 setup.
The Helix launcher remains unchanged.
