# Dependencies and attribution

Python3.11. Tested project environment: PyTorch2.2.0a0+git6c8c5ad (CINECA
cineca-ai/4.3.0),NumPy1.26.4,SciPy1.12.0,nibabel5.3.2,NetworkX3.2.1,
pandas2.2.1,PyYAML6.0.1,scikit-image0.22.0,matplotlib3.8.3,Plotly5.20.0.
requirements-vesseltok.txt pins the scientific Python versions and allows the
corresponding public PyTorch2.2 series; the cluster prerelease is not a public
pip requirement. A different PyTorch build requires rerunning deterministic
regression checks; exact bitwise CUDA replay across builds is not promised.

Use an existing suitable CUDA-enabled environment,or install the requirements
in a virtual environment. Select the PyTorch CUDA build compatible with the
cluster drivers. The CPU tests require no GPU or subject files. There are no
compiled FPS/PyTorch3D/torch-cluster,ANTs or SimpleITK dependencies for this module.

On Leonardo,the existing environment can be loaded externally before scripts:
`module load profile/deeplrn cineca-ai/4.3.0`,then activate your virtualenv.
Scripts honor optional VATL_ENV_ACTIVATE (path to an activation script) and do
not hardcode another worktree or personal environment. GPU jobs must be allocated
through Slurm; pass your allocation account at submission.

Method reference: VesselTok https://arxiv.org/pdf/2603.18797 and official source
https://github.com/chinmay5/vessel_tok at
`d71397b378ed28db7941c1e126e564c4a9d37ab5`.
This is an independent Stage1 implementation inspired by that encoder/decoder,
not a full upstream reproduction. Upstream attribution and its supplied
CC BY-NC4.0/third-party notices are retained as UPSTREAM_LICENSE and
UPSTREAM_THIRD_PARTY_NOTICES.md. Those notices describe upstream components;
this checkout does not copy the upstream diffusion/flow stack,Hunyuan3D or
compiled FPS code. Do not interpret these notices as a new license grant for
the entire pre-existing VATL_MM repository.
