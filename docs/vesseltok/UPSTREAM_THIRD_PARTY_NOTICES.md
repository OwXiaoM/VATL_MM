# Third-Party Notices

This repository adapts or embeds code from the following third-party projects.
Their original license terms apply to the listed files in addition to this
repository's own [LICENSE](LICENSE).

## Tencent Hunyuan3D (Non-Commercial)

- **Files:** `diffusion_on_implicit_fn/flow_matching_model/hunyuan_dit.py`
- **Source:** Tencent Hunyuan3D
- **License:** Tencent Hunyuan Non-Commercial License Agreement (see the
  header comment retained at the top of the file). This is the component
  that requires the whole repository to be released under a noncommercial
  license — see [LICENSE](LICENSE).
- Usage of `sageattention` for accelerated attention in this file is
  opt-in via the `USE_SAGEATTN=1` environment variable and is not enabled
  by default.

## SiT (Scalable Interpolant Transformer)

- **Files:** `diffusion_on_implicit_fn/flow_matching_model/transport/*`,
  `diffusion_on_implicit_fn/gnn_baselines/sit_model.py`
- **Source:** https://github.com/willisma/SiT
- **License:** MIT License, Copyright (c) Meta Platforms, Inc. and affiliates.
- `sit_model.py` additionally references implementation ideas from
  [GLIDE](https://github.com/openai/glide-text2im) and
  [MAE](https://github.com/facebookresearch/mae) (see file header).
