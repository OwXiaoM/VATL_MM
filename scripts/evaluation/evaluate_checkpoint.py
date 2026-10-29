"""Evaluate an original VATL checkpoint using its saved architecture and native grid.

Launch with torchrun on allocated GPUs. Does not train population model weights.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
import yaml
from build_atlas_ddp import AtlasBuilderDDP
from native_evaluation import clean_json
from run_ddp import set_nested


class CheckpointEvaluation(AtlasBuilderDDP):
    def train_on_data(self):
        self.broadcast_latents_from_rank0()
        if self.rank == 0:
            self.validate(int(self.args['load_model']['epoch']))
        dist.barrier()

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split='train'):
        if split == 'train':
            return [] # This command evaluates held-out subjects only.
        return super().generate_subjects_from_df(idcs_df, epoch, split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, help='New output directory')
    p.add_argument('--evaluation-config', default=str(ROOT/'configs/evaluation.yaml'))
    p.add_argument('--subject-manifest')
    p.add_argument('--subjects', type=int)
    p.add_argument('--adaptation-epochs', type=int)
    p.add_argument('--save-images', action='store_true')
    cli = p.parse_args()
    if 'RANK' not in os.environ or not torch.cuda.is_available():
        p.error('Launch with torchrun on an allocated GPU (one GPU/rank is sufficient)')
    checkpoint = Path(cli.checkpoint).resolve()
    match = re.fullmatch(r'checkpoint_epoch_(\d+)\.pth', checkpoint.name)
    if not match:
        p.error('Use the original checkpoint_epoch_N.pth filename')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'inr_decoder' not in saved or 'args' not in saved:
        p.error('Expected an original single-GPU/DDP VATL checkpoint with saved args')
    args = copy.deepcopy(saved['args'])
    args['_evaluation_source_config'] = copy.deepcopy(saved['args'])
    del saved
    with open(cli.evaluation_config) as stream:
        overrides = yaml.safe_load(stream)
    for key, value in overrides.items():
        if isinstance(value, (dict, list)) or value is None:
            p.error('Evaluation config must contain scalar CLI-style overrides')
        set_nested(args, key, value)
    args.setdefault('evaluation', {})
    if cli.subject_manifest:
        args['evaluation']['subject_manifest'] = str(Path(cli.subject_manifest).resolve())
    if cli.subjects is not None:
        args['n_subjects']['val'] = cli.subjects
    if cli.adaptation_epochs is not None:
        args['epochs']['val'] = cli.adaptation_epochs
    if args['epochs']['val'] < 0:
        p.error('Adaptation epochs must be nonnegative')
    args['save_imgs']['val'] = cli.save_images or args['save_imgs']['val']
    args['compute_metrics'] = True
    args.setdefault('test_time_adaptation', {})['enabled'] = True
    args['save_model'] = False
    args['load_model'] = {'path': str(checkpoint.parent), 'epoch': int(match[1])}
    args['evaluation']['evaluated_checkpoint'] = str(checkpoint)
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    args.update(output_dir=str(Path(cli.output).resolve()), device=f'cuda:{local_rank}',
                rank=rank, local_rank=local_rank, world_size=dist.get_world_size())
    try:
        if rank == 0:
            Path(args['output_dir']).mkdir(parents=True, exist_ok=False)
            (Path(args['output_dir'])/'evaluation_config.json').write_text(json.dumps(clean_json(args), indent=2) + '\n')
        dist.barrier()
        # Keep the existing initialization/adaptation algorithm and its seed.
        torch.manual_seed(args['seed'])
        import numpy as np
        np.random.seed(args['seed'])
        CheckpointEvaluation(args)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
