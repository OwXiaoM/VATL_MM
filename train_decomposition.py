"""Opt-in SDF decomposition training. Original run_ddp.py remains unchanged."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from data_paths import relocate_path
from data_loading.dataset import Data
from data_loading.graph_store import GraphStore
from models.sdf_decomposition import AtlasSubjectSDF
from models.decomposition_losses import decomposition_losses, difference_samples
from utils import Criterion, graph_supervision_loss

ROOT = Path(__file__).resolve().parent


def merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path, resolve_subject_ids=True):
    with (ROOT / 'configs/config_atlas.yaml').open() as f:
        args = yaml.safe_load(f)
    with open(path) as f:
        overlay = yaml.safe_load(f) or {}
    parent = overlay.pop('extends', None)
    if parent:
        args = load_config(Path(path).parent / parent, resolve_subject_ids=False)
    merge(args, overlay)
    name = args.get('config_data')
    with (ROOT / 'configs/config_data.yaml').open() as f:
        data = yaml.safe_load(f)[name]
    # Dataset options in overlays override defaults, including sampling budget.
    args['dataset'] = merge(data, args.get('dataset', {}))
    if resolve_subject_ids and isinstance(args['dataset']['subject_ids'], str):
        with open(relocate_path(args['dataset']['subject_ids'])) as f:
            args['dataset']['subject_ids'] = yaml.safe_load(f)[args['dataset']['dataset_name']]['subject_ids']
    args['inr_decoder']['cond_dims'] = sum(args['dataset']['conditions'].values())
    out = args['inr_decoder']['out_dim']
    args['inr_decoder']['out_dim'] = [*out[:-1], len(args['dataset']['class_weights'])]
    # config_atlas stores [2] until the baseline entry point resolves class count.
    if len(out) == 1:
        args['inr_decoder']['out_dim'] = [len(args['dataset']['modalities']) - 1, len(args['dataset']['class_weights'])]
    return args


def checkpoint_payload(model, optimizer, args, epoch, dataframe):
    return {'format': 'vatl_sdf_atlas_rigid_v2', 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'args': args, 'epoch': epoch,
            'subject_ids': model.subject_ids, 'dataset_df': dataframe,
            'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def optimizer_for(model, args):
    cfg = args['decomposition']
    base = float(args['optimizer']['lr_inr'])
    return torch.optim.Adam([
        {'params': model.shared.parameters(), 'lr': float(cfg.get('lr_atlas', base)), 'name': 'atlas'},
        {'params': model.transforms.parameters(), 'lr': float(cfg.get('lr_registration', 1e-4)), 'name': 'registration'}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True, help='New experiment directory; never overwrite')
    parser.add_argument('--initialize-baseline', help='Existing SDF checkpoint for partial initialization')
    parser.add_argument('--resume', help='Decomposition checkpoint; output must be a new directory')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    cli = parser.parse_args()
    if cli.resume and cli.initialize_baseline:
        parser.error('Choose resume or baseline initialization')
    device = torch.device(cli.device, int(os.environ.get('LOCAL_RANK', 0))) if cli.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    distributed = 'RANK' in os.environ
    if distributed:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    rank = dist.get_rank() if distributed else 0
    args = load_config(cli.config)
    resume = torch.load(cli.resume, map_location='cpu', weights_only=False) if cli.resume else None
    if resume and resume.get('format') != 'vatl_sdf_atlas_rigid_v2':
        raise ValueError('Residual-era checkpoints cannot resume the atlas-rigid v2 architecture; start a fresh run')
    if resume:
        # Resume the saved experiment, not a subtly different CLI architecture.
        args = resume['args']
    seed = int(args['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args['device'] = str(device)
    args['output_dir'] = cli.output
    args['rank'], args['local_rank'], args['world_size'] = rank, int(os.environ.get('LOCAL_RANK', 0)), dist.get_world_size() if distributed else 1
    output = Path(cli.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        with (output / 'config_final.yaml').open('w') as f:
            yaml.safe_dump(args, f)
    if distributed:
        dist.barrier()
    diagnostic_eval = args.get('decomposition', {}).get('debug_evaluation', {})
    if diagnostic_eval.get('enabled', False) and distributed:
        raise ValueError('Diagnostic evaluation currently requires a single-process run')
    if not args.get('decomposition', {}).get('enabled', False):
        if not distributed or device.type != 'cuda':
            raise ValueError('Original baseline requires torchrun on an allocated GPU')
        from build_atlas_ddp import AtlasBuilderDDP
        AtlasBuilderDDP(args)
        dist.destroy_process_group()
        return
    if args.get('amp', False):
        raise ValueError('Decomposition initially uses FP32; set amp: false')
    if args['data_augmentation']['activate']:
        raise ValueError('Disable spatial augmentation until SDF-distance consistency is verified')
    frame = pd.read_csv(relocate_path(args['dataset']['tsv_file']), sep='\t')
    dataset_type = Data
    if args['dataset'].get('coverage_key'):
        from data_loading.coverage_dataset import CoverageData
        dataset_type = CoverageData
    dataset = dataset_type(args, frame, 'train', df_loaded=resume['dataset_df'] if resume else None)
    model = AtlasSubjectSDF(args, dataset.df['subject_id'].astype(str).tolist(), device).to(device)
    if resume:
        if resume['subject_ids'] != model.subject_ids:
            raise ValueError('Checkpoint subject ordering mismatch')
        model.load_state_dict(resume['model'])
    if cli.initialize_baseline:
        model.initialize_baseline(torch.load(cli.initialize_baseline, map_location=device, weights_only=False))
    optimizer = optimizer_for(model, args)
    start = 0
    if resume:
        optimizer.load_state_dict(resume['optimizer'])
        start = resume['epoch'] + 1
        torch.set_rng_state(resume['torch_rng'])
        np.random.set_state(resume['numpy_rng'])
        random.setstate(resume['python_rng'])
        if resume.get('cuda_rng') is not None and device.type == 'cuda':
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
    criterion = Criterion(args).to(device)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args['batch_size'], sampler=sampler,
                        shuffle=sampler is None, num_workers=args['num_workers'], collate_fn=dataset.collate_fn)
    topology = bool(args['decomposition'].get('topology_enabled', False))
    graph = GraphStore(args, dataset.df) if topology else None
    wrapper, previous_stage = model, None
    best_reconstruction_dice = -1.
    for epoch in range(start, args['epochs']['train']):
        stage = model.set_stage(epoch)
        # Rebuild reducer when the set of trainable parameters changes between stages.
        if distributed and stage != previous_stage:
            wrapper = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, find_unused_parameters=True)
        previous_stage = stage
        if sampler:
            sampler.set_epoch(epoch)
        weights = copy.deepcopy(args['decomposition']['loss'])
        merge(weights, args['decomposition'].get('stages', {}).get('loss_overrides', {}).get(stage, {}))
        for group in optimizer.param_groups:
            base_lr = args['decomposition'].get('lr_' + group['name'], args['optimizer']['lr_inr'] if group['name'] == 'atlas' else 1e-4)
            group['lr'] = float(base_lr) * float(args['decomposition'].get('stages', {}).get('lr_scales', {}).get(stage, {}).get(group['name'], 1.))
        for batch_index, batch in enumerate(loader):
            if isinstance(batch, dict):
                raise ValueError('Use existing vessel-aware sdf_sampling, not task_specific_sampling, for this entry point')
            for start_point in range(0, len(batch[0]), args['n_samples']):
                coords, values, conditions, ids = [x[start_point:start_point + args['n_samples']].to(device) for x in batch]
                ids = ids.reshape(-1).long()
                target = values[:, len(args['dataset']['modalities'])]
                n = len(coords)
                clouds, conds, indices = [coords], [conditions], [ids]
                diff = None
                if weights['weights'].get('eikonal', 0):
                    diff = difference_samples(coords, conditions, ids, target, model.registration.half_bbox, weights)
                    if diff:
                        clouds.append(diff[0]); conds.append(diff[1]); indices.append(diff[2])
                n_diff = len(diff[0]) if diff else 0
                gtargets = []
                if topology:
                    for idx in torch.unique(ids):
                        gc, gt = graph.sample_nodes(int(idx), args['graph_supervision']['points_per_subject'], device=device, return_sdf=True)
                        clouds.append(gc)
                        conds.append(conditions[ids == idx][0].expand(len(gc), -1))
                        indices.append(idx.expand(len(gc)))
                        gtargets.append(gt)
                optimizer.zero_grad(set_to_none=True)
                all_components = wrapper(torch.cat(clouds), torch.cat(conds), torch.cat(indices))
                components = {k: v[:n] for k, v in all_components.items() if k != 'moe'}
                differences = {k: all_components[k][n:n+n_diff] for k in ('sdf_mm',)} if n_diff else None
                loss = decomposition_losses(model, components, target, ids, weights, differences)
                # Reuse established image/segmentation/MoE terms; SDF is replaced above.
                auxiliary = criterion(components['output'], values, None, moe_loss=all_components['moe'],
                                      sdf_weight=0., apply_sdf=False,
                                      sr_weight=float(args['decomposition'].get('image_weight', 1.)),
                                      seg_weight=float(args['decomposition'].get('seg_weight', 1.)))
                loss['image'] = auxiliary['sr']
                loss['segmentation'] = auxiliary['seg_weighted']
                loss['moe'] = auxiliary['moe']
                loss['total'] = loss['total'] + auxiliary['total']
                if topology:
                    gl = graph_supervision_loss(args, all_components['output'][n+n_diff:], torch.cat(gtargets), None)
                    loss.update({k: v for k, v in gl.items() if k != 'graph'})
                    loss['topology'] = gl['graph']
                    loss['total'] = loss['total'] + gl['graph']
                if not torch.isfinite(loss['total']):
                    raise FloatingPointError('Non-finite decomposition loss')
                loss['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args['decomposition'].get('gradient_clip', 10.)))
                optimizer.step()
                if rank == 0:
                    record = {'epoch': epoch, 'batch': batch_index, 'stage': stage,
                              **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in loss.items()},
                              'atlas_abs_mm': float(components['warped_atlas_mm'].detach().abs().mean()),
                              'atlas_sdf_mae_mm': float((components['warped_atlas_mm'].detach() - target.clamp(-model.shared.tau, model.shared.tau)).abs().mean()),
                              'final_sdf_mae_mm': float((components['sdf_mm'].detach() - target.clamp(-model.shared.tau, model.shared.tau)).abs().mean())}
                    with (output / 'loss_components.jsonl').open('a') as f:
                        f.write(json.dumps(record) + '\n')
            if rank == 0:
                print(f"epoch={epoch} stage={stage} batch={batch_index} total={float(loss['total'].detach()):.5f}", flush=True)
        if rank == 0 and ((epoch + 1) % args['save_every'] == 0 or epoch + 1 == args['epochs']['train']):
            torch.save(checkpoint_payload(model, optimizer, args, epoch, dataset.df), output / f'checkpoint_epoch_{epoch}.pth')
        if rank == 0 and diagnostic_eval.get('enabled', False) and (
                epoch == start or (epoch + 1) % int(diagnostic_eval.get('every', 50)) == 0
                or epoch + 1 == args['epochs']['train']):
            from models.decomposition_evaluation import evaluate_training
            # Full-grid metrics must not perturb subsequent training random samples.
            rng_state = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
                         torch.cuda.get_rng_state_all() if device.type == 'cuda' else None)
            score = evaluate_training(model, dataset.df, output, epoch,
                                      int(diagnostic_eval.get('subjects', 1)), int(diagnostic_eval.get('chunk', 8192)))
            torch.set_rng_state(rng_state[0]); np.random.set_state(rng_state[1]); random.setstate(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])
            print(f'epoch={epoch} training_reconstruction_Dice={score}', flush=True)
            if score is not None and score > best_reconstruction_dice:
                best_reconstruction_dice = score
                torch.save(checkpoint_payload(model, optimizer, args, epoch, dataset.df), output / 'best_training_reconstruction.pth')
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
