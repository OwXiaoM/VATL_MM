"""Create a pinned, reviewable small-cohort ablation study from an existing run."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import yaml
from data_paths import relocate_path
from train_decomposition import merge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=500)
    cli = p.parse_args()
    if cli.epochs < 1:
        p.error('--epochs must be positive')
    checkpoint = Path(cli.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') not in ('vatl_sdf_decomposition_v1', 'vatl_sdf_atlas_rigid_v2'):
        p.error('Expected decomposition checkpoint')
    ids = list(map(str, saved['subject_ids'][:4]))
    if len(ids) < 4:
        p.error('Study requires at least four training subjects in source checkpoint')
    frame = saved['dataset_df']
    for subject in ids:
        row = frame[frame.subject_id.astype(str)==subject].iloc[0]
        for key in [*saved['args']['dataset']['modalities'], saved['args']['dataset']['sdf_key']]:
            if not Path(relocate_path(row[key])).is_file():
                raise FileNotFoundError(f'{subject}: {key}: {row[key]}')
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'configs').mkdir()
    base = copy.deepcopy(saved['args'])
    base.update(batch_size=1, num_workers=0, logging=False, compute_metrics=False, save_every=100)
    base['epochs']['train'] = cli.epochs
    # A new atlas-rigid study, never an extension of a residual-era experiment.
    obsolete = {'residual_enabled','latent_enabled','alpha','latent_dim','hidden','depth','fourier_bands','lr_residual','lr_latent'}
    for key in obsolete:
        base['decomposition'].pop(key, None)
    for key in ('residual', 'smoothness', 'latent'):
        base['decomposition']['loss']['weights'].pop(key, None)
    base['decomposition']['stages'] = {'enabled': False}
    base['decomposition']['debug_evaluation'] = {'enabled': True, 'every': 50, 'subjects': 4, 'chunk': 8192}
    base['dataset']['subject_ids']['train'] = ids
    base['n_subjects']['train'] = 4
    # Small experiments deliberately share architecture, subject IDs, seed and budget.
    # Each branch starts from scratch, avoiding architecture-dependent transfer bias.
    variants = {
        'single_sdf': {'n_subjects': {'train': 1}, 'dataset': {'subject_ids': {'train': ids[:1]}},
            'decomposition': {'registration_enabled': False,
                'topology_enabled': False, 'image_weight': 0., 'seg_weight': 0.,
                'loss': {'weights': {'surface': 0., 'eikonal': 0., 'registration': 0.}}},
            'optimizer': {'moe_weight': 0.}},
        'single_aux': {'n_subjects': {'train': 1}, 'dataset': {'subject_ids': {'train': ids[:1]}},
            'decomposition': {'registration_enabled': False,
                'topology_enabled': False, 'image_weight': 1., 'seg_weight': 1.,
                'loss': {'weights': {'surface': 0., 'eikonal': 0., 'registration': 0.}}},
            'optimizer': {'moe_weight': 0.}},
        'small_reference': {},
        'small_uniform50': {'dataset': {'sdf_sampling': {'uniform_fraction': .5}}},
        'small_less_interior': {'dataset': {'sdf_sampling': {'fractions': {'interior': .05, 'surface': .35, 'tissue': .4, 'background': .2}}}},
        'small_rotvec': {'decomposition': {'rotation': 'rotvec'}},
        'small_no_eikonal': {'decomposition': {'loss': {'weights': {'eikonal': 0.}}}},
    }
    for name, changes in variants.items():
        args = merge(copy.deepcopy(base), changes)
        if name.startswith('single_'):
            args['decomposition']['debug_evaluation']['subjects'] = 1
        with (output/'configs'/f'{name}.yaml').open('w') as f:
            yaml.safe_dump(args, f)
    plan = {'source_checkpoint': str(checkpoint), 'seed': base['seed'], 'epochs': cli.epochs,
            'training_subject_ids': ids, 'experiments': list(variants),
            'initial_jobs': ['single_sdf'],
            'protocol': 'training reconstruction diagnostics; not population performance estimates',
            'controls': 'single_aux vs single_sdf; all small_* vs small_reference; full-grid evaluation every 50 epochs',
            'architecture': 'vatl_sdf_atlas_rigid_v2',
            'next_gate': 'Check one-subject convergence, then joint rigid population fitting and sampling'}
    (output/'study.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
