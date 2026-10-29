"""Tiny synthetic CPU integration: loader, joint atlas/rigid training, checkpoint, exports, adaptation.

No patient volumes or GPUs are used. --ddp exercises two CPU gloo workers.
"""
import argparse
import json
import os
import pickle
import networkx as nx
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import nibabel as nib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--topology', action='store_true')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--resume-check', action='store_true')
    parser.add_argument('--diagnostics', action='store_true', help='Exercise opt-in evaluation, rotvec and uniform sampling')
    cli = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='vatl_decomposition_') as temporary:
        path = Path(temporary)
        shape = (12, 12, 12)
        affine = np.diag([.8, 1., 1.2, 1.])
        ijk = np.indices(shape).reshape(3, -1).T
        physical = nib.affines.apply_affine(affine, ijk - np.asarray(shape) / 2)
        sdf = (np.linalg.norm(physical[:, :2], axis=-1) - 1.2).reshape(shape).astype(np.float32)
        row = {'subject_id': 'synthetic', 'scan_age': 35., 'sex': 'F'}
        for key, array in [('MRA', np.exp(-sdf*sdf)), ('T1', np.ones(shape)),
                           ('Seg', np.where(sdf < 0, 1, 2)), ('SDF', sdf)]:
            dest = path / f'{key}.nii.gz'
            image = nib.Nifti1Image(array.astype(np.float32), affine)
            image.header.set_xyzt_units('mm')
            nib.save(image, dest)
            row[key] = str(dest)
        if cli.topology:
            graph = nx.path_graph(8)
            for i in graph.nodes:
                graph.nodes[i]['pos'] = nib.affines.apply_affine(affine, [6, 6, i+2])
                graph.nodes[i]['radius'] = 1.2
            graph_path = path / 'Graph.pkl'
            with graph_path.open('wb') as stream:
                pickle.dump(graph, stream)
            row['Graph'] = str(graph_path)
        metadata = path / 'metadata.tsv'
        pd.DataFrame([row]).to_csv(metadata, sep='\t', index=False)
        config = {'extends': str(ROOT / 'configs/decomposition/smoke.yaml'),
                  'dataset': {'tsv_file': str(metadata), 'subject_ids': {'train': ['synthetic'], 'val': [], 'test': []}},
                  'decomposition': {'topology_enabled': cli.topology}, 'num_workers': 0}
        config_path = path / 'config.yaml'
        if cli.diagnostics:
            if cli.ddp:
                parser.error('--diagnostics requires a single process')
            config['decomposition'].update(rotation='rotvec',
                debug_evaluation={'enabled': True, 'every': 1, 'subjects': 1, 'chunk': 256})
            config['dataset']['sdf_sampling'] = {'uniform_fraction': .5}
        config_path.write_text(yaml.safe_dump(config))
        command = [sys.executable]
        if cli.ddp:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
        command += [str(ROOT / 'train_decomposition.py'), '--config', str(config_path),
                    '--output', str(path / 'training'), '--device', 'cpu']
        subprocess.run(command, check=True, cwd=ROOT, env=env)
        checkpoint = path / 'training/checkpoint_epoch_2.pth'
        assert checkpoint.is_file()
        if cli.resume_check:
            resumed = command.copy()
            resumed[resumed.index('--output') + 1] = str(path / 'resumed')
            resumed += ['--resume', str(path / 'training/checkpoint_epoch_0.pth')]
            subprocess.run(resumed, check=True, cwd=ROOT, env=env)
            assert (path / 'resumed/checkpoint_epoch_2.pth').is_file()
            full = [json.loads(line) for line in (path / 'training/loss_components.jsonl').read_text().splitlines()]
            continued = [json.loads(line) for line in (path / 'resumed/loss_components.jsonl').read_text().splitlines()]
            assert continued == [record for record in full if record['epoch'] > 0], 'Resumed losses differ'
        if cli.training_only:
            print('Synthetic training passed (topology=%s, resume=%s)' % (cli.topology, cli.resume_check), flush=True)
            return
        infer = [sys.executable, str(ROOT / 'infer_decomposition.py'), '--checkpoint', str(checkpoint), '--device', 'cpu', '--chunk', '256']
        subprocess.run(infer + ['--mode', 'atlas_and_components', '--subject-id', 'synthetic',
                               '--output', str(path / 'components'), '--metrics'], check=True, cwd=ROOT, env=env)
        atlas = nib.load(path / 'components/warped_atlas_mm.nii.gz').get_fdata()
        final = nib.load(path / 'components/sdf_mm.nii.gz').get_fdata()
        np.testing.assert_allclose(atlas, final, equal_nan=True)
        subprocess.run(infer + ['--mode', 'atlas_only', '--subject-id', 'synthetic',
                               '--output', str(path / 'atlas')], check=True, cwd=ROOT, env=env)
        row['subject_id'] = 'unseen'
        pd.DataFrame([row]).to_csv(metadata, sep='\t', index=False)
        subprocess.run(infer + ['--mode', 'subject_reconstruction', '--subject-id', 'unseen',
                               '--metadata', str(metadata), '--adapt-steps', '2', '--adapt-points', '32',
                               '--output', str(path / 'unseen')], check=True, cwd=ROOT, env=env)
        assert (path / 'unseen/adaptation.pth').is_file()
        print('Synthetic pipeline passed: joint training, checkpoint, component and atlas exports, unseen adaptation', flush=True)


if __name__ == '__main__':
    main()
