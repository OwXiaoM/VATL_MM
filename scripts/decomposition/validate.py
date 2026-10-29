"""Validate decomposition checkpoints on the predefined held-out subject split."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--subjects', type=int, default=3)
    p.add_argument('--adapt-steps', type=int, default=100)
    p.add_argument('--adapt-points', type=int, default=8192)
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--subject-manifest')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    cli = p.parse_args()
    if min(cli.subjects, cli.adapt_steps, cli.adapt_points, cli.chunk) <= 0:
        p.error('Subjects, adaptation steps, adaptation points and chunk must be positive')

    import csv
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch
    from data_loading.dataset import Data
    from data_paths import relocate_path
    from native_evaluation import binary_vessel_metrics, geometry_resample, mm_image, clean_json
    from models.vascular_metrics import vascular_metrics

    checkpoint = Path(cli.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'vatl_sdf_atlas_rigid_v2':
        p.error('Expected a vatl_sdf_atlas_rigid_v2 checkpoint')
    args = copy.deepcopy(saved['args'])
    output = Path(cli.output).resolve()
    if output.exists():
        p.error('Output directory must be new')
    args.update(output_dir=str(output), rank=0)
    args['n_subjects']['val'] = cli.subjects
    args.setdefault('evaluation', {})['subject_manifest'] = cli.subject_manifest
    frame = pd.read_csv(relocate_path(args['dataset']['tsv_file']), sep='\t')
    # Data selects the ordered predefined validation split and verifies target files.
    dataset = Data(args, frame, 'val')
    ids = dataset.df.subject_id.astype(str).tolist()
    overlap = set(ids) & set(map(str, saved['subject_ids']))
    if overlap:
        raise ValueError(f'Validation IDs overlap training checkpoint: {sorted(overlap)}')
    metadata = output / 'validation_metadata.tsv'
    dataset.df.to_csv(metadata, sep='\t', index=False)
    protocol = {**vars(cli), 'checkpoint': str(checkpoint), 'checkpoint_epoch': saved['epoch'],
                'subject_ids': ids, 'evaluation_region': 'full_segmentation_reference_grid',
                'adaptation': 'Fit rigid transform to subject SDF samples; population weights frozen',
                'outside_model_crop': 'background', 'surface_tolerance_mm': 1.0}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    records = []
    for subject in ids:
        subject_output = output / ('subject_' + str(len(records)).zfill(3))
        command = [sys.executable, str(ROOT / 'infer_decomposition.py'),
                   '--checkpoint', str(checkpoint), '--mode', 'atlas_and_components',
                   '--metadata', str(metadata), '--subject-id', subject,
                   '--output', str(subject_output), '--device', cli.device,
                   '--adapt-steps', str(cli.adapt_steps), '--adapt-points', str(cli.adapt_points),
                   '--chunk', str(cli.chunk), '--seed', str(cli.seed)]
        subprocess.run(command, cwd=ROOT, check=True)
        row = dataset.df[dataset.df.subject_id.astype(str) == subject].iloc[0]
        reference = mm_image(nib.load(relocate_path(row[args['dataset']['modalities'][-1]])))
        truth = reference.get_fdata() == args['dataset']['label_names'].index('Vessel')
        mask_image = mm_image(nib.load(subject_output / 'vessel_mask.nii.gz'))
        aligned, _ = geometry_resample(mask_image, reference, labels=True)
        pred = aligned > 0
        record = {'subject_id': subject, **binary_vessel_metrics(pred, truth, reference.affine)}
        linear = reference.affine[:3, :3]
        spacing = np.linalg.norm(linear, axis=0)
        if np.allclose((linear / spacing).T @ (linear / spacing), np.eye(3), atol=1e-4):
            record.update(vascular_metrics(pred, truth, spacing))
        record['valid_crop_fraction'] = json.loads((subject_output / 'inference.json').read_text())['valid_crop_fraction']
        records.append(record)
        with (output / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
            writer.writeheader()
            writer.writerows(records)
        summary = {}
        for key in dict.fromkeys(k for r in records for k in r):
            values = [r[key] for r in records if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
            if values:
                summary[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'count': len(values)}
        (output / 'metrics_summary.json').write_text(json.dumps(clean_json(summary), indent=2) + '\n')
        print(json.dumps(clean_json(record)), flush=True)
    print(f'Validation complete: {output}', flush=True)


if __name__ == '__main__':
    main()
