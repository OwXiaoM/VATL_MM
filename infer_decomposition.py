"""Export atlas, registered atlas and zero-level masks as NIfTI."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from data_paths import relocate_path
from data_loading.dataset import Data
from models.sdf_decomposition import AtlasSubjectSDF, sdf_to_mask
from models.decomposition_inference import adapt_subject
from models.vascular_metrics import vascular_metrics
from utils import save_img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mode', choices=['atlas_only', 'subject_reconstruction', 'atlas_and_components'], required=True)
    p.add_argument('--subject-id', help='Known training ID or an ID in --metadata')
    p.add_argument('--metadata', help='TSV for unseen subjects, same dataset schema')
    p.add_argument('--reference', help='Output NIfTI geometry; default subject SDF')
    p.add_argument('--age', type=float, help='Condition override in dataset age units (years for IXI)')
    p.add_argument('--sex', choices=['M', 'F'])
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=42, help='Sampling and adaptation seed')
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--adapt-steps', type=int, default=0)
    p.add_argument('--adapt-points', type=int, default=2048)
    p.add_argument('--adaptation-state', help='Previously fitted transform')
    p.add_argument('--metrics', action='store_true', help='Compare output mask to subject SDF on matching grid')
    cli = p.parse_args()
    if cli.chunk <= 0 or cli.adapt_points <= 0 or cli.adapt_steps < 0:
        p.error('Chunk/point counts must be positive; adaptation steps nonnegative')
    import random
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != 'vatl_sdf_atlas_rigid_v2':
        p.error('Expected atlas-rigid v2 checkpoint; residual-era models require their original implementation')
    args = checkpoint['args']
    device = torch.device(cli.device)
    model = AtlasSubjectSDF(args, checkpoint['subject_ids'], device).to(device)
    model.load_state_dict(checkpoint['model'])
    model.set_stage(int(checkpoint.get('epoch', 0)))
    model.eval()
    row = None
    frame = pd.read_csv(cli.metadata, sep='\t') if cli.metadata else checkpoint['dataset_df']
    if cli.subject_id:
        rows = frame[frame.subject_id.astype(str) == cli.subject_id]
        if len(rows) != 1:
            p.error('Subject ID must select exactly one metadata row')
        row = rows.iloc[0].to_dict()
    if row is None:
        if cli.mode != 'atlas_only' or not cli.reference or cli.age is None or cli.sex is None:
            p.error('Without --subject-id, atlas_only requires --reference, --age and --sex')
        row = {'scan_age': cli.age, 'sex': cli.sex}
    if cli.age is not None:
        row['scan_age'] = cli.age
    if cli.sex is not None:
        row['sex'] = cli.sex
    condition = Data.load_conditions(SimpleNamespace(args=args), row.copy()).to(device)
    reference = cli.reference or relocate_path(row[args['dataset']['sdf_key']])
    image = nib.load(reference)
    if len(image.shape) != 3:
        p.error('Reference must be 3D')
    shape = image.shape
    half = np.asarray(args['dataset']['world_bbox']) / 2
    center = nib.affines.apply_affine(image.affine, np.asarray(shape) / 2.)
    if cli.mode != 'atlas_only':
        native = nib.load(relocate_path(row[args['dataset']['sdf_key']]))
        center = nib.affines.apply_affine(native.affine, np.asarray(native.shape) / 2.)
    # Generate chunks by flat index to avoid allocating a full N x 3 GPU grid.
    def normalized(indices):
        return (nib.affines.apply_affine(image.affine, indices) - center) / half
    known = cli.subject_id in model.subject_ids
    index = model.subject_ids.index(cli.subject_id) if known else None
    adaptation = None
    output = Path(cli.output)
    output.mkdir(parents=True, exist_ok=False)
    if cli.adaptation_state:
        adaptation = torch.load(cli.adaptation_state, map_location=device, weights_only=False)
    if cli.adapt_steps:
        if cli.mode == 'atlas_only':
            p.error('Adaptation is for subject reconstruction')
        target_nii = nib.load(relocate_path(row[args['dataset']['sdf_key']]))
        target = target_nii.get_fdata()
        # Reuse the actual dataset's inside/surface/tissue/background sampling logic.
        segmentation = nib.load(relocate_path(row['Seg'])).get_fdata()
        context = SimpleNamespace(args=args, world_bbox=2 * half)
        indices = Data.sample_sdf_indices(context, target, segmentation, target_nii.affine, num_points=cli.adapt_points)
        target_center = nib.affines.apply_affine(target_nii.affine, np.array(target.shape) / 2.)
        coords = (nib.affines.apply_affine(target_nii.affine, indices) - target_center) / half
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        targets = torch.as_tensor(target[tuple(indices.T)], dtype=torch.float32, device=device)
        adaptation = adapt_subject(model, coords, condition.expand(len(coords), -1), targets, steps=cli.adapt_steps)
        torch.save(adaptation, output / 'adaptation.pth')
    if cli.mode != 'atlas_only' and not known and adaptation is None:
        p.error('Unseen subjects require --adapt-steps or --adaptation-state')
    keys = ('warped_atlas_mm', 'sdf_mm') if cli.mode == 'atlas_and_components' else ('sdf_mm',)
    fields = {k: np.full(int(np.prod(shape)), np.nan, dtype=np.float32) for k in keys}
    with torch.no_grad():
        for start in range(0, np.prod(shape), cli.chunk):
            stop = min(start + cli.chunk, np.prod(shape))
            indices = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
            coords = torch.as_tensor(normalized(indices), dtype=torch.float32, device=device)
            valid = (coords.abs() <= 1).all(-1)
            if not valid.any():
                continue
            coords = coords[valid]
            kwargs = {}
            if cli.mode != 'atlas_only':
                if adaptation is not None:
                    kwargs = {'transforms': adaptation['transform'].to(device).expand(len(coords), -1)}
                else:
                    kwargs = {'subject_indices': torch.full((len(coords),), index, dtype=torch.long, device=device)}
            pred = model(coords, condition.expand(len(coords), -1), mode=cli.mode,
                         **kwargs)
            for key in keys:
                fields[key][start:stop][valid.cpu().numpy()] = pred[key].cpu().numpy()
    for key, field in fields.items():
        save_img(field.reshape(shape), image.affine, str(output), key + '.nii.gz', spatial_units='mm')
    sdf = fields['sdf_mm'].reshape(shape)
    mask = sdf_to_mask(torch.from_numpy(sdf)).numpy()
    save_img(mask.astype(np.uint8), image.affine, str(output), 'vessel_mask.nii.gz', spatial_units='mm')
    valid = np.isfinite(sdf)
    save_img(valid.astype(np.uint8), image.affine, str(output), 'valid_crop.nii.gz', spatial_units='mm')
    if cli.metrics:
        if cli.mode == 'atlas_only':
            p.error('Native subject metrics are not meaningful for an unregistered atlas')
        target = nib.load(relocate_path(row[args['dataset']['sdf_key']]))
        if target.shape != shape or not np.allclose(target.affine, image.affine):
            p.error('Metrics require output and ground truth on the same physical grid')
        linear = image.affine[:3, :3]
        lengths = np.linalg.norm(linear, axis=0)
        if not np.allclose((linear / lengths).T @ (linear / lengths), np.eye(3), atol=1e-4):
            p.error('Surface metrics require an orthogonal voxel grid')
        result = vascular_metrics(mask & valid, (target.get_fdata() < 0), lengths)
        result['valid_crop_fraction'] = float(valid.mean())
        with (output / 'metrics.json').open('w') as f:
            json.dump(result, f, indent=2)
    with (output / 'inference.json').open('w') as f:
        json.dump({'mode': cli.mode, 'subject_id': cli.subject_id,
                   'valid_crop_fraction': float(valid.mean()), 'reference': reference}, f, indent=2)


if __name__ == '__main__':
    main()
