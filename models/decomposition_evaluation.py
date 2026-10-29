"""Full segmentation-grid reconstruction metrics for decomposition models."""
import json
from pathlib import Path
from types import SimpleNamespace
import nibabel as nib
import numpy as np
import torch
from data_paths import relocate_path
from data_loading.dataset import Data
from native_evaluation import binary_vessel_metrics, mm_image, clean_json


def evaluate_subject(model, row, subject_index=None, adaptation=None, chunk=8192, topology=False):
    """Evaluate in world mm, centered on native SDF as in infer_decomposition.py.

    No fitting, resampling, threshold tuning, or crop-restricted truth is used here.
    """
    if chunk <= 0:
        raise ValueError('chunk must be positive')
    args = model.args
    device = next(model.parameters()).device
    native = mm_image(nib.load(relocate_path(row[args['dataset']['sdf_key']])))
    reference = mm_image(nib.load(relocate_path(row[args['dataset']['modalities'][-1]])))
    truth = reference.get_fdata() == args['dataset']['label_names'].index('Vessel')
    observed = np.ones(truth.shape, dtype=bool)
    coverage_key = args['dataset'].get('coverage_key')
    if coverage_key:
        coverage = mm_image(nib.load(relocate_path(row[coverage_key])))
        if coverage.shape != reference.shape or not np.allclose(coverage.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError('Evaluation coverage/reference grid mismatch')
        observed = coverage.get_fdata() > float(args['dataset'].get('coverage_threshold', .999))
        if not observed.any():
            raise ValueError('No observed reference coverage')
        truth = truth & observed
    half = np.asarray(args['dataset']['world_bbox']) / 2
    center = nib.affines.apply_affine(native.affine, np.asarray(native.shape) / 2)
    condition = Data.load_conditions(SimpleNamespace(args=args), dict(row)).to(device)
    masks = {k: np.zeros(truth.size, dtype=bool) for k in ('full', 'atlas')}
    valid_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, truth.size, chunk):
                stop = min(start + chunk, truth.size)
                voxels = np.column_stack(np.unravel_index(np.arange(start, stop), truth.shape))
                coords = (nib.affines.apply_affine(reference.affine, voxels) - center) / half
                valid = (np.abs(coords) <= 1).all(-1) & observed.reshape(-1)[start:stop]
                if not valid.any():
                    continue
                coords = torch.as_tensor(coords[valid], dtype=torch.float32, device=device)
                if adaptation is None:
                    if subject_index is None:
                        raise ValueError('Known subject index or adaptation state required')
                    kwargs = {'subject_indices': torch.full((len(coords),), subject_index, dtype=torch.long, device=device)}
                else:
                    kwargs = {'transforms': adaptation['transform'].to(device).expand(len(coords), -1)}
                components = model(coords, condition.expand(len(coords), -1), **kwargs)
                for name, key in [('full', 'sdf_mm'), ('atlas', 'warped_atlas_mm')]:
                    masks[name][start:stop][valid] = (components[key] < 0).cpu().numpy()
                valid_count += len(coords)
    finally:
        model.train(was_training)
    pred = masks['full'].reshape(truth.shape)
    atlas = masks['atlas'].reshape(truth.shape)
    record = {'subject_id': str(row['subject_id']), **binary_vessel_metrics(pred, truth, reference.affine)}
    tp = int((pred & truth).sum())
    record.update(precision=tp/max(1,int(pred.sum())), recall=tp/max(1,int(truth.sum())),
                  atlas_Dice=binary_vessel_metrics(atlas, truth, reference.affine)['Dice'],
                  volume_ratio=float(pred.sum()/max(1,int(truth.sum()))),
                  valid_crop_fraction=valid_count/int(observed.sum()))
    if coverage_key:
        record.update(evaluation_region='observed_full_reference_grid',
                      observed_fraction=float(observed.mean()),
                      unobserved_policy='excluded from prediction and truth; observed truth outside model crop still counted',
                      surface_note='surfaces clipped to observed support')
    if topology:
        from models.vascular_metrics import vascular_metrics
        linear = reference.affine[:3,:3]
        spacing = np.linalg.norm(linear, axis=0)
        if np.allclose((linear / spacing).T @ (linear / spacing), np.eye(3), atol=1e-4):
            record.update(vascular_metrics(pred, truth, spacing))
    return clean_json(record)


def evaluate_training(model, frame, output, epoch, max_subjects=1, chunk=8192):
    """Known training-subject reconstruction; never call this held-out validation."""
    records = []
    for subject in model.subject_ids[:max_subjects]:
        row = frame[frame.subject_id.astype(str) == subject].iloc[0].to_dict()
        records.append(evaluate_subject(model, row, model.subject_ids.index(subject), chunk=chunk))
    payload = {'epoch': epoch, 'protocol': 'training_reconstruction_no_adaptation', 'subjects': records}
    path = Path(output) / 'reconstruction_metrics.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(payload, allow_nan=False) + '\n')
    values = [r['Dice'] for r in records if r['Dice'] is not None]
    return float(np.mean(values)) if values else None
