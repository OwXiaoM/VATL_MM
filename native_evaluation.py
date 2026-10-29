"""Deterministic native-grid metrics. No registration optimizer is used here."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity
from data_loading.validation_selection import is_primary, save_subject_manifest


VERSION = 'native_full_reference_v1'
DEFAULT_EVALUATION = {'subject_manifest': None, 'intensity_normalization': 'none',
                      'data_range': 1., 'assume_mm_if_unknown': False,
                      'hash_checkpoint': False}


def validate_affine(affine):
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
        raise ValueError('Invalid NIfTI affine')
    if np.linalg.matrix_rank(affine[:3, :3]) < 3:
        raise ValueError('Singular NIfTI affine')
    return affine


def geometry_resample(image, target, labels=False, outside=0.):
    """Resample solely from the supplied world affines, or return an exact-grid view."""
    validate_affine(image.affine)
    validate_affine(target.affine)
    if len(image.shape) != 3 or len(target.shape) != 3:
        raise ValueError('Evaluation requires 3D volumes')
    if image.shape == target.shape and np.allclose(image.affine, target.affine, rtol=0, atol=1e-5):
        return image.get_fdata(), 'identical_grid'
    result = resample_from_to(image, (target.shape, target.affine), order=0 if labels else 1,
                              mode='constant', cval=float(outside))
    return result.get_fdata(), 'affine_resampled'


def mm_image(image, assume_mm=False):
    validate_affine(image.affine)
    unit = image.header.get_xyzt_units()[0]
    scales = {'mm': 1., 'meter': 1000., 'micron': .001}
    if unit not in scales:
        if not assume_mm:
            raise ValueError(f'Unknown spatial units {unit!r}; cannot establish physical millimetres')
        scale = 1.
    else:
        scale = scales[unit]
    affine = image.affine.copy()
    affine[:3] *= scale
    # Preserve the original proxy dtype; labels can be integer-valued.
    result = nib.Nifti1Image(image.get_fdata(), affine)
    result.header.set_xyzt_units('mm')
    return result


def binary_vessel_metrics(pred, truth, affine):
    """Symmetric voxel-surface ASD/HD95 in world mm, including oblique/sheared grids.

    HD95 = max of the two directed 95th percentiles. ASD weights both surface
    point sets equally per point. Empty distances are undefined, not perfect.
    """
    validate_affine(affine)
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError('Vessel masks must have the same 3D shape')
    p, t = int(pred.sum()), int(truth.sum())
    result = {'Dice': None, 'HD95_mm': None, 'ASD_mm': None,
              'pred_voxels': p, 'truth_voxels': t, 'status': 'ok'}
    if p + t == 0:
        result['status'] = 'both_empty'
        return result
    result['Dice'] = float(2 * np.count_nonzero(pred & truth) / (p + t))
    if not p or not t:
        result['status'] = 'prediction_empty' if not p else 'reference_empty'
        return result
    surfaces = []
    for mask in (pred, truth):
        boundary = mask & ~ndi.binary_erosion(mask, border_value=0)
        surfaces.append(nib.affines.apply_affine(affine, np.argwhere(boundary)))
    pt = cKDTree(surfaces[1]).query(surfaces[0], workers=1)[0]
    tp = cKDTree(surfaces[0]).query(surfaces[1], workers=1)[0]
    result['HD95_mm'] = float(max(np.percentile(pt, 95), np.percentile(tp, 95)))
    result['ASD_mm'] = float((pt.sum() + tp.sum()) / (len(pt) + len(tp)))
    return result


def multiclass_dice(pred, truth, background=0):
    labels = np.union1d(np.unique(pred), np.unique(truth))
    labels = [label for label in labels if label not in (0, background)]
    if not labels:
        return None
    return float(np.mean([2 * np.count_nonzero((pred == label) & (truth == label)) /
                          (np.count_nonzero(pred == label) + np.count_nonzero(truth == label)) for label in labels]))


def image_metrics(pred, truth, config):
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Non-finite image values')
    normalization = config.get('intensity_normalization', 'none')
    if normalization == 'independent_minmax':
        # Explicit legacy option; never silently normalize away scale errors.
        def normalized(array):
            span = np.ptp(array)
            return (array - array.min()) / span if span > 0 else array
        pred, truth = normalized(pred), normalized(truth)
    elif normalization != 'none':
        raise ValueError(f'Unsupported evaluation normalization: {normalization}')
    data_range = float(config.get('data_range', 1.))
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError('evaluation.data_range must be finite and positive')
    mse = float(np.mean((pred - truth) ** 2))
    psnr = float(10 * np.log10(data_range ** 2 / mse)) if mse else float('inf')
    window = min(7, min(truth.shape))
    window -= 1 - window % 2
    ssim = float(structural_similarity(truth, pred, data_range=data_range, win_size=window)) if window >= 3 else None
    return psnr, ssim, 'ssim_grid_too_small' if ssim is None else ('perfect_match' if mse == 0 else 'ok')


def compute_native_metrics(args, pred, affine, row, epoch, split):
    config = {**DEFAULT_EVALUATION, **args.get('evaluation', {})}
    modalities = args['dataset']['modalities']
    labels = args['dataset']['label_names']
    sr_dims = len(modalities) - 1
    result = {'Subject': str(row['subject_id']), 'PSNR': [], 'SSIM': [], 'DICE': [],
              'epoch': int(epoch), 'evaluation_region': 'full_reference_grid'}
    for modality in modalities[:-1]:
        result.update({modality + '_PSNR': None, modality + '_SSIM': None})
    for prefix in ('Seg_Vessel', 'SDF_Vessel'):
        for key in ('Dice', 'HD95_mm', 'ASD_mm'):
            result[prefix + '_' + key] = None
        result[prefix + '_status'] = 'unavailable'
    result['Dice_multiclass'] = None
    result['SDF_Vessel_status'] = 'sdf_disabled'
    try:
        validate_affine(affine)
        if pred.ndim != 4 or pred.shape[-1] < len(modalities):
            raise ValueError('Prediction is missing image/segmentation channels')
        native_header = nib.load(row[modalities[0]]).header
        unit = native_header.get_xyzt_units()[0]
        def prediction_image(array):
            image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine)
            image.header.set_xyzt_units(unit)
            return mm_image(image, config.get('assume_mm_if_unknown', False))
        # Validate source geometry/units even when all modality references are invalid.
        prediction_image(pred[..., 0])
    except (ValueError, OSError, nib.filebasedimages.ImageFileError) as error:
        result['geometry_error'] = str(error)
        result['Seg_Vessel_status'] = result['SDF_Vessel_status'] = 'invalid_prediction_geometry'
        return result
    for i, modality in enumerate(modalities):
        is_seg = i == sr_dims
        try:
            reference = mm_image(nib.load(row[modality]), config.get('assume_mm_if_unknown', False))
            truth = reference.get_fdata()
            predicted, grid = geometry_resample(prediction_image(pred[..., i]), reference, labels=is_seg)
            if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
                raise ValueError('Non-finite reference/prediction values')
            result[modality + '_grid'] = grid
            result[modality + '_reference_voxels'] = int(truth.size)
            if not is_seg:
                psnr, ssim, status = image_metrics(predicted, truth, config)
                result[modality + '_PSNR'], result[modality + '_SSIM'] = psnr, ssim
                result[modality + '_status'] = status
                result['PSNR'].append(psnr)
                result['SSIM'].append(ssim)
                continue
            if not np.equal(truth, np.round(truth)).all() or not np.equal(predicted, np.round(predicted)).all():
                raise ValueError('Discrete segmentation contains non-integer labels')
            bg = labels.index('Background') if 'Background' in labels else 0
            result['Dice_multiclass'] = multiclass_dice(predicted, truth, bg)
            result['DICE'] = [result['Dice_multiclass']]
            if 'Vessel' not in labels:
                result['Seg_Vessel_status'] = result['SDF_Vessel_status'] = 'vessel_label_absent'
                continue
            vessel = labels.index('Vessel')
            binary = truth == vessel
            for key, value in binary_vessel_metrics(predicted == vessel, binary, reference.affine).items():
                result['Seg_Vessel_' + key] = value
            has_sdf = args['inr_decoder'].get('sdf_dim', 0) == 1 and bool(args['dataset'].get('sdf_key'))
            if has_sdf:
                sdf_index = sr_dims + 1 + args['inr_decoder']['out_dim'][-1]
                if pred.shape[-1] <= sdf_index:
                    result['SDF_Vessel_status'] = 'sdf_channel_missing'
                else:
                    sdf = pred[..., sdf_index]
                    if not np.isfinite(sdf).all():
                        result['SDF_Vessel_status'] = 'nonfinite_sdf'
                    else:
                        aligned, _ = geometry_resample(prediction_image(sdf), reference, outside=float(args['optimizer'].get('sdf_trunc_mm', 5.)))
                        for key, value in binary_vessel_metrics(aligned < 0, binary, reference.affine).items():
                            result['SDF_Vessel_' + key] = value
        except (ValueError, OSError, KeyError, nib.filebasedimages.ImageFileError) as error:
            result[modality + '_status'] = 'invalid: ' + str(error)
            if is_seg:
                result['Seg_Vessel_status'] = result['SDF_Vessel_status'] = 'invalid_segmentation_reference_or_geometry'
    return result


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_metric_results(args, metrics, epoch, df=None, split='val'):
    if not is_primary(args) or not metrics:
        return {}
    # Also accept historical PSNR/SSIM lists from external callers, expanding
    # every modality rather than silently taking list element zero.
    metrics = [dict(m) for m in metrics]
    for metric in metrics:
        for index, modality in enumerate(args['dataset']['modalities'][:-1]):
            for name in ('PSNR', 'SSIM'):
                values = metric.get(name, [])
                if isinstance(values, (list, tuple)) and index < len(values):
                    metric.setdefault(modality + '_' + name, values[index])
        if isinstance(metric.get('DICE'), (list, tuple)) and metric['DICE']:
            metric.setdefault('Dice_multiclass', metric['DICE'][0])
    ids = [str(m['Subject']) for m in metrics]
    root = Path(args['output_dir']) / split
    snapshot = root / f'epoch_{epoch:04d}'
    snapshot.mkdir(parents=True, exist_ok=True)
    save_subject_manifest(args, ids, split, directory=root)
    save_subject_manifest(args, ids, split, directory=snapshot)
    load = args.get('load_model', {})
    checkpoint = Path(load['path']) / f"checkpoint_epoch_{load['epoch']}.pth" if load.get('path') and epoch == 0 else Path(args['output_dir']) / f'checkpoint_epoch_{epoch}.pth'
    explicit_checkpoint = args.get('evaluation', {}).get('evaluated_checkpoint')
    if explicit_checkpoint:
        checkpoint = Path(explicit_checkpoint)
    fingerprint = None
    # Training checkpoints can be large. Hash only when explicitly requested.
    if args.get('evaluation', {}).get('hash_checkpoint', False) and checkpoint.is_file():
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
    context = {'protocol': VERSION, 'training_epoch': int(epoch),
               'checkpoint': str(checkpoint.resolve()), 'checkpoint_exists': checkpoint.is_file(),
               'checkpoint_sha256': fingerprint,
               'checkpoint_source': 'checkpoint_file' if checkpoint.is_file() else 'in_memory_weights_checkpoint_not_saved',
               'adaptation_epochs': int(args['epochs']['val']) if split in ('val', 'test') else 0,
               'subject_ids': ids, 'evaluation': {**DEFAULT_EVALUATION, **args.get('evaluation', {})},
               'region': 'full_reference_grid',
               'configuration': {k: v for k, v in args.items() if k != '_evaluation_source_config'},
               'source_experiment_configuration': args.get('_evaluation_source_config')}
    rows = [{**{k: v for k, v in m.items() if not isinstance(v, (list, dict, tuple))},
             'training_epoch': int(epoch), 'adaptation_epochs': context['adaptation_epochs'],
             'checkpoint': context['checkpoint']} for m in metrics]
    # Numeric metrics exclude identity, provenance, voxel counts and epoch counters.
    keys = sorted({k for row in rows for k in row if k.endswith(('_PSNR', '_SSIM', '_Dice', '_HD95_mm', '_ASD_mm')) or k == 'Dice_multiclass'})
    summary = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        valid = [float(v) for v in values if isinstance(v, (int, float, np.number)) and np.isfinite(v)]
        summary[key] = {'mean': float(np.mean(valid)) if valid else None,
                        'std': float(np.std(valid, ddof=0)) if valid else None,
                        'valid_count': len(valid), 'undefined_count': sum(v is None for v in values),
                        'positive_infinite_count': int(sum(v is not None and np.isposinf(v) for v in values)),
                        'nonfinite_count': sum(v is not None and not np.isfinite(v) for v in values)}
        print(f'[{split}] {key}: {summary[key]}')
    payload = clean_json({'metrics': summary, 'evaluation_configuration': context})
    for directory in (root, snapshot):
        with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        (directory / 'metrics_summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    # Retain the historical JSON/dataframe filenames as well as named scalar CSVs.
    (root / f'{split}_metrics_ep={epoch}.json').write_text(json.dumps(clean_json(metrics), indent=2, allow_nan=False) + '\n')
    if df is not None:
        df.to_csv(root / f'{split}_df.csv', index=False)
    return summary
