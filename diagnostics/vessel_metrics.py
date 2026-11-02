"""Vessel metrics on native grids. Distances use full voxel-to-mm affines."""
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def components(mask):
    labels, count = ndi.label(mask, ndi.generate_binary_structure(3, 1))
    sizes = np.sort(np.bincount(labels.ravel())[1:])[::-1]
    return {'count_6_connected': int(count), 'voxels': int(mask.sum()),
            'largest_fraction': float(sizes[0] / mask.sum()) if sizes.size else 0.,
            'sizes_voxels': sizes.tolist()}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def vessel_metrics(prediction, reference, affine, observed=None, coverage_mm=1.):
    if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite vessel mask')
    if not np.isfinite(coverage_mm) or coverage_mm <= 0:
        raise ValueError('Coverage tolerance must be positive finite mm')
    prediction, reference = np.asarray(prediction, bool), np.asarray(reference, bool)
    if prediction.ndim != 3 or reference.shape != prediction.shape:
        raise ValueError('Matching 3D grids required')
    affine = np.asarray(affine, float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError('Valid voxel-to-mm affine required')
    observed = np.ones_like(reference) if observed is None else np.asarray(observed, bool)
    if observed.shape != reference.shape or not observed.any():
        raise ValueError('Observed FOV must be nonempty and share evaluation grid')
    outside = {'voxels': int((~observed).sum()),
               'reference_vessel_voxels': int((reference & ~observed).sum()),
               'predicted_vessel_voxels': int((prediction & ~observed).sum())}
    # Compute centerlines before FOV intersection; clipping must not create endpoints.
    sp = skeletonize(prediction).astype(bool) & observed
    sr = skeletonize(reference).astype(bool) & observed
    p, r = prediction & observed, reference & observed
    tp = int((p & r).sum())
    top_precision, top_recall = _ratio((sp & r).sum(), sp.sum()), _ratio((sr & p).sum(), sr.sum())
    cldice = (2 * top_precision * top_recall / (top_precision + top_recall)
              if top_precision is not None and top_recall is not None and top_precision + top_recall else 0.)
    skeleton_status = 'ok'
    if not p.any() and not r.any():
        cldice = 1.
    elif p.any() and r.any() and (not sp.any() or not sr.any()):
        cldice = None
        skeleton_status = 'empty_skeleton_for_nonempty_mask'
    surface_p = p & ~ndi.binary_erosion(p)
    surface_r = r & ~ndi.binary_erosion(r)
    def points(mask):
        return nib.affines.apply_affine(affine, np.argwhere(mask))
    hd95 = asd = coverage = None
    if surface_p.any() and surface_r.any():
        pp, pr = points(surface_p), points(surface_r)
        dp, dr = cKDTree(pr).query(pp)[0], cKDTree(pp).query(pr)[0]
        # Symmetric pooled surface-distance percentile and pooled mean.
        distances = np.concatenate((dp, dr))
        hd95, asd = float(np.percentile(distances, 95)), float(distances.mean())
    if sr.any():
        coverage = float((cKDTree(points(p)).query(points(sr))[0] <= coverage_mm).mean()) if p.any() else 0.
    voxel_volume = abs(np.linalg.det(affine[:3, :3]))
    return {'dice': _ratio(2 * tp, int(p.sum() + r.sum())) if p.any() or r.any() else 1.,
            'precision': _ratio(tp, p.sum()), 'recall': _ratio(tp, r.sum()),
            'volume_ratio': _ratio(p.sum(), r.sum()), 'cldice': cldice, 'skeleton_metric_status': skeleton_status,
            'hd95_mm': hd95, 'asd_mm': asd, 'centerline_coverage': coverage,
            'coverage_tolerance_mm': coverage_mm, 'observed_voxels': int(observed.sum()),
            'prediction_volume_mm3': float(p.sum() * voxel_volume),
            'reference_volume_mm3': float(r.sum() * voxel_volume),
            'prediction_components': components(p), 'reference_components': components(r),
            'out_of_fov': outside, 'surface_metric_status': 'ok' if hd95 is not None else 'empty_surface',
            'fov_boundary_surface_voxels': int(((surface_p | surface_r) & ~ndi.binary_erosion(observed)).sum())}
