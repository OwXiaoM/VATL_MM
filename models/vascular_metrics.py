"""Voxel-grid vascular metrics; these measure topology, not enforce it."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from skimage.measure import euler_number


def vascular_metrics(pred, truth, spacing, tolerance_mm=1.):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape:
        raise ValueError('Metrics require matching grids')
    if not pred.any() or not truth.any():
        return {'dice': float(2 * (pred & truth).sum() / max(1, pred.sum() + truth.sum())),
                'surface_dice': None, 'asd_mm': None, 'cldice': None,
                'centerline_recall': None, 'components_pred': int(ndi.label(pred, np.ones((3, 3, 3)))[1]),
                'components_truth': int(ndi.label(truth, np.ones((3, 3, 3)))[1])}
    ps = pred & ~ndi.binary_erosion(pred)
    ts = truth & ~ndi.binary_erosion(truth)
    dt = ndi.distance_transform_edt(~ts, sampling=spacing)[ps]
    dp = ndi.distance_transform_edt(~ps, sampling=spacing)[ts]
    skp, skt = skeletonize(pred), skeletonize(truth)
    precision = float((skp & truth).sum() / max(1, skp.sum()))
    recall = float((skt & pred).sum() / max(1, skt.sum()))
    cp = int(ndi.label(pred, np.ones((3, 3, 3)))[1])
    ct = int(ndi.label(truth, np.ones((3, 3, 3)))[1])
    ep, et = int(euler_number(pred, connectivity=3)), int(euler_number(truth, connectivity=3))
    return {'dice': float(2 * (pred & truth).sum() / (pred.sum() + truth.sum())),
            'surface_dice': float(((dt <= tolerance_mm).sum() + (dp <= tolerance_mm).sum()) / (len(dt) + len(dp))),
            'asd_mm': float((dt.sum() + dp.sum()) / (len(dt) + len(dp))),
            'cldice': 2 * precision * recall / max(precision + recall, 1e-12) if skp.any() and skt.any() else None,
            'centerline_recall': recall if skt.any() else None, 'components_pred': cp, 'components_truth': ct,
            'component_count_error': abs(cp-ct), 'euler_pred': ep, 'euler_truth': et,
            'euler_error': abs(ep-et)}
