"""Nested graph-only point sampling for a controlled input-cardinality study."""
import numpy as np
from scipy.spatial import cKDTree


def subject_cloud(tube, center, config, rng):
    maximum=config['input_points'];half=config['patch_mm']/2;halo=config['halo_mm']
    policy=config.get('input_sampling')
    if policy is None:return tube.patch_cloud(center,half,halo,maximum,rng)
    if policy!='nested_4096':raise ValueError('Unknown input sampling policy')
    return nested_cloud(tube,center,half,halo,maximum,rng,base_points=4096)


def nested_cloud(tube,center,half,halo,maximum,rng,base_points=4096):
    if maximum<base_points:raise ValueError('Nested maximum must include baseline count')
    base=tube.patch_cloud(center,half,halo,base_points,rng)
    if maximum==base_points or not len(base):return base
    # No random draws when collecting the complete physical crop.
    candidates=tube.patch_cloud(center,half,halo,len(tube.cloud),rng)
    remaining=candidates[cKDTree(base).query(candidates)[0]>1e-8]
    count=min(maximum-len(base),len(remaining))
    extra=remaining[rng.choice(len(remaining),count,replace=False)] if count else remaining[:0]
    result=np.concatenate((base,extra))
    return result[np.lexsort(result.T[::-1])]
