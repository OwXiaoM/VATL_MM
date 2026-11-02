"""Fixed-anchor feature-use controls; no weights or source tokens are modified."""
import copy
import numpy as np
from scipy.spatial import cKDTree
from vesseltok.geometry import local_to_world


def ablate_record(record,mode,seed=42):
    if mode not in ['original','zero','shuffle','anchor_balls']:raise ValueError('Unknown feature control')
    result=copy.deepcopy(record);features=np.asarray(record['mean'],np.float32);valid=np.flatnonzero(~np.asarray(record['padding'],bool))
    permutation=np.arange(len(features))
    if mode=='zero':features=np.zeros_like(features)
    if mode=='shuffle':
        if len(valid)<2:raise ValueError('Feature shuffle requires at least two valid tokens')
        rng=np.random.default_rng(seed)
        for _ in range(100):
            shuffled=rng.permutation(valid)
            if np.all(shuffled!=valid):break
        else:raise RuntimeError('Could not construct derangement')
        permutation[valid]=shuffled;features=features[permutation]
    result['mean']=features.tolist()
    return result,permutation


def world_anchors(record,half_size):
    return local_to_world(np.asarray(record['anchors_local'])[~np.asarray(record['padding'],bool)],record['center_mm'],half_size)


def anchor_occupancy(points,anchors,radius_mm=1.):
    if radius_mm<=0 or not np.isfinite(radius_mm):raise ValueError('Positive finite physical radius required')
    return cKDTree(anchors).query(np.asarray(points,dtype=np.float32))[0]<=radius_mm
