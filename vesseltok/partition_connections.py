"""Supplementary node-pair connectivity relative to an explicit target partition."""
import numpy as np


def connected_pairs(labels):
    labels=np.asarray(labels);_,counts=np.unique(labels[labels>0],return_counts=True)
    return int(np.sum(counts*(counts-1)//2))


def partition_connections(prediction,target):
    p=np.asarray(prediction);t=np.asarray(target)
    if p.ndim!=1 or p.shape!=t.shape or np.any(p<0) or np.any(t<0):raise ValueError('Aligned nonnegative component labels required')
    valid=(p>0)&(t>0)
    _,counts=np.unique(np.c_[p[valid],t[valid]],axis=0,return_counts=True)
    joint=int(np.sum(counts*(counts-1)//2));pc=connected_pairs(p);tc=connected_pairs(t);total=len(p)*(len(p)-1)//2
    return {'predicted_connected_pairs':pc,'target_connected_pairs':tc,'shared_connected_pairs':joint,
            'excess_connected_pairs_vs_target':pc-joint,'target_connections_missing':tc-joint,
            'excess_fraction_of_target_disconnected_pairs':(pc-joint)/(total-tc) if total>tc else None,
            'missing_fraction_of_target_connected_pairs':(tc-joint)/tc if tc else None}
