"""Pure geometry/partition helpers for a read-only break-localization audit."""
import numpy as np
from scipy.spatial import cKDTree
from diagnostics.provenance import canonical_id
from vesseltok.partition_connections import connected_pairs,partition_connections


def require_audit_scope(subjects,splits):
    allowed={canonical_id(s) for key in ['train','validation'] for s in splits[key]}
    actual=[canonical_id(s) for s in subjects]
    if len(set(actual))!=len(actual) or not set(actual)<=allowed:raise ValueError('Audit permits training/selection-validation subjects only')


def arc_weights(arc):
    arc=np.asarray(arc,float)
    if arc.ndim!=1 or len(arc)<2 or np.any(np.diff(arc)<0):raise ValueError('Nondecreasing branch arc required')
    result=np.zeros(len(arc));interval=np.diff(arc)/2
    result[:-1]+=interval;result[1:]+=interval
    return result


def true_runs(mask):
    mask=np.asarray(mask,bool)
    boundaries=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(boundaries==1).tolist(),np.flatnonzero(boundaries==-1).tolist()))


def joint_pairs(*labels):
    arrays=[np.asarray(v) for v in labels];keep=np.logical_and.reduce([v>0 for v in arrays])
    if not keep.any():return 0
    _,counts=np.unique(np.stack([v[keep] for v in arrays],axis=1),axis=0,return_counts=True)
    return int(np.sum(counts*(counts-1)//2))


def pair_transitions(a,b,target):
    a,b,target=map(np.asarray,(a,b,target))
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=target.shape or any(np.any(v<0) for v in [a,b,target]):raise ValueError('Aligned nonnegative partitions required')
    ab=joint_pairs(a,b);at=joint_pairs(a,target);bt=joint_pairs(b,target);abt=joint_pairs(a,b,target)
    result={'removed_excess_pairs':connected_pairs(a)-ab-at+abt,'gained_excess_pairs':connected_pairs(b)-ab-bt+abt,
            'removed_target_connections':at-abt,'gained_target_connections':bt-abt,
            'A':partition_connections(a,target),'B':partition_connections(b,target)}
    assert result['B']['excess_connected_pairs_vs_target']-result['A']['excess_connected_pairs_vs_target']==result['gained_excess_pairs']-result['removed_excess_pairs']
    assert result['B']['target_connections_missing']-result['A']['target_connections_missing']==result['removed_target_connections']-result['gained_target_connections']
    return result


def neighbor_branch_distance(points,branch_ids,coarse,paths,radius=6.):
    """Nearest nonincident .25mm source sample within radius; inf is censored."""
    points=np.asarray(points);branch_ids=np.asarray(branch_ids);coarse=np.asarray(coarse,bool)
    nodes=[set(p) for p in paths];allowed=np.array([[not bool(a&b) for b in nodes] for a in nodes])
    reference=points[coarse];reference_ids=branch_ids[coarse];tree=cKDTree(reference)
    result=np.full(len(points),np.inf);k=min(64,len(reference))
    for start in range(0,len(points),2048):
        q=points[start:start+2048];ids=branch_ids[start:start+2048]
        distance,index=tree.query(q,k=k)
        if k==1:distance=distance[:,None];index=index[:,None]
        valid=allowed[ids[:,None],reference_ids[index]]&(distance<=radius)
        found=valid.any(1);result[start:start+len(q)][found]=np.where(valid,distance,np.inf).min(1)[found]
        unresolved=np.flatnonzero(~found&(distance[:,-1]<radius))
        for i in unresolved:
            candidates=np.asarray(tree.query_ball_point(q[i],radius),int)
            candidates=candidates[allowed[ids[i],reference_ids[candidates]]]
            if len(candidates):result[start+i]=np.linalg.norm(reference[candidates]-q[i],axis=1).min()
    return result


def weighted_summary(weights,pa,pb,keep):
    weights=np.asarray(weights);keep=np.asarray(keep,bool);w=weights[keep];total=float(w.sum())
    return {'points':int(keep.sum()),'baseline_supported_arc_mm':total,
            'new_loss_arc_mm':float(weights[keep&(pb<.5)].sum()),
            'new_loss_arc_fraction':float(np.dot(w,pb[keep]<.5)/total) if total else None,
            'mean_probability_delta':float(np.dot(w,(pb-pa)[keep])/total) if total else None}


def exposure_bins(distance,edges,weights,pa,pb,base):
    rows=[]
    for low,high in zip(edges[:-1],edges[1:]):
        keep=base&(distance>=low)&((distance<high) if np.isfinite(high) else ~np.isnan(distance))
        rows.append({'lower_mm':low,'upper_mm':high if np.isfinite(high) else None,**weighted_summary(weights,pa,pb,keep)})
    return rows
