"""Explicit digital topology and source-branch support; no cleanup/registration."""
import numpy as np
import nibabel as nib
import networkx as nx
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import euler_number
from skimage.morphology import skeletonize
from diagnostics.vessel_metrics import vessel_metrics


def betti(mask):
    mask=np.asarray(mask,bool)
    b0=int(ndi.label(mask,np.ones((3,3,3)))[1])
    filled=ndi.binary_fill_holes(mask)  # dual background 6-connectivity
    b2=int(ndi.label(filled & ~mask,ndi.generate_binary_structure(3,1))[1])
    chi=int(euler_number(mask,connectivity=3))
    return {'betti0_26_connected':b0,'betti1':b0+b2-chi,'betti2':b2,'euler':chi}


def skeleton_graph(mask,affine):
    skel=skeletonize(np.asarray(mask,bool)).astype(bool)
    voxels=np.argwhere(skel); lookup={tuple(p):i for i,p in enumerate(voxels)}
    edges=[]
    for i,voxel in enumerate(voxels):
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    if (dx,dy,dz)==(0,0,0):continue
                    j=lookup.get(tuple(voxel+np.array([dx,dy,dz])))
                    if j is not None and j>i:edges.append((i,j))
    points=nib.affines.apply_affine(affine,voxels) if len(voxels) else np.empty((0,3))
    return skel,points,np.asarray(edges,dtype=int).reshape(-1,2)


def source_branches(graph):
    """Maximal degree-two chains, including all-degree-two closed cycles."""
    seen=set();branches=[]
    for origin in list(n for n in graph if graph.degree(n)!=2)+list(graph):
        for neighbor in graph.neighbors(origin):
            edge=frozenset((origin,neighbor))
            if edge in seen:continue
            path=[origin,neighbor];seen.add(edge);previous,current=origin,neighbor
            while graph.degree(current)==2 and current!=origin:
                candidates=[v for v in graph.neighbors(current) if v!=previous]
                next_node=candidates[0];edge=frozenset((current,next_node))
                if edge in seen:break
                seen.add(edge);path.append(next_node);previous,current=current,next_node
            branches.append(path)
    return branches


def branch_metrics(prediction,tube,affine,observed=None):
    """Exact-voxel branch support and cross-component merge counts (lower bounds).

    Same-component endpoint reachability alone cannot certify correct paths.
    A branch is preserved only if all densely sampled edge points are foreground
    and its endpoints are in the same predicted 26-connected component.
    """
    labels,count=ndi.label(prediction,np.ones((3,3,3)))
    inverse=np.linalg.inv(affine)
    def lookup(points):
        voxel=np.rint(nib.affines.apply_affine(inverse,points)).astype(int)
        inside=np.all((voxel>=0)&(voxel<np.asarray(prediction.shape)),axis=1)
        if observed is not None:
            selected=np.flatnonzero(inside);inside[selected]&=observed[tuple(voxel[selected].T)]
        result=np.zeros(len(voxel),int);result[inside]=labels[tuple(voxel[inside].T)]
        return result,inside
    node_labels,node_inside=lookup(tube.points)
    kept=broken=excluded=0;coverage=[]
    for path in source_branches(tube.graph):
        pieces=[]
        for a,b in zip(path[:-1],path[1:]):
            n=max(2,int(np.ceil(np.linalg.norm(tube.points[b]-tube.points[a])/.25))+1)
            pieces.append(np.linspace(tube.points[a],tube.points[b],n))
        samples=np.concatenate(pieces); hit,inside=lookup(samples)
        if not inside.all():excluded+=1;continue
        coverage.append(float((hit>0).mean()))
        if np.all(hit>0) and node_labels[path[0]]>0 and node_labels[path[0]]==node_labels[path[-1]]:kept+=1
        else:broken+=1
    gt_component=np.zeros(len(tube.points),int)
    for index,nodes in enumerate(nx.connected_components(tube.graph)):gt_component[list(nodes)]=index
    # Count all source-node pairs from distinct GT components merged by prediction.
    false_pairs=0
    for component in np.unique(node_labels[node_labels>0]):
        memberships=gt_component[node_labels==component];sizes=np.unique(memberships,return_counts=True)[1]
        false_pairs+=int((len(memberships)**2-(sizes*sizes).sum())//2)
    sizes=np.unique(gt_component[node_inside],return_counts=True)[1]
    eligible=int((node_inside.sum()**2-(sizes*sizes).sum())//2)
    return {'source_branches_preserved':kept,'source_branches_broken':broken,'branches_excluded_outside_fov':excluded,
            'branch_preservation_fraction':kept/(kept+broken) if kept+broken else None,
            'mean_branch_point_coverage':float(np.mean(coverage)) if coverage else None,
            'false_connections_cross_component_node_pairs':false_pairs,
            'eligible_cross_component_node_pairs':eligible,
            'false_connection_pair_fraction':false_pairs/eligible if eligible else None,
            'definition':'Exact-voxel dense path support plus endpoint connectivity; false connections count cross-source-component merges only, not all intracomponent shortcuts'}


def reconstruction_metrics(prediction,target,affine,tube=None,observed=None):
    result=vessel_metrics(prediction,target,affine,observed)
    if observed is not None:
        prediction=prediction & observed;target=target & observed
    union=int((prediction|target).sum());result['iou']=float((prediction&target).sum()/union) if union else 1.
    pred_betti,gt_betti=betti(prediction),betti(target)
    result['topology_prediction']=pred_betti;result['topology_target']=gt_betti
    result['betti0_absolute_error']=abs(pred_betti['betti0_26_connected']-gt_betti['betti0_26_connected'])
    result['betti1_absolute_error']=abs(pred_betti['betti1']-gt_betti['betti1'])
    p=skeletonize(prediction).astype(bool);r=skeletonize(target).astype(bool)
    if p.any() and r.any():
        pp=nib.affines.apply_affine(affine,np.argwhere(p));rr=nib.affines.apply_affine(affine,np.argwhere(r))
        result['centerline_chamfer_symmetric_mean_mm']=float((cKDTree(pp).query(rr)[0].mean()+cKDTree(rr).query(pp)[0].mean())/2)
    else:result['centerline_chamfer_symmetric_mean_mm']=None
    if tube is not None:result['source_branch_connectivity']=branch_metrics(prediction,tube,affine,observed)
    return result
