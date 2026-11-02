"""Endpoint reachability complements, but cannot replace, exact source-path support."""
from collections import Counter
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def endpoint_diagnosis(mask,tube,affine,branches):
    components,_=ndi.label(np.asarray(mask,bool),np.ones((3,3,3)))
    voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),tube.points)).astype(int)
    inside=np.all((voxel>=0)&(voxel<np.asarray(mask.shape)),axis=1)
    node_labels=np.zeros(len(tube.points),int);node_labels[inside]=components[tuple(voxel[inside].T)]
    result=[]
    for branch in branches:
        a,b=branch['nodes'][0],branch['nodes'][-1];first,last=int(node_labels[a]),int(node_labels[b])
        if not inside[[a,b]].all():status='outside_fov'
        elif not first or not last:status='endpoint_not_foreground'
        elif first!=last:status='endpoints_in_different_components'
        else:status='endpoints_in_same_component'
        result.append({'branch_id':branch['branch_id'],'native_branch_preserved':branch['native_branch_preserved'],
                       'field_category':branch['category'],'endpoint_labels':[first,last],'status':status})
    return {'counts':dict(Counter(r['status'] for r in result)),
            'broken_source_paths_with_connected_endpoints':sum(not r['native_branch_preserved'] and r['status']=='endpoints_in_same_component' for r in result),
            'branches':result,'interpretation':'Reachability can use an incorrect detour or shortcut; same component does not certify source-branch preservation. Source-path field gaps do not alone establish global disconnection.'}
