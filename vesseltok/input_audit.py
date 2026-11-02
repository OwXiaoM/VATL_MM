"""Descriptive input/FPS spatial coverage; never a decoder fidelity bound."""
import argparse
import json
from itertools import product
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from vesseltok.geometry import TubeGraph, local_to_world
from vesseltok.input_sampling import subject_cloud


def summarize_distance(distance):
    if not len(distance) or not np.isfinite(distance).all():raise ValueError('Uncovered or invalid centerline')
    return {'mean_mm':float(distance.mean()),'p95_mm':float(np.percentile(distance,95)),
            'fraction_within_0_5mm':float((distance<=.5).mean()),'fraction_within_1mm':float((distance<=1).mean())}


def audit_subject(root,subject,config,metadata):
    image=nib.load(metadata.loc[subject,'MRA']);tube=TubeGraph.load(metadata.loc[subject,'Graph'],config['edge_spacing_mm'])
    tokens=json.loads((root/subject/'tokens.json').read_text())['patch_tokens'];points=tube.cloud
    voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),points)
    keep=np.all((voxel>=0)&(voxel<=np.asarray(image.shape)-1),axis=1);points=points[keep]
    corners=np.array(list(product(*[(0,n-1) for n in image.shape])),dtype=np.float32)
    world=nib.affines.apply_affine(image.affine,corners).astype(np.float32);low=world.min(0)
    stride=config.get('inference_stride_mm',config['patch_mm']*.75);dimensions=np.floor((world.max(0)-low)/stride).astype(int)+1
    input_distance=np.full(len(points),np.inf);anchor_distance=input_distance.copy();count=0;anchors_count=0;largest_error=0.
    for record in tokens:
        center=np.asarray(record['center_mm']);half=config['patch_mm']/2
        if config.get('representation')=='subject_tokens':seed=config['seed']
        else:
            cell=np.rint((center-low)/stride-.5).astype(int);seed=config['seed']+int(np.ravel_multi_index(cell,dimensions))
        cloud=subject_cloud(tube,center,config,np.random.default_rng(seed))
        anchors=local_to_world(np.asarray(record['anchors_local'])[~np.asarray(record['padding'])],center,half)
        error=float(cKDTree(cloud).query(anchors)[0].max());largest_error=max(largest_error,error)
        if error>1e-4:raise AssertionError('Saved FPS anchors do not match regenerated inference inputs')
        ids=np.flatnonzero(np.all(np.abs(points-center)<half,axis=1));count+=len(cloud);anchors_count+=len(anchors)
        input_distance[ids]=np.minimum(input_distance[ids],cKDTree(cloud).query(points[ids])[0])
        anchor_distance[ids]=np.minimum(anchor_distance[ids],cKDTree(anchors).query(points[ids])[0])
    return {'subject_id':subject,'observed_source_dense_points':len(points),'outside_fov_source_dense_points':int((~keep).sum()),
            'patches':len(tokens),'input_points_summed_over_patches':count,'valid_fps_anchors_summed_over_patches':anchors_count,
            'regenerated_input_anchor_max_error_mm':largest_error,'input_point_proximity':summarize_distance(input_distance),
            'fps_anchor_proximity':summarize_distance(anchor_distance)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--roots',nargs='+',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    results=[]
    for path in args.roots:
        root=Path(path);manifest=json.loads((root/'manifest.json').read_text());checkpoint=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)
        config=checkpoint['config'];metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id')
        rows=[audit_subject(root,s,config,metadata) for s in manifest['subjects']]
        results.append({'prediction_root':str(root),'subjects':rows})
    record={'interpretation':'Nearest-point proximity is descriptive, not a reconstruction bound. Encoder attends all input points; FPS anchors are not the entire token information. Overlapping patches duplicate information and are not a matched whole-subject compression comparison. No segmentation/SDF is read.','runs':results}
    Path(args.output).write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
