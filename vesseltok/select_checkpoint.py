"""Native-geometry selection using validation predictions only; immutable alias."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from diagnostics.provenance import sha256,canonical_id,validate_splits
from vesseltok.calibrate import require_validation
from vesseltok.geometry import TubeGraph,assert_grid
from vesseltok.topology import branch_metrics
from vesseltok.selection_policy import POLICY,policy_sha256,geometric_score,select_candidate


def training_run_root(checkpoint):
    parent=Path(checkpoint).resolve().parent
    return parent.parent if parent.name=='candidates' and (parent.parent/'provenance.json').exists() else parent


def score_candidate(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if sha256(manifest['checkpoint'])!=manifest['checkpoint_sha256']:raise ValueError('Candidate checkpoint hash mismatch')
    checkpoint=torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)
    require_validation(manifest['subjects'],checkpoint['splits']);validate_splits(checkpoint['splits'])
    if manifest.get('ablation_mode','original')!='original':raise ValueError('Ablated fields cannot select pretrained checkpoints')
    if {canonical_id(s) for s in manifest['training_subjects']}!={canonical_id(s) for s in checkpoint['splits']['train']}:raise ValueError('Training membership mismatch')
    if manifest.get('learned_transforms') is not False or manifest['threshold']!=POLICY['threshold']:raise ValueError('Uncontrolled validation prediction')
    config=checkpoint['config'];metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id');subjects=[]
    for sid in manifest['subjects']:
        image=nib.load(metadata.loc[sid,'MRA']);prediction=nib.load(root/sid/'occupancy_probability.nii.gz');assert_grid(image);assert_grid(prediction)
        if image.shape!=prediction.shape or not np.allclose(image.affine,prediction.affine,atol=1e-4):raise ValueError('Validation native-grid mismatch')
        p=prediction.get_fdata(dtype=np.float32)
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid validation probabilities')
        mask=p>=.5;saved=nib.load(root/sid/'occupancy_mask.nii.gz')
        if saved.shape!=image.shape or not np.allclose(saved.affine,image.affine) or not np.array_equal(saved.get_fdata()>0,mask):raise ValueError('Validation mask is not the declared raw threshold')
        tube=TubeGraph.load(metadata.loc[sid,'Graph'],config['edge_spacing_mm'])
        ijk=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T;world=nib.affines.apply_affine(image.affine,ijk)
        target=tube.occupancy(world,config['pseudo_radius_mm']).reshape(image.shape)
        dice=2*np.count_nonzero(mask&target)/max(1,int(mask.sum())+int(target.sum()));branch=branch_metrics(mask,tube,image.affine)
        b=branch['branch_preservation_fraction']
        if b is None:raise ValueError('Validation subject has no observed branches')
        subjects.append({'subject_id':sid,'pseudo_dice':float(dice),'branch_preservation':b,'geometry_score':geometric_score(float(dice),b),
                         'false_pair_fraction':branch['false_connection_pair_fraction'],'branch_metrics':branch,'graph_sha256':sha256(metadata.loc[sid,'Graph']),
                         'probability_sha256':sha256(root/sid/'occupancy_probability.nii.gz')})
    false=[r['false_pair_fraction'] for r in subjects if r['false_pair_fraction'] is not None]
    return {'checkpoint':str(Path(manifest['checkpoint']).resolve()),'checkpoint_sha256':manifest['checkpoint_sha256'],'step':int(checkpoint['step']),
            'training_run':str(training_run_root(manifest['checkpoint'])),'validation_prediction_root':str(root),'manifest_sha256':sha256(root/'manifest.json'),
            'score':float(np.mean([r['geometry_score'] for r in subjects])),'false_pair_fraction':float(np.mean(false)) if false else None,
            'subjects':subjects,'splits':checkpoint['splits'],'config':config,'initial_model_sha256':checkpoint['provenance']['initial_model_sha256']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidates',nargs='+',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'policy.json').write_text(json.dumps(POLICY,indent=2))
    candidates=[score_candidate(p) for p in args.candidates]
    for key in ['training_run','splits','config','initial_model_sha256']:
        if any(c[key]!=candidates[0][key] for c in candidates):raise ValueError('Checkpoint candidates do not share '+key)
    selected=select_candidate(candidates)
    record={'policy':POLICY,'policy_sha256':policy_sha256(),'selector_source_sha256':sha256(__file__),'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'selection_subjects':selected['splits']['validation'],'evaluation_used':False,'segmentation_or_sdf_read':False,
            'selected_checkpoint':selected['checkpoint'],'selected_checkpoint_sha256':selected['checkpoint_sha256'],'selected_step':selected['step'],
            'candidates':candidates,'scope':'Only supplied saved checkpoints; not a claim of optimality over unsaved steps'}
    (output/'selection.json').write_text(json.dumps(record,indent=2,allow_nan=False));(output/'selected.pt').symlink_to(Path(selected['checkpoint']).resolve())
    print(json.dumps({'policy_sha256':record['policy_sha256'],'selected_checkpoint':selected['checkpoint'],'selected_step':selected['step'],'candidate_scores':[(c['step'],c['score']) for c in candidates]},indent=2),flush=True)


if __name__=='__main__':main()
