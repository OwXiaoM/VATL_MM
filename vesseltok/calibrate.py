"""Validation-only threshold selection, then immutable derived evaluation masks."""
import argparse
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from diagnostics.provenance import canonical_id, sha256, validate_splits
from vesseltok.geometry import TubeGraph, assert_grid
from vesseltok.infer import save_nifti


THRESHOLDS = [.3, .4, .5, .6, .7, .8, .9, .95]


def require_validation(subjects, splits):
    validate_splits(splits)
    actual = {canonical_id(s) for s in subjects}
    expected = {canonical_id(s) for s in splits['validation']}
    if len(actual) != len(subjects) or actual != expected:
        raise ValueError('Calibration requires exactly the checkpoint validation people')


def threshold_scores(probability, target, thresholds=THRESHOLDS):
    if probability.shape != target.shape or not np.isfinite(probability).all():
        raise ValueError('Invalid probability geometry or values')
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be bounded')
    positive = int(target.sum()); result = []
    for threshold in thresholds:
        pred = probability >= threshold; tp = int(np.count_nonzero(pred & target)); size = int(pred.sum())
        result.append({'threshold':float(threshold), 'dice':2*tp/max(1,size+positive),
                       'precision':tp/max(1,size), 'recall':tp/max(1,positive),
                       'volume_ratio':size/max(1,positive)})
    return result


def choose_threshold(rows):
    # Fixed macro Dice objective; ties resolve to the lower predeclared threshold.
    scores = [np.mean([row[i]['dice'] for row in rows]) for i in range(len(rows[0]))]
    return rows[0][int(np.argmax(scores))]['threshold'], list(map(float,scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation',required=True); parser.add_argument('--evaluation',required=True)
    parser.add_argument('--output',required=True); args=parser.parse_args()
    validation=Path(args.validation); evaluation=Path(args.evaluation); output=Path(args.output)
    vm=json.loads((validation/'manifest.json').read_text()); em=json.loads((evaluation/'manifest.json').read_text())
    if vm['checkpoint_sha256'] != em['checkpoint_sha256'] or sha256(vm['checkpoint']) != vm['checkpoint_sha256']:
        raise ValueError('Calibration/evaluation checkpoint mismatch')
    checkpoint=torch.load(vm['checkpoint'],map_location='cpu',weights_only=False)
    require_validation(vm['subjects'],checkpoint['splits'])
    validate_splits({'train':checkpoint['splits']['train'],'validation':vm['subjects'],'evaluation':em['subjects']})
    # Lock scope: no new confirmation subjects enter this development comparison.
    if set(em['subjects']) != set(checkpoint['splits']['evaluation']):
        raise ValueError('Evaluation must match pre-existing exploratory cohort')
    output.mkdir(parents=True,exist_ok=False)
    config=checkpoint['config']; metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id')
    rows=[]; subjects=[]
    for subject in vm['subjects']:
        image=nib.load(validation/subject/'occupancy_probability.nii.gz');assert_grid(image)
        reference=nib.load(metadata.loc[subject,'MRA'])
        if image.shape!=reference.shape or not np.allclose(image.affine,reference.affine):raise ValueError('Native geometry mismatch')
        graph=TubeGraph.load(metadata.loc[subject,'Graph'],config['edge_spacing_mm'])
        ijk=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T
        target=graph.occupancy(nib.affines.apply_affine(image.affine,ijk),config['pseudo_radius_mm']).reshape(image.shape)
        row=threshold_scores(image.get_fdata(dtype=np.float32),target);rows.append(row)
        subjects.append({'subject_id':subject,'scores':row})
    selected,macro=choose_threshold(rows)
    record={'objective':'macro graph-pseudo Dice on checkpoint validation people only','threshold_grid':THRESHOLDS,
            'selected_threshold':selected,'macro_validation_dice':macro,'subjects':subjects,
            'checkpoint_sha256':vm['checkpoint_sha256'],'evaluation_used_for_selection':False}
    (output/'calibration.json').write_text(json.dumps(record,indent=2))
    # Never modify original predictions. Derived directories share read-only input files.
    for name,threshold in [('fixed_050',.5),('validation_selected',selected)]:
        dest=output/name;dest.mkdir()
        manifest=dict(em);manifest.update(threshold=threshold,threshold_calibration=str(output/'calibration.json'),source_predictions=str(evaluation))
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for subject in em['subjects']:
            folder=dest/subject;folder.mkdir();source=evaluation/subject
            for filename in ['occupancy_probability.nii.gz','tokens.json']:
                os.link(source/filename,folder/filename)
            image=nib.load(source/'occupancy_probability.nii.gz');values=image.get_fdata(dtype=np.float32)
            save_nifti(folder/'occupancy_mask.nii.gz',(values>=threshold).astype(np.uint8),image.affine)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
