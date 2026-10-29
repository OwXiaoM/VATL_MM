"""Compare completed stages on identical, conservative physical coverage regions."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from pipeline import pair_metrics,dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
    p.add_argument('--stages',nargs='+',default=['R0','R1','R2']);p.add_argument('--output',required=True)
    args=p.parse_args();study=Path(args.study);cfg=json.loads((study/'study.json').read_text())
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    ids=[s['subject_id'] for s in cfg['subjects']];masks={};covers={};deforms={}
    first=nib.load(study/args.stages[0]/ids[0]/'mask.nii.gz');spacing=nib.affines.voxel_sizes(first.affine)
    for stage in args.stages:
        masks[stage]=[];covers[stage]=[];deforms[stage]=[]
        for sid in ids:
            folder=study/stage/sid
            im=nib.load(folder/'mask.nii.gz')
            if im.shape!=first.shape or not np.allclose(im.affine,first.affine):raise ValueError('Different comparison grids')
            masks[stage].append(im.get_fdata()>.5);covers[stage].append(nib.load(folder/'coverage.nii.gz').get_fdata()>.999)
            if (folder/'transforms.json').exists():deforms[stage].append(json.loads((folder/'transforms.json').read_text()))
    shared=[np.logical_and.reduce([covers[stage][i] for stage in args.stages]) for i in range(len(ids))]
    allvalid=np.logical_and.reduce(shared)
    output={}
    for stage in args.stages:
        pairs=[]
        for i,j in itertools.combinations(range(len(ids)),2):
            pairs.append({'a':ids[i],'b':ids[j],**pair_metrics(masks[stage][i],masks[stage][j],shared[i],shared[j],spacing)})
        pmap=np.mean([m[allvalid] for m in masks[stage]],axis=0)
        concentration=float(np.sum(pmap*pmap)/max(1e-12,np.sum(pmap)))
        output[stage]={'pairwise':pairs,'mean_Dice':float(np.mean([r['Dice'] for r in pairs])),
            'mean_clDice':float(np.mean([r['clDice'] for r in pairs])),
            'centerline_mean_mm':float(np.mean([r['centerline_mean_mm'] for r in pairs])),
            'concentration_shared_fov':concentration,'all_subjects_all_stages_valid_voxels':int(allvalid.sum()),
            'max_fold_fraction':max([d['covered']['fold_fraction'] for d in deforms[stage]],default=0),
            'max_displacement_p95_mm':max([d['covered']['displacement_p95_mm'] for d in deforms[stage]],default=0),
            'min_roundtrip_native_Dice':min([d['roundtrip_native_Dice'] for d in deforms[stage]],default=None)}
    dump(out/'comparison.json',{'stages':output,'region':'Per-pair coverage intersection across ALL compared stages; atlas concentration within all-subject/all-stage intersection',
        'interpretation':'Registration diagnostics fitted on these same four subjects; no generalization, normative modelling or neural SDF improvement established'})
    lines=['# Matched-coverage registration comparison','', '| Stage | Dice | clDice | Centerline mean mm | Concentration | Max fold fraction | Max displacement p95 mm |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in output.items():lines.append(f"| {name} | {r['mean_Dice']:.5f} | {r['mean_clDice']:.5f} | {r['centerline_mean_mm']:.3f} | {r['concentration_shared_fov']:.4f} | {r['max_fold_fraction']:.6g} | {r['max_displacement_p95_mm']:.3f} |")
    lines+=['','Pairwise metrics use identical coverage regions across compared stages. These are in-sample registration diagnostics. Concentration alone does not establish preserved topology or a valid population distribution.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
