"""Frozen training reconstructions in observed aligned and original native space.

No SDF adaptation, threshold tuning, image encoder or population generalization
claim. Predictions are evaluated on the original full native segmentation grids.
"""
import argparse,csv,json,sys,os
from pathlib import Path
REPO=Path(os.environ.get('SDF_REPO',Path(__file__).resolve().parents[4])).resolve();sys.path.insert(0,str(REPO))
import ants
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from models.sdf_decomposition import AtlasSubjectSDF
from data_loading.dataset import Data
from native_evaluation import binary_vessel_metrics,clean_json
from models.vascular_metrics import vascular_metrics


def save(path,array,affine):
    im=nib.Nifti1Image(array,affine);im.header.set_xyzt_units('mm');nib.save(im,path)


def predict(model,row,subject,chunk=8192):
    image=nib.load(row['SDF']);shape=image.shape;aff=image.affine
    covered=nib.load(row['Coverage']).get_fdata()>.999
    half=np.asarray(model.args['dataset']['world_bbox'])/2;center=nib.affines.apply_affine(aff,np.array(shape)/2)
    device=next(model.parameters()).device
    condition=Data.load_conditions(SimpleNamespace(args=model.args),dict(row)).to(device)
    field=np.full(np.prod(shape),model.shared.tau,np.float32)
    with torch.no_grad():
        for start in range(0,len(field),chunk):
            stop=min(start+chunk,len(field));flat=np.arange(start,stop)
            vox=np.column_stack(np.unravel_index(flat,shape));coords=(nib.affines.apply_affine(aff,vox)-center)/half
            valid=covered.reshape(-1)[start:stop] & (np.abs(coords)<=1).all(1)
            if not valid.any():continue
            x=torch.as_tensor(coords[valid],dtype=torch.float32,device=device)
            result=model(x,condition.expand(len(x),-1),subject_indices=torch.full((len(x),),model.subject_ids.index(subject),dtype=torch.long,device=device))
            field[start:stop][valid]=result['sdf_mm'].cpu().numpy()
    return field.reshape(shape),image,covered


def scores(pred,truth,affine):
    spacing=nib.affines.voxel_sizes(affine);tp=int((pred&truth).sum())
    return {**binary_vessel_metrics(pred,truth,affine),**vascular_metrics(pred,truth,spacing),
        'precision':tp/max(1,int(pred.sum())),'recall':tp/max(1,int(truth.sum())),
        'volume_ratio':float(pred.sum()/max(1,int(truth.sum())))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);a=p.parse_args()
    root=Path(a.study).resolve();cfg=json.loads((root/'followup.json').read_text());fine=Path(cfg['fine_study'])
    if not json.loads((root/'preflight/gate.json').read_text())['passed']:raise ValueError('Preflight gate is closed')
    out=root/'evaluation';out.mkdir(exist_ok=False);records=[]
    epochs={}
    for arm,stage in cfg['arms'].items():
        checkpoint=root/'fits'/arm/f"checkpoint_epoch_{cfg['epochs']-1}.pth"
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('format')!='vatl_sdf_atlas_rigid_v2':raise ValueError('Fresh v2 checkpoint required')
        if saved['args']['decomposition'].get('registration_enabled',True):raise ValueError('Controlled comparison requires fixed preprocessing in BOTH arms')
        model=AtlasSubjectSDF(saved['args'],saved['subject_ids'],'cuda').cuda();model.load_state_dict(saved['model']);model.eval()
        if float(model.transforms.weight.detach().abs().max())!=0:raise ValueError('Unexpected learned transform in fixed-alignment experiment')
        frame=saved['dataset_df'];epochs[arm]=saved['epoch']
        for subject in saved['subject_ids']:
            row=frame[frame.subject_id.astype(str)==subject].iloc[0].to_dict();directory=out/arm/subject;directory.mkdir(parents=True)
            field,image,covered=predict(model,row,subject)
            save(directory/'aligned_prediction_sdf_mm.nii.gz',field,image.affine)
            truth=nib.load(row['Seg']).get_fdata()==1
            aligned=scores((field<0)&covered,truth&covered,image.affine)
            transform=json.loads((fine/stage/subject/'transforms.json').read_text())
            native_centered=ants.image_read(str(fine/'audit/native_centered'/subject/'mask.nii.gz'))
            native_field=ants.apply_transforms(native_centered,ants.image_read(str(directory/'aligned_prediction_sdf_mm.nii.gz')),
                transform['inverse_image_transformlist'],whichtoinvert=transform['inversion_flags_inverse'],interpolator='linear',defaultvalue=model.shared.tau,singleprecision=True)
            original=next(s for s in json.loads((fine/'study.json').read_text())['subjects'] if s['subject_id']==subject)
            original_seg=nib.load(original['paths']['Seg']);native_truth=original_seg.get_fdata()==1
            native_pred=native_field.numpy()<0
            if native_pred.shape!=native_truth.shape:raise ValueError('Original native reference shape mismatch')
            save(directory/'native_prediction_mask.nii.gz',native_pred.astype(np.uint8),original_seg.affine)
            native=scores(native_pred,native_truth,original_seg.affine)
            records.append({'arm':arm,'subject_id':subject,'aligned_observed':aligned,'native_full_reference':native})
            print(arm,subject,'aligned Dice',aligned['Dice'],'native Dice',native['Dice'],flush=True)
        del model
        torch.cuda.empty_cache()
    fields=['Dice','precision','recall','volume_ratio','HD95_mm','ASD_mm','cldice']
    summary={}
    for arm in cfg['arms']:
        summary[arm]={}
        for region in ['aligned_observed','native_full_reference']:
            summary[arm][region]={}
            for field in fields:
                values=[r[region].get(field) for r in records if r['arm']==arm]
                values=[float(v) for v in values if v is not None and np.isfinite(v)]
                summary[arm][region][field]={'mean':float(np.mean(values)) if values else None,'count':len(values)}
    result=clean_json({'subjects':records,'summary':summary,'epochs':epochs,
        'protocol':'training reconstruction under fixed preprocessing; no held-out adaptation or image-only segmentation claim',
        'aligned_region':'each subject observed coverage; observed truth outside model crop counted',
        'native_region':'full original segmentation reference, original physical spacing/orientation; positive default SDF outside modeled output',
        'interpolation':'single final inverse level-set interpolation; inverse values are not assumed to be a true native-space SDF; compare preflight ideal-SDF and label controls',
        'model_selection':'final matched epoch; best-training checkpoints are not used as held-out validation'})
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Fresh fixed-alignment shared-SDF comparison','', '| Arm | Native Dice | Precision | Recall | Volume ratio |','|---|---:|---:|---:|---:|']
    for arm in cfg['arms']:
        s=summary[arm]['native_full_reference'];lines.append(f"| {arm} | {s['Dice']['mean']:.5f} | {s['precision']['mean']:.5f} | {s['recall']['mean']:.5f} | {s['volume_ratio']['mean']:.4f} |")
    lines+=['','These are four-subject TRAINING reconstructions, not population normative validity or held-out segmentation. Inspect learning curves and native-space errors before occupancy-model submission.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
