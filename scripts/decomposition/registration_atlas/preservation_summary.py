"""Summarize native-space additions/losses after forward/inverse label resampling."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from pipeline import dump
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--stages',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 study=Path(args.study);cfg=json.loads((study/'study.json').read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 rows=[];headers=[]
 for subject in cfg['subjects']:
  sid=subject['subject_id'];im=nib.load(study/'audit/native_centered'/sid/'mask.nii.gz');truth=im.get_fdata()>.5;sk=skeletonize(truth)
  for key in ['MRA','T1','Seg','SDF']:
   if subject['exists'].get(key):
    original=nib.load(subject['paths'][key]);q,qc=original.get_qform(coded=True);s,sc=original.get_sform(coded=True)
    headers.append({'subject_id':sid,'modality':key,'qform_code':int(qc),'sform_code':int(sc),
     'qform_sform_max_abs_difference':float(np.abs(q-s).max()) if q is not None and s is not None else None})
  for stage in args.stages:
   backim=nib.load(study/stage/sid/'roundtrip_native_mask.nii.gz')
   if backim.shape!=im.shape or not np.allclose(backim.affine,im.affine,rtol=0,atol=1e-4):raise ValueError('Native round-trip grid mismatch')
   back=backim.get_fdata()>.5;tp=int((truth&back).sum());fp=int((back&~truth).sum());fn=int((truth&~back).sum())
   rows.append({'stage':stage,'subject_id':sid,'tp':tp,'interpolation_false_positive_additions':fp,'interpolation_false_negative_losses':fn,
    'Dice':2*tp/max(1,int(truth.sum()+back.sum())),'precision':tp/max(1,int(back.sum())),'recall':tp/max(1,int(truth.sum())),
    'centerline_recall':float((sk&back).sum()/max(1,sk.sum())),
    'native_components_26':int(ndi.label(truth,np.ones((3,3,3)))[1]),'roundtrip_components_26':int(ndi.label(back,np.ones((3,3,3)))[1])})
 structural_overlays(study,cfg,args.stages,out)
 dump(out/'preservation.json',{'rows':rows,'interpretation':'Added/lost native voxels after round-trip are interpolation diagnostics, not a true count of biological branch loss or segmentation false positives'})
 dump(out/'headers.json',headers);print(json.dumps(rows,indent=2))
def structural_overlays(study,cfg,stages,out):
 ref=json.loads((study/'audit/audit.json').read_text())['reference_subject']
 reference=nib.load(study/'R1'/ref/'T1.nii.gz');r=reference.get_fdata();aff=reference.affine
 def normalize(x):
  v=x[np.isfinite(x)&(x>0)];lo,hi=np.percentile(v,[1,99])
  return np.clip((x-lo)/max(1e-6,hi-lo),0,1)
 r=normalize(r);extent=[aff[0,3],aff[0,3]+(r.shape[0]-1)*aff[0,0],aff[1,3],aff[1,3]+(r.shape[1]-1)*aff[1,1]]
 for subject in cfg['subjects']:
  sid=subject['subject_id'];fig,axes=plt.subplots(len(stages),4,figsize=(16,4*len(stages)),squeeze=False)
  for row,stage in enumerate(stages):
   path=study/stage/sid/'T1.nii.gz'
   if not path.exists():continue
   moving=normalize(nib.load(path).get_fdata())
   for ax,z in zip(axes[row],[-20,-10,0,10]):
    k=int(np.clip(round((z-aff[2,3])/aff[2,2]),0,r.shape[2]-1))
    rgb=np.stack([moving[:,:,k],r[:,:,k],moving[:,:,k]],axis=-1)
    ax.imshow(rgb.transpose(1,0,2),origin='lower',extent=extent)
    ax.set_title(f'{stage}, z={z} mm');ax.set_xlabel('R mm');ax.set_ylabel('A mm')
  fig.suptitle(sid+' vs '+ref+'; T1: reference green, subject magenta, equal signal white')
  fig.tight_layout();fig.savefig(out/(sid+'_structural.png'),dpi=130);plt.close(fig)

if __name__=='__main__':main()
