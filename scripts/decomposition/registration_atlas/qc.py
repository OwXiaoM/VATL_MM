"""Native image/label overlays and identity-resampling small-vessel controls."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import dump,skeleton_stats

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);args=p.parse_args()
 study=Path(args.study);cfg=json.loads((study/'study.json').read_text());out=study/'resampling_qc';out.mkdir(exist_ok=False)
 rows=[]
 for s in cfg['subjects']:
  sid=s['subject_id'];native=study/'audit/native_centered'/sid
  original=nib.load(native/'mask.nii.gz');truth=original.get_fdata()>.5
  mra=nib.load(native/'MRA.nii.gz').get_fdata()
  registered=nib.load(study/'R0'/sid/'mask.nii.gz')
  # This deliberately measures two label interpolations, not an absence of anatomy.
  back=np.asarray(resample_from_to(registered,(original.shape,original.affine),order=0).dataobj)>.5
  spacing=nib.affines.voxel_sizes(original.affine);sk,st=skeleton_stats(truth,spacing)
  _,stback=skeleton_stats(back,spacing)
  g=registered.get_fdata()>.5
  nativevol=truth.sum()*np.prod(spacing);gridvol=g.sum()*np.prod(nib.affines.voxel_sizes(registered.affine))
  rows.append({'subject_id':sid,'native_stats':st,'identity_roundtrip_stats':stback,
    'native_vessel_volume_mm3':float(nativevol),'R0_volume_mm3':float(gridvol),'R0_over_native_volume':float(gridvol/nativevol),
    'identity_roundtrip_Dice':float(2*(truth&back).sum()/max(1,truth.sum()+back.sum())),
    'identity_roundtrip_recall':float((truth&back).sum()/truth.sum()),
    'identity_roundtrip_centerline_recall':float((sk&back).sum()/max(1,sk.sum())),
    'note':'NN resampling control; differences are interpolation/grid effects, not biological branch counts'})
  fig,axes=plt.subplots(2,3,figsize=(13,8))
  # Fixed quantiles of occupied native slices avoid selecting attractive examples.
  z=np.argwhere(truth)[:,2];slices=np.quantile(z,[.25,.5,.75]).astype(int)
  lo,hi=np.percentile(mra[np.isfinite(mra)],[10,99.8])
  for ax,k in zip(axes[0],slices):
   ax.imshow(mra[:,:,k].T,cmap='gray',origin='lower',vmin=lo,vmax=hi)
   ax.contour(truth[:,:,k].T,levels=[.5],colors=['red'],linewidths=.45)
   ax.set_title(f'Native MRA + vessel mask, slice {k}')
  for ax,axis in zip(axes[1],[2,1,0]):
   ax.imshow(np.max(mra,axis=axis).T,cmap='gray',origin='lower',vmin=lo,vmax=hi)
   overlay=np.max(truth,axis=axis).T
   ax.imshow(np.ma.masked_where(~overlay,overlay),cmap='autumn',origin='lower',vmin=0,vmax=1,alpha=.45)
   ax.set_title(f'Native MRA/vessel MIP axis {axis}')
  fig.suptitle(sid+'; native voxels, no registration');fig.tight_layout();fig.savefig(out/(sid+'.png'),dpi=160);plt.close(fig)
 dump(out/'identity_resampling.json',{'subjects':rows,'resegmentation_status':'Not run: no compatible pretrained CereVessPro model verified in this repository; no original masks overwritten'})
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
