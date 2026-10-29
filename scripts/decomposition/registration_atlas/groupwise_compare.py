"""Matched-coverage groupwise comparison and measured incremental bias correction."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import ants
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import compare
from pipeline import dump

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);a=p.parse_args()
 study=Path(a.study);cfg=json.loads((study/'study.json').read_text());ref=json.loads((study/'audit/audit.json').read_text())['reference_subject']
 group=study/('groupwise_'+ref);iterations=sorted([d for d in group.glob('iteration_*') if (d/'metrics.json').is_file()])
 out=study/'comparison_groupwise';out.mkdir(exist_ok=False)
 diagnostics=[]
 for iteration in iterations:
  folder=out/iteration.name;folder.mkdir()
  mean=ants.image_read(str(iteration/'mean_incremental_sampling_warp.nii.gz'))
  # A scalar domain with the mean field's exact geometry, preserving LPS axes.
  domain=ants.from_numpy(np.zeros(mean.shape,np.float32),origin=mean.origin,spacing=mean.spacing,direction=mean.direction)
  fields=[];coverage=[]
  for s in cfg['subjects']:
   sid=s['subject_id'];record=json.loads((iteration/sid/'transforms.json').read_text())
   # Prefix is mean correction followed by the current increment. Structural and
   # previous iteration components are excluded from this incremental bias check.
   paths=record['forward_image_transformlist'][:2]
   f=ants.apply_transforms(domain,domain,paths,whichtoinvert=[False]*len(paths),compose=str(folder/(sid+'_')),singleprecision=True)
   fields.append(ants.image_read(f).numpy())
   c=ants.image_read(str(iteration/sid/'coverage.nii.gz'))
   coverage.append(ants.resample_image_to_target(c,domain,interp_type='nearestNeighbor').numpy()>.999)
  valid=ndi.binary_erosion(np.logical_and.reduce(coverage),iterations=2)
  pre=mean.numpy();post=np.mean(fields,axis=0)
  before=float(np.sqrt(np.mean(np.sum(pre[valid]**2,axis=-1))))
  after=float(np.sqrt(np.mean(np.sum(post[valid]**2,axis=-1))))
  diagnostics.append({'iteration':iteration.name,'common_eroded_fov_voxels':int(valid.sum()),
    'mean_incremental_sampling_warp_rms_before_mm':before,
    'mean_incremental_sampling_warp_rms_after_mm':after,
    'rms_ratio_after_over_before':after/before if before>1e-12 else None,
    'limitation':'Incremental first-order bias diagnostic on common observed space; does not remove structural reference bias or establish an unbiased atlas'})
 dump(out/'bias_correction.json',diagnostics)
 stages=['R0_ants','R1','R2_incremental']+[str(d.relative_to(study)) for d in iterations]
 sys.argv=['compare.py','--study',str(study),'--stages',*stages,'--output',str(out/'matched_fov')];compare.main()
 results=json.loads((out/'matched_fov/comparison.json').read_text())['stages']
 labels=['R0','R1','R2']+[f'Group {i+1}' for i in range(len(iterations))]
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 for axis,key,title in zip(axes,['Dice','clDice','centerline_mean_mm'],['Pairwise Dice','Pairwise clDice','Centerline distance (mm)']):
  values=np.array([[r[key] for r in results[stage]['pairwise']] for stage in stages],float)
  axis.plot(np.arange(len(stages)),values,color='.7',linewidth=.7,marker='.',alpha=.8)
  axis.plot(np.arange(len(stages)),values.mean(1),color='darkred',marker='o',label='Six-pair mean')
  axis.set_xticks(np.arange(len(stages)),labels);axis.set_title(title);axis.grid(alpha=.2)
 axes[0].legend();fig.suptitle('Identical coverage across stages; pairwise lines are not independent subject replicates')
 fig.tight_layout();fig.savefig(out/'correspondence.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
