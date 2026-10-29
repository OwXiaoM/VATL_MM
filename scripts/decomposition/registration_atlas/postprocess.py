"""Matched-FOV comparisons and incremental vascular deformation QC."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import nibabel as nib
import ants
import compare
from pipeline import dump,jacobian_from_displacement,save

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--vascular-stage',default='R2_incremental');a=p.parse_args()
 study=Path(a.study);cfg=json.loads((study/'study.json').read_text());out=study/'comparison_registration';out.mkdir(exist_ok=False)
 rows=[];fixed=ants.image_read(str(study/'audit/reference_grid.nii.gz'))
 for s in cfg['subjects']:
  sid=s['subject_id'];folder=out/sid;folder.mkdir()
  record=json.loads((study/a.vascular_stage/sid/'transforms.json').read_text())
  # Separate-increment R2 records the incremental H5 first, followed by R1.
  increment=[record['forward_image_transformlist'][0]]
  path=ants.apply_transforms(fixed,fixed,increment,whichtoinvert=[False],compose=str(folder/'vascular_increment_'),singleprecision=True)
  field=ants.image_read(path);v=field.numpy();d=jacobian_from_displacement(v,np.asarray(field.direction)@np.diag(field.spacing))
  cov=ants.image_read(str(study/a.vascular_stage/sid/'coverage.nii.gz')).numpy()>.999
  magnitude=np.linalg.norm(v,axis=-1)
  save(folder/'incremental_jacobian.nii.gz',d,nib.load(study/'audit/reference_grid.nii.gz').affine)
  rows.append({'subject_id':sid,'vascular_increment_displacement_mean_mm':float(magnitude[cov].mean()),
   'vascular_increment_displacement_p95_mm':float(np.percentile(magnitude[cov],95)),
   'vascular_increment_jacobian_min':float(d[cov].min()),'vascular_increment_fold_fraction':float((d[cov]<=0).mean()),
   'vascular_increment_jacobian_p01':float(np.percentile(d[cov],1)),'vascular_increment_jacobian_p99':float(np.percentile(d[cov],99))})
 dump(out/'incremental_deformation.json',rows)
 sys.argv=['compare.py','--study',str(study),'--stages','R0_ants','R1',a.vascular_stage,'--output',str(out/'matched_fov')]
 compare.main()
if __name__=='__main__':main()
