"""Known physical rigid perturbation of one real vessel image, with no resampling."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import ants
from scipy.spatial.transform import Rotation
from pipeline import registration_feature,dump,as_list

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);a=p.parse_args()
 study=Path(a.study);cfg=json.loads((study/'study.json').read_text());out=study/'synthetic_image_rigid';out.mkdir(exist_ok=False)
 sid=cfg['subjects'][0]['subject_id'];path=study/'audit/native_centered'/sid/'mask.nii.gz';im=nib.load(path)
 T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([.03,-.02,.04]).as_matrix();T[:3,3]=[3.,-2.,1.]
 synthetic=nib.Nifti1Image(np.asarray(im.dataobj),T@im.affine);synthetic.header.set_xyzt_units('mm');nib.save(synthetic,out/'moving_known_rigid.nii.gz')
 fixed=registration_feature(ants.image_read(str(path)),'distance',2.)
 moving=registration_feature(ants.image_read(str(out/'moving_known_rigid.nii.gz')),'distance',2.)
 fixed=ants.resample_image(fixed,(2.,)*3,False,0);moving=ants.resample_image(moving,(2.,)*3,False,0)
 reg=ants.registration(fixed,moving,type_of_transform='Rigid',initial_transform='Identity',
   aff_metric='meansquares',aff_sampling=32,aff_random_sampling_rate=1.,
   aff_iterations=(200,100,60,20),aff_shrink_factors=(8,4,2,1),aff_smoothing_sigmas=(3,2,1,0),
   random_seed=cfg['seed'],outprefix=str(out/'rigid_'),write_composite_transform=True)
 rng=np.random.default_rng(cfg['seed']);vox=np.argwhere(np.asarray(im.dataobj)>.5);vox=vox[rng.choice(len(vox),min(512,len(vox)),False)]
 ras=nib.affines.apply_affine(im.affine,vox);expected=nib.affines.apply_affine(T,ras);lps=ras*np.array([-1,-1,1]);expected_lps=expected*np.array([-1,-1,1])
 fitted=ants.apply_transforms_to_points(3,pd.DataFrame(lps,columns=['x','y','z']),as_list(reg['fwdtransforms']),whichtoinvert=[False])
 errors=np.linalg.norm(fitted[['x','y','z']].to_numpy()-expected_lps,axis=1)
 fi=ants.image_read(str(path));mi=ants.image_read(str(out/'moving_known_rigid.nii.gz'))
 warped=ants.apply_transforms(fi,mi,as_list(reg['fwdtransforms']),interpolator='nearestNeighbor',whichtoinvert=[False])
 ants.image_write(warped,str(out/'recovered_mask.nii.gz'));b=fi.numpy()>.5;m=warped.numpy()>.5
 initial=ants.resample_image_to_target(mi,fi,interp_type='nearestNeighbor').numpy()>.5
 result={'subject_id':sid,'known_fixed_to_moving_ras_mm':T,'forward_image_transformlist':as_list(reg['fwdtransforms']),
  'inverse_image_transformlist':as_list(reg['invtransforms']),
  'point_mapping_mean_error_mm':float(errors.mean()),'point_mapping_p95_error_mm':float(np.percentile(errors,95)),
  'initial_Dice':float(2*(initial&b).sum()/max(1,initial.sum()+b.sum())),
  'recovered_Dice':float(2*(m&b).sum()/max(1,m.sum()+b.sum())),
  'note':'Same real vessel anatomy with a known rigid affine perturbation; source voxel values unchanged, no synthetic interpolation before fitting'}
 dump(out/'results.json',result);print(json.dumps(result,indent=2,default=lambda x:x.tolist()))
if __name__=='__main__':main()
