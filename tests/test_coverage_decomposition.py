"""Observed support must protect every target and every physical stencil."""
import copy,json,tempfile,unittest
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from data_loading.coverage_dataset import trusted_support,draw_indices,build_cache,CoverageData,POOL_NAMES
from test_sdf_decomposition import tiny_args

class CoverageTests(unittest.TestCase):
 def test_clearance_and_stencil_on_oblique_grid(self):
  x=np.indices((19,19,19));c=np.zeros((19,19,19),np.float32);c[2:17,2:17,2:17]=1
  sdf=np.zeros_like(c);theta=.25;r=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
  aff=np.eye(4);aff[:3,:3]=r@np.diag([.8,1.,1.2]);aff[:3,3]=[-8,-9,-10]
  valid,_,_=trusted_support(c,sdf,aff,[30,30,30],5.,.25)
  self.assertTrue(valid.any());vox=np.argwhere(valid);world=nib.affines.apply_affine(aff,vox)
  for axis in range(3):
   for sign in [-1,1]:
    moved=world.copy();moved[:,axis]+=.25*sign
    index=np.rint(nib.affines.apply_affine(np.linalg.inv(aff),moved)).astype(int)
    self.assertTrue((c[tuple(index.T)]==1).all())
  # An all-ones array still has an unobserved boundary beyond its grid.
  full,_,_=trusted_support(np.ones_like(c),sdf,aff,[30,30,30],5.,.25)
  self.assertFalse(full[0].any())

 def test_pool_mixture_never_leaks_and_uniform_endpoints(self):
  pools={'valid':np.array([3,7,11]),'interior':np.array([3]),'surface':np.array([3,7]),'tissue':np.array([],int),'background':np.array([11])}
  fractions=dict(zip(POOL_NAMES,[.25,.35,.25,.15]))
  for f in [0.,.5,1.]:
   selected=draw_indices(pools,101,fractions,f,np.random.default_rng(42))
   self.assertEqual(len(selected),101);self.assertTrue(np.isin(selected,pools['valid']).all())
  with self.assertRaises(ValueError):draw_indices({**pools,'valid':np.array([],int)},4,fractions)

 def test_cached_targets_and_source_change_rejection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);args=tiny_args();args['dataset'].update(coverage_key='Coverage',coverage_threshold=.999,world_bbox=[30,30,30],subject_ids={'train':['one'],'val':[],'test':[]})
   args['n_subjects']['train']=1;args['decomposition']['topology_enabled']=False;args['mask_reconstruction']=False
   args['dataset']['sdf_sampling'].update(num_points=128,surface_band_mm=2.)
   x=np.indices((19,19,19));vessel=((x-9)**2).sum(0)<4
   from scipy.ndimage import distance_transform_edt
   sdf=(distance_transform_edt(~vessel)-distance_transform_edt(vessel)).astype(np.float32)
   coverage=np.zeros(vessel.shape,np.float32);coverage[1:18,1:18,1:18]=1
   sdf[coverage==0]=np.nan
   seg=vessel.astype(np.float32);seg[coverage==0]=99
   row={'subject_id':'one','scan_age':25.,'sex':'F'}
   for name,array in [('MRA',np.ones_like(sdf)),('T1',np.ones_like(sdf)),('Seg',seg),('SDF',sdf),('Coverage',coverage)]:
    p=root/(name+'.nii.gz');im=nib.Nifti1Image(array,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,p);row[name]=str(p)
   cache=root/'cache';stats=build_cache(args,row,cache);row['CoverageCache']=str(cache)
   data=CoverageData(args,pd.DataFrame([row]),'train');coords,values,conditions,ids=data[0]
   self.assertEqual(values.shape,(128,4));self.assertTrue(torch.isfinite(values).all());self.assertLessEqual(float(values[:,2].max()),1)
   self.assertLessEqual(float(values[:,3].abs().max()),5.)
   self.assertTrue(stats['trusted_voxels']<stats['observed_voxels'])
   # Changing the trusted-support configuration cannot reuse an old cache.
   changed=copy.deepcopy(args);changed['decomposition']['loss']['difference_step_mm']=.5
   with self.assertRaisesRegex(ValueError,'cache/config mismatch'):CoverageData(changed,pd.DataFrame([row]),'train')

 def test_evaluation_excludes_unobserved_but_counts_crop_misses(self):
  from unittest.mock import patch
  from models.decomposition_evaluation import evaluate_subject
  class Negative(torch.nn.Module):
   def __init__(self):
    super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
    self.args={'dataset':{'sdf_key':'SDF','modalities':['Seg'],'label_names':['Background','Vessel'],'world_bbox':[20,20,20],'coverage_key':'Coverage'}}
   def forward(self,x,c,**kwargs):
    field=-torch.ones(len(x));return {'sdf_mm':field,'warped_atlas_mm':field}
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cover=np.zeros((9,9,9),np.float32);cover[3:6]=1
   truth=np.zeros_like(cover);truth[4]=1;truth[:2]=1
   row={'subject_id':'constant'}
   for key,array in [('SDF',np.zeros_like(cover)),('Seg',truth),('Coverage',cover)]:
    path=root/(key+'.nii.gz');image=nib.Nifti1Image(array,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,path);row[key]=str(path)
   model=Negative()
   with patch('models.decomposition_evaluation.Data.load_conditions',return_value=torch.zeros(2)):
    result=evaluate_subject(model,row,0,chunk=43)
   self.assertAlmostEqual(result['Dice'],.5);self.assertEqual(result['truth_voxels'],81)
   # Observed truth beyond the model crop remains a false negative.
   truth[:]=0;truth[3]=1;image=nib.Nifti1Image(truth,np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,row['Seg'])
   model.args['dataset']['world_bbox']=[2,20,20]
   with patch('models.decomposition_evaluation.Data.load_conditions',return_value=torch.zeros(2)):
    result=evaluate_subject(model,row,0,chunk=43)
   self.assertEqual(result['recall'],0.);self.assertEqual(result['truth_voxels'],81)

if __name__=='__main__':unittest.main()
