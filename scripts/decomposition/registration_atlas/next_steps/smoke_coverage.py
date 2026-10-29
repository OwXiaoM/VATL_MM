"""Tiny CPU training+resume integration with unobserved poisoned targets."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from test_sdf_decomposition import tiny_args
from data_loading.coverage_dataset import build_cache


def main():
 with tempfile.TemporaryDirectory(prefix='vatl_coverage_smoke_') as tmp:
  root=Path(tmp);args=tiny_args();args['decomposition'].update(topology_enabled=False,registration_enabled=False,debug_evaluation={'enabled':True,'every':1,'subjects':1,'chunk':256})
  args['dataset'].update(coverage_key='Coverage',world_bbox=[30,30,30],subject_ids={'train':['one'],'val':[],'test':[]})
  args['n_subjects']['train']=1;args['epochs']['train']=3;args.update(batch_size=1,num_workers=0,n_samples=128,save_every=1,mask_reconstruction=False)
  args['dataset']['sdf_sampling'].update(num_points=256,surface_band_mm=2.)
  x=np.indices((19,19,19));mask=((x-9)**2).sum(0)<9;coverage=np.zeros(mask.shape,np.float32);coverage[1:18,1:18,1:18]=1
  sdf=(ndi.distance_transform_edt(~mask)-ndi.distance_transform_edt(mask)).astype(np.float32);sdf[coverage==0]=np.nan
  seg=mask.astype(np.float32);seg[coverage==0]=99
  row={'subject_id':'one','scan_age':25.,'sex':'F'}
  for key,array in [('MRA',mask.astype(np.float32)),('T1',np.ones(mask.shape,np.float32)),('Seg',seg),('SDF',sdf),('Coverage',coverage)]:
   path=root/(key+'.nii.gz');im=nib.Nifti1Image(array,np.eye(4));im.header.set_xyzt_units('mm');nib.save(im,path);row[key]=str(path)
  build_cache(args,row,root/'cache');row['CoverageCache']=str(root/'cache')
  metadata=root/'metadata.tsv';pd.DataFrame([row]).to_csv(metadata,sep='\t',index=False);args['dataset']['tsv_file']=str(metadata)
  config=root/'config.yaml';config.write_text(yaml.safe_dump(args))
  command=[sys.executable,str(ROOT/'train_decomposition.py'),'--config',str(config),'--device','cpu']
  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  subprocess.run([*command,'--output',str(root/'full')],cwd=ROOT,env=env,check=True)
  subprocess.run([*command,'--output',str(root/'resumed'),'--resume',str(root/'full/checkpoint_epoch_0.pth')],cwd=ROOT,env=env,check=True)
  records=[json.loads(s) for s in (root/'full/loss_components.jsonl').read_text().splitlines()]
  resumed=[json.loads(s) for s in (root/'resumed/loss_components.jsonl').read_text().splitlines()]
  assert resumed==[r for r in records if r['epoch']>0]
  history=[json.loads(s) for s in (root/'full/reconstruction_metrics.jsonl').read_text().splitlines()]
  assert all(r['subjects'][0]['evaluation_region']=='observed_full_reference_grid' for r in history)
  print('Coverage smoke passed: poisoned missing-FOV targets excluded, all-head/Eikonal training finite, observed evaluation, exact resume')
if __name__=='__main__':main()
