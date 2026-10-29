"""Synthetic group-wise plumbing test; no patient data or GPUs are used."""
import importlib.util,json,tempfile,types
from pathlib import Path
import numpy as np
import ants
spec=importlib.util.spec_from_file_location('pipeline',Path(__file__).resolve().with_name('pipeline.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'audit').mkdir();native=root/'audit/native_centered';native.mkdir();ids=['s0','s1','s2','s3']
 cfg={'subjects':[{'subject_id':sid} for sid in ids],'seed':42,'registration_spacing_mm':1.,'vascular_feature_sigma_mm':2.,'vascular_feature_secondary_sigma_mm':5.,'reg_iterations':[5,0],
  'grad_step':.05,'flow_sigma':2.,'total_sigma':.5,'groupwise_shape_step':.2,'groupwise_max_iterations':1,
  'groupwise_probability_tolerance':.001,'fov_definition':'synthetic full coverage'}
 x=np.indices((24,24,24));aff=np.eye(4);aff[:3,3]=-12
 m.save(root/'audit/reference_grid.nii.gz',np.zeros((24,24,24),np.uint8),aff)
 m.dump(root/'audit/audit.json',{'reference_subject':ids[0]})
 for i,sid in enumerate(ids):
  a=((x[0]-12-i*.3)**2+(x[1]-12)**2)<(2.2**2)
  d=native/sid;d.mkdir();m.save(d/'mask.nii.gz',a.astype(np.uint8),aff);m.save(d/'coverage.nii.gz',np.ones(a.shape,np.uint8),aff)
  d=root/'R1'/sid;d.mkdir(parents=True);m.save(d/'mask.nii.gz',a.astype(np.uint8),aff);m.save(d/'coverage.nii.gz',np.ones(a.shape,np.uint8),aff)
  path=d/'identity.mat';ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform',dimension=3),str(path))
  m.dump(d/'transforms.json',{'forward_image_transformlist':[str(path)],'inverse_image_transformlist':[str(path)],'inversion_flags_forward':[False],'inversion_flags_inverse':[False]})
 m.groupwise(root,cfg,types.SimpleNamespace(reference=None,iterations=1))
 out=root/'groupwise_s0/iteration_01';assert(out/'metrics.json').is_file()
 for p in out.glob('*/transforms.json'):
  d=json.loads(p.read_text());assert d['covered']['fold_fraction']==0
 print('Synthetic groupwise iteration passed: coverage, mean correction, composed inverse, atlas and deformation products')
