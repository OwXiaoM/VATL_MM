"""Re-export fixed transforms from original observations on a NEW finer grid.

This creates data products, not a training run. The existing training loader must
first be extended to honor Coverage for every sampling/loss path.
"""
import argparse,copy,csv,json
from pathlib import Path
import numpy as np
import nibabel as nib
import ants
from pipeline import dump,save,mm_image,warp_products,atlas

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--output',required=True)
 p.add_argument('--spacing',type=float,default=.8);p.add_argument('--stages',nargs='+',default=['R0_ants','R1','R2_incremental']);args=p.parse_args()
 if args.spacing<=0:raise ValueError('Positive physical spacing required')
 source=Path(args.study).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((source/'study.json').read_text());cfg=copy.deepcopy(cfg);cfg['analysis_spacing_mm']=args.spacing;cfg['derived_from_study']=str(source)
 cfg['derivation']='Fixed-transform single resampling from original centered observations; no new registration optimization'
 dump(out/'study.json',cfg);audit=out/'audit';audit.mkdir();(audit/'native_centered').symlink_to(source/'audit/native_centered',target_is_directory=True)
 reference=nib.load(source/'audit/reference_grid.nii.gz');step=args.spacing
 shape=tuple((np.ceil((np.asarray(reference.shape)-1)*nib.affines.voxel_sizes(reference.affine)/step).astype(int)+1).tolist())
 aff=reference.affine.copy();aff[:3,:3]=np.diag([step]*3);save(audit/'reference_grid.nii.gz',np.zeros(shape,np.uint8),aff)
 info=json.loads((source/'audit/audit.json').read_text());info.update(analysis_shape=shape,analysis_affine_ras_mm=aff)
 dump(audit/'audit.json',info);labels=out/'source_labels';labels.mkdir()
 for subject in cfg['subjects']:
  im,a=mm_image(subject['paths']['Seg']);center=nib.affines.apply_affine(a,np.array(im.shape)/2);a=a.copy();a[:3,3]-=center
  save(labels/(subject['subject_id']+'.nii.gz'),np.asarray(im.dataobj,dtype=np.float32),a)
 for stage in args.stages:
  directory=out/stage;directory.mkdir();rows=[]
  for subject in cfg['subjects']:
   sid=subject['subject_id'];folder=directory/sid;folder.mkdir()
   record=json.loads((source/stage/sid/'transforms.json').read_text());fwd=record['forward_image_transformlist'];inv=record['inverse_image_transformlist']
   warp_products(out,cfg,folder,sid,fwd,inv,{'source_transform_metadata':str(source/stage/sid/'transforms.json'),'registration_optimization':'unchanged'})
   fixed=ants.image_read(str(audit/'reference_grid.nii.gz'));moving=ants.image_read(str(labels/(sid+'.nii.gz')))
   seg=ants.apply_transforms(fixed,moving,fwd,whichtoinvert=record['inversion_flags_forward'],interpolator='nearestNeighbor',singleprecision=True)
   ants.image_write(seg,str(folder/'Seg.nii.gz'))
   if np.count_nonzero((seg.numpy()==1)!=(ants.image_read(str(folder/'mask.nii.gz')).numpy()>.5)):
    raise ValueError('Warped vessel/multiclass label inconsistency; do not train on these products')
   rows.append({'subject_id':sid,'scan_age':subject['age'],'sex':subject['sex'],
    **{k:str(folder/(name+'.nii.gz')) for k,name in [('MRA','MRA'),('T1','T1'),('Seg','Seg'),('SDF','aligned_sdf_mm'),('Coverage','coverage')]}})
  with (directory/'metadata.tsv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
  atlas(directory,cfg['subjects'],info['reference_subject'])
 dump(out/'training_gate.json',{'ready_for_training':False,'requirements':['Coverage-aware sampling and losses in both controlled arms','Matched graph policy: transform graphs consistently or disable in both arms','Verify fine-grid inverse-space preservation','Use fresh v2 checkpoints, not incompatible old weights']})
if __name__=='__main__':main()
