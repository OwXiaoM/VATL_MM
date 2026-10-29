"""Quantify thin-vessel loss on finer canonical grids before choosing neural targets."""
import argparse,json
from pathlib import Path
import numpy as np
import ants
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from pipeline import dump

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);a=p.parse_args()
 study=Path(a.study);cfg=json.loads((study/'study.json').read_text());out=study/'resolution_control';out.mkdir(exist_ok=False)
 base=ants.image_read(str(study/'audit/reference_grid.nii.gz'))
 rows=[]
 for step in [1.,.8,.5]:
  shape=tuple((np.ceil((np.asarray(base.shape)-1)*np.asarray(base.spacing)/step).astype(int)+1).tolist())
  fixed=ants.from_numpy(np.zeros(shape,np.float32),origin=base.origin,spacing=(step,)*3,direction=base.direction)
  for s in cfg['subjects']:
   sid=s['subject_id'];im=ants.image_read(str(study/'audit/native_centered'/sid/'mask.nii.gz'))
   mask=im.numpy()>.5;sk=skeletonize(mask)
   warped=ants.apply_transforms(fixed,im,[],interpolator='nearestNeighbor',singleprecision=True)
   back=ants.apply_transforms(im,warped,[],interpolator='nearestNeighbor',singleprecision=True).numpy()>.5
   rows.append({'subject_id':sid,'analysis_spacing_mm':step,'analysis_shape':shape,
    'roundtrip_Dice':float(2*(mask&back).sum()/max(1,mask.sum()+back.sum())),
    'roundtrip_recall':float((mask&back).sum()/mask.sum()),
    'roundtrip_centerline_recall':float((sk&back).sum()/max(1,sk.sum())),
    'native_components_26':int(ndi.label(mask,np.ones((3,3,3)))[1]),
    'roundtrip_components_26':int(ndi.label(back,np.ones((3,3,3)))[1]),
    'volume_ratio':float(np.count_nonzero(warped.numpy()>.5)*step**3/(mask.sum()*np.prod(im.spacing)))})
   print(json.dumps(rows[-1]),flush=True)
   del warped,back,mask,sk,im
  del fixed
 dump(out/'metrics.json',{'rows':rows,'note':'Identity maps only; same ANTs backend, canonical centered geometry, nearest-neighbor labels; no biological topology conclusion'})
if __name__=='__main__':main()
