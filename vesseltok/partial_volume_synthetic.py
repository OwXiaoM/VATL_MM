"""Fixed synthetic target tests; no fitting or threshold tuning."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from vesseltok.geometry import TubeGraph
from vesseltok.partial_volume import voxel_fractions
from vesseltok.topology import branch_metrics


def lattice(axis,basis):
    shape=tuple(len(v) for v in axis);centers=np.stack(np.meshgrid(*axis,indexing='ij'),axis=-1).reshape(-1,3)
    affine=np.eye(4);affine[:3,:3]=basis;affine[:3,3]=centers[0]
    return centers,shape,affine


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cases=[]
    for gap in [.1,.5,1.]:
        x=1+gap/2;tube=TubeGraph([[-x,0,-1],[-x,0,1],[x,0,-1],[x,0,1]],[[0,1],[2,3]])
        centers,shape,affine=lattice([np.arange(-3,3.01,.5),np.arange(-2,2.01,.5),np.arange(-2,2.01,.5)],np.eye(3)*.5)
        hard=tube.occupancy(centers,1.).reshape(shape);f=voxel_fractions(tube,centers,np.eye(3)*.5,16)
        masks={'point':hard,'fraction_mid':(f['fraction']>=.5).reshape(shape),'certified_lower':(f['lower']>=.5).reshape(shape),'optimistic_upper':(f['upper']>=.5).reshape(shape)}
        witness=[]
        for n in [8,16,32,64]:
            q=voxel_fractions(tube,np.zeros((1,3)),np.eye(3)*.5,n);witness.append({'n':n,'fraction':float(q['fraction'][0]),'lower':float(q['lower'][0]),'upper':float(q['upper'][0])})
        metrics={name:{'components26':int(ndi.label(mask,np.ones((3,3,3)))[1]),'paths':branch_metrics(mask,tube,affine)} for name,mask in masks.items()}
        np.savez_compressed(out/('gap_'+str(gap)+'.npz'),centers=centers,shape=shape,point=hard,fraction=f['fraction'].reshape(shape),lower=f['lower'].reshape(shape),upper=f['upper'].reshape(shape))
        cases.append({'kind':'separation','radius_mm':1.,'gap_mm':gap,'cell_mm':[.5,.5,.5],'midgap_point_occupancy':False,'witness':witness,'metrics':metrics})
        print('gap',gap,metrics,witness,flush=True)
    for radius,width in [(.2,1.),(1.,5.)]:
        tube=TubeGraph([[0,0,-2],[0,0,2]],[[0,1]]);basis=np.diag([width,width,1.]);centers,shape,affine=lattice([np.arange(-1,2)*width,np.arange(-1,2)*width,np.arange(-3,4)],basis)
        hard=tube.occupancy(centers,radius).reshape(shape);f=voxel_fractions(tube,centers,basis,32,radius_mm=radius);soft=(f['fraction']>=.5).reshape(shape);optimistic=(f['upper']>=.5).reshape(shape)
        witness=[]
        for n in [8,16,32,64,128]:
            q=voxel_fractions(tube,np.zeros((1,3)),basis,n,radius_mm=radius);witness.append({'n':n,'fraction':float(q['fraction'][0]),'lower':float(q['lower'][0]),'upper':float(q['upper'][0])})
        cases.append({'kind':'small_vessel','radius_mm':radius,'cell_mm':[width,width,1.],'analytic_center_fraction':float(np.pi*radius**2/width**2),'witness':witness,
                      'point_paths':branch_metrics(hard,tube,affine),'fraction_paths':branch_metrics(soft,tube,affine),'optimistic_fraction_paths':branch_metrics(optimistic,tube,affine),'fraction_peak_upper':float(f['upper'].max())})
        np.savez_compressed(out/('small_'+str(radius)+'.npz'),centers=centers,shape=shape,point=hard,fraction=f['fraction'].reshape(shape),lower=f['lower'].reshape(shape),upper=f['upper'].reshape(shape));print('small',radius,witness[-1],flush=True)
    tube=TubeGraph([[0,0,-2],[0,0,2]],[[0,1]]);centers,shape,affine=lattice([np.arange(-1.5,1.51,.5),np.arange(-1.5,1.51,.5),np.arange(-3.5,3.51,.5)],np.eye(3)*.5)
    volume=[];exact=4*np.pi+4*np.pi/3
    for n in [8,16]:
        q=voxel_fractions(tube,centers,np.eye(3)*.5,n);estimated=float(q['fraction'].sum()*.125);volume.append({'n':n,'estimated_mm3':estimated,'exact_mm3':float(exact),'relative_error':abs(estimated-exact)/exact,'lower_mm3':float(q['lower'].sum()*.125),'upper_mm3':float(q['upper'].sum()*.125)})
    result={'cases':cases,'capsule_volume':volume,'threshold':.5,'supervision':'deterministic voxel volume fraction,not vessel-presence uncertainty','training_run':False}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(volume),flush=True)


if __name__=='__main__':main()
