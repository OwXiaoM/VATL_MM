"""CPU-only pre-training validation of physical voxel-fraction supervision."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import yaml
from diagnostics.provenance import sha256
from vesseltok.geometry import TubeGraph,assert_grid
from vesseltok.partial_volume import voxel_fractions,circumradius,capped_distance
from vesseltok.topology import branch_metrics,source_branches
from vesseltok.partition_connections import partition_connections
from vesseltok.field_probe import branch_samples
from vesseltok.break_helpers import require_audit_scope


def refine_raster(tube,centers,basis,output,radius=1.):
    base=voxel_fractions(tube,centers,basis,8,radius_mm=radius);fraction=base['fraction'].copy();lower=base['lower'].copy();upper=base['upper'].copy();records=[]
    for n in [16,32]:
        uncertain=(lower<.5)&(upper>=.5);count=int(uncertain.sum());previous=fraction.copy()
        if count:
            refined=voxel_fractions(tube,centers[uncertain],basis,n,radius_mm=radius)
            fraction[uncertain]=refined['fraction'];lower[uncertain]=refined['lower'];upper[uncertain]=refined['upper']
        records.append({'n':n,'refined_threshold_ambiguous_cells':count,'mask_disagreements_previous_level':int(((previous>=.5)!=(fraction>=.5)).sum()),
                        'remaining_uncertified_cells':int(((lower<.5)&(upper>=.5)).sum())})
        print('refined',n,count,'cells',flush=True)
    np.savez_compressed(output/'fractions.npz',fraction=fraction,lower=lower,upper=upper,base8_fraction=base['fraction'])
    return fraction,lower,upper,records


def raster_metrics(mask,tube,affine,hard):
    native=branch_metrics(mask,tube,affine);labels=ndi.label(mask,np.ones((3,3,3)))[0];target_labels=ndi.label(hard,np.ones((3,3,3)))[0]
    voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),tube.points)).astype(int);inside=np.all((voxel>=0)&(voxel<np.asarray(mask.shape)),axis=1);v=voxel[inside]
    overlap=int((mask&hard).sum());union=int((mask|hard).sum())
    return {'source_native_paths':native,'components26':int(labels.max()),'components6':int(ndi.label(mask)[1]),'volume_voxels':int(mask.sum()),
            'volume_ratio_vs_point_target':float(mask.sum()/hard.sum()) if hard.any() else None,'dice_vs_point_target':2*overlap/max(1,int(mask.sum())+int(hard.sum())),
            'iou_vs_point_target':overlap/max(1,union),'connections_vs_point_target':partition_connections(labels[tuple(v.T)],target_labels[tuple(v.T)])}


def conservative_roi(tube,image,radius):
    h=circumradius(image.affine[:3,:3]);low=tube.points.min(0)-radius-h;high=tube.points.max(0)+radius+h
    import itertools
    corners=np.array(list(itertools.product(*zip(low,high))));voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),corners)
    begin=np.maximum(0,np.floor(voxel.min(0)).astype(int)-2);end=np.minimum(np.asarray(image.shape),np.ceil(voxel.max(0)).astype(int)+3)
    shape=end-begin
    if np.any(shape<=0):raise ValueError('Graph outside observed FOV')
    affine=image.affine.copy();affine[:3,3]=nib.affines.apply_affine(image.affine,begin)
    ijk=np.indices(shape,dtype=np.float32).reshape(3,-1).T;centers=nib.affines.apply_affine(affine,ijk)
    np.testing.assert_allclose(nib.affines.apply_affine(np.linalg.inv(image.affine),centers[:100]),ijk[:100]+begin,atol=1e-7)
    return centers,shape,affine,begin,end


def graph_case(sid,split,index,config,splits,out,plan):
    require_audit_scope([sid],splits);metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id');row=metadata.loc[sid]
    image=nib.load(row['MRA']);assert_grid(image);tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm']);basis=image.affine[:3,:3]
    centers,shape,affine,begin,end=conservative_roi(tube,image,1.);print(sid,'shape',shape,'centers',len(centers),'spacing',nib.affines.voxel_sizes(image.affine),flush=True)
    hard=tube.occupancy(centers,1.).reshape(shape);fraction,lower,upper,refinement=refine_raster(tube,centers,basis,out)
    distance=capped_distance(tube,centers,1.+circumradius(basis));boundary=np.flatnonzero((distance+circumradius(basis)>1.-1e-10)&(distance-circumradius(basis)<=1.+1e-10))
    rng=np.random.default_rng(61102+index);selected=rng.choice(boundary,min(128,len(boundary)),replace=False)
    paths=source_branches(tube.graph);parts=[];bounds=[];cursor=0
    for path in paths:
        p,_,_,_,_=branch_samples(tube.points,path,.25);parts.append(p);bounds.append([cursor,cursor+len(p)]);cursor+=len(p)
    points=np.concatenate(parts);voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),points);inside=np.all((np.rint(voxel)>=0)&(np.rint(voxel)<np.asarray(image.shape)),axis=1)
    native_path=np.rint(nib.affines.apply_affine(np.linalg.inv(affine),points[inside])).astype(int);unique=np.unique(np.ravel_multi_index(native_path.T,shape));source_selected=rng.choice(unique,min(128,len(unique)),replace=False)
    numerical=[];prev=None
    for n in [4,8,16,32]:
        value=voxel_fractions(tube,centers[selected],basis,n);source_value=voxel_fractions(tube,centers[source_selected],basis,n)
        numerical.append({'n':n,'boundary_values':value['fraction'].tolist(),'source_voxel_values':source_value['fraction'].tolist(),
                          'boundary_mean_abs_change':float(np.abs(value['fraction']-prev).mean()) if prev is not None else None,
                          'boundary_max_abs_change':float(np.abs(value['fraction']-prev).max()) if prev is not None else None,'boundary_mean_bound_width':float((value['upper']-value['lower']).mean())})
        prev=value['fraction'];print(sid,'numeric',n,flush=True)
    check=voxel_fractions(tube,points,basis,8);continuous=[]
    for start,stop in bounds:
        if not inside[start:stop].all():continue
        continuous.append({'point':True,'midpoint_all_supported':bool((check['fraction'][start:stop]>=.5).all()),'certified_all_supported':bool((check['lower'][start:stop]>=.5).all()),'optimistic_all_supported':bool((check['upper'][start:stop]>=.5).all())})
    metrics={'hard_point':raster_metrics(hard,tube,affine,hard),'fraction_midpoint':raster_metrics((fraction>=.5).reshape(shape),tube,affine,hard),
             'fraction_certified_lower':raster_metrics((lower>=.5).reshape(shape),tube,affine,hard),'fraction_optimistic_upper':raster_metrics((upper>=.5).reshape(shape),tube,affine,hard)}
    for name,field in [('point_target',hard.astype(np.uint8)),('fraction_midpoint',fraction.reshape(shape).astype(np.float32)),('fraction_lower',lower.reshape(shape).astype(np.float32)),('fraction_upper',upper.reshape(shape).astype(np.float32))]:
        target=nib.Nifti1Image(field,affine);target.header.set_xyzt_units('mm');nib.save(target,out/(name+'.nii.gz'))
    result={'subject_id':sid,'split':split,'native_shape':list(map(int,image.shape)),'native_affine':image.affine.tolist(),'voxel_sizes_mm':nib.affines.voxel_sizes(image.affine).tolist(),
            'voxel_volume_mm3':float(abs(np.linalg.det(basis))),'crop_begin':begin.tolist(),'crop_end':end.tolist(),'shape':shape.tolist(),'graph_sha256':sha256(row['Graph']),
            'metrics':metrics,'fraction_integrated_volume_mm3':float(fraction.sum()*abs(np.linalg.det(basis))),'fraction_volume_bounds_mm3':[float(lower.sum()*abs(np.linalg.det(basis))),float(upper.sum()*abs(np.linalg.det(basis)))],
            'fraction_integrated_volume_ratio_vs_point_count':float(fraction.sum()/hard.sum()),'threshold_ambiguous_cells':int(((lower<.5)&(upper>=.5)).sum()),'refinement':refinement,
            'numerical':numerical,'numeric_gate_passed':bool(numerical[-1]['boundary_mean_abs_change']<=.005 and numerical[-1]['boundary_max_abs_change']<=.03),
            'source_continuous_observed_paths':len(continuous),'source_continuous_midpoint_supported':sum(x['midpoint_all_supported'] for x in continuous),'source_continuous_certified_supported':sum(x['certified_all_supported'] for x in continuous),
            'source_continuous_optimistic_supported':sum(x['optimistic_all_supported'] for x in continuous),'source_continuous_excluded_outside_fov':len(paths)-len(continuous),
            'segmentation_or_sdf_read':False,'final_evaluation_read':False,'plan_sha256':sha256(plan),'finished_utc':datetime.now(timezone.utc).isoformat()}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(4),required=True);p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--splits',required=True);p.add_argument('--plan',required=True);args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());splits=json.loads(Path(args.splits).read_text())
    subjects=[('train',s) for s in splits['train'][:2]]+[('validation',s) for s in splits['validation']];split,sid=subjects[args.case];out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=graph_case(sid,split,args.case,config,splits,out,args.plan);print('Finished',sid,'numeric gate',result['numeric_gate_passed'],flush=True)


if __name__=='__main__':main()
