"""CPU-only, physical-space registration and FOV-aware atlas diagnostics.

ANTs image transforms: fwdtransforms resample moving into fixed geometry.
NIfTI RAS arrays pass to ANTs through files, allowing ITK's RAS/LPS conversion.
ANTs displacement vectors are LPS mm, not normalized neural coordinates.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump(path, value):
    def cast(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=cast, allow_nan=False)+'\n')


def save(path, data, affine):
    im = nib.Nifti1Image(np.asarray(data), affine)
    im.header.set_xyzt_units('mm')
    nib.save(im, str(path))


def mm_image(path):
    im = nib.load(str(path))
    if len(im.shape) != 3: raise ValueError(f'Expected 3D image: {path}')
    units = im.header.get_xyzt_units()[0]
    if units not in ('mm', 'meter', 'micron'): raise ValueError(f'Unknown spatial units: {path}')
    scale = {'mm':1., 'meter':1000., 'micron':.001}[units]
    aff = im.affine.copy(); aff[:3] *= scale
    if not np.isfinite(aff).all() or abs(np.linalg.det(aff[:3,:3])) < 1e-10:
        raise ValueError(f'Invalid affine: {path}')
    return im, aff


def signed_distance(mask, spacing):
    return (ndi.distance_transform_edt(~mask, sampling=spacing)-ndi.distance_transform_edt(mask, sampling=spacing)).astype(np.float32)


def resample(im, geometry, order=0):
    return np.asarray(resample_from_to(im, geometry, order=order, mode='constant', cval=0).dataobj)


def fov_probability(masks, coverage):
    count = np.sum(coverage, axis=0, dtype=np.float32)
    numerator = np.sum(np.asarray(masks, np.float32)*coverage, axis=0)
    probability = np.full(count.shape, np.nan, np.float32)
    np.divide(numerator, count, out=probability, where=count>0)
    return probability, count


def skeleton_stats(mask, spacing):
    sk = skeletonize(mask)
    neighbors = ndi.convolve(sk.astype(np.uint8), np.ones((3,3,3),np.uint8),mode='constant')-sk
    return sk, {'vessel_voxels':int(mask.sum()), 'vessel_volume_mm3':float(mask.sum()*np.prod(spacing)),
                'skeleton_voxels':int(sk.sum()),'components_26':int(ndi.label(mask,np.ones((3,3,3)))[1]),
                'endpoint_voxels':int((sk & (neighbors==1)).sum()),
                'bifurcation_clusters_proxy':int(ndi.label(sk & (neighbors>2),np.ones((3,3,3)))[1])}


def pair_metrics(a, b, ca, cb, spacing):
    valid = ca & cb
    a, b = a & valid, b & valid
    sa, _ = skeleton_stats(a, spacing); sb, _ = skeleton_stats(b, spacing)
    tp = int((a & b).sum())
    precision = float((sa & b).sum()/max(1,sa.sum()))
    recall = float((sb & a).sum()/max(1,sb.sum()))
    d = None
    if sa.any() and sb.any():
        pa = np.argwhere(sa)*spacing; pb=np.argwhere(sb)*spacing
        da=cKDTree(pb).query(pa)[0]; db=cKDTree(pa).query(pb)[0]
        ds=np.r_[da,db]; d={'centerline_mean_mm':float(ds.mean()),'centerline_p95_mm':float(np.percentile(ds,95))}
    return {'Dice':2*tp/max(1,int(a.sum()+b.sum())),
            'clDice':2*precision*recall/max(1e-12,precision+recall) if sa.any() and sb.any() else None,
            'intersection_fov_voxels':int(valid.sum()),'a_vessel_voxels_in_joint_fov':int(a.sum()),
            'b_vessel_voxels_in_joint_fov':int(b.sum()),
            **(d or {'centerline_mean_mm':None,'centerline_p95_mm':None})}


def overlays(out, masks, p, aff, title):
    colors=np.array([[1,.2,.2],[.2,1,.2],[.25,.5,1],[1,.7,.1]])
    fig, axes=plt.subplots(2,3,figsize=(15,8))
    for axidx,axis in enumerate([2,1,0]):
        rgb=np.zeros((*np.max(masks[0],axis=axis).T.shape,3))
        for i,m in enumerate(masks): rgb += np.max(m,axis=axis).T[...,None]*colors[i%len(colors)]
        axes[0,axidx].imshow(np.clip(rgb,0,1),origin='lower')
        axes[0,axidx].set_title(f'Occupancy MIP, projection axis {axis}')
        axes[1,axidx].imshow(np.nanmax(np.nan_to_num(p),axis=axis).T,origin='lower',vmin=0,vmax=1,cmap='magma')
        axes[1,axidx].set_title('Probability MIP (not a topology assessment)')
    fig.suptitle(title+f'; centered physical RAS, {nib.affines.voxel_sizes(aff)[0]:g}-mm analysis voxels')
    fig.tight_layout();fig.savefig(out/'overlays.png',dpi=150);plt.close(fig)
    # Fixed physical slices shared across experiments; no case-specific best slice selection.
    zvalues=[-20.,-10.,0.,10.]
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,z in zip(axes,zvalues):
        k=int(round((z-aff[2,3])/aff[2,2]));k=np.clip(k,0,p.shape[2]-1)
        rgb=np.zeros((*masks[0][:,:,k].T.shape,3))
        for i,m in enumerate(masks): rgb+=m[:,:,k].T[...,None]*colors[i%len(colors)]
        ax.imshow(np.clip(rgb,0,1),origin='lower');ax.set_title(f'z={z:g} mm; not labeled CoW')
    fig.tight_layout();fig.savefig(out/'axial_slices.png',dpi=150);plt.close(fig)


def atlas(stage, subjects, reference=None):
    ids=[s['subject_id'] for s in subjects]
    im=nib.load(stage/ids[0]/'mask.nii.gz'); aff=im.affine; spacing=nib.affines.voxel_sizes(aff)
    masks=[nib.load(stage/sid/'mask.nii.gz').get_fdata()>.5 for sid in ids]
    covers=[nib.load(stage/sid/'coverage.nii.gz').get_fdata()>.999 for sid in ids]
    p,count=fov_probability(masks,np.asarray(covers))
    save(stage/'probability.nii.gz',p,aff);save(stage/'coverage_count.nii.gz',count,aff)
    save(stage/'contributing_vessel_count.nii.gz',np.sum([m&c for m,c in zip(masks,covers)],axis=0).astype(np.uint8),aff)
    pairs=[]
    for i,j in itertools.combinations(range(len(ids)),2):
        pairs.append({'a':ids[i],'b':ids[j],**pair_metrics(masks[i],masks[j],covers[i],covers[j],spacing)})
    allvalid=count==len(ids)
    scores={}
    for label,valid in [('any_coverage',count>0),('all_four_coverage',allvalid)]:
        v=p[valid].astype(float);total=float(v.sum())
        scores[label]={'voxels':int(valid.sum()),'expected_vessel_volume_mm3':total*float(np.prod(spacing)),
            'occupancy_concentration':float((v*v).sum()/max(total,1e-12)),
            'occupancy_disagreement':float((v*(1-v)).sum()/max(total,1e-12)),
            'probability_at_least_half_voxels':int((v>=.5).sum()),
            'entropy_per_covered_voxel':float((-v*np.log(np.clip(v,1e-8,1))-(1-v)*np.log(np.clip(1-v,1e-8,1))).mean()) if len(v) else None}
    stats=[]; sks=[]
    for sid,m,c in zip(ids,masks,covers):
        sk,st=skeleton_stats(m & c,spacing);sks.append(sk)
        save(stage/sid/'centerline.nii.gz',sk.astype(np.uint8),aff)
        stats.append({'subject_id':sid,**st})
    skp,_=fov_probability(sks,np.asarray(covers));save(stage/'centerline_probability.nii.gz',skp,aff)
    summary={'stage':stage.name,'reference':reference,'pairwise':pairs,'subject_metrics':stats,'atlas_metrics':scores,
        'mean_pairwise_Dice':float(np.mean([r['Dice'] for r in pairs])),
        'mean_pairwise_clDice':float(np.mean([r['clDice'] for r in pairs if r['clDice'] is not None])) if any(r['clDice'] is not None for r in pairs) else None,
        'valid_clDice_pairs':sum(r['clDice'] is not None for r in pairs),
        'mean_centerline_distance_mm':float(np.mean([r['centerline_mean_mm'] for r in pairs if r['centerline_mean_mm'] is not None])) if any(r['centerline_mean_mm'] is not None for r in pairs) else None,
        'valid_centerline_pairs':sum(r['centerline_mean_mm'] is not None for r in pairs),
        'coverage_note':'Only pairwise shared geometric FOV contributes to pair metrics; probability NaN where no observations',
        'topology_note':'endpoint/bifurcation proxies are grid-dependent, not homologous branch counts'}
    dump(stage/'metrics.json',summary)
    overlays(stage,masks,p,aff,stage.name)
    print(json.dumps({k:summary[k] for k in ['stage','mean_pairwise_Dice','mean_pairwise_clDice','mean_centerline_distance_mm']}),flush=True)
    return summary


def audit(study,cfg):
    out=study/'audit';out.mkdir(exist_ok=False)
    native=out/'native_centered';native.mkdir()
    records=[]; bounds=[]
    for subject in cfg['subjects']:
        sid=subject['subject_id'];folder=native/sid;folder.mkdir()
        ims={};geometries={}
        for key in ['Seg','SDF','MRA','T1']:
            if subject['exists'][key]:
                im,aff=mm_image(subject['paths'][key]);ims[key]=(im,aff)
                geometries[key]={'path':subject['paths'][key],'shape':im.shape,'affine_ras_mm':aff,
                    'orientation':nib.aff2axcodes(aff),'spacing_mm':nib.affines.voxel_sizes(aff),
                    'header_units':im.header.get_xyzt_units(),'qform_code':int(im.header['qform_code']),
                    'sform_code':int(im.header['sform_code']),
                    'orthogonality_error':float(np.abs((aff[:3,:3]/nib.affines.voxel_sizes(aff)).T@(aff[:3,:3]/nib.affines.voxel_sizes(aff))-np.eye(3)).max())}
        im,aff=ims['Seg'];seg=np.asarray(im.dataobj);mask=seg==1
        center=nib.affines.apply_affine(aff,np.asarray(im.shape)/2)
        centered=aff.copy();centered[:3,3]-=center
        corners=np.array(list(itertools.product(*[(0,s-1) for s in im.shape])))
        bounds.append(nib.affines.apply_affine(centered,corners))
        spacing=nib.affines.voxel_sizes(aff)
        if geometries['Seg']['orthogonality_error']>1e-4: raise ValueError('EDT requires orthogonal voxel axes')
        sdf=np.asarray(ims['SDF'][0].dataobj,dtype=np.float32)
        matching={k:bool(x.shape==im.shape and np.allclose(a,aff,rtol=0,atol=1e-4)) for k,(x,a) in ims.items()}
        if not all(matching.values()): raise ValueError(f'Cross-modality geometry mismatch: {sid} {matching}')
        recomputed=signed_distance(mask,spacing)
        bands={}
        for name,valid in [('inside',mask),('outside_0_5mm',(~mask)&(recomputed<=5))]:
            err=np.abs(sdf[valid]-recomputed[valid]);bands[name]={'n':int(valid.sum()),'mae_mm':float(err.mean()),'max_mm':float(err.max())}
        xyz=nib.affines.apply_affine(centered,np.argwhere(mask))
        in_crop=(np.abs(xyz)<=np.asarray(cfg['world_bbox_mm'])/2).all(-1)
        info={'subject_id':sid,'age':subject['age'],'sex':subject['sex'],'geometries':geometries,
            'matching_modalities':matching,'native_world_center_mm':center,
            'native_ras_to_centered_ras_translation_mm':-center,
            'sdf_sign_mismatch_voxels':int(((sdf<0)!=mask).sum()),'sdf_distance_audit':bands,
            'vessel_voxels':int(mask.sum()),'vessel_volume_mm3':float(mask.sum()*np.prod(spacing)),
            'vessel_model_crop_coverage':float(in_crop.mean()),
            'native_centerline_and_components':skeleton_stats(mask,spacing)[1]}
        for key in ['MRA','T1']:
            if key in ims: save(folder/(key+'.nii.gz'),np.asarray(ims[key][0].dataobj,dtype=np.float32),centered)
        save(folder/'mask.nii.gz',mask.astype(np.uint8),centered)
        save(folder/'coverage.nii.gz',np.ones(mask.shape,np.uint8),centered)
        dump(folder/'native_geometry.json',info)
        records.append(info)
        print('AUDIT',sid,'crop',info['vessel_model_crop_coverage'],'sdf',bands,flush=True)
    spacing=cfg['analysis_spacing_mm'];points=np.concatenate(bounds)
    low=np.floor((points.min(0)-cfg['padding_mm'])/spacing)*spacing
    high=np.ceil((points.max(0)+cfg['padding_mm'])/spacing)*spacing
    shape=tuple((np.rint((high-low)/spacing).astype(int)+1).tolist())
    aff=np.diag([spacing]*3+[1.]);aff[:3,3]=low
    save(out/'reference_grid.nii.gz',np.zeros(shape,np.uint8),aff)
    r0=study/'R0';r0.mkdir()
    native_world=out/'native_world_identity';native_world.mkdir()
    for subject in cfg['subjects']:
        sid=subject['subject_id'];folder=r0/sid;folder.mkdir()
        for key in ['mask','coverage','MRA','T1']:
            path=native/sid/(key+'.nii.gz')
            if path.is_file(): save(folder/(key+'.nii.gz'),resample(nib.load(path),(shape,aff),0 if key in ('mask','coverage') else 1).astype(np.float32),aff)
        # Separate header-world comparison on the same grid, preserving absolute translations.
        original,original_aff=mm_image(subject['paths']['Seg'])
        # Explicit diagnostic only: absolute header coordinates are not certified atlas space.
    summary=atlas(r0,cfg['subjects'])
    similarities={sid:np.mean([r['Dice'] for r in summary['pairwise'] if sid in (r['a'],r['b'])]) for sid in [s['subject_id'] for s in cfg['subjects']]}
    representative=max(similarities,key=similarities.get)
    dump(out/'audit.json',{'subjects':records,'analysis_shape':shape,'analysis_affine_ras_mm':aff,
        'reference_subject':representative,'reference_scores':similarities,'reference_selection':cfg['reference_selection'],
        'confirmed':'Per-subject centering retains directions/spacing but removes absolute origin; no anatomical nonlinear alignment is applied by the loader',
        'unknown':'Preprocessing transform provenance and pre-padding acquisition masks are not available in current v1 tree; header RAS orientation alone does not establish common anatomy',
        'structural_limitation':'T1 files share the MRA slab grid; do not assume whole-brain native T1 coverage',
        'fov_definition':cfg['fov_definition']})
    synthetic_rigid(out)


def synthetic_rigid(out):
    # Test the actual neural physical map, independent of image registration optimization.
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    # Snapshot path lives under study/source; use the repository explicitly if needed.
    repo=os.environ.get('SDF_REPO','/lustre/fswork/projects/rech/vnc/uil92qd/projects/VATL_SDF/VATL_MM')
    sys.path.insert(0,repo)
    import torch
    from models.sdf_decomposition import SubjectRegistration
    torch.set_num_threads(1);torch.manual_seed(42)
    coords=torch.rand(512,3)*1.2-.6
    expected=torch.tensor([[.045,-.035,.025,2.,-1.5,1.]])
    results={}
    for rotation in ['legacy_euler','rotvec']:
        reg=SubjectRegistration([240.,240.,120.],rotation)
        target=reg.subject_to_atlas(coords,expected.expand(len(coords),-1))
        fitted=torch.nn.Parameter(torch.zeros(1,6));opt=torch.optim.Adam([fitted],lr=.025)
        history=[]
        for i in range(1800):
            opt.zero_grad();pred=reg.subject_to_atlas(coords,fitted.expand(len(coords),-1))
            loss=((pred-target)*reg.half_bbox).square().mean();loss.backward();opt.step()
            if i in (0,1799): history.append(float(loss.detach()))
        pred=reg.subject_to_atlas(coords,fitted.expand(len(coords),-1))
        inv=reg.atlas_to_subject(target,expected.expand(len(coords),-1))
        results[rotation]={'target_parameters':expected.detach().numpy(),'fitted_parameters':fitted.detach().numpy(),
            'initial_and_final_mse_mm2':history,'rms_coordinate_recovery_error_mm':float(((pred-target)*reg.half_bbox).square().mean().sqrt().detach()),
            'inverse_max_error_mm':float(((inv-coords)*reg.half_bbox).abs().max()),
            'note':'Known point correspondences, not proof of image-objective convergence or clinical alignment'}
    dump(out/'synthetic_rigid.json',results)


def ants_read(path):
    import ants
    return ants.image_read(str(path), pixeltype='float')


def as_list(paths):
    return [paths] if isinstance(paths,str) else list(paths)


def registration_feature(image, kind, sigma):
    import ants
    a=image.numpy()>.5
    spacing=image.spacing
    if kind=='binary': v=a.astype(np.float32)
    elif kind=='density': v=ndi.gaussian_filter(a.astype(np.float32),np.asarray([sigma]*3)/spacing)
    else:
        distance=ndi.distance_transform_edt(~a,sampling=spacing)
        v=np.exp(-.5*(distance/sigma)**2).astype(np.float32)
    return ants.from_numpy(v,origin=image.origin,spacing=image.spacing,direction=image.direction)


def jacobian_from_displacement(displacement, index_to_world):
    """Unclamped det(I + du/dx); vector and world components must share convention."""
    inv=np.linalg.inv(np.asarray(index_to_world,float))
    jac=np.zeros(displacement.shape[:-1]+(3,3),np.float32)
    for comp in range(3):
        grad=np.stack(np.gradient(displacement[...,comp],edge_order=1),axis=-1)
        jac[...,comp,:]=np.einsum('...k,kj->...j',grad,inv)
        jac[...,comp,comp]+=1
    return np.linalg.det(jac)


def warp_products(study,cfg,folder,sid,fwd,inv,extra=None):
    import ants
    fixed=ants_read(study/'audit/reference_grid.nii.gz')
    native=study/'audit/native_centered'/sid
    original=ants_read(native/'mask.nii.gz')
    for key in ['mask','coverage','MRA','T1']:
        path=native/(key+'.nii.gz')
        if path.exists():
            interp='nearestNeighbor' if key=='mask' else 'linear'
            warped=ants.apply_transforms(fixed,ants_read(path),fwd,whichtoinvert=[False]*len(fwd),interpolator=interp,singleprecision=True)
            ants.image_write(warped,str(folder/(key+'.nii.gz')))
    maskim=ants_read(folder/'mask.nii.gz');m=maskim.numpy()>.5
    coverage=ants_read(folder/'coverage.nii.gz').numpy()>.999
    aff=nib.load(folder/'mask.nii.gz').affine
    save(folder/'aligned_sdf_mm.nii.gz',signed_distance(m,maskim.spacing),aff)
    field=ants.apply_transforms(fixed,original,fwd,whichtoinvert=[False]*len(fwd),compose=str(folder/'forward_sampling_'),singleprecision=True)
    inverse_field=ants.apply_transforms(original,fixed,inv,whichtoinvert=[False]*len(inv),compose=str(folder/'inverse_sampling_'),singleprecision=True)
    displacement=ants_read(field)
    vector=displacement.numpy()
    determinant=jacobian_from_displacement(vector,np.asarray(displacement.direction)@np.diag(displacement.spacing))
    save(folder/'jacobian_determinant.nii.gz',determinant,aff)
    magnitude=np.linalg.norm(vector,axis=-1)
    save(folder/'displacement_magnitude_mm.nii.gz',magnitude.astype(np.float32),aff)
    roundtrip=ants.apply_transforms(original,maskim,inv,whichtoinvert=[False]*len(inv),interpolator='nearestNeighbor',singleprecision=True)
    ants.image_write(roundtrip,str(folder/'roundtrip_native_mask.nii.gz'))
    a=roundtrip.numpy()>.5;b=original.numpy()>.5
    # Point mappings use LPS mm and the same image-sampling chain maps fixed -> moving.
    import pandas as pd
    rng=np.random.default_rng(cfg['seed']);eligible=np.argwhere(coverage & ndi.binary_dilation(m,iterations=2))
    chosen=eligible[rng.choice(len(eligible),min(512,len(eligible)),replace=False)]
    coords=chosen@ (np.asarray(fixed.direction)@np.diag(fixed.spacing)).T+fixed.origin
    points=pd.DataFrame(coords,columns=['x','y','z'])
    moved=ants.apply_transforms_to_points(3,points,fwd,whichtoinvert=[False]*len(fwd))
    back=ants.apply_transforms_to_points(3,moved,inv,whichtoinvert=[False]*len(inv))
    error=np.linalg.norm(back[['x','y','z']].to_numpy()-coords,axis=1)
    stats={'subject_id':sid,'forward_image_transformlist':fwd,'inverse_image_transformlist':inv,
        'inversion_flags_forward':[False]*len(fwd),'inversion_flags_inverse':[False]*len(inv),
        'forward_sampling_displacement':field,'inverse_sampling_displacement':inverse_field,
        'displacement_vector_units':'LPS physical mm, fixed-grid sampling coordinates into original centered subject',
        'roundtrip_native_Dice':float(2*(a&b).sum()/max(1,a.sum()+b.sum())),
        'roundtrip_native_vessel_recall':float((a&b).sum()/max(1,b.sum())),
        'inverse_consistency_point_mean_mm':float(error.mean()),'inverse_consistency_point_p95_mm':float(np.percentile(error,95)),
        'registered_volume_over_native_volume':float(m.sum()*np.prod(maskim.spacing)/(b.sum()*np.prod(original.spacing))),
        'roundtrip_note':'Includes two label interpolations; volume change alone is not a branch-loss count',
        'coverage_note':cfg['fov_definition'],**(extra or {})}
    for name,valid in [('covered',coverage),('vessel_neighborhood',coverage & ndi.binary_dilation(m,iterations=5))]:
        d=determinant[valid];v=magnitude[valid]
        stats[name]={'voxels':int(valid.sum()),'jacobian_min':float(d.min()),'jacobian_p01':float(np.percentile(d,1)),
            'jacobian_p50':float(np.median(d)),'jacobian_p99':float(np.percentile(d,99)),
            'fold_fraction':float((d<=0).mean()),'jacobian_below_0_5_fraction':float((d<.5).mean()),
            'jacobian_above_2_fraction':float((d>2).mean()),
            'displacement_mean_mm':float(v.mean()),'displacement_p95_mm':float(np.percentile(v,95)),'displacement_max_mm':float(v.max())}
    dump(folder/'transforms.json',stats)
    print('REGISTRATION_COMPLETE',sid,json.dumps({k:stats[k] for k in ['roundtrip_native_Dice','registered_volume_over_native_volume','covered']}),flush=True)
    return stats


def register(study,cfg,args):
    import ants
    sid=args.subject
    if sid not in [s['subject_id'] for s in cfg['subjects']]: raise ValueError('Subject outside pinned cohort')
    audit_info=json.loads((study/'audit/audit.json').read_text());ref=audit_info['reference_subject']
    stage=args.output_stage or (args.stage if args.stage=='R1' or args.feature=='distance' else args.stage+'_'+args.feature)
    root=study/stage;root.mkdir(exist_ok=True);folder=root/sid;folder.mkdir(exist_ok=False)
    native=study/'audit/native_centered';step=cfg['registration_spacing_mm']
    start=time.monotonic()
    base_forward=[];base_inverse=[]
    if args.stage=='R1':
        if not all(s['exists']['T1'] for s in cfg['subjects']):
            dump(folder/'skipped.json',{'reason':'Paired T1 unavailable'});return
        fi=ants_read(native/ref/'T1.nii.gz');mi=ants_read(native/sid/'T1.nii.gz')
        fixed=ants.resample_image(fi,(step,)*3,use_voxels=False,interp_type=0)
        moving=ants.resample_image(mi,(step,)*3,use_voxels=False,interp_type=0)
        initial='Identity';kind='SyN';extras=None
    else:
        if args.base == 'R0':
            initial='Identity'
        else:
            base=json.loads((study/args.base/sid/'transforms.json').read_text())
            initial=base['forward_image_transformlist']
            base_forward=initial
            base_inverse=base['inverse_image_transformlist']
        fi=ants_read(native/ref/'mask.nii.gz');mi=ants_read(native/sid/'mask.nii.gz')
        fixed=registration_feature(fi,args.feature,cfg['vascular_feature_sigma_mm'])
        moving=registration_feature(mi,args.feature,cfg['vascular_feature_sigma_mm'])
        fixed=ants.resample_image(fixed,(step,)*3,use_voxels=False,interp_type=0)
        moving=ants.resample_image(moving,(step,)*3,use_voxels=False,interp_type=0)
        kind='SyNOnly';extras=None
        if args.feature=='distance':
            ff=ants.resample_image(registration_feature(fi,'distance',cfg['vascular_feature_secondary_sigma_mm']),(step,)*3,False,0)
            mf=ants.resample_image(registration_feature(mi,'distance',cfg['vascular_feature_secondary_sigma_mm']),(step,)*3,False,0)
            extras=[('MeanSquares',ff,mf,.5,0)]
    fixed_coverage=ants.resample_image_to_target(ants_read(native/ref/'coverage.nii.gz'),fixed,interp_type='nearestNeighbor')
    moving_coverage=ants.resample_image_to_target(ants_read(native/sid/'coverage.nii.gz'),moving,interp_type='nearestNeighbor')
    if args.stage=='R2' and initial!='Identity':
        shared=ants.apply_transforms(fixed,moving_coverage,initial,whichtoinvert=[False]*len(initial),interpolator='nearestNeighbor')
        metric_mask=(fixed_coverage>.5)*(shared>.5)
    else:
        shared=ants.resample_image_to_target(moving_coverage,fixed,interp_type='nearestNeighbor')
        metric_mask=(fixed_coverage>.5)*(shared>.5)
    if args.stage=='R2' and base_forward:
        # ANTs cannot always serialize an inverse for a loaded nonlinear H5
        # initializer. Estimate a separate invertible increment in fixed space.
        moving=ants.apply_transforms(fixed,moving,base_forward,whichtoinvert=[False]*len(base_forward),singleprecision=True)
        moving_coverage=ants.apply_transforms(fixed,moving_coverage,base_forward,whichtoinvert=[False]*len(base_forward),singleprecision=True)
        if extras:
            extras=[(name,ff,ants.apply_transforms(ff,mf,base_forward,whichtoinvert=[False]*len(base_forward),singleprecision=True),weight,radius) for name,ff,mf,weight,radius in extras]
        initial='Identity'
    if sid==ref:
        identity=ants.create_ants_transform(transform_type='AffineTransform',dimension=3)
        ip=folder/'identity.mat';ants.write_transform(identity,str(ip));fwd=[str(ip)];inv=[str(ip)]
    else:
        result=ants.registration(fixed,moving,type_of_transform=kind,initial_transform=initial,
            outprefix=str(folder/'registration_'),write_composite_transform=True,
            reg_iterations=tuple(cfg['reg_iterations']),grad_step=cfg['grad_step'],
            flow_sigma=cfg['flow_sigma'],total_sigma=cfg['total_sigma'],
            syn_metric='CC' if args.stage=='R1' else 'meansquares',syn_sampling=2 if args.stage=='R1' else 0,
            aff_iterations=(100,60,30,0),aff_shrink_factors=(8,4,2,1),aff_smoothing_sigmas=(3,2,1,0),
            aff_metric='mattes',aff_random_sampling_rate=1.,random_seed=cfg['seed'],
            multivariate_extras=extras,mask=metric_mask,moving_mask=moving_coverage,mask_all_stages=True,
            singleprecision=True,verbose=False)
        increment_forward=as_list(result['fwdtransforms']);increment_inverse=as_list(result['invtransforms'])
        if not all(Path(path).is_file() for path in increment_forward+increment_inverse):
            raise RuntimeError('Registration did not produce BOTH forward and inverse transforms')
        fwd=increment_forward;inv=increment_inverse
        check=ants.apply_transforms(fixed,moving,fwd,whichtoinvert=[False]*len(fwd),singleprecision=True)
        error=float(np.max(np.abs(check.numpy()-result['warpedmovout'].numpy())))
        dump(folder/'composite_validation.json',{'max_absolute_error_vs_registration_warpedmovout':error,'validation_scope':'incremental warp on optimization features; original labels/intensities resampled only once with the composed chain'})
        # This is essential when R2 embeds the precomputed R1 chain.
        if error>1e-3: raise RuntimeError(f'Composite failed to reproduce registration output: {error}')
        if base_forward:
            fwd,inv=compose_image_chains(base_forward,base_inverse,increment_forward,increment_inverse)
    extra={'registration_stage':args.stage,'feature':args.feature if args.stage=='R2' else 'T1',
        'initial_alignment':args.base if args.stage=='R2' else 'current centered geometry',
        'reference_subject':ref,'antspy_version':ants.__version__,
        'registration_spacing_mm':step,'nominal_pyramid_spacings_mm':[16,8,4,2],
        'algorithm_note':'ANTs SyN baseline; not the paper ANIMAL implementation',
        'elapsed_seconds':time.monotonic()-start}
    warp_products(study,cfg,folder,sid,fwd,inv,extra)


def compose_image_chains(previous_forward, previous_inverse, increment_forward, increment_inverse):
    """Compose pull maps previous(increment(x)) in ANTs list order."""
    return list(increment_forward)+list(previous_forward), list(previous_inverse)+list(increment_inverse)


def groupwise(study,cfg,args):
    """Deformable-only group updates with a damped inverse mean sampling warp.

    This is a small-deformation, first-order mean-bias correction, not an exact
    unbiased diffeomorphic Frechet mean. Its residual bias is measured each round.
    """
    import ants
    ref=args.reference or json.loads((study/'audit/audit.json').read_text())['reference_subject']
    root=study/('groupwise_'+ref);root.mkdir(exist_ok=False)
    native=study/'audit/native_centered'
    initial=ants_read(study/'R1'/ref/'mask.nii.gz')
    template=registration_feature(initial,'distance',cfg['vascular_feature_sigma_mm'])
    template_secondary=registration_feature(initial,'distance',cfg['vascular_feature_secondary_sigma_mm'])
    template_coverage=ants_read(study/'R1'/ref/'coverage.nii.gz')
    chains={s['subject_id']:json.loads((study/'R1'/s['subject_id']/'transforms.json').read_text()) for s in cfg['subjects']}
    previous_probability=None
    for iteration in range(args.iterations or cfg['groupwise_max_iterations']):
        out=root/f'iteration_{iteration+1:02d}';out.mkdir()
        updates={};warps=[];warp_support=[]
        step=cfg['registration_spacing_mm']
        fixed=ants.resample_image(template,(step,)*3,False,0)
        fixed_secondary=ants.resample_image(template_secondary,(step,)*3,False,0)
        fixed_support=ants.resample_image_to_target(template_coverage,fixed,interp_type='nearestNeighbor')
        for subject in cfg['subjects']:
            sid=subject['subject_id'];folder=out/sid;folder.mkdir()
            # Always optimize and resample from original centered observations.
            moving=registration_feature(ants_read(native/sid/'mask.nii.gz'),'distance',cfg['vascular_feature_sigma_mm'])
            moving=ants.resample_image(moving,(step,)*3,False,0)
            # Prewarp features at optimization resolution only, so the incremental
            # warp is separately available for population mean-bias correction.
            pre=ants.apply_transforms(fixed,moving,chains[sid]['forward_image_transformlist'],
                whichtoinvert=chains[sid]['inversion_flags_forward'],singleprecision=True)
            precoverage=ants.apply_transforms(fixed,ants_read(native/sid/'coverage.nii.gz'),chains[sid]['forward_image_transformlist'],
                whichtoinvert=chains[sid]['inversion_flags_forward'],singleprecision=True)
            metric_mask=(precoverage>.999)*(fixed_support>.999)
            moving_secondary=registration_feature(ants_read(native/sid/'mask.nii.gz'),'distance',cfg['vascular_feature_secondary_sigma_mm'])
            moving_secondary=ants.resample_image(moving_secondary,(step,)*3,False,0)
            pre_secondary=ants.apply_transforms(fixed_secondary,moving_secondary,chains[sid]['forward_image_transformlist'],whichtoinvert=chains[sid]['inversion_flags_forward'],singleprecision=True)
            reg=ants.registration(fixed,pre,type_of_transform='SyNOnly',initial_transform='Identity',mask=metric_mask,moving_mask=metric_mask,mask_all_stages=True,
                outprefix=str(folder/'increment_'),reg_iterations=tuple(cfg['reg_iterations']),multivariate_extras=[('MeanSquares',fixed_secondary,pre_secondary,.5,0)],
                syn_metric='meansquares',syn_sampling=0,grad_step=cfg['grad_step'],
                flow_sigma=cfg['flow_sigma'],total_sigma=cfg['total_sigma'],random_seed=cfg['seed'])
            inc=as_list(reg['fwdtransforms']);inci=as_list(reg['invtransforms'])
            warps.append(ants.image_read(inc[0]));warp_support.append(metric_mask.numpy()>.5)
            # Image pull chains are composed in reverse coordinate application order:
            # old pull(new pull(x)); verified by the synthetic chain test.
            forward,reverse=compose_image_chains(chains[sid]['forward_image_transformlist'],chains[sid]['inverse_image_transformlist'],inc,inci)
            updates[sid]={'forward_image_transformlist':forward,'inverse_image_transformlist':reverse,
                          'inversion_flags_forward':[False]*len(forward),'inversion_flags_inverse':[False]*len(reverse)}
        support=np.sum(warp_support,axis=0).astype(np.float32)
        values=np.sum([w.numpy()*m[...,None] for w,m in zip(warps,warp_support)],axis=0)
        mean_values=np.zeros_like(values);np.divide(values,support[...,None],out=mean_values,where=support[...,None]>0)
        meanwarp=ants.from_numpy(mean_values,origin=warps[0].origin,spacing=warps[0].spacing,direction=warps[0].direction,has_components=True)
        ants.image_write(meanwarp,str(out/'mean_incremental_sampling_warp.nii.gz'))
        # Compute the actual numerical inverse of the small mean warp rather than
        # treating the negative displacement as an exact inverse.
        damped=ants.from_numpy(meanwarp.numpy()*cfg['groupwise_shape_step'],origin=meanwarp.origin,spacing=meanwarp.spacing,direction=meanwarp.direction,has_components=True)
        correction=ants.invert_displacement_field(damped,damped*0,maximum_number_of_iterations=80,mean_error_tolerance_threshold=1e-5,max_error_tolerance_threshold=.01)
        corrpath=out/'mean_bias_correction_inverse.nii.gz';ants.image_write(correction,str(corrpath))
        forwardcorr=out/'mean_bias_correction_forward.nii.gz';ants.image_write(damped,str(forwardcorr))
        # Corrected template and every mask use the same coordinate update.
        for subject in cfg['subjects']:
            sid=subject['subject_id'];u=updates[sid]
            u['forward_image_transformlist'],u['inverse_image_transformlist']=compose_image_chains(
                u['forward_image_transformlist'],u['inverse_image_transformlist'],[str(corrpath)],[str(forwardcorr)])
            u['inversion_flags_forward']=[False]*len(u['forward_image_transformlist'])
            u['inversion_flags_inverse']=[False]*len(u['inverse_image_transformlist'])
            warp_products(study,cfg,out/sid,sid,u['forward_image_transformlist'],u['inverse_image_transformlist'],{'iteration':iteration+1,'reference_initialization':ref,'mean_bias_correction':'damped inverse mean incremental sampling displacement; first-order groupwise shape update'})
        metrics=atlas(out,cfg['subjects'],ref)
        probim=ants_read(out/'probability.nii.gz');prob=probim.numpy();cov=ants_read(out/'coverage_count.nii.gz').numpy()
        if previous_probability is not None:
            valid=(cov==len(cfg['subjects'])) & np.isfinite(previous_probability) & np.isfinite(prob)
            delta=float(np.sum(np.abs(prob[valid]-previous_probability[valid]))/max(1e-12,np.sum(previous_probability[valid])))
        else:delta=None
        dump(out/'iteration.json',{'iteration':iteration+1,'probability_mass_normalized_change_all_covered':delta,
            'mean_incremental_displacement_mean_mm':float(np.linalg.norm(meanwarp.numpy(),axis=-1).mean()),
            'shape_step':cfg['groupwise_shape_step'],'convergence_note':'Probability change is coverage-matched; inspect correspondence and deformation too'})
        previous_probability=prob.copy()
        # FOV-weighted population proximity fields keep target/source feature
        # units consistent. The binary occupancy probability remains a separate output.
        valid=(cov>0).astype(np.float32)
        features=[]
        for sigma in [cfg['vascular_feature_sigma_mm'],cfg['vascular_feature_secondary_sigma_mm']]:
            numerator=np.zeros_like(prob,dtype=np.float32)
            for subject in cfg['subjects']:
                sid=subject['subject_id']
                c=ants_read(out/sid/'coverage.nii.gz').numpy()>.999
                f=registration_feature(ants_read(out/sid/'mask.nii.gz'),'distance',sigma).numpy()
                numerator+=f*c
            feature=np.zeros_like(prob);np.divide(numerator,cov,out=feature,where=cov>0)
            image=ants.from_numpy(feature,origin=probim.origin,spacing=probim.spacing,direction=probim.direction)
            ants.image_write(image,str(out/f'registration_proximity_{sigma:g}mm.nii.gz'))
            features.append(image)
        template,template_secondary=features
        template_coverage=ants.from_numpy(valid,origin=probim.origin,spacing=probim.spacing,direction=probim.direction)
        chains=updates
        if delta is not None and delta<cfg['groupwise_probability_tolerance'] and float(np.mean(np.sum(meanwarp.numpy()**2,axis=-1))**.5)<.05:break


def identity_control(study,cfg):
    import ants
    stage=study/'R0_ants';stage.mkdir(exist_ok=False)
    for subject in cfg['subjects']:
        sid=subject['subject_id'];folder=stage/sid;folder.mkdir()
        path=folder/'identity.mat'
        ants.write_transform(ants.create_ants_transform(transform_type='AffineTransform',dimension=3),str(path))
        warp_products(study,cfg,folder,sid,[str(path)],[str(path)],{'registration_stage':'R0_ants','note':'Same geometry/interpolation backend as R1/R2, no alignment optimization'})
    atlas(stage,cfg['subjects'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('audit');sub.add_parser('identity');a=sub.add_parser('register');a.add_argument('--stage',choices=['R1','R2'],required=True);a.add_argument('--subject',required=True);a.add_argument('--feature',choices=['distance','density','binary'],default='distance');a.add_argument('--base',default='R1');a.add_argument('--output-stage')
    a=sub.add_parser('atlas');a.add_argument('--stage',required=True)
    a=sub.add_parser('groupwise');a.add_argument('--iterations',type=int);a.add_argument('--reference')
    args=p.parse_args();study=Path(args.study).resolve();cfg=json.loads((study/'study.json').read_text())
    if args.command=='audit': audit(study,cfg)
    elif args.command=='identity': identity_control(study,cfg)
    elif args.command=='atlas': atlas(study/args.stage,cfg['subjects'],json.loads((study/'audit/audit.json').read_text())['reference_subject'])
    elif args.command=='register': register(study,cfg,args)
    elif args.command=='groupwise': groupwise(study,cfg,args)

if __name__=='__main__': main()
