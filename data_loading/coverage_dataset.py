"""Opt-in observed-FOV sampling for fixed-prealignment decomposition studies.

All supervised heads see only trusted observed points. Finite-difference stencils
remain inside observed support and the neural crop. No coverage channel is ever
passed as a reconstruction target. Existing Data behavior is unchanged.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage as ndi
from data_loading.dataset import Data

POOL_NAMES=('interior','surface','tissue','background')


def options_for(args):
    cfg=args['dataset'];loss=args['decomposition']['loss']
    tau=float(loss.get('trunc_mm',args['optimizer'].get('sdf_trunc_mm',5.)))
    if not np.isfinite(tau) or tau<=0 or tau!=float(args['optimizer'].get('sdf_trunc_mm',5.)):
        raise ValueError('Coverage cache and model must use identical positive SDF truncation')
    eps=float(loss.get('difference_step_mm',.25))
    if not np.isfinite(eps) or eps<=0:raise ValueError('Positive physical finite-difference step required')
    return {'world_bbox_mm':list(cfg['world_bbox']),'trunc_mm':tau,'difference_step_mm':eps,
        'threshold':float(cfg.get('coverage_threshold',.999)),
        'surface_band_mm':float(cfg.get('sdf_sampling',{}).get('surface_band_mm',2.)),
        'target_keys':list(cfg['modalities'])+[cfg['sdf_key']],
        'mask_reconstruction':bool(args.get('mask_reconstruction',False)),
        'sdf_targets':'clipped physical mm; same truncated target controls near-surface boost in both arms',
        'support_rule':'padded-EDT clearance > min(abs(raw_sdf),tau)+difference_step+half_voxel_diagonal; crop includes stencil margin'}


def trusted_support(coverage,sdf,affine,world_bbox,tau,eps,threshold=.999):
    if coverage.shape!=sdf.shape or sdf.ndim!=3:raise ValueError('Matching 3D coverage and SDF required')
    if not np.isfinite(coverage).all() or coverage.min()<0 or coverage.max()>1.00001:
        raise ValueError('Coverage must be finite in [0,1]')
    if not np.isfinite(threshold) or not 0<=threshold<1:raise ValueError('Coverage threshold must be in [0,1)')
    observed=coverage>threshold
    if not observed.any():raise ValueError('No observed FOV; refusing synthetic zero targets')
    if not np.isfinite(sdf[observed]).all():raise ValueError('Nonfinite observed SDF')
    spacing=nib.affines.voxel_sizes(affine)
    direction=affine[:3,:3]/spacing
    if not np.allclose(direction.T@direction,np.eye(3),atol=1e-4):raise ValueError('Physical EDT requires orthogonal voxel axes')
    clearance=ndi.distance_transform_edt(np.pad(observed,1),sampling=spacing)[1:-1,1:-1,1:-1].astype(np.float32)
    needed=np.minimum(np.abs(sdf),tau)+eps+.5*np.linalg.norm(spacing)
    trusted=observed & (clearance>needed)
    crop=np.zeros(sdf.size,bool);half=np.asarray(world_bbox,float)/2
    if (half<=eps).any():raise ValueError('World box smaller than stencil')
    center=nib.affines.apply_affine(affine,np.asarray(sdf.shape)/2)
    for start in range(0,sdf.size,262144):
        stop=min(start+262144,sdf.size)
        ijk=np.column_stack(np.unravel_index(np.arange(start,stop),sdf.shape))
        xyz=nib.affines.apply_affine(affine,ijk)-center
        crop[start:stop]=(np.abs(xyz)<=half-eps).all(1)
    trusted &= crop.reshape(sdf.shape)
    if not trusted.any():raise ValueError('No trusted observed points after physical/crop margins')
    return trusted,observed,crop.reshape(sdf.shape)


def draw_indices(pools,n,fractions,uniform_fraction=0.,rng=None):
    rng=np.random if rng is None else rng
    if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or n<=0:raise ValueError('Positive point count required')
    if set(fractions)!=set(POOL_NAMES):raise ValueError('Invalid pool names')
    weights=np.array([fractions[k] for k in POOL_NAMES],float)
    if not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1):raise ValueError('Pool fractions must sum to one')
    if not np.isfinite(uniform_fraction) or not 0<=uniform_fraction<=1:raise ValueError('Invalid uniform fraction')
    if not len(pools['valid']):raise ValueError('No trusted points')
    un=int(n*uniform_fraction);mixed=n-un
    weights*=np.array([len(pools[k])>0 for k in POOL_NAMES])
    pieces=[]
    if weights.sum()==0:
        pieces.append(rng.choice(pools['valid'],mixed,replace=len(pools['valid'])<mixed))
    else:
        weights/=weights.sum();counts=np.floor(weights*mixed).astype(int)
        counts[np.argsort(-(weights*mixed-counts),kind='stable')[:mixed-counts.sum()]]+=1
        for name,count in zip(POOL_NAMES,counts):
            if count:pieces.append(rng.choice(pools[name],count,replace=len(pools[name])<count))
    if un:pieces.append(rng.choice(pools['valid'],un,replace=len(pools['valid'])<un))
    selected=np.concatenate(pieces)
    return selected[rng.permutation(len(selected))]


def build_cache(args,row,output):
    """One CPU preprocessing pass; mmap arrays avoid repeated image/EDT work."""
    options=options_for(args);keys=options['target_keys'];coverage_key=args['dataset'].get('coverage_key','Coverage')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    context=SimpleNamespace(args=args,modality_keys=args['dataset']['modalities'],target_keys=keys)
    modalities=Data.load_modalities(context,row)
    reference=modalities[args['dataset']['sdf_key']]
    covim=nib.load(row[coverage_key])
    if covim.shape!=reference.shape or not np.allclose(covim.affine,reference.affine,rtol=0,atol=1e-4):raise ValueError('Coverage grid mismatch')
    for key in [*keys,coverage_key]:
        if nib.load(row[key]).header.get_xyzt_units()[0]!='mm':raise ValueError('Coverage study requires explicit mm input units')
    sdf=np.asarray(reference.dataobj,dtype=np.float32);coverage=np.asarray(covim.dataobj,dtype=np.float32)
    trusted,observed,crop=trusted_support(coverage,sdf,reference.affine,options['world_bbox_mm'],options['trunc_mm'],options['difference_step_mm'],options['threshold'])
    seg=np.asarray(modalities[args['dataset']['modalities'][-1]].dataobj)
    if ((sdf[observed]<0)!=(seg[observed]==args['dataset']['label_names'].index('Vessel'))).any():raise ValueError('SDF sign and observed vessel labels disagree')
    labels=seg[trusted]
    if (labels<0).any() or (labels>=len(args['dataset']['label_names'])).any() or not np.equal(labels,np.rint(labels)).all():raise ValueError('Invalid trusted tissue label')
    tau=options['trunc_mm'];band=options['surface_band_mm']
    if not np.isfinite(band) or not 0<band<tau:raise ValueError('Surface sampling band must be below truncation')
    masks={'valid':trusted,'interior':trusted&(sdf<0),'surface':trusted&(np.abs(sdf)<=band),
        'tissue':trusted&(sdf>band)&(seg>0),'background':trusted&(sdf>band)&(seg==0)}
    for name,mask in masks.items():
        dtype=np.uint32 if sdf.size<2**32 else np.uint64
        np.save(out/('pool_'+name+'.npy'),np.flatnonzero(mask).astype(dtype))
    arrays={}
    for key,image in modalities.items():
        array=np.asarray(image.dataobj,dtype=np.float32)
        if not np.isfinite(array[trusted]).all():raise ValueError(f'Nonfinite trusted {key} target')
        if key==args['dataset']['sdf_key']:array=np.clip(array,-tau,tau)
        arrays[key]='target_'+key+'.npy';np.save(out/arrays[key],array.reshape(-1))
    stats={'subject_id':str(row['subject_id']),'observed_voxels':int(observed.sum()),'trusted_voxels':int(trusted.sum()),
        'pool_counts':{k:int(v.sum()) for k,v in masks.items()},
        'observed_vessel_voxels':int(((seg==1)&observed).sum()),
        'trusted_vessel_fraction':float(((seg==1)&trusted).sum()/max(1,((seg==1)&observed).sum())),
        'observed_vessel_crop_fraction':float(((seg==1)&observed&crop).sum()/max(1,((seg==1)&observed).sum()))}
    fingerprints={key:{'path':str(Path(row[key]).resolve()),'bytes':Path(row[key]).stat().st_size,'mtime_ns':Path(row[key]).stat().st_mtime_ns} for key in [*keys,coverage_key]}
    manifest={'format':'vatl_coverage_cache_v1','options':options,'shape':reference.shape,'affine':reference.affine.tolist(),
        'arrays':arrays,'sources':fingerprints,'stats':stats,
        'cache_files':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in out.glob('*.npy')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return stats


class CoverageData(Data):
    def __init__(self,args,tsv_file,split,df_loaded=None):
        if args['data_augmentation'].get('activate',False):raise ValueError('Coverage cache requires disabled augmentation')
        if args.get('task_specific_sampling',{}).get('enabled',False):raise ValueError('Coverage adapter requires standard shared-point sampling')
        if args['decomposition'].get('topology_enabled',False):raise ValueError('Coverage graph policy: disable graph terms in BOTH arms until transformed graph coverage is implemented')
        if split!='train':raise ValueError('CoverageData is a training adapter; evaluation uses explicit observed-FOV masks')
        super().__init__(args,tsv_file,split,df_loaded)
        self.coverage_caches={}
        for i,(_,row) in enumerate(self.df.iterrows()):
            root=Path(row.get('CoverageCache',''));path=root/'manifest.json'
            if not path.is_file():raise ValueError('Prepare a CPU CoverageCache before training')
            manifest=json.loads(path.read_text())
            if manifest['format']!='vatl_coverage_cache_v1' or manifest['options']!=options_for(args):raise ValueError('Coverage cache/config mismatch; rebuild in a new directory')
            for key,source in manifest['sources'].items():
                path=Path(row[key])
                if str(path.resolve())!=source['path'] or path.stat().st_size!=source['bytes'] or path.stat().st_mtime_ns!=source['mtime_ns']:raise ValueError('Coverage input changed since cache preparation')
            for name,info in manifest['cache_files'].items():
                path=root/name
                if path.stat().st_size!=info['bytes'] or path.stat().st_mtime_ns!=info['mtime_ns']:raise ValueError('Prepared coverage cache changed')
            pools={name:np.load(root/('pool_'+name+'.npy'),mmap_mode='r') for name in ['valid',*POOL_NAMES]}
            arrays={key:np.load(root/name,mmap_mode='r') for key,name in manifest['arrays'].items()}
            self.coverage_caches[i]=(manifest,pools,arrays)

    def __getitem__(self,idx):
        manifest,pools,arrays=self.coverage_caches[idx]
        sampling=self.args['dataset'].get('sdf_sampling',{})
        selected=draw_indices(pools,sampling.get('num_points',32768),sampling.get('fractions',dict(zip(POOL_NAMES,[.25,.35,.25,.15]))),float(sampling.get('uniform_fraction',0.)))
        vox=np.column_stack(np.unravel_index(selected,manifest['shape']))
        aff=np.asarray(manifest['affine']);center=nib.affines.apply_affine(aff,np.asarray(manifest['shape'])/2)
        coords=(nib.affines.apply_affine(aff,vox)-center)/(self.world_bbox/2)
        values=np.stack([arrays[key][selected] for key in self.target_keys],axis=-1)
        if not np.isfinite(values).all():raise FloatingPointError('Invalid coverage-cache sample')
        conditions=self.load_conditions(self.df.iloc[idx].to_dict())[None,:].expand(len(coords),-1)
        return torch.as_tensor(coords,dtype=torch.float32),torch.as_tensor(values,dtype=torch.float32),conditions,torch.full((len(coords),1),idx,dtype=torch.int32)
