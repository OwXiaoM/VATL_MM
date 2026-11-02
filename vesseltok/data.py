"""Fixed clean subject cohorts and graph-only cached patch supervision."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from diagnostics.provenance import canonical_id,validate_splits
from vesseltok.geometry import TubeGraph,world_to_local,sample_queries,assert_grid


def cohorts(config):
    manifest=yaml.safe_load(Path(config['manifest']).read_text())
    original=next(iter(manifest.values()))['subject_ids']
    validate_splits(original)
    rng=np.random.default_rng(config['seed'])
    train=sorted(rng.choice(original['train'],config['train_subjects'],replace=False).tolist())
    evaluation=original['val'][:config['evaluation_subjects']]
    validation=original['val'][config['evaluation_subjects']:config['evaluation_subjects']+config['validation_subjects']]
    splits={'train':train,'validation':validation,'evaluation':evaluation}
    validate_splits(splits)
    metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id',drop=False)
    if not metadata.index.is_unique: raise ValueError('Duplicate metadata subject IDs')
    return splits,metadata


def demographics(row):
    if row['sex'] not in ('M','F') or not np.isfinite(float(row['scan_age'])): raise ValueError('Recorded age/sex required')
    return np.array([((float(row['scan_age'])-19)/68*2-1)*.05,(-1 if row['sex']=='M' else 1)*.05],np.float32)


def padded_cloud(points,center,config):
    maximum=config['input_points']; cloud=np.zeros((maximum,3),np.float32); padding=np.ones(maximum,bool)
    cloud[:len(points)]=world_to_local(points,center,config['patch_mm']/2)
    padding[:len(points)]=False
    return cloud,padding


def prepare(config,directory):
    if config.get('input_sampling')=='nested_4096':raise ValueError('Use density_prepare to preserve paired supervision')
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    splits,metadata=cohorts(config)
    (directory/'splits.json').write_text(json.dumps(splits,indent=2))
    (directory/'config.yaml').write_text(yaml.safe_dump(config))
    records=[]
    for split in ('train','validation'):
        clouds=[]; masks=[]; queries=[]; labels=[]; conditions=[]; centers=[]; subjects=[]
        for subject_idx,subject in enumerate(splits[split]):
            row=metadata.loc[subject]; image=nib.load(row['MRA']); assert_grid(image)
            inverse=np.linalg.inv(image.affine)
            tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm'])
            rng=np.random.default_rng(config['seed']+subject_idx+10000*(split=='validation'))
            count=(1 if config.get('representation')=='subject_tokens' else (config['patches_per_subject'] if split=='train' else config['validation_patches']))
            for index in range(count):
                center=(nib.affines.apply_affine(image.affine,np.asarray(image.shape)/2.) if config.get('representation')=='subject_tokens'
                        else tube.cloud[rng.integers(len(tube.cloud))]+rng.uniform(-config['patch_mm']/4,config['patch_mm']/4,3))
                points=tube.patch_cloud(center,config['patch_mm']/2,config['halo_mm'],config['input_points'],rng)
                cloud,mask=padded_cloud(points,center,config)
                if mask.all(): raise ValueError('Empty training patch')
                qpool=[]; ypool=[]; remaining=config['query_pool']
                for _ in range(100):
                    q,y=sample_queries(tube,center,config['patch_mm']/2,config['pseudo_radius_mm'],max(remaining,1024),rng,strategy=config.get('sampling_strategy','isotropic_jitter'))
                    voxel=nib.affines.apply_affine(inverse,q)
                    keep=np.all((voxel>=0)&(voxel<=np.asarray(image.shape)-1),axis=1)
                    q,y=q[keep][:remaining],y[keep][:remaining]
                    qpool.append(q);ypool.append(y);remaining-=len(q)
                    if remaining==0: break
                if remaining: raise ValueError('Insufficient observed FOV in training patch')
                clouds.append(cloud); masks.append(mask); queries.append(world_to_local(np.concatenate(qpool),center,config['patch_mm']/2).astype(np.float32))
                labels.append(np.concatenate(ypool));conditions.append(demographics(row));centers.append(center);subjects.append(subject)
            records.append({'subject_id':subject,'split':split,'graph_nodes':len(tube.points),'graph_edges':len(tube.edges),'cloud_points':len(tube.cloud)})
            print('prepared',split,subject,'patches',count,flush=True)
        np.savez_compressed(directory/(split+'.npz'),cloud=np.stack(clouds),padding=np.stack(masks),queries=np.stack(queries),labels=np.stack(labels),
                            conditions=np.stack(conditions),centers_mm=np.stack(centers),subjects=np.array(subjects))
    (directory/'records.json').write_text(json.dumps(records,indent=2))


class PatchSet(Dataset):
    def __init__(self,path,queries_per_step,balanced=False,sampling_pool=None,inter_anchor_mode=None,replacements=64,private_seed=9215):
        data=np.load(path); self.data={key:torch.from_numpy(data[key]) for key in ['cloud','padding','queries','labels','conditions']}
        self.queries_per_step=queries_per_step;self.balanced=balanced;self.pool=None
        if inter_anchor_mode not in (None,'control','replace'):raise ValueError('Unknown inter-anchor mode')
        if inter_anchor_mode is not None:
            if sampling_pool is None or not balanced or queries_per_step%2 or not 0<=replacements<=queries_per_step//2:raise ValueError('Invalid paired replacement setup')
            with np.load(sampling_pool) as pool:
                if not np.array_equal(pool['subjects'],data['subjects']):raise ValueError('Sampling pool must match exactly the training subjects')
                self.pool={key:torch.from_numpy(pool[key]) for key in ['queries','weights','eligible','anchors','anchor_padding']}
            if self.pool['eligible'].shape!=self.data['labels'].shape or torch.any(self.pool['eligible']&(self.data['labels']<=.5)):raise ValueError('Only positive slots may be eligible')
            self.private_generator=torch.Generator().manual_seed(private_seed);self.inter_anchor_mode=inter_anchor_mode;self.replacements=replacements
        if balanced:
            self.strata=[(torch.where(row>0.5)[0],torch.where(row<=0.5)[0]) for row in self.data['labels']]
            if any(not len(positive) or not len(negative) for positive,negative in self.strata):raise ValueError('Class-balanced supervision requires both occupancy classes')
    def __len__(self): return len(self.data['cloud'])
    def __getitem__(self,index):
        if self.balanced:
            positive,negative=self.strata[index];half=self.queries_per_step//2
            selected=torch.cat((positive[torch.randint(len(positive),(half,))],negative[torch.randint(len(negative),(self.queries_per_step-half,))]))
            selected=selected[torch.randperm(len(selected))]
        else:selected=torch.randint(self.data['queries'].shape[1],(self.queries_per_step,))
        result={key:(value[index,selected] if key in ('queries','labels') else value[index]) for key,value in self.data.items()}
        if self.pool is not None:
            eligible=self.pool['eligible'][index,selected];positions=torch.where(eligible)[0]
            count=min(self.replacements,len(positions)) if self.pool['weights'][index].sum()>0 else 0
            planned=torch.zeros(self.queries_per_step,dtype=torch.bool);proposed=result['queries'].clone()
            if count:
                positions=positions[torch.randperm(len(positions),generator=self.private_generator)[:count]]
                chosen=torch.multinomial(self.pool['weights'][index],count,replacement=True,generator=self.private_generator)
                planned[positions]=True;proposed[positions]=self.pool['queries'][index,chosen]
            if self.inter_anchor_mode=='replace':result['queries']=proposed.clone()
            result.update(subject_index=torch.tensor(index),base_query_indices=selected,planned_replacement_mask=planned,proposed_queries=proposed,
                          interior_eligible_mask=eligible,replacement_count=torch.tensor(count if self.inter_anchor_mode=='replace' else 0),
                          reference_anchors_local=self.pool['anchors'][index],reference_anchor_padding=self.pool['anchor_padding'][index])
        return result
