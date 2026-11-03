"""Actual-data sampler replay and exact label/coordinate gates BEFORE training."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
from diagnostics.provenance import sha256
from vesseltok.model import GraphOccupancyAE
from vesseltok.data import PatchSet
from vesseltok.local_positive import LocalPositiveSet,LocalPositiveStreams
from vesseltok.geometry import TubeGraph,local_to_world
from vesseltok.break_prepare import fingerprint



def run(config,pool,protocol):
    torch.manual_seed(config['seed']);model=GraphOccupancyAE(**config['model']);initial=fingerprint(model);del model
    prepared=Path(protocol['prepared']);data=LocalPositiveSet(prepared/'train.npz',512,True,protocol['positive_pool'],'replace',64,9215,gap_pool=pool/'gap_pool.npz',gap_mode='replace',gap_replacements=16,gap_seed=28411,
                   local_pool=pool/'local_positive_pool.npz',local_mode=config['local_positive_mode'],local_cap=16,local_seed=61103)
    validation=PatchSet(prepared/'validation.npz',512,True);loader=DataLoader(data,batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(config['seed']));iterator=iter(loader)
    state=torch.get_rng_state();torch.manual_seed(config['seed']+123);_=[validation[i] for i in range(len(validation))];torch.set_rng_state(state)
    streams=LocalPositiveStreams();points=[[] for _ in range(len(data))];partners=[[] for _ in range(len(data))]
    for step in range(2000):
        try:b=next(iterator)
        except StopIteration:iterator=iter(loader);b=next(iterator)
        streams.update(b)
        for row,index in enumerate(b['subject_index'].tolist()):
            slots=torch.where(b['local_plan'][row])[0];negative=b['local_partner_gap_slot'][row,slots]
            points[index].append(b['local_proposal'][row,slots].numpy());partners[index].append(b['queries'][row,negative].numpy())
    return streams,torch.get_rng_state(),initial,[np.concatenate(x) for x in points],[np.concatenate(x) for x in partners]


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--reference',required=True);p.add_argument('--config-A',required=True);p.add_argument('--config-B',required=True);args=p.parse_args();reference_root=Path(args.reference);config_paths={'A':Path(args.config_A),'B':Path(args.config_B)};pool=Path(args.pool);out=pool/'preflight';out.mkdir(exist_ok=False);protocol=json.loads((pool/'pool_manifest.json').read_text())
    configs={a:yaml.safe_load(config_paths[a].read_text()) for a in ['A','B']}
    normalized={a:{k:v for k,v in c.items() if k!='local_positive_mode'} for a,c in configs.items()};assert normalized['A']==normalized['B']
    assert all(c['model']['condition_dim']==0 and not c['model']['variational'] and c['steps']==2000 and c['gap_mode']=='replace' for c in configs.values())
    outputs={a:run(c,pool,protocol) for a,c in configs.items()};reference=torch.load(reference_root/'last.pt',map_location='cpu',weights_only=False);old=json.loads((reference_root/'sampling_streams.json').read_text())
    streams={}
    for a,(stream,rng,initial,points,partners) in outputs.items():
        assert initial==reference['provenance']['initial_model_sha256'] and torch.equal(rng,reference['torch_rng'])
        stream.save(out/('streams_'+a+'.json'));streams[a]=json.loads((out/('streams_'+a+'.json')).read_text())
        assert stream.positive==stream.negative==1024000 and sum(stream.gap_applied)==64000 and sum(stream.applied)==255987
    assert streams['A']['hashes']==old['hashes'] and streams['A']['gap']['hashes']==old['gap']['hashes']
    assert streams['A']['local_positive']['hashes']==streams['B']['local_positive']['hashes']
    assert sum(streams['A']['local_positive']['applied_per_step'])==0 and streams['A']['local_positive']['planned_per_step']==streams['B']['local_positive']['applied_per_step']
    assert 0<sum(streams['B']['local_positive']['applied_per_step'])<=64000
    prepared=Path(protocol['prepared']);metadata=pd.read_csv(configs['B']['metadata'],sep='\t').set_index('subject_id')
    with np.load(prepared/'train.npz') as z:centers=z['centers_mm']
    records=[];arrays={}
    for i,sid in enumerate(protocol['train_subjects']):
        points=outputs['B'][3][i];partners=outputs['B'][4][i];world=local_to_world(points,centers[i],128.);negative=local_to_world(partners,centers[i],128.)
        tube=TubeGraph.load(metadata.loc[sid,'Graph'],.5);image=nib.load(metadata.loc[sid,'MRA'])
        assert tube.occupancy(world,1.).all() and not tube.occupancy(negative,1.).any()
        distance=np.linalg.norm(world-negative,axis=1);assert (distance<=3.).all()
        voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),world);error=float(np.abs(voxel-np.rint(voxel)).max()) if len(world) else 0.
        assert error<=1e-4 and np.all((np.rint(voxel)>=0)&(np.rint(voxel)<np.asarray(image.shape))) and np.all(np.abs(points)<=1)
        arrays['positive_'+str(i)]=points;arrays['negative_'+str(i)]=partners
        records.append({'subject_id':sid,'protected_queries':len(points),'maximum_pair_distance_mm':float(distance.max()) if len(distance) else None,'maximum_native_index_error':error})
    np.savez_compressed(out/'replayed_local_queries.npz',**arrays)
    critical=['train','model','geometry','data','gap_sampling','local_positive','inter_anchor_streams','local_positive_preflight']
    result={'passed':True,'finished_utc':datetime.now(timezone.utc).isoformat(),'steps_replayed':2000,'same_global_rng_as_reference':True,'same_initial_weights':True,
            'control_original_sampler_hashes_equal':True,'paired_local_streams_equal':True,'binary_native_positive_labels_and_strict3mm_partners_verified':True,
            'negative_queries_per_arm':64000,'inter_anchor_queries_per_arm':255987,'local_queries_A':0,'local_queries_B':sum(streams['B']['local_positive']['applied_per_step']),
            'protocol_sha256':sha256(pool/'pool_manifest.json'),'pool_sha256':sha256(pool/'local_positive_pool.npz'),'critical_source_sha256':{'vesseltok/'+n+'.py':sha256('vesseltok/'+n+'.py') for n in critical},
            'config_sha256':{a:sha256(config_paths[a]) for a in configs},'subjects':records,'optimizer_short_replay':'Required passing CPU regression:3updates bitwise equality,expected frozen variance head,no missing/duplicate optimizer parameters.'}
    (out/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
