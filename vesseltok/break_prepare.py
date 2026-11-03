"""CPU-only frozen sampler replay and training/validation diagnostic geometry."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from diagnostics.provenance import sha256
from vesseltok.model import GraphOccupancyAE
from vesseltok.data import PatchSet
from vesseltok.gap_sampling import GapPatchSet,GapStreams,gap_candidates
from vesseltok.geometry import TubeGraph,local_to_world
from vesseltok.field_probe import branch_samples
from vesseltok.topology import source_branches
from vesseltok.break_helpers import arc_weights,neighbor_branch_distance,require_audit_scope



def fingerprint(model):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in model.state_dict().items())).hexdigest()


def replay(run,protocol,out,gap_pool):
    saved=torch.load(run/'selection/selected.pt',map_location='cpu',weights_only=False);config=saved['config'];prepared=Path(protocol['prepared'])
    assert not config['model']['variational'] and config['gap_replacements']==16
    torch.manual_seed(config['seed']);model=GraphOccupancyAE(**config['model'])
    assert fingerprint(model)==saved['provenance']['initial_model_sha256'];del model
    data=GapPatchSet(prepared/'train.npz',512,True,protocol['positive_pool'],'replace',64,9215,gap_pool=gap_pool,gap_mode='replace',gap_replacements=16,gap_seed=28411)
    validation=PatchSet(prepared/'validation.npz',512,True)
    loader=DataLoader(data,batch_size=2,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(config['seed']));iterator=iter(loader)
    state=torch.get_rng_state();torch.manual_seed(config['seed']+123)
    _=[validation[i] for i in range(len(validation))];torch.set_rng_state(state)
    streams=GapStreams();used=[[] for _ in range(len(data))]
    for step in range(2000):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        streams.update(batch)
        for row,index in enumerate(batch['subject_index'].tolist()):used[index].append(batch['queries'][row,batch['gap_plan'][row]].numpy())
    expected=json.loads((run/'sampling_streams.json').read_text())
    assert {k:v.hexdigest() for k,v in streams.gap_hashes.items()}==expected['gap']['hashes']
    assert {k:v.hexdigest() for k,v in streams.hashes.items()}==expected['hashes']
    assert streams.gap_applied==expected['gap']['applied_per_step'] and torch.equal(torch.get_rng_state(),saved['torch_rng'])
    arrays={str(i):np.concatenate(q) for i,q in enumerate(used)};np.savez_compressed(out/'realized_negative_queries_local.npz',**arrays)
    result={'all_sampler_hashes_equal':True,'final_cpu_rng_equal':True,'initial_weights_equal':True,'negative_queries':sum(len(q) for q in arrays.values()),'steps':2000,
            'replay_method':'No model forward/backward/optimizer; variational false,dropout0; exact sampler and final RNG assertions passed.'}
    (out/'replay.json').write_text(json.dumps(result,indent=2));return arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--A',required=True);parser.add_argument('--B',required=True);parser.add_argument('--gap-pool',required=True);parser.add_argument('--plan',required=True);args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    runs={'A':Path(args.A),'B':Path(args.B)};gap_pool=Path(args.gap_pool)
    protocol=json.loads((gap_pool.parent/'pool_manifest.json').read_text())
    selected={a:json.loads((r/'selection/selection.json').read_text()) for a,r in runs.items()};checkpoints={a:torch.load(r/'selection/selected.pt',map_location='cpu',weights_only=False) for a,r in runs.items()}
    assert all(s['selected_step']==2000 and not s['evaluation_used'] for s in selected.values())
    config=checkpoints['B']['config'];splits=checkpoints['B']['splits'];subjects=splits['train'][:6]+splits['validation'];require_audit_scope(subjects,splits)
    assert splits==checkpoints['A']['splits'];assert sha256(Path(protocol['prepared'])/'train.npz')==protocol['source_train_sha256']
    used=replay(runs['B'],protocol,out,gap_pool)
    with np.load(Path(protocol['prepared'])/'train.npz') as data:train={k:data[k] for k in data.files}
    with np.load(gap_pool) as p:pool={k:p[k] for k in p.files}
    metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id');records=[]
    for sid in subjects:
        split='train' if sid in splits['train'] else 'validation';index=splits[split].index(sid);dest=out/sid;dest.mkdir()
        row=metadata.loc[sid];image=nib.load(row['MRA']);tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm']);paths=source_branches(tube.graph)
        points=[];weights=[];arcs=[];coarse=[];ids=[];bounds=[];cursor=0
        for branch,path in enumerate(paths):
            p,t,arc,cm,length=branch_samples(tube.points,path,.125);points.append(p);weights.append(arc_weights(arc));arcs.append(arc);coarse.append(cm);ids.append(np.full(len(p),branch,int));bounds.append([cursor,cursor+len(p)]);cursor+=len(p)
        points=np.concatenate(points);weights=np.concatenate(weights);arcs=np.concatenate(arcs);coarse=np.concatenate(coarse);ids=np.concatenate(ids)
        voxel=nib.affines.apply_affine(np.linalg.inv(image.affine),points);rounded=np.rint(voxel).astype(int);inside=np.all((rounded>=0)&(rounded<np.asarray(image.shape)),axis=1)
        if split=='train':
            center=train['centers_mm'][index];candidates=local_to_world(pool['queries'][index,pool['weights'][index]>0],center,128.);actual=local_to_world(used[str(index)],center,128.)
            np.savez_compressed(dest/'cached_input.npz',cloud=train['cloud'][index],padding=train['padding'][index],conditions=train['conditions'][index],center=center)
        else:
            center=nib.affines.apply_affine(image.affine,np.asarray(image.shape)/2.)
            candidates,_=gap_candidates(tube,np.random.default_rng(49001+index));cv=nib.affines.apply_affine(np.linalg.inv(image.affine),candidates)
            candidates=candidates[np.all((cv>=0)&(cv<=np.asarray(image.shape)-1),axis=1)&np.all(np.abs((candidates-center)/128.)<=1,axis=1)];actual=np.empty((0,3))
        from scipy.spatial import cKDTree
        tree=cKDTree(candidates);gap_distance=tree.query(points)[0];used_distance=cKDTree(actual).query(points)[0] if len(actual) else np.full(len(points),np.nan)
        junctions=tube.points[[n for n in tube.graph if tube.graph.degree(n)>2]];junction_distance=cKDTree(junctions).query(points)[0] if len(junctions) else np.full(len(points),np.inf)
        neighbor_distance=neighbor_branch_distance(points,ids,coarse,paths)
        np.savez_compressed(dest/'geometry.npz',points_mm=points,weights_mm=weights,arc_mm=arcs,coarse=coarse,branch_id=ids,inside_fov=inside,
                            gap_candidates_mm=candidates,realized_negatives_mm=actual,gap_distance_mm=gap_distance,used_distance_mm=used_distance,junction_distance_mm=junction_distance,neighbor_distance_mm=neighbor_distance)
        record={'subject_id':sid,'split':split,'row_index':index,'branches':paths,'bounds':bounds,'MRA':row['MRA'],'Graph':row['Graph'],'graph_sha256':sha256(row['Graph']),
                'gap_geometry_role':'actual frozen training pool' if split=='train' else 'hypothetical same-rule validation candidates,never training exposure',
                'source_points':len(points),'candidate_points':len(candidates),'realized_negative_queries':len(actual),'branch_neighbor_censor_radius_mm':6.}
        (dest/'subject.json').write_text(json.dumps(record,indent=2));records.append(record);print(split,sid,len(points),'points',flush=True)
    manifest={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha256(args.plan),'subjects':records,'splits':splits,
              'runs':{a:str(r) for a,r in runs.items()},'checkpoints':{a:str(r/'selection/selected.pt') for a,r in runs.items()},
              'checkpoint_sha256':{a:sha256(r/'selection/selected.pt') for a,r in runs.items()},'selection_sha256':{a:sha256(r/'selection/selection.json') for a,r in runs.items()},
              'positive_pool':protocol['positive_pool'],'train_pool_sha256':protocol['gap_pool_sha256'],'segmentation_or_sdf_read':False,'final_evaluation_read':False,'optimizer_or_updates':False,'threshold':.5}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('CPU replay/geometry complete',out,flush=True)


if __name__=='__main__':main()
