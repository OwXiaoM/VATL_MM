"""Training-only native-positive pools/CSR and untouched confirmation reservation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import yaml
from diagnostics.provenance import sha256,canonical_id,validate_splits
from vesseltok.geometry import TubeGraph,local_to_world,world_to_local,assert_grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',required=True);p.add_argument('--prepared',required=True);p.add_argument('--gap-pool-root',required=True);p.add_argument('--prediction-root',required=True);p.add_argument('--plan',required=True);p.add_argument('--confirmation-start',type=int,default=21);args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    base=Path(args.prediction_root);source=Path(args.gap_pool_root)
    protocol=json.loads((source/'pool_manifest.json').read_text());assert sha256(source/'gap_pool.npz')==protocol['gap_pool_sha256']
    config=yaml.safe_load(Path(args.config).read_text());prepared=Path(args.prepared);splits=json.loads((prepared/'splits.json').read_text())
    assert sha256(prepared/'train.npz')==protocol['source_train_sha256'];metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id')
    original=next(iter(yaml.safe_load(Path(config['manifest']).read_text()).values()))['subject_ids'];fresh=original['val'][args.confirmation_start:args.confirmation_start+3];seen=set()
    for path in base.rglob('manifest.json'):
        try:r=json.loads(path.read_text())
        except (ValueError,OSError):continue
        if isinstance(r.get('subjects'),list):seen.update(canonical_id(s) for s in r['subjects'] if isinstance(s,str))
    validate_splits(dict(splits,confirmation=fresh))
    if len(fresh)!=3 or any(canonical_id(s) in seen for s in fresh):raise ValueError('Reserved confirmation is already seen;stop before fitting')
    for sid in fresh:
        if not all(Path(metadata.loc[sid,k]).is_file() for k in ['MRA','Graph','Seg','SDF']):raise FileNotFoundError(sid)
    with np.load(prepared/'train.npz') as z:train={k:z[k] for k in z.files}
    with np.load(source/'gap_pool.npz') as z:gap={k:z[k] for k in z.files}
    assert train['subjects'].tolist()==splits['train'] and np.array_equal(train['subjects'],gap['subjects']);arrays={'subjects':train['subjects']};records=[]
    for i,sid in enumerate(splits['train']):
        row=metadata.loc[sid];image=nib.load(row['MRA']);assert_grid(image);tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm']);center=train['centers_mm'][i]
        negative=gap['queries'][i,gap['weights'][i]>0];negative_world=local_to_world(negative,center,128.)
        ijk=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T;world=nib.affines.apply_affine(image.affine,ijk)
        near=cKDTree(negative_world).query(world,distance_upper_bound=3.0001)[0]<=3.0001
        ijk=ijk[near];world=world[near];local=world_to_local(world,center,128.).astype(np.float32);recovered=local_to_world(local,center,128.)
        keep=np.all(np.abs(local)<=1,axis=1)&tube.occupancy(recovered,1.)
        local=local[keep];recovered=recovered[keep];ijk=ijk[keep]
        # Remove search-prefilter margin:final pool is truly within3mm of a gap.
        keep=cKDTree(negative_world).query(recovered)[0]<=3.
        local,recovered,ijk=local[keep],recovered[keep],ijk[keep]
        if len(local):
            error=float(np.max(np.abs(nib.affines.apply_affine(np.linalg.inv(image.affine),recovered)-ijk)))
            assert error<=1e-4 and tube.occupancy(recovered,1.).all()
        else:error=0.
        tree=cKDTree(recovered);lists=tree.query_ball_point(negative_world,3.);offsets=[0];neighbors=[];maximum=0.
        for j,items in enumerate(lists):
            ids=np.asarray(sorted(items),int)
            if len(ids):
                distances=np.linalg.norm(recovered[ids]-negative_world[j],axis=1);ids=ids[distances<=3.]
                if len(ids):maximum=max(maximum,float(np.linalg.norm(recovered[ids]-negative_world[j],axis=1).max()))
            neighbors.extend(ids.tolist());offsets.append(len(neighbors))
        arrays.update({'queries_'+str(i):local,'labels_'+str(i):np.ones(len(local),np.float32),'negative_'+str(i):negative,'offsets_'+str(i):np.asarray(offsets,np.int64),'neighbors_'+str(i):np.asarray(neighbors,np.int64)})
        records.append({'subject_id':sid,'positive_voxels':len(local),'gap_candidates':len(negative),'partnerable_gaps':int(np.count_nonzero(np.diff(offsets))),'adjacency_edges':len(neighbors),
                        'maximum_partner_distance_mm':maximum,'maximum_native_roundtrip_error_voxels':error,'graph_sha256':sha256(row['Graph'])})
        print(records[-1],flush=True)
    if not sum(r['adjacency_edges'] for r in records):raise ValueError('No positive partners;do not launch an empty intervention')
    np.savez_compressed(out/'local_positive_pool.npz',**arrays);shutil.copy2(source/'gap_pool.npz',out/'gap_pool.npz')
    protocol.update(prepared=str(prepared.resolve()),frozen_at_utc=datetime.now(timezone.utc).isoformat(),confirmation_subjects=fresh,confirmation_positions=list(range(args.confirmation_start,args.confirmation_start+3)),previously_predicted_people=sorted(seen),
                    local_positive_pool_sha256=sha256(out/'local_positive_pool.npz'),local_positive_cap=16,local_positive_seed=61103,local_positive_subjects=records,
                    parent_protocol_sha256=sha256(source/'pool_manifest.json'),plan_sha256=sha256(args.plan),
                    rule='Same16gap negatives/64inter-anchor positives;only B applies<=16other interior-positive slots to binary graph-positive native voxels within3mm of actual gap queries,private61103;A shadow only.',
                    segmentation_or_sdf_read=False,prediction_mining=False,soft_targets=False)
    (out/'pool_manifest.json').write_text(json.dumps(protocol,indent=2));print('Frozen confirmation',fresh,flush=True)


if __name__=='__main__':main()
