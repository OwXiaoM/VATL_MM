"""Training-only graph-gap candidates and paired negative-query intervention."""
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
import torch
from vesseltok.data import PatchSet
from vesseltok.inter_anchor_streams import SamplingStreams


def gap_candidates(tube, rng, maximum_sources=2048, neighbors_per_source=16):
    """Midpoints of nearby but geodesically distant source nodes, in world mm.

    Euclidean separation [2.6,6]mm; weighted graph distance >10mm (including
    disconnected pairs). Reject every midpoint within 1.2501mm of ANY edge.
    Candidate construction uses graph geometry only, never prediction or GT.
    """
    graph=tube.graph.copy()
    for a,b in graph.edges:graph[a][b]['length_mm']=float(np.linalg.norm(tube.points[a]-tube.points[b]))
    tree=cKDTree(tube.points);pairs=set()
    sources=rng.choice(len(tube.points),min(maximum_sources,len(tube.points)),replace=False)
    for a in sources:
        candidates=np.asarray(sorted(tree.query_ball_point(tube.points[a],6.)),int)
        distances=np.linalg.norm(tube.points[candidates]-tube.points[a],axis=1)
        candidates=candidates[distances>=2.6]
        if not len(candidates):continue
        nearby=nx.single_source_dijkstra_path_length(graph,int(a),cutoff=10.,weight='length_mm')
        candidates=np.asarray([b for b in candidates if int(b) not in nearby],int)
        if len(candidates)>neighbors_per_source:candidates=rng.choice(candidates,neighbors_per_source,replace=False)
        pairs.update(tuple(sorted((int(a),int(b)))) for b in candidates)
    pairs=np.asarray(sorted(pairs),int).reshape(-1,2)
    points=tube.points[pairs].mean(axis=1)
    keep=~tube.occupancy(points,1.2501)
    return points[keep],pairs[keep]


class GapPatchSet(PatchSet):
    """Keep prior targeted positives; alter only original >3mm background slots."""
    def __init__(self,*args,gap_pool,gap_mode,gap_replacements=64,gap_seed=28411,**kwargs):
        super().__init__(*args,**kwargs)
        if self.pool is None or self.inter_anchor_mode!='replace':raise ValueError('Gap arms require identical targeted positives')
        if gap_mode not in ('control','replace') or not 0<=gap_replacements<=self.queries_per_step//2:raise ValueError('Invalid gap setup')
        with np.load(args[0]) as data,np.load(gap_pool) as pool:
            if not np.array_equal(data['subjects'],pool['subjects']):raise ValueError('Gap pool must match training cohort')
            self.gap={k:torch.from_numpy(pool[k]) for k in ['queries','weights','eligible']}
        if self.gap['eligible'].shape!=self.data['labels'].shape or torch.any(self.gap['eligible']&(self.data['labels']>.5)):raise ValueError('Gap eligibility must contain negatives only')
        self.gap_generator=torch.Generator().manual_seed(gap_seed);self.gap_mode=gap_mode;self.gap_cap=gap_replacements

    def __getitem__(self,index):
        result=super().__getitem__(index)
        eligible=self.gap['eligible'][index,result['base_query_indices']]
        positions=torch.where(eligible)[0];count=min(self.gap_cap,len(positions)) if self.gap['weights'][index].sum()>0 else 0
        plan=torch.zeros(self.queries_per_step,dtype=torch.bool);proposal=result['queries'].clone()
        if count:
            positions=positions[torch.randperm(len(positions),generator=self.gap_generator)[:count]]
            chosen=torch.multinomial(self.gap['weights'][index],count,replacement=True,generator=self.gap_generator)
            plan[positions]=True;proposal[positions]=self.gap['queries'][index,chosen]
        if self.gap_mode=='replace':result['queries']=proposal.clone()
        result.update(gap_eligible_mask=eligible,gap_plan=plan,gap_proposal=proposal,gap_applied=torch.tensor(count if self.gap_mode=='replace' else 0))
        return result


class GapStreams(SamplingStreams):
    def __init__(self):
        super().__init__()
        self.gap_hashes={key:hashlib.sha256() for key in ['subjects','base_indices','labels','positive_queries','positive_plans','positive_proposals','gap_plans','gap_proposals','retained_queries','preserved_near_negatives']}
        self.gap_planned=[];self.gap_applied=[]

    def update(self,b):
        super().update(b)
        values={'subjects':b['subject_index'],'base_indices':b['base_query_indices'],'labels':b['labels'],
                'positive_queries':b['queries'][b['labels']>.5],'positive_plans':b['planned_replacement_mask'],'positive_proposals':b['proposed_queries'],
                'gap_plans':b['gap_plan'],'gap_proposals':b['gap_proposal'],'retained_queries':b['queries'][~b['gap_plan']],
                'preserved_near_negatives':b['queries'][(b['labels']<=.5)&~b['gap_eligible_mask']]}
        for key,value in values.items():self.gap_hashes[key].update(value.contiguous().numpy().tobytes())
        self.gap_planned.append(int(b['gap_plan'].sum()));self.gap_applied.append(int(b['gap_applied'].sum()))

    def save(self,path):
        super().save(path)
        result=json.loads(Path(path).read_text());result['gap']={'hashes':{k:v.hexdigest() for k,v in self.gap_hashes.items()},'planned_per_step':self.gap_planned,'applied_per_step':self.gap_applied}
        Path(path).write_text(json.dumps(result,indent=2))
