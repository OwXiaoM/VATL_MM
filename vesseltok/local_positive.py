"""Paired binary native-voxel protection;preserve gap and inter-anchor queries."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from vesseltok.gap_sampling import GapPatchSet,GapStreams


class LocalPositiveSet(GapPatchSet):
    def __init__(self,*args,local_pool,local_mode,local_cap=16,local_seed=61103,**kwargs):
        super().__init__(*args,**kwargs)
        if self.gap_mode!='replace' or local_mode not in ('control','replace') or not 0<=local_cap<=self.queries_per_step//2:raise ValueError('Invalid local-positive control setup')
        if not torch.all((self.data['labels']==0)|(self.data['labels']==1)):raise ValueError('Only exact binary labels are permitted')
        with np.load(args[0]) as source,np.load(local_pool) as pool:
            if not np.array_equal(source['subjects'],pool['subjects']):raise ValueError('Local pool must match training subject order')
            self.local=[]
            for i in range(len(self)):
                q=pool['queries_'+str(i)];labels=pool['labels_'+str(i)];negative=pool['negative_'+str(i)];offset=pool['offsets_'+str(i)];neighbors=pool['neighbors_'+str(i)]
                if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or labels.shape!=(len(q),) or not np.all(labels==1):raise ValueError('Local native pool requires finite binary-positive queries')
                original=self.gap['queries'][i][self.gap['weights'][i]>0].numpy()
                if not np.array_equal(negative,original):raise ValueError('Local adjacency must match frozen gap queries')
                if not np.issubdtype(offset.dtype,np.integer) or not np.issubdtype(neighbors.dtype,np.integer) or offset.shape!=(len(negative)+1,) or offset[0]!=0 or offset[-1]!=len(neighbors) or np.any(np.diff(offset)<0) or np.any(neighbors<0) or np.any(neighbors>=len(q)):raise ValueError('Invalid local CSR adjacency')
                lookup={}
                for j,p in enumerate(negative):
                    key=tuple(p.tolist())
                    if key in lookup:
                        previous=lookup[key]
                        if not np.array_equal(neighbors[offset[previous]:offset[previous+1]],neighbors[offset[j]:offset[j+1]]):raise ValueError('Duplicate gap rows must share adjacency')
                    else:lookup[key]=j
                self.local.append({'queries':torch.from_numpy(q),'offsets':offset,'neighbors':neighbors,'lookup':lookup})
        self.local_mode=local_mode;self.local_cap=local_cap;self.local_generator=torch.Generator().manual_seed(local_seed)

    def __getitem__(self,index):
        result=super().__getitem__(index);pool=self.local[index]
        slots=torch.where(result['interior_eligible_mask']&~result['planned_replacement_mask']&(result['labels']>.5))[0]
        possible=[]
        for position in torch.where(result['gap_plan'])[0].tolist():
            key=tuple(result['queries'][position].tolist())
            if key not in pool['lookup']:raise ValueError('Gap proposal absent from local adjacency')
            j=pool['lookup'][key];start,stop=pool['offsets'][j:j+2]
            if stop>start:possible.append((position,int(start),int(stop)))
        count=min(self.local_cap,len(slots),len(possible));plan=torch.zeros(self.queries_per_step,dtype=torch.bool);proposal=result['queries'].clone();partners=torch.full((self.queries_per_step,),-1,dtype=torch.long)
        if count:
            slots=slots[torch.randperm(len(slots),generator=self.local_generator)[:count]]
            for slot,(negative,start,stop) in zip(slots.tolist(),possible[:count]):
                selected=start+int(torch.randint(stop-start,(1,),generator=self.local_generator));partner=int(pool['neighbors'][selected]);plan[slot]=True;proposal[slot]=pool['queries'][partner];partners[slot]=negative
        assert not torch.any(plan&(result['gap_plan']|result['planned_replacement_mask'])) and torch.all(result['labels'][plan]==1)
        if self.local_mode=='replace':result['queries']=proposal.clone()
        result.update(local_plan=plan,local_proposal=proposal,local_partner_gap_slot=partners,local_applied=torch.tensor(count if self.local_mode=='replace' else 0),
                      local_available_positive_slots=torch.tensor(len(torch.where(result['interior_eligible_mask']&~result['planned_replacement_mask'])[0])),local_partnerable_gaps=torch.tensor(len(possible)))
        return result


class LocalPositiveStreams(GapStreams):
    def __init__(self):
        super().__init__();self.local_hashes={k:hashlib.sha256() for k in ['subjects','base_indices','labels','cloud','padding','conditions','negative_queries','inter_anchor_queries','gap_queries','plans','proposals','partners','retained_queries','boundary_queries']}
        self.local_planned=[];self.local_applied=[];self.local_slots=[];self.local_partnerable=[]

    def update(self,b):
        super().update(b)
        values={'subjects':b['subject_index'],'base_indices':b['base_query_indices'],'labels':b['labels'],'cloud':b['cloud'],'padding':b['padding'],'conditions':b['conditions'],
                'negative_queries':b['queries'][b['labels']==0],'inter_anchor_queries':b['queries'][b['planned_replacement_mask']],'gap_queries':b['queries'][b['gap_plan']],
                'plans':b['local_plan'],'proposals':b['local_proposal'],'partners':b['local_partner_gap_slot'],'retained_queries':b['queries'][~b['local_plan']],
                'boundary_queries':b['queries'][~b['interior_eligible_mask']]}
        for key,value in values.items():self.local_hashes[key].update(value.contiguous().numpy().tobytes())
        self.local_planned.append(int(b['local_plan'].sum()));self.local_applied.append(int(b['local_applied'].sum()));self.local_slots.append(int(b['local_available_positive_slots'].sum()));self.local_partnerable.append(int(b['local_partnerable_gaps'].sum()))

    def save(self,path):
        super().save(path);result=json.loads(Path(path).read_text());result['local_positive']={'hashes':{k:v.hexdigest() for k,v in self.local_hashes.items()},'planned_per_step':self.local_planned,'applied_per_step':self.local_applied,'eligible_slots_per_step':self.local_slots,'partnerable_gaps_per_step':self.local_partnerable}
        Path(path).write_text(json.dumps(result,indent=2))
