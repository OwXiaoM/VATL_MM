"""Audit pairing before any device transfer; no RNG use or loss contribution."""
import hashlib
import json
from pathlib import Path


class SamplingStreams:
    def __init__(self):
        self.hashes={key:hashlib.sha256() for key in ['subjects','base_indices','labels','plans','proposals','retained_queries','negative_queries','boundary_and_background_queries']}
        self.planned=[];self.applied=[];self.positive=0;self.negative=0;self.verified=set()

    def update(self,batch):
        values={'subjects':batch['subject_index'],'base_indices':batch['base_query_indices'],'labels':batch['labels'],'plans':batch['planned_replacement_mask'],
                'proposals':batch['proposed_queries'],'retained_queries':batch['queries'][~batch['planned_replacement_mask']],
                'negative_queries':batch['queries'][batch['labels']<=.5],'boundary_and_background_queries':batch['queries'][~batch['interior_eligible_mask']]}
        for key,value in values.items():self.hashes[key].update(value.contiguous().numpy().tobytes())
        self.planned.append(int(batch['planned_replacement_mask'].sum()));self.applied.append(int(batch['replacement_count'].sum()))
        self.positive+=int((batch['labels']>.5).sum());self.negative+=int((batch['labels']<=.5).sum())

    def verify_anchors(self,encoded,batch):
        import torch
        for row,index in enumerate(batch['subject_index'].tolist()):
            if index not in self.verified:
                if not torch.equal(encoded['anchors'][row].detach().cpu(),batch['reference_anchors_local'][row]) or not torch.equal(encoded['padding'][row].detach().cpu(),batch['reference_anchor_padding'][row]):raise AssertionError('Pool FPS anchors differ from actual encoder FPS')
                self.verified.add(index)

    def save(self,path):
        Path(path).write_text(json.dumps({'hashes':{k:v.hexdigest() for k,v in self.hashes.items()},'planned_replacements_per_step':self.planned,'applied_replacements_per_step':self.applied,
                                         'positive_queries_total':self.positive,'negative_queries_total':self.negative,'encoder_anchor_verified_subject_indices':sorted(self.verified)},indent=2))
