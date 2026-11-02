"""Read-only, world-mm probes of saved graph tokens and frozen implicit fields."""
import numpy as np
import torch
from vesseltok.geometry import world_to_local


class WorldDecoder:
    def __init__(self,model,records,config,device,chunk=4096):
        if config.get('representation')!='subject_tokens' or len(records)!=1:
            raise ValueError('This diagnostic supports one whole-subject token set')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Diagnostic decoder must be eval and frozen')
        self.model=model;self.config=config;self.device=device;self.chunk=chunk
        record=records[0];self.center=np.asarray(record['center_mm'])
        self.code={'tokens':torch.tensor(record['mean'],dtype=torch.float32,device=device)[None],
                   'padding':torch.tensor(record['padding'],dtype=torch.bool,device=device)[None],
                   'anchors':torch.tensor(record['anchors_local'],dtype=torch.float32,device=device)[None]}
        with torch.no_grad():self.prepared=model.prepare_decoder(self.code)

    def __call__(self,points):
        points=np.asarray(points,dtype=np.float32)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite world-mm points required')
        result=[]
        with torch.no_grad():
            for start in range(0,len(points),self.chunk):
                q=world_to_local(points[start:start+self.chunk],self.center,self.config['patch_mm']/2).astype(np.float32)
                result.append(self.model.decode(torch.from_numpy(q)[None].to(self.device),self.code,self.prepared).sigmoid()[0].cpu().numpy())
        return np.concatenate(result) if result else np.empty(0,np.float32)


def branch_samples(points,path,spacing):
    coordinates=[];tangents=[];arcs=[];coarse=[];length=0.
    for a,b in zip(path[:-1],path[1:]):
        delta=points[b]-points[a];distance=float(np.linalg.norm(delta));n=max(2,int(np.ceil(distance/.25))+1)
        # Refine each original .25-mm interval equally, retaining every coarse point.
        factor=max(1,int(np.ceil(.25/spacing)));fine=(n-1)*factor+1
        coordinates.append(np.linspace(points[a],points[b],fine));arcs.append(length+np.linspace(0,distance,fine))
        tangents.append(np.tile(delta/max(distance,1e-12),(fine,1)))
        coarse.append(np.arange(fine)%factor==0);length+=distance
    return np.concatenate(coordinates),np.concatenate(tangents),np.concatenate(arcs),np.concatenate(coarse),length


def orthogonal_frame(tangents):
    tangents=np.asarray(tangents,float);norm=np.linalg.norm(tangents,axis=1)
    if np.any(norm<=1e-12):raise ValueError('Nonzero physical tangent required')
    tangent=tangents/norm[:,None];axis=np.eye(3)[np.argmin(np.abs(tangent),axis=1)]
    u=np.cross(tangent,axis);u/=np.linalg.norm(u,axis=1)[:,None];v=np.cross(tangent,u)
    return u,v


def classify_branch(continuous,voxel,endpoint_connected,threshold=.5):
    continuous=np.asarray(continuous);voxel=np.asarray(voxel,bool)
    if voxel.all() and endpoint_connected:return 'voxel_branch_preserved'
    if np.any(continuous<threshold):return 'continuous_field_gap'
    if not voxel.all():return 'voxelization_candidate'
    return 'endpoint_connectivity_failure'


def radial_crossings(probability,radii,threshold=.5):
    """First sampled outward crossing; NaN means missing center or censored radius."""
    probability=np.asarray(probability);radii=np.asarray(radii)
    if probability.shape[-1]!=len(radii) or radii[0]!=0 or np.any(np.diff(radii)<=0):raise ValueError('Increasing radii starting at zero required')
    center=probability[...,0]>=threshold;inside=probability>=threshold
    crossing=np.full(center.shape,np.nan);censored=center & inside.all(-1)
    for i in range(1,len(radii)):
        chosen=center & np.isnan(crossing) & inside[...,i-1] & ~inside[...,i]
        fraction=(probability[...,i-1]-threshold)/np.maximum(probability[...,i-1]-probability[...,i],1e-12)
        crossing[chosen]=radii[i-1]+fraction[chosen]*(radii[i]-radii[i-1])
    return crossing,~center,censored,np.any(~inside[...,:-1]&inside[...,1:],axis=-1)


def binned(values,failed,edges):
    values=np.asarray(values);failed=np.asarray(failed,bool);rows=[]
    for low,high in zip(edges[:-1],edges[1:]):
        selected=(values>=low)&(values<high);n=int(selected.sum())
        rows.append({'lower':float(low),'upper':None if np.isinf(high) else float(high),'count':n,
                     'failure_fraction':float(failed[selected].mean()) if n else None})
    return rows
