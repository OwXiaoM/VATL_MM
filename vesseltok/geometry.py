"""Exact pseudo-radius capsules in world mm; isotropic patch coordinates only."""
import pickle
import numpy as np
import nibabel as nib
import networkx as nx
from scipy.spatial import cKDTree


def world_to_local(points, center, half_size_mm):
    if not np.isfinite(half_size_mm) or half_size_mm <= 0:
        raise ValueError('Positive isotropic physical half-size required')
    return (np.asarray(points)-np.asarray(center))/half_size_mm


def local_to_world(points, center, half_size_mm):
    return np.asarray(points)*half_size_mm+np.asarray(center)


def rigid_world(points, rotation, translation, center):
    rotation=np.asarray(rotation)
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6):
        raise ValueError('Physical rotation must be proper orthogonal')
    return (np.asarray(points)-center)@rotation.T+center+translation


class TubeGraph:
    """Union of finite edge capsules and node balls; stored radii are NOT targets.

    Edges are split into <=spacing_mm segments. Occupancy's midpoint-ball search
    is exact: any segment within radius has midpoint within radius+half_length.
    No nearest-k approximation can drop a long or crowded edge.
    """
    def __init__(self, points, edges, spacing_mm=.5):
        self.points=np.asarray(points,dtype=np.float64)
        self.edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or not len(self.points) or not np.isfinite(self.points).all():
            raise ValueError('Finite nonempty world-mm point cloud required')
        if not np.isfinite(spacing_mm) or spacing_mm<=0: raise ValueError('Positive edge discretization required')
        if self.edges.size and (self.edges.min()<0 or self.edges.max()>=len(self.points)): raise ValueError('Invalid graph endpoint')
        starts=[self.points]; ends=[self.points]  # include isolated nodes and endpoint balls
        for a,b in self.edges:
            source,target=self.points[a],self.points[b]
            count=max(1,int(np.ceil(np.linalg.norm(target-source)/spacing_mm)))
            line=np.linspace(source,target,count+1)
            starts.append(line[:-1]); ends.append(line[1:])
        self.start=np.concatenate(starts); self.end=np.concatenate(ends)
        self.delta=self.end-self.start
        self.length2=(self.delta**2).sum(1)
        self.mid=(self.start+self.end)/2
        self.half_max=float(np.sqrt(self.length2).max()/2)
        self.tree=cKDTree(self.mid)
        self.cloud=np.unique(np.concatenate((self.start,self.end)),axis=0)
        self.cloud_tree=cKDTree(self.cloud)
        graph=nx.Graph(); graph.add_nodes_from(range(len(self.points))); graph.add_edges_from(self.edges.tolist())
        self.graph=graph

    @classmethod
    def load(cls,path,spacing_mm=.5):
        # Only trusted existing project graph pickles are accepted.
        with open(path,'rb') as stream: graph=pickle.load(stream)
        graph=graph.get('graph') if isinstance(graph,dict) else graph
        if not isinstance(graph,nx.Graph): raise TypeError('Expected a NetworkX graph')
        node_ids=list(graph.nodes); lookup={node:i for i,node in enumerate(node_ids)}
        points=np.asarray([graph.nodes[node]['pos'] for node in node_ids],float)
        # Project graphs have straight dense edges and weight only; reject hidden
        # polyline geometry instead of silently replacing curves by endpoint chords.
        for _,_,data in graph.edges(data=True):
            if any(key in data for key in ('pts','points','polyline','path')):
                raise ValueError('Explicit edge polylines require a verified world-mm adapter')
        edges=[(lookup[a],lookup[b]) for a,b in graph.edges]
        return cls(points,edges,spacing_mm)

    def occupancy(self,points,radius_mm,chunk=32768):
        points=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(points).all() or not np.isfinite(radius_mm) or radius_mm<=0: raise ValueError('Finite queries and positive pseudo-radius required')
        result=np.zeros(len(points),bool)
        bound=radius_mm+self.half_max+1e-9
        for begin in range(0,len(points),chunk):
            query=points[begin:begin+chunk]
            possible=np.flatnonzero(self.tree.query(query)[0]<=bound)
            if not len(possible): continue
            neighbors=self.tree.query_ball_point(query[possible],bound)
            sizes=np.fromiter((len(items) for items in neighbors),dtype=np.int64)
            qi=np.repeat(np.arange(len(possible)),sizes)
            si=np.concatenate(neighbors).astype(np.int64)
            displacement=query[possible][qi]-self.start[si]
            fraction=np.divide((displacement*self.delta[si]).sum(1),self.length2[si],out=np.zeros(len(si)),where=self.length2[si]>0)
            fraction=np.clip(fraction,0,1)
            d2=((displacement-fraction[:,None]*self.delta[si])**2).sum(1)
            minimum=np.full(len(possible),np.inf)
            np.minimum.at(minimum,qi,d2)
            result[begin+possible]=minimum<=radius_mm**2+1e-12
        return result

    def patch_cloud(self,center,half_size_mm,halo_mm,max_points,rng):
        candidates=self.cloud_tree.query_ball_point(center,np.sqrt(3)*(half_size_mm+halo_mm))
        points=self.cloud[np.asarray(candidates,dtype=int)]
        points=points[np.all(np.abs(points-center)<=half_size_mm+halo_mm,axis=1)]
        if len(points)>max_points: points=points[rng.choice(len(points),max_points,replace=False)]
        # Canonical input ordering makes deterministic FPS tie-breaking reproducible.
        if len(points): points=points[np.lexsort(points.T[::-1])]
        return points


def sample_queries(tube,center,half_size_mm,radius_mm,count,rng,strategy="perpendicular_boundary"):
    cloud=tube.patch_cloud(center,half_size_mm,0,100000,rng)
    if not len(cloud): raise ValueError('Training patches must contain an observed centerline')
    # 25% near centerline/interior, 50% boundary band, 25% uniform background.
    ni=count//4; nb=count//2; nu=count-ni-nb
    if strategy == 'perpendicular_boundary' and np.any(tube.length2>1e-12):
        segments=np.flatnonzero(tube.length2>1e-12)
        probabilities=np.sqrt(tube.length2[segments]);probabilities/=probabilities.sum()
        chosen=rng.choice(segments,ni+nb,p=probabilities)
        # Condition candidate segments on this physical patch; otherwise sampling
        # a whole-subject edge then clipping it produces artificial cube-face labels.
        inside=np.all(np.abs(tube.mid[chosen]-center)<=half_size_mm,axis=1)
        patch_segments=segments[np.all(np.abs(tube.mid[segments]-center)<=half_size_mm,axis=1)]
        if not len(patch_segments):patch_segments=segments
        lengths=np.sqrt(tube.length2[patch_segments]);lengths/=lengths.sum()
        chosen=rng.choice(patch_segments,ni+nb,p=lengths)
        source=tube.start[chosen]+rng.uniform(size=(ni+nb,1))*tube.delta[chosen]
        tangent=tube.delta[chosen]/np.sqrt(tube.length2[chosen])[:,None]
        directions=rng.normal(size=(ni+nb,3));directions-=tangent*(directions*tangent).sum(1)[:,None]
    elif strategy in ('isotropic_jitter','perpendicular_boundary'):
        source=cloud[rng.integers(len(cloud),size=ni+nb)]
        directions=rng.normal(size=(ni+nb,3))
    else: raise ValueError('Unknown graph query sampling strategy')
    directions/=np.linalg.norm(directions,axis=1,keepdims=True).clip(1e-12)
    radii=np.r_[rng.uniform(0,.75*radius_mm,ni),rng.uniform(.75*radius_mm,1.25*radius_mm,nb)]
    near=source+directions*radii[:,None]
    near=np.clip(near,center-half_size_mm,center+half_size_mm)
    uniform=rng.uniform(center-half_size_mm,center+half_size_mm,size=(nu,3))
    query=np.concatenate((near,uniform)); labels=tube.occupancy(query,radius_mm)
    order=rng.permutation(count)
    return query[order].astype(np.float32),labels[order].astype(np.float32)


def assert_grid(image):
    if len(image.shape)!=3 or image.header.get_xyzt_units()[0]!='mm' or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3]))<1e-10:
        raise ValueError('Scalar 3D image with valid world-mm affine required')


def graph_alignment(tube,image,seg=None,sdf=None):
    """Diagnostics only; never changes graph or chooses training samples from GT."""
    from scipy.ndimage import map_coordinates
    assert_grid(image)
    vox=nib.affines.apply_affine(np.linalg.inv(image.affine),tube.points)
    inside=np.all((vox>=-1e-4)&(vox<=np.asarray(image.shape)-1+1e-4),axis=1)
    result={'nodes':len(tube.points),'edges':len(tube.edges),'world_mm_bounds':[tube.points.min(0).tolist(),tube.points.max(0).tolist()],
            'nodes_inside_image_fraction':float(inside.mean()),'max_edge_length_mm':float(np.linalg.norm(tube.points[tube.edges[:,1]]-tube.points[tube.edges[:,0]],axis=1).max()) if len(tube.edges) else 0.,
            'source_graph_betti0':nx.number_connected_components(tube.graph),
            'source_graph_betti1':tube.graph.number_of_edges()-tube.graph.number_of_nodes()+nx.number_connected_components(tube.graph)}
    for name,target in [('seg',seg),('sdf',sdf)]:
        if target is None: continue
        if target.shape!=image.shape or not np.allclose(target.affine,image.affine,atol=1e-4): raise ValueError('Graph/image/annotation grids differ')
        samples=map_coordinates(target.get_fdata(),vox[inside].T,order=0 if name=='seg' else 1,mode='nearest')
        result['node_'+name+'_inside_fraction']=float((samples==1).mean() if name=='seg' else (samples<0).mean())
        edge_vox=nib.affines.apply_affine(np.linalg.inv(image.affine),tube.cloud)
        edge_inside=np.all((edge_vox>=-1e-4)&(edge_vox<=np.asarray(image.shape)-1+1e-4),axis=1)
        edge_samples=map_coordinates(target.get_fdata(),edge_vox[edge_inside].T,order=0 if name=='seg' else 1,mode='nearest')
        result['densified_edge_'+name+'_inside_fraction']=float((edge_samples==1).mean() if name=='seg' else (edge_samples<0).mean())
        result['densified_edge_observed_fov_fraction']=float(edge_inside.mean())

    return result
