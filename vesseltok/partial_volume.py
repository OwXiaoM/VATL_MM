"""Physical voxel fractions of graph tubes,not vessel-presence uncertainty."""
import itertools
import numpy as np


def cell_basis(value):
    basis=np.asarray(value,float)
    if basis.shape!=(3,3) or not np.isfinite(basis).all() or abs(np.linalg.det(basis))<1e-12:raise ValueError('Finite nonsingular physical voxel basis required')
    return basis


def circumradius(basis):
    basis=cell_basis(basis)
    return float(np.linalg.norm(np.array(list(itertools.product([-.5,.5],repeat=3)))@basis.T,axis=1).max())


def capped_distance(tube,points,limit,chunk=4096):
    """Exact distance if <=limit;larger distances may be inf,never a false inside."""
    points=np.asarray(points,float).reshape(-1,3)
    if not np.isfinite(points).all() or not np.isfinite(limit) or limit<=0:raise ValueError('Finite coordinates and positive search radius required')
    result=np.full(len(points),np.inf);bound=limit+tube.half_max+1e-9
    for begin in range(0,len(points),chunk):
        q=points[begin:begin+chunk];possible=np.flatnonzero(tube.tree.query(q)[0]<=bound)
        if not len(possible):continue
        neighbors=tube.tree.query_ball_point(q[possible],bound);sizes=np.array([len(x) for x in neighbors],int)
        qi=np.repeat(np.arange(len(possible)),sizes);si=np.concatenate(neighbors).astype(int)
        delta=q[possible][qi]-tube.start[si]
        fraction=np.divide((delta*tube.delta[si]).sum(1),tube.length2[si],out=np.zeros(len(si)),where=tube.length2[si]>0).clip(0,1)
        d2=((delta-fraction[:,None]*tube.delta[si])**2).sum(1);minimum=np.full(len(possible),np.inf);np.minimum.at(minimum,qi,d2)
        result[begin+possible]=np.sqrt(minimum)
    return result


def voxel_fractions(tube,centers,basis,n=8,radius_mm=1.,batch=16):
    """Deterministic n^3 midpoint fractions with rigorous geometric volume bounds.

    Dyadic subdivision skips subcells certified wholly in/out using the distance
    function's1-Lipschitz property. This equals naive midpoint sampling at n.
    Bounds bracket the EXACT fraction,not numerical confidence intervals.
    """
    centers=np.asarray(centers,float);basis=cell_basis(basis)
    if centers.ndim!=2 or centers.shape[1]!=3 or not np.isfinite(centers).all():raise ValueError('Finite physical centers required')
    if not isinstance(n,int) or n<1 or n&(n-1) or not np.isfinite(radius_mm) or radius_mm<=0:raise ValueError('Positive radius and dyadic sampling n required')
    h=circumradius(basis);initial=capped_distance(tube,centers,radius_mm+h+1e-9)
    inside=initial+h<=radius_mm-1e-10;outside=initial-h>radius_mm+1e-10
    lower=inside.astype(float);upper=lower.copy();estimate=lower.copy();unknown=np.flatnonzero(~inside&~outside);depth_max=n.bit_length()-1
    child=np.array(list(itertools.product([-.25,.25],repeat=3)))@basis.T
    for begin in range(0,len(unknown),batch):
        roots=unknown[begin:begin+batch];q=centers[roots].copy();owner=np.arange(len(roots));lo=np.zeros(len(roots))
        for depth in range(depth_max+1):
            half=h/(2**depth);d=capped_distance(tube,q,radius_mm+half+1e-9)
            inn=d+half<=radius_mm-1e-10;out=d-half>radius_mm+1e-10;ambiguous=~inn&~out;mass=1./(8**depth)
            lo+=np.bincount(owner[inn],minlength=len(roots))*mass
            if depth==depth_max or not ambiguous.any():
                up=lo+np.bincount(owner[ambiguous],minlength=len(roots))*mass
                mid=lo+np.bincount(owner[ambiguous&(d*d<=radius_mm**2+1e-12)],minlength=len(roots))*mass
                lower[roots]=lo;upper[roots]=up;estimate[roots]=mid;break
            q=(q[ambiguous,None,:]+child[None,:,:]/(2**depth)).reshape(-1,3);owner=np.repeat(owner[ambiguous],8)
    assert np.all(lower<=estimate+1e-12) and np.all(estimate<=upper+1e-12) and np.all(upper<=1+1e-12)
    return {'fraction':estimate,'lower':lower,'upper':upper,'n':n}


def naive_fractions(tube,centers,basis,n,radius_mm=1.):
    """Independent reference quadrature for small numerical tests only."""
    grid=(np.arange(n)+.5)/n-.5;offsets=np.array(list(itertools.product(grid,repeat=3)))@cell_basis(basis).T
    return np.array([tube.occupancy(c+offsets,radius_mm).mean() for c in np.asarray(centers)])
