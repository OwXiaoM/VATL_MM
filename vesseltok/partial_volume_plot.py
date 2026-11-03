"""Standalone physical target plots;no prediction/threshold changes."""
import argparse
from pathlib import Path
import numpy as np


def plot_targets(root,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,8),layout='constrained');subfigs=fig.subfigures(2,1)
    examples=[('gap_0.1.npz','Pseudo-tubes: radius 1 mm, gap 0.1 mm; voxel width 0.5 mm'),('small_0.2.npz','Synthetic resolution stress: radius 0.2 mm; voxel width 1 mm')]
    for sf,(filename,title) in zip(subfigs,examples):
        sf.suptitle(title,fontsize=13);axes=sf.subplots(1,3)
        with np.load(Path(root)/'synthetic'/filename) as z:
            shape=z['shape'];centers=z['centers'].reshape(*shape,3);fraction=z['fraction'];point=z['point'];middle=int(shape[2]//2)
            spacing=float(centers[1,0,0,0]-centers[0,0,0,0]);extent=[float(centers[0,0,0,0]-spacing/2),float(centers[-1,0,0,0]+spacing/2),float(centers[0,0,0,1]-spacing/2),float(centers[0,-1,0,1]+spacing/2)]
            for ax,(name,field) in zip(axes,[('Point occupancy',point),('Voxel volume fraction',fraction),('Fraction >= 0.5',fraction>=.5)]):
                im=ax.imshow(field[:,:,middle].T,origin='lower',extent=extent,vmin=0,vmax=1,cmap='viridis',interpolation='nearest');ax.set(title=name,xlabel='World x (mm)',ylabel='World y (mm)')
            sf.colorbar(im,ax=axes.tolist(),label='Target value (not uncertainty)',shrink=.85)
    fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot_targets(a.root,a.output)
