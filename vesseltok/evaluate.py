"""Separate pseudo-occupancy and anatomical GT evaluation of saved predictions."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from skimage.measure import marching_cubes
from vesseltok.geometry import TubeGraph,graph_alignment
from vesseltok.topology import reconstruction_metrics,skeleton_graph,branch_metrics
from vesseltok.infer import save_nifti
from diagnostics.provenance import canonical_id


def export_geometry(mask,affine,output,probability=None,threshold=.5):
    skeleton,points,edges=skeleton_graph(mask,affine)
    np.savez_compressed(output/'centerline_graph.npz',points_mm=points,edges=edges)
    save_nifti(output/'centerline_skeleton.nii.gz',skeleton.astype(np.uint8),affine)
    if mask.any() and (~mask).any():
        field=mask.astype(np.float32) if probability is None else np.asarray(probability,np.float32)
        vertices,faces,_,_=marching_cubes(field,level=(.5 if probability is None else threshold))
        vertices=nib.affines.apply_affine(affine,vertices)
        with open(output/'vessel_surface.ply','w') as stream:
            stream.write('ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n'.format(len(vertices),len(faces)))
            np.savetxt(stream,vertices,fmt='%.6f');np.savetxt(stream,np.c_[np.full(len(faces),3),faces],fmt='%d')


def qualitative(prediction,pseudo,anatomical,affine,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=['Predicted graph occupancy','Graph pseudo-radius target','Real vascular segmentation'])
    raster,axes=plt.subplots(1,3,figsize=(12,4))
    for column,(name,mask,color) in enumerate([('Prediction',prediction,'#e25151'),('Pseudo occupancy',pseudo,'#307abb'),('Anatomical vessel',anatomical,'#23926d')]):
        points=nib.affines.apply_affine(affine,np.argwhere(mask))
        selected=points[::max(1,len(points)//40000)]
        figure.add_trace(go.Scatter3d(x=selected[:,0],y=selected[:,1],z=selected[:,2],mode='markers',marker={'size':1,'color':color,'opacity':.6},name=name),row=1,col=column+1)
        axes[column].imshow(mask.max(2).T,origin='lower',cmap='gray');axes[column].set_title(name);axes[column].axis('off')
    figure.update_layout(title='Unseen graph-conditioned reconstruction: pseudo-radius is not anatomical radius',height=650,showlegend=False)
    figure.update_scenes(aspectmode='data',xaxis_title='RAS x (mm)',yaxis_title='RAS y (mm)',zaxis_title='RAS z (mm)')
    figure.write_html(str(output/'reconstruction_3d.html'),include_plotlyjs=True)
    raster.tight_layout();raster.savefig(output/'reconstruction_mip.png',dpi=160);plt.close(raster)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',required=True);parser.add_argument('--config',required=True);args=parser.parse_args()
    root=Path(args.predictions);manifest=json.loads((root/'manifest.json').read_text());config=yaml.safe_load(Path(args.config).read_text())
    trained={canonical_id(s) for s in manifest['training_subjects']}
    if any(canonical_id(s) in trained for s in manifest['subjects']):raise ValueError('Evaluation cannot label trained subjects as unseen')
    metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id');results=[]
    for subject in manifest['subjects']:
        output=root/subject;row=metadata.loc[subject]
        image=nib.load(row['MRA']);seg=nib.load(row['Seg']);sdf=nib.load(row['SDF']);prediction_image=nib.load(output/'occupancy_mask.nii.gz')
        if prediction_image.shape!=image.shape or not np.allclose(prediction_image.affine,image.affine,atol=1e-4):raise ValueError('Prediction native grid mismatch')
        tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm'])
        alignment=graph_alignment(tube,image,seg,sdf)
        indices=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T
        world=nib.affines.apply_affine(image.affine,indices)
        pseudo=tube.occupancy(world,manifest['pseudo_radius_mm']).reshape(image.shape)
        seg_values=seg.get_fdata();pred_values=prediction_image.get_fdata()
        if not np.isfinite(seg_values).all() or not np.isfinite(pred_values).all() or not np.isfinite(sdf.get_fdata()).all():raise ValueError('Nonfinite evaluation volume')
        anatomical=seg_values==1;prediction=pred_values>0
        save_nifti(output/'graph_pseudo_target.nii.gz',pseudo.astype(np.uint8),image.affine)
        record={'subject_id':subject,'unseen_training_membership':True,'alignment':alignment,
                'graph_pseudo_occupancy':reconstruction_metrics(prediction,pseudo,image.affine,tube),
                'real_anatomical_vessel':reconstruction_metrics(prediction,anatomical,image.affine,tube),
                'target_pseudo_vs_anatomical':reconstruction_metrics(pseudo,anatomical,image.affine),
                'target_pseudo_connectivity_to_source_graph':branch_metrics(pseudo,tube,image.affine),
                'real_sdf_sign_vs_seg_dice':float(2*((sdf.get_fdata()<0)&anatomical).sum()/max(1,(sdf.get_fdata()<0).sum()+anatomical.sum()))}
        results.append(record);(output/'metrics.json').write_text(json.dumps(record,indent=2,allow_nan=False))
        export_geometry(prediction,image.affine,output,nib.load(output/'occupancy_probability.nii.gz').get_fdata(),manifest['threshold']);qualitative(prediction,pseudo,anatomical,image.affine,output)
        print(subject,'pseudo Dice',record['graph_pseudo_occupancy']['dice'],'anatomical Dice',record['real_anatomical_vessel']['dice'],flush=True)
    summary={'protocol':'graph-conditioned reconstruction, not image-only prediction','manifest':manifest,'subjects':results}
    (root/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
