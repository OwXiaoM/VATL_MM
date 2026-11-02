"""Unseen graph-conditioned inference. No segmentation/SDF or TTA is used."""
import argparse
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from vesseltok.geometry import TubeGraph,world_to_local,assert_grid
from vesseltok.data import padded_cloud,demographics
from vesseltok.model import GraphOccupancyAE
from vesseltok.input_sampling import subject_cloud
from diagnostics.provenance import canonical_id,sha256


def save_nifti(path,array,affine):
    image=nib.Nifti1Image(array,affine);image.header.set_xyzt_units('mm');nib.save(image,str(path))


def predict_subject(model,tube,image,config,device,chunk=4096):
    assert_grid(image)
    indices=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T
    world=nib.affines.apply_affine(image.affine,indices).astype(np.float32)
    if config.get('representation')=='subject_tokens':
        center=nib.affines.apply_affine(image.affine,np.asarray(image.shape)/2.)
        rng=np.random.default_rng(config['seed'])
        cloud=subject_cloud(tube,center,config,rng)
        if not len(cloud): raise ValueError('Empty subject graph')
        points,padding=padded_cloud(cloud,center,config)
        c=torch.from_numpy(points)[None].to(device);m=torch.from_numpy(padding)[None].to(device)
        conditions=torch.as_tensor(config['_conditions'],device=device)[None] if model.condition_dim else None
        probability=np.empty(len(world),np.float32)
        with torch.no_grad():
            encoded=model.encode(c,m,conditions,sample=False);prepared=model.prepare_decoder(encoded)
            for start in range(0,len(world),chunk):
                queries=world_to_local(world[start:start+chunk],center,config['patch_mm']/2).astype(np.float32)
                probability[start:start+chunk]=model.decode(torch.from_numpy(queries)[None].to(device),encoded,prepared).sigmoid()[0].cpu().numpy()
        records=[{'center_mm':center.tolist(),'mean':encoded['mean'][0].cpu().tolist(),
                  'anchors_local':encoded['anchors'][0].cpu().tolist(),'padding':encoded['padding'][0].cpu().tolist()}]
        return probability.reshape(image.shape),records
    low=world.min(0)
    stride=config.get('inference_stride_mm',config['patch_mm']*.75)
    if not 0 < stride < config['patch_mm']: raise ValueError('Overlapping physical inference patches required')
    tile=np.floor((world-low)/stride).astype(int)
    dimensions=tile.max(0)+1;codes=np.ravel_multi_index(tile.T,dimensions)
    order=np.argsort(codes,kind='stable');unique,begin=np.unique(codes[order],return_index=True)
    stops=np.r_[begin[1:],len(world)]
    ranges={int(code):(int(start),int(stop)) for code,start,stop in zip(unique,begin,stops)}
    probability=np.zeros(len(world),np.float32);total_weight=np.zeros(len(world),np.float32)
    token_records=[]
    with torch.no_grad():
        for code in unique:
            cell=np.array(np.unravel_index(code,dimensions));center=low+(cell+.5)*stride
            candidates=[]
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for dz in (-1,0,1):
                        neighbor=cell+[dx,dy,dz]
                        if np.any(neighbor<0) or np.any(neighbor>=dimensions):continue
                        neighbor_code=int(np.ravel_multi_index(neighbor,dimensions))
                        if neighbor_code in ranges:
                            start,stop=ranges[neighbor_code];candidates.append(order[start:stop])
            chosen=np.concatenate(candidates)
            local=world_to_local(world[chosen],center,config['patch_mm']/2)
            inside=np.all(np.abs(local)<1,axis=1);chosen=chosen[inside];local=local[inside]
            weights=np.prod(1-np.abs(local),axis=1).astype(np.float32)
            total_weight[chosen]+=weights
            # Per-patch seed makes cloud/FPS independent of query chunk/order.
            rng=np.random.default_rng(config['seed']+int(code))
            cloud=subject_cloud(tube,center,config,rng)
            if not len(cloud):continue
            points,padding=padded_cloud(cloud,center,config)
            c=torch.from_numpy(points)[None].to(device);m=torch.from_numpy(padding)[None].to(device)
            conditions=None
            if model.condition_dim:conditions=torch.as_tensor(config['_conditions'],device=device)[None]
            encoded=model.encode(c,m,conditions,sample=False);prepared=model.prepare_decoder(encoded)
            token_records.append({'center_mm':center.tolist(),'mean':encoded['mean'][0].cpu().tolist(),
                                  'anchors_local':encoded['anchors'][0].cpu().tolist(),'padding':encoded['padding'][0].cpu().tolist()})
            for offset in range(0,len(chosen),chunk):
                ids=chosen[offset:offset+chunk]
                query=local[offset:offset+chunk].astype(np.float32)
                logits=model.decode(torch.from_numpy(query)[None].to(device),encoded,prepared)
                probability[ids]+=logits.sigmoid()[0].cpu().numpy()*weights[offset:offset+chunk]
    if not np.all(total_weight>0): raise ValueError('Inference patch coverage incomplete')
    probability/=total_weight
    return probability.reshape(image.shape),token_records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--subjects',nargs='+');parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.device.startswith('cuda') and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU work requires allocated Slurm compute')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=checkpoint['config'];subjects=args.subjects or checkpoint['splits']['evaluation']
    trained={canonical_id(s) for s in checkpoint['splits']['train']}
    if any(canonical_id(s) in trained for s in subjects):raise ValueError('Unseen evaluation requested a trained person')
    model=GraphOccupancyAE(**config['model']).to(args.device).eval();model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
    metadata=pd.read_csv(config['metadata'],sep='\t').set_index('subject_id')
    manifest={'checkpoint':args.checkpoint,'checkpoint_sha256':sha256(args.checkpoint),'subjects':subjects,
              'training_subjects':checkpoint['splits']['train'],'input':'graph centerlines and image geometry; no segmentation/SDF','learned_transforms':False,
              'threshold':config['threshold'],'pseudo_radius_mm':config['pseudo_radius_mm'],'patch_mm':config['patch_mm'],'step':checkpoint['step']}
    selection_file=Path(args.checkpoint).parent/'selection.json'
    if Path(args.checkpoint).name=='selected.pt':
        selection=json.loads(selection_file.read_text())
        if selection['selected_checkpoint_sha256']!=manifest['checkpoint_sha256'] or selection['evaluation_used']:raise ValueError('Invalid geometry selection alias')
        manifest['checkpoint_selection']={'policy_sha256':selection['policy_sha256'],'selection_record':str(selection_file),'selection_record_sha256':sha256(selection_file),'selected_via_validation_only':True}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for subject in subjects:
        row=metadata.loc[subject];image=nib.load(row['MRA']);tube=TubeGraph.load(row['Graph'],config['edge_spacing_mm'])
        config['_conditions']=demographics(row).tolist()
        probability,tokens=predict_subject(model,tube,image,config,args.device)
        output=root/subject;output.mkdir()
        save_nifti(output/'occupancy_probability.nii.gz',probability,image.affine)
        save_nifti(output/'occupancy_mask.nii.gz',(probability>=config['threshold']).astype(np.uint8),image.affine)
        (output/'tokens.json').write_text(json.dumps({'known_demographics':{'scan_age':float(row['scan_age']),'sex':row['sex']},'patch_tokens':tokens}))
        print('predicted unseen graph',subject,'voxels',int((probability>=config['threshold']).sum()),'patches',len(tokens),flush=True)


if __name__=='__main__':main()
