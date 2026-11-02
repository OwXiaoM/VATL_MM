"""Decode saved tokens with image geometry; no graph, images, masks or SDF values."""
import argparse
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from vesseltok.geometry import world_to_local,assert_grid
from vesseltok.model import GraphOccupancyAE
from vesseltok.infer import save_nifti


def render_tokens(model,records,image,config,device,chunk=4096):
    assert_grid(image)
    indices=np.indices(image.shape,dtype=np.float32).reshape(3,-1).T
    world=nib.affines.apply_affine(image.affine,indices).astype(np.float32)
    probability=np.zeros(len(world),np.float32)
    def encoded(record):
        return {'tokens':torch.tensor(record['mean'],dtype=torch.float32,device=device)[None],
                'padding':torch.tensor(record['padding'],dtype=torch.bool,device=device)[None],
                'anchors':torch.tensor(record['anchors_local'],dtype=torch.float32,device=device)[None]}
    if config.get('representation')=='subject_tokens':
        if len(records)!=1:raise ValueError('Subject representation requires one fixed token set')
        record=records[0];code=encoded(record)
        with torch.no_grad():
            prepared=model.prepare_decoder(code)
            for start in range(0,len(world),chunk):
                queries=world_to_local(world[start:start+chunk],record['center_mm'],config['patch_mm']/2).astype(np.float32)
                probability[start:start+chunk]=model.decode(torch.from_numpy(queries)[None].to(device),code,prepared).sigmoid()[0].cpu().numpy()
    else:
        low=world.min(0);stride=config.get('inference_stride_mm',config['patch_mm']*.75)
        cell=np.floor((world-low)/stride).astype(int);dimensions=cell.max(0)+1
        total=np.zeros(len(world),np.float32)
        # Empty lattice blocks must still contribute their tent denominator, as
        # in graph inference. Their field is zero, without needing the source graph.
        occupied_codes=np.unique(np.ravel_multi_index(cell.T,dimensions))
        for code in occupied_codes:
            coordinate=np.unravel_index(code,dimensions)
            center=low+(np.asarray(coordinate)+.5)*stride
            local=world_to_local(world,center,config['patch_mm']/2)
            selected=np.flatnonzero(np.all(np.abs(local)<1,axis=1))
            weights=np.prod(1-np.abs(local[selected]),axis=1).astype(np.float32);total[selected]+=weights
        with torch.no_grad():
            for record in records:
                code=encoded(record);prepared=model.prepare_decoder(code)
                local=world_to_local(world,record['center_mm'],config['patch_mm']/2)
                selected=np.flatnonzero(np.all(np.abs(local)<1,axis=1))
                weights=np.prod(1-np.abs(local[selected]),axis=1).astype(np.float32)
                for start in range(0,len(selected),chunk):
                    ids=selected[start:start+chunk];queries=local[ids].astype(np.float32)
                    probability[ids]+=model.decode(torch.from_numpy(queries)[None].to(device),code,prepared).sigmoid()[0].cpu().numpy()*weights[start:start+chunk]
        if not np.all(total>0):raise ValueError('Incomplete token lattice coverage')
        probability/=total
    return probability.reshape(image.shape)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--tokens',required=True);parser.add_argument('--geometry',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.device.startswith('cuda') and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU allocation required')
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve previous decoded output')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=GraphOccupancyAE(**checkpoint['config']['model']).to(args.device).eval().requires_grad_(False);model.load_state_dict(checkpoint['model'])
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.use_deterministic_algorithms(True)
    records=json.loads(Path(args.tokens).read_text())['patch_tokens']
    probability=render_tokens(model,records,nib.load(args.geometry),checkpoint['config'],args.device)
    image=nib.load(args.geometry);save_nifti(output,probability,image.affine)


if __name__=='__main__':main()
