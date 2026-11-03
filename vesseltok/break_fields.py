"""Allocated-GPU frozen fields on cached train inputs and saved validation tokens."""
import argparse
import json
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from diagnostics.provenance import sha256
from vesseltok.break_helpers import require_audit_scope
from vesseltok.break_prepare import fingerprint
from vesseltok.geometry import local_to_world
from vesseltok.model import GraphOccupancyAE
from vesseltok.field_probe import WorldDecoder
from vesseltok.decode_tokens import render_tokens
from vesseltok.infer import save_nifti


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',required=True);parser.add_argument('--arm',choices=['A','B'],required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Allocated Slurm GPU required')
    root=Path(args.prepared);manifest=json.loads((root/'manifest.json').read_text());arm=args.arm;out=root/('fields_'+arm);out.mkdir(exist_ok=False)
    require_audit_scope([r['subject_id'] for r in manifest['subjects']],manifest['splits'])
    assert sha256(manifest['checkpoints'][arm])==manifest['checkpoint_sha256'][arm]
    saved=torch.load(manifest['checkpoints'][arm],map_location='cpu',weights_only=False);config=saved['config'];assert config['model']['condition_dim']==0 and not saved['provenance']['learned_transforms']
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.use_deterministic_algorithms(True)
    model=GraphOccupancyAE(**config['model']).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);before=fingerprint(model);results=[]
    with np.load(manifest['positive_pool']) as p:anchors=p['anchors'];padding=p['anchor_padding']
    for record in manifest['subjects']:
        sid=record['subject_id'];destination=out/sid;destination.mkdir();image=nib.load(record['MRA'])
        if record['split']=='train':
            with np.load(root/sid/'cached_input.npz') as c:
                with torch.no_grad():encoded=model.encode(torch.from_numpy(c['cloud'])[None].cuda(),torch.from_numpy(c['padding'])[None].cuda(),sample=False)
                index=record['row_index'];assert torch.equal(encoded['anchors'][0].cpu(),torch.from_numpy(anchors[index])) and torch.equal(encoded['padding'][0].cpu(),torch.from_numpy(padding[index]))
                tokens=[{'center_mm':c['center'].tolist(),'mean':encoded['mean'][0].cpu().tolist(),'anchors_local':encoded['anchors'][0].cpu().tolist(),'padding':encoded['padding'][0].cpu().tolist()}]
            probability=render_tokens(model,tokens,image,config,'cuda');field_source='New frozen native render using exact cached training cloud'
            save_nifti(destination/'occupancy_probability.nii.gz',probability,image.affine)
        else:
            original=Path(manifest['runs'][arm])/'native_validation/step_002000'/sid
            tokens=json.loads((original/'tokens.json').read_text())['patch_tokens'];field=nib.load(original/'occupancy_probability.nii.gz')
            assert field.shape==image.shape and np.allclose(field.affine,image.affine)
            probability=field.get_fdata(dtype=np.float32);field_source=str(original/'occupancy_probability.nii.gz')
            (destination/'occupancy_probability.nii.gz').symlink_to(original/'occupancy_probability.nii.gz')
        (destination/'tokens.json').write_text(json.dumps({'patch_tokens':tokens},indent=2))
        decoder=WorldDecoder(model,tokens,config,'cuda')
        with np.load(root/sid/'geometry.npz') as g:
            points=g['points_mm'];inside=g['inside_fov'];cp=decoder(points);candidates=g['gap_candidates_mm'];actual=g['realized_negatives_mm']
            gap_probability=decoder(candidates);used_probability=decoder(actual)
        voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine),points)).astype(int)
        nearest=np.full(len(points),np.nan,np.float32);nearest[inside]=probability[tuple(voxel[inside].T)]
        grid_world=nib.affines.apply_affine(image.affine,voxel[inside]).astype(np.float32);direct=decoder(grid_world)
        error=float(np.max(np.abs(direct-nearest[inside])));assert error<=1e-5,error
        anchor_world=local_to_world(np.asarray(tokens[0]['anchors_local'])[~np.asarray(tokens[0]['padding'])],tokens[0]['center_mm'],128.)
        from scipy.spatial import cKDTree
        distance=cKDTree(anchor_world).query(points)[0]
        np.savez_compressed(destination/'probes.npz',continuous_probability=cp,nearest_voxel_probability=nearest,anchor_distance_mm=distance,gap_probability=gap_probability,realized_negative_probability=used_probability)
        results.append({'subject_id':sid,'split':record['split'],'source':field_source,'maximum_grid_error':error,'native_probability_sha256':sha256(destination/'occupancy_probability.nii.gz')})
        print(arm,record['split'],sid,'grid_error',error,flush=True)
    after=fingerprint(model);assert before==after and all(p.grad is None for p in model.parameters())
    result={'arm':arm,'subjects':results,'parameter_sha256_before':before,'parameter_sha256_after':after,'checkpoint_sha256':sha256(manifest['checkpoints'][arm]),'slurm_job_id':os.environ['SLURM_JOB_ID'],'optimizer_or_updates':False,'segmentation_or_sdf_read':False}
    (out/'manifest.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
