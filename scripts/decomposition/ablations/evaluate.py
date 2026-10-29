"""Evaluate known training subjects or SDF-supervised held-out adaptation."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from data_paths import relocate_path
from data_loading.dataset import Data
from models.sdf_decomposition import AtlasSubjectSDF
from models.decomposition_evaluation import evaluate_subject
from models.decomposition_inference import adapt_subject


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--subjects', type=int, default=3)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--points', type=int, default=8192)
    p.add_argument('--chunk', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    cli = p.parse_args()
    if min(cli.subjects,cli.points,cli.chunk,cli.steps) < 1:
        p.error('counts must be positive')
    random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)
    output = Path(cli.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format') != 'vatl_sdf_atlas_rigid_v2':
        p.error('Expected decomposition checkpoint')
    args = saved['args']
    model = AtlasSubjectSDF(args, saved['subject_ids'], cli.device).to(cli.device)
    model.load_state_dict(saved['model']); model.set_stage(saved['epoch']); model.eval()
    if cli.split == 'train':
        ids = saved['subject_ids'][:cli.subjects]
        frame = saved['dataset_df']
    else:
        args.update(output_dir=str(output), rank=0)
        args['n_subjects']['val'] = cli.subjects
        args.setdefault('evaluation', {})['subject_manifest'] = None
        dataset = Data(args, pd.read_csv(relocate_path(args['dataset']['tsv_file']), sep='\t'), 'val')
        frame = dataset.df
        ids = frame.subject_id.astype(str).tolist()
        if set(ids) & set(saved['subject_ids']):
            raise ValueError('Held-out subjects overlap training IDs')
    protocol = {**vars(cli), 'checkpoint': str(Path(cli.checkpoint).resolve()), 'epoch': saved['epoch'],
                'subject_ids': ids, 'evaluation_region': 'full_segmentation_reference_grid',
                'adaptation': 'none_saved_training_parameters' if cli.split=='train' else 'ground_truth_SDF_supervised',
                'outside_model_crop': 'background', 'threshold_mm': 0.}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records=[]
    for subject in ids:
        row = frame[frame.subject_id.astype(str)==subject].iloc[0].to_dict()
        adaptation = None
        index = saved['subject_ids'].index(subject) if cli.split=='train' else None
        if cli.split == 'val':
            target_img = nib.load(relocate_path(row[args['dataset']['sdf_key']]))
            target = target_img.get_fdata()
            seg = nib.load(relocate_path(row[args['dataset']['modalities'][-1]] )).get_fdata()
            half = np.asarray(args['dataset']['world_bbox'])/2
            context = SimpleNamespace(args=args,world_bbox=2*half)
            indices = Data.sample_sdf_indices(context,target,seg,target_img.affine,num_points=cli.points)
            center = nib.affines.apply_affine(target_img.affine,np.asarray(target.shape)/2)
            coords = torch.as_tensor((nib.affines.apply_affine(target_img.affine,indices)-center)/half,dtype=torch.float32,device=cli.device)
            targets = torch.as_tensor(target[tuple(indices.T)],dtype=torch.float32,device=cli.device)
            condition = Data.load_conditions(SimpleNamespace(args=args),row.copy()).to(cli.device)
            adaptation = adapt_subject(model,coords,condition.expand(len(coords),-1),targets,steps=cli.steps)
            torch.save(adaptation,output/f'adaptation_{subject}.pth')
        record=evaluate_subject(model,row,index,adaptation,cli.chunk,topology=True)
        records.append(record)
        fields=list(dict.fromkeys(k for r in records for k in r))
        with (output/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
        (output/'metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
