"""Small reproducible Stage-1 training, with no test-time subject optimization."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from diagnostics.provenance import sha256,validate_splits
from vesseltok.data import PatchSet,prepare
from vesseltok.model import GraphOccupancyAE
from vesseltok.objectives import occupancy_loss
from vesseltok.selection_policy import POLICY,policy_sha256
from vesseltok.inter_anchor_streams import SamplingStreams


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--prepared',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--sampling-pool');parser.add_argument('--gap-pool');parser.add_argument('--steps',type=int); parser.add_argument('--device',default='cuda');parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    selection_mode=config.get('checkpoint_selection','bce')
    if selection_mode not in ['bce','native_geometry_v1']:raise ValueError('Unknown checkpoint selection policy')
    if args.prepare_only:
        prepare(config,args.prepared); return
    if args.device.startswith('cuda') and not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU work requires allocated Slurm compute')
    if args.steps is not None: config['steps']=args.steps
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    if selection_mode=='native_geometry_v1':
        (root/'candidates').mkdir()
        (root/'selection_required.json').write_text(json.dumps({'policy':POLICY,'policy_sha256':policy_sha256(),'status':'pending_native_validation','best_bce_is_not_geometry_selected':True},indent=2))
    prepared=Path(args.prepared); splits=json.loads((prepared/'splits.json').read_text()); validate_splits(splits)
    expected=yaml.safe_load((prepared/'config.yaml').read_text())
    for key in ('patch_mm','input_points','pseudo_radius_mm','edge_spacing_mm','query_pool','seed'):
        if config[key]!=expected[key]: raise ValueError('Prepared geometry/config mismatch: '+key)
    for key in ('sampling_strategy','representation','input_sampling'):
        if config.get(key)!=expected.get(key):raise ValueError('Prepared configuration mismatch: '+key)
    seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    model=GraphOccupancyAE(**config['model']).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    mode=config.get('inter_anchor_mode')
    if mode is not None:
        if args.sampling_pool is None:raise ValueError('Inter-anchor arms require a frozen training pool')
        pool_manifest=json.loads((Path(args.sampling_pool).parent/'pool_manifest.json').read_text())
        if sha256(prepared/'train.npz')!=pool_manifest['source_train_sha256'] or sha256(args.sampling_pool)!=pool_manifest['pool_sha256']:raise ValueError('Sampling pool/cache mismatch')
        if pool_manifest['train_subjects']!=splits['train']:raise ValueError('Pool includes a different training cohort')
    dataset_class=PatchSet;extra={}
    if config.get('gap_mode') is not None:
        from vesseltok.gap_sampling import GapPatchSet,GapStreams
        if not args.gap_pool:raise ValueError('Gap arms require a frozen gap pool')
        gap_manifest=json.loads((Path(args.gap_pool).parent/'pool_manifest.json').read_text())
        if sha256(prepared/'train.npz')!=gap_manifest['source_train_sha256'] or sha256(args.gap_pool)!=gap_manifest['gap_pool_sha256'] or splits['train']!=gap_manifest['train_subjects']:raise ValueError('Gap pool/cache mismatch')
        if sha256(args.sampling_pool)!=gap_manifest['positive_pool_sha256']:raise ValueError('Positive intervention changed')
        dataset_class=GapPatchSet;extra=dict(gap_pool=args.gap_pool,gap_mode=config['gap_mode'],gap_replacements=config['gap_replacements'],gap_seed=config['gap_private_seed'])
    train=dataset_class(prepared/'train.npz',config['queries_per_step'],config.get('class_balanced_batches',False),sampling_pool=args.sampling_pool,inter_anchor_mode=mode,replacements=config.get('inter_anchor_replacements',64),private_seed=config.get('inter_anchor_private_seed',9215),**extra)
    validation=PatchSet(prepared/'validation.npz',config['queries_per_step'],config.get('class_balanced_batches',False))
    streams=SamplingStreams() if mode is not None else None
    if config.get('gap_mode') is not None:streams=GapStreams()
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=config['batch_size'],shuffle=True,num_workers=0,generator=generator)
    iterator=iter(loader)
    # Fixed validation queries, not random resampling on each checkpoint decision.
    state=torch.get_rng_state(); torch.manual_seed(seed+123)
    validation_batches=[validation[i] for i in range(len(validation))];torch.set_rng_state(state)
    history=[];best=float('inf');start=time.time()
    provenance={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_sha256':{str(p):sha256(p) for p in Path('vesseltok').glob('*.py')},
                'metadata_sha256':sha256(config['metadata']),'manifest_sha256':sha256(config['manifest']),
                'prepared_sha256':{name:sha256(prepared/name) for name in ['splits.json','train.npz','validation.npz']},
                'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'initialization':'scratch','learned_transforms':False,'checkpoint_selection':selection_mode,
                'condition_dim':config['model'].get('condition_dim',0),'torch_version':torch.__version__,
                'initial_model_sha256':hashlib.sha256(b''.join(key.encode()+value.detach().cpu().numpy().tobytes() for key,value in model.state_dict().items())).hexdigest()}
    if mode is not None:provenance['sampling_intervention']={'mode':mode,'pool_sha256':sha256(args.sampling_pool),'pool_manifest_sha256':sha256(Path(args.sampling_pool).parent/'pool_manifest.json'),'validation_unchanged':True}
    if config.get('gap_mode') is not None:provenance['gap_intervention']={'mode':config['gap_mode'],'pool_sha256':sha256(args.gap_pool),'protocol_sha256':sha256(Path(args.gap_pool).parent/'pool_manifest.json'),'validation_unchanged':True}
    (root/'config.yaml').write_text(yaml.safe_dump(config));(root/'splits.json').write_text(json.dumps(splits,indent=2));(root/'provenance.json').write_text(json.dumps(provenance,indent=2))
    def checkpoint(step,name):
        torch.save({'model':model.state_dict(),'config':config,'step':step,'splits':splits,'provenance':provenance,
                    'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'history':history},root/name)
    for step in range(1,config['steps']+1):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader);batch=next(iterator)
        cpu_batch=batch
        if streams is not None:streams.update(cpu_batch)
        batch={key:value.to(args.device) for key,value in batch.items() if key in ['cloud','padding','queries','labels','conditions']}
        model.train();optimizer.zero_grad(set_to_none=True)
        output=model(batch['cloud'],batch['padding'],batch['queries'],batch['conditions'] if model.condition_dim else None)
        if streams is not None:streams.verify_anchors(output['encoded'],cpu_batch)
        reconstruction=occupancy_loss(output['logits'],batch['labels'],config.get('negative_weight',1.))
        beta=config['kl_weight']*min(1.,step/config['kl_warmup_steps'])
        loss=reconstruction+beta*output['kl']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Stage-1 loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if step==1 or step%config['validate_every']==0 or step==config['steps']:
            model.eval();values=[]
            with torch.no_grad():
                for sample in validation_batches:
                    b={key:value[None].to(args.device) for key,value in sample.items()}
                    prediction=model(b['cloud'],b['padding'],b['queries'],b['conditions'] if model.condition_dim else None)
                    values.append(float(F.binary_cross_entropy_with_logits(prediction['logits'],b['labels'])))
            record={'step':step,'train_bce':float(F.binary_cross_entropy_with_logits(output['logits'].detach(),batch['labels'])),'train_reconstruction':float(reconstruction.detach()),'kl':float(output['kl'].detach()),'beta':beta,'validation_bce':float(np.mean(values)),'negative_weight':config.get('negative_weight',1.),'elapsed_seconds':time.time()-start}
            history.append(record);print(json.dumps(record),flush=True)
            (root/'history.json').write_text(json.dumps(history,indent=2))
            if streams is not None:streams.save(root/'sampling_streams.json')
            if record['validation_bce']<best:
                best=record['validation_bce'];checkpoint(step,'best_bce.pt' if selection_mode=='native_geometry_v1' else 'best.pt')
            checkpoint(step,'last.pt')
            if selection_mode=='native_geometry_v1' and (step%config.get('retain_checkpoints_every',500)==0 or step==config['steps']):
                checkpoint(step,'candidates/step_{:06d}.pt'.format(step))
    print('Training complete',root,flush=True)


if __name__=='__main__':main()
