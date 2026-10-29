"""CPU fine-grid preservation gate, trusted-FOV caches and matched fit configs."""
import argparse,copy,csv,json,sys
from pathlib import Path
import os
REPO=Path(os.environ.get('SDF_REPO',Path(__file__).resolve().parents[4])).resolve()
sys.path.insert(0,str(REPO))
import ants
import numpy as np
import nibabel as nib
import yaml
from skimage.morphology import skeletonize
from data_loading.coverage_dataset import build_cache


def matched_args(root,cfg,metadata):
    template=Path(cfg['source_study']).parent/'diagnostic_study_20261027/small_reference/config_final.yaml'
    args=yaml.safe_load(template.read_text())
    args.update(batch_size=1,num_workers=0,logging=False,compute_metrics=False,save_every=100,seed=cfg['seed'],amp=False)
    args['epochs']['train']=cfg['epochs']
    args['dataset'].update(tsv_file=str(metadata),coverage_key='Coverage',coverage_threshold=.999,
        subject_ids={'train':[s['subject_id'] for s in json.loads((Path(cfg['fine_study'])/'study.json').read_text())['subjects']],'val':[],'test':[]})
    args['n_subjects']['train']=4
    decomp=args['decomposition']
    for key in ['residual_enabled','latent_enabled','alpha','latent_dim','hidden','depth','fourier_bands','lr_residual','lr_latent']:
        decomp.pop(key,None)
    for key in ['residual','smoothness','latent']:decomp['loss']['weights'].pop(key,None)
    decomp.update(registration_enabled=False,topology_enabled=False,stages={'enabled':False},
        debug_evaluation={'enabled':True,'every':50,'subjects':4,'chunk':8192})
    args['graph_supervision']['activate']=False
    args['data_augmentation']['activate']=False
    args.setdefault('task_specific_sampling',{})['enabled']=False
    args['output_dir']=str(root/'fits')
    return args


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);a=p.parse_args()
    root=Path(a.study).resolve();cfg=json.loads((root/'followup.json').read_text());fine=Path(cfg['fine_study'])
    if not (fine/'training_gate.json').is_file():raise ValueError('Fine export must finish before preflight')
    out=root/'preflight';out.mkdir(exist_ok=False);configs=root/'configs';configs.mkdir(exist_ok=False)
    cache=root/'caches';cache.mkdir(exist_ok=False)
    gate=dict(cfg['preservation_gate']);gate['ideal_sdf_native_Dice_min']=.95
    checks=[];stats=[];byarm={};allargs={}
    for arm,stage in cfg['arms'].items():
        with (fine/stage/'metadata.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        metadata=out/(arm+'_metadata.tsv');args=matched_args(root,cfg,metadata)
        for row in rows:
            sid=row['subject_id'];folder=fine/stage/sid
            transform=json.loads((folder/'transforms.json').read_text())
            native=nib.load(fine/'audit/native_centered'/sid/'mask.nii.gz');truth=native.get_fdata()>.5
            backim=nib.load(folder/'roundtrip_native_mask.nii.gz')
            if backim.shape!=native.shape or not np.allclose(backim.affine,native.affine,rtol=0,atol=1e-4):raise ValueError('Native inverse grid mismatch')
            back=backim.get_fdata()>.5;sk=skeletonize(truth)
            retained=float((sk&back).sum()/max(1,sk.sum()))
            dice=float(2*(back&truth).sum()/max(1,back.sum()+truth.sum()))
            prefix='identity' if arm=='existing' else 'prealigned'
            for name,value,passed in [
                ('roundtrip_Dice',dice,dice>=gate[prefix+'_Dice_min']),
                ('centerline_recall',retained,retained>=gate[prefix+'_centerline_recall_min']),
                ('fold_fraction',transform['covered']['fold_fraction'],transform['covered']['fold_fraction']<=gate['fold_fraction_max']),
                ('inverse_consistency_p95_mm',transform['inverse_consistency_point_p95_mm'],transform['inverse_consistency_point_p95_mm']<=gate['inverse_consistency_p95_max_mm'])]:
                checks.append({'arm':arm,'subject_id':sid,'check':name,'value':value,'passed':bool(passed)})
            # Control for the exact inverse level-set interpolation used in neural evaluation.
            ideal=ants.apply_transforms(ants.image_read(str(fine/'audit/native_centered'/sid/'mask.nii.gz')),
                ants.image_read(str(folder/'aligned_sdf_mm.nii.gz')),transform['inverse_image_transformlist'],
                whichtoinvert=transform['inversion_flags_inverse'],interpolator='linear',defaultvalue=5.,singleprecision=True).numpy()<0
            ideal_dice=float(2*(ideal&truth).sum()/max(1,ideal.sum()+truth.sum()))
            checks.append({'arm':arm,'subject_id':sid,'check':'ideal_sdf_native_Dice','value':ideal_dice,'passed':ideal_dice>=gate['ideal_sdf_native_Dice_min']})
            target=cache/arm/sid;target.parent.mkdir(exist_ok=True)
            summary=build_cache(args,row,target);row['CoverageCache']=str(target);stats.append({'arm':arm,**summary})
            checks.append({'arm':arm,'subject_id':sid,'check':'trusted_vessel_fraction','value':summary['trusted_vessel_fraction'],
                'passed':summary['trusted_vessel_fraction']>=gate['trusted_vessel_fraction_min']})
            print(arm,sid,'Dice',dice,'centerline',retained,'trusted vessel',summary['trusted_vessel_fraction'],flush=True)
        with metadata.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
        (configs/(arm+'.yaml')).write_text(yaml.safe_dump(args))
        allargs[arm]=args;byarm[arm]=str(metadata)
    # Correctly labeled fine-grid overlays are separate from the export snapshots.
    sys.path.insert(0,str(REPO/'scripts/decomposition/registration_atlas'))
    from pipeline import overlays
    for arm,stage in cfg['arms'].items():
        directory=out/(arm+'_figures');directory.mkdir()
        subjects=json.loads((fine/'study.json').read_text())['subjects']
        masks=[nib.load(fine/stage/s['subject_id']/'mask.nii.gz').get_fdata()>.5 for s in subjects]
        im=nib.load(fine/stage/'probability.nii.gz')
        overlays(directory,masks,im.get_fdata(),im.affine,stage)
    # Only the data frame path changes; common constraints, objectives and budgets stay identical.
    left=copy.deepcopy(allargs['existing']);right=copy.deepcopy(allargs['prealigned'])
    left['dataset'].pop('tsv_file');right['dataset'].pop('tsv_file')
    if left!=right:raise ValueError('Unexpected non-data config differences in matched pair')
    result={'passed':all(c['passed'] for c in checks),'thresholds':gate,'checks':checks,'cache_stats':stats,'metadata':byarm,
        'config_differences':['dataset.tsv_file'],'source_snapshot':str(REPO),
        'protocol':{'subjects':4,'seed':cfg['seed'],'epochs':cfg['epochs'],'registration':'fixed preprocessing; learned rigid disabled BOTH arms',
         'graph':'disabled BOTH arms','heads':'same shared SDF/image/tissue model and losses',
         'coverage':'every target sampled only on trusted observed support; all 6 physical stencil queries remain inside support/crop',
         'sdf':'targets clipped to identical physical truncation before all distance loss terms in BOTH arms',
         'evaluation':'training reconstruction, observed aligned grid and original native full reference; no held-out generalization claim'},
        'note':'Engineering gates do not certify homologous branches or normative validity; do not automatically relax failed thresholds'}
    (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'SUMMARY.md').write_text('# Fine-grid preflight\n\nGate: '+('PASS' if result['passed'] else 'FAIL')+'\n\nSee gate.json for all subjects/checks. GPU launch is blocked if any check fails.\n')
    if not result['passed']:
        print('GPU GATE CLOSED: inspect preflight/gate.json; no automatic threshold relaxation',flush=True)
    else:print('GPU GATE PASSED: matched fit scripts may now be submitted',flush=True)
if __name__=='__main__':main()
