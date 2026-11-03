"""Post-hoc read-only checks of raster ambiguity and geometric confounding."""
import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from vesseltok.geometry import TubeGraph
from vesseltok.break_helpers import require_audit_scope,weighted_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);args=p.parse_args();root=Path(args.prepared);report=json.loads((root/'report/summary.json').read_text());rows=[]
    require_audit_scope([r['subject_id'] for r in report['protocol']['subjects']],report['protocol']['splits'])
    for record,result in zip(report['protocol']['subjects'],report['subjects']):
        sid=record['subject_id'];dest=root/'report'/sid
        with np.load(root/sid/'geometry.npz') as z:g={k:z[k] for k in z.files}
        with np.load(root/'fields_A'/sid/'probes.npz') as z:a=z['continuous_probability'];va=z['nearest_voxel_probability']
        with np.load(root/'fields_B'/sid/'probes.npz') as z:b=z['continuous_probability'];vb=z['nearest_voxel_probability']
        image=nib.load(record['MRA']);voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine),g['points_mm'])).astype(int);world=nib.affines.apply_affine(image.affine,voxel)
        tube=TubeGraph.load(record['Graph'],.5);teacher=tube.occupancy(world,1.)
        branches=list(csv.DictReader((dest/'branches.csv').open()));counts={'native_lost':0,'native_lost_continuous_B_supported':0,'supported_B_teacher_native_path_supported':0,
                 'supported_B_teacher_native_path_broken':0,'native_lost_teacher_path_supported':0,'native_lost_teacher_path_broken':0,
                 'missing_native_coarse_samples_teacher_positive':0,'missing_native_coarse_samples_teacher_negative':0}
        classification=[]
        for branch,(start,stop) in enumerate(record['bounds']):
            r=branches[branch]
            if not (r['observed']=='True' and r['native_A']=='True' and r['native_B']=='False'):continue
            sl=slice(start,stop);coarse=g['coarse'][sl];path_voxels=voxel[sl][coarse];adjacent=bool(np.all(np.abs(np.diff(path_voxels,axis=0))<=1))
            teacher_path=bool(teacher[sl][coarse].all() and adjacent);continuous_supported=r['continuous_B']=='True';missing=(vb[sl]<.5)&coarse
            counts['native_lost']+=1;counts['native_lost_teacher_path_supported' if teacher_path else 'native_lost_teacher_path_broken']+=1
            if continuous_supported:
                counts['native_lost_continuous_B_supported']+=1;counts['supported_B_teacher_native_path_supported' if teacher_path else 'supported_B_teacher_native_path_broken']+=1
            counts['missing_native_coarse_samples_teacher_positive']+=int((missing&teacher[sl]).sum());counts['missing_native_coarse_samples_teacher_negative']+=int((missing&~teacher[sl]).sum())
            classification.append({'branch_id':branch,'B_continuous_sampled_path_supported':continuous_supported,'teacher_native_path_supported':teacher_path,'coarse_voxel_sequence_26_adjacent':adjacent,
                                   'missing_teacher_positive_samples':int((missing&teacher[sl]).sum()),'missing_teacher_negative_samples':int((missing&~teacher[sl]).sum())})
        (dest/'teacher_raster_checks.json').write_text(json.dumps(classification,indent=2))
        # Same-weight near/far rates within common anchor/junction strata. No fitting.
        adjusted=[]
        for minimum in [0,10,50]:
            near=[];far=[];mass=[]
            cells=result['gap_anchor_junction_strata']
            for al in [0,1,2,4,8]:
                for jl in [0,2,5,10]:
                    n=next(c for c in cells if c['gap_lower']==0 and c['anchor_lower']==al and c['junction_lower']==jl)
                    f=next(c for c in cells if c['gap_lower']==6 and c['anchor_lower']==al and c['junction_lower']==jl)
                    weight=min(n['baseline_supported_arc_mm'],f['baseline_supported_arc_mm'])
                    if weight>minimum:near.append(n['new_loss_arc_fraction']);far.append(f['new_loss_arc_fraction']);mass.append(weight)
            adjusted.append({'minimum_common_arc_mm':minimum,'common_strata':len(mass),'common_weight_mm':sum(mass),'near_rate':float(np.average(near,weights=mass)) if mass else None,'far_rate':float(np.average(far,weights=mass)) if mass else None})
        base=g['inside_fov']&(a>=.5);margin=[]
        for low,high in zip([.5,.6,.7,.8,.9],[.6,.7,.8,.9,1.000001]):
            margin.append({'baseline_probability_lower':low,'upper':min(high,1.),**weighted_summary(g['weights_mm'],a,b,base&(a>=low)&(a<high))})
        used={}
        for arm in ['A','B']:
            with np.load(root/('fields_'+arm)/sid/'probes.npz') as z:q=z['realized_negative_probability'];used[arm]=float(q.mean()) if len(q) else None
        rows.append({'subject_id':sid,'split':record['split'],'teacher_raster':counts,'candidate_proximity_standardized_for_anchor_and_junction':adjusted,'baseline_margin_bins':margin,'realized_negative_mean_probability':used})
        print(sid,counts,flush=True)
    summary={'scope':'Additional post-hoc diagnostic checks after initial frozen audit tables; no model/sampling/threshold/cohort changes or evaluation reads. Common-stratum standardization is descriptive,not causal adjustment. Teacher is graph pseudo occupancy,not anatomical segmentation. Centerline support is sampled at.125mm,not an analytic guarantee between samples.','subjects':rows}
    (root/'report/supplement.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
