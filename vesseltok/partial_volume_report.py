"""Local target-validation report and training gate;never submits training."""
import argparse
import json
import math
from pathlib import Path
import shutil
import numpy as np
from diagnostics.provenance import sha256



def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cases',nargs=4,required=True);p.add_argument('--synthetic',required=True);p.add_argument('--plan',required=True);args=p.parse_args();cases=list(map(Path,args.cases));synthetic_root=Path(args.synthetic);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    subjects=[]
    for source in cases:
        r=json.loads((source/'result.json').read_text());subjects.append(r);dest=out/r['subject_id'];dest.mkdir()
        for path in source.iterdir():
            if path.is_file():shutil.copy2(path,dest/path.name)
    synthetic=json.loads((synthetic_root/'result.json').read_text());shutil.copytree(synthetic_root,out/'synthetic')
    gap=next(x for x in synthetic['cases'] if x['kind']=='separation' and x['gap_mm']==.1);a=.25;c=1.05
    analytic_gap=2*((a-c)*2*a+a*math.sqrt(1-a*a)+math.asin(a))/(4*a*a)
    gap['analytic_midgap_fraction']=analytic_gap
    separation_failed=gap['witness'][-1]['lower']>.5 and gap['metrics']['certified_lower']['paths']['false_connections_cross_component_node_pairs']>0
    small=[x for x in synthetic['cases'] if x['kind']=='small_vessel'];small_failed=all(x['fraction_peak_upper']<.5 and x['optimistic_fraction_paths']['source_branches_preserved']==0 for x in small)
    numerical=all(x['numeric_gate_passed'] for x in subjects) and all(abs(x['witness'][-1]['fraction']-x['analytic_center_fraction'])<=.002 for x in small)
    numerical&=synthetic['capsule_volume'][-1]['relative_error']<.01
    real_gate=all(x['metrics']['fraction_certified_lower']['source_native_paths']['source_branches_preserved']==x['metrics']['hard_point']['source_native_paths']['source_branches_preserved'] and x['metrics']['fraction_optimistic_upper']['connections_vs_point_target']['excess_connected_pairs_vs_target']==0 and x['source_continuous_certified_supported']==x['source_continuous_observed_paths'] for x in subjects)
    decision={'real_target_gate_passed':bool(real_gate),'numerical_gate_passed_after_refinement':bool(numerical),'synthetic_separation_gate_passed':not separation_failed,'small_vessel_path_gate_passed':not small_failed,
              'training_supported':bool(numerical and real_gate and not separation_failed and not small_failed),'training_launched':False,'reason':'Certified representation/target-semantics failure at fixed.5,not corrected by more subvoxel samples.',
              'final_evaluation_read':False,'segmentation_or_sdf_read':False,'threshold':.5,'activation_changed':False,'plan_sha256':sha256(args.plan)}
    assert numerical,'Numerical failure must be investigated before interpreting representation gates'
    assert not decision['training_supported'],'Review conditional training protocol before any fitting;this reporter never submits jobs'
    summary={'decision':decision,'subjects':subjects,'synthetic':synthetic,'source_paths':list(map(str,cases))+[str(synthetic_root)]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    from vesseltok.partial_volume_plot import plot_targets
    plot_targets(out,out/'synthetic_target_failures.png')
    print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
