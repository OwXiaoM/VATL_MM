"""CPU-only maps and complete-path/connection transitions; no model selection."""
import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from diagnostics.provenance import sha256
from vesseltok.geometry import TubeGraph
from vesseltok.break_helpers import require_audit_scope,true_runs,pair_transitions,weighted_summary,exposure_bins

BINS={'gap_distance_mm':[0,1.25,2,3,4,6,np.inf],'used_distance_mm':[0,1.25,2,3,4,6,np.inf],
      'anchor_distance_mm':[0,1,2,4,8,np.inf],'junction_distance_mm':[0,2,5,10,np.inf],'neighbor_distance_mm':[0,2,4,6,np.inf]}


def write_csv(path,rows):
    if not rows:path.write_text('');return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def transition(a,b,valid):
    return {'both_preserved':int((valid&a&b).sum()),'A_preserved_B_broken':int((valid&a&~b).sum()),'A_broken_B_preserved':int((valid&~a&b).sum()),
            'both_broken':int((valid&~a&~b).sum()),'observed_branches':int(valid.sum()),'excluded_outside_fov':int((~valid).sum())}


def removed_islands(a,b,target,la,lb,node_a,node_b,node_target,affine,candidates,actual):
    removed=a&~b&~target;labels,n=ndi.label(removed,np.ones((3,3,3)));slices=ndi.find_objects(labels);rows=[];footprint=np.zeros(a.shape,np.uint8)
    for i,box in enumerate(slices,1):
        if box is None:continue
        expanded=tuple(slice(max(0,s.start-1),min(a.shape[k],s.stop+1)) for k,s in enumerate(box));blob=labels[expanded]==i
        coordinates=np.argwhere(blob)+np.array([s.start for s in expanded]);center=nib.affines.apply_affine(affine,coordinates.mean(0));component=int(la[tuple(coordinates[0])])
        contacts=ndi.binary_dilation(blob,np.ones((3,3,3)))&~blob&a[expanded]&b[expanded];neighbors=np.unique(lb[expanded][contacts]);neighbors=neighbors[neighbors>0]
        groups={int(c):set(map(int,node_target[(node_b==c)&(node_a==component)&(node_target>0)])) for c in neighbors}
        bearing=[c for c in neighbors if groups[int(c)]];false_witness=False;true_witness=False
        for p,c in enumerate(bearing):
            for d in bearing[:p]:
                false_witness|=any(x!=y for x in groups[int(c)] for y in groups[int(d)])
                true_witness|=bool(groups[int(c)]&groups[int(d)])
        if false_witness:footprint[tuple(coordinates.T)]=1
        rows.append({'island_id':i,'voxels':len(coordinates),'volume_mm3':float(len(coordinates)*abs(np.linalg.det(affine[:3,:3]))),
                     'x_mm':float(center[0]),'y_mm':float(center[1]),'z_mm':float(center[2]),'baseline_component':component,'adjacent_B_components':len(neighbors),
                     'source_bearing_B_components_in_same_A_component':len(bearing),'removed_excess_pair_witness':bool(false_witness),'removed_target_connection_witness':bool(true_witness),
                     'nearest_candidate_mm':float(cKDTree(candidates).query(center)[0]) if len(candidates) else None,
                     'nearest_realized_negative_mm':float(cKDTree(actual).query(center)[0]) if len(actual) else None})
    return rows,removed,footprint


def witnesses(node_points,a,b,t,original_ids):
    rows=[]
    for c in np.unique(b[b>0]):
        members=np.flatnonzero(b==c);pair=None
        for i in members:
            other=members[(t[members]!=t[i])|(t[i]==0)];other=other[other!=i]
            if len(other):pair=(int(i),int(other[0]));break
        if pair is not None:
            i,j=pair;rows.append({'B_component':int(c),'node_i':int(original_ids[i]),'node_j':int(original_ids[j]),'A_connected':bool(a[i]>0 and a[i]==a[j]),'target_i':int(t[i]),'target_j':int(t[j]),
                                 'point_i_mm':node_points[i].tolist(),'point_j_mm':node_points[j].tolist(),'scope':'Witness node pair,not an asserted straight false edge'})
    return rows


def figures(dest,record,g,pa,pb,new,branch_rows,islands):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lost=[r for r in branch_rows if r['observed'] and r['native_A'] and not r['native_B']][:8]
    if lost:
        fig,axes=plt.subplots(len(lost),2,figsize=(12,2.5*len(lost)),squeeze=False)
        for row,r in enumerate(lost):
            sl=slice(*record['bounds'][r['branch_id']]);arc=g['arc_mm'][sl];axes[row,0].plot(arc,pa[sl],label='A');axes[row,0].plot(arc,pb[sl],label='B');axes[row,0].axhline(.5,c='k',ls='--');axes[row,0].set(title='Lost native branch '+str(r['branch_id']),ylabel='Occupancy probability',ylim=(0,1));axes[row,0].legend()
            for key,label in [('gap_distance_mm','Gap candidates'),('anchor_distance_mm','FPS'),('junction_distance_mm','Junction'),('neighbor_distance_mm','Nonincident branch (within6mm)')]:
                values=g[key][sl].copy()
                if key=='neighbor_distance_mm':values[~np.isfinite(values)]=np.nan
                axes[row,1].plot(arc,np.minimum(values,12),label=label)
            axes[row,1].set(xlabel='Arc mm',ylabel='Distance mm (capped12 for display)');axes[row,1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(dest/'lost_branch_profiles.png',dpi=130);plt.close(fig)
    import plotly.graph_objects as go
    points=g['points_mm'];valid=g['inside_fov'];idx=np.flatnonzero(valid)[::max(1,int(valid.sum())//40000)];c=g['gap_candidates_mm']
    fig=go.Figure()
    for name,p,color,size in [('Source paths',points[idx],'gray',1),('New A-supported/B-broken field samples',points[new],'red',2),('Gap candidates',c,'orange',1)]:
        fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name=name,marker={'size':size,'color':color,'opacity':.5}))
    bridge=[r for r in islands if r['removed_excess_pair_witness']]
    if bridge:
        p=np.array([[r['x_mm'],r['y_mm'],r['z_mm']] for r in bridge]);fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',name='Candidate removed-bridge islands',marker={'size':4,'color':'blue'}))
    fig.update_layout(title=record['split']+' '+record['subject_id']+' (diagnostic geometry;not anatomical GT)',scene={'aspectmode':'data','xaxis_title':'RAS x mm','yaxis_title':'RAS y mm','zaxis_title':'RAS z mm'})
    fig.write_html(str(dest/'break_map_3d.html'),include_plotlyjs=True)


def analyze(root,record,dest):
    sid=record['subject_id'];dest.mkdir();tube=TubeGraph.load(record['Graph'],.5);assert sha256(record['Graph'])==record['graph_sha256']
    with np.load(root/sid/'geometry.npz') as data:g={k:data[k] for k in data.files}
    probes={};prob={};labs={};tokens={}
    for arm in ['A','B']:
        p=root/('fields_'+arm)/sid
        with np.load(p/'probes.npz') as z:probes[arm]={k:z[k] for k in z.files}
        image=nib.load(p/'occupancy_probability.nii.gz');prob[arm]=image.get_fdata(dtype=np.float32);labs[arm]=ndi.label(prob[arm]>=.5,np.ones((3,3,3)))[0]
        tokens[arm]=json.loads((p/'tokens.json').read_text())['patch_tokens'][0]
    for key in ['center_mm','anchors_local','padding']:assert tokens['A'][key]==tokens['B'][key]
    assert np.array_equal(probes['A']['anchor_distance_mm'],probes['B']['anchor_distance_mm']);g['anchor_distance_mm']=probes['A']['anchor_distance_mm']
    points=g['points_mm'];weights=g['weights_mm'];pa=probes['A']['continuous_probability'];pb=probes['B']['continuous_probability'];inside=g['inside_fov'];base=inside&(pa>=.5);new=base&(pb<.5)
    voxel=np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine),points)).astype(int);hit={a:np.zeros(len(points),int) for a in ['A','B']}
    for a in hit:hit[a][inside]=labs[a][tuple(voxel[inside].T)]
    branches=[];intervals=[];observed=[];native={a:[] for a in ['A','B']};continuous={a:[] for a in ['A','B']}
    for branch,(start,stop) in enumerate(record['bounds']):
        sl=slice(start,stop);valid=bool(inside[sl].all());observed.append(valid);row={'branch_id':branch,'observed':valid,'length_mm':float(weights[sl].sum())}
        for a,cp in [('A',pa),('B',pb)]:
            h=hit[a][sl];c=g['coarse'][sl];preserved=bool((h[c]>0).all() and h[0]>0 and h[0]==h[-1]);native[a].append(preserved);continuous[a].append(bool((cp[sl]>=.5).all()))
            row.update({'native_'+a:preserved,'continuous_'+a:continuous[a][-1],'minimum_probability_'+a:float(cp[sl].min()),'coarse_coverage_'+a:float((h[c]>0).mean())})
        row['B_has_continuous_gap']=not row['continuous_B'];row['uncertain_threshold_margin']=min(abs(row['minimum_probability_A']-.5),abs(row['minimum_probability_B']-.5))<1e-5
        branches.append(row)
        if not valid:continue
        for kind,mask in [('new_continuous_loss',new[sl]),('new_voxel_loss',(hit['A'][sl]>0)&(hit['B'][sl]==0))]:
            for begin,end in true_runs(mask):
                ids=np.arange(start+begin,start+end);mid=ids[len(ids)//2];entry={'branch_id':branch,'kind':kind,'baseline_native_branch_preserved':row['native_A'],'B_native_branch_preserved':row['native_B'],
                     'arc_start_mm':float(g['arc_mm'][ids[0]]),'arc_stop_mm':float(g['arc_mm'][ids[-1]]),'weighted_arc_mm':float(weights[ids].sum()),'points':len(ids),
                     'x_mm':float(points[mid,0]),'y_mm':float(points[mid,1]),'z_mm':float(points[mid,2]),'mean_probability_A':float(pa[ids].mean()),'mean_probability_B':float(pb[ids].mean())}
                for key in BINS:
                    values=g[key][ids];finite=values[np.isfinite(values)];entry['minimum_'+key]=float(finite.min()) if len(finite) else None
                intervals.append(entry)
    observed=np.array(observed);native={a:np.array(v) for a,v in native.items()};continuous={a:np.array(v) for a,v in continuous.items()}
    exposure={key:exposure_bins(g[key],edges,weights,pa,pb,base) for key,edges in BINS.items()}
    contrasts={key:{'near_le3mm':weighted_summary(weights,pa,pb,base&(g[key]<=3)), 'far_ge6mm':weighted_summary(weights,pa,pb,base&(g[key]>=6))} for key in ['gap_distance_mm','used_distance_mm']}
    strata=[]
    for gl,gh in [(0,3),(3,6),(6,np.inf)]:
        for al,ah in zip([0,1,2,4,8],[1,2,4,8,np.inf]):
            for jl,jh in zip([0,2,5,10],[2,5,10,np.inf]):
                keep=base&(g['gap_distance_mm']>=gl)&(g['gap_distance_mm']<gh)&(g['anchor_distance_mm']>=al)&(g['anchor_distance_mm']<ah)&(g['junction_distance_mm']>=jl)&((g['junction_distance_mm']<jh) if np.isfinite(jh) else True)
                strata.append({'gap_lower':gl,'gap_upper':None if np.isinf(gh) else gh,'anchor_lower':al,'anchor_upper':None if np.isinf(ah) else ah,'junction_lower':jl,'junction_upper':None if np.isinf(jh) else jh,**weighted_summary(weights,pa,pb,keep)})
    # Independent native pseudo-target and source-node connectivity. No Seg/SDF.
    target=np.empty(image.shape,bool)
    for z in range(image.shape[2]):
        ij=np.indices(image.shape[:2],dtype=np.float32).reshape(2,-1).T;ijk=np.c_[ij,np.full(len(ij),z)];world=nib.affines.apply_affine(image.affine,ijk);target[:,:,z]=tube.occupancy(world,1.).reshape(image.shape[:2])
    target_labels=ndi.label(target,np.ones((3,3,3)))[0];nv=np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine),tube.points)).astype(int);ni=np.all((nv>=0)&(nv<np.array(image.shape)),axis=1);v=nv[ni]
    na=labs['A'][tuple(v.T)];nb=labs['B'][tuple(v.T)];nt=target_labels[tuple(v.T)];source=np.zeros(len(tube.points),int)
    for label,nodes in enumerate(nx.connected_components(tube.graph),1):source[list(nodes)]=label
    connections={'relative_to_pseudo':pair_transitions(na,nb,nt),'relative_to_source_components':pair_transitions(na,nb,source[ni]),'observed_nodes':int(ni.sum()),'outside_fov_nodes':int((~ni).sum())}
    islands,removed,bridges=removed_islands(prob['A']>=.5,prob['B']>=.5,target,labs['A'],labs['B'],na,nb,nt,image.affine,g['gap_candidates_mm'],g['realized_negatives_mm'])
    for name,mask in [('removed_pseudo_false_foreground',removed),('candidate_removed_bridge_footprints',bridges)]:
        im=nib.Nifti1Image(mask.astype(np.uint8),image.affine);im.header.set_xyzt_units('mm');nib.save(im,dest/(name+'.nii.gz'))
    retained=witnesses(tube.points[ni],na,nb,nt,np.flatnonzero(ni));(dest/'retained_excess_pair_witnesses.json').write_text(json.dumps(retained,indent=2))
    write_csv(dest/'branches.csv',branches);write_csv(dest/'lost_intervals.csv',intervals);write_csv(dest/'removed_false_foreground_islands.csv',islands)
    np.savez_compressed(dest/'localized_points.npz',points_mm=points,weights_mm=weights,branch_id=g['branch_id'],inside_fov=inside,new_continuous_loss=new,
                        probability_A=pa,probability_B=pb,**{key:g[key] for key in BINS})
    lost=observed&native['A']&~native['B'];voxelization=lost&continuous['B'];field_gap=lost&~continuous['B']
    total=float(weights[inside].sum());baseline=float(weights[base].sum())
    result={'subject_id':sid,'split':record['split'],'native_branch_transitions':transition(native['A'],native['B'],observed),'continuous_branch_transitions':transition(continuous['A'],continuous['B'],observed),
            'native_lost_B_has_field_gap':int(field_gap.sum()),'native_lost_B_continuously_supported':int(voxelization.sum()),'uncertain_threshold_branches':sum(r['uncertain_threshold_margin'] for r in branches if r['observed']),
            'observed_arc_mm':total,'baseline_supported_arc_mm':baseline,'new_loss':weighted_summary(weights,pa,pb,base),'global_probability_delta':float(np.dot(weights[inside],(pb-pa)[inside])/total),
            'exposure':exposure,'near_far':contrasts,'gap_anchor_junction_strata':strata,'connections':connections,
            'removed_false_foreground_islands':len(islands),'candidate_removed_bridge_islands':sum(r['removed_excess_pair_witness'] for r in islands),'retained_excess_component_witnesses':len(retained),
            'candidate_count':len(g['gap_candidates_mm']),'realized_negative_draws':len(g['realized_negatives_mm']),'neighbor_distance_censored_arc_mm':float(weights[inside&np.isinf(g['neighbor_distance_mm'])].sum()),
            'gap_mean_probability_A':float(probes['A']['gap_probability'].mean()),'gap_mean_probability_B':float(probes['B']['gap_probability'].mean()),
            'scope':'Training actual cached inputs;validation selected saved native inputs. Candidate/used proximity is observational,not causal.'}
    if record['split']=='validation' or sid==json.loads((root/'manifest.json').read_text())['subjects'][0]['subject_id']:figures(dest,record,g,pa,pb,new,branches,islands)
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',required=True);args=parser.parse_args();root=Path(args.prepared);manifest=json.loads((root/'manifest.json').read_text());out=root/'report';out.mkdir(exist_ok=False)
    require_audit_scope([r['subject_id'] for r in manifest['subjects']],manifest['splits']);fields={a:json.loads((root/('fields_'+a)/'manifest.json').read_text()) for a in ['A','B']}
    for a,f in fields.items():assert f['parameter_sha256_before']==f['parameter_sha256_after'] and f['checkpoint_sha256']==manifest['checkpoint_sha256'][a]
    rows=[]
    for record in manifest['subjects']:
        rows.append(analyze(root,record,out/record['subject_id']));print(record['split'],record['subject_id'],rows[-1]['native_branch_transitions'],flush=True)
    result={'subjects':rows,'fields':fields,'protocol':manifest,'segmentation_or_sdf_read':False,'final_evaluation_read':False,'updates':False,'macro':[]}
    for group in ['train','validation']:
        rr=[r for r in rows if r['split']==group];m={'split':group,'subjects':len(rr)}
        for mode in ['native','continuous']:
            for status in ['A_preserved_B_broken','A_broken_B_preserved','both_preserved','both_broken']:
                m[mode+'_'+status+'_fraction']=float(np.mean([r[mode+'_branch_transitions'][status]/r[mode+'_branch_transitions']['observed_branches'] for r in rr]))
        for key in ['native_lost_B_has_field_gap','native_lost_B_continuously_supported','candidate_removed_bridge_islands','retained_excess_component_witnesses']:m[key+'_sum']=sum(r[key] for r in rr)
        m['new_loss_arc_fraction']=float(np.mean([r['new_loss']['new_loss_arc_fraction'] for r in rr]));m['global_probability_delta']=float(np.mean([r['global_probability_delta'] for r in rr]))
        for exposure in ['gap_distance_mm','used_distance_mm']:
            m[exposure]={}
            for region in ['near_le3mm','far_ge6mm']:
                values=[r['near_far'][exposure][region]['new_loss_arc_fraction'] for r in rr];values=[v for v in values if v is not None]
                m[exposure][region]={'subjects_with_support':len(values),'mean_new_loss_arc_fraction':float(np.mean(values)) if values else None,'total_baseline_supported_arc_mm':sum(r['near_far'][exposure][region]['baseline_supported_arc_mm'] for r in rr)}
        result['macro'].append(m)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['macro'],indent=2),flush=True)


if __name__=='__main__':main()
