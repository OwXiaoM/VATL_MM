import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import networkx as nx
import nibabel as nib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vesseltok.geometry import TubeGraph,world_to_local,local_to_world,rigid_world,sample_queries,graph_alignment
from vesseltok.model import GraphOccupancyAE,farthest_points
from vesseltok.topology import betti,branch_metrics,source_branches,reconstruction_metrics
from vesseltok.data import cohorts


class GeometryTests(unittest.TestCase):
    def test_exact_edge_capsules_include_long_edges_and_isolated_nodes(self):
        graph=TubeGraph([[0,0,0],[100,0,0],[0,10,0]],[[0,1]],spacing_mm=1.)
        query=np.array([[50,.9,0],[50,1.1,0],[-.9,0,0],[100.9,0,0],[0,10,.5],[0,12,0]])
        np.testing.assert_array_equal(graph.occupancy(query,1.),[True,False,True,True,True,False])

    def test_rotation_translation_inverse_physical_distance(self):
        rng=np.random.default_rng(1);points=rng.normal(size=(30,3))*[100,50,20]
        center=np.array([12.,-4.,3.]);translation=np.array([5.,-3.,2.]);rotation=Rotation.from_rotvec([.3,-.2,.5]).as_matrix()
        mapped=rigid_world(points,rotation,translation,center)
        recovered=rigid_world(mapped,rotation.T,-rotation.T@translation,center)
        np.testing.assert_allclose(recovered,points,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(mapped[1:]-mapped[:-1],axis=1),np.linalg.norm(points[1:]-points[:-1],axis=1))
        np.testing.assert_allclose(local_to_world(world_to_local(points,center,16),center,16),points)
        with self.assertRaises(ValueError):rigid_world(points,np.diag([1,1,2]),translation,center)

    def test_occupancy_rigid_equivariance(self):
        points=np.array([[0,0,0],[5,0,0],[5,5,0]],float);q=np.random.default_rng(1).uniform(-2,7,(100,3))
        rotation=Rotation.from_rotvec([.4,.3,-.2]).as_matrix();t=np.array([2.,4.,-7.]);center=points.mean(0)
        before=TubeGraph(points,[[0,1],[1,2]])
        after=TubeGraph(rigid_world(points,rotation,t,center),[[0,1],[1,2]])
        np.testing.assert_array_equal(before.occupancy(q,1),after.occupancy(rigid_world(q,rotation,t,center),1))

    def test_sampling_is_reproducible_and_graph_supervised(self):
        graph=TubeGraph([[0,0,-5],[0,0,5]],[[0,1]])
        first=sample_queries(graph,np.zeros(3),8,1,2000,np.random.default_rng(42))
        second=sample_queries(graph,np.zeros(3),8,1,2000,np.random.default_rng(42))
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(first[1].astype(bool),graph.occupancy(first[0],1))
        self.assertGreater(first[1].mean(),.3);self.assertLess(first[1].mean(),.8)

    def test_input_geometry_moments_transform_as_physical_covariance(self):
        from vesseltok.model import local_moments
        direction=np.array([.3,.5,.8]);direction/=np.linalg.norm(direction)
        points=torch.tensor(np.linspace(-4,4,31)[:,None]*direction,dtype=torch.float64)[None]
        anchors=torch.zeros(1,1,3,dtype=torch.float64);padding=torch.zeros(1,31,dtype=torch.bool)
        first=local_moments(points,anchors,padding,2.)[0,0]
        def matrix(v):return torch.stack((torch.stack((v[0],v[3],v[4])),torch.stack((v[3],v[1],v[5])),torch.stack((v[4],v[5],v[2]))))
        rotation=torch.tensor(Rotation.from_rotvec([.2,-.4,.1]).as_matrix(),dtype=torch.float64)
        translation=torch.tensor([5.,-3.,2.],dtype=torch.float64)
        after=local_moments(points@rotation.T+translation,anchors@rotation.T+translation,padding,2.)[0,0]
        torch.testing.assert_close(matrix(after),rotation@matrix(first)@rotation.T)

    def test_perpendicular_boundary_sampling_is_balanced(self):
        tube=TubeGraph([[0,0,-7],[0,0,7]],[[0,1]])
        _,labels=sample_queries(tube,np.zeros(3),8,1,10000,np.random.default_rng(42),strategy='perpendicular_boundary')
        self.assertLess(abs(labels.mean()-.5),.03)

    def test_voxel_world_alignment_and_sdf_sign(self):
        affine=np.diag([.7,1.2,2.,1.]);affine[:3,3]=[10,-3,4]
        values=np.zeros((10,10,10),np.float32);values[4,4,2:8]=1
        image=nib.Nifti1Image(values,affine);image.header.set_xyzt_units('mm')
        sdf=nib.Nifti1Image(np.where(values>0,-1.,1.),affine);sdf.header.set_xyzt_units('mm')
        points=nib.affines.apply_affine(affine,np.array([[4,4,2],[4,4,7]]))
        result=graph_alignment(TubeGraph(points,[[0,1]]),image,image,sdf)
        self.assertEqual(result['nodes_inside_image_fraction'],1.)
        self.assertEqual(result['node_seg_inside_fraction'],1.);self.assertEqual(result['node_sdf_inside_fraction'],1.)


class ModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def model(self,**kwargs):return GraphOccupancyAE(width=24,latent_dim=4,tokens=8,heads=4,encoder_layers=1,decoder_layers=1,bands=3,**kwargs)
    def test_padding_empty_and_fps(self):
        points=torch.rand(2,16,3);padding=torch.zeros(2,16,dtype=torch.bool);padding[1,3:]=True
        anchors,mask=farthest_points(points,padding,8)
        self.assertEqual(mask[1].sum(),5)
        padding[0]=True
        with self.assertRaises(ValueError):farthest_points(points,padding,8)
    def test_encoder_decoder_gradients_kl_and_no_transforms(self):
        model=self.model();cloud=torch.rand(2,16,3);padding=torch.zeros(2,16,dtype=torch.bool);queries=torch.rand(2,24,3)
        output=model(cloud,padding,queries)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(output['logits'],torch.rand(2,24))+1e-3*output['kl'];loss.backward()
        self.assertGreater(output['kl'].item(),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertFalse(any('transform' in key for key in model.state_dict()))
        self.assertGreater(model.encoder_cross.attention.in_proj_weight.grad.abs().sum(),0)
        self.assertGreater(model.decoder_cross.attention.in_proj_weight.grad.abs().sum(),0)
    def test_deterministic_chunked_inference_and_input_sensitivity(self):
        model=self.model().eval();cloud=torch.rand(1,20,3);padding=torch.zeros(1,20,dtype=torch.bool);queries=torch.rand(1,25,3)
        with torch.no_grad():
            first=model.encode(cloud,padding);other=model.encode(cloud+.2,padding)
            output=model.decode(queries,first)
            chunks=torch.cat([model.decode(queries[:,i:i+7],first) for i in range(0,25,7)],1)
            torch.testing.assert_close(output,chunks)
            self.assertFalse(torch.allclose(output,model.decode(queries,other)))
            torch.testing.assert_close(first['tokens'],model.encode(cloud,padding)['tokens'],rtol=0,atol=0)
    def test_optional_known_conditions_are_detached(self):
        model=self.model(condition_dim=2);conditions=torch.tensor([[.01,-.05]],requires_grad=True)
        model(torch.rand(1,16,3),torch.zeros(1,16,dtype=torch.bool),torch.rand(1,10,3),conditions)['logits'].sum().backward()
        self.assertIsNone(conditions.grad)
    def test_decoder_uses_tokens_not_original_points_or_fps_anchors(self):
        model=self.model().eval();cloud=torch.rand(1,16,3);padding=torch.zeros(1,16,dtype=torch.bool);queries=torch.rand(1,10,3)
        with torch.no_grad():
            encoded=model.encode(cloud,padding);first=model.decode(queries,encoded)
            encoded['anchors'].fill_(12345)
            torch.testing.assert_close(first,model.decode(queries,encoded),rtol=0,atol=0)

    def test_deterministic_autoencoder_mode_has_no_kl_or_posterior_noise(self):
        model=self.model(variational=False,decoder_query_residual=False,cross_heads=1,output_layer_norm=False)
        cloud=torch.rand(1,16,3);padding=torch.zeros(1,16,dtype=torch.bool);query=torch.rand(1,20,3)
        first=model(cloud,padding,query);second=model(cloud,padding,query)
        torch.testing.assert_close(first['logits'],second['logits'],rtol=0,atol=0)
        self.assertEqual(float(first['kl']),0.)
        self.assertIsInstance(model.output,torch.nn.Linear)
        first['logits'].square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_paper_style_decoder_and_graph_free_token_roundtrip(self):
        from vesseltok.infer import predict_subject
        from vesseltok.decode_tokens import render_tokens
        model=self.model(decoder_query_residual=False,cross_heads=1).eval()
        image=nib.Nifti1Image(np.ones((8,8,8),np.float32),np.eye(4));image.header.set_xyzt_units('mm')
        tube=TubeGraph([[4,4,1],[4,4,6]],[[0,1]])
        config={'representation':'subject_tokens','patch_mm':16.,'halo_mm':3.,'input_points':32,'seed':42}
        with torch.no_grad():
            probability,records=predict_subject(model,tube,image,config,'cpu',chunk=71)
            reconstructed=render_tokens(model,records,image,config,'cpu',chunk=37)
        np.testing.assert_allclose(probability,reconstructed,atol=1e-6)
        self.assertFalse(hasattr(model.decoder_cross,'ff'))

    def test_spatial_tokens_have_learned_encoder_decoder_gradients(self):
        model=self.model(spatial_anchors=True,attention_scale=.1,relative_feature_scale=.03,geometric_moments=True)
        output=model(torch.rand(1,16,3),torch.zeros(1,16,dtype=torch.bool),torch.rand(1,20,3))
        (output['logits'].square().mean()+.001*output['kl']).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            encoded=model.encode(torch.rand(1,16,3),torch.zeros(1,16,dtype=torch.bool));queries=torch.rand(1,11,3)
            first=model.decode(queries,encoded);zero=dict(encoded);zero['tokens']=torch.zeros_like(encoded['tokens'])
            self.assertFalse(torch.allclose(first,model.decode(queries,zero)))
            chunked=torch.cat([model.decode(queries[:,i:i+3],encoded) for i in range(0,11,3)],1)
            torch.testing.assert_close(first,chunked)

    def test_subject_and_blended_patch_inference(self):
        from vesseltok.infer import predict_subject
        image=nib.Nifti1Image(np.ones((8,8,8),np.float32),np.eye(4));image.header.set_xyzt_units('mm')
        tube=TubeGraph([[4,4,0],[4,4,7]],[[0,1]])
        config={'representation':'subject_tokens','patch_mm':16.,'halo_mm':3.,'input_points':32,'seed':42}
        model=self.model().eval()
        with torch.no_grad():
            first,tokens=predict_subject(model,tube,image,config,'cpu',chunk=73)
            second,_=predict_subject(model,tube,image,config,'cpu',chunk=41)
        np.testing.assert_allclose(first,second,atol=1e-6);self.assertEqual(len(tokens),1)
        class ConstantField:
            condition_dim=0
            def encode(self,*args,**kwargs):return {'mean':torch.zeros(1,2,1),'anchors':torch.zeros(1,2,3),'padding':torch.zeros(1,2,dtype=torch.bool)}
            def prepare_decoder(self,*args):return None
            def decode(self,queries,*args):return torch.zeros(queries.shape[:2])
        config.pop('representation');config['patch_mm']=6.;config['inference_stride_mm']=4.
        blended,records=predict_subject(ConstantField(),tube,image,config,'cpu',chunk=41)
        np.testing.assert_allclose(blended,.5,atol=1e-6)
        from vesseltok.decode_tokens import render_tokens
        np.testing.assert_allclose(blended,render_tokens(ConstantField(),records,image,config,'cpu'),atol=1e-6)
        affine=np.eye(4);affine[:3,:3]=Rotation.from_rotvec([0,0,.5]).as_matrix()@np.diag([.7,1.2,2.])
        oblique=nib.Nifti1Image(np.ones((8,8,8),np.float32),affine);oblique.header.set_xyzt_units('mm')
        points=nib.affines.apply_affine(affine,[[4,4,0],[4,4,7]])
        prediction,records=predict_subject(ConstantField(),TubeGraph(points,[[0,1]]),oblique,config,'cpu')
        np.testing.assert_allclose(prediction,render_tokens(ConstantField(),records,oblique,config,'cpu'),atol=1e-6)


    def test_small_graph_padding_invariance(self):
        model=self.model().eval();cloud=torch.rand(1,16,3);padding=torch.zeros(1,16,dtype=torch.bool);padding[:,4:]=True
        other=cloud.clone();other[:,4:]=10000
        with torch.no_grad():
            torch.testing.assert_close(model.encode(cloud,padding)['tokens'],model.encode(other,padding)['tokens'])


class DataTests(unittest.TestCase):
    def test_scoring_independently_rejects_training_person(self):
        import json
        from vesseltok.evaluate import main
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'manifest.json').write_text(json.dumps({'training_subjects':['IXI001-A'],'subjects':['IXI001-B']}));(root/'config.yaml').write_text('{}')
            with patch('sys.argv',['evaluate','--predictions',str(root),'--config',str(root/'config.yaml')]),self.assertRaises(ValueError):main()

    def test_preparation_needs_no_segmentation_or_sdf(self):
        import pickle
        import pandas as pd
        import yaml
        from vesseltok.data import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); image=nib.Nifti1Image(np.ones((12,12,12),np.float32),np.eye(4));image.header.set_xyzt_units('mm');nib.save(image,root/'mra.nii.gz')
            graph=nx.Graph();graph.add_node('a',pos=np.array([6,6,3.]));graph.add_node('b',pos=np.array([6,6,9.]));graph.add_edge('a','b',weight=6.)
            with open(root/'graph.pkl','wb') as stream:pickle.dump(graph,stream)
            frame=pd.DataFrame([{'subject_id':key,'scan_age':30.,'sex':'F','MRA':str(root/'mra.nii.gz'),'Graph':str(root/'graph.pkl')} for key in ['P1','P2','P3']]);frame.to_csv(root/'metadata.tsv',sep='\t',index=False)
            (root/'ids.yaml').write_text(yaml.safe_dump({'dataset':{'subject_ids':{'train':['P1'],'val':['P2','P3'],'test':[]}}}))
            config={'manifest':str(root/'ids.yaml'),'metadata':str(root/'metadata.tsv'),'seed':42,'train_subjects':1,'evaluation_subjects':1,'validation_subjects':1,
                    'edge_spacing_mm':.5,'pseudo_radius_mm':1.,'patch_mm':8.,'halo_mm':2.,'input_points':32,'patches_per_subject':2,'validation_patches':2,'query_pool':64,'sampling_strategy':'perpendicular_boundary'}
            prepare(config,root/'prepared')
            cache=np.load(root/'prepared/train.npz');self.assertEqual(cache['cloud'].shape,(2,32,3));self.assertTrue(np.isfinite(cache['queries']).all())
            from vesseltok.data import PatchSet
            sample=PatchSet(root/'prepared/train.npz',20,balanced=True)[0]
            self.assertEqual(int(sample['labels'].sum()),10)



class TopologyTests(unittest.TestCase):
    def test_betti_components_loop_and_cavity(self):
        mask=np.zeros((15,15,15),bool);mask[3:6,3:6,3:6]=True;mask[9:12,9:12,9:12]=True
        self.assertEqual(betti(mask)['betti0_26_connected'],2)
        ring=np.zeros_like(mask);ring[3:12,3:12,7]=True;ring[5:10,5:10,7]=False
        self.assertEqual(betti(ring)['betti1'],1)
        shell=np.zeros_like(mask);shell[3:12,3:12,3:12]=True;shell[5:10,5:10,5:10]=False
        self.assertEqual(betti(shell)['betti2'],1)
    def test_broken_branch_and_false_connection(self):
        points=np.array([[2,3,3],[10,3,3],[2,6,3],[10,6,3]])
        tube=TubeGraph(points,[[0,1],[2,3]])
        mask=np.zeros((14,12,8),bool);mask[2:11,3,3]=True;mask[2:11,6,3]=True
        metric=branch_metrics(mask,tube,np.eye(4));self.assertEqual(metric['source_branches_preserved'],2);self.assertEqual(metric['false_connections_cross_component_node_pairs'],0)
        mask[6,3:7,3]=True
        self.assertGreater(branch_metrics(mask,tube,np.eye(4))['false_connections_cross_component_node_pairs'],0)
        mask[7,3,3]=False
        self.assertGreater(branch_metrics(mask,tube,np.eye(4))['source_branches_broken'],0)
    def test_perfect_metrics(self):
        mask=np.zeros((15,15,15),bool);mask[7,7,2:13]=True
        result=reconstruction_metrics(mask,mask,np.diag([1,2,3,1]))
        self.assertEqual(result['dice'],1.);self.assertEqual(result['iou'],1.);self.assertEqual(result['betti1_absolute_error'],0)
        self.assertEqual(result['centerline_chamfer_symmetric_mean_mm'],0.)


if __name__=='__main__':unittest.main()


class ThicknessTests(unittest.TestCase):
    def test_default_objective_and_false_positive_gradient(self):
        from vesseltok.objectives import occupancy_loss
        logits=torch.tensor([[0.,0.]],requires_grad=True);labels=torch.tensor([[1.,0.]])
        base=occupancy_loss(logits,labels)
        torch.testing.assert_close(base,torch.nn.functional.binary_cross_entropy_with_logits(logits,labels))
        weighted=occupancy_loss(logits,labels,2.);weighted.backward()
        torch.testing.assert_close(logits.grad,torch.tensor([[-.25,.5]]))
        with self.assertRaises(ValueError):occupancy_loss(logits,labels,0.)

    def test_calibration_excludes_evaluation_and_training(self):
        from vesseltok.calibrate import require_validation
        splits={'train':['IXI001-A'],'validation':['IXI002-B'],'evaluation':['IXI003-C']}
        require_validation(['IXI002-B'],splits)
        for subject in ['IXI001-A','IXI003-C']:
            with self.assertRaises(ValueError):require_validation([subject],splits)

    def test_threshold_objective_uses_macro_subject_dice(self):
        from vesseltok.calibrate import threshold_scores,choose_threshold
        rows=[threshold_scores(np.array([.9,.6,.1]),np.array([True,False,False]),[.5,.8])]
        threshold,means=choose_threshold(rows)
        self.assertEqual(threshold,.8);self.assertEqual(means,[2/3,1.])
        with self.assertRaises(ValueError):threshold_scores(np.array([np.nan]),np.array([False]))

    def test_export_mesh_respects_calibrated_threshold(self):
        from vesseltok.evaluate import export_geometry
        field=np.broadcast_to(np.linspace(0,1,5)[:,None,None],(5,5,5)).copy()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            export_geometry(field>=.75,np.eye(4),root,field,threshold=.75)
            lines=(root/'vessel_surface.ply').read_text().splitlines()
            count=int(next(s for s in lines if s.startswith('element vertex')).split()[-1])
            offset=lines.index('end_header')+1
            points=np.array([[float(v) for v in s.split()] for s in lines[offset:offset+count]])
            np.testing.assert_allclose(points[:,0],3.,atol=1e-6)

    def test_input_audit_regenerates_physical_anchor_locations(self):
        import json
        import pandas as pd
        from vesseltok.input_audit import audit_subject
        graph=TubeGraph([[1,2,2],[3,2,2]],[[0,1]],spacing_mm=.5)
        config={'seed':42,'patch_mm':16.,'halo_mm':0.,'input_points':16,'edge_spacing_mm':.5,'representation':'subject_tokens'}
        center=np.array([2.5,2.5,2.5]);cloud=graph.patch_cloud(center,8,0,16,np.random.default_rng(42))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'IXI001-A').mkdir();image=root/'image.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((5,5,5)),np.eye(4)),image)
            tokens={'patch_tokens':[{'center_mm':center.tolist(),'anchors_local':((cloud-center)/8).tolist(),'padding':[False]*len(cloud)}]}
            (root/'IXI001-A'/'tokens.json').write_text(json.dumps(tokens))
            metadata=pd.DataFrame([{'subject_id':'IXI001-A','MRA':str(image),'Graph':'mock'}]).set_index('subject_id')
            with patch('vesseltok.input_audit.TubeGraph.load',return_value=graph):result=audit_subject(root,'IXI001-A',config,metadata)
            self.assertEqual(result['input_point_proximity']['mean_mm'],0.)
            self.assertEqual(result['fps_anchor_proximity']['fraction_within_1mm'],1.)


class InputDensityTests(unittest.TestCase):
    def test_nested_cloud_contains_exact_baseline_without_changing_default(self):
        from vesseltok.input_sampling import nested_cloud,subject_cloud
        from scipy.spatial import cKDTree
        graph=TubeGraph([[0,0,0],[10,0,0]],[[0,1]],spacing_mm=.1)
        center=np.array([5.,0,0]);seed=42
        baseline=graph.patch_cloud(center,8,0,8,np.random.default_rng(seed))
        small=nested_cloud(graph,center,8,0,8,np.random.default_rng(seed),base_points=8)
        large=nested_cloud(graph,center,8,0,16,np.random.default_rng(seed),base_points=8)
        np.testing.assert_array_equal(small,baseline)
        np.testing.assert_array_equal(cKDTree(large).query(small)[0],0.)
        self.assertEqual(len(large),16);self.assertEqual(len(np.unique(large,axis=0)),16)
        again=nested_cloud(graph,center,8,0,16,np.random.default_rng(seed),base_points=8)
        np.testing.assert_array_equal(large,again)
        config={'input_points':8,'patch_mm':16.,'halo_mm':0.}
        np.testing.assert_array_equal(subject_cloud(graph,center,config,np.random.default_rng(seed)),baseline)

    def test_nested_cloud_exhaustion_and_invalid_policy(self):
        from vesseltok.input_sampling import nested_cloud,subject_cloud
        graph=TubeGraph([[0,0,0],[1,0,0]],[[0,1]],spacing_mm=.5)
        points=nested_cloud(graph,np.zeros(3),2,0,16,np.random.default_rng(1),base_points=8)
        self.assertEqual(len(points),len(graph.cloud))
        with self.assertRaises(ValueError):nested_cloud(graph,np.zeros(3),2,0,4,np.random.default_rng(1),base_points=8)
        with self.assertRaises(ValueError):subject_cloud(graph,np.zeros(3),{'input_points':8,'patch_mm':4,'halo_mm':0,'input_sampling':'unknown'},np.random.default_rng(1))

    def test_paired_configs_cannot_silently_resample_supervision(self):
        from vesseltok.data import prepare
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'forbidden'
            with self.assertRaises(ValueError):prepare({'input_sampling':'nested_4096'},destination)
            self.assertFalse(destination.exists())

    def test_pair_verification_rejects_changed_supervision(self):
        from vesseltok.cache_validation import verify_pair
        with tempfile.TemporaryDirectory() as directory:
            small=Path(directory)/'small.npz';large=Path(directory)/'large.npz'
            common={'queries':np.zeros((1,4,3),np.float32),'labels':np.array([[0,1,0,1]],np.float32)}
            a=dict(common,cloud=np.array([[[0.,0,0],[1.,0,0]]]),padding=np.array([[False,False]]))
            b=dict(common,cloud=np.array([[[0.,0,0],[.5,0,0],[1.,0,0]]]),padding=np.array([[False,False,False]]))
            np.savez(small,**a);np.savez(large,**b)
            proof=verify_pair(small,large);self.assertEqual(proof['rows_with_exact_nested_clouds'],1)
            b['labels']=np.zeros((1,4),np.float32);np.savez(large,**b)
            with self.assertRaises(AssertionError):verify_pair(small,large)


class FieldDiagnosticTests(unittest.TestCase):
    def test_coarse_branch_samples_survive_physical_refinement(self):
        from vesseltok.field_probe import branch_samples
        points=np.array([[0.,0,0],[1.1,0,0],[1.1,.7,0]])
        q,tangent,arc,coarse,length=branch_samples(points,[0,1,2],.125)
        expected=np.concatenate([np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.25))+1)) for a,b in zip(points[:-1],points[1:])])
        np.testing.assert_allclose(q[coarse],expected,atol=1e-14)
        self.assertLessEqual(np.linalg.norm(np.diff(q,axis=0),axis=1).max(),.125+1e-12)
        self.assertAlmostEqual(length,1.8);self.assertTrue(np.all(np.diff(arc)>=0))

    def test_thin_off_grid_cylinder_is_a_voxelization_candidate(self):
        from vesseltok.field_probe import classify_branch
        centerline=np.c_[np.linspace(1,3,17),np.full(17,.5),np.full(17,.5)]
        nearest=np.rint(centerline)
        distance=np.linalg.norm(nearest[:,1:]-[.5,.5],axis=1)
        voxel=distance<.2;continuous=np.ones(17)
        self.assertEqual(classify_branch(continuous,voxel,False),'voxelization_candidate')
        continuous[8]=.2
        self.assertEqual(classify_branch(continuous,voxel,False),'continuous_field_gap')
        self.assertEqual(classify_branch(continuous,np.ones(17,bool),True),'voxel_branch_preserved')
        self.assertEqual(classify_branch(np.ones(17),np.ones(17,bool),False),'endpoint_connectivity_failure')

    def test_cross_sections_are_orthonormal_in_mm_and_crossings_are_censored(self):
        from vesseltok.field_probe import orthogonal_frame,radial_crossings
        tangent=np.array([[1.,2,3],[0,0,1]])
        u,v=orthogonal_frame(tangent);t=tangent/np.linalg.norm(tangent,axis=1)[:,None]
        np.testing.assert_allclose(np.linalg.norm(u,axis=1),1);np.testing.assert_allclose(np.linalg.norm(v,axis=1),1)
        np.testing.assert_allclose((u*t).sum(1),0,atol=1e-14);np.testing.assert_allclose((v*t).sum(1),0,atol=1e-14)
        np.testing.assert_allclose((u*v).sum(1),0,atol=1e-14)
        radii=np.array([0,.25,.5,1.]);p=np.array([[1,.75,.5,0],[.2,.1,.1,.1],[.9,.9,.9,.9],[.9,.1,.8,.1]])
        crossing,missing,censored,nonmonotone=radial_crossings(p,radii)
        self.assertAlmostEqual(crossing[0],.5);self.assertTrue(missing[1]);self.assertTrue(censored[2]);self.assertTrue(nonmonotone[3])
        self.assertTrue(np.isnan(crossing[1:3]).all())

    def test_arbitrary_world_queries_match_local_decoder_and_freeze_parameters(self):
        from vesseltok.field_probe import WorldDecoder
        from vesseltok.geometry import local_to_world
        model=GraphOccupancyAE(width=16,latent_dim=4,tokens=4,heads=2,encoder_layers=1,decoder_layers=1,bands=2,variational=False).eval()
        cloud=torch.randn(1,8,3);padding=torch.zeros(1,8,dtype=torch.bool)
        with torch.no_grad():code=model.encode(cloud,padding,sample=False)
        record={'center_mm':[12.,-4.,3.],'mean':code['mean'][0].tolist(),'padding':code['padding'][0].tolist(),'anchors_local':code['anchors'][0].tolist()}
        config={'representation':'subject_tokens','patch_mm':32.}
        with self.assertRaises(ValueError):WorldDecoder(model,[record],config,'cpu')
        model.requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
        query=torch.tensor([[[.1,.2,.3],[-.3,.1,.4]]]);world=local_to_world(query[0].numpy(),record['center_mm'],16)
        decoder=WorldDecoder(model,[record],config,'cpu',chunk=1)
        with torch.no_grad():expected=model.decode(query,code,model.prepare_decoder(code)).sigmoid()[0].numpy()
        np.testing.assert_allclose(decoder(world),expected,atol=1e-6)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_source_path_gap_can_have_connected_endpoints_via_detour(self):
        from vesseltok.field_connectivity import endpoint_diagnosis
        tube=TubeGraph([[1,3,1],[5,3,1]],[[0,1]])
        branch={'branch_id':0,'nodes':[0,1],'native_branch_preserved':False,'category':'continuous_field_gap'}
        mask=np.zeros((7,7,3),bool);mask[1:6,3,1]=True;mask[3,3,1]=False;mask[2:5,4,1]=True
        result=endpoint_diagnosis(mask,tube,np.eye(4),[branch])
        self.assertEqual(result['branches'][0]['status'],'endpoints_in_same_component')
        self.assertEqual(result['broken_source_paths_with_connected_endpoints'],1)
        mask[:,4,:]=False
        self.assertEqual(endpoint_diagnosis(mask,tube,np.eye(4),[branch])['branches'][0]['status'],'endpoints_in_different_components')
        mask[1,3,1]=False
        self.assertEqual(endpoint_diagnosis(mask,tube,np.eye(4),[branch])['branches'][0]['status'],'endpoint_not_foreground')


class FeatureSelectionTests(unittest.TestCase):
    def test_feature_controls_preserve_anchors_padding_and_original(self):
        from vesseltok.feature_controls import ablate_record
        record={'mean':[[1.,2],[3.,4],[5.,6],[0.,0]],'anchors_local':[[0,0,0],[1,0,0],[2,0,0],[0,0,0]],'padding':[False,False,False,True],'center_mm':[0,0,0]}
        before=copy.deepcopy(record)
        zero,_=ablate_record(record,'zero');self.assertTrue(np.all(np.asarray(zero['mean'])==0))
        shuffled,perm=ablate_record(record,'shuffle');self.assertTrue(np.all(perm[:3]!=np.arange(3)))
        self.assertEqual(sorted(map(tuple,shuffled['mean'][:3])),sorted(map(tuple,record['mean'][:3])))
        self.assertEqual(shuffled['anchors_local'],record['anchors_local']);self.assertEqual(shuffled['padding'],record['padding']);self.assertEqual(record,before)
        self.assertEqual(ablate_record(record,'shuffle')[0],shuffled)

    def test_anchor_spheres_use_physical_mm(self):
        from vesseltok.feature_controls import anchor_occupancy
        anchors=np.array([[0.,0,0],[2,0,0]]);q=np.array([[0,.9,0],[0,1.1,0],[1,0,0],[5,0,0]])
        np.testing.assert_array_equal(anchor_occupancy(q,anchors),[True,False,True,False])
        rotation=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();translation=np.array([12.,-4,3])
        np.testing.assert_array_equal(anchor_occupancy(q[:2]@rotation.T+translation,anchors@rotation.T+translation),[True,False])

    def test_selection_balances_path_and_dice_and_has_deterministic_ties(self):
        from vesseltok.selection_policy import geometric_score,select_candidate
        self.assertEqual(geometric_score(1.,0.),0.)
        self.assertGreater(geometric_score(.7,.4),geometric_score(.9,.1))
        with self.assertRaises(ValueError):geometric_score(float('nan'),.5)
        a={'score':.4,'false_pair_fraction':.2,'step':100,'checkpoint_sha256':'a'}
        b=dict(a,false_pair_fraction=.1,step=200,checkpoint_sha256='b')
        self.assertEqual(select_candidate([a,b]),b);self.assertEqual(select_candidate([b,a]),b)

    def test_selector_rejects_evaluation_before_loading_metadata(self):
        import json
        from diagnostics.provenance import sha256
        from vesseltok.select_checkpoint import score_candidate
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'checkpoint.pt'
            torch.save({'splits':{'train':['IXI001-A'],'validation':['IXI002-B'],'evaluation':['IXI003-C']},'config':{}},checkpoint)
            (root/'manifest.json').write_text(json.dumps({'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'subjects':['IXI003-C']}))
            with self.assertRaises(ValueError):score_candidate(root)

    def test_native_selection_training_retains_candidates_and_does_not_fake_best(self):
        import contextlib,io,json,sys,yaml
        from vesseltok.train import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir();metadata=root/'metadata';metadata.write_text('fixture');manifest=root/'manifest';manifest.write_text('fixture')
            config={'model':{'width':16,'latent_dim':4,'tokens':4,'heads':2,'encoder_layers':1,'decoder_layers':1,'bands':2,'variational':False},'steps':2,'validate_every':1,'checkpoint_selection':'native_geometry_v1','retain_checkpoints_every':1,'seed':42,'lr':.001,'weight_decay':0.,'batch_size':1,'queries_per_step':8,'class_balanced_batches':True,'kl_weight':0.,'kl_warmup_steps':1,'patch_mm':16,'input_points':8,'pseudo_radius_mm':1,'edge_spacing_mm':.5,'query_pool':16,'metadata':str(metadata),'manifest':str(manifest)}
            (prepared/'config.yaml').write_text(yaml.safe_dump(config));path=root/'config.yaml';path.write_text(yaml.safe_dump(config))
            (prepared/'splits.json').write_text(json.dumps({'train':['IXI001-A'],'validation':['IXI002-B'],'evaluation':['IXI003-C']}))
            rng=np.random.default_rng(1)
            for split in ['train','validation']:
                np.savez(prepared/(split+'.npz'),cloud=rng.normal(size=(1,8,3)).astype(np.float32),padding=np.zeros((1,8),bool),queries=rng.normal(size=(1,16,3)).astype(np.float32),labels=np.tile([0.,1.],(1,8)).astype(np.float32),conditions=np.zeros((1,2),np.float32))
            result=root/'run'
            with patch.object(sys,'argv',['train','--config',str(path),'--prepared',str(prepared),'--output',str(result),'--device','cpu']),contextlib.redirect_stdout(io.StringIO()):main()
            self.assertTrue((result/'best_bce.pt').exists());self.assertFalse((result/'best.pt').exists())
            self.assertTrue((result/'candidates/step_000001.pt').exists());self.assertTrue((result/'candidates/step_000002.pt').exists())
            self.assertEqual(json.loads((result/'selection_required.json').read_text())['status'],'pending_native_validation')

    def test_retained_candidates_resolve_to_the_same_training_run(self):
        from vesseltok.select_checkpoint import training_run_root
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'candidates').mkdir();(root/'provenance.json').write_text('{}')
            self.assertEqual(training_run_root(root/'best_bce.pt'),training_run_root(root/'candidates/step_000500.pt'))


class InterAnchorSamplingTests(unittest.TestCase):
    def fixture(self,root,empty=False,wrong_subject=False):
        source=root/'train.npz';pool=root/'pool.npz';rng=np.random.default_rng(7)
        subjects=np.array(['IXI001-A','IXI002-B']);labels=np.tile([1.,1.,1.,1.,0.,0.,0.,0.],(2,1)).astype(np.float32)
        np.savez(source,subjects=subjects,cloud=rng.normal(size=(2,8,3)).astype(np.float32),padding=np.zeros((2,8),bool),queries=rng.normal(size=(2,8,3)).astype(np.float32),labels=labels,conditions=np.zeros((2,2),np.float32))
        eligible=labels>.5
        if empty:eligible[:]=False
        np.savez(pool,subjects=np.array(['IXI003-C','IXI002-B']) if wrong_subject else subjects,queries=np.full((2,4,3),10.,np.float32),weights=np.ones((2,4)),eligible=eligible,anchors=np.zeros((2,4,3),np.float32),anchor_padding=np.zeros((2,4),bool))
        return source,pool

    def test_shadow_sampling_pairs_rng_background_labels_and_retained_queries(self):
        from vesseltok.data import PatchSet
        from vesseltok.inter_anchor_streams import SamplingStreams
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,pool=self.fixture(root)
            a=PatchSet(source,8,True,pool,'control',2,9215);b=PatchSet(source,8,True,pool,'replace',2,9215)
            torch.manual_seed(42);arows=[a[i%2] for i in range(40)];after_a=torch.get_rng_state().clone()
            torch.manual_seed(42);brows=[b[i%2] for i in range(40)];after_b=torch.get_rng_state().clone()
            torch.testing.assert_close(after_a,after_b,rtol=0,atol=0)
            sa,sb=SamplingStreams(),SamplingStreams()
            for x,y in zip(arows,brows):
                for key in ['base_query_indices','labels','cloud','padding','planned_replacement_mask','proposed_queries']:
                    torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
                mask=x['planned_replacement_mask'];self.assertTrue(torch.all(x['labels'][mask]>.5));self.assertEqual(int(mask.sum()),2)
                torch.testing.assert_close(x['queries'][~mask],y['queries'][~mask],rtol=0,atol=0)
                self.assertEqual(int(x['replacement_count']),0);self.assertEqual(int(y['replacement_count']),2)
                self.assertEqual(int((y['labels']>.5).sum()),4)
                sa.update({k:v[None] for k,v in x.items()});sb.update({k:v[None] for k,v in y.items()})
            self.assertEqual({k:v.hexdigest() for k,v in sa.hashes.items()},{k:v.hexdigest() for k,v in sb.hashes.items()})
            with np.load(source) as saved:np.testing.assert_array_equal(saved['queries'],a.data['queries'].numpy())

    def test_no_eligible_slots_does_not_change_batch_or_drop_subject(self):
        from vesseltok.data import PatchSet
        with tempfile.TemporaryDirectory() as directory:
            source,pool=self.fixture(Path(directory),empty=True)
            dataset=PatchSet(source,8,True,pool,'replace',2)
            item=dataset[0];self.assertEqual(int(item['replacement_count']),0);self.assertEqual(len(item['queries']),8)
            self.assertEqual(int((item['labels']>.5).sum()),4)

    def test_validation_or_wrong_subject_pool_is_rejected(self):
        from vesseltok.data import PatchSet
        with tempfile.TemporaryDirectory() as directory:
            source,pool=self.fixture(Path(directory),wrong_subject=True)
            with self.assertRaises(ValueError):PatchSet(source,8,True,pool,'replace',2)
            with self.assertRaises(ValueError):PatchSet(source,8,False,pool,'replace',2)


class PartitionConnectionTests(unittest.TestCase):
    def test_excess_merges_and_missing_connections_are_separated(self):
        from vesseltok.partition_connections import partition_connections
        target=np.array([1,1,2,2])
        result=partition_connections([1,1,1,2],target)
        self.assertEqual(result['excess_connected_pairs_vs_target'],2);self.assertEqual(result['target_connections_missing'],1)
        self.assertEqual(result['excess_fraction_of_target_disconnected_pairs'],.5)
        self.assertEqual(partition_connections([0,0,1,1],target)['target_connections_missing'],1)
        self.assertEqual(partition_connections([1,1,1,1],target)['excess_fraction_of_target_disconnected_pairs'],1.)
        self.assertEqual(partition_connections([0,0,0,0],target)['missing_fraction_of_target_connected_pairs'],1.)
