"""Regression checks for physical evaluation and opt-in ablation behavior."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from data_loading.dataset import Data
from models.sdf_decomposition import SubjectRegistration, AtlasSubjectSDF
from models.decomposition_evaluation import evaluate_subject
from test_sdf_decomposition import tiny_args


class DiagnosticTests(unittest.TestCase):
    def test_rotvec_identity_rank_inverse_and_recovery(self):
        vector = torch.zeros(3, requires_grad=True)
        jacobian = torch.autograd.functional.jacobian(lambda v: SubjectRegistration.rotation_matrix(v[None]).flatten(), vector)
        self.assertEqual(int(torch.linalg.matrix_rank(jacobian)), 3)
        reg = SubjectRegistration([20,30,40], 'rotvec')
        coords = torch.randn(64,3)*.2
        target_tf = torch.tensor([[.08,-.12,.04,1.,-2.,.5]])
        target = reg.subject_to_atlas(coords, target_tf.expand(64,-1)).detach()
        torch.testing.assert_close(reg.atlas_to_subject(target,target_tf.expand(64,-1)),coords,atol=1e-6,rtol=1e-5)
        fitted = torch.nn.Parameter(torch.zeros(1,6))
        optimizer = torch.optim.Adam([fitted],lr=.04)
        for _ in range(250):
            optimizer.zero_grad()
            loss = (reg.subject_to_atlas(coords,fitted.expand(64,-1))-target).square().mean()
            loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()),1e-9)

    def test_default_checkpoint_state_and_no_residual_parameters(self):
        args = tiny_args()
        old = AtlasSubjectSDF(args,['one'])
        explicit = copy.deepcopy(args)
        explicit['decomposition'].update(rotation='legacy_euler')
        same = AtlasSubjectSDF(explicit,['one'])
        same.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(30,3)*.2; c=torch.zeros(30,2); ids=torch.zeros(30,dtype=torch.long)
        torch.testing.assert_close(old(x,c,ids)['sdf_mm'],same(x,c,ids)['sdf_mm'])
        self.assertFalse(any(k.startswith(('residual.', 'codes.')) for k in same.state_dict()))

    def test_uniform_sampling_endpoints_and_crop(self):
        grid=np.indices((9,9,9)).transpose(1,2,3,0)
        sdf=np.linalg.norm(grid-4,axis=-1)-2
        seg=(sdf<0).astype(float)
        args={'dataset':{'sdf_sampling':{'uniform_fraction':0.}}}
        ctx=SimpleNamespace(args=args,world_bbox=np.array([6,6,6]))
        np.random.seed(4)
        original=Data.sample_sdf_indices(ctx,sdf,seg,np.eye(4),num_points=400)
        del args['dataset']['sdf_sampling']['uniform_fraction']
        np.random.seed(4)
        default=Data.sample_sdf_indices(ctx,sdf,seg,np.eye(4),num_points=400)
        np.testing.assert_array_equal(original,default)
        for fraction in [0.,.5,1.]:
            args['dataset']['sdf_sampling']['uniform_fraction']=fraction
            indices=Data.sample_sdf_indices(ctx,sdf,seg,np.eye(4),num_points=400)
            self.assertEqual(indices.shape,(400,3))
            self.assertTrue((np.abs((indices-4.5)/3)<=1).all())

    def test_full_reference_grid_metric_includes_crop_misses(self):
        class Sphere(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.parameter=torch.nn.Parameter(torch.tensor(0.))
                self.args={'dataset':{'sdf_key':'SDF','modalities':['Seg'],
                    'label_names':['Background','Vessel'],'world_bbox':[4,12,12]}}
            def forward(self,x,c,**kwargs):
                field=torch.linalg.vector_norm(x*torch.tensor([2.,6.,6.]),dim=-1)-3.1
                return {'sdf_mm':field,'warped_atlas_mm':field}
        with tempfile.TemporaryDirectory() as directory:
            affine=np.diag([1.,1.,1.,1.]); affine[:3,3]=[14,-9,7]
            coords=np.indices((12,12,12)).transpose(1,2,3,0)-6
            sdf=np.linalg.norm(coords,axis=-1)-3.1
            truth=sdf<0
            for name,data in [('sdf',sdf.astype(np.float32)),('seg',truth.astype(np.uint8))]:
                img=nib.Nifti1Image(data,affine);img.header.set_xyzt_units('mm');nib.save(img,Path(directory)/f'{name}.nii.gz')
            row={'subject_id':'analytic','SDF':str(Path(directory)/'sdf.nii.gz'),'Seg':str(Path(directory)/'seg.nii.gz')}
            with patch('models.decomposition_evaluation.Data.load_conditions',return_value=torch.zeros(2)):
                result=evaluate_subject(Sphere(),row,0,chunk=43)
            predicted=truth & (np.abs(coords[...,0])<=2)
            expected=2*predicted.sum()/(predicted.sum()+truth.sum())
            self.assertAlmostEqual(result['Dice'],expected)
            self.assertLess(result['recall'],1.)
            self.assertEqual(result['truth_voxels'],int(truth.sum()))


if __name__=='__main__':
    unittest.main()
