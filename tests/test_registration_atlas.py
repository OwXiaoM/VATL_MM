"""Scientific invariants for FOV accounting and physical transform conventions."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import nibabel as nib

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('registration_atlas',ROOT/'scripts/decomposition/registration_atlas/pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)

class RegistrationAtlasTests(unittest.TestCase):
    def test_missing_fov_is_not_background(self):
        masks=np.array([[1,1,0],[0,0,0]],bool)
        covers=np.array([[1,1,0],[0,1,0]],bool)
        p,c=pipeline.fov_probability(masks,covers)
        np.testing.assert_allclose(p[:2],[1.,.5])
        self.assertTrue(np.isnan(p[2]));np.testing.assert_array_equal(c,[1,2,0])

    def test_jacobian_world_units_and_orientation(self):
        shape=(9,10,11);idx=np.indices(shape).transpose(1,2,3,0)
        # Oblique anisotropic physical grid in LPS millimetres.
        t=.3;r=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        b=r@np.diag([.8,1.4,2.])
        x=idx@b.T
        a=np.array([[1.1,.1,0],[0,.9,0],[0,0,1.2]])
        u=x@(a-np.eye(3)).T+np.array([2.,-1.,4.])
        det=pipeline.jacobian_from_displacement(u,b)
        np.testing.assert_allclose(det,np.linalg.det(a),rtol=2e-6,atol=2e-6)
        a[0,0]=-1
        u=x@(a-np.eye(3)).T
        self.assertTrue((pipeline.jacobian_from_displacement(u,b)<0).all())

    def test_empty_centerline_is_explicitly_undefined(self):
        empty=np.zeros((9,9,9),bool);coverage=np.ones_like(empty)
        r=pipeline.pair_metrics(empty,empty,coverage,coverage,np.ones(3))
        self.assertIsNone(r['clDice']);self.assertIsNone(r['centerline_mean_mm'])

    def test_signed_distance_is_physical(self):
        a=np.zeros((9,9,9),bool);a[3:6,3:6,3:6]=True
        d=pipeline.signed_distance(a,[.8,1.5,2.])
        self.assertAlmostEqual(float(d[2,4,4]),.8,places=6)
        self.assertTrue((d[a]<0).all());self.assertTrue((d[~a]>0).all())

    def test_image_chain_and_inverse_order(self):
        import ants
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            a=ants.create_ants_transform(transform_type='AffineTransform',dimension=3,
                matrix=[[1, .1, 0],[0,1,0],[0,0,1]],translation=[2.,0.,0.])
            b=ants.create_ants_transform(transform_type='AffineTransform',dimension=3,
                matrix=[[1,0,0],[0,1,0],[0,0,1]],translation=[0.,3.,0.])
            paths=[];inverses=[]
            for name,tx in [('old',a),('increment',b)]:
                path=p/(name+'.mat');ants.write_transform(tx,str(path));paths.append(str(path))
                path=p/(name+'_inverse.mat');ants.write_transform(tx.invert(),str(path));inverses.append(str(path))
            # Composition must sample original through old(increment(x)).
            points=pd.DataFrame([[2.,4.,6.]],columns=['x','y','z'])
            chain,inverse_chain=pipeline.compose_image_chains([paths[0]],[inverses[0]],[paths[1]],[inverses[1]])
            mapped=ants.apply_transforms_to_points(3,points,chain,whichtoinvert=[False,False])
            expected=a.apply_to_point(b.apply_to_point((2.,4.,6.)))
            np.testing.assert_allclose(mapped[['x','y','z']],np.array([expected]),atol=1e-5)
            returned=ants.apply_transforms_to_points(3,mapped,inverse_chain,whichtoinvert=[False,False])
            np.testing.assert_allclose(returned[['x','y','z']],points,atol=1e-5)

if __name__=='__main__':unittest.main()
