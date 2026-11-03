import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from vesseltok.geometry import TubeGraph,rigid_world
from vesseltok.partial_volume import voxel_fractions,naive_fractions,capped_distance


class PartialVolumeTests(unittest.TestCase):
    def test_adaptive_equals_naive_in_sheared_anisotropic_cells(self):
        tube=TubeGraph([[0,0,-3],[0,0,3],[3,0,0]],[[0,1]])
        basis=np.array([[.7,.1,0],[0,.9,.12],[0,0,1.3]]);q=np.random.default_rng(42).uniform(-1.7,1.7,(14,3))
        for n in [1,2,4,8]:
            r=voxel_fractions(tube,q,basis,n);np.testing.assert_allclose(r['fraction'],naive_fractions(tube,q,basis,n),atol=1e-12)

    def test_rotation_translation_and_inverse_consistency(self):
        tube=TubeGraph([[0,0,-3],[0,0,3]],[[0,1]]);q=np.array([[.913,.137,.223],[1.137,.211,-.17]]);basis=np.diag([.5,.7,1.1]);R=Rotation.from_rotvec([.3,-.2,.4]).as_matrix();t=np.array([51.,-23.,12.])
        transformed=TubeGraph(rigid_world(tube.points,R,t,np.zeros(3)),tube.edges)
        a=voxel_fractions(tube,q,basis,16);b=voxel_fractions(transformed,q@R.T+t,R@basis,16)
        for key in ['fraction','lower','upper']:np.testing.assert_allclose(a[key],b[key],atol=1e-12)
        np.testing.assert_allclose((q@R.T+t-t)@R,q,atol=1e-12)

    def test_certificates_and_distance_include_isolated_nodes(self):
        tube=TubeGraph([[0,0,-5],[0,0,5],[10,0,0]],[[0,1]])
        q=np.array([[0,0,0],[5,0,0],[10,0,0]]);r=voxel_fractions(tube,q,np.eye(3)*.1,8)
        np.testing.assert_array_equal(r['lower'],[1,0,1]);np.testing.assert_array_equal(r['upper'],[1,0,1])
        distance=capped_distance(tube,[[0,.3,0],[10,.2,0],[30,0,0]],1.)
        np.testing.assert_allclose(distance[:2],[.3,.2]);self.assertTrue(np.isinf(distance[-1]))

    def test_small_tube_fraction_is_volume_not_presence_uncertainty(self):
        tube=TubeGraph([[0,0,-4],[0,0,4]],[[0,1]]);r=voxel_fractions(tube,np.zeros((1,3)),np.eye(3),128,radius_mm=.2)
        self.assertTrue(tube.occupancy([[0,0,0]],.2)[0]);self.assertLess(r['upper'][0],.5)
        self.assertAlmostEqual(r['fraction'][0],np.pi*.2**2,delta=.002)

    def test_nearby_disjoint_tubes_can_fill_gap_voxel_majority(self):
        tube=TubeGraph([[-1.05,0,-4],[-1.05,0,4],[1.05,0,-4],[1.05,0,4]],[[0,1],[2,3]])
        self.assertFalse(tube.occupancy([[0,0,0]],1.)[0]);r=voxel_fractions(tube,np.zeros((1,3)),np.eye(3)*.5,32)
        self.assertGreater(r['lower'][0],.5)

    def test_axis_permutations_preserve_physical_cell(self):
        tube=TubeGraph([[0,0,-5],[0,0,5]],[[0,1]]);q=np.array([[.85,.11,.27]]);basis=np.array([[.5,.1,0],[0,.7,0],[0,0,.9]])
        a=voxel_fractions(tube,q,basis,16);b=voxel_fractions(tube,q,basis[:,[2,0,1]]@np.diag([-1,1,-1]),16)
        np.testing.assert_array_equal(a['fraction'],b['fraction'])

    def test_bound_width_decreases_and_brackets_fine_reference(self):
        tube=TubeGraph([[0,0,-4],[0,0,4]],[[0,1]]);q=np.array([[1.013,.117,.223]])
        previous=1.
        for n in [2,4,8,16,32]:
            r=voxel_fractions(tube,q,np.eye(3)*.6,n);width=r['upper'][0]-r['lower'][0];self.assertLessEqual(width,previous+1e-12);previous=width
            fine=naive_fractions(tube,q,np.eye(3)*.6,32)[0];self.assertLessEqual(r['lower'][0],fine);self.assertGreaterEqual(r['upper'][0],fine)

    def test_invalid_geometry_or_n_rejected(self):
        tube=TubeGraph([[0,0,0]],[])
        for basis,n in [(np.zeros((3,3)),8),(np.eye(3),3)]:
            with self.assertRaises(ValueError):voxel_fractions(tube,np.zeros((1,3)),basis,n)


if __name__=='__main__':unittest.main()
