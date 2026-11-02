import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from vesseltok.geometry import TubeGraph
from vesseltok.gap_sampling import gap_candidates,GapPatchSet,GapStreams
from vesseltok import tests as baseline_tests


class GapTests(unittest.TestCase):
    def test_midpoints_are_true_gap_negatives_and_geodesic_filter_is_physical(self):
        tube=TubeGraph([[0,0,0],[0,10,0],[4,0,0],[4,10,0]],[[0,1],[2,3]])
        points,pairs=gap_candidates(tube,np.random.default_rng(1))
        self.assertEqual(len(points),2)
        np.testing.assert_allclose(points,[[2,0,0],[2,10,0]])
        self.assertFalse(tube.occupancy(points,1.25).any())
        connected=TubeGraph(tube.points,[[0,1],[2,3],[0,2],[1,3]])
        self.assertEqual(len(gap_candidates(connected,np.random.default_rng(1))[0]),0)

    def test_other_edge_rejects_false_negative_gap_midpoint(self):
        tube=TubeGraph([[0,0,0],[0,10,0],[4,0,0],[4,10,0],[2,0,0],[2,10,0]],[[0,1],[2,3],[4,5]])
        self.assertEqual(len(gap_candidates(tube,np.random.default_rng(1))[0]),0)

    def fixture(self,root,empty=False,wrong=False,positive=False):
        source,pool=baseline_tests.InterAnchorSamplingTests().fixture(root)
        with np.load(source) as data:
            eligible=data['labels']>.5 if positive else data['labels']<=.5
            if empty:eligible[:]=False
            np.savez(root/'gap.npz',subjects=np.array(['bad','other']) if wrong else data['subjects'],eligible=eligible,queries=np.full((2,3,3),-7.,np.float32),weights=np.ones((2,3)))
        return source,pool,root/'gap.npz'

    def test_pairing_preserves_positives_boundary_rng_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source,pool,gap=self.fixture(Path(directory));a=GapPatchSet(source,8,True,pool,'replace',2,gap_pool=gap,gap_mode='control',gap_replacements=2);b=GapPatchSet(source,8,True,pool,'replace',2,gap_pool=gap,gap_mode='replace',gap_replacements=2)
            torch.manual_seed(42);aa=[a[i%2] for i in range(40)];state_a=torch.get_rng_state()
            torch.manual_seed(42);bb=[b[i%2] for i in range(40)];state_b=torch.get_rng_state()
            self.assertTrue(torch.equal(state_a,state_b));sa,sb=GapStreams(),GapStreams()
            for x,y in zip(aa,bb):
                for key in ['labels','cloud','padding','base_query_indices','gap_plan','gap_proposal','planned_replacement_mask']:
                    self.assertTrue(torch.equal(x[key],y[key]),key)
                self.assertTrue(torch.equal(x['queries'][~x['gap_plan']],y['queries'][~y['gap_plan']]))
                self.assertTrue(torch.all(y['labels'][y['gap_plan']]<=.5));self.assertEqual(int(y['gap_applied']),2)
                self.assertEqual(int(x['gap_applied']),0);self.assertEqual(int((y['labels']>.5).sum()),4)
                sa.update({k:v[None] for k,v in x.items()});sb.update({k:v[None] for k,v in y.items()})
            self.assertEqual({k:v.hexdigest() for k,v in sa.gap_hashes.items()},{k:v.hexdigest() for k,v in sb.gap_hashes.items()})

    def test_no_eligible_negatives_retains_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            source,pool,gap=self.fixture(Path(directory),empty=True)
            sample=GapPatchSet(source,8,True,pool,'replace',2,gap_pool=gap,gap_mode='replace',gap_replacements=2)[0]
            self.assertEqual(int(sample['gap_applied']),0);self.assertEqual(len(sample['queries']),8)

    def test_reject_positive_eligibility_and_cohort_mismatch(self):
        for kwargs in [dict(wrong=True),dict(positive=True)]:
            with tempfile.TemporaryDirectory() as directory:
                source,pool,gap=self.fixture(Path(directory),**kwargs)
                message='match training cohort' if kwargs.get('wrong') else 'negatives only'
                with self.assertRaisesRegex(ValueError,message):GapPatchSet(source,8,True,pool,'replace',2,gap_pool=gap,gap_mode='replace',gap_replacements=2)

    def test_near_negative_queries_preserved_and_partial_cap_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            source,pool,gap=self.fixture(Path(directory))
            with np.load(gap) as saved:arrays={k:saved[k] for k in saved.files}
            arrays['eligible'][:,5:]=False;np.savez(gap,**arrays)
            dataset=GapPatchSet(source,8,True,pool,'replace',2,gap_pool=gap,gap_mode='replace',gap_replacements=2)
            saw_near=saw_partial=False
            for _ in range(40):
                sample=dataset[0];near=(sample['labels']<=.5)&~sample['gap_eligible_mask'];saw_near|=bool(near.any())
                with np.load(source) as saved:original=torch.from_numpy(saved['queries'][0])[sample['base_query_indices']]
                self.assertTrue(torch.equal(sample['queries'][near],original[near]))
                count=min(2,int(sample['gap_eligible_mask'].sum()));self.assertEqual(int(sample['gap_applied']),count)
                saw_partial|=0<count<2
            self.assertTrue(saw_near and saw_partial)


if __name__=='__main__':unittest.main()
