"""Regression for the actual512-query,16-negative-dose configuration."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from vesseltok.gap_sampling import GapPatchSet


class Gap16Tests(unittest.TestCase):
    def test_actual_query_balance_cap16_and_paired_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rng=np.random.default_rng(18);subjects=np.array(['IXI001-A']);labels=np.r_[np.ones(256),np.zeros(256)].astype(np.float32)[None]
            np.savez(root/'train.npz',subjects=subjects,cloud=np.zeros((1,8,3),np.float32),padding=np.zeros((1,8),bool),queries=rng.normal(size=(1,512,3)).astype(np.float32),labels=labels,conditions=np.zeros((1,2),np.float32))
            np.savez(root/'positive.npz',subjects=subjects,queries=np.ones((1,8,3),np.float32),weights=np.ones((1,8)),eligible=labels>.5,anchors=np.zeros((1,4,3),np.float32),anchor_padding=np.zeros((1,4),bool))
            np.savez(root/'gap.npz',subjects=subjects,queries=np.full((1,8,3),-2.,np.float32),weights=np.ones((1,8)),eligible=labels<=.5)
            datasets=[GapPatchSet(root/'train.npz',512,True,root/'positive.npz','replace',64,gap_pool=root/'gap.npz',gap_mode=mode,gap_replacements=16) for mode in ['control','replace']]
            torch.manual_seed(42);a=datasets[0][0];state_a=torch.get_rng_state()
            torch.manual_seed(42);b=datasets[1][0];state_b=torch.get_rng_state()
            self.assertTrue(torch.equal(state_a,state_b));self.assertEqual(int(b['gap_applied']),16);self.assertEqual(int(a['gap_applied']),0)
            self.assertEqual(int(b['replacement_count']),64);self.assertEqual(int((b['labels']>.5).sum()),256)
            for key in ['base_query_indices','labels','gap_plan','gap_proposal','planned_replacement_mask']:self.assertTrue(torch.equal(a[key],b[key]),key)
            self.assertTrue(torch.equal(a['queries'][~a['gap_plan']],b['queries'][~b['gap_plan']]))
            self.assertTrue(torch.all(b['labels'][b['gap_plan']]<=.5))


if __name__=='__main__':unittest.main()
