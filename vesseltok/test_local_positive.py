import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from vesseltok.gap_sampling import GapPatchSet
from vesseltok.local_positive import LocalPositiveSet,LocalPositiveStreams
from vesseltok.model import GraphOccupancyAE
from vesseltok.geometry import TubeGraph


def fixture(root,empty=False,wrong=False,bad_label=False,bad_csr=False):
    subjects=np.array(['IXI001-Synthetic']);labels=np.r_[np.ones(256),np.zeros(256)].astype(np.float32)[None]
    queries=np.zeros((1,512,3),np.float32);queries[:,256:]=.1
    np.savez(root/'train.npz',subjects=subjects,cloud=np.random.default_rng(42).normal(size=(1,8,3)).astype(np.float32)*.01,padding=np.zeros((1,8),bool),queries=queries,labels=labels,conditions=np.zeros((1,2),np.float32))
    np.savez(root/'inter.npz',subjects=subjects,queries=np.tile([.005,0,0],(1,4,1)).astype(np.float32),weights=np.ones((1,4)),eligible=labels>.5,anchors=np.zeros((1,4,3),np.float32),anchor_padding=np.zeros((1,4),bool))
    negative=np.array([[0,.015,0],[.003,.018,0]],np.float32)
    np.savez(root/'gap.npz',subjects=subjects,queries=negative[None],weights=np.ones((1,2)),eligible=labels<=.5)
    positive=np.array([[0,0,0],[.001,0,0],[.002,.002,0],[-.002,0,0]],np.float32)
    np.savez(root/'local.npz',subjects=np.array(['IXI999-other']) if wrong else subjects,queries_0=positive,labels_0=np.array([0,1,1,1]) if bad_label else np.ones(4),negative_0=negative,
             offsets_0=np.zeros(3,int) if empty else np.array([0,4,8]),neighbors_0=np.array([],int) if empty else (np.full(8,99) if bad_csr else np.tile(np.arange(4),2)))
    return root


def dataset(root,mode,**kwargs):
    return LocalPositiveSet(root/'train.npz',512,True,root/'inter.npz','replace',kwargs.pop('replacements',64),gap_pool=root/'gap.npz',gap_mode='replace',gap_replacements=16,
                            local_pool=root/'local.npz',local_mode=mode,local_cap=16,**kwargs)


class LocalPositiveTests(unittest.TestCase):
    def test_paired_actual512_queries_binary_geometry_and_private_rng(self):
        with tempfile.TemporaryDirectory() as d:
            root=fixture(Path(d));a=dataset(root,'control');b=dataset(root,'replace')
            torch.manual_seed(42);aa=[a[0] for _ in range(30)];state=torch.get_rng_state()
            torch.manual_seed(42);bb=[b[0] for _ in range(30)];self.assertTrue(torch.equal(state,torch.get_rng_state()));sa,sb=LocalPositiveStreams(),LocalPositiveStreams()
            tube=TubeGraph([[-1,0,0],[1,0,0]],[[0,1]])
            for x,y in zip(aa,bb):
                self.assertEqual(int(x['local_applied']),0);self.assertEqual(int(y['local_applied']),16);self.assertEqual(int(y['gap_applied']),16);self.assertEqual(int(y['replacement_count']),64)
                self.assertEqual(int((y['labels']==1).sum()),256);self.assertTrue(torch.equal(x['labels'],y['labels']))
                for mask in [y['labels']==0,y['planned_replacement_mask'],~y['local_plan']]:self.assertTrue(torch.equal(x['queries'][mask],y['queries'][mask]))
                self.assertFalse(bool((y['local_plan']&(y['gap_plan']|y['planned_replacement_mask'])).any()))
                p=y['queries'][y['local_plan']];n=y['queries'][y['local_partner_gap_slot'][y['local_plan']]]
                self.assertTrue(torch.all(torch.linalg.vector_norm((p-n).double()*128,dim=1)<=3))
                self.assertTrue(tube.occupancy(p.numpy()*128,1.).all());self.assertFalse(tube.occupancy(n.numpy()*128,1.).any())
                sa.update({k:v[None] for k,v in x.items()});sb.update({k:v[None] for k,v in y.items()})
            self.assertEqual({k:v.hexdigest() for k,v in sa.local_hashes.items()},{k:v.hexdigest() for k,v in sb.local_hashes.items()})

    def test_control_optimizer_and_gradients_bitwise_replay_unwrapped_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root=fixture(Path(d));states=[];rng=[]
            for protected in [False,True]:
                torch.manual_seed(42);model=GraphOccupancyAE(width=16,tokens=4,latent_dim=4,heads=4,cross_heads=1,encoder_layers=1,decoder_layers=1,bands=2,variational=False,spatial_anchors=True)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
                ids=[id(p) for group in optimizer.param_groups for p in group['params']];self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters()})
                data=dataset(root,'control') if protected else GapPatchSet(root/'train.npz',512,True,root/'inter.npz','replace',64,gap_pool=root/'gap.npz',gap_mode='replace',gap_replacements=16)
                for _ in range(3):
                    batch=data[0];optimizer.zero_grad(set_to_none=True);out=model(batch['cloud'][None],batch['padding'][None],batch['queries'][None]);loss=F.binary_cross_entropy_with_logits(out['logits'],batch['labels'][None]);loss.backward()
                    grads={n:p.grad for n,p in model.named_parameters()};self.assertTrue(any(g is not None and bool(g.abs().sum()>0) for n,g in grads.items() if n.startswith('encoder_')))
                    self.assertTrue(any(g is not None and bool(g.abs().sum()>0) for n,g in grads.items() if n.startswith('decoder_')))
                    self.assertTrue(all(g is None for n,g in grads.items() if n.startswith('logvar')));optimizer.step()
                states.append({k:v.clone() for k,v in model.state_dict().items()});rng.append(torch.get_rng_state())
            self.assertTrue(torch.equal(rng[0],rng[1]));self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))

    def test_empty_partner_pool_retains_subject_and_all_queries(self):
        with tempfile.TemporaryDirectory() as d:
            root=fixture(Path(d),empty=True);item=dataset(root,'replace')[0]
            self.assertEqual(int(item['local_applied']),0);self.assertEqual(int(item['gap_applied']),16);self.assertEqual(len(item['queries']),512)

    def test_no_remaining_positive_slots_retains_existing_inter_anchor_queries(self):
        with tempfile.TemporaryDirectory() as d:
            root=fixture(Path(d));item=dataset(root,'replace',replacements=256)[0]
            self.assertEqual(int(item['local_applied']),0);self.assertEqual(int(item['replacement_count']),256)

    def test_corrupt_labels_cohort_and_csr_rejected(self):
        for option,message in [('wrong','subject order'),('bad_label','binary-positive'),('bad_csr','CSR')]:
            with tempfile.TemporaryDirectory() as d:
                root=fixture(Path(d),**{option:True})
                with self.assertRaisesRegex(ValueError,message):dataset(root,'replace')

    def test_nonbinary_cache_targets_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=fixture(Path(d))
            with np.load(root/'train.npz') as z:arrays={k:z[k] for k in z.files}
            arrays['labels'][0,0]=.75;np.savez(root/'train.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'exact binary'):dataset(root,'replace')


if __name__=='__main__':unittest.main()
