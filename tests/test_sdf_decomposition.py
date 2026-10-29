import copy
import io
import unittest
import numpy as np
import torch
from models.sdf_decomposition import AtlasSubjectSDF, SubjectRegistration, sdf_to_mask
from models.decomposition_losses import difference_samples, decomposition_losses
from models.decomposition_inference import adapt_subject
from models.inr_decoder import INR_Decoder
from models.vascular_metrics import vascular_metrics
from train_decomposition import load_config, optimizer_for, checkpoint_payload


def tiny_args():
    a = load_config('configs/decomposition/smoke.yaml', resolve_subject_ids=False)
    a['device'] = 'cpu'
    a['inr_decoder']['use_moe'] = False
    a['inr_decoder']['latent_dim'] = [4, 1, 1, 1]
    a['inr_decoder']['hidden_size'] = 16
    a['inr_decoder']['num_hidden_layers'] = 1
    a['decomposition']['stages']['enabled'] = False
    a['dataset']['world_bbox'] = [20, 30, 40]
    return a


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.args = tiny_args()
        self.model = AtlasSubjectSDF(self.args, ['s1', 's2'])
        self.x = torch.randn(24, 3) * .2
        self.c = torch.zeros(24, 2)
        self.ids = torch.arange(24) % 2

    def test_shapes_identity_zero_and_sign(self):
        out = self.model(self.x, self.c, self.ids)
        self.assertEqual(out['sdf_mm'].shape, (24,))
        self.assertTrue(torch.isfinite(out['output']).all())
        self.assertFalse(any(k.startswith(('residual.', 'codes.')) for k in self.model.state_dict()))
        zeros = out
        atlas = self.model(self.x, self.c, mode='atlas_only')
        torch.testing.assert_close(zeros['sdf_mm'], atlas['sdf_mm'])
        self.assertEqual(sdf_to_mask(torch.tensor([-1., 0., 1.])).tolist(), [True, False, False])
        # Shared atlas predictions cannot change with a subject ID.
        torch.testing.assert_close(self.model(self.x, self.c, self.ids, mode='atlas_only')['sdf_mm'], atlas['sdf_mm'])

    def test_physical_rigid_inverse(self):
        reg = SubjectRegistration([20., 30., 40.])
        tf = torch.tensor([[.2, -.3, .1, 1., 2., -3.]]).expand(24, -1)
        warped = reg.subject_to_atlas(self.x, tf)
        torch.testing.assert_close(reg.atlas_to_subject(warped, tf), self.x, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(reg.subject_to_atlas(self.x, torch.zeros_like(tf)), self.x)
        origin = torch.zeros(1, 3)
        shift = reg.subject_to_atlas(origin, torch.tensor([[0., 0., 0., 1., 2., 3.]]))
        torch.testing.assert_close(shift, torch.tensor([[.1, 2/15, .15]]))
        a = (self.x[0]-self.x[1]) * reg.half_bbox
        b = (warped[0]-warped[1]) * reg.half_bbox
        torch.testing.assert_close(a.norm(), b.norm())

    def test_gradients_and_losses(self):
        cfg = self.args['decomposition']['loss']
        target = torch.linspace(-1, 1, len(self.x))
        diff = difference_samples(self.x, self.c, self.ids, target, self.model.registration.half_bbox, cfg)
        out = self.model(torch.cat([self.x, diff[0]]), torch.cat([self.c, diff[1]]), torch.cat([self.ids, diff[2]]))
        base = {k: v[:len(self.x)] for k, v in out.items() if k != 'moe'}
        fd = {k: out[k][len(self.x):] for k in ('sdf_mm',)}
        loss = decomposition_losses(self.model, base, target, self.ids, cfg, fd)
        loss['total'].backward()
        for module in (self.model.shared, self.model.transforms):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_physical_eikonal_linear_field(self):
        # Analytic d(x)=world_x has exactly unit physical gradient.
        cfg = self.args['decomposition']['loss']
        diff = difference_samples(self.x, self.c, self.ids, torch.zeros(24), self.model.registration.half_bbox, cfg)
        out = self.model(self.x, self.c, self.ids)
        fd = {'sdf_mm': diff[0][:, 0] * 10}
        loss = decomposition_losses(self.model, out, torch.zeros(24), self.ids, cfg, fd)
        self.assertLess(float(loss['eikonal_raw']), 1e-10)

    def test_joint_training_and_registration_switch(self):
        for epoch in (0, 1, 20):
            self.assertEqual(self.model.set_stage(epoch), 'C')
            self.assertTrue(self.model.shared.grid.requires_grad)
            self.assertTrue(self.model.transforms.weight.requires_grad)
        self.model.cfg['registration_enabled'] = False
        self.model.set_stage(0)
        self.assertFalse(self.model.transforms.weight.requires_grad)

    def test_old_config_and_initialization(self):
        args = copy.deepcopy(self.args)
        args.pop('decomposition')
        baseline = INR_Decoder(args, 'cpu')
        latents = torch.randn(2, *args['inr_decoder']['latent_dim'])
        self.model.initialize_baseline({'inr_decoder': baseline.state_dict(), 'latents': latents})
        torch.testing.assert_close(self.model.shared.grid, latents.mean(0, keepdim=True))
        for k, v in baseline.state_dict().items():
            torch.testing.assert_close(self.model.shared.decoder.state_dict()[k], v)
        baseline.load_state_dict(baseline.state_dict(), strict=True)

    def test_ablation_configs(self):
        for name in ('full', 'registration_only', 'shared_only', 'no_topology', 'no_eikonal', 'unstaged', 'baseline'):
            args = load_config('configs/decomposition/' + name + '.yaml', resolve_subject_ids=False)
            self.assertIn('enabled', args['decomposition'])
        for switch in ('registration_enabled',):
            args = copy.deepcopy(self.args)
            args['decomposition'][switch] = False
            model = AtlasSubjectSDF(args, ['s1', 's2'])
            model.set_stage(2)
            result = model(self.x, self.c, self.ids)
            self.assertTrue(torch.isfinite(result['sdf_mm']).all())

    def test_retired_configuration_rejected(self):
        args = copy.deepcopy(self.args)
        args['decomposition']['residual_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'Residual-era configuration is retired'):
            AtlasSubjectSDF(args, ['s1'])

    def test_checkpoint_roundtrip(self):
        optim = optimizer_for(self.model, self.args)
        before = self.model(self.x, self.c, self.ids)['sdf_mm']
        buffer = io.BytesIO()
        torch.save(checkpoint_payload(self.model, optim, self.args, 2, None), buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=False)
        restored = AtlasSubjectSDF(checkpoint['args'], checkpoint['subject_ids'])
        restored.load_state_dict(checkpoint['model'])
        torch.testing.assert_close(restored(self.x, self.c, self.ids)['sdf_mm'], before)
        torch.testing.assert_close(restored.transforms.weight, self.model.transforms.weight)
        optim2 = optimizer_for(restored, self.args)
        optim2.load_state_dict(checkpoint['optimizer'])

    def test_unseen_adaptation_keeps_population_fixed(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            target = self.model(self.x, self.c, self.ids)['sdf_mm'] + .1
        fit = adapt_subject(self.model, self.x, self.c, target, steps=8)
        self.assertEqual(fit['transform'].shape, (1, 6))
        self.assertNotIn('code', fit)
        self.assertTrue(np.isfinite(fit['history']).all())
        self.assertLess(fit['history'][-1], fit['history'][0])
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_metrics(self):
        tube = np.zeros((12, 12, 12), bool)
        tube[4:7, 4:7, 2:10] = True
        m = vascular_metrics(tube, tube, [1, 1, 1])
        self.assertEqual(m['dice'], 1.)
        self.assertEqual(m['cldice'], 1.)
        self.assertEqual(m['euler_error'], 0)

    def test_one_subject_thin_vessel_overfit(self):
        # A fixed small sample around a 0.8-mm radius cylinder: capacity smoke,
        # not a clinical experiment or population/baseline performance claim.
        torch.manual_seed(4)
        coords = torch.rand(128, 3) * 2 - 1
        coords[:, :2] *= .12
        physical = coords * self.model.registration.half_bbox
        target = physical[:, :2].norm(dim=-1) - .8
        ids = torch.zeros(128, dtype=torch.long)
        cond = torch.zeros(128, 2)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.003)
        history = []
        for _ in range(180):
            optimizer.zero_grad()
            out = self.model(coords, cond, ids)
            loss = (out['sdf_mm']-target).square().mean()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        print(f'Synthetic thin-cylinder MSE: {history[0]:.6f} -> {history[-1]:.6f}')
        self.assertLess(history[-1], history[0] * .15)
        with torch.no_grad():
            pred = self.model(coords, cond, ids)['sdf_mm']
        self.assertGreater(float(((pred < 0) == (target < 0)).float().mean()), .9)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
