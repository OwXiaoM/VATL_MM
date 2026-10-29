"""CPU checks for integration with both existing builders; networks are stubbed."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import utils
from build_atlas import AtlasBuilder
from build_atlas_ddp import AtlasBuilderDDP
from data_loading.dataset import Data
import test_native_evaluation as fixtures


class ValidationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeMetricsTests()
        self.fixture.setUp()
        self.args = copy.deepcopy(self.fixture.args)
        self.args.update({'compute_metrics': True, 'save_imgs': {'val': False, 'train': False},
                          'generate_cond_atlas': False, 'test_time_adaptation': {'enabled': True},
                          'save_certainty_maps': False, 'seed': 42})
        self.args['dataset']['world_bbox'] = [20, 20, 20]
        self.args['dataset']['conditions'] = {}

    def tearDown(self):
        self.fixture.tearDown()

    def test_no_registration_during_metrics_export_or_legacy_resampling(self):
        self.assertIsNone(utils.compute_dice(np.zeros((3,3,3)), np.zeros((3,3,3)), 0))
        self.args['save_imgs']['val'] = True
        with patch('utils.ants.registration', side_effect=AssertionError('registration forbidden')) as registration:
            utils.compute_metrics(self.args, self.fixture.pred, np.eye(4), self.fixture.row, 0, 'val')
            utils.save_subject(self.args, self.fixture.pred, np.eye(4), sub_row_df=self.fixture.row, epoch=0, split='val')
            image = nib.load(self.fixture.row['MRA'])
            _, aligned, transform = utils.reg_imgs(image, image, reg_type='Rigid')
            np.testing.assert_allclose(image.get_fdata(), aligned.get_fdata())
            self.assertIsNone(transform)
            registration.assert_not_called()
        self.assertTrue((Path(self.args['output_dir'])/'val/MRA_synthetic_ref.nii.gz').exists())

    def test_generation_metrics_and_saving_independent_both_builders(self):
        for builder_class in (AtlasBuilder, AtlasBuilderDDP):
            for metrics, save in ((True, False), (False, True), (True, True), (False, False)):
                args = copy.deepcopy(self.args)
                args.update(compute_metrics=metrics, save_imgs={'val': save})
                builder = builder_class.__new__(builder_class)
                builder.args, builder.device = args, 'cpu'
                dataset = SimpleNamespace(df=pd.DataFrame([self.fixture.row]), load_conditions=lambda row: torch.zeros(2))
                builder.datasets = {'val': dataset}
                model = Mock()
                model.inference.return_value = torch.as_tensor(self.fixture.pred, dtype=torch.float32)
                builder.inr_decoder = {'val': model}
                builder.latents = {'val': torch.zeros(1, 2, 1, 1, 1)}
                builder.transformations = {'val': torch.ones(1, 6) * .125}
                original_tf = builder.transformations['val'].clone()
                module = 'build_atlas_ddp' if builder_class is AtlasBuilderDDP else 'build_atlas'
                with patch(module + '.save_subject') as save_call, patch('utils.ants.registration', side_effect=AssertionError('registration forbidden')):
                    result = builder.generate_subjects_from_df([0], epoch=0, split='val')
                self.assertEqual(len(result), int(metrics))
                self.assertEqual(save_call.call_count, int(save))
                self.assertEqual(model.inference.call_count, int(metrics or save))
                torch.testing.assert_close(builder.transformations['val'], original_tf)
                with patch.dict(os.environ, {'RANK': '1'}):
                    self.assertEqual(builder.generate_subjects_from_df([0], split='val'), [])

    def test_metrics_generated_only_after_adaptation_finishes(self):
        for builder_class in (AtlasBuilder, AtlasBuilderDDP):
            builder = builder_class.__new__(builder_class)
            builder.args = copy.deepcopy(self.args)
            builder.args['epochs']['val'] = 3
            dataset = SimpleNamespace(df=pd.DataFrame([self.fixture.row]), __len__=lambda: 1)
            # validate() calls len(dataset), so use a minimal concrete wrapper.
            class Dataset:
                df = dataset.df
                def __len__(self):
                    return 1
            builder.datasets = {'train': Dataset(), 'val': Dataset()}
            builder.save_state = Mock()
            builder._init_validation = Mock()
            completed = []
            builder.train_epoch = lambda epoch, split: completed.append(epoch)
            builder._update_scheduler = Mock()
            builder.analyze_latent_space = Mock()
            def generate(idcs_df, epoch, split):
                if split == 'val':
                    self.assertEqual(completed, [0, 1, 2])
                return []
            builder.generate_subjects_from_df = Mock(side_effect=generate)
            builder.validate(7)
            self.assertEqual(builder.generate_subjects_from_df.call_count, 2)

    def test_dataset_eligible_order_and_training_sampler_unchanged(self):
        args = copy.deepcopy(self.args)
        ids = ['c', 'a', 'b']
        args['dataset'].update(subject_ids={'val': ids, 'train': ids}, constraints={})
        args['n_subjects'] = {'val': 2, 'train': 2}
        rows = []
        for subject in ('a', 'b', 'c'):
            row = dict(self.fixture.row, subject_id=subject)
            row['SDF'] = self.fixture.row['MRA']
            rows.append(row)
        frame = pd.DataFrame(rows)
        # c has no T1 file: eligibility must skip it, preserving a then b.
        frame.loc[frame.subject_id == 'c', 'T1'] = str(self.fixture.root/'missing.nii.gz')
        context = Data.__new__(Data)
        context.args = args
        context.split = 'val'
        context.sdf_key = 'SDF'
        context.target_keys = ['MRA', 'T1', 'Seg', 'SDF']
        with patch.object(context, 'sample_subjects', side_effect=AssertionError('validation must not sample')):
            for seed in (1, 99):
                np.random.seed(seed)
                result = context.filter_dataframe(frame.sample(frac=1))
                self.assertEqual(result.subject_id.tolist(), ['a', 'b'])
        context.split = 'train'
        with patch.object(context, 'sample_subjects', return_value=frame.iloc[:2]) as sampler:
            context.filter_dataframe(frame)
            sampler.assert_called_once()

    def test_legacy_metric_lists_log_all_modalities(self):
        metrics = [{'Subject': 'synthetic', 'PSNR': [10., 20.], 'SSIM': [.2, .8], 'DICE': [.5]}]
        utils.log_metrics(self.args, metrics, 2, split='val')
        path = Path(self.args['output_dir'])/'val/metrics_summary.json'
        result = json.loads(path.read_text())['metrics']
        self.assertEqual(result['MRA_PSNR']['mean'], 10.)
        self.assertEqual(result['T1_PSNR']['mean'], 20.)
        self.assertEqual(result['T1_SSIM']['mean'], .8)


if __name__ == '__main__':
    unittest.main()
