"""Synthetic CPU metrics and ordered selection tests; no model/GPU/data required."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib
import pandas as pd
from native_evaluation import (compute_native_metrics, geometry_resample, binary_vessel_metrics,
                               write_metric_results, mm_image)
from data_loading.validation_selection import select_validation, save_subject_manifest


def nifti(data, affine=None):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units('mm')
    return image


class NativeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.args = {'output_dir': str(self.root / 'experiment'), 'rank': 0,
                     'logging': False, 'epochs': {'val': 5}, 'load_model': {'path': '', 'epoch': 0},
                     'evaluation': {'data_range': 1., 'intensity_normalization': 'none'},
                     'dataset': {'modalities': ['MRA', 'T1', 'Seg'], 'sdf_key': 'SDF',
                                 'label_names': ['Background', 'Vessel', 'Tissue']},
                     'inr_decoder': {'out_dim': [2, 3], 'sdf_dim': 1},
                     'optimizer': {'sdf_trunc_mm': 5.}}
        shape = (9, 9, 9)
        self.mra = np.linspace(0, 1, np.prod(shape)).reshape(shape)
        self.t1 = self.mra[::-1].copy()
        self.seg = np.full(shape, 2.)
        self.seg[4, 4, 2:7] = 1
        self.row = {'subject_id': 'synthetic'}
        for name, data in [('MRA', self.mra), ('T1', self.t1), ('Seg', self.seg)]:
            path = self.root / (name + '.nii.gz')
            nib.save(nifti(data), path)
            self.row[name] = str(path)
        self.pred = np.zeros(shape + (8,)) # image(2), hardSeg(1), softSeg(3), SDF, mask
        self.pred[..., 0] = self.mra
        self.pred[..., 1] = self.t1
        self.pred[..., 2] = self.seg
        self.pred[..., 6] = np.where(self.seg == 1, -1., 1.)

    def tearDown(self):
        self.directory.cleanup()

    def test_identical_grid_no_resampling_and_separate_modalities(self):
        with patch('native_evaluation.resample_from_to', side_effect=AssertionError('unexpected resampling')):
            metric = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 4, 'val')
        self.assertEqual(metric['MRA_grid'], 'identical_grid')
        self.assertGreater(metric['MRA_PSNR'], 130)
        self.assertGreater(metric['T1_SSIM'], .99999)
        self.assertEqual(metric['Seg_Vessel_Dice'], 1.)
        self.assertEqual(metric['SDF_Vessel_Dice'], 1.)
        self.assertEqual(metric['Seg_Vessel_HD95_mm'], 0.)
        self.pred[..., 1] = 0
        other = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 4, 'val')
        self.assertNotEqual(other['MRA_PSNR'], other['T1_PSNR'])
        self.assertEqual(other['MRA_reference_voxels'], self.mra.size)

    def test_geometry_resampling_translation_and_labels(self):
        source = np.zeros((5, 5, 5))
        source[1, 2, 2] = 3
        shifted = np.eye(4)
        shifted[0, 3] = 1
        values, mode = geometry_resample(nifti(source, shifted), nifti(np.zeros_like(source)), labels=True)
        self.assertEqual(mode, 'affine_resampled')
        self.assertEqual(values[2, 2, 2], 3)
        self.assertEqual(set(np.unique(values)), {0, 3})
        continuous = np.indices((5, 5, 5))[0].astype(float)
        shifted[0, 3] = .5
        values, _ = geometry_resample(nifti(continuous), nifti(source, shifted), labels=False)
        self.assertAlmostEqual(values[2, 2, 2], 2.5)

    def test_independent_sdf_and_segmentation_and_false_positives(self):
        self.pred[..., 2] = 0
        metric = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 0, 'val')
        self.assertEqual(metric['Seg_Vessel_Dice'], 0.)
        self.assertIsNone(metric['Seg_Vessel_HD95_mm'])
        self.assertEqual(metric['SDF_Vessel_Dice'], 1.)
        # A false positive far outside the reference vessel bbox must be counted.
        self.pred[..., 2] = self.seg
        self.pred[0, 0, 0, 2] = 1
        metric = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 0, 'val')
        self.assertAlmostEqual(metric['Seg_Vessel_Dice'], 10/11)

    def test_physical_distances_empty_masks_and_units(self):
        pred = np.zeros((7, 7, 7), bool)
        truth = pred.copy()
        pred[3, 3, 3], truth[2, 3, 3] = True, True
        affine = np.diag([2., 3., 4., 1.])
        score = binary_vessel_metrics(pred, truth, affine)
        self.assertEqual(score['HD95_mm'], 2.)
        self.assertEqual(score['ASD_mm'], 2.)
        empty = binary_vessel_metrics(pred * 0, truth * 0, affine)
        self.assertIsNone(empty['Dice'])
        self.assertEqual(empty['status'], 'both_empty')
        bad = np.eye(4); bad[0] = 0
        with self.assertRaisesRegex(ValueError, 'Singular'):
            binary_vessel_metrics(pred, truth, bad)
        unknown = nib.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
        with self.assertRaisesRegex(ValueError, 'Unknown spatial units'):
            mm_image(unknown)
        metres = nifti(np.zeros((5, 5, 5)), np.diag([.002, .003, .004, 1.]))
        metres.header.set_xyzt_units('meter')
        np.testing.assert_allclose(mm_image(metres).affine, affine)

    def test_no_sdf_and_invalid_geometry_reported(self):
        args = copy.deepcopy(self.args)
        args['inr_decoder']['sdf_dim'] = 0
        metric = compute_native_metrics(args, self.pred[..., :6], np.eye(4), self.row, 0, 'val')
        self.assertEqual(metric['SDF_Vessel_status'], 'sdf_disabled')
        self.assertIsNone(metric['SDF_Vessel_Dice'])
        invalid = np.eye(4); invalid[0, 0] = np.nan
        metric = compute_native_metrics(args, self.pred, invalid, self.row, 0, 'val')
        self.assertIn('geometry_error', metric)
        self.assertIsNone(metric['Seg_Vessel_Dice'])

    def test_results_summary_full_modality_lists_and_rank_zero(self):
        metric = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 4, 'val')
        summary = write_metric_results(self.args, [metric], 10)
        self.assertIn('MRA_PSNR', summary)
        self.assertIn('T1_PSNR', summary)
        out = self.root / 'experiment/val'
        with (out / 'metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]['Subject'], 'synthetic')
        self.assertEqual(rows[0]['adaptation_epochs'], '5')
        self.assertEqual(json.loads((out / 'subject_ids.json').read_text()), ['synthetic'])
        config = json.loads((out / 'metrics_summary.json').read_text())
        self.assertEqual(config['metrics']['T1_SSIM']['valid_count'], 1)
        self.assertEqual(config['evaluation_configuration']['training_epoch'], 10)
        self.assertTrue((out / 'epoch_0010/metrics.csv').exists())
        other = dict(self.args, output_dir=str(self.root/'rank_one'))
        with patch.dict(os.environ, {'RANK': '1'}):
            write_metric_results(other, [metric], 10)
        self.assertFalse((self.root/'rank_one').exists())

    def test_undefined_metrics_are_not_perfect_scores(self):
        self.pred[..., 2] = 0
        self.pred[..., 6] = 1
        nib.save(nifti(np.zeros_like(self.seg)), self.row['Seg'])
        metric = compute_native_metrics(self.args, self.pred, np.eye(4), self.row, 0, 'val')
        self.assertIsNone(metric['Seg_Vessel_Dice'])
        self.assertIsNone(metric['Dice_multiclass'])
        metric['MRA_PSNR'] = float('inf')
        summary = write_metric_results(self.args, [metric], 0)
        self.assertEqual(summary['Seg_Vessel_Dice']['valid_count'], 0)
        self.assertEqual(summary['MRA_PSNR']['positive_infinite_count'], 1)
        self.assertIsNone(summary['MRA_PSNR']['mean'])


class ValidationSelectionTests(unittest.TestCase):
    def setUp(self):
        self.args = {'n_subjects': {'val': 2},
                     'dataset': {'subject_ids': {'val': ['c', 'a', 'b']}}, 'evaluation': {}}
        self.frame = pd.DataFrame({'subject_id': ['a', 'b', 'c'], 'scan_age': [30, 40, 50]})

    def test_order_and_seed_independence_after_eligibility(self):
        for seed in (1, 42, 999):
            np.random.seed(seed)
            shuffled = self.frame.sample(frac=1)
            self.assertEqual(select_validation(self.args, shuffled).subject_id.tolist(), ['c', 'a'])
        eligible = self.frame[self.frame.subject_id != 'c']
        self.assertEqual(select_validation(self.args, eligible).subject_id.tolist(), ['a', 'b'])

    def test_required_manifest_and_insufficient_eligible(self):
        with tempfile.TemporaryDirectory() as path:
            manifest = Path(path) / 'ids.json'
            manifest.write_text(json.dumps(['c', 'a']))
            self.args['evaluation']['subject_manifest'] = str(manifest)
            self.assertEqual(select_validation(self.args, self.frame).subject_id.tolist(), ['c', 'a'])
            with self.assertRaisesRegex(ValueError, 'missing or ineligible.*c'):
                select_validation(self.args, self.frame[self.frame.subject_id != 'c'])
            manifest.write_text(json.dumps(['a', 'c']))
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                select_validation(self.args, self.frame)
        self.args['evaluation'] = {}
        with self.assertRaisesRegex(ValueError, 'only 1'):
            select_validation(self.args, self.frame.iloc[:1])

    def test_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as path:
            args = dict(self.args, output_dir=path)
            save_subject_manifest(args, ['c', 'a'])
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                save_subject_manifest(args, ['a', 'c'])


if __name__ == '__main__':
    unittest.main()
