"""CPU metadata/file eligibility check; writes the exact ordered val manifest."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml
from data_paths import relocate_path
from data_loading.dataset import Data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-data', default='mra_atlas_synthseg_label_t1_fast_sdf')
    p.add_argument('--subjects', type=int)
    p.add_argument('--subject-manifest')
    p.add_argument('--output', required=True)
    cli = p.parse_args()
    with (ROOT/'configs/config_atlas.yaml').open() as stream:
        args = yaml.safe_load(stream)
    with (ROOT/'configs/config_data.yaml').open() as stream:
        args['dataset'] = yaml.safe_load(stream)[cli.config_data]
    with open(relocate_path(args['dataset']['subject_ids'])) as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']
    if cli.subjects is not None:
        args['n_subjects']['val'] = cli.subjects
    if cli.subject_manifest:
        args.setdefault('evaluation', {})['subject_manifest'] = cli.subject_manifest
    args['output_dir'] = cli.output
    frame = pd.read_csv(relocate_path(args['dataset']['tsv_file']), sep='\t')
    Data(args, frame, split='val') # no __getitem__: no volumes or model weights loaded


if __name__ == '__main__':
    main()
