"""Checkpoint cohort audit. Saved training dataframe is authoritative for membership."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import yaml


def canonical_id(value):
    # Repeated sessions of one person may not cross cohort boundaries.
    value = str(value)
    return value.split('-')[0] if value.startswith('IXI') else value


def validate_splits(splits):
    groups = {key: {canonical_id(x) for x in values} for key, values in splits.items() if values}
    for left in groups:
        for right in groups:
            if left < right and groups[left] & groups[right]:
                raise ValueError(f'Subject leakage across {left}/{right}: {sorted(groups[left] & groups[right])}')


def checkpoint_membership(checkpoint, requested):
    frame = checkpoint.get('dataset_df')
    if frame is None or 'subject_id' not in frame:
        raise ValueError('Missing checkpoint training dataframe: unseen status cannot be verified')
    trained = {canonical_id(x) for x in frame.subject_id}
    return {str(x): ('trained' if canonical_id(x) in trained else 'unseen') for x in requested}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(Path(args.manifest).read_text())
    splits = next(iter(manifest.values()))['subject_ids']
    validate_splits(splits)
    validation = splits['val']
    records = []
    for config in sorted(Path(args.root).rglob('config_final.yaml')):
        files = list(config.parent.glob('checkpoint_epoch_*.pth'))
        if not files:
            continue
        checkpoint_path = max(files, key=lambda p: int(p.stem.split('_')[-1]))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        membership = checkpoint_membership(checkpoint, validation)
        record = {'checkpoint': str(checkpoint_path), 'sha256': sha256(checkpoint_path),
                  'epoch': checkpoint['epoch'], 'training_subjects': len(checkpoint['dataset_df']),
                  'main_validation_membership': membership,
                  'main_validation_trained_count': sum(x == 'trained' for x in membership.values()),
                  'spatial_convention': checkpoint['args']['inr_decoder'].get('spatial_convention', 'legacy_normalized')}
        records.append(record)
        print(json.dumps(record), flush=True)
        del checkpoint
    Path(args.output).write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
