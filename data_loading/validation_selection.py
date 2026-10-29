"""Ordered validation selection and rank-zero subject manifests; no RNG calls."""
import json
import os
from pathlib import Path


def is_primary(args):
    return int(os.environ.get('RANK', args.get('rank', 0))) == 0


def read_manifest(path):
    with open(path, encoding='utf-8') as stream:
        ids = json.load(stream)
    if not isinstance(ids, list) or any(not isinstance(s, str) for s in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid ordered subject manifest: {path}')
    return ids


def select_validation(args, eligible):
    count = args['n_subjects']['val']
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('n_subjects.val must be a nonnegative integer')
    ordered = [str(s) for s in (args['dataset']['subject_ids'].get('val') or [])]
    if len(set(ordered)) != len(ordered):
        raise ValueError('Validation split contains duplicate subject IDs')
    indexed = eligible.copy()
    indexed['subject_id'] = indexed['subject_id'].astype(str)
    indexed = indexed[indexed.subject_id.isin(ordered)]
    if indexed.subject_id.duplicated().any():
        raise ValueError('Validation metadata contains ambiguous duplicate subject IDs')
    indexed = indexed.set_index('subject_id', drop=False)
    available = [s for s in ordered if s in indexed.index]
    selected = available[:count]
    manifest = args.get('evaluation', {}).get('subject_manifest')
    if manifest:
        required = read_manifest(manifest)
        missing = [s for s in required if s not in available]
        if missing:
            raise ValueError(f'Required validation subjects missing or ineligible: {missing}')
        if selected != required:
            raise ValueError(f'Validation subject manifest mismatch: selected={selected}, required={required}; check n_subjects.val and split ordering')
    if len(selected) != count:
        raise ValueError(f'Requested {count} validation subjects but only {len(available)} are eligible in the predefined split')
    return indexed.loc[selected].reset_index(drop=True)


def save_subject_manifest(args, ids, split='val', directory=None):
    ids = [str(s) for s in ids]
    if not is_primary(args):
        return
    path = Path(directory) if directory else Path(args['output_dir']) / split
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / 'subject_ids.json'
    if manifest.exists() and read_manifest(manifest) != ids:
        raise ValueError(f'Refusing to overwrite a different evaluation subject list: {manifest}')
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(ids, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest)
    print(f'[{split}] Ordered evaluation subjects: {ids}', flush=True)
