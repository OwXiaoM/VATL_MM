"""Create an isolated four-subject registration study; never read checkpoint weights."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_paths import relocate_path

IDS = ['IXI425-IOP-0988', 'IXI070-Guys-0767', 'IXI278-HH-1771', 'IXI202-HH-1526']

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--metadata', default='/lustre/fsn1/projects/rech/vnc/uil92qd/datasets/data/v1/ixi_metadata_label_t1_fast_sdf.tsv')
    a = p.parse_args()
    with open(relocate_path(a.metadata)) as stream:
        records = list(csv.DictReader(stream, delimiter='\t'))
    subjects = []
    for sid in IDS:
        rows = [r for r in records if r['subject_id'] == sid]
        if len(rows) != 1:
            raise ValueError(f'Expected one metadata row: {sid}')
        row = rows[0]
        paths = {k: relocate_path(row.get(k, '')) for k in ['MRA', 'T1', 'Seg', 'SDF', 'Graph']}
        for k in ['Seg', 'SDF', 'MRA']:
            if not Path(paths[k]).is_file():
                raise FileNotFoundError(paths[k])
        subjects.append({'subject_id': sid, 'age': float(row['scan_age']), 'sex': row['sex'],
                         'paths': paths, 'exists': {k: bool(v) and Path(v).is_file() for k,v in paths.items()}})
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'metadata': relocate_path(a.metadata), 'subjects': subjects, 'seed': 42,
        'analysis_spacing_mm': 1.0, 'registration_spacing_mm': 2.0, 'padding_mm': 8.0,
        'world_bbox_mm': [240,240,120], 'reg_iterations': [60,40,20,0],
        'grad_step': 0.1, 'flow_sigma': 3.0, 'total_sigma': 0.5,
        'vascular_feature_sigma_mm': 2.0, 'vascular_feature_secondary_sigma_mm': 5.0,
        'reference_selection': 'maximum mean R0 pairwise Dice on pairwise intersecting valid FOV; fixed ID order breaks ties',
        'coordinate_frame': 'NIfTI RAS world millimetres minus each native image affine(shape/2); matches model inputs before normalization',
        'fov_definition': 'geometric acquisition support; no original non-padding coverage masks were supplied',
        'purpose': 'registration and population-atlas correspondence diagnostic, not held-out inference',
        'groupwise_max_iterations': 3, 'groupwise_shape_step': 0.2,
        'groupwise_probability_tolerance': 0.001}
    (out/'study.json').write_text(json.dumps(config,indent=2)+'\n')
    (out/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    (out/'git_head.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    source = out/'source'; source.mkdir()
    for path in Path(__file__).resolve().parent.glob('*'):
        if path.is_file() and path.suffix in ('.py','.sh','.slurm','.md'):
            (source/path.name).write_bytes(path.read_bytes())
    (out/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},indent=2)+'\n')
    print(out)

if __name__ == '__main__': main()
