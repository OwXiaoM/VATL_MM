"""Render retained checkpoint candidates on validation people only; allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from vesseltok.infer import main as infer_main


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retained-only',action='store_true');parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Allocated GPU required')
    run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=sorted((run/'candidates').glob('step_*.pt'))
    if not args.retained_only:files += [run/'best_bce.pt',run/'last.pt']
    found={}
    for path in files:
        if not path.exists():continue
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c['config'].get('checkpoint_selection')!='native_geometry_v1':raise ValueError('Not a native geometry training run')
        found.setdefault(c['step'],(path,c['splits']['validation']))
    if not found:raise ValueError('No retained native-validation candidates')
    roots=[]
    for step,(checkpoint,subjects) in sorted(found.items()):
        root=out/('step_{:06d}'.format(step));roots.append(str(root))
        with patch.object(sys,'argv',['infer','--checkpoint',str(checkpoint),'--output',str(root),'--subjects',*subjects]):infer_main()
    (out/'candidate_roots.json').write_text(json.dumps(roots,indent=2))
    print('Validation-only candidate roots:',*roots,flush=True)


if __name__=='__main__':main()
