"""Submit a single gated follow-up phase and record immutable code plus job order."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path


def snapshot_repo(root):
    destination=root/'source/repo'
    if destination.exists():return destination
    repo=Path(__file__).resolve().parents[4];destination.mkdir(parents=True,exist_ok=False)
    paths=subprocess.check_output(['rg','--files','-g','*.py','-g','*.yaml','-g','*.yml','-g','*.sh','-g','*.slurm'],cwd=repo,text=True).splitlines()
    hashes={}
    for name in paths:
        source=repo/name;target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'source/repo_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
    p.add_argument('--phase',choices=['preflight','existing','prealigned','evaluate'],required=True);a=p.parse_args()
    root=Path(a.study).resolve();cfg=json.loads((root/'followup.json').read_text())
    jobs=dict(line.split('\t') for line in (root/'jobs.tsv').read_text().splitlines())
    if a.phase in jobs:raise ValueError('Already submitted; inspect jobs.tsv instead of duplicating')
    repo=snapshot_repo(root)
    env=dict(os.environ,SDF_REPO=str(repo),SDF_FOLLOWUP=str(root))
    dependency=None
    if a.phase=='preflight':
        dependency=jobs['export'];env.update(SDF_REG_STUDY=str(root),SDF_REG_PIPELINE=str(repo/'scripts/decomposition/registration_atlas/next_steps/preflight.py'))
        command=[str(repo/'scripts/decomposition/registration_atlas/cpu.slurm')]
        extra=['--partition=cpu_p1,prepost','--time=01:00:00']
    else:
        gate_path=root/'preflight/gate.json'
        if not gate_path.is_file() or not json.loads(gate_path.read_text())['passed']:raise ValueError('GPU gate closed: inspect completed preflight first')
        if a.phase=='prealigned':
            if 'existing' not in jobs:raise ValueError('Submit the existing-alignment control first')
            dependency=jobs['existing']
        if a.phase=='evaluate':
            if not all(x in jobs for x in ['existing','prealigned']):raise ValueError('Both controlled fits must be submitted first')
            dependency=jobs['prealigned']
        command=[str(repo/'scripts/decomposition/registration_atlas/next_steps/gpu.slurm'),a.phase]
        extra=['--time=04:00:00'] if cfg['spacing_mm']<=.5 and a.phase!='evaluate' else []
    sbatch=['sbatch','--parsable','--job-name=atlas_'+a.phase,'--output='+str(root/(a.phase+'_%j.log')),*extra]
    if dependency:sbatch+=['--dependency=afterok:'+dependency]
    job=subprocess.check_output([*sbatch,*command],env=env,text=True).strip().split(';')[0]
    with (root/'jobs.tsv').open('a') as f:f.write(a.phase+'\t'+job+'\n')
    print(a.phase,job,'after',dependency)
if __name__=='__main__':main()
