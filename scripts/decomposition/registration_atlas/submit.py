"""Submit one inspected registration phase, with immutable code snapshots and job IDs."""
import argparse,json,os,shutil,subprocess,hashlib
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
 p.add_argument('--stage',choices=['R1','R2','groupwise'],required=True);p.add_argument('--base',choices=['R0','R1'],default='R1')
 p.add_argument('--vascular-stage',default='R2');p.add_argument('--feature',choices=['distance','density','binary'],default='distance')
 p.add_argument('--name');p.add_argument('--iterations',type=int,default=2);p.add_argument('--reference');a=p.parse_args()
 study=Path(a.study).resolve();cfg=json.loads((study/'study.json').read_text())
 if not (study/'audit/audit.json').is_file() or not (study/'audit/synthetic_rigid.json').is_file():raise ValueError('Inspect completed audit before registration')
 if a.stage=='R2' and not (study/a.base/'metrics.json').is_file():raise ValueError('Inspect baseline metrics before vessel refinement')
 if a.stage=='groupwise' and not (study/a.vascular_stage/'metrics.json').is_file():raise ValueError('Inspect R2 before groupwise experiments')
 name=a.stage+('_'+a.feature if a.stage=='R2' and a.feature!='distance' else '')
 if a.name:name=a.name
 if a.stage=='groupwise':name+='_'+(a.reference or 'medoid')
 snapshot=study/'sources'/name;snapshot.mkdir(parents=True,exist_ok=False)
 for path in Path(__file__).resolve().parent.glob('*'):
  if path.is_file() and path.suffix in ('.py','.slurm','.md'):shutil.copy2(path,snapshot/path.name)
 (snapshot/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir() if p.is_file()},indent=2)+'\n')
 env=dict(os.environ,SDF_REG_STUDY=str(study),SDF_REG_PIPELINE=str(snapshot/'pipeline.py'))
 jobs=[]
 def submit(label,command,dependency=None):
  cmd=['sbatch','--parsable','--partition=cpu_p1,prepost','--time=00:45:00','--job-name=vascular_'+label,'--output='+str(study/(label+'_%j.log'))]
  if dependency:cmd+=['--dependency=afterok:'+':'.join(dependency)]
  cmd+=[str(snapshot/'cpu.slurm'),*command]
  job=subprocess.check_output(cmd,text=True,env=env).strip().split(';')[0]
  with (study/'jobs.tsv').open('a') as f:f.write(label+'\t'+job+'\n')
  print(label,job,flush=True);return job
 if a.stage=='groupwise':
  cmd=['groupwise','--iterations',str(a.iterations)]
  if a.reference:cmd+=['--reference',a.reference]
  submit(name,cmd)
 else:
  for s in cfg['subjects']:
   jobs.append(submit(name+'_'+s['subject_id'],['register','--stage',a.stage,'--subject',s['subject_id'],'--base',a.base,'--feature',a.feature,'--output-stage',name]))
  submit(name+'_atlas',['atlas','--stage',name],jobs)
if __name__=='__main__':main()
