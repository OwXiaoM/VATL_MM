"""Start the finer-grid phase in a NEW isolated follow-up; CPU only."""
import argparse,datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path

def submit_prepared(out):
 cfg=json.loads((out/'followup.json').read_text())
 ledger=out/'jobs.tsv'
 if ledger.exists() and any(line.startswith('export\t') for line in ledger.read_text().splitlines()):raise ValueError('Export already submitted')
 if Path(cfg['fine_study']).exists():raise ValueError('Export output already exists; choose a NEW follow-up')
 snap=out/'source/export'
 env=dict(os.environ,SDF_REG_STUDY=cfg['source_study'],SDF_REG_PIPELINE=str(snap/'regrid.py'))
 # Jean Zay allocates memory with CPU cores; explicit --mem flags are forbidden.
 resources=['--time=02:00:00','--cpus-per-task=16'] if cfg['spacing_mm']<.8 else ['--time=01:00:00']
 cmd=['sbatch','--parsable','--partition=cpu_p1,prepost',*resources,'--output='+str(out/'export_%j.log'),str(snap/'cpu.slurm'),'--output',cfg['fine_study'],'--spacing',str(cfg['spacing_mm'])]
 job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
 with ledger.open('a') as f:f.write('export\t'+job+'\n')
 print(str(out));print('export',job)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--submit-prepared',action='store_true')
 p.add_argument('--spacing',type=float,default=.8);p.add_argument('--epochs',type=int,default=500)
 a=p.parse_args();out=Path(a.output).resolve()
 if a.submit_prepared:
  submit_prepared(out);return
 if not a.source:p.error('--source is required when creating a new follow-up')
 source=Path(a.source).resolve()
 if a.spacing<=0 or a.epochs<1:raise ValueError('Positive spacing and epoch budget required')
 for stage in ['R0_ants','R1','R2_incremental']:
  if not (source/stage/'metrics.json').is_file():raise ValueError('Completed source registration required')
 out.mkdir(parents=True,exist_ok=False);snap=out/'source/export';snap.mkdir(parents=True)
 directory=Path(__file__).resolve().parents[1]
 for name in ['pipeline.py','regrid.py','cpu.slurm']:
  shutil.copy2(directory/name,snap/name)
 cfg={'source_study':str(source),'fine_study':str(out/'fine'),'spacing_mm':a.spacing,'epochs':a.epochs,'seed':42,
  'arms':{'existing':'R0_ants','prealigned':'R2_incremental'},'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'gpu_submission':'requires passing QC/cache preflight; no old checkpoint resume',
  'preservation_gate':{'identity_Dice_min':.98,'identity_centerline_recall_min':.98,'prealigned_Dice_min':.95,'prealigned_centerline_recall_min':.95,
   'fold_fraction_max':0.,'inverse_consistency_p95_max_mm':.1,'trusted_vessel_fraction_min':.8},
  'gate_note':'Predeclared engineering screening thresholds, not clinical or normative validation. Never relax automatically.'}
 (out/'followup.json').write_text(json.dumps(cfg,indent=2)+'\n')
 (snap/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir()},indent=2)+'\n')
 submit_prepared(out)
if __name__=='__main__':main()
