"""Collect completed and partial small-cohort experiments without hiding failures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True)
    cli=p.parse_args()
    study=Path(cli.study)
    plan=json.loads((study/'study.json').read_text())
    results=[]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for name in ['existing_train',*plan['experiments']]:
        folder=study/name
        row={'experiment':name,'status':'not_started','epoch':None,'mean_Dice':None,
             'mean_precision':None,'mean_recall':None,'mean_volume_ratio':None,
             'subjects':0}
        if (folder/'metrics.json').is_file():
            records=json.loads((folder/'metrics.json').read_text())
            protocol=json.loads((folder/'protocol.json').read_text())
            row.update(status='evaluated',epoch=protocol['epoch'],subjects=len(records))
        elif (folder/'reconstruction_metrics.jsonl').is_file():
            history=[json.loads(line) for line in (folder/'reconstruction_metrics.jsonl').read_text().splitlines() if line.strip()]
            latest=history[-1]
            records=latest['subjects']
            complete=latest['epoch']+1==plan['epochs']
            row.update(status='completed' if complete else 'partial',epoch=latest['epoch'],subjects=len(records))
            epochs=[r['epoch']+1 for r in history]
            scores=[np.mean([s['Dice'] for s in r['subjects'] if s['Dice'] is not None]) for r in history]
            axes[0].plot(epochs,scores,label=name)
            ratios=[np.mean([s['volume_ratio'] for s in r['subjects']]) for r in history]
            axes[1].plot(epochs,ratios,label=name)
        else:
            if folder.exists(): row['status']='started_no_full_grid_metrics'
            results.append(row)
            continue
        for dest,source in [('mean_Dice','Dice'),('mean_precision','precision'),('mean_recall','recall'),('mean_volume_ratio','volume_ratio')]:
            values=[r[source] for r in records if r.get(source) is not None]
            row[dest]=float(np.mean(values)) if values else None
        results.append(row)
    with (study/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (study/'comparison.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    axes[0].set_ylabel('Full-grid training reconstruction Dice');axes[1].set_ylabel('Predicted/reference vessel volume ratio')
    for axis in axes:
        axis.set_xlabel('Epoch')
        if axis.lines: axis.legend(fontsize=7)
    fig.tight_layout();fig.savefig(study/'learning_curves.png',dpi=160)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
