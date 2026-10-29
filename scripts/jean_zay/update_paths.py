"""Apply the Jean Zay path migration; safe to run again."""
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
base = '/lustre/fswork/projects/rech/vnc/uil92qd/projects'
for name in ('config_data.yaml', 'config_atlas.yaml'):
    p = repo / 'configs' / name
    text = p.read_text().replace('/medias/medical/data/Atlas_training/v1', '/lustre/fsn1/projects/rech/vnc/uil92qd/datasets/data/v1').replace('/medias/medical/caravel/zhangx/training_outputs', base + '/VATL_SDF/training_outputs')
    p.write_text(text)
for p in (repo / 'training_script').rglob('*.sh'):
    text = p.read_text().replace('/users/zhangx/VATL_MM', str(repo)).replace('/medias/medical/caravel/zhangx/training_outputs', base + '/VATL_SDF/training_outputs')
    text = text.replace('#SBATCH -p l40s', '#SBATCH --account=vnc@a100\n#SBATCH --qos=qos_gpu_a100-t3\n#SBATCH -C a100')
    p.write_text(text)
for name in ('run.py', 'run_ddp.py'):
    p = repo / name
    text = p.read_text()
    if 'from data_paths import relocate_path' not in text:
        text = 'from data_paths import relocate_path\n' + text if name == 'run.py' else text.replace('import os\n', 'import os\nfrom data_paths import relocate_path\n', 1)
        if name == 'run.py':
            text = text.replace('with open(args["dataset"]["subject_ids"],', 'with open(relocate_path(args["dataset"]["subject_ids"]),')
        else:
            text = text.replace('subject_ids_path = args["dataset"]["subject_ids"]', 'subject_ids_path = relocate_path(args["dataset"]["subject_ids"])')
        p.write_text(text)
p = repo / 'data_loading/dataset.py'
text = p.read_text()
if 'from data_paths import relocate_dataframe' not in text:
    text = 'from data_paths import relocate_dataframe\n' + text
    text = text.replace('self.tsv_file = tsv_file', 'self.tsv_file = relocate_dataframe(tsv_file)').replace('else df_loaded\n', 'else relocate_dataframe(df_loaded)\n')
    p.write_text(text)
for name in ('build_atlas.py', 'build_atlas_ddp.py'):
    p = repo / name
    text = p.read_text()
    if 'from data_paths import relocate_path' not in text:
        text = 'from data_paths import relocate_path\n' + text
        text = text.replace('pd.read_csv(self.args["dataset"]["tsv_file"],', 'pd.read_csv(relocate_path(self.args["dataset"]["tsv_file"]),')
        p.write_text(text)
