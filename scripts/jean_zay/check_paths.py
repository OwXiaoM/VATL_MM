"""Check metadata and referenced files without loading volumes or using GPUs."""
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_paths import DEFAULT_DATA_ROOT, relocate_path
import os
root = Path(os.environ.get('SDF_DATA_ROOT', DEFAULT_DATA_ROOT))
if not root.is_dir():
    sys.exit(f'Dataset directory missing: {root}')
missing = set()
for name in ('subject_ids_label_t1_fast.yaml', 'ixi_metadata_label_t1_fast_sdf.tsv'):
    if not (root / name).is_file():
        missing.add(str(root / name))
metadata = root / 'ixi_metadata_label_t1_fast_sdf.tsv'
rows = 0
if metadata.is_file():
    with metadata.open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            rows += 1
            for key in ('MRA', 'T1', 'Seg', 'SDF', 'Graph'):
                value = row.get(key)
                if value and not Path(relocate_path(value)).is_file():
                    missing.add(relocate_path(value))
for path in sorted(missing):
    print(f'Missing: {path}')
print(f'Checked {rows} metadata rows; {len(missing)} missing files')
sys.exit(bool(missing))
