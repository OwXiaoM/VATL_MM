"""Paired-cache integrity checks used by training regression tests."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from diagnostics.provenance import sha256

def verify_pair(small_path,large_path):
    with np.load(small_path) as a,np.load(large_path) as b:
        assert set(a.files)==set(b.files)
        equal=[]
        for key in a.files:
            if key not in ['cloud','padding']:
                np.testing.assert_array_equal(a[key],b[key]);equal.append(key)
        assert len(a['cloud'])==len(b['cloud'])
        for small,smask,large,lmask in zip(a['cloud'],a['padding'],b['cloud'],b['padding']):
            np.testing.assert_array_equal(cKDTree(large[~lmask]).query(small[~smask])[0],0.)
        return {'identical_arrays':equal,'rows_with_exact_nested_clouds':len(a['cloud']),
                'small_sha256':sha256(small_path),'large_sha256':sha256(large_path)}
