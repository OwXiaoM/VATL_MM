"""Relocate v1 metadata paths without modifying the shared dataset."""
import os
from pathlib import Path

DEFAULT_DATA_ROOT = "/lustre/fsn1/projects/rech/vnc/uil92qd/datasets/data/v1"


def relocate_path(value):
    if not isinstance(value, str):
        return value
    root = Path(os.environ.get("SDF_DATA_ROOT", DEFAULT_DATA_ROOT))
    # Preserve the relative directory structure, including graph and SDF files.
    if "/v1/" in value:
        return str(root / value.split("/v1/", 1)[1])
    return value


def relocate_dataframe(frame):
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(relocate_path)
    return frame
