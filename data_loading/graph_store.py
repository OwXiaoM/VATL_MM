"""Graph supervision utilities for VATL_MM (V1).

V1 goal
-------
Keep the existing VATL-MM architecture unchanged and add only positive
vessel supervision at graph-node coordinates.

The graph pickle is expected to contain either:
    1) a dict with key ``"graph"`` holding a NetworkX graph, or
    2) a NetworkX graph directly.

For the IXI graph format inspected for this project, every node contains:
    - ``pos``: world / physical coordinate in mm, shape (3,)
    - ``radius``: vessel radius (kept in the cache, not used by V1)

The normalization implemented here intentionally mirrors
``Data.load_coords_and_values`` in ``data_loading/dataset.py``:

    normalized_coord = (world_coord - geometric_center) / (world_bbox / 2)

where ``geometric_center`` is obtained by applying the segmentation affine
(the last configured modality) to the center voxel index defined from the
first modality shape. The existing INR_Decoder then applies the learnable
subject rigid transform, exactly as it does for ordinary voxel coordinates.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch


class GraphStore:
    """Lazy CPU cache for subject-specific vessel graphs.

    Parameters
    ----------
    args:
        Full VATL configuration dictionary.
    dataframe:
        The *already filtered/sampled* DataFrame used by ``Data``.  Its row
        order must therefore be identical to the ``idx_df`` indices produced
        by the current Dataset.
    graph_key:
        TSV column containing Graph.pkl paths. Defaults to
        ``args['graph_supervision']['graph_key']`` and then ``"Graph"``.

    Notes
    -----
    - Graphs are loaded lazily, once per subject, and cached on CPU.
    - V1 samples graph nodes uniformly.
    - No edge/path/branch loss is implemented here yet.
    - ``radius`` is cached for future versions but is deliberately unused in V1.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        dataframe: pd.DataFrame,
        graph_key: Optional[str] = None,
    ) -> None:
        self.args = args
        self.df = dataframe.reset_index(drop=True)

        graph_cfg = args.get("graph_supervision", {})
        configured_key = graph_key or graph_cfg.get("graph_key")
        if configured_key is not None:
            self.graph_key = configured_key
        elif "graph" in self.df.columns:
            self.graph_key = "graph"
        elif "Graph" in self.df.columns:
            self.graph_key = "Graph"
        else:
            self.graph_key = "graph"

        modalities = args["dataset"]["modalities"]
        if len(modalities) < 2:
            raise ValueError(
                "GraphStore expects at least one image modality and one segmentation modality."
            )

        # Match Data.load_coords_and_values exactly:
        #   img_shape <- first modality
        #   affine    <- last modality (Seg)
        self.shape_reference_key = modalities[0]
        self.affine_reference_key = modalities[-1]

        self.world_bbox = np.asarray(
            args["dataset"]["world_bbox"], dtype=np.float64
        )
        if self.world_bbox.shape != (3,) or np.any(self.world_bbox <= 0):
            raise ValueError(
                f"dataset.world_bbox must be three positive values, got {self.world_bbox}."
            )

        if self.graph_key not in self.df.columns:
            raise KeyError(
                f"Graph column '{self.graph_key}' not found in TSV/DataFrame. "
                f"Available columns: {list(self.df.columns)}"
            )

        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _extract_graph(pickle_object: Any) -> nx.Graph:
        """Extract a NetworkX graph from the supported pickle layouts."""
        if isinstance(pickle_object, dict) and "graph" in pickle_object:
            graph = pickle_object["graph"]
        else:
            graph = pickle_object

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "Graph.pkl must contain a NetworkX graph directly or under key 'graph'; "
                f"got {type(graph)!r}."
            )

        if graph.number_of_nodes() == 0:
            raise ValueError("Graph contains no nodes.")

        return graph

    def _get_row(self, idx_df: int) -> pd.Series:
        idx_df = int(idx_df)
        if idx_df < 0 or idx_df >= len(self.df):
            raise IndexError(
                f"idx_df={idx_df} outside GraphStore range [0, {len(self.df) - 1}]."
            )
        return self.df.iloc[idx_df]

    def _get_subject_geometry(self, row: pd.Series):
        """Return geometric center in world mm using VATL's current convention."""
        shape_path = row[self.shape_reference_key]
        affine_path = row[self.affine_reference_key]

        if not isinstance(shape_path, str) or not shape_path:
            raise ValueError(
                f"Invalid {self.shape_reference_key} path: {shape_path!r}"
            )
        if not isinstance(affine_path, str) or not affine_path:
            raise ValueError(
                f"Invalid {self.affine_reference_key} path: {affine_path!r}"
            )

        # Lazy import keeps graph-format inspection usable even in lightweight
        # environments; VATL_MM itself already depends on nibabel.
        import nibabel as nib

        shape_nii = nib.load(shape_path)
        affine_nii = nib.load(affine_path)

        if len(shape_nii.shape) < 3:
            raise ValueError(
                f"Expected >=3D image for {shape_path}, got shape {shape_nii.shape}."
            )

        img_shape = np.asarray(shape_nii.shape[:3], dtype=np.float64)
        affine = np.asarray(affine_nii.affine, dtype=np.float64)

        # This reproduces dataset.py:
        #   img_center_index = np.array(img_shape) / 2.0
        #   geometric_center = nib.affines.apply_affine(affine, img_center_index)
        img_center_index = img_shape / 2.0
        geometric_center = nib.affines.apply_affine(affine, img_center_index)

        return geometric_center

    def normalize_world_coords(
        self,
        pos_world: np.ndarray,
        row: pd.Series,
    ) -> np.ndarray:
        """Convert graph world coordinates to VATL normalized coordinates."""
        pos_world = np.asarray(pos_world, dtype=np.float64)
        if pos_world.ndim != 2 or pos_world.shape[1] != 3:
            raise ValueError(
                f"Expected graph positions with shape (N, 3), got {pos_world.shape}."
            )

        geometric_center = self._get_subject_geometry(row)
        coords = pos_world - geometric_center[None, :]
        coords = coords / (self.world_bbox[None, :] / 2.0)
        return coords

    def _load_subject(self, idx_df: int) -> Dict[str, Any]:
        """Load, normalize, filter, and cache one subject graph."""
        idx_df = int(idx_df)
        if idx_df in self._cache:
            return self._cache[idx_df]

        row = self._get_row(idx_df)
        graph_path_value = row[self.graph_key]

        if not isinstance(graph_path_value, str) or not graph_path_value.strip():
            raise ValueError(
                f"Subject row {idx_df} has invalid graph path in column "
                f"'{self.graph_key}': {graph_path_value!r}"
            )

        graph_path = Path(graph_path_value)
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Graph file does not exist for idx_df={idx_df}: {graph_path}"
            )

        with graph_path.open("rb") as f:
            pickle_object = pickle.load(f)

        graph = self._extract_graph(pickle_object)
        node_ids = list(graph.nodes())

        try:
            pos_world = np.asarray(
                [graph.nodes[node_id]["pos"] for node_id in node_ids],
                dtype=np.float64,
            )
        except KeyError as exc:
            raise KeyError(
                f"Every graph node must contain node['pos']; failed for {graph_path}."
            ) from exc

        if pos_world.shape != (len(node_ids), 3):
            raise ValueError(
                f"Node positions in {graph_path} must have shape (N, 3); "
                f"got {pos_world.shape}."
            )

        if not np.all(np.isfinite(pos_world)):
            raise ValueError(f"Non-finite graph node coordinates found in {graph_path}.")

        # Radius is not used by V1, but retaining it now avoids re-reading the
        # pickle when later experiments start using vessel calibre.
        radius = np.asarray(
            [graph.nodes[node_id].get("radius", np.nan) for node_id in node_ids],
            dtype=np.float32,
        )

        coords_norm = self.normalize_world_coords(pos_world, row)

        # Match the current dataset world-BBox policy: graph nodes outside the
        # INR field of view are excluded from V1 supervision.
        inside = np.all(np.abs(coords_norm) <= 1.0, axis=1)

        valid_indices = np.flatnonzero(inside)
        coords_norm_valid = coords_norm[inside].astype(np.float32, copy=False)
        pos_world_valid = pos_world[inside].astype(np.float32, copy=False)
        radius_valid = radius[inside]
        node_ids_valid = [node_ids[i] for i in valid_indices]

        if len(coords_norm_valid) == 0:
            raise ValueError(
                f"No graph nodes from {graph_path} fall inside VATL world_bbox "
                f"{self.world_bbox.tolist()}. Check graph/NIfTI coordinate alignment."
            )

        cached = {
            # Fields used by V1
            "coords": torch.from_numpy(coords_norm_valid),
            # Debug/future fields
            "pos_world": torch.from_numpy(pos_world_valid),
            "radius": torch.from_numpy(radius_valid),
            "node_ids": node_ids_valid,
            "n_nodes_total": len(node_ids),
            "n_nodes_inside": len(node_ids_valid),
            "inside_fraction": float(inside.mean()),
            "graph_path": str(graph_path),
            "n_edges": graph.number_of_edges(),
        }

        self._cache[idx_df] = cached
        return cached

    def get_subject(self, idx_df: int) -> Dict[str, Any]:
        """Return the cached graph record for one subject."""
        return self._load_subject(idx_df)

    def sample_nodes(
        self,
        idx_df: int,
        n_points: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Uniformly sample V1 graph-node coordinates for one subject.

        Sampling is without replacement when enough nodes are available and
        with replacement otherwise. Returned shape is ``(n_points, 3)``.
        """
        n_points = int(n_points)
        if n_points <= 0:
            raise ValueError(f"n_points must be > 0, got {n_points}.")

        record = self._load_subject(int(idx_df))
        coords = record["coords"]
        n_available = coords.shape[0]

        if n_available >= n_points:
            sample_idx = torch.randperm(n_available)[:n_points]
        else:
            sample_idx = torch.randint(0, n_available, (n_points,))

        sampled = coords[sample_idx]
        if device is not None:
            sampled = sampled.to(device=device, non_blocking=True)
        return sampled

    def summary(self, idx_df: int) -> Dict[str, Any]:
        """Small diagnostic summary useful before starting training."""
        record = self._load_subject(idx_df)
        return {
            "idx_df": int(idx_df),
            "graph_path": record["graph_path"],
            "n_nodes_total": record["n_nodes_total"],
            "n_nodes_inside": record["n_nodes_inside"],
            "inside_fraction": record["inside_fraction"],
            "n_edges": record["n_edges"],
            "coord_min": record["coords"].min(dim=0).values.tolist(),
            "coord_max": record["coords"].max(dim=0).values.tolist(),
        }

    def clear(self) -> None:
        """Drop all cached graphs."""
        self._cache.clear()