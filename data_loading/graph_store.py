"""Cache and sample subject vessel graphs in normalized atlas coordinates.

Graph positions are world coordinates in mm. Optional native-space SDF
targets are interpolated before learned coordinate transforms. Radius
constraints are enabled separately and checked against the current SDF.
Normalization matches Data: (world - geometric_center) / (world_bbox / 2).
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
    """Lazy CPU cache indexed by the filtered training DataFrame row order.

    Accept NetworkX graphs directly or in a pickle dictionary under "graph".
    Nodes contain world-space "pos" and optional scalar "radius" attributes.
    Only graph nodes inside world_bbox are retained. No edge loss is applied.
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

        self.shape_reference_key = modalities[0]
        self.affine_reference_key = modalities[-1]

        self.world_bbox = np.asarray(args["dataset"]["world_bbox"], dtype=np.float64)
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
            raise ValueError(f"Invalid {self.shape_reference_key} path: {shape_path!r}")
        if not isinstance(affine_path, str) or not affine_path:
            raise ValueError(
                f"Invalid {self.affine_reference_key} path: {affine_path!r}"
            )

        import nibabel as nib

        shape_nii = nib.load(shape_path)
        affine_nii = nib.load(affine_path)

        if len(shape_nii.shape) < 3:
            raise ValueError(
                f"Expected >=3D image for {shape_path}, got shape {shape_nii.shape}."
            )

        img_shape = np.asarray(shape_nii.shape[:3], dtype=np.float64)
        affine = np.asarray(affine_nii.affine, dtype=np.float64)

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
            raise ValueError(
                f"Non-finite graph node coordinates found in {graph_path}."
            )

        radius = np.asarray(
            [graph.nodes[node_id].get("radius", np.nan) for node_id in node_ids],
            dtype=np.float32,
        )

        coords_norm = self.normalize_world_coords(pos_world, row)

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
            "coords": torch.from_numpy(coords_norm_valid),
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
        return_sdf: bool = False,
        return_radius: bool = False,
    ):
        """Sample matched graph coordinates and optional scalar targets.

        Sample without replacement when enough nodes exist, otherwise repeat.
        The default return is coords (N, 3). Flags append sdf_mm and/or radius_mm
        to the tuple in that order. Invalid radii remain invalid for loss filtering.
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
        result = [sampled]
        if return_sdf:
            targets = self._load_sdf_targets(int(idx_df), record)[sample_idx]
            if device is not None:
                targets = targets.to(device=device, non_blocking=True)
            result.append(targets)
        if return_radius:
            scale = float(
                self.args.get("graph_supervision", {})
                .get("radius", {})
                .get("scale_to_mm", 1.0)
            )
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "graph_supervision.radius.scale_to_mm must be finite and positive"
                )
            if record["radius"].shape != (n_available,):
                raise ValueError("Every graph radius must be a scalar")
            radius = record["radius"][sample_idx] * scale
            if device is not None:
                radius = radius.to(device=device, non_blocking=True)
            result.append(radius)
        return tuple(result) if len(result) > 1 else sampled

    def _load_sdf_targets(self, idx_df, record):
        """Cache trilinear SDF samples, not the full SDF volume, on CPU."""
        if "sdf_mm" in record:
            return record["sdf_mm"]
        import nibabel as nib
        from scipy.ndimage import map_coordinates

        key = self.args["dataset"].get("sdf_key")
        row = self._get_row(idx_df)
        if not key or key not in row or not isinstance(row[key], str) or not row[key]:
            raise ValueError(
                "Graph SDF supervision requires dataset.sdf_key and SDF paths in the TSV"
            )
        image = nib.load(row[key])
        reference = nib.load(row[self.affine_reference_key])
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-2
        ):
            raise ValueError("Graph SDF and segmentation must share shape and affine")
        sdf = image.get_fdata(dtype=np.float32)
        if sdf.ndim != 3 or not np.isfinite(sdf).all():
            raise ValueError("Graph SDF targets require a finite 3D distance volume")
        voxel_coords = nib.affines.apply_affine(
            np.linalg.inv(image.affine), record["pos_world"].numpy()
        )
        upper = np.array(sdf.shape) - 1
        if np.any(voxel_coords < -1e-4) or np.any(voxel_coords > upper + 1e-4):
            raise ValueError(
                "Graph nodes lie outside the SDF voxel grid; check graph/NIfTI alignment"
            )
        voxel_coords = np.clip(voxel_coords, 0, upper)
        targets = map_coordinates(
            sdf, voxel_coords.T, order=1, mode="nearest", prefilter=False
        )
        record["sdf_mm"] = torch.from_numpy(targets.astype(np.float32, copy=False))
        return record["sdf_mm"]

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
