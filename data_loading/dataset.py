import os
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import pandas as pd
import torchio as tio
from torch.utils.data import Dataset
from utils import *


class Data(Dataset):
    def __init__(self, args, tsv_file, split, df_loaded=None):
        self.args = args
        self.split = split
        self.modality_keys = self.args["dataset"]["modalities"]
        self.sdf_key = self.args["dataset"].get("sdf_key")
        if self.sdf_key in self.modality_keys:
            raise ValueError("SDF must be separate from dataset.modalities")
        self.target_keys = list(self.modality_keys) + (
            [self.sdf_key] if self.sdf_key else []
        )
        self.world_bbox = np.array(self.args["dataset"]["world_bbox"])
        self.tsv_file = tsv_file
        self.df = (
            self.filter_dataframe(self.tsv_file) if df_loaded is None else df_loaded
        )

        self.cache = {}
        # Cache remains disabled; augmentation is applied after loading.
        self.use_cache = False
        if self.split == "train":
            print(f"[{split}] Data Caching Enabled: {self.use_cache}")

        self._init_data_augmentation()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            modalities = self.cache[idx]
        else:
            row_dict = self.df.iloc[idx].to_dict()
            modalities = self.load_modalities(row_dict)
            if self.use_cache:
                self.cache[idx] = modalities

        if self.split == "train" and self.args.get("task_specific_sampling", {}).get("enabled", False):
            return self.task_specific_item(modalities, idx)
        coords, values = self.load_coords_and_values(modalities)

        coords = torch.tensor(coords, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        row_dict = self.df.iloc[idx].to_dict()
        conditions = self.load_conditions(row_dict)[None, :].expand(coords.shape[0], -1)
        idx_df = (
            torch.tensor(idx, dtype=torch.int32)
            .unsqueeze(0)
            .expand(coords.shape[0], -1)
        )
        return coords, values, conditions, idx_df

    def collate_fn(self, batch, shuffle=True):
        if isinstance(batch[0], dict):
            return {"image": self.collate_fn([b["image"] for b in batch], shuffle),
                    "sdf": self.collate_sdf([b["sdf"] for b in batch], shuffle),
                    "sampling": [b["sampling"] for b in batch]}
        coords = torch.concat([b[0] for b in batch], dim=0)
        values = torch.concat([b[1] for b in batch], dim=0)
        conditions = torch.concat([b[2] for b in batch], dim=0)
        idx_df = torch.concat([b[3] for b in batch], dim=0)
        if shuffle:
            perm = torch.randperm(coords.shape[0])
            coords = coords[perm]
            values = values[perm]
            conditions = conditions[perm]
            idx_df = idx_df[perm]
        return coords, values, conditions, idx_df

    def collate_sdf(self, batch, shuffle):
        arrays = self.collate_fn(batch, shuffle=False)
        if shuffle:
            # Separate RNG: adding the SDF shuffle must not perturb image/graph RNG.
            generator = torch.Generator().manual_seed(torch.initial_seed())
            perm = torch.randperm(len(arrays[0]), generator=generator)
            arrays = tuple(x[perm] for x in arrays)
        return arrays

    def load_modalities(self, row_dict):
        modalities = {}
        for i, mod_key in enumerate(self.target_keys):
            if mod_key not in row_dict:
                raise ValueError(f"Modality {mod_key} not found in row_dict")
            if row_dict[mod_key] == "":
                raise ValueError(f"Modality {mod_key} is empty")
            modalities[mod_key] = nib.load(row_dict[mod_key])
            modalities[mod_key] = squeeze_modalities(
                modalities[mod_key], path=row_dict[mod_key], write2disk=False
            )
        shapes = [modalities[mod_key].shape for mod_key in modalities]
        if len(set(shapes)) != 1:
            raise ValueError(f"Modalities have different shapes: {shapes}")

        affines = [modalities[mod_key].affine for mod_key in modalities]
        for i, affine in enumerate(affines):
            if i == 0 or np.allclose(affines[i], affines[i - 1], atol=1e-2):
                continue
            else:
                raise ValueError(f"Modalities have different affines: {affines}")

        if self.args["mask_reconstruction"]:
            mask = modalities[self.modality_keys[-1]].get_fdata() > 0
            for mod_key in self.modality_keys[:-1]:
                modalities[mod_key] = mask_nifti(modalities[mod_key], mask)
            modalities[self.modality_keys[-1]] = add_background_halo(
                self.args["dataset"]["label_names"], modalities[self.modality_keys[-1]]
            )

        return modalities

    def sample_sdf_indices(
        self, sdf, segmentation, affine, normalize=True, num_points=500000
    ):
        """Sample a training mixture using unscaled distances in millimetres.

        Interior and surface pools overlap intentionally. Empty pools donate their
        quotas proportionally to the remaining nonempty pools. Small pools are
        sampled with replacement. Every returned index refers to a real voxel.
        """
        config = self.args["dataset"].get("sdf_sampling", {})
        band_mm = float(config.get("surface_band_mm", 2.0))
        names = ("interior", "surface", "tissue", "background")
        defaults = (0.25, 0.35, 0.25, 0.15)
        fractions = config.get("fractions", dict(zip(names, defaults)))
        if set(fractions) != set(names):
            raise ValueError(f"sdf_sampling.fractions must contain exactly {names}")
        weights = np.array([fractions[name] for name in names], dtype=float)
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError(
                "SDF sampling fractions must be finite, nonnegative, and sum to 1"
            )
        if not np.isfinite(band_mm) or band_mm <= 0:
            raise ValueError("sdf_sampling.surface_band_mm must be finite and positive")
        if (
            isinstance(num_points, bool)
            or not isinstance(num_points, (int, np.integer))
            or num_points <= 0
        ):
            raise ValueError("sdf_sampling.num_points must be a positive integer")
        if sdf.shape != segmentation.shape or sdf.ndim != 3:
            raise ValueError("SDF and segmentation must have the same 3D shape")
        if not np.isfinite(sdf).all():
            raise ValueError("SDF sampling requires finite distances")

        # Check the crop in chunks rather than allocating a full-volume coordinate grid.
        valid = np.ones(sdf.size, dtype=bool)
        if normalize:
            center = nib.affines.apply_affine(affine, np.array(sdf.shape) / 2.0)
            half_bbox = self.world_bbox / 2.0
            if not np.isfinite(half_bbox).all() or np.any(half_bbox <= 0):
                raise ValueError("world_bbox must contain positive finite sizes")
            for start in range(0, sdf.size, 262144):
                stop = min(start + 262144, sdf.size)
                indices = np.column_stack(
                    np.unravel_index(np.arange(start, stop), sdf.shape)
                )
                coords = (
                    nib.affines.apply_affine(affine, indices) - center
                ) / half_bbox
                valid[start:stop] = np.all(np.abs(coords) <= 1.0, axis=1)
        valid = valid.reshape(sdf.shape)
        if not valid.any():
            raise ValueError("No SDF voxels lie inside world_bbox")

        # Interior and surface pools overlap to preserve thin-vessel coverage.
        def pool_mask(name):
            if name == "interior":
                return valid & (sdf < 0)
            if name == "surface":
                return valid & (np.abs(sdf) <= band_mm)
            if name == "tissue":
                return valid & (sdf > band_mm) & (segmentation > 0)
            return valid & (sdf > band_mm) & (segmentation == 0)

        available = np.array([np.any(pool_mask(name)) for name in names])
        weights *= available
        # Fall back to real in-bounds voxels if every requested pool is empty.
        if weights.sum() == 0:
            pool = np.flatnonzero(valid)
            selected = np.random.choice(
                pool, num_points, replace=pool.size < num_points
            )
        else:
            weights /= weights.sum()
            counts = np.floor(weights * num_points).astype(int)
            remainder = num_points - counts.sum()
            order = np.argsort(-(weights * num_points - counts), kind="stable")
            counts[order[:remainder]] += 1
            samples = []
            for name, count in zip(names, counts):
                if count:
                    pool = np.flatnonzero(pool_mask(name))
                    samples.append(
                        np.random.choice(pool, count, replace=pool.size < count)
                    )
            selected = np.concatenate(samples)
        return np.column_stack(np.unravel_index(selected, sdf.shape))

    def load_coords_and_values(self, modalities, normalize=True, prepared=None, original_count=None):
        """
        Return aligned coordinates and targets with a fixed per-subject count.

        SDF training uses a crop-aware mixture of vessel, surface, tissue, and
        background voxels. Other modes retain foreground/background sampling.
        """
        modalities_data = self.augment_modalities(modalities) if prepared is None else prepared
        mra_data = modalities_data[self.modality_keys[0]]
        seg_data = modalities_data[self.modality_keys[-1]]
        affine = modalities[self.modality_keys[-1]].affine
        img_shape = mra_data.shape

        # Keep a fixed sample count per subject for DDP batching.
        sampling = self.args["dataset"].get("sdf_sampling", {})
        FINAL_TARGET_POINTS = (
            sampling.get("num_points", 500000)
            if self.sdf_key and self.split == "train" else 500000
        )
        if original_count is not None:
            FINAL_TARGET_POINTS = original_count
        if (isinstance(FINAL_TARGET_POINTS, bool)
                or not isinstance(FINAL_TARGET_POINTS, (int, np.integer))
                or FINAL_TARGET_POINTS <= 0):
            raise ValueError("sdf_sampling.num_points must be a positive integer")
        use_sdf_sampling = (
            bool(self.sdf_key) and self.split == "train"
            and sampling.get("enabled", True) and original_count is None
        )
        if use_sdf_sampling:
            sampling = self.args["dataset"].get("sdf_sampling", {})
            FINAL_TARGET_POINTS = sampling.get("num_points", FINAL_TARGET_POINTS)
            c_nz = self.sample_sdf_indices(
                modalities_data[self.sdf_key],
                seg_data,
                affine,
                normalize=normalize,
                num_points=FINAL_TARGET_POINTS,
            )
        else:
            INITIAL_BUFFER = int(FINAL_TARGET_POINTS * 1.5)

            fg_indices = np.argwhere(seg_data > 0)
            n_fg_raw = len(fg_indices)

            target_bg_raw = n_fg_raw if n_fg_raw < INITIAL_BUFFER else INITIAL_BUFFER

            n_candidates = target_bg_raw * 3 + 10000
            rand_coords = np.random.randint(
                0, [img_shape[0], img_shape[1], img_shape[2]], size=(n_candidates, 3)
            )

            vals_seg = seg_data[rand_coords[:, 0], rand_coords[:, 1], rand_coords[:, 2]]
            bg_indices = rand_coords[vals_seg == 0]
            if len(bg_indices) > target_bg_raw:
                bg_indices = bg_indices[:target_bg_raw]

            if len(bg_indices) > 0:
                if n_fg_raw > 0:
                    c_nz = np.vstack((fg_indices, bg_indices))
                else:
                    c_nz = bg_indices
            else:
                c_nz = fg_indices

            if len(c_nz) == 0:
                c_nz = np.array(
                    [[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2]]
                )

        values = np.stack(
            [
                modalities_data[mod][c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]].flatten()
                for mod in self.target_keys
            ],
            axis=-1,
        )

        c_nz_phys = nib.affines.apply_affine(affine, c_nz)

        # Use the same geometric center as native-grid inference and graph normalization.
        img_center_index = np.array(img_shape) / 2.0
        geometric_center = nib.affines.apply_affine(affine, img_center_index)

        coords = c_nz_phys - geometric_center

        if normalize:
            wb_center = self.world_bbox / 2
            coords = coords / wb_center

            mask_inside = np.all(np.abs(coords) <= 1.0, axis=1)

            coords = coords[mask_inside]
            values = values[mask_inside]

        # Preserve the fixed batch length after cropping in non-SDF sampling modes.
        current_len = coords.shape[0]

        if current_len > FINAL_TARGET_POINTS:
            idx = np.random.choice(current_len, FINAL_TARGET_POINTS, replace=False)
            coords = coords[idx]
            values = values[idx]

        elif current_len < FINAL_TARGET_POINTS:
            if current_len == 0:
                coords = np.zeros((FINAL_TARGET_POINTS, 3), dtype=np.float32)
                values = np.zeros(
                    (FINAL_TARGET_POINTS, values.shape[1]), dtype=np.float32
                )
            else:
                repeat_factor = (FINAL_TARGET_POINTS // current_len) + 1

                coords = np.tile(coords, (repeat_factor, 1))
                values = np.tile(values, (repeat_factor, 1))

                coords = coords[:FINAL_TARGET_POINTS]
                values = values[:FINAL_TARGET_POINTS]

        perm = np.random.permutation(FINAL_TARGET_POINTS)
        coords = coords[perm]
        values = values[perm]

        return coords, values

    def task_specific_item(self, modalities, idx):
        from data_loading.task_sampling import sample_vessel, sampling_stats
        cfg = self.args["task_specific_sampling"]
        if not self.sdf_key or self.args["inr_decoder"].get("sdf_dim") != 1:
            raise ValueError("Task-specific sampling requires a vessel SDF target/head")
        if cfg["image"].get("strategy", "original") != "original" or cfg["sdf"].get("strategy", "vessel_aware") != "vessel_aware":
            raise ValueError("Unsupported task-specific sampling strategy")
        # One augmentation shared by both branches; image RNG and legacy sampler unchanged.
        arrays = self.augment_modalities(modalities)
        coords, values = self.load_coords_and_values(
            modalities, prepared=arrays, original_count=cfg["image"]["num_points"])
        affine = modalities[self.modality_keys[-1]].affine
        state = np.random.get_state()
        rng = np.random.default_rng(np.random.SeedSequence([int(state[1][0]), int(state[2]), int(idx)]))
        indices, stats = sample_vessel(arrays[self.sdf_key], affine, self.world_bbox, cfg["sdf"], rng)
        center = nib.affines.apply_affine(affine, np.array(arrays[self.sdf_key].shape) / 2.)
        sdf_coords = (nib.affines.apply_affine(affine, indices) - center) / (self.world_bbox / 2.)
        sdf_values = arrays[self.sdf_key][tuple(indices.T)][:, None]
        condition = self.load_conditions(self.df.iloc[idx].to_dict())
        def pack(c, v):
            n = len(c)
            return (torch.as_tensor(c, dtype=torch.float32), torch.as_tensor(v, dtype=torch.float32),
                    condition[None].expand(n, -1), torch.full((n, 1), idx, dtype=torch.int32))
        image_stats = sampling_stats(coords)
        labels, counts = np.unique(values[:, len(self.modality_keys)-1], return_counts=True)
        image_stats["labels"] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
        return {"image": pack(coords, values), "sdf": pack(sdf_coords, sdf_values),
                "sampling": {"subject": int(idx), "image": image_stats, "sdf": stats}}

    def augment_modalities(self, modalities):
        if self.data_augmentation:
            tio_sub = {}
            for mod_key in modalities:
                data = modalities[mod_key].get_fdata()
                affine = modalities[mod_key].affine
                tensor_data = torch.from_numpy(data).type(torch.float32).unsqueeze(0)
                if "Seg" in mod_key:
                    tio_sub[mod_key] = tio.LabelMap(tensor=tensor_data, affine=affine)
                else:
                    tio_sub[mod_key] = tio.ScalarImage(
                        tensor=tensor_data, affine=affine
                    )

            tio_sub = tio.Subject(tio_sub)
            tio_sub = self.tio_transform(tio_sub)
            modalities_data = {
                mod_key: tio_sub[mod_key].data.squeeze().numpy()
                for mod_key in modalities
            }
        else:
            modalities_data = {
                mod_key: modalities[mod_key].get_fdata() for mod_key in modalities
            }
        return modalities_data

    def load_conditions(self, row_dict, normalize=True):
        conditions = []
        for key in self.args["dataset"]["conditions"]:
            if self.args["dataset"]["conditions"][key]:
                constraint = self.args["dataset"]["constraints"][key]
                val = row_dict[key]

                if constraint.get("type") == "categoric":
                    possible_values = constraint["values"]
                    if val in possible_values:
                        val = float(possible_values.index(val))
                    else:
                        val = 0.0

                    c_min = 0.0
                    c_max = float(len(possible_values) - 1)

                else:
                    val = float(val)
                    c_min = constraint["min"]
                    c_max = constraint["max"]

                if normalize:
                    if c_max > c_min:
                        norm_val = (
                            ((val - c_min) / (c_max - c_min)) * 2 - 1
                        ) * self.args["atlas_gen"]["cond_scale"]
                    else:
                        norm_val = 0.0

                    row_dict[key] = norm_val
                else:
                    row_dict[key] = val

                conditions.append(row_dict[key])

        return torch.tensor(conditions, dtype=torch.float32)

    def filter_dataframe(self, df):
        """
        Filter the dataframe based on the constraints in the args.
        Also check for missing modalities and remove subjects with missing modalities.
        Returns a dataframe with the filtered data, i.e. the final subjects
        that will be used for training, validation, or testing.
        """
        print("--------------------------------")
        print("Sampling data for split ", self.split, "\n")
        df = self.sample_subject_ids(df)
        df = self.remove_missing_modalities(df)
        df = self.check_constraints(df)
        df = self.sample_subjects(df)
        print("Sampled ", len(df), " subjects for split ", self.split, "\n")
        print("--------------------------------")
        return df

    def sample_subject_ids(self, df, verbose=True):
        """
        Sample subjects from subject_ids list provided by the args_data['subject_ids']
        """
        if self.args["dataset"]["subject_ids"][self.split]:
            print(
                f"Sampling subjects from subject_ids list. Number of subjects in dataframe: {len(df)}, "
                f"Number of subjects in subject_ids list: {len(self.args['dataset']['subject_ids'][self.split])}"
            )
            df = df[
                df["subject_id"].isin(self.args["dataset"]["subject_ids"][self.split])
            ]
            print(f"Number of subjects in dataframe after sampling: {len(df)} \n")
        return df

    def remove_missing_modalities(self, df, verbose=True):
        """
        Removes rows if any required modality entry is missing or empty in the DataFrame.
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        modalities = self.target_keys
        if self.sdf_key and self.sdf_key not in df.columns:
            raise ValueError(f"Missing SDF column {self.sdf_key!r} in metadata TSV")
        keep_mask = np.ones(len(df), dtype=bool)

        for modality in modalities:
            if modality not in df.columns:
                print(
                    f"Warning: modality column '{modality}' not in DataFrame. Skipping."
                )
                continue

            col_values = df[modality]
            is_valid = col_values.notnull() & (col_values != "")
            keep_mask &= is_valid.to_numpy(dtype=bool)

        dropped_count = (~keep_mask).sum()
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} subjects missing modality entries.")

        df = df[keep_mask].reset_index(drop=True)
        if verbose:
            print("Number of subjects after removing missing modalities:", len(df))

        return df

    def check_constraints(self, df, verbose=True):
        """
        Drops rows if any numeric constraint is outside [min, max].
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        constraints_dict = self.args["dataset"].get("constraints", {})

        keep_mask = np.ones(len(df), dtype=bool)

        for ckey, cinfo in constraints_dict.items():
            if cinfo.get("type") == "numeric":
                cmin = cinfo.get("min", None)
                cmax = cinfo.get("max", None)
                if cmin is not None and cmax is not None and ckey in df.columns:
                    vals = df[ckey].values
                    this_mask = (vals >= cmin) & (vals <= cmax)
                    keep_mask &= this_mask
            elif cinfo.get("type") == "categoric":
                if ckey in df.columns:
                    this_mask = df[ckey].isin(cinfo.get("values"))
                    keep_mask &= this_mask
                else:
                    raise ValueError(f"Constraint column {ckey} not found in dataframe")
            else:
                raise ValueError(f"Constraint type {cinfo.get('type')} not supported")

        dropped_count = np.count_nonzero(~keep_mask)
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} subjects outside constraint ranges.")
        df = df[keep_mask].reset_index(drop=True)

        if verbose:
            print("Number of subjects after constraints check:", len(df))

        return df

    def get_condition_values(self, condition_key, normed=True, device=None):
        """
        Get all values together with their entry_idx of a condition_key from the dataframe.
        """
        values = torch.tensor(
            self.df[condition_key].values, dtype=torch.float32, device=device
        )
        if normed:
            values = normalize_condition(self.args, condition_key, values)
        return values, torch.arange(len(self.df), device=device)

    def sample_subjects(self, df, verbose=True):
        """Sample subjects using prioritized metadata distributions."""
        if self.args["n_subjects"][self.split] > len(df):
            print(
                "max_num_subjects must be greater than the number of subjects in the dataframe"
            )
            print(
                f"max_num_subjects: {self.args['n_subjects'][self.split]}, number of subjects in the dataframe: {len(df)}"
            )
            self.args["n_subjects"][self.split] = len(df)

        max_num_subjects = self.args["n_subjects"][self.split]
        constraints_with_prio = []
        for cname, cinfo in self.args["dataset"].get("constraints", {}).items():
            dist_info = cinfo.get("distribution", {})
            if "priority" in dist_info:
                constraints_with_prio.append((cname, cinfo))

        constraints_with_prio.sort(
            key=lambda x: x[1]["distribution"]["priority"], reverse=True
        )

        if verbose:
            print(f"[sample_subjects] # subjects before sampling: {len(df)}")
            print(
                "[sample_subjects] Constraints in priority order:",
                [c[0] for c in constraints_with_prio],
            )

        constraints_list = [(cname, cinfo) for (cname, cinfo) in constraints_with_prio]
        df_sampled = self._shallow_sampling(
            df, constraints_list, max_num_subjects, verbose
        )

        if len(df_sampled) > max_num_subjects:
            df_sampled = df_sampled.sample(
                n=max_num_subjects, random_state=42
            ).reset_index(drop=True)
            if verbose:
                print(
                    f"[sample_subjects] Truncated final set to max_num_subjects={max_num_subjects}"
                )

        if verbose:
            print(
                f"[sample_subjects] # subjects after nested sampling: {len(df_sampled)}"
            )

        if verbose:
            for constraint in constraints_list:
                self._print_and_save_hist(df_sampled, constraint)

        return df_sampled

    def _shallow_sampling(self, df, constraints_list, max_num_subjects, verbose=True):
        """
        Sample uniformly over the highest priority constraint. No nested sampling supported just yet.
        """
        current_constraint = constraints_list[0]
        current_constraint_name = current_constraint[0]
        current_constraint_info = current_constraint[1]
        c_min = current_constraint_info.get("min", None)
        c_max = current_constraint_info.get("max", None)
        bins = current_constraint_info.get("distribution").get("bins")
        if bins is None:
            bins = c_max - c_min
        edges = np.linspace(c_min, c_max, bins + 1)
        edges[-1] += 1e-6
        values = df[current_constraint_name].values
        bin_idx = np.digitize(values, edges) - 1
        bin_sizes = np.bincount(bin_idx)
        bin_labels = [f"[{edges[i]},{edges[i+1]}]" for i in range(len(edges) - 1)]
        if verbose:
            print(f"Bin labels: {bin_labels}")
            print(f"Bin sizes: {bin_sizes}")

        samples_drawn_per_bin = [0] * bins
        drained_bins = []
        while np.sum(samples_drawn_per_bin) < max_num_subjects:
            num_remaining_bins = bins - len(np.unique(drained_bins))
            num_remaining_subjects = max_num_subjects - np.sum(samples_drawn_per_bin)
            required_samples_per_bin = math.ceil(
                num_remaining_subjects / num_remaining_bins
            )
            print(
                f"Required (additional) samples per remaining bin: {required_samples_per_bin}"
            )
            for i in range(bins):
                bin_df = df[bin_idx == i]
                if i in drained_bins:
                    continue
                n_samples = min(
                    required_samples_per_bin, len(bin_df) - samples_drawn_per_bin[i]
                )
                samples_drawn_per_bin[i] += n_samples
                if samples_drawn_per_bin[i] == len(bin_df):
                    drained_bins.append(i)
        print(f"Samples drawn per bin: {samples_drawn_per_bin}")
        df_sampled = []
        for i, n_samples in enumerate(samples_drawn_per_bin):
            if n_samples > 0:
                assert len(df[bin_idx == i]) >= n_samples
                df_sampled.append(df[bin_idx == i].sample(n=n_samples, random_state=42))
        df_sampled = pd.concat(df_sampled, ignore_index=True)
        return df_sampled

    def _print_and_save_hist(self, df, constraint):
        """
        Print the histogram data (counts per bin) for `column_name` in `df`,
        and save a bar plot as an image. Uses 30 bins by default, or fewer if df is small.
        """
        return
        if self.args.get("rank", 0) != 0:
            return

        indent = "  "
        current_cname, current_cinfo = constraint
        col_data = df[current_cname].dropna().values
        if len(col_data) == 0:
            print(f"{indent}[Histogram] No data for column '{current_cname}'")
            return

        dist_info = current_cinfo.get("distribution", {})
        dist_type = dist_info.get("type")
        c_min = current_cinfo.get("min", None)
        c_max = current_cinfo.get("max", None)
        bins = dist_info.get("bins")
        if bins is None:
            bins = (
                (c_max - c_min + 1) if (c_min is not None and c_max is not None) else 10
            )
        edges = np.linspace(c_min, c_max, bins)
        edges[-1] += 1e-9

        counts, bin_edges = np.histogram(col_data, bins=edges)
        print(f"{indent}[Histogram] {current_cname} counts: {counts.tolist()}")
        print(f"{indent}[Histogram] {current_cname} bin_edges: {bin_edges.tolist()}")

        plt.figure(figsize=(15, 6))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9)
        plt.xticks(bin_centers, rotation=45)
        if max(counts) < 20:
            plt.yticks(range(0, max(counts) + 1))

        plt.title(f"Histogram of {current_cname} with {dist_type} distribution")
        plt.xlabel(current_cname)
        plt.ylabel("Count")

        out_path = os.path.join(
            self.args["output_dir"],
            f"hist_{current_cname}_dist_{dist_type}_{self.split}.png",
        )
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"{indent}Saved histogram to {out_path}")

    def _init_data_augmentation(self):
        """
        Initialize data augmentation.
        """
        self.data_augmentation = self.args["data_augmentation"]["activate"] * (
            self.split == "train"
        )
        if self.data_augmentation:
            self.transforms = []
            args_aug = self.args["data_augmentation"]
            if self.sdf_key and args_aug["augment_deformation"]["p"] > 0.0:
                raise ValueError(
                    "Elastic augmentation requires recomputing SDF distances; disable augment_deformation for SDF training"
                )
            if args_aug["augment_deformation"]["p"] > 0.0:
                args = args_aug["augment_deformation"]
                self.transforms.append(
                    tio.RandomElasticDeformation(
                        p=args["p"],
                        num_control_points=args["num_control_points"],
                        max_displacement=args["max_displacement"],
                    )
                )
            if args_aug["augment_motion"]["p"] > 0.0:
                args = args_aug["augment_motion"]
                self.transforms.append(
                    tio.RandomMotion(
                        p=args["p"],
                        degrees=args["degrees"],
                        translation=args["translation"],
                        include=self.modality_keys[:-1],
                        num_transforms=args["num_transforms"],
                    )
                )
            if args_aug["augment_noise"]["p"] > 0.0:
                args = args_aug["augment_noise"]
                self.transforms.append(
                    tio.RandomNoise(
                        p=args["p"],
                        mean=args["mean"],
                        std=args["std"],
                        include=self.modality_keys[:-1],
                    )
                )
            if args_aug["augment_biasfield"]["p"] > 0.0:
                args = args_aug["augment_biasfield"]
                self.transforms.append(
                    tio.RandomBiasField(
                        p=args["p"],
                        coefficients=args["coeff"],
                        include=self.modality_keys[:-1],
                    )
                )
            if args_aug["augment_gamma"]["p"] > 0.0:
                args = args_aug["augment_gamma"]
                self.transforms.append(
                    tio.RandomGamma(
                        p=args["p"],
                        log_gamma=args["log_gamma"],
                        include=self.modality_keys[:-1],
                    )
                )
            self.transforms.append(
                MaskSubjectTransform(
                    segmentation_key=self.modality_keys[-1],
                    intensity_keys=self.modality_keys[:-1],
                )
            )
            self.tio_transform = (
                tio.Compose(self.transforms) if self.transforms else None
            )
