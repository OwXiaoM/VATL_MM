import math
import json
from types import SimpleNamespace
import ants
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
import wandb as wd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import scipy.ndimage as ndi
import torchio as tio


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def dict_to_simplenamespace(d):
    """Recursively convert dictionaries to SimpleNamespace objects."""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    else:
        return d


class MaskSubjectTransform(tio.Transform):
    """Mask selected intensity images using the subject segmentation."""

    def __init__(self, segmentation_key, intensity_keys=None, **kwargs):
        super().__init__(**kwargs)
        self.segmentation_key = segmentation_key
        self.intensity_keys = intensity_keys

    def apply_transform(self, subject):
        seg_image = subject[self.segmentation_key]
        seg_data = seg_image.data

        if self.intensity_keys is None:
            keys_to_mask = [
                k for k in subject.get_images_names() if k != self.segmentation_key
            ]
        else:
            keys_to_mask = self.intensity_keys

        for key in keys_to_mask:
            image = subject[key]
            image.data = image.data * (seg_data > 0)

        return subject


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tf_weight = args["optimizer"]["tf_weight"]
        self.moe_weight = args["optimizer"].get("moe_weight", 0.1)

        self.n_classes = args["inr_decoder"]["out_dim"][-1] - 1
        self.sr_dims = sum(args["inr_decoder"]["out_dim"][:-1])
        self.seg_end = self.sr_dims + args["inr_decoder"]["out_dim"][-1]
        self.sdf_enabled = bool(args["dataset"].get("sdf_key"))
        self.sdf_weight = (
            float(args["optimizer"].get("sdf_weight", 1.0)) if self.sdf_enabled else 0.0
        )
        self.sdf_trunc_mm = float(args["optimizer"].get("sdf_trunc_mm", 5.0))
        if not math.isfinite(self.sdf_weight) or self.sdf_weight < 0:
            raise ValueError("optimizer.sdf_weight must be finite and nonnegative")
        if not math.isfinite(self.sdf_trunc_mm) or self.sdf_trunc_mm <= 0:
            raise ValueError("optimizer.sdf_trunc_mm must be finite and positive")
        if self.sdf_enabled and args["inr_decoder"].get("sdf_dim", 0) != 1:
            raise ValueError("SDF supervision requires inr_decoder.sdf_dim: 1")
        self.criterion_sr = (
            nn.MSELoss() if args["optimizer"]["loss_metric"] == "mse" else nn.L1Loss()
        )
        self.ce_weights = (
            torch.tensor(
                args["dataset"]["class_weights"],
                dtype=torch.float32,
                device=args["device"],
            )
            if args["dataset"]["class_weights"] is not None
            else None
        )
        self.criterion_seg = nn.CrossEntropyLoss(weight=self.ce_weights)

    def voxel_sdf_loss(self, output, sdf_mm):
        if not torch.isfinite(sdf_mm).all():
            raise ValueError("SDF targets contain non-finite distances")
        target = sdf_mm.clamp(-self.sdf_trunc_mm, self.sdf_trunc_mm) / self.sdf_trunc_mm
        raw = F.l1_loss(output[..., self.seg_end], target)
        return raw, self.sdf_weight * raw

    def forward(
        self,
        output,
        target,
        tfs,
        moe_loss=None,
        sr_weight=1.0,
        seg_weight=1.0,
        sdf_weight=None,
        apply_sdf=True,
    ):
        # Targets contain image intensities, one tissue label, then optional SDF in mm.
        expected_targets = self.sr_dims + 1 + int(self.sdf_enabled)
        if target.shape[-1] != expected_targets:
            raise ValueError(
                f"Expected {expected_targets} target channels, got {target.shape[-1]}"
            )
        if output.shape[-1] < self.seg_end + int(self.sdf_enabled):
            raise ValueError("Decoder output is missing required tissue/SDF channels")
        zero = output.new_zeros(())
        loss = {
            "seg": zero,
            "sdf": zero,
            "sdf_raw": zero,
            "tf": zero,
            "sr": sr_weight
            * self.criterion_sr(
                output[..., : self.sr_dims], target[..., : self.sr_dims]
            ),
            "trafo": zero,
            "moe": zero,
            "total": 0.0,
        }

        if seg_weight > 0:
            loss["seg"] = self.criterion_seg(
                output[..., self.sr_dims : self.seg_end],
                target[..., self.sr_dims].to(torch.int64),
            )

        sdf_weight = self.sdf_weight if sdf_weight is None else sdf_weight
        if self.sdf_enabled and apply_sdf:
            sdf_mm = target[..., self.sr_dims + 1]
            if not torch.isfinite(sdf_mm).all():
                raise ValueError("SDF targets contain non-finite distances")
            sdf_target = (
                sdf_mm.clamp(-self.sdf_trunc_mm, self.sdf_trunc_mm) / self.sdf_trunc_mm
            )
            loss["sdf_raw"] = F.l1_loss(output[..., self.seg_end], sdf_target)
            loss["sdf"] = sdf_weight * loss["sdf_raw"]

        if tfs is not None:
            loss["tf_rot"] = torch.mean(tfs[..., :3] ** 2)
            loss["tf_trans"] = torch.mean(tfs[..., 3:6] ** 2)
            loss["tf_scale"] = (
                torch.mean(tfs[..., 6:9] ** 2) if tfs.shape[-1] == 9 else zero
            )
            loss["tf"] = loss["tf_rot"] + loss["tf_trans"] + loss["tf_scale"]

        if moe_loss is not None and self.moe_weight > 0:
            loss["moe"] = self.moe_weight * moe_loss

        loss["sr_raw"] = loss["sr"] / sr_weight if sr_weight else zero
        loss["seg_raw"] = loss["seg"]
        loss["seg_weighted"] = seg_weight * loss["seg"]
        loss["tf_weighted"] = self.tf_weight * loss["tf"]
        loss["moe_raw"] = moe_loss if moe_loss is not None else zero
        loss["total"] = (
            loss["sr"]
            + seg_weight * loss["seg"]
            + loss["sdf"]
            + self.tf_weight * loss["tf"]
            + loss["moe"]
        )
        return loss


def graph_radius_enabled(args):
    return bool(
        args.get("graph_supervision", {}).get("radius", {}).get("activate", False)
    )


def graph_supervision_mode(args):
    mode = args.get("graph_supervision", {}).get("mode", "ce")
    if mode not in ("ce", "sdf", "both"):
        raise ValueError("graph_supervision.mode must be ce, sdf, or both")
    if (mode in ("sdf", "both") or graph_radius_enabled(args)) and (
        not args["dataset"].get("sdf_key") or args["inr_decoder"].get("sdf_dim", 0) != 1
    ):
        raise ValueError(
            "Graph SDF supervision requires dataset.sdf_key and inr_decoder.sdf_dim: 1"
        )
    return mode


def graph_supervision_loss(args, output, sdf_target_mm=None, radius_target_mm=None):
    """Graph CE/SDF losses plus an optional consistency-gated radius prior."""
    mode = graph_supervision_mode(args)
    config = args.get("graph_supervision", {})
    sr_dims = sum(args["inr_decoder"]["out_dim"][:-1])
    seg_end = sr_dims + args["inr_decoder"]["out_dim"][-1]
    zero = output.new_zeros(())
    loss = dict(
        graph_ce_raw=zero,
        graph_sdf_raw=zero,
        graph_ce=zero,
        graph_sdf=zero,
        graph_radius_raw=zero,
        graph_radius=zero,
        graph_radius_valid_fraction=zero,
    )
    if mode in ("ce", "both"):
        label = config.get("vessel_label", "Vessel")
        if label not in args["dataset"]["label_names"]:
            raise ValueError(
                f"Graph vessel label {label!r} not found in dataset.label_names"
            )
        vessel_idx = args["dataset"]["label_names"].index(label)
        weight = float(config.get("weight", 0.1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("graph_supervision.weight must be finite and nonnegative")
        target = torch.full(
            (output.shape[0],), vessel_idx, dtype=torch.long, device=output.device
        )
        loss["graph_ce_raw"] = F.cross_entropy(output[..., sr_dims:seg_end], target)
        loss["graph_ce"] = weight * loss["graph_ce_raw"]
    if mode in ("sdf", "both"):
        if sdf_target_mm is None or sdf_target_mm.shape != output.shape[:-1]:
            raise ValueError("Graph SDF targets must match the sampled node count")
        if output.shape[-1] != seg_end + 1 or not torch.isfinite(sdf_target_mm).all():
            raise ValueError(
                "Graph SDF supervision requires one SDF output and finite distance targets"
            )
        tau = float(args["optimizer"].get("sdf_trunc_mm", 5.0))
        weight = float(config.get("sdf_weight", 0.1))
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("optimizer.sdf_trunc_mm must be finite and positive")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                "graph_supervision.sdf_weight must be finite and nonnegative"
            )
        target = sdf_target_mm.clamp(-tau, tau) / tau
        loss["graph_sdf_raw"] = F.l1_loss(output[..., seg_end], target)
        loss["graph_sdf"] = weight * loss["graph_sdf_raw"]
    if graph_radius_enabled(args):
        radius_config = config["radius"]
        weight = float(radius_config.get("weight", 0.05))
        tolerance = float(radius_config.get("max_sdf_disagreement_mm", 0.5))
        scale = float(radius_config.get("scale_to_mm", 1.0))
        tau = float(args["optimizer"].get("sdf_trunc_mm", 5.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                "graph_supervision.radius.weight must be finite and nonnegative"
            )
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "radius.max_sdf_disagreement_mm must be finite and nonnegative"
            )
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError(
                "Radius unit scale and SDF truncation must be finite and positive"
            )
        if (
            radius_target_mm is None
            or sdf_target_mm is None
            or radius_target_mm.shape != output.shape[:-1]
            or sdf_target_mm.shape != output.shape[:-1]
        ):
            raise ValueError(
                "Radius supervision requires matched SDF and radius targets for every graph node"
            )
        if output.shape[-1] != seg_end + 1 or not torch.isfinite(sdf_target_mm).all():
            raise ValueError(
                "Radius supervision requires one SDF output and finite SDF targets"
            )
        # Check radius agreement in physical mm before truncation can hide disagreements.
        valid = (
            torch.isfinite(radius_target_mm)
            & (radius_target_mm > 0)
            & (sdf_target_mm < 0)
            & ((sdf_target_mm + radius_target_mm).abs() <= tolerance)
        )
        loss["graph_radius_valid_fraction"] = valid.float().mean()
        if valid.any():
            target = -radius_target_mm[valid].clamp(max=tau) / tau
            loss["graph_radius_raw"] = F.smooth_l1_loss(
                output[..., seg_end][valid], target
            )
        else:
            loss["graph_radius_raw"] = output[..., seg_end].sum() * 0.0
        loss["graph_radius"] = weight * loss["graph_radius_raw"]
    loss["graph_raw"] = (
        loss["graph_ce_raw"] + loss["graph_sdf_raw"] + loss["graph_radius_raw"]
    )
    loss["graph"] = loss["graph_ce"] + loss["graph_sdf"] + loss["graph_radius"]
    return loss


def scale_affine(affine, new_spacing, bbox):
    old_spacing = torch.sqrt(torch.sum(affine[:3, :3] ** 2, axis=0))
    scale = torch.tensor(new_spacing) / old_spacing
    bbox_size = bbox["max"] - bbox["min"]
    shape = torch.ceil_(bbox_size / scale).int()
    affine[:3, :3] = torch.matmul(affine[:3, :3], torch.diag(scale))
    affine[:3, 3] = affine[:3, 3] + (bbox["min"] * old_spacing)
    return affine, shape


def compute_ncc(prediction, reference):
    mean_pred = np.mean(prediction)
    mean_ref = np.mean(reference)
    numerator = np.sum((prediction - mean_pred) * (reference - mean_ref))
    denominator = np.sqrt(
        np.sum((prediction - mean_pred) ** 2) * np.sum((reference - mean_ref) ** 2)
    )
    ncc = numerator / denominator
    return ncc.astype(np.float64)


def embed2affine(embed):
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:6]
    if embed.shape[-1] >= 9:
        S = torch.diag_embed(1.0 + embed[..., 6:9])
        R = torch.matmul(R, S)
    if embed.shape[-1] == 12:
        S_x = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_y = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_z = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_x[..., 0, 1] = embed[..., 10]
        S_x[..., 0, 2] = embed[..., 11]
        S_y[..., 1, 0] = embed[..., 9]
        S_y[..., 1, 2] = embed[..., 11]
        S_z[..., 2, 0] = embed[..., 9]
        S_z[..., 2, 1] = embed[..., 10]
        R = torch.matmul(R, S_x)
        R = torch.matmul(R, S_y)
        R = torch.matmul(R, S_z)
    return R, t


def euler2rot(theta):
    c1 = torch.cos(theta[..., 0])
    s1 = torch.sin(theta[..., 0])
    c2 = torch.cos(theta[..., 1])
    s2 = torch.sin(theta[..., 1])
    c3 = torch.cos(theta[..., 2])
    s3 = torch.sin(theta[..., 2])
    r11 = c1 * c3 - c2 * s1 * s3
    r12 = -c1 * s3 - c2 * c3 * s1
    r13 = s1 * s2
    r21 = c3 * s1 + c1 * c2 * s3
    r22 = c1 * c2 * c3 - s1 * s3
    r23 = -c1 * s2
    r31 = s2 * s3
    r32 = c3 * s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    R = R.view(R.shape[:-1] + (3, 3))
    return R


def harmonize_labels(subject_seg, dataset):
    if "dhcp_neo" in dataset:
        subject_seg[subject_seg == 4] = 0
        subject_seg[subject_seg == 8] = 6
        subject_seg[subject_seg == 9] = 7
    elif "dhcp_fetal" in dataset or "marsfet" in dataset:
        subject_seg[subject_seg == 4] = 0
        subject_seg[subject_seg == 5] = 4
        subject_seg[subject_seg == 3] = 3
        subject_seg[subject_seg == 6] = 3
        subject_seg[subject_seg == 7] = 3
        subject_seg[subject_seg == 8] = 3
        subject_seg[subject_seg == 9] = 3
    else:
        raise NotImplementedError("Atlas Type not implemented.")
    return subject_seg


def to_device(x, device="cuda"):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def normalize_condition(args, condition_key, condition_values, cond_scale=None):
    c_scale = args["atlas_gen"]["cond_scale"] if cond_scale is None else cond_scale
    constraint = args["dataset"]["constraints"][condition_key]

    if constraint.get("type") == "categoric":
        c_min = 0.0
        c_max = float(len(constraint["values"]) - 1)
    else:
        c_min = constraint["min"]
        c_max = constraint["max"]

    if c_max > c_min:
        cv = 2 * ((condition_values - c_min) / (c_max - c_min) - 0.5)
    else:
        cv = 0.0

    cv = cv * c_scale
    return cv


def generate_combinations(
    args_data, conditions, keys=None, idx=0, current=None, results=None
):
    if conditions is None:
        return [[]]
    if keys is None:
        keys = list(conditions.keys())
    if current is None:
        current = []
    if results is None:
        results = []

    key = keys[idx]
    values = conditions[key]["values"]

    for value in values:
        if not conditions[key]["normed_values"]:
            value = normalize_condition(args_data, key, value)
        else:
            value = value * args_data["atlas_gen"]["cond_scale"]
        next_current = current + [value]
        if idx == len(keys) - 1:
            results.append(next_current)
        else:
            generate_combinations(
                args_data, conditions, keys, idx + 1, next_current, results
            )

    return results


def generate_world_grid(args, normed=True, device="cpu"):
    world_bbox = args["dataset"]["world_bbox"]
    spacing = args["atlas_gen"]["spacing"]

    x = torch.arange(0, world_bbox[0], spacing[0], device=device)
    y = torch.arange(0, world_bbox[1], spacing[1], device=device)
    z = torch.arange(0, world_bbox[2], spacing[2], device=device)

    if normed:
        x = x / world_bbox[0] * 2 - 1
        y = y / world_bbox[1] * 2 - 1
        z = z / world_bbox[2] * 2 - 1

    grid = torch.meshgrid(x, y, z, indexing="ij")
    grid_shape = list(grid[0].shape)
    grid_coords = torch.stack(grid, dim=-1).reshape(-1, 3)

    affine = torch.diag(
        torch.tensor([spacing[0], spacing[1], spacing[2], 1.0], device=device)
    )
    origin_offset = -torch.tensor(world_bbox, device=device) / 2.0
    affine[:3, 3] = origin_offset

    return grid_coords, grid_shape, affine


def sdf_export_enabled(args):
    """Only export SDF for a model configured with SDF supervision."""
    return (
        bool(args["dataset"].get("sdf_key"))
        and args["inr_decoder"].get("sdf_dim", 0) == 1
    )


def reconstruction_channels(args):
    """Explicit (file label, channel index, dtype) mapping for inference volumes."""
    modalities = args["dataset"]["modalities"]
    sr_dims = sum(args["inr_decoder"]["out_dim"][:-1])
    n_classes = args["inr_decoder"]["out_dim"][-1]
    if len(modalities) != sr_dims + 1:
        raise ValueError(
            "Reconstruction modalities must contain images followed by Seg"
        )
    channels = [
        (name, i, np.int16 if i == sr_dims else np.float32)
        for i, name in enumerate(modalities)
    ]
    if args.get("save_certainty_maps", False):
        labels = args["dataset"]["label_names"]
        if len(labels) != n_classes:
            raise ValueError("Tissue label count does not match decoder channels")
        channels += [
            (f"CertaintyMaps/{label}", sr_dims + 1 + i, np.float32)
            for i, label in enumerate(labels)
            if label.lower() != "background"
        ]
    if sdf_export_enabled(args):
        sdf_index = sr_dims + 1 + n_classes
        channels += [
            ("SDF", sdf_index, np.float32),
            ("VesselMask", sdf_index + 1, np.uint8),
        ]
    return channels


def save_atlas(args, atlas, affine, temp_steps, condition_vectors, epoch):
    if isinstance(atlas, torch.Tensor):
        atlas = atlas.detach().cpu().numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    channels = reconstruction_channels(args)
    if atlas.ndim != 6 or atlas.shape[3] <= max(i for _, i, _ in channels):
        raise ValueError("Atlas is missing required reconstruction channels")
    n_conds = atlas.shape[4]
    for c in range(n_conds):
        for name, i, dtype in channels:
            filename = (
                f"{name}={temp_steps[0]}-{temp_steps[-1]}_cond={c}_ep={epoch}.nii.gz"
            )
            save_img(
                atlas[..., i, c, :].astype(dtype),
                affine=affine,
                output_path=args["output_dir"],
                filename=filename,
                spatial_units="mm" if name in ("SDF", "VesselMask") else None,
            )
    print("Atlas saved to {}".format(args["output_dir"]))


def typecheck_img_affine(img, affine):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if img.dtype == "int64" or img.dtype == "int32":
        img = img.astype(np.int16)
    elif img.dtype == "float64":
        img = img.astype(np.float32)
    return img, affine


def save_subject(
    args, img, affine, sub_row_df=None, sub_name=None, epoch=0, split="train"
):
    if sub_row_df is not None:
        sub_name = sub_row_df["subject_id"]
    elif sub_name is None:
        print("No subject name provided, assigning random subject name.")
        sub_name = "subject_" + str(np.random.randint(100000))

    img, affine = typecheck_img_affine(img, affine)
    channels = reconstruction_channels(args)
    if img.shape[-1] <= max(i for _, i, _ in channels):
        raise ValueError("Subject volume is missing required reconstruction channels")
    modalities = args["dataset"]["modalities"]
    for i, mod in enumerate(modalities):
        if i == len(modalities) - 1:
            img_mod = img[..., i].astype(np.int16)
        else:
            img_mod = img[..., i].astype(np.float32)
        filename = f"{split}/{mod}_{sub_name}_ep={epoch}.nii.gz"
        save_img(img_mod, affine, args["output_dir"], filename)
        if epoch == 0 and sub_row_df is not None:
            # Export the untouched native reference; never estimate a transform.
            ref_mod_nii = nib.load(sub_row_df[mod])
            filename_ref = f"{split}/{mod}_{sub_name}_ref.nii.gz"
            save_img(
                ref_mod_nii.get_fdata(),
                ref_mod_nii.affine,
                args["output_dir"],
                filename_ref,
            )

    save_subject_extras(args, img, affine, sub_name, epoch, split)


def save_subject_extras(args, img, affine, sub_name, epoch=0, split="train"):
    """Save tissue probabilities and SDF outputs in the inference grid."""
    img, affine = typecheck_img_affine(img, affine)
    channels = reconstruction_channels(args)
    if img.shape[-1] <= max(i for _, i, _ in channels):
        raise ValueError("Subject volume is missing required reconstruction channels")
    for name, i, dtype in channels:
        if i < len(args["dataset"]["modalities"]):
            continue
        filename = f"{split}/{name}_{sub_name}_ep={epoch}.nii.gz"
        save_img(
            img[..., i].astype(dtype),
            affine,
            args["output_dir"],
            filename,
            spatial_units="mm" if name in ("SDF", "VesselMask") else None,
        )


def save_img(img, affine, output_path, filename, spatial_units=None):
    if img.dtype == "int64" or img.dtype == "int32":
        img = img.astype(np.int16)
    elif img.dtype == "float64":
        img = img.astype(np.float32)
    full_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    img_nii = nib.Nifti1Image(img, affine)
    if spatial_units is not None:
        img_nii.header.set_xyzt_units(spatial_units)
    nib.save(img_nii, full_path)
    print(f"Saved {filename} to {output_path}")


def assert_correct_coord_normalization(coords, min_val=-1.0, max_val=1.0):
    min_x, min_y, min_z = coords.min(axis=0)
    max_x, max_y, max_z = coords.max(axis=0)
    assert min_x >= min_val, f"min_x = {min_x} is less than min_val = {min_val}"
    assert max_x <= max_val, f"max_x = {max_x} is greater than max_val = {max_val}"
    assert min_y >= min_val, f"min_y = {min_y} is less than min_val = {min_val}"
    assert max_y <= max_val, f"max_y = {max_y} is greater than max_val = {max_val}"
    assert min_z >= min_val, f"min_z = {min_z} is less than min_val = {min_val}"
    assert max_z <= max_val, f"max_z = {max_z} is greater than max_val = {max_val}"


def squeeze_modalities(modality, path, write2disk=False):
    if modality.ndim == 4:
        mod_np = modality.get_fdata()
        mod_np = np.squeeze(mod_np)
        mod_nii = nib.Nifti1Image(mod_np, modality.affine)
        if write2disk:
            mod_nii.to_filename(path)
        return mod_nii
    else:
        return modality


def add_background_halo(
    label_names, seg_nii, halo_width=1.5, background_label_str="Background"
):
    seg = seg_nii.get_fdata()
    bg_label = label_names.index(background_label_str)
    mask_bg = (seg > 0).astype(np.float32)
    mask_bg = (ndi.gaussian_filter(mask_bg, sigma=halo_width) > 0.001).astype(
        np.uint8
    ) * bg_label
    mask_bg[seg > 0] = seg[seg > 0]
    return nib.Nifti1Image(mask_bg, seg_nii.affine)


def mask_nifti(nii, mask):
    data = nii.get_fdata()
    data *= mask
    return nib.Nifti1Image(data, nii.affine)


def nib_to_ants(nii_img):
    """Convert a NIfTI image to ANTs while preserving its physical geometry."""
    data = nii_img.get_fdata().astype(np.float32)

    affine = nii_img.affine

    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    origin = affine[:3, 3]
    spacing_nonzero = spacing.copy()
    spacing_nonzero[spacing_nonzero == 0] = 1.0
    direction = affine[:3, :3] @ np.diag(1 / spacing_nonzero)

    img_ants = ants.from_numpy(
        data=data, origin=origin.tolist(), spacing=spacing.tolist(), direction=direction
    )
    return img_ants


def compute_metrics(
    args, pred, affine, df_row_dict, epoch, split, reg_type=None, bg_label=None
):
    """Native full-reference-grid evaluation after adaptation; no registration.

    reg_type/bg_label remain accepted for call compatibility. Labels are resolved
    from dataset.label_names. Image saving is handled independently by the caller.
    """
    from native_evaluation import compute_native_metrics
    pred, affine = typecheck_img_affine(pred, affine)
    return compute_native_metrics(args, pred, affine, df_row_dict, epoch, split)


def reg_imgs(
    fix_nii, mov_nii, mask_mov_nii=None, mytx=None, reg_type=None, is_seg=False
):
    """Legacy tuple interface, now geometry-only: no estimated registration.

    Existing reg_type/mask arguments are accepted but do not alter the metric
    region. Applying learned STN maps remains the decoder's responsibility.
    """
    from native_evaluation import geometry_resample
    if mytx is not None:
        raise ValueError("Evaluation does not accept estimated registration transforms")
    values, _ = geometry_resample(mov_nii, fix_nii, labels=is_seg)
    return fix_nii, nib.Nifti1Image(values, fix_nii.affine), None


def get_bbox(img):
    if isinstance(img, nib.Nifti1Image):
        img = img.get_fdata()

    non_zero_voxels = np.argwhere(img != 0)

    if non_zero_voxels.size == 0:
        shape = img.shape
        min_coords = np.array([0, 0, 0])
        max_coords = np.array([shape[0] - 1, shape[1] - 1, shape[2] - 1])
        return np.stack([min_coords, max_coords], axis=0), img

    min_coords = non_zero_voxels.min(axis=0)
    max_coords = non_zero_voxels.max(axis=0)
    img_cropped = img[
        min_coords[0] : max_coords[0] + 1,
        min_coords[1] : max_coords[1] + 1,
        min_coords[2] : max_coords[2] + 1,
    ]
    return np.stack([min_coords, max_coords], axis=0), img_cropped


def compute_dice(pred, ref, bg_label):
    """Legacy multiclass Dice API; an empty foreground union is undefined."""
    from native_evaluation import multiclass_dice
    if pred.shape != ref.shape:
        raise ValueError("The shape of pred and ref must be the same.")
    return multiclass_dice(pred, ref, bg_label)


def log_metrics(args, metrics, epoch, df=None, split="train"):
    from native_evaluation import write_metric_results
    summary = write_metric_results(args, metrics, epoch, df, split)
    if args.get("logging", False) and summary:
        wd.log({f"{split}/{name}": stats["mean"] for name, stats in summary.items()
                if stats["valid_count"] > 0})


def log_loss(loss, epoch, split, log=True):
    if log:
        wd.log({f"{split}/loss": loss["total"].item()})

        wd.log({f"{split}/loss_sr": loss["sr"].item()})

        wd.log({f"{split}/loss_seg": loss["seg"].item()})

        if "sdf" in loss:
            wd.log(
                {
                    f"{split}/loss_sdf": loss["sdf"].item(),
                    f"{split}/loss_sdf_raw": loss["sdf_raw"].item(),
                }
            )

        if "graph" in loss:
            wd.log({f"{split}/loss_graph": loss["graph"].item()})

        for key in (
            "graph_ce_raw",
            "graph_sdf_raw",
            "graph_ce",
            "graph_sdf",
            "graph_radius_raw",
            "graph_radius",
            "graph_radius_valid_fraction",
        ):
            if key in loss:
                wd.log({f"{split}/loss_{key}": loss[key].item()})

        if "graph_raw" in loss:
            wd.log({f"{split}/loss_graph_raw": loss["graph_raw"].item()})

        wd.log({f"{split}/loss_moe": loss["moe"].item()})


def normalize_intensities(values, norm_type):
    values = np.clip(values, 0, None)
    values_mod = values[..., :-1]
    if norm_type == "minmax":
        v_min, v_max = values_mod.min(axis=0), values_mod.max(axis=0)
        values_mod = (values_mod - v_min) / (v_max - v_min)
    elif norm_type == "zscore":
        v_mean, v_std = values_mod.mean(axis=0), values_mod.std(axis=0)
        values_mod = (values_mod - v_mean) / (v_std + 1e-5)
    values[..., :-1] = values_mod
    return values


def ants_to_nib(ants_img):
    """Convert an ANTs image to NIfTI while preserving its physical geometry."""
    data = ants_img.numpy().astype(np.float32)

    spacing = np.array(ants_img.spacing)
    origin = np.array(ants_img.origin)
    direction = np.array(ants_img.direction)

    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin

    return nib.Nifti1Image(data, affine)


def denormalize_conditions(args, cond_key, values):
    constraint = args["dataset"]["constraints"][cond_key]

    if constraint.get("type") == "categoric":
        c_min = 0.0
        c_max = float(len(constraint["values"]) - 1)
    else:
        c_min = constraint["min"]
        c_max = constraint["max"]

    c_scale = args["atlas_gen"]["cond_scale"]

    if c_max > c_min:
        values = (values / c_scale + 1) / 2 * (c_max - c_min) + c_min
    else:
        values = values * 0 + c_min

    return values
