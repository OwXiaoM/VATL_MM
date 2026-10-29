from data_paths import relocate_path
import wandb as wd
import pandas as pd
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.inr_decoder import INR_Decoder, LatentRegressor
from data_loading.dataset import Data
from utils import *
import torch.nn.functional as F
from data_loading.graph_store import GraphStore


class AtlasBuilderDDP:
    def __init__(self, args):
        self.args = args
        self.device = args["device"]
        self.rank = args["rank"]
        self.world_size = args["world_size"]

        self.loss_criterion = Criterion(args).to(self.device)
        self._init_atlas_training()
        self.train_on_data()

    def train_on_data(self):
        self.broadcast_latents_from_rank0()

        if len(self.args["load_model"]["path"]) > 0 and self.rank == 0:
            self.validate(epoch_train=0)

        dist.barrier()

        loss_hist_epochs = []
        start_time = time.time()
        total_epochs = self.args["epochs"]["train"]

        for epoch in range(total_epochs):
            if "train" in self.dataloaders and hasattr(
                self.dataloaders["train"], "sampler"
            ):
                self.dataloaders["train"].sampler.set_epoch(epoch)

            if self.args["optimizer"]["re_init_latents"]:
                self.re_init_latents()

            loss = self.train_epoch(epoch, split="train")
            loss_hist_epochs.append(loss)

            if self.rank == 0:
                print(
                    f"Training: Epoch: {epoch}, Loss: {np.mean(loss_hist_epochs):.4f}, Total Time Epoch: {time.time() - start_time:.2f}s"
                )

            if epoch > 0 and (
                epoch % self.args["validate_every"] == 0 or epoch == total_epochs - 1
            ):
                self.synchronize_latents(split="train")

                if self.rank == 0:
                    self.validate(epoch)

                dist.barrier()

            self._update_scheduler(split="train")

        if total_epochs > 0:
            self.synchronize_latents(split="train")
            self.save_state(total_epochs - 1)
        return np.mean(loss_hist_epochs)

    def train_epoch(self, epoch, split):
        self.inr_decoder[split].train() if split == "train" else self.inr_decoder[
            split
        ].eval()
        loss_hist_batches = []

        for i, batch in enumerate(self.dataloaders[split]):
            if self.rank == 0 and i == 0:
                pass

            loss = self.train_batch(batch, epoch, split)
            loss_hist_batches.append(loss)

            if self.rank == 0 and (i % 50 == 0):
                print(
                    f"Split: {split}, Epoch: {epoch}, Batch: {i}/{len(self.dataloaders[split])}, Loss: {loss:.4f}"
                )

        return np.mean(loss_hist_batches)

    def train_batch(self, batch, epoch, split="train"):
        loss_hist_samples = []
        n_smpls = self.args["n_samples"]
        # Adapt validation/test latents without tissue or SDF supervision.
        seg_weight = self.args["optimizer"]["seg_weight"] if split == "train" else 0.0

        task_batch = isinstance(batch, dict)
        sampling_records = batch["sampling"] if task_batch else None
        sdf_batch = to_device(batch["sdf"], self.device) if task_batch else None
        coords_batch, values_batch, conditions_batch, idx_df_batch = to_device(
            batch["image"] if task_batch else batch, self.device
        )
        if task_batch and split != "train":
            raise ValueError("Task-specific batches are training-only")
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)

        for i, smpls in enumerate(sample_iterator):
            step_started = time.perf_counter()
            self.optimizers[split].zero_grad()

            coords = coords_batch[smpls : smpls + n_smpls]
            values = values_batch[smpls : smpls + n_smpls]

            idx_df = idx_df_batch[smpls : smpls + n_smpls].reshape(-1).long()

            if split == "train":
                conditions = conditions_batch[smpls : smpls + n_smpls]
            else:
                conditions = self.conditions_val[idx_df]

            n_voxel = coords.shape[0]
            n_sdf = 0
            sdf_end = 0
            if task_batch:
                # Partition the independently shuffled SDF batch proportionally across
                # the existing image microsteps; never add optimizer steps.
                start = smpls * len(sdf_batch[0]) // len(coords_batch)
                sdf_end = min(smpls + n_smpls, len(coords_batch)) * len(sdf_batch[0]) // len(coords_batch)
                if sdf_end == start:
                    raise ValueError("SDF budget too small for the image microstep count")
                scoords, starget, sconditions, sids = [x[start:sdf_end] for x in sdf_batch]
                sids = sids.reshape(-1).long()
                n_sdf = len(scoords)

            graph_batch_active = False
            graph_coords = None
            graph_conditions = None
            graph_idx_df = None
            graph_sdf_target = None
            graph_radius_target = None
            graph_cfg = self.args.get("graph_supervision", {})

            if (
                split == "train"
                and graph_cfg.get("activate", False)
                and getattr(self, "graph_store", None) is not None
            ):

                points_per_subject = int(graph_cfg.get("points_per_subject", 256))
                use_graph_radius = graph_radius_enabled(self.args)
                use_graph_sdf = (
                    graph_supervision_mode(self.args) in ("sdf", "both")
                    or use_graph_radius
                )
                graph_radius_list = []
                graph_sdf_list = []
                graph_coords_list = []
                graph_conditions_list = []
                graph_idx_list = []

                unique_subjects = torch.unique(idx_df)

                for subject_idx_tensor in unique_subjects:
                    subject_idx = int(subject_idx_tensor.item())

                    graph_sample = self.graph_store.sample_nodes(
                        idx_df=subject_idx,
                        n_points=points_per_subject,
                        device=self.device,
                        return_sdf=use_graph_sdf,
                        return_radius=use_graph_radius,
                    )
                    if use_graph_radius:
                        gcoords, gsdf, gradius = graph_sample
                        graph_sdf_list.append(gsdf)
                        graph_radius_list.append(gradius)
                    elif use_graph_sdf:
                        gcoords, gsdf = graph_sample
                        graph_sdf_list.append(gsdf)
                    else:
                        gcoords = graph_sample

                    n_graph_subject = gcoords.shape[0]

                    subject_mask = idx_df == subject_idx
                    if not torch.any(subject_mask):
                        continue

                    subject_condition = conditions[subject_mask][0]
                    gconditions = subject_condition.unsqueeze(0).expand(
                        n_graph_subject, -1
                    )

                    gids = torch.full(
                        (n_graph_subject,),
                        subject_idx,
                        dtype=torch.long,
                        device=self.device,
                    )

                    graph_coords_list.append(gcoords)
                    graph_conditions_list.append(gconditions)
                    graph_idx_list.append(gids)

                if len(graph_coords_list) > 0:
                    graph_coords = torch.cat(graph_coords_list, dim=0)
                    graph_conditions = torch.cat(graph_conditions_list, dim=0)
                    graph_idx_df = torch.cat(graph_idx_list, dim=0)
                    if use_graph_sdf:
                        graph_sdf_target = torch.cat(graph_sdf_list, dim=0)
                    if use_graph_radius:
                        graph_radius_target = torch.cat(graph_radius_list, dim=0)
                    graph_batch_active = True

            if task_batch:
                coords = torch.cat([coords, scoords], dim=0)
                conditions = torch.cat([conditions, sconditions], dim=0)
                idx_df_forward = torch.cat([idx_df, sids], dim=0)
            else:
                idx_df_forward = idx_df
            if graph_batch_active:
                coords_forward = torch.cat([coords, graph_coords], dim=0)
                conditions_forward = torch.cat([conditions, graph_conditions], dim=0)
                idx_forward = torch.cat([idx_df_forward, graph_idx_df], dim=0)
            else:
                coords_forward = coords
                conditions_forward = conditions
                idx_forward = idx_df_forward

            with torch.autocast(device_type="cuda", enabled=self.args["amp"]):
                output_forward, aux_loss = self.inr_decoder[split](
                    coords_forward,
                    self.latents[split],
                    conditions_forward,
                    self.transformations[split][idx_forward],
                    idcs_df=idx_forward,
                )

                values_p = output_forward[:n_voxel]

                loss = self.loss_criterion(
                    values_p,
                    values,
                    self.transformations[split][idx_df],
                    moe_loss=aux_loss,
                    seg_weight=seg_weight,
                    sdf_weight=None if split == "train" and not task_batch else 0.0,
                    apply_sdf=not task_batch,
                )

                if task_batch:
                    sdf_loss = self.loss_criterion.voxel_sdf_loss(
                        output_forward[n_voxel:n_voxel+n_sdf], starget[:, 0])
                    loss["sdf_raw"], loss["sdf"] = sdf_loss
                    loss["total"] = loss["total"] + loss["sdf"]

                if graph_batch_active:
                    graph_loss = graph_supervision_loss(
                        self.args,
                        output_forward[n_voxel+n_sdf:],
                        graph_sdf_target,
                        graph_radius_target,
                    )
                    loss.update(graph_loss)
                    loss["total"] = loss["total"] + graph_loss["graph"]
                else:
                    loss["graph"] = output_forward.new_zeros(())
                    loss["graph_raw"] = output_forward.new_zeros(())

            every = int(self.args.get("loss_diagnostics", {}).get("every", 0))
            self.loss_diagnostic_step = getattr(self, "loss_diagnostic_step", 0) + 1
            if every > 0 and self.loss_diagnostic_step % every == 0 and self.rank == 0:
                import json
                record = {"epoch": epoch, "split": split,
                          "step": self.loss_diagnostic_step, "rank": self.rank}
                if task_batch:
                    record["sampling"] = sampling_records
                    record["points_image"] = n_voxel
                    record["points_sdf"] = n_sdf
                    record["points_graph"] = len(graph_coords) if graph_batch_active else 0
                    record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device) if torch.device(self.device).type == "cuda" else 0
                record.update({k: float(v.detach()) if torch.is_tensor(v) else float(v)
                               for k, v in loss.items()})
                path = os.path.join(self.args["output_dir"], "loss_components.jsonl")
                with open(path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("[Loss components]", json.dumps(record))

            if self.args["amp"]:
                self.grad_scalers[split].scale(loss["total"]).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss["total"].backward()
                self.optimizers[split].step()

            if task_batch and every > 0 and self.loss_diagnostic_step % every == 0 and self.rank == 0:
                import json
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
                perf = {"epoch": epoch, "step": self.loss_diagnostic_step,
                        "step_seconds": time.perf_counter() - step_started,
                        "points_image": n_voxel, "points_sdf": n_sdf,
                        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if torch.device(self.device).type == "cuda" else 0}
                with open(os.path.join(self.args["output_dir"], "task_sampling_cost.jsonl"), "a") as stream:
                    stream.write(json.dumps(perf) + "\n")
            loss_hist_samples.append(loss["total"].item())

            if (i % 100 == 0 or i == (len(sample_iterator) - 1)) and self.rank == 0:
                log_loss(loss, epoch, split, self.args["logging"])

                if graph_batch_active:
                    print(
                        f"[DDP Graph V1] Epoch {epoch}, Step {i}/{len(sample_iterator)}, "
                        f"Total={loss['total'].item():.4f}, GraphRaw={loss['graph_raw'].item():.4f}, "
                        f"GraphWeighted={loss['graph'].item():.4f}, GraphPoints={graph_coords.shape[0]}"
                    )

        return np.mean(loss_hist_samples)

    def _is_distributed(self):
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def _get_rank_subject_indices(self, split="train"):
        total_len = len(self.datasets[split])

        if not self._is_distributed():
            return torch.arange(total_len, device=self.device, dtype=torch.long)

        sampler = getattr(self.dataloaders[split], "sampler", None)

        if isinstance(sampler, DistributedSampler):
            my_indices = list(iter(sampler))
        else:
            my_indices = list(range(self.rank, total_len, self.world_size))

        return torch.as_tensor(my_indices, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def synchronize_latents(self, split="train"):
        if not self._is_distributed():
            return

        if split not in self.latents:
            return

        total_len = len(self.datasets[split])
        my_indices = self._get_rank_subject_indices(split)

        local_latents = torch.zeros_like(self.latents[split].data)
        latent_counts = torch.zeros(
            total_len,
            dtype=self.latents[split].dtype,
            device=self.device,
        )

        if my_indices.numel() > 0:
            local_latents.index_add_(
                0, my_indices, self.latents[split].data[my_indices]
            )
            latent_counts.index_add_(
                0,
                my_indices,
                torch.ones(
                    my_indices.numel(), dtype=latent_counts.dtype, device=self.device
                ),
            )

        dist.all_reduce(local_latents, op=dist.ReduceOp.SUM)
        dist.all_reduce(latent_counts, op=dist.ReduceOp.SUM)

        denom = latent_counts.clamp_min(1.0).view(
            total_len, *([1] * (local_latents.dim() - 1))
        )
        self.latents[split].data.copy_(local_latents / denom)

        if (
            split in self.transformations
            and isinstance(self.transformations[split], torch.Tensor)
            and self.args["inr_decoder"]["tf_dim"] > 0
        ):
            local_tfs = torch.zeros_like(self.transformations[split].data)
            tf_counts = torch.zeros(
                total_len,
                dtype=local_tfs.dtype,
                device=self.device,
            )

            if my_indices.numel() > 0:
                local_tfs.index_add_(
                    0, my_indices, self.transformations[split].data[my_indices]
                )
                tf_counts.index_add_(
                    0,
                    my_indices,
                    torch.ones(
                        my_indices.numel(), dtype=tf_counts.dtype, device=self.device
                    ),
                )

            dist.all_reduce(local_tfs, op=dist.ReduceOp.SUM)
            dist.all_reduce(tf_counts, op=dist.ReduceOp.SUM)

            denom_tf = tf_counts.clamp_min(1.0).view(
                total_len, *([1] * (local_tfs.dim() - 1))
            )
            self.transformations[split].data.copy_(local_tfs / denom_tf)

        if self.rank == 0:
            print(
                f"[DDP] Latents and transformations synchronized for {split}.",
                flush=True,
            )

    @torch.no_grad()
    def broadcast_latents_from_rank0(self, split="train"):
        if not self._is_distributed():
            return

        if split in self.latents:
            dist.broadcast(self.latents[split].data, src=0)

        if (
            split in self.transformations
            and isinstance(self.transformations[split], torch.Tensor)
            and self.args["inr_decoder"]["tf_dim"] > 0
        ):
            dist.broadcast(self.transformations[split].data, src=0)

        if self.rank == 0:
            print(
                f"[DDP] Broadcast latents and transformations from rank0 for {split}.",
                flush=True,
            )

    def validate(self, epoch_train):
        self.save_state(epoch_train)
        if self.args["generate_cond_atlas"]:
            self.generate_atlas(epoch_train, n_max=100)

        print(f"Starting inference for Epoch {epoch_train}...")

        num_train = len(self.datasets["train"])
        train_indices = [0, 2, 3] if num_train > 3 else list(range(num_train))
        metrics_train = self.generate_subjects_from_df(
            idcs_df=train_indices, epoch=epoch_train, split="train"
        )
        log_metrics(
            self.args,
            metrics_train,
            epoch_train,
            df=self.datasets["train"].df,
            split="train",
        )

        if not self.args.get("test_time_adaptation", {}).get("enabled", True):
            print("Test-time adaptation disabled; skipping held-out subject inference.")
            return

        self._init_validation()

        epoch_val = 0
        for epoch_val in range(self.args["epochs"]["val"]):
            self.train_epoch(epoch=epoch_val, split="val")
            self._update_scheduler(split="val")
            self.analyze_latent_space(epoch_train, epoch_val=epoch_val)

        metrics_val = self.generate_subjects_from_df(
            idcs_df=range(len(self.datasets["val"])), epoch=epoch_val, split="val"
        )
        log_metrics(
            self.args, metrics_val, epoch_train, df=self.datasets["val"].df, split="val"
        )

    def _init_inr(self, state_dict=None, split="train"):
        self.args["inr_decoder"]["cond_dims"] = sum(
            [
                self.args["dataset"]["conditions"][c]
                for c in self.args["dataset"]["conditions"]
            ]
        )

        model = INR_Decoder(self.args, self.device).to(self.device)

        if state_dict is not None:
            new_state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(new_state_dict)

        if split == "train":
            self.inr_decoder[split] = DDP(
                model,
                device_ids=[self.args["local_rank"]],
                output_device=self.args["local_rank"],
                find_unused_parameters=True,
            )
        else:
            self.inr_decoder[split] = model

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split="train"):
        tsv_file = (
            pd.read_csv(relocate_path(self.args["dataset"]["tsv_file"]), sep="\t")
            if tsv_file is None
            else tsv_file
        )
        self.datasets[split] = Data(
            self.args, tsv_file, split=split, df_loaded=df_loaded
        )

        sampler = None
        shuffle = split == "train"

        if split == "train":
            sampler = DistributedSampler(
                self.datasets[split],
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            shuffle = False

        self.dataloaders[split] = DataLoader(
            self.datasets[split],
            batch_size=self.args["batch_size"],
            num_workers=self.args["num_workers"],
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=self.datasets[split].collate_fn,
            pin_memory=True,
            persistent_workers=self.args["num_workers"] > 0,
            prefetch_factor=2 if self.args["num_workers"] > 0 else None,
        )

        if self.rank == 0:
            print(
                f"Initialized DDP dataloader for {split} with {len(self.datasets[split])} subjects."
            )

    def save_state(self, epoch, split="train"):
        if self.rank != 0:
            return

        if self.args["save_model"]:
            log_dir = self.args["output_dir"]

            model_to_save = self.inr_decoder[split]
            if isinstance(model_to_save, DDP):
                model_to_save = model_to_save.module

            torch.save(
                {
                    "epoch": epoch,
                    "latents": self.latents[split].detach().cpu(),
                    "transformations": self.transformations[split].detach().cpu(),
                    "inr_decoder": model_to_save.state_dict(),
                    "tsv_file": self.datasets[split].tsv_file,
                    "dataset_df": self.datasets[split].df,
                    "args": self.args,
                },
                os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth"),
            )
            print(
                f'Saved model state to {os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")}'
            )

    def generate_subject_from_latent(
        self, latent_vec, condition_vector, transformation=None, split="train"
    ):
        grid_coords, grid_shape, affine = generate_world_grid(
            self.args, device=self.device
        )
        model = self.inr_decoder[split]
        if isinstance(model, DDP):
            model = model.module

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.args["amp"]):
                volume_inf = model.inference(
                    grid_coords,
                    latent_vec,
                    condition_vector,
                    grid_shape,
                    transformation,
                    include_sdf=sdf_export_enabled(self.args),
                )
        return volume_inf

    def generate_subjects_from_df(self, idcs_df=None, epoch=0, split="train"):
        import nibabel as nib

        metrics = []
        from data_loading.validation_selection import is_primary
        if not is_primary(self.args):
            return metrics
        if not (self.args["compute_metrics"] or self.args["save_imgs"][split]):
            return metrics
        model = self.inr_decoder[split]
        if isinstance(model, DDP):
            model = model.module

        def generate_native_grid(header_nii, world_bbox):
            shape = header_nii.shape
            affine = header_nii.affine
            i = torch.arange(0, shape[0], device=self.device)
            j = torch.arange(0, shape[1], device=self.device)
            k = torch.arange(0, shape[2], device=self.device)
            grid = torch.meshgrid(i, j, k, indexing="ij")
            grid_coords_idx = torch.stack(grid, dim=-1).reshape(-1, 3).float()
            affine_torch = torch.tensor(affine, dtype=torch.float32, device=self.device)
            ones = torch.ones((grid_coords_idx.shape[0], 1), device=self.device)
            grid_coords_homo = torch.cat([grid_coords_idx, ones], dim=1)
            grid_coords_phys = (affine_torch @ grid_coords_homo.T).T[:, :3]
            img_center_index = torch.tensor(shape, device=self.device) / 2.0
            center_homo = torch.cat(
                [img_center_index, torch.tensor([1.0], device=self.device)]
            )
            geometric_center = (affine_torch @ center_homo)[:3]
            grid_coords_norm = grid_coords_phys - geometric_center
            wb_torch = torch.tensor(world_bbox, dtype=torch.float32, device=self.device)
            grid_coords_norm = grid_coords_norm / (wb_torch / 2.0)
            return grid_coords_norm, list(shape), affine

        for idx_df in idcs_df:
            df_row_dict = self.datasets[split].df.iloc[idx_df].to_dict()
            ref_mod_path = df_row_dict[self.args["dataset"]["modalities"][0]]
            ref_nii = nib.load(ref_mod_path)
            grid_coords, grid_shape, affine = generate_native_grid(
                ref_nii, self.args["dataset"]["world_bbox"]
            )

            with torch.no_grad():
                transformations = self.transformations[split][idx_df, None]
                conditions = (
                    self.datasets[split].load_conditions(df_row_dict).to(self.device)
                )

                volume_inf = model.inference(
                    grid_coords,
                    self.latents[split][idx_df : idx_df + 1],
                    conditions,
                    grid_shape,
                    transformations,
                    include_sdf=sdf_export_enabled(self.args),
                )

            if (self.args.get("subject_to_atlas", {}).get("enabled", False)
                    and self.args["save_imgs"][split]):
                from subject_to_atlas import export_subject_to_atlas
                export_subject_to_atlas(
                    self.args, df_row_dict, transformations, epoch, split
                )

            if self.args["compute_metrics"]:
                metrics.append(
                    compute_metrics(
                        self.args, volume_inf, affine, df_row_dict, epoch, split
                    )
                )
            if self.args["save_imgs"][split]:
                save_subject(
                    self.args,
                    volume_inf,
                    affine,
                    sub_row_df=df_row_dict,
                    epoch=epoch,
                    split=split,
                )

        return metrics

    def generate_atlas(self, epoch=0, n_max=100):
        model = self.inr_decoder["train"]
        if isinstance(model, DDP):
            model = model.module
        model.eval()

        print(f"Generating atlases...\n")
        grid_coords, grid_shape, affine = generate_world_grid(
            self.args, device=self.device
        )
        temp_steps = self.args["atlas_gen"]["temporal_values"]
        atlas_list = []

        with torch.no_grad():
            for temp_step in temp_steps:
                temp_step_normed = normalize_condition(self.args, "scan_age", temp_step)
                mean_latent = self.get_mean_latent(
                    "scan_age", temp_step_normed, n_max=n_max
                )
                condition_vectors = generate_combinations(
                    self.args, self.args["atlas_gen"]["conditions"]
                )
                cond_list = []
                for c_v in condition_vectors:
                    if self.args["dataset"]["conditions"].get("scan_age", False):
                        c_v = [temp_step_normed] + c_v
                    c_v = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                    values_p = model.inference(
                        grid_coords,
                        mean_latent,
                        c_v,
                        grid_shape,
                        None,
                        include_sdf=sdf_export_enabled(self.args),
                    )
                    cond_list.append(values_p.detach().cpu())
                    torch.cuda.empty_cache()
                atlas_list.append(torch.stack(cond_list, dim=-1))
        atlas_list = torch.stack(atlas_list, dim=-1)
        save_atlas(
            self.args, atlas_list, affine, temp_steps, condition_vectors, epoch=epoch
        )

    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split="train"):
        c_ratio = 2 / (
            self.args["dataset"]["constraints"][condition_key]["max"]
            - self.args["dataset"]["constraints"][condition_key]["min"]
        )
        span_weeks = self.args["atlas_gen"]["gaussian_span"]
        sigma = 0.5 * span_weeks * c_ratio * self.args["atlas_gen"]["cond_scale"]
        latents = self.latents[split]
        condition_values, df_idcs = self.datasets[split].get_condition_values(
            condition_key, normed=True, device=self.device
        )
        weights = torch.exp(
            -((condition_values - condition_mean) ** 2) / (2 * (sigma**2))
        )
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        weights = weights[:, None, None, None, None]
        mean_latent = torch.sum(latents * weights, dim=0, keepdim=True)
        return mean_latent

    def analyze_latent_space(self, epoch, epoch_val=0):
        pass

    def load_checkpoint(self, chkp_path=None, epoch=None):
        chkp_path = os.path.join(chkp_path, f"checkpoint_epoch_{epoch}.pth")
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f"State file {chkp_path} not found!")
        chkp = torch.load(chkp_path, weights_only=False)
        self._init_dataloading(chkp["tsv_file"], chkp["dataset_df"])
        self._init_inr(chkp["inr_decoder"], split="train")
        self._init_transformations(chkp["transformations"])
        self._init_latents(chkp["latents"])
        if self.rank == 0:
            print(f"Loaded state from {chkp_path}")

    def _init_atlas_training(self):
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}

        chkp_path = self.args["load_model"]["path"]

        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args["load_model"]["epoch"])
        else:
            self._init_dataloading(split="train")
            self._init_inr(split="train")
            self._init_transformations(split="train")
            self._init_latents(split="train")

        self._init_optimizer(split="train")
        self._init_dataloading(split="val")

        self.graph_store = None

        graph_cfg = self.args.get("graph_supervision", {})

        if graph_cfg.get("activate", False):

            self.graph_store = GraphStore(self.args, self.datasets["train"].df)

            if self.rank == 0:
                print(
                    f"Initialized GraphStore for "
                    f"{len(self.datasets['train'])} training subjects."
                )

                for idx in range(min(3, len(self.datasets["train"]))):
                    print("[Graph QC]", self.graph_store.summary(idx))

    def _init_validation(self):
        self._seed()
        self._init_latents(split="val")
        self._init_transformations(split="val")
        self._init_optimizer(split="val")

        model_train = self.inr_decoder["train"]
        if isinstance(model_train, DDP):
            model_train = model_train.module

        self.inr_decoder["val"] = copy.deepcopy(model_train)
        self.inr_decoder["val"].eval()

    def _init_transformations(self, tfs=None, split="train"):
        shape = (len(self.datasets[split]), max(self.args["inr_decoder"]["tf_dim"], 6))
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = (
            nn.Parameter(tfs) if self.args["inr_decoder"]["tf_dim"] > 0 else tfs
        )

    def _init_latents(self, lats=None, split="train"):
        shape = (len(self.datasets[split]), *self.args["inr_decoder"]["latent_dim"])
        lats = (
            torch.normal(0, 0.01, size=shape).to(self.device)
            if lats is None
            else lats.to(self.device)
        )
        self.latents[split] = nn.Parameter(lats)
        if split == "val":
            shape_cond_val = (
                len(self.datasets["val"]),
                self.args["inr_decoder"]["cond_dims"],
            )
            self.conditions_val = nn.Parameter(
                torch.normal(0, 0.01, size=shape_cond_val).to(self.device)
            )

    def re_init_latents(self, split="train"):
        self.latents[split].data.normal_(0, 0.01)
        self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()

    def _init_optimizer(self, split="train"):
        params = [
            {
                "name": f"latents_{split}",
                "params": self.latents[split],
                "lr": self.args["optimizer"]["lr_latent"],
                "weight_decay": self.args["optimizer"]["latent_weight_decay"],
            }
        ]

        if self.args["inr_decoder"]["tf_dim"] > 0:
            params.append(
                {
                    "name": f"transformations_{split}",
                    "params": self.transformations[split],
                    "lr": self.args["optimizer"]["lr_tf"],
                    "weight_decay": self.args["optimizer"]["tf_weight_decay"],
                }
            )
        if split == "train":
            params.append(
                {
                    "name": f"inr_decoder",
                    "params": self.inr_decoder[split].parameters(),
                    "lr": self.args["optimizer"]["lr_inr"],
                    "weight_decay": self.args["optimizer"]["inr_weight_decay"],
                }
            )
        if split == "val":
            params.append(
                {
                    "name": f"conditions_val",
                    "params": self.conditions_val,
                    "lr": self.args["optimizer"]["lr_latent"],
                    "weight_decay": self.args["optimizer"]["latent_weight_decay"],
                }
            )
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args["amp"] else None
        if self.args["optimizer"]["scheduler"]["type"] == "cosine":
            self.schedulers[split] = CosineAnnealingLR(
                self.optimizers[split],
                T_max=self.args["epochs"][split],
                eta_min=self.args["optimizer"]["scheduler"]["eta_min"],
            )
        else:
            self.schedulers[split] = None

    def _update_scheduler(self, split="train"):
        if self.schedulers[split] is not None:
            self.schedulers[split].step()

    def _seed(self):
        torch.manual_seed(self.args["seed"])
        torch.cuda.manual_seed(self.args["seed"])
        np.random.seed(self.args["seed"])
