#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP launcher for CINeMA / VATL_MM atlas training.

Main features:
1. Works with torchrun / torch.distributed.run on SLURM.
2. Broadcasts output_dir from rank 0 to all ranks.
3. Supports command-line MoE overrides:
   --num_experts, --moe_k, --use_moe
4. Supports generic nested overrides using "__", e.g.:
   --inr_decoder__hidden_size 512
   --optimizer__lr_inr 1e-4
5. Saves config_final.yaml after all command-line overrides.
"""

import os

# Set thread limits BEFORE importing torch / numpy-heavy libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

import sys
import yaml
import argparse
import wandb as wd
import torch
import torch.distributed as dist
from datetime import datetime, timedelta

from build_atlas_ddp import AtlasBuilderDDP


class Logger:
    """Mirror stdout/stderr to a log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def str2bool(value):
    """Robust bool parser for command-line args."""
    if isinstance(value, bool):
        return value

    value = str(value).lower().strip()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def auto_cast_value(value):
    """
    Cast unknown CLI override values to bool/int/float when possible.

    This helps generic overrides such as:
      --optimizer__lr_inr 1e-4
      --logging false
      --batch_size 16
    """
    if value is None:
        return value

    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    low = text.lower()

    if low in {"true", "yes", "y"}:
        return True
    if low in {"false", "no", "n"}:
        return False

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return value


def set_nested(config, key_path, value):
    """
    Set nested config value using "__" separated key path.

    Example:
      key_path = "inr_decoder__num_experts"
      config["inr_decoder"]["num_experts"] = value
    """
    keys = key_path.split("__")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def override_args(config_args, cmd_args):
    """
    Apply command-line overrides.

    Rules:
    - internal DDP keys are skipped;
    - explicit MoE shortcut keys are handled later in apply_moe_overrides();
    - keys with "__" modify nested dictionaries;
    - top-level keys modify top-level config.
    """
    skip_keys = {
        "rank",
        "local_rank",
        "world_size",
        "is_distributed",
        "num_experts",
        "moe_k",
        "use_moe",
    }

    for key, value in cmd_args.items():
        if key in skip_keys:
            continue

        if value is None:
            continue

        value = auto_cast_value(value)

        if "__" in key:
            set_nested(config_args, key, value)
        else:
            config_args[key] = value

    return config_args


def apply_moe_overrides(args, cmd_args):
    """
    Shortcut overrides for MoE config.

    These directly modify:
      args["inr_decoder"]["num_experts"]
      args["inr_decoder"]["moe_k"]
      args["inr_decoder"]["use_moe"]
    """
    if "inr_decoder" not in args:
        args["inr_decoder"] = {}

    if cmd_args.get("num_experts") is not None:
        args["inr_decoder"]["num_experts"] = int(cmd_args["num_experts"])

    if cmd_args.get("moe_k") is not None:
        args["inr_decoder"]["moe_k"] = int(cmd_args["moe_k"])

    if cmd_args.get("use_moe") is not None:
        args["inr_decoder"]["use_moe"] = str2bool(cmd_args["use_moe"])

    # Safety check
    if args["inr_decoder"].get("use_moe", False):
        e = int(args["inr_decoder"].get("num_experts", 1))
        k = int(args["inr_decoder"].get("moe_k", 1))
        if k > e:
            raise ValueError(f"Invalid MoE config: moe_k={k} cannot be larger than num_experts={e}.")
        if k < 1:
            raise ValueError(f"Invalid MoE config: moe_k={k} must be >= 1.")
        if e < 1:
            raise ValueError(f"Invalid MoE config: num_experts={e} must be >= 1.")

    return args


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, sort_keys=False, allow_unicode=True)


def initial_setup(cmd_args):
    """
    Load YAML config, apply command-line overrides, create/broadcast output_dir,
    and save final config on rank 0.
    """
    rank = int(cmd_args.get("rank", 0))

    args_atlas = load_yaml("./configs/config_atlas.yaml")

    config_data_name = cmd_args.get("config_data", args_atlas.get("config_data"))
    config_data_all = load_yaml("./configs/config_data.yaml")

    if config_data_name not in config_data_all:
        raise KeyError(
            f"config_data={config_data_name} not found in ./configs/config_data.yaml. "
            f"Available keys: {list(config_data_all.keys())}"
        )

    args_data = {"dataset": config_data_all[config_data_name]}

    # Merge dataset config and atlas config.
    args = {**args_data, **args_atlas}
    args["config_data"] = config_data_name

    # Load subject IDs.
    subject_ids_path = args["dataset"]["subject_ids"]
    subject_ids_yaml = load_yaml(subject_ids_path)
    dataset_name = args["dataset"]["dataset_name"]

    if dataset_name in subject_ids_yaml:
        args["dataset"]["subject_ids"] = subject_ids_yaml[dataset_name]["subject_ids"]
    else:
        raise KeyError(
            f"dataset_name={dataset_name} not found in subject_ids file: {subject_ids_path}"
        )

    # Apply command-line overrides.
    args = override_args(args, cmd_args)
    args = apply_moe_overrides(args, cmd_args)

    # Rank 0 creates output dir.
    run_dir = args["output_dir"]

    if rank == 0:
        job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args['config_data']}_{time_stamp}_{job_id}"
        run_dir = os.path.join(args["output_dir"], run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Output directory created: {run_dir}", flush=True)

    # Broadcast output dir to all ranks.
    if dist.is_available() and dist.is_initialized():
        obj_container = [run_dir]
        dist.broadcast_object_list(obj_container, src=0)
        run_dir = obj_container[0]

    args["output_dir"] = run_dir

    # Rank 0 saves configs and initializes wandb.
    if rank == 0:
        os.makedirs(args["output_dir"], exist_ok=True)

        # Save original selected dataset block for readability.
        save_yaml(args_data, os.path.join(args["output_dir"], "config_data.yaml"))

        # Save original atlas config before overrides.
        save_yaml(args_atlas, os.path.join(args["output_dir"], "config_atlas_original.yaml"))

        # Save final effective config after all overrides.
        save_yaml(args, os.path.join(args["output_dir"], "config_final.yaml"))

        moe_cfg = args.get("inr_decoder", {})
        print(
            "[Config] MoE effective config: "
            f"use_moe={moe_cfg.get('use_moe')}, "
            f"num_experts={moe_cfg.get('num_experts')}, "
            f"moe_k={moe_cfg.get('moe_k')}",
            flush=True,
        )

        if args.get("logging", False):
            run_name = os.path.basename(run_dir)
            wd.init(
                config=args,
                project=args["project_name"],
                entity=args.get("wandb_entity"),
                name=run_name,
            )

    return args


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="CINeMA Atlas Builder DDP")

    # Common args
    parser.add_argument("--config_data", type=str, default=None, help="Dataset config key in configs/config_data.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # torchrun compatibility:
    # torchrun usually provides LOCAL_RANK by env, but older launchers may pass args.
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)

    # MoE shortcuts
    parser.add_argument("--num_experts", type=int, default=None, help="Override inr_decoder.num_experts")
    parser.add_argument("--moe_k", type=int, default=None, help="Override inr_decoder.moe_k")
    parser.add_argument("--use_moe", type=str, default=None, help="Override inr_decoder.use_moe: True/False")

    # Parse known args and keep unknown args for generic override.
    args, unknown = parser.parse_known_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}

    # Generic nested overrides:
    # Example:
    #   --inr_decoder__hidden_size 512
    #   --optimizer__lr_inr 1e-4
    #   --batch_size 16
    i = 0
    while i < len(unknown):
        item = unknown[i]

        if not item.startswith("--"):
            i += 1
            continue

        key = item[2:]

        # Boolean flag style: --logging
        if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
            cmd_args[key] = True
            i += 1
        else:
            cmd_args[key] = unknown[i + 1]
            i += 2

    return cmd_args


def init_distributed():
    """
    Initialize DDP from torchrun environment variables.

    Returns:
      rank, local_rank, world_size, device
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
        )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        return rank, local_rank, world_size, device

    print("Not running in Distributed Mode. Fallback to single GPU.", flush=True)
    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return rank, local_rank, world_size, device


def main():
    rank, local_rank, world_size, device = init_distributed()

    cmd_args = parse_cmd_args()
    cmd_args["rank"] = rank
    cmd_args["local_rank"] = local_rank
    cmd_args["world_size"] = world_size

    args = initial_setup(cmd_args)

    args["device"] = device
    args["rank"] = rank
    args["local_rank"] = local_rank
    args["world_size"] = world_size
    args["is_distributed"] = dist.is_available() and dist.is_initialized()

    if rank == 0:
        log_dir = os.path.join(args["output_dir"], "train")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training_log.txt")

        sys.stdout = Logger(log_file)
        sys.stderr = sys.stdout

        print(f"Logging initialized. All outputs will be saved to {log_file}", flush=True)
        print(f"DDP Enabled: {args['is_distributed']}. World Size: {world_size}", flush=True)
        print(f"Device: {device}", flush=True)

        moe_cfg = args.get("inr_decoder", {})
        print(
            "[Config] Effective MoE: "
            f"use_moe={moe_cfg.get('use_moe')}, "
            f"num_experts={moe_cfg.get('num_experts')}, "
            f"moe_k={moe_cfg.get('moe_k')}",
            flush=True,
        )

    AtlasBuilderDDP(args)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
