import os
from datetime import datetime
import yaml
import argparse
import wandb as wd
from build_atlas import AtlasBuilder
os.environ["WANDB__SERVICE_WAIT"] = "500"
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入文件，防止程序崩溃时丢失日志

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def sync_out_dim_with_dataset(args, rank=None):
    """
    Complete inr_decoder.out_dim using dataset.class_weights.

    Example:
        config_atlas.yaml: out_dim: [1]
        config_data.yaml: class_weights length = 11

    Then:
        args["inr_decoder"]["out_dim"] becomes [1, 11]
    """
    dataset = args["dataset"]
    inr_decoder = args["inr_decoder"]

    class_weights = dataset.get("class_weights", None)
    label_names = dataset.get("label_names", None)

    if class_weights is None:
        raise ValueError(
            "dataset.class_weights is required to infer inr_decoder.out_dim[-1]."
        )

    n_seg_channels = len(class_weights)

    should_print = rank is None or rank == 0

    if label_names is not None and len(label_names) != n_seg_channels:
        if should_print:
            print(
                f"WARNING: dataset.label_names has {len(label_names)} entries, "
                f"but dataset.class_weights has {n_seg_channels} entries."
            )

    out_dim = list(inr_decoder.get("out_dim", []))

    if len(out_dim) == 0:
        out_dim = [len(dataset["modalities"]) - 1, n_seg_channels]
    elif len(out_dim) == 1:
        out_dim = [out_dim[0], n_seg_channels]
    else:
        if out_dim[-1] != n_seg_channels and should_print:
            print(
                f"INFO: overriding inr_decoder.out_dim[-1] from {out_dim[-1]} "
                f"to {n_seg_channels} based on dataset.class_weights."
            )
        out_dim[-1] = n_seg_channels

    inr_decoder["out_dim"] = out_dim
    return args

def initial_setup(cmd_args=None):
    with open('./configs/config_atlas.yaml', 'r') as stream:
        args_atlas = yaml.safe_load(stream)
    with open('./configs/config_data.yaml', 'r') as stream:
        config_data = cmd_args['config_data'] if 'config_data' in cmd_args else args_atlas['config_data']
        args_data = {'dataset': yaml.safe_load(stream)[config_data]}
    args = {**args_data, **args_atlas}
    with open(args['dataset']['subject_ids'], 'r') as stream:
        args['dataset']['subject_ids'] = yaml.safe_load(stream)[args['dataset']['dataset_name']]['subject_ids']
    if cmd_args is not None:
        args = override_args(args, cmd_args)

    args = sync_out_dim_with_dataset(args)
    job_id = os.getenv("SLURM_JOB_ID", "loc")[-3:]
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dsetup = args['config_data']

    run_name =f"{dsetup}_{time_stamp}_{job_id}"
    args['output_dir'] = f"{args['output_dir']}/{run_name}"
    os.makedirs(args['output_dir'], exist_ok=True)
    print(f"Output directory: {args['output_dir']}")

    # save config files
    with open(os.path.join(args['output_dir'], 'config_data.yaml'), 'w') as f:
        yaml.dump(args_data, f)
    with open(os.path.join(args['output_dir'], 'config_atlas.yaml'), 'w') as f:
        yaml.dump(args_atlas, f)
    print(f"Saved config files to {args['output_dir']}")

    if args['inr_decoder']['out_dim'][0] != len(args['dataset']['modalities'])-1:
        print(f"WARNING: The number of output dimensions ({args['inr_decoder']['out_dim'][0]}) " 
              f"might not match the number of modalities ({len(args['dataset']['modalities'])-1}).")
    if args['atlas_gen']['conditions'] is not None: # check if all atlas conditions are set True in the dataset config
        for key in args['atlas_gen']['conditions'].keys():
            if not args['dataset']['conditions'][key]:
                print(f"WARNING: The atlas condition {key} is not set True in the dataset config."
                    f"Turning off the atlas generation for {key}.")
                args['atlas_gen']['conditions'].pop(key)

    if args['logging']: # init weights and biases if logging is True
        wd.init(config=args, project=args['project_name'], 
                                entity=args['wandb_entity'], name=run_name)
    return args


def override_args(config_args, cmd_args):
    for key, value in cmd_args.items():
        key1, key2 = key.split("__") if "__" in key else (key, None)
        if key2 is None:
            if value is not None:
                config_args[key] = value
        else:
            if value is not None:
                config_args[key1][key2] = value
    return config_args


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="CINeMA Atlas Builder")
    parser.add_argument("--config_data", type=str, help="Configuration data")
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--inr_decoder__out_dim", type=int, nargs='+', help="Number of output dimensions [#modalities, #classes of segmentation]")
    parser.add_argument("--inr_decoder__tf_dim", type=int, help="Degrees of freedom for the transformation")
    parser.add_argument("--inr_decoder__cnn_kernel_size", type=int, help="Kernel size for the CNN for spatial modulation")
    parser.add_argument("--inr_decoder__latent_dim", type=int, nargs='+', help="Latent dimension [c,x,y,z]")
    parser.add_argument("--inr_decoder__hidden_size", type=int, help="Hidden size of the sr network")
    parser.add_argument("--inr_decoder__num_hidden_layers", type=int, help="Number of hidden layers of the sr network")
    parser.add_argument("--inr_decoder__modulated_layers", type=int, nargs='+', help="Modulated layers")
    parser.add_argument("--atlas_gen__cond_scale", type=float, help="Scale of the condition vector")
    args = parser.parse_args()
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    return cmd_args


def main():
    cmd_args = parse_cmd_args()
    args = initial_setup(cmd_args)
    
    # ================= [新增代码开始] =================
    # 1. 确保 train 目录存在
    log_dir = os.path.join(args['output_dir'], 'train')
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. 设置日志文件路径
    log_file = os.path.join(log_dir, 'training_log.txt')
    
    # 3. 重定向 stdout 和 stderr
    # 这样所有的 print() 和 错误信息 都会同时显示在屏幕并保存到文件中
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout 
    
    print(f"Logging initialized. All outputs will be saved to {log_file}")
    # ================= [新增代码结束] =================

    print(args['inr_decoder'])
    atlas_builder = AtlasBuilder(args)

if __name__ == "__main__":
    main()
